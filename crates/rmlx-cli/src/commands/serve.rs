// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
//! `rmlx serve` — load a model and start the OpenAI-compatible HTTP server.
//!
//! [`run_serve`] is the single entry point. It:
//! 1. Resolves project caps from `projects.toml` (CLI flags override file).
//! 2. Acquires the single-MLX-process claim file via [`rmlx_server::try_claim`].
//! 3. Loads the model (or a multi-model registry) and warms prompt-cache slots.
//! 4. Launches the Axum HTTP server, then drives the idle-eviction loop until
//!    the process is signalled.
//!
//! # Public API
//!
//! - [`run_serve`] — main entry point called from the CLI dispatch table.
//! - [`apply_turbo_flags`] — bridge TurboFlash CLI booleans into the
//!   `OnceLock`-backed global used by the decode kernel.
//!
//! # See also
//!
//! - `docs/PROJECTS_CONFIG.md` — per-project cap resolution spec.
//! - `docs/SSD_TIER.md` — SSD KV spill tier configuration.

#![allow(
    clippy::cognitive_complexity,
    clippy::duration_suboptimal_units,
    clippy::fn_params_excessive_bools,
    clippy::too_many_lines,
    trivial_casts
)]
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use rmlx_core::runinfo::make_run_id;
use rmlx_loader::{discover_kv_calibration, load_config, load_head_budgets};
use rmlx_metrics::events::EventRecorder;
use rmlx_mlx::Device;
use rmlx_server::{
    register_ssd_prom_hooks, spawn_drainer, AppState, ArchGenerator, KeepAlivePolicy,
    ModelLoadConfig, ModelLoader, ModelRegistry, RegistryConfig, SpeculativeGenerator, TtftStore,
};
use tracing::{info, warn};

/// `--turbo-flash` tri-state. Default `Auto` resolves **OFF** on every host.
///
/// The kernel is correct — it is a throughput loss. TurboFlash is the only
/// consumer of the K8V4 flash path (`turbo_flash_should_run`: K8V4 storage,
/// `q_seq == 1`, `kv_seq > 4096`, `head_dim ∈ {128, 256}`), and on the one
/// shape where it engages it decodes several times slower than the generic
/// path it replaces while producing byte-identical tokens:
///
/// | cell (Bonsai-8B, `kv_h=8`, `head_dim=128`) | off | on | |
/// |---|---|---|---|
/// | k8v4 @16k, `--max-ctx 65536` | 89.4 TPS | 19.3 TPS | 4.6× slower |
/// | k8v4 @16k, `--max-ctx 16640` | 82.8 TPS | 24.2 TPS | 3.4× slower |
/// | k8v4 @32k, `--max-ctx 65536` | 61.3 TPS | 10.5 TPS | 5.9× slower |
///
/// Same binary, same cell, back to back, temp=0, `--turbo-flash` the only
/// difference; the generated-token digest is identical in both arms, so this is
/// pure cost, not a fidelity trade. Tightening the ring 4× recovers part of it
/// but not the bulk, so it is not a `--max-ctx` sizing artefact. The `on` arm
/// also holds ~722 MB more resident KV (the persistent head-major flash buffers
/// sit on top of the bf16 mirror and the packed store). On a shared-KV /
/// windowed arch (gemma-4-e2b, `kv_h=1`, `head_dim=256`, SWA 512) the kernel is
/// inert: ±0.3%, inside the 1.3% noise floor measured at `k8v4`@4k where the
/// `kv_seq > 4096` gate keeps it from firing at all.
///
/// `Auto` therefore holds OFF — the same HOLD posture
/// [`PlanarFlashDecodeMode`] already takes for the same reason. `on` remains
/// the explicit opt-in for ablation and for the re-validation that would lift
/// the HOLD. Nothing else changes: the kernel, its tests, and the
/// `head_dim = 256` hazard re-validation
/// (`docs/reports/apple10-head-dim-256-revalidation.md`, a *crash/fidelity*
/// clearance, never a throughput one) all stand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum TurboFlashMode {
    /// Force the TurboFlash MSL kernel on. Sets `RMLX_TURBO_FLASH=1`.
    On,
    /// Force the TurboFlash MSL kernel off. Does NOT clear an existing
    /// `RMLX_TURBO_FLASH=1` env-var — explicit env wins for back-compat.
    Off,
    /// Resolves to OFF on every host: the kernel is a measured 3.4–5.9× decode
    /// loss on the one codec it serves. HOLD until a re-measurement clears it.
    #[default]
    Auto,
}

impl std::fmt::Display for TurboFlashMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TurboFlashMode::On => f.write_str("on"),
            TurboFlashMode::Off => f.write_str("off"),
            TurboFlashMode::Auto => f.write_str("auto"),
        }
    }
}

/// Bridge the CLI `--turbo-flash` / `--turbo-flash-lock` flags into the
/// process environment so that the existing `OnceLock`-cached consumers in
/// `turbo_flash_msl.rs` pick them up on first read.
///
/// Semantics:
/// - [`TurboFlashMode::On`] → force-set `RMLX_TURBO_FLASH=1`.
/// - [`TurboFlashMode::Off`] → **hard override**: remove `RMLX_TURBO_FLASH`
///   from the environment so a stale `RMLX_TURBO_FLASH=1` in the shell
///   cannot latch the `OnceLock` to true after we've explicitly asked for
///   OFF. Previously this arm was a no-op, which meant `--turbo-flash off`
///   was silently a no-op whenever the shell had `RMLX_TURBO_FLASH=1` set.
/// - [`TurboFlashMode::Auto`] → resolves OFF on every host (see
///   [`TurboFlashMode`] for the measurements). The Apple family is still probed
///   and logged so an operator can see which host the HOLD applied to. `Auto`
///   is the only mode that honours a pre-existing `RMLX_TURBO_FLASH=1` env
///   (back-compat path) — it does not remove the var, so a shell that opts in
///   still gets the kernel.
///
/// `turbo_flash_lock` is unchanged: `true` → force-set, `false` → leave env
/// untouched.
///
/// Called from `main` before subcommand dispatch, so the `OnceLock` is not yet
/// initialised and every command — server and measurement alike — sees the same
/// resolution.
pub(crate) fn apply_turbo_flags(turbo_flash: TurboFlashMode, turbo_flash_lock: bool) {
    let family = rmlx_core::apple_gpu::apple_silicon_generation();
    apply_turbo_flags_inner(turbo_flash, turbo_flash_lock, family);
}

/// Pure inner of [`apply_turbo_flags`] parameterised on the Apple-Silicon GPU
/// family. Splitting the family probe out lets unit tests drive every Auto arm
/// (recognised family vs unknown host) regardless of which family the CI host
/// actually reports. `Auto` resolves OFF for both, so the parameter now only
/// selects which log line is emitted — it is kept so a future per-family flip
/// has a seam and a test surface.
///
/// `family = None` mirrors `apple_silicon_generation()` returning `None`
/// (sysctl probe failed / non-Apple-Silicon).
pub(crate) fn apply_turbo_flags_inner(
    turbo_flash: TurboFlashMode,
    turbo_flash_lock: bool,
    family: Option<u8>,
) {
    // Explicit Off is a hard override — remove the env var so a stale
    // RMLX_TURBO_FLASH=1 in the shell cannot latch the OnceLock to true.
    // Safe here because apply_turbo_flags runs from `main` before the tokio
    // runtime is built and before any other thread exists.
    if turbo_flash == TurboFlashMode::Off {
        std::env::remove_var("RMLX_TURBO_FLASH");
    }
    let resolved_on = match turbo_flash {
        TurboFlashMode::On => true,
        TurboFlashMode::Off => false,
        // Auto is a HOLD on every host. The kernel is correct — its output
        // digest matches the generic path token for token — but on the one
        // storage it serves (K8V4, kv_seq > 4096) it decodes 3.4-5.9x slower
        // than the path it replaces and carries several hundred MB of extra
        // resident KV for the privilege. Defaulting a measured decode loss ON
        // is worse than shipping no kernel at all, so Auto stays OFF until a
        // re-measurement clears it. See `TurboFlashMode` for the cells.
        // The family is still probed so the log names the host the HOLD
        // applied to.
        TurboFlashMode::Auto => {
            if let Some(f) = family {
                tracing::info!(
                    family = f,
                    "--turbo-flash=auto on Apple{f} — resolved OFF (HOLD: the \
                     kernel decodes 3.4-5.9x slower than the generic K8V4 path \
                     at kv_seq > 4096 for byte-identical output). Use \
                     --turbo-flash on to override."
                );
            } else {
                tracing::warn!(
                    "--turbo-flash=auto on unknown host (sysctl probe failed) — \
                     resolved OFF (conservative). Use --turbo-flash on to override."
                );
            }
            false
        }
    };
    if resolved_on {
        // SAFETY: set_var is safe here: apply_turbo_flags is called from `main`
        // before the tokio runtime is constructed and before any subcommand
        // dispatch. No other thread exists that could concurrently read the
        // environment.
        std::env::set_var("RMLX_TURBO_FLASH", "1");
        tracing::info!(
            mode = %turbo_flash,
            "--turbo-flash resolved ON; RMLX_TURBO_FLASH=1 applied"
        );
    } else if turbo_flash == TurboFlashMode::Off {
        tracing::info!(
            mode = %turbo_flash,
            "--turbo-flash resolved OFF (hard override); \
             RMLX_TURBO_FLASH removed from env"
        );
    } else {
        tracing::info!(
            mode = %turbo_flash,
            "--turbo-flash resolved OFF; env untouched (pre-existing \
             RMLX_TURBO_FLASH=1 still honoured for Auto back-compat)"
        );
    }
    if turbo_flash_lock {
        std::env::set_var("RMLX_TURBO_FLASH_LOCK", "1");
        tracing::info!("--turbo-flash-lock flag set; RMLX_TURBO_FLASH_LOCK=1 applied");
    }
}

/// `--planar-flash-decode` tri-state. Default `Auto` resolves at startup
/// based on the host's Apple GPU family — same Apple ≤9 vs ≥10 hazard
/// policy as `TurboFlashMode`. The two flash kernels share the same family
/// of register-pressure / threadgroup-memory failure modes, so until the
/// planar-flash decode is proven on Apple ≥10 we mirror the TurboFlash
/// defaults conservatively.
///
/// Added with the planar_flash_decode MSL kernel (2026-05). Defaults OFF;
/// validation found no measurable speedup (-0.19% at 4k canary) and NIAH was
/// blocked by a pre-existing bug. HOLD until both are resolved.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum PlanarFlashDecodeMode {
    /// Force the planar_flash_decode kernel on. Sets `RMLX_PLANAR_FLASH_DECODE=1`.
    On,
    /// Force the planar_flash_decode kernel off (hard override: removes
    /// `RMLX_PLANAR_FLASH_DECODE` from env so a stale `=1` cannot latch
    /// the `OnceLock` true on first read).
    Off,
    /// Resolves to OFF on every host. Validation complete: HOLD.
    #[default]
    Auto,
}

impl std::fmt::Display for PlanarFlashDecodeMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PlanarFlashDecodeMode::On => f.write_str("on"),
            PlanarFlashDecodeMode::Off => f.write_str("off"),
            PlanarFlashDecodeMode::Auto => f.write_str("auto"),
        }
    }
}

/// Bridge the CLI `--planar-flash-decode` flag into the process environment
/// so that `OnceLock`-cached consumers in
/// `rmlx_kv_quant::planar_flash_decode_msl::planar_flash_decode_enabled`
/// pick the resolved value up on first read.
///
/// Semantics (mirrors [`apply_turbo_flags`]):
/// - [`PlanarFlashDecodeMode::On`] → force-set `RMLX_PLANAR_FLASH_DECODE=1`.
/// - [`PlanarFlashDecodeMode::Off`] → **hard override**: remove
///   `RMLX_PLANAR_FLASH_DECODE` so a stale `=1` in the shell cannot latch
///   the `OnceLock` to true.
/// - [`PlanarFlashDecodeMode::Auto`] → currently resolves to OFF on every
///   host. Validation confirmed dispatch_delta > 0 on Bonsai+PlanarK and
///   byte-identical decoded output
///   vs the prior chain. The warm-TTFT bf16-K shortcut (see
///   `docs/reports/planar-chunked-prefill-fix.md`) unblocked the NIAH
///   correctness anchor but as a side effect: when the prefill bf16 K seed is
///   live (the normal post-`exit_prefill` decode flow) the PlanarK fused-QK /
///   flash-decode kernels intentionally do NOT fire — the dispatcher falls
///   through to bf16 SDPA so PlanarK matches the warm-TTFT semantics of
///   K8V4/K8V8/Planar/Mixed/K8VTurbo*/Iso*/Rotor*/TurboSym*. As a consequence
///   the planar-flash-decode kernel does not contribute to TPS in normal
///   generate flows, and the ≥10% Auto-flip gate cannot be met from a routine
///   prompt-cache miss. Auto therefore stays OFF until either (a) a seedless
///   workload (PPL eval / future prompt-cache hits that skip `exit_prefill`)
///   demonstrates a measurable speedup, or (b) the kernel is rewired to seed
///   itself from `decode_fp16_k` (mirroring TurboFlash).
///   Pre-existing `RMLX_PLANAR_FLASH_DECODE=1` in the shell is honoured for
///   back-compat.
///
/// Must be called before any inference (i.e. before the server starts
/// accepting requests), ensuring the `OnceLock` is not yet initialised.
pub(crate) fn apply_planar_flash_decode_flags(mode: PlanarFlashDecodeMode) {
    // Hard override on Off — remove the env var so a stale shell value
    // cannot latch the OnceLock.
    if mode == PlanarFlashDecodeMode::Off {
        std::env::remove_var("RMLX_PLANAR_FLASH_DECODE");
    }
    let resolved_on = match mode {
        PlanarFlashDecodeMode::On => true,
        PlanarFlashDecodeMode::Off => false,
        PlanarFlashDecodeMode::Auto => {
            // Auto stays OFF on every host. The warm-TTFT bf16-K shortcut
            // added by the PlanarK chunked-prefill fix bypasses the
            // planar-flash-decode kernel in the normal generate flow (the
            // bf16 prefill K seed is live for every post-`exit_prefill`
            // decode step, and the dispatcher honours it). The kernel still
            // works on seedless caches but no production flow currently
            // exercises that path, so there is no measurable TPS win to flip
            // Auto for.
            match rmlx_core::apple_gpu::apple_silicon_generation() {
                Some(family) => {
                    tracing::info!(
                        family,
                        "--planar-flash-decode=auto on Apple{family} — \
                         resolved OFF (no measurable speedup at canary shape). \
                         Use --planar-flash-decode on to override."
                    );
                }
                None => {
                    tracing::warn!(
                        "--planar-flash-decode=auto on unknown host — \
                         defaulting OFF (conservative)."
                    );
                }
            }
            false
        }
    };
    if resolved_on {
        // SAFETY: called at the top of `run_serve` before the tokio runtime
        // or any worker thread exists; no concurrent env reader.
        std::env::set_var("RMLX_PLANAR_FLASH_DECODE", "1");
        tracing::info!(
            mode = %mode,
            "--planar-flash-decode resolved ON; RMLX_PLANAR_FLASH_DECODE=1 applied"
        );
    } else if mode == PlanarFlashDecodeMode::Off {
        tracing::info!(
            mode = %mode,
            "--planar-flash-decode resolved OFF (hard override); \
             RMLX_PLANAR_FLASH_DECODE removed from env"
        );
    } else {
        tracing::info!(
            mode = %mode,
            "--planar-flash-decode resolved OFF; env untouched \
             (pre-existing RMLX_PLANAR_FLASH_DECODE=1 still honoured for \
             Auto back-compat)"
        );
    }
}

/// `--fused-qk` tri-state. Default `Auto` resolves OFF on every host
/// (HOLD pattern — kernels ship as stubs; codec implementations fill in
/// and flip `Auto` once the NIAH gate passes per codec).
///
/// Mirrors [`PlanarFlashDecodeMode`] exactly — same env-var / OnceLock
/// pattern, same Auto-HOLD rationale.
///
/// Added with the fused-QK kernel skeleton (2026-05).
/// Auto stays OFF until all five codec kernels pass their NIAH gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum FusedQkMode {
    /// Force the fused-QK kernels on. Sets `RMLX_FUSED_QK=1`.
    On,
    /// Force the fused-QK kernels off (hard override: removes
    /// `RMLX_FUSED_QK` from env so a stale `=1` cannot latch
    /// the `OnceLock` true on first read).
    Off,
    /// Resolves to OFF on every host. HOLD: kernel stubs not yet
    /// dispatching. Auto flips once codec implementations land and NIAH
    /// gates pass per codec.
    #[default]
    Auto,
}

impl std::fmt::Display for FusedQkMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FusedQkMode::On => f.write_str("on"),
            FusedQkMode::Off => f.write_str("off"),
            FusedQkMode::Auto => f.write_str("auto"),
        }
    }
}

/// Bridge the CLI `--fused-qk` flag into the process environment so that
/// `OnceLock`-cached consumers in `rmlx_kv_quant::fused_qk_enabled`
/// pick the resolved value up on first read.
///
/// Semantics (mirrors [`apply_planar_flash_decode_flags`]):
/// - [`FusedQkMode::On`] → force-set `RMLX_FUSED_QK=1`.
/// - [`FusedQkMode::Off`] → **hard override**: remove `RMLX_FUSED_QK` so a
///   stale `=1` in the shell cannot latch the `OnceLock` to true.
/// - [`FusedQkMode::Auto`] → currently resolves to OFF on every host.
///   Auto stays OFF until Executors B-F land and NIAH gates pass.
///   Pre-existing `RMLX_FUSED_QK=1` in the shell is honoured for back-compat.
///
/// Must be called before any inference (i.e. before the server starts
/// accepting requests), ensuring the `OnceLock` is not yet initialised.
pub(crate) fn apply_fused_qk_flags(mode: FusedQkMode) {
    // Hard override on Off — remove the env var so a stale shell value
    // cannot latch the OnceLock.
    if mode == FusedQkMode::Off {
        std::env::remove_var("RMLX_FUSED_QK");
    }
    let resolved_on = match mode {
        FusedQkMode::On => true,
        FusedQkMode::Off => false,
        FusedQkMode::Auto => {
            // HOLD — kernel stubs not yet dispatching. Auto stays OFF on
            // every host until each codec passes its NIAH gate.
            match rmlx_core::apple_gpu::apple_silicon_generation() {
                Some(family) => {
                    tracing::info!(
                        family,
                        "--fused-qk=auto on Apple{family} — resolved OFF \
                         (HOLD: kernel stubs not yet dispatching). \
                         Use --fused-qk on to override."
                    );
                }
                None => {
                    tracing::warn!(
                        "--fused-qk=auto on unknown host — defaulting OFF (conservative)."
                    );
                }
            }
            false
        }
    };
    if resolved_on {
        // SAFETY: called at the top of `run_serve` before the tokio runtime
        // or any worker thread exists; no concurrent env reader.
        std::env::set_var("RMLX_FUSED_QK", "1");
        tracing::info!(
            mode = %mode,
            "--fused-qk resolved ON; RMLX_FUSED_QK=1 applied"
        );
    } else if mode == FusedQkMode::Off {
        tracing::info!(
            mode = %mode,
            "--fused-qk resolved OFF (hard override); RMLX_FUSED_QK removed from env"
        );
    } else {
        tracing::info!(
            mode = %mode,
            "--fused-qk resolved OFF; env untouched \
             (pre-existing RMLX_FUSED_QK=1 still honoured for Auto back-compat)"
        );
    }
}

/// `--sparse-attn` tri-state. Default `Auto` resolves OFF on every host.
///
/// The `phase1_score` and `phase2_sparse_attend` MSL kernels shipped with
/// cosine parity ≥0.9997 vs the dense reference (3 configs). The
/// `--recipe head_budget` calibration writer has been validated on Bonsai.
///
/// **Audit verdict (2026-06)**: sparse-attn is **warm-TTFT dormant by
/// design** (Path C). The two-phase kernels operate over PlanarQuant-K
/// packed buffers; every production decode path uses the warm-TTFT bf16-K
/// seed materialised by `exit_prefill` (see
/// `docs/reports/planar-chunked-prefill-fix.md`), so the
/// sparse-attn dispatcher does not fire on the normal generate flow. The
/// kernels are reserved for **seedless** workloads (synthetic PlanarK
/// caches, PPL eval, future prompt-cache hits that skip prefill). This
/// matches the PlanarFlashDecode posture and is a generalisation of the
/// warm-TTFT cross-codec audit. Auto therefore stays OFF on every host —
/// same posture as `PlanarFlashDecodeMode::Auto`.
///
/// Mirrors [`FusedQkMode`] exactly — same env-var / `OnceLock` pattern.
/// The dispatch counter aggregator (`sparse_attn_total_dispatch_count`),
/// dormancy invariant tests, and the seedless dispatch test are in
/// `crates/rmlx-kv-quant/src/sparse_attn*.rs`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
pub(crate) enum SparseAttnMode {
    /// Force the two-phase sparse-attention dispatch on. Sets
    /// `RMLX_SPARSE_ATTN=1`.
    On,
    /// Force the sparse-attention dispatch off (hard override: removes
    /// `RMLX_SPARSE_ATTN` from env so a stale `=1` cannot latch the
    /// `OnceLock` true on first read).
    Off,
    /// Resolves to OFF on every host. Sparse-attn is warm-TTFT dormant by
    /// design (Path C): the production `update_and_sdpa` path always
    /// shortcuts through the bf16-K seed, so the two-phase kernels are
    /// reserved for seedless workloads. Same posture as
    /// `PlanarFlashDecodeMode::Auto`.
    #[default]
    Auto,
}

impl std::fmt::Display for SparseAttnMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SparseAttnMode::On => f.write_str("on"),
            SparseAttnMode::Off => f.write_str("off"),
            SparseAttnMode::Auto => f.write_str("auto"),
        }
    }
}

/// Bridge the CLI `--sparse-attn` flag into the process environment so that
/// `OnceLock`-cached consumers in `rmlx_kv_quant::sparse_attn_enabled`
/// pick the resolved value up on first read.
///
/// Semantics (mirrors [`apply_fused_qk_flags`]):
/// - [`SparseAttnMode::On`] → force-set `RMLX_SPARSE_ATTN=1`.
/// - [`SparseAttnMode::Off`] → **hard override**: remove `RMLX_SPARSE_ATTN`
///   so a stale `=1` in the shell cannot latch the `OnceLock` to true.
/// - [`SparseAttnMode::Auto`] → currently resolves to OFF on every host.
///   Auto stays OFF until Exec B/C land and the NIAH gate passes.
///   Pre-existing `RMLX_SPARSE_ATTN=1` in the shell is honoured for
///   back-compat.
///
/// Must be called before any inference (i.e. before the server starts
/// accepting requests), ensuring the `OnceLock` is not yet initialised.
pub(crate) fn apply_sparse_attn_flags(mode: SparseAttnMode) {
    // Hard override on Off — remove the env var so a stale shell value
    // cannot latch the OnceLock.
    if mode == SparseAttnMode::Off {
        std::env::remove_var("RMLX_SPARSE_ATTN");
    }
    let resolved_on = match mode {
        SparseAttnMode::On => true,
        SparseAttnMode::Off => false,
        SparseAttnMode::Auto => {
            // Sparse-attn is warm-TTFT dormant by design (Path C). The
            // production decode path shortcuts through the bf16-K seed, so
            // the two-phase kernels stay reserved for seedless workloads
            // (PPL eval, future prompt-cache hits). Auto resolves OFF on
            // every host — same posture as PlanarFlashDecodeMode::Auto.
            // The On override still routes through `sparse_attn_enabled()`
            // for callers that exercise the kernels directly (the
            // calibration runner, seedless integration tests).
            match rmlx_core::apple_gpu::apple_silicon_generation() {
                Some(family) => {
                    tracing::info!(
                        family,
                        "--sparse-attn=auto on Apple{family} — resolved OFF \
                         (warm-TTFT dormant by design). \
                         Use --sparse-attn on for seedless workloads."
                    );
                }
                None => {
                    tracing::warn!(
                        "--sparse-attn=auto on unknown host — \
                         defaulting OFF (conservative; warm-TTFT dormant)."
                    );
                }
            }
            false
        }
    };
    if resolved_on {
        // SAFETY: called at the top of `run_serve` before the tokio runtime
        // or any worker thread exists; no concurrent env reader.
        std::env::set_var("RMLX_SPARSE_ATTN", "1");
        tracing::info!(
            mode = %mode,
            "--sparse-attn resolved ON; RMLX_SPARSE_ATTN=1 applied"
        );
    } else if mode == SparseAttnMode::Off {
        tracing::info!(
            mode = %mode,
            "--sparse-attn resolved OFF (hard override); RMLX_SPARSE_ATTN removed from env"
        );
    } else {
        tracing::info!(
            mode = %mode,
            "--sparse-attn resolved OFF; env untouched \
             (pre-existing RMLX_SPARSE_ATTN=1 still honoured for Auto back-compat)"
        );
    }
}

/// Start the HTTP server synchronously (builds its own tokio runtime).
///
/// - `--model` → single-snapshot mode (Stage-1 behavior preserved).
/// - `--registry` → multi-model JSON config.
/// - Neither → empty registry, diagnostics only.
///
/// All registry entries are eagerly pre-loaded at startup before the
/// server begins accepting connections, so the first real request does not pay
/// model-load overhead in its TTFT. Fallback: if preload fails, the on-demand
/// path (`POST /v1/models/{id}/load` or first request) retries the load.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex::lock() cannot poison; Option/Result unwrap on values established by construction in this fn"
)]
#[allow(
    clippy::needless_pass_by_value,
    reason = "callers transfer ownership of CLI-parsed Option<String> params; refactoring to references would require lifetime parameters across the whole serve entry-point"
)]
pub(crate) fn run_serve(
    model: Option<&Path>,
    registry_file: Option<&Path>,
    host: &str,
    port: u16,
    device_str: &str,
    kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
    max_ctx_override: Option<i32>,
    idle_timeout_spec: Option<String>,
    prompt_cache_slots: usize,
    draft_model: Option<&Path>,
    draft_kind: Option<rmlx_models::DraftKind>,
    draft_block_size: Option<usize>,
    max_tokens_cap: u32,
    max_timeout_secs: u64,
    require_smoke_probe: bool,
    max_loaded_models: usize,
    max_queue_depth: usize,
    default_temperature: Option<f32>,
    enable_thinking: Option<bool>,
    kv_ssd_cache_gb: f64,
    kv_ssd_global_gb: f64,
    project: Option<String>,
    prompt_cache_ram_gb: Option<f64>,
    paged_kv: bool,
    paged_kv_page_tokens: Option<i32>,
    prefix_index_kind: rmlx_models::prefix_index::PrefixIndexKind,
    // Enable the in-process adaptive admission controller (default OFF).
    adaptive_admission: bool,
    // TTFT SLA target in ms for the adaptive controller (default 500 ms).
    ttft_target_ms: u64,
    // ITL SLA target in ms for the adaptive controller (default 50 ms).
    itl_target_ms: u64,
    // Enable adaptive prefill-chunk sizing (default OFF).
    adaptive_prefill_chunk: bool,
    whisper_model_path: Option<std::path::PathBuf>,
    whisper_tokenizer_path: Option<std::path::PathBuf>,
    tts_model_path: Option<std::path::PathBuf>,
    tts_tokenizer_path: Option<std::path::PathBuf>,
    // Multimodal encoder-output cache byte budget. `0` disables.
    // Default 512 MiB (set by main.rs default_value_t).
    mm_cache_bytes: usize,
    // Maximum number of sessions in the LRU session cache. Default 64.
    session_cache_max_sessions: usize,
    // Runtime YARN RoPE override for Qwen3 models that lack rope_scaling.
    // None = no override (default).
    yarn_override: Option<rmlx_models::qwen3::YarnOverride>,
    // Server-startup default image-token budget for Gemma4-unified vision
    // (--image-max-tokens). None = use the snapshot's processor_config.json
    // max_soft_tokens. A per-request image_max_tokens field overrides this.
    image_max_tokens: Option<usize>,
    sink: &EventRecorder,
) -> anyhow::Result<()> {
    // The TurboFlash / planar-flash-decode gates are resolved in `main`, before
    // any subcommand dispatch, alongside `--fused-qk` and `--sparse-attn`. They
    // are process-wide `OnceLock`s, so resolving them per-subcommand would let
    // `serve` and the measurement commands disagree on the kernel set.

    // load projects.toml and resolve caps via the precedence chain:
    // CLI flag > [project.<name>] > [global] > built-in default.
    // Missing file is a silent no-op. Malformed file is a startup error.
    let projects_cfg =
        rmlx_core::projects_config::load().map_err(|e| anyhow::anyhow!("projects.toml: {e}"))?;

    // kv_ssd_global_gb / kv_ssd_cache_gb have `default_value_t = 0.0` on the
    // clap arg, so 0.0 means "user did not pass the flag" — treat as None so
    // the file's [global] value can fill in. Only pass Some when the user
    // explicitly chose a non-zero value (CLI wins via precedence chain).
    let cli_caps = rmlx_core::projects_config::CliCaps {
        ssd_pool_gb: (kv_ssd_global_gb > 0.0).then_some(kv_ssd_global_gb),
        ssd_cap_gb: (kv_ssd_cache_gb > 0.0).then_some(kv_ssd_cache_gb),
        ram_prompt_cache_gb: prompt_cache_ram_gb,
    };
    let resolved =
        rmlx_core::projects_config::resolve_caps(&cli_caps, &projects_cfg, project.as_deref());

    // Log the applied config when any section was actually loaded.
    let any_global = projects_cfg.global.ssd_pool_gb.is_some()
        || projects_cfg.global.ram_prompt_cache_gb.is_some();
    let project_section_found = project
        .as_deref()
        .and_then(|n| projects_cfg.project.get(n))
        .is_some();
    let config_path = rmlx_core::paths::projects_toml_path();
    let exists = config_path.exists();
    if any_global || project_section_found {
        tracing::info!(
            path = %config_path.display(),
            global_applied = any_global,
            project_applied = project_section_found,
            project_name = project.as_deref().unwrap_or("(none)"),
            resolved_ssd_pool_gb = resolved.ssd_pool_gb,
            resolved_ssd_cap_gb = resolved.ssd_cap_gb,
            resolved_ram_prompt_cache_gb = ?resolved.ram_prompt_cache_gb,
            "projects.toml applied"
        );
    } else {
        tracing::debug!(
            config_path = %config_path.display(),
            project = ?project,
            ssd_pool_gb = resolved.ssd_pool_gb,
            ssd_cap_gb = resolved.ssd_cap_gb,
            ram_prompt_cache_gb = ?resolved.ram_prompt_cache_gb,
            reason = if exists { "no matching project section" } else { "no projects.toml" },
            "projects.toml: using defaults",
        );
    }

    // Use resolved caps for the SSD tier and RAM cap from this point on.
    let effective_kv_ssd_cache_gb = resolved.ssd_cap_gb;
    let effective_kv_ssd_global_gb = resolved.ssd_pool_gb;
    let effective_prompt_cache_ram_gb = resolved.ram_prompt_cache_gb;

    // install the process-global RAM cap (now from resolved caps).
    rmlx_models::prompt_cache::install_ram_cap(effective_prompt_cache_ram_gb);

    // install the process-global prompt-cache prefix-index strategy
    // before any model loads. Every `PromptCache<E>::new` built after this
    // point picks up the selected kind. Default is Linear; --prefix-index
    // radix opts into the radix-tree accelerator.
    rmlx_models::prefix_index::install_prefix_index_kind(prefix_index_kind);

    // install the paged-KV CLI config before any KvStorage::new
    // call reads the cached env.
    rmlx_kv_quant::paged::install_paged_kv(paged_kv, paged_kv_page_tokens);

    // + : install the process-global SSD prompt-cache tier config
    // before any model loads. Per-namespace budget 0 AND global budget 0 →
    // tier OFF (no spiller/hydrator installed, decode byte-identical to the
    // RAM-only path). The model-load path
    // (`ArchGenerator::from_snapshot_with_id` → `ssd_tier::attach_at_load`)
    // reads this config to attach the spiller + hydrator.
    //
    // per_project_budgets populated from projects.toml sections.
    let per_ns_bytes: u64 = if effective_kv_ssd_cache_gb > 0.0 {
        (effective_kv_ssd_cache_gb * 1024.0 * 1024.0 * 1024.0) as u64
    } else {
        0
    };
    let global_bytes: u64 = if effective_kv_ssd_global_gb > 0.0 {
        (effective_kv_ssd_global_gb * 1024.0 * 1024.0 * 1024.0) as u64
    } else {
        0
    };
    // Build per_project_budgets from loaded project sections.
    let mut per_project_budgets: std::collections::BTreeMap<String, u64> = projects_cfg
        .project
        .iter()
        .filter_map(|(name, pc)| {
            pc.ssd_cap_gb.map(|gb| {
                let bytes = (gb * 1024.0 * 1024.0 * 1024.0) as u64;
                (name.clone(), bytes)
            })
        })
        .collect();
    // CLI/resolver precedence wins: the active project's entry mirrors per_namespace_budget_bytes,
    // not the raw file value.
    if let Some(name) = &project {
        per_project_budgets.insert(name.clone(), per_ns_bytes);
    }
    rmlx_kv_ssd::ssd_tier::install_config(rmlx_kv_ssd::ssd_tier::SsdTierConfig {
        per_namespace_budget_bytes: per_ns_bytes,
        global_budget_bytes: global_bytes,
        default_namespace: project,
        per_project_budgets,
    })
    .map_err(|e| anyhow::anyhow!("ssd-tier config: {e}"))?;

    // --draft-model activates SpeculativeGenerator
    // (greedy acceptance, re-prefill rollback). Single-model path runs
    // unchanged when --draft-model is absent.
    if let Some(p) = draft_model {
        info!(
            draft_model = %p.display(),
            draft_kind = ?draft_kind,
            draft_block_size = ?draft_block_size,
            "--draft-model active — SpeculativeGenerator will be loaded \
             on first request (kind + block_size stored for drafter loaders)"
        );
    }
    // Parse device flag.
    let device = match device_str {
        "cpu" => Device::Cpu,
        "gpu" => Device::Gpu,
        other => {
            return Err(anyhow::anyhow!(
                "--device must be 'cpu' or 'gpu', got '{other}'"
            ));
        }
    };

    // Byte-to-byte port of mlx-lm server.py startup —
    // if mx.metal.is_available():
    // wired_limit = mx.device_info()["max_recommended_working_set_size"]
    // mx.set_wired_limit(wired_limit)
    // Locks pages to GPU residency, eliminating page-fault stalls during decode.
    match rmlx_mlx::metal::set_wired_limit_to_recommended() {
        Ok(Some((recommended, old))) => {
            info!(
                wired_limit_bytes = recommended,
                wired_limit_gib = (recommended as f64) / (1024.0 * 1024.0 * 1024.0),
                previous_wired_limit_bytes = old,
                "set_wired_limit(max_recommended_working_set_size)"
            );
        }
        Ok(None) => {
            info!("Metal backend not available; skipping set_wired_limit");
        }
        Err(e) => {
            warn!(error = %e, "set_wired_limit failed (continuing)");
        }
    }

    // Log the kv_quant and max_ctx selections.
    info!(
        ?kv_quant_override,
        ?max_ctx_override,
        "run_serve: kv_quant_override and max_ctx_override (engine uses auto=None if unset)"
    );

    // Build the registry from --model or --registry.
    let registry: Arc<ModelRegistry> = if let Some(reg_path) = registry_file {
        let cfg = RegistryConfig::from_file(reg_path)
            .map_err(|e| anyhow::anyhow!("registry file: {e}"))?;
        info!(
            path = %reg_path.display(),
            n = cfg.models.len(),
            "run_serve: loaded registry config"
        );
        Arc::new(ModelRegistry::from_config(&cfg))
    } else if let Some(p) = model {
        // B3: early architecture validation — fail fast before registry/chat-template
        // setup can mask the error. Reads config.json, checks architectures[0]
        // against KNOWN_ARCHS, and exits non-zero if unsupported.
        let model_cfg = load_config(p).map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
        let arch_name = model_cfg
            .architectures
            .first()
            .map_or("(empty)", String::as_str);
        if !rmlx_models::is_arch_supported(arch_name) {
            tracing::error!(
                arch = arch_name,
                model = %p.display(),
                "architecture '{}' not yet supported in v0.0.1; \
                 see crates/rmlx-models/src/arch.rs for how to add it",
                arch_name
            );
            eprintln!("error: architecture '{arch_name}' not yet supported in v0.0.1");
            std::process::exit(1);
        }
        Arc::new(ModelRegistry::from_paths(&[p.to_path_buf()]))
    } else {
        warn!("no --model or --registry; empty registry (diagnostics only)");
        Arc::new(ModelRegistry::default())
    };

    // The loader closure — called by AppState::ensure_loaded on demand.
    // When `--draft-model` is set, build a
    // SpeculativeGenerator instead of the single-model ArchGenerator.
    // bundle the shared model-load args into one config built from the
    // CLI-resolved flags. `gpu_gate` stays a separate handle (shared resource,
    // cloned per construction — not a load-config value).
    // Build the shared multimodal encoder-output cache before any model
    // loads. One `Arc<MultimodalCache>` is threaded into every generator
    // (via `ModelLoadConfig`) and into `AppState` for the jina-v4
    // `/v1/embeddings` path.
    let mm_cache = Arc::new(rmlx_models::multimodal_cache::MultimodalCache::new(
        mm_cache_bytes,
    ));
    info!(
        mm_cache_bytes,
        disabled = mm_cache.is_disabled(),
        "multimodal encoder-output cache initialised"
    );
    let load_cfg = ModelLoadConfig {
        device,
        kv_quant: kv_quant_override,
        max_ctx: max_ctx_override,
        prompt_cache_slots,
        mm_cache: Some(Arc::clone(&mm_cache)),
        // Calibration is per-model-path and is discovered inside the loader
        // closure below where `path` is known. Default None here.
        calibration: None,
        // YARN override — propagated from --yarn-factor / --yarn-original-max.
        yarn: yarn_override,
        // Server-startup image-token budget default (--image-max-tokens).
        image_max_tokens,
    };
    let draft_path: Option<std::path::PathBuf> = draft_model.map(Path::to_path_buf);
    // capture draft_kind + draft_block_size for the loader closure.
    // /14/15 loaders will branch on kind to select the right drafter.
    let loader_draft_kind = draft_kind;
    let loader_draft_block_size = draft_block_size;
    // C4: one process-wide GPU serialisation gate. A clone is injected into
    // every generator the loader builds so the existing try_lock/warn/lock
    // critical section in `Generator::generate` serialises across ALL
    // resident models (single Metal context per process).
    let gpu_gate: Arc<parking_lot::Mutex<()>> = Arc::new(parking_lot::Mutex::new(()));
    let gpu_gate_for_loader = Arc::clone(&gpu_gate);
    let loader: ModelLoader = Arc::new(move |path: &Path, id: &str| {
        // Probe kv_calib.json next to the model snapshot.
        // Requires head_size from config.json; load_config is cheap (already
        // done again inside the generator constructors, but needed here for
        // head_dim before construction begins). Explicit match arms so that
        // operator gets a warn when kv_calib.json is present but unusable.
        let calib_path = path.join("kv_calib.json");
        let mut calibration = if calib_path.is_file() {
            match load_config(path) {
                Ok(cfg) => {
                    if let Some(head_dim) = cfg.head_dim() {
                        discover_kv_calibration(path, head_dim as u32)
                    } else {
                        tracing::warn!(
                            path = %calib_path.display(),
                            "kv_calib.json present but head_dim unresolved from config; skipping"
                        );
                        None
                    }
                }
                Err(err) => {
                    tracing::warn!(
                        path = %calib_path.display(),
                        error = %err,
                        "kv_calib.json present but config.json load failed; skipping"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Probe head_budgets.json alongside kv_calib.json and attach it to
        // the calibration (schema v1.2 runtime-only field).
        // Missing file → Ok(None) → no-op. Malformed file → Err → warn and
        // skip so a bad calibration cannot bring down the loader.
        let hb_path = path.join("head_budgets.json");
        match load_head_budgets(&hb_path) {
            Ok(Some(hb)) => {
                tracing::info!(
                    path = %hb_path.display(),
                    num_layers = hb.num_layers,
                    num_heads = hb.num_heads,
                    "head_budgets.json loaded successfully"
                );
                if let Some(ref mut calib) = calibration {
                    calib.head_budgets = Some(hb);
                } else {
                    tracing::warn!(
                        path = %hb_path.display(),
                        "head_budgets.json present but no kv_calib.json — \
                         budgets ignored (sparse-attn needs both)"
                    );
                }
            }
            Ok(None) => { /* no head_budgets.json next to snapshot — common case */ }
            Err(err) => {
                tracing::warn!(
                    path = %hb_path.display(),
                    error = %err,
                    "head_budgets.json present but failed to load; skipping"
                );
            }
        }

        // Build a per-call load config with the discovered calibration.
        let effective_cfg = ModelLoadConfig {
            calibration,
            ..load_cfg.clone()
        };

        if let Some(dp) = draft_path.as_deref() {
            tracing::info!(
                model_id = %id,
                verifier = %path.display(),
                draft = %dp.display(),
                device = ?effective_cfg.device,
                kv_quant = ?effective_cfg.kv_quant,
                max_ctx = ?effective_cfg.max_ctx,
                cache_slots = effective_cfg.prompt_cache_slots,
                has_calibration = effective_cfg.calibration.is_some(),
                draft_kind = ?loader_draft_kind,
                draft_block_size = ?loader_draft_block_size,
                "loader: SpeculativeGenerator::from_snapshots"
            );
            let gen = SpeculativeGenerator::from_snapshots_with_id(
                path,
                dp,
                Some(id),
                &effective_cfg,
                Arc::clone(&gpu_gate_for_loader),
                loader_draft_kind,
                loader_draft_block_size,
            )?;
            Ok(Box::new(gen) as Box<dyn rmlx_server::Generator>)
        } else {
            tracing::info!(
                model_id = %id,
                device = ?effective_cfg.device,
                kv_quant = ?effective_cfg.kv_quant,
                max_ctx = ?effective_cfg.max_ctx,
                cache_slots = effective_cfg.prompt_cache_slots,
                has_calibration = effective_cfg.calibration.is_some(),
                "loader: ArchGenerator::from_snapshot"
            );
            let gen = ArchGenerator::from_snapshot_with_id(
                path,
                Some(id),
                &effective_cfg,
                Arc::clone(&gpu_gate_for_loader),
            )?;
            Ok(Box::new(gen) as Box<dyn rmlx_server::Generator>)
        }
    });

    // Open a shared metrics sink for the server (JSONL / CSV legacy path).
    let run_id_serve = make_run_id();
    let serve_sink = EventRecorder::open(&run_id_serve)
        .map_err(|e| anyhow::anyhow!("metrics open for serve: {e}"))?;
    let serve_sink = Arc::new(serve_sink);

    // step 2: wire the process-global SSD event recorder so that spill
    // and hydrate events are captured in the metrics DB. Must be called before
    // any model load (the loader closure that spawns spill threads runs below).
    rmlx_kv_ssd::set_ssd_event_recorder(Arc::clone(&serve_sink));

    // Hook the multimodal cache up to the same metrics sink so that
    // hit/miss events land in the `events` table.
    mm_cache.set_recorder(Arc::clone(&serve_sink), "global");

    // step 2: install the Prometheus histogram observation hooks so that
    // spill/hydrate durations are reflected in /metrics immediately after each
    // event, without going through the async drainer channel.
    register_ssd_prom_hooks();

    let addr_str = format!("{host}:{port}");
    println!("rmlx serve  {addr_str}");

    sink.record(&rmlx_metrics::events::Measurement {
        model_path: model.and_then(|p| p.to_str()).unwrap_or("(none)"),
        quant_mode: "n/a",
        stage: "stage3.5",
        op: "serve_start",
        value_unit: "bool",
        value: 1.0,
        notes: &addr_str,
    })
    .map_err(|e| anyhow::anyhow!("metrics record: {e}"))?;

    // Resolve the effective keep-alive policy from the precedence chain:
    //   --idle-timeout-secs > default 15 min.
    // Per-request `keep_alive` field overrides further at request time.
    let flag_policy = match idle_timeout_spec.as_deref() {
        Some(s) => Some(
            rmlx_server::parse_duration_spec(s)
                .map_err(|e| anyhow::anyhow!("--idle-timeout-secs '{s}': {e}"))?,
        ),
        None => None,
    };
    let idle_policy = KeepAlivePolicy::resolve(None, flag_policy);
    info!(
        address = %addr_str,
        idle_policy = ?idle_policy,
        ttl_secs = idle_policy.ttl_secs_for_log(),
        "rmlx serve starting"
    );

    // Run identity for the SPSC drainer comes from RunIdentity::rmlx() inside
    // the drainer itself — serve does not assemble it.
    let drainer_db_path = rmlx_core::paths::metrics_db_path();

    // F6/L18: SPSC async drainer db path for per-request SQLite metrics.
    // Spawned inside the tokio runtime below so `tokio::spawn` is available.

    // Cap worker threads to 4. On M5 Max the default (num_cpus ≈ 16)
    // wastes cores that MLX needs for CPU dispatch during prefill/decode.
    // HTTP + SSE + idle-eviction never saturate more than 4 async workers;
    // blocking inference runs in the separate blocking-thread pool regardless.
    //
    // `max_blocking_threads` + `thread_keep_alive`: every `spawn_blocking`
    // closure that touches MLX (generate, embeddings, image, audio,
    // transcribe, speculative) registers a per-thread default CPU/GPU stream
    // via `ensure_cpu_default_stream` / `ensure_gpu_default_stream`
    // (`docs/FFI.md` § stream context) — deliberately leaked for the thread's
    // lifetime, each backed by its own MLX-internal OS thread. Tokio's
    // blocking pool has no cap and idle-reaps threads after 10s by default;
    // under sporadic load that would create an unbounded, ever-growing set of
    // distinct worker threads over long serve uptime, each leaking its own
    // stream(s) — eventually exhausting the ~2048 per-process pthread ceiling
    // (`docs/FFI.md`). Bounding `max_blocking_threads` caps the worst-case
    // cumulative leak; a `thread_keep_alive` far longer than any realistic
    // idle gap between requests keeps that bounded set of workers alive
    // (reused, not reaped-and-replaced) so the cap is actually load-bearing.
    // 128 is 2× the default `--max-queue-depth` (64), leaving headroom for
    // concurrent audio/embeddings `spawn_blocking` calls alongside admitted
    // generate requests — GPU work itself stays single-flight via the
    // `gpu_queue` semaphore, so a modest cap does not throttle throughput.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .max_blocking_threads(128)
        .thread_keep_alive(Duration::from_secs(24 * 60 * 60))
        .enable_all()
        .build()
        .map_err(|e| anyhow::anyhow!("tokio runtime: {e}"))?;
    rt.block_on(async {
        // F6/L18: spawn the SPSC metrics drainer task.
        // Must be inside block_on so tokio::spawn is available.
        let drainer_handle = spawn_drainer(drainer_db_path);

        let mut state = AppState {
            registry,
            slots: Arc::new(parking_lot::RwLock::new(Vec::new())),
            embed_slot: Arc::new(parking_lot::RwLock::new(None)),
            // Clone the cache built above into AppState so the
            // /v1/embeddings (jina-v4) path can reach it.
            mm_cache: Arc::clone(&mm_cache),
            gpu_gate,
            // C5 Slice A: 1-permit semaphore = FIFO single-GPU serialisation.
            gpu_queue: Arc::new(tokio::sync::Semaphore::new(1)),
            gpu_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            max_queue_depth,
            max_loaded_models,
            loader,
            metrics: Some(serve_sink),
            idle_policy,
            max_tokens_cap,
            // A8: per-request HTTP timeout cap (seconds). 0 = disabled.
            max_timeout_secs,
            session_cache: Arc::new(parking_lot::Mutex::new(rmlx_server::SessionCache::new(
                session_cache_max_sessions,
            ))),
            // L6: TTFT ring-buffer — empty at startup, populated on first request.
            ttft_store: TtftStore::default(),
            // M30: ITL ring-buffer — empty at startup, populated after first decode.
            itl_store: rmlx_server::ItlStore::default(),
            // F6/L18: SPSC drainer for per-request SQLite metrics.
            metrics_drainer: Some(drainer_handle),
            // B5: --require-smoke-probe gate (default-OFF).
            require_smoke_probe,
            // G4: --default-temperature (None = absent = unchanged behaviour).
            default_temperature,
            // --enable-thinking (None = absent = thinking enabled by template default).
            default_enable_thinking: enable_thinking,
            // --image-max-tokens (None = absent = use snapshot processor_config default).
            default_image_max_tokens: image_max_tokens,
            // Process-lifetime cumulative token counters (prompt + completion).
            tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            // Per-category HTTP error lifetime counters.
            error_counts: rmlx_server::ApiErrorCounters::new(),
            // server startup timestamp + request lifecycle counters.
            started_at: std::time::Instant::now(),
            requests_started: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
            // Adaptive admission controller (default OFF).
            // When OFF, this is None and the open-loop FIFO path is unchanged.
            admission_controller: None,
            // Tick-task handle — set below when --adaptive-admission is used.
            admission_handle: None,
            // Whisper audio paths (None = audio disabled).
            whisper_model_path,
            whisper_tokenizer_path,
            // Whisper model cache — populated on first request.
            audio_model: Arc::new(parking_lot::RwLock::new(None)),
            // TTS paths (None = TTS disabled).
            tts_model_path,
            tts_tokenizer_path,
            // TTS model cache — populated on first TTS request.
            tts_model: Arc::new(parking_lot::RwLock::new(None)),
        };

        // Eager model preload — warm the resident set before serving requests
        // so cold TTFT does not include model-load overhead. Bounded to AT MOST
        // `max_loaded_models` entries (the `cap` alphabetically-first ids, since
        // `registry.list()` iterates a BTreeMap sorted by id — not JSON order):
        // anything beyond the resident cap would be evicted by the next
        // `ensure_loaded` (see `AppState::ensure_loaded` LRU swap), so preloading
        // it is pure load-cost + transient memory pressure with nothing kept.
        // The rest stay lazy — the documented load-on-demand / idle-unload path
        // handles them on first request.
        // `ensure_loaded` is synchronous (CPU-bound disk + dequant); run it in
        // the blocking-thread pool so we do not stall the async runtime.
        // Best-effort: a load failure logs a warning but does not abort startup
        // (the first real request will attempt the load again via the normal
        // on-demand path and surface a 503 if it still fails).
        {
            let cap = max_loaded_models.max(1);
            let ids: Vec<String> = state
                .registry
                .list()
                .iter()
                .take(cap)
                .map(|e| e.id.clone())
                .collect();
            let state_ref = state.clone();
            tokio::task::spawn_blocking(move || {
                for id in &ids {
                    tracing::info!(model_id = %id, "eager preload starting");
                    let t = std::time::Instant::now();
                    match state_ref.ensure_loaded(id) {
                        Ok(_) => tracing::info!(
                            model_id = %id,
                            load_ms = t.elapsed().as_millis(),
                            "eager preload complete"
                        ),
                        Err(e) => tracing::warn!(
                            model_id = %id,
                            error = %e,
                            "eager preload failed; model will load on first request"
                        ),
                    }
                }
            })
            .await
            .unwrap_or_else(
                |e| tracing::warn!(error = %e, "eager preload: spawn_blocking panicked"),
            );
        }

        // Install the adaptive admission controller when --adaptive-admission is set.
        // Must happen after AppState is built but before accepting connections.
        // Mutates state.admission_controller and spawns the background tick task.
        if adaptive_admission {
            // Resolve the primary model's prefill-chunk arch key from the registry
            // (the per-model config class name, mapped to a module-style key). The
            // controller's prefill-chunk fallback is process-wide; with a single
            // --model there is exactly one entry. Empty string → conservative
            // FALLBACK chunk, never the oversized gemma4 default.
            let prefill_arch = state
                .registry
                .list()
                .first()
                .map_or("", |e| {
                    rmlx_models::prefill_chunk::module_key_for_class(&e.arch)
                })
                .to_owned();
            let ctrl = rmlx_server::ControllerHandle::new(
                rmlx_server::ControllerConfig::new(
                    ttft_target_ms, // M2: flag --ttft-target-ms maps to step_target_ms
                    itl_target_ms,
                    max_queue_depth.max(1),
                    prefill_arch,
                )
                .with_adaptive_prefill_chunk(adaptive_prefill_chunk),
                state.metrics.clone(),
            );
            info!(
                step_target_ms = ttft_target_ms,
                itl_target_ms,
                initial_queue_depth = max_queue_depth,
                adaptive_prefill_chunk,
                "adaptive admission controller enabled"
            );
            // Store the handle on AppState so it is aborted when AppState is
            // dropped (graceful shutdown / runtime teardown).
            let admission_handle = rmlx_server::spawn_controller_task(ctrl.clone());
            state.admission_handle = Some(Arc::new(admission_handle));
            state.admission_controller = Some(ctrl);
        }

        // Per-model timers are armed inside `ensure_loaded`; no global poll
        // loop is needed. The eager preload above already armed timers for
        // every preloaded slot via `ensure_loaded`.

        rmlx_server::serve(state, host, port).await
    })
}

#[cfg(test)]
#[path = "serve_tests.rs"]
mod tests;
