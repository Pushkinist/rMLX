// unsafe_code: Metal kernel byte cast — slice::from_raw_parts over kernel input data for MSL dispatch
#![allow(unsafe_code)]

//! TurboFlash p1/p2 split-K FlashAttention kernel.
//!
//! # Apple10 (M5+) hazard — historical, cleared 2026-06
//!
//! TheTom's original TurboFlash kernel was default-OFF on Apple10 (M5+) due
//! to corruption producing garbage output (commit `67f076f2e` in
//! llama-cpp-turboquant). `67f076f2e` is a 4-line `return true`→
//! `return false` default-flip, NOT a kernel fix — no upstream TurboFlash
//! corruption fix exists.
//!
//! On rMLX, the initial B1 validation reproduced a more severe failure on
//! M5 Max: an `EXC_BAD_ACCESS (SIGSEGV) KERN_INVALID_ADDRESS at 0x0` the
//! instant the K8V4 flash path dispatched at 32k context on
//! Qwen3.6-35B-A3B-8bit (`head_dim = 256`). Null `Buffer::raw_ptr()` in
//! `Array::to_bytes` reading back the kernel output. The `apply_turbo_flags`
//! `Auto` arm therefore landed with a `family ≥ 10 → OFF` clause.
//!
//! **Re-validation (2026-06)**: the hazard was re-driven on M5 Max via
//! `crates/rmlx-kv-quant/tests/apple10_head_dim_256.rs` — a synthetic K8V4
//! cache at `head_dim = 256` driven through the public
//! `KvCache::update_and_sdpa` chain with `RMLX_TURBO_FLASH=1`, smoke +
//! 16-step decode stress + TF=0 control. Result:
//!
//! * smoke (1 dispatch, kv_seq=65): no SIGSEGV, cosine min 0.997 vs bf16.
//! * stress (16 dispatches, kv_seq up to 80): no SIGSEGV, cosine min 0.997.
//! * control (TF=0): dispatch dormant (delta=0).
//!
//! The 0.997 SDPA cosine vs the K8V4 fused-QK 0.999998 floor is the
//! **codec floor**, not a kernel issue. A CPU baseline test
//! (`tests/apple10_cpu_baseline.rs`) confirmed the V turbo-4 codec
//! encode→decode round-trip cosine alone is 0.997 (~identical at
//! head_dim ∈ {128, 256}). The K8V4 fused-QK 0.999998 measures Q·K^T
//! (K-dominated); the full SDPA (softmax @ V) shows V's turbo-4 codec
//! floor. Same numerics at both head dims.
//!
//! The documented hazard does not reproduce against the current kernel
//! surface. `apply_turbo_flags::Auto` now resolves ON across Apple7 through
//! Apple10+ (Apple11+ is optimistically ON with an operator-visible info log
//! until that family is hw-validated). See
//! `docs/reports/apple10-head-dim-256-revalidation.md` for the verbatim
//! numbers and the kernel changes that almost certainly closed the original
//! failure mode.
//!
//! The smoke-probe trip-wire below is retained as armour against any future
//! drift in the kernel that might revive the `!!!!!!`-style garbage-token
//! signature — it is cheap (≥4 consecutive identical token IDs check) and
//! gives a soft fallback if it ever fires.
//!
//! # What this is
//!
//! A 2-pass split-K FlashAttention kernel for rMLX's native KV storage format:
//! - K: q8_0 affine 8-bit, group_size=128, f32 scale, i8 codes packed as u32.
//! - V: TurboQuant 4-bit Lloyd-Max, group_size=32, f32 scale, 4-bit in u32.
//!
//! Architecture (adapted from TheTom `ggml-metal.metal:8843-9168`):
//!
//! **Pass 1** (`rmlx_turbo_flash_p1`): Each threadgroup processes a block of
//! `BLOCK_SIZE=64` KV tokens for one query head.
//! - Loads Q into registers (one lane per head_dim/32 slice).
//! - Dequants K (rMLX q8_0 format: i8 codes × f32 scale, group_size=128).
//! - Computes Q·K scores with attention mask.
//! - Online softmax within the block (running max + exp sum).
//! - Dequants V (rMLX turbo4 format: 4-bit Lloyd-Max × f32 scale, group_size=32).
//! - Accumulates softmax-weighted V in registers.
//! - Writes per-block partial {output[dim], block_max, block_sum} to DRAM.
//!
//! **Pass 2** (`rmlx_turbo_flash_p2`): One threadgroup per query head.
//! - Scans per-block {max, sum} to find global max.
//! - Rescales each block's output by `exp(block_max - global_max)`.
//! - Normalises by global sum and writes final output.
//! - NO inverse WHT: rMLX turbo4 V does NOT use WHT rotation.
//!
//! # Key differences from TheTom's kernel
//!
//! TheTom kernel uses `block_q8_0` (f16 scale, 32-element groups) and
//! `block_turbo3_0` (3-bit + sign array + WHT rotation). rMLX uses:
//! - K: `f32 scale` per 128 elements, i8 codes packed as 4 per u32.
//! - V: `f32 scale` per 32 elements, 4-bit Lloyd-Max codes (8 per u32).
//! - No WHT rotation on V — Lloyd-Max is a scalar codebook, not rotated.
//!
//! # Activation condition
//!
//! Only active for:
//! - `RMLX_TURBO_FLASH=1` env var.
//! - Decode step (q_seq = 1).
//! - K format: q8_0 (k_bits = 8 in KvQuant::K8V4).
//! - V format: turbo4 (KvQuant::K8V4 — the only mode with turbo V).
//! - kv_seq > 4096 (split-K wins when K-seq is long; below this threshold
//!   the existing `mixed_quantized_sdpa` is faster due to launch overhead).
//!
//! # Reference
//!
//! TheTom `ggml-metal.metal:8843` (p1), `:9034` (p2). N69 §1.
//! Commit `67f076f2e` disables it on Apple10. N73 §3.
//!
//! # Single-process GPU claim
//!
//! Per CLAUDE.md "Single MLX process per Mac" rule, callers must hold the
//! `/tmp/rmlx.<port>.claim` lock before dispatching GPU kernels.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::OnceLock;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};

// ── Env-var gate ──────────────────────────────────────────────────────────────

/// Returns true when the TurboFlash kernel is enabled by env var.
///
/// Default OFF at the env level; the CLI `--turbo-flash auto` (default)
/// resolves to `RMLX_TURBO_FLASH=1` via `rmlx_cli::commands::serve::apply_turbo_flags`
/// on every recognised Apple GPU family after the 2026-06 re-validation — see the
/// module-level rustdoc for the Apple10 hazard timeline.
pub fn turbo_flash_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("RMLX_TURBO_FLASH").as_deref(), Ok("1")))
}

/// Returns true when "lock-on" mode is enabled via `RMLX_TURBO_FLASH_LOCK=1`.
///
/// In lock-on mode the K8V4 decode path skips `update_decode_fp16` (bf16
/// K/V buffer maintenance) once the head-major persistent flash buffers
/// have been seeded. New K/V tokens are quantised directly into
/// `flash_k_codes` / `flash_v_codes` without ever touching the bf16 mirror.
///
/// **Trade-off**: lock-on disables the standard-SDPA fallback path for the
/// remainder of the request, because the bf16 buffer stops growing past
/// the seed point. Only enable when the TurboFlash kernel is known to be
/// correct on this hardware (VG.2 NIAH PASS at the target context length).
///
/// Lock-on has no effect unless `RMLX_TURBO_FLASH=1` is also set.
pub fn turbo_flash_lock_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("RMLX_TURBO_FLASH_LOCK").as_deref(), Ok("1")))
}

/// Default minimum KV sequence length for TurboFlash to activate.
///
/// Below this threshold the split-K dispatch overhead outweighs the
/// parallelism benefit. 4096 is conservative; TheTom's benchmarks show
/// the split-K crossover at ~4K tokens on M5 Max.
///
/// Override via `RMLX_TURBO_FLASH_MIN` env var. This is a perf gate, not a
/// correctness gate — proof runs lower it to 0 so dispatch fires on
/// short prompts. Production keeps the default.
pub const TURBO_FLASH_MIN_KV_SEQ: i32 = 4096;

/// Resolve the active TurboFlash min kv_seq threshold.
///
/// Reads `RMLX_TURBO_FLASH_MIN` once at first call; subsequent calls return
/// the cached value (`OnceLock`).  Parse failure falls back to the default
/// with a single `tracing::warn!`.
pub fn turbo_flash_min_kv_seq() -> i32 {
    use std::sync::OnceLock;
    static V: OnceLock<i32> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("RMLX_TURBO_FLASH_MIN") {
        Ok(s) => match s.parse::<i32>() {
            Ok(n) => {
                if n < 0 {
                    tracing::warn!(
                        value = n,
                        "RMLX_TURBO_FLASH_MIN: negative value clamped to 0"
                    );
                    0
                } else {
                    n
                }
            }
            Err(e) => {
                tracing::warn!(
                    value = %s,
                    error = %e,
                    default = TURBO_FLASH_MIN_KV_SEQ,
                    "RMLX_TURBO_FLASH_MIN: parse failed, using default"
                );
                TURBO_FLASH_MIN_KV_SEQ
            }
        },
        Err(_) => TURBO_FLASH_MIN_KV_SEQ,
    })
}

// ── Smoke-probe state ─────────────────────────────────────────────────────────

/// Set to true if the smoke-probe detected corruption and forced fallback.
static FORCED_FALLBACK: AtomicBool = AtomicBool::new(false);

/// Returns true if the smoke-probe triggered corruption fallback.
pub fn turbo_flash_corrupted() -> bool {
    FORCED_FALLBACK.load(Ordering::Relaxed)
}

// ── Dispatch counter ──────────────────────────────────────────────────────────
//
// Incremented exactly once per `turbo_flash_sdpa` invocation that reaches the
// P1 kernel-enqueue point. Used by the NIAH harness to *prove* the MSL kernel
// actually fired (vs. silently falling back to `mixed_quantized_sdpa`).
//
// Validation gates (head_dim, t_stride) live above the increment site, so a
// kernel-dispatch failure due to a bad gate increments only after those checks
// pass — i.e. the counter reflects real MSL P1 enqueues, not pre-check calls.
static TURBO_FLASH_DISPATCHES: AtomicU64 = AtomicU64::new(0);

/// Returns the process-lifetime count of TurboFlash P1 kernel dispatches.
///
/// NIAH harness uses this to assert the kernel actually fired (delta > 0 on
/// ON cells, delta == 0 on OFF cells). Production code does NOT consult this
/// counter.
pub fn turbo_flash_dispatch_count() -> u64 {
    TURBO_FLASH_DISPATCHES.load(Ordering::Relaxed)
}

/// Inspect the first `n_tokens` of the generated output for corruption.
///
/// Corruption signature: ≥4 consecutive identical token IDs (the `!!!!!!`
/// pattern observed on M5 Max with TheTom's kernel at commit `67f076f2e`).
///
/// If corruption is detected, sets `FORCED_FALLBACK` and returns `true`.
/// Once set, `turbo_flash_should_run` returns false for the process lifetime.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn smoke_probe_check(token_ids: &[u32]) -> bool {
    if token_ids.len() < 4 {
        return false;
    }
    // Check for ≥4 consecutive identical tokens.
    let mut run = 1usize;
    for i in 1..token_ids.len() {
        if token_ids[i] == token_ids[i - 1] {
            run += 1;
            if run >= 4 {
                tracing::warn!(
                    "TurboFlash: corruption detected — {} consecutive identical tokens \
                     (token_id={}). Apple10/M5 corruption (TheTom `67f076f2e`). \
                     Falling back to mixed_quantized_sdpa for remainder of process. \
                     Set RMLX_TURBO_FLASH=0 to suppress this probe.",
                    run,
                    token_ids[i]
                );
                FORCED_FALLBACK.store(true, Ordering::Relaxed);
                return true;
            }
        } else {
            run = 1;
        }
    }
    false
}

/// Returns true if TurboFlash should run for this decode step.
///
/// Conditions:
/// 1. `RMLX_TURBO_FLASH=1` env var.
/// 2. Smoke-probe has not forced fallback.
/// 3. q_seq == 1 (decode step only).
/// 4. kv_seq > turbo_flash_min_kv_seq() (default TURBO_FLASH_MIN_KV_SEQ=4096,
///    override via RMLX_TURBO_FLASH_MIN env var; values < 0 clamped to 0).
pub fn turbo_flash_should_run(q_seq: i32, kv_seq: i32) -> bool {
    turbo_flash_enabled()
        && !turbo_flash_corrupted()
        && q_seq == 1
        && kv_seq > turbo_flash_min_kv_seq()
}

// ── MSL constants ─────────────────────────────────────────────────────────────

/// Block size (tokens per P1 threadgroup). Matches TheTom's optimal B=64.
const BLOCK_SIZE: i32 = 64;

/// Threads per threadgroup in P1: 32 (one SIMD group).
const TG_SIZE: i32 = 32;

// ── Lloyd-Max 4-bit codebook (same as turboquant_msl.rs) ─────────────────────
//
// Embedded as hex f32 bit patterns for bit-exact MSL literals.
// Source: turboquant.rs::CODEBOOK_4BIT (Lloyd-Max N(0,1)).
const KERNEL_HEADER: &str = include_str!("metal/turbo_flash_header.metal");

// ── Pass 1 MSL source ─────────────────────────────────────────────────────────
//
// Grid: (n_bh × n_blocks, 1, 1) where n_bh = B × n_q_heads, n_blocks = ceil(T_active / BLOCK_SIZE).
// Threadgroup: (TG_SIZE, 1, 1) = (32, 1, 1) — exactly one SIMD group.
//
// Buffer layout (must match `add_input` order in `turbo_flash_p1`):
// 0. q_flat: f32 [B × n_q_heads × head_dim] — scaled queries (q_seq=1)
// 1. k_codes: u32 [B × n_kv_heads × T_stride × (head_dim/4)] — i8 codes, 4/u32
// 2. k_scales: f32 [B × n_kv_heads × T_stride × (head_dim/128)] — q8_0 scales
// 3. v_codes: u32 [B × n_kv_heads × T_stride × (head_dim/8)] — turbo4 codes, 8/u32
// 4. v_scales: f32 [B × n_kv_heads × T_stride × (head_dim/32)] — turbo4 scales
// 5. mask_flat: f32 [B × n_q_heads × T_active] or empty if no mask
// 6. params_p1: u32 [11] — {B, n_q_heads, n_kv_heads, n_repeats, T_active, head_dim, n_blocks, has_mask, q8_words_per_tok, tq4_words_per_tok, T_stride}
//
// `T_active` is the count of valid tokens (iteration bound + mask length).
// `T_stride` is the per-head row stride in K/V code/scale buffers — equal to
// the persistent buffer's `max_seq` dimension for the head-major P2.A.1 layout
// (slots `[T_active..T_stride)` are zero-init and never read). Pre-P2.A.1 the
// caller passed `T_stride == T_active` (chunk-major layout requantised fresh
// per dispatch); both modes share the same kernel.
//
// Output:
// partial_out: f32 [B × n_q_heads × n_blocks × head_dim]
// partial_ms: f32 [B × n_q_heads × n_blocks × 2] — {block_max, block_sum}
const P1_SOURCE: &str = include_str!("metal/turbo_flash_p1.metal");

// ── Pass 2 MSL source ─────────────────────────────────────────────────────────
//
// Grid: (B × n_q_heads, 1, 1).
// Threadgroup: (max(head_dim, 32), 1, 1) — covers all head_dim dims in one TG.
//
// Buffer layout (must match `add_input` order in `turbo_flash_p2`):
// 0. partial_out: f32 [B × n_q_heads × n_blocks × head_dim]
// 1. partial_ms: f32 [B × n_q_heads × n_blocks × 2]
// 2. params_p2: u32 [4] — {n_blocks, head_dim, B_times_n_q_heads, _pad}
//
// Output:
// dst: f32 [B × n_q_heads × head_dim]
const P2_SOURCE: &str = include_str!("metal/turbo_flash_p2.metal");

// ── Kernel singletons ─────────────────────────────────────────────────────────

static P1_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();
static P2_KERNEL: OnceLock<std::result::Result<MetalKernel, String>> = OnceLock::new();

fn p1_kernel() -> Result<&'static MetalKernel> {
    P1_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_turbo_flash_p1",
                KERNEL_HEADER,
                P1_SOURCE,
                &[
                    "q_flat",
                    "k_codes",
                    "k_scales",
                    "v_codes",
                    "v_scales",
                    "mask_flat",
                    "params_p1",
                ],
                &["partial_out", "partial_ms"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("turbo_flash_p1 kernel init: {e}")))
}

fn p2_kernel() -> Result<&'static MetalKernel> {
    P2_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_turbo_flash_p2",
                KERNEL_HEADER,
                P2_SOURCE,
                &["partial_out", "partial_ms", "params_p2"],
                &["dst"],
            )
            .map_err(|e| format!("{e}"))
        })
        .as_ref()
        .map_err(|e| Error::Mlx(format!("turbo_flash_p2 kernel init: {e}")))
}

// ── Public API ────────────────────────────────────────────────────────────────

/// 2-pass split-K FlashAttention for rMLX's native q8_0 K + turbo4 V format.
///
/// Replaces `mixed_quantized_sdpa` when `RMLX_TURBO_FLASH=1` and
/// `kv_seq > TURBO_FLASH_MIN_KV_SEQ` for the K8V4 decode path.
///
/// # Arguments
///
/// - `queries`: f32 `[B, n_q_heads, 1, head_dim]` — scaled queries (q_seq=1).
/// - `k_codes_flat`: u32 flat — rMLX q8_0 K codes (`i8` packed 4/u32).
/// - `k_scales_flat`: f32 flat — rMLX q8_0 K scales (one per Q8_GROUP=128 elems).
/// - `v_codes_flat`: u32 flat — rMLX turbo4 V codes (4-bit, 8/u32).
/// - `v_scales_flat`: f32 flat — rMLX turbo4 V scales (one per TQ4_GROUP=32 elems).
/// - `additive_mask`: optional causal mask, f32 `[B, n_q_heads, 1, T_active]`.
/// - `b`, `n_q_heads`, `n_kv_heads`, `head_dim`: dimensions.
/// - `t_active`: number of valid KV tokens (iteration bound + mask length).
/// - `t_stride`: per-head row stride of the K/V buffers in tokens. For the
///   chunk-major fresh-quantise path use `t_stride = t_active`. For the
///   Head-major persistent buffer use `t_stride = max_seq` —
///   slots `[t_active..t_stride)` are zero-init and never read.
/// - `device`: MLX device.
///
/// # Returns
///
/// f32 array of shape `[B, n_q_heads, 1, head_dim]`.
///
/// # Errors
///
/// Returns `Error::Mlx` if kernel dispatch fails.
/// Returns `Error::Quant` if head_dim is not in `{128, 256}` or not divisible
/// by 32. The kernel's register arrays (`q_vals`, `o_state`, `v_decoded`) are
/// sized for the larger of the two supported dims (head_dim=256, dims_per_lane=8).
#[allow(clippy::too_many_arguments)]
pub fn turbo_flash_sdpa(
    queries: &Array,
    k_codes_flat: &Array,
    k_scales_flat: &Array,
    v_codes_flat: &Array,
    v_scales_flat: &Array,
    additive_mask: Option<&Array>,
    b: i32,
    n_q_heads: i32,
    n_kv_heads: i32,
    t_active: i32,
    t_stride: i32,
    head_dim: i32,
    device: Device,
) -> Result<Array> {
    // Validate head_dim.
    if head_dim % 32 != 0 {
        return Err(Error::Quant(format!(
            "turbo_flash_sdpa: head_dim={head_dim} not divisible by 32"
        )));
    }
    if head_dim != 128 && head_dim != 256 {
        return Err(Error::Quant(format!(
            "turbo_flash_sdpa: head_dim={head_dim} not supported \
             (only 128 and 256 are wired). Fallback to mixed_quantized_sdpa."
        )));
    }
    if t_stride < t_active {
        return Err(Error::Quant(format!(
            "turbo_flash_sdpa: t_stride={t_stride} < t_active={t_active}"
        )));
    }

    let n_repeats = n_q_heads / n_kv_heads;
    let n_blocks = (t_active + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // ── Flatten queries to 1D f32 ─────────────────────────────────────────────
    let q_flat = {
        let q = queries.reshape(&[b * n_q_heads * head_dim], device)?;
        if q.dtype() == Dtype::F32 {
            q
        } else {
            q.astype(Dtype::F32, device)?
        }
    };

    // ── Mask flat (or dummy 1-element zero) ───────────────────────────────────
    let (mask_flat, has_mask) = if let Some(m) = additive_mask {
        let flat_len = b * n_q_heads * t_active;
        let m_f = if m.dtype() == Dtype::F32 {
            m.reshape(&[flat_len], device)?
        } else {
            m.astype(Dtype::F32, device)?.reshape(&[flat_len], device)?
        };
        (m_f, 1u32)
    } else {
        // Provide a dummy zero scalar — the kernel checks has_mask before reading.
        let zero_bytes = [0u8; 4];
        let dummy = Array::from_bytes(&zero_bytes, &[1], Dtype::F32)
            .map_err(|e| Error::Mlx(format!("turbo_flash dummy mask: {e}")))?;
        (dummy, 0u32)
    };

    // ── P1 params: see kernel layout above (11 u32 entries). ─────────────────
    let q8_words_per_tok = head_dim / 4; // i8 codes packed 4/u32
    let tq4_words_per_tok = head_dim / 8; // 4-bit codes packed 8/u32
    let params_p1: [u32; 11] = [
        b as u32,
        n_q_heads as u32,
        n_kv_heads as u32,
        n_repeats as u32,
        t_active as u32,
        head_dim as u32,
        n_blocks as u32,
        has_mask,
        q8_words_per_tok as u32,
        tq4_words_per_tok as u32,
        t_stride as u32,
    ];
    let params_p1_arr = {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(params_p1.as_ptr().cast::<u8>(), 11 * 4) };
        Array::from_bytes(bytes, &[11], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("turbo_flash p1 params: {e}")))?
    };

    // ── Dispatch P1 ───────────────────────────────────────────────────────────
    let kern_p1 = p1_kernel()?;
    let mut inv_p1 = MetalKernelInvoke::new();
    inv_p1.add_input(&q_flat)?;
    inv_p1.add_input(k_codes_flat)?;
    inv_p1.add_input(k_scales_flat)?;
    inv_p1.add_input(v_codes_flat)?;
    inv_p1.add_input(v_scales_flat)?;
    inv_p1.add_input(&mask_flat)?;
    inv_p1.add_input(&params_p1_arr)?;

    // partial_out: f32 [B × n_q_heads × n_blocks × head_dim]
    let partial_out_len = b * n_q_heads * n_blocks * head_dim;
    inv_p1.add_output_shape(&[partial_out_len], Dtype::F32)?;
    // partial_ms: f32 [B × n_q_heads × n_blocks × 2]
    let partial_ms_len = b * n_q_heads * n_blocks * 2;
    inv_p1.add_output_shape(&[partial_ms_len], Dtype::F32)?;

    // Grid is total threads (mlx-c `mlx_fast_metal_kernel` uses dispatchThreads
    // semantics — see https://ml-explore.github.io/mlx/build/html/dev/custom_metal_kernels.html).
    // We want exactly `B*n_q_heads * n_blocks` threadgroups, each with TG_SIZE=32
    // threads in the x-axis. So x-grid must be `(B*n_q_heads) * TG_SIZE`.
    //
    // Earlier the grid was set to `(B*n_q_heads, n_blocks, 1)` which produced
    // a single threadgroup of 16 threads (B*n_q_heads=16 for Qwen35B): only
    // head 0 was computed, simd_sum read 16 inactive lanes, output garbage.
    inv_p1.set_grid(b * n_q_heads * TG_SIZE, n_blocks, 1)?;
    inv_p1.set_thread_group(TG_SIZE, 1, 1)?;

    // Counter increment sits at the actual P1 enqueue point — after all
    // validation gates above, immediately before the kernel `.apply()` call
    // that submits work to the Metal command queue.
    // Reads via `turbo_flash_dispatch_count()` (NIAH harness only).
    TURBO_FLASH_DISPATCHES.fetch_add(1, Ordering::Relaxed);
    let mut p1_outs = kern_p1.apply(inv_p1, device)?;
    if p1_outs.len() < 2 {
        return Err(Error::Mlx("turbo_flash_p1: expected 2 outputs".to_owned()));
    }
    let partial_out = p1_outs.remove(0);
    let partial_ms = p1_outs.remove(0);

    // ── P2 params: [n_blocks, head_dim, B×n_q_heads, _pad] ──────────────────
    let params_p2: [u32; 4] = [
        n_blocks as u32,
        head_dim as u32,
        (b * n_q_heads) as u32,
        0u32,
    ];
    let params_p2_arr = {
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(params_p2.as_ptr().cast::<u8>(), 4 * 4) };
        Array::from_bytes(bytes, &[4], Dtype::U32)
            .map_err(|e| Error::Mlx(format!("turbo_flash p2 params: {e}")))?
    };

    // ── Dispatch P2 ───────────────────────────────────────────────────────────
    let kern_p2 = p2_kernel()?;
    let mut inv_p2 = MetalKernelInvoke::new();
    inv_p2.add_input(&partial_out)?;
    inv_p2.add_input(&partial_ms)?;
    inv_p2.add_input(&params_p2_arr)?;

    // dst: f32 [B × n_q_heads × head_dim]
    let dst_len = b * n_q_heads * head_dim;
    inv_p2.add_output_shape(&[dst_len], Dtype::F32)?;

    // Grid is total threads (dispatchThreads). We want `B*n_q_heads`
    // threadgroups, each with `tg_size = max(head_dim, 32)` threads.
    let tg_size = head_dim.max(32);
    inv_p2.set_grid(b * n_q_heads * tg_size, 1, 1)?;
    inv_p2.set_thread_group(tg_size, 1, 1)?;

    let mut p2_outs = kern_p2.apply(inv_p2, device)?;
    if p2_outs.is_empty() {
        return Err(Error::Mlx("turbo_flash_p2: expected 1 output".to_owned()));
    }
    let dst_flat = p2_outs.remove(0);

    // Reshape to [B, n_q_heads, 1, head_dim].
    dst_flat.reshape(&[b, n_q_heads, 1, head_dim], device)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "turbo_flash_msl_tests.rs"]
mod tests;
