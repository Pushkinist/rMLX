//! KV-cache quantization codecs, storage, MSL kernels, and per-layer cache.
//!
//! Extracted from `rmlx-quant` (KV-side weight-quant codecs) and
//! `rmlx-models::kv_cache` (storage + MSL wrappers + builder pieces). Owned by
//! this crate:
//!
//! * `turboquant` / `planarquant` — CPU codecs (formerly in `rmlx-quant`).
//! * `q8`, `q8_msl`, `turboquant_msl`, `planarquant_msl` — q8_0 helpers + MSL
//!   wrappers.
//! * `rot_k`, `rot_k_msl`, `mixed_quant`, `k8vturbo3_append_msl` — rotation +
//!   mixed-precision codec families.
//! * `turbo_flash_msl`, `sparse_v_msl` — TurboFlash split-K and sparse-V MSL
//!   kernels used inside the KV update + SDPA dispatch.
//! * `storage` — `QuantK` / `QuantV` / `QuantPlanarV` / `KvStorage` enum.
//! * `kvcache` — `KvCache`, the per-layer cache struct.
//! * `linear_attn` — `LinearAttnCache`, recurrent state for GatedDeltaNet.
//! * `paged` — paged-KV block table + page allocator.
//!
//! Higher-level wiring (`KvQuant`, `KvCacheBuilder`, SSD spill/hydrate, arch
//! entries) remains in `rmlx-models::kv_cache`. This crate is the
//! self-contained codec layer that downstream consumers can lift wholesale.
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::float_cmp,
        clippy::disallowed_methods,
        clippy::ignore_without_reason,
    )
)]

#[cfg(test)]
pub(crate) mod test_utils;

pub mod clifford;
pub(crate) mod fused_qk_common;
pub mod iso_fused_qk_msl;
pub mod isoquant;
pub mod isoquant_msl;
pub mod isoquant_msl_v4;
pub mod k8vturbo3_append_msl;
pub mod kvcache;
pub mod linear_attn;
pub mod mixed_quant;
pub mod paged;
pub mod planar_flash_decode_msl;
pub mod planar_fused_qk;
pub mod planar_fused_qk_msl;
pub mod planarquant;
pub mod planarquant_msl;
pub mod precompile;
pub mod q8;
pub mod q8_fused_qk_msl;
pub mod q8_msl;
pub mod quant;
pub mod rot_k;
pub mod rot_k_msl;
pub mod rotating;
pub mod rotor_flash_decode_msl;
pub mod rotor_flash_decode_symv_msl;
pub mod rotor_fused_qk_msl;
pub mod rotor_qjl;
pub mod rotorquant;
pub mod rotorquant_msl;
pub mod sparse_attn;
pub mod sparse_v_msl;
pub mod storage;
pub mod tcq;
pub mod tcq_v2_msl;
pub mod tcq_v_msl;
pub mod turbo2_v_msl;
pub mod turbo_flash_msl;
pub mod turbo_k3_fused_qk_msl;
pub mod turbo_k4_fused_qk_msl;
pub mod turboquant;
pub mod turboquant_msl;

pub use kvcache::{KvCache, SharedKv};
pub use linear_attn::LinearAttnCache;
pub use quant::{validate_rotor_k_asym_v, KvQuant, KvQuantParseError, KV_MAX_SEQ_DEFAULT};

// ── Fused-QK global enable gate ──────────────────────────────────────────────

/// Returns `true` when the generalized fused-QK kernels are enabled via the
/// `RMLX_FUSED_QK=1` environment variable.
///
/// Default OFF (env var absent or not `"1"`).  Mirrors the
/// [`planar_flash_decode_msl::planar_flash_decode_enabled`] OnceLock pattern:
/// the value is latched on first call and cached for the process lifetime.
///
/// The CLI flag `--fused-qk {on|off|auto}` in `rmlx-cli::commands::serve`
/// sets `RMLX_FUSED_QK=1` (or removes it) before the first inference, which
/// ensures this OnceLock reads the resolved value on first call.
pub fn fused_qk_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("RMLX_FUSED_QK").as_deref(), Ok("1")))
}

// ── Sparse-attention global enable gate ──────────────────────────────────────

/// Returns `true` when the two-phase sparse-attention dispatch is enabled via
/// the `RMLX_SPARSE_ATTN=1` environment variable.
///
/// Default OFF (env var absent or not `"1"`). Mirrors the
/// [`fused_qk_enabled`] OnceLock pattern: the value is latched on first call
/// and cached for the process lifetime.
///
/// The CLI flag `--sparse-attn {on|off|auto}` in `rmlx-cli::commands::serve`
/// sets `RMLX_SPARSE_ATTN=1` (or removes it) before the first inference, which
/// ensures this `OnceLock` reads the resolved value on first call.
///
/// **Audit verdict** (Path C, warm-TTFT dormant): the two-phase kernels are
/// wired and dispatch-counter-instrumented, but the production
/// `update_and_sdpa` path always shortcuts through the bf16-K seed materialised
/// by `exit_prefill`. Setting this gate to true does NOT make sparse-attn fire
/// on the normal generate flow; the kernels are reserved for **seedless**
/// workloads (synthetic PlanarK caches, PPL eval, future prompt-cache hits that
/// skip prefill). See [`sparse_attn::sparse_attn_total_dispatch_count`] for
/// the dispatch counter aggregator.
pub fn sparse_attn_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| matches!(std::env::var("RMLX_SPARSE_ATTN").as_deref(), Ok("1")))
}

// ── GPU-resident iso-blocks mirror gate ──────────────────────────────────────

/// Returns `true` when the GPU-resident `QuantIsoV3` mirror is enabled.
///
/// **Hardcoded OFF** (bench-driven decision; no env-var opt-in). A/B bench
/// showed deltas within noise on the `update_iso3` hot path because the
/// warm-TTFT bf16 seed absorbs the dequant cost before the mirror is reached.
/// See `docs/PERF_BASELINE.md` for bench numbers. The gate exists as a
/// forward-compatibility hook for future seedless decode paths where
/// `decode_fp16_k.is_none()` during steady-state decode.
#[cfg(not(test))]
pub fn gpu_resident_iso_enabled() -> bool {
    false
}

/// Test-only override for `gpu_resident_iso_enabled`. Latches the value for
/// the lifetime of the test binary (OnceLock semantics preserved). Call before
/// any `append_gpu` invocation. GPU mirror tests require `--test-threads=1`.
#[cfg(test)]
pub fn gpu_resident_iso_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| GPU_RESIDENT_ISO_FOR_TEST.load(std::sync::atomic::Ordering::Relaxed))
}

/// Test-only setter for the GPU-resident ISO gate. Set to `true` before the
/// first call to `gpu_resident_iso_enabled()` in the test binary. Requires
/// `--test-threads=1` (OnceLock latches on first read).
#[cfg(test)]
pub(crate) static GPU_RESIDENT_ISO_FOR_TEST: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Enables the GPU-resident ISO mirror for the remainder of this test binary.
/// Must be called before any `QuantIsoV3::append_gpu` (OnceLock latches on
/// first read of `gpu_resident_iso_enabled`). Run tests with `--test-threads=1`.
#[cfg(test)]
pub(crate) fn set_gpu_resident_iso_for_test(enabled: bool) {
    GPU_RESIDENT_ISO_FOR_TEST.store(enabled, std::sync::atomic::Ordering::Relaxed);
}
