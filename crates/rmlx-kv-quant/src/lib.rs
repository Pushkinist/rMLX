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
//! * `turbo_flash_msl` — TurboFlash split-K MSL kernel used inside the KV
//!   update + SDPA dispatch.
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

/// Rotation-quality gates: incoherence reduction and outlier-fixture cosine for
/// every codec family that applies an orthogonal transform.
#[cfg(test)]
mod rotation_fidelity_tests;

/// Rate-distortion reference: SQNR of every scalar-codebook codec against the
/// fixed-rate Lloyd-Max Gaussian anchor for its bit width.
#[cfg(test)]
mod rate_distortion_tests;

/// Stored-rate ceiling: bits per value every KV store family actually spends,
/// measured from real encoder output, against the bf16 floor.
#[cfg(test)]
mod kv_rate_tests;

pub(crate) mod bytes;
pub mod clifford;
pub(crate) mod flash_decode_common;
pub(crate) mod fused_qk_common;
pub mod iso_flash_decode_msl;
pub mod iso_flash_decode_symv_msl;
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
/// Deliberately out-of-bounds kernel used as the positive control for the
/// shader-validation gate. Off by default; see `scripts/run_gpu_tests.sh`.
#[cfg(feature = "shader-validation-canary")]
pub mod shader_validation_canary;
pub mod sparse_attn;
pub mod storage;
pub mod tcq;
pub mod tcq_v_msl;
pub mod turbo2_v_msl;
pub mod turbo_flash_msl;
pub mod turbo_k3_fused_qk_msl;
pub mod turbo_k4_fused_qk_msl;
pub mod turboquant;
pub mod turboquant_msl;

pub use kvcache::{KvCache, SharedKv};
pub use linear_attn::LinearAttnCache;
pub use quant::{
    validate_rotor_k_asym_v, KvQuant, KvQuantParseError, ALL_KV_QUANTS, KV_MAX_SEQ_DEFAULT,
};

// ── Kernel-path selection ────────────────────────────────────────────────────
//
// The fused-QK, sparse-attention, TurboFlash, planar-flash-decode and rot_k
// fused-FWHT kernel gates all live on [`rmlx_core::DispatchPolicy`]. Every
// `KvCache` captures a policy at construction ([`KvCache::dispatch_policy`])
// and the dispatch sites read it from there, so two caches built under
// different policies stay independent and both can run in one process.
//
// **Sparse-attention audit verdict** (warm-TTFT dormant): the two-phase
// kernels are wired and dispatch-counter-instrumented, but the production
// `update_and_sdpa` path always shortcuts through the bf16-K seed materialised
// by `exit_prefill`. Setting `DispatchPolicy::sparse_attn` does NOT make
// sparse-attn fire on the normal generate flow; the kernels are reserved for
// **seedless** workloads (synthetic PlanarK caches, PPL eval, future
// prompt-cache hits that skip prefill). See
// [`sparse_attn::sparse_attn_total_dispatch_count`] for the dispatch counter
// aggregator.

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
