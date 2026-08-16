//! Which optional kernel paths a KV cache dispatches through.
//!
//! Every field selects between a fused MSL kernel and the generic path it
//! replaces. The value is a plain `Copy` struct: it is resolved once (from the
//! CLI surface, or from the environment when rMLX is driven as a library),
//! captured by each [`crate`]-downstream KV cache at construction, and read
//! from there at dispatch time. Nothing about it is latched, so two caches
//! built under different policies stay independent for their whole lifetime
//! and both can be exercised in one process — which is what an interleaved
//! A/B comparison of two kernel paths needs.
//!
//! # Environment fallback
//!
//! [`DispatchPolicy::from_env`] reads the variables that predate the CLI
//! flags. They remain the fallback for the `auto` arm of every flag and for
//! embedders that never construct a policy, so an exported
//! `RMLX_TURBO_FLASH=1` still turns the kernel on.
//!
//! | Variable | Field | Accepted |
//! |---|---|---|
//! | `RMLX_FUSED_QK` | `fused_qk` | `1` |
//! | `RMLX_FUSED_QK_MIN` | `fused_qk_min_kv_seq` | integer |
//! | `RMLX_SPARSE_ATTN` | `sparse_attn` | `1` |
//! | `RMLX_TURBO_FLASH` | `turbo_flash` | `1` |
//! | `RMLX_TURBO_FLASH_LOCK` | `turbo_flash_lock` | `1` |
//! | `RMLX_TURBO_FLASH_MIN` | `turbo_flash_min_kv_seq` | integer, negatives clamped to 0 |
//! | `RMLX_PLANAR_FLASH_DECODE` | `planar_flash_decode` | `1` |
//! | `RMLX_ROT_K_FUSED` | `rot_k_fused` | `1` |

use std::sync::{LazyLock, PoisonError, RwLock};

/// Default minimum `kv_seq` for the generalized fused-QK kernels to dispatch.
///
/// Below this the per-step head-major K encode costs more than the fused
/// kernel saves.
pub const DEFAULT_FUSED_QK_MIN_KV_SEQ: i32 = 512;

/// Default minimum `kv_seq` for the TurboFlash split-K kernel to dispatch.
///
/// Below this the split-K dispatch overhead outweighs the parallelism
/// benefit; the generic `mixed_quantized_sdpa` path is faster.
pub const DEFAULT_TURBO_FLASH_MIN_KV_SEQ: i32 = 4096;

/// Kernel-path selections for one KV cache.
///
/// Every boolean defaults to `false` — the generic path — and every threshold
/// to its `DEFAULT_*` constant. [`DispatchPolicy::from_env`] is the only
/// constructor that consults the environment.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed policy value — callers build it field-by-field (CLI resolution, benches, tests); adding a field is a deliberate review point at every construction site"
)]
#[allow(
    clippy::struct_excessive_bools,
    reason = "one flat bool per kernel gate is the point: the gates are independent, and folding them into enums would invent states the dispatch sites cannot express"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DispatchPolicy {
    /// Generalized fused-QK kernels for the q8 / turbo3 / turbo4 / iso /
    /// rotor K-packed caches.
    pub fused_qk: bool,
    /// Minimum `kv_seq` for `fused_qk` to dispatch.
    pub fused_qk_min_kv_seq: i32,
    /// Two-phase sparse-attention dispatch (phase-1 score + phase-2 attend).
    pub sparse_attn: bool,
    /// TurboFlash split-K FlashAttention kernel on K8V4 storage.
    pub turbo_flash: bool,
    /// TurboFlash "lock-on" mode: stop maintaining the bf16 K/V mirror once
    /// the head-major flash buffers are seeded. No effect unless
    /// `turbo_flash` is also set.
    pub turbo_flash_lock: bool,
    /// Minimum `kv_seq` for `turbo_flash` to dispatch.
    pub turbo_flash_min_kv_seq: i32,
    /// PlanarQuant single-pass flash-decode kernel on PlanarK storage.
    pub planar_flash_decode: bool,
    /// Fused FWHT + affine-quantize kernel on the rot_k codec families,
    /// replacing the matmul-against-a-Hadamard-matrix path.
    pub rot_k_fused: bool,
}

impl Default for DispatchPolicy {
    fn default() -> Self {
        Self {
            fused_qk: false,
            fused_qk_min_kv_seq: DEFAULT_FUSED_QK_MIN_KV_SEQ,
            sparse_attn: false,
            turbo_flash: false,
            turbo_flash_lock: false,
            turbo_flash_min_kv_seq: DEFAULT_TURBO_FLASH_MIN_KV_SEQ,
            planar_flash_decode: false,
            rot_k_fused: false,
        }
    }
}

impl DispatchPolicy {
    /// Resolve a policy from the environment.
    ///
    /// Each boolean is on only for the exact value `"1"`; every threshold
    /// falls back to its default when unset, and a threshold that fails to
    /// parse warns once and falls back rather than aborting the run.
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            fused_qk: env_flag("RMLX_FUSED_QK"),
            fused_qk_min_kv_seq: env_threshold(
                "RMLX_FUSED_QK_MIN",
                DEFAULT_FUSED_QK_MIN_KV_SEQ,
                false,
            ),
            sparse_attn: env_flag("RMLX_SPARSE_ATTN"),
            turbo_flash: env_flag("RMLX_TURBO_FLASH"),
            turbo_flash_lock: env_flag("RMLX_TURBO_FLASH_LOCK"),
            turbo_flash_min_kv_seq: env_threshold(
                "RMLX_TURBO_FLASH_MIN",
                DEFAULT_TURBO_FLASH_MIN_KV_SEQ,
                true,
            ),
            planar_flash_decode: env_flag("RMLX_PLANAR_FLASH_DECODE"),
            rot_k_fused: env_flag("RMLX_ROT_K_FUSED"),
        }
    }
}

/// `true` when `name` is set to exactly `"1"`.
fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1"))
}

/// Parse an integer threshold, falling back to `default` when unset or
/// unparseable. `clamp_negative` reproduces the `RMLX_TURBO_FLASH_MIN`
/// contract, where a negative value means "no threshold" rather than an
/// error.
fn env_threshold(name: &str, default: i32, clamp_negative: bool) -> i32 {
    let Ok(raw) = std::env::var(name) else {
        return default;
    };
    match raw.parse::<i32>() {
        Ok(n) if clamp_negative && n < 0 => {
            tracing::warn!(var = name, value = n, "negative threshold clamped to 0");
            0
        }
        Ok(n) => n,
        Err(e) => {
            tracing::warn!(
                var = name,
                value = %raw,
                error = %e,
                default,
                "threshold parse failed, using default"
            );
            default
        }
    }
}

/// Process-wide default, handed to every KV cache that is not given an
/// explicit policy.
///
/// Seeded from the environment on first read and replaceable at any point via
/// [`set_dispatch_policy`] — it is a default, not a latch. A cache captures
/// the value at construction, so replacing it never disturbs caches that are
/// already live.
static PROCESS_DEFAULT: LazyLock<RwLock<DispatchPolicy>> =
    LazyLock::new(|| RwLock::new(DispatchPolicy::from_env()));

/// The current process-wide default policy.
#[must_use]
pub fn dispatch_policy() -> DispatchPolicy {
    *PROCESS_DEFAULT
        .read()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Replace the process-wide default policy.
///
/// Called by `rmlx-cli` once, after flag parsing and before any cache is
/// built. Benches and tests call it between arms to build the next arm's
/// caches under a different policy; caches built earlier keep the policy they
/// captured.
pub fn set_dispatch_policy(policy: DispatchPolicy) {
    *PROCESS_DEFAULT
        .write()
        .unwrap_or_else(PoisonError::into_inner) = policy;
}

#[cfg(test)]
#[path = "dispatch_policy_tests.rs"]
mod dispatch_policy_tests;
