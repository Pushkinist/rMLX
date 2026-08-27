//! MapleRMSNorm: fp32 weight multiply (`maple.py` MapleRMSNorm).
//!
//! The scale is upcast once at load. A per-step `weight.astype(f32)` was 97
//! extra casts per decode token (2 layer norms + 2 QK norms × 24 layers +
//! final). `compile_shapeless` of this path was A/B'd at ~123 vs ~190 TPS
//! and is not used — compiled.apply per norm lost to the unfused graph.

use rmlx_core::error::Result;
use rmlx_mlx::{rms_norm, Array, Device, Dtype};

/// RMSNorm with the weight multiply in float32 (`MapleRMSNorm` in maple.py).
///
/// `mx.fast.rms_norm` on bf16 rounds the normalized activation before the
/// weight multiply (~1% per element vs the training reference). Casting the
/// activation to f32 against a **load-time** f32 weight, then casting back,
/// matches the reference.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed layer struct — weight + eps is the full MapleRMSNorm contract"
)]
#[allow(missing_debug_implementations)]
pub(super) struct MapleRmsNorm {
    /// Learned scale, stored f32 so the per-step path does not recast it.
    pub(super) weight: Array,
    /// Variance epsilon (1e-6 in the snapshot).
    pub(super) eps: f32,
}

impl MapleRmsNorm {
    /// `weight` is `[dims]` (hidden for layer norms, `head_dim` for Q/K norms).
    /// Upcast to f32 once at load.
    pub(super) fn new(weight: Array, eps: f32, device: Device) -> Result<Self> {
        let weight = if weight.dtype() == Dtype::F32 {
            weight
        } else {
            weight.astype(Dtype::F32, device)?
        };
        Ok(Self { weight, eps })
    }

    /// `rms_norm(x.f32, weight.f32, eps).astype(x.dtype)`.
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let x_f32 = x.astype(Dtype::F32, device)?;
        let out = rms_norm(&x_f32, Some(&self.weight), self.eps, device)?;
        out.astype(x.dtype(), device)
    }
}

#[cfg(test)]
#[path = "rms_tests.rs"]
mod rms_tests;
