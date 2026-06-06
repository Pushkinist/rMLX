// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! `RmsNormShifted` — RMSNorm with the Gemma2/3 "weight + 1.0" convention.
//!
//! Gemma3 stores RMSNorm gammas centred at 0.0 so the effective scale is
//! `1.0 + weight`. We materialise the shifted weight once at load time so the
//! forward call is identical to the plain-gamma path.
//!
//! Reference: `mlx-lm/mlx_lm/models/gemma3_text.py` `RMSNorm.__call__`:
//! ```text
//! return mx.fast.rms_norm(x, 1.0 + self.weight, self.eps)
//! ```
//!
//! Lifted verbatim from `crates/rmlx-models/src/gemma3/layers.rs`. The
//! gemma3 module currently keeps a private copy of this type (with `pub(super)`
//! fields) — Stage 1 of the runtime extraction adds this version alongside,
//! and migrates gemma3 to it. Other archs that adopt Gemma2/3-style norms in
//! the future can use this directly.

use rmlx_core::error::Result;
use rmlx_mlx::{add, Array, Device, Dtype};

/// RMSNorm layer using the Gemma2/3 `weight + 1.0` scale convention.
#[allow(missing_debug_implementations)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal layer implementation — fields are the complete RMSNorm+1 contract; adding a field requires updating from_weight and forward"
)]
pub struct RmsNormShifted {
    /// Stores `raw_weight + 1.0` so forward is identical to `rmlx_models::layers::RmsNorm`.
    pub shifted_weight: Array,
    /// Epsilon added to the RMS denominator for numerical stability.
    pub eps: f32,
}

impl RmsNormShifted {
    /// Build from a raw weight array. Computes `shifted = weight + 1.0` once.
    ///
    /// `weight` may be BF16 or F32; `1.0` is built as F32 then cast to match.
    pub fn from_weight(weight: &Array, eps: f32) -> Result<Self> {
        let shape = weight.shape();
        let n = shape.iter().map(|&d| d as usize).product::<usize>();
        let data: Vec<f32> = vec![1.0_f32; n];
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), n * 4) };
        let ones = Array::from_bytes(bytes, &shape, Dtype::F32)?;
        let ones_cast = ones.astype(weight.dtype(), Device::Cpu)?;
        let shifted_weight = add(&ones_cast, weight, Device::Cpu)?;
        Ok(Self {
            shifted_weight,
            eps,
        })
    }

    /// Apply shifted RMSNorm to `x` using the pre-computed `shifted_weight`.
    pub fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        rmlx_mlx::rms_norm(x, Some(&self.shifted_weight), self.eps, device)
    }
}

#[cfg(test)]
#[path = "rmsnorm_tests.rs"]
mod tests;
