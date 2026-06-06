//! RMS normalisation layer.

use rmlx_core::error::Result;
use rmlx_mlx::{rms_norm, Array, Device};

/// RMSNorm with optional scale weight.
///
/// Uses the plain-gamma convention: rms_norm(x) * weight.
/// Pass `weight: None` for RMSNormNoScale layers (Gemma4 v_norm).
///
/// Gemma2/3 use weight+1 (shifted gamma). That is a per-arch wrapper,
/// not modeled here, to keep this type simple.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed layer struct — fields are the complete RMSNorm contract; adding a field requires updating all arch RmsNorm construction sites"
)]
#[allow(missing_debug_implementations)]
/// RMS normalisation layer (weight + epsilon).
pub struct RmsNorm {
    /// Optional learned scale weight (`None` = unscaled).
    pub weight: Option<Array>,
    /// Epsilon added to the variance denominator for numerical stability.
    pub eps: f32,
}

impl RmsNorm {
    /// Apply RMS normalization to `x`.
    pub fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        rms_norm(x, self.weight.as_ref(), self.eps, device)
    }
}
