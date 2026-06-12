//! Smoke-probe types re-exported from the shared decode loop, plus byte-level logit helpers.

use rmlx_mlx::Dtype;

pub use crate::decode_loop::{ProbeStep, SmokeVerdict};

// ---------------------------------------------------------------------------
// Byte-level logit helpers
// ---------------------------------------------------------------------------

/// Count NaN values in a byte buffer of floats (F32 or Bf16).
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub(super) fn count_nan_in_bytes(bytes: &[u8], dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .filter(|c| f32::from_le_bytes((*c).try_into().unwrap()).is_nan())
            .count(),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .filter(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                let f32_bits = u32::from(raw) << 16;
                f32::from_bits(f32_bits).is_nan()
            })
            .count(),
        _ => 0,
    }
}

/// Compute max(|logit|) from a byte buffer. Returns 0.0 on empty or unknown dtype.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub(super) fn max_abs_from_bytes(bytes: &[u8], dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes((*c).try_into().unwrap()).abs())
            .fold(0.0_f32, f32::max),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                let f32_bits = u32::from(raw) << 16;
                f32::from_bits(f32_bits).abs()
            })
            .fold(0.0_f32, f32::max),
        _ => 0.0,
    }
}
