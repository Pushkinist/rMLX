//! Byte-level helpers for inspecting logit tensors.
//!
//! Extracted from the four per-arch copies in `qwen2.rs`, `qwen3.rs`,
//! `gemma3/generate.rs`, and `laguna/generate.rs`. Bit-for-bit identical
//! to those copies; `gemma4/generate.rs` keeps its own `pub(super)` versions
//! that other archs already import.
//!
//! These are deliberately tiny pure functions — extracting them is mainly
//! about removing copy-paste, not perf.

use rmlx_mlx::Dtype;

/// Count NaN values in a byte buffer of floats (F32 or Bf16).
///
/// Returns 0 for unsupported dtypes (matches per-arch behaviour).
#[allow(
    clippy::unwrap_used,
    reason = "try_into on a fixed-size chunk from chunks_exact — size is guaranteed by the chunk iterator; infallible"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "returning 0 for unrecognised dtypes is the correct and intentional fallback matching per-arch behaviour"
)]
pub fn count_nan_in_bytes(bytes: &[u8], dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .filter(|c| f32::from_le_bytes((*c).try_into().unwrap()).is_nan())
            .count(),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .filter(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                f32::from_bits(u32::from(raw) << 16).is_nan()
            })
            .count(),
        _ => 0,
    }
}

/// Compute `max(|logit|)` from a byte buffer. Returns `0.0` on empty or
/// unsupported dtype.
#[allow(
    clippy::unwrap_used,
    reason = "try_into on a fixed-size chunk from chunks_exact — size is guaranteed by the chunk iterator; infallible"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "returning 0.0 for unrecognised dtypes is the correct and intentional fallback matching per-arch behaviour"
)]
pub fn max_abs_from_bytes(bytes: &[u8], dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes((*c).try_into().unwrap()).abs())
            .fold(0.0_f32, f32::max),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                f32::from_bits(u32::from(raw) << 16).abs()
            })
            .fold(0.0_f32, f32::max),
        _ => 0.0,
    }
}

#[cfg(test)]
#[path = "probe_tests.rs"]
mod tests;
