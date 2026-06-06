//! Minimal bf16 ↔ f32 helpers for dequant.
//!
//! BF16 layout: the upper 16 bits of an IEEE 754 f32.
//! Encoding: sign(1) | exponent(8) | mantissa(7)
//!
//! MLX always stores tensors little-endian on Apple Silicon.

use rmlx_core::Error;
use rmlx_core::Result;

/// Decode one bf16 element from 2 little-endian bytes.
///
/// BF16 is the upper 16 bits of an f32 → shift left by 16.
#[inline]
pub fn bf16_to_f32(le_bytes: [u8; 2]) -> f32 {
    let bits = u16::from_le_bytes(le_bytes);
    f32::from_bits(u32::from(bits) << 16)
}

/// Decode `le` (a sequence of 2-byte LE bf16 values) into `out`.
///
/// Requires `le.len() == 2 * out.len()`.
pub fn bf16_decode_into(le: &[u8], out: &mut [f32]) -> Result<()> {
    if le.len() != 2 * out.len() {
        return Err(Error::Quant(format!(
            "bf16_decode_into: input length {} != 2 * output length {}",
            le.len(),
            out.len()
        )));
    }
    for (i, slot) in out.iter_mut().enumerate() {
        // le.len() == 2 * out.len() is checked above; i < out.len(), so 2*i+1 < le.len().
        #[allow(
            clippy::indexing_slicing,
            reason = "2*i+1 < le.len(): length guard above asserts le.len()==2*out.len(); i < out.len() from iter_mut"
        )]
        let lo = le[2 * i];
        #[allow(
            clippy::indexing_slicing,
            reason = "2*i+1 < le.len(): length guard above asserts le.len()==2*out.len(); i < out.len() from iter_mut"
        )]
        let hi = le[2 * i + 1];
        *slot = bf16_to_f32([lo, hi]);
    }
    Ok(())
}

#[cfg(test)]
#[path = "bf16_tests.rs"]
mod tests;
