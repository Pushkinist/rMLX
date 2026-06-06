//! OCP E2M1 fp4 decoder for mxfp4 / nvfp4 elements.
//!
//! E2M1: 1 sign + 2 exponent (bias 1) + 1 mantissa bit = 4 bits.
//!
//! OCP-specified value set (16 entries):
//! {+0, +0.5, +1, +1.5, +2, +3, +4, +6, -0, -0.5, -1, -1.5, -2, -3, -4, -6}
//!
//! No NaN, no Infinity in E2M1.

// ── E2M1 table ───────────────────────────────────────────────────────────────

/// All 16 E2M1 fp4 values, indexed by the 4-bit nibble (0..=15).
///
/// Bit layout: nibble = `s(1) | e(2) | m(1)`.
/// - s = nibble >> 3
/// - e = (nibble >> 1) & 0x3
/// - m = nibble & 0x1
///
/// Exponent bias = 1.
///
/// Normal (`e >= 1`): `(-1)^s * 2^(e-1) * (1 + m/2)`
/// Subnormal (`e == 0`): `(-1)^s * 2^0 * (m/2) = (-1)^s * m/2`
/// (OCP subnormal base exponent = 2^(1-bias) = 2^0 = 1; no leading 1)
///
/// Derivation for each nibble:
///
/// |nibble| s | e | m | formula | value |
/// |------|---|---|---|-------------------------------|--------|
/// | 0x0 | 0 | 0 | 0 | +2^0 * (0/2) | +0.0 |
/// | 0x1 | 0 | 0 | 1 | +2^0 * (1/2) | +0.5 |
/// | 0x2 | 0 | 1 | 0 | +2^(1-1) * (1+0/2) = +1*1 | +1.0 |
/// | 0x3 | 0 | 1 | 1 | +2^(1-1) * (1+1/2) = +1*1.5 | +1.5 |
/// | 0x4 | 0 | 2 | 0 | +2^(2-1) * (1+0/2) = +2*1 | +2.0 |
/// | 0x5 | 0 | 2 | 1 | +2^(2-1) * (1+1/2) = +2*1.5 | +3.0 |
/// | 0x6 | 0 | 3 | 0 | +2^(3-1) * (1+0/2) = +4*1 | +4.0 |
/// | 0x7 | 0 | 3 | 1 | +2^(3-1) * (1+1/2) = +4*1.5 | +6.0 |
/// | 0x8 | 1 | 0 | 0 | -2^0 * (0/2) | -0.0 |
/// | 0x9 | 1 | 0 | 1 | -2^0 * (1/2) | -0.5 |
/// | 0xA | 1 | 1 | 0 | -2^(1-1) * (1+0/2) = -1*1 | -1.0 |
/// | 0xB | 1 | 1 | 1 | -2^(1-1) * (1+1/2) = -1*1.5 | -1.5 |
/// | 0xC | 1 | 2 | 0 | -2^(2-1) * (1+0/2) = -2*1 | -2.0 |
/// | 0xD | 1 | 2 | 1 | -2^(2-1) * (1+1/2) = -2*1.5 | -3.0 |
/// | 0xE | 1 | 3 | 0 | -2^(3-1) * (1+0/2) = -4*1 | -4.0 |
/// | 0xF | 1 | 3 | 1 | -2^(3-1) * (1+1/2) = -4*1.5 | -6.0 |
const E2M1_TABLE: [f32; 16] = [
    0.0_f32,  // 0x0: +0
    0.5_f32,  // 0x1: +0.5
    1.0_f32,  // 0x2: +1.0
    1.5_f32,  // 0x3: +1.5
    2.0_f32,  // 0x4: +2.0
    3.0_f32,  // 0x5: +3.0
    4.0_f32,  // 0x6: +4.0
    6.0_f32,  // 0x7: +6.0
    -0.0_f32, // 0x8: -0
    -0.5_f32, // 0x9: -0.5
    -1.0_f32, // 0xA: -1.0
    -1.5_f32, // 0xB: -1.5
    -2.0_f32, // 0xC: -2.0
    -3.0_f32, // 0xD: -3.0
    -4.0_f32, // 0xE: -4.0
    -6.0_f32, // 0xF: -6.0
];

/// Decode one E2M1 nibble (low 4 bits of `nibble`) into f32.
///
/// The upper 4 bits of `nibble` are ignored — callers are responsible for
/// masking when unpacking two nibbles from a byte.
///
/// Uses a 16-entry constant table — single array lookup, branch-free.
#[inline]
pub fn e2m1_decode(nibble: u8) -> f32 {
    // nibble & 0xF is always in [0, 15]; E2M1_TABLE has exactly 16 entries.
    #[allow(
        clippy::indexing_slicing,
        reason = "nibble & 0xF ∈ [0,15]; E2M1_TABLE is a 16-entry constant array — index always in bounds"
    )]
    E2M1_TABLE[(nibble & 0xF) as usize]
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "fp4_tests.rs"]
mod tests;
