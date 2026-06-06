//! OCP E4M3 and Blackwell UE4M3 / E8M0 scalar decoders.
//!
//! These are pure-Rust CPU reference implementations. They serve as
//! correctness oracles for future Metal kernels.
//!
//! ## E8M0 (scale for mxfp4 / mxfp8)
//!
//! 8-bit unsigned exponent, bias 127. No mantissa, no sign.
//! Decodes to `2^(e - 127)`. Special value `0xFF` is NaN.
//!
//! ## E4M3 (element dtype for mxfp8)
//!
//! OCP FP8: 1 sign + 4 exponent (bias 7) + 3 mantissa bits.
//! Special: `0x7F` and `0xFF` (s=0/1, e=0xF, m=0x7) are NaN. No infinities.
//!
//! ## UE4M3 (scale dtype for nvfp4 — Blackwell correct)
//!
//! Unsigned: 4 exponent (bias 7) + 3 mantissa bits, no sign bit.
//! The 8-bit byte layout is `e[3:0] = byte[6:3]`, `m[2:0] = byte[2:0]`;
//! bit 7 is part of the exponent in a full-range unsigned interpretation.
//! Effective exponent field = `(byte >> 3) & 0xF` (4 bits), mantissa = `byte & 0x7`.
//!
//! No NaN reservations. Max finite value ≈ 480 (e=15, m=7).
//!
//! ### Why MLX uses signed E4M3 instead (the bug)
//!
//! ml-explore/mlx#2962: MLX's nvfp4 path decodes the scale byte as signed
//! E4M3 (treating bit 7 as sign). This gives 137× less dynamic range than
//! the Blackwell spec's UE4M3. rMLX defaults to UE4M3 and offers a
//! `compat_mlx_signed_scale` opt-in toggle that selects the broken signed-E4M3
//! path so users can load MLX-produced nvfp4 snapshots with the same scaling
//! behaviour the Python stack used.

// ── E8M0 ─────────────────────────────────────────────────────────────────────

/// Decode one E8M0 byte into f32.
///
/// E8M0 is exponent-only with bias 127: value = `2^(e - 127)`.
/// Special: `0xFF` → NaN (caller should treat this as a broken snapshot).
/// `0x00` → `2^-127` (denormal-flush-to-zero by convention; not
/// rejected here — log at debug level in higher layers).
#[inline]
pub fn e8m0_decode(byte: u8) -> f32 {
    if byte == 0xFF {
        return f32::NAN;
    }
    // pow2(e - 127): encode the exponent directly in f32 bits.
    // f32 layout: s(1) | exp(8, bias 127) | mantissa(23).
    // A pure power of two has mantissa = 0, sign = 0, exponent = (e - 127) + 127 = e.
    // Special cases e == 0 (0x00): unbiased exp = -127, which is subnormal in f32.
    // We handle it via ldexpf logic to stay correct.
    let unbiased: i32 = i32::from(byte) - 127;
    // 2^unbiased as f32: use bit manipulation.
    // f32 biased exponent = unbiased + 127. Valid f32 range: [1..254] for normals.
    let biased_exp = unbiased + 127; // = byte as i32
    if biased_exp <= 0 {
        // Subnormal or zero f32. Use f32::from_bits with mantissa trick.
        // 2^(unbiased) where unbiased <= -127: encode as subnormal or flush to zero.
        // Smallest normal f32 = 2^-126. For unbiased = -127, result = 2^-127.
        // f32 subnormal: sign=0, exp=0, mantissa = 2^(23 + unbiased + 126) = 2^(unbiased + 149).
        let shift: i32 = unbiased + 149; // = byte as i32 - 127 + 149 = byte as i32 + 22
        if shift < 0 {
            return 0.0_f32; // underflow to zero
        }
        return f32::from_bits(1u32 << (shift as u32));
    }
    if biased_exp >= 255 {
        return f32::INFINITY; // overflow (shouldn't happen for byte < 0xFF)
    }
    f32::from_bits((biased_exp as u32) << 23)
}

// ── E4M3 ─────────────────────────────────────────────────────────────────────

/// Decode one OCP E4M3 byte (signed) into f32.
///
/// Layout: `s(1) | e(4) | m(3)`. Exponent bias = 7.
///
/// Normal (`1 ≤ e ≤ 14`): `(-1)^s * 2^(e - 7) * (1 + m/8)`
/// Subnormal (`e == 0`): `(-1)^s * 2^-6 * (m/8)`
/// NaN: `e == 0xF, m == 0x7` → bytes `0x7F` and `0xFF`. Returns `f32::NAN`.
/// Infinity is not part of OCP E4M3; `e == 0xF, m != 0x7` is finite.
///
/// Note: OCP E4M3 has a specific "FN" (Finite + NaN) profile. Only the two
/// bit patterns where all exponent and mantissa bits are 1 are NaN; all others
/// including `e=15, m<7` are finite normal values.
#[inline]
pub fn e4m3_decode(byte: u8) -> f32 {
    let s = (byte >> 7) & 0x1; // sign bit
    let e = (byte >> 3) & 0xF; // 4 exponent bits
    let m = byte & 0x7; // 3 mantissa bits

    // NaN: e=0xF, m=0x7 (bytes 0x7F and 0xFF)
    if e == 0xF && m == 0x7 {
        return f32::NAN;
    }

    let sign = if s == 1 { -1.0_f32 } else { 1.0_f32 };

    if e == 0 {
        // Subnormal: (-1)^s * 2^-6 * (m / 8)
        // 2^-6 = 1/64
        sign * f32::from(m) / 8.0 / 64.0
    } else {
        // Normal: (-1)^s * 2^(e - 7) * (1 + m/8)
        let mantissa_f = 1.0_f32 + f32::from(m) / 8.0;
        let exp_val = e8m0_decode(e.wrapping_add(127u8 - 7u8));
        sign * exp_val * mantissa_f
    }
}

// ── UE4M3 ────────────────────────────────────────────────────────────────────

/// Decode one Blackwell UE4M3 byte (unsigned) into f32.
///
/// Layout: `e[3:0] = byte[6:3]` (4 bits), `m[2:0] = byte[2:0]` (3 bits).
/// Bit 7 of the byte is the MSB of the exponent field, making the effective
/// exponent 4 bits wide: `e = (byte >> 3) & 0xF`.
///
/// Exponent bias = 7. No sign bit. Always non-negative.
///
/// Normal (`1 ≤ e ≤ 15`): `2^(e - 7) * (1 + m/8)`. Range up to 480.
/// Subnormal (`e == 0`): `2^-6 * (m/8)`.
/// No NaN reservations.
///
/// **MLX bug contrast**: MLX interprets this byte as signed E4M3 (treating
/// bit 7 as sign), giving a range of roughly ±240 vs the unsigned 0..480.
/// The `compat_mlx_signed_scale` toggle in `MxParams` selects the signed
/// interpretation for parity with MLX-produced snapshots.
#[inline]
pub fn ue4m3_decode(byte: u8) -> f32 {
    let e = (byte >> 3) & 0xF; // 4 exponent bits (bits 6:3)
    let m = byte & 0x7; // 3 mantissa bits (bits 2:0)

    if e == 0 {
        // Subnormal: 2^-6 * (m/8)
        f32::from(m) / 8.0 / 64.0
    } else {
        // Normal: 2^(e - 7) * (1 + m/8)
        let mantissa_f = 1.0_f32 + f32::from(m) / 8.0;
        let exp_val = e8m0_decode(e.wrapping_add(127u8 - 7u8));
        exp_val * mantissa_f
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "fp8_tests.rs"]
mod tests;
