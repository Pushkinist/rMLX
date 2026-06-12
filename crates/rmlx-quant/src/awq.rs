//! AWQ → MLX weight-conversion byte math (for PARO checkpoints).
//!
//! Pure byte-level codecs: unpack AutoAWQ-packed INT4 words, re-pack into MLX
//! sequential order, convert AWQ qzeros/scales into MLX affine scales/biases,
//! and F16↔F32 bit-pattern helpers used throughout. These functions take and
//! return raw byte buffers plus scalars (no MLX `Array`), so they live in the
//! weight-quant crate; the `Array`-side assembly that consumes them stays in
//! `rmlx-models`.

use rmlx_core::error::{Error, Result};

/// Unpack 8 × 4-bit nibbles from one I32 word using the AutoAWQ interleave order.
///
/// AutoAWQ CUDA packing stores output elements in nibble positions with a column
/// interleave: output elements `[0, 2, 4, 6, 1, 3, 5, 7]` are stored at nibble
/// positions `[0, 1, 2, 3, 4, 5, 6, 7]`.
///
/// To recover sequential elements `[0, 1, 2, 3, 4, 5, 6, 7]`:
/// element 0 ← nibble position 0
/// element 1 ← nibble position 4
/// element 2 ← nibble position 1
/// element 3 ← nibble position 5
/// element 4 ← nibble position 2
/// element 5 ← nibble position 6
/// element 6 ← nibble position 3
/// element 7 ← nibble position 7
///
/// Reference: `_unpack_and_reorder` in `paroquant/inference/backends/mlx/load.py`.
#[inline]
pub fn awq_unpack_word(word: u32) -> [u8; 8] {
    let raw: [u8; 8] = [
        (word & 0xF) as u8,
        ((word >> 4) & 0xF) as u8,
        ((word >> 8) & 0xF) as u8,
        ((word >> 12) & 0xF) as u8,
        ((word >> 16) & 0xF) as u8,
        ((word >> 20) & 0xF) as u8,
        ((word >> 24) & 0xF) as u8,
        ((word >> 28) & 0xF) as u8,
    ];
    // Return in sequential output order [0, 1, 2, 3, 4, 5, 6, 7]:
    // out[i] = raw[AWQ_REORDER[i]] where AWQ_REORDER = [0,4,1,5,2,6,3,7]
    [
        raw[0], raw[4], raw[1], raw[5], // out positions 0,1,2,3
        raw[2], raw[6], raw[3], raw[7], // out positions 4,5,6,7
    ]
}

/// Re-pack 8 nibbles (u8, values 0..15) into one I32 in MLX sequential order.
///
/// MLX sequential packing: nibble for element e occupies bits [e*4 .. e*4+3].
#[inline]
pub fn mlx_pack_word(nibbles: &[u8; 8]) -> u32 {
    u32::from(nibbles[0])
        | (u32::from(nibbles[1]) << 4)
        | (u32::from(nibbles[2]) << 8)
        | (u32::from(nibbles[3]) << 12)
        | (u32::from(nibbles[4]) << 16)
        | (u32::from(nibbles[5]) << 20)
        | (u32::from(nibbles[6]) << 24)
        | (u32::from(nibbles[7]) << 28)
}

/// Convert AutoAWQ-packed I32 qweight → MLX-sequential I32 weight, transposed.
///
/// AWQ layout: `[in_features, out_features*bits/32]` — packed over the output dimension.
/// MLX `quantized_matmul` expects: `[out_features, in_features*bits/32]` — packed over input.
///
/// Steps: unpack all nibbles → `[in, out]` element matrix → transpose → `[out, in]`
/// → repack in MLX sequential order → `[out, in*bits/32]`.
///
/// Returns the converted weight bytes with shape `[out_features, in_features*bits/32]`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn convert_awq_qweight(
    qweight_bytes: &[u8],
    in_features: usize,
    out_features: usize,
    bits: usize,
) -> Result<Vec<u8>> {
    let words_per_in_row = out_features * bits / 32; // AWQ: each input row has this many I32s
    let expected = in_features * words_per_in_row * 4;
    if qweight_bytes.len() != expected {
        return Err(Error::Loader(format!(
            "convert_awq_qweight: expected {expected} bytes, got {}",
            qweight_bytes.len()
        )));
    }

    // 1. Unpack all nibbles into a flat [in, out] u8 matrix.
    let mut unpacked = vec![0u8; in_features * out_features];
    for i in 0..in_features {
        for j in 0..words_per_in_row {
            let offset = (i * words_per_in_row + j) * 4;
            let word = u32::from_le_bytes([
                qweight_bytes[offset],
                qweight_bytes[offset + 1],
                qweight_bytes[offset + 2],
                qweight_bytes[offset + 3],
            ]);
            let nibbles = awq_unpack_word(word);
            let out_base = j * 8;
            for k in 0..8usize {
                if out_base + k < out_features {
                    unpacked[i * out_features + out_base + k] = nibbles[k];
                }
            }
        }
    }

    // 2. Transpose to [out, in] and repack in MLX sequential order → [out, in*bits/32].
    let words_per_out_row = in_features * bits / 32; // MLX: each output row has this many I32s
    let out_bytes_len = out_features * words_per_out_row * 4;
    let mut out = vec![0u8; out_bytes_len];
    for o in 0..out_features {
        for j in 0..words_per_out_row {
            let in_base = j * 8;
            let mut nibbles = [0u8; 8];
            for k in 0..8usize {
                if in_base + k < in_features {
                    nibbles[k] = unpacked[(in_base + k) * out_features + o];
                }
            }
            let repacked = mlx_pack_word(&nibbles);
            let offset = (o * words_per_out_row + j) * 4;
            out[offset..offset + 4].copy_from_slice(&repacked.to_le_bytes());
        }
    }
    Ok(out)
}

/// Convert AWQ qzeros + F16 scales → MLX affine scales and biases, transposed.
///
/// AWQ layout: `qzeros [num_groups, out_features*bits/32]`, `scales [num_groups, out_features]`.
/// MLX `quantized_matmul` expects: `scales [out_features, num_groups]`, `biases [out_features, num_groups]`.
///
/// Output:
/// - `scales_out`: F16 `[out_features, num_groups]` — transposed AWQ scales.
/// - `biases_out`: F16 `[out_features, num_groups]` — `biases = -scales * zeros`.
///
/// Reference: `_convert_awq_linear` in load.py.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn convert_awq_qzeros_to_biases(
    qzeros_bytes: &[u8],
    scales_bytes: &[u8],
    num_groups: usize,
    out_features: usize,
    bits: usize,
) -> Result<(Vec<u8>, Vec<u8>)> {
    let words_per_row = out_features * bits / 32;
    let expected_z = num_groups * words_per_row * 4;
    let expected_s = num_groups * out_features * 2; // F16
    if qzeros_bytes.len() != expected_z {
        return Err(Error::Loader(format!(
            "convert_awq_qzeros: expected {expected_z} qzeros bytes, got {}",
            qzeros_bytes.len()
        )));
    }
    if scales_bytes.len() != expected_s {
        return Err(Error::Loader(format!(
            "convert_awq_qzeros: expected {expected_s} scales bytes, got {}",
            scales_bytes.len()
        )));
    }

    // Unpack zeros into u8 nibbles: [num_groups, out_features].
    let mut zeros = vec![0u8; num_groups * out_features];
    for g in 0..num_groups {
        for j in 0..words_per_row {
            let offset = (g * words_per_row + j) * 4;
            let word = u32::from_le_bytes([
                qzeros_bytes[offset],
                qzeros_bytes[offset + 1],
                qzeros_bytes[offset + 2],
                qzeros_bytes[offset + 3],
            ]);
            let nibbles = awq_unpack_word(word);
            let out_base = j * 8;
            for k in 0..8usize {
                if out_base + k < out_features {
                    zeros[g * out_features + out_base + k] = nibbles[k];
                }
            }
        }
    }

    // Transpose and compute:
    // scales_out[o, g] = scales[g, o]
    // biases_out[o, g] = -scales[g, o] * zeros[g, o]
    // Output shape: [out_features, num_groups] F16.
    let n = out_features * num_groups * 2;
    let mut scales_out = vec![0u8; n];
    let mut biases_out = vec![0u8; n];
    for o in 0..out_features {
        for g in 0..num_groups {
            let src_si = (g * out_features + o) * 2;
            let dst_si = (o * num_groups + g) * 2;
            let scale_bits = u16::from_le_bytes([scales_bytes[src_si], scales_bytes[src_si + 1]]);
            let scale_f32 = f16_bits_to_f32(scale_bits);
            let zero_f32 = f32::from(zeros[g * out_features + o]);
            let bias_bits = f32_to_f16_bits(-scale_f32 * zero_f32);
            scales_out[dst_si..dst_si + 2].copy_from_slice(&scale_bits.to_le_bytes());
            biases_out[dst_si..dst_si + 2].copy_from_slice(&bias_bits.to_le_bytes());
        }
    }

    Ok((scales_out, biases_out))
}

/// F16 bit-pattern to f32.
#[inline]
pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x3FF);

    if exp == 0 {
        // Subnormal or zero.
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: scale by 2^(1-15-10) = 2^(-24). The sign must be applied
        // as a factor of ±1.0 — multiplying by f32::from_bits(sign) (±0.0)
        // collapses every subnormal to a signed zero.
        let sign_f = if bits >> 15 != 0 { -1.0_f32 } else { 1.0_f32 };
        return sign_f * (mantissa as f32) * (1.0 / 16777216.0);
    } else if exp == 31 {
        // Inf or NaN.
        return f32::from_bits(sign | 0x7F800000 | (mantissa << 13));
    }

    let f32_exp = (exp + 127 - 15) << 23;
    let f32_mantissa = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_mantissa)
}

/// f32 to F16 bit-pattern, round-to-nearest-even (pub for gemma4 PARO loader).
///
/// Previous implementation used truncation ("no rounding for simplicity"),
/// which accumulated 1-ULP bias errors across groups. For a 5120-element
/// sum this caused a ~0.06 systematic output error — outside F16 tolerance.
/// This implementation matches numpy's round-to-nearest-even.
#[inline]
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 31) as u16) << 15;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mantissa = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf or NaN.
        return sign | 0x7C00 | ((mantissa >> 13) as u16);
    }
    let f16_exp = exp - 127 + 15;
    if f16_exp >= 31 {
        return sign | 0x7C00; // Overflow → Inf.
    }
    if f16_exp <= 0 {
        // Subnormal or underflow.
        if f16_exp < -10 {
            return sign;
        }
        // Shift into subnormal position: add implicit leading 1, then shift.
        let shift = (1 - f16_exp) as u32;
        let full = (mantissa | 0x800000) >> shift;
        // Round-to-nearest-even: inspect the dropped bits.
        let round_bit = (mantissa | 0x800000) >> (shift - 1) & 1;
        let sticky = if shift > 1 {
            (mantissa | 0x800000) & ((1u32 << (shift - 1)) - 1)
        } else {
            0
        };
        let half = round_bit != 0;
        let above_half = sticky != 0;
        let rounded = if half && (above_half || (full & 1) != 0) {
            full + 1
        } else {
            full
        };
        return sign | (rounded >> 13) as u16;
    }
    // Normal range: 13 bits of mantissa are dropped (F32 has 23, F16 has 10).
    let m_hi = mantissa >> 13; // kept bits
    let round_bit = (mantissa >> 12) & 1;
    let sticky = mantissa & 0xFFF;
    let half = round_bit != 0;
    let above_half = sticky != 0;
    let rounded_m = if half && (above_half || (m_hi & 1) != 0) {
        m_hi + 1
    } else {
        m_hi
    };
    // rounded_m may overflow 10 bits → carry into exponent.
    if rounded_m >= 0x400 {
        let new_exp = f16_exp as u16 + 1;
        if new_exp >= 31 {
            return sign | 0x7C00; // Overflow → Inf.
        }
        sign | (new_exp << 10) | (rounded_m & 0x3FF) as u16
    } else {
        sign | ((f16_exp as u16) << 10) | rounded_m as u16
    }
}

/// Quantize a F16 weight matrix `[rows, cols]` to MLX affine INT4 format.
///
/// Exact port of MLX's CPU `quantize()` function in
/// `mlx/backend/cpu/quantized.cpp`.
///
/// MLX algorithm per group:
/// 1. Find w_min, w_max in f32.
/// 2. mask = |w_min| > |w_max|
/// 3. scale = (w_max - w_min) / (2^bits - 1), negated unless mask
/// 4. edge = w_min if mask, else w_max
/// 5. q0 = round(edge / scale)
/// 6. If q0 != 0: refine scale = edge / q0, bias = edge (this is the key!)
/// 7. nibble = round((x - bias) / scale), clamped to [0, 15]
///
/// This refinement step makes the scale exact for the dominant edge, avoiding
/// 1-ULP rounding drift that would otherwise cause different dequant values.
///
/// Outputs:
/// weight_out: `[rows, cols * 4 / 32]` U32 — 8 nibbles per U32 (LSB first).
/// scales_out: `[rows, groups]` F16 — stored as F16 (cast from f32).
/// biases_out: `[rows, groups]` F16 — stored as F16 (cast from f32).
///
/// where `groups = cols / group_size`.
///
/// # Errors
///
/// Returns `Err` if `cols % group_size != 0` or `w_bytes` has wrong length.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn quantize_f16_affine_int4(
    w_bytes: &[u8],
    rows: usize,
    cols: usize,
    group_size: usize,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    if group_size == 0 || !cols.is_multiple_of(group_size) {
        return Err(Error::Quant(format!(
            "quantize_f16_affine_int4: cols={cols} not divisible by group_size={group_size}"
        )));
    }
    let expected_bytes = rows * cols * 2;
    if w_bytes.len() != expected_bytes {
        return Err(Error::Quant(format!(
            "quantize_f16_affine_int4: expected {expected_bytes} bytes for [{rows},{cols}] F16, got {}",
            w_bytes.len()
        )));
    }

    let num_groups = cols / group_size;
    let nibbles_per_word = 8usize; // 32 bits / 4 bits per nibble
    let words_per_row = cols / nibbles_per_word;
    let n_bins = 15.0_f32; // 2^4 - 1
    let eps = 1e-7_f32;

    let mut weight_out = vec![0u8; rows * words_per_row * 4];
    let mut scales_out = vec![0u8; rows * num_groups * 2];
    let mut biases_out = vec![0u8; rows * num_groups * 2];

    for row in 0..rows {
        for g in 0..num_groups {
            let col_start = g * group_size;

            // Step 1: find w_min, w_max in f32.
            let mut w_min = f32::INFINITY;
            let mut w_max = f32::NEG_INFINITY;
            for c in 0..group_size {
                let idx = (row * cols + col_start + c) * 2;
                let bits_val = u16::from_le_bytes([w_bytes[idx], w_bytes[idx + 1]]);
                let v = f16_bits_to_f32(bits_val);
                if v < w_min {
                    w_min = v;
                }
                if v > w_max {
                    w_max = v;
                }
            }

            // Step 2-3: mask and initial scale.
            let mask = w_min.abs() > w_max.abs();
            let raw_scale = (w_max - w_min) / n_bins;
            let mut scale = raw_scale.max(eps);
            if !mask {
                scale = -scale;
            }

            // Step 4-6: refine scale so dominant edge maps exactly.
            let edge = if mask { w_min } else { w_max };
            let q0 = (edge / scale).round();
            let mut bias = 0.0_f32;
            if q0 != 0.0 {
                scale = edge / q0;
                bias = edge;
            }

            // Store scale and bias as F16.
            let scale_f16 = f32_to_f16_bits(scale);
            let bias_f16 = f32_to_f16_bits(bias);
            let sg_off = (row * num_groups + g) * 2;
            scales_out[sg_off..sg_off + 2].copy_from_slice(&scale_f16.to_le_bytes());
            biases_out[sg_off..sg_off + 2].copy_from_slice(&bias_f16.to_le_bytes());

            // Step 7: quantize each element and pack 8 nibbles per U32.
            for c in 0..group_size {
                let col = col_start + c;
                let src_idx = (row * cols + col) * 2;
                let bits_val = u16::from_le_bytes([w_bytes[src_idx], w_bytes[src_idx + 1]]);
                let v = f16_bits_to_f32(bits_val);
                let nibble_f32 = ((v - bias) / scale).round();
                let nibble = (nibble_f32 as i32).clamp(0, n_bins as i32) as u32;

                let word_idx = col / nibbles_per_word;
                let nibble_pos = col % nibbles_per_word;
                let byte_off = (row * words_per_row + word_idx) * 4;
                let mut word = u32::from_le_bytes([
                    weight_out[byte_off],
                    weight_out[byte_off + 1],
                    weight_out[byte_off + 2],
                    weight_out[byte_off + 3],
                ]);
                word |= nibble << (nibble_pos * 4);
                weight_out[byte_off..byte_off + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
    }

    Ok((weight_out, scales_out, biases_out))
}

#[cfg(test)]
#[path = "awq_tests.rs"]
mod awq_tests;
