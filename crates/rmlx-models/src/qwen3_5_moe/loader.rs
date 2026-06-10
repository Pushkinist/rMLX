//! Loaders for Qwen3_5Moe (MoE standard) and Qwen3_5 (PARO dense) checkpoints.
//!
//! Provides two public entry points: [`load_from_path`] for standard MoE
//! checkpoints (bf16, affine-4bit, mxfp8) and [`load_from_path_paro`] for
//! PARO-quantized dense variants. Also contains AWQ → MLX weight conversion
//! helpers used during checkpoint loading.
//!
//! # Public API
//!
//! - [`load_from_path`] — load a standard Qwen3.5-MoE checkpoint.
//! - [`load_from_path_paro`] — load a PARO-quantized Qwen3.5 checkpoint.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref
)]
use std::path::Path;

use rmlx_core::error::{Error, Result};
use rmlx_loader::{load_config, load_shard_index, ShardSet};
use rmlx_mlx::Array;
use tracing::info;

use crate::layers::{resolve_quant, QuantParams};

use super::attention::FullAttention;
use super::config::Qwen3_5MoeConfig;
use super::decoder_layer::{AttnBlock, DecoderLayer, MlpBlock};
use super::gated_delta_net::GatedDeltaNet;
use super::layers::{Embedding, Linear, ParoRotation, RmsNorm};
use super::model::Qwen3_5MoeText;
use super::moe::{DenseMlp, SharedExpert, SparseMoeBlock, SwitchMlp};

// ---------------------------------------------------------------------------
// AWQ → MLX weight conversion helpers (for PARO checkpoints)
// ---------------------------------------------------------------------------

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
pub(crate) fn awq_unpack_word(word: u32) -> [u8; 8] {
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
pub(crate) fn mlx_pack_word(nibbles: &[u8; 8]) -> u32 {
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
pub(crate) fn convert_awq_qweight(
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
pub(crate) fn convert_awq_qzeros_to_biases(
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
pub(crate) fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = u32::from((bits >> 10) & 0x1F);
    let mantissa = u32::from(bits & 0x3FF);

    if exp == 0 {
        // Subnormal or zero.
        if mantissa == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal: scale by 2^(1-15-10) = 2^(-24).
        return f32::from_bits(sign) * (mantissa as f32) * (1.0 / 16777216.0);
    } else if exp == 31 {
        // Inf or NaN.
        return f32::from_bits(sign | 0x7F800000 | (mantissa << 13));
    }

    let f32_exp = (exp + 127 - 15) << 23;
    let f32_mantissa = mantissa << 13;
    f32::from_bits(sign | f32_exp | f32_mantissa)
}

/// f32 to F16 bit-pattern, round-to-nearest-even (pub(crate) for gemma4 PARO loader).
///
/// Previous implementation used truncation ("no rounding for simplicity"),
/// which accumulated 1-ULP bias errors across groups. For a 5120-element
/// sum this caused a ~0.06 systematic output error — outside F16 tolerance.
/// This implementation matches numpy's round-to-nearest-even.
#[inline]
pub(crate) fn f32_to_f16_bits(v: f32) -> u16 {
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
pub(crate) fn quantize_f16_affine_int4(
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

/// Load a PARO linear layer from raw tensor bytes.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn load_paro_linear(
    base: &str,
    qweight_bytes: &[u8],
    qweight_shape: &[usize],
    scales_bytes: &[u8],
    scales_shape: &[usize],
    qzeros_bytes: &[u8],
    theta_bytes: &[u8],
    theta_shape: &[usize],
    pairs_bytes: &[u8],
    channel_scales_bytes: &[u8],
    group_size: usize,
) -> Result<Linear> {
    if scales_shape.len() != 2 || qweight_shape.len() != 2 {
        return Err(Error::Loader(format!(
            "load_paro_linear '{base}': unexpected tensor rank"
        )));
    }

    let num_groups = scales_shape[0];
    let out_features = scales_shape[1];
    let in_features = qweight_shape[0];

    let mlx_weight_bytes = convert_awq_qweight(qweight_bytes, in_features, out_features, 4)?;
    let weight = Array::from_bytes(
        &mlx_weight_bytes,
        &[out_features as i32, (in_features * 4 / 32) as i32],
        rmlx_mlx::Dtype::U32,
    )?;

    let (scales_bytes_t, biases_bytes_t) =
        convert_awq_qzeros_to_biases(qzeros_bytes, scales_bytes, num_groups, out_features, 4)?;
    let scales = Array::from_bytes(
        &scales_bytes_t,
        &[out_features as i32, num_groups as i32],
        rmlx_mlx::Dtype::F16,
    )?;
    let biases = Array::from_bytes(
        &biases_bytes_t,
        &[out_features as i32, num_groups as i32],
        rmlx_mlx::Dtype::F16,
    )?;

    if theta_shape.len() != 2 {
        return Err(Error::Loader(format!(
            "load_paro_linear '{base}': theta shape unexpected: {theta_shape:?}"
        )));
    }
    let krot = theta_shape[0];
    let half_hidden = theta_shape[1];
    let hidden = half_hidden * 2;

    let n_theta = krot * half_hidden;
    if theta_bytes.len() != n_theta * 2 {
        return Err(Error::Loader(format!(
            "load_paro_linear '{base}': theta bytes length {} != expected {}",
            theta_bytes.len(),
            n_theta * 2
        )));
    }
    let mut cos_bytes = vec![0u8; n_theta * 2];
    let mut sin_bytes = vec![0u8; n_theta * 2];
    for i in 0..n_theta {
        let th_bits = u16::from_le_bytes([theta_bytes[i * 2], theta_bytes[i * 2 + 1]]);
        let th_f32 = f16_bits_to_f32(th_bits);
        let cos_f16 = f32_to_f16_bits(th_f32.cos());
        let sin_f16 = f32_to_f16_bits(th_f32.sin());
        cos_bytes[i * 2..i * 2 + 2].copy_from_slice(&cos_f16.to_le_bytes());
        sin_bytes[i * 2..i * 2 + 2].copy_from_slice(&sin_f16.to_le_bytes());
    }
    let cos_theta = Array::from_bytes(
        &cos_bytes,
        &[krot as i32, half_hidden as i32],
        rmlx_mlx::Dtype::F16,
    )?;
    let sin_theta = Array::from_bytes(
        &sin_bytes,
        &[krot as i32, half_hidden as i32],
        rmlx_mlx::Dtype::F16,
    )?;

    let packed = crate::paroquant_msl::pack_pairs_cpu(pairs_bytes, krot, hidden, group_size)?;
    let packed_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 4) };
    let packed_pairs = Array::from_bytes(
        packed_bytes,
        &[krot as i32, half_hidden as i32],
        rmlx_mlx::Dtype::I32,
    )?;

    let cs = Array::from_bytes(
        channel_scales_bytes,
        &[1i32, hidden as i32],
        rmlx_mlx::Dtype::F16,
    )?;

    Ok(Linear::Paro {
        rotation: ParoRotation {
            packed_pairs,
            cos_theta,
            sin_theta,
            channel_scales: cs,
            krot,
            group_size,
        },
        weight,
        scales,
        biases,
    })
}

// ---------------------------------------------------------------------------
// load_from_path — standard MoE checkpoint
// ---------------------------------------------------------------------------

/// Load a Qwen3_5Moe model from a snapshot directory.
///
/// Expects `config.architectures[0] == "Qwen3_5MoeForConditionalGeneration"`.
/// Tensor prefix: `language_model.model`.
pub fn load_from_path(model_dir: &Path) -> Result<Qwen3_5MoeText> {
    let cfg_raw = load_config(model_dir)?;

    let raw_json: serde_json::Value = {
        let path = model_dir.join("config.json");
        let data = std::fs::read(&path)
            .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;
        serde_json::from_slice(&data)
            .map_err(|e| Error::Loader(format!("malformed config.json: {e}")))?
    };

    let raw_quant = raw_json.get("quantization");
    let raw_text_config = raw_json.get("text_config");

    let cfg = Qwen3_5MoeConfig::from_model_config(&cfg_raw, raw_quant, raw_text_config)?;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        num_experts = cfg.num_experts,
        num_experts_per_tok = cfg.num_experts_per_tok,
        quant_mode = %cfg.quant_mode,
        quant_overrides = cfg.quant_overrides.len(),
        "Qwen3_5Moe: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

    let load_array = |name: &str| -> Result<Array> {
        for (_, handle) in shards.iter() {
            let st = handle.safetensors()?;
            if let Ok(t) = st.tensor(name) {
                let tv = rmlx_loader::TensorView {
                    name,
                    dtype: t.dtype(),
                    shape: t.shape().to_vec(),
                    bytes: t.data(),
                };
                return Array::from_safetensor_view(&tv);
            }
        }
        Err(Error::Loader(format!(
            "tensor '{name}' not found in any shard"
        )))
    };

    let has_tensor = |name: &str| -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    };

    let defaults = QuantParams::global(cfg.quant_group_size, cfg.quant_bits, &cfg.quant_mode);

    let load_linear = |base: &str| -> Result<Linear> {
        let w = load_array(&format!("{base}.weight"))?;
        let s_name = format!("{base}.scales");
        if has_tensor(&s_name) {
            let s = load_array(&s_name)?;
            let biases = if has_tensor(&format!("{base}.biases")) {
                Some(load_array(&format!("{base}.biases"))?)
            } else {
                None
            };
            // The shared resolver owns the `.biases`-sibling affine rule.
            let qp = resolve_quant(base, biases.is_some(), &defaults, &cfg.quant_overrides)?;
            Ok(Linear::Quantized {
                weight: w,
                scales: s,
                biases,
                group_size: qp.group_size,
                bits: qp.bits,
                mode: qp.mode,
            })
        } else {
            Ok(Linear::Plain { weight: w })
        }
    };

    let load_rms = |name: &str| -> Result<RmsNorm> {
        Ok(RmsNorm {
            weight: load_array(&format!("{name}.weight"))?,
            eps: cfg.rms_norm_eps,
        })
    };

    let pfx = "language_model.model";

    let embed_tokens = {
        let base = format!("{pfx}.embed_tokens");
        if has_tensor(&format!("{base}.scales")) {
            let w = load_array(&format!("{base}.weight"))?;
            let s = load_array(&format!("{base}.scales"))?;
            let biases = if has_tensor(&format!("{base}.biases")) {
                Some(load_array(&format!("{base}.biases"))?)
            } else {
                None
            };
            let qp = resolve_quant(&base, biases.is_some(), &defaults, &cfg.quant_overrides)?;
            Embedding::Quantized {
                weight: w,
                scales: s,
                biases,
                group_size: qp.group_size,
                bits: qp.bits,
                mode: qp.mode,
            }
        } else {
            Embedding::Plain {
                weight: load_array(&format!("{base}.weight"))?,
            }
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    let lm_head = if cfg.tie_word_embeddings {
        info!("Qwen3_5Moe: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let candidates = [
            "language_model.lm_head",
            "lm_head",
            &format!("{pfx}.lm_head"),
        ];
        let base = candidates
            .iter()
            .find(|b| has_tensor(&format!("{b}.weight")))
            .copied()
            .unwrap_or("language_model.lm_head");
        info!(%base, "Qwen3_5Moe: loading lm_head");
        Some(load_linear(base)?)
    };

    let attn_scale = (cfg.head_dim as f32).powf(-0.5);
    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let is_linear = (i + 1) % cfg.full_attention_interval != 0;

        let attn = if is_linear {
            let la = format!("{base}.linear_attn");
            let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;

            let a_log_raw = load_array(&format!("{la}.A_log"))?;
            let hv = cfg.linear_num_value_heads as i32;
            let a_log_3d = a_log_raw.reshape(&[1, 1, hv], rmlx_mlx::Device::Cpu)?;
            let a_log_f32 = a_log_3d.astype(rmlx_mlx::Dtype::F32, rmlx_mlx::Device::Cpu)?;
            let exp_a_log_f32 = rmlx_mlx::exp(&a_log_f32, rmlx_mlx::Device::Cpu)?;
            exp_a_log_f32.eval()?;

            let dt_bias_raw = load_array(&format!("{la}.dt_bias"))?;
            let dt_bias_3d = dt_bias_raw.reshape(&[1, 1, hv], rmlx_mlx::Device::Cpu)?;
            dt_bias_3d.eval()?;

            let inv_scale = (cfg.linear_key_head_dim as f32).powf(-0.5);
            let inv_scale_sq_arr = rmlx_mlx::scalar_f32(inv_scale * inv_scale)
                .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
            inv_scale_sq_arr.eval()?;
            let inv_scale_arr = rmlx_mlx::scalar_f32(inv_scale)
                .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
            inv_scale_arr.eval()?;

            AttnBlock::Linear(GatedDeltaNet {
                in_proj_qkv: load_linear(&format!("{la}.in_proj_qkv"))?,
                in_proj_z: load_linear(&format!("{la}.in_proj_z"))?,
                in_proj_b: load_linear(&format!("{la}.in_proj_b"))?,
                in_proj_a: load_linear(&format!("{la}.in_proj_a"))?,
                conv1d_weight: load_array(&format!("{la}.conv1d.weight"))?,
                norm_weight: load_array(&format!("{la}.norm.weight"))?,
                exp_a_log_f32,
                dt_bias_3d,
                inv_scale_sq_arr,
                inv_scale_arr,
                out_proj: load_linear(&format!("{la}.out_proj"))?,
                num_k_heads: cfg.linear_num_key_heads,
                num_v_heads: cfg.linear_num_value_heads,
                head_k_dim: cfg.linear_key_head_dim,
                head_v_dim: cfg.linear_value_head_dim,
                key_dim,
                value_dim,
                eps: cfg.rms_norm_eps,
            })
        } else {
            let sa = format!("{base}.self_attn");
            AttnBlock::Full(FullAttention {
                q_proj: load_linear(&format!("{sa}.q_proj"))?,
                k_proj: load_linear(&format!("{sa}.k_proj"))?,
                v_proj: load_linear(&format!("{sa}.v_proj"))?,
                o_proj: load_linear(&format!("{sa}.o_proj"))?,
                q_norm: load_rms(&format!("{sa}.q_norm"))?,
                k_norm: load_rms(&format!("{sa}.k_norm"))?,
                n_heads: cfg.num_attention_heads,
                n_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                scale: attn_scale,
                rope_theta: cfg.rope_theta,
                rope_dims: cfg.rope_dims,
            })
        };

        let m = format!("{base}.mlp");
        let mlp = MlpBlock::Moe(Box::new(SparseMoeBlock {
            gate: load_linear(&format!("{m}.gate"))?,
            switch_mlp: SwitchMlp {
                gate_proj: load_linear(&format!("{m}.switch_mlp.gate_proj"))?,
                up_proj: load_linear(&format!("{m}.switch_mlp.up_proj"))?,
                down_proj: load_linear(&format!("{m}.switch_mlp.down_proj"))?,
            },
            shared_expert: SharedExpert {
                gate_proj: load_linear(&format!("{m}.shared_expert.gate_proj"))?,
                up_proj: load_linear(&format!("{m}.shared_expert.up_proj"))?,
                down_proj: load_linear(&format!("{m}.shared_expert.down_proj"))?,
            },
            shared_expert_gate: load_linear(&format!("{m}.shared_expert_gate"))?,
            num_experts: cfg.num_experts,
            top_k: cfg.num_experts_per_tok,
            norm_topk_prob: cfg.norm_topk_prob,
        }));

        layers.push(DecoderLayer {
            input_layernorm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attention_layernorm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });
    }

    Ok(Qwen3_5MoeText {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        cached_hot_head: std::sync::OnceLock::new(),
    })
}

// ---------------------------------------------------------------------------
// load_from_path_paro — PARO dense checkpoint
// ---------------------------------------------------------------------------

/// Load a Qwen3_5 (dense, PARO) model from a snapshot directory.
///
/// Handles `Qwen3_5ForConditionalGeneration` with ParoQuant INT4 weights.
/// Tensor prefix: `model.language_model` (same as MoE variant).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn load_from_path_paro(model_dir: &Path) -> Result<Qwen3_5MoeText> {
    let cfg_raw = load_config(model_dir)?;

    let raw_json: serde_json::Value = {
        let path = model_dir.join("config.json");
        let data = std::fs::read(&path)
            .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;
        serde_json::from_slice(&data)
            .map_err(|e| Error::Loader(format!("malformed config.json: {e}")))?
    };

    let raw_text_config = raw_json.get("text_config");
    let raw_quant = None;

    let mut cfg = Qwen3_5MoeConfig::from_model_config(&cfg_raw, raw_quant, raw_text_config)
        .or_else(|_| {
            let mut tc = raw_text_config
                .and_then(|v| v.as_object())
                .map(|m| {
                    let mut map = serde_json::Map::new();
                    map.extend(m.clone());
                    map
                })
                .unwrap_or_default();
            let inter = tc
                .get("intermediate_size")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(17408);
            tc.entry("num_experts").or_insert(serde_json::json!(1));
            tc.entry("num_experts_per_tok")
                .or_insert(serde_json::json!(1));
            tc.entry("moe_intermediate_size")
                .or_insert(serde_json::json!(inter));
            tc.entry("shared_expert_intermediate_size")
                .or_insert(serde_json::json!(inter));
            tc.entry("norm_topk_prob")
                .or_insert(serde_json::json!(false));
            let patched = serde_json::Value::Object(tc);
            Qwen3_5MoeConfig::from_model_config(&cfg_raw, raw_quant, Some(&patched))
        })?;

    cfg.num_experts = 0;
    cfg.num_experts_per_tok = 1;

    let paro_qc = cfg_raw.quantization_config.as_ref().ok_or_else(|| {
        Error::Config("PARO loader: missing quantization_config in config.json".to_owned())
    })?;
    let paro_bits = paro_qc.bits.unwrap_or(4) as usize;
    let paro_group_size = paro_qc.group_size.unwrap_or(128) as usize;
    let paro_krot = paro_qc.krot.unwrap_or(8) as usize;

    info!(
        num_hidden_layers = cfg.num_hidden_layers,
        hidden_size = cfg.hidden_size,
        paro_bits,
        paro_group_size,
        paro_krot,
        "Qwen3_5 PARO: loading model"
    );

    let idx = load_shard_index(model_dir)?;
    let shards = ShardSet::open(model_dir, &idx)?;

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
    )]
    fn load_raw(
        shards: &ShardSet,
        name: &str,
    ) -> Result<(Vec<u8>, Vec<usize>, safetensors::Dtype)> {
        for (_, handle) in shards.iter() {
            let st = handle.safetensors()?;
            if let Ok(t) = st.tensor(name) {
                return Ok((t.data().to_vec(), t.shape().to_vec(), t.dtype()));
            }
        }
        Err(Error::Loader(format!(
            "tensor '{name}' not found in any shard"
        )))
    }

    let has_tensor = |name: &str| -> bool {
        shards
            .iter()
            .any(|(_, h)| h.safetensors().is_ok_and(|st| st.tensor(name).is_ok()))
    };

    let load_array = |name: &str| -> Result<Array> {
        let (bytes, shape, dtype) = load_raw(&shards, name)?;
        let shape_i32: Vec<i32> = shape.iter().map(|&d| d as i32).collect();
        let mlx_dtype = match dtype {
            safetensors::Dtype::F16 => rmlx_mlx::Dtype::F16,
            safetensors::Dtype::BF16 => rmlx_mlx::Dtype::Bf16,
            safetensors::Dtype::F32 => rmlx_mlx::Dtype::F32,
            safetensors::Dtype::I32 => rmlx_mlx::Dtype::I32,
            safetensors::Dtype::U32 => rmlx_mlx::Dtype::U32,
            other => {
                return Err(Error::Loader(format!(
                    "load_array '{name}': unsupported dtype {other:?}"
                )));
            }
        };
        Array::from_bytes(&bytes, &shape_i32, mlx_dtype)
    };

    let load_rms = |name: &str| -> Result<RmsNorm> {
        let wname = format!("{name}.weight");
        let (w_bytes, w_shape, _) = load_raw(&shards, &wname)?;
        let n = w_shape.iter().product::<usize>();
        let mut shifted = vec![0u8; n * 2];
        for i in 0..n {
            let bits = u16::from_le_bytes([w_bytes[i * 2], w_bytes[i * 2 + 1]]);
            let v = f16_bits_to_f32(bits) + 1.0_f32;
            let out_bits = f32_to_f16_bits(v);
            shifted[i * 2..i * 2 + 2].copy_from_slice(&out_bits.to_le_bytes());
        }
        let shape_i32: Vec<i32> = w_shape.iter().map(|&d| d as i32).collect();
        let w = Array::from_bytes(&shifted, &shape_i32, rmlx_mlx::Dtype::F16)?;
        Ok(RmsNorm {
            weight: w,
            eps: cfg.rms_norm_eps,
        })
    };

    let load_plain_linear = |base: &str| -> Result<Linear> {
        let w_name = format!("{base}.weight");
        let (w_bytes, w_shape, _) = load_raw(&shards, &w_name)?;
        let w_shape_i32: Vec<i32> = w_shape.iter().map(|&d| d as i32).collect();
        let w = Array::from_bytes(&w_bytes, &w_shape_i32, rmlx_mlx::Dtype::F16)?;
        Ok(Linear::Plain { weight: w })
    };

    let load_paro = |base: &str| -> Result<Linear> {
        let (qw_bytes, qw_shape, _) = load_raw(&shards, &format!("{base}.qweight"))?;
        let (sc_bytes, sc_shape, _) = load_raw(&shards, &format!("{base}.scales"))?;
        let (qz_bytes, _, _) = load_raw(&shards, &format!("{base}.qzeros"))?;
        let (th_bytes, th_shape, _) = load_raw(&shards, &format!("{base}.theta"))?;
        let (pa_bytes, _, _) = load_raw(&shards, &format!("{base}.pairs"))?;
        let (cs_bytes, _, _) = load_raw(&shards, &format!("{base}.channel_scales"))?;
        load_paro_linear(
            base,
            &qw_bytes,
            &qw_shape,
            &sc_bytes,
            &sc_shape,
            &qz_bytes,
            &th_bytes,
            &th_shape,
            &pa_bytes,
            &cs_bytes,
            paro_group_size,
        )
    };

    let load_auto_linear = |base: &str| -> Result<Linear> {
        if has_tensor(&format!("{base}.pairs")) {
            load_paro(base)
        } else {
            load_plain_linear(base)
        }
    };

    let pfx = "model.language_model";

    let embed_tokens = {
        let (w_bytes, w_shape, _) = load_raw(&shards, &format!("{pfx}.embed_tokens.weight"))?;
        if w_shape.len() != 2 {
            return Err(Error::Loader(format!(
                "embed_tokens.weight: expected 2-D, got shape {w_shape:?}"
            )));
        }
        let vocab = w_shape[0];
        let hidden = w_shape[1];
        let num_groups = hidden / paro_group_size;
        let (wq_bytes, sc_bytes, bi_bytes) =
            quantize_f16_affine_int4(&w_bytes, vocab, hidden, paro_group_size)?;
        let w = Array::from_bytes(
            &wq_bytes,
            &[vocab as i32, (hidden * paro_bits / 32) as i32],
            rmlx_mlx::Dtype::U32,
        )?;
        let s = Array::from_bytes(
            &sc_bytes,
            &[vocab as i32, num_groups as i32],
            rmlx_mlx::Dtype::F16,
        )?;
        let b = Array::from_bytes(
            &bi_bytes,
            &[vocab as i32, num_groups as i32],
            rmlx_mlx::Dtype::F16,
        )?;
        Embedding::Quantized {
            weight: w,
            scales: s,
            biases: Some(b),
            group_size: paro_group_size as i32,
            bits: paro_bits as i32,
            mode: "affine".to_owned(),
        }
    };

    let final_norm = load_rms(&format!("{pfx}.norm"))?;

    let lm_head = if cfg.tie_word_embeddings {
        info!("Qwen3_5 PARO: tie_word_embeddings=true, using embed_tokens as lm_head");
        None
    } else {
        let candidates = ["lm_head", &format!("{pfx}.lm_head")];
        let base = candidates
            .iter()
            .find(|b| has_tensor(&format!("{b}.weight")))
            .copied()
            .unwrap_or("lm_head");
        info!(%base, "Qwen3_5 PARO: loading lm_head (INT4 quantized to match Python loader)");
        let (lm_bytes, lm_shape, _) = load_raw(&shards, &format!("{base}.weight"))?;
        let lm_vocab = lm_shape[0];
        let lm_hidden = lm_shape[1];
        let lm_groups = lm_hidden / paro_group_size;
        let (lm_wq, lm_sc, lm_bi) =
            quantize_f16_affine_int4(&lm_bytes, lm_vocab, lm_hidden, paro_group_size)?;
        let lm_w = Array::from_bytes(
            &lm_wq,
            &[lm_vocab as i32, (lm_hidden * paro_bits / 32) as i32],
            rmlx_mlx::Dtype::U32,
        )?;
        let lm_s = Array::from_bytes(
            &lm_sc,
            &[lm_vocab as i32, lm_groups as i32],
            rmlx_mlx::Dtype::F16,
        )?;
        let lm_b = Array::from_bytes(
            &lm_bi,
            &[lm_vocab as i32, lm_groups as i32],
            rmlx_mlx::Dtype::F16,
        )?;
        Some(Linear::Quantized {
            weight: lm_w,
            scales: lm_s,
            biases: Some(lm_b),
            group_size: paro_group_size as i32,
            bits: paro_bits as i32,
            mode: "affine".to_owned(),
        })
    };

    let attn_scale = (cfg.head_dim as f32).powf(-0.5);
    let _ = paro_krot;

    let mut layers = Vec::with_capacity(cfg.num_hidden_layers);

    for i in 0..cfg.num_hidden_layers {
        let base = format!("{pfx}.layers.{i}");
        let is_linear = (i + 1) % cfg.full_attention_interval != 0;

        let attn = if is_linear {
            let la = format!("{base}.linear_attn");
            let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
            let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;

            let a_log_raw = load_array(&format!("{la}.A_log"))?;
            let hv = cfg.linear_num_value_heads as i32;
            let a_log_3d = a_log_raw.reshape(&[1, 1, hv], rmlx_mlx::Device::Cpu)?;
            let a_log_f32 = a_log_3d.astype(rmlx_mlx::Dtype::F32, rmlx_mlx::Device::Cpu)?;
            let exp_a_log_f32 = rmlx_mlx::exp(&a_log_f32, rmlx_mlx::Device::Cpu)?;
            exp_a_log_f32.eval()?;

            let dt_bias_raw = load_array(&format!("{la}.dt_bias"))?;
            let dt_bias_3d = dt_bias_raw.reshape(&[1, 1, hv], rmlx_mlx::Device::Cpu)?;
            dt_bias_3d.eval()?;

            let inv_scale = (cfg.linear_key_head_dim as f32).powf(-0.5);
            let inv_scale_sq_arr = rmlx_mlx::scalar_f32(inv_scale * inv_scale)
                .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
            inv_scale_sq_arr.eval()?;
            let inv_scale_arr = rmlx_mlx::scalar_f32(inv_scale)
                .astype(rmlx_mlx::Dtype::Bf16, rmlx_mlx::Device::Cpu)?;
            inv_scale_arr.eval()?;

            AttnBlock::Linear(GatedDeltaNet {
                in_proj_qkv: load_auto_linear(&format!("{la}.in_proj_qkv"))?,
                in_proj_z: load_auto_linear(&format!("{la}.in_proj_z"))?,
                in_proj_b: load_plain_linear(&format!("{la}.in_proj_b"))?,
                in_proj_a: load_plain_linear(&format!("{la}.in_proj_a"))?,
                conv1d_weight: {
                    let w = load_array(&format!("{la}.conv1d.weight"))?;
                    let s = w.shape();
                    if s.len() == 3 && s[1] < s[2] {
                        w.transpose(&[0, 2, 1], rmlx_mlx::Device::Cpu)?
                    } else {
                        w
                    }
                },
                norm_weight: load_array(&format!("{la}.norm.weight"))?,
                exp_a_log_f32,
                dt_bias_3d,
                inv_scale_sq_arr,
                inv_scale_arr,
                out_proj: load_auto_linear(&format!("{la}.out_proj"))?,
                num_k_heads: cfg.linear_num_key_heads,
                num_v_heads: cfg.linear_num_value_heads,
                head_k_dim: cfg.linear_key_head_dim,
                head_v_dim: cfg.linear_value_head_dim,
                key_dim,
                value_dim,
                eps: cfg.rms_norm_eps,
            })
        } else {
            let sa = format!("{base}.self_attn");
            AttnBlock::Full(FullAttention {
                q_proj: load_auto_linear(&format!("{sa}.q_proj"))?,
                k_proj: load_auto_linear(&format!("{sa}.k_proj"))?,
                v_proj: load_auto_linear(&format!("{sa}.v_proj"))?,
                o_proj: load_auto_linear(&format!("{sa}.o_proj"))?,
                q_norm: load_rms(&format!("{sa}.q_norm"))?,
                k_norm: load_rms(&format!("{sa}.k_norm"))?,
                n_heads: cfg.num_attention_heads,
                n_kv_heads: cfg.num_key_value_heads,
                head_dim: cfg.head_dim,
                scale: attn_scale,
                rope_theta: cfg.rope_theta,
                rope_dims: cfg.rope_dims,
            })
        };

        let m = format!("{base}.mlp");
        let mlp = MlpBlock::Dense(Box::new(DenseMlp {
            gate_proj: load_auto_linear(&format!("{m}.gate_proj"))?,
            up_proj: load_auto_linear(&format!("{m}.up_proj"))?,
            down_proj: load_auto_linear(&format!("{m}.down_proj"))?,
        }));

        layers.push(DecoderLayer {
            input_layernorm: load_rms(&format!("{base}.input_layernorm"))?,
            post_attention_layernorm: load_rms(&format!("{base}.post_attention_layernorm"))?,
            attn,
            mlp,
        });
    }

    Ok(Qwen3_5MoeText {
        cfg,
        embed_tokens,
        layers,
        final_norm,
        lm_head,
        cached_hot_head: std::sync::OnceLock::new(),
    })
}
