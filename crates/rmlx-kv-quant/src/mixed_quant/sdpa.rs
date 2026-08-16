// unsafe_code: mlx-rs Array zero-copy view
#![allow(unsafe_code)]

//! SDPA helpers for Mixed and RotKTq4V KV modes.
//!
//! - [`mixed_quantized_sdpa`]: fused quantized SDPA for K8/V4 mixed-precision.
//! - [`rot_k_tq4v_sdpa`]: dequant-then-SDPA for RotKTq4V hybrid.

use crate::rot_k_msl::rot_k_fwht_rotate_gpu;
use rmlx_core::error::{Error, Result};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{
    add, dequantize, expand_dims, greater_equal, multiply, quantized_matmul, scalar_f32,
    scaled_dot_product_attention, softmax_precise, where_cond, Array, Device, Dtype,
};

use super::state::MixedTuple;

// ── Sparse-V threshold ────────────────────────────────────────────────────────

/// Threshold below which a softmax probability is treated as zero for V-row
/// dequant (sparse-V cheap path).
///
/// Hardcoded `1e-6` (`RMLX_SPARSE_V_THRESHOLD` env var removed in PASS 3).
/// Matches TheTom `experimental_decode_speed_tests` `TURBO_SPARSE_V` default.
#[inline]
fn sparse_v_threshold() -> f32 {
    1e-6_f32
}

/// Byte-for-byte port of `mixed_quantized_scaled_dot_product_attention`
/// (`mlx_lm/models/base.py:108-157`).
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn mixed_quantized_sdpa(
    queries: &Array,
    q_keys: &MixedTuple,
    q_values: &MixedTuple,
    scale: f32,
    additive_mask: Option<&Array>,
    k_group_size: i32,
    k_bits: i32,
    v_group_size: i32,
    v_bits: i32,
    k_rotation: Option<&Array>,
    device: Device,
    policy: DispatchPolicy,
) -> Result<Array> {
    // pre-rotate Q by the same R the K cache was rotated with, so the
    // rotations cancel in the score matmul: (Q Rt)(K Rt)t = Q Kt. K is stored
    // (and quantized) in the rotated basis and is never inverse-rotated.
    //
    // When the policy selects the fused path and D is in the supported FWHT
    // set, use the fused FWHT rotate kernel (O(D log D) vs O(D^2) matmul).
    // Falls back to the v1 rotate_last_axis matmul on error or unsupported D.
    let queries_owned;
    let queries: &Array = match k_rotation {
        Some(r) => {
            let d = *queries
                .shape()
                .last()
                .ok_or_else(|| Error::Mlx("mixed_quantized_sdpa: empty Q shape".into()))?
                as usize;
            let try_fused = policy.rot_k_fused && crate::rot_k_msl::is_supported_d(d);
            queries_owned = if try_fused {
                match rot_k_fwht_rotate_gpu(queries, device) {
                    Ok(q_rot) => q_rot,
                    Err(e) => {
                        tracing::warn!(
                            reason = %e,
                            "rot_k_fwht_rotate_gpu failed; falling back to v1 matmul Q rotation"
                        );
                        super::super::rot_k::rotate_last_axis(queries, r, device)?
                    }
                }
            } else {
                super::super::rot_k::rotate_last_axis(queries, r, device)?
            };
            &queries_owned
        }
        None => queries,
    };

    let q_shape = queries.shape();
    let b = q_shape[0];
    let n_q_heads = q_shape[1];
    let l = q_shape[2];
    let d = q_shape[3];

    let n_kv_heads = q_keys.codes.shape()[1];
    let n_repeats = n_q_heads / n_kv_heads;

    // queries *= scale
    let scale_arr_f32 = scalar_f32(scale);
    let scale_arr = if queries.dtype() == Dtype::F32 {
        scale_arr_f32
    } else {
        scale_arr_f32.astype(queries.dtype(), device)?
    };
    let queries_scaled = multiply(queries, &scale_arr, device)?;

    // GQA expansion: reshape queries to [B, n_kv_heads, n_repeats, L, D] and
    // add a new axis=-3 to each of q_keys / q_values for broadcast.
    let (queries_eff, k_eff, v_eff) = if n_repeats > 1 {
        let q_r = queries_scaled.reshape(&[b, n_kv_heads, n_repeats, l, d], device)?;
        let k_e = MixedTuple {
            codes: expand_dims(&q_keys.codes, -3, device)?,
            scales: expand_dims(&q_keys.scales, -3, device)?,
            biases: expand_dims(&q_keys.biases, -3, device)?,
        };
        let v_e = MixedTuple {
            codes: expand_dims(&q_values.codes, -3, device)?,
            scales: expand_dims(&q_values.scales, -3, device)?,
            biases: expand_dims(&q_values.biases, -3, device)?,
        };
        (q_r, k_e, v_e)
    } else {
        (queries_scaled, q_keys.try_clone()?, q_values.try_clone()?)
    };

    let scores = quantized_matmul(
        &queries_eff,
        &k_eff.codes,
        &k_eff.scales,
        Some(&k_eff.biases),
        k_group_size,
        k_bits,
        "affine",
        true,
        device,
    )?;

    let scores_masked = match additive_mask {
        Some(m) => {
            let mask = if m.dtype() == scores.dtype() {
                m.try_clone()?
            } else {
                m.astype(scores.dtype(), device)?
            };
            add(&scores, &mask, device)?
        }
        None => scores,
    };

    let probs_raw = softmax_precise(&scores_masked, -1, device)?;

    // Sparse-V cheap path — zero out softmax probs below threshold before the
    // V-row dequant. Rows with zero weight are skipped by `quantized_matmul`
    // without altering any non-zero rows.
    let probs = {
        let threshold = sparse_v_threshold();
        if threshold > 0.0 {
            let t_arr = scalar_f32(threshold);
            let t_arr = if probs_raw.dtype() == Dtype::F32 {
                t_arr
            } else {
                t_arr.astype(probs_raw.dtype(), device)?
            };
            let zeros_arr = scalar_f32(0.0);
            let zeros_arr = if probs_raw.dtype() == Dtype::F32 {
                zeros_arr
            } else {
                zeros_arr.astype(probs_raw.dtype(), device)?
            };
            // mask = probs >= threshold (bool/U8)
            let mask = greater_equal(&probs_raw, &t_arr, device)?;
            // probs_sparse = where(mask, probs, 0)
            where_cond(&mask, &probs_raw, &zeros_arr, device)?
        } else {
            probs_raw
        }
    };

    let out = quantized_matmul(
        &probs,
        &v_eff.codes,
        &v_eff.scales,
        Some(&v_eff.biases),
        v_group_size,
        v_bits,
        "affine",
        false,
        device,
    )?;

    if n_repeats > 1 {
        out.reshape(&[b, n_q_heads, l, d], device)
    } else {
        Ok(out)
    }
}

/// SDPA for the RotKTq4V hybrid variant.
///
/// K is stored as an MLX affine 3-tuple `(codes, scales, biases)` in the rotated
/// basis (same as `KvQuant::RotK`). V is stored as TurboFlash tq4 (rMLX MSL
/// symmetric 4-bit, `QuantV`).
///
/// Steps:
/// 1. Pre-rotate Q by the same `R` used for K (so scores cancel: `Q Kt`).
/// 2. Dequantize K from its affine 3-tuple back to bf16 (`mx.dequantize`).
/// 3. Dequantize V from TurboFlash to bf16 (`QuantV::dequantize_choice`).
/// 4. Run standard `scaled_dot_product_attention` on the bf16 K/V.
///
/// `k_rotation` must be `Some(&R)` (always, for RotKTq4V); pass `None` only
/// to disable Q rotation for testing purposes.
#[allow(clippy::too_many_arguments)]
pub fn rot_k_tq4v_sdpa(
    queries: &Array,
    k_tuple: &MixedTuple,
    v_bf16: &Array,
    scale: f32,
    additive_mask: Option<&Array>,
    k_group_size: i32,
    k_bits: i32,
    k_rotation: Option<&Array>,
    device: Device,
    policy: DispatchPolicy,
) -> Result<Array> {
    // 1. Pre-rotate Q (same logic as mixed_quantized_sdpa).
    let queries_owned;
    let queries: &Array = match k_rotation {
        Some(r) => {
            let d = *queries
                .shape()
                .last()
                .ok_or_else(|| Error::Mlx("rot_k_tq4v_sdpa: empty Q shape".into()))?
                as usize;
            let try_fused = policy.rot_k_fused && crate::rot_k_msl::is_supported_d(d);
            queries_owned = if try_fused {
                match rot_k_fwht_rotate_gpu(queries, device) {
                    Ok(q_rot) => q_rot,
                    Err(e) => {
                        tracing::warn!(
                            reason = %e,
                            "rot_k_fwht_rotate_gpu failed in rot_k_tq4v_sdpa; \
                             falling back to v1 matmul Q rotation"
                        );
                        super::super::rot_k::rotate_last_axis(queries, r, device)?
                    }
                }
            } else {
                super::super::rot_k::rotate_last_axis(queries, r, device)?
            };
            &queries_owned
        }
        None => queries,
    };

    // 2. Dequantize K from its affine 3-tuple back to bf16.
    // K was stored in the rotated basis (rotate_k_and_quantize); dequantizing
    // gives K_rot in bf16. Q is also rotated (step 1), so the score matmul
    // Q_rot * K_rot^T = Q * K^T (rotations cancel).
    let k_bf16 = dequantize(
        &k_tuple.codes,
        &k_tuple.scales,
        Some(&k_tuple.biases),
        k_group_size,
        k_bits,
        "affine",
        device,
    )?;

    // 3. V is already dequantized by the caller (QuantV::dequantize_choice).
    // `v_bf16` is [B, kv_h, T_seq, D].

    // 4. Standard scaled dot-product attention.
    let mask_mode = if additive_mask.is_some() {
        "array"
    } else {
        "causal"
    };
    scaled_dot_product_attention(
        queries,
        &k_bf16,
        v_bf16,
        scale,
        mask_mode,
        additive_mask,
        device,
    )
}

#[cfg(test)]
#[path = "sdpa_tests.rs"]
mod rot_k_tests;
