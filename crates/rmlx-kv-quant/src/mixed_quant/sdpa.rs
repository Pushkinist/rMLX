// unsafe_code: mlx-rs Array zero-copy view
#![allow(unsafe_code)]

//! SDPA helper for the Mixed / RotK KV modes.
//!
//! - [`mixed_quantized_sdpa`]: fused quantized SDPA for K8/V4 mixed-precision.

use crate::rot_k_msl::rot_k_fwht_rotate_gpu;
use rmlx_core::error::{Error, Result};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::{
    add, expand_dims, multiply, quantized_matmul, scalar_f32, softmax_precise, Array, Device,
};

use super::state::MixedTuple;

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

    // queries *= scale. Canonical guarded form: `astype` to the same dtype is
    // a no-op in MLX, so this is identical to branching on it, and the guard is
    // on the statement `check-no-scalar-f32-leak` reads.
    let scale_arr = scalar_f32(scale).astype(queries.dtype(), device)?;
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

    // The full softmax distribution goes into the V matmul. An earlier revision
    // truncated probabilities below 1e-6 to zero here, on the theory that a
    // zeroed row costs nothing downstream; `quantized_matmul` is opaque and
    // reads every V row regardless, so the truncation bought no bandwidth while
    // dropping attention mass that it never renormalised — an error that grows
    // with context, which is the regime the codec exists for.
    let probs = softmax_precise(&scores_masked, -1, device)?;

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

#[cfg(test)]
#[path = "sdpa_tests.rs"]
mod rot_k_tests;
