// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Local layer primitives for qwen3_vl_moe: quantized `Linear` (plain /
//! affine-quantized) with batched expert dispatch (`gather_forward`), and a
//! quantized `Embedding`.
//!
//! These mirror the proven primitives in [`crate::qwen3_5_moe::layers`] — the
//! Qwen3-VL text decoder uses the identical MLX 4-bit affine quant layout
//! (`weight` U32 + `scales`/`biases` BF16, `switch_mlp.*_proj` pre-split MoE
//! experts). Kept local rather than shared because the qwen3_5_moe versions are
//! `pub(super)` and carry Qwen3-Next-specific PARO/gate machinery not needed
//! here.

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

#[allow(missing_debug_implementations)]
pub(super) enum Linear {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
}

impl Linear {
    /// `x`: `[.., in]` -> `[.., out]`.
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Linear::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::quantized_matmul(
                x,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                true,
                device,
            ),
        }
    }

    /// Batched expert dispatch via `gather_qmm`.
    /// `x`: `[n_tokens, 1, 1, hidden]`; `rhs_indices`: `[n_tokens, top_k]`.
    /// Returns `[n_tokens, top_k, 1, out_dim]`. Mirrors
    /// [`crate::qwen3_5_moe::layers::Linear::gather_forward`].
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn gather_forward(
        &self,
        x: &Array,
        rhs_indices: &Array,
        device: Device,
    ) -> Result<Array> {
        match self {
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => rmlx_mlx::gather_qmm(
                x,
                weight,
                scales,
                biases.as_ref(),
                None,
                rhs_indices,
                *group_size,
                *bits,
                mode,
                false,
                device,
            ),
            // Slow fallback for unquantized experts (not hit in practice — all
            // MoE expert projections are 4-bit affine in the target snapshot).
            Linear::Plain { weight } => {
                let s = x.shape();
                let n_tokens = s[0];
                let tk = rhs_indices.shape()[1];
                let hidden_in = *s.last().unwrap_or(&0);
                let x_flat = x.reshape(&[n_tokens, hidden_in], device)?;
                let rhs_flat = rhs_indices.reshape(&[n_tokens * tk], device)?;
                let w_sel = weight.take(&rhs_flat, 0, device)?;
                let mut tok_data = vec![0i32; (n_tokens * tk) as usize];
                for i in 0..(n_tokens as usize) {
                    for j in 0..(tk as usize) {
                        tok_data[i * tk as usize + j] = i as i32;
                    }
                }
                let tok_bytes = unsafe {
                    std::slice::from_raw_parts(tok_data.as_ptr().cast::<u8>(), tok_data.len() * 4)
                };
                let tok_idx = Array::from_bytes(tok_bytes, &[n_tokens * tk], Dtype::I32)?;
                let x_sel = x_flat.take(&tok_idx, 0, device)?;
                let out = rmlx_mlx::matmul(&x_sel, &w_sel.transpose(&[0, 2, 1], device)?, device)?;
                let out_dim = out.shape()[1];
                out.reshape(&[n_tokens, tk, 1, out_dim], device)
            }
        }
    }
}

#[allow(missing_debug_implementations)]
pub(super) enum Embedding {
    Plain {
        weight: Array,
    },
    Quantized {
        weight: Array,
        scales: Array,
        biases: Option<Array>,
        group_size: i32,
        bits: i32,
        mode: String,
    },
}

impl Embedding {
    pub(super) fn forward(&self, ids: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => weight.take(ids, 0, device),
            Embedding::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => embed_lookup(
                ids,
                weight,
                scales,
                biases.as_ref(),
                *group_size,
                *bits,
                mode,
                device,
            ),
        }
    }
}

/// Quantized embedding lookup: dequantize the selected rows via an identity
/// `quantized_matmul` on CPU, then move to `device`. Mirrors
/// [`crate::qwen3_5_moe::layers::embed_lookup`].
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn embed_lookup(
    ids: &Array,
    weight: &Array,
    scales: &Array,
    biases: Option<&Array>,
    group_size: i32,
    bits: i32,
    mode: &str,
    device: Device,
) -> Result<Array> {
    let cpu = Device::Cpu;
    let w_rows = weight.take(ids, 0, cpu)?;
    let s_rows = scales.take(ids, 0, cpu)?;
    let b_rows = biases.map(|b| b.take(ids, 0, cpu)).transpose()?;

    let seq = ids.dim(0)? as usize;
    let mut eye_data = vec![0.0_f32; seq * seq];
    for i in 0..seq {
        eye_data[i * seq + i] = 1.0;
    }
    let eye_bytes =
        unsafe { std::slice::from_raw_parts(eye_data.as_ptr().cast::<u8>(), eye_data.len() * 4) };
    let eye = Array::from_bytes(eye_bytes, &[seq as i32, seq as i32], Dtype::F32)?;
    let eye_bf16 = eye.astype(Dtype::Bf16, cpu)?;

    let result = rmlx_mlx::quantized_matmul(
        &eye_bf16,
        &w_rows,
        &s_rows,
        b_rows.as_ref(),
        group_size,
        bits,
        mode,
        false,
        cpu,
    )?;
    if device == cpu {
        Ok(result)
    } else {
        result.astype(result.dtype(), device)
    }
}
