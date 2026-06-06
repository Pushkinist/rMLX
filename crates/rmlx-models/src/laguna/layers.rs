//! Laguna-local layer types: Linear, Embedding, RmsNorm, DenseMlp.

// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(clippy::implicit_clone, clippy::struct_field_names)]
use rmlx_core::error::Result;
use rmlx_mlx::{gather_qmm, multiply, rms_norm, silu, Array, Device, Dtype};

// ---------------------------------------------------------------------------
// Local Linear + Embedding with biases support
// ---------------------------------------------------------------------------

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

    /// Batched expert dispatch via gather_qmm.
    ///
    /// Follows the mlx-lm Python pattern (switch_layers.py / QuantizedSwitchLinear.__call__):
    /// - `x` is already pre-expanded: `[..., 1, hidden]` (caller adds a 1-dim before hidden)
    /// - `rhs_indices`: `[n_tokens, top_k]` (expert indices per token)
    /// - Output: `[n_tokens, top_k, 1, intermediate]`
    ///   Caller is responsible for squeezing the `1` dim if needed.
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
            Linear::Plain { weight } => {
                // Plain fallback: [n_tokens, top_k] indices, x already expanded.
                // Flatten, gather weight + token rows, matmul, then reshape.
                let s = x.shape(); // [..., 1, hidden]
                let nd = s.len();
                let n_batch: i32 = s[..nd - 2].iter().product();
                let tk = rhs_indices.shape()[rhs_indices.shape().len() - 1];
                let idx_flat = rhs_indices.reshape(&[n_batch * tk], device)?;
                let w_sel = weight.take(&idx_flat, 0, device)?; // [n*tk, out, in]
                let x_flat = x.reshape(&[n_batch, s[nd - 1]], device)?;
                let mut tok_data = vec![0i32; (n_batch * tk) as usize];
                for i in 0..n_batch as usize {
                    for j in 0..tk as usize {
                        tok_data[i * tk as usize + j] = i as i32;
                    }
                }
                let tok_bytes = unsafe {
                    std::slice::from_raw_parts(tok_data.as_ptr().cast::<u8>(), tok_data.len() * 4)
                };
                let tok_idx = Array::from_bytes(tok_bytes, &[n_batch * tk], Dtype::I32)?;
                let x_sel = x_flat.take(&tok_idx, 0, device)?;
                let out_flat =
                    rmlx_mlx::matmul(&x_sel, &w_sel.transpose(&[0, 2, 1], device)?, device)?;
                let rhs_s = rhs_indices.shape();
                let mut out_s: Vec<i32> = rhs_s.to_vec();
                out_s.push(1);
                out_s.push(out_flat.shape()[1]);
                out_flat.reshape(&out_s, device)
            }
            Linear::Quantized {
                weight,
                scales,
                biases,
                group_size,
                bits,
                mode,
            } => {
                // No lhs_indices: MLX uses identity lhs gather.
                // Output shape: rhs_indices.shape + [x.shape(-2), w_outer]
                gather_qmm(
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
                )
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
            } => laguna_embed_lookup(
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

    pub(super) fn as_linear(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Embedding::Plain { weight } => {
                rmlx_mlx::matmul(x, &weight.transpose(&[1, 0], device)?, device)
            }
            Embedding::Quantized {
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
}

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn laguna_embed_lookup(
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
    let weight_rows = weight.take(ids, 0, cpu)?;
    let scales_rows = scales.take(ids, 0, cpu)?;
    let biases_rows = biases.map(|b| b.take(ids, 0, cpu)).transpose()?;

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
        &weight_rows,
        &scales_rows,
        biases_rows.as_ref(),
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

// ---------------------------------------------------------------------------
// RmsNorm
// ---------------------------------------------------------------------------

pub(super) struct RmsNorm {
    pub(super) weight: Array,
    pub(super) eps: f32,
}

impl RmsNorm {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        rms_norm(x, Some(&self.weight), self.eps, device)
    }
}

// ---------------------------------------------------------------------------
// DenseMlp (layer 0)
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct DenseMlp {
    pub(super) gate_proj: Linear,
    pub(super) up_proj: Linear,
    pub(super) down_proj: Linear,
}

impl DenseMlp {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let gate = self.gate_proj.forward(x, device)?;
        let gate = silu(&gate, device)?;
        let up = self.up_proj.forward(x, device)?;
        let gated = multiply(&gate, &up, device)?;
        self.down_proj.forward(&gated, device)
    }
}
