//! Plain-GQA attention for the Qwen3-VL text decoder with **3D interleaved
//! M-RoPE** applied via precomputed per-token `cos`/`sin` tables.
//!
//! Unlike [`crate::qwen3_5_moe::attention::FullAttention`] (gated, scalar-offset
//! `rope()` kernel), Qwen3-VL needs per-token 3D rotary positions that the
//! scalar-offset kernel cannot express. We therefore apply the rotation in the
//! `q*cos + rotate_half(q)*sin` form (faithful to
//! `language.py::apply_multimodal_rotary_pos_emb`), with the `cos`/`sin` tables
//! built host-side by [`super::mrope::build_interleaved_mrope_tables`] for the
//! tokens in this forward chunk.
//!
//! No attention gate (plain Qwen3-MoE), per-head q_norm/k_norm RMSNorm.

use rmlx_core::error::Result;
use rmlx_mlx::{add, concatenate, multiply, negative, Array, Device};

use crate::layers::RmsNorm;
use rmlx_kv_quant::KvCache;

use super::layers::Linear;

#[allow(missing_debug_implementations)]
pub(super) struct Attention {
    pub(super) q_proj: Linear,
    pub(super) k_proj: Linear,
    pub(super) v_proj: Linear,
    pub(super) o_proj: Linear,
    pub(super) q_norm: RmsNorm,
    pub(super) k_norm: RmsNorm,
    pub(super) n_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) scale: f32,
}

/// `rotate_half(x)` = `concat([-x2, x1])` over the last axis, where `x1`/`x2`
/// are the first / second halves.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn rotate_half(x: &Array, device: Device) -> Result<Array> {
    let s = x.shape();
    let last = s.len() - 1;
    let d = s[last];
    let half = d / 2;
    let mut start = vec![0i32; s.len()];
    let mut stop: Vec<i32> = s.clone();
    let stride = vec![1i32; s.len()];
    // x1 = x[..., :half]
    stop[last] = half;
    let x1 = x.slice(&start, &stop, &stride, device)?;
    // x2 = x[..., half:]
    start[last] = half;
    stop[last] = d;
    let x2 = x.slice(&start, &stop, &stride, device)?;
    let neg_x2 = negative(&x2, device)?;
    concatenate(&[&neg_x2, &x1], last as i32, device)
}

/// Apply M-RoPE: `out = x*cos + rotate_half(x)*sin`.
/// `x`: `[B, H, S, head_dim]`; `cos`/`sin`: `[1, 1, S, head_dim]`.
fn apply_rotary(x: &Array, cos: &Array, sin: &Array, device: Device) -> Result<Array> {
    let xc = multiply(x, cos, device)?;
    let rh = rotate_half(x, device)?;
    let rs = multiply(&rh, sin, device)?;
    add(&xc, &rs, device)
}

impl Attention {
    /// `x`: `[B, S, hidden]`. `cos`/`sin`: `[1, 1, S, head_dim]` already shaped
    /// for broadcast over `[B, H, S, head_dim]`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        cos: &Array,
        sin: &Array,
        cache: Option<&mut KvCache>,
        prebuilt_mask: Option<&Array>,
        mask_mode: &str,
        device: Device,
    ) -> Result<Array> {
        let s = x.shape();
        let batch = s[0];
        let seq = s[1];
        let hd = self.head_dim as i32;
        let nh = self.n_heads as i32;
        let nkv = self.n_kv_heads as i32;

        let queries = self.q_proj.forward(x, device)?;
        let keys = self.k_proj.forward(x, device)?;
        let values = self.v_proj.forward(x, device)?;

        // [B, S, H, D]
        let queries = queries.reshape(&[batch, seq, nh, hd], device)?;
        let keys = keys.reshape(&[batch, seq, nkv, hd], device)?;
        let values = values.reshape(&[batch, seq, nkv, hd], device)?;

        // Per-head q_norm / k_norm over head_dim.
        let queries = self.q_norm.forward(&queries, device)?;
        let keys = self.k_norm.forward(&keys, device)?;

        // -> [B, H, S, D]
        let queries = queries.transpose(&[0, 2, 1, 3], device)?;
        let keys = keys.transpose(&[0, 2, 1, 3], device)?;
        let values = values.transpose(&[0, 2, 1, 3], device)?;

        // 3D interleaved M-RoPE.
        let queries = apply_rotary(&queries, cos, sin, device)?;
        let keys = apply_rotary(&keys, cos, sin, device)?;

        // Cast a prebuilt mask to Q dtype if needed (mirrors qwen3_5_moe).
        let additive_mask: Option<Array> = if mask_mode == "array" {
            let q_dtype = queries.dtype();
            match prebuilt_mask {
                Some(m) if m.dtype() != q_dtype => Some(m.astype(q_dtype, device)?),
                _ => None,
            }
        } else {
            None
        };
        let mask_ref = if mask_mode == "array" {
            additive_mask.as_ref().or(prebuilt_mask)
        } else {
            None
        };

        let attn = if let Some(c) = cache {
            c.update_and_sdpa(
                &queries, &keys, &values, self.scale, mask_mode, mask_ref, device,
            )?
        } else {
            rmlx_mlx::scaled_dot_product_attention(
                &queries, &keys, &values, self.scale, mask_mode, mask_ref, device,
            )?
        };

        let attn = attn.transpose(&[0, 2, 1, 3], device)?;
        let attn = attn.reshape(&[batch, seq, nh * hd], device)?;
        self.o_proj.forward(&attn, device)
    }
}
