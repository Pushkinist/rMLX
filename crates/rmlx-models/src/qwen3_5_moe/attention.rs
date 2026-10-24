//! Full-Attention (Qwen3Next style).
//!
//! q_proj output = n_heads * head_dim * 2; first half = queries, second half = gate.
//! gate is sigmoid-applied and multiplies the attention output before o_proj.
//! q/k have per-head RMSNorm. Partial RoPE (rope_dims << head_dim).

use rmlx_core::error::Result;
use rmlx_mlx::{
    attention_node_query_rows, multiply, rope, scaled_dot_product_attention, sigmoid, Array,
    Device, Dtype,
};

use rmlx_kv_quant::KvCache;

use super::layers::{qk_norm_fused, repeat_kv, Linear, RmsNorm, SharedInput};

#[allow(missing_debug_implementations)]
pub(super) struct FullAttention {
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
    pub(super) rope_theta: f32,
    pub(super) rope_dims: usize,
}

/// The array mask of a chunked prefill for the full-attention layers, or
/// `None` when the step needs none.
///
/// It has the rows of the attention node the call reaches MLX as, so no layer
/// copies it, and `dtype`, the hidden state's dtype. A layer whose queries
/// have another dtype casts it, which MLX requires of a mask.
pub(super) fn prefill_attention_mask(
    head_dim: usize,
    offset: i32,
    seq: i32,
    dtype: Dtype,
    device: Device,
) -> Result<Option<Array>> {
    let mask_mode = crate::layers::pick_attn_mask_mode(offset, seq);
    if mask_mode != "array" {
        return Ok(None);
    }
    let head_dim = head_dim as i32;
    let rows = attention_node_query_rows(seq, head_dim, head_dim, mask_mode, device);
    let mask = crate::layers::build_chunked_prefill_mask_rows(offset, seq, rows, device)?;
    Ok(Some(mask.astype(dtype, device)?))
}

impl FullAttention {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        offset: i32,
        cache: Option<&mut KvCache>,
        prebuilt_mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        let s = x.shape();
        let batch = s[0];
        let seq = s[1];

        let mut input = SharedInput::new(x);
        // q_proj: [B, S, n_heads * head_dim * 2]
        let q_proj_out = input.forward(&self.q_proj, device)?;
        // Reshape to [B, S, n_heads, head_dim * 2] then split on last dim via slice.
        let q_full = q_proj_out.reshape(
            &[batch, seq, self.n_heads as i32, (self.head_dim * 2) as i32],
            device,
        )?;
        let hd = self.head_dim as i32;
        let nh = self.n_heads as i32;
        let queries = q_full.slice(&[0, 0, 0, 0], &[batch, seq, nh, hd], &[1, 1, 1, 1], device)?; // [B, S, n_heads, head_dim]
        let gate_heads = q_full.slice(
            &[0, 0, 0, hd],
            &[batch, seq, nh, hd * 2],
            &[1, 1, 1, 1],
            device,
        )?; // [B, S, n_heads, head_dim]
            // gate: [B, S, n_heads * head_dim] for broadcasting later.
        let gate =
            gate_heads.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;

        // K, V.
        let k = input.forward(&self.k_proj, device)?;
        let k = k.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let v = input.forward(&self.v_proj, device)?;
        let v = v.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;

        // q_norm on queries [B, S, n_heads, head_dim]; k_norm on k.
        // Fuse the two rms_norm dispatches into one compiled program.
        let (queries, k) = qk_norm_fused(
            &queries,
            &k,
            &self.q_norm.weight,
            &self.k_norm.weight,
            self.q_norm.eps,
            device,
        )?;

        // Transpose to [B, H, S, D].
        let queries = queries.transpose(&[0, 2, 1, 3], device)?;
        let k = k.transpose(&[0, 2, 1, 3], device)?;
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        // Partial RoPE.
        let dims = self.rope_dims as i32;
        let queries = rope(&queries, dims, false, self.rope_theta, 1.0, offset, device)?;
        let k = rope(&k, dims, false, self.rope_theta, 1.0, offset, device)?;

        // GQA: MLX's fast SDPA kernel handles head broadcasting natively when
        // n_q_heads % n_kv_heads == 0. Skip the manual repeat_kv expand to
        // avoid a broadcast+reshape that materialises the repeated cache.
        let _ = repeat_kv;
        let mask_mode = crate::layers::pick_attn_mask_mode(offset, seq);
        // The caller (forward_arr) builds the array mask once per forward and
        // threads it down; this layer builds its own only without one.
        let owned_mask = if prebuilt_mask.is_none() {
            prefill_attention_mask(self.head_dim, offset, seq, queries.dtype(), device)?
        } else {
            None
        };
        let shared_mask = if mask_mode == "array" {
            prebuilt_mask.or(owned_mask.as_ref())
        } else {
            None
        };
        let query_dtype = queries.dtype();
        let cast_mask = match shared_mask {
            Some(m) if m.dtype() != query_dtype => Some(m.astype(query_dtype, device)?),
            _ => None,
        };
        let additive_mask = cast_mask.as_ref().or(shared_mask);

        // Route through `KvCache::update_and_sdpa` —
        // the universal wrapper that fuses cache update + SDPA and dispatches
        // to the Mixed, K8V4-flash, or legacy path based on quant type.
        // No cross-layer KV sharing in Qwen3.5MoE, so the `_shared_source`
        // sibling is not needed.
        let attn = if let Some(c) = cache {
            c.update_and_sdpa(
                &queries,
                &k,
                &v,
                self.scale,
                mask_mode,
                additive_mask,
                device,
            )?
        } else {
            scaled_dot_product_attention(
                &queries,
                &k,
                &v,
                self.scale,
                mask_mode,
                additive_mask,
                device,
            )?
        };
        // [B, H, S, D] -> [B, S, H*D]
        let attn = attn.transpose(&[0, 2, 1, 3], device)?;
        let attn = attn.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;

        // Apply sigmoid gate: attn * sigmoid(gate).
        let gate_sig = sigmoid(&gate, device)?;
        let attn_gated = multiply(&attn, &gate_sig, device)?;

        self.o_proj.forward(&attn_gated, device)
    }
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod attention_tests;
