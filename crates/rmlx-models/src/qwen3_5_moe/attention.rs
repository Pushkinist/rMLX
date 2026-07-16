//! Full-Attention (Qwen3Next style).
//!
//! q_proj output = n_heads * head_dim * 2; first half = queries, second half = gate.
//! gate is sigmoid-applied and multiplies the attention output before o_proj.
//! q/k have per-head RMSNorm. Partial RoPE (rope_dims << head_dim).

use rmlx_core::error::Result;
use rmlx_mlx::{multiply, rope, scaled_dot_product_attention, sigmoid, Array, Device};

use rmlx_kv_quant::KvCache;

use super::layers::{qk_norm_fused, repeat_kv, Linear, RmsNorm};

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

        // q_proj: [B, S, n_heads * head_dim * 2]
        let q_proj_out = self.q_proj.forward(x, device)?;
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
        let k = self.k_proj.forward(x, device)?;
        let k = k.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let v = self.v_proj.forward(x, device)?;
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
        // Mask sharing across layers: caller (forward_arr) builds the array
        // mask once per chunk and threads it down; FullAttention only
        // builds its own as a fallback. 60→6 mask constructions for a
        // 7-chunk, 10-FA-layer prefill.
        let owned_mask: Option<Array> = if mask_mode == "array" && prebuilt_mask.is_none() {
            Some(crate::layers::build_chunked_prefill_mask(
                offset, seq, device,
            )?)
        } else {
            None
        };
        // MLX SDPA requires mask dtype to be promotable to Q/K/V dtype.
        // The prebuilt mask is Bf16 (built for Bf16 models); for F16 models
        // (e.g. PARO INT4 affine), cast the mask to F16 to match.
        let q_dtype = queries.dtype();
        let mask_owned: Option<Array> = if mask_mode == "array" {
            let raw = prebuilt_mask.or(owned_mask.as_ref());
            match raw {
                Some(m) if m.dtype() != q_dtype => Some(m.astype(q_dtype, device)?),
                _ => None,
            }
        } else {
            None
        };
        // Final mask reference: prefer cast copy, then owned, then prebuilt.
        let additive_mask = if mask_mode == "array" {
            mask_owned
                .as_ref()
                .or(prebuilt_mask)
                .or(owned_mask.as_ref())
        } else {
            None
        };

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
