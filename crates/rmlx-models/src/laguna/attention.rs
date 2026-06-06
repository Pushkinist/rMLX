//! Laguna attention block with per-head q/k norms, partial RoPE, and g_proj gating.

use rmlx_core::error::Result;
use rmlx_mlx::{exp, log1p, multiply, rope, scaled_dot_product_attention, Array, Device};

use rmlx_kv_quant::KvCache;

use super::config::LayerKind;
use super::layers::{Linear, RmsNorm};

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Laguna attention with per-head q/k norms, partial RoPE, and g_proj gating.
#[allow(missing_debug_implementations)]
pub(super) struct Attention {
    pub(super) q_proj: Linear,
    pub(super) k_proj: Linear,
    pub(super) v_proj: Linear,
    pub(super) o_proj: Linear,
    pub(super) g_proj: Linear,
    pub(super) q_norm: RmsNorm,
    pub(super) k_norm: RmsNorm,
    pub(super) n_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) scale: f32,
    pub(super) layer_kind: LayerKind,
    pub(super) sliding_window: usize,
    pub(super) rope_theta_full: f32,
    pub(super) rope_theta_sliding: f32,
    pub(super) rope_dims_full: usize,
}

impl Attention {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn forward(
        &self,
        x: &Array,
        offset: i32,
        cache: Option<&mut KvCache>,
        device: Device,
    ) -> Result<Array> {
        let shape = x.shape();
        let batch = shape[0];
        let seq = shape[1];

        // Q: project -> reshape [B,S,H,D] -> q_norm -> transpose [B,H,S,D] -> rope.
        let q = self.q_proj.forward(x, device)?;
        let q = q.reshape(
            &[batch, seq, self.n_heads as i32, self.head_dim as i32],
            device,
        )?;
        let q = self.q_norm.forward(&q, device)?;
        let q = q.transpose(&[0, 2, 1, 3], device)?;

        // K: same path.
        let k = self.k_proj.forward(x, device)?;
        let k = k.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let k = self.k_norm.forward(&k, device)?;
        let k = k.transpose(&[0, 2, 1, 3], device)?;

        // V: no norm.
        let v = self.v_proj.forward(x, device)?;
        let v = v.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        // RoPE: partial rotation for full-attention, full for sliding.
        let (q, k) = match self.layer_kind {
            LayerKind::FullAttention => {
                let dims = self.rope_dims_full as i32;
                let qr = rope(&q, dims, false, self.rope_theta_full, 1.0, offset, device)?;
                let kr = rope(&k, dims, false, self.rope_theta_full, 1.0, offset, device)?;
                (qr, kr)
            }
            LayerKind::SlidingAttention => {
                let dims = self.head_dim as i32;
                let qr = rope(
                    &q,
                    dims,
                    false,
                    self.rope_theta_sliding,
                    1.0,
                    offset,
                    device,
                )?;
                let kr = rope(
                    &k,
                    dims,
                    false,
                    self.rope_theta_sliding,
                    1.0,
                    offset,
                    device,
                )?;
                (qr, kr)
            }
        };

        // Total K length after cache update: offset + seq tokens (pre-computed
        // before calling update_and_sdpa so the SWA decode mask can be built first).
        let total_kv_len = offset + seq;

        // Build SDPA mask before the cache interaction (mirrors gemma3 approach).
        //
        // For SlidingAttention layers: banded-causal mask (prefill) or decode mask.
        // - Decode step: mask old keys only if total_kv_len > sliding_window.
        // - Prefill: banded-causal mask clipped to sliding_window.
        // For FullAttention layers: standard causal / chunked-prefill logic.
        let mask_holder: Option<Array>;
        let mask_mode: &str;
        if self.layer_kind == LayerKind::SlidingAttention {
            if seq == 1 {
                mask_holder = crate::layers::build_swa_decode_mask(
                    total_kv_len,
                    self.sliding_window,
                    device,
                )?;
                mask_mode = if mask_holder.is_some() { "array" } else { "" };
            } else {
                mask_holder = Some(crate::layers::build_swa_prefill_mask(
                    offset,
                    seq,
                    self.sliding_window,
                    device,
                )?);
                mask_mode = "array";
            }
        } else {
            let mode = crate::layers::pick_attn_mask_mode(offset, seq);
            if mode == "array" {
                mask_holder = Some(crate::layers::build_chunked_prefill_mask(
                    offset, seq, device,
                )?);
            } else {
                mask_holder = None;
            }
            mask_mode = mode;
        }
        let mask_ref = mask_holder.as_ref();

        // GQA: MLX's fast SDPA kernel handles head broadcasting natively when
        // `n_q_heads % n_kv_heads == 0`. Skip the manual `repeat_kv` expand to
        // avoid the broadcast+reshape that materialises the repeated cache.
        let _ = repeat_kv;

        // Route through `KvCache::update_and_sdpa` —
        // the universal wrapper that fuses cache update + SDPA and dispatches
        // to the Mixed, K8V4-flash, or legacy path based on quant type.
        // No cross-layer KV sharing in Laguna, so the `_returning_kv` sibling
        // is not needed.
        let attn_out = if let Some(c) = cache {
            c.update_and_sdpa(&q, &k, &v, self.scale, mask_mode, mask_ref, device)?
        } else {
            scaled_dot_product_attention(&q, &k, &v, self.scale, mask_mode, mask_ref, device)?
        };
        // [B,H,S,D] -> [B,S,H,D] -> [B,S,H*D]
        let attn_out = attn_out.transpose(&[0, 2, 1, 3], device)?;
        let attn_out =
            attn_out.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;

        // Gating: softplus(g_proj(x)) -> [B, S, n_heads, 1] * attn per-head.
        let g = self.g_proj.forward(x, device)?; // [B, S, n_heads]
        let g = softplus(&g, device)?;
        let g = g.reshape(&[batch, seq, self.n_heads as i32, 1], device)?;
        let attn_g = attn_out.reshape(
            &[batch, seq, self.n_heads as i32, self.head_dim as i32],
            device,
        )?;
        let attn_g = multiply(&attn_g, &g, device)?;
        let attn_g =
            attn_g.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;

        self.o_proj.forward(&attn_g, device)
    }
}

/// Softplus: log(1 + exp(x)).
pub(super) fn softplus(x: &Array, device: Device) -> Result<Array> {
    let ex = exp(x, device)?;
    log1p(&ex, device)
}

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn repeat_kv(x: &Array, repeat: usize, device: Device) -> Result<Array> {
    let s = x.shape();
    let (b, kv_h, seq, d) = (s[0], s[1], s[2], s[3]);
    let x5 = rmlx_mlx::expand_dims(x, 2, device)?;
    let bc = rmlx_mlx::broadcast_to(&x5, &[b, kv_h, repeat as i32, seq, d], device)?;
    bc.reshape(&[b, kv_h * repeat as i32, seq, d], device)
}
