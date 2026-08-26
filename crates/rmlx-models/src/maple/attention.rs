//! Maple attention: fused QKV + O, per-head QK RMSNorm (fp32 multiply),
//! partial RoPE on SWA layers only (full-attention layers are NoPE).

use rmlx_core::error::Result;
use rmlx_kv_quant::KvCache;
use rmlx_mlx::{rope, scaled_dot_product_attention, Array, Device};

use crate::layers::Linear;

use super::config::MapleConfig;
pub(super) use super::rms::MapleRmsNorm;

/// Maple GQA attention (16/4, head_dim 128). Fused qkv + o, no bias.
#[allow(missing_debug_implementations)]
pub(super) struct MapleAttention {
    /// Concatenated `q_proj`/`k_proj`/`v_proj` along the output axis
    /// (`maple.py` `sanitize`). One `quantized_matmul` per step.
    pub(super) qkv_proj: Linear,
    pub(super) o_proj: Linear,
    pub(super) q_norm: MapleRmsNorm,
    pub(super) k_norm: MapleRmsNorm,
    /// RoPE on SWA layers only; full-attention layers are NoPE.
    pub(super) use_rope: bool,
    /// Per-head Q/K RMSNorm before RoPE (snapshot default true).
    pub(super) use_qk_norm: bool,
    pub(super) scale: f32,
    pub(super) head_dim: usize,
    pub(super) n_q: usize,
    pub(super) n_kv: usize,
    /// `n_q * head_dim` — Q slice of the fused QKV last dim.
    pub(super) q_out: i32,
    /// `n_kv * head_dim` — each of K and V.
    pub(super) kv_out: i32,
    pub(super) rope_theta: f32,
    pub(super) rope_dims: i32,
}

impl MapleAttention {
    /// Wire fused QKV + O + norms; `use_rope` comes from `cfg.is_swa_layer`.
    pub(super) fn new(
        cfg: &MapleConfig,
        layer_idx: usize,
        qkv_proj: Linear,
        o_proj: Linear,
        q_norm: MapleRmsNorm,
        k_norm: MapleRmsNorm,
    ) -> Self {
        let head_dim = cfg.head_dim as usize;
        let n_q = cfg.num_attention_heads as usize;
        let n_kv = cfg.num_key_value_heads as usize;
        Self {
            qkv_proj,
            o_proj,
            q_norm,
            k_norm,
            use_rope: cfg.is_swa_layer(layer_idx),
            use_qk_norm: cfg.use_qk_norm,
            scale: (head_dim as f32).sqrt().recip(),
            head_dim,
            n_q,
            n_kv,
            q_out: (n_q * head_dim) as i32,
            kv_out: (n_kv * head_dim) as i32,
            rope_theta: cfg.rope_theta,
            rope_dims: cfg.rope_dims(),
        }
    }

    /// `x`: `[B, S, hidden]`. Caller chooses the SWA vs full additive mask.
    ///
    /// When `prebuilt_mask` is `Some`, SDPA uses `mask_mode="array"`. Otherwise
    /// the causal/decode fallback from `pick_attn_mask_mode` applies (full
    /// attention prefill / single-token decode with no explicit mask).
    #[allow(
        clippy::indexing_slicing,
        reason = "x is rank-3 [B, S, H] by the decoder contract"
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
        let hd = self.head_dim as i32;
        let nq = self.n_q as i32;
        let nkv = self.n_kv as i32;
        let q_out = self.q_out;
        let kv_out = self.kv_out;
        let total = q_out + 2 * kv_out;

        let qkv = self.qkv_proj.forward(x, device)?;
        let qkv_last = qkv.shape()[2];
        if qkv_last != total {
            return Err(rmlx_core::error::Error::Mlx(format!(
                "maple fused qkv last dim {qkv_last} != q+k+v {total}"
            )));
        }
        let queries = qkv.slice(&[0, 0, 0], &[batch, seq, q_out], &[1, 1, 1], device)?;
        let keys = qkv.slice(
            &[0, 0, q_out],
            &[batch, seq, q_out + kv_out],
            &[1, 1, 1],
            device,
        )?;
        let values = qkv.slice(
            &[0, 0, q_out + kv_out],
            &[batch, seq, total],
            &[1, 1, 1],
            device,
        )?;

        let mut queries = queries.reshape(&[batch, seq, nq, hd], device)?;
        let mut keys = keys.reshape(&[batch, seq, nkv, hd], device)?;
        let values = values.reshape(&[batch, seq, nkv, hd], device)?;

        if self.use_qk_norm {
            queries = self.q_norm.forward(&queries, device)?;
            keys = self.k_norm.forward(&keys, device)?;
        }

        let mut queries = queries.transpose(&[0, 2, 1, 3], device)?;
        let mut keys = keys.transpose(&[0, 2, 1, 3], device)?;
        let values = values.transpose(&[0, 2, 1, 3], device)?;

        if self.use_rope {
            let rope_offset = match &cache {
                Some(c) => c.offset(),
                None => offset,
            };
            queries = rope(
                &queries,
                self.rope_dims,
                false,
                self.rope_theta,
                1.0,
                rope_offset,
                device,
            )?;
            keys = rope(
                &keys,
                self.rope_dims,
                false,
                self.rope_theta,
                1.0,
                rope_offset,
                device,
            )?;
        }

        let q_dtype = queries.dtype();
        let mask_owned: Option<Array> = match prebuilt_mask {
            Some(m) if m.dtype() != q_dtype => Some(m.astype(q_dtype, device)?),
            _ => None,
        };
        let mask_mode = if prebuilt_mask.is_some() {
            "array"
        } else {
            crate::layers::pick_attn_mask_mode(offset, seq)
        };
        let additive_mask = if mask_mode == "array" {
            mask_owned.as_ref().or(prebuilt_mask)
        } else {
            None
        };

        let attn = if let Some(c) = cache {
            c.update_and_sdpa(
                &queries,
                &keys,
                &values,
                self.scale,
                mask_mode,
                additive_mask,
                device,
            )?
        } else {
            scaled_dot_product_attention(
                &queries,
                &keys,
                &values,
                self.scale,
                mask_mode,
                additive_mask,
                device,
            )?
        };

        let attn = attn.transpose(&[0, 2, 1, 3], device)?;
        let attn = attn.reshape(&[batch, seq, nq * hd], device)?;
        self.o_proj.forward(&attn, device)
    }
}

#[cfg(test)]
#[path = "attention_tests.rs"]
mod attention_tests;
