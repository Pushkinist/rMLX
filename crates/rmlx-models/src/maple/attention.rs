//! Maple attention: split Q/K/V/O, per-head QK RMSNorm (fp32 multiply),
//! partial RoPE on SWA layers only (full-attention layers are NoPE).

use rmlx_core::error::Result;
use rmlx_kv_quant::KvCache;
use rmlx_mlx::{rms_norm, rope, scaled_dot_product_attention, Array, Device, Dtype};

use crate::layers::Linear;

use super::config::MapleConfig;

/// RMSNorm with the weight multiply in float32 (`MapleRMSNorm` in maple.py).
///
/// `mx.fast.rms_norm` on bf16 rounds the normalized activation before the
/// weight multiply (~1% per element vs the training reference). Casting both
/// operands to f32, then casting the product back, matches the reference.
#[allow(
    clippy::exhaustive_structs,
    reason = "closed layer struct — weight + eps is the full MapleRMSNorm contract"
)]
#[allow(missing_debug_implementations)]
pub(super) struct MapleRmsNorm {
    /// Learned scale, shape `[dims]`.
    pub(super) weight: Array,
    /// Variance epsilon (1e-6 in the snapshot).
    pub(super) eps: f32,
}

impl MapleRmsNorm {
    /// `weight` is `[dims]` (hidden for layer norms, `head_dim` for Q/K norms).
    pub(super) fn new(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
    }

    /// `rms_norm(x.f32, weight.f32, eps).astype(x.dtype)`.
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        let x_f32 = x.astype(Dtype::F32, device)?;
        let w_f32 = self.weight.astype(Dtype::F32, device)?;
        let out = rms_norm(&x_f32, Some(&w_f32), self.eps, device)?;
        out.astype(x.dtype(), device)
    }
}

/// Maple GQA attention (16/4, head_dim 128). Split q/k/v/o, no bias.
#[allow(missing_debug_implementations)]
pub(super) struct MapleAttention {
    pub(super) q_proj: Linear,
    pub(super) k_proj: Linear,
    pub(super) v_proj: Linear,
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
    pub(super) rope_theta: f32,
    pub(super) rope_dims: i32,
}

impl MapleAttention {
    /// Wire projections + norms; `use_rope` comes from `cfg.is_swa_layer`.
    #[allow(
        clippy::too_many_arguments,
        reason = "loader passes the four projections and two norms; grouping them would be a single-caller wrapper"
    )]
    pub(super) fn new(
        cfg: &MapleConfig,
        layer_idx: usize,
        q_proj: Linear,
        k_proj: Linear,
        v_proj: Linear,
        o_proj: Linear,
        q_norm: MapleRmsNorm,
        k_norm: MapleRmsNorm,
    ) -> Self {
        let head_dim = cfg.head_dim as usize;
        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            use_rope: cfg.is_swa_layer(layer_idx),
            use_qk_norm: cfg.use_qk_norm,
            scale: (head_dim as f32).sqrt().recip(),
            head_dim,
            n_q: cfg.num_attention_heads as usize,
            n_kv: cfg.num_key_value_heads as usize,
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

        let queries = self.q_proj.forward(x, device)?;
        let keys = self.k_proj.forward(x, device)?;
        let values = self.v_proj.forward(x, device)?;

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
