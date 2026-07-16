//! Gemma3 attention block.

#![allow(clippy::unnecessary_semicolon)]
use rmlx_core::error::Result;
use rmlx_mlx::{rope, scaled_dot_product_attention, Array, Device};

use rmlx_kv_quant::KvCache;

use super::layers::{qk_norm_fused, Linear, RmsNormShifted};

// `repeat_kv` lives in `rmlx_runtime::attention::repeat_kv`. Local function
// below is a thin shim so the rest of this module compiles unchanged. The
// runtime version is byte-identical.

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct Attention {
    pub(super) q_proj: Linear,
    pub(super) k_proj: Linear,
    pub(super) v_proj: Linear,
    pub(super) o_proj: Linear,
    pub(super) q_norm: RmsNormShifted,
    pub(super) k_norm: RmsNormShifted,
    pub(super) n_heads: usize,
    pub(super) n_kv_heads: usize,
    pub(super) head_dim: usize,
    pub(super) scale: f32,
    pub(super) rope_theta: f32, // local theta for sliding, global theta for full-attention layers
    /// Whether this layer uses Sliding-Window Attention (SWA).
    /// When true, the attention mask is banded: each query may only attend to keys
    /// within the last `sliding_window` positions. Set from `layer_types` in config.
    pub(super) is_sliding: bool,
    /// SWA window size in tokens. Only used when `is_sliding == true`.
    /// Set from `sliding_window` in the model config (e.g. 1024 for medgemma/Gemma4-31B,
    /// 512 for Gemma4-e4b).
    pub(super) sliding_window: usize,
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
        let shape = x.shape(); // [batch, seq, hidden]
        let batch = shape[0];
        let seq = shape[1];

        // Capture rotating-cache flag up-front. Consumed by the SWA
        // mask path below to size the prefill mask to the ring-buffer's actual
        // K shape and to skip the per-step decode mask.
        let attn_is_rotating = matches!(&cache, Some(c) if c.is_rotating());

        // Q + K: project -> reshape [B,S,H,D] -> mx.compile-fused per-head RMSNorm
        // (qk_norm_fused: 2 rms_norm dispatches collapsed to 1 compiled program;
        // port of the Qwen3 QK-norm fusion to Gemma3). Gemma3 uses RmsNormShifted
        // (gamma+1 trick); callers pass `shifted_weight` (already `raw + 1.0`)
        // so the closure body is identical to the plain-gamma path.
        // V has no per-head norm in Gemma3.
        let q = self.q_proj.forward(x, device)?;
        let q = q.reshape(
            &[batch, seq, self.n_heads as i32, self.head_dim as i32],
            device,
        )?;
        let k = self.k_proj.forward(x, device)?;
        let k = k.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let (q, k) = qk_norm_fused(
            &q,
            &k,
            &self.q_norm.shifted_weight,
            &self.k_norm.shifted_weight,
            self.q_norm.eps,
            device,
        )?;
        let q = q.transpose(&[0, 2, 1, 3], device)?; // [B, H, S, D]
        let k = k.transpose(&[0, 2, 1, 3], device)?;

        // V projection (no per-head norm in Gemma3).
        let v = self.v_proj.forward(x, device)?;
        let v = v.reshape(
            &[batch, seq, self.n_kv_heads as i32, self.head_dim as i32],
            device,
        )?;
        let v = v.transpose(&[0, 2, 1, 3], device)?;

        // RoPE: full rotation on all layers (both sliding and full-attention).
        // Gemma3 rotates `dims = head_dim` (no ProportionalRoPE / partial rotation).
        // Reference: gemma3_text.py Attention.__init__ lines 57-70:
        // sliding: `rope(dims=head_dim, base=rope_local_base_freq)`
        // full: `rope(dims=head_dim, base=rope_theta)`
        let rope_dims = self.head_dim as i32;
        let q = rope(&q, rope_dims, false, self.rope_theta, 1.0, offset, device)?;
        let k = rope(&k, rope_dims, false, self.rope_theta, 1.0, offset, device)?;

        // Build the additive mask before the cache update so it can be passed
        // directly into `update_and_sdpa`. The mask shape depends on the
        // post-update K length, which is `effective_offset + seq` for the
        // wrapper call (mirrors the Gemma4 approach).
        //
        // `effective_offset`: for rotating caches the K shape is capped at
        // `min(sliding_window - 1, offset) + seq` — mirror mlx-lm
        // `RotatingKVCache.make_mask` (`cache.py:559`).
        let effective_offset = if attn_is_rotating {
            offset.min(self.sliding_window as i32 - 1)
        } else {
            offset
        };

        // Build SDPA mask.
        //
        // For full-attention layers: standard causal / chunked-prefill logic.
        // For SWA layers: banded-causal mask (prefill) or decode mask (single step).
        //
        // Decode path (seq == 1):
        // - Full or SWA within window: mask_mode="" (attend all K).
        // - SWA beyond window: mask_mode="array" with SWA decode mask.
        //
        // Prefill path (seq > 1):
        // - Full (offset==0): mask_mode="causal".
        // - Full (offset>0, chunked): mask_mode="array" with chunked mask.
        // - SWA any offset: mask_mode="array" with banded-causal mask.
        //
        // `mask_holder` owns the Array (if any); `mask_ref` borrows it.
        let total_kv_len_pre = effective_offset + seq;
        let mask_holder: Option<Array>;
        let mask_mode: &str;
        if self.is_sliding {
            if seq == 1 {
                if attn_is_rotating {
                    // Rotating cache caps K at `sliding_window`, single
                    // decode query may attend everything. No mask needed.
                    mask_holder = None;
                    mask_mode = "";
                } else {
                    // Decode step: mask old keys only if total_kv_len > window.
                    mask_holder = crate::layers::build_swa_decode_mask(
                        total_kv_len_pre,
                        self.sliding_window,
                        device,
                    )?;
                    mask_mode = if mask_holder.is_some() { "array" } else { "" };
                }
            } else {
                // Prefill or chunked-prefill: banded-causal mask sized by the
                // capped effective offset.
                mask_holder = Some(crate::layers::build_swa_prefill_mask(
                    effective_offset,
                    seq,
                    self.sliding_window,
                    device,
                )?);
                mask_mode = "array";
            }
        } else {
            // Full-attention layer: standard causal mask logic.
            let mode = crate::layers::pick_attn_mask_mode(effective_offset, seq);
            if mode == "array" {
                mask_holder = Some(crate::layers::build_chunked_prefill_mask(
                    effective_offset,
                    seq,
                    device,
                )?);
            } else {
                mask_holder = None;
            }
            mask_mode = mode;
        };
        let mask_ref = mask_holder.as_ref();

        // GQA: MLX's fast SDPA kernel handles head broadcasting natively when
        // `n_q_heads % n_kv_heads == 0`. Skip the manual `repeat_kv` expand to
        // avoid two broadcast+reshape ops that materialise the repeated cache
        // (2 ops × 34 layers per decode step on medgemma).
        // Reference implementation kept below for documentation purposes.
        let _ = repeat_kv;

        // Route through `KvCache::update_and_sdpa` —
        // the universal wrapper that fuses cache update + SDPA and dispatches
        // to the Mixed, K8V4-flash, or legacy path based on quant type.
        // No cross-layer KV sharing in Gemma3, so the `_shared_source` sibling
        // is not needed.
        let attn_out = if let Some(c) = cache {
            c.update_and_sdpa(&q, &k, &v, self.scale, mask_mode, mask_ref, device)?
        } else {
            scaled_dot_product_attention(&q, &k, &v, self.scale, mask_mode, mask_ref, device)?
        };
        let attn_out = attn_out.transpose(&[0, 2, 1, 3], device)?;
        let attn_out =
            attn_out.reshape(&[batch, seq, (self.n_heads * self.head_dim) as i32], device)?;

        self.o_proj.forward(&attn_out, device)
    }
}

/// Expand K/V from [B, kv_heads, S, D] to [B, q_heads, S, D] by repeating.
///
/// Delegates to `rmlx_runtime::attention::repeat_kv`.
pub(super) fn repeat_kv(x: &Array, repeat: usize, device: Device) -> Result<Array> {
    rmlx_runtime::repeat_kv(x, repeat, device)
}
