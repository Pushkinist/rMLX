//! Gemma3 decoder layer.

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device};

use rmlx_kv_quant::KvCache;

use super::attention::Attention;
use super::layers::{clip_residual_fused, Mlp, RmsNormShifted};

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct DecoderLayer {
    pub(super) input_norm: RmsNormShifted,
    pub(super) post_attn_norm: RmsNormShifted,
    pub(super) pre_ffn_norm: RmsNormShifted,
    pub(super) post_ffn_norm: RmsNormShifted,
    pub(super) attn: Attention,
    pub(super) mlp: Mlp,
}

impl DecoderLayer {
    pub(super) fn forward(
        &self,
        x: &Array,
        offset: i32,
        cache: Option<&mut KvCache>,
        device: Device,
    ) -> Result<Array> {
        // Attention sub-layer with pre-norm + post-norm + residual.
        // Residual add is `clip_residual_fused` — mx.compile-fused
        // port of mlx-lm `gemma3_text.clip_residual`. For BF16 (medgemma)
        // the body is `x + y`, but compiled (drops per-call FFI dispatch).
        let residual = x.try_clone()?;
        let h = self.input_norm.forward(x, device)?;
        let h = self.attn.forward(&h, offset, cache, device)?;
        let h = self.post_attn_norm.forward(&h, device)?;
        let h = clip_residual_fused(&residual, &h, device)?;

        // FFN sub-layer with pre-norm + post-norm + residual.
        let residual = h.try_clone()?;
        let h = self.pre_ffn_norm.forward(&h, device)?;
        let h = self.mlp.forward(&h, device)?;
        let h = self.post_ffn_norm.forward(&h, device)?;
        clip_residual_fused(&residual, &h, device)
    }
}
