//! Maple decoder layer: pre-norm attention + pre-norm MoE, unfused residual.

use rmlx_core::error::Result;
use rmlx_kv_quant::KvCache;
use rmlx_mlx::{add, Array, Device};

use super::attention::{MapleAttention, MapleRmsNorm};
use super::moe::MapleSparseMoeBlock;

/// One Maple transformer block.
///
/// Residual (portable path, no fused add+RMS):
/// `h = x + attn(ln1(x)); return h + mlp(ln2(h))`.
#[allow(missing_debug_implementations)]
pub(super) struct MapleDecoderLayer {
    pub(super) input_layernorm: MapleRmsNorm,
    pub(super) self_attn: MapleAttention,
    pub(super) post_attention_layernorm: MapleRmsNorm,
    pub(super) mlp: MapleSparseMoeBlock,
}

impl MapleDecoderLayer {
    /// Compose pre-norm attention and the MoE FFN. All Maple-Preview layers are MoE.
    pub(super) fn new(
        input_layernorm: MapleRmsNorm,
        self_attn: MapleAttention,
        post_attention_layernorm: MapleRmsNorm,
        mlp: MapleSparseMoeBlock,
    ) -> Self {
        Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        }
    }

    /// `x`: `[B, S, hidden]`. `prebuilt_mask` is the SWA or full mask for this layer.
    pub(super) fn forward(
        &self,
        x: &Array,
        offset: i32,
        kv_cache: Option<&mut KvCache>,
        prebuilt_mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        let residual = x.try_clone()?;
        let h = self.input_layernorm.forward(x, device)?;
        let h = self
            .self_attn
            .forward(&h, offset, kv_cache, prebuilt_mask, device)?;
        let h = add(&residual, &h, device)?;

        let residual = h.try_clone()?;
        let h = self.post_attention_layernorm.forward(&h, device)?;
        let h = self.mlp.forward(&h, device)?;
        add(&residual, &h, device)
    }
}
