//! DecoderLayer: composes AttnBlock (GDN or FA) with MlpBlock (MoE or Dense).

use rmlx_core::error::Result;
use rmlx_mlx::{add, Array, Device};

use super::attention::FullAttention;
use super::gated_delta_net::GatedDeltaNet;
use super::layers::RmsNorm;
use super::moe::{DenseMlp, SparseMoeBlock};
use rmlx_kv_quant::{KvCache, LinearAttnCache};

#[allow(missing_debug_implementations)]
pub(super) enum AttnBlock {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

/// MLP variant: sparse MoE (Qwen3.5-35B-A3B) or dense SwiGLU (PARO 27B).
#[allow(missing_debug_implementations)]
pub(super) enum MlpBlock {
    Moe(Box<SparseMoeBlock>),
    Dense(Box<DenseMlp>),
}

#[allow(missing_debug_implementations)]
pub(super) struct DecoderLayer {
    pub(super) input_layernorm: RmsNorm,
    pub(super) post_attention_layernorm: RmsNorm,
    pub(super) attn: AttnBlock,
    pub(super) mlp: MlpBlock,
}

impl DecoderLayer {
    pub(super) fn forward(
        &self,
        x: &Array,
        offset: i32,
        kv_cache: Option<&mut KvCache>,
        lin_cache: Option<&mut LinearAttnCache>,
        prebuilt_mask: Option<&Array>,
        device: Device,
    ) -> Result<Array> {
        let residual = x.try_clone()?;
        let h = self.input_layernorm.forward(x, device)?;
        // GatedDeltaNet uses `lin_cache` (conv tail + delta state); FullAttention
        // uses `kv_cache` (K/V tensors). The two are passed independently so the
        // existing FullAttention KV cache machinery (kv-quant none|k8v8|k8v4|planar)
        // stays untouched.
        let h = match &self.attn {
            AttnBlock::Linear(gdn) => gdn.forward(&h, lin_cache, device)?,
            AttnBlock::Full(fa) => fa.forward(&h, offset, kv_cache, prebuilt_mask, device)?,
        };
        let h = add(&residual, &h, device)?;

        let residual = h.try_clone()?;
        let h = self.post_attention_layernorm.forward(&h, device)?;
        let h = match &self.mlp {
            MlpBlock::Moe(moe) => moe.forward(&h, device)?,
            MlpBlock::Dense(dense) => dense.forward(&h, device)?,
        };
        add(&residual, &h, device)
    }
}
