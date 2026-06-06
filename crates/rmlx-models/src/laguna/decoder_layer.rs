//! Laguna decoder layer.

use rmlx_core::error::Result;
use rmlx_mlx::{add, Array, Device};

use rmlx_kv_quant::KvCache;

use super::attention::Attention;
use super::layers::{DenseMlp, RmsNorm};
use super::moe::SparseMoeBlock;

// ---------------------------------------------------------------------------
// Mlp enum
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) enum Mlp {
    Dense(DenseMlp),
    Sparse(Box<SparseMoeBlock>),
}

impl Mlp {
    pub(super) fn forward(&self, x: &Array, device: Device) -> Result<Array> {
        match self {
            Mlp::Dense(m) => m.forward(x, device),
            Mlp::Sparse(m) => m.forward(x, device),
        }
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[allow(missing_debug_implementations)]
pub(super) struct DecoderLayer {
    pub(super) input_norm: RmsNorm,
    pub(super) post_attn_norm: RmsNorm,
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
        let residual = x.try_clone()?;
        let h = self.input_norm.forward(x, device)?;
        let h = self.attn.forward(&h, offset, cache, device)?;
        let h = add(&residual, &h, device)?;

        let residual = h.try_clone()?;
        let h = self.post_attn_norm.forward(&h, device)?;
        let h = self.mlp.forward(&h, device)?;
        add(&residual, &h, device)
    }
}
