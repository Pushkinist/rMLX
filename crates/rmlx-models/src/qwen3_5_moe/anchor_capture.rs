//! Capturing the recurrent state at an anchor while a prompt is prefilled.
//!
//! The recurrent state of a layer cannot be cut back to an earlier position, so
//! it is taken while the prompt goes by. The prefill splits the prompt at the
//! anchor ([`crate::prefill_chunk::prefill_pieces`]), which puts the state at
//! the anchor between two forwards. The capture clones and evaluates it there.

use rmlx_core::error::Result;
use rmlx_kv_quant::{KvCache, LinearAttnCache};
use rmlx_mlx::{Array, Device};

use super::model::Qwen3_5MoeText;
use crate::prompt_cache::Anchor;

/// Takes the recurrent state at `target` tokens, once, as the prompt goes by.
pub(crate) struct AnchorCapture {
    target: Option<usize>,
    captured: Option<Anchor>,
}

impl AnchorCapture {
    /// `target` is the absolute token position to capture at, or `None` to
    /// capture nothing. A prefill that captures at `target` must split there.
    pub(crate) fn new(target: Option<usize>) -> Self {
        Self {
            target,
            captured: None,
        }
    }

    /// The captured state, or `None` when nothing was captured.
    pub(crate) fn into_anchor(self) -> Option<Anchor> {
        self.captured
    }

    /// Forward `piece`, which starts at absolute position `start`. When the
    /// target is `start`, the state is taken first.
    ///
    /// The prefill loop evaluates the K/V caches between pieces; the capture
    /// evaluates its own clone of the recurrent state.
    ///
    /// Returns the logits of the piece's last position.
    pub(crate) fn forward(
        &mut self,
        model: &Qwen3_5MoeText,
        piece: &[u32],
        start: usize,
        kv_caches: &mut [KvCache],
        lin_caches: &mut [LinearAttnCache],
        device: Device,
    ) -> Result<Array> {
        if self.target == Some(start) {
            self.capture(start, lin_caches)?;
        }
        model.forward_seq_with_cache(piece, Some(kv_caches), Some(lin_caches), device)
    }

    fn capture(&mut self, pos: usize, lin_caches: &[LinearAttnCache]) -> Result<()> {
        let cloned = lin_caches
            .iter()
            .map(LinearAttnCache::try_deep_clone)
            .collect::<Result<Vec<_>>>()?;
        cloned
            .iter()
            .try_for_each(LinearAttnCache::eval_for_spill)?;
        tracing::debug!(pos, "qwen3_5moe: saved the recurrent state at the anchor");
        self.captured = Some(Anchor {
            pos,
            lin_caches: cloned,
        });
        Ok(())
    }
}
