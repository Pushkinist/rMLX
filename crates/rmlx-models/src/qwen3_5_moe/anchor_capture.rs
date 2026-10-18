//! Capturing the recurrent state at an anchor while a prompt is prefilled.
//!
//! A prefill forwards the prompt in chunks that rarely end on a block boundary,
//! and the recurrent state of a layer cannot be cut back to one afterwards. So
//! the chunk that holds the anchor position is forwarded in two parts, with the
//! state cloned and evaluated between them.

use rmlx_core::error::Result;
use rmlx_kv_quant::{KvCache, LinearAttnCache};
use rmlx_mlx::{Array, Device};

use super::model::Qwen3_5MoeText;
use crate::prompt_cache::Anchor;

/// Where a prompt of `prompt_len` tokens captures its anchor when it is forwarded
/// from `resume_from`: the anchor position of its length, if that is not behind
/// the resume point and the entry can be stored
/// ([`crate::prompt_cache::entry_is_storable`]).
pub(crate) fn anchor_target(
    prompt_len: usize,
    storable: bool,
    resume_from: usize,
) -> Option<usize> {
    if !storable {
        return None;
    }
    Anchor::pos_for(prompt_len).filter(|&pos| pos >= resume_from)
}

/// Takes the recurrent state at `target` tokens, once, as the prompt goes by.
pub(crate) struct AnchorCapture {
    target: Option<usize>,
    /// Flushes a KV cache's pending state between the two parts of a split
    /// chunk, in the form the caches are in (mid-prefill or past `exit_prefill`).
    eval_kv: fn(&KvCache) -> Result<()>,
    captured: Option<Anchor>,
}

impl AnchorCapture {
    /// `target` is the absolute token position to capture at, or `None` to
    /// capture nothing.
    pub(crate) fn new(target: Option<usize>, eval_kv: fn(&KvCache) -> Result<()>) -> Self {
        Self {
            target,
            eval_kv,
            captured: None,
        }
    }

    /// The captured state, or `None` when nothing was captured.
    pub(crate) fn into_anchor(self) -> Option<Anchor> {
        self.captured
    }

    /// How many leading tokens of a chunk the state is taken after: the chunk of
    /// `len` tokens starting at absolute position `start` holds the target when
    /// `start <= target <= start + len`.
    fn cut_in(&self, start: usize, len: usize) -> Option<usize> {
        let target = self.target.filter(|_| self.captured.is_none())?;
        (start..=start + len)
            .contains(&target)
            .then(|| target - start)
    }

    /// Forward `chunk`, which starts at absolute position `start`, and take the
    /// state when the target falls inside it.
    ///
    /// Returns the logits of the chunk's last position.
    pub(crate) fn forward(
        &mut self,
        model: &Qwen3_5MoeText,
        chunk: &[u32],
        start: usize,
        kv_caches: &mut [KvCache],
        lin_caches: &mut [LinearAttnCache],
        device: Device,
    ) -> Result<Array> {
        let Some(cut) = self.cut_in(start, chunk.len()) else {
            return model.forward_seq_with_cache(chunk, Some(kv_caches), Some(lin_caches), device);
        };
        let (head, tail) = chunk.split_at(cut);
        let head_logits = if head.is_empty() {
            None
        } else {
            Some(model.forward_seq_with_cache(
                head,
                Some(&mut *kv_caches),
                Some(&mut *lin_caches),
                device,
            )?)
        };
        if !tail.is_empty() {
            kv_caches.iter().try_for_each(self.eval_kv)?;
        }
        self.capture(start + cut, lin_caches)?;
        match head_logits {
            Some(logits) if tail.is_empty() => Ok(logits),
            _ => model.forward_seq_with_cache(tail, Some(kv_caches), Some(lin_caches), device),
        }
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

#[cfg(test)]
#[path = "anchor_capture_tests.rs"]
mod anchor_capture_tests;
