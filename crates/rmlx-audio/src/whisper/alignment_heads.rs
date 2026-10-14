//! The `alignment_heads` entry of `weights.npz`: the decoder cross-attention
//! heads whose weights line up text tokens with audio frames, as integer
//! (layer, head) pairs of shape `(N, 2)`. The large-v3 snapshot stores them as
//! `<i8`, which the npy reader gives as `I32`.
//!
//! Nothing reads the heads yet, so an entry in another form, or with a pair
//! outside the decoder, gives one warning and no heads, and the load goes on.

use tracing::{info, warn};

use super::{WhisperConfig, WhisperError};
use crate::npz::WeightMap;
use rmlx_mlx::{Array, Dtype};

const ENTRY: &str = "alignment_heads";

/// The (layer, head) pairs of the `alignment_heads` entry. `None` when the
/// snapshot has no such entry, or when the entry is not usable pairs.
pub(super) fn read_alignment_heads(
    weights: &WeightMap,
    cfg: &WhisperConfig,
) -> Result<Option<Vec<(usize, usize)>>, WhisperError> {
    let heads = weights
        .get(ENTRY)
        .map(|entry| usable_pairs(entry, cfg))
        .transpose()?
        .flatten();
    info!(
        n_alignment_heads = heads.as_ref().map_or(0, Vec::len),
        loaded = heads.is_some(),
        "Whisper alignment heads"
    );
    Ok(heads)
}

/// The pairs of `entry`, or `None` after one warning that gives the fact.
fn usable_pairs(
    entry: &Array,
    cfg: &WhisperConfig,
) -> Result<Option<Vec<(usize, usize)>>, WhisperError> {
    let (dtype, shape) = (entry.dtype(), entry.shape());
    if dtype != Dtype::I32 || !matches!(shape.as_slice(), [_, 2]) {
        warn!(
            ?dtype,
            ?shape,
            "alignment_heads is not int32 or int64 pairs of shape (N, 2); no alignment heads loaded"
        );
        return Ok(None);
    }

    let bytes = entry
        .to_bytes()
        .map_err(|e| WhisperError::Mlx(e.to_string()))?;
    let mut values = bytes.chunks_exact(4).map(|chunk| {
        let mut word = [0_u8; 4];
        word.copy_from_slice(chunk);
        i32::from_le_bytes(word)
    });
    let below = |value: i32, limit: usize| usize::try_from(value).ok().filter(|&v| v < limit);
    let mut heads = Vec::with_capacity(bytes.len() / 8);
    while let (Some(layer), Some(head)) = (values.next(), values.next()) {
        let (Some(layer_idx), Some(head_idx)) =
            (below(layer, cfg.n_text_layer), below(head, cfg.n_text_head))
        else {
            warn!(
                layer,
                head,
                n_text_layer = cfg.n_text_layer,
                n_text_head = cfg.n_text_head,
                "alignment_heads has a pair outside the decoder; no alignment heads loaded"
            );
            return Ok(None);
        };
        heads.push((layer_idx, head_idx));
    }
    Ok(Some(heads))
}

#[cfg(test)]
#[path = "alignment_heads_tests.rs"]
mod alignment_heads_tests;
