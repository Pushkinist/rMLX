// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for image bytes into Array
#![allow(unsafe_code)]

//! Image-input plumbing for qwen3_vl_moe: vision-feature scatter + the
//! deepstack injection index math. Host-side / MLX-array helpers shared by the
//! model forward and the generator.
//!
//! Mirrors `mlx-vlm/.../qwen3_vl_moe/qwen3_vl_moe.py::merge_input_ids_with_image_features`
//! (the masked-scatter at `image_token_id` positions) and
//! `language.py::_deepstack_process` (additive injection of deepstack visual
//! embeds at the visual-token positions for the first-N decoder layers).

// consumed by the qwen3_vl_moe model forward / generator (pending the
// weight-bearing layers + model download). Unit-tested standalone; allow until
// the consumers are wired so clippy -D warnings stays green.
#![allow(dead_code)]

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

/// Scatter `vision_feats` (`[num_merged, hidden]`) into `inputs_embeds`
/// (`[1, seq, hidden]`) at the `image_token_id` positions, in order.
///
/// Faithful to `merge_input_ids_with_image_features` (masked_scatter over the
/// image/video token positions). The number of image-pad positions must equal
/// the number of vision-feature rows. The image-pad positions are contiguous
/// for a single image span, so a single `slice_update` writes the block; the
/// contiguity assertion keeps us faithful to the in-order scatter.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn scatter_vision_features(
    inputs_embeds: &Array, // [1, seq, hidden]
    vision_feats: &Array,  // [num_merged, hidden]
    visual_positions: &[usize],
    device: Device,
) -> Result<Array> {
    let es = inputs_embeds.shape(); // [1, seq, hidden]
    let hid = es[2];
    let vs = vision_feats.shape(); // [num_merged, hidden]
    let n_feat = vs[0] as usize;

    if visual_positions.len() != n_feat {
        return Err(Error::Model(format!(
            "qwen3_vl_moe: {} image_pad positions != {n_feat} vision features",
            visual_positions.len()
        )));
    }
    if vs[1] != hid {
        return Err(Error::Model(format!(
            "qwen3_vl_moe: vision feature dim {} != embed hidden {hid}",
            vs[1]
        )));
    }
    let first = visual_positions[0];
    let contiguous = visual_positions
        .iter()
        .enumerate()
        .all(|(k, &p)| p == first + k);
    if !contiguous {
        return Err(Error::Model(
            "qwen3_vl_moe: image_pad positions are not contiguous".into(),
        ));
    }
    let block = vision_feats
        .astype(inputs_embeds.dtype(), device)?
        .reshape(&[1, n_feat as i32, hid], device)?;
    inputs_embeds.slice_update(
        &block,
        &[0, first as i32, 0],
        &[1, (first + n_feat) as i32, hid],
        &[1, 1, 1],
        device,
    )
}

/// Locate the `image_token_id` positions in `input_ids` (the visual-token
/// positions used for scatter and deepstack injection).
pub(super) fn visual_token_positions(input_ids: &[i64], image_token_id: i64) -> Vec<usize> {
    input_ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == image_token_id)
        .map(|(i, _)| i)
        .collect()
}

/// Additively inject a deepstack `visual_embeds` block (`[n_visual, hidden]`)
/// into `hidden` (`[1, seq, hidden]`) at the contiguous `visual_positions`.
///
/// Mirrors `_deepstack_process`: `hidden[visual_positions] += visual_embeds`.
/// For a single contiguous image span this is `hidden[:, first:first+n] += block`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn deepstack_inject(
    hidden: &Array,        // [1, seq, hidden]
    visual_embeds: &Array, // [n_visual, hidden]
    visual_positions: &[usize],
    device: Device,
) -> Result<Array> {
    let hs = hidden.shape();
    let hid = hs[2];
    let n = visual_embeds.shape()[0] as usize;
    if visual_positions.len() != n {
        return Err(Error::Model(format!(
            "qwen3_vl_moe deepstack: {} positions != {n} embeds",
            visual_positions.len()
        )));
    }
    if n == 0 {
        return hidden.try_clone();
    }
    let first = visual_positions[0];
    let contiguous = visual_positions
        .iter()
        .enumerate()
        .all(|(k, &p)| p == first + k);
    if !contiguous {
        return Err(Error::Model(
            "qwen3_vl_moe deepstack: visual positions not contiguous".into(),
        ));
    }
    // Read the current slice, add, write back.
    let cur = hidden.slice(
        &[0, first as i32, 0],
        &[1, (first + n) as i32, hid],
        &[1, 1, 1],
        device,
    )?;
    let add_block = visual_embeds
        .astype(hidden.dtype(), device)?
        .reshape(&[1, n as i32, hid], device)?;
    let summed = rmlx_mlx::add(&cur, &add_block, device)?;
    hidden.slice_update(
        &summed,
        &[0, first as i32, 0],
        &[1, (first + n) as i32, hid],
        &[1, 1, 1],
        device,
    )
}

#[cfg(test)]
#[path = "image_tests.rs"]
mod tests;
