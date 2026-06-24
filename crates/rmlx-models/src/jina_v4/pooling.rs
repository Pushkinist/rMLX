//! jina-embeddings-v4 pooling + L2-normalize + matryoshka (text path).
//!
//! Port of `modeling_jina_embeddings_v4.py`:
//! - single-vector (text, no image): mean-pool the last hidden over the
//!   attention mask, then `F.normalize(p=2, dim=-1)` — lines 217-251.
//! - multi-vector: `multi_vector_projector(hidden)` then
//!   `F.normalize(p=2, dim=-1)` then `* attention_mask` — lines 253-266.
//! - matryoshka (single-vector only): `emb[:, :truncate_dim]` then
//!   **re-L2-normalize**; `truncate_dim ∈ matryoshka_dims` (validated) —
//!   lines 349-353, 392-395.
//!
//! ## Text-path simplification (single, non-padded input)
//!
//! rMLX serves one non-padded text sequence at a time. The
//! `attention_mask` is therefore all-ones, so:
//! - single-vector mean-pool over the mask == a plain mean over the seq axis;
//! - multi-vector `* attention_mask` == identity (no-op).
//!
//! This matches the reference exactly for the all-ones-mask case; batched /
//! padded inputs are an endpoint concern, not implemented here.
//!
//! ## L2-normalize
//!
//! `‖x‖₂ = sqrt(sum(x²) + 1e-12)` using the `rmlx_mlx::sqrt` op. The
//! `1e-12` floor matches `torch.nn.functional.normalize`'s default `eps`
//! denominator clamp. (An earlier implementation used the `exp(0.5·ln(·))`
//! identity; the numeric result is unchanged to < 1e-6.)

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{
    add, divide, expand_dims, multiply, scalar_f32, sqrt, sum_axis, Array, Device, Dtype,
};

/// L2-normalize `x` over its last axis: `x / sqrt(sum(x^2, last))`.
///
/// `x` may be any rank ≥ 1; the reduction is over the final axis and the
/// divisor is broadcast back (keep-dim emulated via `expand_dims`, since
/// `sum_axis` drops the reduced axis). Faithful to `F.normalize(p=2, dim=-1)`.
fn l2_normalize_last(x: &Array, device: Device) -> Result<Array> {
    let last = (x.shape().len() as i32) - 1;
    let sq = multiply(x, x, device)?;
    // `sum_axis` drops the reduced axis; `expand_dims` restores keep-dim so
    // the per-row norm broadcasts back over the last axis.
    let s = sum_axis(&sq, last, device)?;
    let s = expand_dims(&s, last, device)?;
    // ‖x‖₂ = sqrt(sum(x²) + 1e-12); 1e-12 matches F.normalize's eps clamp.
    let s = add(&s, &scalar_f32(1e-12), device)?; // f32-ok: pooling output is converted to Vec<f32>; f32 precision is intentional here
    let norm = sqrt(&s, device)?;
    divide(x, &norm, device)
}

/// Validate a requested matryoshka truncation dim against the model's allowed
/// set (`config.matryoshka_dims`, jina default `[128,256,512,1024,2048]`).
///
/// Mirrors `_validate_encoding_params` (ref lines 392-395): any dim not in the
/// set is a hard error with a clear, listing message.
pub(super) fn validate_truncate_dim(truncate_dim: usize, matryoshka_dims: &[usize]) -> Result<()> {
    if matryoshka_dims.contains(&truncate_dim) {
        Ok(())
    } else {
        Err(Error::Config(format!(
            "jina-v4: invalid truncate_dim {truncate_dim}; must be one of {matryoshka_dims:?}"
        )))
    }
}

/// Materialize a contiguous f32 vector from an [`Array`] (CPU).
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn to_f32_vec(arr: &Array) -> Result<Vec<f32>> {
    let f = arr.astype(Dtype::F32, Device::Cpu)?;
    f.eval()?;
    Ok(f.to_bytes()?
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().expect("4-byte f32 chunk")))
        .collect())
}

/// Single-vector text embedding: mean-pool over the sequence axis, then
/// L2-normalize; optionally matryoshka-truncate (`emb[:dim]`) + re-normalize.
///
/// `hidden` is the post-final-norm tensor `[1, seq, hidden]` (with the active
/// task's LoRA already live). Returns a `[hidden]` (or `[truncate_dim]`) vec.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn single_vector(
    hidden: &Array,
    matryoshka_dims: &[usize],
    truncate_dim: Option<usize>,
    device: Device,
) -> Result<Vec<f32>> {
    // Mean over seq (axis 1). attention_mask is all-ones for one non-padded
    // text input, so masked-sum / mask-sum == plain mean (ref lines 246-249).
    let shape = hidden.shape(); // [1, seq, hidden]
    let seq = shape[1];
    let summed = sum_axis(hidden, 1, device)?; // -> [1, hidden]
    let pooled = divide(&summed, &scalar_f32(seq as f32), device)?; // f32-ok: output is Vec<f32> via to_f32_vec; f32 precision intentional
    let normed = l2_normalize_last(&pooled, device)?; // [1, hidden]

    match truncate_dim {
        None => to_f32_vec(&normed),
        Some(dim) => {
            validate_truncate_dim(dim, matryoshka_dims)?;
            let full = normed.shape(); // [1, hidden]
            let hidden_dim = full[1] as usize;
            if dim > hidden_dim {
                return Err(Error::Config(format!(
                    "jina-v4: truncate_dim {dim} exceeds embedding dim {hidden_dim}"
                )));
            }
            // emb[:, :dim] then re-L2-normalize (ref lines 350-353).
            let sliced = normed.slice(&[0, 0], &[1, dim as i32], &[1, 1], device)?;
            let renorm = l2_normalize_last(&sliced, device)?;
            to_f32_vec(&renorm)
        }
    }
}

/// Single-vector **image** embedding: mean-pool the hidden states over the
/// `[vision_start, vision_end]` span (inclusive), L2-normalize; optionally
/// matryoshka-truncate + re-normalize.
///
/// Port of `modeling_jina_embeddings_v4.py:226-251` (the image branch of
/// `get_single_vector_embeddings`): the `image_mask` selects positions
/// `vision_start_idx ..= vision_end_idx` and pools
/// `sum(h * mask) / mask.sum()`. `hidden` is `[1, seq, hidden]`;
/// `start`/`end` are the (inclusive) token indices of `<|vision_start|>` /
/// `<|vision_end|>`. Returns a `[hidden]` (or `[truncate_dim]`) vec.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn single_vector_image_span(
    hidden: &Array,
    start: usize,
    end: usize,
    matryoshka_dims: &[usize],
    truncate_dim: Option<usize>,
    device: Device,
) -> Result<Vec<f32>> {
    let shape = hidden.shape(); // [1, seq, hidden]
    let seq = shape[1] as usize;
    let hid = shape[2];
    if start > end || end >= seq {
        return Err(Error::Model(format!(
            "jina-v4: image span [{start},{end}] out of range for seq {seq}"
        )));
    }
    let span = (end - start + 1) as i32;
    // hidden[:, start..=end, :] -> [1, span, hidden]
    let slice = hidden.slice(
        &[0, start as i32, 0],
        &[1, end as i32 + 1, hid],
        &[1, 1, 1],
        device,
    )?;
    let summed = sum_axis(&slice, 1, device)?; // -> [1, hidden]
    let pooled = divide(&summed, &scalar_f32(span as f32), device)?; // f32-ok: output is Vec<f32> via to_f32_vec; f32 precision intentional
    let normed = l2_normalize_last(&pooled, device)?; // [1, hidden]

    match truncate_dim {
        None => to_f32_vec(&normed),
        Some(dim) => {
            validate_truncate_dim(dim, matryoshka_dims)?;
            let hidden_dim = normed.shape()[1] as usize;
            if dim > hidden_dim {
                return Err(Error::Config(format!(
                    "jina-v4: truncate_dim {dim} exceeds embedding dim {hidden_dim}"
                )));
            }
            let sliced = normed.slice(&[0, 0], &[1, dim as i32], &[1, 1], device)?;
            let renorm = l2_normalize_last(&sliced, device)?;
            to_f32_vec(&renorm)
        }
    }
}

/// Multi-vector text embedding: per-token L2-normalize the projected hidden,
/// then `* attention_mask` (identity for one non-padded text input). Returns
/// `[seq][projector_dim]`.
///
/// Matryoshka truncation is **not** applied to multi-vector embeddings — the
/// reference `_process_batches` only truncates `single_vec_emb` (line 350);
/// the `multi_vec_emb` branch (line 355) never touches `truncate_dim`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn multi_vector(
    projected: &Array, // [1, seq, projector_dim] (post multi_vector_projector)
    device: Device,
) -> Result<Vec<Vec<f32>>> {
    let normed = l2_normalize_last(projected, device)?; // per-token p=2
    let shape = normed.shape(); // [1, seq, proj_dim]
    let seq = shape[1] as usize;
    let proj_dim = shape[2] as usize;
    let flat = to_f32_vec(&normed)?; // row-major [1*seq*proj_dim]
    debug_assert_eq!(flat.len(), seq * proj_dim);
    Ok(flat.chunks_exact(proj_dim).map(<[f32]>::to_vec).collect())
}

// ---------------------------------------------------------------------------
// Tests (gated on the on-disk jina-v4 snapshot; skip-with-msg if absent)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "pooling_tests.rs"]
mod tests;
