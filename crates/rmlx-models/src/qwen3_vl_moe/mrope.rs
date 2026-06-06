//! Interleaved 3D M-RoPE for Qwen3-VL — host-side, pure (no weights, no GPU).
//!
//! Faithful port of `mlx-vlm/.../qwen3_vl_moe/language.py`:
//! - `get_rope_index` — 3D (T, H, W) position ids from `image_grid_thw`.
//! - `Qwen3VLMoERotaryEmbedding.apply_interleaved_mrope` — the **interleaved**
//!   section layout (NOT the chunked `[TTT…HHH…WWW]` layout the jina /
//!   Qwen2.5-VL path uses; see `jina_v4/image.rs::build_mrope_tables`).
//!
//! ## Interleaved vs chunked (the critical difference)
//!
//! Upstream `apply_interleaved_mrope(freqs, mrope_section)` over
//! `freqs: (3, …, head_dim/2)`:
//! ```text
//! freqs_t = freqs[0] # T everywhere by default
//! for dim, offset in [(1,1),(2,2)]: # H then W
//! length = mrope_section[dim] * 3
//! idx = slice(offset, length, 3) # offset, offset+3, offset+6, …
//! freqs_t[..., idx] = freqs[dim, ..., idx]
//! ```
//! So for each frequency channel `c` in `0..head_dim/2`, the position dim that
//! drives the angle is:
//! - **H** if `c % 3 == 1` and `c < mrope_section[1]*3`
//! - **W** if `c % 3 == 2` and `c < mrope_section[2]*3`
//! - **T** otherwise
//!
//! The frequency itself is always `inv_freq[c]` (the channel index — unlike the
//! chunked layout, channels are NOT re-indexed). `emb = cat(freqs, freqs)` then
//! doubles the table over `head_dim`, so channel `c` and `c + head_dim/2` share
//! the same angle. This matches `apply_multimodal_rotary_pos_emb`'s
//! `rotate_half` convention exactly.

// these helpers are consumed by the qwen3_vl_moe decoder / generator,
// which land with the weight-bearing layers (pending the model download — see
// the module-root status note). Unit-tested standalone here; allow until the
// consumers are wired so clippy -D warnings stays green on the verified core.
#![allow(dead_code)]
#![allow(clippy::float_cmp)]
use rmlx_core::error::{Error, Result};

/// 3D M-RoPE position ids `(t, h, w)`, each `[seq]` (i64 to match HF).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct RopeIndex3D {
    pub t: Vec<i64>,
    pub h: Vec<i64>,
    pub w: Vec<i64>,
}

/// Compute 3D M-RoPE position ids for a single-batch `input_ids` sequence that
/// may contain image spans described by `image_grids` (each `(t, h, w)` in
/// *patch* units; the LLM grid divides h/w by `spatial_merge_size`).
///
/// Faithful port of `LanguageModel.get_rope_index` for the single-batch,
/// no-padding case with zero or more image spans (text-only -> sequential ids
/// in all three dims).
///
/// Algorithm (per `get_rope_index`, image branch):
/// - leading text before an image: `arange(text_len) + st_idx` in all 3 dims;
/// - the image block: `t_index` (constant per temporal frame), `h_index`,
///   `w_index` each offset by `text_len + st_idx`;
/// - text after the last image: continues from `max(prev positions) + 1`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(crate) fn get_rope_index(
    input_ids: &[i64],
    image_grids: &[(usize, usize, usize)],
    image_token_id: i64,
    spatial_merge_size: usize,
) -> Result<RopeIndex3D> {
    let total = input_ids.len();

    // Text-only fast path: sequential ids in all three dims.
    if image_grids.is_empty() {
        let seq: Vec<i64> = (0..total as i64).collect();
        return Ok(RopeIndex3D {
            t: seq.clone(),
            h: seq.clone(),
            w: seq,
        });
    }

    let mut t = vec![0i64; total];
    let mut h = vec![0i64; total];
    let mut w = vec![0i64; total];

    // Running max position (== st_idx for the next text block) and cursor.
    let mut st = 0usize;
    let mut image_index = 0usize;
    // st_idx for the FIRST block is 0; afterwards it is max(prev)+1.
    let mut next_st_idx = 0i64;

    // Walk image spans in order. `ed` is the index of the first image_pad token
    // of the current image span.
    while image_index < image_grids.len() {
        let ed = input_ids[st..]
            .iter()
            .position(|&tok| tok == image_token_id)
            .map(|p| st + p)
            .ok_or_else(|| {
                Error::Model(format!(
                    "qwen3_vl_moe: expected {} image span(s) but found fewer image_pad tokens",
                    image_grids.len()
                ))
            })?;

        let (gt, gh, gw) = image_grids[image_index];
        let llm_t = gt;
        let llm_h = gh / spatial_merge_size;
        let llm_w = gw / spatial_merge_size;
        let vision_len = llm_t * llm_h * llm_w;

        // Leading text block: [st, ed) — arange(text_len) + st_idx.
        let text_len = ed - st;
        let st_idx = next_st_idx;
        for (k, idx) in (st..ed).enumerate() {
            let p = st_idx + k as i64;
            t[idx] = p;
            h[idx] = p;
            w[idx] = p;
        }

        // Image block: (t_index, h_index, w_index) + text_len + st_idx.
        let offset = text_len as i64 + st_idx;
        let mut vi = ed;
        for tt in 0..llm_t {
            for hh in 0..llm_h {
                for ww in 0..llm_w {
                    t[vi] = tt as i64 + offset;
                    h[vi] = hh as i64 + offset;
                    w[vi] = ww as i64 + offset;
                    vi += 1;
                }
            }
        }

        // Advance: next text block starts at max(prev positions) + 1.
        let max_so_far = (0..vi)
            .map(|i| t[i].max(h[i]).max(w[i]))
            .max()
            .unwrap_or(-1);
        next_st_idx = max_so_far + 1;
        st = ed + vision_len;
        image_index += 1;
    }

    // Trailing text block after the last image.
    if st < total {
        let st_idx = next_st_idx;
        for (k, idx) in (st..total).enumerate() {
            let p = st_idx + k as i64;
            t[idx] = p;
            h[idx] = p;
            w[idx] = p;
        }
    }

    Ok(RopeIndex3D { t, h, w })
}

/// Per-channel position-dim selector for the **interleaved** M-RoPE layout.
///
/// Returns `sec[c]` for `c in 0..head_dim/2`: 0=T, 1=H, 2=W. Exactly mirrors
/// `apply_interleaved_mrope`'s `slice(offset, mrope_section[dim]*3, 3)`
/// overwrites on top of an all-T base.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn interleaved_section_map(half: usize, mrope_section: &[usize]) -> Vec<usize> {
    let mut sec = vec![0usize; half]; // default T
                                      // H: channels offset=1, step 3, up to mrope_section[1]*3 (exclusive).
    let h_len = mrope_section[1] * 3;
    let mut c = 1usize;
    while c < h_len && c < half {
        sec[c] = 1;
        c += 3;
    }
    // W: channels offset=2, step 3, up to mrope_section[2]*3 (exclusive).
    let w_len = mrope_section[2] * 3;
    let mut c = 2usize;
    while c < w_len && c < half {
        sec[c] = 2;
        c += 3;
    }
    sec
}

/// Build the per-token interleaved M-RoPE `cos`/`sin` tables `[seq, head_dim]`
/// from 3D position ids — host-side, pure.
///
/// `inv_freq[c] = 1 / rope_theta^(2c/head_dim)` for `c in 0..head_dim/2`.
/// For each token and channel `c`, the angle uses the position id selected by
/// `interleaved_section_map`. `emb = cat(freqs, freqs)` doubles over `head_dim`
/// (channel `c` and `c + head_dim/2` share the angle), matching the
/// `rotate_half` convention in `apply_multimodal_rotary_pos_emb`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn build_interleaved_mrope_tables(
    pos: &RopeIndex3D,
    head_dim: usize,
    rope_theta: f64,
    mrope_section: &[usize],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let seq = pos.t.len();
    let half = head_dim / 2;
    if mrope_section.len() != 3 {
        return Err(Error::Model(format!(
            "qwen3_vl_moe: mrope_section must have 3 entries, got {}",
            mrope_section.len()
        )));
    }
    if mrope_section.iter().sum::<usize>() != half {
        return Err(Error::Model(format!(
            "qwen3_vl_moe: mrope_section {mrope_section:?} sums to {} != head_dim/2 {half}",
            mrope_section.iter().sum::<usize>()
        )));
    }

    let inv_freq: Vec<f64> = (0..half)
        .map(|c| 1.0 / rope_theta.powf((2 * c) as f64 / head_dim as f64))
        .collect();
    let sec = interleaved_section_map(half, mrope_section);

    let mut cos = vec![0.0f32; seq * head_dim];
    let mut sin = vec![0.0f32; seq * head_dim];
    for tok in 0..seq {
        let base = tok * head_dim;
        for c in 0..half {
            let pos_val = match sec[c] {
                0 => pos.t[tok],
                1 => pos.h[tok],
                _ => pos.w[tok],
            } as f64;
            let angle = pos_val * inv_freq[c];
            let (cc, ss) = (angle.cos() as f32, angle.sin() as f32);
            cos[base + c] = cc;
            sin[base + c] = ss;
            cos[base + half + c] = cc;
            sin[base + half + c] = ss;
        }
    }
    Ok((cos, sin))
}

#[cfg(test)]
#[path = "mrope_tests.rs"]
mod tests;
