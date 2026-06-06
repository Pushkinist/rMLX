// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for image bytes into Array
#![allow(unsafe_code)]

//! jina-embeddings-v4 image embedding path.
//!
//! Closes full-multimodal jina-v4: build the image input sequence, scatter
//! the ViT vision features into the text embeddings at the `<|image_pad|>`
//! positions, compute 3D M-RoPE position ids, run the text decoder with an
//! M-RoPE-aware forward, then pool exactly as the reference.
//!
//! ## Sequence construction (faithful to `process_images`)
//!
//! jina's `process_images` (`modeling_jina_embeddings_v4.py:42-89`) uses the
//! fixed prompt
//! `"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n"`.
//! The Qwen2.5-VL processor expands the single `<|image_pad|>` placeholder to
//! `num_merged_tokens = grid_t * grid_h * grid_w / spatial_merge_size^2`
//! copies. Verified end-to-end: tokenizing the fixed prompt yields
//! `[151644, 872, 198, 151652, 151655, 151653, 74785, 279, 2168, 13,
//! 151645, 198]` — replacing the single `151655` with `N` copies reproduces
//! the exact `process_images` `input_ids` bit-for-bit.
//!
//! ## 3D M-RoPE position ids (faithful to `get_rope_index`)
//!
//! `qwen2_5_vl.py::get_rope_index` (1721+) — text tokens get sequential
//! `(t,h,w)=(p,p,p)` ids; the image span gets a `(t,h,w)` grid where
//! `t` is constant (`second_per_grid_t = 0` for images), `h`/`w` enumerate
//! the merged grid. The text *after* the image continues from
//! `max(prev positions) + 1`. Verified against the Python reference for a
//! `grid_thw=[1,4,4]` image (positions matched exactly).
//!
//! ## Merge (faithful to colqwen2_5 / qwen2_5_vl.forward)
//!
//! `mlx_embeddings/models/colqwen2_5.py:152-183` and
//! `qwen2_5_vl.py:1979-1996`: replace the `embed_tokens` embeddings at the
//! `<|image_pad|>` (151655) positions, in order, with the ViT
//! `vision_embed()` rows; keep text token embeddings as-is.
//!
//! ## Pooling (faithful to `get_single_vector_embeddings` image branch)
//!
//! Single-vector: mean-pool hidden over `[vision_start, vision_end]`
//! inclusive then L2-norm (+ optional matryoshka). Multi-vector:
//! `multi_vector_projector(hidden)` then per-token L2-norm. Task LoRA flows
//! exactly as in the text path (same `Linear` seams).

#![allow(clippy::float_cmp)]
use std::mem::size_of_val;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};

use super::config::JinaV4Config;

/// The fixed image prompt jina's `process_images` uses (single
/// `<|image_pad|>` placeholder — expanded per image below).
pub(super) const IMAGE_PROMPT: &str =
    "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Describe the image.<|im_end|>\n";

/// Expand the single `<|image_pad|>` (`image_token_id`) placeholder in
/// `prompt_ids` to `num_merged_tokens` copies — reproducing the Qwen2.5-VL
/// processor's image-token expansion.
///
/// `prompt_ids` is the tokenization of [`IMAGE_PROMPT`] (exactly one
/// `image_token_id`). Returns the full input-id sequence.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn expand_image_pad(
    prompt_ids: &[i64],
    image_token_id: i64,
    num_merged_tokens: usize,
) -> Result<Vec<i64>> {
    let pad_positions: Vec<usize> = prompt_ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == image_token_id)
        .map(|(i, _)| i)
        .collect();
    if pad_positions.len() != 1 {
        return Err(Error::Model(format!(
            "jina-v4 image: expected exactly one image_pad ({image_token_id}) in the \
             prompt tokenization, found {}",
            pad_positions.len()
        )));
    }
    if num_merged_tokens == 0 {
        return Err(Error::Model(
            "jina-v4 image: num_merged_tokens is 0 (empty vision grid)".into(),
        ));
    }
    let pos = pad_positions[0];
    let mut out = Vec::with_capacity(prompt_ids.len() - 1 + num_merged_tokens);
    out.extend_from_slice(&prompt_ids[..pos]);
    out.extend(std::iter::repeat_n(image_token_id, num_merged_tokens));
    out.extend_from_slice(&prompt_ids[pos + 1..]);
    Ok(out)
}

/// 3D M-RoPE position ids `(t, h, w)`, each `[seq]`. Faithful host port of
/// `qwen2_5_vl.py::get_rope_index` for the single-image, full-attention
/// (no padding) case jina image embedding uses.
#[derive(Debug, Clone)]
pub(super) struct RopeIndex {
    pub t: Vec<i64>,
    pub h: Vec<i64>,
    pub w: Vec<i64>,
}

/// Compute the 3D M-RoPE position ids for `input_ids` containing exactly one
/// image span (`image_grid_thw = (grid_t, grid_h, grid_w)` — the *patch*
/// grid; the LLM grid divides h/w by `spatial_merge_size`).
///
/// Faithful port of `get_rope_index` (the `image_grid_thw is not None`
/// branch, single batch, no padding): leading text gets sequential ids;
/// the image block gets `t_index` (constant — `second_per_grid_t = 0` for
/// images), `h_index`, `w_index` offset by `text_len + st_idx`; trailing
/// text continues from `max(prev) + 1`. Verified against the Python
/// reference (`grid_thw=[1,4,4]` -> positions matched exactly).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn get_rope_index(
    input_ids: &[i64],
    image_token_id: i64,
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    spatial_merge_size: usize,
) -> Result<RopeIndex> {
    let llm_grid_t = grid_t;
    let llm_grid_h = grid_h / spatial_merge_size;
    let llm_grid_w = grid_w / spatial_merge_size;
    let vision_len = llm_grid_t * llm_grid_h * llm_grid_w;

    let ed_image = input_ids
        .iter()
        .position(|&t| t == image_token_id)
        .ok_or_else(|| Error::Model("jina-v4 image: no image_token_id in input_ids".into()))?;
    let n_pad = input_ids.iter().filter(|&&t| t == image_token_id).count();
    if n_pad != vision_len {
        return Err(Error::Model(format!(
            "jina-v4 image: image_pad count {n_pad} != llm grid t*h*w {vision_len}"
        )));
    }

    let total = input_ids.len();
    let mut t = vec![0i64; total];
    let mut h = vec![0i64; total];
    let mut w = vec![0i64; total];

    // ---- leading text block: arange(text_len) + st_idx (st_idx = 0) -------
    let st = 0usize;
    let ed = ed_image;
    let text_len = ed - st; // tokens [0, ed)
    let st_idx = 0i64;
    for (k, idx) in (st..ed).enumerate() {
        let p = st_idx + k as i64;
        t[idx] = p;
        h[idx] = p;
        w[idx] = p;
    }

    // ---- image block: (t_index, h_index, w_index) + text_len + st_idx ----
    // t_index: range(llm_grid_t) expanded over h*w, * second_per_grid_t(=0)
    // * tokens_per_second -> all zero for an image.
    // h_index: arange(h).view(1,-1,1).expand(t,h,w).flatten()
    // w_index: arange(w).view(1,1,-1).expand(t,h,w).flatten()
    let offset = text_len as i64 + st_idx;
    let mut vi = ed; // first image_pad position
    for _tt in 0..llm_grid_t {
        let t_pos = 0i64; // second_per_grid_t == 0 for images
        for hh in 0..llm_grid_h {
            for ww in 0..llm_grid_w {
                t[vi] = t_pos + offset;
                h[vi] = hh as i64 + offset;
                w[vi] = ww as i64 + offset;
                vi += 1;
            }
        }
    }

    // ---- trailing text block: arange(text_len) + (max(prev) + 1) ---------
    let st = ed + vision_len;
    if st < total {
        let prev_max = (0..st)
            .map(|i| t[i].max(h[i]).max(w[i]))
            .max()
            .unwrap_or(-1);
        let st_idx = prev_max + 1;
        for (k, idx) in (st..total).enumerate() {
            let p = st_idx + k as i64;
            t[idx] = p;
            h[idx] = p;
            w[idx] = p;
        }
    }

    Ok(RopeIndex { t, h, w })
}

/// Build the per-token M-RoPE `cos`/`sin` tables `[seq, head_dim]` from the
/// 3D position ids — host-side, pure (mirrors
/// `Qwen2_5_VLRotaryEmbedding.forward` + `apply_multimodal_rotary_pos_emb`).
///
/// `inv_freq[j] = 1 / rope_theta^(2j/head_dim)` for `j in 0..head_dim/2`
/// (default RoPE init, `attention_scaling = 1.0`). `emb = cat(freqs, freqs)`
/// over `head_dim`. The mrope split selects, per channel `d`, which of the 3
/// position ids feeds the angle: with `mrope_section = [16,24,24]`, doubled
/// to `[16,24,24,16,24,24]`, channels are partitioned as
/// `T:[0,16) H:[16,40) W:[40,64) T:[64,80) H:[80,104) W:[104,128)` and the
/// frequency index is `d % (head_dim/2)`. Exactly reproduces
/// `cat([m[i%3] for i,m in enumerate(cos.split(mrope_section*2,-1))])`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn build_mrope_tables(
    pos: &RopeIndex,
    head_dim: usize,
    rope_theta: f64,
    mrope_section: &[usize],
) -> Result<(Vec<f32>, Vec<f32>)> {
    let seq = pos.t.len();
    let half = head_dim / 2;
    if mrope_section.iter().sum::<usize>() != half {
        return Err(Error::Model(format!(
            "jina-v4 image: mrope_section {mrope_section:?} sums to {} != head_dim/2 {half}",
            mrope_section.iter().sum::<usize>()
        )));
    }
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| 1.0 / rope_theta.powf((2 * j) as f64 / head_dim as f64))
        .collect();

    // sec[d] (d in 0..half) = which position dim (0=T,1=H,2=W) feeds channel
    // d, per the doubled-section split. The second half (d+half) repeats the
    // same selection because emb = cat(freqs, freqs) AND the doubled section
    // pattern is [s0,s1,s2,s0,s1,s2] -> i%3 over chunk index.
    let mut sec = vec![0usize; half];
    {
        let mut d = 0usize;
        for (chunk_i, &sz) in mrope_section.iter().enumerate() {
            let which = chunk_i % 3;
            for _ in 0..sz {
                sec[d] = which;
                d += 1;
            }
        }
        debug_assert_eq!(d, half);
    }

    let mut cos = vec![0.0f32; seq * head_dim];
    let mut sin = vec![0.0f32; seq * head_dim];
    for tok in 0..seq {
        let base = tok * head_dim;
        for d in 0..half {
            let pos_val = match sec[d] {
                0 => pos.t[tok],
                1 => pos.h[tok],
                _ => pos.w[tok],
            } as f64;
            let angle = pos_val * inv_freq[d];
            let (c, s) = (angle.cos() as f32, angle.sin() as f32);
            // emb = cat(freqs, freqs): channel d and d+half share the angle.
            cos[base + d] = c;
            sin[base + d] = s;
            cos[base + half + d] = c;
            sin[base + half + d] = s;
        }
    }
    Ok((cos, sin))
}

/// Scatter `vision_feats` (`[num_merged, hidden]`) into `inputs_embeds`
/// (`[1, seq, hidden]`) at the `image_token_id` positions, in order.
///
/// Faithful to `colqwen2_5.py:get_input_embeddings_batch` /
/// `qwen2_5_vl.py:1990-1996` (`masked_scatter` over `image_token_id`).
/// The number of `<|image_pad|>` positions must equal `vision_feats` rows.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(super) fn scatter_vision_features(
    inputs_embeds: &Array, // [1, seq, hidden]
    vision_feats: &Array,  // [num_merged, hidden]
    input_ids: &[i64],
    image_token_id: i64,
    device: Device,
) -> Result<Array> {
    let es = inputs_embeds.shape(); // [1, seq, hidden]
    let hid = es[2];
    let vs = vision_feats.shape(); // [num_merged, hidden]
    let n_feat = vs[0] as usize;

    let pad_idx: Vec<usize> = input_ids
        .iter()
        .enumerate()
        .filter(|(_, &t)| t == image_token_id)
        .map(|(i, _)| i)
        .collect();
    if pad_idx.len() != n_feat {
        return Err(Error::Model(format!(
            "jina-v4 image: {} image_pad positions != {n_feat} vision features",
            pad_idx.len()
        )));
    }
    if vs[1] != hid {
        return Err(Error::Model(format!(
            "jina-v4 image: vision feature dim {} != embed hidden {hid}",
            vs[1]
        )));
    }
    // The image_pad positions are contiguous (the prompt has one run of
    // expanded pads), so a single slice_update writes the whole block. Assert
    // contiguity to stay faithful to the in-order masked_scatter.
    let first = pad_idx[0];
    let contiguous = pad_idx.iter().enumerate().all(|(k, &p)| p == first + k);
    if !contiguous {
        return Err(Error::Model(
            "jina-v4 image: image_pad positions are not contiguous".into(),
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

/// Upload a host f32 buffer as a bf16 [`Array`] of `shape` (RoPE tables are
/// applied against bf16 activations — match the dtype, as the vision tower
/// does for its precomputed tables).
pub(super) fn upload_bf16(buf: &[f32], shape: &[i32], device: Device) -> Result<Array> {
    let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast::<u8>(), size_of_val(buf)) };
    Array::from_bytes(bytes, shape, Dtype::F32)?.astype(Dtype::Bf16, device)
}

/// Locate the (inclusive) `<|vision_start|>` / `<|vision_end|>` token indices
/// in `input_ids` (exactly one each — single image). Used for the
/// single-vector image-span mean-pool.
pub(super) fn vision_span(
    input_ids: &[i64],
    vision_start_id: i64,
    vision_end_id: i64,
) -> Result<(usize, usize)> {
    let start = input_ids
        .iter()
        .position(|&t| t == vision_start_id)
        .ok_or_else(|| Error::Model("jina-v4 image: no <|vision_start|> token".into()))?;
    let end = input_ids
        .iter()
        .position(|&t| t == vision_end_id)
        .ok_or_else(|| Error::Model("jina-v4 image: no <|vision_end|> token".into()))?;
    if start > end {
        return Err(Error::Model(format!(
            "jina-v4 image: vision_start idx {start} after vision_end idx {end}"
        )));
    }
    Ok((start, end))
}

/// `(head_dim, rope_theta, mrope_section)` for the text tower, read from the
/// parsed config. `mrope_section` defaults to jina's `[16, 24, 24]` (sums to
/// `head_dim/2 = 64`) when the config omits `rope_scaling.mrope_section`.
pub(super) fn mrope_params(cfg: &JinaV4Config) -> (usize, f64, Vec<usize>) {
    let tc = &cfg.text_config;
    (
        tc.head_dim,
        tc.rope_theta,
        cfg.mrope_section
            .clone()
            .unwrap_or_else(|| vec![16, 24, 24]),
    )
}

// ---------------------------------------------------------------------------
// Tests (pure host logic — no model / GPU needed)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "image_tests.rs"]
mod tests;
