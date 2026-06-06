// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]

//! Greedy autoregressive generation for Qwen3-VL-MoE — text + image branches.
//!
//! Self-contained, deliberately simpler than the qwen3_5_moe generator: no
//! prompt-cache reuse, no GatedDeltaNet, no MTP. The Qwen3-VL text decoder is a
//! plain GQA stack with 3D interleaved M-RoPE, so a fresh per-layer
//! [`KvCache`] + sequential prefill/decode is correct and fast enough for the
//! single-image serve path. Sampling / penalty / constraint hooks mirror the
//! other arch generators so the server's decode-loop call site is uniform.
//!
//! The 3D position bookkeeping is the only Qwen3-VL-specific wrinkle:
//! - text-only: positions are sequential in all three (t,h,w) dims;
//! - image: [`super::mrope::get_rope_index`] builds the prompt positions; the
//!   decode continues from `max(prompt positions) + 1`, incrementing in all
//!   three dims (mirroring `language.py` trailing-text positions).

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};

use crate::constraint::ConstraintEngine;
use crate::gemma4::ProbeStep;
use crate::sampler::{apply_mask_argmax, sample_token_array, Pcg32, PenaltyConfig, SamplerConfig};
use rmlx_kv_quant::{KvCache, KvQuant};

use super::image::{scatter_vision_features, visual_token_positions};
use super::model::Qwen3VlMoeText;
use super::mrope::{get_rope_index, RopeIndex3D};
use super::vision::VisionOutput;

/// Trailing-window size for repetition penalties (matches the other archs).
const PENALTY_WINDOW: usize = 20;

/// Pick the next token: greedy argmax (temp<=0 and no penalties) else sample.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn pick_token(
    logits_flat: &Array,
    vocab: usize,
    sampler_cfg: &SamplerConfig,
    penalty_cfg: &PenaltyConfig,
    constraint: &mut Option<&mut dyn ConstraintEngine>,
    token_history: &[u32],
    rng: &mut Pcg32,
    device: Device,
) -> Result<u32> {
    // Build the constraint mask if the engine wants one this step.
    let mask: Option<Vec<bool>> = match constraint {
        Some(c) if c.wants_mask() => Some(c.step_mask(vocab).to_vec()),
        _ => None,
    };
    let greedy = sampler_cfg.temperature <= 0.0 && !penalty_cfg.penalties_active();
    let chosen = if greedy {
        match &mask {
            Some(m) => apply_mask_argmax(logits_flat, m, device)?,
            None => rmlx_mlx::argmax(logits_flat, -1, device)?,
        }
    } else {
        let recent = if token_history.len() > PENALTY_WINDOW {
            &token_history[token_history.len() - PENALTY_WINDOW..]
        } else {
            token_history
        };
        sample_token_array(
            logits_flat,
            sampler_cfg,
            mask.as_deref(),
            penalty_cfg,
            recent,
            rng,
            device,
        )?
    };
    Array::eval(&chosen)?;
    let b = chosen.to_bytes()?;
    Ok(i32::from_le_bytes(b[..4].try_into().unwrap()) as u32)
}

/// Emit one decode step (token id + piece). Logit stats are left at defaults —
/// the smoke classifier only needs ids + pieces, and the hot loop avoids the
/// extra GPU→host transfer of the full logit row.
fn make_step(token_id: u32, tokenizer: &tokenizers::Tokenizer) -> ProbeStep {
    ProbeStep {
        token_id,
        piece: tokenizer
            .id_to_token(token_id)
            .unwrap_or_default()
            .replace('\u{2581}', " ")
            .into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    }
}

/// Greedy generation from plain text token ids.
#[allow(clippy::too_many_arguments)]
pub fn generate_greedy(
    model: &Qwen3VlMoeText,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: KvQuant,
    _max_ctx_override: Option<i32>,
    _prompt_cache_slots: usize,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    mut constraint: Option<&mut dyn ConstraintEngine>,
    sampler_cfg: &SamplerConfig,
    rng: &mut Pcg32,
    penalty_cfg: &PenaltyConfig,
    token_history: &mut Vec<u32>,
) -> Result<Vec<ProbeStep>> {
    tracing::info!(
        arch = "Qwen3VLMoeForConditionalGeneration",
        ?kv_quant,
        prompt_len = prompt_ids.len(),
        n_tokens,
        "qwen3_vl_moe::generate_greedy (text)"
    );
    if n_tokens == 0 {
        return Ok(vec![]);
    }
    let vocab = model.cfg.vocab_size;
    let n_layers = model.cfg.num_hidden_layers;
    let mut kv: Vec<KvCache> = (0..n_layers)
        .enumerate()
        .map(|(i, _)| KvCache::with_quant(kv_quant).with_layer_idx(i))
        .collect();

    let mut steps = Vec::with_capacity(n_tokens);
    let logits = model.forward_seq_with_cache(prompt_ids, Some(&mut kv), device)?;
    let first = pick_token(
        &logits,
        vocab,
        sampler_cfg,
        penalty_cfg,
        &mut constraint,
        token_history,
        rng,
        device,
    )?;

    let mut next = first;
    for _ in 0..n_tokens {
        token_history.push(next);
        let step = make_step(next, tokenizer);
        let forced = step_fn(&step);
        steps.push(step);
        if eos_ids.contains(&next) {
            break;
        }
        if let Some(c) = constraint.as_mut() {
            c.advance(next);
        }
        let feed = forced.unwrap_or(next);
        let logits = model.forward_seq_with_cache(&[feed], Some(&mut kv), device)?;
        next = pick_token(
            &logits,
            vocab,
            sampler_cfg,
            penalty_cfg,
            &mut constraint,
            token_history,
            rng,
            device,
        )?;
    }
    Ok(steps)
}

/// Image-branch generation: ViT features already produced by the caller and
/// passed as `vision`. The caller also supplies the augmented prompt ids
/// (`aug_ids`) containing the image-token spans, and the per-image
/// `image_grids` `(t,h,w)` (patch units) for the 3D M-RoPE index.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub fn generate_image(
    model: &Qwen3VlMoeText,
    tokenizer: &tokenizers::Tokenizer,
    aug_ids: &[u32],
    vision: &VisionOutput,
    image_grids: &[(usize, usize, usize)],
    image_token_id: i64,
    spatial_merge_size: usize,
    n_tokens: usize,
    device: Device,
    kv_quant: KvQuant,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    mut constraint: Option<&mut dyn ConstraintEngine>,
    sampler_cfg: &SamplerConfig,
    rng: &mut Pcg32,
    penalty_cfg: &PenaltyConfig,
    token_history: &mut Vec<u32>,
) -> Result<Vec<ProbeStep>> {
    tracing::info!(
        arch = "Qwen3VLMoeForConditionalGeneration",
        ?kv_quant,
        prompt_len = aug_ids.len(),
        n_tokens,
        "qwen3_vl_moe::generate_image"
    );
    if n_tokens == 0 {
        return Ok(vec![]);
    }
    let vocab = model.cfg.vocab_size;
    let seq = aug_ids.len() as i32;

    // 1. text embeddings + scatter vision features at the image-token positions.
    let ids_i64: Vec<i64> = aug_ids.iter().map(|&t| i64::from(t)).collect();
    let visual_positions = visual_token_positions(&ids_i64, image_token_id);
    let inputs_embeds = model.embed_ids(aug_ids, device)?;
    let inputs_embeds = scatter_vision_features(
        &inputs_embeds,
        &vision.image_embeds,
        &visual_positions,
        device,
    )?;

    // 2. 3D M-RoPE positions for the augmented sequence.
    let pos = get_rope_index(&ids_i64, image_grids, image_token_id, spatial_merge_size)?;

    // 3. prefill (deepstack injected after layers 0..len(deepstack)).
    let n_layers = model.cfg.num_hidden_layers;
    let mut kv: Vec<KvCache> = (0..n_layers)
        .enumerate()
        .map(|(i, _)| KvCache::with_quant(kv_quant).with_layer_idx(i))
        .collect();
    let logits = model.forward_embeds(
        &inputs_embeds,
        seq,
        &pos,
        0,
        &vision.deepstack_embeds,
        &visual_positions,
        Some(&mut kv),
        device,
    )?;

    let mut steps = Vec::with_capacity(n_tokens);
    let first = pick_token(
        &logits,
        vocab,
        sampler_cfg,
        penalty_cfg,
        &mut constraint,
        token_history,
        rng,
        device,
    )?;

    // Decode continues from max(prompt positions)+1 in all three dims (the
    // trailing-text rule); the image tokens compress the position range so the
    // decode base differs from the raw cache offset — drive it explicitly.
    let max_pos = pos
        .t
        .iter()
        .chain(pos.h.iter())
        .chain(pos.w.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let decode_base = max_pos + 1;

    let mut next = first;
    for g in 0..n_tokens {
        let pos_val = decode_base + g as i64;
        token_history.push(next);
        let step = make_step(next, tokenizer);
        let forced = step_fn(&step);
        steps.push(step);
        if eos_ids.contains(&next) {
            break;
        }
        if let Some(c) = constraint.as_mut() {
            c.advance(next);
        }
        let feed = forced.unwrap_or(next);
        let ids_i32 = [feed as i32];
        let ids_bytes = unsafe { std::slice::from_raw_parts(ids_i32.as_ptr().cast::<u8>(), 4) };
        let ids_arr = Array::from_bytes(ids_bytes, &[1], Dtype::I32)?;
        let base_offset = kv[0].offset();
        let dpos = RopeIndex3D {
            t: vec![pos_val],
            h: vec![pos_val],
            w: vec![pos_val],
        };
        let logits = model.forward_arr(&ids_arr, 1, &dpos, base_offset, Some(&mut kv), device)?;
        next = pick_token(
            &logits,
            vocab,
            sampler_cfg,
            penalty_cfg,
            &mut constraint,
            token_history,
            rng,
            device,
        )?;
    }
    Ok(steps)
}
