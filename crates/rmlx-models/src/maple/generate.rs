//! Autoregressive generation for Maple: chunked prefill and KV-cached decode.
//!
//! Hybrid SWA + full KV (rotating ring on sliding layers, unbounded on full
//! layers). No GatedDeltaNet. Signature matches
//! [`crate::qwen3_5_moe::generate_greedy`].

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::too_many_lines
)]
use std::time::Instant;

use rmlx_core::error::Result;
use rmlx_mlx::Device;
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};

use crate::constraint::ConstraintEngine;
use crate::decode_loop::{
    capture_logprobs, choose_token, chunked_prefill, pipelined_decode, reject_nan_prefill,
    DecodeCtx,
};
use crate::kv_cache::{kv_max_seq_and_ceiling, warn_if_kv_codec_net_negative, KvLayerShape};
use crate::prompt_cache::{chained_block_hashes_seeded, Consumed, ReusePolicy};
use rmlx_kv_quant::{KvCache, KvQuant};

use super::model::MapleText;
use super::prompt_cache::{active_layout_key, ensure_prompt_cache, MapleEntry, PROMPT_CACHE};

const ARCH: &str = "MapleForCausalLM";

/// Greedy autoregressive generation using KV-cache prefill + decode.
///
/// Argument list matches [`crate::qwen3_5_moe::generate_greedy`].
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free; remaining unwrap is on a 4-byte argmax slab"
)]
pub fn generate_greedy<'a>(
    model: &MapleText,
    tokenizer: &'a tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: KvQuant,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: &'a [u32],
    step_fn: &'a mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
    mut constraint: Option<&'a mut dyn ConstraintEngine>,
    sampler_cfg: &'a crate::sampler::SamplerConfig,
    rng: &'a mut crate::sampler::Pcg32,
    penalty_cfg: &'a crate::sampler::PenaltyConfig,
    token_history: &'a mut Vec<u32>,
) -> Result<Vec<crate::decode_loop::ProbeStep>> {
    use crate::decode_loop::ProbeStep;

    tracing::info!(
        arch = ARCH,
        ?kv_quant,
        ?max_ctx_override,
        prompt_cache_slots,
        "generate_greedy: selected KV cache quant"
    );

    if n_tokens == 0 {
        return Ok(vec![]);
    }

    let vocab = model.cfg.vocab_size;
    let mut steps = Vec::with_capacity(n_tokens);

    ensure_prompt_cache(prompt_cache_slots);

    assert_eq!(
        PROMPT_CACHE.policy(),
        ReusePolicy::ExactOnly,
        "Maple prompt cache must be ExactOnly — SWA rotating rings are only \
         safe to reuse via a full RAM deep_clone; partial-prefix reuse is not enabled",
    );

    if let Consumed::Exact(cloned) = PROMPT_CACHE.consume(
        prompt_ids,
        kv_quant,
        model.n_layers(),
        false,
        model.model_sig,
    ) {
        return exact_hit_decode(
            model,
            tokenizer,
            cloned,
            prompt_ids,
            n_tokens,
            vocab,
            device,
            eos_ids,
            step_fn,
            constraint,
            sampler_cfg,
            rng,
            penalty_cfg,
            token_history,
        );
    }

    let prefill_t0 = Instant::now();
    let n_layers = model.n_layers();
    let (initial_max_seq, max_seq_ceiling) =
        kv_max_seq_and_ceiling(max_ctx_override, model.cfg.max_position_embeddings);
    let mut caches = model.make_cache(kv_quant, initial_max_seq, max_seq_ceiling);

    {
        let window_u64 = model.cfg.sliding_window.max(0) as u64;
        let layer_shapes: Vec<KvLayerShape> = (0..n_layers)
            .map(|i| KvLayerShape {
                head_dim: model.cfg.head_dim.max(0) as u64,
                kv_heads: model.cfg.num_key_value_heads.max(0) as u64,
                window: if model.cfg.is_swa_layer(i) {
                    Some(window_u64)
                } else {
                    None
                },
            })
            .collect();
        let eff_seq = (max_seq_ceiling.max(0) as u64).max(prompt_ids.len() as u64);
        warn_if_kv_codec_net_negative(kv_quant, &layer_shapes, eff_seq);
    }

    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("maple");
    let prefill_logits = chunked_prefill(
        &mut caches,
        prompt_ids,
        prefill_chunk,
        device,
        ARCH,
        |chunk, caches| model.forward_seq_with_cache(chunk, Some(caches), device),
    )?;

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;
    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());
    reject_nan_prefill(
        ARCH,
        logits_flat.dtype(),
        nan_count,
        max_abs_logit,
        prompt_ids.len(),
    )?;

    let lp_k = sampler_cfg.top_logprobs_k as usize;
    let mut ctx = DecodeCtx {
        tokenizer,
        vocab,
        n_tokens,
        device,
        eos_ids,
        step_fn,
        constraint: constraint.take(),
        sampler_cfg,
        rng,
        penalty_cfg,
        token_history,
        arch: ARCH,
        resolve_pieces: false,
    };

    let mask_active = ctx.constraint.as_ref().is_some_and(|c| c.wants_mask());
    let top = choose_token(&mut ctx, &logits_flat, mask_active)?;
    top.eval()?;
    let top_bytes = top.to_bytes()?;
    let last_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    if let Some(c) = ctx.constraint.as_mut() {
        c.advance(last_id);
    }
    ctx.token_history.push(last_id);
    let prefill_total_ns = prefill_t0.elapsed().as_nanos();

    let piece = tokenizer
        .id_to_token(last_id)
        .unwrap_or_else(|| format!("<unk:{last_id}>"));

    tracing::debug!(
        step = 0,
        token_id = last_id,
        piece = %piece,
        prompt_len = prompt_ids.len(),
        "maple generate_greedy prefill"
    );

    {
        let kv_snap: Result<Vec<KvCache>> = caches.iter().map(KvCache::try_deep_clone).collect();
        if let Ok(kvs) = kv_snap {
            match kvs.iter().try_for_each(KvCache::eval_for_spill) {
                Ok(()) => {
                    let lk = active_layout_key();
                    let block_hashes = chained_block_hashes_seeded(
                        prompt_ids,
                        crate::prompt_cache::request_cache_seed(
                            lk,
                            kv_quant,
                            n_layers,
                            model.model_sig,
                        ),
                    );
                    let entry = MapleEntry {
                        prompt_token_ids: prompt_ids.to_vec(),
                        block_hashes,
                        kv_caches: kvs,
                        first_id: last_id,
                        first_piece: piece.clone(),
                        kv_quant: Some(kv_quant),
                        is_ssd_hydrated: false,
                    };
                    PROMPT_CACHE.with_inner_mut(|guard| {
                        if let Some(cache) = guard.as_mut() {
                            if cache.push(entry).is_some() {
                                tracing::debug!(
                                    prompt_len = prompt_ids.len(),
                                    token_id = last_id,
                                    "maple generate_greedy: prompt cache MISS — saved snapshot"
                                );
                            }
                        }
                    });
                }
                Err(e) => tracing::warn!(
                    error = %e,
                    "maple generate_greedy: pre-eval of KV caches failed, skipping prompt cache store"
                ),
            }
        }
    }

    let prefill_logprobs = if lp_k > 0 {
        capture_logprobs(&logits_flat, &top, lp_k)
    } else {
        None
    };
    (ctx.step_fn)(steps.push_mut(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit,
        nan_count,
        logprobs: prefill_logprobs,
    }));

    if eos_ids.contains(&last_id) {
        let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
        tracing::info!(
            target: "decode_profile",
            arch = ARCH,
            n_steps = 0,
            prefill_ms,
            "decode_profile (prefill-EOS)"
        );
        return Ok(steps);
    }

    let (stats, post) = pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
        model.forward_step(y, &mut caches, device)
    })?;

    let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
    let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
    let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
    let step_ms = (stats.step_total_ns as f64) / 1.0e6;
    let n = f64::from(stats.decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = ARCH,
        n_steps = stats.decode_steps,
        prefill_ms,
        forward_total_ms = forward_ms,
        eval_total_ms = eval_ms,
        step_total_ms = step_ms,
        forward_per_step_ms = forward_ms / n,
        eval_per_step_ms = eval_ms / n,
        cache_path = "miss",
        "decode_profile"
    );

    model
        .kv_bytes
        .store(caches.iter().map(KvCache::resident_bytes).sum(), post);

    Ok(steps)
}

/// Exact prompt-cache hit: skip re-prefill, replay the stored first token.
#[allow(clippy::too_many_arguments)]
fn exact_hit_decode<'a>(
    model: &MapleText,
    tokenizer: &'a tokenizers::Tokenizer,
    cloned: MapleEntry,
    prompt_ids: &[u32],
    n_tokens: usize,
    vocab: i32,
    device: Device,
    eos_ids: &'a [u32],
    step_fn: &'a mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
    mut constraint: Option<&'a mut dyn ConstraintEngine>,
    sampler_cfg: &'a crate::sampler::SamplerConfig,
    rng: &'a mut crate::sampler::Pcg32,
    penalty_cfg: &'a crate::sampler::PenaltyConfig,
    token_history: &'a mut Vec<u32>,
) -> Result<Vec<crate::decode_loop::ProbeStep>> {
    use crate::decode_loop::ProbeStep;

    let MapleEntry {
        kv_caches: mut caches,
        first_id: last_id,
        first_piece: piece,
        ..
    } = cloned;
    let mut steps = Vec::with_capacity(n_tokens);

    tracing::debug!(
        prompt_len = prompt_ids.len(),
        token_id = last_id,
        "maple generate_greedy: prompt cache EXACT HIT"
    );

    step_fn(steps.push_mut(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    }));
    token_history.push(last_id);

    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    let (stats, post) = {
        let mut ctx = DecodeCtx {
            tokenizer,
            vocab,
            n_tokens,
            device,
            eos_ids,
            step_fn,
            constraint: constraint.take(),
            sampler_cfg,
            rng,
            penalty_cfg,
            token_history,
            arch: ARCH,
            resolve_pieces: false,
        };
        pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
            model.forward_step(y, &mut caches, device)
        })?
    };

    let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
    let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
    let step_ms = (stats.step_total_ns as f64) / 1.0e6;
    let n = f64::from(stats.decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = ARCH,
        n_steps = stats.decode_steps,
        prefill_ms = 0.0_f64,
        forward_total_ms = forward_ms,
        eval_total_ms = eval_ms,
        step_total_ms = step_ms,
        forward_per_step_ms = forward_ms / n,
        eval_per_step_ms = eval_ms / n,
        cache_path = "exact",
        "decode_profile"
    );

    model
        .kv_bytes
        .store(caches.iter().map(KvCache::resident_bytes).sum(), post);

    Ok(steps)
}
