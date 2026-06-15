//! BitNet greedy generation loop.

#![allow(
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::items_after_statements,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines,
    reason = "greedy decode loop: large branchy state machine over sampler/penalty/mask combos; splitting hides the per-step decision tree"
)]

use std::time::Instant;

use rmlx_core::error::Result;
use rmlx_mlx::{argmax, Array, Device, Dtype};
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};
use tracing::{info, warn};

use crate::constraint::ConstraintEngine;
use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};
use crate::prompt_cache::{chained_block_hashes_seeded, Consumed, ReusePolicy, FNV_OFFSET};
use crate::sampler::apply_mask_argmax;
use rmlx_kv_quant::{KvCache, KV_MAX_SEQ_DEFAULT};

use super::model::BitNetText;
use super::prompt_cache::{
    active_layout_key, ensure_prompt_cache, store_kv_cache_bytes, BitNetEntry, PROMPT_CACHE,
};

// ---------------------------------------------------------------------------
// Greedy generation
// ---------------------------------------------------------------------------

/// Greedy autoregressive generation for BitNetForCausalLM.
///
/// Returns `Vec<ProbeStep>` — same shape as `gemma4::generate_greedy`.
///
/// `prompt_cache_slots` — number of post-prefill KV snapshots kept across
/// requests. Pass 1 for single-slot; pass N for multi-slot prefix matching.
/// Recommended: 4. Only the Exact hit path is active (identical-prompt repeat
/// skips re-prefill entirely, same `ReusePolicy::ExactOnly` contract as Qwen2
/// and Qwen3 dense).
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "steps.last().unwrap() immediately preceded by steps.push(...), so Vec is non-empty"
)]
pub fn generate_greedy(
    model: &BitNetText,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: rmlx_kv_quant::KvQuant,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
    mut constraint: Option<&mut dyn ConstraintEngine>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    rng: &mut crate::sampler::Pcg32,
    penalty_cfg: &crate::sampler::PenaltyConfig,
    token_history: &mut Vec<u32>,
) -> Result<Vec<crate::decode_loop::ProbeStep>> {
    use crate::decode_loop::ProbeStep;

    tracing::info!(
        arch = "BitNetForCausalLM",
        ?kv_quant,
        ?max_ctx_override,
        "generate_greedy: selected KV cache quant"
    );

    if n_tokens == 0 {
        return Ok(vec![]);
    }

    let vocab = model.cfg.vocab_size as i32;
    let mut steps = Vec::with_capacity(n_tokens);

    ensure_prompt_cache(prompt_cache_slots);

    // Decode profile timers.
    let mut forward_total_ns: u128 = 0;
    let mut decode_steps_count: u32 = 0;

    // ------------------------------------------------------------------
    // Prompt cache lookup via the shared consume() engine. BitNet is a dense
    // pure-attention arch with no recurrent state and uses
    // ReusePolicy::ExactOnly (it overrides none of the reuse hooks), so the only
    // reachable outcomes are Exact (identical-prompt repeat skips re-prefill)
    // and Miss (full re-prefill). The engine owns the find → SSD-hydrate retry
    // → quant-mismatch guard → SSD-hydrated exclusion → Exact decision and
    // traces every degrade branch. The ExactOnly policy tripwire lives here at
    // the call site — not in the generic engine — because the engine is
    // policy-agnostic and shared across architectures that may use different
    // policies.
    // ------------------------------------------------------------------
    assert_eq!(
        PROMPT_CACHE.policy(),
        ReusePolicy::ExactOnly,
        "BitNet prompt cache must be ExactOnly — pure-attention with no recurrent \
         state and an Exact-hit-dominated workload; partial-prefix reuse is not wired",
    );
    // BitNet is text-only (no vision tower), so there is never an image prompt;
    // the engine's has_image bypass is belt-and-suspenders.
    let consumed = PROMPT_CACHE.consume(prompt_ids, kv_quant, false);

    // Path A: exact cache hit — skip re-prefill, replay the stored first token,
    // then run the shared decode loop on the cloned caches.
    if let Consumed::Exact(cloned) = consumed {
        let BitNetEntry {
            kv_caches: mut caches,
            first_id: last_id,
            first_piece: piece,
            ..
        } = cloned;
        tracing::debug!(
            prompt_len = prompt_ids.len(),
            token_id = last_id,
            "bitnet generate_greedy: prompt cache EXACT HIT"
        );
        steps.push(ProbeStep {
            token_id: last_id,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: None,
        });
        step_fn(steps.last().unwrap());
        token_history.push(last_id);

        if eos_ids.contains(&last_id) {
            return Ok(steps);
        }

        decode_loop(
            model,
            tokenizer,
            &mut caches,
            last_id,
            n_tokens,
            vocab,
            device,
            eos_ids,
            &mut steps,
            step_fn,
            &mut constraint,
            sampler_cfg,
            rng,
            penalty_cfg,
            token_history,
            &mut forward_total_ns,
            &mut decode_steps_count,
        )?;

        {
            let kv_bytes: u64 = caches.iter().map(|c| c.resident_bytes()).sum();
            store_kv_cache_bytes(kv_bytes);
        }
        info!(
            arch = "BitNetForCausalLM",
            total_tokens = steps.len(),
            decode_steps = decode_steps_count,
            "generate_greedy: complete (exact hit)"
        );
        return Ok(steps);
    }

    // Path B (Miss): full re-prefill from scratch.
    let prefill_t0 = Instant::now();
    let max_seq = max_ctx_override.unwrap_or(KV_MAX_SEQ_DEFAULT);

    let n_layers = model.cfg.num_hidden_layers;
    let mut caches: Vec<KvCache> = (0..n_layers)
        .map(|i| {
            let q = kv_quant_for_layer(
                i,
                n_layers,
                kv_quant,
                LAYER_ADAPTIVE_TAIL_N,
                LAYER_ADAPTIVE_HEAD_N,
            );
            KvCache::with_quant_max_seq_window(q, max_seq, None).with_layer_idx(i)
        })
        .collect();

    // Prefill in chunks.
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("bitnet");
    for c in &mut caches {
        c.enter_prefill();
    }

    let mut last_logits: Option<Array> = None;
    let mut prefill_ok = true;
    let n_chunks = prompt_ids.len().div_ceil(prefill_chunk);
    'prefill: for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {
        let is_last = chunk_idx + 1 == n_chunks;
        match model.forward_seq_with_cache(chunk, Some(&mut caches), device) {
            Ok(logits) => {
                if is_last {
                    last_logits = Some(logits);
                } else {
                    for c in &caches {
                        if let Err(e) = c.eval_prefill_state() {
                            warn!(
                                error = %e,
                                chunk_len = chunk.len(),
                                "bitnet generate_greedy: prefill chunk cache evaluation failed"
                            );
                            prefill_ok = false;
                            break 'prefill;
                        }
                    }
                }
            }
            Err(e) => {
                warn!(
                    error = %e,
                    prompt_len = prompt_ids.len(),
                    "bitnet generate_greedy: prefill chunk failed, returning empty"
                );
                prefill_ok = false;
                break 'prefill;
            }
        }
    }

    for c in &mut caches {
        if let Err(e) = c.exit_prefill(device) {
            warn!(error = %e, "bitnet generate_greedy: exit_prefill quantization failed");
            prefill_ok = false;
            break;
        }
    }

    if !prefill_ok {
        return Ok(steps);
    }
    let Some(prefill_logits) = last_logits else {
        return Ok(steps);
    };

    let prefill_ns = prefill_t0.elapsed().as_nanos();
    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());

    let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
    let sampling_active = sampler_cfg.sampling_active();
    let penalties_active = penalty_cfg.penalties_active();
    let win_start = token_history.len().saturating_sub(20);
    let recent = &token_history[win_start..];

    let top = if sampling_active {
        let mask_opt: Option<&[bool]> = if mask_active {
            if let Some(c) = constraint.as_mut() {
                Some(c.step_mask(vocab as usize))
            } else {
                None
            }
        } else {
            None
        };
        crate::sampler::sample_token_array(
            &logits_flat,
            sampler_cfg,
            mask_opt,
            penalty_cfg,
            recent,
            rng,
            device,
        )?
    } else if penalties_active {
        let mask_opt: Option<&[bool]> = if mask_active {
            if let Some(c) = constraint.as_mut() {
                Some(c.step_mask(vocab as usize))
            } else {
                None
            }
        } else {
            None
        };
        crate::sampler::argmax_with_penalties(&logits_flat, mask_opt, penalty_cfg, recent, device)?
    } else if mask_active {
        if let Some(c) = constraint.as_mut() {
            let m = c.step_mask(vocab as usize);
            apply_mask_argmax(&logits_flat, m, device)?
        } else {
            argmax(&logits_flat, -1, device)?
        }
    } else {
        argmax(&logits_flat, -1, device)?
    };

    top.eval()?;
    let top_bytes = top.to_bytes()?;
    let last_id =
        i32::from_le_bytes([top_bytes[0], top_bytes[1], top_bytes[2], top_bytes[3]]) as u32;

    if let Some(c) = constraint.as_mut() {
        c.advance(last_id);
    }
    token_history.push(last_id);

    let piece = tokenizer
        .id_to_token(last_id)
        .unwrap_or_else(|| format!("<unk:{last_id}>"));

    tracing::debug!(
        step = 0,
        token_id = last_id,
        piece = %piece,
        max_abs_logit,
        nan_count,
        prompt_len = prompt_ids.len(),
        prefill_ns,
        "bitnet generate_greedy prefill"
    );

    steps.push(ProbeStep {
        token_id: last_id,
        piece: piece.clone().into_boxed_str(),
        max_abs_logit,
        nan_count,
        logprobs: None,
    });
    step_fn(steps.last().unwrap());

    if nan_count > 0 {
        return Ok(steps);
    }

    // Push this prefill snapshot to the prompt cache (Miss → store). Clone the
    // post-prefill KV caches (refcount bump, no data copy) before the decode
    // loop starts writing new decode-step K/V into them. Materialize the GPU
    // arrays on the current inference thread first so a later eviction on a
    // different tokio/Metal thread can re-eval them as a no-op.
    {
        let kv_bytes: u64 = caches.iter().map(|c| c.resident_bytes()).sum();
        store_kv_cache_bytes(kv_bytes);
        let cloned_caches: Result<Vec<KvCache>> =
            caches.iter().map(|c| c.try_deep_clone()).collect();
        if let Ok(kv_snapshot) = cloned_caches {
            match kv_snapshot.iter().try_for_each(|c| c.eval_for_spill()) {
                Ok(()) => {
                    // Salt the chained block-hash walk with the active layout_key
                    // + KV codec so a slot stored under a different codec / layout
                    // never cross-serves. Identical to the consume() seed.
                    let lk = active_layout_key();
                    let block_hashes = chained_block_hashes_seeded(
                        prompt_ids,
                        FNV_OFFSET ^ lk ^ kv_quant.cache_key_salt(),
                    );
                    let entry = BitNetEntry {
                        prompt_token_ids: prompt_ids.to_vec(),
                        block_hashes,
                        kv_caches: kv_snapshot,
                        first_id: last_id,
                        // Last use of `piece` — move it (the prefill ProbeStep
                        // above already consumed its own clone via into_boxed_str).
                        first_piece: piece,
                        kv_quant: Some(kv_quant),
                        is_ssd_hydrated: false,
                    };
                    PROMPT_CACHE.with_inner_mut(|guard| {
                        if let Some(cache) = guard.as_mut() {
                            cache.push(entry);
                            let stats = cache.stats();
                            tracing::debug!(
                                prompt_len = prompt_ids.len(),
                                cache_hits = stats.hits,
                                cache_misses = stats.misses,
                                cache_bytes = stats.bytes,
                                "bitnet generate_greedy: pushed snapshot to prompt cache (miss path)"
                            );
                        }
                    });
                }
                Err(e) => tracing::warn!(error = %e,
                    "bitnet generate_greedy: pre-eval of KV caches failed, skipping prompt cache store"),
            }
        }
    }

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    decode_loop(
        model,
        tokenizer,
        &mut caches,
        last_id,
        n_tokens,
        vocab,
        device,
        eos_ids,
        &mut steps,
        step_fn,
        &mut constraint,
        sampler_cfg,
        rng,
        penalty_cfg,
        token_history,
        &mut forward_total_ns,
        &mut decode_steps_count,
    )?;

    info!(
        arch = "BitNetForCausalLM",
        total_tokens = steps.len(),
        decode_steps = decode_steps_count,
        avg_decode_ns = if decode_steps_count > 0 {
            forward_total_ns / u128::from(decode_steps_count)
        } else {
            0
        },
        "generate_greedy: complete"
    );

    Ok(steps)
}

/// Shared sequential decode loop for both the Exact-hit and Miss paths.
///
/// `last_id` is the already-emitted step-0 token (prefill argmax or replayed
/// cache first token); `steps` already holds its `ProbeStep`. This drives steps
/// `1..n_tokens`.
///
/// The per-step decode math is byte-identical to the original inline loop; only
/// the wrapping (function boundary + accumulated counters) differs.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: argmax byte slices are sized 4 by the I32 dtype; loop indices bounded by slice length"
)]
#[allow(
    clippy::unwrap_used,
    reason = "steps.last().unwrap() immediately preceded by steps.push(...), so Vec is non-empty"
)]
fn decode_loop(
    model: &BitNetText,
    tokenizer: &tokenizers::Tokenizer,
    caches: &mut [KvCache],
    last_id: u32,
    n_tokens: usize,
    vocab: i32,
    device: Device,
    eos_ids: &[u32],
    steps: &mut Vec<crate::decode_loop::ProbeStep>,
    step_fn: &mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
    constraint: &mut Option<&mut dyn ConstraintEngine>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    rng: &mut crate::sampler::Pcg32,
    penalty_cfg: &crate::sampler::PenaltyConfig,
    token_history: &mut Vec<u32>,
    forward_total_ns: &mut u128,
    decode_steps_count: &mut u32,
) -> Result<()> {
    use crate::decode_loop::ProbeStep;

    let mut y: Array = {
        let id_i32 = last_id as i32;
        let bytes = id_i32.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::I32)?
    };
    y.eval()?;

    for step_idx in 1..n_tokens {
        let fwd_t0 = Instant::now();
        let decode_logits = match model.forward_arr(&y, 1, Some(&mut *caches), device) {
            Ok(l) => l,
            Err(e) => {
                warn!(
                    step = step_idx,
                    error = %e,
                    "bitnet generate_greedy: decode step failed, stopping early"
                );
                break;
            }
        };
        *forward_total_ns += fwd_t0.elapsed().as_nanos();

        let logits_flat = decode_logits.reshape(&[1, vocab], device)?;

        let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
        let sampling_active = sampler_cfg.sampling_active();
        let penalties_active = penalty_cfg.penalties_active();

        if mask_active {
            logits_flat.eval()?;
        }

        let win_start = token_history.len().saturating_sub(20);
        let recent = &token_history[win_start..];

        let next_y = if sampling_active {
            let mask_opt: Option<&[bool]> = if mask_active {
                if let Some(c) = constraint.as_mut() {
                    Some(c.step_mask(vocab as usize))
                } else {
                    None
                }
            } else {
                None
            };
            crate::sampler::sample_token_array(
                &logits_flat,
                sampler_cfg,
                mask_opt,
                penalty_cfg,
                recent,
                rng,
                device,
            )?
        } else if penalties_active {
            let mask_opt: Option<&[bool]> = if mask_active {
                if let Some(c) = constraint.as_mut() {
                    Some(c.step_mask(vocab as usize))
                } else {
                    None
                }
            } else {
                None
            };
            crate::sampler::argmax_with_penalties(
                &logits_flat,
                mask_opt,
                penalty_cfg,
                recent,
                device,
            )?
        } else if mask_active {
            if let Some(c) = constraint.as_mut() {
                let m = c.step_mask(vocab as usize);
                apply_mask_argmax(&logits_flat, m, device)?
            } else {
                argmax(&logits_flat, -1, device)?
            }
        } else {
            argmax(&logits_flat, -1, device)?
        };

        next_y.eval()?;
        *decode_steps_count += 1;

        let top_bytes = next_y.to_bytes()?;
        let next_id =
            i32::from_le_bytes([top_bytes[0], top_bytes[1], top_bytes[2], top_bytes[3]]) as u32;

        if let Some(c) = constraint.as_mut() {
            c.advance(next_id);
        }
        token_history.push(next_id);

        let piece = tokenizer
            .id_to_token(next_id)
            .unwrap_or_else(|| format!("<unk:{next_id}>"));

        tracing::debug!(
            step = step_idx,
            token_id = next_id,
            piece = %piece,
            "bitnet generate_greedy decode step"
        );

        steps.push(ProbeStep {
            token_id: next_id,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: None,
        });
        step_fn(steps.last().unwrap());

        y = next_y;

        if eos_ids.contains(&next_id) {
            break;
        }
    }

    Ok(())
}
