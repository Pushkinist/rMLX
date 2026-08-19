//! Laguna greedy autoregressive generation loop.
//!
//! [`generate_greedy`] drives prefill and decode for the Laguna architecture,
//! including chunked prefill, KV-cache management, and streaming token delivery.
//!
//! # Public API
//!
//! - [`generate_greedy`] — main generation entry point.

#![allow(
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines
)]
use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, Array, Device, Dtype};

use crate::constraint::ConstraintEngine;
use crate::decode_loop::reject_nan_prefill;
use crate::kv_cache::kv_layer_quants;
use crate::prompt_cache::{chained_block_hashes_seeded, Consumed, ReusePolicy};
use crate::sampler::apply_mask_argmax;
use rmlx_kv_quant::{KvCache, KV_MAX_SEQ_DEFAULT};

use super::model::LagunaText;
use super::prompt_cache::{active_layout_key, ensure_prompt_cache, LagunaEntry, PROMPT_CACHE};

// ---------------------------------------------------------------------------
// Smoke probe -- generate_greedy
// ---------------------------------------------------------------------------

/// Count NaN values in a byte buffer of floats (F32 or Bf16).
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn count_nan_in_bytes(bytes: &[u8], dtype: Dtype) -> usize {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .filter(|c| f32::from_le_bytes((*c).try_into().unwrap()).is_nan())
            .count(),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .filter(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                f32::from_bits(u32::from(raw) << 16).is_nan()
            })
            .count(),
        _ => 0,
    }
}

/// Compute max(|logit|) from a byte buffer. Returns 0.0 on empty or unknown dtype.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
fn max_abs_from_bytes(bytes: &[u8], dtype: Dtype) -> f32 {
    match dtype {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes((*c).try_into().unwrap()).abs())
            .fold(0.0_f32, f32::max),
        Dtype::Bf16 => bytes
            .chunks_exact(2)
            .map(|c| {
                let raw = u16::from_le_bytes((*c).try_into().unwrap());
                f32::from_bits(u32::from(raw) << 16).abs()
            })
            .fold(0.0_f32, f32::max),
        _ => 0.0,
    }
}

/// Greedy autoregressive generation using KV-cache prefill + decode.
///
/// Returns `Vec<ProbeStep>` -- same shape as `gemma4::generate_greedy`.
///
/// `prompt_cache_slots` — number of post-prefill KV snapshots kept across
/// requests. Pass 1 for single-slot; pass N for multi-slot prefix matching.
/// Recommended: 4. Only the Exact hit path is active (identical-prompt repeat
/// skips re-prefill entirely, same `ReusePolicy::ExactOnly` contract as Qwen2
/// and BitNet).
/// Pass 0 to disable the cache: nothing is stored, so every request prefills.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub fn generate_greedy(
    model: &LagunaText,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: rmlx_kv_quant::KvQuant,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
    // A6.2: optional sampler constraint. See gemma4::generate_greedy.
    mut constraint: Option<&mut dyn ConstraintEngine>,
    // A7.2: sampling config + per-request RNG. See gemma3::generate_greedy.
    sampler_cfg: &crate::sampler::SamplerConfig,
    rng: &mut crate::sampler::Pcg32,
    // A7.3: logit-penalty configuration + per-request token history.
    penalty_cfg: &crate::sampler::PenaltyConfig,
    token_history: &mut Vec<u32>,
) -> Result<Vec<crate::decode_loop::ProbeStep>> {
    use crate::decode_loop::ProbeStep;

    tracing::info!(
        arch = "LagunaForCausalLM",
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

    // Decode profile timers. See gemma4::generate_greedy for rationale.
    let mut forward_total_ns: u128 = 0;
    let mut eval_total_ns: u128 = 0;
    let mut step_total_ns: u128 = 0;
    let mut decode_steps: u32 = 0;

    // ------------------------------------------------------------------
    // Prompt cache lookup via the shared consume() engine. Laguna is a
    // pure-attention sparse-MoE arch with no recurrent state and uses
    // ReusePolicy::ExactOnly (it overrides none of the reuse hooks), so the
    // only reachable outcomes are Exact (identical-prompt repeat skips
    // re-prefill) and Miss (full re-prefill). The engine owns the find →
    // SSD-hydrate retry → quant-mismatch guard → SSD-hydrated exclusion →
    // Exact decision and traces every degrade branch. The ExactOnly policy
    // tripwire lives here at the call site — not in the generic engine —
    // because the engine is policy-agnostic and shared across architectures
    // that may use different policies.
    // ------------------------------------------------------------------
    assert_eq!(
        PROMPT_CACHE.policy(),
        ReusePolicy::ExactOnly,
        "Laguna prompt cache must be ExactOnly — pure-attention with no recurrent \
         state and an Exact-hit-dominated workload; partial-prefix reuse is not wired",
    );
    // Laguna is text-only (no vision tower), so there is never an image prompt;
    // the engine's has_image bypass is belt-and-suspenders.
    let consumed = PROMPT_CACHE.consume(
        prompt_ids,
        kv_quant,
        model.cfg.num_hidden_layers,
        false,
        model.model_sig,
    );

    // Path A: exact cache hit — skip re-prefill, replay the stored first token,
    // then run the shared decode loop on the cloned caches.
    if let Consumed::Exact(cloned) = consumed {
        let LagunaEntry {
            kv_caches: mut caches,
            first_id: last_id,
            first_piece: piece,
            ..
        } = cloned;
        tracing::debug!(
            prompt_len = prompt_ids.len(),
            token_id = last_id,
            "laguna generate_greedy: prompt cache EXACT HIT"
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

        let post = decode_loop(
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
            &mut eval_total_ns,
            &mut step_total_ns,
            &mut decode_steps,
        )?;

        {
            let kv_bytes: u64 = caches.iter().map(|c| c.resident_bytes()).sum();
            model.kv_bytes.store(kv_bytes, post);
        }
        tracing::info!(
            arch = "LagunaForCausalLM",
            total_tokens = steps.len(),
            decode_steps,
            avg_decode_ns = if decode_steps > 0 {
                forward_total_ns / u128::from(decode_steps)
            } else {
                0
            },
            "generate_greedy: complete (exact hit)"
        );
        return Ok(steps);
    }

    // Path B (Miss): full re-prefill from scratch.
    let prefill_t0 = Instant::now();
    let max_seq = max_ctx_override.unwrap_or(KV_MAX_SEQ_DEFAULT);

    // Allocate one KvCache per decoder layer using the selected quant mode.
    // Force K8V8 for boundary layers (first head_n + last tail_n).
    let n_layers = model.cfg.num_hidden_layers;
    let mut caches: Vec<KvCache> = kv_layer_quants(n_layers, kv_quant)
        .into_iter()
        .enumerate()
        .map(|(i, q)| KvCache::with_quant_max_seq(q, max_seq).with_layer_idx(i))
        .collect();

    // Prefill: encode the prompt in fixed-size chunks. Per chunk we
    // eval only the KV-cache prefill_raw buffers, not the logits, so MLX
    // lazily skips the lm_head matmul on every non-final chunk. The Metal
    // command buffer still flushes between chunks via the cache evals.
    //
    // enter_prefill() / exit_prefill() bracket the loop so K/V are stored
    // as raw BF16 during chunked prefill instead of being
    // quantize-dequantized on every chunk.
    //
    // Chunk size is per-arch; default 256 for laguna (preserves pre-tuning
    // value, no laguna bench data yet), override via `RMLX_PREFILL_CHUNK`
    // (global) or `RMLX_PREFILL_CHUNK_LAGUNA` (per-arch).
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("laguna");
    for c in &mut caches {
        c.enter_prefill();
    }
    // A failed prefill aborts the request. Returning the (empty) step list
    // instead reaches the operator as the engine's generic zero-token backstop,
    // which names nothing — the real cause (a Metal fault, a store refusing an
    // append) is erased. Propagate it verbatim.
    let prefill_logits = {
        let mut last_logits: Option<Array> = None;
        let n_chunks = prompt_ids.len().div_ceil(prefill_chunk);
        for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            match model.forward_seq_with_cache(chunk, Some(&mut caches), device) {
                Ok(logits) => {
                    if is_last {
                        last_logits = Some(logits);
                    } else {
                        for c in &caches {
                            if let Err(e) = c.eval_prefill_state() {
                                tracing::error!(
                                    error = %e,
                                    chunk_len = chunk.len(),
                                    "laguna generate_greedy: prefill chunk cache eval failed, \
                                     aborting generation"
                                );
                                return Err(e);
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::error!(
                        error = %e,
                        prompt_len = prompt_ids.len(),
                        "laguna generate_greedy: prefill chunk failed, aborting generation"
                    );
                    return Err(e);
                }
            }
        }
        for c in &mut caches {
            if let Err(e) = c.exit_prefill(device) {
                tracing::error!(
                    error = %e,
                    "laguna generate_greedy: exit_prefill quantization failed, aborting generation"
                );
                return Err(e);
            }
        }
        let Some(l) = last_logits else {
            return Err(Error::Other(
                "laguna generate_greedy: prefill produced no logits (empty prompt)".to_owned(),
            ));
        };
        l
    };

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());
    reject_nan_prefill(
        "LagunaForCausalLM",
        logits_flat.dtype(),
        nan_count,
        max_abs_logit,
        prompt_ids.len(),
    )?;

    // A6.2 masked-argmax fork (first emit).
    let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
    // A7.2: temp<=0 keeps the exact greedy block below; temp>0 host-samples.
    let sampling_active = sampler_cfg.sampling_active();
    let penalties_active = penalty_cfg.penalties_active();
    // A7.3: trailing-20 window (empty at prefill step 0).
    let win_start = token_history.len().saturating_sub(20);
    let recent = &token_history[win_start..];
    let top = if sampling_active {
        let mask_opt: Option<&[bool]> = if mask_active {
            Some(constraint.as_mut().unwrap().step_mask(vocab as usize))
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
    } else {
        if penalties_active {
            let mask_opt: Option<&[bool]> = if mask_active {
                Some(constraint.as_mut().unwrap().step_mask(vocab as usize))
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
            let c = constraint.as_mut().unwrap();
            let m = c.step_mask(vocab as usize);
            apply_mask_argmax(&logits_flat, m, device)?
        } else {
            argmax(&logits_flat, -1, device)?
        }
    };
    top.eval()?;
    let top_bytes = top.to_bytes()?;
    let last_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    // A6.2: advance constraint with emitted id.
    if let Some(c) = constraint.as_mut() {
        c.advance(last_id);
    }
    // A7.3: push prefill token into history.
    token_history.push(last_id);
    let prefill_total_ns = prefill_t0.elapsed().as_nanos();

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
        "laguna generate_greedy prefill"
    );

    steps.push(ProbeStep {
        token_id: last_id,
        piece: piece.clone().into_boxed_str(),
        max_abs_logit,
        nan_count,
        logprobs: None,
    });
    step_fn(steps.last().unwrap());

    // Push this prefill snapshot to the prompt cache (Miss → store). Clone the
    // post-prefill KV caches (refcount bump, no data copy) before the decode
    // loop starts writing new decode-step K/V into them. Materialize the GPU
    // arrays on the current inference thread first so a later eviction on a
    // different tokio/Metal thread can re-eval them as a no-op.
    //
    // `kv_cache_bytes` is NOT sampled here — this is the prefill snapshot,
    // before the decode ring is allocated. It is recorded post-decode below.
    {
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
                        crate::prompt_cache::request_cache_seed(
                            lk,
                            kv_quant,
                            model.cfg.num_hidden_layers,
                            model.model_sig,
                        ),
                    );
                    let entry = LagunaEntry {
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
                            if cache.push(entry).is_some() {
                                let stats = cache.stats();
                                tracing::debug!(
                                    prompt_len = prompt_ids.len(),
                                    cache_hits = stats.hits,
                                    cache_misses = stats.misses,
                                    cache_bytes = stats.bytes,
                                    "laguna generate_greedy: pushed snapshot to prompt cache (miss path)"
                                );
                            }
                        }
                    });
                }
                Err(e) => tracing::warn!(error = %e,
                    "laguna generate_greedy: pre-eval of KV caches failed, skipping prompt cache store"),
            }
        }
    }

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    let post = decode_loop(
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
        &mut eval_total_ns,
        &mut step_total_ns,
        &mut decode_steps,
    )?;

    // Store KV-cache bytes post-decode: the decode ring is resident now, so the
    // sample includes it on ring-backed codecs. Same lifecycle point as the
    // exact-hit path above.
    {
        let kv_bytes: u64 = caches.iter().map(|c| c.resident_bytes()).sum();
        model.kv_bytes.store(kv_bytes, post);
    }

    let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
    let forward_ms = (forward_total_ns as f64) / 1.0e6;
    let eval_ms = (eval_total_ns as f64) / 1.0e6;
    let step_ms = (step_total_ns as f64) / 1.0e6;
    let n = f64::from(decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = "LagunaForCausalLM",
        n_steps = decode_steps,
        prefill_ms,
        forward_total_ms = forward_ms,
        eval_total_ms = eval_ms,
        step_total_ms = step_ms,
        forward_per_step_ms = forward_ms / n,
        eval_per_step_ms = eval_ms / n,
        "decode_profile"
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
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "steps.last().unwrap() immediately preceded by steps.push(...), so Vec is non-empty"
)]
fn decode_loop(
    model: &LagunaText,
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
    eval_total_ns: &mut u128,
    step_total_ns: &mut u128,
    decode_steps: &mut u32,
) -> Result<crate::decode_loop::PostDecode> {
    use crate::decode_loop::ProbeStep;

    // The starting token for step 1 is the last emitted (step 0) token.
    let mut cur_id = last_id;

    // Decode: one token at a time.
    for step_idx in 1..n_tokens {
        let step_t0 = Instant::now();
        let fwd_t0 = Instant::now();
        // A failed forward aborts the request: returning the tokens produced so
        // far reports as a normal short generation and hides the failure from
        // the caller. Propagate to the server's error channel.
        let decode_logits = match model.forward_seq_with_cache(&[cur_id], Some(caches), device) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(
                    step = step_idx,
                    emitted = steps.len(),
                    error = %e,
                    "laguna generate_greedy: decode step failed, aborting generation"
                );
                return Err(e);
            }
        };
        let fwd_dt = fwd_t0.elapsed().as_nanos();

        let eval_t0 = Instant::now();
        let logits_flat = decode_logits.reshape(&[1, vocab], device)?;
        logits_flat.eval()?;

        // NaN mid-decode aborts the request, exactly as a failed forward does
        // twenty lines above. Truncating instead — emitting the junk token this
        // row selects and breaking — hands back a short generation the server
        // reports as finish_reason="length", indistinguishable from a clean
        // token-cap stop, with one fabricated token on the end.
        //
        // Unlike the prefill sites, `steps.len()` tokens are already delivered
        // here, so the "nothing has reached the wire" argument does not apply.
        // What holds instead: the guard fires before THIS step's token is
        // pushed, so the delivered prefix is entirely healthy, and at temp=0 a
        // replay reproduces it and continues past the fault point. The retry
        // envelope's prefix-identity assertion is what covers the case where it
        // does not — see docs/SERVER.md § Retry Envelope.
        let logit_bytes = logits_flat.to_bytes()?;
        let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
        let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());
        if nan_count > 0 {
            let e = Error::Other(format!(
                "laguna generate_greedy: decode logits contain {nan_count} NaN cells at step \
                 {step_idx} (max|logit| = {max_abs_logit}), aborting generation"
            ));
            tracing::error!(
                error = %e,
                step = step_idx,
                emitted = steps.len(),
                nan_count,
                max_abs_logit,
                "laguna generate_greedy: NaN in decode logits, aborting generation"
            );
            return Err(e);
        }

        // A6.3: only apply mask when engine is engaged (wants_mask).
        let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
        // A7.2: temp<=0 keeps the exact greedy block below; temp>0 host-samples.
        let sampling_active = sampler_cfg.sampling_active();
        let penalties_active = penalty_cfg.penalties_active();
        // A7.3: trailing-20 window for penalty context.
        let win_start = token_history.len().saturating_sub(20);
        let recent = &token_history[win_start..];
        let top = if sampling_active {
            let mask_opt: Option<&[bool]> = if mask_active {
                Some(constraint.as_mut().unwrap().step_mask(vocab as usize))
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
        } else {
            if penalties_active {
                let mask_opt: Option<&[bool]> = if mask_active {
                    Some(constraint.as_mut().unwrap().step_mask(vocab as usize))
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
                let c = constraint.as_mut().unwrap();
                let m = c.step_mask(vocab as usize);
                apply_mask_argmax(&logits_flat, m, device)?
            } else {
                argmax(&logits_flat, -1, device)?
            }
        };
        top.eval()?;
        let top_bytes = top.to_bytes()?;
        let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
        // A6.2: advance constraint with emitted id.
        if let Some(c) = constraint.as_mut() {
            c.advance(next_id);
        }
        // A7.3: accumulate emitted token into history.
        token_history.push(next_id);
        let eval_dt = eval_t0.elapsed().as_nanos();
        *forward_total_ns += fwd_dt;
        *eval_total_ns += eval_dt;
        *step_total_ns += step_t0.elapsed().as_nanos();
        *decode_steps += 1;

        let piece = tokenizer
            .id_to_token(next_id)
            .unwrap_or_else(|| format!("<unk:{next_id}>"));

        tracing::debug!(
            step = step_idx,
            token_id = next_id,
            piece = %piece,
            max_abs_logit,
            nan_count,
            "laguna generate_greedy decode step"
        );

        steps.push(ProbeStep {
            token_id: next_id,
            piece: piece.into_boxed_str(),
            max_abs_logit,
            nan_count,
            logprobs: None,
        });
        step_fn(steps.last().unwrap());

        // EOS-stop in the decode loop.
        if eos_ids.contains(&next_id) {
            tracing::debug!(
                step = step_idx,
                token_id = next_id,
                "laguna generate_greedy: EOS emitted, stopping decode loop"
            );
            break;
        }

        cur_id = next_id;
    }

    // Final act of the decode phase: mint the post-decode witness for the
    // caller's `store_kv_cache_bytes`.
    Ok(crate::decode_loop::PostDecode::seal())
}
