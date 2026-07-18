//! Qwen2 greedy generation loop.

#![allow(clippy::too_many_arguments)]
#![allow(
    clippy::cloned_instead_of_copied,
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::items_after_statements,
    clippy::redundant_closure_for_method_calls,
    clippy::struct_field_names,
    clippy::too_many_lines,
    clippy::indexing_slicing
)]

use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, Array, Device, Dtype};

use crate::constraint::ConstraintEngine;
use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};
use crate::prompt_cache::{chained_block_hashes_seeded, Consumed, ReusePolicy, FNV_OFFSET};
use crate::sampler::apply_mask_argmax;
use rmlx_kv_quant::{KvCache, KV_MAX_SEQ_DEFAULT};

use super::model::Qwen2Text;
use super::prompt_cache::{
    active_layout_key, ensure_prompt_cache, store_kv_cache_bytes, Qwen2Entry, PROMPT_CACHE,
};

// ---------------------------------------------------------------------------
// Smoke probe — generate_greedy
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
/// Returns `Vec<ProbeStep>` — same shape as `gemma4::generate_greedy`.
///
/// `prompt_cache_slots` — number of post-prefill KV snapshots kept across
/// requests. Pass 1 for single-slot; pass N for multi-slot prefix matching.
/// Recommended: 4. Only the Exact hit path is active (identical-prompt repeat
/// skips re-prefill entirely, same `ReusePolicy::ExactOnly` contract as Qwen3
/// dense).
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
    model: &Qwen2Text,
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
        arch = "Qwen2ForCausalLM",
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
    // Prompt cache lookup via the shared consume() engine. Qwen2 is a dense
    // pure-attention arch with no recurrent state and uses
    // ReusePolicy::ExactOnly (it overrides none of the reuse hooks), so the only
    // reachable outcomes are Exact (identical-prompt repeat skips re-prefill) and
    // Miss (full re-prefill). The engine owns the find → SSD-hydrate retry →
    // quant-mismatch guard → SSD-hydrated exclusion → Exact decision and traces
    // every degrade branch. The ExactOnly policy tripwire lives here at the call
    // site — not in the generic engine — because the engine is policy-agnostic
    // and shared across architectures that may use different policies.
    // ------------------------------------------------------------------
    assert_eq!(
        PROMPT_CACHE.policy(),
        ReusePolicy::ExactOnly,
        "Qwen2 prompt cache must be ExactOnly — pure-attention with no recurrent \
         state and an Exact-hit-dominated workload; partial-prefix reuse is not wired",
    );
    // Qwen2 is text-only (no vision tower), so there is never an image prompt;
    // the engine's has_image bypass is belt-and-suspenders.
    let consumed = PROMPT_CACHE.consume(prompt_ids, kv_quant, false);

    // Path A: exact cache hit — skip re-prefill, replay the stored first token,
    // then run the shared decode loop on the cloned caches.
    if let Consumed::Exact(cloned) = consumed {
        let Qwen2Entry {
            kv_caches: mut caches,
            first_id: last_id,
            first_piece: piece,
            ..
        } = cloned;
        tracing::debug!(
            prompt_len = prompt_ids.len(),
            token_id = last_id,
            "qwen2 generate_greedy: prompt cache EXACT HIT"
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
            store_kv_cache_bytes(kv_bytes, post);
        }
        log_decode_profile(
            0.0,
            forward_total_ns,
            eval_total_ns,
            step_total_ns,
            decode_steps,
        );
        return Ok(steps);
    }

    // Path B (Miss): full re-prefill from scratch.
    let prefill_t0 = Instant::now();

    let max_seq = max_ctx_override.unwrap_or(KV_MAX_SEQ_DEFAULT);

    // Allocate one KvCache per decoder layer using the selected quant mode.
    // Force K8V8 for boundary layers (first head_n + last tail_n).
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
            KvCache::with_quant_max_seq(q, max_seq).with_layer_idx(i)
        })
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
    // Chunk size is per-arch; default 256 for qwen2, override via
    // `RMLX_PREFILL_CHUNK` (global) or `RMLX_PREFILL_CHUNK_QWEN2` (per-arch).
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen2");
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
                                    "qwen2 generate_greedy: prefill chunk cache eval failed, \
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
                        "qwen2 generate_greedy: prefill chunk failed, aborting generation"
                    );
                    return Err(e);
                }
            }
        }
        for c in &mut caches {
            if let Err(e) = c.exit_prefill(device) {
                tracing::error!(
                    error = %e,
                    "qwen2 generate_greedy: exit_prefill quantization failed, aborting generation"
                );
                return Err(e);
            }
        }
        let Some(l) = last_logits else {
            return Err(Error::Other(
                "qwen2 generate_greedy: prefill produced no logits (empty prompt)".to_owned(),
            ));
        };
        l
    };

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());

    // A6.2 masked-argmax fork (first emit).
    // A7.2: temp<=0 keeps the exact greedy match below; temp>0 host-samples.
    let sampling_active = sampler_cfg.sampling_active();
    let penalties_active = penalty_cfg.penalties_active();
    // A7.3: trailing-20 window (empty at prefill step 0).
    let win_start = token_history.len().saturating_sub(20);
    let recent = &token_history[win_start..];
    let top = if sampling_active {
        let mask_opt: Option<&[bool]> = constraint
            .as_mut()
            .map(|c| c.step_mask(vocab as usize) as &[bool]);
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
            let mask_opt: Option<&[bool]> = constraint
                .as_mut()
                .map(|c| c.step_mask(vocab as usize) as &[bool]);
            crate::sampler::argmax_with_penalties(
                &logits_flat,
                mask_opt,
                penalty_cfg,
                recent,
                device,
            )?
        } else {
            match constraint.as_mut() {
                None => argmax(&logits_flat, -1, device)?,
                Some(c) => {
                    let m = c.step_mask(vocab as usize);
                    apply_mask_argmax(&logits_flat, m, device)?
                }
            }
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
        "qwen2 generate_greedy prefill"
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
    // different tokio/Metal thread can re-eval them as a no-op (see qwen3.rs).
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
                        FNV_OFFSET ^ lk ^ kv_quant.cache_key_salt(),
                    );
                    let entry = Qwen2Entry {
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
                                    "qwen2 generate_greedy: pushed snapshot to prompt cache (miss path)"
                                );
                            }
                        }
                    });
                }
                Err(e) => tracing::warn!(error = %e,
                    "qwen2 generate_greedy: pre-eval of KV caches failed, skipping prompt cache store"),
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
        store_kv_cache_bytes(kv_bytes, post);
    }

    log_decode_profile(
        (prefill_total_ns as f64) / 1.0e6,
        forward_total_ns,
        eval_total_ns,
        step_total_ns,
        decode_steps,
    );

    Ok(steps)
}

/// Shared pipelined decode loop for both the Exact-hit and Miss paths.
///
/// `last_id` is the already-emitted step-0 token (prefill argmax or replayed
/// cache first token); `steps` already holds its `ProbeStep`. This drives steps
/// `1..n_tokens` with the single-slot async_eval pipeline and the final drain,
/// accumulating decode timers into the caller's counters.
///
/// async_eval + single-slot pending pipeline (mirrors qwen3.rs). Each iteration
/// dispatches step i+1's forward while step i's argmax materialises in the
/// background; GPU sync only happens on the *previous* step's `pending` via
/// `to_bytes()`. The `last_id` u32 is only used to build the initial Array for
/// the very first decode step.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: argmax byte slices are sized 4 by the I32 dtype; loop indices bounded by slice length"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Option/Result unwrap is on values established by construction earlier in this fn (pending pipeline / try_into on a 4-byte slice)"
)]
fn decode_loop(
    model: &Qwen2Text,
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

    let mut y: Array = {
        let id_i32 = last_id as i32;
        let bytes = id_i32.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::I32)?
    };
    y.eval()?;
    let mut pending: Option<Array> = None;
    let mut early_stop = false;

    for step_idx in 1..n_tokens {
        let step_t0 = Instant::now();
        let fwd_t0 = Instant::now();
        // A failed forward aborts the request: returning the tokens produced so
        // far reports as a normal short generation and hides the failure from
        // the caller. Propagate to the server's error channel.
        let decode_logits = match model.forward_arr(&y, 1, Some(&mut *caches), device) {
            Ok(l) => l,
            Err(e) => {
                tracing::error!(
                    step = step_idx,
                    emitted = steps.len(),
                    error = %e,
                    "qwen2 generate_greedy: decode step failed, aborting generation"
                );
                return Err(e);
            }
        };

        let logits_flat = decode_logits.reshape(&[1, vocab], device)?;
        // A6.3: only run the masked pre-drain pipeline when the engine is
        // actively masking (wants_mask=true). During warm-up the engine
        // is inert; we fall through to the unconstrained pipeline so
        // argmax dtype + pipelining are bit-identical to the no-constraint
        // path. `advance()` is still called from the post-drain branch so
        // warm-up engagement detection works.
        let mask_active = constraint.as_deref().is_some_and(|c| c.wants_mask());
        // A7.2: temp>0 reads logits to host each step (no async pipelining
        // benefit), so it shares the masked branch's pre-drain. temp<=0
        // keeps the exact `mask_active`-gated pipelined path byte-for-byte.
        let sampling_active = sampler_cfg.sampling_active();
        // A7.3: penalties also require logits on host → fold into drain_now.
        let penalties_active = penalty_cfg.penalties_active();
        let drain_now = mask_active || sampling_active || penalties_active;
        // Force eager eval of logits on the masked path to prevent stale GPU data.
        if mask_active {
            logits_flat.eval()?;
        }
        let fwd_dt = fwd_t0.elapsed().as_nanos();

        let eval_t0 = Instant::now();
        let pre_drain_eos = if drain_now {
            if let Some(p) = pending.take() {
                let top_bytes = p.to_bytes()?;
                let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                if let Some(c) = constraint.as_deref_mut() {
                    c.advance(next_id);
                }
                // A7.3: accumulate emitted token into history.
                token_history.push(next_id);
                let piece = tokenizer
                    .id_to_token(next_id)
                    .unwrap_or_else(|| format!("<unk:{next_id}>"));
                tracing::debug!(
                    step = step_idx - 1,
                    token_id = next_id,
                    piece = %piece,
                    "qwen2 generate_greedy decode step (pre-drain emit)"
                );
                steps.push(ProbeStep {
                    token_id: next_id,
                    piece: piece.into_boxed_str(),
                    max_abs_logit: 0.0,
                    nan_count: 0,
                    logprobs: None,
                });
                step_fn(steps.last().unwrap());
                eos_ids.contains(&next_id)
            } else {
                false
            }
        } else {
            false
        };
        if pre_drain_eos {
            early_stop = true;
            tracing::debug!(
                step = step_idx - 1,
                "qwen2 generate_greedy: EOS emitted (pre-drain, masked), stopping"
            );
            break;
        }
        // A7.3: trailing-20 window for penalty context.
        let win_start = token_history.len().saturating_sub(20);
        let recent = &token_history[win_start..];
        let next_y = if sampling_active {
            let mask_opt: Option<&[bool]> = if mask_active {
                Some(constraint.as_deref_mut().unwrap().step_mask(vocab as usize))
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
                    Some(constraint.as_deref_mut().unwrap().step_mask(vocab as usize))
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
                let c = constraint.as_deref_mut().unwrap();
                let m = c.step_mask(vocab as usize);
                apply_mask_argmax(&logits_flat, m, device)?
            } else {
                argmax(&logits_flat, -1, device)?
            }
        };
        let _ = next_y.async_eval();
        let fwd_dt_total = fwd_dt;

        // Consume the previous step's pending argmax (unconstrained branch +
        // warm-up branch). On the masked / sampling / penalty branch we
        // already drained above.
        let mut emitted_eos = false;
        if !drain_now {
            if let Some(p) = pending.take() {
                let top_bytes = p.to_bytes()?;
                let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                if let Some(c) = constraint.as_deref_mut() {
                    c.advance(next_id);
                }
                // A7.3: accumulate pipelined token into history.
                token_history.push(next_id);
                let piece = tokenizer
                    .id_to_token(next_id)
                    .unwrap_or_else(|| format!("<unk:{next_id}>"));
                tracing::debug!(
                    step = step_idx - 1,
                    token_id = next_id,
                    piece = %piece,
                    "qwen2 generate_greedy decode step (pipelined emit)"
                );
                steps.push(ProbeStep {
                    token_id: next_id,
                    piece: piece.into_boxed_str(),
                    max_abs_logit: 0.0,
                    nan_count: 0,
                    logprobs: None,
                });
                step_fn(steps.last().unwrap());
                emitted_eos = eos_ids.contains(&next_id);
            }
        }
        let eval_dt = eval_t0.elapsed().as_nanos();

        *forward_total_ns += fwd_dt_total;
        *eval_total_ns += eval_dt;
        *step_total_ns += step_t0.elapsed().as_nanos();
        *decode_steps += 1;

        if emitted_eos {
            early_stop = true;
            tracing::debug!(
                step = step_idx - 1,
                "qwen2 generate_greedy: EOS emitted, stopping decode loop"
            );
            break;
        }

        // Feed next step's forward with the (still pending) argmax Array.
        // try_clone shares the underlying data; pending keeps ownership of
        // the GPU buffer until the next iteration consumes it.
        y = next_y.try_clone()?;
        pending = Some(next_y);
    }

    // Drain any final pending token that has not yet been emitted.
    // Skip drain on early_stop.
    if !early_stop {
        if let Some(p) = pending {
            let drain_t0 = Instant::now();
            p.eval()?;
            let top_bytes = p.to_bytes()?;
            *eval_total_ns += drain_t0.elapsed().as_nanos();
            let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
            if let Some(c) = constraint.as_mut() {
                c.advance(next_id);
            }
            // A7.3: final drain token into history.
            token_history.push(next_id);
            let piece = tokenizer
                .id_to_token(next_id)
                .unwrap_or_else(|| format!("<unk:{next_id}>"));
            steps.push(ProbeStep {
                token_id: next_id,
                piece: piece.into_boxed_str(),
                max_abs_logit: 0.0,
                nan_count: 0,
                logprobs: None,
            });
            step_fn(steps.last().unwrap());
        }
    }

    // Final act of the decode phase: mint the post-decode witness for the
    // caller's `store_kv_cache_bytes`.
    Ok(crate::decode_loop::PostDecode::seal())
}

/// Emit the `decode_profile` info event for one generate call.
///
/// `prefill_ms` is `0.0` on the Exact-hit path (no prefill ran).
fn log_decode_profile(
    prefill_ms: f64,
    forward_total_ns: u128,
    eval_total_ns: u128,
    step_total_ns: u128,
    decode_steps: u32,
) {
    let forward_ms = (forward_total_ns as f64) / 1.0e6;
    let eval_ms = (eval_total_ns as f64) / 1.0e6;
    let step_ms = (step_total_ns as f64) / 1.0e6;
    let n = f64::from(decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = "Qwen2ForCausalLM",
        n_steps = decode_steps,
        prefill_ms,
        forward_total_ms = forward_ms,
        eval_total_ms = eval_ms,
        step_total_ms = step_ms,
        forward_per_step_ms = forward_ms / n,
        eval_per_step_ms = eval_ms / n,
        "decode_profile"
    );
}
