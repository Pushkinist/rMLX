//! Autoregressive generation for Qwen3.5-MoE: chunked prefill and KV-cached decode.
//!
//! [`generate_greedy`] drives the full token-streaming pipeline for both
//! the standard `Qwen3_5MoeForConditionalGeneration` architecture and the
//! PARO-dense variant. Covers chunked prefill, thinking-budget enforcement,
//! speculative-draft integration, per-token sampling, and streaming delivery.
//!
//! # Public API
//!
//! - [`generate_greedy`] — main generation entry point.
//
// LOC-exempt: the engine deliberately holds three explicit, sequential serving
// paths inline — Path A (Exact cache hit), Path B (SSD HydratedTail), Path C
// (cold prefill) — each with its own prefill + decode loop. Per the project's
// "straight-forward core backend: sequential, sync, explicit" rule, these are
// kept unrolled rather than factored behind a shared decode helper; the small
// duplication is intentional and easier to audit than a parameterised loop.

#![allow(
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::items_after_statements,
    clippy::match_same_arms,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines
)]
use std::time::Instant;

use rmlx_core::error::Result;
use rmlx_mlx::{argmax, Array, Device, Dtype};

use crate::constraint::ConstraintEngine;
use crate::kv_cache::{
    kv_max_seq_and_ceiling, kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N,
};
use crate::sampler::{apply_mask_argmax, TokenLogprobs};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

use super::model::Qwen3_5MoeText;
use super::prompt_cache::{
    ensure_prompt_cache, store_kv_cache_bytes, Qwen35MoeEntry, PROMPT_CACHE,
};
use crate::prompt_cache::PromptCacheEntry as _;

/// capture top-`k` logprobs for an already-chosen token.
///
/// Called ONLY behind a `k > 0` guard at every decode-loop call site, so the
/// default decode path (no logprobs requested) never touches log-softmax /
/// top-k and stays byte-identical. `chosen` is the sampled/argmax token Array
/// (`[1] I32`); it is materialised here so we can pin the chosen token's own
/// logprob to the same logits the rest of `top` is computed from. Errors are
/// downgraded to `None` — logprob capture must never abort generation.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn capture_logprobs(logits_flat: &Array, chosen: &Array, k: usize) -> Option<TokenLogprobs> {
    let top_bytes = match chosen.to_bytes() {
        Ok(b) if b.len() >= 4 => b,
        _ => return None,
    };
    let chosen_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    crate::sampler::compute_top_logprobs(logits_flat, chosen_id, k).ok()
}

/// Greedy autoregressive generation using KV-cache prefill + decode.
///
/// Returns `Vec<ProbeStep>` — same shape as `gemma4::generate_greedy`.
///
/// `prompt_cache_slots` controls the number of post-prefill KV snapshots kept
/// across requests. Pass 1 for the legacy single-slot behaviour; pass N for
/// multi-slot prefix matching. Default recommended value: 4.
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
    model: &Qwen3_5MoeText,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: KvQuant,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&crate::gemma4::ProbeStep) -> Option<u32>,
    // A6.2: optional sampler constraint. See gemma4::generate_greedy for
    // the hot-path-cost-and-correctness contract.
    mut constraint: Option<&mut dyn ConstraintEngine>,
    // A7.2: sampling config + per-request RNG. See gemma3::generate_greedy.
    sampler_cfg: &crate::sampler::SamplerConfig,
    rng: &mut crate::sampler::Pcg32,
    // A7.3: logit-penalty configuration + per-request token history.
    penalty_cfg: &crate::sampler::PenaltyConfig,
    token_history: &mut Vec<u32>,
) -> Result<Vec<crate::gemma4::ProbeStep>> {
    use crate::gemma4::ProbeStep;

    tracing::info!(
        arch = "Qwen3_5MoeForConditionalGeneration",
        ?kv_quant,
        ?max_ctx_override,
        "generate_greedy: selected KV cache quant"
    );

    if n_tokens == 0 {
        return Ok(vec![]);
    }

    let vocab = model.cfg.vocab_size as i32;
    let mut steps = Vec::with_capacity(n_tokens);

    // Decode profile timers. For Qwen3.5MoE we only instrument the
    // cache-MISS path C (full prefill from scratch). Path A (Exact) and
    // Path B (HydratedTail) are intentionally uninstrumented: Path A skips
    // prefill entirely, and Path B's tail-prefill latency is dominated by
    // model forward cost (not the cache machinery), so decode_profile events
    // would only clutter the log for these paths.
    //
    // Note: Qwen3.5MoE pipelines argmax with `async_eval` (line ~485), so the
    // GPU sync materialises at the *next* step's `to_bytes()` on `pending`.
    // We bracket the `to_bytes` call as the eval boundary to capture this.
    let mut forward_total_ns: u128 = 0;
    let mut eval_total_ns: u128 = 0;
    let mut step_total_ns: u128 = 0;
    let mut decode_steps: u32 = 0;

    ensure_prompt_cache(prompt_cache_slots);

    // No `Prefix` variant for qwen3_5_moe: the block-truncate + tail-reprefill
    // partial path is unsafe for the recurrent GDN `lin_caches` (see the
    // `find_best_prefix` match below). Genuine partial hits fall back to Miss
    // (full re-prefill). Only full-token-equality reuse (Exact) is optimized.
    //
    // `HydratedTail` is the exception: when a SSD-hydrated entry is a STRICT
    // PREFIX of the incoming prompt (i.e. its token ids are a byte-identical
    // leading subsequence of `prompt_ids`), the block-aligned KV + GDN
    // lin_caches represent recurrent state at exactly t=prefix_len of THIS
    // same prompt, so re-prefilling only `prompt_ids[prefix_len..]` on top
    // is sequentially correct. This is safe because the strict-prefix check
    // guarantees the tail is the continuation of the exact same prompt (not a
    // divergent sequence), which is the unsafe case ExactOnly guards against.
    enum CacheLookup {
        Exact {
            kv_caches: Vec<KvCache>,
            lin_caches: Vec<LinearAttnCache>,
            last_id: u32,
            piece: String,
        },
        /// SSD-hydrated block-aligned prefix; caller must re-prefill the tail.
        HydratedTail {
            kv_caches: Vec<KvCache>,
            lin_caches: Vec<LinearAttnCache>,
            prefix_len: usize,
        },
        Miss,
    }

    // hard runtime gate: Qwen3.5-MoE must be on the ExactOnly policy.
    // A wrong policy here would expose the unsafe partial-prefix path on GDN
    // `lin_caches` (see Qwen35MoeEntry docs). Compile-time would also fail at
    // SsdHydrate impl resolution, but this catches a misconfigured arch table
    // before any cache lookup happens.
    // Promoted from debug_assert_eq! so the tripwire fires in release-perf too.
    assert_eq!(
        PROMPT_CACHE.policy(),
        crate::prompt_cache::ReusePolicy::ExactOnly,
        "qwen3_5_moe ArchPromptCache must use ReusePolicy::ExactOnly — \
         partial-prefix reuse is unsafe for GDN lin_caches",
    );

    // Issue #26: codec-partitioned prompt-cache key — see gemma4/generate/mod.rs
    // for the full rationale. Query digest stream salted by
    // `FNV_OFFSET ^ layout_key ^ codec_salt(kv_quant)` (matches the push seed).
    let cache_seed = crate::prompt_cache::FNV_OFFSET
        ^ crate::qwen3_5_moe::prompt_cache::active_layout_key()
        ^ kv_quant.cache_key_salt();
    let lookup: CacheLookup = PROMPT_CACHE.with_inner_mut(|guard| {
        let cache = guard.as_mut().unwrap();
        let mut raw_match = cache.find_best_prefix(prompt_ids, cache_seed);
        // on a RAM miss, try the SSD tier (no-op when no source attached
        // — tier OFF). A hit promotes the block into RAM; re-run find_best_prefix
        // so the promoted slot is matched + quant-checked by the path below.
        if raw_match.is_none() && cache.hydrate_from_ssd(prompt_ids).is_some() {
            raw_match = cache.find_best_prefix(prompt_ids, cache_seed);
        }
        // Plan §D8 / Task 11.5: stored `KvQuant` must match runtime; on
        // mismatch evict + warn + degrade to Miss. See `gemma4/generate.rs`
        // for the full rationale.
        let safe_match = match raw_match {
            Some((slot_idx, block_count)) => {
                let stored = cache.slots[slot_idx].entry.kv_quant;
                if stored == Some(kv_quant) {
                    Some((slot_idx, block_count))
                } else {
                    tracing::warn!(
                        stored = ?stored,
                        runtime = ?kv_quant,
                        prompt_len = prompt_ids.len(),
                        "prompt cache KV quant mismatch — evicting entry, \
                         degrading to re-prefill"
                    );
                    cache.evict_slot(slot_idx);
                    None
                }
            }
            None => None,
        };
        match safe_match {
            // Exact match: the cached entry's FULL token id list equals the
            // incoming prompt. This is verified at token granularity, NOT by
            // `block_count * BLOCK_TOKENS == len` — the block-floored test is
            // essentially never true (only when len % 256 == 0) and was
            // misrouting identical-prompt repeats into the Prefix path. For
            // qwen3_5_moe that path is unsafe: the GDN `lin_caches` recurrent
            // state cannot be reconstructed by truncating KV to a block
            // boundary and re-prefilling a short tail (`truncate_kv_to_block`
            // deliberately leaves `lin_caches` untouched), so it emitted
            // corrupted output (258 -> 9 tokens for an identical re-request).
            // True full-equality reuse sidesteps truncation entirely.
            //
            // BUG-1 guard: a SSD-hydrated entry whose block-aligned prefix
            // length happens to equal `prompt_ids.len()` (prompt is an exact
            // multiple of BLOCK_TOKENS → no tail → `prefix_len == len`) must
            // NOT be served as Exact: `first_id` is the placeholder 0 set in
            // `SsdHydrate::hydrate`, not a real decode token. Excluding it
            // here causes the match to fall through to `Miss` → full re-prefill
            // re-derives the real `first_id`. Correctness over the micro-opt.
            Some((slot_idx, _block_count))
                if !cache.slots[slot_idx].entry.is_ssd_hydrated
                    && cache.slots[slot_idx].entry.prompt_token_ids() == prompt_ids =>
            {
                let slot = &cache.slots[slot_idx].entry;
                match slot.deep_clone() {
                    Ok(cloned) => CacheLookup::Exact {
                        kv_caches: cloned.kv_caches,
                        lin_caches: cloned.lin_caches,
                        last_id: cloned.first_id,
                        piece: cloned.first_piece,
                    },
                    Err(_) => CacheLookup::Miss,
                }
            }
            // HydratedTail: SSD-hydrated block-aligned prefix that is a
            // STRICT PREFIX of the incoming prompt. The hydrated KV + GDN
            // lin_caches represent recurrent state at t=prefix_len of this
            // exact prompt; re-prefilling only `prompt_ids[prefix_len..]`
            // on top is sequentially correct (identical to pausing/resuming
            // the original prefill at the block boundary).
            //
            // Guards (all must hold):
            // 1. Entry was promoted from SSD (`is_ssd_hydrated == true`).
            // 2. Stored ids are shorter than the incoming prompt (strict prefix,
            //    not a full match — the Exact arm handles the equal-length case).
            // 3. Stored ids are byte-identical to the matching leading subsequence
            //    of `prompt_ids` (guarantees the tail is the same prompt's
            //    continuation, not a divergent one).
            //
            // If deep_clone fails, fall through to Miss (full re-prefill).
            Some((slot_idx, _block_count))
                if {
                    let e = &cache.slots[slot_idx].entry;
                    let stored = e.prompt_token_ids();
                    e.is_ssd_hydrated
                        && stored.len() < prompt_ids.len()
                        && prompt_ids.starts_with(stored)
                } =>
            {
                let slot = &cache.slots[slot_idx].entry;
                match slot.deep_clone() {
                    Ok(cloned) => CacheLookup::HydratedTail {
                        kv_caches: cloned.kv_caches,
                        lin_caches: cloned.lin_caches,
                        prefix_len: slot.prompt_token_ids().len(),
                    },
                    Err(_) => CacheLookup::Miss,
                }
            }
            // Genuine partial block-prefix hit (shares leading blocks then
            // diverges). The block-truncate + tail-reprefill Prefix path is
            // NOT correct for qwen3_5_moe: `truncate_kv_to_block` truncates
            // only `kv_caches`; the recurrent GDN `lin_caches` would still
            // hold the full original-prompt state, which is wrong for the new
            // (diverged) tail. Reconstructing recurrent linear-attn state for
            // a block-truncated prefix is not straightforward, so fall back to
            // a full re-prefill (Miss). Correctness over speed — the partial
            // optimization is forfeited for this hybrid arch on purpose.
            Some(_) => CacheLookup::Miss,
            None => CacheLookup::Miss,
        }
    });

    // Path A: exact match
    if let CacheLookup::Exact {
        mut kv_caches,
        mut lin_caches,
        last_id,
        piece,
    } = lookup
    {
        tracing::debug!(
            prompt_len = prompt_ids.len(),
            token_id = last_id,
            "qwen3_5moe generate_greedy: prompt cache EXACT HIT"
        );
        steps.push(ProbeStep {
            token_id: last_id,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: None,
        });
        step_fn(steps.last().unwrap());
        // A7.3: exact-hit token into history.
        token_history.push(last_id);

        // EOS-stop. If prefill emitted an EOS already, no decode steps.
        if eos_ids.contains(&last_id) {
            return Ok(steps);
        }

        let mut y: Array = {
            let id_i32 = last_id as i32;
            let id_bytes = id_i32.to_le_bytes();
            Array::from_bytes(&id_bytes, &[1], Dtype::I32)?
        };
        y.eval()?;
        let mut pending: Option<Array> = None;
        // logprobs captured at sample time travel with `pending` (the
        // token they belong to is emitted one iteration later). `None` on the
        // disabled path (`lp_k == 0`) — never allocated, never read.
        let mut pending_logprobs: Option<TokenLogprobs> = None;
        let mut early_stop = false;
        // forced `</think>` injection on thinking-budget overflow —
        // mirrors the Miss-path (Path C) `forced_next` contract exactly.
        // `None` (no budget) leaves the pipeline byte-for-byte unchanged.
        let mut forced_next: Option<u32> = None;
        for step_idx in 1..n_tokens {
            let decode_logits =
                match model.forward_arr(&y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
                {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!(
                            step = step_idx,
                            error = %e,
                            "qwen3_5moe generate_greedy (exact-hit): decode step failed"
                        );
                        break;
                    }
                };
            let logits_flat = decode_logits.reshape(&[1, vocab], device)?;
            // A6.3: masked path engages only when constraint wants mask
            // (engaged, not in warm-up). Otherwise behave like the
            // unconstrained path so argmax dtype + pipelining match
            // bit-for-bit — critical for greedy reproducibility during
            // the model's pre-JSON reasoning phase.
            let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
            // A7.2: temp>0 reads logits to host each step (no async
            // pipelining benefit), so it shares the masked branch's
            // pre-drain. temp<=0 keeps the exact `mask_active`-gated
            // pipelined path byte-for-byte.
            let sampling_active = sampler_cfg.sampling_active();
            // A7.3: penalties also require logits on host → fold into drain_now.
            let penalties_active = penalty_cfg.penalties_active();
            // logprob capture needs host logits per step → also drains.
            let lp_k = sampler_cfg.top_logprobs_k as usize;
            let drain_now = mask_active || sampling_active || penalties_active || lp_k > 0;
            // Force eager eval of logits on the masked path to prevent stale GPU data.
            if mask_active {
                logits_flat.eval()?;
            }
            let pre_drain_eos = if drain_now {
                if let Some(p) = pending.take() {
                    let top_bytes = p.to_bytes()?;
                    let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                    if let Some(c) = constraint.as_mut() {
                        c.advance(next_id);
                    }
                    // A7.3: accumulate emitted token into history.
                    token_history.push(next_id);
                    steps.push(ProbeStep {
                        token_id: next_id,
                        piece: String::new().into_boxed_str(),
                        max_abs_logit: 0.0,
                        nan_count: 0,
                        // logprobs computed when this token was sampled.
                        logprobs: pending_logprobs.take(),
                    });
                    // capture forced injection request from step_fn.
                    forced_next = forced_next.or(step_fn(steps.last().unwrap()));
                    eos_ids.contains(&next_id)
                } else {
                    false
                }
            } else {
                false
            };
            if pre_drain_eos {
                early_stop = true;
                break;
            }
            // A7.3: trailing-20 window for penalty context.
            let win_start = token_history.len().saturating_sub(20);
            let recent = &token_history[win_start..];
            let next_y = if sampling_active {
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
            let _ = next_y.async_eval();
            // capture this step's logprobs (gated on lp_k>0).
            if lp_k > 0 {
                pending_logprobs = capture_logprobs(&logits_flat, &next_y, lp_k);
            }
            let mut emitted_eos = false;
            if !drain_now {
                if let Some(p) = pending.take() {
                    // Task 11: mlx_sync span — measures the GPU-sync cost of
                    // materializing the previous step's pipelined argmax token.
                    let top_bytes = {
                        let _sync = tracing::debug_span!(
                            "mlx_sync",
                            step = step_idx - 1,
                            reason = "pipelined_argmax_drain",
                        )
                        .entered();
                        p.to_bytes()?
                    };
                    let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                    if let Some(c) = constraint.as_mut() {
                        c.advance(next_id);
                    }
                    // A7.3: accumulate pipelined token into history.
                    token_history.push(next_id);
                    steps.push(ProbeStep {
                        token_id: next_id,
                        piece: String::new().into_boxed_str(),
                        max_abs_logit: 0.0,
                        nan_count: 0,
                        // only reached when drain_now == false (lp_k == 0).
                        logprobs: None,
                    });
                    // capture forced injection request from step_fn.
                    forced_next = forced_next.or(step_fn(steps.last().unwrap()));
                    emitted_eos = eos_ids.contains(&next_id);
                }
            }
            if emitted_eos {
                early_stop = true;
                break;
            }
            // forced `</think>` injection. Discard the model's pipelined
            // successor and feed the forced id as both next input and next drained
            // token, so the model resumes on the answer channel. No-op when
            // `forced_next` is `None` (budget-unset hot path).
            if let Some(forced_id) = forced_next.take() {
                let bytes = (forced_id as i32).to_le_bytes();
                let forced_arr = Array::from_bytes(&bytes, &[1], Dtype::I32)?;
                let _ = forced_arr.async_eval();
                y = forced_arr.try_clone()?;
                pending = Some(forced_arr);
                // Forced tokens are external overrides — no meaningful logprobs.
                pending_logprobs = None;
            } else {
                y = next_y.try_clone()?;
                pending = Some(next_y);
            }
        }
        if !early_stop {
            if let Some(p) = pending {
                p.eval()?;
                let top_bytes = p.to_bytes()?;
                let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                // A6.2: advance constraint with emitted id.
                if let Some(c) = constraint.as_mut() {
                    c.advance(next_id);
                }
                // A7.3: final drain token into history.
                token_history.push(next_id);
                steps.push(ProbeStep {
                    token_id: next_id,
                    piece: String::new().into_boxed_str(),
                    max_abs_logit: 0.0,
                    nan_count: 0,
                    // final pending token carries its sample-time logprobs.
                    logprobs: pending_logprobs.take(),
                });
                step_fn(steps.last().unwrap());
            }
        }
        // N16: store KV-cache bytes (KV + linear-attn state) for /metrics/cache.
        let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
            + lin_caches.iter().map(|c| c.approx_bytes()).sum::<u64>();
        store_kv_cache_bytes(kv_bytes);
        return Ok(steps);
    }

    // Path B: HydratedTail — SSD block-aligned prefix in RAM, tail needs forwarding.
    //
    // The hydrated caches are in POST-exit_prefill (decode) mode: the prefix
    // K/V at positions 0..prefix_len are already stored in the quantised
    // payload buffers with `offset == prefix_len`. Running
    // `forward_seq_with_cache(tail, ...)` in this state (WITHOUT calling
    // enter_prefill) correctly:
    //
    //   1. appends the tail K/V into the existing storage via the decode path
    //      (`update_none` / `update_k8v8` etc.), advancing `offset` to
    //      prefix_len + tail_len.
    //   2. produces logits where each tail position attends causally over ALL
    //      prefix + earlier-tail positions — identical to what a full cold
    //      prefill would produce.
    //
    // This is exactly the mechanism that `forward_seq_last_k_with_cache`
    // (tested in `tests/qwen3_5_moe_forward_seq_last_k.rs`) uses to split a
    // prefill into two sequential calls and produce byte-identical logits.
    //
    // IMPORTANT: do NOT call enter_prefill() here. enter_prefill() resets the
    // raw accumulation buffers (`prefill_raw_k/v = None`) but leaves the
    // quantised payload intact. The subsequent tail forward would then attempt
    // to grow the prefill_raw buffer from zero, ignoring the existing payload,
    // which produces wrong attention and a guard error on resumed caches.
    //
    // Chunking (same prefill_chunk as Path C): keeps GDN ts < 256 per chunk
    // to avoid Metal watchdog timeouts on long tails.
    if let CacheLookup::HydratedTail {
        mut kv_caches,
        mut lin_caches,
        prefix_len,
    } = lookup
    {
        let tail_len = prompt_ids.len() - prefix_len;
        tracing::info!(
            arch = "Qwen3_5MoeForConditionalGeneration",
            prefix_len,
            tail_len,
            "qwen3_5moe: hydrated-tail hit — forwarding tail from SSD-restored KV"
        );

        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe");

        let prefill_logits = {
            let mut last_logits: Option<Array> = None;
            let mut prefill_ok = true;
            let tail = &prompt_ids[prefix_len..];
            let n_chunks = tail.len().div_ceil(prefill_chunk);
            for (chunk_idx, chunk) in tail.chunks(prefill_chunk).enumerate() {
                let is_last = chunk_idx + 1 == n_chunks;
                match model.forward_seq_with_cache(
                    chunk,
                    Some(&mut kv_caches),
                    Some(&mut lin_caches),
                    device,
                ) {
                    Ok(logits) => {
                        if is_last {
                            last_logits = Some(logits);
                        } else {
                            // Force-evaluate the KV cache state between chunks
                            // (avoids lazy-graph explosion, mirrors Path C).
                            for c in &kv_caches {
                                if let Err(e) = c.eval_prefill_state() {
                                    tracing::warn!(
                                        error = %e,
                                        chunk_len = chunk.len(),
                                        "qwen3_5moe generate_greedy (hydrated-tail): tail chunk eval failed"
                                    );
                                    prefill_ok = false;
                                    break;
                                }
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!(
                            error = %e,
                            prefix_len,
                            tail_len,
                            "qwen3_5moe generate_greedy (hydrated-tail): tail chunk failed, returning empty"
                        );
                        prefill_ok = false;
                        break;
                    }
                }
                if !prefill_ok {
                    break;
                }
            }

            if !prefill_ok || last_logits.is_none() {
                return Ok(steps);
            }
            last_logits.unwrap()
        };

        let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
        let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
        let sampling_active = sampler_cfg.sampling_active();
        let penalties_active = penalty_cfg.penalties_active();
        let lp_k = sampler_cfg.top_logprobs_k as usize;
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
        if let Some(c) = constraint.as_mut() {
            c.advance(last_id);
        }

        let piece = tokenizer
            .id_to_token(last_id)
            .unwrap_or_else(|| format!("<unk:{last_id}>"));

        // Push the completed full-length snapshot (both prefix + tail) so
        // future requests can serve this prompt as an Exact hit from RAM.
        {
            let kv_snap: Result<Vec<_>> = kv_caches.iter().map(|c| c.try_deep_clone()).collect();
            let lin_snap: Result<Vec<_>> = lin_caches.iter().map(|c| c.try_deep_clone()).collect();
            if let (Ok(kvs), Ok(lins)) = (kv_snap, lin_snap) {
                match kvs
                    .iter()
                    .try_for_each(|c| c.eval_for_spill())
                    .and_then(|()| lins.iter().try_for_each(|c| c.eval_for_spill()))
                {
                    Ok(()) => {
                        PROMPT_CACHE.with_inner_mut(|guard| {
                            if let Some(cache) = guard.as_mut() {
                                let lk = crate::qwen3_5_moe::prompt_cache::active_layout_key();
                                cache.push(Qwen35MoeEntry {
                                    prompt_token_ids: prompt_ids.to_vec(),
                                    block_hashes: crate::prompt_cache::chained_block_hashes_seeded(
                                        prompt_ids,
                                        crate::prompt_cache::FNV_OFFSET
                                            ^ lk
                                            ^ kv_quant.cache_key_salt(),
                                    ),
                                    kv_caches: kvs,
                                    lin_caches: lins,
                                    first_id: last_id,
                                    first_piece: piece.clone(),
                                    kv_quant: Some(kv_quant),
                                    is_ssd_hydrated: false,
                                });
                                tracing::debug!(
                                    prompt_len = prompt_ids.len(),
                                    token_id = last_id,
                                    n_slots = cache.slots.len(),
                                    "qwen3_5moe generate_greedy: hydrated-tail — saved full snapshot"
                                );
                            }
                        });
                    }
                    Err(e) => tracing::warn!(error = %e,
                        "qwen3_5moe generate_greedy (hydrated-tail): pre-eval failed, skipping prompt cache store"),
                }
            }
        }

        let prefill_logprobs = if lp_k > 0 {
            capture_logprobs(&logits_flat, &top, lp_k)
        } else {
            None
        };
        steps.push(ProbeStep {
            token_id: last_id,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: prefill_logprobs,
        });
        step_fn(steps.last().unwrap());
        token_history.push(last_id);

        if eos_ids.contains(&last_id) {
            return Ok(steps);
        }

        let mut next_id;
        let _ = last_id;
        let mut y: Array = {
            let id_i32 = last_id as i32;
            let bytes = id_i32.to_le_bytes();
            Array::from_bytes(&bytes, &[1], Dtype::I32)?
        };
        y.eval()?;

        let mut pending: Option<Array> = None;
        let mut pending_logprobs: Option<TokenLogprobs> = None;
        let mut early_stop = false;
        let mut forced_next: Option<u32> = None;

        for step_idx in 1..n_tokens {
            let decode_logits =
                match model.forward_arr(&y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
                {
                    Ok(l) => l,
                    Err(e) => {
                        tracing::warn!(
                            step = step_idx,
                            error = %e,
                            "qwen3_5moe generate_greedy (hydrated-tail): decode step failed"
                        );
                        break;
                    }
                };
            let logits_flat = decode_logits.reshape(&[1, vocab], device)?;
            let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
            let sampling_active = sampler_cfg.sampling_active();
            let penalties_active = penalty_cfg.penalties_active();
            let lp_k = sampler_cfg.top_logprobs_k as usize;
            let drain_now = mask_active || sampling_active || penalties_active || lp_k > 0;
            if mask_active {
                logits_flat.eval()?;
            }
            let pre_drain_eos = if drain_now {
                if let Some(p) = pending.take() {
                    let top_bytes = p.to_bytes()?;
                    next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                    if let Some(c) = constraint.as_mut() {
                        c.advance(next_id);
                    }
                    token_history.push(next_id);
                    steps.push(ProbeStep {
                        token_id: next_id,
                        piece: String::new().into_boxed_str(),
                        max_abs_logit: 0.0,
                        nan_count: 0,
                        logprobs: pending_logprobs.take(),
                    });
                    forced_next = forced_next.or(step_fn(steps.last().unwrap()));
                    eos_ids.contains(&next_id)
                } else {
                    false
                }
            } else {
                false
            };
            if pre_drain_eos {
                early_stop = true;
                break;
            }
            let win_start = token_history.len().saturating_sub(20);
            let recent = &token_history[win_start..];
            let next_y = if sampling_active {
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
            let _ = next_y.async_eval();
            if lp_k > 0 {
                pending_logprobs = capture_logprobs(&logits_flat, &next_y, lp_k);
            }
            let mut emitted_eos = false;
            if !drain_now {
                if let Some(p) = pending.take() {
                    let top_bytes = p.to_bytes()?;
                    next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                    if let Some(c) = constraint.as_mut() {
                        c.advance(next_id);
                    }
                    token_history.push(next_id);
                    steps.push(ProbeStep {
                        token_id: next_id,
                        piece: String::new().into_boxed_str(),
                        max_abs_logit: 0.0,
                        nan_count: 0,
                        logprobs: None,
                    });
                    forced_next = forced_next.or(step_fn(steps.last().unwrap()));
                    emitted_eos = eos_ids.contains(&next_id);
                }
            }
            if emitted_eos {
                early_stop = true;
                break;
            }
            if let Some(forced_id) = forced_next.take() {
                let bytes = (forced_id as i32).to_le_bytes();
                let forced_arr = Array::from_bytes(&bytes, &[1], Dtype::I32)?;
                let _ = forced_arr.async_eval();
                y = forced_arr.try_clone()?;
                pending = Some(forced_arr);
                pending_logprobs = None;
            } else {
                y = next_y.try_clone()?;
                pending = Some(next_y);
            }
        }

        if !early_stop {
            if let Some(p) = pending {
                p.eval()?;
                let top_bytes = p.to_bytes()?;
                next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                if let Some(c) = constraint.as_mut() {
                    c.advance(next_id);
                }
                token_history.push(next_id);
                steps.push(ProbeStep {
                    token_id: next_id,
                    piece: String::new().into_boxed_str(),
                    max_abs_logit: 0.0,
                    nan_count: 0,
                    logprobs: pending_logprobs.take(),
                });
                step_fn(steps.last().unwrap());
            }
        }

        let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
            + lin_caches.iter().map(|c| c.approx_bytes()).sum::<u64>();
        store_kv_cache_bytes(kv_bytes);
        return Ok(steps);
    }

    // Issue #25: `--max-ctx` is a virtual ceiling the KV ring grows lazily up
    // to, not an eager allocation. `initial_max_seq` is the lazy start;
    // `max_seq_ceiling` caps growth and rejects over-long prompts.
    let (initial_max_seq, max_seq_ceiling) =
        kv_max_seq_and_ceiling(max_ctx_override, model.cfg.max_position_embeddings as i32);
    // 64-token chunks keep GDN ts < 256 threshold → sequential MSL kernel, no lazy graph explosion.
    // 2048-token chunks (e0231a4) trigger gated_delta_prefill_ops at ts=2048 → ~1.47M lazy nodes
    // across 30 GDN layers → Metal GPU watchdog timeout.
    //
    // Compile cache is in place (gated_delta_msl.rs). Raising
    // chunk size to 256 routes chunks to gated_delta_prefill_ops, which on the FIRST
    // call traces T=256 (~184K lazy nodes) — acceptable for the watchdog, but the trace
    // cost (~6s) lands on the critical path of the first request without warmup.
    // The GDN pre-warm (`arch::load_model`) now traces gated_delta_prefill_ops at
    // model load time, so the first chunk's compile cost is paid at startup.
    //
    // p0b-ttft bench: even with pre-warm, chunk=256
    // REGRESSED cold TTFT by +15% at 8K and +80% at 32K vs chunk=64. Per-chunk
    // gated_delta_prefill_ops at T=256 is genuinely slower (much heavier lazy
    // graph per call) than per-chunk at T=64. Default stays at 64; the env
    // override remains available for users who want to experiment.
    //
    // Override via `RMLX_PREFILL_CHUNK` (global) or
    // `RMLX_PREFILL_CHUNK_QWEN3_5_MOE` (per-arch). Note: the pre-warm uses
    // the chunk size resolved at load time
    // (`gdn_warmup_t = prefill_chunk_for("qwen3_5_moe")` in arch.rs); set
    // the env BEFORE `rmlx serve` for the warmup to match.
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe");

    // Path C: cache miss — full prefill from scratch
    let prefill_t0 = Instant::now();
    let n_layers = model.cfg.num_hidden_layers;
    // Force K8V8 for boundary layers (first head_n + last tail_n)
    // to protect output quality with aggressive V compression (planar/k8v4).
    let mut kv_caches: Vec<KvCache> = (0..n_layers)
        .map(|i| {
            let q = kv_quant_for_layer(
                i,
                n_layers,
                kv_quant,
                LAYER_ADAPTIVE_TAIL_N,
                LAYER_ADAPTIVE_HEAD_N,
            );
            KvCache::with_quant_max_seq(q, initial_max_seq)
                .with_max_seq_ceiling(max_seq_ceiling)
                .with_layer_idx(i)
        })
        .collect();
    let mut lin_caches: Vec<LinearAttnCache> = (0..model.cfg.num_hidden_layers)
        .map(|_| LinearAttnCache::new())
        .collect();

    for c in &mut kv_caches {
        c.enter_prefill();
    }

    let prefill_logits = {
        let mut last_logits: Option<Array> = None;
        let mut prefill_ok = true;
        let n_chunks = prompt_ids.len().div_ceil(prefill_chunk);
        'prefill: for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            match model.forward_seq_with_cache(
                chunk,
                Some(&mut kv_caches),
                Some(&mut lin_caches),
                device,
            ) {
                Ok(logits) => {
                    if is_last {
                        last_logits = Some(logits);
                    } else {
                        // Eval only cache state, skip lm_head matmul
                        // for non-final chunks via lazy-graph pruning.
                        for c in &kv_caches {
                            if let Err(e) = c.eval_prefill_state() {
                                tracing::warn!(
                                    error = %e,
                                    chunk_len = chunk.len(),
                                    "qwen3_5moe generate_greedy: prefill chunk cache eval failed"
                                );
                                prefill_ok = false;
                                break 'prefill;
                            }
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        error = %e,
                        prompt_len = prompt_ids.len(),
                        "qwen3_5moe generate_greedy: prefill chunk failed, returning empty"
                    );
                    prefill_ok = false;
                    break;
                }
            }
        }

        for c in &mut kv_caches {
            if let Err(e) = c.exit_prefill(device) {
                tracing::warn!(error = %e, "prefill: exit_prefill quantization failed");
                prefill_ok = false;
                break;
            }
        }

        if !prefill_ok || last_logits.is_none() {
            return Ok(steps);
        }
        last_logits.unwrap()
    };

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    // A6.3: see wants_mask gating rationale on the exact-hit decode.
    let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
    // A7.2: temp<=0 keeps the exact greedy block below; temp>0 host-samples.
    let sampling_active = sampler_cfg.sampling_active();
    // A7.3: penalties require logits on host.
    let penalties_active = penalty_cfg.penalties_active();
    // top-k logprob capture (0 = disabled, hot-loop zero-overhead).
    let lp_k = sampler_cfg.top_logprobs_k as usize;
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
    // A6.3: advance constraint regardless of mask state (warm-up scans).
    if let Some(c) = constraint.as_mut() {
        c.advance(last_id);
    }
    let prefill_total_ns = prefill_t0.elapsed().as_nanos();

    let piece = tokenizer
        .id_to_token(last_id)
        .unwrap_or_else(|| format!("<unk:{last_id}>"));

    tracing::debug!(
        step = 0,
        token_id = last_id,
        piece = %piece,
        prompt_len = prompt_ids.len(),
        "qwen3_5moe generate_greedy prefill (fast path)"
    );

    {
        let kv_snap: Result<Vec<_>> = kv_caches.iter().map(|c| c.try_deep_clone()).collect();
        let lin_snap: Result<Vec<_>> = lin_caches.iter().map(|c| c.try_deep_clone()).collect();
        if let (Ok(kvs), Ok(lins)) = (kv_snap, lin_snap) {
            // Materialize GPU arrays on the current inference thread before
            // storing in the prompt cache.  Each spawn_blocking request runs
            // on its own tokio thread, which has its own Metal GPU stream
            // (registered by ensure_gpu_default_stream() in arch::generate_greedy).
            // If these lazy arrays are stored as-is and later evicted on a
            // *different* inference thread (a subsequent request), that thread's
            // eval_for_spill call will fail with "There is no Stream(gpu, N) in
            // current thread" because stream N is only registered here.
            // Pre-eval on this thread makes eval() a no-op from any future thread.
            match kvs
                .iter()
                .try_for_each(|c| c.eval_for_spill())
                .and_then(|()| lins.iter().try_for_each(|c| c.eval_for_spill()))
            {
                Ok(()) => {
                    PROMPT_CACHE.with_inner_mut(|guard| {
                        if let Some(cache) = guard.as_mut() {
                            // salt chained walk with the active layout_key. When
                            // tier is OFF, helper returns 0 ⇒ legacy un-salted digests.
                            let lk = crate::qwen3_5_moe::prompt_cache::active_layout_key();
                            cache.push(Qwen35MoeEntry {
                                prompt_token_ids: prompt_ids.to_vec(),
                                block_hashes: crate::prompt_cache::chained_block_hashes_seeded(
                                    prompt_ids,
                                    crate::prompt_cache::FNV_OFFSET
                                        ^ lk
                                        ^ kv_quant.cache_key_salt(),
                                ),
                                kv_caches: kvs,
                                lin_caches: lins,
                                first_id: last_id,
                                first_piece: piece.clone(),
                                kv_quant: Some(kv_quant),
                                is_ssd_hydrated: false,
                            });
                            tracing::debug!(
                                prompt_len = prompt_ids.len(),
                                token_id = last_id,
                                n_slots = cache.slots.len(),
                                "qwen3_5moe generate_greedy: prompt cache MISS — saved snapshot"
                            );
                        }
                    });
                }
                Err(e) => tracing::warn!(error = %e,
                    "qwen3_5moe generate_greedy: pre-eval of KV/lin caches failed, skipping prompt cache store"),
            }
        }
    }

    // prefill is non-pipelined (last_id already materialised), so the
    // prefill token's logprobs come straight from this step's logits.
    let prefill_logprobs = if lp_k > 0 {
        capture_logprobs(&logits_flat, &top, lp_k)
    } else {
        None
    };
    steps.push(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: prefill_logprobs,
    });
    step_fn(steps.last().unwrap());
    // A7.3: prefill first token into history.
    token_history.push(last_id);

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if eos_ids.contains(&last_id) {
        let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
        tracing::info!(
            target: "decode_profile",
            arch = "Qwen3_5MoeForConditionalGeneration",
            n_steps = 0,
            prefill_ms,
            "decode_profile (prefill-EOS)"
        );
        return Ok(steps);
    }

    let mut next_id;
    let _ = last_id;
    let mut y: Array = {
        let id_i32 = last_id as i32;
        let bytes = id_i32.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::I32)?
    };
    y.eval()?;

    let mut pending: Option<Array> = None;
    // logprobs captured at sample time travel with `pending` (the token
    // they belong to is emitted one iteration later). `None` on the disabled
    // path (`lp_k == 0`) — never allocated, never read.
    let mut pending_logprobs: Option<TokenLogprobs> = None;
    let mut early_stop = false;
    // when `step_fn` returns `Some(id)` (thinking budget exceeded), the
    // loop discards the model's pipelined successor and feeds `id` (`</think>`)
    // as the next input, materialising it as the next emitted token so the
    // model resumes on the answer channel. `None` (no budget) leaves the
    // pipeline byte-for-byte unchanged.
    let mut forced_next: Option<u32> = None;

    for step_idx in 1..n_tokens {
        let step_t0 = Instant::now();
        let fwd_t0 = Instant::now();
        let decode_logits =
            match model.forward_arr(&y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device) {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!(
                        step = step_idx,
                        error = %e,
                        "qwen3_5moe generate_greedy: decode step failed, stopping early"
                    );
                    break;
                }
            };

        let logits_flat = decode_logits.reshape(&[1, vocab], device)?;
        // A6.3: see exact-hit decode for the wants_mask gating rationale.
        let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
        // A7.2: temp>0 reads logits to host each step (no async pipelining
        // benefit), so it shares the masked branch's pre-drain. temp<=0
        // keeps the exact `mask_active`-gated pipelined path byte-for-byte.
        let sampling_active = sampler_cfg.sampling_active();
        // A7.3: penalties also require logits on host → fold into drain_now.
        let penalties_active = penalty_cfg.penalties_active();
        // logprob capture needs host logits per step → also drains here.
        let lp_k = sampler_cfg.top_logprobs_k as usize;
        let drain_now = mask_active || sampling_active || penalties_active || lp_k > 0;
        // Force eager eval of logits on the masked path to prevent stale GPU data.
        if mask_active {
            logits_flat.eval()?;
        }
        let pre_drain_eos = if drain_now {
            if let Some(p) = pending.take() {
                let top_bytes = p.to_bytes()?;
                next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                if let Some(c) = constraint.as_mut() {
                    c.advance(next_id);
                }
                // A7.3: accumulate emitted token into history.
                token_history.push(next_id);
                steps.push(ProbeStep {
                    token_id: next_id,
                    piece: String::new().into_boxed_str(),
                    max_abs_logit: 0.0,
                    nan_count: 0,
                    // logprobs computed when this token was sampled.
                    logprobs: pending_logprobs.take(),
                });
                forced_next = forced_next.or(step_fn(steps.last().unwrap()));
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
                "qwen3_5moe generate_greedy: EOS emitted (pre-drain, masked), stopping"
            );
            break;
        }
        // A7.3: trailing-20 window for penalty context.
        let win_start = token_history.len().saturating_sub(20);
        let recent = &token_history[win_start..];
        let next_y = if sampling_active {
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
        let _ = next_y.async_eval();
        // capture this step's logprobs from `logits_flat` + the chosen
        // token. Stashed in `pending_logprobs`; emitted with the matching
        // ProbeStep next iteration (pipelined token identity). Gated on lp_k>0.
        if lp_k > 0 {
            pending_logprobs = capture_logprobs(&logits_flat, &next_y, lp_k);
        }
        let fwd_dt = fwd_t0.elapsed().as_nanos();

        // The actual GPU sync materialises here on the *previous* step's pending.
        let eval_t0 = Instant::now();
        let mut emitted_eos = false;
        if !drain_now {
            if let Some(p) = pending.take() {
                let top_bytes = p.to_bytes()?;
                next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                if let Some(c) = constraint.as_mut() {
                    c.advance(next_id);
                }
                // A7.3: accumulate pipelined token into history.
                token_history.push(next_id);
                steps.push(ProbeStep {
                    token_id: next_id,
                    piece: String::new().into_boxed_str(),
                    max_abs_logit: 0.0,
                    nan_count: 0,
                    logprobs: None,
                });
                forced_next = forced_next.or(step_fn(steps.last().unwrap()));
                emitted_eos = eos_ids.contains(&next_id);
            }
        }
        let eval_dt = eval_t0.elapsed().as_nanos();

        forward_total_ns += fwd_dt;
        eval_total_ns += eval_dt;
        step_total_ns += step_t0.elapsed().as_nanos();
        decode_steps += 1;

        if emitted_eos {
            early_stop = true;
            tracing::debug!(
                step = step_idx - 1,
                "qwen3_5moe generate_greedy: EOS emitted, stopping decode loop"
            );
            break;
        }

        // forced `</think>` injection. Discard the model's pipelined
        // successor (`next_y`) and feed the forced id as BOTH the next input
        // (`y`) and the next drained token (`pending`), so the next iteration
        // emits `</think>` and resumes natural answer generation. No-op when
        // `forced_next` is `None` (the budget-unset hot path).
        if let Some(forced_id) = forced_next.take() {
            let bytes = (forced_id as i32).to_le_bytes();
            let forced_arr = Array::from_bytes(&bytes, &[1], Dtype::I32)?;
            let _ = forced_arr.async_eval();
            y = forced_arr.try_clone()?;
            pending = Some(forced_arr);
            // Forced tokens are external overrides — no meaningful logprobs.
            pending_logprobs = None;
        } else {
            y = next_y.try_clone()?;
            pending = Some(next_y);
        }
    }

    if !early_stop {
        if let Some(p) = pending {
            let drain_t0 = Instant::now();
            p.eval()?;
            let top_bytes = p.to_bytes()?;
            eval_total_ns += drain_t0.elapsed().as_nanos();
            next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
            // A6.2: advance constraint on final drain.
            if let Some(c) = constraint.as_mut() {
                c.advance(next_id);
            }
            // A7.3: final drain token into history.
            token_history.push(next_id);
            steps.push(ProbeStep {
                token_id: next_id,
                piece: String::new().into_boxed_str(),
                max_abs_logit: 0.0,
                nan_count: 0,
                // final pending token carries its sample-time logprobs.
                logprobs: pending_logprobs.take(),
            });
            step_fn(steps.last().unwrap());
        }
    }

    let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
    let forward_ms = (forward_total_ns as f64) / 1.0e6;
    let eval_ms = (eval_total_ns as f64) / 1.0e6;
    let step_ms = (step_total_ns as f64) / 1.0e6;
    let n = f64::from(decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = "Qwen3_5MoeForConditionalGeneration",
        n_steps = decode_steps,
        prefill_ms,
        forward_total_ms = forward_ms,
        eval_total_ms = eval_ms,
        step_total_ms = step_ms,
        forward_per_step_ms = forward_ms / n,
        eval_per_step_ms = eval_ms / n,
        "decode_profile"
    );

    // N16: store KV-cache bytes (KV + linear-attn state) for /metrics/cache.
    let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
        + lin_caches.iter().map(|c| c.approx_bytes()).sum::<u64>();
    store_kv_cache_bytes(kv_bytes);

    Ok(steps)
}
