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

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::match_same_arms,
    clippy::redundant_closure_for_method_calls,
    clippy::too_many_lines
)]
use std::time::Instant;

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device};

use crate::constraint::ConstraintEngine;
use crate::decode_loop::{
    capture_logprobs, choose_token, chunked_prefill, pipelined_decode, DecodeCtx,
};
use crate::kv_cache::{
    kv_max_seq_and_ceiling, kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

use super::model::Qwen3_5MoeText;
use super::prompt_cache::{
    ensure_prompt_cache, store_kv_cache_bytes, Qwen35MoeEntry, PROMPT_CACHE,
};
use crate::prompt_cache::PromptCacheEntry as _;

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
pub fn generate_greedy<'a>(
    model: &Qwen3_5MoeText,
    tokenizer: &'a tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: KvQuant,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
    eos_ids: &'a [u32],
    // A6.2: optional sampler constraint. See gemma4::generate_greedy for
    // the hot-path-cost-and-correctness contract.
    // The shared `DecodeCtx` bundles every per-request borrow under one
    // lifetime, so these references share `'a` (a `&mut dyn` trait-object
    // reborrow is invariant and cannot be re-unified once split).
    step_fn: &'a mut dyn FnMut(&crate::gemma4::ProbeStep) -> Option<u32>,
    mut constraint: Option<&'a mut dyn ConstraintEngine>,
    // A7.2: sampling config + per-request RNG. See gemma3::generate_greedy.
    sampler_cfg: &'a crate::sampler::SamplerConfig,
    rng: &'a mut crate::sampler::Pcg32,
    // A7.3: logit-penalty configuration + per-request token history.
    penalty_cfg: &'a crate::sampler::PenaltyConfig,
    token_history: &'a mut Vec<u32>,
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

        // Shared pipelined decode loop. The exact-hit funnel is just
        // "caches + last_id → decode"; the loop is the call site. The GDN
        // `lin_caches` are CLOSURE-CAPTURED — the shared loop never learns
        // they exist; only the forward closure threads them through.
        {
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
                arch: "Qwen3_5MoeForConditionalGeneration",
                resolve_pieces: false,
            };
            pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
                model.forward_arr(y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
            })?;
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
        let lp_k = sampler_cfg.top_logprobs_k as usize;

        // Build the shared decode context ONCE for the tail-prefill token
        // selection AND the decode loop, so the per-request state is borrowed a
        // single time (a `&mut dyn` trait-object reborrow is invariant in its
        // lifetime — two separate `DecodeCtx` over the same params would not
        // compile). The GDN `lin_caches` stay closure-captured below.
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
            arch: "Qwen3_5MoeForConditionalGeneration",
            resolve_pieces: false,
        };

        // Tail-prefill token selection via the shared sampling fork. Gate the
        // mask ONCE here, before the post-selection `advance()` below — matching
        // the old tail timing (wants_mask can flip on engagement).
        let mask_active = ctx.constraint.as_ref().is_some_and(|c| c.wants_mask());
        let top = choose_token(&mut ctx, &logits_flat, mask_active)?;
        top.eval()?;
        let top_bytes = top.to_bytes()?;
        let last_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
        if let Some(c) = ctx.constraint.as_mut() {
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
        (ctx.step_fn)(steps.push_mut(ProbeStep {
            token_id: last_id,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: prefill_logprobs,
        }));
        ctx.token_history.push(last_id);

        if eos_ids.contains(&last_id) {
            return Ok(steps);
        }

        // Shared pipelined decode loop. GDN `lin_caches` are closure-captured —
        // the loop never learns they exist.
        pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
            model.forward_arr(y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
        })?;

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

    // Fresh chunked prefill via the shared helper. It brackets the loop with
    // enter_prefill() / exit_prefill(), evals only the cache state on non-final
    // chunks, and returns None on rejection. The GDN `lin_caches` are
    // closure-captured — the helper only owns `kv_caches`.
    let Some(prefill_logits) = chunked_prefill(
        &mut kv_caches,
        prompt_ids,
        prefill_chunk,
        device,
        "Qwen3_5MoeForConditionalGeneration",
        |chunk, kv| model.forward_seq_with_cache(chunk, Some(kv), Some(&mut lin_caches), device),
    )?
    else {
        return Ok(steps);
    };

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    // top-k logprob capture (0 = disabled, hot-loop zero-overhead).
    let lp_k = sampler_cfg.top_logprobs_k as usize;

    // Build the shared decode context ONCE for the prefill-tail selection AND
    // the Miss decode loop, so the per-request state is borrowed a single time
    // (a `&mut dyn` trait-object reborrow is invariant in its lifetime — two
    // separate `DecodeCtx` over the same params would not compile). The GDN
    // `lin_caches` stay closure-captured in the loop below.
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
        arch: "Qwen3_5MoeForConditionalGeneration",
        resolve_pieces: false,
    };

    // Prefill-tail token selection via the shared sampling fork. Gate the mask
    // ONCE here, before the post-selection `advance()` below — matching the old
    // prefill-tail timing (wants_mask can flip on engagement).
    let mask_active = ctx.constraint.as_ref().is_some_and(|c| c.wants_mask());
    let top = choose_token(&mut ctx, &logits_flat, mask_active)?;
    top.eval()?;
    let top_bytes = top.to_bytes()?;
    let last_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    // A6.3: advance constraint regardless of mask state (warm-up scans).
    if let Some(c) = ctx.constraint.as_mut() {
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
    (ctx.step_fn)(steps.push_mut(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: prefill_logprobs,
    }));
    // prefill first token into history.
    ctx.token_history.push(last_id);

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

    // Decode: shared pipelined async loop, reusing the prefill-tail `ctx`. The
    // Miss funnel is just "caches + last_id → decode"; the loop is the call
    // site. The GDN `lin_caches` are closure-captured — the loop never learns
    // they exist.
    let stats = pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
        model.forward_arr(y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
    })?;

    let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
    let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
    let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
    let step_ms = (stats.step_total_ns as f64) / 1.0e6;
    let n = f64::from(stats.decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = "Qwen3_5MoeForConditionalGeneration",
        n_steps = stats.decode_steps,
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
