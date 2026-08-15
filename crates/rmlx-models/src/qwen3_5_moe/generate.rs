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

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};

use crate::constraint::ConstraintEngine;
use crate::decode_loop::{
    capture_logprobs, choose_token, chunked_prefill, pipelined_decode, DecodeCtx,
};
use crate::kv_cache::{
    kv_max_seq_and_ceiling, kv_quant_for_layer, warn_if_kv_codec_net_negative, KvLayerShape,
    LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

use super::model::Qwen3_5MoeText;
use super::prompt_cache::{ensure_prompt_cache, Qwen35MoeEntry, PROMPT_CACHE};
use crate::prompt_cache::{Consumed, ReuseKind, ReusePolicy};

/// Greedy autoregressive generation using KV-cache prefill + decode.
///
/// Returns `Vec<ProbeStep>` — same shape as `gemma4::generate_greedy`.
///
/// `prompt_cache_slots` controls the number of post-prefill KV snapshots kept
/// across requests. Pass 1 for the legacy single-slot behaviour; pass N for
/// multi-slot prefix matching. Default recommended value: 4.
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
    step_fn: &'a mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
    mut constraint: Option<&'a mut dyn ConstraintEngine>,
    // A7.2: sampling config + per-request RNG. See gemma3::generate_greedy.
    sampler_cfg: &'a crate::sampler::SamplerConfig,
    rng: &'a mut crate::sampler::Pcg32,
    // A7.3: logit-penalty configuration + per-request token history.
    penalty_cfg: &'a crate::sampler::PenaltyConfig,
    token_history: &'a mut Vec<u32>,
) -> Result<Vec<crate::decode_loop::ProbeStep>> {
    use crate::decode_loop::ProbeStep;

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

    // ------------------------------------------------------------------
    // Prompt cache lookup via the shared consume() engine. Qwen3.5-MoE is a
    // hybrid GDN arch and uses `ReusePolicy::ExactOnly`. The reachable outcomes:
    //   - `Exact`  = cached prompt token-for-token equal to this prompt (full
    //                reuse, no re-prefill).
    //   - `Reuse{StrictPrefix}` = HydratedTail: an SSD-hydrated entry that is a
    //                STRICT PREFIX of the incoming prompt. Its block-aligned KV +
    //                GDN `lin_caches` are the recurrent state at t=prefix_len of
    //                THIS same prompt, so re-prefilling only the tail on top is
    //                sequentially correct (the engine's policy gate permits a
    //                hydrated strict-prefix even under ExactOnly).
    //   - `Miss`   = no usable entry — a non-hydrated partial match (forbidden by
    //                ExactOnly for the GDN `lin_caches`), the block-aligned
    //                equal-length hydrated case (placeholder first token — must
    //                recompute), an incomplete hydrate, or a quant mismatch.
    // `BlockTruncate` is never produced by moe (the GDN `lin_caches` cannot be
    // reconstructed from a block-truncated KV), so that arm is unreachable here.
    // The engine owns find → SSD-hydrate retry → quant-mismatch guard →
    // SSD-hydrated Exact exclusion → reuse-gate, and traces every degrade branch.
    // ------------------------------------------------------------------

    // hard runtime gate: Qwen3.5-MoE must be on the ExactOnly policy. A wrong
    // policy here would expose the unsafe partial-prefix path on the GDN
    // `lin_caches` (see Qwen35MoeEntry docs). Compile-time would also fail at
    // SsdHydrate impl resolution, but this catches a misconfigured arch table
    // before any cache lookup happens. A real (not debug_) assert so the
    // tripwire fires under release-perf too.
    assert_eq!(
        PROMPT_CACHE.policy(),
        ReusePolicy::ExactOnly,
        "qwen3_5_moe ArchPromptCache must use ReusePolicy::ExactOnly — \
         partial-prefix reuse is unsafe for GDN lin_caches",
    );

    // image prompts never reach this arch (text path only), so `has_image` is
    // always false here; the engine bypass is belt-and-suspenders.
    let consumed = PROMPT_CACHE.consume(prompt_ids, kv_quant, false, model.model_sig);

    // `exact_hit` carries the Path-A locals; `hydrated_tail` carries the Path-B
    // tail-re-prefill locals. `Consumed::Miss` leaves both `None` and falls
    // through to Path C (full re-prefill).
    let mut exact_hit: Option<(Vec<KvCache>, Vec<LinearAttnCache>, u32, String)> = None;
    let mut hydrated_tail: Option<(Vec<KvCache>, Vec<LinearAttnCache>, usize)> = None;
    match consumed {
        Consumed::Exact(cloned) => {
            exact_hit = Some((
                cloned.kv_caches,
                cloned.lin_caches,
                cloned.first_id,
                cloned.first_piece,
            ));
        }
        // HydratedTail: SSD-hydrated block-aligned prefix that is a strict
        // prefix of the incoming prompt; tail re-prefills on top of the
        // restored KV + GDN lin state (no truncation — recurrent state is
        // resumed at the block boundary).
        Consumed::Reuse {
            entry: cloned,
            kind: ReuseKind::StrictPrefix { prefix_len },
        } => {
            hydrated_tail = Some((cloned.kv_caches, cloned.lin_caches, prefix_len));
        }
        // Unreachable for moe: the ExactOnly policy never permits a non-hydrated
        // partial match, and the hydrated-prefix hook only ever yields
        // `StrictPrefix` (block-truncating a GDN `lin_caches` is unsafe and
        // structurally forbidden). Treat defensively as a Miss (full re-prefill)
        // rather than panicking, since a Miss is always correctness-safe.
        Consumed::Reuse {
            kind: ReuseKind::BlockTruncate { .. },
            ..
        } => {
            tracing::warn!(
                prompt_len = prompt_ids.len(),
                "qwen3_5moe generate_greedy: unexpected BlockTruncate reuse for a GDN arch — \
                 degrading to full re-prefill"
            );
        }
        Consumed::Miss => {}
    }

    // Path A: exact match
    if let Some((mut kv_caches, mut lin_caches, last_id, piece)) = exact_hit {
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
        let (_, post) = {
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
            })?
        };
        // N16: store KV-cache bytes (KV + linear-attn state) for /metrics/cache (post-decode).
        let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
            + lin_caches.iter().map(|c| c.resident_bytes()).sum::<u64>();
        model.kv_bytes.store(kv_bytes, post);
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
    // Chunking (same prefill_chunk as Path C): bounds the per-chunk forward
    // (KV write + GDN kernel) so long tails stay under the Metal
    // command-buffer budget.
    if let Some((mut kv_caches, mut lin_caches, prefix_len)) = hydrated_tail {
        let tail_len = prompt_ids.len() - prefix_len;
        tracing::info!(
            arch = "Qwen3_5MoeForConditionalGeneration",
            prefix_len,
            tail_len,
            "qwen3_5moe: hydrated-tail hit — forwarding tail from SSD-restored KV"
        );

        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe");

        // No enter_prefill above (see the note on resumed caches) means no
        // exit_prefill sweep is owed here, so a failure returns its cause
        // immediately (unlike the shared helper, which must capture and sweep
        // first).
        let prefill_logits = {
            let mut last_logits: Option<Array> = None;
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
                            // Stops at the first failing cache.
                            if let Some(e) =
                                kv_caches.iter().find_map(|c| c.eval_prefill_state().err())
                            {
                                tracing::error!(
                                    error = %e,
                                    chunk_len = chunk.len(),
                                    "qwen3_5moe generate_greedy (hydrated-tail): tail chunk eval failed, aborting generation"
                                );
                                return Err(e);
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            prefix_len,
                            tail_len,
                            "qwen3_5moe generate_greedy (hydrated-tail): tail chunk failed, aborting generation"
                        );
                        return Err(e);
                    }
                }
            }

            last_logits.ok_or_else(|| {
                Error::Model(
                    "qwen3_5moe generate_greedy (hydrated-tail): tail produced no logits"
                        .to_owned(),
                )
            })?
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
                                let stored = cache.push(Qwen35MoeEntry {
                                    prompt_token_ids: prompt_ids.to_vec(),
                                    block_hashes: crate::prompt_cache::chained_block_hashes_seeded(
                                    prompt_ids,
                                    crate::prompt_cache::cache_seed(lk, kv_quant, model.model_sig),
                                ),
                                    kv_caches: kvs,
                                    lin_caches: lins,
                                    first_id: last_id,
                                    first_piece: piece.clone(),
                                    kv_quant: Some(kv_quant),
                                    is_ssd_hydrated: false,
                                });
                                if stored.is_some() {
                                    tracing::debug!(
                                        prompt_len = prompt_ids.len(),
                                        token_id = last_id,
                                        n_slots = cache.slots.len(),
                                        "qwen3_5moe generate_greedy: hydrated-tail — saved full snapshot"
                                    );
                                }
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
        let (_, post) = pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
            model.forward_arr(y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
        })?;

        let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
            + lin_caches.iter().map(|c| c.resident_bytes()).sum::<u64>();
        model.kv_bytes.store(kv_bytes, post);
        return Ok(steps);
    }

    // Issue #25: `--max-ctx` is a virtual ceiling the KV ring grows lazily up
    // to, not an eager allocation. `initial_max_seq` is the lazy start;
    // `max_seq_ceiling` caps growth and rejects over-long prompts.
    let (initial_max_seq, max_seq_ceiling) =
        kv_max_seq_and_ceiling(max_ctx_override, model.cfg.max_position_embeddings as i32);
    // Prefill chunk for qwen3_5_moe defaults to 2048 (see prefill_chunk.rs for
    // the sweep behind that number). The GDN recurrence runs the
    // `gated_delta_step_gpu` kernel at any T, so a large chunk does NOT route to
    // the ops-graph path that used to explode the lazy graph — it just means
    // fewer, bigger forward passes and far fewer per-chunk KV-state evals, which
    // is where the prefill/TTFT win comes from. Override via
    // `RMLX_PREFILL_CHUNK` (global) or `RMLX_PREFILL_CHUNK_QWEN3_5_MOE`
    // (per-arch); the GDN kernel pre-warm in `arch::load_model` reads the same
    // resolved chunk, so set the env BEFORE `rmlx serve` for the warmup shape to
    // match.
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

    // Advise once if the resolved codec is estimated to increase resident KV vs
    // bf16. Keyed on geometry + codec only, exactly as the other arches call it
    // — a codec that costs more memory than bf16 does so because of its store
    // layout, which no architecture can change, so the operator has to hear it
    // on every arch and not only where it happened to be wired first. Only the
    // full-attention layers hold a token-indexed KV cache; the linear-attention
    // (GDN) layers carry a fixed-size recurrent state the codec never touches.
    {
        let layer_shapes: Vec<KvLayerShape> = (0..n_layers)
            .filter(|i| (i + 1).is_multiple_of(model.cfg.full_attention_interval))
            .map(|_| KvLayerShape {
                head_dim: model.cfg.head_dim as u64,
                kv_heads: model.cfg.num_key_value_heads as u64,
                window: None,
            })
            .collect();
        let eff_seq = (max_seq_ceiling.max(0) as u64).max(prompt_ids.len() as u64);
        warn_if_kv_codec_net_negative(kv_quant, &layer_shapes, eff_seq);
    }

    // Fresh chunked prefill via the shared helper. It brackets the loop with
    // enter_prefill() / exit_prefill(), evals only the cache state on non-final
    // chunks, and propagates the cause on rejection. The GDN `lin_caches` are
    // closure-captured — the helper only owns `kv_caches`.
    let prefill_logits = chunked_prefill(
        &mut kv_caches,
        prompt_ids,
        prefill_chunk,
        device,
        "Qwen3_5MoeForConditionalGeneration",
        |chunk, kv| model.forward_seq_with_cache(chunk, Some(kv), Some(&mut lin_caches), device),
    )?;

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
                            let stored = cache.push(Qwen35MoeEntry {
                                prompt_token_ids: prompt_ids.to_vec(),
                                block_hashes: crate::prompt_cache::chained_block_hashes_seeded(
                                    prompt_ids,
                                    crate::prompt_cache::cache_seed(lk, kv_quant, model.model_sig),
                                ),
                                kv_caches: kvs,
                                lin_caches: lins,
                                first_id: last_id,
                                first_piece: piece.clone(),
                                kv_quant: Some(kv_quant),
                                is_ssd_hydrated: false,
                            });
                            if stored.is_some() {
                                tracing::debug!(
                                    prompt_len = prompt_ids.len(),
                                    token_id = last_id,
                                    n_slots = cache.slots.len(),
                                    "qwen3_5moe generate_greedy: prompt cache MISS — saved snapshot"
                                );
                            }
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
    let (stats, post) = pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
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

    // N16: store KV-cache bytes (KV + linear-attn state) for /metrics/cache (post-decode).
    let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
        + lin_caches.iter().map(|c| c.resident_bytes()).sum::<u64>();
    model.kv_bytes.store(kv_bytes, post);

    Ok(steps)
}
