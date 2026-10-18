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
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use rmlx_core::cancel::ensure_not_cancelled;
use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device};
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};

use crate::constraint::ConstraintEngine;
use crate::decode_loop::{
    capture_first_logprobs, capture_logprobs, choose_token, chunked_prefill, emit_token,
    log_prefill, pipelined_decode, reject_nan_prefill, replay_first_logprobs, DecodeCtx,
    PrefillReport,
};
use crate::kv_cache::{kv_layer_quants, warn_if_kv_codec_net_negative, KvLayerShape};
use crate::prefill_chunk::prefill_pieces;
use crate::sampler::TokenLogprobs;
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};

use super::anchor_capture::AnchorCapture;
use super::model::Qwen3_5MoeText;
use super::prompt_cache::{ensure_prompt_cache, Qwen35MoeEntry, PROMPT_CACHE};
use crate::prompt_cache::{
    anchor_target, entry_is_storable, Anchor, Consumed, ReuseKind, BLOCK_TOKENS,
};

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
    // Optional sampler constraint. See gemma4::generate_greedy for
    // the hot-path-cost-and-correctness contract.
    // The shared `DecodeCtx` bundles every per-request borrow under one
    // lifetime, so these references share `'a` (a `&mut dyn` trait-object
    // reborrow is invariant and cannot be re-unified once split).
    step_fn: &'a mut dyn FnMut(&crate::decode_loop::ProbeStep) -> crate::decode_loop::StepVerdict,
    mut constraint: Option<&'a mut dyn ConstraintEngine>,
    // Sampling config + per-request RNG. See gemma3::generate_greedy.
    sampler_cfg: &'a crate::sampler::SamplerConfig,
    rng: &'a mut crate::sampler::Pcg32,
    // Logit-penalty configuration + per-request token history.
    penalty_cfg: &'a crate::sampler::PenaltyConfig,
    token_history: &'a mut Vec<u32>,
    cancel: &AtomicBool,
) -> Result<Vec<crate::decode_loop::ProbeStep>> {
    use crate::decode_loop::ProbeStep;

    // One resolved label for the whole call. This module serves both Qwen3.5
    // shapes, so a hardcoded string made a dense run emit two different `arch`
    // values in one run's log — splitting exact-field search on it.
    let arch_label = model.arch_class();

    tracing::info!(
        arch = arch_label,
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
    let storable = entry_is_storable(prompt_cache_slots, rmlx_kv_quant::paged::paged_kv_enabled());

    // Prompt cache lookup via the shared consume() engine. The outcomes:
    //   - `Exact`  = cached prompt token-for-token equal to this prompt (full
    //                reuse, no re-prefill).
    //   - `Reuse`  = a state of THIS prompt's own prefix, resumed at
    //                `prefix_len`: the whole stored prompt (`StrictPrefix`) or
    //                the stored anchor (`BlockTruncate`, K/V cut back to it).
    //                Only the tail is forwarded on top.
    //   - `Miss`   = no usable entry.
    // The engine owns find → SSD-hydrate retry → quant-mismatch guard →
    // SSD-hydrated Exact exclusion → reuse-gate, and traces every degrade branch.

    // `--max-ctx` is a virtual ceiling the KV ring grows lazily up
    // to, not an eager allocation. `initial_max_seq` is the lazy start;
    // `max_seq_ceiling` caps growth and rejects over-long prompts. It is
    // resolved before the lookup, as a cold run does, so that a reused entry is
    // bound to this request's ceiling, whatever ceiling it was filled under.
    let resolved_ctx = crate::context::resolve_context(&model.cfg.context, max_ctx_override)?;
    let (initial_max_seq, max_seq_ceiling) = (resolved_ctx.initial_max_seq, resolved_ctx.ceiling);
    let bind =
        |kv_caches| crate::context::rebind_to_ceiling(kv_caches, max_seq_ceiling, prompt_ids.len());

    // image prompts never reach this arch (text path only), so `has_image` is
    // always false here; the engine bypass is belt-and-suspenders.
    let consumed = PROMPT_CACHE.consume(
        prompt_ids,
        kv_quant,
        model.cfg.num_hidden_layers,
        false,
        model.model_sig,
    );

    // `exact_hit` carries the Path-A locals; `resumed` carries the Path-B
    // tail-forward locals. `Consumed::Miss` leaves both `None` and falls
    // through to Path C (full re-prefill).
    let mut exact_hit: Option<ExactHit> = None;
    let mut resumed: Option<Resumed> = None;
    match consumed {
        Consumed::Exact(cloned) => {
            exact_hit = Some(ExactHit {
                kv_caches: bind(cloned.kv_caches)?,
                lin_caches: cloned.lin_caches,
                first_id: cloned.first_id,
                first_piece: cloned.first_piece,
                first_logprobs: cloned.first_logprobs,
            });
        }
        Consumed::Reuse {
            entry: cloned,
            kind,
        } => {
            let (prefix_len, kind) = match kind {
                ReuseKind::StrictPrefix { prefix_len } => (prefix_len, "strict_prefix"),
                ReuseKind::BlockTruncate { effective_blocks } => {
                    (effective_blocks * BLOCK_TOKENS, "anchor")
                }
            };
            resumed = Some(Resumed {
                kv_caches: bind(cloned.kv_caches)?,
                lin_caches: cloned.lin_caches,
                anchor: cloned.anchor,
                prefix_len,
                kind,
            });
        }
        Consumed::Miss(_) => {}
    }

    // Path A: exact match
    if let Some(ExactHit {
        mut kv_caches,
        mut lin_caches,
        first_id: last_id,
        first_piece: piece,
        first_logprobs,
    }) = exact_hit
    {
        tracing::debug!(
            prompt_len = prompt_ids.len(),
            token_id = last_id,
            "qwen3_5moe generate_greedy: prompt cache EXACT HIT"
        );
        log_prefill(arch_label, &PrefillReport::exact(prompt_ids.len()));
        let ends = crate::decode_loop::emit_replayed_token(
            &mut constraint,
            step_fn,
            &mut steps,
            token_history,
            eos_ids,
            ProbeStep {
                token_id: last_id,
                piece: piece.into_boxed_str(),
                max_abs_logit: 0.0,
                nan_count: 0,
                logprobs: replay_first_logprobs(first_logprobs, sampler_cfg.top_logprobs_k),
            },
        );
        if ends {
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
                arch: arch_label,
                resolve_pieces: false,
            };
            pipelined_decode(&mut ctx, last_id, &mut steps, cancel, |y| {
                model.forward_arr(y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
            })?
        };
        // Store KV-cache bytes (KV + linear-attn state) for /metrics/cache (post-decode).
        let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
            + lin_caches.iter().map(|c| c.resident_bytes()).sum::<u64>();
        model.kv_bytes.store(kv_bytes, post);
        return Ok(steps);
    }

    // Path B: resume — the caches hold a prefix of this prompt, the tail needs forwarding.
    //
    // The resumed caches are in POST-exit_prefill (decode) mode: the prefix
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
    if let Some(Resumed {
        mut kv_caches,
        mut lin_caches,
        anchor: inherited_anchor,
        prefix_len,
        kind,
    }) = resumed
    {
        let prefill_t0 = Instant::now();
        let tail_len = prompt_ids.len() - prefix_len;
        tracing::info!(
            arch = arch_label,
            kind,
            prefix_len,
            tail_len,
            "qwen3_5moe: prompt-cache resume — forwarding the tail on the saved state"
        );

        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe");

        // The new entry's anchor is taken from this tail when the tail reaches
        // it; otherwise the resumed entry's own anchor is the right one.
        let cut = anchor_target(prompt_ids.len(), prefix_len);
        let mut anchor_capture = AnchorCapture::new(cut.filter(|_| storable));
        let pieces = prefill_pieces(prefix_len, prompt_ids.len(), prefill_chunk, cut);
        let n_chunks = pieces.len();

        // No enter_prefill above (see the note on resumed caches) means no
        // exit_prefill sweep is owed here, so a failure or a stop returns its
        // cause immediately (unlike the shared helper, which must capture and
        // sweep first).
        let prefill_logits = {
            let mut last_logits: Option<Array> = None;
            for (chunk_idx, piece) in pieces.iter().enumerate() {
                ensure_not_cancelled(cancel)?;
                let is_last = chunk_idx + 1 == n_chunks;
                let chunk = &prompt_ids[piece.clone()];
                match anchor_capture.forward(
                    model,
                    chunk,
                    piece.start,
                    &mut kv_caches,
                    &mut lin_caches,
                    device,
                ) {
                    Ok(logits) => {
                        if is_last {
                            last_logits = Some(logits);
                        } else {
                            // Force-evaluate the KV cache state between chunks
                            // (avoids lazy-graph explosion). These caches are
                            // past `exit_prefill`: `eval_prefill_state` reads
                            // the prefill buffers, which are empty here.
                            // Stops at the first failing cache.
                            if let Some(e) = kv_caches.iter().find_map(|c| c.eval_gpu_state().err())
                            {
                                tracing::error!(
                                    error = %e,
                                    chunk_len = chunk.len(),
                                    "qwen3_5moe generate_greedy (resume): tail chunk eval failed, aborting generation"
                                );
                                return Err(e);
                            }
                            crate::decode_loop::release_chunk_buffers("qwen3_5_moe", chunk_idx);
                        }
                    }
                    Err(e) => {
                        tracing::error!(
                            error = %e,
                            prefix_len,
                            tail_len,
                            "qwen3_5moe generate_greedy (resume): tail chunk failed, aborting generation"
                        );
                        return Err(e);
                    }
                }
            }

            last_logits.ok_or_else(|| {
                Error::Model(
                    "qwen3_5moe generate_greedy (resume): tail produced no logits".to_owned(),
                )
            })?
        };

        let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
        // Same prefill logit stats + NaN guard as the Miss path below: a tail
        // forward off SSD-restored KV can produce a NaN row just as a fresh
        // prefill can, and this path feeds the same decode loop.
        logits_flat.eval()?;
        let prefill_ns = prefill_t0.elapsed().as_nanos();
        let logit_bytes = logits_flat.to_bytes()?;
        let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
        let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());
        reject_nan_prefill(
            arch_label,
            logits_flat.dtype(),
            nan_count,
            max_abs_logit,
            prompt_ids.len(),
        )?;

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
            arch: arch_label,
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
        log_prefill(
            arch_label,
            &PrefillReport::reusing(prompt_ids.len(), prefix_len, prefill_ns),
        );

        let piece = tokenizer
            .id_to_token(last_id)
            .unwrap_or_else(|| format!("<unk:{last_id}>"));

        // Save the completed full-length snapshot (prefix + tail) so future
        // requests can serve this prompt as an Exact hit from RAM.
        let first_logprobs = storable
            .then(|| capture_first_logprobs(&logits_flat, &top))
            .flatten();
        store_snapshot(
            model,
            prompt_ids,
            kv_quant,
            (&kv_caches, &lin_caches),
            anchor_capture.into_anchor().or(inherited_anchor),
            (last_id, &piece, first_logprobs),
            "resume",
        );

        let prefill_logprobs = if lp_k > 0 {
            capture_logprobs(&logits_flat, &top, lp_k)
        } else {
            None
        };
        let first = emit_token(
            &mut *ctx.step_fn,
            &mut steps,
            eos_ids,
            ProbeStep {
                token_id: last_id,
                piece: piece.into_boxed_str(),
                max_abs_logit,
                nan_count,
                logprobs: prefill_logprobs,
            },
        );
        ctx.token_history.push(last_id);

        if first.ends {
            return Ok(steps);
        }

        // Shared pipelined decode loop. GDN `lin_caches` are closure-captured —
        // the loop never learns they exist.
        let (_, post) = pipelined_decode(&mut ctx, last_id, &mut steps, cancel, |y| {
            model.forward_arr(y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
        })?;

        let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
            + lin_caches.iter().map(|c| c.resident_bytes()).sum::<u64>();
        model.kv_bytes.store(kv_bytes, post);
        return Ok(steps);
    }

    // Prefill chunk for qwen3_5_moe comes from `arch_default` in
    // `prefill_chunk.rs`, which records the sweep behind it. The GDN recurrence runs the
    // `gated_delta_step_gpu` kernel at any T, so a large chunk does NOT route to
    // the ops-graph path that used to explode the lazy graph — it just means
    // fewer, bigger forward passes and far fewer per-chunk KV-state evals, which
    // is where the prefill/TTFT win comes from. Override via
    // `RMLX_PREFILL_CHUNK` (global) or `RMLX_PREFILL_CHUNK_QWEN3_5_MOE`
    // (per-arch); the GDN kernel pre-warm in `arch::load_model` reads the same
    // resolved chunk, so set the env BEFORE `rmlx serve` for the warmup shape to
    // match.
    let prefill_chunk = crate::prefill_chunk::resolve("qwen3_5_moe");

    // Path C: cache miss — full prefill from scratch
    let prefill_t0 = Instant::now();
    let n_layers = model.cfg.num_hidden_layers;
    // Force K8V8 for boundary layers (first head_n + last tail_n)
    // to protect output quality with aggressive V compression (planar/k8v4).
    let mut kv_caches: Vec<KvCache> = kv_layer_quants(
        n_layers,
        kv_quant,
        crate::qwen3_5_moe::SHARES_KV_ACROSS_LAYERS,
    )
    .into_iter()
    .enumerate()
    .map(|(i, q)| {
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
        warn_if_kv_codec_net_negative(kv_quant, &layer_shapes, eff_seq, false);
    }

    // Fresh chunked prefill via the shared helper. It brackets the loop with
    // enter_prefill() / exit_prefill(), evals only the cache state on non-final
    // chunks, and propagates the cause on rejection. The GDN `lin_caches` are
    // closure-captured — the helper only owns `kv_caches`.
    let cut = anchor_target(prompt_ids.len(), 0);
    let mut anchor_capture = AnchorCapture::new(cut.filter(|_| storable));
    let mut chunk_start = 0usize;
    let prefill_logits = chunked_prefill(
        &mut kv_caches,
        prompt_ids,
        prefill_chunk,
        cut,
        device,
        "qwen3_5_moe",
        cancel,
        |chunk, kv| {
            let start = chunk_start;
            chunk_start += chunk.len();
            anchor_capture.forward(model, chunk, start, kv, &mut lin_caches, device)
        },
    )?;

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    // Prefill logit stats, the same pair every other arch computes. One host
    // readback of the vocab row per request — not per token — which is what
    // keeps it off the decode hot path while still giving the NaN guard
    // something to read. Without it a NaN prefill on this arch runs to full
    // length and returns garbage with no guard at all.
    //
    // `max_abs_logit` is the only one of the two the emitted ProbeStep can carry
    // a non-trivial value for: the guard returns Err a few lines below, so a
    // step that reaches the classifier always has nan_count == 0 by
    // construction. See SmokeVerdict::BrokenNan.
    logits_flat.eval()?;
    let prefill_ns = prefill_t0.elapsed().as_nanos();
    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());
    reject_nan_prefill(
        arch_label,
        logits_flat.dtype(),
        nan_count,
        max_abs_logit,
        prompt_ids.len(),
    )?;

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
        arch: arch_label,
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
    // Advance constraint regardless of mask state (warm-up scans).
    if let Some(c) = ctx.constraint.as_mut() {
        c.advance(last_id);
    }
    log_prefill(
        arch_label,
        &PrefillReport::cold(prompt_ids.len(), prefill_ns),
    );

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

    let first_logprobs = storable
        .then(|| capture_first_logprobs(&logits_flat, &top))
        .flatten();
    store_snapshot(
        model,
        prompt_ids,
        kv_quant,
        (&kv_caches, &lin_caches),
        anchor_capture.into_anchor(),
        (last_id, &piece, first_logprobs),
        "miss",
    );

    // prefill is non-pipelined (last_id already materialised), so the
    // prefill token's logprobs come straight from this step's logits.
    let prefill_logprobs = if lp_k > 0 {
        capture_logprobs(&logits_flat, &top, lp_k)
    } else {
        None
    };
    let first = emit_token(
        &mut *ctx.step_fn,
        &mut steps,
        eos_ids,
        ProbeStep {
            token_id: last_id,
            piece: piece.into_boxed_str(),
            max_abs_logit,
            nan_count,
            logprobs: prefill_logprobs,
        },
    );
    // prefill first token into history.
    ctx.token_history.push(last_id);

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if first.ends {
        return Ok(steps);
    }

    // Decode: shared pipelined async loop, reusing the prefill-tail `ctx`. The
    // Miss funnel is just "caches + last_id → decode"; the loop is the call
    // site. The GDN `lin_caches` are closure-captured — the loop never learns
    // they exist.
    let (stats, post) = pipelined_decode(&mut ctx, last_id, &mut steps, cancel, |y| {
        model.forward_arr(y, 1, Some(&mut kv_caches), Some(&mut lin_caches), device)
    })?;

    let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
    let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
    let step_ms = (stats.step_total_ns as f64) / 1.0e6;
    let n = f64::from(stats.decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = arch_label,
        n_steps = stats.decode_steps,
        forward_total_ms = forward_ms,
        eval_total_ms = eval_ms,
        step_total_ms = step_ms,
        forward_per_step_ms = forward_ms / n,
        eval_per_step_ms = eval_ms / n,
        "decode_profile"
    );

    // Store KV-cache bytes (KV + linear-attn state) for /metrics/cache (post-decode).
    let kv_bytes: u64 = kv_caches.iter().map(|c| c.resident_bytes()).sum::<u64>()
        + lin_caches.iter().map(|c| c.resident_bytes()).sum::<u64>();
    model.kv_bytes.store(kv_bytes, post);

    Ok(steps)
}

/// What an exact prompt-cache hit hands the decode path: the saved caches and
/// the first token with its logprobs.
struct ExactHit {
    kv_caches: Vec<KvCache>,
    lin_caches: Vec<LinearAttnCache>,
    first_id: u32,
    first_piece: String,
    first_logprobs: Option<TokenLogprobs>,
}

/// What a prompt-cache reuse hands the tail path: caches that hold a prefix of
/// the request's prompt, and the anchor the stored entry carried.
struct Resumed {
    kv_caches: Vec<KvCache>,
    lin_caches: Vec<LinearAttnCache>,
    anchor: Option<Anchor>,
    prefix_len: usize,
    /// Which reuse arm produced it, for the resume event.
    kind: &'static str,
}

/// Save the state a prefill left, so a later request can reuse it.
///
/// `caches` are copied and evaluated first, so the entry holds its own buffers
/// and not a graph over the live caches. `first` is the first decoded token, its
/// piece and its logprobs; `path` names the prefill that produced the state, for
/// the event.
fn store_snapshot(
    model: &Qwen3_5MoeText,
    prompt_ids: &[u32],
    kv_quant: KvQuant,
    caches: (&[KvCache], &[LinearAttnCache]),
    anchor: Option<Anchor>,
    first: (u32, &str, Option<TokenLogprobs>),
    path: &'static str,
) {
    let (kv_caches, lin_caches) = caches;
    let (first_id, first_piece, first_logprobs) = first;
    let arch = PROMPT_CACHE.arch_name();
    let (Some(kvs), Some(lins)) = (
        crate::prompt_cache::snapshot_clone(arch, kv_caches, KvCache::try_deep_clone),
        crate::prompt_cache::snapshot_clone(arch, lin_caches, LinearAttnCache::try_deep_clone),
    ) else {
        return;
    };
    if let Err(e) = kvs
        .iter()
        .try_for_each(|c| c.eval_for_spill())
        .and_then(|()| lins.iter().try_for_each(|c| c.eval_for_spill()))
    {
        tracing::warn!(error = %e, path,
            "qwen3_5moe generate_greedy: pre-eval of KV/lin caches failed, skipping prompt cache store");
        return;
    }
    PROMPT_CACHE.with_inner_mut(|guard| {
        let Some(cache) = guard.as_mut() else {
            return;
        };
        // Salt the chained walk with the active layout_key. With the tier off
        // the helper returns 0, which keeps the digests un-salted.
        let lk = crate::qwen3_5_moe::prompt_cache::active_layout_key();
        let anchor_pos = anchor.as_ref().map(|a| a.pos);
        let stored = cache.push(Qwen35MoeEntry {
            prompt_token_ids: prompt_ids.to_vec(),
            block_hashes: crate::prompt_cache::chained_block_hashes_seeded(
                prompt_ids,
                crate::prompt_cache::request_cache_seed(
                    lk,
                    kv_quant,
                    model.cfg.num_hidden_layers,
                    crate::qwen3_5_moe::SHARES_KV_ACROSS_LAYERS,
                    model.model_sig,
                ),
            ),
            kv_caches: kvs,
            lin_caches: lins,
            anchor,
            first_id,
            first_piece: first_piece.to_owned(),
            first_logprobs,
            kv_quant: Some(kv_quant),
            is_ssd_hydrated: false,
        });
        if stored.is_some() {
            tracing::debug!(
                path,
                prompt_len = prompt_ids.len(),
                token_id = first_id,
                anchor_pos,
                n_slots = cache.slots.len(),
                "qwen3_5moe generate_greedy: prompt cache saved snapshot"
            );
        }
    });
}
