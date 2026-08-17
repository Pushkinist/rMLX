//! Gemma3 greedy generation loop and single-step forward probe.
//!
//! [`generate_greedy`] drives the full autoregressive decode pipeline for
//! Gemma3 models: chunked prefill, KV-cache management, per-token sampling,
//! and streaming token delivery. [`probe_forward`] runs a single forward pass
//! for smoke-probe validation without loading the full generator state.
//!
//! # Public API
//!
//! - [`generate_greedy`] — main generation entry point.
//! - [`probe_forward`] — single-step forward pass for smoke-probe use.

#![allow(
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::items_after_statements,
    clippy::too_many_lines
)]
use std::path::Path;
use std::time::Instant;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::{argmax, max_axis, Array, Device, Dtype};
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};
use tracing::{info, warn};

use crate::constraint::ConstraintEngine;
use crate::decode_loop::{
    capture_logprobs, choose_token, chunked_prefill, pipelined_decode, reject_nan_prefill,
    DecodeCtx,
};
use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};
use crate::prompt_cache::{chained_block_hashes_seeded, Consumed, ReusePolicy};
use rmlx_kv_quant::{KvCache, KV_MAX_SEQ_DEFAULT};

use super::loader::load_from_path;
use super::model::Gemma3Text;
use super::prompt_cache::{active_layout_key, ensure_prompt_cache, Gemma3Entry, PROMPT_CACHE};

// ---------------------------------------------------------------------------
// probe_forward -- CLI entry point
// ---------------------------------------------------------------------------

/// Load a Gemma3 model and run a single-token forward probe.
///
/// Returns `(top_token_id, max_logit)`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn probe_forward(model_dir: &Path, token_id: u32, device: Device) -> Result<(u32, f32)> {
    let model = load_from_path(model_dir)?;
    info!(token_id, "gemma3 forward probe: single-token forward pass");

    let logits = model.forward_seq(&[token_id], device)?;
    let vocab = model.cfg.vocab_size as i32;
    let logits_flat = logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let top = argmax(&logits_flat, -1, device)?;
    top.eval()?;
    let max_val = max_axis(&logits_flat, -1, device)?;
    max_val.eval()?;

    let top_bytes = top.to_bytes()?;
    let top_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;

    let max_bytes = max_val.to_bytes()?;
    let max_f32 = match logits_flat.dtype() {
        Dtype::F32 => f32::from_le_bytes(max_bytes[..4].try_into().unwrap()),
        Dtype::Bf16 => {
            let raw = u16::from_le_bytes(max_bytes[..2].try_into().unwrap());
            f32::from_bits(u32::from(raw) << 16)
        }
        _ => {
            warn!("unexpected logits dtype {:?}", logits_flat.dtype());
            0.0
        }
    };

    Ok((top_id, max_f32))
}

// ---------------------------------------------------------------------------
// Smoke probe -- generate_greedy
// ---------------------------------------------------------------------------

// `count_nan_in_bytes` and `max_abs_from_bytes` are imported from
// `rmlx_runtime::probe`. The byte-level decoder logic
// is shared with qwen2/qwen3/gemma3/laguna.

/// Greedy autoregressive generation using KV-cache prefill + decode.
///
/// `prompt_cache_slots` — number of post-prefill KV snapshots kept across
/// requests. Pass 1 for single-slot; pass N for multi-slot prefix matching.
/// Pass 0 to disable the cache: nothing is stored, so every request prefills.
///
/// Returns `Vec<ProbeStep>` -- same shape as `gemma4::generate_greedy`.
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
    model: &Gemma3Text,
    tokenizer: &'a tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: rmlx_kv_quant::KvQuant,
    max_ctx_override: Option<i32>,
    // number of post-prefill KV snapshots kept across requests. Pass 1 for
    // single-slot; pass N for multi-slot exact-match. Recommended: 4. Only the
    // Exact-hit path is active for gemma3 (ReusePolicy::ExactOnly); an
    // identical-prompt repeat skips re-prefill entirely.
    prompt_cache_slots: usize,
    eos_ids: &'a [u32],
    // The shared `DecodeCtx` bundles every per-request borrow under one
    // lifetime, so these references share `'a` (a `&mut dyn` trait-object
    // reborrow is invariant and cannot be re-unified once split).
    step_fn: &'a mut dyn FnMut(&crate::decode_loop::ProbeStep) -> Option<u32>,
    // A6.2: optional sampler constraint. See gemma4::generate_greedy.
    mut constraint: Option<&'a mut dyn ConstraintEngine>,
    // A7.2: sampling config + per-request RNG. `temperature <= 0.0` keeps the
    // untouched greedy argmax path (`sampler_cfg.sampling_active() == false`).
    sampler_cfg: &'a crate::sampler::SamplerConfig,
    rng: &'a mut crate::sampler::Pcg32,
    // A7.3: logit-penalty configuration + per-request token history.
    penalty_cfg: &'a crate::sampler::PenaltyConfig,
    token_history: &'a mut Vec<u32>,
    // optional precomputed multimodal prefill. `Some(embeds)` carries
    // the scatter-merged `inputs_embeds` `[1, seq, hidden]` from
    // `gemma3::build_inputs_embeds`. When present the prompt cache is bypassed
    // (image prompts are one-shot) and prefill runs from the embeds in one
    // forward instead of chunked token-id forwards.
    image_prefill: Option<Array>,
) -> Result<Vec<crate::decode_loop::ProbeStep>> {
    use crate::decode_loop::ProbeStep;

    tracing::info!(
        arch = "Gemma3ForConditionalGeneration",
        ?kv_quant,
        ?max_ctx_override,
        prompt_cache_slots,
        has_image = image_prefill.is_some(),
        "generate_greedy: selected KV cache quant"
    );

    if n_tokens == 0 {
        return Ok(vec![]);
    }

    // capture before `image_prefill` is moved into the prefill branch.
    // Image prompts must not be served from or saved to the token-id-keyed
    // prompt cache (the K/V depends on scattered vision features, not just the
    // token ids).
    let has_image = image_prefill.is_some();

    let vocab = model.cfg.vocab_size as i32;
    let mut steps = Vec::with_capacity(n_tokens);

    ensure_prompt_cache(prompt_cache_slots);

    // ------------------------------------------------------------------
    // Prompt cache lookup via the shared consume() engine. Gemma3 is a
    // pure-attention arch (no recurrent state) with sliding-window-attention
    // layers, and uses ReusePolicy::ExactOnly: it overrides none of the
    // prefix-reuse hooks, so the only reachable consume outcomes are Exact
    // (identical-prompt repeat skips re-prefill — a full in-memory deep_clone of
    // every layer cache including the SWA ring, so it is safe) and Miss (full
    // re-prefill). Gemma3's ring / SWA-mask differs from gemma4, so the
    // gemma4-style partial / strict-prefix snapshot-restore path is NOT enabled
    // here yet (promotion to Partial is a separate follow-up). The engine owns
    // the find → SSD-hydrate retry → quant-mismatch guard → SSD-hydrated
    // exclusion → Exact decision and traces every degrade branch. The ExactOnly
    // tripwire lives here at the call site — not in the generic engine — because
    // the engine is policy-agnostic and shared across architectures.
    // ------------------------------------------------------------------
    assert_eq!(
        PROMPT_CACHE.policy(),
        ReusePolicy::ExactOnly,
        "Gemma3 prompt cache must be ExactOnly — SWA-first: gemma3's sliding-window ring \
         and mask differ from gemma4, so only a full-token-equality Exact hit (RAM deep_clone, \
         SWA ring included) is reused; partial / strict-prefix reuse is a separate follow-up",
    );

    // image prompts bypass the prompt cache: pass `has_image` so the engine
    // returns Miss without touching the cache (mirrored by the `!has_image`
    // store gate below).
    if !has_image {
        if let Consumed::Exact(cloned) =
            PROMPT_CACHE.consume(prompt_ids, kv_quant, false, model.model_sig)
        {
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
    }

    // Prefill timer; decode timers come from `pipelined_decode`'s `DecodeStats`.
    let prefill_t0 = Instant::now();

    let max_seq = max_ctx_override.unwrap_or(KV_MAX_SEQ_DEFAULT);

    // Allocate one KvCache per decoder layer using the selected quant mode.
    //
    // SWA layers in bf16-KV mode (`KvQuant::None`) use the byte-for-byte
    // RotatingKvCache port. medgemma has the SWA pattern (5 sliding + 1 full).
    use super::config::LayerType;
    let sliding_window_i32 = model.cfg.sliding_window as i32;
    let n_layers = model.cfg.num_hidden_layers;
    // Force K8V8 for boundary layers (first head_n + last tail_n).
    let mut caches: Vec<KvCache> = (0..n_layers)
        .map(|i| {
            let window = match model.cfg.layer_types[i] {
                LayerType::SlidingAttention => Some(sliding_window_i32),
                LayerType::FullAttention => None,
            };
            let q = kv_quant_for_layer(
                i,
                n_layers,
                kv_quant,
                LAYER_ADAPTIVE_TAIL_N,
                LAYER_ADAPTIVE_HEAD_N,
            );
            KvCache::with_quant_max_seq_window(q, max_seq, window).with_layer_idx(i)
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
    // Chunk size is per-arch; default 256 for gemma3, override via
    // `RMLX_PREFILL_CHUNK` (global) or `RMLX_PREFILL_CHUNK_GEMMA3` (per-arch).
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("gemma3");
    // Two prefill flush protocols, kept per-arch:
    //   - image : one forward over the scatter-merged embeds, fresh caches
    //             enter_prefill → forward → exit_prefill (one-shot prompt,
    //             no prompt-cache reuse).
    //   - text  : Fresh full prefill via the shared `chunked_prefill`
    //             (enter_prefill → per-chunk `eval_prefill_state` → exit).
    let prefill_logits = if let Some(embeds) = image_prefill {
        for c in &mut caches {
            c.enter_prefill();
        }
        let seq_i32 = prompt_ids.len() as i32;
        // Every cache entered prefill above, so the exit_prefill sweep below is
        // mandatory even when the forward fails: the cause is captured, the
        // sweep runs, then the first cause propagates.
        let mut first_err: Option<Error> = None;
        let forward = match model.forward_arr_embeds(embeds, seq_i32, Some(&mut caches), device) {
            Ok(l) => Some(l),
            Err(e) => {
                tracing::error!(error = %e, prompt_len = prompt_ids.len(), "gemma3 generate_greedy: image prefill failed, aborting generation");
                first_err = Some(e);
                None
            }
        };
        for c in &mut caches {
            if let Err(e) = c.exit_prefill(device) {
                tracing::error!(error = %e, "gemma3 generate_greedy: image exit_prefill quantization failed, aborting generation");
                if first_err.is_none() {
                    first_err = Some(e);
                }
            }
        }
        if let Some(e) = first_err {
            return Err(e);
        }
        forward.ok_or_else(|| {
            Error::Model("gemma3 generate_greedy: image prefill produced no logits".to_owned())
        })?
    } else {
        chunked_prefill(
            &mut caches,
            prompt_ids,
            prefill_chunk,
            device,
            "Gemma3ForConditionalGeneration",
            |chunk, caches| model.forward_seq_with_cache(chunk, Some(caches), device),
        )?
    };

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());
    reject_nan_prefill(
        "Gemma3ForConditionalGeneration",
        nan_count,
        max_abs_logit,
        prompt_ids.len(),
    )?;

    // top-k logprob capture (0 = disabled, hot-loop zero-overhead).
    let lp_k = sampler_cfg.top_logprobs_k as usize;

    // Build the shared decode context ONCE for the prefill-tail selection AND
    // the decode loop, so the per-request state is borrowed a single time (a
    // `&mut dyn` trait-object reborrow is invariant in its lifetime — two
    // separate `DecodeCtx` over the same params would not compile).
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
        arch: "Gemma3ForConditionalGeneration",
        resolve_pieces: true,
    };

    // Prefill-tail token selection via the shared sampling fork. Gate the mask
    // ONCE here, before the post-selection `advance()` below — `wants_mask` can
    // flip on engagement, so a post-advance recompute would diverge.
    let mask_active = ctx.constraint.as_ref().is_some_and(|c| c.wants_mask());
    let top = choose_token(&mut ctx, &logits_flat, mask_active)?;
    top.eval()?;
    let top_bytes = top.to_bytes()?;
    let last_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    // A6.3: advance constraint regardless of mask state (warm-up scans).
    if let Some(c) = ctx.constraint.as_mut() {
        c.advance(last_id);
    }
    // A7.3: push prefill token into history.
    ctx.token_history.push(last_id);
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
        "gemma3 generate_greedy prefill"
    );

    // Push this prefill snapshot to the prompt cache (Miss → store), gated
    // `!has_image`: an image prompt's K/V is not reconstructible from the
    // token-id cache key, so it must neither be served from nor stored into the
    // cache. Clone the post-prefill KV caches (refcount bump, no data copy)
    // before the decode loop starts writing new decode-step K/V into them.
    // Materialize the GPU arrays on the current inference thread first so a
    // later eviction on a different tokio/Metal thread can re-eval them as a
    // no-op (see gemma4/generate/mod.rs).
    if !has_image {
        let kv_snap: Result<Vec<KvCache>> = caches.iter().map(KvCache::try_deep_clone).collect();
        if let Ok(kvs) = kv_snap {
            match kvs.iter().try_for_each(KvCache::eval_for_spill) {
                Ok(()) => {
                    // Salt the chained block-hash walk with the active layout_key
                    // + KV codec so a slot stored under a different codec / layout
                    // never cross-serves. Identical to the consume() seed.
                    let lk = active_layout_key();
                    let block_hashes = chained_block_hashes_seeded(
                        prompt_ids,
                        crate::prompt_cache::cache_seed(lk, kv_quant, model.model_sig),
                    );
                    let entry = Gemma3Entry {
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
                                let stats = cache.stats();
                                tracing::debug!(
                                    prompt_len = prompt_ids.len(),
                                    cache_hits = stats.hits,
                                    cache_misses = stats.misses,
                                    cache_bytes = stats.bytes,
                                    "gemma3 generate_greedy: pushed snapshot to prompt cache (miss path)"
                                );
                            }
                        }
                    });
                }
                Err(e) => tracing::warn!(error = %e,
                    "gemma3 generate_greedy: pre-eval of KV caches failed, skipping prompt cache store"),
            }
        }
    }

    // prefill is non-pipelined (last_id already materialised), so the prefill
    // token's logprobs come straight from this step's logits.
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

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    // Decode: shared pipelined async loop, reusing the prefill-tail `ctx`. The
    // pipeline ordering (choose_token → async_eval → drain previous pending →
    // feed) overlaps host sampling with the in-flight GPU forward; see
    // decode_loop.rs. gemma3 is pure-KV, so the closure threads only `caches`.
    let (stats, post) = pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
        model.forward_arr(y, 1, Some(&mut caches), device)
    })?;

    let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
    let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
    let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
    let step_ms = (stats.step_total_ns as f64) / 1.0e6;
    let n = f64::from(stats.decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = "Gemma3ForConditionalGeneration",
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

    // store KV-cache bytes for the /metrics/cache endpoint (post-decode).
    model
        .kv_bytes
        .store(caches.iter().map(KvCache::resident_bytes).sum(), post);

    Ok(steps)
}

/// Exact prompt-cache hit: skip re-prefill, replay the stored first token, then
/// run the shared pipelined decode loop on the cloned caches (SWA ring
/// included). `cloned` is the deep-cloned `Gemma3Entry` the consume engine
/// returned; its `kv_caches` are post-prefill state for `prompt_ids`.
#[allow(clippy::too_many_arguments)]
fn exact_hit_decode<'a>(
    model: &Gemma3Text,
    tokenizer: &'a tokenizers::Tokenizer,
    cloned: Gemma3Entry,
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

    let Gemma3Entry {
        kv_caches: mut caches,
        first_id: last_id,
        first_piece: piece,
        ..
    } = cloned;
    let mut steps = Vec::with_capacity(n_tokens);

    tracing::debug!(
        prompt_len = prompt_ids.len(),
        token_id = last_id,
        "gemma3 generate_greedy: prompt cache EXACT HIT"
    );

    step_fn(steps.push_mut(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    }));
    // exact-hit token into history.
    token_history.push(last_id);

    // EOS-stop. If the cached first token is an EOS, no decode steps.
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
            arch: "Gemma3ForConditionalGeneration",
            resolve_pieces: true,
        };
        pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
            model.forward_arr(y, 1, Some(&mut caches), device)
        })?
    };

    let forward_ms = (stats.forward_total_ns as f64) / 1.0e6;
    let eval_ms = (stats.eval_total_ns as f64) / 1.0e6;
    let step_ms = (stats.step_total_ns as f64) / 1.0e6;
    let n = f64::from(stats.decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = "Gemma3ForConditionalGeneration",
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

    // store KV-cache bytes for the /metrics/cache endpoint (post-decode).
    model
        .kv_bytes
        .store(caches.iter().map(KvCache::resident_bytes).sum(), post);

    Ok(steps)
}
