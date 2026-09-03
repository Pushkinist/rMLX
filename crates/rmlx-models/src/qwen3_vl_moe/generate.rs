// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret for Array::from_bytes
#![allow(unsafe_code)]
#![allow(clippy::redundant_closure_for_method_calls)]

//! Greedy autoregressive generation for Qwen3-VL-MoE — text + image branches.
//!
//! The Qwen3-VL text decoder is a plain GQA stack with 3D interleaved M-RoPE
//! (the MoE lives in the FFN and never touches the KV cache), so a per-layer
//! [`KvCache`] one-shot prefill + sequential decode is correct and fast enough
//! for the single-image serve path. Sampling / penalty / constraint hooks mirror
//! the other arch generators so the server's decode-loop call site is uniform.
//!
//! ## Prompt cache (text path only)
//!
//! The text [`generate_greedy`] routes through the shared
//! [`crate::prompt_cache::ArchPromptCache::consume`] engine under
//! [`crate::prompt_cache::ReusePolicy::ExactOnly`]: an identical-prompt repeat
//! skips re-prefill (Exact), and every other request re-prefills and snapshots
//! the post-prefill KV (Miss → store). Image turns route to [`generate_image`]
//! (the text path never sees image ids), and both the consume bypass and the
//! store-back are additionally `has_image`-gated as belt-and-suspenders — the
//! token-id cache key is unsafe across image spans.
//!
//! The 3D position bookkeeping is the only Qwen3-VL-specific wrinkle:
//! - text-only: positions are sequential in all three (t,h,w) dims;
//! - image: [`super::mrope::get_rope_index`] builds the prompt positions; the
//!   decode continues from `max(prompt positions) + 1`, incrementing in all
//!   three dims (mirroring `language.py` trailing-text positions).

// kv-layer-quants: uniform — this arch builds every layer at the requested
// codec and does not apply the boundary promotion, unlike the seven arches
// that do. Pre-existing and left alone deliberately: changing it would move
// this arch's KV mixture (memory and, on a quantizing codec, output), which is
// a behaviour change that needs its own proof. Consequence to know: the SSD
// `layout_key` folds the NOMINAL vector, so for this arch it over-describes
// what is built. That is safe in the direction that matters — the key still
// moves whenever the policy or the shape moves — but it will invalidate this
// arch's blocks on a policy change that cannot affect them.

use rmlx_core::error::Result;
use rmlx_mlx::{Array, Device, Dtype};
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};

use crate::constraint::ConstraintEngine;
use crate::context::{resolve_context, ResolvedContext};
use crate::decode_loop::{reject_nan_prefill, ProbeStep};
use crate::prompt_cache::{chained_block_hashes_seeded, Consumed, ReusePolicy};
use crate::sampler::{apply_mask_argmax, sample_token_array, Pcg32, PenaltyConfig, SamplerConfig};
use rmlx_kv_quant::{KvCache, KvQuant};

use super::image::{scatter_vision_features, visual_token_positions};
use super::model::Qwen3VlMoeText;
use super::mrope::{get_rope_index, RopeIndex3D};
use super::prompt_cache::{active_layout_key, ensure_prompt_cache, Qwen3VlMoeEntry, PROMPT_CACHE};
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

/// Decode one token id to its display piece (SentencePiece `▁` → space). Shared
/// by `make_step` and the Miss-path snapshot store-back so a later Exact-hit
/// replays a byte-identical piece without re-decoding.
fn piece_for(token_id: u32, tokenizer: &tokenizers::Tokenizer) -> String {
    tokenizer
        .id_to_token(token_id)
        .unwrap_or_default()
        .replace('\u{2581}', " ")
}

/// Abort the request when the prefill logit row carries NaN.
///
/// Called by both prefill paths (text and image), which feed the same decode
/// loop and select through the same argmax. One host readback of the vocab row
/// per request — not per token, so the decode hot path is untouched. Without it
/// a NaN prefill on this arch runs to full length and returns garbage with no
/// guard and no verdict.
fn guard_prefill_logits(logits: &Array, prompt_len: usize) -> Result<()> {
    Array::eval(logits)?;
    let bytes = logits.to_bytes()?;
    let nan_count = count_nan_in_bytes(&bytes, logits.dtype());
    let max_abs_logit = max_abs_from_bytes(&bytes, logits.dtype());
    reject_nan_prefill(
        "Qwen3VLMoeForConditionalGeneration",
        logits.dtype(),
        nan_count,
        max_abs_logit,
        prompt_len,
    )
}

/// Resolve this checkpoint's context bounds. Qwen3-VL declares no RoPE
/// scaling, so its positional capacity is the trained window.
fn resolved_context(
    model: &Qwen3VlMoeText,
    max_ctx_override: Option<i32>,
) -> Result<ResolvedContext> {
    resolve_context(&model.cfg.context, max_ctx_override)
}

/// Emit one decode step (token id + piece). Logit stats are left at defaults —
/// the smoke classifier only needs ids + pieces, and the hot loop avoids the
/// extra GPU→host transfer of the full logit row.
fn make_step(token_id: u32, tokenizer: &tokenizers::Tokenizer) -> ProbeStep {
    ProbeStep {
        token_id,
        piece: piece_for(token_id, tokenizer).into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    }
}

/// Greedy generation from plain text token ids.
///
/// `prompt_cache_slots` — number of post-prefill KV snapshots kept across
/// requests. Pass 1 for single-slot; pass N for multi-slot prefix matching.
/// Recommended: 4. Only the Exact-hit path is active under
/// [`ReusePolicy::ExactOnly`] (identical-prompt repeat skips re-prefill
/// entirely, same contract as Qwen2 / Qwen3 dense). Pass 0 to disable the cache:
/// nothing is stored, so every request prefills. `max_ctx_override` sizes the
/// KV ring ceiling so a long prompt (up to the effective `--max-ctx`) grows to
/// fit and an over-cap prompt is rejected cleanly; see [`generate_image`].
#[allow(clippy::too_many_arguments)]
pub fn generate_greedy(
    model: &Qwen3VlMoeText,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: KvQuant,
    max_ctx_override: Option<i32>,
    prompt_cache_slots: usize,
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
    let mut steps = Vec::with_capacity(n_tokens);

    ensure_prompt_cache(prompt_cache_slots);

    // ------------------------------------------------------------------
    // Prompt cache lookup via the shared consume() engine. The Qwen3-VL text
    // decoder is plain GQA with no recurrent state and uses
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
        "Qwen3-VL-MoE prompt cache must be ExactOnly — pure-attention text decoder \
         with no recurrent state and an Exact-hit-dominated workload; partial-prefix \
         reuse is not wired",
    );
    // The text path never carries image ids (images route to generate_image), so
    // has_image is always false here; the engine's bypass is belt-and-suspenders.
    let has_image = false;
    let consumed = PROMPT_CACHE.consume(prompt_ids, kv_quant, n_layers, has_image, model.model_sig);

    // Path A: exact cache hit — skip re-prefill, replay the stored first token,
    // then run the shared decode loop on the cloned caches.
    if let Consumed::Exact(cloned) = consumed {
        let Qwen3VlMoeEntry {
            kv_caches: mut kv,
            first_id: first,
            first_piece,
            ..
        } = cloned;
        tracing::debug!(
            prompt_len = prompt_ids.len(),
            token_id = first,
            "qwen3_vl_moe generate_greedy: prompt cache EXACT HIT"
        );
        // The cached first_id is the same token the prefill argmax produced, and
        // first_piece is its piece_for(...) string (stored at Miss-path push), so
        // replaying it on the cloned caches yields the same decoded sequence as a
        // cold run. Subsequent steps re-derive their piece via make_step.
        let post = decode_from(
            model,
            tokenizer,
            &mut kv,
            first,
            Some(first_piece),
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
        )?;
        let kv_bytes: u64 = kv.iter().map(|c| c.resident_bytes()).sum();
        model.kv_bytes.store(kv_bytes, post);
        return Ok(steps);
    }

    // Path B (Miss): chunked prefill from scratch. Size the KV ring from the
    // effective `--max-ctx` (lazy start + growth ceiling) so a prompt longer
    // than the lazy KV_MAX_SEQ_DEFAULT=4096 start grows to fit instead of
    // overflowing the fixed decode buffer; an over-cap prompt is rejected
    // cleanly with KvCeilingExceeded (→ context_overflow). The shared
    // chunked_prefill helper brackets enter_prefill()/exit_prefill() and flushes
    // each chunk's command buffer under the ~10s Metal GPU watchdog — a
    // single-shot forward over a multi-thousand-token prompt trips it. Mirrors
    // the other arch text paths (qwen3_5_moe / gemma4).
    let ctx = resolved_context(model, max_ctx_override)?;
    let (initial_max_seq, max_seq_ceiling) = (ctx.initial_max_seq, ctx.ceiling);
    let mut kv: Vec<KvCache> = (0..n_layers)
        .map(|i| {
            KvCache::with_quant_max_seq(kv_quant, initial_max_seq)
                .with_max_seq_ceiling(max_seq_ceiling)
                .with_layer_idx(i)
        })
        .collect();

    let prefill_chunk = crate::prefill_chunk::resolve("qwen3_vl_moe");
    let logits = crate::decode_loop::chunked_prefill(
        &mut kv,
        prompt_ids,
        prefill_chunk,
        device,
        "Qwen3VLMoeForConditionalGeneration",
        |chunk, kv| model.forward_seq_with_cache(chunk, Some(kv), device),
    )?;
    guard_prefill_logits(&logits, prompt_ids.len())?;
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

    // Push this prefill snapshot to the prompt cache (Miss → store), gated on
    // !has_image so an image turn can never pollute the text cache. Clone the
    // post-prefill KV caches (refcount bump, no data copy) before the decode loop
    // starts writing new decode-step K/V into them. Materialize the GPU arrays on
    // the current inference thread first so a later eviction on a different
    // tokio/Metal thread can re-eval them as a no-op (see qwen3.rs).
    //
    // `kv_cache_bytes` is NOT sampled here — this is the prefill snapshot,
    // before the decode ring is allocated. It is recorded post-decode below,
    // for every request (text and image), at the same lifecycle point as the
    // exact-hit path.
    if !has_image {
        let cloned_caches: Result<Vec<KvCache>> = kv.iter().map(|c| c.try_deep_clone()).collect();
        if let Ok(kv_snapshot) = cloned_caches {
            match kv_snapshot.iter().try_for_each(|c| c.eval_for_spill()) {
                Ok(()) => {
                    // Salt the chained block-hash walk with the active layout_key
                    // + KV codec so a slot stored under a different codec / layout
                    // never cross-serves. Identical to the consume() seed, so
                    // find_best_prefix matches what is stored here.
                    let seed = crate::prompt_cache::request_cache_seed(
                        active_layout_key(),
                        kv_quant,
                        n_layers,
                        super::SHARES_KV_ACROSS_LAYERS,
                        model.model_sig,
                    );
                    let block_hashes = chained_block_hashes_seeded(prompt_ids, seed);
                    let first_piece = piece_for(first, tokenizer);
                    let entry = Qwen3VlMoeEntry {
                        prompt_token_ids: prompt_ids.to_vec(),
                        block_hashes,
                        kv_caches: kv_snapshot,
                        first_id: first,
                        first_piece,
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
                                    "qwen3_vl_moe generate_greedy: pushed snapshot to prompt cache (miss path)"
                                );
                            }
                        }
                    });
                }
                Err(e) => tracing::warn!(error = %e,
                    "qwen3_vl_moe generate_greedy: pre-eval of KV caches failed, skipping prompt cache store"),
            }
        }
    }

    let post = decode_from(
        model,
        tokenizer,
        &mut kv,
        first,
        None,
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
    )?;
    // Store KV-cache bytes post-decode: the decode ring is resident now, so the
    // sample includes it on ring-backed codecs. Same lifecycle point as the
    // exact-hit path above.
    let kv_bytes: u64 = kv.iter().map(|c| c.resident_bytes()).sum();
    model.kv_bytes.store(kv_bytes, post);
    Ok(steps)
}

/// Shared sequential decode loop for both the Exact-hit and Miss text paths.
///
/// `first` is the step-0 token (prefill argmax or replayed cache first token);
/// the loop emits it, then drives the remaining steps with the per-token
/// `forward_seq_with_cache(&[feed], …)` continuation. `first_piece` is the
/// stored display piece for `first` on the Exact path (set by `piece_for` at the
/// Miss-path push, so it is byte-identical to what `make_step` produces); `None`
/// re-derives it via `make_step`, matching the Miss path exactly.
///
/// The per-step decode math (token push, `step_fn` forced-feed, EOS break,
/// constraint advance, `pick_token`) is byte-identical to the original inline
/// loop; only the function boundary + the step-0 piece source differ (and that
/// source produces the same string on both paths via `piece_for`).
#[allow(clippy::too_many_arguments)]
fn decode_from(
    model: &Qwen3VlMoeText,
    tokenizer: &tokenizers::Tokenizer,
    kv: &mut [KvCache],
    first: u32,
    first_piece: Option<String>,
    n_tokens: usize,
    vocab: usize,
    device: Device,
    eos_ids: &[u32],
    steps: &mut Vec<ProbeStep>,
    step_fn: &mut dyn FnMut(&ProbeStep) -> Option<u32>,
    constraint: &mut Option<&mut dyn ConstraintEngine>,
    sampler_cfg: &SamplerConfig,
    rng: &mut Pcg32,
    penalty_cfg: &PenaltyConfig,
    token_history: &mut Vec<u32>,
) -> Result<crate::decode_loop::PostDecode> {
    let mut next = first;
    let mut first_piece = first_piece;
    for _ in 0..n_tokens {
        token_history.push(next);
        let step = match first_piece.take() {
            Some(piece) => ProbeStep {
                token_id: next,
                piece: piece.into_boxed_str(),
                max_abs_logit: 0.0,
                nan_count: 0,
                logprobs: None,
            },
            None => make_step(next, tokenizer),
        };
        let forced = step_fn(&step);
        steps.push(step);
        if eos_ids.contains(&next) {
            break;
        }
        if let Some(c) = constraint.as_mut() {
            c.advance(next);
        }
        let feed = forced.unwrap_or(next);
        let logits = model.forward_seq_with_cache(&[feed], Some(&mut *kv), device)?;
        next = pick_token(
            &logits,
            vocab,
            sampler_cfg,
            penalty_cfg,
            constraint,
            token_history,
            rng,
            device,
        )?;
    }
    // Final act of the decode phase: mint the post-decode witness for the
    // caller's `store_kv_cache_bytes`.
    Ok(crate::decode_loop::PostDecode::seal())
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
    max_ctx_override: Option<i32>,
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
    // Materialize the scatter-merged embeds before the chunked prefill so the
    // full-sequence scatter is its own command buffer, not folded into the first
    // prefill chunk.
    Array::eval(&inputs_embeds)?;

    // 2. 3D M-RoPE positions for the augmented sequence.
    let pos = get_rope_index(&ids_i64, image_grids, image_token_id, spatial_merge_size)?;

    // 3. prefill (deepstack injected after layers 0..len(deepstack)).
    //
    // The augmented image prompt is long (thousands of image soft tokens for
    // native Qwen3-VL tiling — e.g. a 2560×2560 image → ~6400 soft tokens),
    // far above the lazy KV_MAX_SEQ_DEFAULT=4096 start. Size the KV ring from
    // the effective `--max-ctx`: `initial_max_seq` is the lazy start and
    // `max_seq_ceiling` caps lazy growth and rejects an over-cap prompt with a
    // clean `KvCeilingExceeded` (→ context_overflow) instead of a cryptic
    // `slice_update` broadcast. Bracketing the chunked forward with
    // enter_prefill()/exit_prefill() routes the prefill through the lazy-grow
    // raw buffer (mirrors the Gemma4 image path) so it grows to fit.
    let ctx = resolved_context(model, max_ctx_override)?;
    let (initial_max_seq, max_seq_ceiling) = (ctx.initial_max_seq, ctx.ceiling);
    let n_layers = model.cfg.num_hidden_layers;
    let mut kv: Vec<KvCache> = (0..n_layers)
        .map(|i| {
            KvCache::with_quant_max_seq(kv_quant, initial_max_seq)
                .with_max_seq_ceiling(max_seq_ceiling)
                .with_layer_idx(i)
        })
        .collect();
    for c in &mut kv {
        c.enter_prefill();
    }
    // Chunk the image prefill so a long augmented prompt (thousands of image
    // soft tokens) does not run a single multi-thousand-token forward in one
    // Metal command buffer (the ~10s GPU watchdog). The chunk size comes from
    // the per-arch prefill-chunk table (512 for this plain-GQA MoE arch).
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_vl_moe");
    let logits = model.forward_embeds_chunked(
        &inputs_embeds,
        seq,
        &pos,
        &vision.deepstack_embeds,
        &visual_positions,
        prefill_chunk,
        &mut kv,
        device,
    )?;
    for c in &mut kv {
        c.exit_prefill(device)?;
    }

    guard_prefill_logits(&logits, aug_ids.len())?;

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

#[cfg(test)]
#[path = "generate_tests.rs"]
mod generate_tests;
