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

use rmlx_core::error::Result;
use rmlx_mlx::{argmax, max_axis, Array, Device, Dtype};
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};
use tracing::{info, warn};

use crate::constraint::ConstraintEngine;
use crate::decode_loop::{
    capture_logprobs, choose_token, chunked_prefill, pipelined_decode, DecodeCtx,
};
use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};
use rmlx_kv_quant::{KvCache, KV_MAX_SEQ_DEFAULT};

use super::loader::load_from_path;
use super::model::Gemma3Text;

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
        "generate_greedy: selected KV cache quant"
    );

    if n_tokens == 0 {
        return Ok(vec![]);
    }

    let vocab = model.cfg.vocab_size as i32;
    let mut steps = Vec::with_capacity(n_tokens);

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
        let logits = match model.forward_arr_embeds(embeds, seq_i32, Some(&mut caches), device) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(error = %e, prompt_len = prompt_ids.len(), "gemma3 generate_greedy: image prefill failed, returning empty");
                for c in &mut caches {
                    let _ = c.exit_prefill(device);
                }
                return Ok(steps);
            }
        };
        for c in &mut caches {
            if let Err(e) = c.exit_prefill(device) {
                tracing::warn!(error = %e, "gemma3 generate_greedy: image exit_prefill quantization failed");
                return Ok(steps);
            }
        }
        logits
    } else {
        let Some(logits) = chunked_prefill(
            &mut caches,
            prompt_ids,
            prefill_chunk,
            device,
            "Gemma3ForConditionalGeneration",
            |chunk, caches| model.forward_seq_with_cache(chunk, Some(caches), device),
        )?
        else {
            return Ok(steps);
        };
        logits
    };

    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());

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

    if nan_count > 0 {
        return Ok(steps);
    }

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    // Decode: shared pipelined async loop, reusing the prefill-tail `ctx`. The
    // pipeline ordering (choose_token → async_eval → drain previous pending →
    // feed) overlaps host sampling with the in-flight GPU forward; see
    // decode_loop.rs. gemma3 is pure-KV, so the closure threads only `caches`.
    let stats = pipelined_decode(&mut ctx, last_id, &mut steps, |y| {
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
        "decode_profile"
    );

    Ok(steps)
}
