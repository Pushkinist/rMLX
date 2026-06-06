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
use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};
use crate::sampler::apply_mask_argmax;
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
pub fn generate_greedy(
    model: &Gemma3Text,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: rmlx_kv_quant::KvQuant,
    max_ctx_override: Option<i32>,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&crate::gemma4::ProbeStep) -> Option<u32>,
    // A6.2: optional sampler constraint. See gemma4::generate_greedy.
    mut constraint: Option<&mut dyn ConstraintEngine>,
    // A7.2: sampling config + per-request RNG. `temperature <= 0.0` keeps the
    // untouched greedy argmax path (`sampler_cfg.sampling_active() == false`).
    sampler_cfg: &crate::sampler::SamplerConfig,
    rng: &mut crate::sampler::Pcg32,
    // A7.3: logit-penalty configuration + per-request token history.
    penalty_cfg: &crate::sampler::PenaltyConfig,
    token_history: &mut Vec<u32>,
    // optional precomputed multimodal prefill. `Some(embeds)` carries
    // the scatter-merged `inputs_embeds` `[1, seq, hidden]` from
    // `gemma3::build_inputs_embeds`. When present the prompt cache is bypassed
    // (image prompts are one-shot) and prefill runs from the embeds in one
    // forward instead of chunked token-id forwards.
    image_prefill: Option<Array>,
) -> Result<Vec<crate::gemma4::ProbeStep>> {
    use crate::gemma4::ProbeStep;

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

    // Decode profile timers. See gemma4::generate_greedy for rationale.
    let mut forward_total_ns: u128 = 0;
    let mut eval_total_ns: u128 = 0;
    let mut step_total_ns: u128 = 0;
    let mut decode_steps: u32 = 0;
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
    for c in &mut caches {
        c.enter_prefill();
    }
    // image prompts prefill from scatter-merged embeds in a single
    // forward (the run is one-shot; no prompt-cache reuse). Text prompts take
    // the chunked token-id path below — byte-identical to pre-.
    let prefill_logits = if let Some(embeds) = image_prefill {
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
                                tracing::warn!(
                                    error = %e,
                                    chunk_len = chunk.len(),
                                    "gemma3 generate_greedy: prefill chunk cache eval failed"
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
                        "gemma3 generate_greedy: prefill chunk failed, returning empty"
                    );
                    prefill_ok = false;
                    break 'prefill;
                }
            }
        }
        for c in &mut caches {
            if let Err(e) = c.exit_prefill(device) {
                tracing::warn!(error = %e, "gemma3 generate_greedy: exit_prefill quantization failed");
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
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());

    // A6.3: only apply mask when the engine is engaged (wants_mask).
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
    // A6.3: advance constraint regardless of mask state (warm-up scans).
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
        "gemma3 generate_greedy prefill"
    );

    steps.push(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit,
        nan_count,
        logprobs: None,
    });
    step_fn(steps.last().unwrap());

    if nan_count > 0 {
        return Ok(steps);
    }

    // EOS-stop. If prefill emitted an EOS already, no decode steps.
    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    // Decode: pipelined async pattern. Each iteration dispatches
    // step i+1's forward while step i's argmax materialises in the
    // background. GPU sync only happens on the *previous* step's `pending`
    // via `to_bytes()`. Mirrors the qwen3 path C decoder.
    //
    // The `last_id` u32 is only used for the very first decode step; from
    // step 2 onward the next-token Array (`y`) is fed to the forward without
    // a CPU readback. Per-step NaN/max-abs diagnostics are dropped from the
    // hot path — the prefill check above still catches catastrophic-quant
    // failures.
    //
    // EOS-stop after pending drain (see gemma4::generate_greedy).
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
        let decode_logits = match model.forward_arr(&y, 1, Some(&mut caches), device) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(
                    step = step_idx,
                    error = %e,
                    "gemma3 generate_greedy: decode step failed, stopping early"
                );
                break;
            }
        };

        let logits_flat = decode_logits.reshape(&[1, vocab], device)?;
        // A6.3: only apply mask when engine is engaged. See qwen3.rs for rationale.
        let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
        // A7.2: temp>0 also needs a per-step host sync (the sampler reads
        // logits to host), so it shares the masked branch's pre-drain path;
        // the pipelined unconstrained drain is skipped. temp<=0 keeps the
        // exact `mask_active`-gated behaviour byte-for-byte.
        let sampling_active = sampler_cfg.sampling_active();
        // A7.3: penalties also require logits on host → fold into drain_now.
        let penalties_active = penalty_cfg.penalties_active();
        let drain_now = mask_active || sampling_active || penalties_active;
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
                "gemma3 generate_greedy: EOS emitted (pre-drain, masked), stopping"
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
        let fwd_dt = fwd_t0.elapsed().as_nanos();

        // Consume the previous step's pending argmax (unconstrained + warm-up).
        let eval_t0 = Instant::now();
        let mut emitted_eos = false;
        if !drain_now {
            if let Some(p) = pending.take() {
                let top_bytes = p.to_bytes()?;
                let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                if let Some(c) = constraint.as_mut() {
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
                    "gemma3 generate_greedy decode step (pipelined emit)"
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

        forward_total_ns += fwd_dt;
        eval_total_ns += eval_dt;
        step_total_ns += step_t0.elapsed().as_nanos();
        decode_steps += 1;

        if emitted_eos {
            early_stop = true;
            tracing::debug!(
                step = step_idx - 1,
                "gemma3 generate_greedy: EOS emitted, stopping decode loop"
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
            eval_total_ns += drain_t0.elapsed().as_nanos();
            let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
            // A6.2: advance constraint on final drain.
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

    let prefill_ms = (prefill_total_ns as f64) / 1.0e6;
    let forward_ms = (forward_total_ns as f64) / 1.0e6;
    let eval_ms = (eval_total_ns as f64) / 1.0e6;
    let step_ms = (step_total_ns as f64) / 1.0e6;
    let n = f64::from(decode_steps.max(1));
    tracing::info!(
        target: "decode_profile",
        arch = "Gemma3ForConditionalGeneration",
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
