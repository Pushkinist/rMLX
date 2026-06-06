//! BitNet greedy generation loop.

#![allow(
    clippy::cognitive_complexity,
    clippy::collapsible_else_if,
    clippy::items_after_statements,
    clippy::too_many_lines,
    reason = "greedy decode loop: large branchy state machine over sampler/penalty/mask combos; splitting hides the per-step decision tree"
)]

use std::time::Instant;

use rmlx_core::error::Result;
use rmlx_mlx::{argmax, Array, Device, Dtype};
use rmlx_runtime::{count_nan_in_bytes, max_abs_from_bytes};
use tracing::{info, warn};

use crate::constraint::ConstraintEngine;
use crate::kv_cache::{kv_quant_for_layer, LAYER_ADAPTIVE_HEAD_N, LAYER_ADAPTIVE_TAIL_N};
use crate::sampler::apply_mask_argmax;
use rmlx_kv_quant::{KvCache, KV_MAX_SEQ_DEFAULT};

use super::model::BitNetText;

// ---------------------------------------------------------------------------
// Greedy generation
// ---------------------------------------------------------------------------

/// Greedy autoregressive generation for BitNetForCausalLM.
///
/// Returns `Vec<ProbeStep>` — same shape as `gemma4::generate_greedy`.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "steps.last().unwrap() at L231/L360: immediately preceded by steps.push(...), so Vec is non-empty"
)]
pub fn generate_greedy(
    model: &BitNetText,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
    device: Device,
    kv_quant: rmlx_kv_quant::KvQuant,
    max_ctx_override: Option<i32>,
    eos_ids: &[u32],
    step_fn: &mut dyn FnMut(&crate::gemma4::ProbeStep) -> Option<u32>,
    mut constraint: Option<&mut dyn ConstraintEngine>,
    sampler_cfg: &crate::sampler::SamplerConfig,
    rng: &mut crate::sampler::Pcg32,
    penalty_cfg: &crate::sampler::PenaltyConfig,
    token_history: &mut Vec<u32>,
) -> Result<Vec<crate::gemma4::ProbeStep>> {
    use crate::gemma4::ProbeStep;

    tracing::info!(
        arch = "BitNetForCausalLM",
        ?kv_quant,
        ?max_ctx_override,
        "generate_greedy: selected KV cache quant"
    );

    if n_tokens == 0 {
        return Ok(vec![]);
    }

    let vocab = model.cfg.vocab_size as i32;
    let mut steps = Vec::with_capacity(n_tokens);

    let prefill_t0 = Instant::now();
    let max_seq = max_ctx_override.unwrap_or(KV_MAX_SEQ_DEFAULT);

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
            KvCache::with_quant_max_seq_window(q, max_seq, None).with_layer_idx(i)
        })
        .collect();

    // Prefill in chunks.
    let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("bitnet");
    for c in &mut caches {
        c.enter_prefill();
    }

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
                            warn!(
                                error = %e,
                                chunk_len = chunk.len(),
                                "bitnet generate_greedy: prefill chunk cache evaluation failed"
                            );
                            prefill_ok = false;
                            break 'prefill;
                        }
                    }
                }
            }
            Err(e) => {
                warn!(
                    error = %e,
                    prompt_len = prompt_ids.len(),
                    "bitnet generate_greedy: prefill chunk failed, returning empty"
                );
                prefill_ok = false;
                break 'prefill;
            }
        }
    }

    for c in &mut caches {
        if let Err(e) = c.exit_prefill(device) {
            warn!(error = %e, "bitnet generate_greedy: exit_prefill quantization failed");
            prefill_ok = false;
            break;
        }
    }

    if !prefill_ok {
        return Ok(steps);
    }
    let Some(prefill_logits) = last_logits else {
        return Ok(steps);
    };

    let prefill_ns = prefill_t0.elapsed().as_nanos();
    let logits_flat = prefill_logits.reshape(&[1, vocab], device)?;
    logits_flat.eval()?;

    let logit_bytes = logits_flat.to_bytes()?;
    let nan_count = count_nan_in_bytes(&logit_bytes, logits_flat.dtype());
    let max_abs_logit = max_abs_from_bytes(&logit_bytes, logits_flat.dtype());

    let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
    let sampling_active = sampler_cfg.sampling_active();
    let penalties_active = penalty_cfg.penalties_active();
    let win_start = token_history.len().saturating_sub(20);
    let recent = &token_history[win_start..];

    let top = if sampling_active {
        let mask_opt: Option<&[bool]> = if mask_active {
            if let Some(c) = constraint.as_mut() {
                Some(c.step_mask(vocab as usize))
            } else {
                None
            }
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
    } else if penalties_active {
        let mask_opt: Option<&[bool]> = if mask_active {
            if let Some(c) = constraint.as_mut() {
                Some(c.step_mask(vocab as usize))
            } else {
                None
            }
        } else {
            None
        };
        crate::sampler::argmax_with_penalties(&logits_flat, mask_opt, penalty_cfg, recent, device)?
    } else if mask_active {
        if let Some(c) = constraint.as_mut() {
            let m = c.step_mask(vocab as usize);
            apply_mask_argmax(&logits_flat, m, device)?
        } else {
            argmax(&logits_flat, -1, device)?
        }
    } else {
        argmax(&logits_flat, -1, device)?
    };

    top.eval()?;
    let top_bytes = top.to_bytes()?;
    let last_id =
        i32::from_le_bytes([top_bytes[0], top_bytes[1], top_bytes[2], top_bytes[3]]) as u32;

    if let Some(c) = constraint.as_mut() {
        c.advance(last_id);
    }
    token_history.push(last_id);

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
        prefill_ns,
        "bitnet generate_greedy prefill"
    );

    steps.push(ProbeStep {
        token_id: last_id,
        piece: piece.into_boxed_str(),
        max_abs_logit,
        nan_count,
        logprobs: None,
    });
    step_fn(steps.last().unwrap());

    // TODO: wire a base-model degeneracy allowlist or
    // repetition-score guard. Current snapshot is base-model; instruct
    // fine-tunes regressing to base-model behaviour would currently pass smoke
    // silently.
    if nan_count > 0 {
        return Ok(steps);
    }
    if eos_ids.contains(&last_id) {
        return Ok(steps);
    }

    // Decode loop — simple sequential (non-pipelined) for clarity.
    let mut y: Array = {
        let id_i32 = last_id as i32;
        let bytes = id_i32.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::I32)?
    };
    y.eval()?;

    let mut decode_steps: u32 = 0;
    let mut forward_total_ns: u128 = 0;

    for step_idx in 1..n_tokens {
        let fwd_t0 = Instant::now();
        let decode_logits = match model.forward_arr(&y, 1, Some(&mut caches), device) {
            Ok(l) => l,
            Err(e) => {
                warn!(
                    step = step_idx,
                    error = %e,
                    "bitnet generate_greedy: decode step failed, stopping early"
                );
                break;
            }
        };
        forward_total_ns += fwd_t0.elapsed().as_nanos();

        let logits_flat = decode_logits.reshape(&[1, vocab], device)?;

        let mask_active = constraint.as_ref().is_some_and(|c| c.wants_mask());
        let sampling_active = sampler_cfg.sampling_active();
        let penalties_active = penalty_cfg.penalties_active();

        if mask_active {
            logits_flat.eval()?;
        }

        let win_start = token_history.len().saturating_sub(20);
        let recent = &token_history[win_start..];

        let next_y = if sampling_active {
            let mask_opt: Option<&[bool]> = if mask_active {
                if let Some(c) = constraint.as_mut() {
                    Some(c.step_mask(vocab as usize))
                } else {
                    None
                }
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
        } else if penalties_active {
            let mask_opt: Option<&[bool]> = if mask_active {
                if let Some(c) = constraint.as_mut() {
                    Some(c.step_mask(vocab as usize))
                } else {
                    None
                }
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
            if let Some(c) = constraint.as_mut() {
                let m = c.step_mask(vocab as usize);
                apply_mask_argmax(&logits_flat, m, device)?
            } else {
                argmax(&logits_flat, -1, device)?
            }
        } else {
            argmax(&logits_flat, -1, device)?
        };

        next_y.eval()?;
        decode_steps += 1;

        let top_bytes = next_y.to_bytes()?;
        let next_id =
            i32::from_le_bytes([top_bytes[0], top_bytes[1], top_bytes[2], top_bytes[3]]) as u32;

        if let Some(c) = constraint.as_mut() {
            c.advance(next_id);
        }
        token_history.push(next_id);

        let piece = tokenizer
            .id_to_token(next_id)
            .unwrap_or_else(|| format!("<unk:{next_id}>"));

        tracing::debug!(
            step = step_idx,
            token_id = next_id,
            piece = %piece,
            "bitnet generate_greedy decode step"
        );

        steps.push(ProbeStep {
            token_id: next_id,
            piece: piece.into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: None,
        });
        step_fn(steps.last().unwrap());

        y = next_y;

        if eos_ids.contains(&next_id) {
            break;
        }
    }

    let avg_decode_ns = if decode_steps > 0 {
        forward_total_ns / u128::from(decode_steps)
    } else {
        0
    };

    info!(
        arch = "BitNetForCausalLM",
        total_tokens = steps.len(),
        decode_steps,
        avg_decode_ns,
        "generate_greedy: complete"
    );

    Ok(steps)
}
