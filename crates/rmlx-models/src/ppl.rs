//! — Native perplexity scorer for sliding-window text evaluation.
//!
//! Computes per-position negative log-likelihood (NLL) of the *actual* prompt
//! token at every interior window position, accumulates the running mean, and
//! returns `PPL = exp(mean(NLL))` over the whole corpus.
//!
//! # Scope
//!
//! Implemented for [`crate::Architecture::Qwen3`] (, Bonsai smoke target)
//! and [`crate::Architecture::Gemma4`]. Other architectures return
//! [`PplError::ArchUnsupported`]; adding parity paths for Qwen3.5MoE / Qwen3VL
//! is a follow-up.
//!
//! # Algorithm
//!
//! Standard fixed-stride sliding-window PPL — matches the `mlx-lm`
//! `perplexity.py` reference math and HF transformers' "Perplexity of fixed
//! length models" pattern:
//!
//! 1. Slide a window of `ctx_window` over `tokens`, stride `stride`.
//! 2. Forward the window with `Qwen3Text::forward_seq_logits_all` -> logits
//!    `[1, S, vocab]`.
//! 3. For every interior position `t in [warmup .. S-1)` (so `t` has a
//!    next-token id at `t+1`), compute `log_softmax(logits[t])[tokens[t+1]]`
//!    and accumulate `-logprob`.
//! 4. `warmup` for the first window is `0`; for subsequent windows it is
//!    `ctx_window - stride` so already-scored positions are not double-counted.
//! 5. `PPL = exp(sum(nll) / count)`.
//!
//! Compute is sync per the project's "compute sync, async only at boundaries"
//! rule. One GPU-to-host transfer per window for the logits buffer.

#![allow(clippy::cognitive_complexity)]
#![allow(trivial_casts)]
use rmlx_core::error::{Error, Result};
use rmlx_mlx::{Array, Device, Dtype};
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

use crate::arch::Architecture;
use crate::gemma4::Gemma4Text;
use crate::qwen3::Qwen3Text;

/// Errors that can short-circuit a PPL scoring run.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum PplError {
    /// The model architecture does not yet have the `forward_seq_logits_all`
    /// path the scorer needs. See module docs.
    ///
    /// implements Qwen3; adds Gemma4.
    #[error("ppl: architecture '{arch}' is not supported (supported: Qwen3, Gemma4)")]
    ArchUnsupported {
        /// The architecture name that is not yet supported.
        arch: String,
    },
    /// Window parameters are not usable (e.g. `stride > ctx_window`, or the
    /// corpus is shorter than one window of useful tokens).
    #[error("ppl: invalid window config: {msg}")]
    InvalidWindow {
        /// Human-readable explanation of the invalid configuration.
        msg: String,
    },
    /// Underlying forward / tensor error.
    #[error("ppl: forward failed: {0}")]
    Forward(#[from] Error),
}

/// One run's PPL result + the accumulator pieces that fed it. Returned by
/// `compute_ppl` so callers can record audit fields (token counts, window
/// count) alongside the headline number.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed report struct — four fields are the complete PPL-run result contract; adding a field requires updating all compute_ppl callers"
)]
#[derive(Debug, Clone, Copy)]
pub struct PplReport {
    /// `exp(sum_nll / count)`.
    pub ppl: f64,
    /// Mean per-token natural-log NLL (`sum_nll / count`).
    pub mean_nll: f64,
    /// Number of scored positions (denominator of `mean_nll`).
    pub scored_tokens: usize,
    /// Number of forward windows the scorer ran.
    pub windows: usize,
}

/// Sliding-window PPL scorer.
///
/// `ctx_window` -- number of tokens forwarded per window (e.g. 4096).
/// `stride` -- gap between consecutive window starts. When
/// `stride < ctx_window` each window's first
/// `(ctx_window - stride)` positions are skipped (they were
/// already scored in the previous window).
///
/// `tokens` must be the corpus tokenized end-to-end. Returns
/// `Err(PplError::ArchUnsupported)` for any architecture other than Qwen3 or
/// Gemma4.
#[instrument(skip(arch, tokens), fields(arch_class = arch.arch_class(), n_tokens = tokens.len(), ctx_window, stride))]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant variants; exhaustive expansion would require updating on every new variant"
)]
pub fn compute_ppl(
    arch: &Architecture,
    tokens: &[u32],
    ctx_window: usize,
    stride: usize,
    device: Device,
) -> std::result::Result<PplReport, PplError> {
    match arch {
        Architecture::Qwen3(m) => compute_ppl_qwen3(m, tokens, ctx_window, stride, device),
        Architecture::Gemma4(m) => compute_ppl_gemma4(m, tokens, ctx_window, stride, device),
        other => Err(PplError::ArchUnsupported {
            arch: other.arch_class().to_string(),
        }),
    }
}

#[instrument(skip(model, tokens), fields(n_tokens = tokens.len(), ctx_window, stride))]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn compute_ppl_qwen3(
    model: &Qwen3Text,
    tokens: &[u32],
    ctx_window: usize,
    stride: usize,
    device: Device,
) -> std::result::Result<PplReport, PplError> {
    if ctx_window < 2 {
        return Err(PplError::InvalidWindow {
            msg: format!("ctx_window must be >= 2, got {ctx_window}"),
        });
    }
    if stride == 0 || stride > ctx_window {
        return Err(PplError::InvalidWindow {
            msg: format!("stride must be in 1..={ctx_window}, got {stride}"),
        });
    }
    if tokens.len() < 2 {
        return Err(PplError::InvalidWindow {
            msg: format!("corpus must have >=2 tokens, got {}", tokens.len()),
        });
    }

    let vocab = model.cfg.vocab_size;
    let mut sum_nll: f64 = 0.0;
    let mut count: usize = 0;
    let mut windows: usize = 0;

    let mut start: usize = 0;
    let mut first = true;
    while start < tokens.len() {
        let end = (start + ctx_window).min(tokens.len());
        let window = &tokens[start..end];
        if window.len() < 2 {
            break;
        }

        // Forward this window -> [1, S, vocab] logits.
        let logits = model.forward_seq_logits_all(window, device)?;
        let host = logits_3d_to_host_f32(&logits, window.len(), vocab)?;

        // warmup = number of leading positions whose NLL was already counted
        // in the previous window. First window: 0. Subsequent windows: the
        // overlap = ctx_window - stride.
        let warmup = if first {
            0
        } else {
            ctx_window.saturating_sub(stride).min(window.len() - 1)
        };

        // Score positions [warmup .. window.len() - 1) -- predicting token at
        // window[t+1] from logits[t].
        for t in warmup..(window.len() - 1) {
            let next_id = window[t + 1] as usize;
            if next_id >= vocab {
                warn!(
                    token_id = next_id,
                    vocab, "ppl: token id out of vocab range -- skipping position"
                );
                continue;
            }
            let row = &host[t * vocab..(t + 1) * vocab];
            let nll = neg_log_softmax_at(row, next_id);
            if nll.is_finite() {
                sum_nll += f64::from(nll);
                count += 1;
            } else {
                warn!(
                    t,
                    token_id = next_id,
                    nll,
                    "ppl: non-finite NLL at position -- skipping"
                );
            }
        }

        windows += 1;
        debug!(
            window_idx = windows - 1,
            start,
            end,
            warmup,
            scored_so_far = count,
            mean_nll_running = if count > 0 {
                sum_nll / count as f64
            } else {
                0.0
            },
            "ppl: window scored"
        );

        // Advance. If the window already reached the corpus end, stop.
        if end == tokens.len() {
            break;
        }
        start += stride;
        first = false;
    }

    if count == 0 {
        return Err(PplError::InvalidWindow {
            msg: "no scored positions -- corpus shorter than ctx_window?".to_owned(),
        });
    }

    let mean_nll = sum_nll / count as f64;
    let ppl_value = mean_nll.exp();
    info!(
        ppl = ppl_value,
        mean_nll,
        scored_tokens = count,
        windows,
        "ppl: run complete"
    );
    Ok(PplReport {
        ppl: ppl_value,
        mean_nll,
        scored_tokens: count,
        windows,
    })
}

/// sliding-window PPL scorer for the Gemma4 family.
///
/// Mirrors `compute_ppl_qwen3` with one Gemma4-specific addition: a BOS token
/// (id = 2) is prepended to the corpus when not already present. Gemma4 is
/// trained with BOS as a start-of-document signal; without it the model
/// produces degenerate logit distributions at position 0 and inflated NLL
/// throughout the sequence. Prepending BOS before the first window matches
/// the convention used by HF `evaluate.load("perplexity")` and
/// lm-evaluation-harness for BOS-bearing models.
///
/// Uses `Gemma4Text::forward_seq_logits_all` to produce the full
/// `[1, seq, vocab]` logit tensor in one Metal dispatch — no cache state,
/// fresh window each call. Final-logit softcapping is applied inside
/// `forward_seq_logits_all` via `apply_softcap`; the NLL computation here
/// is unaffected.
#[instrument(skip(model, tokens), fields(n_tokens = tokens.len(), ctx_window, stride))]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn compute_ppl_gemma4(
    model: &Gemma4Text,
    tokens: &[u32],
    ctx_window: usize,
    stride: usize,
    device: Device,
) -> std::result::Result<PplReport, PplError> {
    if ctx_window < 2 {
        return Err(PplError::InvalidWindow {
            msg: format!("ctx_window must be >= 2, got {ctx_window}"),
        });
    }
    if stride == 0 || stride > ctx_window {
        return Err(PplError::InvalidWindow {
            msg: format!("stride must be in 1..={ctx_window}, got {stride}"),
        });
    }
    if tokens.len() < 2 {
        return Err(PplError::InvalidWindow {
            msg: format!("corpus must have >=2 tokens, got {}", tokens.len()),
        });
    }

    // Gemma4 is trained with a BOS token (id=2). When evaluating raw
    // text without a chat template the model has no start-of-document signal and
    // produces degenerate logit distributions at position 0 (and, to a lesser
    // degree, throughout the sequence). Standard LM perplexity evaluation for
    // BOS-trained models prepends the BOS token so the model knows it is at the
    // beginning of a document. We do that here, mirroring the convention used
    // by HF `evaluate.load("perplexity")` and lm-evaluation-harness for models
    // with a BOS token.
    //
    // Gemma4 config.json: "bos_token_id": 2 (field lives on the outer
    // GenerationConfig / text_config, not on Gemma4TextConfig as exposed by
    // rMLX; no struct field to read from so the literal is hard-coded here).
    // We only prepend if the caller didn't already supply BOS at position 0.
    const GEMMA4_BOS: u32 = 2;
    let bos_prepended: Vec<u32>;
    let tokens: &[u32] = if tokens.first().copied() == Some(GEMMA4_BOS) {
        tokens
    } else {
        bos_prepended = std::iter::once(GEMMA4_BOS)
            .chain(tokens.iter().copied())
            .collect();
        &bos_prepended
    };

    let vocab = model.cfg.vocab_size;
    let mut sum_nll: f64 = 0.0;
    let mut count: usize = 0;
    let mut windows: usize = 0;

    let mut start: usize = 0;
    let mut first = true;
    while start < tokens.len() {
        let end = (start + ctx_window).min(tokens.len());
        let window = &tokens[start..end];
        if window.len() < 2 {
            break;
        }

        // Forward this window -> [1, S, vocab] logits.
        let logits = model.forward_seq_logits_all(window, device)?;
        let host = logits_3d_to_host_f32(&logits, window.len(), vocab)?;

        // warmup = number of leading positions whose NLL was already counted
        // in the previous window. First window: 0. Subsequent windows: the
        // overlap = ctx_window - stride.
        let warmup = if first {
            0
        } else {
            ctx_window.saturating_sub(stride).min(window.len() - 1)
        };

        // Score positions [warmup .. window.len() - 1) -- predicting token at
        // window[t+1] from logits[t].
        for t in warmup..(window.len() - 1) {
            let next_id = window[t + 1] as usize;
            if next_id >= vocab {
                warn!(
                    token_id = next_id,
                    vocab, "ppl: token id out of vocab range -- skipping position"
                );
                continue;
            }
            let row = &host[t * vocab..(t + 1) * vocab];
            let nll = neg_log_softmax_at(row, next_id);
            if nll.is_finite() {
                sum_nll += f64::from(nll);
                count += 1;
            } else {
                warn!(
                    t,
                    token_id = next_id,
                    nll,
                    "ppl: non-finite NLL at position -- skipping"
                );
            }
        }

        windows += 1;
        debug!(
            window_idx = windows - 1,
            start,
            end,
            warmup,
            scored_so_far = count,
            mean_nll_running = if count > 0 {
                sum_nll / count as f64
            } else {
                0.0
            },
            "ppl: window scored"
        );

        // Advance. If the window already reached the corpus end, stop.
        if end == tokens.len() {
            break;
        }
        start += stride;
        first = false;
    }

    if count == 0 {
        return Err(PplError::InvalidWindow {
            msg: "no scored positions -- corpus shorter than ctx_window?".to_owned(),
        });
    }

    let mean_nll = sum_nll / count as f64;
    let ppl_value = mean_nll.exp();
    info!(
        ppl = ppl_value,
        mean_nll,
        scored_tokens = count,
        windows,
        "ppl: run complete"
    );
    Ok(PplReport {
        ppl: ppl_value,
        mean_nll,
        scored_tokens: count,
        windows,
    })
}

/// Numerically-stable `-log_softmax(row)[idx]` over a vocab row.
///
/// Pure host work -- vocab is at most ~150k for the architectures we serve,
/// so a single pass over the row + an `idx` lookup is faster than going back
/// to the GPU. Mirrors the math in `crate::sampler::compute_top_logprobs`.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
pub(crate) fn neg_log_softmax_at(row: &[f32], idx: usize) -> f32 {
    let mut max = f32::NEG_INFINITY;
    for &l in row {
        if l > max {
            max = l;
        }
    }
    let mut sum_exp = 0.0f64;
    for &l in row {
        sum_exp += f64::from(l - max).exp();
    }
    let lse = f64::from(max) + sum_exp.ln();
    let logprob = f64::from(row[idx]) - lse;
    -logprob as f32
}

/// Read a `[1, seq, vocab]` logits array into a host `Vec<f32>` of length
/// `seq * vocab`. Supports `F32` and `BF16` (the two dtypes the model graphs
/// produce).
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
fn logits_3d_to_host_f32(logits: &Array, seq: usize, vocab: usize) -> Result<Vec<f32>> {
    logits.eval()?;
    let bytes = logits.to_bytes()?;
    let total = seq * vocab;
    match logits.dtype() {
        Dtype::F32 => {
            if bytes.len() < total * 4 {
                return Err(Error::Other(format!(
                    "ppl: F32 logits buffer too small ({} < {})",
                    bytes.len(),
                    total * 4
                )));
            }
            let out: Vec<f32> = bytes[..total * 4]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            Ok(out)
        }
        Dtype::Bf16 => {
            if bytes.len() < total * 2 {
                return Err(Error::Other(format!(
                    "ppl: BF16 logits buffer too small ({} < {})",
                    bytes.len(),
                    total * 2
                )));
            }
            let mut out = Vec::with_capacity(total);
            for i in 0..total {
                let o = i * 2;
                let raw = u16::from_le_bytes(bytes[o..o + 2].try_into().unwrap());
                out.push(f32::from_bits(u32::from(raw) << 16));
            }
            Ok(out)
        }
        other => Err(Error::Other(format!(
            "ppl: unsupported logits dtype {other:?}; expected F32 or BF16"
        ))),
    }
}

#[cfg(test)]
#[path = "ppl_tests.rs"]
mod tests;
