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

use rmlx_kv_quant::{KvCache, KvQuant};

use crate::arch::Architecture;
use crate::decode_loop::chunked_prefill;
use crate::gemma4::{Gemma4Text, LayerType};
use crate::kv_cache::kv_layer_quants;
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
///
/// `kv_quant` selects between the two scorers:
///
/// * `None` — the cacheless full-window forward described above. Every KV codec
///   is out of the picture: nothing is stored, so nothing is quantized.
/// * `Some(codec)` — the window is teacher-forced through a real per-layer KV
///   cache built at `codec` by [`kv_layer_quants`], the same vector the decode
///   loop builds. The prompt prefix is prefilled and every scored position is a
///   single-token step, so each NLL is read off the decode path the served
///   model runs. This is the only shape in which a KV codec can affect a
///   perplexity number at all — a scorer that keeps no cache cannot measure one.
///
/// **The two are not interchangeable on Gemma4.** At a bf16 cache the modes
/// should differ only by floating-point noise, and they agree bit-for-bit at
/// `ctx_window = 8` on every architecture. On Qwen3 they stay within ±0.003 of
/// `mean_nll` at every window from 32 up. On Gemma4 they part by a margin that
/// grows with the attended context, reaching −0.123 at `ctx_window = 512` on
/// `gemma-4-12B-it-mxfp8` — a disagreement between
/// [`Gemma4Text::forward_seq_logits_all`] and
/// `Gemma4Text::forward_seq_with_cache`, both of which predate this parameter
/// and neither of which this parameter changes. Compare a cached number only
/// against another cached number of the same architecture until that is
/// resolved.
#[instrument(skip(arch, tokens), fields(arch_class = arch.arch_class(), n_tokens = tokens.len(), ctx_window, stride, ?kv_quant))]
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
    kv_quant: Option<KvQuant>,
) -> std::result::Result<PplReport, PplError> {
    match arch {
        Architecture::Qwen3(m) => {
            compute_ppl_qwen3(m, tokens, ctx_window, stride, device, kv_quant)
        }
        Architecture::Gemma4(m) => {
            compute_ppl_gemma4(m, tokens, ctx_window, stride, device, kv_quant)
        }
        other => Err(PplError::ArchUnsupported {
            arch: other.arch_class().to_string(),
        }),
    }
}

/// Accumulate the NLL of `win[t + 1]` under the logit row at window position
/// `t`, skipping an out-of-vocab target or a non-finite result.
///
/// One implementation for four loops (two architectures × cached / cacheless):
/// the skip rules are what decides the denominator of `mean_nll`, so two copies
/// of them are two perplexities.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn accumulate_position_nll(
    row: &[f32],
    next_id: usize,
    vocab: usize,
    t: usize,
    sum_nll: &mut f64,
    count: &mut usize,
) {
    if next_id >= vocab {
        warn!(
            token_id = next_id,
            vocab, "ppl: token id out of vocab range -- skipping position"
        );
        return;
    }
    let nll = neg_log_softmax_at(row, next_id);
    if nll.is_finite() {
        *sum_nll += f64::from(nll);
        *count += 1;
    } else {
        warn!(
            t,
            token_id = next_id,
            nll,
            "ppl: non-finite NLL at position -- skipping"
        );
    }
}

/// Score one window by teacher-forcing it through `caches`.
///
/// `win[0..=warmup]` is prefilled through the arch's own chunked-prefill
/// protocol; its final-position logits score position `warmup`. Every later
/// scored position is a single-token forward — the decode step, with the
/// cache's codec on the read path.
///
/// `forward` is the arch's cache-taking forward returning last-position logits.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::indexing_slicing,
    reason = "every index below is behind the `warmup + 1 < win.len()` guard at the top or the loop's own `t < last_scored` bound"
)]
fn score_window_through_cache(
    win: &[u32],
    warmup: usize,
    vocab: usize,
    caches: &mut Vec<KvCache>,
    arch: &'static str,
    device: Device,
    sum_nll: &mut f64,
    count: &mut usize,
    mut forward: impl FnMut(&[u32], &mut Vec<KvCache>) -> Result<Array>,
) -> Result<()> {
    // A window whose warm-up prefix reaches its last position has no scored
    // position at all: `t` predicts `win[t + 1]`, and there is no `t + 1`. The
    // callers clamp `warmup` to `win.len() - 1`, so the final window of a
    // corpus hits that equality whenever it is one token longer than the
    // overlap. The cacheless path's `for t in warmup..(win.len() - 1)` is
    // already empty there; this is the same statement, made before any GPU
    // work rather than by indexing past the end.
    if win.get(warmup + 1).is_none() {
        return Ok(());
    }
    let last_scored = win.len() - 1; // exclusive: position t predicts win[t+1]
    let prefill_logits = chunked_prefill(
        caches,
        &win[..=warmup],
        crate::prefill_chunk::resolve(arch),
        device,
        arch,
        &mut forward,
    )?;
    let host = logits_3d_to_host_f32(&prefill_logits, 1, vocab)?;
    accumulate_position_nll(
        &host,
        win[warmup + 1] as usize,
        vocab,
        warmup,
        sum_nll,
        count,
    );

    for t in (warmup + 1)..last_scored {
        let logits = forward(&win[t..=t], caches)?;
        let host = logits_3d_to_host_f32(&logits, 1, vocab)?;
        accumulate_position_nll(&host, win[t + 1] as usize, vocab, t, sum_nll, count);
    }
    Ok(())
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
    kv_quant: Option<KvQuant>,
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
        match kv_quant {
            None => {
                let logits = model.forward_seq_logits_all(window, device)?;
                let host = logits_3d_to_host_f32(&logits, window.len(), vocab)?;
                for t in warmup..(window.len() - 1) {
                    let row = &host[t * vocab..(t + 1) * vocab];
                    accumulate_position_nll(
                        row,
                        window[t + 1] as usize,
                        vocab,
                        t,
                        &mut sum_nll,
                        &mut count,
                    );
                }
            }
            Some(q) => {
                let mut caches = qwen3_ppl_caches(model, q, window.len());
                score_window_through_cache(
                    window,
                    warmup,
                    vocab,
                    &mut caches,
                    "qwen3",
                    device,
                    &mut sum_nll,
                    &mut count,
                    |ids, cs| model.forward_seq_with_cache(ids, Some(cs.as_mut_slice()), device),
                )?;
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

/// Sliding-window PPL scorer for the Gemma4 family.
///
/// Gemma4 is trained with a BOS token (id=2) as a start-of-document signal.
/// Without BOS at the start of a forward pass, the model produces degenerate
/// logit distributions — high NLL that worsens progressively across windows as
/// no window after the first would have a BOS start-signal.
///
/// **BOS is prepended to every sliding window**, not just the first. For each
/// window at corpus position `start`, the forward input is
/// `[BOS, tokens[start], ..., tokens[start + ctx_window - 2]]` (length
/// `ctx_window`). This ensures the model always starts from a known
/// start-of-document state. The warmup positions (first `ctx_window - stride`
/// positions in each non-first window) are skipped in scoring as usual — only
/// the second half of each window is scored — so BOS context is always present
/// for scored positions.
///
/// The scored positions in window i (BOS-prefixed view, 0-indexed):
/// - window 0: positions `[0..ctx_window-2]` (predicts `tokens[0..ctx_window-1]`)
/// - window i>0: positions `[warmup..ctx_window-2]` where `warmup = ctx_window - stride`
///
/// Each scored position `t` predicts `bos_window[t+1] = tokens[start + t]` —
/// the corpus token immediately following position `t`.
///
/// Uses `Gemma4Text::forward_seq_logits_all` to produce the full
/// `[1, seq, vocab]` logit tensor in one Metal dispatch — no cache state,
/// fresh window each call. Final-logit softcapping is applied inside
/// `forward_seq_logits_all` via `apply_softcap`.
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
    kv_quant: Option<KvQuant>,
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

    // Gemma4 config.json: "bos_token_id": 2.  The field lives on the outer
    // GenerationConfig / text_config; no struct field in Gemma4TextConfig,
    // so the literal is hard-coded here.
    const GEMMA4_BOS: u32 = 2;

    // Reusable BOS-prefixed window buffer.  Allocated once and reused across
    // windows to avoid per-window heap allocation for a ctx_window-sized Vec.
    let mut bos_window: Vec<u32> = Vec::with_capacity(ctx_window);

    let vocab = model.cfg.vocab_size;
    let mut sum_nll: f64 = 0.0;
    let mut count: usize = 0;
    let mut windows: usize = 0;

    let mut start: usize = 0;
    let mut first = true;
    while start < tokens.len() {
        // Build a BOS-prefixed window of length min(ctx_window, available+1).
        // Layout: [BOS, tokens[start], tokens[start+1], ..., tokens[start+ctx_window-2]]
        // The window always starts with BOS so the model has a start-of-document
        // signal regardless of where in the corpus this window begins.
        let available = tokens.len() - start;
        let content_len = available.min(ctx_window - 1); // corpus tokens after BOS
        if content_len == 0 {
            break; // no corpus tokens left to score
        }
        bos_window.clear();
        bos_window.push(GEMMA4_BOS);
        bos_window.extend_from_slice(&tokens[start..start + content_len]);
        let win_len = bos_window.len(); // 1 + content_len

        // BOS guarantees win_len = 1 + content_len >= 2 (content_len >= 1).

        // warmup = leading positions to skip (already scored in the previous window).
        // First window: 0. Subsequent windows: the overlap region is
        // ctx_window - stride positions, but the prepended BOS shifts every
        // corpus target one slot earlier in the window, so the overlap to skip
        // is one smaller: (ctx_window - stride - 1).
        let warmup = if first {
            0
        } else {
            ctx_window
                .saturating_sub(stride)
                .saturating_sub(1)
                .min(win_len - 1)
        };

        // Score positions [warmup .. win_len-1) predicting bos_window[t+1].
        // bos_window[t+1] = tokens[start + t]  (for t >= 0, since bos_window[1..] = tokens[start..]).
        match kv_quant {
            None => {
                let logits = model.forward_seq_logits_all(&bos_window, device)?;
                let host = logits_3d_to_host_f32(&logits, win_len, vocab)?;
                for t in warmup..(win_len - 1) {
                    let row = &host[t * vocab..(t + 1) * vocab];
                    accumulate_position_nll(
                        row,
                        bos_window[t + 1] as usize,
                        vocab,
                        t,
                        &mut sum_nll,
                        &mut count,
                    );
                }
            }
            Some(q) => {
                let mut caches = gemma4_ppl_caches(model, q, win_len);
                score_window_through_cache(
                    &bos_window,
                    warmup,
                    vocab,
                    &mut caches,
                    "gemma4",
                    device,
                    &mut sum_nll,
                    &mut count,
                    |ids, cs| model.forward_seq_with_cache(ids, Some(cs.as_mut_slice()), device),
                )?;
            }
        }

        windows += 1;
        debug!(
            window_idx = windows - 1,
            start,
            end = start + content_len,
            warmup,
            win_len,
            scored_so_far = count,
            mean_nll_running = if count > 0 {
                sum_nll / count as f64
            } else {
                0.0
            },
            "ppl: window scored"
        );

        // Advance. If we consumed the last available corpus tokens, stop.
        if content_len < ctx_window - 1 {
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

/// The per-layer cache stack the Qwen3 scorer teacher-forces a window through.
///
/// Mirrors `qwen3::generate_greedy`'s fresh-cache construction: the same
/// producer, the same `shares_kv`, the same per-layer codec vector. `max_seq`
/// is the window length because a window is the whole sequence this stack ever
/// sees — the scorer builds a fresh stack per window rather than resetting one,
/// which is what "fresh window each call" means once there is a cache.
fn qwen3_ppl_caches(model: &Qwen3Text, kv_quant: KvQuant, win_len: usize) -> Vec<KvCache> {
    kv_layer_quants(
        model.cfg.num_hidden_layers,
        kv_quant,
        crate::qwen3::SHARES_KV_ACROSS_LAYERS,
    )
    .into_iter()
    .enumerate()
    .map(|(i, q)| KvCache::with_quant_max_seq(q, win_len as i32).with_layer_idx(i))
    .collect()
}

/// The per-layer cache stack the Gemma4 scorer teacher-forces a window through.
///
/// Mirrors `gemma4::generate_greedy`'s fresh-cache construction, windows
/// included: a sliding-attention layer gets its rotating ring here exactly as
/// it does when serving, so the scored NLL sees the same layer mix a request
/// would.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn gemma4_ppl_caches(model: &Gemma4Text, kv_quant: KvQuant, win_len: usize) -> Vec<KvCache> {
    let sliding_window_i32 = model.cfg.sliding_window as i32;
    kv_layer_quants(
        model.cfg.num_hidden_layers,
        kv_quant,
        crate::gemma4::SHARES_KV_ACROSS_LAYERS,
    )
    .into_iter()
    .enumerate()
    .map(|(i, q)| {
        let window = match model.cfg.layer_types[i] {
            LayerType::SlidingAttention => Some(sliding_window_i32),
            LayerType::FullAttention => None,
        };
        KvCache::with_quant_max_seq_window(q, win_len as i32, window)
            .with_layer_idx(i)
            .with_shares_kv(crate::gemma4::SHARES_KV_ACROSS_LAYERS)
    })
    .collect()
}

/// Returns the set of corpus-target indices scored by `compute_ppl_gemma4` for a
/// corpus of `n_tokens` tokens with the given window parameters.
///
/// Each entry is a corpus position `c` such that the scorer evaluated the NLL
/// of `tokens[c]` (predicted by logits at the previous BOS-window slot).
///
/// This is pure index arithmetic — no model, no GPU, no `Array`.  It mirrors
/// the `start` / `content_len` / `win_len` / `warmup` logic in
/// `compute_ppl_gemma4` exactly and serves as the single source of truth for
/// the windowing-coverage unit test.
#[cfg(test)]
pub(crate) fn gemma4_scored_indices(
    n_tokens: usize,
    ctx_window: usize,
    stride: usize,
) -> Vec<usize> {
    let mut out = Vec::new();
    let mut start: usize = 0;
    let mut first = true;
    while start < n_tokens {
        let available = n_tokens - start;
        let content_len = available.min(ctx_window - 1);
        if content_len == 0 {
            break;
        }
        let win_len = 1 + content_len; // BOS + content_len corpus tokens

        let warmup = if first {
            0
        } else {
            ctx_window
                .saturating_sub(stride)
                .saturating_sub(1)
                .min(win_len - 1)
        };

        // Scored positions t in [warmup .. win_len-1): corpus index = start + t.
        for t in warmup..(win_len - 1) {
            out.push(start + t);
        }

        if content_len < ctx_window - 1 {
            break; // last (partial) window
        }
        start += stride;
        first = false;
    }
    out
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
