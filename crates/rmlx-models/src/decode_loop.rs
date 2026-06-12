// unsafe_code: none here; the byte-reinterpret helpers used by the loop live in
// the per-arch modules. This file is pure orchestration over already-built Arrays.
//! Shared, model-agnostic pipelined decode loop.
//!
//! One copy of the autoregressive decode pipeline every pipelined arch (qwen3,
//! qwen3_5_moe, gemma4, gemma3, qwen2) previously hand-copied. The pipeline
//! ordering — `choose_token` → `async_eval(next_y)` → drain the *previous*
//! step's pending token via `to_bytes()` → `try_clone`/feed — overlaps host
//! sampling/penalty work with the in-flight GPU forward. Reordering (draining
//! before dispatching the next forward) serializes CPU/GPU and is a measurable
//! decode regression; do not.
//!
//! The only per-arch hole is `forward_step` (an `impl FnMut`, monomorphized —
//! never a `Box<dyn>` on the per-token hot path). Prompt-cache policy, cache
//! construction, prefill flush protocol, and `decode_profile` emission all stay
//! arch-side; the loop returns [`DecodeStats`] for the arch to emit.
//!
//! `resolve_pieces` parameterizes the one emission difference between the source
//! loops: per-step `piece` resolution via `tokenizer.id_to_token` plus a
//! per-step `debug!` (gemma4/qwen2/qwen3 = true) vs. an empty `Box<str>` piece
//! with no per-step debug (qwen3_5_moe = false). Both are preserved byte-for-byte.

use std::time::Instant;

use rmlx_mlx::{argmax, Array, Device, Dtype};
use tracing::info_span;

use rmlx_core::error::Result;

use crate::constraint::ConstraintEngine;
use crate::sampler::{
    apply_mask_argmax, compute_top_logprobs, Pcg32, PenaltyConfig, SamplerConfig, TokenLogprobs,
};
use rmlx_kv_quant::KvCache;

// ---------------------------------------------------------------------------
// Smoke probe types
// ---------------------------------------------------------------------------

/// One record per autoregressive step produced by `generate_greedy`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed per-step probe record — five fields are the complete decode-step observability contract; adding a field requires updating all generate_greedy call sites"
)]
#[derive(Debug, Clone)]
pub struct ProbeStep {
    /// Token id selected by greedy argmax.
    pub token_id: u32,
    /// Single-piece string from `tokenizer.id_to_token(token_id)` — used for
    /// display and the broken-punct-loop heuristic.
    ///
    /// `Box<str>` saves 8 B per step vs `String`: 2 words (ptr + len) vs 3 words
    /// (ptr + len + capacity). For 4096-token sequences: ~32 KB saved + 4096 fewer
    /// allocator capacity slots. Construction: `piece.into_boxed_str()`.
    pub piece: Box<str>,
    /// `max(|logits|)` at this step. Finite normally; NaN/Inf signals a hazard.
    pub max_abs_logit: f32,
    /// Number of NaN cells in the logit vector at this step.
    pub nan_count: usize,
    /// per-token top-k logprobs. `None` unless the request set
    /// `top_logprobs_k > 0` (the zero-overhead default leaves this `None` and
    /// never runs the log-softmax / top-k path).
    pub logprobs: Option<TokenLogprobs>,
}

/// Verdict returned by `classify_smoke`.
///
/// Heuristic: after the seeded 8-token probe, the snapshot is flagged broken on
/// a degenerate repeat — see `classify_smoke` for the exact (B5b-widened) rule.
/// Mirrors the Qwen3.6-35B-A3B-mxfp8 pattern in CLAUDE.md "mxfp8 broken-snapshot
/// hazard" and the gemma-4-26b-a4b single-CJK-token loop (B5b audit).
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — four smoke outcomes (Ok/BrokenPunctLoop/BrokenNan/Inconclusive); adding an outcome requires updating classify_smoke and all SmokeVerdict match arms in the serve path"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
/// Smoke-probe outcome for a Gemma4 forward pass.
pub enum SmokeVerdict {
    /// Generation succeeded and output looks coherent.
    Ok,
    /// A degenerate repeat loop: ≤ 2 distinct ids + single-char punct piece, OR
    /// ≥ `LOOP_K` consecutive identical ids, OR a single-char piece (any
    /// category) dominating ≥ `LOOP_K` of the window. Name kept (not just
    /// punct) to keep exit-code / HTTP-503 / test mapping stable.
    BrokenPunctLoop {
        /// The punctuation piece that dominated the output.
        dominant_piece: String,
        /// Number of distinct token ids in the sample window.
        distinct_ids: usize,
    },
    /// A NaN appeared in the logit vector.
    BrokenNan {
        /// Decode step at which the NaN was detected.
        at_step: usize,
    },
    /// Generation stopped early but neither hazard fired (e.g. EOS at step 1).
    Inconclusive {
        /// Human-readable explanation of why the verdict is inconclusive.
        reason: String,
    },
}

/// Per-request borrow bundle for the shared decode loop. Decode-only — prefill
/// has its own free function ([`chunked_prefill`]) and does NOT take this struct.
pub(crate) struct DecodeCtx<'a> {
    pub tokenizer: &'a tokenizers::Tokenizer,
    pub vocab: i32,
    pub n_tokens: usize,
    pub device: Device,
    pub eos_ids: &'a [u32],
    /// dyn today (`arch/mod.rs`) — unchanged. Returns `Some(id)` to force the
    /// next emitted token (thinking-budget injection), `None` otherwise.
    pub step_fn: &'a mut dyn FnMut(&ProbeStep) -> Option<u32>,
    /// dyn today — unchanged. Optional grammar/JSON constraint engine.
    pub constraint: Option<&'a mut dyn ConstraintEngine>,
    pub sampler_cfg: &'a SamplerConfig,
    pub rng: &'a mut Pcg32,
    pub penalty_cfg: &'a PenaltyConfig,
    pub token_history: &'a mut Vec<u32>,
    /// tracing field only — names the arch in the per-step `debug!`.
    pub arch: &'static str,
    /// gemma4/qwen2/qwen3: true (resolve `piece` per token via `id_to_token`,
    /// emit a per-step `debug!`); qwen3_5_moe: false (push an empty `Box<str>`,
    /// no per-step debug). Both byte-for-byte preserved.
    pub resolve_pieces: bool,
}

/// Decode-loop timing accumulators, returned for arch-side `decode_profile`
/// emission. Nanoseconds; the arch converts to the milliseconds field shapes
/// `scripts/aggregate_decode_profile.py` parses.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct DecodeStats {
    pub forward_total_ns: u128,
    pub eval_total_ns: u128,
    pub step_total_ns: u128,
    pub decode_steps: u32,
}

/// capture top-`k` logprobs for an already-chosen token. Called ONLY behind a
/// `k > 0` guard, so the default decode path (no logprobs requested) never
/// touches log-softmax / top-k. Errors downgrade to `None` — logprob capture
/// must never abort generation.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: chosen token Array is [1] I32, sliced to its first 4 bytes only after a length check"
)]
#[allow(
    clippy::unwrap_used,
    reason = "the 4-byte slice is length-checked immediately above the try_into"
)]
pub(crate) fn capture_logprobs(
    logits_flat: &Array,
    chosen: &Array,
    k: usize,
) -> Option<TokenLogprobs> {
    let top_bytes = match chosen.to_bytes() {
        Ok(b) if b.len() >= 4 => b,
        _ => return None,
    };
    let chosen_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
    compute_top_logprobs(logits_flat, chosen_id, k).ok()
}

/// Shared sampling fork: penalties → temperature/top-k/top-p OR argmax, with the
/// optional constraint mask. Transplant of the ~60-line if/else duplicated at
/// every prefill tail and inside every decode loop. Returns the chosen-token
/// Array (possibly still lazy).
///
/// `mask_active` is computed by the caller ONCE per iteration, BEFORE the
/// pre-drain `constraint.advance()`. It MUST NOT be recomputed here:
/// `ConstraintEngine::wants_mask` flips on warm-up engagement, so recomputing
/// after `advance()` would mask/unmask one token differently than the
/// pre-advance gate the `eval()` protective branch uses — a silent divergence
/// on constrained + temp>0 / penalty paths. Caller hoists the single value.
///
/// The trailing-20 token window for penalty context is read from
/// `ctx.token_history` internally (zero per-step allocation — the read borrow of
/// `ctx.token_history` is disjoint from the `&mut` borrows of `ctx.constraint` /
/// `ctx.rng`). This function does NOT push to `token_history` — the caller
/// materializes the id and pushes it (the loop's drain site, or the prefill
/// tail). Never push in two places.
#[allow(clippy::collapsible_else_if)]
#[allow(
    clippy::indexing_slicing,
    reason = "win_start = len.saturating_sub(20) is <= len by construction, so [win_start..] never panics"
)]
#[allow(
    clippy::unwrap_used,
    reason = "constraint.as_mut().unwrap() is guarded by mask_active = constraint.is_some_and(wants_mask), so Some is structural"
)]
pub(crate) fn choose_token(
    ctx: &mut DecodeCtx<'_>,
    logits_flat: &Array,
    mask_active: bool,
) -> Result<Array> {
    let sampling_active = ctx.sampler_cfg.sampling_active();
    let penalties_active = ctx.penalty_cfg.penalties_active();
    let vocab = ctx.vocab;
    let device = ctx.device;
    // trailing-20 window for penalty context. Borrowed read-only here; disjoint
    // from the `&mut ctx.constraint` / `&mut ctx.rng` field borrows below.
    let win_start = ctx.token_history.len().saturating_sub(20);
    let recent: &[u32] = &ctx.token_history[win_start..];
    if sampling_active {
        let mask_opt: Option<&[bool]> = if mask_active {
            Some(ctx.constraint.as_mut().unwrap().step_mask(vocab as usize))
        } else {
            None
        };
        crate::sampler::sample_token_array(
            logits_flat,
            ctx.sampler_cfg,
            mask_opt,
            ctx.penalty_cfg,
            recent,
            ctx.rng,
            device,
        )
    } else {
        if penalties_active {
            let mask_opt: Option<&[bool]> = if mask_active {
                Some(ctx.constraint.as_mut().unwrap().step_mask(vocab as usize))
            } else {
                None
            };
            crate::sampler::argmax_with_penalties(
                logits_flat,
                mask_opt,
                ctx.penalty_cfg,
                recent,
                device,
            )
        } else if mask_active {
            let c = ctx.constraint.as_mut().unwrap();
            let m = c.step_mask(vocab as usize);
            apply_mask_argmax(logits_flat, m, device)
        } else {
            argmax(logits_flat, -1, device)
        }
    }
}

/// THE pipelined decode loop. Wraps the whole loop in `info_span!("decode")`.
///
/// `first_id` is the prefill / exact-hit first token already pushed to `steps`
/// and `token_history` by the caller; this fn drives steps `1..n_tokens`.
/// `forward_step` is the ONLY per-arch hole — it is `impl FnMut` (monomorphized,
/// zero dispatch); never accept a `Box<dyn>` here.
#[allow(
    clippy::unwrap_used,
    reason = "the remaining unwraps are the 4-byte `try_into` on a [..4] slice of a [1] I32 Array — length is structural, not data-dependent"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "the 4-byte to_le_bytes drain slices [..4] of a [1] I32 Array — length structural"
)]
#[allow(clippy::cognitive_complexity)]
#[allow(
    clippy::too_many_lines,
    reason = "single cohesive pipelined loop body; splitting the pre-drain / sample / post-drain phases would break the borrow flow and the async/eval interleave that is the perf contract"
)]
pub(crate) fn pipelined_decode(
    ctx: &mut DecodeCtx<'_>,
    first_id: u32,
    steps: &mut Vec<ProbeStep>,
    mut forward_step: impl FnMut(&Array) -> Result<Array>,
) -> Result<DecodeStats> {
    let mut stats = DecodeStats::default();
    let device = ctx.device;
    let vocab = ctx.vocab;

    let _decode_span = info_span!("decode", n_tokens = ctx.n_tokens).entered();

    let mut y: Array = {
        let id_i32 = first_id as i32;
        let bytes = id_i32.to_le_bytes();
        Array::from_bytes(&bytes, &[1], Dtype::I32)?
    };
    y.eval()?;
    let mut pending: Option<Array> = None;
    // logprobs captured at sample time travel with `pending` (the token they
    // belong to is emitted one iteration later). `None` on the disabled path
    // (`lp_k == 0`) — never allocated, never read.
    let mut pending_logprobs: Option<TokenLogprobs> = None;
    let mut early_stop = false;
    // when `step_fn` returns `Some(id)` (thinking budget exceeded), the loop
    // discards the model's pipelined successor and feeds `id` (`</think>`) as the
    // next input, materializing it as the next emitted token. `None` (no budget)
    // leaves the pipeline byte-for-byte unchanged.
    let mut forced_next: Option<u32> = None;

    for step_idx in 1..ctx.n_tokens {
        let step_t0 = Instant::now();
        let fwd_t0 = Instant::now();
        let decode_logits = match forward_step(&y) {
            Ok(l) => l,
            Err(e) => {
                tracing::warn!(
                    arch = ctx.arch,
                    step = step_idx,
                    error = %e,
                    "decode step failed, stopping early"
                );
                break;
            }
        };

        let logits_flat = decode_logits.reshape(&[1, vocab], device)?;
        // only run the masked pre-drain pipeline when the engine is actively
        // masking (wants_mask=true). During warm-up the engine is inert; we fall
        // through to the unconstrained pipeline so argmax dtype + pipelining are
        // bit-identical. `advance()` is still called from the post-drain branch.
        let mask_active = ctx.constraint.as_ref().is_some_and(|c| c.wants_mask());
        // temp>0 reads logits to host each step (no async pipelining benefit), so
        // it shares the masked branch's pre-drain. temp<=0 keeps the exact
        // `mask_active`-gated pipelined path byte-for-byte.
        let sampling_active = ctx.sampler_cfg.sampling_active();
        // penalties also require logits on host → fold into drain_now.
        let penalties_active = ctx.penalty_cfg.penalties_active();
        // logprob capture needs host logits per step → also drains here.
        let lp_k = ctx.sampler_cfg.top_logprobs_k as usize;
        let drain_now = mask_active || sampling_active || penalties_active || lp_k > 0;
        // Force eager eval of logits on the masked path to prevent stale GPU data.
        if mask_active {
            logits_flat.eval()?;
        }
        let pre_drain_eos = if drain_now {
            if let Some(p) = pending.take() {
                let top_bytes = p.to_bytes()?;
                let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                if let Some(c) = ctx.constraint.as_mut() {
                    c.advance(next_id);
                }
                ctx.token_history.push(next_id);
                let piece = resolve_piece(ctx, next_id);
                let last = steps.push_mut(ProbeStep {
                    token_id: next_id,
                    piece,
                    max_abs_logit: 0.0,
                    nan_count: 0,
                    // logprobs computed when this token was sampled.
                    logprobs: pending_logprobs.take(),
                });
                forced_next = forced_next.or((ctx.step_fn)(&*last));
                ctx.eos_ids.contains(&next_id)
            } else {
                false
            }
        } else {
            false
        };
        if pre_drain_eos {
            early_stop = true;
            tracing::debug!(
                arch = ctx.arch,
                step = step_idx - 1,
                "EOS emitted (pre-drain, masked), stopping"
            );
            break;
        }
        // Reuse the SAME pre-advance `mask_active` the eval() gate and pre-drain
        // computed above; the pre-drain may have called `constraint.advance()`,
        // which can flip `wants_mask`, so recomputing here would diverge.
        let next_y = choose_token(ctx, &logits_flat, mask_active)?;
        let _ = next_y.async_eval();
        // capture this step's logprobs from `logits_flat` + the chosen token.
        // Stashed in `pending_logprobs`; emitted with the matching ProbeStep next
        // iteration (pipelined token identity). Gated on lp_k>0.
        if lp_k > 0 {
            pending_logprobs = capture_logprobs(&logits_flat, &next_y, lp_k);
        }
        let fwd_dt = fwd_t0.elapsed().as_nanos();

        // The actual GPU sync materializes here on the *previous* step's pending
        // (unconstrained / warm-up branch). The masked / sampling / penalty branch
        // already drained above.
        let eval_t0 = Instant::now();
        let mut emitted_eos = false;
        if !drain_now {
            if let Some(p) = pending.take() {
                // mlx_sync span — measures the GPU-sync cost of materializing the
                // previous step's pipelined argmax token.
                let top_bytes = {
                    let _sync = tracing::debug_span!(
                        "mlx_sync",
                        step = step_idx - 1,
                        reason = "pipelined_argmax_drain",
                    )
                    .entered();
                    p.to_bytes()?
                };
                let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
                if let Some(c) = ctx.constraint.as_mut() {
                    c.advance(next_id);
                }
                ctx.token_history.push(next_id);
                let piece = resolve_piece(ctx, next_id);
                if ctx.resolve_pieces {
                    tracing::debug!(
                        arch = ctx.arch,
                        step = step_idx - 1,
                        token_id = next_id,
                        piece = %piece,
                        "decode step (pipelined emit)"
                    );
                }
                let last = steps.push_mut(ProbeStep {
                    token_id: next_id,
                    piece,
                    max_abs_logit: 0.0,
                    nan_count: 0,
                    // this branch only runs when drain_now == false (lp_k == 0).
                    logprobs: None,
                });
                forced_next = forced_next.or((ctx.step_fn)(&*last));
                emitted_eos = ctx.eos_ids.contains(&next_id);
            }
        }
        let eval_dt = eval_t0.elapsed().as_nanos();

        stats.forward_total_ns += fwd_dt;
        stats.eval_total_ns += eval_dt;
        stats.step_total_ns += step_t0.elapsed().as_nanos();
        stats.decode_steps += 1;

        if emitted_eos {
            early_stop = true;
            tracing::debug!(
                arch = ctx.arch,
                step = step_idx - 1,
                "EOS emitted, stopping decode loop"
            );
            break;
        }

        // forced `</think>` injection. Discard the model's pipelined successor and
        // feed the forced id as BOTH the next input (`y`) and the next drained
        // token (`pending`), so the next iteration emits `</think>` and resumes
        // natural answer generation. No-op when `forced_next` is `None`.
        if let Some(forced_id) = forced_next.take() {
            let bytes = (forced_id as i32).to_le_bytes();
            let forced_arr = Array::from_bytes(&bytes, &[1], Dtype::I32)?;
            let _ = forced_arr.async_eval();
            y = forced_arr.try_clone()?;
            pending = Some(forced_arr);
            // Forced tokens are external overrides — no meaningful logprobs.
            pending_logprobs = None;
        } else {
            // Feed next step's forward with the (still pending) argmax Array.
            // try_clone shares the underlying data; pending keeps ownership of the
            // GPU buffer until the next iteration consumes it.
            y = next_y.try_clone()?;
            pending = Some(next_y);
        }
    }

    // Drain any final pending token that has not yet been emitted. Skip on
    // early_stop.
    if !early_stop {
        if let Some(p) = pending {
            let drain_t0 = Instant::now();
            p.eval()?;
            let top_bytes = p.to_bytes()?;
            stats.eval_total_ns += drain_t0.elapsed().as_nanos();
            let next_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;
            if let Some(c) = ctx.constraint.as_mut() {
                c.advance(next_id);
            }
            ctx.token_history.push(next_id);
            let piece = resolve_piece(ctx, next_id);
            (ctx.step_fn)(&*steps.push_mut(ProbeStep {
                token_id: next_id,
                piece,
                max_abs_logit: 0.0,
                nan_count: 0,
                // final pending token carries its sample-time logprobs.
                logprobs: pending_logprobs.take(),
            }));
        }
    }

    Ok(stats)
}

/// Resolve the per-token `piece`: `tokenizer.id_to_token` when `resolve_pieces`,
/// otherwise an empty `Box<str>` (qwen3_5_moe behavior — pieces are display-only
/// and recomputed downstream from the id stream there).
#[inline]
fn resolve_piece(ctx: &DecodeCtx<'_>, id: u32) -> Box<str> {
    if ctx.resolve_pieces {
        ctx.tokenizer
            .id_to_token(id)
            .unwrap_or_else(|| format!("<unk:{id}>"))
            .into_boxed_str()
    } else {
        String::new().into_boxed_str()
    }
}

/// Fresh chunked prefill: `enter_prefill` → per-chunk forward + `eval_prefill_state`
/// → `exit_prefill`. Returns the final-position logits, or `None` when prefill was
/// rejected (caller returns `Ok(steps)` as today, unchanged).
///
/// `caches` is borrowed `&mut` by this fn (it brackets enter/exit and evals
/// non-final chunk state) AND re-borrowed into `forward_chunk` per chunk — the
/// closure takes `&mut Vec<KvCache>` so the arch's forward writes into the same
/// caches this fn brackets.
///
/// Deliberately Fresh-only. gemma4's prefix-append flush (`eval_gpu_state`, no
/// enter/exit) and qwen3_5_moe's HydratedTail append (`eval_prefill_state`, no
/// enter/exit) are two different flush protocols and stay per-arch — there is no
/// `PrefillKind` enum.
#[allow(
    clippy::unnecessary_wraps,
    reason = "Result is the documented contract: callers `?`-chain it, and the per-chunk forward / exit_prefill failures it folds into Ok(None) are genuinely fallible — the wrapper keeps the rejection path uniform with the loop call site"
)]
pub(crate) fn chunked_prefill(
    caches: &mut Vec<KvCache>,
    ids: &[u32],
    chunk_size: usize,
    device: Device,
    arch: &'static str,
    mut forward_chunk: impl FnMut(&[u32], &mut Vec<KvCache>) -> Result<Array>,
) -> Result<Option<Array>> {
    for c in caches.iter_mut() {
        c.enter_prefill();
    }
    let mut last_logits: Option<Array> = None;
    let mut prefill_ok = true;
    let n_chunks = ids.len().div_ceil(chunk_size.max(1));
    'prefill: for (chunk_idx, chunk) in ids.chunks(chunk_size.max(1)).enumerate() {
        let is_last = chunk_idx + 1 == n_chunks;
        match forward_chunk(chunk, caches) {
            Ok(logits) => {
                if is_last {
                    last_logits = Some(logits);
                } else {
                    // Eval only the cache prefill state, skip the lm_head matmul
                    // for non-final chunks via lazy-graph pruning.
                    for c in caches.iter() {
                        if let Err(e) = c.eval_prefill_state() {
                            tracing::warn!(
                                arch,
                                error = %e,
                                chunk_len = chunk.len(),
                                "prefill chunk cache eval failed"
                            );
                            prefill_ok = false;
                            break 'prefill;
                        }
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    arch,
                    error = %e,
                    prompt_len = ids.len(),
                    "prefill chunk failed, returning empty"
                );
                prefill_ok = false;
                break;
            }
        }
    }
    for c in caches.iter_mut() {
        if let Err(e) = c.exit_prefill(device) {
            tracing::warn!(arch, error = %e, "prefill: exit_prefill quantization failed");
            prefill_ok = false;
            // No break: every cache entered prefill above and must run
            // exit_prefill, even after a failure — skipping it leaves the
            // remaining caches with un-finalized prefill state (no decode
            // seed / un-quantized storage) that corrupts any later reuse of
            // this Vec.
        }
    }
    if !prefill_ok || last_logits.is_none() {
        return Ok(None);
    }
    Ok(last_logits)
}

#[cfg(test)]
#[path = "decode_loop_tests.rs"]
mod decode_loop_tests;
