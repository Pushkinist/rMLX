//! Token-replay retry envelope for canceled / watchdog-killed streams.
//!
//! ## Design
//!
//! The public entry point is [`replay_stream`]. It wraps a generator call and
//! returns a transparent stream of [`GenerationToken`] items. Internally it
//! drives a Tokio task that:
//!
//! 1. Runs attempt 1 using the original [`GenerationRequest`] (preserving the
//!    GPU admission permit for its lifetime).
//! 2. Forwards each token through an MPSC channel to the caller.
//! 3. Records each delivered `token_id` in a `Vec<u32>` kept inside the task.
//! 4. On a [`RetryClass::Migratable`] error: rebuilds the request via
//!    [`RequestPlan::build_request`], increments the attempt counter, and
//!    restarts from step 1 (attempts 2..=N use the plan, not the original req).
//!    The new attempt re-runs from the *original* prompt at `temperature == 0`,
//!    so it deterministically re-emits the already-delivered tokens as its
//!    first outputs, then continues past the fault point. The task then *skips*
//!    the first `delivered.len()` tokens and forwards only the new
//!    continuation, asserting prefix identity.
//! 5. On [`RetryClass::Fatal`] or attempt exhaustion: sends the error through
//!    the channel and exits.
//! 6. When the channel send fails (client disconnected / HTTP drop): exits
//!    silently without retrying — that is intentional cancellation, not a
//!    transient fault.
//!
//! ## Client-cancel safety
//!
//! The returned [`ReplayStream`] owns a `JoinHandle` and calls
//! `handle.abort()` on drop. Dropping the stream (HTTP client cancel)
//! stops the spawned engine task at the next `tx.send().await` yield point
//! with no further engine work.
//!
//! ## RequestPlan
//!
//! [`RequestPlan`] is a cheap shadow of [`GenerationRequest`] that captures
//! only the fields that (a) are needed to reconstruct the request and (b) can
//! be cheaply cloned. The non-clonable fields (`constraint`, `gpu_admission`)
//! are excluded: `constraint` disqualifies the request from replay via
//! [`is_replayable`]; `gpu_admission` is held by the original request (attempt 1)
//! and released when that blocking task exits. Retry attempts run without it.
//!
//! ## Prompt-continuation strategy
//!
//! `build_request` re-issues the *original* prompt unchanged. At
//! `temperature == 0` the decode is deterministic, so the engine re-emits the
//! already-delivered tokens as its first `delivered.len()` outputs and then
//! continues past the fault point; the skip logic drops that reproduced prefix
//! so the client sees a seamless stream. The delivered tokens are **not** also
//! appended to the prompt: doing so would double-count them (consumed as
//! prompt *and* skipped on output), so the engine's continuation would no
//! longer line up with the delivered prefix and every legitimate
//! partial-delivery replay would spuriously report a prefix divergence.
//!
//! ## single-emit invariant
//!
//! The engine gates `kv_cache_bytes` and `itl_p*_ms` events-table writes on
//! `steps_result.is_ok()`. On a Migratable error the engine sends `Err` and
//! returns; the emit sites are not reached for failed attempts. The one
//! successful attempt is therefore the unique emitter — no flag needed.
//! TTFT is emitted by the HTTP handler layer on the first token it ever sees,
//! so it fires exactly once regardless of retry count.
//!
//! ## Skip condition
//!
//! Callers must check [`is_replayable`] before calling [`replay_stream`].
//! When the skip conditions hold (temp > 0, n > 1, guided decoding), callers
//! should call `generator.generate(req)` directly.

#![allow(
    clippy::items_after_statements,
    clippy::match_same_arms,
    clippy::semicolon_if_nothing_returned
)]
use std::pin::Pin;
use std::sync::Arc;

use futures::StreamExt;
use rmlx_core::Error as RmlxError;

use crate::engine::{
    GenerationRequest, GenerationToken, Generator, NormalizedResponseFormat, NormalizedTool,
    NormalizedToolChoice, SamplingParams,
};
use crate::metrics_drainer::DrainerHandle;
use crate::openai::ItlStore;

// ── Retry classification ─────────────────────────────────────────────────────

/// Whether an [`RmlxError`] permits a transparent token-replay retry.
///
/// `Migratable` — the error is transient and the stream can be reconstructed
/// by re-issuing the request with already-delivered tokens appended to the
/// prompt.
///
/// `Fatal` — the error is permanent or intentional (e.g. client cancelled,
/// logic error). No retry should be attempted.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two retry classification outcomes (Migratable/Fatal); adding a class requires updating classify() and all RetryClass match arms in the serve path"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetryClass {
    /// Transient failure; replay is permitted.
    Migratable,
    /// Permanent or intentional failure; do not retry.
    Fatal,
}

/// Classify an [`RmlxError`] as [`RetryClass::Migratable`] or
/// [`RetryClass::Fatal`].
///
/// Delegates to [`RmlxError::is_migratable`]. The transient/permanent decision
/// lives on the error type itself, where the match over the variants is
/// exhaustive (no wildcard) in the crate that defines them — so a newly added
/// error variant fails the build until it is explicitly classified, rather than
/// a catch-all here silently defaulting an unclassified error to `Fatal`.
///
/// - Migratable: [`RmlxError::Mlx`] (any Metal-level fault) and
///   [`RmlxError::Other`] (a recovered engine panic). Both may succeed on a
///   fresh attempt.
/// - Fatal: every other variant — structural, configuration, OOM, smoke-probe,
///   and the KV ceiling / hard-cap rejections (the bound is the same on every
///   attempt, whichever phase crossed it).
pub fn classify(err: &RmlxError) -> RetryClass {
    if err.is_migratable() {
        RetryClass::Migratable
    } else {
        RetryClass::Fatal
    }
}

// ── Skip-condition check ─────────────────────────────────────────────────────

/// Return `true` iff token-replay retry is legal for this request.
///
/// Replay is disabled when **any** skip condition holds:
/// - `temperature > 0` — the continuation is non-deterministic.
/// - `has_guided_decoding` — the FSM resets on every request; replaying would
///   restart the grammar and produce malformed or duplicated JSON.
///
/// The `n > 1` check (OpenAI `n` field) is evaluated at the call site where
/// the raw request value is still in scope; pass `false` for `n_choices_ok`
/// when the request set `n > 1`.
pub fn is_replayable(req: &GenerationRequest, n_choices_ok: bool) -> bool {
    if req.sampling.temperature > 0.0 {
        return false;
    }
    if !n_choices_ok {
        return false;
    }
    if req.constraint.is_some() {
        return false;
    }
    true
}

// ── Default retry limit ───────────────────────────────────────────────────────

/// Default maximum number of replay attempts per logical request.
///
/// Matches Dynamo's production default (2 retries = 3 total attempts).
pub const DEFAULT_MAX_RETRIES: u32 = 2;

// ── RequestPlan ───────────────────────────────────────────────────────────────

/// Lightweight shadow of [`GenerationRequest`] that holds only the fields
/// needed to rebuild a request for a retry attempt.
///
/// Non-clonable fields are excluded:
/// - `constraint` / `gpu_admission` — `constraint` disqualifies via
///   [`is_replayable`]; `gpu_admission` is released when the previous
///   attempt's blocking task exits, and retry attempts run without it.
///
/// See the module doc for the full design rationale.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed request-plan struct — all fields are the complete per-request replay contract; adding a field requires updating all RequestPlan construction sites in the OpenAI handler"
)]
#[derive(Debug, Clone)]
pub struct RequestPlan {
    /// Registry model id identifying the generator for retry attempts.
    pub model_id: String,
    /// Token ids of the original (pre-retry) prompt, without any delivered tokens.
    pub original_prompt_tokens: Vec<u32>,
    /// Maximum new tokens from the original request, before subtracting delivered tokens.
    pub original_max_tokens: u32,
    /// Sampling parameters (temperature, top-p, etc.) frozen at plan creation.
    pub sampling: SamplingParams,
    /// Stop strings; generation halts on the first match.
    pub stop: Vec<String>,
    /// Whether the response is streamed via SSE.
    pub stream: bool,
    /// Optional system prompt injected before user turns.
    pub system: Option<String>,
    /// Client-supplied session id from the `X-Session-Id` header.
    pub session_id: Option<String>,
    /// Number of prompt-cache slots that were effective on the first attempt.
    pub effective_prompt_cache_slots: Option<usize>,
    /// Handle to the async metrics drainer for this request.
    pub metrics_drainer: Option<DrainerHandle>,
    /// Per-request inter-token latency accumulator.
    pub itl_store: Option<ItlStore>,
    /// Shared event recorder for structured runtime events.
    pub event_recorder: Option<Arc<rmlx_metrics::events::EventRecorder>>,
    /// Normalised tool definitions available to the model for this request.
    pub tools: Option<Vec<NormalizedTool>>,
    /// Normalised tool-choice directive (`auto`, `required`, or a named function).
    pub tool_choice: Option<NormalizedToolChoice>,
    /// Structured output / JSON-schema constraint for this request.
    pub response_format: Option<NormalizedResponseFormat>,
    /// Shared flag indicating whether the model is currently inside a `<think>` block.
    pub is_thinking_handle: Option<Arc<std::sync::atomic::AtomicBool>>,
    /// Cap on reasoning-channel tokens; `None` = uncapped.
    pub thinking_budget: Option<u32>,
    /// Token id of the thinking-block close delimiter for budget enforcement.
    pub thinking_end_token_id: Option<u32>,
    /// Per-request thinking mode; `Some(false)` disables `<think>` for Qwen3-family.
    pub enable_thinking: Option<bool>,
    /// Whether to emit raw `<tool_call>` / `</tool_call>` markers in the stream.
    pub emit_tool_markers: bool,
    /// Override for the thinking-block open delimiter; `None` uses `"<think>"`.
    pub thinking_start_token: Option<String>,
    /// Override for the thinking-block close delimiter; `None` uses `"</think>"`.
    pub thinking_end_token: Option<String>,
    /// Base64-encoded images attached to this request (vision input).
    pub images: Vec<String>,
    /// Base64-encoded audio payloads attached to this request.
    pub audio_b64: Vec<String>,
    /// Issue #26: per-request KV-quant codec override (`None` = launch default).
    pub kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
    /// Issue #26: per-request max-ctx ceiling override (`None` = launch default).
    pub max_ctx_override: Option<i32>,
    /// Per-request image-token budget override (`None` = launch default).
    pub image_max_tokens: Option<usize>,
}

impl RequestPlan {
    /// Snapshot the clonable fields of a [`GenerationRequest`] into a plan.
    ///
    /// Non-clonable fields (`constraint`, `gpu_admission`) are excluded:
    /// `constraint` disqualifies via [`is_replayable`]; `gpu_admission` is held
    /// by the original request (attempt 1) and must not be cloned.
    pub fn from_gen_req(req: &GenerationRequest) -> Self {
        Self {
            model_id: req.model_id.clone(),
            original_prompt_tokens: req.prompt_tokens.clone(),
            original_max_tokens: req.max_tokens,
            sampling: req.sampling.clone(),
            stop: req.stop.clone(),
            stream: req.stream,
            system: req.system.clone(),
            session_id: req.session_id.clone(),
            effective_prompt_cache_slots: req.effective_prompt_cache_slots,
            metrics_drainer: req.metrics_drainer.clone(),
            itl_store: req.itl_store.clone(),
            event_recorder: req.event_recorder.clone(),
            tools: req.tools.clone(),
            tool_choice: req.tool_choice.clone(),
            response_format: req.response_format.clone(),
            is_thinking_handle: req.is_thinking_handle.clone(),
            thinking_budget: req.thinking_budget,
            thinking_end_token_id: req.thinking_end_token_id,
            enable_thinking: req.enable_thinking,
            emit_tool_markers: req.emit_tool_markers,
            thinking_start_token: req.thinking_start_token.clone(),
            thinking_end_token: req.thinking_end_token.clone(),
            images: req.images.clone(),
            audio_b64: req.audio_b64.clone(),
            kv_quant_override: req.kv_quant_override,
            max_ctx_override: req.max_ctx_override,
            image_max_tokens: req.image_max_tokens,
        }
    }

    /// Build a [`GenerationRequest`] for a retry attempt (attempt 2 onward).
    ///
    /// The replay re-issues the **original** prompt unchanged. At
    /// `temperature == 0` the engine deterministically re-emits the
    /// already-delivered tokens as its first outputs; the replay loop skips
    /// exactly `delivered.len()` of them (its `skip_count`) while asserting
    /// prefix identity, then forwards the continuation. The delivered tokens
    /// are therefore **not** appended to the prompt, and the token budget stays
    /// at the original value — the engine re-generates the delivered prefix and
    /// then the remaining continuation, so the total new-token count still
    /// equals `original_max_tokens`. Appending the delivered tokens to the
    /// prompt (or shrinking the budget by their count) would double-count them
    /// and make every legitimate partial-delivery replay spuriously diverge.
    pub fn build_request(&self) -> GenerationRequest {
        GenerationRequest {
            model_id: self.model_id.clone(),
            prompt_tokens: self.original_prompt_tokens.clone(),
            max_tokens: self.original_max_tokens.max(1),
            sampling: self.sampling.clone(),
            stop: self.stop.clone(),
            stream: self.stream,
            system: self.system.clone(),
            session_id: self.session_id.clone(),
            effective_prompt_cache_slots: self.effective_prompt_cache_slots,
            metrics_drainer: self.metrics_drainer.clone(),
            itl_store: self.itl_store.clone(),
            event_recorder: self.event_recorder.clone(),
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            response_format: self.response_format.clone(),
            // constraint is always None for replay-eligible requests
            constraint: None,
            is_thinking_handle: self.is_thinking_handle.clone(),
            thinking_budget: self.thinking_budget,
            thinking_end_token_id: self.thinking_end_token_id,
            enable_thinking: self.enable_thinking,
            emit_tool_markers: self.emit_tool_markers,
            thinking_start_token: self.thinking_start_token.clone(),
            thinking_end_token: self.thinking_end_token.clone(),
            // gpu_admission is not held across retries; the original req's
            // blocking task released the permit on exit.
            gpu_admission: None,
            images: self.images.clone(),
            audio_b64: self.audio_b64.clone(),
            kv_quant_override: self.kv_quant_override,
            max_ctx_override: self.max_ctx_override,
            image_max_tokens: self.image_max_tokens,
        }
    }
}

// ── replay_stream ─────────────────────────────────────────────────────────────

/// RAII wrapper that aborts the inner task when dropped.
///
/// Tokio's `JoinHandle` detaches the task on drop. Wrapping it here ensures
/// that dropping the returned [`ReplayStream`] (e.g. HTTP client cancel)
/// stops the spawned engine task at the next `tx.send().await` yield point,
/// releasing GPU resources promptly.
struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Stream returned by [`replay_stream`].
///
/// Owns the abort handle so that dropping the stream cancels the background
/// engine task.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed stream wrapper — private receiver + RAII handle fields; public API is the futures::Stream impl; adding a field requires updating the spawn_replay_stream constructor"
)]
#[allow(missing_debug_implementations)]
pub struct ReplayStream {
    rx: tokio::sync::mpsc::Receiver<rmlx_core::Result<GenerationToken>>,
    _handle: AbortOnDrop,
}

impl futures::stream::Stream for ReplayStream {
    type Item = rmlx_core::Result<GenerationToken>;

    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        self.rx.poll_recv(cx)
    }
}

/// Wrap a generator call in a transparent token-replay envelope.
///
/// Returns a [`ReplayStream`] of [`GenerationToken`] items identical to what
/// `generator.generate(req)` would return — but with automatic retry on
/// [`RetryClass::Migratable`] errors, up to `max_retries` additional attempts
/// (total `max_retries + 1` tries).
///
/// ### Caller contract
///
/// - Call [`is_replayable`] before this function. If the skip conditions hold,
///   call `generator.generate(req)` directly — do not use this envelope.
/// - Pass the original [`GenerationRequest`] as `initial_req` (preserves the
///   GPU admission permit for attempt 1). Supply a [`RequestPlan`] built from
///   the same request for attempts 2 onward.
///
/// ### Client-cancel behaviour
///
/// Dropping the [`ReplayStream`] aborts the background task — **no retry** is
/// attempted on a client cancel.
///
/// ### invariant
///
/// `kv_cache_bytes` and `itl_p*_ms` events-table writes are gated on
/// `steps_result.is_ok()` inside the engine. On error the engine returns
/// before reaching the emit sites; only the one successful attempt emits.
/// TTFT is emitted by the HTTP handler layer and is not affected here.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
pub fn replay_stream(
    generator: Arc<dyn Generator>,
    initial_req: GenerationRequest,
    plan: RequestPlan,
    max_retries: u32,
) -> ReplayStream {
    // Channel cap=4 mirrors the engine's own channel.
    let (tx, rx) = tokio::sync::mpsc::channel::<rmlx_core::Result<GenerationToken>>(4);

    let handle = tokio::spawn(async move {
        let mut delivered: Vec<u32> = Vec::new();
        let mut attempts_remaining = max_retries + 1; // total attempts including the first
        let mut next_req: Option<GenerationRequest> = Some(initial_req);
        // The real engine error that triggered the current replay attempt. A
        // replay that then diverges from — or underruns — the delivered prefix
        // must surface THIS cause, not a synthetic "prefix divergence" /
        // "underrun" message that launders the true fault (e.g. a decode-step
        // crash) into an unrelated error. A finish_reason-keying client cannot
        // tell a laundered error from a clean short stop, so the real cause has
        // to travel with the failure. `None` on attempt 1 (no prior error).
        let mut root_error: Option<RmlxError> = None;

        loop {
            // Attempt 1 uses initial_req (holds the GPU admission permit).
            // Subsequent attempts build a fresh request from the plan.
            let req = match next_req.take() {
                Some(r) => r,
                None => plan.build_request(),
            };
            let skip_count = delivered.len();
            let model_id = &plan.model_id;

            tracing::debug!(
                model_id = %model_id,
                attempt = (max_retries + 2).saturating_sub(attempts_remaining),
                max_attempts = max_retries + 1,
                skip_prefix = skip_count,
                remaining_budget = req.max_tokens,
                "replay_stream starting attempt"
            );

            let mut token_stream = generator.generate(req);
            let mut skipped = 0usize;
            let mut attempt_error: Option<RmlxError> = None;

            while let Some(item) = token_stream.next().await {
                match item {
                    Err(e) => {
                        attempt_error = Some(e);
                        break;
                    }
                    Ok(tok) => {
                        // Skip already-delivered prefix tokens on retry attempts.
                        if skipped < skip_count {
                            // Assert prefix identity: at temp=0 the model must
                            // reproduce the same token at each position.
                            let expected = delivered[skipped];
                            if tok.token_id != expected {
                                tracing::error!(
                                    expected,
                                    actual = tok.token_id,
                                    position = skipped,
                                    "replay prefix divergence; aborting retry"
                                );
                                // Surface the real fault that triggered this
                                // replay (a decode-step crash reproduces at the
                                // same boundary on retry). Sending only the
                                // divergence message here would launder that
                                // crash into an unrelated error — the exact
                                // masked-failure shape the loud-error contract
                                // exists to prevent. Fall back to the divergence
                                // message only when there is no prior cause.
                                let err = root_error.take().unwrap_or_else(|| {
                                    RmlxError::Other(format!(
                                        "replay prefix divergence at {skipped}: \
                                         expected {expected} got {}",
                                        tok.token_id
                                    ))
                                });
                                let _ = tx.send(Err(err)).await;
                                return;
                            }
                            skipped += 1;
                            continue;
                        }
                        // Record delivered token_id for future retry attempts.
                        delivered.push(tok.token_id);
                        // Forward to the caller; exit on channel closed (client drop).
                        if tx.send(Ok(tok)).await.is_err() {
                            tracing::debug!(
                                model_id = %model_id,
                                "replay_stream receiver dropped (client cancel), stopping"
                            );
                            return;
                        }
                    }
                }
            }

            // Underrun: retry terminated before reproducing the delivered prefix.
            if attempt_error.is_none() && skipped < skip_count {
                tracing::error!(
                    skipped,
                    skip_count,
                    "retry terminated before reproducing delivered prefix"
                );
                // Same laundering guard as the divergence path: surface the real
                // cause that triggered this replay rather than a bare underrun
                // message that hides it.
                let err = root_error.take().unwrap_or_else(|| {
                    RmlxError::Other(
                        "replay underrun: retry EOS before prefix reproduced".to_owned(),
                    )
                });
                let _ = tx.send(Err(err)).await;
                return;
            }

            // Stream completed successfully — we're done.
            if attempt_error.is_none() {
                return;
            }

            let Some(err) = attempt_error else { return };
            attempts_remaining = attempts_remaining.saturating_sub(1);

            match classify(&err) {
                RetryClass::Fatal => {
                    tracing::debug!(
                        model_id = %model_id,
                        error = %err,
                        "replay_stream fatal error, not retrying"
                    );
                    let _ = tx.send(Err(err)).await;
                    return;
                }
                RetryClass::Migratable if attempts_remaining == 0 => {
                    tracing::warn!(
                        model_id = %model_id,
                        error = %err,
                        delivered_tokens = delivered.len(),
                        "replay_stream migratable error, retries exhausted"
                    );
                    let _ = tx.send(Err(err)).await;
                    return;
                }
                RetryClass::Migratable => {
                    tracing::info!(
                        model_id = %model_id,
                        error = %err,
                        delivered_tokens = delivered.len(),
                        attempts_remaining,
                        "replay_stream migratable error, retrying"
                    );
                    // Preserve the real cause so a divergence / underrun on the
                    // coming attempt surfaces it rather than a synthetic message.
                    root_error = Some(err);
                    // next_req is None — loop will call plan.build_request.
                }
            }
        }
    });

    ReplayStream {
        rx,
        _handle: AbortOnDrop(handle),
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "retry_tests.rs"]
mod retry_tests;
