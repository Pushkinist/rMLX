//! SSE streaming path for the Anthropic Messages API.
//!
//! - `generate_streaming` — streaming generation entry point
//! - `BlockKind` — kind of the currently-open Anthropic content block
//! - `enqueue_text_or_thinking_delta` — append a text/thinking delta to the event queue
//! - `enqueue_tool_use_block` — emit a complete tool_use block sequence
//! - `AnthropicState` — state machine threaded through the `unfold` stream
//! - `sse_event` — serialise a named SSE event

#![allow(clippy::too_many_lines)]

use std::sync::Arc;
use std::time::Instant;

use axum::http::HeaderValue;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use serde_json::{json, Value};

use crate::engine::{GenerationRequest, GenerationToken};
use crate::openai::errors::engine_error_category;
use crate::tool_parser::{ParsedToolCall, ToolCallFormat, ToolCallStreamParser};

use super::errors::engine_error_response;
use super::route::{map_stop_reason, select_anthropic_stop_reason};
use crate::openai::AppState;

// ── SSE helper ────────────────────────────────────────────────────────────────

pub(super) fn sse_event(event_name: &str, payload: &Value) -> Event {
    Event::default()
        .event(event_name)
        .data(serde_json::to_string(payload).unwrap_or_default())
}

// ── BlockKind ─────────────────────────────────────────────────────────────────

/// Kind of the currently-open Anthropic content block.
///
/// Note: tool_use blocks are opened/streamed/closed atomically by
/// `enqueue_tool_use_block` and never linger as `current_block`, so this
/// enum has only the two long-lived kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockKind {
    Text,
    Thinking,
}

impl BlockKind {
    fn delta_type(self) -> &'static str {
        // Anthropic public SSE spec: text content blocks use `text_delta`
        // (key: `text`); extended-thinking blocks use `thinking_delta`
        // (key: `thinking`).
        match self {
            BlockKind::Text => "text_delta",
            BlockKind::Thinking => "thinking_delta",
        }
    }
    fn delta_text_key(self) -> &'static str {
        match self {
            BlockKind::Text => "text",
            BlockKind::Thinking => "thinking",
        }
    }
    pub(crate) fn cb_start_event(self, index: u32) -> Event {
        let body = match self {
            BlockKind::Text => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "text", "text": ""}
            }),
            BlockKind::Thinking => json!({
                "type": "content_block_start",
                "index": index,
                "content_block": {"type": "thinking", "thinking": ""}
            }),
        };
        sse_event("content_block_start", &body)
    }
}

// ── Block event helpers ───────────────────────────────────────────────────────

/// Append a text-or-thinking content_block_delta to `queue`, opening a new
/// block (and closing the previous one) if `want != current_block`.
pub(crate) fn enqueue_text_or_thinking_delta(
    queue: &mut std::collections::VecDeque<Result<Event, std::convert::Infallible>>,
    current_block: &mut Option<BlockKind>,
    current_index: &mut u32,
    want: BlockKind,
    text: &str,
) {
    // ToolUse blocks should never call into this helper.
    debug_assert!(matches!(want, BlockKind::Text | BlockKind::Thinking));
    match *current_block {
        None => {
            queue.push_back(Ok(want.cb_start_event(*current_index)));
            *current_block = Some(want);
        }
        Some(cur) if cur != want => {
            let cb_stop = json!({"type": "content_block_stop", "index": *current_index});
            queue.push_back(Ok(sse_event("content_block_stop", &cb_stop)));
            *current_index += 1;
            queue.push_back(Ok(want.cb_start_event(*current_index)));
            *current_block = Some(want);
        }
        Some(_) => {}
    }
    let delta = json!({
        "type": "content_block_delta",
        "index": *current_index,
        "delta": {
            "type": want.delta_type(),
            want.delta_text_key(): text,
        }
    });
    queue.push_back(Ok(sse_event("content_block_delta", &delta)));
}

/// Append the full sequence for one tool_use content block:
/// content_block_stop (prior, if any) →
/// content_block_start (tool_use, with id+name; empty input) →
/// content_block_delta (input_json_delta, partial_json = full JSON) →
/// content_block_stop (tool_use)
///
/// v1 emits the input as a single `input_json_delta` chunk per call. Per
/// the Anthropic spec, clients accumulate `partial_json` across all
/// `input_json_delta` events in a block and JSON-parse the concatenation
/// at `content_block_stop` — one chunk is fully valid.
///
/// After this returns, `current_block = None` so the next non-thinking text
/// piece (if any) opens a fresh text block.
pub(crate) fn enqueue_tool_use_block(
    queue: &mut std::collections::VecDeque<Result<Event, std::convert::Infallible>>,
    current_block: &mut Option<BlockKind>,
    current_index: &mut u32,
    parsed: &ParsedToolCall,
) {
    // Close any currently-open block first.
    if current_block.is_some() {
        let cb_stop = json!({"type": "content_block_stop", "index": *current_index});
        queue.push_back(Ok(sse_event("content_block_stop", &cb_stop)));
        *current_index += 1;
    }

    let cb_start = json!({
        "type": "content_block_start",
        "index": *current_index,
        "content_block": {
            "type": "tool_use",
            "id": parsed.id,
            "name": parsed.name,
            "input": {},
        }
    });
    queue.push_back(Ok(sse_event("content_block_start", &cb_start)));

    // v1: single input_json_delta chunk containing the full JSON-stringified
    // input. Clients concatenate partial_json fragments → parse at stop.
    let input_obj = Value::Object(parsed.arguments.clone());
    let partial_json = serde_json::to_string(&input_obj).unwrap_or_else(|_| "{}".to_owned());
    let delta = json!({
        "type": "content_block_delta",
        "index": *current_index,
        "delta": {
            "type": "input_json_delta",
            "partial_json": partial_json,
        }
    });
    queue.push_back(Ok(sse_event("content_block_delta", &delta)));

    let cb_stop = json!({"type": "content_block_stop", "index": *current_index});
    queue.push_back(Ok(sse_event("content_block_stop", &cb_stop)));
    *current_index += 1;
    // Force a fresh block next time around.
    *current_block = None;
}

// ── AnthropicState ────────────────────────────────────────────────────────────

/// Streaming-side state machine. Lazily opens a content block on the
/// first visible token, transitions blocks when `is_thinking` flips, and
/// (A5.5) interleaves tool_use blocks with text/thinking blocks. Emits the
/// final `content_block_stop` + `message_delta` + `message_stop` triplet on
/// stream end.
///
/// The variants differ in size (Streaming carries the parser + token stream;
/// Epilogue is a small VecDeque), but the enum is per-request transient
/// state — boxing the hot variant would add a heap-alloc per state
/// transition for no real win, so the lint is suppressed.
#[allow(clippy::large_enum_variant)]
enum AnthropicState {
    Streaming {
        token_stream: futures::stream::BoxStream<'static, rmlx_core::Result<GenerationToken>>,
        /// A5.5: parser when tools are enabled for this request.
        parser: Option<ToolCallStreamParser>,
        output_tokens: u32,
        finish_reason: Option<String>,
        /// `None` until the first non-empty piece arrives, or after a
        /// tool_use block closes (force a fresh block).
        current_block: Option<BlockKind>,
        /// Index of the currently-open block (0 for the first block,
        /// increments on every transition or tool_use close).
        current_index: u32,
        /// A5.5: tracks whether any tool_use block has been emitted, used
        /// to upgrade the terminal `stop_reason`.
        any_tool_use: bool,
        /// Pending events queued by the most recent step (e.g. a
        /// block transition emits 3 events: stop-old, start-new,
        /// delta-on-new). Drained before pulling the next token.
        pending: std::collections::VecDeque<Result<Event, std::convert::Infallible>>,
        /// F1b: drainer handle for PromptTokens/CompletionTokens emit at epilogue.
        drainer: Option<crate::metrics_drainer::DrainerHandle>,
        /// F1b: snapshot basename for MetricEvent.model_id.
        metrics_model_id: String,
        /// F1b: prompt token count from request (before generate consumed it).
        input_token_count: u32,
        /// F1b/F2: effective_max_ctx for MetricEvent.ctx_max.
        metrics_ctx_max: i64,
        /// F14: process-lifetime prompt-token counter shared with AppState.
        lifetime_tokens_in: Arc<std::sync::atomic::AtomicU64>,
        /// F14: process-lifetime completion-token counter shared with AppState.
        lifetime_tokens_out: Arc<std::sync::atomic::AtomicU64>,
        /// F8: per-category error counters shared with AppState.
        error_counts: crate::openai::ApiErrorCounters,
        /// process-lifetime completed-request counter shared with AppState.
        lifetime_requests_completed: Arc<std::sync::atomic::AtomicU64>,
        /// process-lifetime failed-request counter shared with AppState.
        lifetime_requests_failed: Arc<std::sync::atomic::AtomicU64>,
        /// Optional adaptive controller handle + admission metadata.
        admission_ctrl: Option<(
            crate::admission::ControllerHandle,
            u64, // queue_depth
            u64, // queue_wait_ms
        )>,
        /// Wall-clock instant the streaming request started (for step_ms).
        request_start: Instant,
        /// Stop-sequence matcher for the Text content channel (inert
        /// when the request set no stop strings). Thinking deltas bypass it.
        stop_matcher: crate::stop_matcher::StopMatcher,
        /// Set once a stop string matched. Suppresses further text and
        /// forces stop_reason="stop_sequence" + the matched `stop_sequence`.
        stop_hit: bool,
        /// The stop string that matched, for the `stop_sequence` field.
        matched_stop: Option<String>,
    },
    Epilogue {
        events: std::collections::VecDeque<Result<Event, std::convert::Infallible>>,
    },
    Done,
}

// ── Streaming generation path ─────────────────────────────────────────────────

/// Streaming generation path.
///
/// Returns 503 BEFORE writing any SSE bytes if the generator is not ready.
/// Once the stream opens, emits the canonical Anthropic SSE event order
/// with one network flush per token:
/// 1. `message_start`
/// 2. `content_block_start`
/// 3. `ping`
/// 4. `content_block_delta` × N (one per token, streamed lazily)
/// 5. `content_block_stop`
/// 6. `message_delta`
/// 7. `message_stop`
///
/// L6: `request_start` is the `Instant` captured at handler entry. TTFT is
/// computed when the first token arrives from the decode thread.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_streaming(
    generator: Arc<dyn crate::engine::Generator>,
    req: GenerationRequest,
    // when Some, the token stream is wrapped in the replay envelope.
    replay_plan: Option<crate::retry::RequestPlan>,
    model_id: &str,
    request_start: Instant,
    state: &AppState,
    parser_format: Option<ToolCallFormat>,
    // F1/F2: server effective_max_ctx for drainer MetricEvent.ctx_max.
    ctx_max_for_metrics: i64,
    // F10: correlation id resolved at handler entry.
    request_id: &str,
    // cold/warm flag for TTFT metric name selection.
    is_cold_request: bool,
    // Queue admission values for StepMetrics recording.
    queue_at_admission: (u64, u64),
    // Decode-lease guard moved into the SSE response's body via
    // GuardedStream — drops when the stream is fully consumed or the client
    // disconnects.
    decode_lease: Option<crate::keep_alive::DecodeLeaseGuard>,
) -> Response {
    // F1b: capture input token count before generator consumes the request.
    let input_token_count = req.prompt_tokens.len() as u32;
    // Capture stop sequences before `req` is moved into the generator.
    let stop_sequences = req.stop.clone();
    // use the replay envelope when eligible; direct generate otherwise.
    let mut token_stream: std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = rmlx_core::Result<GenerationToken>> + Send>,
    > = match replay_plan {
        Some(plan) => Box::pin(crate::retry::replay_stream(
            generator,
            req,
            plan,
            crate::retry::DEFAULT_MAX_RETRIES,
        )),
        None => generator.generate(req),
    };

    // Peek first item — return error response before any SSE bytes.
    let first_ok: Option<GenerationToken> = match token_stream.next().await {
        None => None,
        Some(Err(e)) => {
            state.error_counts.increment(engine_error_category(&e));
            // count engine errors as failed requests.
            state
                .requests_failed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return engine_error_response(&e);
        }
        Some(Ok(tok)) => {
            // L6: TTFT captured immediately when the first token arrives from
            // the decode thread — before SSE serialisation or TCP flush.
            let ttft_ms = request_start.elapsed().as_millis() as u64;
            tracing::info!(
                model_id,
                ttft_ms,
                "generate_streaming (anthropic): TTFT (L6)"
            );
            {
                use crate::openai::{TtftSample, TTFT_RING_CAPACITY};
                let mut ring = state.ttft_store.lock();
                if ring.len() >= TTFT_RING_CAPACITY {
                    ring.pop_front();
                }
                ring.push_back(TtftSample {
                    model_id: model_id.to_owned(),
                    ttft_ms,
                });
            }
            // F1a: emit TtftMs to SQLite via SPSC drainer (single emit per request).
            if let Some(ref drainer) = state.metrics_drainer {
                use crate::metrics_drainer::{MetricEvent, MetricKind};
                drainer.try_emit(MetricEvent {
                    model_id: model_id.to_owned(),
                    kv_quant: "none".into(),
                    ts_utc: chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                    ctx_max: ctx_max_for_metrics,
                    kind: MetricKind::TtftMs(ttft_ms),
                });
            }
            // write TTFT to events table off the tokio worker via
            // spawn_blocking so SQLite I/O never stalls the async executor.
            // also emit `prefill_duration_ms` (same value, distinct op).
            tracing::debug!(
                model_id,
                phase = ?crate::engine::Phase::Decode,
                ttft_ms,
                "phase transition Prefill -> Decode (anthropic streaming)"
            );
            if let Some(rec) = state.metrics.clone() {
                let model_id_owned = model_id.to_owned();
                tokio::task::spawn_blocking(move || {
                    crate::engine::record_ttft_and_prefill(
                        &rec,
                        &model_id_owned,
                        is_cold_request,
                        ttft_ms,
                    );
                });
            }
            Some(tok)
        }
    };

    // F10: use the correlation id resolved at handler entry so that the id
    // embedded in every SSE event matches the X-Request-Id response header.
    let id = format!("msg_{request_id}");
    let model = model_id.to_owned();

    // ── message_start preamble (always emitted) ──────────────────────────────
    //
    // The content_block_start is emitted lazily by the unfold state machine
    // below on the FIRST visible delta, so that the block kind (`text` vs
    // `thinking`) matches the first piece's channel. Anthropic's public
    // streaming API permits this — content_block_start fires when the
    // server commits to a block, not unconditionally up front.
    let msg_start = json!({
        "type": "message_start",
        "message": {
            "id": id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": model,
            "stop_reason": null,
            "stop_sequence": null,
            "usage": {"input_tokens": 0, "output_tokens": 0}
        }
    });
    let ping = json!({"type": "ping"});

    let preamble = futures::stream::iter(vec![
        Ok::<Event, std::convert::Infallible>(sse_event("message_start", &msg_start)),
        Ok(sse_event("ping", &ping)),
    ]);

    // ── Per-token deltas: streamed lazily ─────────────────────────────────────
    // Reconstruct the full token stream by re-attaching the peeked first token.
    let full_token_stream: futures::stream::BoxStream<'static, rmlx_core::Result<GenerationToken>> =
        match first_ok {
            None => futures::stream::empty().boxed(),
            Some(tok) => futures::stream::once(async move { Ok(tok) })
                .chain(token_stream)
                .boxed(),
        };

    // A5.5: instantiate parser when the caller supplied a format.
    let parser_init: Option<ToolCallStreamParser> = parser_format.map(ToolCallStreamParser::new);

    // F1b: clone drainer + capture context for PromptTokens/CompletionTokens
    // emit at the epilogue boundary (inside the `unfold` async closure).
    let drainer_for_stream = state.metrics_drainer.clone();
    let model_id_for_stream = model_id.to_owned();

    let token_and_epilogue = futures::stream::unfold(
        AnthropicState::Streaming {
            token_stream: full_token_stream,
            parser: parser_init,
            output_tokens: 0,
            finish_reason: None,
            current_block: None,
            current_index: 0,
            any_tool_use: false,
            pending: std::collections::VecDeque::new(),
            drainer: drainer_for_stream,
            metrics_model_id: model_id_for_stream,
            input_token_count,
            metrics_ctx_max: ctx_max_for_metrics,
            // F14: share process-lifetime counters from AppState.
            lifetime_tokens_in: Arc::clone(&state.tokens_in),
            lifetime_tokens_out: Arc::clone(&state.tokens_out),
            // F8: share per-category error counters from AppState.
            error_counts: state.error_counts.clone(),
            // share request lifecycle counters from AppState.
            lifetime_requests_completed: Arc::clone(&state.requests_completed),
            lifetime_requests_failed: Arc::clone(&state.requests_failed),
            // Adaptive controller handle + admission metadata.
            admission_ctrl: state
                .admission_controller
                .as_ref()
                .map(|ctrl| (ctrl.clone(), queue_at_admission.0, queue_at_admission.1)),
            request_start,
            // Text-channel stop matcher (inert when no stop strings).
            stop_matcher: crate::stop_matcher::StopMatcher::new(&stop_sequences),
            stop_hit: false,
            matched_stop: None,
        },
        |state| async move {
            match state {
                AnthropicState::Done => None,
                AnthropicState::Epilogue { mut events } => {
                    let ev = events.pop_front()?;
                    if events.is_empty() {
                        Some((ev, AnthropicState::Done))
                    } else {
                        Some((ev, AnthropicState::Epilogue { events }))
                    }
                }
                AnthropicState::Streaming {
                    mut token_stream,
                    mut parser,
                    mut output_tokens,
                    mut finish_reason,
                    mut current_block,
                    mut current_index,
                    mut any_tool_use,
                    mut pending,
                    drainer,
                    metrics_model_id,
                    input_token_count,
                    metrics_ctx_max,
                    lifetime_tokens_in,
                    lifetime_tokens_out,
                    error_counts,
                    lifetime_requests_completed,
                    lifetime_requests_failed,
                    admission_ctrl,
                    request_start,
                    mut stop_matcher,
                    mut stop_hit,
                    mut matched_stop,
                } => {
                    // Drain any queued events from the previous step first.
                    if let Some(ev) = pending.pop_front() {
                        return Some((
                            ev,
                            AnthropicState::Streaming {
                                token_stream,
                                parser,
                                output_tokens,
                                finish_reason,
                                current_block,
                                current_index,
                                any_tool_use,
                                pending,
                                drainer,
                                metrics_model_id,
                                input_token_count,
                                metrics_ctx_max,
                                lifetime_tokens_in,
                                lifetime_tokens_out,
                                error_counts,
                                lifetime_requests_completed,
                                lifetime_requests_failed,
                                admission_ctrl,
                                request_start,
                                stop_matcher,
                                stop_hit,
                                matched_stop,
                            },
                        ));
                    }

                    // `loop { match ... }` is intentional: the loop has
                    // multiple control-flow exits (return, continue,
                    // break). Refactoring to `while let Some(Ok(tok)) =
                    // ...` would lose the `continue` for empty pieces.
                    #[allow(clippy::while_let_loop)]
                    loop {
                        match token_stream.next().await {
                            Some(Ok(tok)) => {
                                output_tokens += 1;
                                if tok.done {
                                    finish_reason.clone_from(&tok.finish_reason);
                                }
                                // A3: skip empty pieces — they would
                                // produce an empty delta, which Anthropic
                                // clients should never see.
                                if tok.piece.is_empty() {
                                    if tok.done {
                                        // Fall through to epilogue below.
                                        break;
                                    }
                                    continue;
                                }

                                // A5.5 / A5.6: feed the parser regardless of
                                // think state. A reasoning model may emit the
                                // tool call without closing `</think>`
                                // (`Ternary-Bonsai`); a thinking-only bypass
                                // would never surface `tool_use`. Parser
                                // passthrough is routed to the block matching
                                // the token's think state (Thinking vs Text),
                                // so genuine reasoning still streams as a
                                // thinking block and Qwen3.6 (closes
                                // `</think>` first) is unaffected.
                                let mut queue: std::collections::VecDeque<
                                    Result<Event, std::convert::Infallible>,
                                > = std::collections::VecDeque::new();

                                let passthrough_kind = if tok.is_thinking {
                                    BlockKind::Thinking
                                } else {
                                    BlockKind::Text
                                };
                                let (drained_text, drained_calls): (
                                    Option<String>,
                                    Vec<ParsedToolCall>,
                                ) = match parser.as_mut() {
                                    Some(p) => {
                                        p.push(&tok.piece);
                                        let text = if p.passthrough_text.is_empty() {
                                            None
                                        } else {
                                            Some(std::mem::take(&mut p.passthrough_text))
                                        };
                                        (text, p.take_parsed())
                                    }
                                    None => (Some(tok.piece.clone()), Vec::new()),
                                };

                                if let Some(text) = drained_text {
                                    // Gate the Text channel through the
                                    // stop matcher; Thinking deltas bypass it.
                                    if passthrough_kind == BlockKind::Text
                                        && stop_matcher.is_active()
                                    {
                                        if !stop_hit {
                                            let pushed = stop_matcher.push(&text);
                                            if !pushed.emit.is_empty() {
                                                enqueue_text_or_thinking_delta(
                                                    &mut queue,
                                                    &mut current_block,
                                                    &mut current_index,
                                                    BlockKind::Text,
                                                    &pushed.emit,
                                                );
                                            }
                                            if pushed.stopped {
                                                stop_hit = true;
                                                matched_stop = pushed.matched;
                                                tracing::debug!(
                                                    stop = ?matched_stop,
                                                    "stop sequence matched (anthropic streaming)"
                                                );
                                            }
                                        }
                                    } else {
                                        enqueue_text_or_thinking_delta(
                                            &mut queue,
                                            &mut current_block,
                                            &mut current_index,
                                            passthrough_kind,
                                            &text,
                                        );
                                    }
                                }
                                for parsed in &drained_calls {
                                    enqueue_tool_use_block(
                                        &mut queue,
                                        &mut current_block,
                                        &mut current_index,
                                        parsed,
                                    );
                                    any_tool_use = true;
                                }

                                tracing::trace!(
                                    token_id = tok.token_id,
                                    piece = %tok.piece,
                                    done = tok.done,
                                    is_thinking = tok.is_thinking,
                                    "sse(anthropic): handled token piece"
                                );

                                // Once a stop matched, drain whatever is
                                // queued and then break to the epilogue — no
                                // further tokens are processed.
                                if stop_hit {
                                    if let Some(ev) = queue.pop_front() {
                                        // Emit the queued pre-stop delta, then
                                        // re-enter with the rest pending and the
                                        // token stream exhausted on next poll.
                                        pending = queue;
                                        return Some((
                                            ev,
                                            AnthropicState::Streaming {
                                                token_stream: futures::stream::empty().boxed(),
                                                parser,
                                                output_tokens,
                                                finish_reason,
                                                current_block,
                                                current_index,
                                                any_tool_use,
                                                pending,
                                                drainer,
                                                metrics_model_id,
                                                input_token_count,
                                                metrics_ctx_max,
                                                lifetime_tokens_in,
                                                lifetime_tokens_out,
                                                error_counts,
                                                lifetime_requests_completed,
                                                lifetime_requests_failed,
                                                admission_ctrl,
                                                request_start,
                                                stop_matcher,
                                                stop_hit,
                                                matched_stop,
                                            },
                                        ));
                                    }
                                    break;
                                }

                                // If this token produced no visible events
                                // (everything buffered inside parser), keep
                                // looping for the next token. Done is also
                                // possible mid-loop and breaks to epilogue.
                                if queue.is_empty() {
                                    if tok.done {
                                        break;
                                    }
                                    continue;
                                }

                                let Some(ev) = queue.pop_front() else {
                                    // Invariant: queue is non-empty (guard above skips empty).
                                    // A None here means state-machine corruption.
                                    tracing::warn!(
                                        model_id = %metrics_model_id,
                                        "sse(anthropic): queue drain invariant violated; continuing"
                                    );
                                    continue;
                                };
                                return Some((
                                    ev,
                                    AnthropicState::Streaming {
                                        token_stream,
                                        parser,
                                        output_tokens,
                                        finish_reason,
                                        current_block,
                                        current_index,
                                        any_tool_use,
                                        pending: queue,
                                        drainer,
                                        metrics_model_id,
                                        input_token_count,
                                        metrics_ctx_max,
                                        lifetime_tokens_in,
                                        lifetime_tokens_out,
                                        error_counts,
                                        lifetime_requests_completed,
                                        lifetime_requests_failed,
                                        admission_ctrl,
                                        request_start,
                                        stop_matcher,
                                        stop_hit,
                                        matched_stop,
                                    },
                                ));
                            }
                            Some(Err(e)) => {
                                // F8: mid-stream engine error — count even
                                // though HTTP 200 is already sent.
                                error_counts.increment(engine_error_category(&e));
                                // count mid-stream failures as failed requests.
                                lifetime_requests_failed
                                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                break;
                            }
                            None => break,
                        }
                    }

                    // ── Epilogue ──────────────────────────────────────────
                    // Drain any residual passthrough + completed tool_calls
                    // from the parser one last time. These produce real
                    // visible blocks that must be emitted before
                    // content_block_stop / message_delta / message_stop.
                    let mut events: std::collections::VecDeque<
                        Result<Event, std::convert::Infallible>,
                    > = std::collections::VecDeque::new();
                    if let Some(p) = parser.as_mut() {
                        if !p.passthrough_text.is_empty() && !stop_hit {
                            let text = std::mem::take(&mut p.passthrough_text);
                            // Route the final tail through the matcher.
                            if stop_matcher.is_active() {
                                let pushed = stop_matcher.push(&text);
                                if !pushed.emit.is_empty() {
                                    enqueue_text_or_thinking_delta(
                                        &mut events,
                                        &mut current_block,
                                        &mut current_index,
                                        BlockKind::Text,
                                        &pushed.emit,
                                    );
                                }
                                if pushed.stopped {
                                    stop_hit = true;
                                    matched_stop = pushed.matched;
                                }
                            } else {
                                enqueue_text_or_thinking_delta(
                                    &mut events,
                                    &mut current_block,
                                    &mut current_index,
                                    BlockKind::Text,
                                    &text,
                                );
                            }
                        } else if !p.passthrough_text.is_empty() {
                            // stop already hit — discard the post-stop tail.
                            p.passthrough_text.clear();
                        }
                        for parsed in &p.take_parsed() {
                            enqueue_tool_use_block(
                                &mut events,
                                &mut current_block,
                                &mut current_index,
                                parsed,
                            );
                            any_tool_use = true;
                        }
                    }
                    // Flush the matcher's held-back tail if no stop hit.
                    if !stop_hit && stop_matcher.is_active() {
                        let tail = stop_matcher.finalize();
                        if !tail.is_empty() {
                            enqueue_text_or_thinking_delta(
                                &mut events,
                                &mut current_block,
                                &mut current_index,
                                BlockKind::Text,
                                &tail,
                            );
                        }
                    }

                    // A5.5: when the previous block was a tool_use, it has
                    // already been closed by `enqueue_tool_use_block` and
                    // `current_block` was reset to `None` with `current_index`
                    // already advanced past the closed block. In that case
                    // skip both the safety-open AND the trailing
                    // `content_block_stop` — the message is already well-
                    // formed.
                    //
                    // The pre-A5.5 safety-open only fires when no block was
                    // ever opened (the model produced only empty pieces —
                    // exotic edge case). Detect that by checking
                    // `current_index == 0 && current_block.is_none()`.
                    let no_block_ever_opened = current_block.is_none() && current_index == 0;
                    let need_trailing_stop = current_block.is_some() || no_block_ever_opened;

                    if no_block_ever_opened {
                        events.push_back(Ok::<Event, std::convert::Infallible>(
                            BlockKind::Text.cb_start_event(0),
                        ));
                    }

                    // A stop-sequence match forces stop_reason
                    // "stop_sequence" and names the matched stop, overriding the
                    // engine's terminal (EOS/length) reason.
                    let (stop_reason, stop_sequence_field) = if stop_hit {
                        (
                            "stop_sequence".to_owned(),
                            Value::from(matched_stop.clone()),
                        )
                    } else {
                        let terminal_stop_reason = map_stop_reason(finish_reason.as_deref());
                        (
                            select_anthropic_stop_reason(any_tool_use, terminal_stop_reason),
                            Value::Null,
                        )
                    };
                    if need_trailing_stop {
                        let cb_stop = json!({"type": "content_block_stop", "index": current_index});
                        events.push_back(Ok(sse_event("content_block_stop", &cb_stop)));
                    }
                    let msg_delta = json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason, "stop_sequence": stop_sequence_field},
                        "usage": {"output_tokens": output_tokens}
                    });
                    let msg_stop = json!({"type": "message_stop"});
                    events.push_back(Ok(sse_event("message_delta", &msg_delta)));
                    events.push_back(Ok(sse_event("message_stop", &msg_stop)));

                    // F1b: emit PromptTokens + CompletionTokens to SQLite via
                    // the SPSC drainer at epilogue boundary — same token counts
                    // that populate the Anthropic `usage` response body.
                    // Single-source, single emit per completed request.
                    if let Some(ref d) = drainer {
                        use crate::metrics_drainer::{MetricEvent, MetricKind};
                        let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
                        d.try_emit(MetricEvent {
                            model_id: metrics_model_id.clone(),
                            kv_quant: "none".into(),
                            ts_utc: ts.clone(),
                            ctx_max: metrics_ctx_max,
                            kind: MetricKind::PromptTokens(input_token_count),
                        });
                        d.try_emit(MetricEvent {
                            model_id: metrics_model_id.clone(),
                            kv_quant: "none".into(),
                            ts_utc: ts,
                            ctx_max: metrics_ctx_max,
                            kind: MetricKind::CompletionTokens(output_tokens),
                        });
                    }
                    // F14: increment process-lifetime token counters (single
                    // source: same values as the SPSC drainer emit above).
                    lifetime_tokens_in.fetch_add(
                        u64::from(input_token_count),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    lifetime_tokens_out.fetch_add(
                        u64::from(output_tokens),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    // count successfully completed streaming requests.
                    lifetime_requests_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // Skip record_step in streaming paths where
                    // kv_cache_bytes is unavailable (generator Arc consumed by
                    // stream). Zero-KV samples bias the regressor; under-sample.
                    if admission_ctrl.is_some() {
                        tracing::debug!(
                            prompt_tokens = input_token_count,
                            "StepMetrics skipped (anthropic streaming — kv_bytes unavailable)"
                        );
                    }
                    // Invariant: events has at least 2 items (pushed just above).
                    let Some(ev) = events.pop_front() else {
                        tracing::warn!(
                            model_id = %metrics_model_id,
                            "sse(anthropic): epilogue events queue unexpectedly empty \
                             — dropping epilogue, terminating stream"
                        );
                        return None;
                    };
                    if events.is_empty() {
                        Some((ev, AnthropicState::Done))
                    } else {
                        Some((ev, AnthropicState::Epilogue { events }))
                    }
                }
            }
        },
    );

    // F10: add the correlation id as a response header on the SSE response.
    // Wrap the composed SSE stream in a GuardedStream so the
    // decode-lease guard drops when the stream is fully consumed or the
    // client disconnects. Box::pin the composed stream so the wrapper meets
    // its `Unpin` bound.
    let boxed: std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(preamble.chain(token_and_epilogue));
    let guarded = crate::keep_alive::GuardedStream::new(boxed, decode_lease);
    let mut resp = Sse::new(guarded).into_response();
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        resp.headers_mut().insert("x-request-id", hv);
    }
    resp
}
