//! SSE streaming helpers for the OpenAI-compatible chat endpoint.
//!
//! - `StreamState` — per-request state threaded through the `unfold` stream.
//! - `make_content_chunk` — build a content / reasoning_content SSE chunk.
//! - `make_tool_call_chunk` — build a tool_calls SSE chunk.
//! - `chunk_to_event` — serialise a chunk to an SSE event.
//! - `handle_streaming_token` — map one engine token to ≥0 SSE events.

use std::sync::Arc;

use axum::response::sse::Event;

use crate::engine::GenerationToken;
use crate::metrics_drainer::DrainerHandle;
use crate::openai::chat::bare_json_to_tool_call;
use crate::openai::errors::engine_error_category;
use crate::tool_parser::{ParsedToolCall, ToolCallStreamParser};

use super::response::{
    select_finish_reason, to_response_tool_call, ChatCompletionChunk, ChatLogprobs, DeltaContent,
    StreamChoice, ToolCall, Usage,
};
use super::state::ApiErrorCounters;

// ── StreamState ───────────────────────────────────────────────────────────────

/// State threaded through the per-request SSE stream via `futures::stream::unfold`.
pub(crate) struct StreamState {
    pub(crate) parser: Option<ToolCallStreamParser>,
    pub(crate) next_tool_index: u32,
    pub(crate) any_tool_calls: bool,
    pub(crate) id: String,
    pub(crate) model: String,
    pub(crate) created: u64,
    /// A6.5: when `true` (json_object / json_schema mode), pre-JSON tokens
    /// are buffered here and discarded if they form a markdown-fence prefix.
    /// Once the first JSON value byte is seen, `json_fence_buf_done` is set
    /// and all subsequent pieces flow directly to content.
    pub(crate) json_object_mode: bool,
    /// Pre-engagement fence buffer for the streaming path (see above).
    pub(crate) json_fence_buf: String,
    /// Set once the fence-buffer phase is complete (first JSON byte seen).
    pub(crate) json_fence_buf_done: bool,
    /// H4: prompt token count captured before `generator.generate` consumes
    /// the `GenerationRequest`. Used for the usage-summary chunk.
    pub(crate) prompt_tokens: u32,
    /// H4: running tally of decode steps (incremented once per non-done
    /// token, including thinking tokens — mirrors the non-streaming counter).
    pub(crate) completion_tokens: u32,
    /// H4: whether to emit a usage-summary chunk before `[DONE]`.
    pub(crate) include_usage: bool,
    /// When true (tool_choice=required/named), the model is constrained
    /// to emit bare JSON. At stream-end, the accumulated JSON is converted
    /// into a tool_calls envelope rather than streamed as content.
    pub(crate) bare_json_tool_call_mode: bool,
    /// Accumulated content for bare_json_tool_call_mode. All non-empty
    /// content pieces are appended here; converted at done-token boundary.
    pub(crate) bare_json_accum: String,
    /// F1b: SPSC drainer handle for emitting PromptTokens/CompletionTokens
    /// once per request at the `done` token boundary. `None` in tests.
    pub(crate) metrics_drainer: Option<DrainerHandle>,
    /// F1b/F2: model snapshot basename (for MetricEvent.model_id).
    pub(crate) metrics_model_id: String,
    /// F1b/F2: server effective_max_ctx clamped to i64 range.
    pub(crate) metrics_ctx_max: i64,
    /// F14: process-lifetime prompt-token counter shared with AppState.
    ///
    /// Incremented once at the done-token boundary (same as the SPSC drainer
    /// emit) so there is exactly one source of truth per request.
    pub(crate) lifetime_tokens_in: Arc<std::sync::atomic::AtomicU64>,
    /// F14: process-lifetime completion-token counter shared with AppState.
    pub(crate) lifetime_tokens_out: Arc<std::sync::atomic::AtomicU64>,
    /// F8: per-category error counters shared with AppState.
    ///
    /// Incremented when an engine error terminates the SSE stream.
    pub(crate) error_counts: ApiErrorCounters,
    /// process-lifetime completed-request counter shared with AppState.
    ///
    /// Incremented once at the done-token boundary (success path).
    pub(crate) lifetime_requests_completed: Arc<std::sync::atomic::AtomicU64>,
    /// process-lifetime failed-request counter shared with AppState.
    ///
    /// Incremented when an engine error terminates the SSE stream.
    pub(crate) lifetime_requests_failed: Arc<std::sync::atomic::AtomicU64>,

    /// Optional adaptive controller handle + admission metadata.
    ///
    /// `None` when `--adaptive-admission` is off (default). When `Some`, a debug
    /// trace is emitted at the done-token boundary. StepMetrics recording is skipped
    /// on the streaming path — kv_bytes unavailable after generator is consumed.
    /// The tuple fields are `(handle, queue_depth, queue_wait_ms)`.
    pub(crate) admission_ctrl: Option<(
        crate::admission::ControllerHandle,
        u64, // queue_depth at admission
        u64, // queue_wait_ms
    )>,

    /// Stop-sequence matcher for the content channel. Inert
    /// (`is_active() == false`) when the request set no stop strings. Holds
    /// back a partial-match tail so a stop straddling token boundaries is
    /// never half-emitted.
    pub(crate) stop_matcher: crate::stop_matcher::StopMatcher,
    /// Set once a stop string matched. Suppresses all further content
    /// and forces `finish_reason="stop"` on the terminal chunk.
    pub(crate) stop_hit: bool,
}

// ── Chunk builders ────────────────────────────────────────────────────────────

/// Build a content / reasoning_content chunk for streaming. Mirrors
/// `make_token_chunk` but takes already-routed `(content, reasoning_content)`.
pub(super) fn make_content_chunk(
    state: &StreamState,
    content: Option<String>,
    reasoning_content: Option<String>,
    done: bool,
    finish: Option<String>,
    // per-token logprobs for the token in this delta, or `None`.
    logprobs: Option<ChatLogprobs>,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: state.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: state.created,
        model: state.model.clone(),
        choices: vec![StreamChoice {
            index: 0,
            delta: DeltaContent {
                role: None,
                content,
                reasoning_content,
                tool_calls: None,
            },
            finish_reason: if done { finish } else { None },
            logprobs,
        }],
        usage: None,
    }
}

/// Build a tool_calls chunk for streaming (one complete call per chunk in v1).
pub(super) fn make_tool_call_chunk(
    state: &StreamState,
    tool_call: ToolCall,
) -> ChatCompletionChunk {
    ChatCompletionChunk {
        id: state.id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created: state.created,
        model: state.model.clone(),
        choices: vec![StreamChoice {
            index: 0,
            delta: DeltaContent {
                role: None,
                content: None,
                reasoning_content: None,
                tool_calls: Some(vec![tool_call]),
            },
            finish_reason: None,
            logprobs: None,
        }],
        usage: None,
    }
}

/// Serialise a chunk to an SSE event.
pub(super) fn chunk_to_event(chunk: &ChatCompletionChunk) -> Event {
    Event::default().data(serde_json::to_string(chunk).unwrap_or_default())
}

// ── Token handler ─────────────────────────────────────────────────────────────

/// Map one input token to zero-or-more SSE events, updating `state`.
pub(crate) fn handle_streaming_token(
    item: rmlx_core::Result<GenerationToken>,
    state: &mut StreamState,
) -> Vec<Result<Event, std::convert::Infallible>> {
    let tok = match item {
        Err(e) => {
            // F8: count mid-stream engine errors even though the HTTP status
            // is already 200 (SSE stream has started). The category is
            // derived from the same classification as the blocking path.
            state.error_counts.increment(engine_error_category(&e));
            // count mid-stream failures as failed requests.
            state
                .lifetime_requests_failed
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return vec![Ok(Event::default().data("[DONE]"))];
        }
        Ok(t) => t,
    };
    let done = tok.done;
    let finish = tok.finish_reason.clone();
    let is_thinking = tok.is_thinking;
    // wrap this token's logprob record (if any) for the content chunk.
    // `.take()`n on the first content emit; reasoning / done / tool deltas
    // carry no logprobs (OpenAI attaches logprobs to content tokens only).
    let mut tok_logprobs: Option<ChatLogprobs> = tok
        .logprobs
        .clone()
        .map(|c| ChatLogprobs { content: vec![c] });
    let piece = tok.piece;
    tracing::trace!(
        token_id = tok.token_id,
        piece = %piece,
        done,
        is_thinking,
        "sse: handling token"
    );

    // H4: count every token (including thinking tokens and the done marker)
    // to mirror the non-streaming counter in `generate_blocking`.
    state.completion_tokens += 1;

    let mut out: Vec<Result<Event, std::convert::Infallible>> = Vec::new();

    // bare_json_tool_call_mode — when tool_choice=required/named the
    // constraint forces bare JSON output. For thinking models (Bonsai/Qwen3)
    // whose template starts inside <think>, the constrained JSON is emitted
    // while is_thinking == true. Route ALL pieces to bare_json_accum in this
    // mode so the done-handler can extract and convert the tool call.
    if is_thinking && state.bare_json_tool_call_mode {
        if !piece.is_empty() {
            state.bare_json_accum.push_str(&piece);
        }
    } else if is_thinking {
        // A5.6: reasoning-channel text. A reasoning model may emit the tool
        // call WITHOUT closing `</think>` (e.g. `Ternary-Bonsai`), so the
        // parser must still scan thinking pieces — otherwise the
        // `<tool_call>` block leaks into `reasoning_content` and no
        // `tool_calls` are streamed. Parser passthrough while thinking is
        // emitted as reasoning content (channel preserved); extracted calls
        // are streamed as tool_call deltas. When no parser is active this
        // is the unchanged plain-reasoning path.
        let (drained_text, drained_calls): (Option<String>, Vec<ParsedToolCall>) =
            match state.parser.as_mut() {
                Some(parser) => {
                    if piece.is_empty() {
                        (None, Vec::new())
                    } else {
                        parser.push(&piece);
                        let text = if parser.passthrough_text.is_empty() {
                            None
                        } else {
                            Some(std::mem::take(&mut parser.passthrough_text))
                        };
                        (text, parser.take_parsed())
                    }
                }
                None => (
                    if piece.is_empty() { None } else { Some(piece) },
                    Vec::new(),
                ),
            };
        if let Some(text) = drained_text {
            // attach this token's logprobs to the reasoning delta too.
            // Thinking-model output (e.g. Bonsai) flows on the reasoning
            // channel, so logprobs must ride along to mirror the non-streaming
            // path (which records logprobs per token regardless of channel).
            let chunk =
                make_content_chunk(state, None, Some(text), false, None, tok_logprobs.take());
            out.push(Ok(chunk_to_event(&chunk)));
        }
        for p in &drained_calls {
            let idx = state.next_tool_index;
            state.next_tool_index += 1;
            state.any_tool_calls = true;
            let tc = to_response_tool_call(p, idx);
            let chunk = make_tool_call_chunk(state, tc);
            out.push(Ok(chunk_to_event(&chunk)));
        }
    } else {
        // A6.5: fence suppression for streaming. When in json_object/json_schema
        // mode and we have not yet seen the first JSON value byte, buffer the
        // piece. Once a JSON-value-starter byte appears, discard everything
        // before it (the fence) and emit only from that byte onward.
        let piece = if state.json_object_mode && !state.json_fence_buf_done && !piece.is_empty() {
            state.json_fence_buf.push_str(&piece);
            // Scan for first JSON value-starter byte.
            let start_idx = state.json_fence_buf.as_bytes().iter().position(|&b| {
                matches!(
                    b,
                    b'{' | b'[' | b'"' | b't' | b'f' | b'n' | b'-' | b'0'..=b'9'
                )
            });
            if let Some(idx) = start_idx {
                state.json_fence_buf_done = true;
                let pre = &state.json_fence_buf[..idx];
                if !crate::constraint_json::schema::is_only_fence_or_whitespace(pre) {
                    tracing::debug!(
                        pre_text = %pre,
                        "A6.5: streaming json_object_mode: non-fence pre-text before JSON; kept"
                    );
                } else if !pre.is_empty() {
                    tracing::debug!(
                        "A6.5: streaming json_object_mode: suppressed fence/whitespace prefix"
                    );
                }
                // Take only from idx onward.
                let flushed = state.json_fence_buf.split_off(idx);
                state.json_fence_buf.clear();
                flushed
            } else {
                // Not yet at a JSON byte — fully buffered, nothing to emit.
                String::new()
            }
        } else {
            piece
        };

        // Drain parser-derived (passthrough_text, new_calls) without holding
        // a parser borrow across the make_*_chunk calls (which read state).
        let (drained_text, drained_calls): (Option<String>, Vec<ParsedToolCall>) =
            match state.parser.as_mut() {
                Some(parser) => {
                    if piece.is_empty() {
                        (None, Vec::new())
                    } else {
                        parser.push(&piece);
                        let text = if parser.passthrough_text.is_empty() {
                            None
                        } else {
                            Some(std::mem::take(&mut parser.passthrough_text))
                        };
                        (text, parser.take_parsed())
                    }
                }
                None => (
                    if piece.is_empty() { None } else { Some(piece) },
                    Vec::new(),
                ),
            };
        if let Some(text) = drained_text {
            if state.bare_json_tool_call_mode {
                // Accumulate rather than stream; converted at done boundary.
                state.bare_json_accum.push_str(&text);
            } else if state.stop_hit {
                // A stop already matched; suppress all further content.
            } else if state.stop_matcher.is_active() && !state.any_tool_calls {
                // Gate content through the stop matcher. It holds back a
                // partial-match tail (token-straddling safe) and signals when a
                // stop string is hit, after which content is suppressed and the
                // terminal chunk carries finish_reason="stop".
                // Stop sequences apply to free-text content only; when a tool
                // call is being emitted (any_tool_calls=true), stop-truncation
                // does not apply — uniform with the non-streaming path.
                let pushed = state.stop_matcher.push(&text);
                // LOW-1: take logprobs unconditionally once the piece is
                // consumed into the matcher. When the whole piece is held back
                // (emit.is_empty()), the logprob for this token is dropped
                // rather than left to ride on the next emitted chunk.
                let lp = tok_logprobs.take();
                if !pushed.emit.is_empty() {
                    let chunk = make_content_chunk(state, Some(pushed.emit), None, false, None, lp);
                    out.push(Ok(chunk_to_event(&chunk)));
                }
                if pushed.stopped {
                    state.stop_hit = true;
                    tracing::debug!(
                        stop = ?pushed.matched,
                        "stop sequence matched (streaming); suppressing rest"
                    );
                }
            } else {
                let chunk =
                    make_content_chunk(state, Some(text), None, false, None, tok_logprobs.take());
                out.push(Ok(chunk_to_event(&chunk)));
            }
        }
        for p in &drained_calls {
            let idx = state.next_tool_index;
            state.next_tool_index += 1;
            state.any_tool_calls = true;
            let tc = to_response_tool_call(p, idx);
            let chunk = make_tool_call_chunk(state, tc);
            out.push(Ok(chunk_to_event(&chunk)));
        }
    }

    if done {
        // A5.4: on the terminal token, drain any final passthrough + tool_calls
        // from the parser, then emit one finish chunk with the upgraded reason.
        let (final_text, final_calls): (Option<String>, Vec<ParsedToolCall>) =
            match state.parser.as_mut() {
                Some(parser) => {
                    parser.finalize();
                    let text = if parser.passthrough_text.is_empty() {
                        None
                    } else {
                        Some(std::mem::take(&mut parser.passthrough_text))
                    };
                    (text, parser.take_parsed())
                }
                None => (None, Vec::new()),
            };
        if let Some(text) = final_text {
            if state.stop_hit {
                // Stop already matched — drop the parser's final tail.
            } else if state.stop_matcher.is_active() {
                let pushed = state.stop_matcher.push(&text);
                if !pushed.emit.is_empty() {
                    let chunk =
                        make_content_chunk(state, Some(pushed.emit), None, false, None, None);
                    out.push(Ok(chunk_to_event(&chunk)));
                }
                if pushed.stopped {
                    state.stop_hit = true;
                }
            } else {
                let chunk = make_content_chunk(state, Some(text), None, false, None, None);
                out.push(Ok(chunk_to_event(&chunk)));
            }
        }
        // If a stop matcher is active and no stop matched, flush the
        // held-back tail (it cannot grow into a stop string at end-of-stream).
        if !state.stop_hit && state.stop_matcher.is_active() {
            let tail = state.stop_matcher.finalize();
            if !tail.is_empty() {
                let chunk = make_content_chunk(state, Some(tail), None, false, None, None);
                out.push(Ok(chunk_to_event(&chunk)));
            }
        }
        for p in &final_calls {
            let idx = state.next_tool_index;
            state.next_tool_index += 1;
            state.any_tool_calls = true;
            let tc = to_response_tool_call(p, idx);
            let chunk = make_tool_call_chunk(state, tc);
            out.push(Ok(chunk_to_event(&chunk)));
        }

        // bare_json_tool_call_mode — convert accumulated JSON to tool call.
        if state.bare_json_tool_call_mode && !state.any_tool_calls {
            let json_str =
                crate::openai::generate::extract_top_level_json_value(&state.bare_json_accum)
                    .unwrap_or_default();
            if let Some(tc_parsed) = bare_json_to_tool_call(&json_str) {
                tracing::debug!(
                    name = %tc_parsed.name,
                    "streaming bare_json_tool_call_mode: synthesised tool call at done"
                );
                let idx = state.next_tool_index;
                state.next_tool_index += 1;
                state.any_tool_calls = true;
                let tc = to_response_tool_call(&tc_parsed, idx);
                let chunk = make_tool_call_chunk(state, tc);
                out.push(Ok(chunk_to_event(&chunk)));
            } else if !state.bare_json_accum.is_empty() {
                tracing::warn!(
                    json = %state.bare_json_accum,
                    "bare_json_tool_call_mode: could not parse constrained output; returning as content"
                );
                // Fallback: emit accumulated text as content.
                let accum = std::mem::take(&mut state.bare_json_accum);
                let chunk = make_content_chunk(state, Some(accum), None, false, None, None);
                out.push(Ok(chunk_to_event(&chunk)));
            }
        }

        // A stop-sequence match forces finish_reason="stop", overriding
        // the engine's terminal reason (which is EOS/length based).
        let finish = if state.stop_hit {
            Some("stop".to_owned())
        } else {
            finish
        };
        let upgraded = select_finish_reason(state.any_tool_calls, finish);
        let chunk = make_content_chunk(state, None, None, true, upgraded, None);
        out.push(Ok(chunk_to_event(&chunk)));

        // H4: when `stream_options.include_usage=true`, emit a final summary
        // chunk with `choices: []` and a fully-populated `usage` object.
        // Order: final-delta(finish_reason) → usage-chunk → then [DONE] from
        // the outer `sse_stream` compose.
        if state.include_usage {
            let total = state.prompt_tokens + state.completion_tokens;
            let usage_chunk = ChatCompletionChunk {
                id: state.id.clone(),
                object: "chat.completion.chunk".to_owned(),
                created: state.created,
                model: state.model.clone(),
                choices: vec![],
                usage: Some(Usage {
                    prompt_tokens: state.prompt_tokens,
                    completion_tokens: state.completion_tokens,
                    total_tokens: total,
                }),
            };
            out.push(Ok(chunk_to_event(&usage_chunk)));
        }

        // F1b: emit PromptTokens + CompletionTokens to SQLite via the SPSC
        // drainer once per completed request. Single-source: the same counters
        // that populate the usage chunk / non-streaming Usage body.
        // Emitted regardless of `include_usage` — DB telemetry is always on.
        if let Some(ref drainer) = state.metrics_drainer {
            use crate::metrics_drainer::{MetricEvent, MetricKind};
            let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
            drainer.try_emit(MetricEvent {
                model_id: state.metrics_model_id.clone(),
                kv_quant: "none".into(),
                ts_utc: ts.clone(),
                ctx_max: state.metrics_ctx_max,
                kind: MetricKind::PromptTokens(state.prompt_tokens),
            });
            drainer.try_emit(MetricEvent {
                model_id: state.metrics_model_id.clone(),
                kv_quant: "none".into(),
                ts_utc: ts,
                ctx_max: state.metrics_ctx_max,
                kind: MetricKind::CompletionTokens(state.completion_tokens),
            });
        }
        // F14: increment process-lifetime token counters (single source: same
        // values as the SPSC drainer emit above; no double-count possible).
        state.lifetime_tokens_in.fetch_add(
            u64::from(state.prompt_tokens),
            std::sync::atomic::Ordering::Relaxed,
        );
        state.lifetime_tokens_out.fetch_add(
            u64::from(state.completion_tokens),
            std::sync::atomic::Ordering::Relaxed,
        );
        // count successfully completed streaming requests.
        state
            .lifetime_requests_completed
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Skip record_step in streaming paths where kv_cache_bytes
        // is unavailable (generator Arc is consumed by the stream, no handle
        // survives to the done-token boundary). Recording a zero-KV sample would
        // bias the 2D OLS regressor toward the constant column; under-sampling is
        // preferable. The blocking path (openai/generate.rs) records full samples.
        if state.admission_ctrl.is_some() {
            tracing::debug!(
                prompt_tokens = state.prompt_tokens,
                "StepMetrics skipped (streaming path — kv_bytes unavailable)"
            );
        }
    }

    out
}
