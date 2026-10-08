//! Non-streaming and streaming generation paths for the OpenAI chat endpoint.
//!
//! - `generate_blocking` — collects all tokens then returns a single JSON response.
//! - `generate_streaming` — returns an SSE stream; each token is a separate event.

#[cfg(test)]
#[path = "generate_tests.rs"]
mod generate_tests;

use std::sync::Arc;
use std::time::Instant;

use axum::http::{HeaderValue, StatusCode};
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use rmlx_metrics::events::EventRecorder;

use crate::engine::{GenerationRequest, GenerationToken, Generator};
use crate::metrics_drainer::DrainerHandle;
use crate::openai::errors::{engine_error_category, error_response};
use crate::openai::state::ApiErrorCategory;
use crate::tool_parser::{ParsedToolCall, ToolCallFormat, ToolCallStreamParser};

use super::chat::bare_json_to_tool_call;
use super::errors::{engine_error_response, unix_now};
use super::json_reply::{JsonReply, NOT_ENGAGED_MESSAGE, NOT_ENGAGED_TYPE};
use super::response::{
    select_finish_reason, to_response_tool_call, ChatCompletionChunk, ChatCompletionsResponse,
    ChatLogprobContent, ChatLogprobs, Choice, DeltaContent, ResponseMessage, StreamChoice, Usage,
};
use super::state::{ApiErrorCounters, AppState, TtftSample, TtftStore, TTFT_RING_CAPACITY};
use super::streaming::{handle_streaming_token, StreamState};

// ── Non-streaming path ────────────────────────────────────────────────────────

/// Non-streaming generation path.
#[allow(clippy::too_many_arguments)]
pub(super) async fn generate_blocking(
    generator: Arc<dyn Generator>,
    req: GenerationRequest,
    // when Some, the token stream is wrapped in the replay envelope.
    replay_plan: Option<crate::retry::RequestPlan>,
    model_id: &str,
    parser_format: Option<ToolCallFormat>,
    // Where the request's constraint engaged, cloned by the route before the
    // engine was moved into the generator. `None` when the request built no
    // constraint that reports it.
    engagement: Option<Arc<rmlx_models::Engagement>>,
    // True for a forced tool call (tool_choice=required/named); the text
    // output (bare JSON) must be converted to a tool_calls envelope.
    bare_json_tool_call_mode: bool,
    // Drainer handle + ctx_max for TTFT/token-count DB emission.
    request_start: Instant,
    metrics_drainer: Option<&DrainerHandle>,
    ctx_max: i64,
    // Process-lifetime token counters shared with AppState.
    tokens_in: &Arc<std::sync::atomic::AtomicU64>,
    tokens_out: &Arc<std::sync::atomic::AtomicU64>,
    // Correlation id resolved at handler entry.
    request_id: &str,
    // Per-category error counters shared with AppState.
    error_counts: &ApiErrorCounters,
    // request lifecycle counters.
    requests_completed: &Arc<std::sync::atomic::AtomicU64>,
    requests_failed: &Arc<std::sync::atomic::AtomicU64>,
    // per-event DB recorder for TTFT writes to the events table.
    event_recorder: Option<Arc<EventRecorder>>,
    // cold/warm flag for TTFT metric name selection.
    is_cold_request: bool,
    // Optional adaptive controller handle for StepMetrics recording.
    // None when --adaptive-admission is off (default). When Some, one StepMetrics
    // observation is pushed to the regressor after the request completes.
    admission_ctrl: Option<(
        crate::admission::ControllerHandle,
        u64, // queue_depth at admission
        u64, // queue_wait_ms
    )>,
    // Rolling ring-buffer for TTFT samples shared with AppState.
    // Written here so non-streaming requests populate the same ring as streaming.
    ttft_store: &TtftStore,
) -> Response {
    let prompt_token_count = req.prompt_tokens.len() as u32;
    // Capture the stop sequences before `req` is moved into the
    // generator. Used to truncate `text` at the first stop-string boundary.
    let stop_sequences = req.stop.clone();
    // Clone generator before the stream match so kv_cache_bytes() is
    // still available after generate/replay_stream takes ownership.
    let generator_for_metrics = Arc::clone(&generator);
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
    let mut text = String::new();
    // Accumulate reasoning text separately. Stays empty for
    // non-reasoning archs (the engine never sets `is_thinking == true`).
    let mut reasoning_text = String::new();
    let mut finish_reason: Option<String> = Some("stop".to_owned());
    let mut completion_tokens: u32 = 0;
    // TTFT for non-streaming — measured when the first token arrives
    // from the decode thread, before any serialisation work. `None` until set.
    let mut ttft_ms_blocking: Option<u64> = None;
    // Instantiate parser when caller supplied a format. `None` means
    // tools-disabled — every non-thinking piece flows straight to `text`.
    let mut parser: Option<ToolCallStreamParser> = parser_format.map(ToolCallStreamParser::new);
    let mut tool_calls_accum: Vec<ParsedToolCall> = Vec::new();
    // per-token logprob records, in emission order. Stays empty unless
    // the request set `logprobs:true` (the engine only attaches `logprobs`
    // to tokens then).
    let mut logprobs_accum: Vec<ChatLogprobContent> = Vec::new();
    let mut json_reply = engagement.map(JsonReply::new);

    while let Some(item) = token_stream.next().await {
        match item {
            Err(e) => {
                error_counts.increment(engine_error_category(&e));
                // count engine errors as failed requests.
                requests_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return engine_error_response(&e);
            }
            Ok(tok) => {
                // Capture TTFT on the very first token (completion_tokens == 0).
                // Capture TTFT on the very first token and
                // immediately persist to the events table off the tokio worker
                // (spawn_blocking) so SQLite I/O never stalls the executor.
                // Hoisted here (not post-loop) so TTFT survives mid-stream errors.
                if completion_tokens == 0 {
                    let ttft_ms = request_start.elapsed().as_millis() as u64;
                    ttft_ms_blocking = Some(ttft_ms);
                    tracing::info!(model_id, ttft_ms, "generate_blocking: TTFT");
                    // Append to the rolling ring-buffer so non-streaming requests
                    // populate the same ring as streaming requests.
                    {
                        let mut ring = ttft_store.lock();
                        if ring.len() >= TTFT_RING_CAPACITY {
                            ring.pop_front();
                        }
                        ring.push_back(TtftSample {
                            model_id: model_id.to_owned(),
                            ttft_ms,
                        });
                    }
                    // phase transition Prefill -> Decode at the natural
                    // TTFT boundary (first OK token). Same timestamp as the
                    // existing TTFT capture — no second `Instant::now()`.
                    tracing::debug!(
                        model_id,
                        phase = ?crate::engine::Phase::Decode,
                        ttft_ms,
                        "phase transition Prefill -> Decode"
                    );
                    if let Some(rec) = event_recorder.clone() {
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
                }
                let in_answer = !tok.is_thinking || bare_json_tool_call_mode;
                if let (Some(reply), true) = (json_reply.as_mut(), in_answer) {
                    reply.receive(completion_tokens + 1, &tok.piece);
                }
                // collect this token's logprob record (present only when
                // `logprobs:true` was requested). An answer token before a
                // `response_format` reply starts has no byte in the content.
                let before_the_reply = in_answer
                    && !bare_json_tool_call_mode
                    && json_reply.as_ref().is_some_and(|r| r.start().is_none());
                if let (Some(lp), false) = (tok.logprobs.clone(), before_the_reply) {
                    logprobs_accum.push(lp);
                }
                // Feed the parser regardless of think state.
                //
                // A reasoning model whose prompt left a `<think>` open can
                // emit the tool call WITHOUT first closing `</think>` — every
                // piece is then `is_thinking == true`. Routing thinking pieces straight
                // to `reasoning_text` (the old behaviour) meant the parser
                // never saw the `<tool_call>` block and no `tool_calls` were
                // produced. Feeding the parser in both states extracts the
                // call wherever it appears; the parser's passthrough is then
                // routed to the channel matching the token's think state, so
                // genuine reasoning text still lands in `reasoning_content`
                // and Qwen3.6 (which DOES close `</think>` before the XML
                // call) is unaffected — its pre-`</think>` text is
                // passthrough-while-thinking → `reasoning_text`, identical
                // to before.
                match parser.as_mut() {
                    Some(p) => {
                        if !tok.piece.is_empty() {
                            p.push(&tok.piece);
                            if !p.passthrough_text.is_empty() {
                                if tok.is_thinking {
                                    reasoning_text.push_str(&p.passthrough_text);
                                } else {
                                    text.push_str(&p.passthrough_text);
                                }
                                p.passthrough_text.clear();
                            }
                        }
                    }
                    None => {
                        // A constrained forced tool call goes to `text`
                        // regardless of is_thinking: when the prompt left a
                        // `<think>` open, the JSON the constraint forced is
                        // emitted while is_thinking == true.
                        if tok.is_thinking && !(bare_json_tool_call_mode && json_reply.is_some()) {
                            reasoning_text.push_str(&tok.piece);
                        } else {
                            text.push_str(&tok.piece);
                        }
                    }
                }
                completion_tokens += 1;
                if tok.done {
                    finish_reason = tok.finish_reason;
                    break;
                }
            }
        }
    }

    // Drain residual passthrough + completed tool_calls from the
    // parser. Call `finalize` first (flips allow_eof_recovery=true) so that a
    // truncated Bonsai/Hermes `<tool_call>{json(unclosed)` at max_tokens/EOS
    // is balanced and recovered before draining.
    if let Some(p) = parser.as_mut() {
        p.finalize();
        if !p.passthrough_text.is_empty() {
            text.push_str(&p.passthrough_text);
            p.passthrough_text.clear();
        }
        tool_calls_accum.extend(p.take_parsed());
    }

    // Truncate the accumulated content at the first stop-sequence
    // boundary (stop string excluded) and force finish_reason="stop". Matching
    // is on the detokenized text, so a stop that straddled token boundaries is
    // handled. Only applies when no tool_calls were produced — a tool-call
    // response carries no free-text content to truncate.
    if tool_calls_accum.is_empty() {
        if let Some(hit) = crate::stop_matcher::find_stop_match(&text, &stop_sequences) {
            tracing::debug!(
                model_id,
                stop = %hit.matched,
                offset = hit.offset,
                "truncated content at stop sequence (non-streaming)"
            );
            text.truncate(hit.offset);
            finish_reason = Some("stop".to_owned());
        }
    }

    // A `response_format` reply is the returned text from the engagement byte
    // on. Text that ends before that byte was never checked: refuse it.
    if let (Some(reply), false, true) = (
        json_reply.as_mut(),
        bare_json_tool_call_mode,
        tool_calls_accum.is_empty(),
    ) {
        let dropped_bytes = reply.release(&text);
        if !reply.engaged() {
            tracing::warn!(
                model_id,
                request_id,
                returned_bytes = text.len(),
                "refusing: the returned text ends before the `response_format` grammar engaged"
            );
            error_counts.increment(ApiErrorCategory::Upstream);
            requests_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return error_response(
                StatusCode::BAD_GATEWAY,
                NOT_ENGAGED_TYPE,
                NOT_ENGAGED_MESSAGE,
            );
        }
        tracing::debug!(
            model_id,
            dropped_bytes,
            "text before the engagement byte is not in the `response_format` reply"
        );
        text.drain(..dropped_bytes);
    }

    // Emit TTFT + token counts to SQLite via the SPSC drainer.
    // Single-source: same counters that populate the `usage` response body.
    // One emit per metric per request — no double-emit possible.
    if let Some(drainer) = metrics_drainer {
        use crate::metrics_drainer::{MetricEvent, MetricKind};
        let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
        if let Some(ttft_ms) = ttft_ms_blocking {
            drainer.try_emit(MetricEvent {
                model_id: model_id.to_owned(),
                kv_quant: "none".into(),
                ts_utc: ts.clone(),
                ctx_max,
                kind: MetricKind::TtftMs(ttft_ms),
            });
        }
        drainer.try_emit(MetricEvent {
            model_id: model_id.to_owned(),
            kv_quant: "none".into(),
            ts_utc: ts.clone(),
            ctx_max,
            kind: MetricKind::PromptTokens(prompt_token_count),
        });
        drainer.try_emit(MetricEvent {
            model_id: model_id.to_owned(),
            kv_quant: "none".into(),
            ts_utc: ts,
            ctx_max,
            kind: MetricKind::CompletionTokens(completion_tokens),
        });
    }
    // TTFT events-table write hoisted to first-token time (finding #3).
    // Increment process-lifetime token counters (single source: same
    // values as the SPSC drainer emit above; no double-count possible).
    tokens_in.fetch_add(
        u64::from(prompt_token_count),
        std::sync::atomic::Ordering::Relaxed,
    );
    tokens_out.fetch_add(
        u64::from(completion_tokens),
        std::sync::atomic::Ordering::Relaxed,
    );
    // count successfully completed non-streaming requests.
    requests_completed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    // Record StepMetrics for the adaptive admission controller.
    // One observation per completed request; gated on the handle being present.
    if let Some((ctrl, queue_depth, queue_wait_ms)) = admission_ctrl {
        let step_ms = request_start.elapsed().as_millis() as u64;
        let decode_kv_bytes = generator_for_metrics.kv_cache_bytes();
        ctrl.record_step(&crate::admission::StepMetrics {
            prompt_tokens: u64::from(prompt_token_count),
            decode_kv_bytes,
            queue_depth,
            queue_wait_ms,
            step_ms,
        });
        tracing::debug!(
            prompt_tokens = prompt_token_count,
            decode_kv_bytes,
            step_ms,
            queue_depth,
            queue_wait_ms,
            "StepMetrics recorded (blocking path)"
        );
    }

    let created = unix_now();
    let reasoning_content = if reasoning_text.is_empty() {
        None
    } else {
        Some(reasoning_text)
    };

    // A forced tool call with no marked call: the text is bare
    // `{"name":"…","arguments":{…}}` JSON. Convert it, from the engagement
    // byte on when a constraint reported one, to a ParsedToolCall so the
    // response envelope has `tool_calls` and `content=""` (not raw JSON).
    let (text, tool_calls_accum) = if bare_json_tool_call_mode && tool_calls_accum.is_empty() {
        let json_str = match json_reply.as_mut() {
            Some(reply) => &text[reply.release(&text)..],
            None => &text,
        };
        if let Some(tc) = bare_json_to_tool_call(json_str) {
            tracing::debug!(
                name = %tc.name,
                "bare_json_tool_call_mode: synthesised tool call from the bare JSON of a forced tool call"
            );
            (String::new(), vec![tc])
        } else {
            tracing::warn!(
                json = %json_str,
                "bare_json_tool_call_mode: a forced tool call did not parse; returning as content"
            );
            (json_str.to_owned(), tool_calls_accum)
        }
    } else {
        (text, tool_calls_accum)
    };

    let tool_calls_out = if tool_calls_accum.is_empty() {
        None
    } else {
        Some(
            tool_calls_accum
                .iter()
                .enumerate()
                .map(|(i, p)| to_response_tool_call(p, i as u32))
                .collect(),
        )
    };
    let any_tool_calls = tool_calls_out.is_some();
    let finish_reason = select_finish_reason(any_tool_calls, finish_reason);
    // `id` uses the correlation id resolved at handler entry so that the
    // response body and the X-Request-Id header always agree.
    // emit `choices[0].logprobs` only when at least one token carried a
    // logprob record (i.e. the request set `logprobs:true`).
    let logprobs = if logprobs_accum.is_empty() {
        None
    } else {
        Some(ChatLogprobs {
            content: logprobs_accum,
        })
    };
    let response = ChatCompletionsResponse {
        id: format!("chatcmpl-{request_id}"),
        object: "chat.completion".to_owned(),
        created,
        model: model_id.to_owned(),
        choices: vec![Choice {
            index: 0,
            message: ResponseMessage {
                role: "assistant".to_owned(),
                content: text,
                reasoning_content,
                tool_calls: tool_calls_out,
            },
            finish_reason,
            logprobs,
        }],
        usage: Usage {
            prompt_tokens: prompt_token_count,
            completion_tokens,
            total_tokens: prompt_token_count + completion_tokens,
        },
    };

    // Echo the correlation id as a response header.
    let mut resp = (StatusCode::OK, Json(response)).into_response();
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        resp.headers_mut().insert("x-request-id", hv);
    }
    resp
}

// ── Streaming path ────────────────────────────────────────────────────────────

/// Streaming generation path — returns 503 before opening SSE if the
/// generator is not ready.
///
/// Each token from the generator is forwarded to SSE immediately.
/// axum flushes one SSE event per `Stream::poll_next`, so clients
/// see tokens trickling in rather than arriving in one network burst.
///
/// `request_start` is the `Instant` captured at handler entry. TTFT is
/// computed as `request_start.elapsed()` when the first token arrives from the
/// decode thread — before any SSE serialisation overhead.
///
/// `include_usage` / `prompt_token_count` support the usage-summary
/// chunk emitted before `[DONE]` when `stream_options.include_usage=true`.
#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::fn_params_excessive_bools,
    reason = "generate_streaming takes several boolean mode flags (bare_json_tool_call_mode, include_usage, is_cold_request) that are structurally distinct; a refactor to a request-opts struct is deferred to a follow-up"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(super) async fn generate_streaming(
    generator: Arc<dyn Generator>,
    req: GenerationRequest,
    // when Some, the token stream is wrapped in the replay envelope.
    replay_plan: Option<crate::retry::RequestPlan>,
    model_id: &str,
    request_start: Instant,
    state: &AppState,
    parser_format: Option<ToolCallFormat>,
    // Where the request's constraint engaged; see `generate_blocking`.
    engagement: Option<Arc<rmlx_models::Engagement>>,
    // True when tool_choice=required/named drives the constraint.
    bare_json_tool_call_mode: bool,
    include_usage: bool,
    prompt_token_count: u32,
    // Server effective_max_ctx for drainer MetricEvent.ctx_max.
    ctx_max_for_metrics: i64,
    // Correlation id resolved at handler entry.
    request_id: &str,
    // cold/warm flag for TTFT metric name selection.
    is_cold_request: bool,
    // Queue metrics captured at admission time for StepMetrics recording.
    // (queue_depth, queue_wait_ms) — used when state.admission_controller is Some.
    queue_at_admission: (u64, u64),
    // Per-request decode-lease guard owned for the lifetime of the
    // SSE stream. Held in StreamState so the guard drops when the stream is
    // fully consumed (or the client connection closes).
    decode_lease: Option<crate::keep_alive::DecodeLeaseGuard>,
) -> Response {
    // Capture stop sequences before `req` is moved into the generator.
    let stop_sequences = req.stop.clone();
    // Peek at the first item. If it's an error, return 503 immediately
    // (before any SSE bytes are written to the connection).
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
            // TTFT captured immediately when the first token arrives from
            // the decode thread. This is the moment the first generated token
            // leaves the blocking decode loop and reaches the async layer —
            // before SSE serialisation, JSON encoding, or TCP flush.
            let ttft_ms = request_start.elapsed().as_millis() as u64;
            tracing::info!(model_id, ttft_ms, "generate_streaming: TTFT");
            // Append to the rolling ring-buffer; evict oldest when full.
            {
                let mut ring = state.ttft_store.lock();
                if ring.len() >= TTFT_RING_CAPACITY {
                    ring.pop_front();
                }
                ring.push_back(TtftSample {
                    model_id: model_id.to_owned(),
                    ttft_ms,
                });
            }
            // Emit TtftMs to SQLite via SPSC drainer (single emit per
            // request; kv_quant="none" matches queue-metric convention at
            // handler level — engine already emits kv-aware KV/ITL metrics).
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
            // phase transition Prefill -> Decode at the same boundary;
            // also write `prefill_duration_ms` (same value, distinct op name)
            // in the same spawn_blocking closure for one round-trip cost.
            tracing::debug!(
                model_id,
                phase = ?crate::engine::Phase::Decode,
                ttft_ms,
                "phase transition Prefill -> Decode (streaming)"
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

    let created = unix_now();
    // Use the correlation id resolved at handler entry so that the id
    // embedded in every SSE chunk matches the X-Request-Id response header.
    let id = format!("chatcmpl-{request_id}");
    let model = model_id.to_owned();

    // ── Role-only preamble chunk ──────────────────────────────────────────────
    let role_chunk = ChatCompletionChunk {
        id: id.clone(),
        object: "chat.completion.chunk".to_owned(),
        created,
        model: model.clone(),
        choices: vec![StreamChoice {
            index: 0,
            delta: DeltaContent {
                role: Some("assistant".to_owned()),
                content: None,
                reasoning_content: None,
                tool_calls: None,
            },
            finish_reason: None,
            logprobs: None,
        }],
        usage: None,
    };
    let role_event: Result<Event, std::convert::Infallible> =
        Ok(Event::default().data(serde_json::to_string(&role_chunk).unwrap_or_default()));

    // ── Lazy token stream → SSE event stream ─────────────────────────────────
    // Re-attach the peeked first token (if any) to the front of the remaining
    // stream, then map each token to zero or more SSE events. This avoids
    // collecting all tokens into a Vec before yielding — axum flushes each
    // event as soon as the stream yields it.
    //
    // Parser is threaded through the stream via `unfold`, which gives
    // us mutable state without `Mutex`. Each input token may yield:
    // - 0 events (e.g. all bytes still buffered inside `<tool_call>` markers),
    // - 1 content / reasoning_content event (existing behaviour),
    // - 1+ tool_call events (one per completed `</tool_call>`),
    // - or a mix (passthrough text before a tool_call, plus the call itself).
    // The terminal `done` token always emits a final chunk carrying
    // `finish_reason` (upgraded to `"tool_calls"` if any calls were emitted).
    let token_events = {
        // Build an iterator of exactly 0 or 1 first tokens, chained with the rest.
        let first_stream: futures::stream::BoxStream<'static, rmlx_core::Result<GenerationToken>> =
            match first_ok {
                None => futures::stream::empty().boxed(),
                Some(tok) => futures::stream::once(async move { Ok(tok) })
                    .chain(token_stream)
                    .boxed(),
            };

        // State threaded across the stream: parser (Some when enabled),
        // next tool-call index, whether any tool_calls have been emitted.
        let parser_init: Option<ToolCallStreamParser> =
            parser_format.map(ToolCallStreamParser::new);
        let init_state = StreamState {
            parser: parser_init,
            next_tool_index: 0,
            any_tool_calls: false,
            id,
            model,
            created,
            json_reply: engagement.map(JsonReply::new),
            prompt_tokens: prompt_token_count,
            completion_tokens: 0,
            include_usage,
            bare_json_tool_call_mode,
            bare_json_accum: String::new(),
            // Drainer handle + context for per-request token-count emit.
            metrics_drainer: state.metrics_drainer.clone(),
            metrics_model_id: model_id.to_owned(),
            metrics_ctx_max: ctx_max_for_metrics,
            // Share the process-lifetime counters from AppState.
            lifetime_tokens_in: Arc::clone(&state.tokens_in),
            lifetime_tokens_out: Arc::clone(&state.tokens_out),
            // Share per-category error counters from AppState.
            error_counts: state.error_counts.clone(),
            // share request lifecycle counters from AppState.
            lifetime_requests_completed: Arc::clone(&state.requests_completed),
            lifetime_requests_failed: Arc::clone(&state.requests_failed),
            // Adaptive controller handle + admission metadata.
            // kv_cache_bytes is unavailable in streaming (generator Arc consumed);
            // record_step is skipped and only a debug trace is emitted.
            admission_ctrl: state
                .admission_controller
                .as_ref()
                .map(|ctrl| (ctrl.clone(), queue_at_admission.0, queue_at_admission.1)),
            // Content-channel stop-sequence matcher (inert when empty).
            stop_matcher: crate::stop_matcher::StopMatcher::new(&stop_sequences),
            stop_hit: false,
        };

        futures::stream::unfold(
            (first_stream, init_state),
            |(mut s, mut state)| async move {
                let item = s.next().await?;
                let events = handle_streaming_token(item, &mut state);
                Some((futures::stream::iter(events), (s, state)))
            },
        )
        .flatten()
    };

    // ── [DONE] sentinel ───────────────────────────────────────────────────────
    let done_event: Result<Event, std::convert::Infallible> = Ok(Event::default().data("[DONE]"));

    // Compose: role preamble → token events → [DONE]
    let sse_stream = futures::stream::once(async move { role_event })
        .chain(token_events)
        .chain(futures::stream::once(async move { done_event }));

    // Add the correlation id as a response header on the SSE response.
    // Wrap the composed SSE stream in a GuardedStream so the
    // decode-lease guard drops when the stream is fully consumed or the
    // client disconnects. Box::pin the composed Chain<Chain<...>> so the
    // wrapper meets its `Unpin` bound.
    let boxed: std::pin::Pin<
        Box<dyn futures::stream::Stream<Item = Result<Event, std::convert::Infallible>> + Send>,
    > = Box::pin(sse_stream);
    let guarded = crate::keep_alive::GuardedStream::new(boxed, decode_lease);
    let mut resp = Sse::new(guarded).into_response();
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        resp.headers_mut().insert("x-request-id", hv);
    }
    resp
}
