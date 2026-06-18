//! Non-streaming generation path for the Anthropic Messages API.

#![allow(clippy::too_many_lines)]

use std::sync::Arc;
use std::time::Instant;

use axum::http::{HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;

use crate::engine::{GenerationRequest, GenerationToken};
use crate::openai::errors::engine_error_category;
use crate::tool_parser::{ParsedToolCall, ToolCallFormat, ToolCallStreamParser};

use super::errors::engine_error_response;
use super::response::{AnthropicUsage, ContentBlock, MessagesResponse};
use super::route::{map_stop_reason, select_anthropic_stop_reason, to_tool_use_block};

// ── Non-streaming path ────────────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
#[allow(
    clippy::cognitive_complexity,
    reason = "streaming token loop + tool-call accumulation + TTFT/finish-reason branching — inherent to the non-streaming path"
)]
pub(super) async fn generate_blocking(
    generator: Arc<dyn crate::engine::Generator>,
    req: GenerationRequest,
    // when Some, the token stream is wrapped in the replay envelope.
    replay_plan: Option<crate::retry::RequestPlan>,
    model_id: &str,
    parser_format: Option<ToolCallFormat>,
    // F1: drainer handle + ctx_max for TTFT/token-count DB emission.
    request_start: Instant,
    metrics_drainer: Option<&crate::metrics_drainer::DrainerHandle>,
    ctx_max: i64,
    // F14: process-lifetime token counters shared with AppState.
    tokens_in: &Arc<std::sync::atomic::AtomicU64>,
    tokens_out: &Arc<std::sync::atomic::AtomicU64>,
    // F10: correlation id resolved at handler entry.
    request_id: &str,
    // F8: per-category error counters shared with AppState.
    error_counts: &crate::openai::ApiErrorCounters,
    // request lifecycle counters.
    requests_completed: &Arc<std::sync::atomic::AtomicU64>,
    requests_failed: &Arc<std::sync::atomic::AtomicU64>,
    // per-event DB recorder for TTFT writes to the events table.
    event_recorder: Option<Arc<rmlx_metrics::events::EventRecorder>>,
    // cold/warm flag for TTFT metric name selection.
    is_cold_request: bool,
    // Optional adaptive controller handle for StepMetrics recording.
    admission_ctrl: Option<(
        crate::admission::ControllerHandle,
        u64, // queue_depth at admission
        u64, // queue_wait_ms
    )>,
    // Rolling ring-buffer for TTFT samples shared with AppState.
    // Written here so non-streaming requests populate the same ring as streaming.
    ttft_store: &crate::openai::state::TtftStore,
) -> Response {
    let input_token_count = req.prompt_tokens.len() as u32;
    // Capture stop sequences before `req` is moved into the generator.
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
    // A3: accumulate reasoning text into a separate buffer; it becomes
    // a leading `thinking` content block in the response when non-empty.
    let mut thinking = String::new();
    let mut finish_reason: Option<String> = None;
    let mut output_tokens: u32 = 0;
    // F1a: TTFT for non-streaming — measured when the first token arrives.
    let mut ttft_ms_blocking: Option<u64> = None;
    // A5.5: instantiate parser when caller supplied a format. `None` means
    // tools-disabled — every non-thinking piece flows straight to `text`.
    let mut parser: Option<ToolCallStreamParser> = parser_format.map(ToolCallStreamParser::new);
    let mut tool_calls_accum: Vec<ParsedToolCall> = Vec::new();

    while let Some(item) = token_stream.next().await {
        match item {
            Err(e) => {
                error_counts.increment(engine_error_category(&e));
                // count engine errors as failed requests.
                requests_failed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return engine_error_response(&e);
            }
            Ok(tok) => {
                // F1a + : capture TTFT on the very first token and
                // immediately persist to the events table off the tokio worker
                // so TTFT survives mid-stream errors.
                if output_tokens == 0 {
                    let ttft_ms = request_start.elapsed().as_millis() as u64;
                    ttft_ms_blocking = Some(ttft_ms);
                    tracing::info!(model_id, ttft_ms, "generate_blocking (anthropic): TTFT");
                    // Append to the rolling ring-buffer so non-streaming requests
                    // populate the same ring as streaming requests.
                    {
                        use crate::openai::{TtftSample, TTFT_RING_CAPACITY};
                        let mut ring = ttft_store.lock();
                        if ring.len() >= TTFT_RING_CAPACITY {
                            ring.pop_front();
                        }
                        ring.push_back(TtftSample {
                            model_id: model_id.to_owned(),
                            ttft_ms,
                        });
                    }
                    tracing::debug!(
                        model_id,
                        phase = ?crate::engine::Phase::Decode,
                        ttft_ms,
                        "phase transition Prefill -> Decode (anthropic non-stream)"
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
                // A5.5 / A5.6: feed the parser regardless of think state —
                // some reasoning models emit the tool call without closing
                // `</think>`, so a thinking-only routing would never let the
                // parser see the `<tool_call>` block. Parser passthrough is
                // routed to the channel matching the token's think state, so
                // genuine reasoning still lands in the `thinking` buffer and
                // Qwen3.6 (which closes `</think>` first) is unaffected.
                match parser.as_mut() {
                    Some(p) => {
                        if !tok.piece.is_empty() {
                            p.push(&tok.piece);
                            if !p.passthrough_text.is_empty() {
                                if tok.is_thinking {
                                    thinking.push_str(&p.passthrough_text);
                                } else {
                                    text.push_str(&p.passthrough_text);
                                }
                                p.passthrough_text.clear();
                            }
                        }
                    }
                    None => {
                        if tok.is_thinking {
                            thinking.push_str(&tok.piece);
                        } else {
                            text.push_str(&tok.piece);
                        }
                    }
                }
                output_tokens += 1;
                if tok.done {
                    finish_reason = tok.finish_reason;
                    break;
                }
            }
        }
    }

    // A5.5: drain residual passthrough + completed tool_calls from the parser.
    if let Some(p) = parser.as_mut() {
        if !p.passthrough_text.is_empty() {
            text.push_str(&p.passthrough_text);
            p.passthrough_text.clear();
        }
        tool_calls_accum.extend(p.take_parsed());
    }

    // Truncate the text block at the first stop-sequence boundary
    // (stop string excluded). On a hit, force stop_reason="stop_sequence" and
    // populate the `stop_sequence` field naming which stop matched. Only when
    // no tool_use block was produced (tool responses carry no free text).
    let matched_stop: Option<String> = if tool_calls_accum.is_empty() {
        if let Some(hit) = crate::stop_matcher::find_stop_match(&text, &stop_sequences) {
            tracing::debug!(
                model_id,
                stop = %hit.matched,
                offset = hit.offset,
                "truncated text block at stop sequence (anthropic non-streaming)"
            );
            text.truncate(hit.offset);
            finish_reason = Some("stop".to_owned());
            Some(hit.matched)
        } else {
            None
        }
    } else {
        None
    };

    // F1: emit TTFT + token counts to SQLite via the SPSC drainer.
    // Single-source: same counters that populate the Anthropic `usage` body.
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
            kind: MetricKind::PromptTokens(input_token_count),
        });
        drainer.try_emit(MetricEvent {
            model_id: model_id.to_owned(),
            kv_quant: "none".into(),
            ts_utc: ts,
            ctx_max,
            kind: MetricKind::CompletionTokens(output_tokens),
        });
    }
    // F14: increment process-lifetime token counters (single source: same
    // values as the SPSC drainer emit above; no double-count possible).
    tokens_in.fetch_add(
        u64::from(input_token_count),
        std::sync::atomic::Ordering::Relaxed,
    );
    tokens_out.fetch_add(
        u64::from(output_tokens),
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
            prompt_tokens: u64::from(input_token_count),
            decode_kv_bytes,
            queue_depth,
            queue_wait_ms,
            step_ms,
        });
        tracing::debug!(
            prompt_tokens = input_token_count,
            decode_kv_bytes,
            step_ms,
            queue_depth,
            queue_wait_ms,
            "StepMetrics recorded (anthropic blocking path)"
        );
    }

    // When a stop-sequence matched (stop-matcher path), use "stop_sequence"
    // directly — map_stop_reason must not be consulted for that branch because
    // it correctly returns "end_turn" for engine finish_reason="stop" (natural
    // EOS), not "stop_sequence". The stop_sequence field is populated from
    // matched_stop below.
    let any_tool_use = !tool_calls_accum.is_empty();
    let stop_reason = if matched_stop.is_some() {
        // Real stop-string match; tool_use cannot co-occur (matched_stop is
        // set only when tool_calls_accum is empty — see guard above).
        "stop_sequence".to_owned()
    } else {
        let terminal = map_stop_reason(finish_reason.as_deref());
        select_anthropic_stop_reason(any_tool_use, terminal)
    };
    // F10: use the correlation id resolved at handler entry so that the
    // response body and the X-Request-Id header always agree.
    let id = format!("msg_{request_id}");

    // A3+A5.5: emit thinking block first (if any), then text block (if any),
    // then a tool_use block per parsed call. The text block is suppressed
    // when empty AND there is at least one tool_use block, so a pure
    // tool-call response is not padded with an empty `text` block.
    let mut content: Vec<ContentBlock> = Vec::with_capacity(2 + tool_calls_accum.len());
    if !thinking.is_empty() {
        content.push(ContentBlock::Thinking { thinking });
    }
    if !text.is_empty() || !any_tool_use {
        // Preserve pre-A5.5 behaviour: when no tool_use is present, always
        // emit the (possibly empty) text block. With tool_use, only emit it
        // when non-empty.
        content.push(ContentBlock::Text { text });
    }
    for parsed in &tool_calls_accum {
        content.push(to_tool_use_block(parsed));
    }

    let response = MessagesResponse {
        id,
        kind: "message".to_owned(),
        role: "assistant".to_owned(),
        model: model_id.to_owned(),
        content,
        stop_reason: Some(stop_reason),
        // Name the matched stop string per Anthropic spec; `None` when
        // generation ended for any other reason (EOS / max_tokens / tool_use).
        stop_sequence: matched_stop,
        usage: AnthropicUsage {
            input_tokens: input_token_count,
            output_tokens,
        },
    };

    // F10: echo the correlation id as a response header.
    let mut resp = (StatusCode::OK, Json(response)).into_response();
    if let Ok(hv) = HeaderValue::from_str(request_id) {
        resp.headers_mut().insert("x-request-id", hv);
    }
    resp
}
