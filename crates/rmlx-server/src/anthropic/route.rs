//! POST /v1/messages route handler and response helpers.
//!
//! - `record_metric` — per-request metric emission helper
//! - `map_stop_reason` / `select_anthropic_stop_reason` / `to_tool_use_block` — stop-reason + tool helpers
//! - `messages` — `POST /v1/messages` entry point

#![allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::manual_let_else
)]

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use tracing::Instrument;

use rmlx_metrics::events::Measurement;

use crate::bounds;
use crate::chat_template::{ChatMessageTpl, RenderOpts};
use crate::engine::{
    normalized_to_jinja_tool, GenerationRequest, NormalizedTool, NormalizedToolChoice,
};
use crate::logged_json::LoggedJson;
use crate::openai::{
    enforce_max_tokens_cap, resolve_request_id, resolve_sampling_params, ApiErrorCategory, AppState,
};
use crate::session_cache::SessionKey;
use crate::tokenizer_io;
use crate::tool_parser::{detect_tool_call_format, ParsedToolCall, ToolCallFormat};

use super::blocking::generate_blocking;
use super::errors::{bad_request, error_response, service_unavailable};
use super::request::{AnthropicContent, MessagesRequest};
use super::response::ContentBlock;
use super::streaming::generate_streaming;

// A5.1: no fields in the Anthropic extra-rejection list — tools + tool_choice
// are now first-class parsed fields. Retain the section header so future
// fields (e.g. computer_use, A6-style response_format) have a home.

/// Emit one metric record; silently drops on error or absent sink.
pub(super) fn record_metric(
    state: &AppState,
    op: &str,
    unit: &str,
    value: f64,
    notes: &str,
    model_path: &str,
) {
    if let Some(sink) = &state.metrics {
        let m = Measurement {
            model_path,
            quant_mode: "n/a",
            stage: "stage1",
            op,
            value_unit: unit,
            value,
            notes,
        };
        if let Err(e) = sink.record(&m) {
            tracing::warn!(error = %e, "failed to record metric");
        }
    }
}

// ── stop_reason mapping ───────────────────────────────────────────────────────

/// Map a generator `finish_reason` to an Anthropic `stop_reason`.
///
/// - `"length"` (token cap hit) → `"max_tokens"`
/// - `"tool_calls"` (engine tool-call finish) → `"tool_use"`
/// - `"stop"` (explicit EOS) and `None` (unmarked clean terminal) → `"end_turn"`
/// - any other / unrecognised finish reason → `"error"`
///
/// The unrecognised branch must **never** collapse to `"end_turn"`: an unknown
/// or future finish reason (e.g. an engine-emitted `"error"` terminal) reported
/// as a normal successful completion is a masked failure. Only reasons this
/// mapper has an explicit success contract for become a successful stop; every
/// other terminal surfaces as an explicit `"error"` stop reason a client or
/// harness can detect.
///
/// Note: `"stop_sequence"` is NOT produced here. A real stop-string match
/// is detected by the stop-matcher path in `blocking.rs` / `streaming.rs`, which
/// sets `stop_reason` / `stop_sequence` explicitly and bypasses this
/// function for that branch.
pub(crate) fn map_stop_reason(finish_reason: Option<&str>) -> String {
    match finish_reason {
        Some("length") => "max_tokens".to_owned(),
        Some("tool_calls") => "tool_use".to_owned(),
        // Natural end of turn: an explicit EOS ("stop") or an unmarked clean
        // terminal (None) are both legitimate successful completions.
        Some("stop") | None => "end_turn".to_owned(),
        // A finish reason this mapper has no success contract for. Do not
        // launder it into "end_turn"; surface it as an explicit error stop
        // reason so a failed/unexpected terminal is distinguishable from a
        // genuine completion.
        Some(other) => {
            tracing::error!(
                finish_reason = other,
                "map_stop_reason: unrecognised finish reason; reporting error stop reason"
            );
            "error".to_owned()
        }
    }
}

/// A5.5: select the wire `stop_reason` after consuming the full token stream.
///
/// Per the Anthropic spec, when any tool_use block was emitted, the response
/// carries `stop_reason="tool_use"` regardless of the natural model finish
/// (which would typically map to `"end_turn"` / `"max_tokens"` / `"stop_sequence"`).
pub(crate) fn select_anthropic_stop_reason(any_tool_use: bool, terminal: String) -> String {
    if any_tool_use {
        "tool_use".to_owned()
    } else {
        terminal
    }
}

/// A5.5: convert a `ParsedToolCall` into the Anthropic `tool_use` content block.
///
/// Note: Anthropic's `input` is a JSON object, NOT a JSON-stringified string
/// (that's the OpenAI `arguments` shape).
pub(crate) fn to_tool_use_block(p: &ParsedToolCall) -> ContentBlock {
    ContentBlock::ToolUse {
        id: p.id.clone(),
        name: p.name.clone(),
        input: Value::Object(p.arguments.clone()),
    }
}

// ── Route: POST /v1/messages ──────────────────────────────────────────────────

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(crate) async fn messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    LoggedJson(req): LoggedJson<MessagesRequest>,
) -> Response {
    // L6: wall-clock anchor for TTFT. Captured immediately after header parse.
    let request_start = Instant::now();

    // F10: resolve correlation id; the span is attached to each generate call
    // via .instrument() so all tracing inside generate_blocking / generate_streaming
    // automatically carry request_id.
    let rid = resolve_request_id(&headers);
    let req_span = tracing::info_span!("request", request_id = %rid, route = "messages");
    tracing::info!(parent: &req_span, "messages: incoming request");

    // Debug-log unknown extra fields (tools + tool_choice now parsed above the
    // flatten catch-all, so they won't appear here).
    for key in req.extra.keys() {
        tracing::debug!(field = %key, "messages: ignoring unknown request field");
    }
    if let Some(meta) = &req.metadata {
        tracing::debug!(metadata = %meta, "messages: ignoring metadata");
    }

    // Validate.
    if req.max_tokens == 0 {
        state.error_counts.increment(ApiErrorCategory::BadRequest);
        return bad_request("max_tokens must be > 0");
    }
    if let Some(t) = req.temperature {
        if !(0.0..=1.0).contains(&t) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("temperature must be in [0.0, 1.0]");
        }
    }
    if let Some(p) = req.top_p {
        if !(0.0..=1.0).contains(&p) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("top_p must be in [0.0, 1.0]");
        }
    }
    if let Some(k) = req.top_k {
        if k == 0 {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("top_k must be >= 1");
        }
    }
    if req.messages.is_empty() {
        state.error_counts.increment(ApiErrorCategory::BadRequest);
        return bad_request("messages must not be empty");
    }

    // Bounded-input checks — fail FAST before any allocation.
    if let Err(e) = bounds::check_messages(req.messages.len()) {
        state.error_counts.increment(ApiErrorCategory::BadRequest);
        return e.into_response();
    }
    if let Some(ref tools) = req.tools {
        if let Err(e) = bounds::check_tools(tools.len()) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return e.into_response();
        }
    }
    // Per-message checks: content_parts, audio bytes, total token estimate.
    {
        let mut total_text_bytes: usize = 0;
        for (idx, msg) in req.messages.iter().enumerate() {
            match &msg.content {
                AnthropicContent::Blocks(blocks) => {
                    if let Err(e) = bounds::check_content_parts(blocks.len(), idx) {
                        state.error_counts.increment(ApiErrorCategory::BadRequest);
                        return e.into_response();
                    }
                    for (pidx, block) in blocks.iter().enumerate() {
                        // Anthropic audio: {type:"input_audio", source:{type:"base64",data:"..."}}
                        if block.get("type").and_then(|t| t.as_str()) == Some("input_audio") {
                            if let Some(b64) = block
                                .get("source")
                                .and_then(|s| s.get("data"))
                                .and_then(|d| d.as_str())
                            {
                                let decoded_estimate = b64.len() * 3 / 4;
                                if let Err(e) =
                                    bounds::check_input_audio_bytes(decoded_estimate, pidx)
                                {
                                    state.error_counts.increment(ApiErrorCategory::BadRequest);
                                    return e.into_response();
                                }
                            }
                        }
                        if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                            if let Some(s) = block.get("text").and_then(|t| t.as_str()) {
                                total_text_bytes = total_text_bytes.saturating_add(s.len());
                            }
                        }
                    }
                }
                AnthropicContent::Text(s) => {
                    total_text_bytes = total_text_bytes.saturating_add(s.len());
                }
            }
        }
        if let Err(e) = bounds::check_total_input_tokens_estimate(total_text_bytes) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return e.into_response();
        }
    }

    let system: Option<String> = req.system.as_ref().map(|s| s.as_text().into_owned());

    // A5.1: normalise Anthropic tools → route-agnostic NormalizedTool.
    // Moved before the prompt pipeline so `jinja_tools` is available for
    // injection into RenderOpts (A5.2). Empty array treated same as absent.
    let norm_tools: Option<Vec<NormalizedTool>> =
        req.tools.filter(|v| !v.is_empty()).map(|tools| {
            tools
                .into_iter()
                .map(|t| NormalizedTool {
                    name: t.name,
                    description: t.description,
                    schema: t.input_schema,
                })
                .collect()
        });

    // Anthropic tool_choice is always an object; map to normalised enum.
    // Unknown kind is rejected with 400 as it indicates a client programming error.
    let norm_tool_choice: Option<NormalizedToolChoice> = match req.tool_choice {
        None => None,
        Some(tc) => match tc.kind.as_str() {
            "auto" => Some(NormalizedToolChoice::Auto),
            "any" => Some(NormalizedToolChoice::Required),
            "tool" => {
                let name = if let Some(n) = tc.name {
                    n
                } else {
                    state.error_counts.increment(ApiErrorCategory::BadRequest);
                    return bad_request("tool_choice with type=tool requires a name field");
                };
                Some(NormalizedToolChoice::Named(name))
            }
            other => {
                state.error_counts.increment(ApiErrorCategory::BadRequest);
                return bad_request(&format!(
                    "unsupported tool_choice type: {other:?}; expected auto, any, or tool"
                ));
            }
        },
    };

    // A5.2: convert normalised tools to OpenAI-shaped JSON values for Jinja.
    // Note: tool_choice:"none" / Anthropic does not have a "none" variant, so
    // tools are always injected when present. Hard suppression is a v1.x item.
    let jinja_tools: Vec<Value> = norm_tools
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(normalized_to_jinja_tool)
        .collect();

    if !jinja_tools.is_empty() {
        tracing::debug!(
            model_id = %req.model,
            tool_count = jinja_tools.len(),
            tool_names = ?jinja_tools.iter().filter_map(|v| v["function"]["name"].as_str()).collect::<Vec<_>>(),
            "chat_template: rendering with tools"
        );
    }

    // A5.5: resolve the tool-call output format from the model's
    // chat_template source (the same arch emits different tool conventions
    // per snapshot), falling back to the coarse arch map. Parser is
    // instantiated downstream only when tools are present AND a format is
    // resolved. Reads the registry entry — same entry is re-fetched by the
    // prompt-pipeline match block below; both are cheap.
    let tool_format: Option<ToolCallFormat> = state
        .registry
        .get(&req.model)
        .and_then(|e| detect_tool_call_format(e.chat_template_src.as_deref(), &e.arch));
    let tools_enabled = !jinja_tools.is_empty() && tool_format.is_some();
    if !jinja_tools.is_empty() && tool_format.is_none() {
        tracing::debug!(
            model_id = %req.model,
            "A5.5: tools requested but no parser for this arch; passthrough"
        );
    }

    // ── Prompt pipeline (S1.7) ────────────────────────────────────────────────
    let (prompt_tokens, prompt_think_open): (Vec<u32>, bool) = match state.registry.get(&req.model)
    {
        None => {
            state.error_counts.increment(ApiErrorCategory::NotFound);
            return error_response(
                StatusCode::NOT_FOUND,
                "not_found_error",
                &format!("model '{}' not found in registry", req.model),
            );
        }
        Some(entry) => {
            if let (Some(tpl), Some(tk)) = (&entry.chat_template, &entry.tokenizer) {
                // Build full message list (system message first if present).
                let mut all_msgs: Vec<(String, String)> = Vec::new();
                if let Some(sys) = &system {
                    all_msgs.push(("system".to_owned(), sys.clone()));
                }
                for m in &req.messages {
                    all_msgs.push((m.role.clone(), m.content.as_text().into_owned()));
                }

                let tpl_msgs: Vec<ChatMessageTpl<'_>> = all_msgs
                    .iter()
                    .map(|(role, content)| ChatMessageTpl {
                        role: role.as_str(),
                        content: content.as_str(),
                        ..Default::default()
                    })
                    .collect();

                let bos = entry.bos_token.as_deref().unwrap_or("");
                let eos = entry.eos_token.as_deref().unwrap_or("");
                // Anthropic route has no per-request enable_thinking;
                // use the server-startup default only.
                let opts = RenderOpts {
                    bos_token: bos,
                    eos_token: eos,
                    add_generation_prompt: true,
                    tools: &jinja_tools,
                    enable_thinking: state.default_enable_thinking,
                };

                let rendered = match tpl.render(&tpl_msgs, &opts) {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(model_id = %req.model, error = %e, "chat_template render failed");
                        state.error_counts.increment(ApiErrorCategory::Upstream);
                        return service_unavailable(&format!("chat template render failed: {e}"));
                    }
                };

                let ids = match tokenizer_io::encode(tk, &rendered.text) {
                    Ok(ids) => ids,
                    Err(e) => {
                        tracing::warn!(model_id = %req.model, error = %e, "tokenizer encode failed");
                        state.error_counts.increment(ApiErrorCategory::Upstream);
                        return service_unavailable(&format!("tokenizer encode failed: {e}"));
                    }
                };

                // Read the initial think channel off the prompt the model will
                // actually see. The Anthropic surface has no per-request
                // delimiter override, so the defaults apply.
                let think_open = crate::engine::think::prompt_leaves_think_open(
                    &rendered.text,
                    "<think>",
                    "</think>",
                );

                let token_count = ids.len();
                tracing::debug!(
                    model_id = %req.model,
                    prompt_token_count = token_count,
                    template_used = true,
                    prompt_think_open = think_open,
                    "prompt pipeline complete"
                );
                record_metric(
                    &state,
                    "prompt_tokens",
                    "count",
                    token_count as f64,
                    &req.model,
                    &entry.abs_path_str,
                );

                (ids, think_open)
            } else {
                tracing::debug!(
                    model_id = %req.model,
                    prompt_token_count = 0,
                    template_used = false,
                    "prompt pipeline skipped (missing chat_template or tokenizer)"
                );
                record_metric(
                    &state,
                    "prompt_pipeline_skip",
                    "count",
                    1.0,
                    &req.model,
                    &entry.abs_path_str,
                );
                state.error_counts.increment(ApiErrorCategory::Upstream);
                return service_unavailable(&format!(
                    "model '{}' is not ready (missing chat_template.jinja or tokenizer.json)",
                    req.model
                ));
            }
        }
    };
    // ─────────────────────────────────────────────────────────────────────────

    // A1: per-request `max_tokens` ceiling is now configurable via
    // `--max-tokens-cap` (default `u32::MAX` = no cap). Requests exceeding
    // the cap return HTTP 400 explicitly rather than being silently clamped.
    let max_tokens = match enforce_max_tokens_cap(req.max_tokens, state.max_tokens_cap, &req.model)
    {
        Ok(v) => v,
        Err(resp) => {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return *resp;
        }
    };

    // ── N2: session KV-reuse ───────────────────────────────────────────────────
    // Same logic as the OpenAI route: extract X-Session-Id, touch session cache,
    // compute effective prompt_cache_slots.
    let session_id: Option<String> = headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_owned);

    let effective_prompt_cache_slots: Option<usize> = if let Some(ref sid) = session_id {
        let key = SessionKey {
            model_id: req.model.clone(),
            session_id: sid.clone(),
        };
        let mut sc = state.session_cache.lock();
        let is_hit = sc.touch(key, prompt_tokens.len());
        // Same rule as the OpenAI route: widen the configured base, and leave a
        // base of 0 disabled — see `openai::chat`.
        let base = state.prompt_cache_slots;
        let effective = crate::session_cache::effective_prompt_cache_slots(base, sc.active_count());
        tracing::debug!(
            model_id = %req.model,
            session_id = %sid,
            is_hit,
            active_sessions = sc.active_count(),
            base_prompt_cache_slots = base,
            effective_prompt_cache_slots = effective,
            "messages: session cache lookup"
        );
        effective
    } else {
        None
    };
    // ─────────────────────────────────────────────────────────────────────────

    // A4/A7.1/G4: resolve all sampling params via request > server_default > model_defaults > hard_coded.
    // Anthropic API supports temperature, top_p, top_k only; the remaining
    // OpenAI-only knobs (min_p, penalties, logit_bias) default to neutral values.
    let gen_defaults = state
        .registry
        .get(&req.model)
        .and_then(|e| e.generation_defaults.as_ref().map(Arc::clone));
    let (sampling, temp_src, top_p_src) = resolve_sampling_params(
        req.temperature,
        req.top_p,
        req.top_k,
        None,       // min_p — Anthropic API does not expose this
        None,       // repetition_penalty — Anthropic API does not expose this
        None,       // frequency_penalty — Anthropic API does not expose this
        None,       // presence_penalty — Anthropic API does not expose this
        Vec::new(), // logit_bias — Anthropic API does not expose this
        None,       // seed — Anthropic API does not expose this
        gen_defaults.as_deref(),
        state.default_temperature, // G4: --default-temperature server flag
    );
    tracing::debug!(
        model_id = %req.model,
        temperature = sampling.temperature,
        temperature_source = temp_src.as_str(),
        top_p = sampling.top_p,
        top_p_source = top_p_src.as_str(),
        top_k = sampling.top_k,
        "messages: resolved sampling params (A7.1)"
    );

    tracing::debug!(
        model_id = %req.model,
        tool_count = norm_tools.as_ref().map_or(0, Vec::len),
        tool_choice = ?norm_tool_choice,
        "messages: tools parsed (A5.1), injected into template (A5.2)"
    );

    let mut gen_req = GenerationRequest {
        model_id: req.model.clone(),
        prompt_tokens,
        max_tokens,
        // A7.1: fully resolved sampling params.
        sampling,
        stop: req.stop_sequences.unwrap_or_default(),
        stream: req.stream,
        system,
        session_id,
        effective_prompt_cache_slots,
        // F6/L18: pass the drainer handle so the blocking thread can emit
        // per-request metrics to SQLite without blocking the decode loop.
        metrics_drainer: state.metrics_drainer.clone(),
        // M30: pass the ITL ring-buffer so the blocking thread can write
        // per-request ITL aggregates readable by /metrics/cache.
        itl_store: Some(Arc::clone(&state.itl_store)),
        // event_recorder wired below after ensure_loaded.
        event_recorder: None,
        // A5.1: normalised tool-calling fields (not yet consumed by engine).
        tools: norm_tools,
        tool_choice: norm_tool_choice,
        // A6.1: Anthropic route has no response_format field — always None here.
        response_format: None,
        // A6.2: Anthropic route never sets response_format, so no constraint
        // engine is instantiated. Anthropic JSON mode is done via prompt +
        // `stop_sequences`, not via sampler masking.
        constraint: None,
        is_thinking_handle: None,
        // Anthropic route has no thinking-budget field — never caps.
        thinking_budget: None,
        thinking_end_token_id: None,
        // ThinkSplitter init channel, read off the rendered prompt above.
        prompt_think_open,
        // A5.6: reconstruct tool-protocol special-token markers into the
        // decoded stream only when a tool-call parser is active (Gemma
        // markers are suppressed by `skip_special`).
        emit_tool_markers: tools_enabled,
        // Anthropic route has no per-request delimiter overrides — always None.
        thinking_start_token: None,
        thinking_end_token: None,
        // C5 Slice A: set below, after FIFO admission acquires the permit.
        gpu_admission: None,
        // Issue #26: the Anthropic Messages surface does not expose per-request
        // KV-config overrides (stricter wire spec); always launch default here.
        kv_quant_override: None,
        max_ctx_override: None,
        // Anthropic route has no multimodal content-part extraction yet.
        images: vec![],
        audio_b64: vec![],
        // The Anthropic surface has no image input yet, but carry the
        // server-startup `--image-max-tokens` default so a future image path
        // honours it; a no-op while `images` is empty.
        image_max_tokens: state.default_image_max_tokens,
    };

    // Ensure the requested model is loaded (auto-swap if needed).
    let (generator, is_cold_request) = match state.ensure_loaded(&req.model) {
        Ok(pair) => pair,
        Err(e) => {
            state.error_counts.increment(ApiErrorCategory::Upstream);
            return service_unavailable(&e);
        }
    };
    // Acquire the active-decode lease (drops on stream-end / fn-exit).
    let decode_lease_guard = state.decode_lease_guard(&req.model);
    // wire event_recorder now that we know the cold/warm flag.
    // Mirrors the OpenAI route so the same metric set is emitted for
    // Anthropic-route traffic.
    gen_req.event_recorder = state.metrics.clone();

    // A2: enforce per-request prompt length ceiling. Mirror of the OpenAI
    // route guard — see `openai.rs` for full rationale. Slot=None or a
    // generator that does not override `effective_max_ctx` returns usize::MAX,
    // letting the existing 503 path catch real runtime overflows.
    {
        let effective_max_ctx = state.effective_max_ctx_for(&req.model);
        let prompt_len = gen_req.prompt_tokens.len();
        if prompt_len > effective_max_ctx {
            state
                .error_counts
                .increment(ApiErrorCategory::ContextOverflow);
            return error_response(
                StatusCode::BAD_REQUEST,
                "context_length_exceeded",
                &format!("prompt has {prompt_len} tokens, max_ctx is {effective_max_ctx}"),
            );
        }
    }

    // A5.5: only pass the parser format when both tools[] was supplied AND
    // the arch has a known parser. Otherwise the decode loop bypasses the
    // parser entirely (same code path as pre-A5.5).
    let parser_format: Option<ToolCallFormat> = tools_enabled.then_some(()).and(tool_format);

    // Anticipatory 503 — same logic as the OpenAI route.
    if let Some(ref ctrl) = state.admission_controller {
        let n_prompt = gen_req.prompt_tokens.len() as u64;
        // M1: only read kv_cache_bytes when a single model is loaded (same as
        // the OpenAI route). Skip in the multi-model case to avoid reading the
        // wrong slot's footprint.
        let current_kv_bytes: u64 = {
            let slots = state.slots.read();
            if slots.len() <= 1 {
                slots.first().map_or(0, |m| m.model.kv_cache_bytes())
            } else {
                tracing::debug!(
                    "admission: multi-model ({} slots); skipping per-model kv_bytes",
                    slots.len()
                );
                0
            }
        };
        if let Some(reason) = ctrl.check_admission(n_prompt, current_kv_bytes) {
            debug_assert_eq!(reason, crate::admission::DecisionReason::AnticipatorySlo503);
            state
                .error_counts
                .increment(ApiErrorCategory::AdmissionSla503);
            tracing::warn!(
                reason = reason.as_str(),
                n_prompt,
                "admission rejected by anticipatory SLA controller (anthropic)"
            );
            let mut resp = error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "admission_sla_exceeded",
                "request rejected: predicted TTFT exceeds SLA threshold; retry after backoff",
            );
            resp.headers_mut().insert(
                axum::http::header::RETRY_AFTER,
                axum::http::HeaderValue::from_static("5"),
            );
            return resp;
        }
    }

    // C5 Slice A: FIFO admission over the single-GPU permit (mirror of the
    // OpenAI route). Bounded-depth 429 + FIFO + queue metrics. Acquired
    // here (async, before spawn_blocking) so the 429 status can be returned.
    //
    // When the adaptive controller is active, read the current depth.
    let effective_queue_depth = state
        .admission_controller
        .as_ref()
        .map_or(state.max_queue_depth, |c| {
            c.current_depth.load(std::sync::atomic::Ordering::Acquire)
        });
    // Initial zeros are overwritten in the Admitted arm before use.
    #[allow(unused_assignments)]
    let (mut admitted_depth, mut admitted_wait_ms): (u64, u64) = (0, 0);
    match crate::engine::admit_request(&state.gpu_queue, &state.gpu_pending, effective_queue_depth)
        .await
    {
        crate::engine::Admission::QueueFull => {
            state.error_counts.increment(ApiErrorCategory::RateLimit);
            return error_response(
                StatusCode::TOO_MANY_REQUESTS,
                "rate_limit_error",
                "server queue full",
            );
        }
        crate::engine::Admission::Admitted {
            guard,
            depth,
            wait_ms,
        } => {
            admitted_depth = depth;
            admitted_wait_ms = wait_ms;
            // count every admitted request (mirrors the OpenAI route).
            state
                .requests_started
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if let Some(ref drainer) = state.metrics_drainer {
                use crate::metrics_drainer::{MetricEvent, MetricKind};
                let ts = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();
                // F2: real ctx_max from the resident generator (clamped to i64 range).
                let ctx_max_val = state.effective_max_ctx_for(&req.model).min(1_048_576) as i64;
                drainer.try_emit(MetricEvent {
                    model_id: req.model.clone(),
                    kv_quant: "none".into(),
                    ts_utc: ts.clone(),
                    ctx_max: ctx_max_val,
                    kind: MetricKind::QueueWaitMs(wait_ms),
                });
                drainer.try_emit(MetricEvent {
                    model_id: req.model.clone(),
                    kv_quant: "none".into(),
                    ts_utc: ts,
                    ctx_max: ctx_max_val,
                    kind: MetricKind::QueueDepth(depth),
                });
            }
            gen_req.gpu_admission = Some(guard);
        }
    }

    // F2: resolve ctx_max once for both paths.
    let ctx_max_for_metrics = state.effective_max_ctx_for(&req.model).min(1_048_576) as i64;

    // evaluate token-replay eligibility. Anthropic does not expose `n`;
    // treat as n=1 (n_choices_ok = true).
    let replay_eligible = crate::retry::is_replayable(&gen_req, true);
    tracing::info!(
        request_id = %rid,
        replay_eligible,
        temperature = gen_req.sampling.temperature,
        has_constraint = gen_req.constraint.is_some(),
        "token-replay eligibility"
    );

    // snapshot the plan before gen_req is consumed.
    // Only constructed when replay is eligible.
    let replay_plan: Option<crate::retry::RequestPlan> =
        replay_eligible.then(|| crate::retry::RequestPlan::from_gen_req(&gen_req));

    if req.stream {
        generate_streaming(
            generator,
            gen_req,
            replay_plan,
            &req.model,
            request_start,
            &state,
            parser_format,
            ctx_max_for_metrics,
            &rid,
            is_cold_request,
            // Queue admission values for StepMetrics.
            (admitted_depth, admitted_wait_ms),
            // Move the lease into the SSE stream — drops on stream-end.
            decode_lease_guard,
        )
        .instrument(req_span)
        .await
    } else {
        // Hold the lease across the blocking generation await.
        let _lease = decode_lease_guard;
        generate_blocking(
            generator,
            gen_req,
            replay_plan,
            &req.model,
            parser_format,
            request_start,
            state.metrics_drainer.as_ref(),
            ctx_max_for_metrics,
            &state.tokens_in,
            &state.tokens_out,
            &rid,
            &state.error_counts,
            &state.requests_completed,
            &state.requests_failed,
            state.metrics.clone(),
            is_cold_request,
            // Controller handle + queue metrics for StepMetrics recording.
            state
                .admission_controller
                .as_ref()
                .map(|ctrl| (ctrl.clone(), admitted_depth, admitted_wait_ms)),
            &state.ttft_store,
        )
        .instrument(req_span)
        .await
    }
}
