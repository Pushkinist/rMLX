//! POST /v1/chat/completions route handler.
//!
//! Entry point: `chat_completions`. Handles request validation, prompt
//! pipeline (template render + tokenize), constraint setup, GPU admission,
//! and dispatches to `generate_blocking` or `generate_streaming`.

use std::sync::Arc;
use std::time::Instant;

use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use serde_json::Value;
use tracing::Instrument;

use crate::bounds;
use crate::chat_template::{ChatMessageTpl, RenderOpts};
use crate::engine::{
    normalized_to_jinja_tool, GenerationRequest, NormalizedResponseFormat, NormalizedTool,
    NormalizedToolChoice,
};
use crate::logged_json::LoggedJson;
use crate::session_cache::SessionKey;
use crate::tokenizer_io;
use crate::tool_parser::{detect_tool_call_format, ToolCallFormat};

use super::errors::{
    bad_request, error_response, internal_error, record_metric, resolve_request_id,
    resolve_sampling_params, service_unavailable,
};
use super::errors::{enforce_max_tokens_cap, parse_logit_bias};
use super::generate::{generate_blocking, generate_streaming};
use super::request::{
    ChatCompletionsRequest, MessageContent, OwnedTplMessage, StopSequences, ToolChoice,
};
use super::state::{ApiErrorCategory, AppState};

// A5.1: "tools" and "tool_choice" removed — now first-class parsed fields.
// A6.1: "response_format" removed — now a first-class parsed field.
// "functions" (legacy v0 OpenAI) still rejected — out of scope for A6.
const REJECTED_EXTRA: &[&str] = &["functions"];

// ── tool_choice=required/named schema synthesis ───────────────────────────────

/// Synthesise a JSON Schema for the constrained-tool-call output shape.
///
/// For `tool_choice=required` with N ≥ 1 tools: produces a `oneOf` where each
/// branch is `{"name": <const "TOOL_NAME">, "arguments": <that tool's schema>}`.
///
/// For `tool_choice=named` (single tool): a single-branch object schema —
/// `{"name": <const "TOOL_NAME">, "arguments": <schema>}`.
///
/// In constrained mode the model emits bare JSON (no `<tool_call>` wrapper).
/// The output is post-processed by [`bare_json_to_tool_call`] in the generate
/// path to synthesise the OpenAI `tool_calls` envelope.
///
/// `tools` must be non-empty and every selected tool must exist in the list.
/// Returns `None` if no matching tool is found (caller falls back to auto-mode).
pub(crate) fn tool_choice_to_schema(
    choice: &NormalizedToolChoice,
    tools: &[NormalizedTool],
) -> Option<Value> {
    // Build a single-branch object schema for a named tool.
    let branch = |tool: &NormalizedTool| -> Value {
        // `arguments` schema: use the tool's declared `parameters` if it's a
        // JSON object; fall back to an empty-object schema (permissive).
        let args_schema = if tool.schema.is_object() {
            tool.schema.clone()
        } else {
            serde_json::json!({"type": "object"})
        };
        serde_json::json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "const": tool.name},
                "arguments": args_schema
            },
            "required": ["name", "arguments"],
            "additionalProperties": false
        })
    };

    match choice {
        NormalizedToolChoice::Named(name) => {
            let tool = tools.iter().find(|t| &t.name == name)?;
            Some(branch(tool))
        }
        NormalizedToolChoice::Required => match tools {
            [] => None,
            [only] => Some(branch(only)),
            many => Some(serde_json::json!({"oneOf": many.iter().map(branch).collect::<Vec<_>>()})),
        },
        // Auto / None: no constraint schema needed.
        NormalizedToolChoice::Auto | NormalizedToolChoice::None => None,
    }
}

/// Parse a bare JSON string produced by the constrained tool-choice path into
/// a `ParsedToolCall`. Expected shape: `{"name":"<fn>","arguments":{...}}`.
///
/// `arguments` may be either a JSON object (direct) or a JSON-encoded string
/// (some models; degrades gracefully).
pub(crate) fn bare_json_to_tool_call(json: &str) -> Option<crate::tool_parser::ParsedToolCall> {
    let v: Value = serde_json::from_str(json.trim()).ok()?;
    let obj = v.as_object()?;
    let name = obj.get("name")?.as_str()?.to_owned();
    if name.is_empty() {
        return None;
    }
    let arguments: serde_json::Map<String, Value> = match obj.get("arguments")? {
        Value::Object(m) => m.clone(),
        Value::String(s) => match serde_json::from_str::<Value>(s) {
            Ok(Value::Object(m)) => m,
            other => {
                tracing::warn!(
                    ?other,
                    "bare_json_to_tool_call: `arguments` string did not decode to a JSON object"
                );
                return None;
            }
        },
        other @ (Value::Null | Value::Bool(_) | Value::Number(_) | Value::Array(_)) => {
            tracing::warn!(
                ?other,
                "bare_json_to_tool_call: `arguments` is neither object nor string"
            );
            return None;
        }
    };
    Some(crate::tool_parser::ParsedToolCall {
        id: crate::tool_parser::new_call_id(),
        name,
        arguments,
    })
}

// ── Route: POST /v1/chat/completions ─────────────────────────────────────────

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(crate) async fn chat_completions(
    State(state): State<AppState>,
    headers: HeaderMap,
    LoggedJson(req): LoggedJson<ChatCompletionsRequest>,
) -> Response {
    // L6: wall-clock anchor for TTFT. Captured immediately after header parse,
    // before any tokenization or model-load work.
    let request_start = Instant::now();

    // F10: resolve correlation id; the span is attached to each generate call
    // via .instrument() so all tracing inside generate_blocking / generate_streaming
    // automatically carry request_id.
    let rid = resolve_request_id(&headers);
    let req_span = tracing::info_span!("request", request_id = %rid, route = "chat_completions");
    tracing::info!(parent: &req_span, "chat_completions: incoming request");

    // Reject known unsupported injection-risk fields.
    for key in REJECTED_EXTRA {
        if req.extra.contains_key(*key) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request(&format!(
                "field `{key}` is not supported (Stage 2+); remove it from your request"
            ));
        }
    }
    // Log unknown extra fields (accept-and-ignore).
    for key in req.extra.keys() {
        tracing::debug!(field = %key, "chat_completions: ignoring unknown request field");
    }

    // Validate ranges.
    if let Some(t) = req.temperature {
        if !(0.0..=2.0).contains(&t) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("temperature must be in [0.0, 2.0]");
        }
    }
    if let Some(p) = req.top_p {
        if !(0.0..=1.0).contains(&p) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("top_p must be in [0.0, 1.0]");
        }
    }
    // A7.1: extended sampling field validation.
    // top_k: any u32 is valid (0 = disabled). No upper bound check needed.
    if let Some(p) = req.min_p {
        if !(0.0..=1.0).contains(&p) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("min_p must be in [0.0, 1.0]");
        }
    }
    if let Some(r) = req.repetition_penalty {
        if r <= 0.0 {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("repetition_penalty must be > 0.0");
        }
    }
    if let Some(f) = req.frequency_penalty {
        if !(-2.0..=2.0).contains(&f) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("frequency_penalty must be in [-2.0, 2.0]");
        }
    }
    if let Some(p) = req.presence_penalty {
        if !(-2.0..=2.0).contains(&p) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("presence_penalty must be in [-2.0, 2.0]");
        }
    }
    // Issue #26: per-request KV-cache config hot-swap. Parse `kv_quant` /
    // `max_ctx` overrides up front so a malformed codec string rejects with a
    // clean 400 before any tokenization or model-load work. `None` (omitted)
    // → fall through to the generator's launch default (zero regression).
    let req_kv_quant_override = match req.kv_quant.as_deref() {
        Some(s) => match crate::engine::parse_request_kv_quant(s) {
            Ok(v) => v,
            Err(e) => {
                state.error_counts.increment(ApiErrorCategory::BadRequest);
                return bad_request(&format!("kv_quant: {e}"));
            }
        },
        None => None,
    };
    if let Some(c) = req.max_ctx {
        if c <= 0 {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("max_ctx must be > 0");
        }
    }
    let req_max_ctx_override = req.max_ctx;
    if req_kv_quant_override.is_some() || req_max_ctx_override.is_some() {
        tracing::info!(
            kv_quant = ?req_kv_quant_override,
            max_ctx = ?req_max_ctx_override,
            "chat_completions: per-request KV-config override (issue #26)"
        );
    }
    // per-request image-token budget. Reject a zero budget with a clean 400;
    // the preprocessor clamps the upper bound. Resolution order (request >
    // `--image-max-tokens` server default > snapshot config) is completed in
    // the generator. `None` here keeps the server default.
    if let Some(n) = req.image_max_tokens {
        if n == 0 {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("image_max_tokens must be > 0");
        }
    }
    let req_image_max_tokens: Option<usize> = req
        .image_max_tokens
        .map(|n| n as usize)
        .or(state.default_image_max_tokens);
    // logprobs / top_logprobs validation (OpenAI semantics).
    // - `top_logprobs` requires `logprobs:true` (else 400).
    // - `top_logprobs` must be in 0..=20.
    let logprobs_enabled = req.logprobs.unwrap_or(false);
    if let Some(n) = req.top_logprobs {
        if !logprobs_enabled {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("top_logprobs requires logprobs to be set to true");
        }
        if n > 20 {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("top_logprobs must be in [0, 20]");
        }
    }
    // Resolve the capture width: `top_logprobs:N` ⇒ N alternatives; bare
    // `logprobs:true` ⇒ 1 (the chosen token's own logprob, no alternatives
    // beyond it). 0 ⇒ disabled (the hot-loop zero-overhead default).
    let top_logprobs_k: u32 = if logprobs_enabled {
        req.top_logprobs.unwrap_or(0).max(1)
    } else {
        0
    };
    // `echo:true` (per-prompt-position logprobs) is parsed for OpenAI
    // wire compatibility but the runtime path is deferred -- see the field
    // comment on `ChatCompletionsRequest::echo`. Reject the request with a
    // 501-style HTTP 400 + clear hint so clients can fall back to the
    // standalone scorer.
    if req.echo.unwrap_or(false) {
        state.error_counts.increment(ApiErrorCategory::BadRequest);
        return bad_request(
            "echo=true is not yet wired into the chat endpoint; \
             use the `rmlx eval ppl` CLI subcommand for per-prompt-position \
             logprobs; a future release will land the HTTP path.",
        );
    }
    // A7.1: parse logit_bias string-keyed map → Vec<(u32, f32)>.
    let logit_bias_parsed = match parse_logit_bias(req.logit_bias.as_ref()) {
        Ok(v) => v,
        Err(msg) => {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request(&msg);
        }
    };
    if let Some(m) = req.max_tokens {
        if m == 0 {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return bad_request("max_tokens must be > 0");
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
    // Per-message checks: tool_calls and content_parts.
    {
        let mut total_text_bytes: usize = 0;
        for (idx, msg) in req.messages.iter().enumerate() {
            if let Some(ref calls) = msg.tool_calls {
                if let Err(e) = bounds::check_tool_calls(calls.len(), idx) {
                    state.error_counts.increment(ApiErrorCategory::BadRequest);
                    return e.into_response();
                }
            }
            match &msg.content {
                Some(MessageContent::Parts(parts)) => {
                    if let Err(e) = bounds::check_content_parts(parts.len(), idx) {
                        state.error_counts.increment(ApiErrorCategory::BadRequest);
                        return e.into_response();
                    }
                    for (pidx, part) in parts.iter().enumerate() {
                        // audio: base64 byte estimate (base64 is ≈4/3 of binary)
                        if part.get("type").and_then(|t| t.as_str()) == Some("input_audio") {
                            if let Some(b64) = part
                                .get("input_audio")
                                .and_then(|a| a.get("data"))
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
                        // accumulate text bytes for the token-estimate check
                        if part.get("type").and_then(|t| t.as_str()) == Some("text") {
                            if let Some(s) = part.get("text").and_then(|t| t.as_str()) {
                                total_text_bytes = total_text_bytes.saturating_add(s.len());
                            }
                        }
                    }
                }
                Some(MessageContent::Text(s)) => {
                    total_text_bytes = total_text_bytes.saturating_add(s.len());
                }
                None => {}
            }
        }
        if let Err(e) = bounds::check_total_input_tokens_estimate(total_text_bytes) {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return e.into_response();
        }
    }

    // Extract the first system-role message (if any) as the canonical system
    // prompt. It is stripped from the prompt-message list so the engine
    // receives it once via `system`, not duplicated inside the conversation
    // history. Anthropic's top-level `system` field maps to the same slot.
    let system: Option<String> = req
        .messages
        .iter()
        .find(|m| m.role == "system")
        .map(|m| m.content_text().into_owned());

    // A5.1: normalise OpenAI tools → route-agnostic NormalizedTool.
    // Moved before the prompt pipeline so `jinja_tools` is available for
    // injection into RenderOpts inside spawn_blocking.
    // Empty array treated same as absent (skip alloc).
    let norm_tools: Option<Vec<NormalizedTool>> =
        req.tools.filter(|v| !v.is_empty()).map(|tools| {
            tools
                .into_iter()
                .map(|t| NormalizedTool {
                    name: t.function.name,
                    description: t.function.description,
                    schema: t.function.parameters,
                })
                .collect()
        });

    let norm_tool_choice: Option<NormalizedToolChoice> = req.tool_choice.map(|tc| match tc {
        ToolChoice::Mode(ref s) if s == "none" => NormalizedToolChoice::None,
        ToolChoice::Mode(ref s) if s == "required" => NormalizedToolChoice::Required,
        ToolChoice::Mode(_) => NormalizedToolChoice::Auto, // "auto" + anything else
        ToolChoice::Named(n) => NormalizedToolChoice::Named(n.function.name),
    });

    // A5.2: convert normalised tools to OpenAI-shaped JSON values for Jinja.
    // Note: tool_choice:"none" still injects tools here — hard suppression of
    // the tools block on tool_choice:none is a v1.x polish item (A5.4+).
    let mut jinja_tools: Vec<Value> = norm_tools
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .map(normalized_to_jinja_tool)
        .collect();

    // A9: runtime guard — if the loaded model's template cannot render a tool
    // context, suppress tool injection and warn once. The request proceeds
    // tool-less (normal content response) instead of 500-ing.
    if !jinja_tools.is_empty() {
        let template_supports_tools = state
            .registry
            .get(&req.model)
            .is_some_and(|e| e.tools_supported);
        if !template_supports_tools {
            tracing::warn!(
                model_id = %req.model,
                tool_count = jinja_tools.len(),
                "A9: template does not support tools — disabling tool injection for this request"
            );
            jinja_tools.clear();
        }
    }

    if !jinja_tools.is_empty() {
        tracing::debug!(
            model_id = %req.model,
            tool_count = jinja_tools.len(),
            tool_names = ?jinja_tools.iter().filter_map(|v| v["function"]["name"].as_str()).collect::<Vec<_>>(),
            "chat_template: rendering with tools"
        );
    }

    // A5.4: resolve the tool-call output format from the model's
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
            "A5.4: tools requested but no parser for this arch; passthrough"
        );
    }

    // / : resolve effective enable_thinking once at the outer scope
    // (request field > server default > absent = enabled). The prompt-pipeline
    // match below uses this for template rendering; the GenerationRequest
    // builder uses it to start the ThinkSplitter in the correct channel
    // (PART 2: Some(false) → answer-mode so output routes to `content`).
    let effective_enable_thinking = req.enable_thinking.or(state.default_enable_thinking);

    // resolve the thinking-end-token id ONCE, only when a thinking
    // budget is requested. Encoding with `add_special_tokens=false` mirrors
    // mlx-vlm's `ThinkingBudgetCriteria` (last id of the encoded literal).
    // `None` when no budget is set (no work) or the literal cannot be encoded
    // (budget degrades to a soft no-op — reasoning runs to `max_tokens`).
    // encode the caller-supplied `thinking_end_token` override when
    // present; otherwise fall back to the default `"</think>"` string so a
    // custom delimiter still caps the budget correctly.
    let thinking_budget = req.thinking_budget;
    let thinking_start_token = req.thinking_start_token.clone();
    let thinking_end_token = req.thinking_end_token.clone();
    let end_token_str: &str = thinking_end_token.as_deref().unwrap_or("</think>");
    let thinking_end_token_id: Option<u32> = thinking_budget.and_then(|_| {
        let tk = state
            .registry
            .get(&req.model)
            .and_then(|e| e.tokenizer.clone())?;
        let enc = tk.encode(end_token_str, false).ok()?;
        let ids = enc.get_ids();
        let id = ids.last().copied();
        if id.is_none() {
            tracing::warn!(
                model_id = %req.model,
                end_token = %end_token_str,
                "could not encode thinking end token — thinking budget will be a soft no-op"
            );
        }
        id
    });

    // ── Prompt pipeline (S1.7) ────────────────────────────────────────────────
    // Look up the model entry and run: render chat template → tokenize.
    // Best-effort: if the entry or pipeline is missing, fall through with
    // empty prompt_tokens and emit a metric + 503 note (generator is still
    // NotReadyGenerator anyway).
    // Delimiters the think-splitter will use for this request, needed at render
    // time to read the initial think channel off the rendered prompt.
    //
    // An empty (or whitespace-only) override is not a usable delimiter: it
    // matches at every offset, so the prompt scan would report the block open
    // for any prompt and the splitter's own scanner would never advance past
    // it. Reject at the boundary rather than letting either consumer inherit
    // the degenerate value.
    for (field, value) in [
        ("thinking_start_token", thinking_start_token.as_deref()),
        ("thinking_end_token", thinking_end_token.as_deref()),
    ] {
        if let Some(v) = value {
            if v.trim().is_empty() {
                state.error_counts.increment(ApiErrorCategory::BadRequest);
                return bad_request(&format!(
                    "`{field}` must be a non-empty, non-whitespace delimiter string"
                ));
            }
        }
    }
    let think_start_delim = thinking_start_token
        .clone()
        .unwrap_or_else(|| "<think>".to_owned());
    let think_end_delim = thinking_end_token
        .clone()
        .unwrap_or_else(|| "</think>".to_owned());
    let (prompt_tokens, prompt_think_open): (Vec<u32>, bool) = match state.registry.get(&req.model)
    {
        None => {
            // Unknown model — 404 before reaching the generator.
            state.error_counts.increment(ApiErrorCategory::NotFound);
            return error_response(
                StatusCode::NOT_FOUND,
                "not_found_error",
                &format!("model '{}' not found in registry", req.model),
            );
        }
        Some(entry) => {
            if let (Some(tpl), Some(tk)) = (&entry.chat_template, &entry.tokenizer) {
                // Render + tokenize moved into spawn_blocking so a
                // slow BPE encode (e.g. 32k-token prompt = tens of ms) does
                // not stall the async worker handling other requests' headers
                // or SSE keepalives.
                //
                // Owned clones of Arc<T> and String data are cheap vs the
                // BPE encode itself.
                let tpl = Arc::clone(tpl);
                let tk = Arc::clone(tk);
                let messages_owned: Vec<OwnedTplMessage> = req
                    .messages
                    .iter()
                    .map(OwnedTplMessage::from_request)
                    .collect();
                let bos = entry.bos_token.clone().unwrap_or_default();
                let eos = entry.eos_token.clone().unwrap_or_default();
                let model_id_log = req.model.clone();
                // A5.2: capture jinja_tools for injection into RenderOpts.
                // Vec<serde_json::Value> is Send + 'static.
                let jinja_tools_capture = jinja_tools.clone();
                // effective enable_thinking resolved at the outer scope
                // (captured by the closure below). Precedence: request field >
                // server default > absent (= enabled / undefined). Only
                // Some(false) changes template behaviour; Some(true) and None
                // both leave enable_thinking undefined in the Jinja context.
                let result = tokio::task::spawn_blocking(move || {
                    let tpl_msgs: Vec<ChatMessageTpl<'_>> =
                        messages_owned.iter().map(OwnedTplMessage::as_tpl).collect();
                    let opts = RenderOpts {
                        bos_token: bos.as_str(),
                        eos_token: eos.as_str(),
                        add_generation_prompt: true,
                        tools: &jinja_tools_capture,
                        enable_thinking: effective_enable_thinking,
                    };
                    let rendered = tpl
                        .render(&tpl_msgs, &opts)
                        .map_err(|e| format!("chat template render failed: {e}"))?;
                    // Read the initial think channel off the assistant-turn
                    // suffix ONLY. Message content is client-controlled and can
                    // contain a literal `<think>`; the suffix is emitted by the
                    // template alone. Re-render without the generation prompt
                    // and take the delta (a Jinja render, no tokenizer work).
                    let no_gen_opts = RenderOpts {
                        add_generation_prompt: false,
                        ..opts
                    };
                    let gen_suffix: &str = match tpl.render(&tpl_msgs, &no_gen_opts) {
                        Ok(base)
                            if rendered.text.len() >= base.text.len()
                                && rendered.text.starts_with(&base.text) =>
                        {
                            &rendered.text[base.text.len()..]
                        }
                        // Template does not simply append for the generation
                        // prompt (or the render failed). Fall back to the whole
                        // prompt and say so — the fallback is defeatable by
                        // message content, so it must not be silent.
                        other => {
                            tracing::warn!(
                                render_ok = other.is_ok(),
                                "chat template is not append-only for \
                                 add_generation_prompt; think-channel detection \
                                 falls back to scanning the whole prompt"
                            );
                            rendered.text.as_str()
                        }
                    };
                    let think_open = crate::engine::think::prompt_leaves_think_open(
                        gen_suffix,
                        &think_start_delim,
                        &think_end_delim,
                    );
                    let ids = tokenizer_io::encode(&tk, &rendered.text)
                        .map_err(|e| format!("tokenizer encode failed: {e}"))?;
                    Ok::<(Vec<u32>, bool), String>((ids, think_open))
                })
                .await;

                match result {
                    Err(join_err) => {
                        tracing::warn!(model_id = %model_id_log, error = %join_err, "prompt pipeline task panicked");
                        state.error_counts.increment(ApiErrorCategory::Internal);
                        return internal_error("prompt pipeline task panicked");
                    }
                    Ok(Err(e)) => {
                        tracing::warn!(model_id = %model_id_log, error = %e, "prompt pipeline failed");
                        state.error_counts.increment(ApiErrorCategory::Upstream);
                        return service_unavailable(&e);
                    }
                    Ok(Ok((ids, think_open))) => {
                        let token_count = ids.len();
                        tracing::debug!(
                            model_id = %model_id_log,
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
                            &model_id_log,
                            &entry.abs_path_str,
                        );
                        (ids, think_open)
                    }
                }
            } else {
                // Template or tokenizer absent for this entry.
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
    let raw_max_tokens = req.max_tokens.unwrap_or(512);
    let max_tokens = match enforce_max_tokens_cap(raw_max_tokens, state.max_tokens_cap, &req.model)
    {
        Ok(v) => v,
        Err(resp) => {
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return *resp;
        }
    };

    // ── N2: session KV-reuse ───────────────────────────────────────────────────
    // Extract optional `X-Session-Id` header. Presence is purely opt-in;
    // absence falls back to the N1 prompt-cache path unchanged.
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
        // The server's configured base widened by one slot per active session,
        // so the FIFO PromptCache never evicts a live session's snapshot.
        //
        // A base of 0 is the disabled cache and stays disabled: a request
        // header must not be able to turn on a cache the operator switched
        // off, and mixing the two capacities would rebuild the cache on every
        // request as `ensure` alternated between them.
        let base = state.prompt_cache_slots;
        let effective = crate::session_cache::effective_prompt_cache_slots(base, sc.active_count());
        tracing::debug!(
            model_id = %req.model,
            session_id = %sid,
            is_hit,
            active_sessions = sc.active_count(),
            base_prompt_cache_slots = base,
            effective_prompt_cache_slots = effective,
            "chat_completions: session cache lookup"
        );
        effective
    } else {
        None
    };
    // ─────────────────────────────────────────────────────────────────────────

    // A4/A7.1/G4: resolve all sampling params via request > server_default > model_defaults > hard_coded.
    let gen_defaults = state
        .registry
        .get(&req.model)
        .and_then(|e| e.generation_defaults.as_ref().map(Arc::clone));
    let (mut sampling, temp_src, top_p_src) = resolve_sampling_params(
        req.temperature,
        req.top_p,
        req.top_k,
        req.min_p,
        req.repetition_penalty,
        req.frequency_penalty,
        req.presence_penalty,
        logit_bias_parsed,
        req.seed,
        gen_defaults.as_deref(),
        state.default_temperature,
    );
    // apply the resolved logprob-capture width (no model-defaults tier).
    sampling.top_logprobs_k = top_logprobs_k;
    tracing::debug!(
        model_id = %req.model,
        temperature = sampling.temperature,
        temperature_source = temp_src.as_str(),
        top_p = sampling.top_p,
        top_p_source = top_p_src.as_str(),
        top_k = sampling.top_k,
        min_p = sampling.min_p,
        repetition_penalty = sampling.repetition_penalty,
        frequency_penalty = sampling.frequency_penalty,
        presence_penalty = sampling.presence_penalty,
        logit_bias_count = sampling.logit_bias.len(),
        "chat_completions: resolved sampling params (A7.1/G4)"
    );

    tracing::debug!(
        model_id = %req.model,
        tool_count = norm_tools.as_ref().map_or(0, Vec::len),
        tool_choice = ?norm_tool_choice,
        "chat_completions: tools parsed (A5.1), injected into template (A5.2)"
    );

    // A6.1: normalise response_format → NormalizedResponseFormat.
    // No enforcement yet; the field is metadata for A6.2+ (logit masking).
    let norm_response_format: Option<NormalizedResponseFormat> =
        req.response_format.as_ref().map(|rf| match rf {
            super::request::ResponseFormat::Text => NormalizedResponseFormat::Text,
            super::request::ResponseFormat::JsonObject => NormalizedResponseFormat::JsonObject,
            super::request::ResponseFormat::JsonSchema { json_schema } => {
                NormalizedResponseFormat::JsonSchema {
                    name: json_schema.name.clone(),
                    strict: json_schema.strict,
                    schema: json_schema.schema.clone(),
                }
            }
        });
    if let Some(ref rf) = norm_response_format {
        match rf {
            NormalizedResponseFormat::JsonObject => {
                tracing::debug!(model_id = %req.model, response_format = "json_object",
                    "chat_completions: response_format parsed (A6.1, no enforcement yet)");
            }
            NormalizedResponseFormat::JsonSchema { name, .. } => {
                tracing::debug!(model_id = %req.model, response_format = "json_schema",
                    schema_name = %name,
                    "chat_completions: response_format parsed (A6.1, no enforcement yet)");
            }
            NormalizedResponseFormat::Text => {
                tracing::debug!(model_id = %req.model, response_format = "text",
                    "chat_completions: response_format=text (no-op)");
            }
        }
    }

    // `tool_choice=required/named` → synthesise a JSON-Schema constraint
    // so the model is forced to emit bare `{"name":"…","arguments":{…}}` JSON.
    // In this mode the marker-based tool parser is bypassed; the post-processor
    // extracts the bare JSON and converts it into the OpenAI tool_calls envelope.
    //
    // `bare_json_tool_call_mode` is true when the constrained path is active and
    // the post-processor needs to interpret the output as a tool call (not content).
    let tool_choice_schema: Option<Value> = norm_tool_choice.as_ref().and_then(|tc| {
        norm_tools
            .as_deref()
            .filter(|t| !t.is_empty())
            .and_then(|tools| tool_choice_to_schema(tc, tools))
    });
    let bare_json_tool_call_mode = tool_choice_schema.is_some();

    // A6.3: instantiate a real sampler constraint engine when the request
    // asks for structured output. JsonObject (and JsonSchema, for now)
    // → tokenizer-aware JSON syntax constraint. The construction cost
    // (~600 ms on Qwen3.6 vocab to precompute the token-bytes map) is
    // paid here, off the unconstrained hot path. `Text` and `None` skip
    // this entirely.
    //
    // `tool_choice=required/named` takes priority and uses a
    // schema-driven constraint (Immediate engage — object root).
    //
    // We also extract the `is_thinking_handle` so the route's step_fn
    // can signal the constraint when the model is inside its reasoning
    // channel — preventing premature engagement on a `{` byte that
    // belongs to example JSON in the chain of thought.
    let (constraint, is_thinking_handle): (
        Option<Box<dyn rmlx_models::ConstraintEngine>>,
        Option<Arc<std::sync::atomic::AtomicBool>>,
    ) = match tool_choice_schema {
        Some(tc_schema) => {
            // tool_choice=required/named constraint (priority over response_format).
            let entry_opt = state.registry.get(&req.model);
            let tk_opt = entry_opt.and_then(|e| e.tokenizer.clone());
            let path_opt = entry_opt.map(|e| e.abs_path.clone());
            if let (Some(tk), Some(p)) = (tk_opt, path_opt) {
                let eos_ids: Vec<u32> = rmlx_loader::load_config(&p)
                    .map(|c| c.eos_token_ids())
                    .unwrap_or_default();
                tracing::info!(
                    model_id = %req.model,
                    request_id = %rid,
                    tool_choice = ?norm_tool_choice,
                    vocab_size = tk.get_vocab_size(true),
                    eos_ids = ?eos_ids,
                    "building SchemaConstraint for tool_choice=required/named (bare JSON mode)"
                );
                // Force Immediate engage so the constraint masks from
                // token 1, before any model-specific prefix (e.g. Gemma4's
                // `<|tool_call|>call:NAME{`) can slip through unconstrained.
                match crate::constraint_json::SchemaConstraint::new(
                    tk,
                    eos_ids,
                    &tc_schema,
                    false,
                    Some(crate::constraint_json::EngagePolicy::Immediate),
                ) {
                    Ok(c) => {
                        let handle = c.is_thinking_handle();
                        (
                            Some(Box::new(c) as Box<dyn rmlx_models::ConstraintEngine>),
                            Some(handle),
                        )
                    }
                    Err(e) => {
                        tracing::warn!(
                            model_id = %req.model,
                            error = %e,
                            "tool_choice schema constraint failed — falling back to unconstrained"
                        );
                        (None, None)
                    }
                }
            } else {
                tracing::warn!(
                    model_id = %req.model,
                    "tool_choice constraint requested but tokenizer/path \
                     missing — falling back to unconstrained"
                );
                (None, None)
            }
        }
        None => match norm_response_format.as_ref() {
            Some(
                NormalizedResponseFormat::JsonObject | NormalizedResponseFormat::JsonSchema { .. },
            ) => {
                let entry_opt = state.registry.get(&req.model);
                let tk_opt = entry_opt.and_then(|e| e.tokenizer.clone());
                let path_opt = entry_opt.map(|e| e.abs_path.clone());
                if let (Some(tk), Some(p)) = (tk_opt, path_opt) {
                    let eos_ids: Vec<u32> = rmlx_loader::load_config(&p)
                        .map(|c| c.eos_token_ids())
                        .unwrap_or_default();
                    // A6.4: schema-driven constraint.
                    if let Some(NormalizedResponseFormat::JsonSchema {
                        name,
                        strict,
                        schema,
                    }) = norm_response_format.as_ref()
                    {
                        tracing::info!(
                            model_id = %req.model,
                            request_id = %rid,
                            schema_name = %name,
                            strict,
                            vocab_size = tk.get_vocab_size(true),
                            eos_ids = ?eos_ids,
                            "A6.4: building SchemaConstraint (TokenBytesMap precompute)"
                        );
                        match crate::constraint_json::SchemaConstraint::new(
                            tk, eos_ids, schema, *strict, None,
                        ) {
                            Ok(c) => {
                                let handle = c.is_thinking_handle();
                                (
                                    Some(Box::new(c) as Box<dyn rmlx_models::ConstraintEngine>),
                                    Some(handle),
                                )
                            }
                            Err(e) if e.is_unsupported_keyword() => {
                                // strict==true + a keyword rMLX cannot
                                // enforce → HTTP 400 with the dedicated
                                // `unsupported_schema_keyword` code so
                                // callers can distinguish "I can't honour
                                // this in strict mode" from "malformed".
                                return error_response(
                                    StatusCode::BAD_REQUEST,
                                    "unsupported_schema_keyword",
                                    &format!(
                                        "response_format.json_schema.schema uses a \
                                         keyword rMLX cannot enforce in strict mode: {e}"
                                    ),
                                );
                            }
                            Err(e) => {
                                return bad_request(&format!(
                                    "response_format.json_schema.schema is not a \
                                     valid JSON Schema: {e}"
                                ));
                            }
                        }
                    } else {
                        // A6.3: schema-less json_object syntax constraint.
                        tracing::info!(
                            model_id = %req.model,
                            request_id = %rid,
                            vocab_size = tk.get_vocab_size(true),
                            eos_ids = ?eos_ids,
                            "A6.3: building JsonObjectConstraint (TokenBytesMap precompute)"
                        );
                        let c = crate::constraint_json::JsonObjectConstraint::new(tk, eos_ids);
                        let handle = c.is_thinking_handle();
                        (
                            Some(Box::new(c) as Box<dyn rmlx_models::ConstraintEngine>),
                            Some(handle),
                        )
                    }
                } else {
                    tracing::warn!(
                        model_id = %req.model,
                        "A6.3/A6.4: response_format requested but tokenizer/path \
                         missing — falling back to NoOpConstraint"
                    );
                    (Some(Box::new(rmlx_models::NoOpConstraint::new())), None)
                }
            }
            _ => (None, None),
        },
    };

    // Clone of the engine's engaged mirror, taken before the box is moved into
    // the generator. The non-streaming path reads it once the stream drains: a
    // `response_format` request whose grammar never engaged was never checked,
    // and nothing has reached the client yet, so it can still be refused.
    // Scoped to `response_format` — `tool_choice` has its own text-parsing
    // fallback and a non-engaged constraint there is not a failed contract.
    let response_format_engaged: Option<Arc<std::sync::atomic::AtomicBool>> =
        if bare_json_tool_call_mode {
            None
        } else {
            constraint.as_ref().and_then(|c| c.engaged_handle())
        };

    // extract image_url / input_audio content parts from user messages.
    // Collected across all user messages in order; will pass them to the
    // vision/audio towers. Text-only requests produce empty Vecs — zero cost.
    let (req_images, req_audio_b64): (Vec<String>, Vec<String>) = {
        let mut imgs = Vec::new();
        let mut audio = Vec::new();
        for msg in &req.messages {
            if msg.role == "user" {
                if let Some(MessageContent::Parts(parts)) = &msg.content {
                    imgs.extend(super::request::extract_image_parts(parts));
                    audio.extend(super::request::extract_audio_parts(parts));
                }
            }
        }
        tracing::debug!(
            image_count = imgs.len(),
            audio_count = audio.len(),
            "content-part extraction"
        );
        (imgs, audio)
    };

    let gen_req = GenerationRequest {
        model_id: req.model.clone(),
        prompt_tokens,
        max_tokens,
        // A7.1: fully resolved sampling params (all fields; decode stays greedy until A7.2).
        sampling,
        stop: req.stop.map(StopSequences::into_vec).unwrap_or_default(),
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
        // event_recorder for engine-side ITL/kv_cache writes.
        // TTFT (cold vs warm) is emitted by the handler layer off-runtime.
        event_recorder: state.metrics.clone(),
        // A5.1: normalised tool-calling fields (not yet consumed by engine).
        tools: norm_tools,
        tool_choice: norm_tool_choice,
        // A6.1: normalised response format (not yet consumed by engine).
        response_format: norm_response_format,
        // A6.2: sampler constraint plumbed end-to-end (real json_object grammar in A6.3).
        constraint,
        // A6.3: handle the route uses to signal `is_thinking` into the constraint.
        // Suppress thinking-channel routing when tool_choice=required/named.
        // In bare_json_tool_call_mode the constraint must engage immediately at
        // token 1 (EngagePolicy::Immediate); if the engine stores `is_thinking=true`
        // into the handle (e.g. Qwen3/Bonsai whose template prefills <think>), the
        // constraint would defer engagement and the tool call would land in
        // reasoning_content instead of text. Passing `None` keeps the handle unset,
        // so is_thinking stays false and Immediate fires on the first decode token.
        is_thinking_handle: if bare_json_tool_call_mode {
            None
        } else {
            is_thinking_handle
        },
        // per-request thinking budget + pre-resolved thinking-end-token id.
        thinking_budget,
        thinking_end_token_id,
        // ThinkSplitter init channel, read off the rendered prompt above.
        prompt_think_open,
        // A5.6: reconstruct tool-protocol special-token markers into the
        // decoded stream only when a tool-call parser is active for this
        // request (Gemma markers are suppressed by `skip_special`).
        emit_tool_markers: tools_enabled,
        // per-request delimiter overrides (None = keep defaults).
        thinking_start_token,
        thinking_end_token,
        // C5 Slice A: set below, after FIFO admission acquires the permit.
        gpu_admission: None,
        // Issue #26: per-request KV-config overrides threaded to the cache
        // builder (None = launch default).
        kv_quant_override: req_kv_quant_override,
        max_ctx_override: req_max_ctx_override,
        // multimodal content-part extraction.
        images: req_images,
        audio_b64: req_audio_b64,
        // per-request image-token budget (request > server default).
        image_max_tokens: req_image_max_tokens,
    };

    // Ensure the requested model is loaded (auto-swap if needed).
    // `is_cold_request` is true when the model was just loaded now
    // (first request after load), used to emit `ttft_cold_ms` vs `ttft_warm_ms`.
    let (generator, is_cold_request) = match state.ensure_loaded(&req.model) {
        Ok(pair) => pair,
        Err(e) => {
            state.error_counts.increment(ApiErrorCategory::Upstream);
            return error_response(
                StatusCode::SERVICE_UNAVAILABLE,
                "service_unavailable",
                &e.to_string(),
            );
        }
    };

    // Acquire the active-decode lease for this request. The guard is
    // held across the entire await of generate_blocking (non-streaming) or
    // moved into the SSE StreamState (streaming) so the keep-alive timer
    // never tears the model down mid-decode. OpenAI/Anthropic-compat routes
    // intentionally do NOT honor a `keep_alive` request field — the
    // ecosystem reserves that field for native /v1/models/{id}/load
    // routes — but the lease still protects in-flight decodes.
    let decode_lease_guard = state.decode_lease_guard(&req.model);

    // Enforce the per-request prompt-length ceiling. Without this guard a
    // prompt that overflows the KV-cache capacity bottoms out as either a 503
    // "generation produced zero tokens" or a 200 with completion_tokens=0.
    // Slot=None (cold-start race) or NotReadyGenerator default → usize::MAX,
    // letting the existing 503 path catch real runtime overflows there.
    {
        // A per-request `max_ctx` override (issue #26) re-sizes the KV-ring
        // ceiling for this one request, so it becomes the guard's ceiling —
        // but only after the same resolution the launch flag goes through
        // refuses one above the model's positional capacity. Refusing here
        // keeps the operator's own numbers in the message instead of surfacing
        // the engine's post-resolution ceiling as if it were the request.
        let effective_max_ctx = match req_max_ctx_override {
            Some(n) => match state.context_limits_for(&req.model) {
                Some(limits) => match rmlx_models::context::resolve_context(&limits, Some(n)) {
                    Ok(ctx) => ctx.ceiling_tokens(),
                    Err(e) => {
                        state
                            .error_counts
                            .increment(ApiErrorCategory::ContextOverflow);
                        return error_response(
                            StatusCode::BAD_REQUEST,
                            "context_length_exceeded",
                            &e.to_string(),
                        );
                    }
                },
                None => n as usize,
            },
            None => state.effective_max_ctx_for(&req.model),
        };
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

    // A5.4: only pass the parser format when both tools[] was supplied AND
    // the arch has a known parser. Otherwise the decode loop bypasses the
    // parser entirely (same code path as pre-A5.4).
    //
    // In bare_json_tool_call_mode (tool_choice=required/named) the
    // constraint drives output; the marker-based parser is bypassed entirely
    // (model emits bare JSON, not `<tool_call>…</tool_call>`).
    let parser_format: Option<ToolCallFormat> = if bare_json_tool_call_mode {
        None
    } else {
        tools_enabled.then_some(()).and(tool_format)
    };

    // A6.3: detect json_object mode for the response post-processor — even
    // with constraint masking, the model may emit a markdown fence wrapper
    // (` ```json ... ``` `) DURING the warm-up phase. The grammar guarantees
    // bytes from the engagement `{` through the matched closing `}` form a
    // valid JSON object; the post-processor extracts that substring before
    // the response is shipped to the client.
    //
    // bare_json_tool_call_mode also requires JSON extraction from text.
    let json_object_mode = bare_json_tool_call_mode
        || matches!(
            gen_req.response_format,
            Some(
                NormalizedResponseFormat::JsonObject | NormalizedResponseFormat::JsonSchema { .. }
            )
        );

    // Anticipatory 503 — when the adaptive controller is enabled, predict
    // TTFT for the incoming request before entering the FIFO queue. If the
    // regression says `est_ttft > 2 × ttft_target`, reject early with 503 +
    // Retry-After so the client knows to back off. This fires before the FIFO
    // semaphore is acquired, so it neither stalls the caller nor bumps gpu_pending.
    // When the controller is absent (default OFF) this block is a no-op.
    if let Some(ref ctrl) = state.admission_controller {
        let n_prompt = gen_req.prompt_tokens.len() as u64;
        // M1: only read kv_cache_bytes when a single model is loaded; with
        // multiple resident models `slots.first()` would read the wrong model's
        // KV footprint, biasing the admission estimate. Skip when multi-model
        // (0 is a safe conservative fallback — less accurate, never over-rejects).
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
            // Only AnticipatorySlo503 comes back from check_admission — assert invariant.
            debug_assert_eq!(reason, crate::admission::DecisionReason::AnticipatorySlo503);
            state
                .error_counts
                .increment(ApiErrorCategory::AdmissionSla503);
            tracing::warn!(
                reason = reason.as_str(),
                n_prompt,
                current_kv_bytes,
                "admission rejected by anticipatory SLA controller"
            );
            // Retry-After: 5 s (one controller tick interval — caller can retry).
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

    // C5 Slice A: FIFO admission over the single-GPU permit. Bounded-depth
    // 429 reject + FIFO fairness + queue-wait/depth metrics. Acquired here
    // (async, before spawn_blocking) so HTTP status can be returned; the
    // guard is moved into `gen_req` and lives until the decode finishes.
    //
    // When the adaptive controller is active, read the current depth
    // from its atomic (updated by the tick loop) instead of the static field.
    let effective_queue_depth = state
        .admission_controller
        .as_ref()
        .map_or(state.max_queue_depth, |c| {
            c.current_depth.load(std::sync::atomic::Ordering::Acquire)
        });
    // Track the queue admission values (depth, wait_ms) so they can be
    // forwarded to StepMetrics after the request completes. Populated in the
    // Admitted arm; only used when admission_controller is Some.
    // Initial zeros are overwritten in the Admitted arm before use.
    #[allow(unused_assignments)]
    let (mut admitted_depth, mut admitted_wait_ms): (u64, u64) = (0, 0);
    let mut gen_req = gen_req;
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
            // count every admitted request (passed depth check + entered queue).
            state
                .requests_started
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            // Emit queue metrics once per admitted request via the existing
            // SPSC drainer (mirrors the engine.rs prompt-cache emit pattern).
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

    // H4: capture prompt length before gen_req is consumed by the generator.
    // Used for the usage-summary chunk in the streaming path.
    let prompt_token_count = gen_req.prompt_tokens.len() as u32;
    let include_usage = req.stream_options.as_ref().is_some_and(|o| o.include_usage);

    // F2: resolve ctx_max once for both paths (non-streaming needs it for drainer).
    let ctx_max_for_metrics = state.effective_max_ctx_for(&req.model).min(1_048_576) as i64;

    // evaluate token-replay eligibility. Replay is only legal at
    // temp=0, n=1, and when no guided-decoding constraint is engaged.
    // OpenAI `n` is not currently parsed; treat as 1 (n_choices_ok = true).
    let replay_eligible = crate::retry::is_replayable(&gen_req, true);
    tracing::info!(
        request_id = %rid,
        replay_eligible,
        temperature = gen_req.sampling.temperature,
        has_constraint = gen_req.constraint.is_some(),
        "token-replay eligibility"
    );

    // snapshot the plan before gen_req is consumed.
    // Only constructed when replay is eligible — free on the non-replay path.
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
            json_object_mode,
            bare_json_tool_call_mode,
            include_usage,
            prompt_token_count,
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
        // Hold the lease for the entire blocking generation. The
        // guard drops when this scope exits.
        let _lease = decode_lease_guard;
        generate_blocking(
            generator,
            gen_req,
            replay_plan,
            &req.model,
            parser_format,
            json_object_mode,
            bare_json_tool_call_mode,
            response_format_engaged,
            request_start,
            state.metrics_drainer.as_ref(),
            ctx_max_for_metrics,
            &state.tokens_in,
            &state.tokens_out,
            &rid,
            &state.error_counts,
            &state.requests_completed,
            &state.requests_failed,
            // event recorder for TTFT writes to events table.
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
