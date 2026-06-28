//! OpenAI chat-completions request types — wire DTOs and message handling.

#![allow(unreachable_pub, dead_code)]

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::chat_template::ChatMessageTpl;

// ── Request / response structs ───────────────────────────────────────────────

/// `content` field for a chat message.
///
/// OpenAI allows a plain string or an array of content parts. For Stage 1
/// we accept both but only store the string representation.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
#[allow(
    clippy::exhaustive_enums,
    reason = "OpenAI API wire DTO — variant set tracks upstream spec; exhaustiveness is the contract"
)]
pub enum MessageContent {
    /// Plain string content.
    Text(String),
    /// Array of typed content parts (text, image_url, input_audio).
    Parts(Vec<Value>),
}

impl MessageContent {
    /// Return the text content, borrowing the inner string for the common
    /// `Text` variant (zero allocation) and producing an owned `String` only
    /// for the `Parts` variant that requires concatenation.
    pub fn as_text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            MessageContent::Text(s) => std::borrow::Cow::Borrowed(s.as_str()),
            MessageContent::Parts(parts) => std::borrow::Cow::Owned(
                parts
                    .iter()
                    .filter_map(|p| {
                        if p.get("type")?.as_str()? == "text" {
                            p.get("text")?.as_str().map(str::to_owned)
                        } else {
                            None
                        }
                    })
                    .collect::<String>(),
            ),
        }
    }
}

/// extract image URLs from a content-parts array.
///
/// Handles two shapes emitted by OpenAI-compatible clients:
/// - `{type:"image_url", image_url:{url:"<url>"}}` — standard OpenAI vision shape.
/// - `{type:"input_image", image_url:"<url>"}` — mlx-vlm / older client shape.
///
/// Returns the URL strings in part order. Non-matching parts are skipped.
pub fn extract_image_parts(parts: &[Value]) -> Vec<String> {
    parts
        .iter()
        .filter_map(|p| {
            let typ = p.get("type")?.as_str()?;
            match typ {
                // Standard OpenAI: {type:"image_url", image_url:{url:"…"}}
                "image_url" => p.get("image_url")?.get("url")?.as_str().map(str::to_owned),
                // mlx-vlm shape: {type:"input_image", image_url:"…"}
                "input_image" => p.get("image_url")?.as_str().map(str::to_owned),
                _ => None,
            }
        })
        .collect()
}

/// extract base64-encoded audio data from a content-parts array.
///
/// Handles the shape: `{type:"input_audio", input_audio:{data:"<b64>"}}`.
///
/// Returns the raw base64 strings in part order. Non-matching parts are skipped.
pub fn extract_audio_parts(parts: &[Value]) -> Vec<String> {
    parts
        .iter()
        .filter_map(|p| {
            if p.get("type")?.as_str()? == "input_audio" {
                p.get("input_audio")?
                    .get("data")?
                    .as_str()
                    .map(str::to_owned)
            } else {
                None
            }
        })
        .collect()
}

/// `tools[].function` sub-object on an inbound assistant tool-call message.
///
/// Mirrors the response-side [`ToolCallFunction`] shape but is `Deserialize`:
/// pi (and every OpenAI client) echoes the prior assistant turn back verbatim,
/// so a multi-turn tool session sends this on `messages[].tool_calls[]`.
/// Per the OpenAI spec `arguments` is a JSON-encoded **string**, not an object.
#[derive(Deserialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct RequestToolCallFunction {
    /// Name of the function that was called.
    pub name: String,
    /// JSON-encoded arguments string echoed from the prior assistant turn.
    #[serde(default)]
    pub arguments: String,
}

/// One entry in an inbound assistant message's `tool_calls` array.
#[derive(Deserialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct RequestToolCall {
    /// Opaque call identifier echoed from the prior assistant turn.
    #[serde(default)]
    pub id: Option<String>,
    /// Always `"function"` per the OpenAI spec when present.
    #[serde(rename = "type", default)]
    pub kind: Option<String>,
    /// The function name and arguments.
    pub function: RequestToolCallFunction,
}

#[derive(Deserialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
/// One message in the OpenAI `messages` array.
pub struct ChatMessage {
    /// Message role: `"user"`, `"assistant"`, `"system"`, or `"tool"`.
    pub role: String,
    /// `null`/absent on assistant tool-call turns and accepted as such — the
    /// untagged enum cannot represent JSON `null`, so `Option` carries it.
    #[serde(default)]
    pub content: Option<MessageContent>,
    /// Present on `assistant` turns that called tools (echoed back by the
    /// client on the next request).
    #[serde(default)]
    pub tool_calls: Option<Vec<RequestToolCall>>,
    /// Present on `tool`-role result turns; links the result to the call id.
    #[serde(default)]
    pub tool_call_id: Option<String>,
    /// Optional function name (tool-role turns / named messages).
    #[serde(default)]
    pub name: Option<String>,
}

impl ChatMessage {
    /// Text content with `None` collapsing to an empty string — matches the
    /// chat templates' `content is none` branch (renders `''`).
    pub fn content_text(&self) -> std::borrow::Cow<'_, str> {
        match &self.content {
            Some(c) => c.as_text(),
            None => std::borrow::Cow::Borrowed(""),
        }
    }
}

/// Owned, `'static` projection of a [`ChatMessage`] for the chat template.
///
/// The render+tokenize step runs in `spawn_blocking`, so the data it borrows
/// must outlive the request future — hence owned `String`s here, with
/// [`as_tpl`](OwnedTplMessage::as_tpl) lending `&str`/`&Value` views into a
/// borrowing [`ChatMessageTpl`].
///
/// `tool_calls_json` is pre-built into the exact shape both Qwen3 family
/// templates expect: `[{"id","type":"function","function":{"name","arguments"}}]`
/// where `arguments` is the **parsed object** (Qwen3.6's template iterates it
/// with `|items`; Ternary-Bonsai's `else`/`tojson` branch also accepts an
/// object). The OpenAI wire `arguments` is a JSON string; a parse failure
/// degrades to `{}` so `|items` stays safe.
pub(crate) struct OwnedTplMessage {
    pub(crate) role: String,
    pub(crate) content: String,
    pub(crate) tool_calls_json: Option<Value>,
    pub(crate) tool_call_id: Option<String>,
    pub(crate) name: Option<String>,
}

impl OwnedTplMessage {
    pub(crate) fn from_request(m: &ChatMessage) -> Self {
        let tool_calls_json = m.tool_calls.as_ref().map(|calls| {
            let arr: Vec<Value> = calls
                .iter()
                .map(|tc| {
                    let args_val: Value = serde_json::from_str(&tc.function.arguments)
                        .unwrap_or_else(|_| serde_json::json!({}));
                    serde_json::json!({
                        "id": tc.id,
                        "type": tc.kind.clone().unwrap_or_else(|| "function".to_owned()),
                        "function": {
                            "name": tc.function.name,
                            "arguments": args_val,
                        }
                    })
                })
                .collect();
            Value::Array(arr)
        });
        OwnedTplMessage {
            role: m.role.clone(),
            content: m.content_text().into_owned(),
            tool_calls_json,
            tool_call_id: m.tool_call_id.clone(),
            name: m.name.clone(),
        }
    }

    pub(crate) fn as_tpl(&self) -> ChatMessageTpl<'_> {
        ChatMessageTpl {
            role: self.role.as_str(),
            content: self.content.as_str(),
            tool_calls: self.tool_calls_json.as_ref(),
            tool_call_id: self.tool_call_id.as_deref(),
            name: self.name.as_deref(),
        }
    }
}

/// `stop` may be a single string or an array of strings.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
#[allow(
    clippy::exhaustive_enums,
    reason = "OpenAI API wire DTO — variant set tracks upstream spec; exhaustiveness is the contract"
)]
pub enum StopSequences {
    /// Single stop string.
    One(String),
    /// Array of stop strings.
    Many(Vec<String>),
}

impl StopSequences {
    pub(crate) fn into_vec(self) -> Vec<String> {
        match self {
            StopSequences::One(s) => vec![s],
            StopSequences::Many(v) => v,
        }
    }
}

// ── A6.1: response_format request types (schema only; no enforcement yet) ────

/// Specification object carried inside `response_format` when `type = "json_schema"`.
#[derive(Deserialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct JsonSchemaSpec {
    /// Schema name; echoed in the response for client correlation.
    pub name: String,
    /// When `true`, disallow extra properties not listed in the schema.
    #[serde(default)]
    pub strict: bool,
    /// The JSON Schema value to enforce.
    pub schema: Value,
    /// Optional human-readable description of the schema.
    #[serde(default)]
    pub description: Option<String>,
}

/// OpenAI `response_format` field.
///
/// Three variants:
/// - `text` — plain-text output (default / no-op).
/// - `json_object` — model is expected to produce any valid JSON object.
/// - `json_schema` — model is expected to produce JSON conforming to `json_schema`.
///
/// Constraint enforcement (logit masking, grammar) is not wired yet; this is
/// parsed and normalised for A6.2+. The enum is closed — unknown type strings
/// produce a serde error (HTTP 422).
#[derive(Deserialize, Debug, Clone)]
#[serde(tag = "type", rename_all = "snake_case")]
#[allow(
    clippy::exhaustive_enums,
    reason = "OpenAI API wire DTO — variant set tracks upstream spec; exhaustiveness is the contract"
)]
pub enum ResponseFormat {
    /// Plain text output (default / no-op).
    Text,
    /// Any valid JSON object.
    JsonObject,
    /// JSON conforming to the supplied schema.
    JsonSchema {
        /// The JSON Schema specification from the request.
        json_schema: JsonSchemaSpec,
    },
}

// ── A5.1: tool-calling request types (schema only; no execution yet) ─────────

/// OpenAI `tools[].function` sub-object.
#[derive(Deserialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct ToolFunction {
    /// Function name matching the OpenAI spec.
    pub name: String,
    /// Optional human-readable description of the function.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the function arguments. Stored verbatim — validation
    /// is the model's responsibility, not ours.
    #[serde(default)]
    pub parameters: Value,
}

/// One entry in the OpenAI `tools` array.
#[derive(Deserialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct Tool {
    /// Tool type; always `"function"` per the current OpenAI spec.
    #[serde(rename = "type")]
    pub kind: String,
    /// The function definition for this tool.
    pub function: ToolFunction,
}

/// Named-function variant for `tool_choice`.
#[derive(Deserialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct NamedToolFunction {
    /// Name of the specific function to call.
    pub name: String,
}

/// Object variant for `tool_choice: {"type": "function", "function": {...}}`.
#[derive(Deserialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct NamedToolChoice {
    /// Always `"function"` per the OpenAI spec.
    #[serde(rename = "type")]
    pub kind: String,
    /// The function selector.
    pub function: NamedToolFunction,
}

/// `tool_choice` field on the chat completions request.
///
/// Either a plain mode string (`"auto"`, `"none"`, `"required"`) or an object
/// selecting a specific named function (`{"type":"function","function":{"name":"…"}}`).
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
#[allow(
    clippy::exhaustive_enums,
    reason = "OpenAI API wire DTO — variant set tracks upstream spec; exhaustiveness is the contract"
)]
pub enum ToolChoice {
    /// Mode string: `"auto"` | `"none"` | `"required"`.
    Mode(String),
    /// Named-function selector object.
    Named(NamedToolChoice),
}

// ─────────────────────────────────────────────────────────────────────────────

#[derive(Deserialize, Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
/// Inbound request body for `POST /v1/chat/completions` (OpenAI Chat Completions API).
pub struct ChatCompletionsRequest {
    /// Model id matching the OpenAI spec.
    pub model: String,
    /// Ordered list of conversation messages.
    pub messages: Vec<ChatMessage>,
    /// When `true`, response is delivered as an SSE stream.
    #[serde(default)]
    pub stream: bool,
    /// Sampling temperature; valid range `[0.0, 2.0]`.
    pub temperature: Option<f32>,
    /// Maximum new tokens to generate.
    pub max_tokens: Option<u32>,
    /// Nucleus sampling probability; valid range `[0.0, 1.0]`.
    pub top_p: Option<f32>,
    /// Random seed for reproducible outputs.
    pub seed: Option<u64>,
    /// Stop string(s); generation halts on the first match.
    pub stop: Option<StopSequences>,

    // A5.1: tool-calling fields — parsed and normalised, not yet executed.
    /// Tool definitions available to the model.
    #[serde(default)]
    pub tools: Option<Vec<Tool>>,
    /// Tool selection preference for this request.
    #[serde(default)]
    pub tool_choice: Option<ToolChoice>,

    // A6.1: response format — parsed and normalised; constraint enforcement
    // follows in A6.2..A6.5. `None` is equivalent to `Text` (plain text).
    /// Desired output format; `None` is plain text.
    #[serde(default)]
    pub response_format: Option<ResponseFormat>,

    // A7.1: extended sampling fields (schema + plumb; decode stays greedy until A7.2/A7.3).
    /// Top-k sampling cutoff; must be `>= 1` when set.
    #[serde(default)]
    pub top_k: Option<u32>,
    /// Minimum token probability (nucleus floor); `0.0` = disabled.
    #[serde(default)]
    pub min_p: Option<f32>,
    /// Repetition penalty multiplier; `1.0` = no penalty.
    #[serde(default)]
    pub repetition_penalty: Option<f32>,
    /// Frequency penalty applied to already-seen tokens.
    #[serde(default)]
    pub frequency_penalty: Option<f32>,
    /// Presence penalty for encouraging novel topics.
    #[serde(default)]
    pub presence_penalty: Option<f32>,
    /// Token-id (as string key) → logit bias. Per the OpenAI spec, keys are
    /// token ids expressed as strings (e.g. `"1234"`). Parsed to `u32` in the
    /// route handler; non-integer keys → 400.
    #[serde(default)]
    pub logit_bias: Option<HashMap<String, f32>>,

    // H4: stream_options — lifted from the extra catch-all so include_usage
    // is accessible by name. Only meaningful when stream=true; ignored for
    // non-streaming requests per the OpenAI spec.
    /// Stream output options; only meaningful when `stream = true`.
    #[serde(default)]
    pub stream_options: Option<StreamOptions>,

    // per-token log-probability output (OpenAI-compatible).
    // `logprobs:true` returns the chosen token's logprob per content token.
    // `top_logprobs:N` (0..=20) additionally returns the N most-likely
    // alternatives per token and REQUIRES `logprobs:true` (else 400).
    /// When `true`, emit per-token logprobs alongside content tokens.
    #[serde(default)]
    pub logprobs: Option<bool>,
    /// Number of top alternative tokens per position (`0..=20`); requires `logprobs:true`.
    #[serde(default)]
    pub top_logprobs: Option<u32>,

    // OpenAI `echo` flag. When paired with `max_tokens=0` and
    // `logprobs:true`, asks the server to return per-token logprobs over the
    // PROMPT (no generation). Used by the wikitext-2 PPL harness.
    //
    // Status: the schema is parsed and validated; the runtime path that
    // computes per-prompt-position logprobs is deferred (fell back to
    // the standalone `rmlx eval ppl` CLI subcommand because exposing
    // per-position prefill logits across every architecture required engine
    // refactor outside this ticket's scope). Requests with `echo:true` are
    // rejected with HTTP 501 and a "use `rmlx eval ppl`" hint until the
    // follow-up lands. // TODO: wire per-arch forward-all-logits path and accept echo+max_tokens=0 here.
    /// When `true` (with `max_tokens=0` + `logprobs:true`), return prompt logprobs instead of generating.
    #[serde(default)]
    pub echo: Option<bool>,

    // per-request thinking control. `Some(false)` suppresses the open
    // `<think>` block on Qwen3-family models (no-think mode); `Some(true)` or
    // `None` leaves the template default in place (open `<think>` block).
    // Precedence: request > AppState::default_enable_thinking > absent (= enabled).
    /// Per-request thinking mode; `Some(false)` disables `<think>` block for Qwen3-family.
    #[serde(default)]
    pub enable_thinking: Option<bool>,

    // per-request thinking-token budget. `Some(n)` caps the model's
    // reasoning channel at `n` tokens; once exceeded the engine injects
    // the thinking end delimiter and resumes answer generation. `None` (the
    // default) leaves reasoning uncapped (runs to `max_tokens`). Only
    // meaningful for thinking-capable archs (Qwen3-family); a no-op elsewhere.
    /// Cap on reasoning-channel tokens; `None` = uncapped.
    #[serde(default)]
    pub thinking_budget: Option<u32>,

    // per-request delimiter overrides for the thinking-block splitter.
    // When present, redirects both the splitter pattern matching and the
    // budget-enforcement forced-injection to the supplied string instead of the
    // default `"<think>"` / `"</think>"`. `None` (omitted) preserves the
    // current default behavior exactly.
    /// Override for the thinking-block open delimiter; `None` uses `"<think>"`.
    #[serde(default)]
    pub thinking_start_token: Option<String>,
    /// Override for the thinking-block close delimiter; `None` uses `"</think>"`.
    #[serde(default)]
    pub thinking_end_token: Option<String>,

    // Issue #26: per-request KV-cache config hot-swap on a resident model.
    // These let a single `rmlx serve` process (weights loaded once) switch the
    // KV codec and context ceiling per request — no weight reload. `None`
    // (omitted) preserves the launch-default behavior exactly (zero regression).
    /// Per-request KV-quant codec override (e.g. `"none"`, `"k8v4"`, `"k8v8"`,
    /// `"planar"`, `"auto"`). `"auto"` selects the per-arch/per-ctx default.
    /// `None` (omitted) uses the server's launch `--kv-quant`. Parsed with the
    /// same grammar as the `--kv-quant` CLI flag.
    #[serde(default)]
    pub kv_quant: Option<String>,
    /// Per-request max-context ceiling override (KV ring grows lazily up to this
    /// ceiling, #25). `None` (omitted) uses the server's launch `--max-ctx`.
    #[serde(default)]
    pub max_ctx: Option<i32>,

    // Per-request image-token budget for Gemma4-unified vision. `Some(n)`
    // raises the soft-token budget for dense images (e.g. tables), preserving
    // more vision resolution; the preprocessor clamps it to the model's safe
    // upper bound. `None` (omitted) uses the server's launch
    // `--image-max-tokens` or the snapshot's `processor_config.json` default.
    // Resolution order: request > CLI flag > config default. A no-op for
    // text-only requests and non-Gemma4-unified vision archs.
    /// Per-request image-token budget override for Gemma4-unified vision;
    /// `None` uses the server default.
    #[serde(default)]
    pub image_max_tokens: Option<u32>,

    // Stage 2+ features — accepted & ignored with a debug log.
    // Explicitly reject only fields that indicate unsafe injection intent.
    /// Catch-all for unknown request fields; debug-logged and ignored.
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

/// H4: `stream_options` request field.
///
/// Currently only `include_usage` is used. The struct is kept open for
/// future additions (e.g. `include_logprobs`) without changing the wire
/// format.
#[derive(Deserialize, Debug, Default, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct StreamOptions {
    /// When `true`, the server appends one extra SSE chunk with
    /// `choices: []` and a populated `usage` object immediately before
    /// the `[DONE]` sentinel. Per the OpenAI spec this chunk always
    /// appears (even when `max_tokens` is reached early).
    #[serde(default)]
    pub include_usage: bool,
}
