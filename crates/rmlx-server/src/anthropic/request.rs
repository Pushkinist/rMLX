//! Anthropic Messages API request types.
//!
//! `AnthropicContent`, `AnthropicSystem`, `AnthropicMessage`, `AnthropicTool`,
//! `AnthropicToolChoice`, and `MessagesRequest`.

#![allow(
    clippy::exhaustive_enums,
    clippy::exhaustive_structs,
    unreachable_pub,
    reason = "Anthropic API wire DTOs — field/variant set tracks upstream spec"
)]

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::openai::request::{text_of_parts, tool_call_json, OwnedTplMessage};

// ── Request types ─────────────────────────────────────────────────────────────

/// `content` field on an Anthropic message.
///
/// Either a plain string or an array of content blocks.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum AnthropicContent {
    /// Plain string content.
    Text(String),
    /// Structured content block array (e.g. text + image blocks).
    Blocks(Vec<Value>),
}

/// The base64 data of an `input_audio` block: `{type:"input_audio",
/// source:{type:"base64", data:"…"}}`.
pub(super) fn input_audio_data(block: &Value) -> Option<&str> {
    if block.get("type").and_then(Value::as_str) != Some("input_audio") {
        return None;
    }
    block.get("source")?.get("data")?.as_str()
}

/// `system` field: plain string or array of text blocks.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum AnthropicSystem {
    /// Plain string system prompt.
    Text(String),
    /// Structured system-prompt block array.
    Blocks(Vec<Value>),
}

impl AnthropicSystem {
    pub(super) fn as_text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            AnthropicSystem::Text(s) => std::borrow::Cow::Borrowed(s.as_str()),
            AnthropicSystem::Blocks(blocks) => std::borrow::Cow::Owned(text_of_parts(blocks)),
        }
    }
}

/// A single message in an Anthropic conversation turn.
#[derive(Deserialize, Debug, Clone)]
pub struct AnthropicMessage {
    /// Role of the speaker: `"user"` or `"assistant"`.
    pub role: String,
    /// Content of the message (text or structured blocks).
    pub content: AnthropicContent,
}

impl AnthropicMessage {
    /// The base64 data of each `input_audio` block, in block order.
    pub(super) fn audio_b64(&self) -> impl Iterator<Item = String> + '_ {
        let blocks: &[Value] = match &self.content {
            AnthropicContent::Blocks(blocks) => blocks,
            AnthropicContent::Text(_) => &[],
        };
        blocks
            .iter()
            .filter_map(input_audio_data)
            .map(str::to_owned)
    }
}

impl OwnedTplMessage {
    /// The template messages of one Anthropic message, with every block value
    /// as the client sent it. Each `tool_result` block becomes a `tool`
    /// message, in block order. They come first, then one message of the
    /// message's own role with its text, `thinking` text and `tool_use` calls.
    ///
    /// `signature`, `redacted_thinking`, `is_error` and the image blocks of a
    /// tool result have no template field and are not rendered.
    pub(crate) fn from_anthropic(m: &AnthropicMessage) -> Vec<OwnedTplMessage> {
        let blocks = match &m.content {
            AnthropicContent::Text(text) => {
                return vec![OwnedTplMessage {
                    role: m.role.clone(),
                    content: text.clone(),
                    ..Default::default()
                }];
            }
            AnthropicContent::Blocks(blocks) => blocks,
        };

        let mut messages = Vec::new();
        let mut reasoning: Option<String> = None;
        let mut calls = Vec::new();
        let no_input = serde_json::json!({});
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("thinking") => {
                    if let Some(text) = block.get("thinking").and_then(Value::as_str) {
                        reasoning.get_or_insert_with(String::new).push_str(text);
                    }
                }
                Some("tool_use") => calls.push(tool_call_json(
                    block.get("id").and_then(Value::as_str),
                    None,
                    block
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    block.get("input").unwrap_or(&no_input),
                )),
                Some("tool_result") => messages.push(OwnedTplMessage {
                    role: "tool".to_owned(),
                    content: match block.get("content") {
                        Some(Value::String(text)) => text.clone(),
                        Some(Value::Array(parts)) => text_of_parts(parts),
                        _ => String::new(),
                    },
                    tool_call_id: block
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .map(str::to_owned),
                    ..Default::default()
                }),
                _ => {}
            }
        }

        let content = text_of_parts(blocks);
        let tool_calls_json = (!calls.is_empty()).then_some(Value::Array(calls));
        if messages.is_empty()
            || !content.is_empty()
            || reasoning.is_some()
            || tool_calls_json.is_some()
        {
            messages.push(OwnedTplMessage {
                role: m.role.clone(),
                content,
                tool_calls_json,
                reasoning_content: reasoning,
                ..Default::default()
            });
        }
        messages
    }
}

// ── Anthropic tool-calling request types (schema only) ─────────────────

/// One entry in the Anthropic `tools` array.
///
/// Note: Anthropic uses `input_schema` (not `parameters`) to hold the JSON
/// Schema. These are kept separate from the OpenAI types — do not share.
#[derive(Deserialize, Debug, Clone)]
pub struct AnthropicTool {
    /// Unique function name for this tool.
    pub name: String,
    /// Human-readable description of what the tool does.
    #[serde(default)]
    pub description: Option<String>,
    /// JSON Schema for the tool's input arguments.
    #[serde(default)]
    pub input_schema: Value,
}

/// Anthropic `tool_choice` field.
///
/// Always an object: `{"type": "auto"}` | `{"type": "any"}` |
/// `{"type": "tool", "name": "fn_name"}`.
#[derive(Deserialize, Debug, Clone)]
pub struct AnthropicToolChoice {
    /// Selection mode: `"auto"`, `"any"`, or `"tool"`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Tool name to force; set when `kind == "tool"`.
    #[serde(default)]
    pub name: Option<String>,
}

// ─────────────────────────────────────────────────────────────────────────────

/// Inbound request body for `POST /v1/messages` (Anthropic Messages API).
#[derive(Deserialize, Debug)]
pub struct MessagesRequest {
    /// Model id matching the Anthropic spec.
    pub model: String,
    /// Required by the Anthropic spec (no default; reject if absent).
    pub max_tokens: u32,
    /// Ordered conversation turns.
    pub messages: Vec<AnthropicMessage>,
    /// Optional system prompt injected before the first user turn.
    pub system: Option<AnthropicSystem>,
    /// Sampling temperature; valid range `[0.0, 1.0]`.
    pub temperature: Option<f32>,
    /// Nucleus sampling probability; valid range `[0.0, 1.0]`.
    pub top_p: Option<f32>,
    /// Top-k sampling cutoff; must be `>= 1` when set.
    pub top_k: Option<u32>,
    /// Stop strings; generation halts on the first match.
    pub stop_sequences: Option<Vec<String>>,
    /// When `true`, response is delivered as an SSE stream.
    #[serde(default)]
    pub stream: bool,
    /// `metadata` — accepted and ignored (debug-logged).
    pub metadata: Option<Value>,

    // Tool-calling fields — parsed and normalised; the client executes tools.
    /// Tool definitions available to the model.
    #[serde(default)]
    pub tools: Option<Vec<AnthropicTool>>,
    /// Tool selection preference for this request.
    #[serde(default)]
    pub tool_choice: Option<AnthropicToolChoice>,

    /// Catch-all for unknown fields. All are debug-logged and ignored.
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}
