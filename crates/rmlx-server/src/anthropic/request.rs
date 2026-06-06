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

impl AnthropicContent {
    /// Return the text content, borrowing for the common `Text` variant (zero
    /// allocation) and producing an owned `String` only for `Blocks`.
    pub fn as_text(&self) -> std::borrow::Cow<'_, str> {
        match self {
            AnthropicContent::Text(s) => std::borrow::Cow::Borrowed(s.as_str()),
            AnthropicContent::Blocks(blocks) => std::borrow::Cow::Owned(
                blocks
                    .iter()
                    .filter_map(|b| {
                        if b.get("type")?.as_str()? == "text" {
                            b.get("text")?.as_str().map(str::to_owned)
                        } else {
                            None
                        }
                    })
                    .collect::<String>(),
            ),
        }
    }
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
            AnthropicSystem::Blocks(blocks) => std::borrow::Cow::Owned(
                blocks
                    .iter()
                    .filter_map(|b| {
                        if b.get("type")?.as_str()? == "text" {
                            b.get("text")?.as_str().map(str::to_owned)
                        } else {
                            None
                        }
                    })
                    .collect::<String>(),
            ),
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

// ── A5.1: Anthropic tool-calling request types (schema only) ─────────────────

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

    // A5.1: tool-calling fields — parsed and normalised, not yet executed.
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
