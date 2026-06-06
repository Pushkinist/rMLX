//! Anthropic Messages API response types.
//!
//! `ContentBlock`, `AnthropicUsage`, `MessagesResponse`.

#![allow(
    clippy::exhaustive_enums,
    clippy::exhaustive_structs,
    unreachable_pub,
    reason = "Anthropic API wire DTOs — field/variant set tracks upstream spec"
)]

use serde::Serialize;
use serde_json::Value;

// ── Response types ────────────────────────────────────────────────────────────

/// One content block in a non-streaming Anthropic `messages` response.
///
/// Anthropic encodes block kinds as a tagged JSON object where the text
/// field key varies by kind: `text` blocks carry `text`, extended-
/// thinking blocks carry `thinking`. A3 surfaces reasoning-capable
/// architectures' `<think>...</think>` output as a leading `thinking`
/// block followed by the normal `text` block. A5.5 adds `tool_use`
/// blocks emitted when the model produces a parsed `<tool_call>`.
///
/// Per-variant `rename` is used instead of `rename_all = "lowercase"`
/// because `ToolUse` must serialise as `"tool_use"` (snake_case), while
/// the other two stay lowercase.
#[derive(Serialize, Debug)]
#[serde(tag = "type")]
pub enum ContentBlock {
    /// Plain text output block.
    #[serde(rename = "text")]
    Text {
        /// Decoded text content.
        text: String,
    },
    /// Extended-thinking (reasoning) block emitted for thinking-capable models.
    #[serde(rename = "thinking")]
    Thinking {
        /// Reasoning text emitted inside `<think>...</think>`.
        thinking: String,
    },
    /// A5.5: tool_use block. `input` is a JSON object (NOT a JSON-stringified
    /// string — that's OpenAI's `arguments` shape). The Anthropic public ID
    /// prefix is `toolu_`, but reusing the parser's `call_<hex>` ID is
    /// acceptable for v1 — clients treat the field as an opaque string.
    #[serde(rename = "tool_use")]
    ToolUse {
        /// Opaque call identifier; format `call_<hex>`.
        id: String,
        /// Name of the function to call.
        name: String,
        /// Parsed arguments as a JSON object (not a JSON-encoded string).
        input: Value,
    },
}

/// Token-count usage object in an Anthropic `messages` response.
#[derive(Serialize, Debug)]
pub struct AnthropicUsage {
    /// Number of prompt (input) tokens consumed.
    pub input_tokens: u32,
    /// Number of completion (output) tokens produced.
    pub output_tokens: u32,
}

/// Non-streaming response body for `POST /v1/messages`.
#[derive(Serialize, Debug)]
pub struct MessagesResponse {
    /// Unique message identifier with `msg_` prefix.
    pub id: String,
    /// Always `"message"` per the Anthropic spec.
    #[serde(rename = "type")]
    pub kind: String,
    /// Always `"assistant"` for model responses.
    pub role: String,
    /// Model id echoed from the request.
    pub model: String,
    /// Ordered list of content blocks produced by the model.
    pub content: Vec<ContentBlock>,
    /// Why generation stopped; e.g. `"end_turn"`, `"max_tokens"`, `"tool_use"`.
    pub stop_reason: Option<String>,
    /// The stop sequence that triggered stop, if applicable.
    pub stop_sequence: Option<String>,
    /// Prompt and completion token counts.
    pub usage: AnthropicUsage,
}
