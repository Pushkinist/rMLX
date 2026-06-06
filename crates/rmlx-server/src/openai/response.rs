//! OpenAI chat-completions response types — wire DTOs for non-streaming
//! and streaming (SSE) responses.

#![allow(unreachable_pub)]

use serde::Serialize;

use crate::tool_parser::ParsedToolCall;

// ── logprobs response shape (OpenAI-compatible) ────────────────────────

/// One alternative token in a `top_logprobs` list.
///
/// Mirrors the OpenAI `choices[].logprobs.content[].top_logprobs[]` entry:
/// `{ token, logprob, bytes }`. `bytes` is the UTF-8 byte sequence of the
/// token surface (OpenAI uses this so clients can reconstruct partial
/// codepoints); `null` is allowed when the surface is unavailable.
#[derive(Serialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream metric spec; exhaustiveness is the contract"
)]
pub struct TopLogprob {
    /// Token surface string for this alternative.
    pub token: String,
    /// Log-probability of this alternative token.
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// UTF-8 byte sequence of the token surface; `null` when unavailable.
    pub bytes: Option<Vec<u8>>,
}

/// One emitted content token's logprob record.
///
/// Mirrors OpenAI `choices[].logprobs.content[]`: the chosen `token`, its
/// `logprob`, the token surface `bytes`, and the `top_logprobs` alternatives.
#[derive(Serialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream metric spec; exhaustiveness is the contract"
)]
pub struct ChatLogprobContent {
    /// Surface string of the chosen token.
    pub token: String,
    /// Log-probability of the chosen token.
    pub logprob: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// UTF-8 byte sequence of the chosen token surface; `null` when unavailable.
    pub bytes: Option<Vec<u8>>,
    /// Top-N alternative tokens and their log-probabilities at this position.
    pub top_logprobs: Vec<TopLogprob>,
}

/// `choices[].logprobs` object. Only the `content` array is populated; the
/// `refusal` channel (OpenAI) is not produced by this backend.
#[derive(Serialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream metric spec; exhaustiveness is the contract"
)]
pub struct ChatLogprobs {
    /// Per-token logprob records for the generated content sequence.
    pub content: Vec<ChatLogprobContent>,
}

/// resolve a decode-loop `TokenLogprobs` (token ids + logprobs) into the
/// OpenAI wire shape, decoding each id to its token surface via the tokenizer.
///
/// `chosen_piece` is the chosen token's already-decoded visible text (carried
/// on `GenerationToken`); we reuse it rather than re-decoding so the `token`
/// field matches the streamed `content`. Alternative-token surfaces come from
/// `tokenizer.id_to_token` (raw piece, may contain the SentencePiece `▁`
/// marker — acceptable: OpenAI clients key off `bytes` for reconstruction).
pub(crate) fn resolve_logprobs(
    lp: &rmlx_models::TokenLogprobs,
    chosen_piece: &str,
    tokenizer: &tokenizers::Tokenizer,
) -> ChatLogprobContent {
    let surface = |id: u32| -> String {
        tokenizer
            .id_to_token(id)
            .unwrap_or_else(|| format!("<unk:{id}>"))
    };
    let top = lp
        .top
        .iter()
        .map(|&(id, logprob)| {
            let token = surface(id);
            TopLogprob {
                bytes: Some(token.as_bytes().to_vec()),
                token,
                logprob,
            }
        })
        .collect();
    ChatLogprobContent {
        token: chosen_piece.to_owned(),
        logprob: lp.token_logprob,
        bytes: Some(chosen_piece.as_bytes().to_vec()),
        top_logprobs: top,
    }
}

/// Non-streaming message in `choices[].message` of a chat completion response.
#[derive(Serialize, Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct ResponseMessage {
    /// Role of the message author, always `"assistant"` for completion responses.
    pub role: String,
    /// Generated text content of the assistant message.
    pub content: String,
    /// A3: accumulated text emitted from inside `<think>...</think>` blocks
    /// for reasoning-capable architectures (Qwen3 family). `None` when the
    /// model never produced any thinking text (non-reasoning archs, or the
    /// model exited the prefilled think block without emitting anything).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// A5.4: parsed tool calls from the model's output, in OpenAI shape.
    /// `None` when the model emitted no `<tool_call>` blocks (or tools were
    /// not enabled). Serialised key is omitted when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

// ── A5.4: tool_call response shape ───────────────────────────────────────────

/// One tool call entry in an OpenAI chat completion response.
///
/// Matches the OpenAI spec for `choices[].message.tool_calls[]` (non-streaming)
/// and `choices[].delta.tool_calls[]` (streaming). The `arguments` field is
/// the function arguments serialised as a JSON **string** per the spec, not a
/// raw JSON object.
#[derive(Serialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct ToolCall {
    /// Monotonic index per choice; v1 emits one complete tool_call per chunk.
    pub index: u32,
    /// Stable ID assigned at parse time. Format `call_<hex>`.
    pub id: String,
    /// Always `"function"` for v1.
    #[serde(rename = "type")]
    pub kind: String,
    /// Name and JSON-stringified arguments of the called function.
    pub function: ToolCallFunction,
}

/// Function name and arguments in a tool call response entry.
#[derive(Serialize, Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct ToolCallFunction {
    /// Name of the function the model invoked.
    pub name: String,
    /// JSON-stringified arguments object (OpenAI spec: arguments is a string).
    pub arguments: String,
}

/// Convert a `tool_parser::ParsedToolCall` into the OpenAI wire shape.
pub(crate) fn to_response_tool_call(p: &ParsedToolCall, index: u32) -> ToolCall {
    let arguments =
        serde_json::to_string(&serde_json::Value::Object(p.arguments.clone())).unwrap_or_default();
    ToolCall {
        index,
        id: p.id.clone(),
        kind: "function".to_owned(),
        function: ToolCallFunction {
            name: p.name.clone(),
            arguments,
        },
    }
}

/// A5.4: select the wire `finish_reason` after consuming the full token stream.
///
/// Per OpenAI spec, when any tool_call was emitted, `finish_reason="tool_calls"`
/// regardless of the natural model finish (which would typically be `"stop"`).
pub(crate) fn select_finish_reason(
    any_tool_calls: bool,
    terminal: Option<String>,
) -> Option<String> {
    if any_tool_calls {
        Some("tool_calls".to_owned())
    } else {
        terminal
    }
}

/// One completion alternative in a non-streaming `ChatCompletionsResponse`.
#[derive(Serialize, Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct Choice {
    /// Zero-based index of this choice in the `choices` array.
    pub index: u32,
    /// Assistant message produced by the model for this choice.
    pub message: ResponseMessage,
    /// Reason the model stopped generating: `"stop"`, `"length"`, `"tool_calls"`, etc.
    pub finish_reason: Option<String>,
    /// per-token logprobs for this choice (OpenAI puts `logprobs` on the
    /// choice, not the message). `None` when the request did not set
    /// `logprobs:true`; omitted from the wire when `None`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

/// Token usage counters for a chat completion response.
#[derive(Serialize, Debug)]
#[allow(
    clippy::exhaustive_structs,
    clippy::struct_field_names,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct Usage {
    /// Number of tokens in the input prompt.
    pub prompt_tokens: u32,
    /// Number of tokens generated by the model.
    pub completion_tokens: u32,
    /// Sum of `prompt_tokens` and `completion_tokens`.
    pub total_tokens: u32,
}

/// Non-streaming response body for `POST /v1/chat/completions`.
#[derive(Serialize, Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct ChatCompletionsResponse {
    /// Unique request identifier in the form `chatcmpl-<hex>`.
    pub id: String,
    /// Always `"chat.completion"` per the OpenAI spec.
    pub object: String,
    /// Unix timestamp (seconds) when the response was created.
    pub created: u64,
    /// Model id echoed from the request.
    pub model: String,
    /// One or more completion alternatives produced by the model.
    pub choices: Vec<Choice>,
    /// Token usage counts for this request.
    pub usage: Usage,
}

/// Delta variant used in SSE chunks.
#[derive(Serialize, Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct DeltaContent {
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Role preamble sent on the first SSE chunk; `None` on all subsequent chunks.
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Incremental text fragment for this chunk; `None` when chunk carries role or tool_call.
    pub content: Option<String>,
    /// A3: reasoning text emitted from inside `<think>...</think>` blocks.
    /// Mutually exclusive with `content` per chunk — exactly one of them is
    /// populated for a non-role token chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    /// A5.4: complete tool_call(s) for this chunk. v1 emits one complete call
    /// per chunk once `</tool_call>` is consumed by the parser. Clients
    /// accumulate by `index`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// One SSE-chunk alternative in a streaming `ChatCompletionChunk`.
#[derive(Serialize, Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct StreamChoice {
    /// Zero-based index of this choice in the `choices` array.
    pub index: u32,
    /// Incremental content delta for this SSE chunk.
    pub delta: DeltaContent,
    /// Reason generation stopped; `null` on all non-terminal chunks.
    pub finish_reason: Option<String>,
    /// per-token logprobs for the token(s) in this chunk's delta.
    /// OpenAI streams logprobs on the choice alongside each content delta.
    /// `None` (omitted) when logprobs were not requested or this chunk
    /// carries no content token (role preamble, tool_call, usage chunk).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<ChatLogprobs>,
}

/// One SSE event body for a streaming `POST /v1/chat/completions` response.
#[derive(Serialize, Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "OpenAI API wire DTO — field set tracks upstream spec; exhaustiveness is the contract"
)]
pub struct ChatCompletionChunk {
    /// Unique request identifier shared across all chunks for this request.
    pub id: String,
    /// Always `"chat.completion.chunk"` per the OpenAI spec.
    pub object: String,
    /// Unix timestamp (seconds) when the response stream started.
    pub created: u64,
    /// Model id echoed from the request.
    pub model: String,
    /// Content delta alternatives for this chunk; empty on the usage-summary chunk.
    pub choices: Vec<StreamChoice>,
    /// H4: populated only on the final usage-summary chunk emitted when
    /// `stream_options.include_usage == true`. All other chunks serialize
    /// this field as absent (not `null`) via `skip_serializing_if`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}
