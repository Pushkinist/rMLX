//! Anthropic Messages API routes:
//! - `POST /v1/messages` (streaming + non-streaming)
//!
//! Both schemas (OpenAI and Anthropic) feed the same `Generator` trait.
//!
//! Handler logic is split across:
//! - `route`    — `messages()` POST /v1/messages entry point, stop-reason helpers
//! - `blocking` — non-streaming generation path
//! - `streaming` — SSE streaming path, `BlockKind`, queue helpers
//!
//! Other submodules:
//! - `request`  — AnthropicContent, AnthropicSystem, AnthropicMessage,
//!   AnthropicTool, AnthropicToolChoice, MessagesRequest
//! - `response` — ContentBlock, AnthropicUsage, MessagesResponse
//! - `errors`   — Anthropic-typed HTTP error helpers (J3 OOM surface)

pub(crate) mod blocking;
pub(crate) mod errors;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod route;
pub(crate) mod streaming;

#[cfg(test)]
mod tests;

// ── Public API surface (frozen — matches lib.rs pub use) ─────────────────────

pub use request::{
    AnthropicContent, AnthropicMessage, AnthropicSystem, AnthropicTool, AnthropicToolChoice,
    MessagesRequest,
};
pub use response::{AnthropicUsage, ContentBlock, MessagesResponse};
pub(crate) use route::messages;

// ── Test-visible re-exports ───────────────────────────────────────────────────

#[cfg(test)]
pub(crate) use crate::tool_parser::ParsedToolCall;
#[cfg(test)]
pub(crate) use axum::http::StatusCode;
#[cfg(test)]
pub(crate) use axum::response::sse::Event;
#[cfg(test)]
pub(crate) use errors::engine_error_response;
#[cfg(test)]
pub(crate) use route::{map_stop_reason, select_anthropic_stop_reason, to_tool_use_block};
#[cfg(test)]
pub(crate) use serde_json::Value;
#[cfg(test)]
pub(crate) use streaming::{enqueue_tool_use_block, BlockKind};
