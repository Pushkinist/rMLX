//! OpenAI-compatible API surface.
//!
//! Submodules:
//! - `state`    — AppState, LoadedModel, SSD histogram, TTFT/ITL rings, error counters
//! - `request`  — ChatCompletionsRequest and all request-side wire types
//! - `response` — ChatCompletionsResponse and all response-side wire types
//! - `errors`   — Error helpers, sampling resolution, middleware
//! - `metrics`  — Metrics snapshot, Prometheus renderer, all /metrics handlers
//! - `handlers` — Route handlers: chat_completions, generate_blocking/streaming,
//!   model lifecycle (list/load/unload/status)

#![allow(
    clippy::cognitive_complexity,
    clippy::implicit_hasher,
    clippy::items_after_statements,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_fields_in_debug,
    clippy::struct_excessive_bools,
    clippy::too_many_lines,
    clippy::unnecessary_map_or,
    clippy::write_with_newline,
    trivial_casts
)]

pub(crate) mod chat;
pub(crate) mod errors;
pub(crate) mod generate;
pub(crate) mod handlers;
pub(crate) mod lifecycle;
pub(crate) mod metrics;
pub(crate) mod request;
pub(crate) mod response;
pub(crate) mod state;
pub(crate) mod streaming;

#[cfg(test)]
mod tests;

// ── Public API surface (frozen — matches lib.rs pub use) ─────────────────────

pub use errors::{compute_effective_timeout, timeout_mw};
pub use state::{
    register_ssd_prom_hooks, ApiErrorCategory, ApiErrorCounters, AppState, ItlSample, ItlStore,
    LoadedModel, ModelLoader, TtftSample, TtftStore, UnloadReason, ITL_RING_CAPACITY,
    TTFT_RING_CAPACITY,
};

// ── Crate-internal re-exports (used by lib.rs route registration) ─────────────

pub(crate) use handlers::{chat_completions, list_models, load_model, model_status, unload_model};
pub(crate) use metrics::{metrics_cache, metrics_prometheus, metrics_v1_summary};

// ── Cross-crate-module re-exports ─────────────────────────────────────────────
// Items from submodules referenced via `crate::openai::*` in sibling modules
// (anthropic.rs, engine.rs, metrics_drainer.rs).

pub(crate) use errors::{enforce_max_tokens_cap, resolve_request_id, resolve_sampling_params};
pub(crate) use response::{resolve_logprobs, ChatLogprobContent};
pub(crate) use state::{
    increment_ssd_evict_total, record_ssd_hydrate_obs, record_ssd_spill_obs, update_ssd_bytes_used,
};

// ── Test-visible re-exports ───────────────────────────────────────────────────
// Items that `use super::*;` in openai/tests.rs needs access to. These were
// visible via the monolithic openai.rs file-scope; now they must be explicit.
// Non-test items above are already visible without re-export; test-only items
// that aren't in the non-test set are gated here.

#[cfg(test)]
pub(crate) use errors::{engine_error_response, parse_logit_bias, SamplingSource};
#[cfg(test)]
pub(crate) use metrics::{gather_metrics, render_prometheus, MetricsSnapshot};
#[cfg(test)]
pub(crate) use request::{
    extract_audio_parts, extract_image_parts, ChatCompletionsRequest, ChatMessage, MessageContent,
    OwnedTplMessage, RequestToolCall, RequestToolCallFunction, ResponseFormat, ToolChoice,
};
#[cfg(test)]
pub(crate) use response::select_finish_reason;
#[cfg(test)]
pub(crate) use response::{
    to_response_tool_call, ChatCompletionChunk, DeltaContent, ResponseMessage, ToolCall, Usage,
};
#[cfg(test)]
pub(crate) use std::collections::HashMap;
// Re-export parking_lot aliases, std types, and axum types used in tests.
// These were available via `use super::*` in the monolithic file because
// the file-level `use` statements brought them into scope; now tests need
// them explicitly from the correct sources. We keep re-exports here so
// the test file retains `use super::*;` as the single import.
#[cfg(test)]
pub(crate) use crate::engine::GenerationToken;
#[cfg(test)]
pub(crate) use crate::engine::Generator;
#[cfg(test)]
pub(crate) use crate::engine::{NormalizedTool, NormalizedToolChoice};
#[cfg(test)]
pub(crate) use crate::registry::ModelRegistry;
#[cfg(test)]
pub(crate) use crate::session_cache::SessionCache;
#[cfg(test)]
pub(crate) use crate::tool_parser::ParsedToolCall;
#[cfg(test)]
pub(crate) use crate::tool_parser::{ToolCallFormat, ToolCallStreamParser};
#[cfg(test)]
pub(crate) use axum::extract::State;
#[cfg(test)]
pub(crate) use axum::http::{HeaderMap, HeaderValue, StatusCode};
#[cfg(test)]
pub(crate) use axum::response::Response;
#[cfg(test)]
pub(crate) use chat::{bare_json_to_tool_call, tool_choice_to_schema};
#[cfg(test)]
pub(crate) use metrics::percentile_u64;
#[cfg(test)]
pub(crate) use parking_lot::{Mutex as PLMutex, RwLock as PLRwLock};
#[cfg(test)]
pub(crate) use serde_json::Value;
#[cfg(test)]
pub(crate) use state::{SsdHistogram, HIST_BUCKETS_US};
#[cfg(test)]
pub(crate) use std::sync::Arc;
#[cfg(test)]
pub(crate) use std::time::Instant;
#[cfg(test)]
pub(crate) use streaming::{handle_streaming_token, StreamState};
