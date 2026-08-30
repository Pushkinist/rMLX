//! Bounded-input checks for adversarial chat-completion payloads.
//!
//! Implements the "Cloudflare lesson": fail FAST on over-large inputs,
//! before allocating tokenisation state. Each bound has one check function
//! that logs a structured `tracing::error!` and returns a typed error on
//! violation. [`BoundError`] implements [`axum::response::IntoResponse`] so
//! callers can return it directly or use `?` in `Result<Response, BoundError>`.
//!
//! # Limits (all sourced below)
//!
//! | Constant | Default | Rationale |
//! |---|---|---|
//! | [`MAX_MESSAGES`] | 4096 | Exceeds OpenAI (1000) and Anthropic (≈100k tokens worth) published caps; defensive upper bound |
//! | [`MAX_TOOL_CALLS`] | 256 | No published per-message cap; 256 covers any realistic agent loop |
//! | [`MAX_TOOLS`] | 128 | OpenAI publishes 128 tools per request as of 2025; match it |
//! | [`MAX_CONTENT_PARTS`] | 1024 | No published cap; 1024 allows large vision batches while bounding memory |
//! | [`MAX_INPUT_AUDIO_BYTES`] | 16 MiB | Practical voice-turn limit; larger uploads should use the files API |
//! | [`MAX_TOTAL_INPUT_TOKENS_ESTIMATE`] | 1 048 576 | 1M-token coarse bound (bytes ÷ 3); prevents 100 GB JSON payloads |
//! | [`MAX_COMPLETION_TOKENS`] | 1 048 576 | A completion cannot outgrow the context holding it; sizes the generator pre-allocation |

use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

// ── Bound constants ───────────────────────────────────────────────────────────

/// Maximum number of messages in a single chat-completion request.
///
/// OpenAI's published context window caps effectively cap messages well below
/// 1000; Anthropic's documentation does not publish a per-request message limit
/// but their context window (~200k tokens) also implies far fewer messages.
/// 4096 is an explicit defensive upper bound that no legitimate client will hit.
pub const MAX_MESSAGES: usize = 4096;

/// Maximum number of `tool_calls` entries on a single assistant message.
///
/// No vendor publishes a per-message tool-calls cap as of 2025. 256 accommodates
/// the most aggressive parallel-tool-calling agent loops seen in the wild.
pub const MAX_TOOL_CALLS: usize = 256;

/// Maximum number of tool definitions in the request-level `tools` array.
///
/// OpenAI published a limit of 128 tools per request (2025 API reference).
/// Anthropic's documentation does not publish a number; matching OpenAI's cap
/// is a safe defensive default for both surfaces.
pub const MAX_TOOLS: usize = 128;

/// Maximum number of parts in a single message's `content` array.
///
/// No vendor publishes a per-message part limit. 1024 is chosen to allow large
/// multi-image/audio vision batches while bounding the per-request allocation.
pub const MAX_CONTENT_PARTS: usize = 1024;

/// Maximum base64-decoded bytes for a single audio input part (16 MiB).
///
/// Practical voice-turn limit. Audio longer than 16 MiB should be submitted
/// through a files API, not inlined in a JSON payload.
pub const MAX_INPUT_AUDIO_BYTES: usize = 16 * 1024 * 1024;

/// Coarse pre-tokeniser token estimate upper bound (1M tokens).
///
/// Computed as `total_text_bytes / 3`. The ÷3 factor assumes ~3 bytes per token
/// on average (UTF-8 prose is slightly above; code is slightly below). This
/// prevents a pathological 3 GB JSON payload from reaching the tokeniser.
pub const MAX_TOTAL_INPUT_TOKENS_ESTIMATE: usize = 1_048_576;

/// Structural ceiling on a single request's `max_tokens` (1 Mi tokens).
///
/// `max_tokens` sizes generator pre-allocations — `Vec::with_capacity(n_tokens)`
/// for the per-step timestamps and the per-arch `steps` vector — so an
/// unbounded value is an unbounded allocation. A completion can never outgrow
/// the context window that holds it, and 1 Mi is above every context window in
/// scope; it is the same ceiling the input side uses
/// ([`MAX_TOTAL_INPUT_TOKENS_ESTIMATE`]) and that the Anthropic route reports
/// `ctx_max` against. `--max-tokens-cap` can lower it, never raise it.
pub const MAX_COMPLETION_TOKENS: u32 = 1_048_576;

// ── Error type ────────────────────────────────────────────────────────────────

/// Returned when any bounded-input limit is exceeded.
///
/// Implements [`axum::response::IntoResponse`] — call `.into_response()` or
/// return via `?` in a `Result<Response, BoundError>`. All violations emit a
/// structured `tracing::error!` **before** returning from the check function so
/// the log is always present even if the caller discards the error.
#[derive(Debug, thiserror::Error)]
#[allow(
    clippy::exhaustive_enums,
    reason = "bounds module internal closed enum — variant set mirrors the three check categories (items, bytes, generic limit); adding a category requires reviewing all check functions and callers"
)]
pub enum BoundError {
    /// An array field exceeded its item count limit.
    #[error("input too large: field `{field}` has {got} items, max is {max}")]
    ItemsExceeded {
        /// Name of the violating field (e.g. `"messages"`).
        field: &'static str,
        /// Actual item count from the request.
        got: usize,
        /// Configured maximum.
        max: usize,
    },
    /// A byte-size field exceeded its byte limit.
    #[error("input too large: field `{field}` has {got} bytes, max is {max}")]
    BytesExceeded {
        /// Name of the violating field (e.g. `"input_audio"`).
        field: &'static str,
        /// Actual byte count.
        got: usize,
        /// Configured maximum.
        max: usize,
    },
    /// A generic numeric limit was exceeded (e.g. an estimated token count).
    #[error("input too large: field `{field}` has {got}, max is {max}")]
    LimitExceeded {
        /// Name of the violating field (e.g. `"total_input_tokens_estimate"`).
        field: &'static str,
        /// Actual value from the request.
        got: usize,
        /// Configured maximum.
        max: usize,
    },
}

impl BoundError {
    /// HTTP status code for this error.
    ///
    /// All bound violations use 413 Payload Too Large per RFC 9110 §15.5.14.
    pub fn status_code(&self) -> StatusCode {
        StatusCode::PAYLOAD_TOO_LARGE
    }

    /// Field name for the structured JSON error body.
    fn field(&self) -> &'static str {
        match self {
            BoundError::ItemsExceeded { field, .. }
            | BoundError::BytesExceeded { field, .. }
            | BoundError::LimitExceeded { field, .. } => field,
        }
    }

    /// Maximum for the structured JSON error body.
    fn max(&self) -> usize {
        match self {
            BoundError::ItemsExceeded { max, .. }
            | BoundError::BytesExceeded { max, .. }
            | BoundError::LimitExceeded { max, .. } => *max,
        }
    }
}

impl IntoResponse for BoundError {
    /// Convert to an axum HTTP response with a structured JSON body.
    ///
    /// Body shape:
    /// ```json
    /// {"error": {"code": "input_too_large", "message": "...", "field": "messages", "max": 4096}}
    /// ```
    fn into_response(self) -> Response {
        let body = json!({
            "error": {
                "code": "input_too_large",
                "message": self.to_string(),
                "field": self.field(),
                "max": self.max(),
            }
        });
        (self.status_code(), Json(body)).into_response()
    }
}

// ── Check functions ───────────────────────────────────────────────────────────

/// Check that `messages` does not exceed [`MAX_MESSAGES`].
///
/// Logs a structured error and returns [`BoundError`] on violation.
/// Call this before any tokenisation or model-load work.
pub fn check_messages(n: usize) -> Result<(), BoundError> {
    if n > MAX_MESSAGES {
        tracing::error!(
            target: "rmlx::server::bounds",
            got = n,
            max = MAX_MESSAGES,
            field = "messages",
            "bounded-input limit exceeded"
        );
        return Err(BoundError::ItemsExceeded {
            field: "messages",
            got: n,
            max: MAX_MESSAGES,
        });
    }
    Ok(())
}

/// Check that a per-assistant-message `tool_calls` array does not exceed
/// [`MAX_TOOL_CALLS`].
///
/// `msg_idx` is included in the tracing event for diagnostics.
pub fn check_tool_calls(n: usize, msg_idx: usize) -> Result<(), BoundError> {
    if n > MAX_TOOL_CALLS {
        tracing::error!(
            target: "rmlx::server::bounds",
            got = n,
            max = MAX_TOOL_CALLS,
            field = "tool_calls",
            msg_idx,
            "bounded-input limit exceeded"
        );
        return Err(BoundError::ItemsExceeded {
            field: "tool_calls",
            got: n,
            max: MAX_TOOL_CALLS,
        });
    }
    Ok(())
}

/// Check that the request-level `tools` definitions array does not exceed
/// [`MAX_TOOLS`].
pub fn check_tools(n: usize) -> Result<(), BoundError> {
    if n > MAX_TOOLS {
        tracing::error!(
            target: "rmlx::server::bounds",
            got = n,
            max = MAX_TOOLS,
            field = "tools",
            "bounded-input limit exceeded"
        );
        return Err(BoundError::ItemsExceeded {
            field: "tools",
            got: n,
            max: MAX_TOOLS,
        });
    }
    Ok(())
}

/// Check that a per-message multimodal `content` array does not exceed
/// [`MAX_CONTENT_PARTS`].
///
/// `msg_idx` is included in the tracing event for diagnostics.
pub fn check_content_parts(n: usize, msg_idx: usize) -> Result<(), BoundError> {
    if n > MAX_CONTENT_PARTS {
        tracing::error!(
            target: "rmlx::server::bounds",
            got = n,
            max = MAX_CONTENT_PARTS,
            field = "content_parts",
            msg_idx,
            "bounded-input limit exceeded"
        );
        return Err(BoundError::ItemsExceeded {
            field: "content_parts",
            got: n,
            max: MAX_CONTENT_PARTS,
        });
    }
    Ok(())
}

/// Check that a single audio input part does not exceed [`MAX_INPUT_AUDIO_BYTES`]
/// **after** base64 decoding.
///
/// `decoded_bytes` is the byte count of the decoded audio data (not the
/// base64-encoded string length). `part_idx` is included for diagnostics.
pub fn check_input_audio_bytes(decoded_bytes: usize, part_idx: usize) -> Result<(), BoundError> {
    if decoded_bytes > MAX_INPUT_AUDIO_BYTES {
        tracing::error!(
            target: "rmlx::server::bounds",
            got = decoded_bytes,
            max = MAX_INPUT_AUDIO_BYTES,
            field = "input_audio",
            part_idx,
            "bounded-input limit exceeded"
        );
        return Err(BoundError::BytesExceeded {
            field: "input_audio",
            got: decoded_bytes,
            max: MAX_INPUT_AUDIO_BYTES,
        });
    }
    Ok(())
}

/// Coarse pre-tokeniser token estimate check.
///
/// `total_text_bytes` is the sum of `.len()` across all message text content.
/// The estimate is `total_text_bytes / 3` (≈3 bytes per token for prose/code
/// mixed UTF-8). Returns an error if the estimate exceeds
/// [`MAX_TOTAL_INPUT_TOKENS_ESTIMATE`].
///
/// The estimate `total_text_bytes / 3` uses integer division (floors);
/// payloads up to `MAX_TOTAL_INPUT_TOKENS_ESTIMATE * 3 + 2` bytes pass.
///
/// This is a **pre-tokeniser sanity bound** only — it does not replace the
/// model's context-length check (which uses real token counts). Its purpose is
/// to reject pathological payloads (e.g. a 3 GB text field) before they reach
/// the tokeniser.
pub fn check_total_input_tokens_estimate(total_text_bytes: usize) -> Result<(), BoundError> {
    let estimate = total_text_bytes / 3;
    if estimate > MAX_TOTAL_INPUT_TOKENS_ESTIMATE {
        tracing::error!(
            target: "rmlx::server::bounds",
            got = estimate,
            max = MAX_TOTAL_INPUT_TOKENS_ESTIMATE,
            field = "total_input_tokens_estimate",
            total_text_bytes,
            "bounded-input limit exceeded"
        );
        return Err(BoundError::LimitExceeded {
            field: "total_input_tokens_estimate",
            got: estimate,
            max: MAX_TOTAL_INPUT_TOKENS_ESTIMATE,
        });
    }
    Ok(())
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "bounds_tests.rs"]
mod tests;
