//! `LoggedJson<T>` — a drop-in replacement for `axum::Json<T>` that emits a
//! `tracing::warn!` when the request body fails to deserialise.
//!
//! On success the extractor is transparent: it returns `T` exactly as
//! `axum::Json<T>` would. On failure it logs
//!
//! ```text
//! WARN route=<route> error=<serde error> body_snippet=<first 2 KiB, lossy UTF-8>
//! ```
//!
//! and then returns the **same** `JsonRejection` axum's built-in `Json<T>`
//! would have returned, so the wire response (status 422 / 400, body) is
//! byte-identical to the previous behaviour.
//!
//! No new crates are needed: this uses only `axum`, `serde_json`, and
//! `tracing`, all already in the workspace.

use axum::body::Bytes;
use axum::extract::rejection::JsonRejection;
use axum::extract::{FromRequest, Request};
use axum::http::header;
use axum::Json;
use serde::de::DeserializeOwned;

/// Maximum number of bytes of the raw body to include in the warning log.
///
/// Prompts can be arbitrarily large; we never want to log the full body.
pub(crate) const SNIPPET_LIMIT: usize = 2048;

/// Extractor wrapping `axum::Json<T>` with rejection logging.
///
/// Usage:
///
/// ```ignore
/// async fn my_handler(
/// State(state): State<AppState>,
/// LoggedJson(req): LoggedJson<MyRequest>,
/// ) -> Response { … }
/// ```
///
/// On a JSON deserialisation failure the extractor emits a `tracing::warn!`
/// with the route URI, the serde error message, and a length-bounded snippet
/// of the offending body (first `SNIPPET_LIMIT` bytes, lossy UTF-8).
///
/// Content-type gating is preserved: a request without
/// `Content-Type: application/json` is delegated back to axum's built-in
/// `Json<T>::from_request`, which returns the canonical 415 rejection.
pub(crate) struct LoggedJson<T>(pub T);

impl<T, S> FromRequest<S> for LoggedJson<T>
where
    T: DeserializeOwned,
    S: Send + Sync,
{
    type Rejection = JsonRejection;

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    async fn from_request(req: Request, state: &S) -> Result<Self, Self::Rejection> {
        // Split off parts so we can inspect headers without consuming the body,
        // then reconstruct the request for `Bytes::from_request`.
        let (parts, body) = req.into_parts();

        let uri = parts.uri.path().to_owned();

        // Content-type gate: if the header is absent or not JSON, delegate the
        // whole request back to `Json::from_request`. That is the canonical
        // path for the 415 rejection; we cannot construct `MissingJsonContentType`
        // from outside axum (it is `#[non_exhaustive]`).
        if !is_json_content_type(&parts.headers) {
            let req = Request::from_parts(parts, body);
            return Json::<T>::from_request(req, state)
                .await
                .map(|Json(v)| LoggedJson(v));
        }

        // Reconstruct and buffer the body. `Bytes::from_request` is what
        // axum's own `Json` extractor uses and applies the correct body limits.
        let req = Request::from_parts(parts, body);
        let bytes = Bytes::from_request(req, state)
            .await
            .map_err(JsonRejection::from)?;

        // Delegate to `Json::from_bytes` — exact same serde pipeline as
        // axum's built-in extractor, same rejection variants and status codes.
        match Json::<T>::from_bytes(&bytes) {
            Ok(Json(value)) => Ok(LoggedJson(value)),
            Err(rejection) => {
                // Bounded, lossy UTF-8 snippet for the log record.
                let raw = &bytes[..bytes.len().min(SNIPPET_LIMIT)];
                let snippet = String::from_utf8_lossy(raw);

                tracing::warn!(
                    route = %uri,
                    error = %rejection,
                    body_snippet = %snippet,
                    "JSON deserialisation rejected"
                );

                Err(rejection)
            }
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Returns `true` when `Content-Type` is `application/json` or
/// `application/*+json` (e.g. `application/cloudevents+json`).
///
/// Mirrors the same check axum's private `json_content_type` function does,
/// without depending on the `mime` crate.
fn is_json_content_type(headers: &axum::http::HeaderMap) -> bool {
    let Some(ct) = headers.get(header::CONTENT_TYPE) else {
        return false;
    };
    let Ok(ct_str) = ct.to_str() else {
        return false;
    };
    // Strip parameters (e.g. "; charset=utf-8") and trim whitespace.
    let media_type = ct_str.split(';').next().unwrap_or("").trim();
    if let Some(subtype) = media_type.strip_prefix("application/") {
        subtype == "json" || subtype.ends_with("+json")
    } else {
        false
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "logged_json_tests.rs"]
mod tests;
