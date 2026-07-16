//! Anthropic-specific HTTP error response helpers.
//!
//! Error type strings differ from OpenAI: `invalid_request_error`,
//! `service_unavailable_error`, `internal_server_error`, plus the J3 typed-OOM
//! surface (`oom_during_load`, `oom_kv_cache`, `oom_mid_stream`).

#![allow(unreachable_pub)]

use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

pub(super) fn error_response(status: StatusCode, error_type: &str, message: &str) -> Response {
    let body = json!({
        "error": {
            "message": message,
            "type": error_type,
        }
    });
    (status, Json(body)).into_response()
}

pub(super) fn bad_request(message: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

pub(super) fn service_unavailable(message: &str) -> Response {
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable_error",
        message,
    )
}

pub(super) fn internal_error(message: &str) -> Response {
    error_response(
        StatusCode::INTERNAL_SERVER_ERROR,
        "internal_server_error",
        message,
    )
}

/// J3: typed-OOM error response (Anthropic error-type strings).
///
/// Mirrors the OpenAI mapping: same `type` strings (`oom_during_load`,
/// `oom_kv_cache`, `oom_mid_stream`), same 507 / 503 + `Retry-After` per
/// phase, same best-effort J4 memory fields.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
pub(super) fn oom_response(
    phase: rmlx_core::OomPhase,
    requested_bytes: Option<u64>,
    peak_alloc_mb: Option<u64>,
    msg: &str,
) -> Response {
    use rmlx_core::OomPhase;

    let (status, type_str, retry_after) = match phase {
        OomPhase::LoadWeights => (StatusCode::INSUFFICIENT_STORAGE, "oom_during_load", true),
        OomPhase::LoadKvCache => (StatusCode::INSUFFICIENT_STORAGE, "oom_kv_cache", true),
        OomPhase::Generation => (StatusCode::SERVICE_UNAVAILABLE, "oom_mid_stream", false),
    };

    // Best-effort process-memory snapshot (J4). Never fail the error path.
    let mem = rmlx_core::mach_mem::read_proc_mem().ok();
    let to_mb = |b: u64| b / (1024 * 1024);
    let process_rss_mb = mem
        .map(|m| json!(to_mb(m.rss_bytes)))
        .unwrap_or(json!(null));
    let phys_footprint_mb = mem
        .map(|m| json!(to_mb(m.phys_footprint_bytes)))
        .unwrap_or(json!(null));
    let compressed_mb = mem
        .map(|m| json!(to_mb(m.compressed_bytes)))
        .unwrap_or(json!(null));

    let body = json!({
           "error": {
               "type": type_str,
               "message": msg,
    // TODO: metal_peak_alloc_mb — telemetry not yet built;
    // emit whatever the call site passed (today always None → null).
               "peak_alloc_mb": peak_alloc_mb.map(|v| json!(v)).unwrap_or(json!(null)),
               "requested_bytes": requested_bytes.map(|v| json!(v)).unwrap_or(json!(null)),
               "process_rss_mb": process_rss_mb,
               "phys_footprint_mb": phys_footprint_mb,
               "compressed_mb": compressed_mb,
           }
       });

    let mut headers = HeaderMap::new();
    if retry_after {
        headers.insert("Retry-After", "5".parse().expect("static header value"));
    }
    (status, headers, Json(body)).into_response()
}

/// Map an `rmlx_core::Error` to an HTTP error response (Anthropic error types).
///
/// `SmokeProbe` (NaN logits) → 500 internal_server_error.
/// `Oom` (J3) → 507 / 503 typed body with `Retry-After` per phase.
/// Everything else → 503 service_unavailable_error.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
)]
pub(crate) fn engine_error_response(e: &rmlx_core::Error) -> Response {
    match e {
        rmlx_core::Error::SmokeProbe(msg) => {
            internal_error(&format!("NaN logits during generation: {msg}"))
        }
        rmlx_core::Error::Oom {
            phase,
            requested_bytes,
            peak_alloc_mb,
            msg,
        } => oom_response(*phase, *requested_bytes, *peak_alloc_mb, msg),
        _ => service_unavailable(&e.to_string()),
    }
}

/// The stable `type` key an engine error carries in the Anthropic error
/// envelope.
///
/// Mirrors the arms of [`engine_error_response`] above. A stream that dies
/// mid-flight cannot reuse that function — the HTTP status and headers are
/// already sent — but must still name the failure identically, so a client sees
/// the same `type` for the same fault whether it streamed the response or not.
///
/// Deliberately NOT shared with the OpenAI surface: these strings carry this
/// surface's `_error` suffix (`service_unavailable_error`,
/// `internal_server_error`) and diverge from OpenAI's by design. Type strings
/// are per-surface; only the OOM keys coincide. Keep this in step with
/// `engine_error_response`, not with the OpenAI mirror.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "mirrors engine_error_response: every unenumerated variant maps to the service_unavailable_error envelope there"
)]
pub(super) fn engine_error_type(e: &rmlx_core::Error) -> &'static str {
    use rmlx_core::OomPhase;
    match e {
        rmlx_core::Error::SmokeProbe(_) => "internal_server_error",
        rmlx_core::Error::Oom { phase, .. } => match phase {
            OomPhase::LoadWeights => "oom_during_load",
            OomPhase::LoadKvCache => "oom_kv_cache",
            OomPhase::Generation => "oom_mid_stream",
        },
        _ => "service_unavailable_error",
    }
}
