//! HTTP error-response helpers, OOM response, engine-error mapping, sampling
//! parameter resolution, and request-id utilities.

#![allow(unreachable_pub)]

use std::collections::HashMap;

use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

use crate::generation_config_io::GenerationConfig;

use super::state::ApiErrorCategory;

// ── Error helpers ─────────────────────────────────────────────────────────────

pub(crate) fn error_response(status: StatusCode, error_type: &str, message: &str) -> Response {
    let body = json!({
        "error": {
            "message": message,
            "type": error_type,
        }
    });
    (status, Json(body)).into_response()
}

pub(crate) fn bad_request(message: &str) -> Response {
    error_response(StatusCode::BAD_REQUEST, "invalid_request_error", message)
}

pub(crate) fn service_unavailable(message: &str) -> Response {
    error_response(
        StatusCode::SERVICE_UNAVAILABLE,
        "service_unavailable",
        message,
    )
}

pub(crate) fn internal_error(message: &str) -> Response {
    error_response(StatusCode::INTERNAL_SERVER_ERROR, "internal_error", message)
}

/// Validate a per-request `max_tokens` against the server-configured cap.
///
/// Returns the validated value when `requested <= cap`. Otherwise returns
/// a ready-to-send HTTP 400 `invalid_request_error` response with a message
/// of the form `"max_tokens N exceeds server cap M"`.
///
/// Sources for a resolved sampling parameter — used in trace logging only.
#[derive(Debug, Clone, Copy, PartialEq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "OpenAI server internal closed enum — variant set mirrors the four-tier sampling fallback; adding a tier requires reviewing all sampling resolution sites"
)]
pub enum SamplingSource {
    /// Parameter came from the inbound request body.
    Request,
    /// G4: `--default-temperature` server-startup flag.
    ServerDefault,
    /// Parameter fell back to the model's `generation_config.json` defaults.
    ModelDefaults,
    /// Parameter fell back to the compile-time hard-coded default.
    HardCoded,
}

impl SamplingSource {
    /// Return a short lowercase string label for trace logging.
    pub fn as_str(self) -> &'static str {
        match self {
            SamplingSource::Request => "request",
            SamplingSource::ServerDefault => "server_default",
            SamplingSource::ModelDefaults => "model_defaults",
            SamplingSource::HardCoded => "hard_coded",
        }
    }
}

/// Resolve all sampling parameters from the four-tier fallback chain:
/// **request > server default (G4) > model `generation_defaults` (A4) > hard-coded default**.
///
/// `temperature` and `top_p` also return a `SamplingSource` label for the
/// existing trace log. The remaining fields (A7.1) never fall back to model
/// defaults for the ones not in `generation_config.json` (`frequency_penalty`,
/// `presence_penalty`, `min_p`, `logit_bias`); `top_k` and `repetition_penalty`
/// do fall back to model defaults because `GenerationConfig` already parses them.
///
/// `server_default_temperature` — from `AppState::default_temperature` (G4
/// `--default-temperature` flag). `None` = absent (behaviour unchanged). When
/// `Some(t)` and the request omits temperature, `t` takes precedence over the A4
/// `generation_config.json` value. An explicit request `temperature` always wins.
///
/// `logit_bias_raw` — the JSON string-keyed map from the OpenAI request. Absent
/// → empty. Non-integer string keys → the caller must already have returned 400
/// before calling this function (parse is performed in the route handler).
///
/// Returns `(SamplingParams, temp_source, top_p_source)`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_sampling_params(
    req_temperature: Option<f32>,
    req_top_p: Option<f32>,
    req_top_k: Option<u32>,
    req_min_p: Option<f32>,
    req_repetition_penalty: Option<f32>,
    req_frequency_penalty: Option<f32>,
    req_presence_penalty: Option<f32>,
    logit_bias_parsed: Vec<(u32, f32)>,
    req_seed: Option<u64>,
    defaults: Option<&GenerationConfig>,
    server_default_temperature: Option<f32>,
) -> (
    crate::engine::SamplingParams,
    SamplingSource,
    SamplingSource,
) {
    let (temperature, temp_src) = if let Some(t) = req_temperature {
        (t, SamplingSource::Request)
    } else if let Some(t) = server_default_temperature {
        (t, SamplingSource::ServerDefault)
    } else if let Some(t) = defaults.and_then(|d| d.temperature) {
        (t, SamplingSource::ModelDefaults)
    } else {
        (1.0_f32, SamplingSource::HardCoded)
    };

    let (top_p, top_p_src) = if let Some(p) = req_top_p {
        (p, SamplingSource::Request)
    } else if let Some(p) = defaults.and_then(|d| d.top_p) {
        (p, SamplingSource::ModelDefaults)
    } else {
        (1.0_f32, SamplingSource::HardCoded)
    };

    // top_k: request > model defaults (GenerationConfig already has this field).
    let top_k = req_top_k
        .or_else(|| defaults.and_then(|d| d.top_k))
        .unwrap_or(0);

    // repetition_penalty: request > model defaults.
    let repetition_penalty = req_repetition_penalty
        .or_else(|| defaults.and_then(|d| d.repetition_penalty))
        .unwrap_or(1.0);

    // min_p, frequency_penalty, presence_penalty: request only (no model-defaults key).
    let min_p = req_min_p.unwrap_or(0.0);
    let frequency_penalty = req_frequency_penalty.unwrap_or(0.0);
    let presence_penalty = req_presence_penalty.unwrap_or(0.0);

    (
        crate::engine::SamplingParams {
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            frequency_penalty,
            presence_penalty,
            logit_bias: logit_bias_parsed,
            seed: req_seed,
            // resolved separately by the route handler from
            // `logprobs` / `top_logprobs` (no model-defaults fallback).
            top_logprobs_k: 0,
        },
        temp_src,
        top_p_src,
    )
}

/// Parse a JSON string-keyed logit_bias map into `Vec<(u32, f32)>`.
///
/// Returns `Err(String)` when any key is not a valid `u32` decimal string, or
/// when any bias value is non-finite. The error string is suitable for a 400
/// `invalid_request_error` message.
pub fn parse_logit_bias(raw: Option<&HashMap<String, f32>>) -> Result<Vec<(u32, f32)>, String> {
    let Some(map) = raw else {
        return Ok(Vec::new());
    };
    let mut out = Vec::with_capacity(map.len());
    for (k, &v) in map {
        let id: u32 = k
            .parse()
            .map_err(|_| format!("logit_bias key {k:?} is not a valid token id (u32)"))?;
        if !v.is_finite() {
            return Err(format!("logit_bias[{k}] = {v} is not finite"));
        }
        out.push((id, v));
    }
    Ok(out)
}

/// Shared between the OpenAI (`/v1/chat/completions`) and Anthropic
/// (`/v1/messages`) routes — both surfaces emit the same error shape
/// (`{"error": {"message": ..., "type": "invalid_request_error"}}`).
///
/// `_model_id` is accepted for symmetry with other server helpers but not
/// embedded in the message (the user already knows which model they POSTed to).
pub(crate) fn enforce_max_tokens_cap(
    requested: u32,
    cap: u32,
    _model_id: &str,
) -> Result<u32, Box<Response>> {
    if requested > cap {
        let msg = format!("max_tokens {requested} exceeds server cap {cap}");
        Err(Box::new(bad_request(&msg)))
    } else {
        Ok(requested)
    }
}

/// J3: build the typed-OOM error response (OpenAI error-spec compatible).
///
/// `type` is the stable automation key; `message` is human. Memory fields are
/// best-effort from J4 `read_proc_mem()` — on read failure they serialize as
/// `null`, the error path never fails because telemetry failed.
///
/// Status / `Retry-After` per phase:
/// - `LoadWeights` / `LoadKvCache` → **507** + `Retry-After: 5` (retryable
///   after eviction frees memory).
/// - `Generation` → **503**, no `Retry-After` (KV cache is corrupt past the
///   failure point — retrying the same stream is unsafe).
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
    use axum::http::HeaderMap;
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
    // TODO: metal_peak_alloc_mb — telemetry not yet built; emit
    // whatever the call site passed (today always None → null). Do not
    // invent a Metal peak number here.
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

/// Map an `rmlx_core::Error` to an HTTP error response.
///
/// `SmokeProbe` (NaN logits, broken snapshot) → 500 internal_error.
/// `Oom` (J3) → 507 / 503 typed body with `Retry-After` per phase.
/// Everything else → 503 service_unavailable (generator not ready / MLX error).
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

/// F8: classify an `rmlx_core::Error` into an `ApiErrorCategory`.
///
/// Mirrors the match arms in `engine_error_response` so the same logic
/// drives both the response shape and the counter.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
)]
pub(crate) fn engine_error_category(e: &rmlx_core::Error) -> ApiErrorCategory {
    use rmlx_core::OomPhase;
    match e {
        rmlx_core::Error::SmokeProbe(_) => ApiErrorCategory::Internal,
        rmlx_core::Error::Oom { phase, .. } => match phase {
            OomPhase::LoadWeights => ApiErrorCategory::OomLoad,
            OomPhase::LoadKvCache => ApiErrorCategory::OomKvCache,
            OomPhase::Generation => ApiErrorCategory::OomMidStream,
        },
        _ => ApiErrorCategory::Upstream,
    }
}

// ── F10: request-id resolution ────────────────────────────────────────────────

/// Resolve a correlation id for this request.
///
/// If the inbound `X-Request-Id` header is present, its value is reused (after
/// trimming, capping at 128 chars, and stripping non-printable-ASCII bytes).
/// Otherwise a fresh `req-<uuid-v4>` string is generated.
///
/// The resolved id is used for:
/// - The `id` field in the OpenAI / Anthropic response body.
/// - The `X-Request-Id` response header (both streaming and non-streaming).
/// - A `tracing::info_span!` that wraps the whole handler body.
pub(crate) fn resolve_request_id(headers: &HeaderMap) -> String {
    if let Some(val) = headers.get("x-request-id") {
        if let Ok(s) = val.to_str() {
            let sanitised: String = s
                .chars()
                .filter(|c| c.is_ascii() && !c.is_ascii_control())
                .take(128)
                .collect();
            let trimmed = sanitised.trim().to_owned();
            if !trimmed.is_empty() {
                return trimmed;
            }
        }
    }
    format!("req-{}", uuid::Uuid::new_v4().simple())
}

// ── Metrics helper ────────────────────────────────────────────────────────────

/// Emit one metric record to the sink if present; silently drop on error.
pub(crate) fn record_metric(
    state: &super::state::AppState,
    op: &str,
    unit: &str,
    value: f64,
    notes: &str,
    model_path: &str,
) {
    if let Some(sink) = &state.metrics {
        let m = rmlx_metrics::events::Measurement {
            model_path,
            quant_mode: "n/a",
            stage: "stage1",
            op,
            value_unit: unit,
            value,
            notes,
        };
        if let Err(e) = sink.record(&m) {
            tracing::warn!(error = %e, "failed to record metric");
        }
    }
}

// ── Per-request HTTP timeout middleware (A8) ──────────────────────────────────

/// Resolve the effective per-request timeout.
///
/// Rules:
/// - `max_secs == 0` → no timeout (returns `None`).
/// - Header absent → `Some(max_secs)`.
/// - Header present, parses to `u64 > 0` → `Some(min(header_value, max_secs))`.
/// - Header present, non-numeric / 0 / negative → `Err(400 response)`.
// Response is axum's standard error carrier for middleware; boxing adds noise with no benefit.
#[allow(clippy::result_large_err)]
pub fn compute_effective_timeout(
    headers: &HeaderMap,
    max_secs: u64,
) -> Result<Option<std::time::Duration>, Response> {
    const HEADER: &str = "x-request-timeout-seconds";

    if max_secs == 0 {
        return Ok(None);
    }

    match headers.get(HEADER) {
        None => Ok(Some(std::time::Duration::from_secs(max_secs))),
        Some(val) => {
            let raw = val.to_str().unwrap_or("");
            match raw.parse::<u64>() {
                Ok(n) if n > 0 => {
                    let effective = n.min(max_secs);
                    Ok(Some(std::time::Duration::from_secs(effective)))
                }
                _ => Err(bad_request(
                    "X-Request-Timeout-Seconds must be a positive integer",
                )),
            }
        }
    }
}

/// Axum middleware: per-request HTTP timeout (A8).
///
/// Wraps the downstream handler in `tokio::time::timeout(effective_dur)`.
/// This bounds the **whole** request — including SSE streams (when
/// `stream: true` is set on `/v1/chat/completions` or `/v1/messages`).
/// A running SSE stream that has already started emitting bytes will be
/// dropped when the deadline fires; the client will see a mid-stream
/// disconnect. This is the intended behaviour: it caps runaway generations.
///
/// Effective timeout: the lesser of the `X-Request-Timeout-Seconds` header
/// value and `AppState::max_timeout_secs` (the server-startup cap). See
/// `compute_effective_timeout` for the full resolution rules.
///
/// On timeout → HTTP 408 with `{"error":{"type":"timeout","message":"..."}}`.
pub async fn timeout_mw(
    axum::extract::State(state): axum::extract::State<super::state::AppState>,
    req: axum::extract::Request,
    next: axum::middleware::Next,
) -> Response {
    let effective = match compute_effective_timeout(req.headers(), state.max_timeout_secs) {
        Ok(d) => d,
        Err(err_resp) => {
            // Bad X-Request-Timeout-Seconds header value → 400 bad_request.
            state.error_counts.increment(ApiErrorCategory::BadRequest);
            return err_resp;
        }
    };

    match effective {
        None => next.run(req).await,
        Some(dur) => match tokio::time::timeout(dur, next.run(req)).await {
            Ok(resp) => resp,
            Err(_elapsed) => {
                let secs = dur.as_secs();
                state.error_counts.increment(ApiErrorCategory::Timeout);
                error_response(
                    StatusCode::REQUEST_TIMEOUT,
                    "timeout",
                    &format!("request exceeded {secs} second timeout"),
                )
            }
        },
    }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

pub(crate) fn unix_now() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}
