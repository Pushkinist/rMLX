//! Model lifecycle route handlers.
//!
//! - `list_models` — GET /v1/models
//! - `load_model` — POST /v1/models/{id}/load
//! - `unload_model` — POST /v1/models/{id}/unload
//! - `model_status` — GET /v1/models/{id}/status

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde::Deserialize;
use serde_json::json;
use tracing::info;

use super::errors::{error_response, service_unavailable, unix_now};
use super::state::{ApiErrorCategory, AppState};
use crate::keep_alive::{policy_from_request_field, KeepAlivePolicy};

// ── /v1/models/{id}/load — request body ──────────────────────────────────────

/// Optional JSON body for `POST /v1/models/{id}/load`.
///
/// This is the **native rMLX surface** for the keep-alive field: the
/// per-request `keep_alive` integer is honored here. OpenAI/Anthropic
/// compat routes (`/v1/chat/completions`, `/v1/messages`, `/v1/embeddings`,
/// `/v1/audio/*`) intentionally do not parse this field — they only reset
/// the timer on use — to match the wider ecosystem (cf. ollama#11458).
///
/// Body shape (Ollama/LM-Studio compatible):
/// ```json
/// { "keep_alive": -1 }   // pin forever
/// { "keep_alive": 0 }    // unload after this response
/// { "keep_alive": 900 }  // idle for 15 min then unload
/// ```
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal native-route DTO — adding a field requires updating the load handler"
)]
pub(crate) struct LoadBody {
    /// Per-request keep-alive override. Negative = pin, `0` = unload-after,
    /// `N>0` = idle TTL in seconds. Absent = use the slot's current policy
    /// (which is in turn `RMLX_KEEP_ALIVE` env > `--idle-timeout-secs` flag
    /// > 15-min default).
    pub keep_alive: Option<i64>,
}

// ── Route: GET /v1/models ─────────────────────────────────────────────────────

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(crate) async fn list_models(State(state): State<AppState>) -> Response {
    // Snapshot resident-model timing under the read lock, then release.
    let now_unix = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let resident: std::collections::HashMap<String, (u64, u64)> = {
        let slots = state.slots.read();
        slots
            .iter()
            .map(|m| {
                (
                    m.id.clone(),
                    (
                        now_unix.saturating_sub(m.loaded_at.elapsed().as_secs()),
                        now_unix.saturating_sub(m.last_used.elapsed().as_secs()),
                    ),
                )
            })
            .collect()
    };

    let data: Vec<serde_json::Value> = state
        .registry
        .list()
        .into_iter()
        .map(|e| {
            let mut obj = json!({
                "id": e.id,
                "object": "model",
                "created": 0,
                "owned_by": "rmlx",
                "loaded": resident.contains_key(&e.id),
            });
            if let Some((loaded_at, last_used)) = resident.get(&e.id) {
                obj["loaded_at"] = (*loaded_at).into();
                obj["last_used"] = (*last_used).into();
            }
            obj
        })
        .collect();

    let body = json!({
        "object": "list",
        "data": data,
    });

    (StatusCode::OK, Json(body)).into_response()
}

// ── Route: POST /v1/models/{id}/load ─────────────────────────────────────────

/// `POST /v1/models/{id}/load` — load the registered model into the slot.
///
/// Accepts an optional `keep_alive` body field (i64). When present,
/// the loaded slot's keep-alive policy is overridden for this request and
/// every subsequent reset until a different policy is supplied.
pub(crate) async fn load_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
    body: Option<Json<LoadBody>>,
) -> Response {
    // Check registry first (fast path, no lock on slot).
    if state.registry.get(&id).is_none() {
        state.error_counts.increment(ApiErrorCategory::NotFound);
        return error_response(
            StatusCode::NOT_FOUND,
            "model_not_found",
            &format!("model '{id}' not found in registry"),
        );
    }

    let override_policy: Option<KeepAlivePolicy> =
        body.and_then(|Json(b)| policy_from_request_field(b.keep_alive));

    // ensure_loaded acquires slot lock, does implicit unload+load if needed.
    match state.ensure_loaded(&id) {
        Ok(_pair) => {
            info!(model_id = %id, "load: model ready in slot");
            // If the body supplied a keep_alive override, rewrite
            // the slot's policy and re-arm the timer accordingly.
            if let Some(policy) = override_policy {
                tracing::info!(
                    model_id = %id,
                    policy = ?policy,
                    ttl_secs = policy.ttl_secs_for_log(),
                    "load: applying per-request keep_alive override"
                );
                state.reset_keep_alive(&id, Some(policy));
            }
            (StatusCode::OK, Json(json!({"ok": true, "model": id}))).into_response()
        }
        Err(e) => {
            tracing::error!(model_id = %id, error = %e, "load: failed");
            state.error_counts.increment(ApiErrorCategory::Upstream);
            service_unavailable(&e)
        }
    }
}

// ── Route: POST /v1/models/{id}/unload ───────────────────────────────────────

pub(crate) async fn unload_model(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    if state.unload(&id) {
        info!(model_id = %id, "unload: model evicted from slot");
        (StatusCode::OK, Json(json!({"ok": true}))).into_response()
    } else {
        info!(model_id = %id, "unload: model was not loaded");
        (
            StatusCode::NOT_FOUND,
            Json(json!({"ok": false, "message": format!("model '{id}' is not loaded")})),
        )
            .into_response()
    }
}

// ── Route: GET /v1/models/{id}/status ────────────────────────────────────────

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
pub(crate) async fn model_status(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Response {
    if state.registry.get(&id).is_none() {
        state.error_counts.increment(ApiErrorCategory::NotFound);
        return error_response(
            StatusCode::NOT_FOUND,
            "model_not_found",
            &format!("model '{id}' not found in registry"),
        );
    }

    {
        let slots = state.slots.read();
        if let Some(loaded) = slots.iter().find(|m| m.id == id) {
            let now_secs = unix_now();
            let loaded_ago = loaded.loaded_at.elapsed().as_secs();
            let used_ago = loaded.last_used.elapsed().as_secs();
            let body = json!({
                "id": id,
                "loaded": true,
                "loaded_at": now_secs.saturating_sub(loaded_ago),
                "last_used": now_secs.saturating_sub(used_ago),
                "idle_secs": used_ago,
            });
            return (StatusCode::OK, Json(body)).into_response();
        }
    }

    let body = json!({
        "id": id,
        "loaded": false,
        "loaded_at": null,
        "last_used": null,
        "idle_secs": null,
    });
    (StatusCode::OK, Json(body)).into_response()
}
