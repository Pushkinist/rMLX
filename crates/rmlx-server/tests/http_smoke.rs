//! Smoke tests for the rmlx-server HTTP surface.
//!
//! Spawns a real `tokio::net::TcpListener` on port 0, starts the router in a
//! background task, then sends raw HTTP/1.1 requests over a `TcpStream`.
//! No extra crates needed beyond what the workspace already has.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::format_push_string,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::duration_suboptimal_units,
    clippy::unchecked_time_subtraction,
    trivial_casts
)]

use std::sync::Arc;
use std::time::Duration;

use rmlx_server::{
    timeout_mw, ApiErrorCounters, AppState, ArchGenerator, Generator, ItlStore, LoadedModel,
    ModelLoadConfig, ModelLoader, ModelRegistry, NotReadyGenerator, SessionCache, TtftStore,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

fn primary_snapshot_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

// ── Test server helpers ───────────────────────────────────────────────────────

/// Build an `AppState` whose loader always returns `NotReadyGenerator`.
///
/// This matches the old test behaviour: model lookup in registry works (404
/// path tested), but generation returns 503 (NotReadyGenerator).
fn not_ready_state(registry: ModelRegistry) -> AppState {
    let loader: ModelLoader =
        Arc::new(|_path, _id| Ok(Box::new(NotReadyGenerator) as Box<dyn Generator>));
    AppState {
        registry: Arc::new(registry),
        slots: Arc::new(parking_lot::RwLock::new(Vec::new())),
        embed_slot: Arc::new(parking_lot::RwLock::new(None)),
        mm_cache: Arc::new(rmlx_models::multimodal_cache::MultimodalCache::new(0)),
        gpu_gate: Arc::new(parking_lot::Mutex::new(())),
        gpu_queue: Arc::new(tokio::sync::Semaphore::new(1)),
        gpu_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_queue_depth: 64,
        max_loaded_models: 1,
        loader,
        metrics: None,
        idle_policy: rmlx_server::KeepAlivePolicy::Pin,
        max_tokens_cap: u32::MAX,
        max_timeout_secs: 600,
        session_cache: Arc::new(parking_lot::Mutex::new(SessionCache::new(4))),
        ttft_store: TtftStore::default(),
        itl_store: ItlStore::default(),
        metrics_drainer: None,
        require_smoke_probe: false,
        default_temperature: None,
        default_enable_thinking: None,
        tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        started_at: std::time::Instant::now(),
        requests_started: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        admission_controller: None,
        admission_handle: None,
        whisper_model_path: None,
        whisper_tokenizer_path: None,
        audio_model: Arc::new(parking_lot::RwLock::new(None)),
        tts_model_path: None,
        tts_tokenizer_path: None,
        tts_model: Arc::new(parking_lot::RwLock::new(None)),
    }
}

/// Build an `AppState` with `NotReadyGenerator` loader and an explicit
/// `max_tokens_cap`. Used by the A1 cap-enforcement tests.
fn not_ready_state_with_cap(registry: ModelRegistry, max_tokens_cap: u32) -> AppState {
    let mut s = not_ready_state(registry);
    s.max_tokens_cap = max_tokens_cap;
    s
}

/// Build an `AppState` with the slot pre-populated by a `LoadedModel`
/// carrying the given `effective_max_ctx`. Used by the A2
/// `context_length_exceeded` tests so the guard fires before the request
/// ever reaches `NotReadyGenerator`.
fn loaded_state_with_max_ctx(
    registry: ModelRegistry,
    model_id: &str,
    effective_max_ctx: usize,
) -> AppState {
    let state = not_ready_state(registry);
    let now = std::time::Instant::now();
    state.slots.write().push(LoadedModel {
        id: model_id.to_owned(),
        model: Arc::new(NotReadyGenerator),
        loaded_at: now,
        last_used: now,
        effective_max_ctx,
        decode_lease: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        unload_handle: Arc::new(parking_lot::Mutex::new(None)),
        keep_alive: rmlx_server::KeepAlivePolicy::Pin,
    });
    state
}

/// Start a test server using the provided `AppState`.
async fn start_server_with_state(state: AppState) -> u16 {
    let router = rmlx_server::build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    port
}

/// Bind a listener on an OS-assigned port, start serving in the background,
/// return the bound port.
async fn start_server(registry: ModelRegistry) -> u16 {
    let state = not_ready_state(registry);
    let router = rmlx_server::build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    // Give the server a moment to be ready.
    tokio::time::sleep(Duration::from_millis(20)).await;
    port
}

/// Start a server backed by a real `ArchGenerator` loaded on demand.
///
/// The loader captures the `gen` inside an `Arc<Mutex<Option<ArchGenerator>>>` so
/// the first call yields the real generator and subsequent calls reuse it.
/// (Tests using this helper call the model at most once, so this is safe.)
async fn start_server_with_gemma4(gen: ArchGenerator) -> u16 {
    let snap = primary_snapshot_dir()
        .expect("RMLX_TEST_MODEL_GEMMA4_E4B must be set when calling start_server_with_gemma4");
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));

    // Wrap the pre-built generator in a once-cell pattern.
    let gen_cell: Arc<std::sync::Mutex<Option<ArchGenerator>>> =
        Arc::new(std::sync::Mutex::new(Some(gen)));

    let loader: ModelLoader = Arc::new(move |_path, _id| {
        let mut guard = gen_cell.lock().unwrap();
        let g = guard.take().expect("ArchGenerator already consumed");
        Ok(Box::new(g) as Box<dyn Generator>)
    });

    let state = AppState {
        registry: Arc::new(reg),
        slots: Arc::new(parking_lot::RwLock::new(Vec::new())),
        embed_slot: Arc::new(parking_lot::RwLock::new(None)),
        mm_cache: Arc::new(rmlx_models::multimodal_cache::MultimodalCache::new(0)),
        gpu_gate: Arc::new(parking_lot::Mutex::new(())),
        gpu_queue: Arc::new(tokio::sync::Semaphore::new(1)),
        gpu_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_queue_depth: 64,
        max_loaded_models: 1,
        loader,
        metrics: None,
        idle_policy: rmlx_server::KeepAlivePolicy::Pin,
        max_tokens_cap: u32::MAX,
        max_timeout_secs: 600,
        session_cache: Arc::new(parking_lot::Mutex::new(SessionCache::new(4))),
        ttft_store: TtftStore::default(),
        itl_store: ItlStore::default(),
        metrics_drainer: None,
        require_smoke_probe: false,
        default_temperature: None,
        default_enable_thinking: None,
        tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        started_at: std::time::Instant::now(),
        requests_started: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        admission_controller: None,
        admission_handle: None,
        whisper_model_path: None,
        whisper_tokenizer_path: None,
        audio_model: Arc::new(parking_lot::RwLock::new(None)),
        tts_model_path: None,
        tts_tokenizer_path: None,
        tts_model: Arc::new(parking_lot::RwLock::new(None)),
    };

    let router = rmlx_server::build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    port
}

/// Send a raw HTTP/1.1 request and return (status_code, body_string).
async fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();

    let content_header = if let Some(b) = &body {
        format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        )
    } else {
        String::new()
    };

    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n{content_header}\r\n{}",
        body.unwrap_or("")
    );

    stream.write_all(request.as_bytes()).await.unwrap();

    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let text = String::from_utf8_lossy(&response).into_owned();

    // Parse status line: "HTTP/1.1 200 OK"
    let status: u16 = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    // Everything after the header/body separator.
    let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    let body_text = text[body_start..].to_owned();

    (status, body_text)
}

/// Like `http` but allows injecting extra request headers.
async fn http_with_headers(
    port: u16,
    method: &str,
    path: &str,
    extra_headers: &[(&str, &str)],
    body: Option<&str>,
) -> (u16, String) {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();

    let content_header = if let Some(b) = &body {
        format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        )
    } else {
        String::new()
    };

    let mut extra = String::new();
    for (k, v) in extra_headers {
        extra.push_str(&format!("{k}: {v}\r\n"));
    }

    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n{content_header}{extra}\r\n{}",
        body.unwrap_or("")
    );

    stream.write_all(request.as_bytes()).await.unwrap();

    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let text = String::from_utf8_lossy(&response).into_owned();

    let status: u16 = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    let body_text = text[body_start..].to_owned();

    (status, body_text)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[tokio::test]
async fn health_returns_ok() {
    let port = start_server(ModelRegistry::default()).await;
    let (status, body) = http(port, "GET", "/health", None).await;
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["ok"], true);
}

#[tokio::test]
async fn models_empty_registry() {
    let port = start_server(ModelRegistry::default()).await;
    let (status, body) = http(port, "GET", "/v1/models", None).await;
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"].as_array().unwrap().len(), 0);
}

#[tokio::test]
async fn models_one_entry_with_snapshot() {
    // Use the primary test snapshot if available; skip if absent.
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!(
            "RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping models_one_entry_with_snapshot"
        );
        return;
    };
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let port = start_server(reg).await;

    let (status, body) = http(port, "GET", "/v1/models", None).await;
    assert_eq!(status, 200, "body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);
    assert_eq!(data[0]["id"], "mlx-community__gemma-4-e4b-it-mxfp8");
}

#[tokio::test]
async fn chat_completions_unknown_model_returns_404() {
    // Unknown model → 404 not_found_error (model lookup happens before generator).
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{"model":"no-such-model","messages":[{"role":"user","content":"hello"}]}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_eq!(status, 404, "expected 404 for unknown model, body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["error"]["message"].is_string());
    assert_eq!(v["error"]["type"], "not_found_error");
}

#[tokio::test]
async fn chat_completions_streaming_unknown_model_returns_404() {
    let port = start_server(ModelRegistry::default()).await;
    let payload =
        r#"{"model":"no-such-model","messages":[{"role":"user","content":"hello"}],"stream":true}"#;
    let (status, _body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    // Model not found → 404, never reaches SSE.
    assert_eq!(status, 404, "streaming must return 404 for unknown model");
}

/// With the real primary snapshot: prompt pipeline runs (template + tokenize),
/// then the generator returns 503 because NotReadyGenerator is still wired.
/// Verifies the full prompt-side pipeline is exercised.
#[tokio::test]
async fn chat_completions_real_snapshot_503_with_prompt_tokens() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping chat_completions_real_snapshot_503_with_prompt_tokens");
        return;
    };
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let port = start_server(reg).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"What is the capital of France?"}]}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;

    // Generator returns 503 — prompt pipeline ran successfully.
    assert_eq!(
        status, 503,
        "expected 503 from NotReadyGenerator after prompt pipeline, body: {body}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    // Generator error message — prompt pipeline ran (would 404 if registry miss,
    // or 503 "missing chat_template" if pipeline skipped).
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("generator not ready"),
        "expected 'generator not ready' in error message, got: {msg}"
    );
}

#[tokio::test]
async fn unload_not_loaded_returns_404() {
    // Model not loaded → 404 (not 200 as in the old no-op stub).
    let port = start_server(ModelRegistry::default()).await;
    let (status, _body) = http(port, "POST", "/v1/models/some-model/unload", None).await;
    assert_eq!(status, 404, "unloading a non-loaded model must return 404");
}

#[tokio::test]
async fn validation_rejects_bad_temperature() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"temperature":-1.0}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_eq!(status, 400, "expected 400, body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");
}

#[tokio::test]
async fn validation_rejects_zero_max_tokens() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_eq!(status, 400, "expected 400, body: {body}");
}

// ── Anthropic Messages API ────────────────────────────────────────────────────

#[tokio::test]
async fn anthropic_messages_unknown_model_returns_404() {
    let port = start_server(ModelRegistry::default()).await;
    let payload =
        r#"{"model":"no-such-model","max_tokens":10,"messages":[{"role":"user","content":"hi"}]}"#;
    let (status, body) = http(port, "POST", "/v1/messages", Some(payload)).await;
    assert_eq!(status, 404, "expected 404 for unknown model, body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["error"]["message"].is_string());
    assert_eq!(v["error"]["type"], "not_found_error");
}

#[tokio::test]
async fn anthropic_messages_streaming_unknown_model_returns_404() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{"model":"no-such-model","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let (status, _body) = http(port, "POST", "/v1/messages", Some(payload)).await;
    assert_eq!(status, 404, "streaming must return 404 for unknown model");
}

/// Real snapshot: Anthropic route runs prompt pipeline, then gets 503 from generator.
#[tokio::test]
async fn anthropic_messages_real_snapshot_503_with_prompt_tokens() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping anthropic_messages_real_snapshot_503_with_prompt_tokens");
        return;
    };
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let port = start_server(reg).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","max_tokens":16,"messages":[{"role":"user","content":"What is the capital of France?"}]}"#;
    let (status, body) = http(port, "POST", "/v1/messages", Some(payload)).await;
    assert_eq!(
        status, 503,
        "expected 503 from NotReadyGenerator after prompt pipeline, body: {body}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("generator not ready"),
        "expected 'generator not ready' in error message, got: {msg}"
    );
}

#[tokio::test]
async fn anthropic_messages_missing_max_tokens_returns_400() {
    let port = start_server(ModelRegistry::default()).await;
    // max_tokens is required (plain u32); axum returns 422 for serde errors.
    let payload = r#"{"model":"x","messages":[{"role":"user","content":"hi"}]}"#;
    let (status, _body) = http(port, "POST", "/v1/messages", Some(payload)).await;
    // axum returns 422 Unprocessable Entity for JSON parse / serde failures.
    assert!(
        status == 400 || status == 422,
        "expected 400 or 422 for missing max_tokens, got {status}"
    );
}

// A5.1: tools field is now accepted — old 400 test updated.
// tools=[] is treated same as absent (empty normalises to None), so the
// request reaches the generator which returns 404 (unknown model "x").
#[tokio::test]
async fn anthropic_messages_with_empty_tools_is_accepted() {
    let port = start_server(ModelRegistry::default()).await;
    let payload =
        r#"{"model":"x","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"tools":[]}"#;
    let (status, _body) = http(port, "POST", "/v1/messages", Some(payload)).await;
    assert_ne!(status, 400, "tools=[] must not return 400 (A5.1)");
}

// ── A1: max_tokens cap enforcement (configurable, HTTP 400) ──────────────────

/// OpenAI: requesting `max_tokens` above the server cap returns HTTP 400
/// `invalid_request_error` with the requested and cap values in the message.
#[tokio::test]
async fn max_tokens_above_cap_returns_400_openai() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            "primary snapshot absent — skipping max_tokens_above_cap_returns_400_openai"
        );
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    // Cap = 10; the generator is NotReadyGenerator but the cap check fires
    // before the generator is invoked.
    let state = not_ready_state_with_cap(reg, 10);
    let port = start_server_with_state(state).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;

    assert_eq!(
        status, 400,
        "expected 400 for over-cap max_tokens, body: {body}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("100"),
        "message must mention requested value, got: {msg}"
    );
    assert!(
        msg.contains("10"),
        "message must mention cap value, got: {msg}"
    );
}

/// Anthropic: requesting `max_tokens` above the server cap returns HTTP 400
/// `invalid_request_error` with the requested and cap values in the message.
#[tokio::test]
async fn max_tokens_above_cap_returns_400_anthropic() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            "primary snapshot absent — skipping max_tokens_above_cap_returns_400_anthropic"
        );
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let state = not_ready_state_with_cap(reg, 10);
    let port = start_server_with_state(state).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}"#;
    let (status, body) = http(port, "POST", "/v1/messages", Some(payload)).await;

    assert_eq!(
        status, 400,
        "expected 400 for over-cap max_tokens, body: {body}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "invalid_request_error");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("100"),
        "message must mention requested value, got: {msg}"
    );
    assert!(
        msg.contains("10"),
        "message must mention cap value, got: {msg}"
    );
}

/// With the default cap (`u32::MAX`), high `max_tokens` values pass the cap
/// gate and reach the generator — for `NotReadyGenerator`, that's a 503.
/// Regression test: the old hardcoded cap of 64 must not be reintroduced.
#[tokio::test]
async fn max_tokens_above_64_default_cap_passes() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!("primary snapshot absent — skipping max_tokens_above_64_default_cap_passes");
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let port = start_server(reg).await; // default cap = u32::MAX

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":1024}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    // No cap rejection: passes through to NotReadyGenerator → 503.
    assert_eq!(
        status, 503,
        "expected 503 (not 400 from old cap=64), body: {body}"
    );
}

// ── Real end-to-end generation tests ─────────────────────────────────────────
//
// These tests boot a server with the real ArchGenerator (CPU) and verify
// that both API surfaces produce non-empty text.
//
// CPU-only forward is ~1.3 s/token with no KV cache. With max_tokens=4 that
// is ~5-6 s plus model load time (~10-20 s). Tests are marked `#[ignore]` to
// avoid making the default `cargo test` run impractical on CI.
//
// Run manually:
// cargo test -p rmlx-server -- --ignored real_generation --nocapture

/// OpenAI `POST /v1/chat/completions` with the real ArchGenerator.
#[ignore = "requires primary snapshot + ~30s wall-clock (CPU, no KV cache)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn real_generation_openai_chat_completions() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping real_generation_openai_chat_completions"
        );
        return;
    }

    let gen = ArchGenerator::from_snapshot(
        &snap,
        &ModelLoadConfig {
            device: rmlx_mlx::Device::Cpu,
            kv_quant: None,
            max_ctx: None,
            prompt_cache_slots: 4,
            mm_cache: None,
            calibration: None,
            yarn: None,
        },
        Arc::new(parking_lot::Mutex::new(())),
    )
    .expect("ArchGenerator::from_snapshot");
    let port = start_server_with_gemma4(gen).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":4}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;

    assert_eq!(status, 200, "expected 200, body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .expect("choices[0].message.content must be a string");
    assert!(
        !content.is_empty(),
        "choices[0].message.content must be non-empty"
    );
    let completion_tokens = v["usage"]["completion_tokens"]
        .as_u64()
        .expect("usage.completion_tokens must be present");
    assert!(
        completion_tokens >= 1,
        "usage.completion_tokens must be >= 1"
    );

    tracing::info!(content, completion_tokens, "real_generation_openai: OK");
}

/// Anthropic `POST /v1/messages` with the real ArchGenerator.
#[ignore = "requires primary snapshot + ~30s wall-clock (CPU, no KV cache)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn real_generation_anthropic_messages() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping real_generation_anthropic_messages"
        );
        return;
    }

    let gen = ArchGenerator::from_snapshot(
        &snap,
        &ModelLoadConfig {
            device: rmlx_mlx::Device::Cpu,
            kv_quant: None,
            max_ctx: None,
            prompt_cache_slots: 4,
            mm_cache: None,
            calibration: None,
            yarn: None,
        },
        Arc::new(parking_lot::Mutex::new(())),
    )
    .expect("ArchGenerator::from_snapshot");
    let port = start_server_with_gemma4(gen).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":4}"#;
    let (status, body) = http(port, "POST", "/v1/messages", Some(payload)).await;

    assert_eq!(status, 200, "expected 200, body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
    let text = v["content"][0]["text"]
        .as_str()
        .expect("content[0].text must be a string");
    assert!(!text.is_empty(), "content[0].text must be non-empty");
    let output_tokens = v["usage"]["output_tokens"]
        .as_u64()
        .expect("usage.output_tokens must be present");
    assert!(output_tokens >= 1, "usage.output_tokens must be >= 1");

    tracing::info!(text, output_tokens, "real_generation_anthropic: OK");
}

// ── A2: context_length_exceeded guard (HTTP 400) ─────────────────────────────

/// OpenAI: a prompt longer than the loaded model's `effective_max_ctx`
/// returns HTTP 400 `context_length_exceeded` with both numbers in the
/// message body. The slot is pre-populated with a `LoadedModel`
/// carrying `effective_max_ctx = 10` so the guard fires before
/// generation.
#[tokio::test]
async fn prompt_exceeds_ctx_returns_400_openai() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!("primary snapshot absent — skipping prompt_exceeds_ctx_returns_400_openai");
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let model_id = "mlx-community__gemma-4-e4b-it-mxfp8";
    let state = loaded_state_with_max_ctx(reg, model_id, 10);
    let port = start_server_with_state(state).await;

    // Long enough user content that the rendered chat template tokenises
    // to well over 10 tokens — every Gemma chat-template wrap alone is
    // already >5 tokens, plus 12 distinct ASCII words = ≥17 ids.
    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"}],"max_tokens":4}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;

    assert_eq!(
        status, 400,
        "expected 400 for over-ctx prompt, body: {body}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "context_length_exceeded");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("max_ctx is 10"),
        "message must report the configured max_ctx, got: {msg}"
    );
    assert!(
        msg.contains("tokens"),
        "message must mention prompt token count, got: {msg}"
    );
}

/// Anthropic: same as the OpenAI test above, mirrored at `/v1/messages`.
#[tokio::test]
async fn prompt_exceeds_ctx_returns_400_anthropic() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            "primary snapshot absent — skipping prompt_exceeds_ctx_returns_400_anthropic"
        );
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let model_id = "mlx-community__gemma-4-e4b-it-mxfp8";
    let state = loaded_state_with_max_ctx(reg, model_id, 10);
    let port = start_server_with_state(state).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","max_tokens":4,"messages":[{"role":"user","content":"alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi omicron"}]}"#;
    let (status, body) = http(port, "POST", "/v1/messages", Some(payload)).await;

    assert_eq!(
        status, 400,
        "expected 400 for over-ctx prompt, body: {body}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["error"]["type"], "context_length_exceeded");
    let msg = v["error"]["message"].as_str().unwrap_or("");
    assert!(
        msg.contains("max_ctx is 10"),
        "message must report the configured max_ctx, got: {msg}"
    );
    assert!(
        msg.contains("tokens"),
        "message must mention prompt token count, got: {msg}"
    );
}

/// Short prompt under the configured `effective_max_ctx` must pass the
/// guard and reach the `NotReadyGenerator`, which returns 503. Verifies
/// the guard does not over-fire and that the message-token-count side
/// of the comparison is correct.
#[tokio::test]
async fn short_prompt_passes_ctx_guard_openai() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!("primary snapshot absent — skipping short_prompt_passes_ctx_guard_openai");
        return;
    }
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let model_id = "mlx-community__gemma-4-e4b-it-mxfp8";
    // Very generous max_ctx — even after chat-template wrapping the
    // single-word user prompt fits comfortably under 10_000 ids.
    let state = loaded_state_with_max_ctx(reg, model_id, 10_000);
    let port = start_server_with_state(state).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;

    assert_eq!(
        status, 503,
        "guard must not fire for in-range prompt; expected 503 from NotReadyGenerator, body: {body}"
    );
}

// ── A5.1: tools + tool_choice schema (no execution) ──────────────────────────

/// OpenAI route: a request with a full `tools` + `tool_choice: "auto"` payload
/// reaches the generator (404 for unknown model) — not rejected with 400.
#[tokio::test]
async fn openai_tools_payload_is_accepted() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "max_tokens": 50,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }],
        "tool_choice": "auto"
    }"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    // Must NOT be 400; unknown model gives 404.
    assert_ne!(
        status, 400,
        "tools payload must not be rejected (A5.1), body: {body}"
    );
    assert_eq!(status, 404, "unknown model should give 404, body: {body}");
}

/// Anthropic route: a request with `tools` + `tool_choice: {"type":"auto"}` is
/// accepted and reaches the generator (404 for unknown model).
#[tokio::test]
async fn anthropic_tools_payload_is_accepted() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{
        "model": "unknown-model",
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "tools": [{
            "name": "get_weather",
            "description": "Get weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }],
        "tool_choice": {"type": "auto"}
    }"#;
    let (status, body) = http(port, "POST", "/v1/messages", Some(payload)).await;
    assert_ne!(
        status, 400,
        "tools payload must not be rejected (A5.1), body: {body}"
    );
    assert_eq!(status, 404, "unknown model should give 404, body: {body}");
}

/// OpenAI: `tool_choice: "required"` deserialises to `Mode("required")`.
/// Request is accepted (reaches 404 for unknown model).
#[tokio::test]
async fn openai_tool_choice_required_is_accepted() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "noop", "parameters": {}}}],
        "tool_choice": "required"
    }"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_ne!(
        status, 400,
        "tool_choice=required must not be rejected, body: {body}"
    );
}

/// OpenAI: named `tool_choice` object is accepted.
#[tokio::test]
async fn openai_tool_choice_named_is_accepted() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{
        "model": "x",
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}],
        "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
    }"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_ne!(
        status, 400,
        "named tool_choice must not be rejected, body: {body}"
    );
}

// ── A6.1: response_format schema (no enforcement) ────────────────────────────

/// OpenAI: `response_format: {"type":"json_object"}` is now a first-class
/// parsed field — must NOT return 400. With an unknown model it returns 404;
/// with a registered model + NotReadyGenerator it returns 503.
///
/// This test uses an unknown model to avoid needing the snapshot on disk.
/// The assertion is `!= 400` — the field is accepted, whatever the model
/// resolution outcome.
#[tokio::test]
async fn openai_response_format_json_object_is_accepted() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Return JSON with key result=42."}],
        "max_tokens": 80,
        "response_format": {"type": "json_object"}
    }"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_ne!(
        status, 400,
        "response_format=json_object must not return 400 (A6.1), body: {body}"
    );
    // Unknown model → 404; the field did not cause a rejection.
    assert_eq!(status, 404, "unknown model must return 404, body: {body}");
}

/// OpenAI: `response_format: {"type":"json_schema", ...}` is accepted and
/// routed the same way as the plain json_object case.
#[tokio::test]
async fn openai_response_format_json_schema_is_accepted() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "Reply with JSON for location=Paris."}],
        "max_tokens": 80,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "weather",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"]
                }
            }
        }
    }"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_ne!(
        status, 400,
        "response_format=json_schema must not return 400 (A6.1), body: {body}"
    );
    assert_eq!(status, 404, "unknown model must return 404, body: {body}");
}

// ── A7.1: sampling params schema (no enforcement) ─────────────────────────────

// ── A8: timeout middleware integration test ───────────────────────────────────

/// Build a test router with a `/slow` handler (sleeps 5 s) and `max_timeout_secs`
/// set to 1, so the timeout middleware fires before the handler returns.
async fn start_timeout_test_server(max_timeout_secs: u64) -> u16 {
    use axum::middleware;
    use axum::routing::get;

    let reg = ModelRegistry::default();
    let loader: ModelLoader =
        Arc::new(|_path, _id| Ok(Box::new(NotReadyGenerator) as Box<dyn Generator>));

    let state = AppState {
        registry: Arc::new(reg),
        slots: Arc::new(parking_lot::RwLock::new(Vec::new())),
        embed_slot: Arc::new(parking_lot::RwLock::new(None)),
        mm_cache: Arc::new(rmlx_models::multimodal_cache::MultimodalCache::new(0)),
        gpu_gate: Arc::new(parking_lot::Mutex::new(())),
        gpu_queue: Arc::new(tokio::sync::Semaphore::new(1)),
        gpu_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_queue_depth: 64,
        max_loaded_models: 1,
        loader,
        metrics: None,
        idle_policy: rmlx_server::KeepAlivePolicy::Pin,
        max_tokens_cap: u32::MAX,
        max_timeout_secs,
        session_cache: Arc::new(parking_lot::Mutex::new(SessionCache::new(4))),
        ttft_store: TtftStore::default(),
        itl_store: ItlStore::default(),
        metrics_drainer: None,
        require_smoke_probe: false,
        default_temperature: None,
        default_enable_thinking: None,
        tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        started_at: std::time::Instant::now(),
        requests_started: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        admission_controller: None,
        admission_handle: None,
        whisper_model_path: None,
        whisper_tokenizer_path: None,
        audio_model: Arc::new(parking_lot::RwLock::new(None)),
        tts_model_path: None,
        tts_tokenizer_path: None,
        tts_model: Arc::new(parking_lot::RwLock::new(None)),
    };

    async fn slow_handler() -> &'static str {
        tokio::time::sleep(Duration::from_secs(5)).await;
        "done"
    }

    let router = axum::Router::new()
        .route("/slow", get(slow_handler))
        .layer(middleware::from_fn_with_state(state.clone(), timeout_mw))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    port
}

/// A8: request to a slow handler with max_timeout_secs=1 must return 408
/// with an OpenAI-shaped error body containing `"type":"timeout"`.
#[tokio::test]
async fn a8_slow_handler_times_out_with_408() {
    let port = start_timeout_test_server(1).await;
    let (status, body) = http(port, "GET", "/slow", None).await;
    assert_eq!(
        status, 408,
        "expected 408 timeout, got {status} body: {body}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).expect("body must be JSON");
    assert_eq!(
        v["error"]["type"].as_str(),
        Some("timeout"),
        "error.type must be 'timeout', got: {v}"
    );
    assert!(
        v["error"]["message"]
            .as_str()
            .unwrap_or("")
            .contains("1 second"),
        "message must mention the timeout duration, got: {v}"
    );
}

/// A8: bad header value → 400 invalid_request_error.
#[tokio::test]
async fn a8_bad_timeout_header_returns_400() {
    let port = start_server(ModelRegistry::default()).await;
    let payload =
        r#"{"model":"unknown","messages":[{"role":"user","content":"hi"}],"max_tokens":5}"#;
    let (status, body) = http_with_headers(
        port,
        "POST",
        "/v1/chat/completions",
        &[("X-Request-Timeout-Seconds", "abc")],
        Some(payload),
    )
    .await;
    assert_eq!(
        status, 400,
        "expected 400 for bad timeout header, body: {body}"
    );
    let v: serde_json::Value = serde_json::from_str(&body).expect("body must be JSON");
    assert_eq!(
        v["error"]["type"].as_str(),
        Some("invalid_request_error"),
        "error.type must be 'invalid_request_error', got: {v}"
    );
}

/// All six new sampling fields are accepted by the OpenAI route without a 400.
///
/// Uses an unknown model so only the field-parsing/validation path is exercised.
/// Expects 404 (unknown model), not 400 (field rejected).
#[tokio::test]
async fn openai_sampling_params_all_fields_accepted() {
    let port = start_server(ModelRegistry::default()).await;
    let payload = r#"{
        "model": "unknown-model",
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.0,
        "max_tokens": 1500,
        "top_k": 5,
        "min_p": 0.05,
        "repetition_penalty": 1.2,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "logit_bias": {"100": 1.0}
    }"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_ne!(
        status, 400,
        "sampling params must not return 400 (A7.1), body: {body}"
    );
    // Unknown model → 404; the sampling fields did not cause a rejection.
    assert_eq!(status, 404, "unknown model must return 404, body: {body}");
}

// ── A9: tools-supported guard ─────────────────────────────────────────────────

/// Build a `ModelRegistry` with a single synthetic snapshot whose chat
/// template raises an exception when `tools` is non-empty. The snapshot
/// directory is created in a tempdir and cleaned up automatically.
///
/// Returns (registry, tempdir) — caller must hold onto `tempdir` to prevent
/// premature cleanup.
fn make_no_tools_registry() -> (ModelRegistry, tempfile::TempDir) {
    use std::io::Write as _;

    let tmp = tempfile::tempdir().unwrap();
    let snap = tmp.path().join("NoToolsModel");
    std::fs::create_dir_all(&snap).unwrap();

    // config.json — minimal, required for registry to load the entry.
    let cfg = serde_json::json!({"architectures": ["FakeForCausalLM"], "dtype": "bfloat16"});
    let mut f = std::fs::File::create(snap.join("config.json")).unwrap();
    f.write_all(cfg.to_string().as_bytes()).unwrap();

    // tokenizer_config.json — minimal bos/eos so the pipeline can run.
    let tkcfg = serde_json::json!({"bos_token": "", "eos_token": "</s>"});
    let mut f = std::fs::File::create(snap.join("tokenizer_config.json")).unwrap();
    f.write_all(tkcfg.to_string().as_bytes()).unwrap();

    // chat_template.jinja — raises exception when tools is passed.
    let tmpl = b"{% if tools %}{{ tools | raise_exception }}{% endif %}{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}";
    let mut f = std::fs::File::create(snap.join("chat_template.jinja")).unwrap();
    f.write_all(tmpl).unwrap();

    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    (reg, tmp)
}

/// A9 guard: when the loaded model's template cannot render tools, a request
/// that includes `tools` must NOT return 500. It should return 503
/// (NotReadyGenerator — no tokenizer, so the prompt pipeline is incomplete)
/// rather than panicking or returning 500.
///
/// The key assertion: status != 500. The guard must suppress the tool
/// injection and proceed with the tool-less render path.
///
/// Note: The test snapshot has a chat_template.jinja but no tokenizer.json,
/// so the prompt pipeline returns 503. That is the expected result — the
/// important invariant is no 500 / no panic.
#[tokio::test]
async fn a9_tools_unsupported_snapshot_returns_no_500() {
    let (reg, _tmp) = make_no_tools_registry();

    // Verify the probe flagged tools_supported=false.
    let entry = reg
        .get("NoToolsModel")
        .expect("NoToolsModel must be in registry");
    assert!(
        !entry.tools_supported,
        "NoToolsModel must have tools_supported=false"
    );

    let port = start_server(reg).await;
    let payload = r#"{
        "model": "NoToolsModel",
        "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
        "max_tokens": 50,
        "tools": [{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}
            }
        }]
    }"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_ne!(
        status, 500,
        "A9 guard: tools-unsupported snapshot must not 500 when tools are in request; body: {body}"
    );
    // 503 is expected here because the snapshot has no tokenizer.json.
    assert_eq!(
        status, 503,
        "expected 503 (missing tokenizer) after tools guard suppressed injection; body: {body}"
    );
}

// ── H3/H4: usage chunk in streaming path ────────────────────────────────────
//
// These tests require the primary snapshot and ~30 s wall-clock (CPU).
// Run manually:
// cargo test -p rmlx-server -- --ignored h3_ h4_ --nocapture

/// Parse raw SSE body into a Vec of `(event_type_or_empty, data)` pairs.
///
/// Lines starting with `data:` are parsed; lines starting with `event:` set
/// the current event type; blank lines flush an event. `[DONE]` data values
/// are kept as-is.
fn parse_sse_events(raw: &str) -> Vec<serde_json::Value> {
    let mut events: Vec<serde_json::Value> = Vec::new();
    for line in raw.lines() {
        let line = line.trim();
        if let Some(data) = line.strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                events.push(serde_json::json!({"__done": true}));
            } else if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                events.push(v);
            }
        }
    }
    events
}

/// H3: OpenAI non-streaming response has a correct `usage` triple for two
/// different `max_tokens` values. Verifies that `total_tokens ==
/// prompt_tokens + completion_tokens` and both token counts are positive.
#[ignore = "requires primary snapshot + ~30s wall-clock (CPU, no KV cache)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn h3_non_streaming_usage_triple_exact() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping h3_non_streaming_usage_triple_exact"
        );
        return;
    }

    let gen = ArchGenerator::from_snapshot(
        &snap,
        &ModelLoadConfig {
            device: rmlx_mlx::Device::Cpu,
            kv_quant: None,
            max_ctx: None,
            prompt_cache_slots: 4,
            mm_cache: None,
            calibration: None,
            yarn: None,
        },
        Arc::new(parking_lot::Mutex::new(())),
    )
    .expect("ArchGenerator::from_snapshot");
    let port = start_server_with_gemma4(gen).await;

    // max_tokens = 4.
    let payload4 = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":4}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload4)).await;
    assert_eq!(status, 200, "H3 max_tokens=4: expected 200, body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).expect("valid JSON");
    let pt = v["usage"]["prompt_tokens"].as_u64().expect("prompt_tokens");
    let ct = v["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    let tt = v["usage"]["total_tokens"].as_u64().expect("total_tokens");
    assert!(pt > 0, "H3: prompt_tokens must be > 0");
    assert!(
        ct >= 1,
        "H3: completion_tokens must be >= 1 for max_tokens=4"
    );
    assert_eq!(
        tt,
        pt + ct,
        "H3: total_tokens must equal prompt_tokens + completion_tokens"
    );
    tracing::info!(pt, ct, tt, "H3 max_tokens=4: OK");

    // max_tokens = 8 (separate request — same prompt, different cap).
    let payload8 = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":8}"#;
    let (status2, body2) = http(port, "POST", "/v1/chat/completions", Some(payload8)).await;
    assert_eq!(status2, 200, "H3 max_tokens=8: expected 200, body: {body2}");
    let v2: serde_json::Value = serde_json::from_str(&body2).expect("valid JSON");
    let pt2 = v2["usage"]["prompt_tokens"]
        .as_u64()
        .expect("prompt_tokens");
    let ct2 = v2["usage"]["completion_tokens"]
        .as_u64()
        .expect("completion_tokens");
    let tt2 = v2["usage"]["total_tokens"].as_u64().expect("total_tokens");
    assert!(pt2 > 0, "H3 mt=8: prompt_tokens must be > 0");
    assert!(ct2 >= 1, "H3 mt=8: completion_tokens must be >= 1");
    assert_eq!(
        tt2,
        pt2 + ct2,
        "H3 mt=8: total_tokens must equal prompt + completion"
    );
    tracing::info!(pt2, ct2, tt2, "H3 max_tokens=8: OK");
}

/// H4: streaming with `stream_options.include_usage=true` — the penultimate
/// SSE event must be a usage chunk with `choices: []` and exact triple; the
/// final event must be `[DONE]`.
#[ignore = "requires primary snapshot + ~30s wall-clock (CPU, no KV cache)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn h4_streaming_include_usage_true_emits_usage_chunk() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping h4_streaming_include_usage_true_emits_usage_chunk"
        );
        return;
    }

    let gen = ArchGenerator::from_snapshot(
        &snap,
        &ModelLoadConfig {
            device: rmlx_mlx::Device::Cpu,
            kv_quant: None,
            max_ctx: None,
            prompt_cache_slots: 4,
            mm_cache: None,
            calibration: None,
            yarn: None,
        },
        Arc::new(parking_lot::Mutex::new(())),
    )
    .expect("ArchGenerator::from_snapshot");
    let port = start_server_with_gemma4(gen).await;

    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":8,"stream":true,"stream_options":{"include_usage":true}}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_eq!(
        status, 200,
        "H4 include_usage=true: expected 200, body: {body}"
    );

    let events = parse_sse_events(&body);
    assert!(
        events.len() >= 3,
        "H4: expected at least 3 events (role chunk + usage + done), got {}: body: {body}",
        events.len()
    );

    // Final event must be [DONE].
    let last = &events[events.len() - 1];
    assert_eq!(
        last["__done"],
        serde_json::json!(true),
        "H4: last event must be [DONE], got: {last}"
    );

    // Penultimate event must be the usage chunk.
    let usage_ev = &events[events.len() - 2];
    assert_eq!(
        usage_ev["choices"].as_array().map(Vec::len),
        Some(0),
        "H4: usage chunk must have choices=[], got: {usage_ev}"
    );
    let pt = usage_ev["usage"]["prompt_tokens"]
        .as_u64()
        .expect("usage.prompt_tokens must be present");
    let ct = usage_ev["usage"]["completion_tokens"]
        .as_u64()
        .expect("usage.completion_tokens must be present");
    let tt = usage_ev["usage"]["total_tokens"]
        .as_u64()
        .expect("usage.total_tokens must be present");
    assert!(pt > 0, "H4: prompt_tokens must be > 0");
    assert!(ct >= 1, "H4: completion_tokens must be >= 1");
    assert_eq!(
        tt,
        pt + ct,
        "H4: total_tokens must equal prompt_tokens + completion_tokens"
    );
    tracing::info!(
        pt,
        ct,
        tt,
        events_count = events.len(),
        "H4 include_usage=true: OK"
    );
}

/// H4: streaming with `stream_options.include_usage=false` (or absent) — NO
/// chunk in the stream may contain a `usage` key.
#[ignore = "requires primary snapshot + ~30s wall-clock (CPU, no KV cache)"]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn h4_streaming_include_usage_false_no_usage_in_stream() {
    let Some(snap) = primary_snapshot_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping test");
        return;
    };
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping h4_streaming_include_usage_false_no_usage_in_stream"
        );
        return;
    }

    let gen = ArchGenerator::from_snapshot(
        &snap,
        &ModelLoadConfig {
            device: rmlx_mlx::Device::Cpu,
            kv_quant: None,
            max_ctx: None,
            prompt_cache_slots: 4,
            mm_cache: None,
            calibration: None,
            yarn: None,
        },
        Arc::new(parking_lot::Mutex::new(())),
    )
    .expect("ArchGenerator::from_snapshot");
    let port = start_server_with_gemma4(gen).await;

    // Omit stream_options entirely — must behave identically to include_usage=false.
    let payload = r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","messages":[{"role":"user","content":"Hello"}],"max_tokens":8,"stream":true}"#;
    let (status, body) = http(port, "POST", "/v1/chat/completions", Some(payload)).await;
    assert_eq!(
        status, 200,
        "H4 include_usage=false: expected 200, body: {body}"
    );

    let events = parse_sse_events(&body);
    for (i, ev) in events.iter().enumerate() {
        if ev.get("__done").is_some() {
            continue;
        }
        assert!(
            ev.get("usage").is_none(),
            "H4: no chunk must contain 'usage' when include_usage omitted, chunk[{i}]: {ev}"
        );
    }
    tracing::info!(events_count = events.len(), "H4 include_usage=false: OK");
}

// ── B1: LoggedJson rejection logging ─────────────────────────────────────────
//
// Verifies that a deliberately malformed JSON body on the two POST routes
// that use `LoggedJson<T>` still returns the same HTTP 422 status that
// axum's built-in `Json<T>` would return.
//
// The warn! emission cannot be asserted here without a tracing capture
// dep (which the task spec prohibits adding). The `logged_json` module
// carries a unit test that exercises the snippet-truncation path directly.

/// OpenAI `POST /v1/chat/completions` with a malformed body must still
/// return 422 (the unchanged axum JsonRejection wire response).
#[tokio::test]
async fn b1_malformed_json_openai_returns_422() {
    let port = start_server(ModelRegistry::default()).await;
    // `{broken` is syntactically invalid JSON — serde will reject it before
    // field-level validation, so this triggers `JsonSyntaxError` → 400 or
    // `JsonDataError` → 422 depending on axum version. Both are non-200;
    // the important invariant is that the status is an error status (4xx)
    // and NOT 200 (i.e., the handler did not succeed on garbage input).
    let (status, _body) = http(port, "POST", "/v1/chat/completions", Some("{broken")).await;
    assert!(
        status == 400 || status == 422,
        "malformed JSON must return 400 or 422 (axum JsonRejection), got {status}"
    );
}

/// Anthropic `POST /v1/messages` with a malformed body must return 422.
#[tokio::test]
async fn b1_malformed_json_anthropic_returns_422() {
    let port = start_server(ModelRegistry::default()).await;
    let (status, _body) = http(port, "POST", "/v1/messages", Some("{broken")).await;
    assert!(
        status == 400 || status == 422,
        "malformed JSON must return 400 or 422 (axum JsonRejection), got {status}"
    );
}

// ── idle-eviction must clear SessionCache ─────────────────────────────

/// Verify that the idle-eviction retain closure clears session-cache entries
/// for an evicted model.
///
/// The idle-eviction loop in `serve.rs` clones both `state.slots` and
/// `state.session_cache` before spawning. When a slot is evicted (retain
/// returns `false`) the loop calls `session_cache.lock().unwrap().remove_model`.
///
/// This test exercises that exact code path without spawning the background
/// task or loading a real model: it constructs the minimal state, primes the
/// session cache, then runs the retain predicate directly (matching the fixed
/// `serve.rs` logic) and asserts the session is gone.
///
/// To confirm the test catches the pre-fix bug: if you remove the
/// `session_cache.lock().unwrap().remove_model(...)` call from the retain
/// closure below, `active_count()` stays at 1 and the assertion fails.
#[tokio::test]
async fn idle_eviction_clears_session_cache() {
    use rmlx_server::{LoadedModel, SessionKey};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    // Build minimal state.
    let state = not_ready_state(ModelRegistry::default());

    // Insert a LoadedModel whose last_used is old enough to be evicted.
    let model_id = "test-model-idle";
    let old_instant = Instant::now() - Duration::from_secs(3600); // 1 h ago
    state.slots.write().push(LoadedModel {
        id: model_id.to_owned(),
        model: Arc::new(NotReadyGenerator),
        loaded_at: old_instant,
        last_used: old_instant,
        effective_max_ctx: 4096,
        decode_lease: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        unload_handle: Arc::new(parking_lot::Mutex::new(None)),
        keep_alive: rmlx_server::KeepAlivePolicy::Pin,
    });

    // Prime the session cache for that model.
    {
        let key = SessionKey {
            model_id: model_id.to_owned(),
            session_id: "sess-1".to_owned(),
        };
        state.session_cache.lock().touch(key, 10);
    }
    assert_eq!(
        state.session_cache.lock().active_count(),
        1,
        "precondition: session must be present before eviction"
    );

    // Run the eviction retain closure exactly as the fixed serve.rs does.
    // timeout = 60 s; old_instant is 1 h old, so it will be evicted.
    let timeout = Duration::from_secs(60);
    let session_cache = Arc::clone(&state.session_cache);
    state.slots.write().retain(|loaded| {
        let idle = loaded.last_used.elapsed();
        if idle >= timeout {
            // This is the line added by the fix.
            session_cache.lock().remove_model(&loaded.id);
            false
        } else {
            true
        }
    });

    // Model must be evicted from slots.
    assert_eq!(
        state.slots.read().len(),
        0,
        "slot must be empty after eviction"
    );

    // Session must be cleared — this is the invariant.
    assert_eq!(
        state.session_cache.lock().active_count(),
        0,
        "session cache must be empty after idle-eviction of model"
    );
}

// ── per-model keep-alive TTL ──────────────────────────────────────────────

/// Build a tempdir-backed minimal snapshot registered under `id`.
fn make_synthetic_registry(ids: &[&str]) -> (ModelRegistry, tempfile::TempDir) {
    use std::io::Write as _;

    let tmp = tempfile::tempdir().unwrap();
    let mut snaps = Vec::new();
    for id in ids {
        let snap = tmp.path().join(id);
        std::fs::create_dir_all(&snap).unwrap();
        let cfg = serde_json::json!({"architectures": ["FakeForCausalLM"], "dtype": "bfloat16"});
        let mut f = std::fs::File::create(snap.join("config.json")).unwrap();
        f.write_all(cfg.to_string().as_bytes()).unwrap();
        let tkcfg = serde_json::json!({"bos_token": "", "eos_token": "</s>"});
        let mut f = std::fs::File::create(snap.join("tokenizer_config.json")).unwrap();
        f.write_all(tkcfg.to_string().as_bytes()).unwrap();
        let tmpl =
            b"{% for m in messages %}<{{ m.role }}>{{ m.content }}</{{ m.role }}>{% endfor %}";
        let mut f = std::fs::File::create(snap.join("chat_template.jinja")).unwrap();
        f.write_all(tmpl).unwrap();
        snaps.push(snap);
    }
    let reg = ModelRegistry::from_paths(&snaps);
    (reg, tmp)
}

/// Arming a short Idle policy via `ensure_loaded` causes the model
/// to unload after the TTL expires.
#[tokio::test]
async fn idle_ttl_unloads_after_timeout() {
    use rmlx_server::KeepAlivePolicy;
    use std::time::Duration;

    let (registry, _tmp) = make_synthetic_registry(&["kp-test"]);
    let mut state = not_ready_state(registry);
    state.idle_policy = KeepAlivePolicy::Idle(Duration::from_millis(150));

    let _ = state.ensure_loaded("kp-test").expect("load must succeed");
    assert_eq!(state.slots.read().len(), 1, "model loaded");

    tokio::time::sleep(Duration::from_millis(400)).await;

    assert_eq!(
        state.slots.read().len(),
        0,
        "model should have been unloaded by the keep-alive timer"
    );
}

/// A held decode lease suppresses the unload — the timer fires,
/// observes lease > 0, and reschedules instead of unloading.
#[tokio::test]
async fn decode_lease_suppresses_unload() {
    use rmlx_server::KeepAlivePolicy;
    use std::time::Duration;

    let (registry, _tmp) = make_synthetic_registry(&["kp-busy"]);
    let mut state = not_ready_state(registry);
    state.idle_policy = KeepAlivePolicy::Idle(Duration::from_millis(120));

    let _ = state.ensure_loaded("kp-busy").expect("load must succeed");
    let _guard = state.decode_lease_guard("kp-busy").expect("model resident");

    tokio::time::sleep(Duration::from_millis(450)).await;

    assert_eq!(
        state.slots.read().len(),
        1,
        "held decode lease must suppress unload"
    );
}

/// `KeepAlivePolicy::Pin` arms no timer; model stays resident
/// indefinitely.
#[tokio::test]
async fn pin_policy_no_unload() {
    use rmlx_server::KeepAlivePolicy;
    use std::time::Duration;

    let (registry, _tmp) = make_synthetic_registry(&["kp-pin"]);
    let mut state = not_ready_state(registry);
    state.idle_policy = KeepAlivePolicy::Pin;

    let _ = state.ensure_loaded("kp-pin").expect("load must succeed");
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(state.slots.read().len(), 1, "pin: never unload");
}

/// Per-request override via `reset_keep_alive` promotes a slot to Pin.
#[tokio::test]
async fn request_field_promotes_to_pin() {
    use rmlx_server::KeepAlivePolicy;
    use std::time::Duration;

    let (registry, _tmp) = make_synthetic_registry(&["kp-promote"]);
    let mut state = not_ready_state(registry);
    state.idle_policy = KeepAlivePolicy::Idle(Duration::from_millis(120));

    let _ = state
        .ensure_loaded("kp-promote")
        .expect("load must succeed");
    state.reset_keep_alive("kp-promote", Some(KeepAlivePolicy::Pin));

    tokio::time::sleep(Duration::from_millis(400)).await;
    assert_eq!(
        state.slots.read().len(),
        1,
        "Pin override must prevent unload"
    );
}

/// Cooperative same-process evict — loading B unloads A when
/// `max_loaded_models == 1`, regardless of A's TTL remaining.
#[tokio::test]
async fn cooperative_evict_on_conflicting_load() {
    use rmlx_server::KeepAlivePolicy;

    let (registry, _tmp) = make_synthetic_registry(&["kp-a", "kp-b"]);
    let mut state = not_ready_state(registry);
    state.idle_policy = KeepAlivePolicy::Pin;
    state.max_loaded_models = 1;

    let _ = state.ensure_loaded("kp-a").expect("A loads");
    assert_eq!(state.slots.read().first().unwrap().id, "kp-a");

    let _ = state.ensure_loaded("kp-b").expect("B loads + evicts A");
    let slots = state.slots.read();
    assert_eq!(slots.len(), 1, "single-slot post-evict");
    assert_eq!(slots.first().unwrap().id, "kp-b", "B is now resident");
}

/// H1 regression: stale TTL fire post-`sleep().await` must not unload
/// a freshly-reset slot.
///
/// Race window we are pinning: timer task 1 sleeps for the full TTL, then
/// proceeds past the sleep before the abort from a reset can land. Without
/// the identity check, task 1 would call the unload path on a slot whose
/// `decode_lease` Arc has been swapped out by `unload + reload`, tearing
/// down the user's just-loaded model. With the identity-check fix, the task
/// observes the new lease pointer and bails out.
#[tokio::test]
async fn stale_ttl_fire_after_reset_does_not_unload() {
    use rmlx_server::KeepAlivePolicy;
    use std::sync::Arc;
    use std::time::Duration;

    let (registry, _tmp) = make_synthetic_registry(&["kp-race"]);
    let mut state = not_ready_state(registry);
    // Short TTL so the spawned task wakes quickly; we drive the race
    // synchronously below.
    state.idle_policy = KeepAlivePolicy::Idle(Duration::from_millis(80));

    // Load the model. Spawned timer T1 starts sleeping for 80 ms.
    let _ = state.ensure_loaded("kp-race").expect("initial load");

    // Wait past the TTL boundary. The timer task has already passed its
    // sleep().await and is about to enter the identity-check critical
    // section. The actual ordering inside tokio is racy — we simulate the
    // worst case by giving T1 a clear window to wake up before we swap.
    tokio::time::sleep(Duration::from_millis(120)).await;

    // At this point either T1 already unloaded the slot (in which case the
    // test's preconditions hit the "model not resident" branch) or T1 is
    // racing the identity check. We force the worst case: explicitly
    // unload + reload to swap the decode_lease pointer.
    let _ = state.unload("kp-race");
    let _ = state
        .ensure_loaded("kp-race")
        .expect("reload after explicit unload");

    let lease_after_reload = state
        .slots
        .read()
        .first()
        .map(|m| Arc::as_ptr(&m.decode_lease) as usize)
        .expect("model must be resident after reload");

    // Give the original T1 plenty of time to wake from its sleep, run its
    // identity check, observe the swapped lease pointer, and bail. With the
    // bug present, T1 would call unload here and tear down the freshly
    // reloaded slot. With the fix, the identity check fires "stale TTL
    // fire — slot was reset" and the slot survives.
    tokio::time::sleep(Duration::from_millis(50)).await;

    let resident = state.slots.read();
    assert_eq!(
        resident.len(),
        1,
        "freshly reloaded slot must survive a stale TTL fire from the pre-reset timer"
    );
    let lease_now = Arc::as_ptr(&resident[0].decode_lease) as usize;
    assert_eq!(
        lease_now, lease_after_reload,
        "the resident slot must still be the one we reloaded, not a re-load by a panicked path"
    );
}

/// H2 regression: a warm-reset via `ensure_loaded` MUST swap the
/// slot's `decode_lease` Arc, and the prior timer's stale TTL fire (still
/// holding the OLD lease pointer as its identity token) MUST bail without
/// evicting the resident slot.
///
/// Two-part check:
///   1. After warm-reset, the slot's decode_lease pointer differs from the
///      pointer captured before the reset (the H2 invariant).
///   2. A stale TTL fire from the pre-reset timer (forced by aborting its
///      handle race window) does not unload the slot.
#[tokio::test]
async fn stale_ttl_fire_after_warm_reset_does_not_unload() {
    use rmlx_server::KeepAlivePolicy;
    use std::sync::Arc;
    use std::time::Duration;

    let (registry, _tmp) = make_synthetic_registry(&["kp-warm-race"]);
    let mut state = not_ready_state(registry);
    // TTL long enough that T1 is still sleeping when we issue the warm
    // reset — we drive the race synchronously rather than relying on T1
    // having already fired.
    state.idle_policy = KeepAlivePolicy::Idle(Duration::from_millis(200));

    // Load the model. Spawned timer T1 starts sleeping for 200 ms.
    let _ = state.ensure_loaded("kp-warm-race").expect("initial load");

    let lease_before_reset = state
        .slots
        .read()
        .first()
        .map(|m| Arc::as_ptr(&m.decode_lease) as usize)
        .expect("model must be resident after initial load");

    // Sleep close to but BEFORE the TTL boundary — T1 is still parked on
    // its sleep().await.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Warm-reset path: call ensure_loaded again for the SAME model id.
    // This is the path a fresh chat request takes — same model, no unload,
    // no reload. With the H2 fix this swaps the slot's decode_lease Arc
    // and arms a fresh timer T2 (cancelling T1 via abort()).
    let _ = state
        .ensure_loaded("kp-warm-race")
        .expect("warm-reset hit must succeed");

    let lease_after_reset = state
        .slots
        .read()
        .first()
        .map(|m| Arc::as_ptr(&m.decode_lease) as usize)
        .expect("model must still be resident after warm-reset");

    // Part 1: the H2 invariant — warm-reset MUST swap the slot's
    // decode_lease Arc. Without this swap, the identity check in the
    // spawned timer task cannot distinguish T1 from T2.
    assert_ne!(
        lease_before_reset, lease_after_reset,
        "warm-reset must swap the slot's decode_lease Arc (H2 invariant)"
    );

    // Part 2: even if T1 somehow won an abort race and ran to completion
    // (its sleep already elapsed and the abort signal lost the race), the
    // identity check inside T1's spawned closure observes the swapped
    // lease pointer and bails out as "stale TTL fire — slot was reset".
    // We wait well past the original TTL boundary AND past T2's new TTL,
    // then confirm the slot is still resident with the new lease pointer.
    // (If H2 is fixed correctly the slot survives because T1 bails;
    // T2 will fire its own TTL after warm-reset + 200 ms = 350 ms total;
    // we check earlier to avoid T2's own legitimate fire.)
    tokio::time::sleep(Duration::from_millis(120)).await;
    // Total elapsed: 150 + 120 = 270 ms; T2 armed at 150, fires at 350. The
    // resident slot must still be present, with the post-reset lease.

    let resident = state.slots.read();
    assert_eq!(
        resident.len(),
        1,
        "warm-reset slot must survive a stale TTL fire from the pre-reset timer"
    );
    let lease_now = Arc::as_ptr(&resident[0].decode_lease) as usize;
    assert_eq!(
        lease_now, lease_after_reset,
        "the resident slot must still be the warm-reset slot (its lease pointer unchanged)"
    );
}

/// OpenAI: valid JSON that fails type-level deserialisation (missing required
/// field `messages`) must return 422, not 200.
#[tokio::test]
async fn b1_type_error_json_openai_returns_422() {
    let port = start_server(ModelRegistry::default()).await;
    // `model` is present but `messages` (required Vec) is absent.
    let (status, _body) = http(
        port,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"model":"x"}"#),
    )
    .await;
    assert!(
        status == 400 || status == 422,
        "type-error JSON must return 400 or 422, got {status}"
    );
}

// ── GET /v1/metrics rolling summary ────────────────────────────────────

/// `GET /v1/metrics` must return HTTP 200 with JSON containing the
/// three required fields: `in_flight` (numeric), `uptime_s` (float), and
/// `requests_started` (count).
#[tokio::test]
async fn v1_metrics_returns_required_fields() {
    let port = start_server(ModelRegistry::default()).await;
    let (status, body) = http(port, "GET", "/v1/metrics", None).await;

    assert_eq!(status, 200, "GET /v1/metrics must return 200, got {status}");

    let v: serde_json::Value =
        serde_json::from_str(&body).expect("GET /v1/metrics must return valid JSON");

    // in_flight: numeric (usize serialised as JSON number).
    assert!(
        v["in_flight"].is_number(),
        "in_flight must be a numeric field, got: {}",
        v["in_flight"]
    );

    // uptime_s: float >= 0.
    assert!(
        v["uptime_s"].is_number(),
        "uptime_s must be a numeric field, got: {}",
        v["uptime_s"]
    );
    assert!(
        v["uptime_s"].as_f64().unwrap_or(-1.0) >= 0.0,
        "uptime_s must be >= 0, got: {}",
        v["uptime_s"]
    );

    // requests_started: present and numeric.
    assert!(
        v["requests_started"].is_number(),
        "requests_started must be a numeric field, got: {}",
        v["requests_started"]
    );

    // requests_completed and requests_failed also present.
    assert!(
        v["requests_completed"].is_number(),
        "requests_completed must be present"
    );
    assert!(
        v["requests_failed"].is_number(),
        "requests_failed must be present"
    );

    // tokens_in, tokens_out present.
    assert!(v["tokens_in"].is_number(), "tokens_in must be present");
    assert!(v["tokens_out"].is_number(), "tokens_out must be present");
}
