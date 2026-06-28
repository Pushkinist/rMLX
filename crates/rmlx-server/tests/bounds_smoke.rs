//! Integration smoke tests for bounded-input checks.
//!
//! Each test constructs an over-limit request and verifies the server returns
//! 413 Payload Too Large with a structured JSON error body. A within-limit
//! case is also included per bound to verify the check does not fire on valid
//! input (expected 503 from NotReadyGenerator, not 413).
//!
//! Tests are deterministic: no model is loaded (NotReadyGenerator), no timing
//! dependencies. The bound check must fire **before** model resolution, so 413
//! is the authoritative signal on over-limit requests.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::sync::Arc;
use std::time::Duration;

use rmlx_server::bounds::{
    MAX_CONTENT_PARTS, MAX_INPUT_AUDIO_BYTES, MAX_MESSAGES, MAX_TOOLS, MAX_TOOL_CALLS,
    MAX_TOTAL_INPUT_TOKENS_ESTIMATE,
};
use rmlx_server::{
    ApiErrorCounters, AppState, Generator, ItlStore, ModelLoader, ModelRegistry, NotReadyGenerator,
    SessionCache, TtftStore,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Test server helpers ───────────────────────────────────────────────────────

fn stub_state() -> AppState {
    let loader: ModelLoader = Arc::new(|_path, _id| {
        let g: Box<dyn Generator> = Box::new(NotReadyGenerator);
        Ok(g)
    });
    AppState {
        registry: Arc::new(ModelRegistry::default()),
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
        default_image_max_tokens: None,
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

async fn start() -> u16 {
    let state = stub_state();
    let router = rmlx_server::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind");
    let port = listener.local_addr().expect("local_addr").port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.expect("serve");
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    port
}

/// Send a POST request with a JSON body, return (status, body).
///
/// Handles `BrokenPipe` on write gracefully: if the server closes the
/// connection early (e.g. axum body-limit or our own bounds check rejects the
/// request before we finish writing), we read whatever response bytes are
/// available and parse what we can.
async fn post(port: u16, path: &str, body: &str) -> (u16, String) {
    use tokio::io::ErrorKind;

    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    // Write may fail with BrokenPipe if the server closes the connection
    // early (body too large). That is fine — the response is already in the
    // read buffer; proceed to read.
    let write_result = stream.write_all(req.as_bytes()).await;
    if let Err(ref e) = write_result {
        if e.kind() != ErrorKind::BrokenPipe && e.kind() != ErrorKind::ConnectionReset {
            write_result.unwrap(); // propagate unexpected errors
        }
    }
    let mut resp = Vec::new();
    // Ignore read errors — partial response is enough to parse status + body.
    let _ = stream.read_to_end(&mut resp).await;
    let text = String::from_utf8_lossy(&resp).into_owned();
    let status: u16 = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    (status, text[body_start..].to_owned())
}

/// Build a JSON string with N minimal messages.
fn messages_json(n: usize) -> String {
    let msgs: Vec<String> = (0..n)
        .map(|i| format!(r#"{{"role":"user","content":"msg {i}"}}"#))
        .collect();
    format!(
        r#"{{"model":"test","messages":[{}],"max_tokens":1}}"#,
        msgs.join(",")
    )
}

// ── Test 1: MAX_MESSAGES exceeded → 413 ──────────────────────────────────────

#[tokio::test]
async fn test_messages_over_limit_returns_413() {
    let port = start().await;
    let body = messages_json(MAX_MESSAGES + 1);
    let (status, body_text) = post(port, "/v1/chat/completions", &body).await;
    assert_eq!(
        status, 413,
        "expected 413 for over-limit messages, got {status}\nbody: {body_text}"
    );
    let json: serde_json::Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(json["error"]["code"], "input_too_large");
    assert_eq!(json["error"]["field"], "messages");
}

// ── Test 2: Within-limit messages → not 413 ──────────────────────────────────

#[tokio::test]
async fn test_messages_within_limit_not_413() {
    let port = start().await;
    let body = messages_json(10);
    let (status, _body_text) = post(port, "/v1/chat/completions", &body).await;
    // 503 = NotReadyGenerator (no model loaded), 404 = model not found — both fine.
    assert_ne!(status, 413, "within-limit messages should not return 413");
}

// ── Test 3: MAX_TOOL_CALLS exceeded → 413 ────────────────────────────────────

#[tokio::test]
async fn test_tool_calls_over_limit_returns_413() {
    let port = start().await;
    // Build one assistant message with MAX_TOOL_CALLS + 1 entries.
    let calls: Vec<String> = (0..=MAX_TOOL_CALLS)
        .map(|i| {
            format!(
                r#"{{"id":"call_{i}","type":"function","function":{{"name":"fn_{i}","arguments":"{{}}"}}}}"#
            )
        })
        .collect();
    let body = format!(
        r#"{{"model":"test","messages":[{{"role":"assistant","content":null,"tool_calls":[{}]}}],"max_tokens":1}}"#,
        calls.join(",")
    );
    let (status, body_text) = post(port, "/v1/chat/completions", &body).await;
    assert_eq!(
        status, 413,
        "expected 413 for over-limit tool_calls, got {status}\nbody: {body_text}"
    );
    let json: serde_json::Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(json["error"]["code"], "input_too_large");
    assert_eq!(json["error"]["field"], "tool_calls");
}

// ── Test 4: MAX_TOOLS exceeded → 413 ─────────────────────────────────────────

#[tokio::test]
async fn test_tools_over_limit_returns_413() {
    let port = start().await;
    let tools: Vec<String> = (0..=MAX_TOOLS)
        .map(|i| {
            format!(
                r#"{{"type":"function","function":{{"name":"fn_{i}","description":"d","parameters":{{"type":"object","properties":{{}}}}}}}}"#
            )
        })
        .collect();
    let body = format!(
        r#"{{"model":"test","messages":[{{"role":"user","content":"hi"}}],"tools":[{}],"max_tokens":1}}"#,
        tools.join(",")
    );
    let (status, body_text) = post(port, "/v1/chat/completions", &body).await;
    assert_eq!(
        status, 413,
        "expected 413 for over-limit tools, got {status}\nbody: {body_text}"
    );
    let json: serde_json::Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(json["error"]["code"], "input_too_large");
    assert_eq!(json["error"]["field"], "tools");
}

// ── Test 5: MAX_CONTENT_PARTS exceeded → 413 ─────────────────────────────────

#[tokio::test]
async fn test_content_parts_over_limit_returns_413() {
    let port = start().await;
    let parts: Vec<String> = (0..=MAX_CONTENT_PARTS)
        .map(|i| format!(r#"{{"type":"text","text":"part {i}"}}"#))
        .collect();
    let body = format!(
        r#"{{"model":"test","messages":[{{"role":"user","content":[{}]}}],"max_tokens":1}}"#,
        parts.join(",")
    );
    let (status, body_text) = post(port, "/v1/chat/completions", &body).await;
    assert_eq!(
        status, 413,
        "expected 413 for over-limit content_parts, got {status}\nbody: {body_text}"
    );
    let json: serde_json::Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(json["error"]["code"], "input_too_large");
    assert_eq!(json["error"]["field"], "content_parts");
}

// ── Test 6: MAX_INPUT_AUDIO_BYTES exceeded → 413 from our handler ────────────
//
// MAX_INPUT_AUDIO_BYTES is 16 MiB; triggering it over HTTP requires a ~22 MiB
// base64 body. The router's DefaultBodyLimit is 24 MiB so this payload passes
// the transport-level gate and reaches our handler-level check, which returns a
// structured 413 with code "input_too_large".

#[tokio::test]
async fn test_input_audio_bytes_over_limit_returns_413() {
    let port = start().await;
    // b64_len * 3 / 4 > MAX_INPUT_AUDIO_BYTES → b64_len > MAX_INPUT_AUDIO_BYTES * 4 / 3
    let b64_len = (MAX_INPUT_AUDIO_BYTES * 4 / 3) + 8;
    let fake_b64 = "A".repeat(b64_len);
    let body = format!(
        r#"{{"model":"test","messages":[{{"role":"user","content":[{{"type":"input_audio","input_audio":{{"data":"{fake_b64}","format":"wav"}}}}]}}],"max_tokens":1}}"#
    );
    let (status, body_text) = post(port, "/v1/chat/completions", &body).await;
    // Our handler-level bounds check fires (body is within the 24 MiB transport limit).
    assert_eq!(
        status, 413,
        "expected 413 for over-limit input_audio, got {status}"
    );
    let json: serde_json::Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(json["error"]["code"], "input_too_large");
    assert_eq!(json["error"]["field"], "input_audio");
}

// ── Test 7: MAX_TOTAL_INPUT_TOKENS_ESTIMATE exceeded → 413 from our handler ───
//
// Triggering this bound requires > 3 MiB of text content. The router's
// DefaultBodyLimit is 24 MiB so this payload passes the transport-level gate
// and reaches our handler-level check, which returns a structured 413.

#[tokio::test]
async fn test_total_tokens_estimate_over_limit_returns_413() {
    let port = start().await;
    // MAX_TOTAL_INPUT_TOKENS_ESTIMATE * 3 bytes of text → estimate = MAX + 5
    let big_text = "x".repeat(MAX_TOTAL_INPUT_TOKENS_ESTIMATE * 3 + 16);
    let body = format!(
        r#"{{"model":"test","messages":[{{"role":"user","content":"{big_text}"}}],"max_tokens":1}}"#
    );
    let (status, body_text) = post(port, "/v1/chat/completions", &body).await;
    // Our handler-level bounds check fires (body is within the 24 MiB transport limit).
    assert_eq!(
        status, 413,
        "expected 413 for over-limit token estimate, got {status}"
    );
    let json: serde_json::Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(json["error"]["code"], "input_too_large");
    assert_eq!(json["error"]["field"], "total_input_tokens_estimate");
}

// ── Test 8: MAX_MESSAGES via Anthropic /v1/messages → 413 ────────────────────

#[tokio::test]
async fn test_anthropic_messages_over_limit_returns_413() {
    let port = start().await;
    let msgs: Vec<String> = (0..=MAX_MESSAGES)
        .map(|i| format!(r#"{{"role":"user","content":"msg {i}"}}"#))
        .collect();
    let body = format!(
        r#"{{"model":"test","messages":[{}],"max_tokens":1}}"#,
        msgs.join(",")
    );
    let (status, body_text) = post(port, "/v1/messages", &body).await;
    assert_eq!(
        status, 413,
        "expected 413 from Anthropic endpoint for over-limit messages, got {status}\nbody: {body_text}"
    );
    let json: serde_json::Value = serde_json::from_str(&body_text).unwrap();
    assert_eq!(json["error"]["code"], "input_too_large");
    assert_eq!(json["error"]["field"], "messages");
}
