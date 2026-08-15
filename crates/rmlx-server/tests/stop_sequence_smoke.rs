//! HTTP-surface smoke tests for stop-sequence truncation.
//!
//! A `ScriptedGenerator` emits a fixed token sequence so the test controls
//! exactly what the model "produces", then asserts the server truncates the
//! content at the first stop-string boundary (stop excluded) and sets the
//! correct finish/stop reason — for OpenAI + Anthropic, streaming +
//! non-streaming, including a token-straddling stop.

// Test harness: panics/unwraps are acceptable in test bodies.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    trivial_casts
)]

use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::{self, Stream};
use rmlx_server::{
    ApiErrorCounters, AppState, GenerationRequest, GenerationToken, Generator, ItlStore,
    LoadedModel, ModelLoader, ModelRegistry, SessionCache, TtftStore,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Scripted generator ──────────────────────────────────────────────────────────

/// A generator that emits a pre-scripted list of pieces, one per token, then a
/// terminal `done` token carrying `finish_reason="length"` (so any "stop"
/// reason in the response is attributable to stop-sequence handling, not EOS).
#[derive(Clone)]
struct ScriptedGenerator {
    pieces: Vec<String>,
}

impl ScriptedGenerator {
    fn new(pieces: &[&str]) -> Self {
        Self {
            pieces: pieces.iter().map(|s| (*s).to_owned()).collect(),
        }
    }
}

impl Generator for ScriptedGenerator {
    fn generate(
        &self,
        _req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        let mut toks: Vec<rmlx_core::Result<GenerationToken>> = self
            .pieces
            .iter()
            .map(|p| {
                Ok(GenerationToken {
                    token_id: 1,
                    piece: p.clone(),
                    done: false,
                    finish_reason: None,
                    is_thinking: false,
                    logprobs: None,
                })
            })
            .collect();
        // Terminal token: empty piece, done=true, reason "length" (max_tokens).
        toks.push(Ok(GenerationToken {
            token_id: 0,
            piece: String::new(),
            done: true,
            finish_reason: Some("length".to_owned()),
            is_thinking: false,
            logprobs: None,
        }));
        Box::pin(stream::iter(toks))
    }
}

// ── AppState + server wiring ─────────────────────────────────────────────────────

fn scripted_state(pieces: &'static [&'static str]) -> (AppState, tempfile::TempDir) {
    // Build a registry from a minimal on-disk snapshot named "scripted". The
    // route handler requires a compilable chat_template.jinja + loadable
    // tokenizer.json (otherwise it 503s "not ready" before generating), so
    // both are written as trivial-but-valid files. The ScriptedGenerator's
    // output — not the tokenized prompt — is what stop-truncation acts on.
    let tmp = tempfile::tempdir().unwrap();
    let snap = tmp.path().join("scripted");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(
        snap.join("config.json"),
        r#"{"architectures":["Scripted"]}"#,
    )
    .unwrap();
    // Trivial Jinja chat template: concatenate message contents.
    std::fs::write(
        snap.join("chat_template.jinja"),
        "{% for m in messages %}{{ m['content'] }}{% endfor %}",
    )
    .unwrap();
    // Minimal WordLevel tokenizer.json the `tokenizers` crate can load.
    std::fs::write(
        snap.join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"x":0,"hi":1,"[UNK]":2},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let loader: ModelLoader = Arc::new(move |_path, _id| {
        Ok(Box::new(ScriptedGenerator::new(pieces)) as Box<dyn Generator>)
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
        prompt_cache_slots: 4,
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
    };
    // Pre-populate the slot so the request never needs the (fake) snapshot on
    // disk — the loader is only invoked on a cold path we avoid by pinning a
    // LoadedModel that delegates to a ScriptedGenerator.
    let now = std::time::Instant::now();
    state.slots.write().push(LoadedModel {
        id: "scripted".to_owned(),
        model: Arc::new(ScriptedGenerator::new(pieces)),
        loaded_at: now,
        last_used: now,
        effective_max_ctx: usize::MAX,
        decode_lease: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        unload_handle: Arc::new(parking_lot::Mutex::new(None)),
        keep_alive: rmlx_server::KeepAlivePolicy::Pin,
    });
    (state, tmp)
}

async fn start_server(pieces: &'static [&'static str]) -> (u16, tempfile::TempDir) {
    let (state, tmp) = scripted_state(pieces);
    let router = rmlx_server::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    // Return the TempDir so it lives for the test duration (registry holds
    // canonicalized paths; the snapshot dir must not be deleted early).
    (port, tmp)
}

async fn http(port: u16, path: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
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
    (status, text[body_start..].to_owned())
}

/// Concatenate all SSE content deltas in an OpenAI streaming body.
fn sse_openai_content(raw: &str) -> String {
    let mut out = String::new();
    for line in raw.lines() {
        if let Some(data) = line.trim().strip_prefix("data:") {
            let data = data.trim();
            if data == "[DONE]" {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                    out.push_str(c);
                }
            }
        }
    }
    out
}

/// Last non-null finish_reason in an OpenAI streaming body.
fn sse_openai_finish(raw: &str) -> Option<String> {
    let mut last = None;
    for line in raw.lines() {
        if let Some(data) = line.trim().strip_prefix("data:") {
            let data = data.trim();
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                if let Some(fr) = v["choices"][0]["finish_reason"].as_str() {
                    last = Some(fr.to_owned());
                }
            }
        }
    }
    last
}

// ── OpenAI non-streaming ──────────────────────────────────────────────────────

#[tokio::test]
async fn openai_nonstream_single_stop_truncates() {
    // Model emits "alpha bravo charlie delta echo"; stop ["charlie"].
    let (port, _tmp) = start_server(&["alpha ", "bravo ", "charlie", " delta echo"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stop":["charlie"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "alpha bravo ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn openai_nonstream_multi_stop_first_match_wins() {
    // "charlie" appears before "echo"; earliest offset wins.
    let (port, _tmp) = start_server(&["a charlie b echo c"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stop":["echo","charlie"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "a ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn openai_nonstream_stop_not_present_keeps_all() {
    let (port, _tmp) = start_server(&["alpha bravo charlie"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stop":["zzz"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "alpha bravo charlie");
    // No stop matched → terminal reason from the engine ("length").
    assert_eq!(v["choices"][0]["finish_reason"], "length");
}

// ── OpenAI streaming ──────────────────────────────────────────────────────────

#[tokio::test]
async fn openai_stream_single_stop_truncates() {
    let (port, _tmp) = start_server(&["alpha ", "bravo ", "charlie", " delta echo"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stream":true,"stop":["charlie"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    assert_eq!(sse_openai_content(&b), "alpha bravo ");
    assert_eq!(sse_openai_finish(&b).as_deref(), Some("stop"));
}

#[tokio::test]
async fn openai_stream_token_straddling_stop_truncates() {
    // "charlie" split across three pieces: must hold back the partial tail.
    let (port, _tmp) = start_server(&["alpha bravo ", "char", "l", "ie", " delta echo"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stream":true,"stop":["charlie"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    assert_eq!(sse_openai_content(&b), "alpha bravo ");
    assert_eq!(sse_openai_finish(&b).as_deref(), Some("stop"));
}

#[tokio::test]
async fn openai_stream_stop_not_present_keeps_all() {
    let (port, _tmp) = start_server(&["alpha ", "bravo ", "charlie"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stream":true,"stop":["zzz"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    assert_eq!(sse_openai_content(&b), "alpha bravo charlie");
}

// ── Anthropic non-streaming ────────────────────────────────────────────────────

#[tokio::test]
async fn anthropic_nonstream_single_stop_truncates_and_names() {
    let (port, _tmp) = start_server(&["alpha ", "bravo ", "charlie", " delta echo"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stop_sequences":["charlie"]}"#;
    let (status, b) = http(port, "/v1/messages", body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    // First content block is the text block.
    assert_eq!(v["content"][0]["text"], "alpha bravo ");
    assert_eq!(v["stop_reason"], "stop_sequence");
    assert_eq!(v["stop_sequence"], "charlie");
}

// ── Anthropic streaming ────────────────────────────────────────────────────────

#[tokio::test]
async fn anthropic_stream_single_stop_truncates_and_names() {
    let (port, _tmp) = start_server(&["alpha ", "bravo ", "charlie", " delta echo"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stream":true,"stop_sequences":["charlie"]}"#;
    let (status, b) = http(port, "/v1/messages", body).await;
    assert_eq!(status, 200, "body: {b}");
    // Concatenate text_delta values.
    let mut text = String::new();
    let mut stop_reason = None;
    let mut stop_sequence = None;
    for line in b.lines() {
        if let Some(data) = line.trim().strip_prefix("data:") {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(data.trim()) {
                if v["delta"]["type"] == "text_delta" {
                    if let Some(t) = v["delta"]["text"].as_str() {
                        text.push_str(t);
                    }
                }
                if v["type"] == "message_delta" {
                    stop_reason = v["delta"]["stop_reason"].as_str().map(str::to_owned);
                    stop_sequence = v["delta"]["stop_sequence"].as_str().map(str::to_owned);
                }
            }
        }
    }
    assert_eq!(text, "alpha bravo ");
    assert_eq!(stop_reason.as_deref(), Some("stop_sequence"));
    assert_eq!(stop_sequence.as_deref(), Some("charlie"));
}
