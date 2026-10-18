//! HTTP-surface smoke tests for stop-sequence handling.
//!
//! A `ScriptedGenerator` emits a fixed token sequence so the test controls
//! exactly what the model "produces", then asserts the server truncates the
//! content at the first stop-string boundary (stop excluded) and sets the
//! correct finish/stop reason — for OpenAI + Anthropic, streaming +
//! non-streaming, including a token-straddling stop.
//!
//! The generator is lazy, counts the steps it runs and honours the end flag
//! of the request as the engine does: once the route has set the flag, it sends
//! its terminal item. A stop-string match must therefore leave the count at the
//! step of the match (or a few steps later when the stub is told to lag). The
//! route reads the generation to its terminal item after a match, because the
//! engine writes its rows only when that item is sent. A request at temperature
//! 0 goes through the replay envelope, whose channel of 4 lets the generator
//! run up to 5 steps ahead of the route; the count is then bounded, not exact.
//!
//! What these tests prove is what the route does: it sets the flag at a match
//! and only then, reads to the terminal item, counts every step that ran, cuts
//! the reply at the match, and writes one finish by one rule. They do not prove
//! that an engine ends when the flag is set. The loop tests of `rmlx-models`
//! and the GPU cells of `stop_ends_decode` hold that.
//!
//! What cannot move: the reply, the finish reason, the step count and the
//! flag of a request whose stop string never matches, whether or not it sends
//! `stop`, a tool call, or temperature 0: `no_match_runs_the_whole_generation`.
//!
//! Mutations and the test that catches each:
//! - a route that never sets the flag, or sets it late: the step count and the
//!   flag column of `a_match_ends_the_generation_on_every_route`.
//! - a route that sets the flag with no match:
//!   `no_match_runs_the_whole_generation` and
//!   `a_stop_string_matches_the_content_channel_only`.
//! - a route that keeps counting past the steps that ran, or stops counting:
//!   `a_late_end_counts_the_steps_that_ran`.
//! - a route that drops the stream at the match: the terminal item is never
//!   read (every test).
//! - a route that matches inside reasoning or tool-call arguments:
//!   `a_stop_string_matches_the_content_channel_only`.
//! - a route that keeps a tool call that came after the match, drops one that
//!   came before it, or writes another finish: `a_tool_call_changes_no_stop_rule`.
//! - a route that skips the end-of-stream flush of the tool parser:
//!   `what_the_tool_parser_holds_at_the_end_reaches_the_reply`.

// Test harness: panics/unwraps are acceptable in test bodies.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    trivial_casts
)]

use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use futures::stream::{self, Stream};
use rmlx_server::{
    ApiErrorCounters, AppState, GenerationRequest, GenerationToken, Generator, ItlStore,
    LoadedModel, ModelLoader, ModelRegistry, SessionCache, TtftStore,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Scripted generator ──────────────────────────────────────────────────────────

/// What the generator did, read by the test after the reply.
#[derive(Default)]
struct Probe {
    /// Steps the generator produced (one per piece it handed out).
    produced: AtomicUsize,
    /// The terminal item was handed out: the route read to the end.
    terminal_read: AtomicBool,
    /// The end flag of the request, kept so the test can read whether the
    /// route ever set it.
    end_requested: Mutex<Option<Arc<AtomicBool>>>,
}

impl Probe {
    fn produced(&self) -> usize {
        self.produced.load(Ordering::SeqCst)
    }

    fn terminal_read(&self) -> bool {
        self.terminal_read.load(Ordering::SeqCst)
    }

    /// Whether the route set the end flag of the request.
    fn end_was_requested(&self) -> bool {
        self.end_requested
            .lock()
            .unwrap()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::SeqCst))
    }
}

/// What the stub generator says: the pieces, how many of them are reasoning,
/// and how many more steps it runs after the end flag is set. An engine runs
/// at least one: the pass that was in flight at the match.
#[derive(Clone)]
struct Script {
    pieces: Vec<String>,
    reasoning: usize,
    lag: usize,
}

impl Script {
    fn of(pieces: &[&str]) -> Self {
        Self {
            pieces: pieces.iter().map(|p| (*p).to_owned()).collect(),
            reasoning: 0,
            lag: 0,
        }
    }
}

/// A generator that emits a pre-scripted list of pieces, one per token, then a
/// terminal `done` token carrying `finish_reason="length"` (so any "stop"
/// reason in the response is attributable to stop-sequence handling, not EOS).
/// It produces an item only when the route asks for it. Once the route has set
/// the end flag of the request and `lag` more steps have run, it sends the
/// terminal item at once, as the engine does.
#[derive(Clone)]
struct ScriptedGenerator {
    script: Script,
    probe: Arc<Probe>,
}

impl Generator for ScriptedGenerator {
    fn generate(
        &self,
        req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        *self.probe.end_requested.lock().unwrap() = Some(Arc::clone(&req.end_requested));
        let state = (
            self.script.clone(),
            Arc::clone(&self.probe),
            req.end_requested,
            0usize,
            0usize,
        );
        Box::pin(stream::unfold(
            state,
            |(script, probe, flag, at, after)| async move {
                let end_asked = flag.load(Ordering::SeqCst);
                let honoured = end_asked && after >= script.lag;
                let n = script.pieces.len();
                if at > n {
                    return None;
                }
                if at < n && !honoured {
                    probe.produced.fetch_add(1, Ordering::SeqCst);
                    let token = GenerationToken {
                        token_id: 1,
                        piece: script.pieces[at].clone(),
                        done: false,
                        finish_reason: None,
                        is_thinking: at < script.reasoning,
                        logprobs: None,
                    };
                    let after = after + usize::from(end_asked);
                    return Some((Ok(token), (script, probe, flag, at + 1, after)));
                }
                probe.terminal_read.store(true, Ordering::SeqCst);
                let terminal = GenerationToken {
                    token_id: 0,
                    piece: String::new(),
                    done: true,
                    finish_reason: Some("length".to_owned()),
                    is_thinking: false,
                    logprobs: None,
                };
                Some((Ok(terminal), (script, probe, flag, n + 1, after)))
            },
        ))
    }
}

// ── AppState + server wiring ─────────────────────────────────────────────────────

fn scripted_state(script: Script, probe: &Arc<Probe>) -> (AppState, tempfile::TempDir) {
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
    // Trivial Jinja chat template: concatenate message contents. The comment
    // holds the marker that makes the server parse tool calls in the Qwen XML
    // form.
    std::fs::write(
        snap.join("chat_template.jinja"),
        "{# <function= #}{% for m in messages %}{{ m['content'] }}{% endfor %}",
    )
    .unwrap();
    // Minimal WordLevel tokenizer.json the `tokenizers` crate can load.
    std::fs::write(
        snap.join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"x":0,"hi":1,"[UNK]":2},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let loader_script = script.clone();
    let loader_probe = Arc::clone(probe);
    let loader: ModelLoader = Arc::new(move |_path, _id| {
        Ok(Box::new(ScriptedGenerator {
            script: loader_script.clone(),
            probe: Arc::clone(&loader_probe),
        }) as Box<dyn Generator>)
    });
    let state = AppState {
        device: rmlx_mlx::Device::Cpu,
        registry: Arc::new(reg),
        slots: Arc::new(parking_lot::RwLock::new(Vec::new())),
        embed_slot: Arc::new(parking_lot::RwLock::new(None)),
        mm_cache: Arc::new(rmlx_models::multimodal_cache::MultimodalCache::new(0)),
        gpu_gate: Arc::new(parking_lot::Mutex::new(())),
        gpu_queue: Arc::new(tokio::sync::Semaphore::new(1)),
        gpu_pending: Arc::new(AtomicUsize::new(0)),
        max_queue_depth: 64,
        max_loaded_models: 1,
        loader,
        load_gate: Arc::default(),
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
        model: Arc::new(ScriptedGenerator {
            script,
            probe: Arc::clone(probe),
        }),
        loaded_at: now,
        last_used: now,
        effective_max_ctx: usize::MAX,
        context_limits: None,
        decode_lease: Arc::new(AtomicUsize::new(0)),
        unload_handle: Arc::new(parking_lot::Mutex::new(None)),
        keep_alive: rmlx_server::KeepAlivePolicy::Pin,
    });
    (state, tmp)
}

async fn start_server(pieces: &[&str]) -> (u16, tempfile::TempDir, Arc<Probe>) {
    start_scripted(Script::of(pieces)).await
}

async fn start_scripted(script: Script) -> (u16, tempfile::TempDir, Arc<Probe>) {
    let probe = Arc::new(Probe::default());
    let (state, tmp) = scripted_state(script, &probe);
    let router = rmlx_server::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    // Return the TempDir so it lives for the test duration (registry holds
    // canonicalized paths; the snapshot dir must not be deleted early).
    (port, tmp, probe)
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
    let (port, _tmp, _) = start_server(&["alpha ", "bravo ", "charlie", " delta echo"]).await;
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
    let (port, _tmp, _) = start_server(&["a charlie b echo c"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stop":["echo","charlie"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "a ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
}

#[tokio::test]
async fn openai_nonstream_stop_not_present_keeps_all() {
    let (port, _tmp, _) = start_server(&["alpha bravo charlie"]).await;
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
    let (port, _tmp, _) = start_server(&["alpha ", "bravo ", "charlie", " delta echo"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stream":true,"stop":["charlie"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    assert_eq!(sse_openai_content(&b), "alpha bravo ");
    assert_eq!(sse_openai_finish(&b).as_deref(), Some("stop"));
}

#[tokio::test]
async fn openai_stream_token_straddling_stop_truncates() {
    // "charlie" split across three pieces: must hold back the partial tail.
    let (port, _tmp, _) = start_server(&["alpha bravo ", "char", "l", "ie", " delta echo"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stream":true,"stop":["charlie"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    assert_eq!(sse_openai_content(&b), "alpha bravo ");
    assert_eq!(sse_openai_finish(&b).as_deref(), Some("stop"));
}

#[tokio::test]
async fn openai_stream_stop_not_present_keeps_all() {
    let (port, _tmp, _) = start_server(&["alpha ", "bravo ", "charlie"]).await;
    let body = r#"{"model":"scripted","messages":[{"role":"user","content":"x"}],"max_tokens":48,"stream":true,"stop":["zzz"]}"#;
    let (status, b) = http(port, "/v1/chat/completions", body).await;
    assert_eq!(status, 200, "body: {b}");
    assert_eq!(sse_openai_content(&b), "alpha bravo charlie");
}

// ── Anthropic non-streaming ────────────────────────────────────────────────────

#[tokio::test]
async fn anthropic_nonstream_single_stop_truncates_and_names() {
    let (port, _tmp, _) = start_server(&["alpha ", "bravo ", "charlie", " delta echo"]).await;
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
    let (port, _tmp, _) = start_server(&["alpha ", "bravo ", "charlie", " delta echo"]).await;
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

// ── The generation ends at a match ───────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum Route {
    OpenAi,
    OpenAiStream,
    Anthropic,
    AnthropicStream,
}

const ROUTES: [Route; 4] = [
    Route::OpenAi,
    Route::OpenAiStream,
    Route::Anthropic,
    Route::AnthropicStream,
];

impl Route {
    fn is_anthropic(self) -> bool {
        matches!(self, Route::Anthropic | Route::AnthropicStream)
    }

    fn streams(self) -> bool {
        matches!(self, Route::OpenAiStream | Route::AnthropicStream)
    }

    fn path(self) -> &'static str {
        if self.is_anthropic() {
            "/v1/messages"
        } else {
            "/v1/chat/completions"
        }
    }

    fn body(self, row: &Row) -> String {
        let schema = serde_json::json!({"type": "object", "properties": {"x": {"type": "string"}}});
        let mut body = serde_json::json!({
            "model": "scripted",
            "messages": [{"role": "user", "content": "x"}],
            "max_tokens": 512,
            "stream": self.streams(),
            "temperature": row.temperature,
        });
        if self.is_anthropic() {
            if let Some(stop) = row.stop {
                body["stop_sequences"] = serde_json::json!([stop]);
            }
            if row.tools {
                body["tools"] =
                    serde_json::json!([{"name": "f", "description": "d", "input_schema": schema}]);
            }
        } else {
            if let Some(stop) = row.stop {
                body["stop"] = serde_json::json!([stop]);
            }
            if row.tools {
                body["tools"] = serde_json::json!([{
                    "type": "function",
                    "function": {"name": "f", "description": "d", "parameters": schema},
                }]);
            }
            if self.streams() {
                body["stream_options"] = serde_json::json!({"include_usage": true});
            }
        }
        body.to_string()
    }
}

/// What a client reads from one reply.
struct Seen {
    text: String,
    reasoning: String,
    tool_calls: usize,
    finish: Option<String>,
    stop_sequence: Option<String>,
    usage: Option<u64>,
}

fn sse_events(raw: &str) -> Vec<serde_json::Value> {
    raw.lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str(data.trim()).ok())
        .collect()
}

fn read(route: Route, raw: &str) -> Seen {
    match route {
        Route::OpenAi => {
            let v: serde_json::Value = serde_json::from_str(raw).unwrap();
            let choice = &v["choices"][0];
            let message = &choice["message"];
            Seen {
                text: message["content"].as_str().unwrap_or_default().to_owned(),
                reasoning: message["reasoning_content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                tool_calls: message["tool_calls"].as_array().map_or(0, Vec::len),
                finish: choice["finish_reason"].as_str().map(str::to_owned),
                stop_sequence: None,
                usage: v["usage"]["completion_tokens"].as_u64(),
            }
        }
        Route::OpenAiStream => {
            let events = sse_events(raw);
            let delta = |e: &serde_json::Value| e["choices"][0]["delta"].clone();
            let joined = |field: &str| -> String {
                events
                    .iter()
                    .filter_map(|e| delta(e)[field].as_str().map(str::to_owned))
                    .collect()
            };
            Seen {
                text: joined("content"),
                reasoning: joined("reasoning_content"),
                tool_calls: events
                    .iter()
                    .filter(|e| delta(e)["tool_calls"][0]["function"]["name"].is_string())
                    .count(),
                finish: events
                    .iter()
                    .filter_map(|e| e["choices"][0]["finish_reason"].as_str())
                    .next_back()
                    .map(str::to_owned),
                stop_sequence: None,
                usage: events
                    .iter()
                    .find_map(|e| e["usage"]["completion_tokens"].as_u64()),
            }
        }
        Route::Anthropic => {
            let v: serde_json::Value = serde_json::from_str(raw).unwrap();
            let blocks = v["content"].as_array().unwrap();
            let joined = |kind: &str, field: &str| -> String {
                blocks
                    .iter()
                    .filter(|b| b["type"] == kind)
                    .filter_map(|b| b[field].as_str())
                    .collect()
            };
            Seen {
                text: joined("text", "text"),
                reasoning: joined("thinking", "thinking"),
                tool_calls: blocks.iter().filter(|b| b["type"] == "tool_use").count(),
                finish: v["stop_reason"].as_str().map(str::to_owned),
                stop_sequence: v["stop_sequence"].as_str().map(str::to_owned),
                usage: v["usage"]["output_tokens"].as_u64(),
            }
        }
        Route::AnthropicStream => {
            let events = sse_events(raw);
            let message_delta = events.iter().find(|e| e["type"] == "message_delta");
            let joined = |kind: &str, field: &str| -> String {
                events
                    .iter()
                    .filter(|e| e["delta"]["type"] == kind)
                    .filter_map(|e| e["delta"][field].as_str())
                    .collect()
            };
            Seen {
                text: joined("text_delta", "text"),
                reasoning: joined("thinking_delta", "thinking"),
                tool_calls: events
                    .iter()
                    .filter(|e| e["content_block"]["type"] == "tool_use")
                    .count(),
                finish: message_delta
                    .and_then(|e| e["delta"]["stop_reason"].as_str())
                    .map(str::to_owned),
                stop_sequence: message_delta
                    .and_then(|e| e["delta"]["stop_sequence"].as_str())
                    .map(str::to_owned),
                usage: message_delta.and_then(|e| e["usage"]["output_tokens"].as_u64()),
            }
        }
    }
}

/// Steps the model would run past the match: far more than any channel holds.
const STEPS_PAST_THE_MATCH: usize = 197;

/// A tool call in the Qwen XML form, as one piece.
const TOOL_CALL: &str =
    "<tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";

/// The same call with the stop string in its argument.
const TOOL_CALL_WITH_THE_STOP: &str =
    "<tool_call>\n<function=f>\n<parameter=x>\ncharlie\n</parameter>\n</function>\n</tool_call>";

/// Text that holds the stop string, and a whole call, in one piece: the call
/// closes in the push that carries the match.
const MATCH_AND_TOOL_CALL: &str = "Sure charlie <tool_call>\n<function=f>\n<parameter=x>\n1\n</parameter>\n</function>\n</tool_call>";

/// A call that the generation cuts before its closing tags.
const CUT_TOOL_CALL: &str = "<tool_call>\n<function=f>\n<parameter=x>\n1";

/// The pieces of `head`, then `STEPS_PAST_THE_MATCH` more steps of text.
fn generation(head: &[&'static str]) -> Vec<&'static str> {
    let mut pieces = head.to_vec();
    pieces.extend(std::iter::repeat_n("echo ", STEPS_PAST_THE_MATCH));
    pieces
}

/// What the client reads as the finish of a reply.
#[derive(Clone, Copy)]
enum Finish {
    /// A stop string matched and no tool call is in the reply.
    Stop,
    /// A tool call is in the reply, whether or not a stop string matched.
    ToolCalls,
    /// The generation ran to its end.
    Length,
}

impl Finish {
    /// The finish reason, and the stop string of an Anthropic reply.
    fn on(self, route: Route) -> (&'static str, Option<&'static str>) {
        match (self, route.is_anthropic()) {
            (Finish::Stop, false) => ("stop", None),
            (Finish::Stop, true) => ("stop_sequence", Some("charlie")),
            (Finish::ToolCalls, false) => ("tool_calls", None),
            (Finish::ToolCalls, true) => ("tool_use", None),
            (Finish::Length, false) => ("length", None),
            (Finish::Length, true) => ("max_tokens", None),
        }
    }
}

/// One request and what a correct server shows for it.
struct Row {
    name: &'static str,
    script: Script,
    stop: Option<&'static str>,
    temperature: f64,
    tools: bool,
    text: &'static str,
    reasoning: &'static str,
    tool_calls: usize,
    /// The steps the generator may run. Every step that runs is counted in
    /// `usage`, so `usage` must equal the count.
    steps: std::ops::RangeInclusive<usize>,
    end_requested: bool,
    finish: Finish,
}

impl Row {
    /// A request for `pieces` with the stop string `charlie` at temperature
    /// 0.7, which keeps it out of the replay envelope. The expectation is a
    /// reply that holds all of the text, runs all the steps and does not ask
    /// the engine to end.
    fn new(name: &'static str, pieces: &[&'static str], text: &'static str) -> Self {
        Self {
            name,
            script: Script::of(pieces),
            stop: Some("charlie"),
            temperature: 0.7,
            tools: true,
            text,
            reasoning: "",
            tool_calls: 0,
            steps: pieces.len()..=pieces.len(),
            end_requested: false,
            finish: Finish::Length,
        }
    }

    /// The row ends at step `steps`: the route asks the engine to end and the
    /// finish is `finish`.
    fn ends_at(mut self, steps: usize, finish: Finish) -> Self {
        self.steps = steps..=steps;
        self.end_requested = true;
        self.finish = finish;
        self
    }
}

/// Run `row` on `route` and list what is wrong with the reply and with what
/// the generator did.
async fn check(route: Route, row: &Row, wrong: &mut Vec<String>) {
    let (port, _tmp, probe) = start_scripted(row.script.clone()).await;
    let (status, raw) = http(port, route.path(), &route.body(row)).await;
    assert_eq!(status, 200, "body: {raw}");
    let seen = read(route, &raw);
    let at = format!("{route:?} / {}", row.name);
    if seen.text != row.text {
        wrong.push(format!(
            "{at}: text {:?}, expected {:?}",
            short(&seen.text),
            row.text
        ));
    }
    if seen.reasoning != row.reasoning {
        wrong.push(format!(
            "{at}: reasoning {:?}, expected {:?}",
            short(&seen.reasoning),
            row.reasoning
        ));
    }
    if seen.tool_calls != row.tool_calls {
        wrong.push(format!(
            "{at}: {} tool calls, expected {}",
            seen.tool_calls, row.tool_calls
        ));
    }
    let ran = probe.produced();
    if !row.steps.contains(&ran) {
        wrong.push(format!(
            "{at}: the generator ran {ran} steps, expected {:?}",
            row.steps
        ));
    }
    if seen.usage != Some(ran as u64) {
        wrong.push(format!(
            "{at}: usage {:?}, the generator ran {ran}",
            seen.usage
        ));
    }
    if !probe.terminal_read() {
        wrong.push(format!("{at}: the route did not read the terminal item"));
    }
    if probe.end_was_requested() != row.end_requested {
        wrong.push(format!(
            "{at}: end flag {}, expected {}",
            probe.end_was_requested(),
            row.end_requested
        ));
    }
    let (finish, stop_sequence) = row.finish.on(route);
    if seen.finish.as_deref() != Some(finish) || seen.stop_sequence.as_deref() != stop_sequence {
        wrong.push(format!(
            "{at}: finish {:?} / stop_sequence {:?}, expected {finish:?} / {stop_sequence:?}",
            seen.finish, seen.stop_sequence
        ));
    }
}

/// The start of a reply, so a failure that holds all 200 steps stays readable.
fn short(text: &str) -> String {
    let head: String = text.chars().take(40).collect();
    if head.len() < text.len() {
        format!("{head}...")
    } else {
        head
    }
}

async fn check_all(rows: &[Row]) {
    let mut wrong = Vec::new();
    for route in ROUTES {
        for row in rows {
            check(route, row, &mut wrong).await;
        }
    }
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// The stop string is in step 3 of 200. The route asks the engine to end, the
/// generator runs 3 steps, the route reads on to the terminal item, `usage`
/// counts the 3 steps, and the reply is cut before the string. Temperature 0.7
/// keeps the request out of the replay envelope, so the count is exact. The
/// second row sends no `tools`.
#[tokio::test]
async fn a_match_ends_the_generation_on_every_route() {
    let head = ["alpha ", "bravo ", "charlie delta "];
    let mut no_tools =
        Row::new("match, no tools", &generation(&head), "alpha bravo ").ends_at(3, Finish::Stop);
    no_tools.tools = false;
    check_all(&[
        Row::new("match", &generation(&head), "alpha bravo ").ends_at(3, Finish::Stop),
        no_tools,
    ])
    .await;
}

/// The same at temperature 0, where the replay envelope holds a channel of 4
/// between the generator and the route. The stub has no channel of its own,
/// so at the match the generator can be 5 steps ahead of the route: 4 in the
/// channel and 1 waiting to enter it.
#[tokio::test]
async fn a_replayed_request_ends_within_the_channel_bound() {
    let mut row = Row::new(
        "match at temperature 0",
        &generation(&["alpha ", "bravo ", "charlie delta "]),
        "alpha bravo ",
    )
    .ends_at(3, Finish::Stop);
    row.temperature = 0.0;
    row.steps = 3..=8;
    check_all(&[row]).await;
}

/// An engine that runs 2 steps after the flag is set: the route counts both,
/// shows neither, and the reply is still cut at the match.
#[tokio::test]
async fn a_late_end_counts_the_steps_that_ran() {
    let mut row = Row::new(
        "match, engine ends 2 steps late",
        &generation(&["alpha ", "bravo ", "charlie delta "]),
        "alpha bravo ",
    )
    .ends_at(5, Finish::Stop);
    row.script.lag = 2;
    check_all(&[row]).await;
}

/// One rule on all four routes, whether or not a tool call is in the reply. A
/// tool call before the match stays, and so does the text around it up to the
/// match: the finish is `tool_calls` or `tool_use`, with no stop string. A
/// tool call after the match is not in the reply, and the finish is `stop` or
/// `stop_sequence` with the string.
#[tokio::test]
async fn a_tool_call_changes_no_stop_rule() {
    check_all(&[
        Row::new(
            "tool call before the match",
            &generation(&["Sure. ", TOOL_CALL, "done charlie tail "]),
            "Sure. done ",
        )
        .ends_at(3, Finish::ToolCalls)
        .with_tool_calls(1),
        Row::new(
            "tool call after the match",
            &generation(&["Sure charlie ", TOOL_CALL, "tail "]),
            "Sure ",
        )
        .ends_at(1, Finish::Stop),
        Row::new(
            "tool call closing in the push of the match",
            &generation(&[MATCH_AND_TOOL_CALL, "tail "]),
            "Sure ",
        )
        .ends_at(1, Finish::Stop),
    ])
    .await;
}

impl Row {
    fn with_tool_calls(mut self, n: usize) -> Self {
        self.tool_calls = n;
        self
    }
}

/// A stop string matches the content channel only. The same string inside the
/// arguments of a tool call, or inside the reasoning, is no match: the
/// generator runs on, the flag stays down, and the reply holds the string.
#[tokio::test]
async fn a_stop_string_matches_the_content_channel_only() {
    let mut in_reasoning = Row::new(
        "stop string in the reasoning",
        &["think charlie ", "more ", "answer "],
        "answer ",
    );
    in_reasoning.script.reasoning = 2;
    in_reasoning.reasoning = "think charlie more ";
    check_all(&[
        Row::new(
            "stop string in the arguments of a tool call",
            &["Sure. ", TOOL_CALL_WITH_THE_STOP, "done "],
            "Sure. done ",
        )
        .with_finish(Finish::ToolCalls)
        .with_tool_calls(1),
        in_reasoning,
    ])
    .await;
}

impl Row {
    fn with_finish(mut self, finish: Finish) -> Self {
        self.finish = finish;
        self
    }
}

/// What the parser holds at the end of the generation reaches the reply on
/// every route: a tool call cut before its closing tags is recovered, and the
/// start of a tag that never completed is text, or reasoning when the
/// generation was in its reasoning.
#[tokio::test]
async fn what_the_tool_parser_holds_at_the_end_reaches_the_reply() {
    let mut cut = Row::new("cut tool call", &[CUT_TOOL_CALL], "")
        .with_finish(Finish::ToolCalls)
        .with_tool_calls(1);
    cut.stop = None;
    let mut prefix = Row::new("held-back tag prefix", &["done <to"], "done <to");
    prefix.stop = None;
    // A reasoning stream cut after `<`: the tail is reasoning, as the pieces
    // before it were.
    let mut reasoning_prefix = Row::new("held-back tag prefix in reasoning", &["think <"], "");
    reasoning_prefix.stop = None;
    reasoning_prefix.script.reasoning = 1;
    reasoning_prefix.reasoning = "think <";
    check_all(&[cut, prefix, reasoning_prefix]).await;
}

/// A request whose stop string never matches changes nothing: the generator
/// runs all its steps, the terminal item is read, the end flag stays down and
/// the reply carries the finish of the generation. Four ways to get there: a
/// stop string that does not occur, no stop string at all, a tool call in the
/// reply, and temperature 0.
#[tokio::test]
async fn no_match_runs_the_whole_generation() {
    let plain = ["alpha ", "bravo ", "delta "];
    let mut no_stop = Row::new("no stop string", &plain, "alpha bravo delta ");
    no_stop.stop = None;
    let mut at_zero = Row::new("temperature 0", &plain, "alpha bravo delta ");
    at_zero.temperature = 0.0;
    check_all(&[
        Row::new("stop string never occurs", &plain, "alpha bravo delta "),
        no_stop,
        Row::new(
            "tool call and no match",
            &["Sure. ", TOOL_CALL, "done "],
            "Sure. done ",
        )
        .with_finish(Finish::ToolCalls)
        .with_tool_calls(1),
        at_zero,
    ])
    .await;
}
