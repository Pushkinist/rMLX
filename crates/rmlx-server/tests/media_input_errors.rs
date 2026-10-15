//! HTTP-surface tests for the status of an engine fault on the two chat routes.
//!
//! The observable is the status and the error `type` a client reads. A media
//! input the client got wrong reaches the route as `Error::InvalidInput` from
//! the engine; a fault of the server reaches it as `Error::Other`. The first is
//! a 400 `invalid_request_error` on both surfaces, so an SDK does not send the
//! same input again. The second stays a 503, which an SDK retries.
//!
//! A scripted generator fails with one fault, either before its first item or
//! after one token. A stream that fails after its first token has sent its
//! status already, so the error event carries the `type`.
//!
//! Mutations and the test that catches each:
//! - the new variant maps to 503 on one surface:
//!   `a_client_fault_before_the_first_token_is_400` fails on that surface's rows.
//! - one `engine_error_type` mirror keeps the old type:
//!   `a_client_fault_after_the_first_token_keeps_its_type` fails on that
//!   surface's stream row.
//! - the server fault maps to 400: `a_server_fault_stays_503`.

// Test harness: panics/unwraps are acceptable in test bodies.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    trivial_casts
)]

use std::pin::Pin;
use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::{self, Stream};
use rmlx_server::metrics_drainer::spawn_drainer;
use rmlx_server::{
    ApiErrorCounters, AppState, GenerationRequest, GenerationToken, Generator, ItlStore,
    LoadedModel, ModelLoader, ModelRegistry, SessionCache, TtftStore,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const CLIENT_FACT: &str = "input_audio decode failed: the file holds no audio track";
const SERVER_FACT: &str = "the audio tower did not answer";

#[derive(Clone, Copy)]
enum Fault {
    Client,
    Server,
    /// A sequence the engine finds past the KV ceiling.
    Ceiling,
}

impl Fault {
    fn error(self) -> rmlx_core::Error {
        match self {
            Fault::Client => rmlx_core::Error::InvalidInput(CLIENT_FACT.to_owned()),
            Fault::Server => rmlx_core::Error::Other(SERVER_FACT.to_owned()),
            Fault::Ceiling => rmlx_core::Error::KvCeilingExceeded {
                requested: 641,
                ceiling: 640,
            },
        }
    }
}

/// Fails with `fault` after `tokens_before` tokens.
#[derive(Clone)]
struct FailingGenerator {
    fault: Fault,
    tokens_before: u32,
}

impl Generator for FailingGenerator {
    fn generate(
        &self,
        _req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        let mut items: Vec<rmlx_core::Result<GenerationToken>> = (0..self.tokens_before)
            .map(|id| {
                Ok(GenerationToken {
                    token_id: id,
                    piece: "a".to_owned(),
                    done: false,
                    finish_reason: None,
                    is_thinking: false,
                    logprobs: None,
                })
            })
            .collect();
        items.push(Err(self.fault.error()));
        Box::pin(stream::iter(items))
    }
}

async fn start_server(generator: FailingGenerator) -> (u16, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let snap = tmp.path().join("scripted");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(
        snap.join("config.json"),
        r#"{"architectures":["Scripted"]}"#,
    )
    .unwrap();
    std::fs::write(
        snap.join("chat_template.jinja"),
        "{% for m in messages %}{{ m['content'] }}{% endfor %}",
    )
    .unwrap();
    std::fs::write(
        snap.join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"x":0,"hi":1,"[UNK]":2},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();
    let metrics_db = tmp.path().join("metrics").join("runs.db");
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let loader_generator = generator.clone();
    let loader: ModelLoader =
        Arc::new(move |_path, _id| Ok(Box::new(loader_generator.clone()) as Box<dyn Generator>));
    let state = AppState {
        device: rmlx_mlx::Device::Cpu,
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
        load_gate: Arc::default(),
        metrics: None,
        idle_policy: rmlx_server::KeepAlivePolicy::Pin,
        max_tokens_cap: u32::MAX,
        max_timeout_secs: 600,
        session_cache: Arc::new(parking_lot::Mutex::new(SessionCache::new(4))),
        prompt_cache_slots: 4,
        ttft_store: TtftStore::default(),
        itl_store: ItlStore::default(),
        metrics_drainer: Some(spawn_drainer(metrics_db)),
        require_smoke_probe: false,
        default_temperature: None,
        default_enable_thinking: None,
        default_image_max_tokens: None,
        tokens_in: Arc::new(AtomicU64::new(0)),
        tokens_out: Arc::new(AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        started_at: std::time::Instant::now(),
        requests_started: Arc::new(AtomicU64::new(0)),
        requests_completed: Arc::new(AtomicU64::new(0)),
        requests_failed: Arc::new(AtomicU64::new(0)),
        admission_controller: None,
        admission_handle: None,
        whisper_model_path: None,
        whisper_tokenizer_path: None,
        audio_model: Arc::new(parking_lot::RwLock::new(None)),
        tts_model_path: None,
        tts_tokenizer_path: None,
        tts_model: Arc::new(parking_lot::RwLock::new(None)),
    };
    let now = std::time::Instant::now();
    state.slots.write().push(LoadedModel {
        id: "scripted".to_owned(),
        model: Arc::new(generator),
        loaded_at: now,
        last_used: now,
        effective_max_ctx: usize::MAX,
        context_limits: None,
        decode_lease: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        unload_handle: Arc::new(parking_lot::Mutex::new(None)),
        keep_alive: rmlx_server::KeepAlivePolicy::Pin,
    });
    let router = rmlx_server::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    (port, tmp)
}

/// The status and the body of one POST.
async fn post(port: u16, path: &str, body: &str) -> (u16, String) {
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

    fn body(self) -> String {
        json!({
            "model": "scripted",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 8,
            "stream": self.streams(),
        })
        .to_string()
    }

    /// The `type` a surface gives a fault.
    fn type_of(self, fault: Fault) -> &'static str {
        match (self.is_anthropic(), fault) {
            (_, Fault::Client) => "invalid_request_error",
            (_, Fault::Ceiling) => "context_length_exceeded",
            (false, Fault::Server) => "service_unavailable",
            (true, Fault::Server) => "service_unavailable_error",
        }
    }
}

/// The `error` object of a body that is a plain JSON error.
fn error_of_json(body: &str) -> Value {
    let v: Value = serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"));
    v["error"].clone()
}

/// The `error` object of the first error event of an SSE body.
fn error_of_sse(body: &str) -> Value {
    body.lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str::<Value>(data.trim()).ok())
        .find_map(|event| event.get("error").cloned())
        .unwrap_or_else(|| panic!("no error event in: {body}"))
}

fn assert_no_rows_wrong(wrong: &[String]) {
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// A media input the client got wrong, refused before the first token, is a 400
/// `invalid_request_error` that names the fact, on both surfaces and both modes.
#[tokio::test]
async fn a_client_fault_before_the_first_token_is_400() {
    let mut wrong = Vec::new();
    for route in ROUTES {
        let (port, _tmp) = start_server(FailingGenerator {
            fault: Fault::Client,
            tokens_before: 0,
        })
        .await;
        let (status, body) = post(port, route.path(), &route.body()).await;
        let err = error_of_json(&body);
        if status != 400 {
            wrong.push(format!("{route:?}: status {status}, expected 400: {body}"));
        }
        if err["type"] != route.type_of(Fault::Client) {
            wrong.push(format!("{route:?}: type {}, body {body}", err["type"]));
        }
        if !err["message"].as_str().unwrap_or("").contains(CLIENT_FACT) {
            wrong.push(format!("{route:?}: the fact is not in: {body}"));
        }
    }
    assert_no_rows_wrong(&wrong);
}

/// A fault of the server stays a 503, which an SDK retries.
#[tokio::test]
async fn a_server_fault_stays_503() {
    let mut wrong = Vec::new();
    for route in ROUTES {
        let (port, _tmp) = start_server(FailingGenerator {
            fault: Fault::Server,
            tokens_before: 0,
        })
        .await;
        let (status, body) = post(port, route.path(), &route.body()).await;
        let err = error_of_json(&body);
        if status != 503 {
            wrong.push(format!("{route:?}: status {status}, expected 503: {body}"));
        }
        if err["type"] != route.type_of(Fault::Server) {
            wrong.push(format!("{route:?}: type {}, body {body}", err["type"]));
        }
    }
    assert_no_rows_wrong(&wrong);
}

/// A fault after one token: a blocking reply is a 400 as before the first
/// token, and a stream, whose status is sent already, carries the same `type`
/// in its error event.
#[tokio::test]
async fn a_client_fault_after_the_first_token_keeps_its_type() {
    let mut wrong = Vec::new();
    for route in ROUTES {
        let (port, _tmp) = start_server(FailingGenerator {
            fault: Fault::Client,
            tokens_before: 1,
        })
        .await;
        let (status, body) = post(port, route.path(), &route.body()).await;
        let (expected_status, err) = if route.streams() {
            (200, error_of_sse(&body))
        } else {
            (400, error_of_json(&body))
        };
        if status != expected_status {
            wrong.push(format!(
                "{route:?}: status {status}, expected {expected_status}: {body}"
            ));
        }
        if err["type"] != route.type_of(Fault::Client) {
            wrong.push(format!("{route:?}: type {}, body {body}", err["type"]));
        }
        if !err["message"].as_str().unwrap_or("").contains(CLIENT_FACT) {
            wrong.push(format!("{route:?}: the fact is not in: {body}"));
        }
    }
    assert_no_rows_wrong(&wrong);
}

/// A sequence past the KV ceiling, which image or audio tokens can cause after
/// the route's prompt check, is the 400 `context_length_exceeded` of that
/// check, before or after the first token, on both surfaces.
#[tokio::test]
async fn a_sequence_past_the_ceiling_is_400_context_length_exceeded() {
    let mut wrong = Vec::new();
    for tokens_before in [0, 1] {
        for route in ROUTES {
            let (port, _tmp) = start_server(FailingGenerator {
                fault: Fault::Ceiling,
                tokens_before,
            })
            .await;
            let (status, body) = post(port, route.path(), &route.body()).await;
            let in_stream = route.streams() && tokens_before > 0;
            let (expected_status, err) = if in_stream {
                (200, error_of_sse(&body))
            } else {
                (400, error_of_json(&body))
            };
            let at = format!("{route:?} after {tokens_before} tokens");
            if status != expected_status {
                wrong.push(format!("{at}: status {status}, expected {expected_status}"));
            }
            if err["type"] != route.type_of(Fault::Ceiling) {
                wrong.push(format!("{at}: type {}, body {body}", err["type"]));
            }
            if err["message"] != "the request needs 641 tokens of context, max_ctx is 640" {
                wrong.push(format!("{at}: message {}", err["message"]));
            }
        }
    }
    assert_no_rows_wrong(&wrong);
}
