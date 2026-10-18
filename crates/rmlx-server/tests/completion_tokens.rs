//! HTTP-surface tests for `completion_tokens` / `output_tokens`.
//!
//! The observable is the figure a client reads: `usage.completion_tokens` of a
//! non-streamed OpenAI reply, the usage chunk of a streamed one, and
//! `usage.output_tokens` of the Anthropic reply and of its `message_delta`
//! event. Two more outputs carry the same figure and are read beside it: the
//! process-lifetime `tokens_out` counter and the `completion_tokens_live`
//! observation that lands in the metrics DB.
//!
//! A scripted generator replays one fixed generation into every path. It
//! mirrors the engine's emit shape: one item per decode step, then one
//! terminal item (`done == true`) that carries the finish reason, the id of
//! the last step and the bytes left over when the generation ends inside a
//! multi-byte character (empty otherwise). The terminal item is no generated
//! token. The expected figure of each case is the number of decode steps in
//! its script, written as a literal and checked against the script.
//!
//! What cannot move: a `length` reply never reports more than `max_tokens`;
//! the figure is the same on the blocking and the streaming path of one API,
//! also when a stop string matches; a step that adds no text (the token that
//! closes reasoning, the end-of-turn token) is a generated token and is
//! counted; reasoning steps are counted; the text on the terminal item reaches
//! the reply and adds no token; `tokens_out` and the metrics observation equal
//! the figure in `usage`.
//!
//! Mutations and the test that catches each:
//! - the OpenAI blocking path counts the terminal item again:
//!   `usage_counts_the_decode_steps`, `a_length_reply_never_exceeds_max_tokens`
//!   and `blocking_and_streaming_agree` fail on the `openai` rows.
//! - the OpenAI streaming path counts it again: the same three tests, on the
//!   `openai-stream` rows.
//! - either Anthropic path counts it again: the same three tests, on the
//!   `anthropic` and `anthropic-stream` rows.
//! - a step with an empty piece stops counting (the end-of-turn token, the
//!   token that closes reasoning): `usage_counts_the_decode_steps` fails on
//!   `end_of_turn` and `after_reasoning`.
//! - a reasoning step stops counting: `usage_counts_the_decode_steps` fails on
//!   `after_reasoning`.
//! - a generation cut below `max_tokens` reaches the client as `stop`, or with
//!   a count other than the cut: `a_cut_at_the_ceiling_reports_length_and_the_cut`.
//! - a path keeps the old figure for the process counter or the metrics
//!   observation: `counter_and_metrics_equal_usage`.
//! - the terminal item counts when it carries text (the predicate reads the
//!   piece as well as `done`): `usage_counts_the_decode_steps` fails on
//!   `mid_character`.
//! - a streaming path stops reading the generation at a stop-string match:
//!   `usage_counts_the_decode_steps` and `blocking_and_streaming_agree` fail
//!   on the `stop_string` rows (the match with an event queued, and the match
//!   with none).

// Test harness: panics/unwraps are acceptable in test bodies.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    trivial_casts
)]

use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
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

// ── Scripted generations ────────────────────────────────────────────────────

/// One decode step: the visible text it adds and the channel it is on.
#[derive(Clone, Copy)]
struct Step {
    piece: &'static str,
    thinking: bool,
}

const fn answer(piece: &'static str) -> Step {
    Step {
        piece,
        thinking: false,
    }
}

const fn reasoning(piece: &'static str) -> Step {
    Step {
        piece,
        thinking: true,
    }
}

/// One generation. `expected` is the number of decode steps, as a literal.
#[derive(Clone, Copy)]
struct Case {
    name: &'static str,
    steps: &'static [Step],
    finish: &'static str,
    max_tokens: u32,
    expected: u64,
    content: &'static str,
    reasoning: &'static str,
    /// The bytes left over when the generation ends inside a character; they
    /// ride on the terminal item.
    tail: &'static str,
    stop: Option<&'static str>,
}

/// Stops at `max_tokens`: three steps, three tokens in the content.
const LENGTH: Case = Case {
    name: "length",
    steps: &[answer("a"), answer("b"), answer("c")],
    finish: "length",
    max_tokens: 3,
    expected: 3,
    content: "abc",
    reasoning: "",
    tail: "",
    stop: None,
};

/// Stops on the end-of-turn token: one token of text, then the end-of-turn
/// step, which adds no text. Two steps, two generated tokens.
const END_OF_TURN: Case = Case {
    name: "end_of_turn",
    steps: &[answer("hi"), answer("")],
    finish: "stop",
    max_tokens: 16,
    expected: 2,
    content: "hi",
    reasoning: "",
    tail: "",
    stop: None,
};

/// Two reasoning steps, the step that closes the reasoning (no text), one
/// answer step, the end-of-turn step. Five steps, five generated tokens.
const AFTER_REASONING: Case = Case {
    name: "after_reasoning",
    steps: &[
        reasoning("Let"),
        reasoning(" me"),
        answer(""),
        answer("ok"),
        answer(""),
    ],
    finish: "stop",
    max_tokens: 16,
    expected: 5,
    content: "ok",
    reasoning: "Let me",
    tail: "",
    stop: None,
};

/// Ends inside a multi-byte character: the last step opens the character and
/// adds no text, and the replacement character for the cut bytes rides on the
/// terminal item. Three steps, three generated tokens, and the text of the
/// terminal item is in the content.
const MID_CHARACTER: Case = Case {
    name: "mid_character",
    steps: &[answer("a"), answer("b"), answer("")],
    finish: "length",
    max_tokens: 3,
    expected: 3,
    content: "ab\u{FFFD}",
    reasoning: "",
    tail: "\u{FFFD}",
    stop: None,
};

/// A stop string matches inside the second step of four, after text of that
/// step ("b") is released: a streaming path has an event queued when the match
/// happens. The engine keeps generating, so the figure is still the four
/// steps; the reply is cut before the match. Two steps follow the match, so a
/// path that reads one more item and then stops is caught too.
const STOP_STRING: Case = Case {
    name: "stop_string",
    steps: &[answer("a"), answer("bc"), answer("d"), answer("e")],
    finish: "length",
    max_tokens: 4,
    expected: 4,
    content: "ab",
    reasoning: "",
    tail: "",
    stop: Some("c"),
};

/// A stop string matches at the start of the second step of four, so the
/// match releases no text and a streaming path has no event queued at the
/// match. The other branch of the same code as `STOP_STRING`.
const STOP_STRING_AT_A_STEP_START: Case = Case {
    name: "stop_string_at_a_step_start",
    steps: &[answer("a"), answer("b"), answer("c"), answer("d")],
    finish: "length",
    max_tokens: 4,
    expected: 4,
    content: "a",
    reasoning: "",
    tail: "",
    stop: Some("b"),
};

/// The engine stops at the context ceiling before `max_tokens`: three steps of
/// a request for 64, ended with `length`. The reply must say `length` and
/// count the three.
const CUT_AT_THE_CEILING: Case = Case {
    name: "cut_at_the_ceiling",
    steps: &[answer("a"), answer("b"), answer("c")],
    finish: "length",
    max_tokens: 64,
    expected: 3,
    content: "abc",
    reasoning: "",
    tail: "",
    stop: None,
};

const CASES: [Case; 7] = [
    LENGTH,
    CUT_AT_THE_CEILING,
    END_OF_TURN,
    AFTER_REASONING,
    MID_CHARACTER,
    STOP_STRING,
    STOP_STRING_AT_A_STEP_START,
];

#[derive(Clone)]
struct ScriptedGenerator {
    case: Case,
}

impl Generator for ScriptedGenerator {
    fn generate(
        &self,
        _req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        let mut items: Vec<rmlx_core::Result<GenerationToken>> = self
            .case
            .steps
            .iter()
            .enumerate()
            .map(|(id, step)| {
                Ok(GenerationToken {
                    token_id: id as u32,
                    piece: step.piece.to_owned(),
                    done: false,
                    finish_reason: None,
                    is_thinking: step.thinking,
                    logprobs: None,
                })
            })
            .collect();
        items.push(Ok(GenerationToken::terminal(
            self.case.steps.len().saturating_sub(1) as u32,
            self.case.finish.to_owned(),
            self.case.tail.to_owned(),
        )));
        Box::pin(stream::iter(items))
    }
}

// ── AppState + server wiring ────────────────────────────────────────────────

struct Server {
    port: u16,
    tokens_out: Arc<AtomicU64>,
    metrics_db: PathBuf,
    _snapshot: tempfile::TempDir,
}

async fn start_server(case: Case) -> Server {
    assert_eq!(
        case.expected,
        case.steps.len() as u64,
        "harness: `expected` of {} is the number of decode steps",
        case.name
    );
    assert!(
        case.steps.len() <= case.max_tokens as usize,
        "harness: the engine never runs more than `max_tokens` steps ({})",
        case.name
    );
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
    let loader: ModelLoader =
        Arc::new(move |_path, _id| Ok(Box::new(ScriptedGenerator { case }) as Box<dyn Generator>));
    let tokens_out = Arc::new(AtomicU64::new(0));
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
        metrics_drainer: Some(spawn_drainer(metrics_db.clone())),
        require_smoke_probe: false,
        default_temperature: None,
        default_enable_thinking: None,
        default_image_max_tokens: None,
        tokens_in: Arc::new(AtomicU64::new(0)),
        tokens_out: Arc::clone(&tokens_out),
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
        model: Arc::new(ScriptedGenerator { case }),
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
    Server {
        port,
        tokens_out,
        metrics_db,
        _snapshot: tmp,
    }
}

async fn post(port: u16, path: &str, body: &str) -> String {
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
    assert_eq!(status, 200, "body: {}", &text[body_start..]);
    text[body_start..].to_owned()
}

/// The `completion_tokens_live` observations in the metrics DB, once the
/// drainer has flushed one.
async fn completion_tokens_observed(db: &Path) -> Vec<f64> {
    for _ in 0..50 {
        if let Ok(conn) = rusqlite::Connection::open(db) {
            let rows: Vec<f64> = conn
                .prepare("SELECT value FROM observations WHERE metric = 'completion_tokens_live'")
                .and_then(|mut q| q.query_map([], |r| r.get(0))?.collect())
                .unwrap_or_default();
            if !rows.is_empty() {
                return rows;
            }
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    Vec::new()
}

// ── The four paths ──────────────────────────────────────────────────────────

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
    fn label(self) -> &'static str {
        match self {
            Route::OpenAi => "openai",
            Route::OpenAiStream => "openai-stream",
            Route::Anthropic => "anthropic",
            Route::AnthropicStream => "anthropic-stream",
        }
    }

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

    fn body(self, case: Case) -> String {
        let mut body = json!({
            "model": "scripted",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": case.max_tokens,
            "stream": self.streams(),
        });
        if let Some(stop) = case.stop {
            let field = if self.is_anthropic() {
                "stop_sequences"
            } else {
                "stop"
            };
            body[field] = json!([stop]);
        }
        if !self.is_anthropic() {
            body["temperature"] = json!(0);
            if self.streams() {
                body["stream_options"] = json!({"include_usage": true});
            }
        }
        body.to_string()
    }
}

/// Every `data:` line of an SSE body that parses as JSON.
fn sse_events(raw: &str) -> Vec<Value> {
    raw.lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .filter_map(|data| serde_json::from_str(data.trim()).ok())
        .collect()
}

/// What one request showed: the figure in `usage`, the text it returned, and
/// the two other outputs that carry the figure.
struct Observed {
    usage: u64,
    /// The finish reason on the wire: `finish_reason` of an OpenAI reply or
    /// `stop_reason` of an Anthropic one.
    finish: String,
    content: String,
    reasoning: String,
    tokens_out: u64,
    metrics: Vec<f64>,
}

async fn observe(route: Route, case: Case) -> Observed {
    let server = start_server(case).await;
    let raw = post(server.port, route.path(), &route.body(case)).await;
    let (usage, finish, content, reasoning) = match route {
        Route::OpenAi => {
            let v: Value = serde_json::from_str(&raw).unwrap();
            let message = &v["choices"][0]["message"];
            (
                v["usage"]["completion_tokens"].as_u64().unwrap(),
                v["choices"][0]["finish_reason"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                message["content"].as_str().unwrap_or_default().to_owned(),
                message["reasoning_content"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
            )
        }
        Route::OpenAiStream => {
            let events = sse_events(&raw);
            let delta = |field: &str| -> String {
                events
                    .iter()
                    .filter_map(|e| e["choices"][0]["delta"][field].as_str())
                    .collect()
            };
            let usage = events
                .iter()
                .find_map(|e| e["usage"]["completion_tokens"].as_u64())
                .unwrap_or_else(|| panic!("no usage chunk in: {raw}"));
            let finish = events
                .iter()
                .find_map(|e| e["choices"][0]["finish_reason"].as_str())
                .unwrap_or_default()
                .to_owned();
            (usage, finish, delta("content"), delta("reasoning_content"))
        }
        Route::Anthropic => {
            let v: Value = serde_json::from_str(&raw).unwrap();
            let block = |kind: &str, field: &str| -> String {
                v["content"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|b| b["type"] == kind)
                    .filter_map(|b| b[field].as_str())
                    .collect()
            };
            (
                v["usage"]["output_tokens"].as_u64().unwrap(),
                v["stop_reason"].as_str().unwrap_or_default().to_owned(),
                block("text", "text"),
                block("thinking", "thinking"),
            )
        }
        Route::AnthropicStream => {
            let events = sse_events(&raw);
            let delta = |kind: &str, field: &str| -> String {
                events
                    .iter()
                    .filter(|e| e["delta"]["type"] == kind)
                    .filter_map(|e| e["delta"][field].as_str())
                    .collect()
            };
            let usage = events
                .iter()
                .find(|e| e["type"] == "message_delta")
                .and_then(|e| e["usage"]["output_tokens"].as_u64())
                .unwrap_or_else(|| panic!("no message_delta usage in: {raw}"));
            let finish = events
                .iter()
                .find(|e| e["type"] == "message_delta")
                .and_then(|e| e["delta"]["stop_reason"].as_str())
                .unwrap_or_default()
                .to_owned();
            (
                usage,
                finish,
                delta("text_delta", "text"),
                delta("thinking_delta", "thinking"),
            )
        }
    };
    let metrics = completion_tokens_observed(&server.metrics_db).await;
    Observed {
        usage,
        finish,
        content,
        reasoning,
        tokens_out: server.tokens_out.load(Ordering::Relaxed),
        metrics,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

/// Fail with every row that is wrong, not only the first.
fn assert_no_rows_wrong(wrong: &[String]) {
    assert!(wrong.is_empty(), "\n{}", wrong.join("\n"));
}

/// The figure in `usage` is the number of decode steps of the script, on every
/// path, and the script played out as written.
#[tokio::test]
async fn usage_counts_the_decode_steps() {
    let mut wrong = Vec::new();
    for route in ROUTES {
        for case in CASES {
            let seen = observe(route, case).await;
            let at = format!("{} / {}", route.label(), case.name);
            if seen.content != case.content || seen.reasoning != case.reasoning {
                wrong.push(format!(
                    "{at}: the script did not play out: content {:?}, reasoning {:?}",
                    seen.content, seen.reasoning
                ));
            }
            if seen.usage != case.expected {
                wrong.push(format!(
                    "{at}: usage {}, expected {} (one per decode step, none for the terminal item)",
                    seen.usage, case.expected
                ));
            }
        }
    }
    assert_no_rows_wrong(&wrong);
}

/// A reply that ends on `max_tokens` never reports more than `max_tokens`.
#[tokio::test]
async fn a_length_reply_never_exceeds_max_tokens() {
    let mut wrong = Vec::new();
    for route in ROUTES {
        for case in [
            LENGTH,
            MID_CHARACTER,
            STOP_STRING,
            STOP_STRING_AT_A_STEP_START,
        ] {
            let seen = observe(route, case).await;
            if seen.usage > u64::from(case.max_tokens) {
                wrong.push(format!(
                    "{} / {}: usage {} exceeds max_tokens {}",
                    route.label(),
                    case.name,
                    seen.usage,
                    case.max_tokens
                ));
            }
        }
    }
    assert_no_rows_wrong(&wrong);
}

/// A generation that the engine ends below `max_tokens`, at the context
/// ceiling, reaches the client as a length stop on every route, with the cut
/// as its token count.
#[tokio::test]
async fn a_cut_at_the_ceiling_reports_length_and_the_cut() {
    let mut wrong = Vec::new();
    for route in ROUTES {
        let seen = observe(route, CUT_AT_THE_CEILING).await;
        let want = if route.is_anthropic() {
            "max_tokens"
        } else {
            "length"
        };
        if seen.finish != want {
            wrong.push(format!(
                "{}: finish {:?}, expected {want:?}",
                route.label(),
                seen.finish
            ));
        }
        if seen.usage != CUT_AT_THE_CEILING.expected {
            wrong.push(format!(
                "{}: usage {}, expected {}",
                route.label(),
                seen.usage,
                CUT_AT_THE_CEILING.expected
            ));
        }
    }
    assert_no_rows_wrong(&wrong);
}

/// One API reports the same figure streamed and not streamed.
#[tokio::test]
async fn blocking_and_streaming_agree() {
    let mut wrong = Vec::new();
    for case in CASES {
        for (blocking, streamed) in [
            (Route::OpenAi, Route::OpenAiStream),
            (Route::Anthropic, Route::AnthropicStream),
        ] {
            let blocking_usage = observe(blocking, case).await.usage;
            let streamed_usage = observe(streamed, case).await.usage;
            if blocking_usage != streamed_usage {
                wrong.push(format!(
                    "{}: {} reports {blocking_usage}, {} reports {streamed_usage}",
                    case.name,
                    blocking.label(),
                    streamed.label()
                ));
            }
        }
    }
    assert_no_rows_wrong(&wrong);
}

/// The process counter and the metrics observation carry the figure of
/// `usage`, once per request.
#[tokio::test]
async fn counter_and_metrics_equal_usage() {
    let mut wrong = Vec::new();
    for route in ROUTES {
        for case in CASES {
            let seen = observe(route, case).await;
            let at = format!("{} / {}", route.label(), case.name);
            if seen.tokens_out != seen.usage {
                wrong.push(format!(
                    "{at}: tokens_out {}, usage {}",
                    seen.tokens_out, seen.usage
                ));
            }
            if seen.metrics != vec![seen.usage as f64] {
                wrong.push(format!(
                    "{at}: completion_tokens_live {:?}, usage {}",
                    seen.metrics, seen.usage
                ));
            }
        }
    }
    assert_no_rows_wrong(&wrong);
}
