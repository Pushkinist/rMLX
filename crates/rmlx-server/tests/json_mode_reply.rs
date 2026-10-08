//! HTTP-surface tests for the reply of a `response_format` request.
//!
//! The observable is what a client reads: the `content` bytes of each SSE
//! delta of a streamed reply, and `message.content` of a non-streamed reply.
//! A scripted generator replays one fixed generation into both paths. It
//! drives the constraint the route built in the way the decode loop does (one
//! `step_mask` and one `advance` for each token), so the grammar engages where
//! it engages on a model, and a script that the grammar refuses is a harness
//! failure.
//!
//! The rule the tests hold: the reply is the generated answer text from the
//! byte at which the grammar engaged to the end of the generation, with no
//! byte dropped and no byte added after that point, on both paths.
//!
//! A test with `#[should_panic]` holds a reply that is wrong today. Its
//! `expected` string is the tag of the one assertion that must fail, so a
//! harness failure does not satisfy it. When the reply is corrected, the test
//! fails with "test did not panic" and the attribute must go.

// Test harness: panics/unwraps are acceptable in test bodies.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    trivial_casts
)]

use std::pin::Pin;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use futures::stream::{self, Stream};
use parking_lot::Mutex;
use rmlx_server::{
    ApiErrorCounters, AppState, GenerationRequest, GenerationToken, Generator, ItlStore,
    LoadedModel, ModelLoader, ModelRegistry, SessionCache, TtftStore,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Vocabulary ──────────────────────────────────────────────────────────────

/// The token id of a piece is its index. Id 0 is the EOS id of the fixture's
/// `config.json`.
const VOCAB: &[&str] = &[
    "<eos>",
    "[UNK]",
    "hi",
    "{",
    "}",
    "[",
    ":",
    ",",
    "\"a\"",
    " 1",
    " {",
    "\"b\"",
    " \"c\"",
    "\"Afghanistan\"",
    " \"Kabul\"",
    " \"Albania\"",
    "10",
    " 20",
    "```",
    "json",
    "\n",
    "\n{",
    "\n\n",
    "Here",
    " is",
    " a",
    " JSON",
    " object",
    " are",
    " 30",
    " capitals",
    "I",
    " cannot",
    " do",
    " that",
    "true",
    "null",
    "42",
    "\"medium\"",
    "\"me",
    "Let",
    " me",
    " think",
    "Note",
    " x",
    " true",
    "\"name\"",
    "\"f\"",
    "\"arguments\"",
    "}\n",
    "dium\"",
    "\"a{b\"",
    "-",
    " {\"x\"}",
];

fn token_id(piece: &str) -> u32 {
    VOCAB
        .iter()
        .position(|p| *p == piece)
        .unwrap_or_else(|| panic!("harness: piece {piece:?} is not in VOCAB")) as u32
}

fn tokenizer_json() -> String {
    let vocab: serde_json::Map<String, Value> = VOCAB
        .iter()
        .enumerate()
        .map(|(id, piece)| ((*piece).to_owned(), json!(id)))
        .collect();
    json!({
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null,
        "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    })
    .to_string()
}

// ── Scripted generations ────────────────────────────────────────────────────

/// One generation: reasoning pieces, answer pieces, the terminal reason.
#[derive(Clone, Copy)]
struct Script {
    thinking: &'static [&'static str],
    answer: &'static [&'static str],
    finish: &'static str,
}

const fn stop(answer: &'static [&'static str]) -> Script {
    Script {
        thinking: &[],
        answer,
        finish: "stop",
    }
}

const fn length(answer: &'static [&'static str]) -> Script {
    Script {
        thinking: &[],
        answer,
        finish: "length",
    }
}

const OBJECT: Script = stop(&["{", "\"a\"", ":", " 1", "}"]);
const OBJECT_AT_LENGTH: Script = length(&["{", "\"a\"", ":", " 1", "}"]);
const OBJECT_AFTER_REASONING: Script = Script {
    thinking: &["Let", " me", " think", " {\"x\"}"],
    answer: &["{", "\"a\"", ":", " 1", "}"],
    finish: "stop",
};
const OBJECT_THEN_NEWLINE: Script = stop(&["{", "\"a\"", ":", " 1", "}\n"]);
const BARE_FENCE: Script = stop(&["```", "\n", "{", "\"a\"", ":", " 1", "}"]);
const BARE_FENCE_MID_TOKEN: Script = stop(&["```", "\n{", "\"a\"", ":", " 1", "}"]);
const JSON_FENCE: Script = stop(&["```", "json", "\n", "{", "\"a\"", ":", " 1", "}"]);
const PROSE: Script = stop(&[
    "Here", " is", " a", " JSON", " object", ":", "\n\n", "{", "\"a\"", ":", " 1", "}",
]);
const PROSE_WITH_NUMBER: Script = stop(&[
    "Here",
    " are",
    " 30",
    " capitals",
    ":",
    "\n\n",
    "{",
    "\"a\"",
    ":",
    " 1",
    "}",
]);
const PROSE_WITH_QUOTED_WORD: Script =
    stop(&["Note", " \"c\"", ":", "\n\n", "{", "\"a\"", ":", " 1", "}"]);
const PROSE_WITH_DASH: Script = stop(&["-", " JSON", ":", "\n", "{", "\"a\"", ":", " 1", "}"]);
const CUT_OBJECT: Script = length(&[
    "{",
    "\"Afghanistan\"",
    ":",
    " \"Kabul\"",
    ",",
    " \"Albania\"",
    ":",
]);
const CUT_OBJECT_IN_JSON_FENCE: Script = length(&[
    "```",
    "json",
    "\n",
    "{",
    "\"Afghanistan\"",
    ":",
    " \"Kabul\"",
    ",",
    " \"Albania\"",
    ":",
]);
const CUT_AFTER_INNER_OBJECT: Script = length(&[
    "{", "\"a\"", ":", " {", "\"b\"", ":", " 1", "}", ",", " \"c\"", ":",
]);
const CUT_ARRAY: Script = length(&["[", "10", ",", " 20", ","]);
const NO_JSON: Script = stop(&["I", " cannot", " do", " that"]);

const SCALAR_TRUE: Script = stop(&["true"]);
const SCALAR_NULL: Script = stop(&["null"]);
const SCALAR_INTEGER: Script = stop(&["42"]);
const SCALAR_STRING: Script = stop(&["\"medium\""]);
const SCALAR_STRING_IN_TWO_PIECES: Script = stop(&["\"me", "dium\""]);
const SCALAR_STRING_WITH_BRACE: Script = stop(&["\"a{b\""]);
const CUT_SCALAR_STRING: Script = length(&["\"me"]);

const PLAIN_TEXT: Script = stop(&["Note", ":", " 30", " {", " x", "}", " true"]);
const TOOL_CALL: Script = stop(&[
    "{",
    "\"name\"",
    ":",
    "\"f\"",
    ",",
    "\"arguments\"",
    ":",
    "{",
    "}",
    "}",
]);

const CUT_TOOL_CALL: Script = length(&["{", "\"name\"", ":", "\"f\"", ",", "\"arguments\"", ":"]);

// ── Scripted generator ──────────────────────────────────────────────────────

/// What the generator saw. `violations` is empty for a sound script.
#[derive(Default)]
struct Seen {
    violations: Vec<String>,
    /// `Some` when the route built a constraint: the engine's `finished()`
    /// after the last token.
    constraint_finished: Option<bool>,
}

#[derive(Clone)]
struct ScriptedGenerator {
    script: Script,
    seen: Arc<Mutex<Seen>>,
}

fn token(piece: &str, is_thinking: bool) -> GenerationToken {
    GenerationToken {
        token_id: token_id(piece),
        piece: piece.to_owned(),
        done: false,
        finish_reason: None,
        is_thinking,
        logprobs: None,
    }
}

impl Generator for ScriptedGenerator {
    fn generate(
        &self,
        mut req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        let mut seen = self.seen.lock();
        let mut constraint = req.constraint.take();
        let mut toks = Vec::new();
        let pieces = self
            .script
            .thinking
            .iter()
            .map(|p| (*p, true))
            .chain(self.script.answer.iter().map(|p| (*p, false)));
        for (piece, is_thinking) in pieces {
            let id = token_id(piece);
            if let Some(flag) = req.is_thinking_handle.as_ref() {
                flag.store(is_thinking, Ordering::Relaxed);
            }
            if let Some(c) = constraint.as_mut() {
                if !c.step_mask(VOCAB.len())[id as usize] {
                    seen.violations
                        .push(format!("the grammar refuses piece {piece:?}"));
                }
                c.advance(id);
            }
            toks.push(Ok(token(piece, is_thinking)));
        }
        seen.constraint_finished = constraint.as_ref().map(|c| c.finished());
        toks.push(Ok(GenerationToken {
            token_id: 0,
            piece: String::new(),
            done: true,
            finish_reason: Some(self.script.finish.to_owned()),
            is_thinking: false,
            logprobs: None,
        }));
        Box::pin(stream::iter(toks))
    }
}

// ── AppState + server wiring ────────────────────────────────────────────────

fn scripted_state(script: Script) -> (AppState, Arc<Mutex<Seen>>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let snap = tmp.path().join("scripted");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(
        snap.join("config.json"),
        r#"{"architectures":["Scripted"],"eos_token_id":0}"#,
    )
    .unwrap();
    std::fs::write(
        snap.join("chat_template.jinja"),
        "{% for m in messages %}{{ m['content'] }}{% endfor %}",
    )
    .unwrap();
    std::fs::write(snap.join("tokenizer.json"), tokenizer_json()).unwrap();

    let seen = Arc::new(Mutex::new(Seen::default()));
    let generator = ScriptedGenerator {
        script,
        seen: Arc::clone(&seen),
    };
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
        gpu_gate: Arc::new(Mutex::new(())),
        gpu_queue: Arc::new(tokio::sync::Semaphore::new(1)),
        gpu_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_queue_depth: 64,
        max_loaded_models: 1,
        loader,
        metrics: None,
        idle_policy: rmlx_server::KeepAlivePolicy::Pin,
        max_tokens_cap: u32::MAX,
        max_timeout_secs: 600,
        session_cache: Arc::new(Mutex::new(SessionCache::new(4))),
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
    let now = std::time::Instant::now();
    state.slots.write().push(LoadedModel {
        id: "scripted".to_owned(),
        model: Arc::new(generator),
        loaded_at: now,
        last_used: now,
        effective_max_ctx: usize::MAX,
        context_limits: None,
        decode_lease: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        unload_handle: Arc::new(Mutex::new(None)),
        keep_alive: rmlx_server::KeepAlivePolicy::Pin,
    });
    (state, seen, tmp)
}

async fn http(port: u16, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
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

// ── The reply a client reads ────────────────────────────────────────────────

/// The request modes under test.
#[derive(Clone, Copy)]
enum Mode {
    JsonObject,
    /// `json_schema` with this root type.
    SchemaRoot(&'static str),
    Text,
    NoFormat,
    RequiredTool,
}

#[derive(Debug, PartialEq, Eq)]
struct Reply {
    /// The content bytes of each delta, in order. A non-streamed reply has
    /// one entry.
    content: Vec<String>,
    reasoning: String,
    finish: String,
    completion_tokens: u64,
    tool_names: Vec<String>,
}

impl Reply {
    fn text(&self) -> String {
        self.content.concat()
    }
}

/// A schema with this root type. The object root accepts `{"a": <integer>}`.
fn schema_of(root: &str) -> Value {
    if root == "object" {
        json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}},
            "required": ["a"],
            "additionalProperties": false
        })
    } else {
        json!({"type": root})
    }
}

fn request_body(mode: Mode, stream: bool) -> String {
    let mut body = json!({
        "model": "scripted",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 64,
        "temperature": 0,
        "stream": stream,
    });
    if stream {
        body["stream_options"] = json!({"include_usage": true});
    }
    match mode {
        Mode::JsonObject => body["response_format"] = json!({"type": "json_object"}),
        Mode::SchemaRoot(root) => {
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {"name": "root", "strict": true, "schema": schema_of(root)}
            });
        }
        Mode::Text => body["response_format"] = json!({"type": "text"}),
        Mode::NoFormat => {}
        Mode::RequiredTool => {
            body["tools"] = json!([{
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {"type": "object", "properties": {}}
                }
            }]);
            body["tool_choice"] = json!("required");
        }
    }
    body.to_string()
}

fn parse_streamed(raw: &str) -> Reply {
    let mut reply = Reply {
        content: Vec::new(),
        reasoning: String::new(),
        finish: String::new(),
        completion_tokens: 0,
        tool_names: Vec::new(),
    };
    for line in raw.lines() {
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let Ok(v) = serde_json::from_str::<Value>(data.trim()) else {
            continue;
        };
        let choice = &v["choices"][0];
        if let Some(c) = choice["delta"]["content"].as_str() {
            reply.content.push(c.to_owned());
        }
        if let Some(r) = choice["delta"]["reasoning_content"].as_str() {
            reply.reasoning.push_str(r);
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            f.clone_into(&mut reply.finish);
        }
        if let Some(n) = v["usage"]["completion_tokens"].as_u64() {
            reply.completion_tokens = n;
        }
        if let Some(calls) = choice["delta"]["tool_calls"].as_array() {
            for call in calls {
                if let Some(name) = call["function"]["name"].as_str() {
                    reply.tool_names.push(name.to_owned());
                }
            }
        }
    }
    reply
}

fn parse_blocking(raw: &str) -> Reply {
    let v: Value = serde_json::from_str(raw).unwrap_or_else(|e| panic!("harness: {e}: {raw}"));
    let choice = &v["choices"][0];
    let message = &choice["message"];
    Reply {
        content: vec![message["content"].as_str().unwrap_or_default().to_owned()],
        reasoning: message["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        finish: choice["finish_reason"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        completion_tokens: v["usage"]["completion_tokens"].as_u64().unwrap_or_default(),
        tool_names: message["tool_calls"]
            .as_array()
            .map(|calls| {
                calls
                    .iter()
                    .filter_map(|c| c["function"]["name"].as_str().map(str::to_owned))
                    .collect()
            })
            .unwrap_or_default(),
    }
}

/// Run `script` through one request and return `(status, raw body, what the
/// generator saw)`.
async fn serve(script: Script, mode: Mode, stream: bool) -> (u16, String, Arc<Mutex<Seen>>) {
    let (state, seen, _tmp) = scripted_state(script);
    let router = rmlx_server::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    let (status, raw) = http(port, &request_body(mode, stream)).await;
    (status, raw, seen)
}

/// The reply of a request that must answer 200 from a sound script.
async fn reply(script: Script, mode: Mode, stream: bool) -> Reply {
    let (status, raw, seen) = serve(script, mode, stream).await;
    assert_eq!(status, 200, "harness: body: {raw}");
    let seen = seen.lock();
    assert!(
        seen.violations.is_empty(),
        "harness: unsound script: {:?}",
        seen.violations
    );
    if stream {
        parse_streamed(&raw)
    } else {
        parse_blocking(&raw)
    }
}

/// The answer pieces from the first piece that holds `{` or `[`, that piece
/// cut at the byte. This is where the `json_object` grammar engages.
fn from_first_container_byte(script: Script) -> Vec<String> {
    let mut out = Vec::new();
    for piece in script.answer {
        if out.is_empty() {
            if let Some(at) = piece.find(['{', '[']) {
                out.push(piece[at..].to_owned());
            }
        } else {
            out.push((*piece).to_owned());
        }
    }
    assert!(!out.is_empty(), "harness: the script holds no `{{` or `[`");
    out
}

fn all_answer_pieces(script: Script) -> Vec<String> {
    script.answer.iter().map(|p| (*p).to_owned()).collect()
}

async fn assert_streamed(script: Script, mode: Mode, deltas: Vec<String>) {
    let got = reply(script, mode, true).await;
    assert_eq!(got.content, deltas, "streamed content deltas");
}

async fn assert_blocking(script: Script, mode: Mode, deltas: &[String]) {
    let got = reply(script, mode, false).await;
    assert_eq!(got.text(), deltas.concat(), "non-streamed content");
}

// ── What must not move: a JSON reply that is correct today ──────────────────

#[tokio::test]
async fn a_bare_object_is_streamed_piece_for_piece() {
    assert_streamed(OBJECT, Mode::JsonObject, all_answer_pieces(OBJECT)).await;
}

#[tokio::test]
async fn a_bare_object_is_returned_whole() {
    assert_blocking(OBJECT, Mode::JsonObject, &all_answer_pieces(OBJECT)).await;
    assert_eq!(all_answer_pieces(OBJECT).concat(), r#"{"a": 1}"#);
}

#[tokio::test]
async fn an_object_that_closes_on_the_last_allowed_token_is_whole_on_both_paths() {
    let whole = all_answer_pieces(OBJECT_AT_LENGTH);
    assert_streamed(OBJECT_AT_LENGTH, Mode::JsonObject, whole.clone()).await;
    assert_blocking(OBJECT_AT_LENGTH, Mode::JsonObject, &whole).await;
}

#[tokio::test]
async fn a_bare_fence_header_is_not_in_the_reply_on_both_paths() {
    for (script, mode) in [
        (BARE_FENCE, Mode::JsonObject),
        (BARE_FENCE_MID_TOKEN, Mode::JsonObject),
        (BARE_FENCE, Mode::SchemaRoot("object")),
    ] {
        let from_brace = from_first_container_byte(script);
        assert_eq!(from_brace.concat(), r#"{"a": 1}"#);
        assert_streamed(script, mode, from_brace.clone()).await;
        assert_blocking(script, mode, &from_brace).await;
    }
}

#[tokio::test]
async fn a_json_fence_header_is_not_in_the_non_streamed_reply() {
    let from_brace = from_first_container_byte(JSON_FENCE);
    assert_blocking(JSON_FENCE, Mode::JsonObject, &from_brace).await;
}

#[tokio::test]
async fn prose_before_the_object_is_not_in_the_non_streamed_reply() {
    let from_brace = from_first_container_byte(PROSE);
    assert_blocking(PROSE, Mode::JsonObject, &from_brace).await;
}

#[tokio::test]
async fn an_object_cut_by_the_token_limit_is_streamed_piece_for_piece() {
    for script in [CUT_OBJECT, CUT_AFTER_INNER_OBJECT, CUT_ARRAY] {
        assert_streamed(script, Mode::JsonObject, all_answer_pieces(script)).await;
    }
}

#[tokio::test]
async fn white_space_after_the_closed_object_is_streamed() {
    let whole = all_answer_pieces(OBJECT_THEN_NEWLINE);
    assert_streamed(OBJECT_THEN_NEWLINE, Mode::JsonObject, whole).await;
}

#[tokio::test]
async fn a_dash_in_the_prose_is_not_returned_in_place_of_the_object() {
    let from_brace = from_first_container_byte(PROSE_WITH_DASH);
    assert_blocking(PROSE_WITH_DASH, Mode::JsonObject, &from_brace).await;
}

#[tokio::test]
async fn reasoning_stays_on_its_channel_and_the_object_is_whole_on_both_paths() {
    let whole = all_answer_pieces(OBJECT_AFTER_REASONING);
    for stream in [true, false] {
        let got = reply(OBJECT_AFTER_REASONING, Mode::JsonObject, stream).await;
        assert_eq!(got.text(), whole.concat(), "stream={stream}");
        assert_eq!(got.reasoning, "Let me think {\"x\"}", "stream={stream}");
    }
}

#[tokio::test]
async fn a_scalar_schema_root_is_returned_unchanged_on_both_paths() {
    for (script, root) in [
        (SCALAR_TRUE, "boolean"),
        (SCALAR_NULL, "null"),
        (SCALAR_INTEGER, "integer"),
        (SCALAR_STRING, "string"),
        (SCALAR_STRING_IN_TWO_PIECES, "string"),
        (SCALAR_STRING_WITH_BRACE, "string"),
        (CUT_SCALAR_STRING, "string"),
    ] {
        let whole = all_answer_pieces(script);
        assert_streamed(script, Mode::SchemaRoot(root), whole.clone()).await;
        assert_blocking(script, Mode::SchemaRoot(root), &whole).await;
    }
}

#[tokio::test]
async fn an_object_schema_root_is_returned_unchanged_on_both_paths() {
    let whole = all_answer_pieces(OBJECT);
    assert_streamed(OBJECT, Mode::SchemaRoot("object"), whole.clone()).await;
    assert_blocking(OBJECT, Mode::SchemaRoot("object"), &whole).await;
}

// ── What must not move: the other fields, and the other modes ───────────────

/// Every JSON-mode script, with the request mode it runs under.
const JSON_MODE_SCRIPTS: &[(Script, Mode)] = &[
    (OBJECT, Mode::JsonObject),
    (OBJECT_AT_LENGTH, Mode::JsonObject),
    (OBJECT_AFTER_REASONING, Mode::JsonObject),
    (OBJECT_THEN_NEWLINE, Mode::JsonObject),
    (PROSE_WITH_QUOTED_WORD, Mode::JsonObject),
    (PROSE_WITH_DASH, Mode::JsonObject),
    (BARE_FENCE, Mode::JsonObject),
    (BARE_FENCE_MID_TOKEN, Mode::JsonObject),
    (JSON_FENCE, Mode::JsonObject),
    (PROSE, Mode::JsonObject),
    (PROSE_WITH_NUMBER, Mode::JsonObject),
    (CUT_OBJECT, Mode::JsonObject),
    (CUT_OBJECT_IN_JSON_FENCE, Mode::JsonObject),
    (CUT_AFTER_INNER_OBJECT, Mode::JsonObject),
    (CUT_ARRAY, Mode::JsonObject),
    (SCALAR_TRUE, Mode::SchemaRoot("boolean")),
    (CUT_SCALAR_STRING, Mode::SchemaRoot("string")),
];

#[tokio::test]
async fn the_finish_reason_and_the_token_count_are_the_generation_s_on_both_paths() {
    for (script, mode) in JSON_MODE_SCRIPTS {
        let tokens = (script.thinking.len() + script.answer.len() + 1) as u64;
        for stream in [true, false] {
            let got = reply(*script, *mode, stream).await;
            let what = format!("stream={stream} answer={:?}", script.answer);
            assert_eq!(got.finish, script.finish, "{what}");
            assert_eq!(got.completion_tokens, tokens, "{what}");
            assert!(got.tool_names.is_empty(), "{what}");
        }
    }
}

#[tokio::test]
async fn the_grammar_reports_a_closed_value_only_for_a_closed_value() {
    for (script, closed) in [(OBJECT, true), (CUT_OBJECT, false), (PROSE, true)] {
        let (status, raw, seen) = serve(script, Mode::JsonObject, false).await;
        assert_eq!(status, 200, "harness: body: {raw}");
        assert_eq!(seen.lock().constraint_finished, Some(closed));
    }
}

#[tokio::test]
async fn a_reply_without_json_mode_is_returned_byte_for_byte_on_both_paths() {
    let whole = all_answer_pieces(PLAIN_TEXT);
    assert_eq!(whole.concat(), "Note: 30 { x} true");
    for mode in [Mode::Text, Mode::NoFormat] {
        assert_streamed(PLAIN_TEXT, mode, whole.clone()).await;
        assert_blocking(PLAIN_TEXT, mode, &whole).await;
        let (_, _, seen) = serve(PLAIN_TEXT, mode, false).await;
        assert_eq!(seen.lock().constraint_finished, None, "no constraint");
    }
}

#[tokio::test]
async fn a_forced_tool_call_becomes_a_tool_call_on_both_paths() {
    for stream in [true, false] {
        let got = reply(TOOL_CALL, Mode::RequiredTool, stream).await;
        assert_eq!(got.tool_names, ["f"], "stream={stream}");
        assert_eq!(got.text(), "", "stream={stream}");
        assert_eq!(got.finish, "tool_calls", "stream={stream}");
    }
}

/// A forced tool call that the token limit cut is not a tool call. Both paths
/// return the text.
#[tokio::test]
async fn a_cut_forced_tool_call_is_returned_as_its_text_on_both_paths() {
    let whole = all_answer_pieces(CUT_TOOL_CALL).concat();
    for stream in [true, false] {
        let got = reply(CUT_TOOL_CALL, Mode::RequiredTool, stream).await;
        assert_eq!(got.text(), whole, "stream={stream}");
        assert!(got.tool_names.is_empty(), "stream={stream}");
        assert_eq!(got.finish, "length", "stream={stream}");
    }
}

#[tokio::test]
async fn a_non_streamed_reply_whose_grammar_never_engaged_is_refused() {
    let (status, raw, _) = serve(NO_JSON, Mode::JsonObject, false).await;
    assert_eq!(status, 502, "body: {raw}");
    assert!(raw.contains("constraint_not_engaged"), "body: {raw}");
}

// ── Wrong today: the streamed reply loses or keeps the wrong text ───────────

#[tokio::test]
#[should_panic(expected = "streamed content deltas")]
async fn a_json_fence_header_is_not_in_the_streamed_reply() {
    let from_brace = from_first_container_byte(JSON_FENCE);
    assert_streamed(JSON_FENCE, Mode::JsonObject, from_brace).await;
}

#[tokio::test]
#[should_panic(expected = "streamed content deltas")]
async fn prose_before_the_object_is_not_cut_at_a_letter_in_the_streamed_reply() {
    let from_brace = from_first_container_byte(PROSE);
    assert_streamed(PROSE, Mode::JsonObject, from_brace).await;
}

#[tokio::test]
#[should_panic(expected = "streamed content deltas")]
async fn prose_with_a_number_is_not_cut_at_the_digit_in_the_streamed_reply() {
    let from_brace = from_first_container_byte(PROSE_WITH_NUMBER);
    assert_streamed(PROSE_WITH_NUMBER, Mode::JsonObject, from_brace).await;
}

#[tokio::test]
#[should_panic(expected = "streamed content deltas")]
async fn prose_with_a_quoted_word_is_not_cut_at_the_quote_in_the_streamed_reply() {
    let from_brace = from_first_container_byte(PROSE_WITH_QUOTED_WORD);
    assert_streamed(PROSE_WITH_QUOTED_WORD, Mode::JsonObject, from_brace).await;
}

#[tokio::test]
#[should_panic(expected = "streamed content deltas")]
async fn prose_that_starts_with_a_dash_is_not_in_the_streamed_reply() {
    let from_brace = from_first_container_byte(PROSE_WITH_DASH);
    assert_streamed(PROSE_WITH_DASH, Mode::JsonObject, from_brace).await;
}

#[tokio::test]
#[should_panic(expected = "streamed content deltas")]
async fn a_cut_object_in_a_json_fence_is_streamed_from_its_brace() {
    let from_brace = from_first_container_byte(CUT_OBJECT_IN_JSON_FENCE);
    assert_streamed(CUT_OBJECT_IN_JSON_FENCE, Mode::JsonObject, from_brace).await;
}

/// The grammar never engaged, so no byte of the reply is JSON-mode output. The
/// stream cannot refuse, and the text the model wrote is all the client has.
#[tokio::test]
#[should_panic(expected = "streamed content of an unengaged reply")]
async fn a_streamed_reply_whose_grammar_never_engaged_keeps_its_text() {
    let got = reply(NO_JSON, Mode::JsonObject, true).await;
    assert_eq!(
        got.text(),
        "I cannot do that",
        "streamed content of an unengaged reply"
    );
}

// ── Wrong today: the non-streamed reply is one smaller value ────────────────

#[tokio::test]
#[should_panic(expected = "non-streamed content")]
async fn an_object_cut_by_the_token_limit_is_not_returned_as_its_first_key() {
    assert_blocking(CUT_OBJECT, Mode::JsonObject, &all_answer_pieces(CUT_OBJECT)).await;
}

#[tokio::test]
#[should_panic(expected = "non-streamed content")]
async fn a_cut_object_in_a_json_fence_is_not_returned_as_its_first_key() {
    let from_brace = from_first_container_byte(CUT_OBJECT_IN_JSON_FENCE);
    assert_blocking(CUT_OBJECT_IN_JSON_FENCE, Mode::JsonObject, &from_brace).await;
}

#[tokio::test]
#[should_panic(expected = "non-streamed content")]
async fn a_cut_object_is_not_returned_as_a_closed_inner_object_or_key() {
    assert_blocking(
        CUT_AFTER_INNER_OBJECT,
        Mode::JsonObject,
        &all_answer_pieces(CUT_AFTER_INNER_OBJECT),
    )
    .await;
}

#[tokio::test]
#[should_panic(expected = "non-streamed content")]
async fn a_cut_array_is_not_returned_as_its_first_number() {
    assert_blocking(CUT_ARRAY, Mode::JsonObject, &all_answer_pieces(CUT_ARRAY)).await;
}

#[tokio::test]
#[should_panic(expected = "non-streamed content")]
async fn a_number_in_the_prose_is_not_returned_in_place_of_the_object() {
    let from_brace = from_first_container_byte(PROSE_WITH_NUMBER);
    assert_blocking(PROSE_WITH_NUMBER, Mode::JsonObject, &from_brace).await;
}

#[tokio::test]
#[should_panic(expected = "non-streamed content")]
async fn a_quoted_word_in_the_prose_is_not_returned_in_place_of_the_object() {
    let from_brace = from_first_container_byte(PROSE_WITH_QUOTED_WORD);
    assert_blocking(PROSE_WITH_QUOTED_WORD, Mode::JsonObject, &from_brace).await;
}

/// The streamed reply holds this white space today, so the two paths differ.
#[tokio::test]
#[should_panic(expected = "non-streamed content")]
async fn white_space_after_the_closed_object_is_in_the_non_streamed_reply() {
    let whole = all_answer_pieces(OBJECT_THEN_NEWLINE);
    assert_blocking(OBJECT_THEN_NEWLINE, Mode::JsonObject, &whole).await;
}
