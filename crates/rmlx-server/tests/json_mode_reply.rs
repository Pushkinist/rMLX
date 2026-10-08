//! HTTP-surface tests for the reply of a `response_format` request.
//!
//! The observable is what a client reads: the `content` bytes of each SSE
//! delta of a streamed reply, its error events and its `finish_reason` chunks,
//! and `message.content` of a non-streamed reply. A scripted generator replays
//! one fixed generation into both paths. It drives the constraint the route
//! built in the order of the decode loop: `step_mask`, `advance`, and only
//! then the `is_thinking` flag of that token. So the grammar engages where it
//! engages on a model, and a script that the grammar refuses is a harness
//! failure.
//!
//! The rule the tests hold: the reply is the generated answer text from the
//! byte at which the grammar engaged to the end of the generation, with no
//! byte dropped and no byte added after that point, on both paths. A reply
//! whose text ends before that byte is not engaged: the non-streamed path
//! answers 502 and the stream ends with an error event.
//!
//! The constraint is the one producer of the engagement byte. This file holds
//! no rule for it: each expected value is a literal, and the harness checks
//! the literal against the piece at which the constraint reported `engaged()`.

// LOC-exempt: one scripted harness and the reply contract it holds on both
// paths; the scripts and the tests that read them stay in one file.

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

use futures::stream::{self, Stream, StreamExt};
use parking_lot::Mutex;
use rmlx_server::{
    ApiErrorCounters, AppState, GenerationRequest, GenerationToken, Generator, ItlStore,
    LoadedModel, ModelLoader, ModelRegistry, SessionCache, TtftStore,
};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;
use tokio::sync::Notify;

// ── Vocabulary ──────────────────────────────────────────────────────────────

/// The token id of a piece is its index. Id 0 is the EOS id of the fixture's
/// `config.json`.
const VOCAB: &[&str] = &[
    "<eos>",
    "[UNK]",
    "hi",
    "</think>",
    "{",
    "}",
    "[",
    "]",
    ":",
    ",",
    "\"a\"",
    "\"b\"",
    " 1",
    " {",
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
    "}\n",
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
    "dium\"",
    "\"a{b\"",
    "Let",
    " me",
    " think",
    " {\"x\"}",
    "Note",
    " x",
    " true",
    "-",
    "\"name\"",
    "\"f\"",
    "\"arguments\"",
    "{\"a\"",
    "<tool_call>",
    "</tool_call>",
    "\n```",
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

/// One generation: reasoning pieces, answer pieces, the terminal reason. A
/// script with reasoning and an answer has one more token between the two:
/// the token that closes the reasoning block, which carries no visible text.
#[derive(Clone, Copy)]
struct Script {
    thinking: &'static [&'static str],
    answer: &'static [&'static str],
    finish: &'static str,
    /// The first answer token is replayed from a prompt cache: the constraint
    /// sees it, and no mask was applied to it.
    replayed_first: bool,
}

impl Script {
    fn closes_its_reasoning(self) -> bool {
        !self.thinking.is_empty() && !self.answer.is_empty()
    }

    fn tokens(self) -> u64 {
        let close = usize::from(self.closes_its_reasoning());
        (self.thinking.len() + close + self.answer.len() + 1) as u64
    }
}

const fn stop(answer: &'static [&'static str]) -> Script {
    Script {
        thinking: &[],
        answer,
        finish: "stop",
        replayed_first: false,
    }
}

const fn length(answer: &'static [&'static str]) -> Script {
    Script {
        thinking: &[],
        answer,
        finish: "length",
        replayed_first: false,
    }
}

const OBJECT_TEXT: &str = r#"{"a": 1}"#;
const OBJECT_PIECES: &[&str] = &["{", "\"a\"", ":", " 1", "}"];

const OBJECT: Script = stop(OBJECT_PIECES);
const OBJECT_AT_LENGTH: Script = length(OBJECT_PIECES);
const OBJECT_AFTER_REASONING: Script = Script {
    thinking: &["Let", " me", " think", " {\"x\"}"],
    answer: OBJECT_PIECES,
    finish: "stop",
    replayed_first: false,
};
const OBJECT_THEN_NEWLINE: Script = stop(&["{", "\"a\"", ":", " 1", "}\n"]);
const TWO_KEYS: Script = stop(&["{", "\"a\"", ":", " 1", ",", "\"b\"", ":", " 1", "}"]);
const FENCE_THEN_BRACE_AND_KEY: Script = stop(&["```", "\n", "{\"a\"", ":", " 1", "}"]);
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
    stop(&["Here", " \"c\"", ":", "\n\n", "{", "\"a\"", ":", " 1", "}"]);
const PROSE_WITH_DASH: Script = stop(&["-", " JSON", ":", "\n", "{", "\"a\"", ":", " 1", "}"]);
const BRACE_PROSE_THEN_ARRAY: Script = stop(&[
    "Here", " {", " x", "}", ":", "\n", "[", "10", ",", " 20", "]",
]);
const BRACKET_PROSE_THEN_OBJECT: Script =
    stop(&["[", "Here", "]", ":", "\n", "{", "\"a\"", ":", " 1", "}"]);
const CUT_OBJECT: Script = length(&[
    "{",
    "\"Afghanistan\"",
    ":",
    " \"Kabul\"",
    ",",
    " \"Albania\"",
    ":",
]);
const CUT_OBJECT_TEXT: &str = r#"{"Afghanistan": "Kabul", "Albania":"#;
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
const CUT_SCHEMA_OBJECT: Script = length(&["{", "\"a\"", ":"]);
const CUT_AT_THE_BRACE: Script = length(&["```", "\n{"]);
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

const MARKED_TOOL_CALL: Script = stop(&[
    "<tool_call>",
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
    "</tool_call>",
]);
const CUT_TOOL_CALL_AFTER_A_REPLAYED_WORD: Script = Script {
    thinking: &[],
    answer: &[
        "Here",
        "{",
        "\"name\"",
        ":",
        "\"f\"",
        ",",
        "\"arguments\"",
        ":",
    ],
    finish: "length",
    replayed_first: true,
};

// ── Scripted generator ──────────────────────────────────────────────────────

/// What the generator saw. `violations` is empty for a sound script.
#[derive(Default)]
struct Seen {
    violations: Vec<String>,
    /// `Some` when the route built a constraint: the engine's `finished()`
    /// after the last token.
    constraint_finished: Option<bool>,
    /// The index of the answer piece after which the constraint first
    /// reported `engaged()`.
    engaged_at: Option<usize>,
}

#[derive(Clone)]
struct ScriptedGenerator {
    script: Script,
    /// When set, the generation waits here after the token at which the
    /// constraint engaged.
    gate: Option<Arc<Notify>>,
    seen: Arc<Mutex<Seen>>,
}

fn token(id: u32, piece: &str, is_thinking: bool) -> GenerationToken {
    GenerationToken {
        token_id: id,
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
        let mut gate_after = None;

        // (token id, visible piece, is_thinking, answer index)
        let mut steps: Vec<(u32, &str, bool, Option<usize>)> = Vec::new();
        for piece in self.script.thinking {
            steps.push((token_id(piece), piece, true, None));
        }
        if self.script.closes_its_reasoning() {
            steps.push((token_id("</think>"), "", false, None));
        }
        for (at, piece) in self.script.answer.iter().enumerate() {
            steps.push((token_id(piece), piece, false, Some(at)));
        }

        for (id, piece, is_thinking, answer_at) in steps {
            let masked = !(self.script.replayed_first && answer_at == Some(0));
            if let Some(c) = constraint.as_mut() {
                if masked && !c.step_mask(VOCAB.len())[id as usize] {
                    seen.violations.push(format!(
                        "the grammar refuses token {:?}",
                        VOCAB[id as usize]
                    ));
                }
                c.advance(id);
                if seen.engaged_at.is_none() && c.engaged() {
                    match answer_at {
                        Some(at) => {
                            seen.engaged_at = Some(at);
                            gate_after = Some(toks.len());
                        }
                        // With no thinking flag from the route (a forced
                        // tool call) the grammar cannot tell the channels.
                        None if req.is_thinking_handle.is_none() => {}
                        None => seen
                            .violations
                            .push("the grammar engaged on a reasoning token".to_owned()),
                    }
                }
            }
            if let Some(flag) = req.is_thinking_handle.as_ref() {
                flag.store(is_thinking, Ordering::Relaxed);
            }
            toks.push(Ok(token(id, piece, is_thinking)));
        }
        if let Some(c) = constraint.as_mut() {
            if self.script.finish == "stop" && !c.step_mask(VOCAB.len())[0] {
                seen.violations
                    .push("the grammar refuses EOS at the end of the script".to_owned());
            }
            seen.constraint_finished = Some(c.finished());
        }
        toks.push(Ok(GenerationToken {
            token_id: 0,
            piece: String::new(),
            done: true,
            finish_reason: Some(self.script.finish.to_owned()),
            is_thinking: false,
            logprobs: None,
        }));

        let Some(gate) = self.gate.clone() else {
            return Box::pin(stream::iter(toks));
        };
        let Some(gate_after) = gate_after else {
            seen.violations
                .push("a gated script did not engage".to_owned());
            return Box::pin(stream::iter(toks));
        };
        let tail = toks.split_off(gate_after + 1);
        let open = stream::once(async move {
            gate.notified().await;
            stream::iter(tail)
        })
        .flatten();
        Box::pin(stream::iter(toks).chain(open))
    }
}

// ── AppState + server wiring ────────────────────────────────────────────────

fn scripted_state(
    script: Script,
    gate: Option<Arc<Notify>>,
    tool_call_parser: bool,
) -> (AppState, Arc<Mutex<Seen>>, tempfile::TempDir) {
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
        // A comment that names the Hermes markers gives the route a tool-call
        // parser for this model.
        if tool_call_parser {
            "{% for m in messages %}{{ m['content'] }}{% endfor %}{# <tool_call>{\"name\" #}"
        } else {
            "{% for m in messages %}{{ m['content'] }}{% endfor %}"
        },
    )
    .unwrap();
    std::fs::write(snap.join("tokenizer.json"), tokenizer_json()).unwrap();

    let seen = Arc::new(Mutex::new(Seen::default()));
    let generator = ScriptedGenerator {
        script,
        gate,
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

/// Send one request. `on_first_content` runs when the first content delta of
/// the reply is in the bytes read so far; with `None` nothing waits.
async fn http(port: u16, body: &str, on_first_content: Option<&Notify>) -> (u16, String) {
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
    if let Some(gate) = on_first_content {
        let mut buf = [0u8; 4096];
        while !String::from_utf8_lossy(&response).contains("\"delta\":{\"content\":") {
            let read = tokio::time::timeout(Duration::from_secs(5), stream.read(&mut buf))
                .await
                .expect("no content delta arrived while the generation was open");
            let n = read.unwrap();
            assert!(n > 0, "the reply ended with no content delta");
            response.extend_from_slice(&buf[..n]);
        }
        gate.notify_one();
    }
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

const SCHEMA_OBJECT: &str = r#"{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false}"#;
const SCHEMA_ARRAY: &str = r#"{"type":"array","items":{"type":"integer"}}"#;
const SCHEMA_OBJECT_OR_NULL: &str = r#"{"anyOf":[{"type":"object","properties":{"a":{"type":"integer"}},"required":["a"],"additionalProperties":false},{"type":"null"}]}"#;
const SCHEMA_TYPE_LIST: &str = r#"{"type":["object","null"]}"#;
const SCHEMA_ANY: &str = "{}";
const SCHEMA_BOOLEAN: &str = r#"{"type":"boolean"}"#;
const SCHEMA_NULL: &str = r#"{"type":"null"}"#;
const SCHEMA_INTEGER: &str = r#"{"type":"integer"}"#;
const SCHEMA_STRING: &str = r#"{"type":"string"}"#;

/// The request modes under test.
#[derive(Clone, Copy)]
enum Mode {
    JsonObject,
    /// `json_schema` with this schema text, not strict.
    Schema(&'static str),
    Text,
    NoFormat,
    RequiredTool,
    /// A forced tool call whose schema the route cannot compile: the request
    /// runs with no constraint.
    RequiredToolUnconstrained,
    /// The same, on a model whose template names no tool-call markers.
    RequiredToolUnconstrainedNoParser,
}

impl Mode {
    fn builds_a_constraint(self) -> bool {
        matches!(
            self,
            Mode::JsonObject | Mode::Schema(_) | Mode::RequiredTool
        )
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Reply {
    /// The content bytes of each delta, in order. A non-streamed reply has
    /// one entry.
    content: Vec<String>,
    reasoning: String,
    /// The `finish_reason` values, one for each chunk that carries one.
    finish: Vec<String>,
    /// The `type` of each error event of a streamed reply.
    errors: Vec<String>,
    completion_tokens: u64,
    tool_names: Vec<String>,
}

impl Reply {
    fn text(&self) -> String {
        self.content.concat()
    }
}

fn request_body(mode: Mode, stop: Option<&str>, stream: bool) -> String {
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
    if let Some(stop) = stop {
        body["stop"] = json!([stop]);
    }
    match mode {
        Mode::JsonObject => body["response_format"] = json!({"type": "json_object"}),
        Mode::Schema(schema) => {
            let schema: Value = serde_json::from_str(schema).unwrap();
            body["response_format"] = json!({
                "type": "json_schema",
                "json_schema": {"name": "root", "strict": false, "schema": schema}
            });
        }
        Mode::Text => body["response_format"] = json!({"type": "text"}),
        Mode::NoFormat => {}
        Mode::RequiredToolUnconstrained | Mode::RequiredToolUnconstrainedNoParser => {
            body["tools"] = json!([{
                "type": "function",
                "function": {
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "$defs": {"P": {"type": "object", "properties": {}}},
                        "properties": {"p": {"$ref": "#/$defs/P"}}
                    }
                }
            }]);
            body["tool_choice"] = json!("required");
        }
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
        finish: Vec::new(),
        errors: Vec::new(),
        completion_tokens: 0,
        tool_names: Vec::new(),
    };
    let mut done = 0;
    for line in raw.lines() {
        let Some(data) = line.trim().strip_prefix("data:") else {
            continue;
        };
        let data = data.trim();
        if data == "[DONE]" {
            done += 1;
            continue;
        }
        let v: Value = serde_json::from_str(data)
            .unwrap_or_else(|e| panic!("harness: a data line is not JSON: {e}: {data}"));
        if let Some(error) = v.get("error") {
            let kind = error["type"]
                .as_str()
                .unwrap_or_else(|| panic!("harness: an error event has no type: {data}"));
            reply.errors.push(kind.to_owned());
            continue;
        }
        let choice = &v["choices"][0];
        if let Some(c) = choice["delta"]["content"].as_str() {
            reply.content.push(c.to_owned());
        }
        if let Some(r) = choice["delta"]["reasoning_content"].as_str() {
            reply.reasoning.push_str(r);
        }
        if let Some(f) = choice["finish_reason"].as_str() {
            reply.finish.push(f.to_owned());
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
    assert_eq!(done, 1, "harness: a stream ends with one [DONE]: {raw}");
    reply
}

fn parse_blocking(raw: &str) -> Reply {
    let v: Value = serde_json::from_str(raw).unwrap_or_else(|e| panic!("harness: {e}: {raw}"));
    let choice = &v["choices"][0];
    let message = &choice["message"];
    let field = |v: &Value, name: &str| -> String {
        v[name]
            .as_str()
            .unwrap_or_else(|| panic!("harness: the reply has no `{name}`: {raw}"))
            .to_owned()
    };
    Reply {
        content: vec![field(message, "content")],
        reasoning: message["reasoning_content"]
            .as_str()
            .unwrap_or_default()
            .to_owned(),
        finish: vec![field(choice, "finish_reason")],
        errors: Vec::new(),
        completion_tokens: v["usage"]["completion_tokens"]
            .as_u64()
            .unwrap_or_else(|| panic!("harness: the reply has no token count: {raw}")),
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

/// One request against one scripted generation.
#[derive(Clone, Copy)]
struct Ask {
    script: Script,
    mode: Mode,
    stop: Option<&'static str>,
    stream: bool,
    /// Hold the generation after the engagement token until the client has
    /// the first content delta.
    gated: bool,
}

const fn ask(script: Script, mode: Mode, stream: bool) -> Ask {
    Ask {
        script,
        mode,
        stop: None,
        stream,
        gated: false,
    }
}

/// Run one request and return `(status, raw body, what the generator saw)`.
async fn serve(ask: Ask) -> (u16, String, Arc<Mutex<Seen>>) {
    let gate = ask.gated.then(|| Arc::new(Notify::new()));
    let parser = !matches!(ask.mode, Mode::RequiredToolUnconstrainedNoParser);
    let (state, seen, _tmp) = scripted_state(ask.script, gate.clone(), parser);
    let router = rmlx_server::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    let body = request_body(ask.mode, ask.stop, ask.stream);
    let (status, raw) = http(port, &body, gate.as_deref()).await;
    {
        let seen = seen.lock();
        assert!(
            seen.violations.is_empty(),
            "harness: unsound script: {:?}",
            seen.violations
        );
        assert_eq!(
            seen.constraint_finished.is_some(),
            ask.mode.builds_a_constraint(),
            "harness: the route built a constraint for this mode"
        );
    }
    (status, raw, seen)
}

/// The reply of a request that answers 200.
async fn reply(ask: Ask) -> (Reply, Arc<Mutex<Seen>>) {
    let (status, raw, seen) = serve(ask).await;
    assert_eq!(status, 200, "harness: body: {raw}");
    let reply = if ask.stream {
        parse_streamed(&raw)
    } else {
        parse_blocking(&raw)
    };
    (reply, seen)
}

/// A literal expected reply must start in the answer piece at which the
/// constraint reported that it engaged.
fn check_literal_against_the_constraint(script: Script, seen: &Seen, literal: &str) {
    let at = seen
        .engaged_at
        .expect("harness: the constraint did not engage");
    let rest = script.answer[at + 1..].concat();
    let starts_in_the_piece = literal
        .strip_suffix(rest.as_str())
        .is_some_and(|head| !head.is_empty() && script.answer[at].ends_with(head));
    assert!(
        starts_in_the_piece,
        "harness: the literal {literal:?} does not start where the constraint engaged \
         (answer piece {at} of {:?})",
        script.answer
    );
}

async fn assert_streamed(script: Script, mode: Mode, deltas: &[&str]) {
    let (got, seen) = reply(ask(script, mode, true)).await;
    if mode.builds_a_constraint() {
        check_literal_against_the_constraint(script, &seen.lock(), &deltas.concat());
    }
    assert_eq!(got.content, deltas, "streamed content deltas");
    assert!(got.errors.is_empty(), "harness: {:?}", got.errors);
}

async fn assert_blocking(script: Script, mode: Mode, text: &str) {
    let (got, seen) = reply(ask(script, mode, false)).await;
    if mode.builds_a_constraint() {
        check_literal_against_the_constraint(script, &seen.lock(), text);
    }
    assert_eq!(got.text(), text, "non-streamed content");
}

/// The stream of a reply that is not engaged: no content, one error event of
/// the type of the non-streamed refusal, and no `finish_reason` chunk.
async fn assert_stream_refuses(ask: Ask) {
    let (got, _) = reply(ask).await;
    assert_eq!(
        (got.content, got.errors, got.finish),
        (
            Vec::<String>::new(),
            vec!["constraint_not_engaged".to_owned()],
            Vec::<String>::new()
        ),
        "streamed reply that is not engaged"
    );
}

async fn assert_blocking_refuses(ask: Ask) {
    let (status, raw, _) = serve(ask).await;
    assert_eq!(status, 502, "non-streamed status: body: {raw}");
    assert!(raw.contains("constraint_not_engaged"), "body: {raw}");
}

// ── What must not move: a JSON reply that is correct today ──────────────────

#[tokio::test]
async fn a_bare_object_is_streamed_piece_for_piece() {
    for mode in [Mode::JsonObject, Mode::Schema(SCHEMA_OBJECT)] {
        assert_streamed(OBJECT, mode, OBJECT_PIECES).await;
    }
}

#[tokio::test]
async fn a_bare_object_is_returned_whole() {
    for mode in [Mode::JsonObject, Mode::Schema(SCHEMA_OBJECT)] {
        assert_blocking(OBJECT, mode, OBJECT_TEXT).await;
    }
}

#[tokio::test]
async fn an_object_that_closes_on_the_last_allowed_token_is_whole_on_both_paths() {
    assert_streamed(OBJECT_AT_LENGTH, Mode::JsonObject, OBJECT_PIECES).await;
    assert_blocking(OBJECT_AT_LENGTH, Mode::JsonObject, OBJECT_TEXT).await;
}

#[tokio::test]
async fn a_bare_fence_header_is_not_in_the_reply_on_both_paths() {
    for (script, mode) in [
        (BARE_FENCE, Mode::JsonObject),
        (BARE_FENCE_MID_TOKEN, Mode::JsonObject),
        (BARE_FENCE, Mode::Schema(SCHEMA_OBJECT)),
        (BARE_FENCE, Mode::Schema(SCHEMA_OBJECT_OR_NULL)),
        (BARE_FENCE, Mode::Schema(SCHEMA_TYPE_LIST)),
        (BARE_FENCE, Mode::Schema(SCHEMA_ANY)),
    ] {
        assert_streamed(script, mode, OBJECT_PIECES).await;
        assert_blocking(script, mode, OBJECT_TEXT).await;
    }
}

/// The engagement token holds text after its opener. The reply starts at the
/// opener, not at a fixed distance from the end of the token.
#[tokio::test]
async fn text_after_the_opener_in_its_token_is_in_the_reply_on_both_paths() {
    for mode in [Mode::JsonObject, Mode::Schema(SCHEMA_OBJECT)] {
        let script = FENCE_THEN_BRACE_AND_KEY;
        assert_streamed(script, mode, &["{\"a\"", ":", " 1", "}"]).await;
        assert_blocking(script, mode, OBJECT_TEXT).await;
    }
}

#[tokio::test]
async fn a_json_fence_header_is_not_in_the_non_streamed_reply() {
    for mode in [Mode::JsonObject, Mode::Schema(SCHEMA_OBJECT)] {
        assert_blocking(JSON_FENCE, mode, OBJECT_TEXT).await;
    }
}

#[tokio::test]
async fn prose_before_the_object_is_not_in_the_non_streamed_reply() {
    for mode in [
        Mode::JsonObject,
        Mode::Schema(SCHEMA_OBJECT),
        Mode::Schema(SCHEMA_OBJECT_OR_NULL),
    ] {
        assert_blocking(PROSE, mode, OBJECT_TEXT).await;
    }
}

#[tokio::test]
async fn a_dash_in_the_prose_is_not_returned_in_place_of_the_object() {
    assert_blocking(PROSE_WITH_DASH, Mode::JsonObject, OBJECT_TEXT).await;
}

#[tokio::test]
async fn a_value_cut_by_the_token_limit_is_streamed_piece_for_piece() {
    for (script, mode) in [
        (CUT_OBJECT, Mode::JsonObject),
        (CUT_AFTER_INNER_OBJECT, Mode::JsonObject),
        (CUT_ARRAY, Mode::JsonObject),
        (CUT_SCHEMA_OBJECT, Mode::Schema(SCHEMA_OBJECT)),
    ] {
        assert_streamed(script, mode, script.answer).await;
    }
}

#[tokio::test]
async fn white_space_after_the_closed_object_is_streamed() {
    assert_streamed(
        OBJECT_THEN_NEWLINE,
        Mode::JsonObject,
        &["{", "\"a\"", ":", " 1", "}\n"],
    )
    .await;
}

#[tokio::test]
async fn reasoning_stays_on_its_channel_and_the_object_is_whole_on_both_paths() {
    for stream in [true, false] {
        let (got, _) = reply(ask(OBJECT_AFTER_REASONING, Mode::JsonObject, stream)).await;
        assert_eq!(got.text(), OBJECT_TEXT, "stream={stream}");
        assert_eq!(got.reasoning, "Let me think {\"x\"}", "stream={stream}");
    }
}

#[tokio::test]
async fn a_scalar_schema_root_is_returned_unchanged_on_both_paths() {
    for (script, schema) in [
        (SCALAR_TRUE, SCHEMA_BOOLEAN),
        (SCALAR_NULL, SCHEMA_NULL),
        (SCALAR_INTEGER, SCHEMA_INTEGER),
        (SCALAR_STRING, SCHEMA_STRING),
        (SCALAR_STRING_IN_TWO_PIECES, SCHEMA_STRING),
        (SCALAR_STRING_WITH_BRACE, SCHEMA_STRING),
        (CUT_SCALAR_STRING, SCHEMA_STRING),
    ] {
        assert_streamed(script, Mode::Schema(schema), script.answer).await;
        assert_blocking(script, Mode::Schema(schema), &script.answer.concat()).await;
    }
}

/// The first content delta reaches the client while the generation is open: a
/// reply is not held until its end.
#[tokio::test]
async fn the_first_delta_of_an_engaged_reply_is_sent_before_the_generation_ends() {
    let gated = Ask {
        gated: true,
        ..ask(BARE_FENCE, Mode::JsonObject, true)
    };
    let (got, _) = reply(gated).await;
    assert_eq!(got.content, OBJECT_PIECES);
}

// ── What must not move: the other fields, and the other modes ───────────────

/// Every JSON-mode script that answers 200, with the request mode it runs
/// under.
const JSON_MODE_SCRIPTS: &[(Script, Mode)] = &[
    (OBJECT, Mode::JsonObject),
    (OBJECT, Mode::Schema(SCHEMA_OBJECT)),
    (OBJECT_AT_LENGTH, Mode::JsonObject),
    (OBJECT_AFTER_REASONING, Mode::JsonObject),
    (OBJECT_THEN_NEWLINE, Mode::JsonObject),
    (TWO_KEYS, Mode::JsonObject),
    (PROSE_WITH_QUOTED_WORD, Mode::JsonObject),
    (PROSE_WITH_DASH, Mode::JsonObject),
    (BARE_FENCE, Mode::JsonObject),
    (BARE_FENCE_MID_TOKEN, Mode::JsonObject),
    (JSON_FENCE, Mode::JsonObject),
    (JSON_FENCE, Mode::Schema(SCHEMA_OBJECT)),
    (PROSE, Mode::JsonObject),
    (PROSE_WITH_NUMBER, Mode::JsonObject),
    (BRACE_PROSE_THEN_ARRAY, Mode::Schema(SCHEMA_ARRAY)),
    (BRACKET_PROSE_THEN_OBJECT, Mode::Schema(SCHEMA_OBJECT)),
    (CUT_OBJECT, Mode::JsonObject),
    (CUT_OBJECT_IN_JSON_FENCE, Mode::JsonObject),
    (CUT_AFTER_INNER_OBJECT, Mode::JsonObject),
    (CUT_ARRAY, Mode::JsonObject),
    (CUT_SCHEMA_OBJECT, Mode::Schema(SCHEMA_OBJECT)),
    (SCALAR_TRUE, Mode::Schema(SCHEMA_BOOLEAN)),
    (CUT_SCALAR_STRING, Mode::Schema(SCHEMA_STRING)),
];

#[tokio::test]
async fn the_finish_reason_and_the_token_count_are_the_generation_s_on_both_paths() {
    for (script, mode) in JSON_MODE_SCRIPTS {
        for stream in [true, false] {
            let (got, _) = reply(ask(*script, *mode, stream)).await;
            let what = format!("stream={stream} answer={:?}", script.answer);
            assert_eq!(got.finish, [script.finish], "{what}");
            assert_eq!(got.completion_tokens, script.tokens(), "{what}");
            assert!(got.tool_names.is_empty(), "{what}");
            assert!(got.errors.is_empty(), "{what}");
        }
    }
}

#[tokio::test]
async fn the_grammar_reports_a_closed_value_only_for_a_closed_value() {
    for (script, closed) in [(OBJECT, true), (CUT_OBJECT, false), (PROSE, true)] {
        let (status, raw, seen) = serve(ask(script, Mode::JsonObject, false)).await;
        assert_eq!(status, 200, "harness: body: {raw}");
        assert_eq!(seen.lock().constraint_finished, Some(closed));
    }
}

#[tokio::test]
async fn a_reply_without_json_mode_is_returned_byte_for_byte_on_both_paths() {
    assert_eq!(PLAIN_TEXT.answer.concat(), "Note: 30 { x} true");
    for mode in [Mode::Text, Mode::NoFormat] {
        assert_streamed(PLAIN_TEXT, mode, PLAIN_TEXT.answer).await;
        assert_blocking(PLAIN_TEXT, mode, "Note: 30 { x} true").await;
    }
}

#[tokio::test]
async fn a_forced_tool_call_becomes_a_tool_call_on_both_paths() {
    for stream in [true, false] {
        let (got, _) = reply(ask(TOOL_CALL, Mode::RequiredTool, stream)).await;
        assert_eq!(got.tool_names, ["f"], "stream={stream}");
        assert_eq!(got.text(), "", "stream={stream}");
        assert_eq!(got.finish, ["tool_calls"], "stream={stream}");
    }
}

/// A forced tool call that the token limit cut is not a tool call. Both paths
/// return the text.
#[tokio::test]
async fn a_cut_forced_tool_call_is_returned_as_its_text_on_both_paths() {
    for stream in [true, false] {
        let (got, _) = reply(ask(CUT_TOOL_CALL, Mode::RequiredTool, stream)).await;
        assert_eq!(got.text(), r#"{"name":"f","arguments":"#, "stream={stream}");
        assert!(got.tool_names.is_empty(), "stream={stream}");
        assert_eq!(got.finish, ["length"], "stream={stream}");
    }
}

/// A model whose prompt left the reasoning block open writes the constrained
/// call on the reasoning channel. It is the tool call on both paths.
#[tokio::test]
async fn a_constrained_forced_tool_call_on_the_reasoning_channel_is_a_tool_call() {
    let script = unconstrained(BARE, &[]);
    for stream in [true, false] {
        let (got, _) = reply(ask(script, Mode::RequiredTool, stream)).await;
        assert_eq!(got.tool_names, ["f"], "stream={stream}");
        assert_eq!(got.text(), "", "stream={stream}");
        assert_eq!(got.reasoning, "", "stream={stream}");
        assert_eq!(got.finish, ["tool_calls"], "stream={stream}");
    }
}

/// The cut call starts at the engagement byte on both paths: a replayed word
/// before it is not in the text.
#[tokio::test]
async fn a_cut_forced_tool_call_after_a_replayed_word_is_the_same_text_on_both_paths() {
    for stream in [true, false] {
        let script = CUT_TOOL_CALL_AFTER_A_REPLAYED_WORD;
        let (got, _) = reply(ask(script, Mode::RequiredTool, stream)).await;
        assert_eq!(got.text(), r#"{"name":"f","arguments":"#, "stream={stream}");
        assert!(got.tool_names.is_empty(), "stream={stream}");
        assert_eq!(got.finish, ["length"], "stream={stream}");
    }
}

const MARKED: &[&str] = MARKED_TOOL_CALL.answer;
const BARE: &[&str] = TOOL_CALL.answer;
const REASONING: &[&str] = &["Let", " me", " think"];

const fn unconstrained(
    thinking: &'static [&'static str],
    answer: &'static [&'static str],
) -> Script {
    Script {
        thinking,
        answer,
        finish: "stop",
        replayed_first: false,
    }
}

/// One reply shape of a forced tool call that runs with no constraint, and
/// what a client reads: tool names, content, reasoning, finish reason.
type ForcedCallShape = (
    &'static str,
    Script,
    &'static [&'static str],
    &'static str,
    &'static str,
    &'static str,
);

const BARE_TEXT: &str = r#"{"name":"f","arguments":{}}"#;

const UNCONSTRAINED_FORCED_CALLS: &[ForcedCallShape] = &[
    (
        "marked",
        unconstrained(&[], MARKED),
        &["f"],
        "",
        "",
        "tool_calls",
    ),
    (
        "bare",
        unconstrained(&[], BARE),
        &["f"],
        "",
        "",
        "tool_calls",
    ),
    (
        "marked_in_thinking",
        unconstrained(MARKED, &[]),
        &["f"],
        "",
        "",
        "tool_calls",
    ),
    (
        "reasoning_then_bare",
        unconstrained(REASONING, BARE),
        &["f"],
        "",
        "Let me think",
        "tool_calls",
    ),
    (
        "reasoning_then_marked",
        unconstrained(REASONING, MARKED),
        &["f"],
        "",
        "Let me think",
        "tool_calls",
    ),
    (
        "prose_then_marked",
        unconstrained(
            &[],
            &[
                "Here",
                "<tool_call>",
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
                "</tool_call>",
            ],
        ),
        &["f"],
        "Here",
        "",
        "tool_calls",
    ),
    (
        "prose_then_bare",
        unconstrained(
            &[],
            &[
                "Here",
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
            ],
        ),
        &[],
        "Here{\"name\":\"f\",\"arguments\":{}}",
        "",
        "stop",
    ),
    (
        "fenced_bare",
        unconstrained(
            &[],
            &[
                "```",
                "json",
                "\n",
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
                "\n```",
            ],
        ),
        &[],
        "```json\n{\"name\":\"f\",\"arguments\":{}}\n```",
        "",
        "stop",
    ),
];

/// With no constraint the reply is read as for `tool_choice: auto`: a marked
/// call is a tool call in either channel, and text outside it is content or
/// reasoning. When no call is marked, the whole content is tried as bare JSON.
/// Bare JSON after prose or inside a code fence is text. The two paths agree
/// in each field.
#[tokio::test]
async fn a_forced_tool_call_with_no_constraint_is_the_same_reply_on_both_paths() {
    assert_eq!(BARE.concat(), BARE_TEXT);
    for (shape, script, tools, content, reasoning, finish) in UNCONSTRAINED_FORCED_CALLS {
        for stream in [true, false] {
            let (got, _) = reply(ask(*script, Mode::RequiredToolUnconstrained, stream)).await;
            assert_eq!(
                (
                    got.tool_names.as_slice(),
                    got.text().as_str(),
                    got.reasoning.as_str(),
                    got.finish.as_slice()
                ),
                (
                    tools
                        .iter()
                        .map(|t| (*t).to_owned())
                        .collect::<Vec<_>>()
                        .as_slice(),
                    *content,
                    *reasoning,
                    [(*finish).to_owned()].as_slice()
                ),
                "{shape} stream={stream}"
            );
        }
    }
}

/// A model with no tool-call parser: the reasoning stays reasoning and the
/// bare call is the tool call, on both paths.
#[tokio::test]
async fn a_forced_tool_call_with_no_constraint_and_no_parser_keeps_its_reasoning() {
    let script = unconstrained(REASONING, BARE);
    for stream in [true, false] {
        let mode = Mode::RequiredToolUnconstrainedNoParser;
        let (got, _) = reply(ask(script, mode, stream)).await;
        assert_eq!(got.tool_names, ["f"], "stream={stream}");
        assert_eq!(got.text(), "", "stream={stream}");
        assert_eq!(got.reasoning, "Let me think", "stream={stream}");
        assert_eq!(got.finish, ["tool_calls"], "stream={stream}");
    }
}

#[tokio::test]
async fn a_non_streamed_reply_whose_grammar_never_engaged_is_refused() {
    assert_blocking_refuses(ask(NO_JSON, Mode::JsonObject, false)).await;
}

/// A stop string inside the JSON ends the streamed reply before it, with
/// `finish_reason: "stop"`.
#[tokio::test]
async fn a_stop_string_inside_the_object_cuts_the_streamed_reply_there() {
    let stopped = Ask {
        stop: Some(","),
        ..ask(TWO_KEYS, Mode::JsonObject, true)
    };
    let (got, _) = reply(stopped).await;
    assert_eq!(got.content, ["{", "\"a\"", ":", " 1"]);
    assert_eq!(got.finish, ["stop"]);
    assert!(got.errors.is_empty(), "{:?}", got.errors);
}

/// A stop string that does not match holds back the end of the text. That
/// text is cut at the engagement byte as all other text is.
#[tokio::test]
async fn text_held_for_a_stop_string_is_cut_at_the_engagement_byte_on_both_paths() {
    for stream in [true, false] {
        let held = Ask {
            stop: Some("\n{x"),
            ..ask(CUT_AT_THE_BRACE, Mode::JsonObject, stream)
        };
        let (got, _) = reply(held).await;
        assert_eq!(got.text(), "{", "stream={stream}");
        assert_eq!(got.finish, ["length"], "stream={stream}");
    }
}

/// A stop string that starts inside a token keeps the text of that token
/// before it.
#[tokio::test]
async fn a_stop_string_inside_a_token_keeps_the_text_before_it_on_both_paths() {
    for stream in [true, false] {
        let stopped = Ask {
            stop: Some("a\""),
            ..ask(TWO_KEYS, Mode::JsonObject, stream)
        };
        let (got, _) = reply(stopped).await;
        assert_eq!(got.text(), "{\"", "stream={stream}");
        assert_eq!(got.finish, ["stop"], "stream={stream}");
    }
}

// ── The streamed reply starts at the engagement byte ────────────────────────

#[tokio::test]
async fn a_json_fence_header_is_not_in_the_streamed_reply() {
    assert_streamed(JSON_FENCE, Mode::JsonObject, OBJECT_PIECES).await;
}

#[tokio::test]
async fn a_json_fence_header_is_not_in_the_streamed_reply_of_an_object_schema() {
    assert_streamed(JSON_FENCE, Mode::Schema(SCHEMA_OBJECT), OBJECT_PIECES).await;
}

#[tokio::test]
async fn prose_before_the_object_is_not_cut_at_a_letter_in_the_streamed_reply() {
    assert_streamed(PROSE, Mode::JsonObject, OBJECT_PIECES).await;
}

#[tokio::test]
async fn prose_before_the_object_is_not_in_the_streamed_reply_of_an_object_schema() {
    assert_streamed(PROSE, Mode::Schema(SCHEMA_OBJECT), OBJECT_PIECES).await;
}

#[tokio::test]
async fn prose_with_a_number_is_not_cut_at_the_digit_in_the_streamed_reply() {
    assert_streamed(PROSE_WITH_NUMBER, Mode::JsonObject, OBJECT_PIECES).await;
}

#[tokio::test]
async fn prose_with_a_quoted_word_is_not_cut_at_the_quote_in_the_streamed_reply() {
    assert_streamed(PROSE_WITH_QUOTED_WORD, Mode::JsonObject, OBJECT_PIECES).await;
}

#[tokio::test]
async fn prose_that_starts_with_a_dash_is_not_in_the_streamed_reply() {
    assert_streamed(PROSE_WITH_DASH, Mode::JsonObject, OBJECT_PIECES).await;
}

#[tokio::test]
async fn a_cut_object_in_a_json_fence_is_streamed_from_its_brace() {
    assert_streamed(
        CUT_OBJECT_IN_JSON_FENCE,
        Mode::JsonObject,
        &[
            "{",
            "\"Afghanistan\"",
            ":",
            " \"Kabul\"",
            ",",
            " \"Albania\"",
            ":",
        ],
    )
    .await;
}

/// The array grammar does not engage at the `{` of the prose.
#[tokio::test]
async fn a_brace_in_the_prose_does_not_start_the_streamed_reply_of_an_array_schema() {
    assert_streamed(
        BRACE_PROSE_THEN_ARRAY,
        Mode::Schema(SCHEMA_ARRAY),
        &["[", "10", ",", " 20", "]"],
    )
    .await;
}

/// The object grammar does not engage at the `[` of the prose.
#[tokio::test]
async fn a_bracket_in_the_prose_does_not_start_the_streamed_reply_of_an_object_schema() {
    assert_streamed(
        BRACKET_PROSE_THEN_OBJECT,
        Mode::Schema(SCHEMA_OBJECT),
        OBJECT_PIECES,
    )
    .await;
}

// ── A streamed reply that is not engaged is refused ─────────────────────────

#[tokio::test]
async fn a_streamed_reply_whose_grammar_never_engaged_ends_with_an_error_event() {
    assert_stream_refuses(ask(NO_JSON, Mode::JsonObject, true)).await;
}

/// The refused stream is a mid-stream failure: the error event, then `[DONE]`.
/// The request asks for a usage chunk, and a failed stream sends none.
#[tokio::test]
async fn a_refused_stream_sends_the_error_event_and_nothing_after_it() {
    let (status, raw, _) = serve(ask(NO_JSON, Mode::JsonObject, true)).await;
    assert_eq!(status, 200, "body: {raw}");
    let data: Vec<&str> = raw
        .lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .map(str::trim)
        .collect();
    let at = data
        .iter()
        .position(|d| d.contains("\"error\""))
        .unwrap_or_else(|| panic!("no error event: {data:?}"));
    let error: Value = serde_json::from_str(data[at]).unwrap();
    assert_eq!(error["error"]["type"], "constraint_not_engaged");
    assert!(error["error"]["message"].is_string());
    assert_eq!(error.as_object().unwrap().len(), 1, "the envelope only");
    assert_eq!(data[at + 1..], ["[DONE]"], "nothing after the error event");
    assert!(!raw.contains("\"usage\":{"), "no usage chunk: {raw}");
}

/// The generation ends at the stop string, before the grammar engaged.
#[tokio::test]
async fn a_stop_string_in_the_prose_ends_the_streamed_reply_with_an_error_event() {
    let stopped = Ask {
        stop: Some(" JSON"),
        ..ask(PROSE, Mode::JsonObject, true)
    };
    assert_stream_refuses(stopped).await;
}

// ── The non-streamed reply is never one smaller value ───────────────────────

#[tokio::test]
async fn an_object_cut_by_the_token_limit_is_not_returned_as_its_first_key() {
    assert_blocking(CUT_OBJECT, Mode::JsonObject, CUT_OBJECT_TEXT).await;
}

#[tokio::test]
async fn a_cut_object_of_an_object_schema_is_not_returned_as_its_first_key() {
    assert_blocking(CUT_SCHEMA_OBJECT, Mode::Schema(SCHEMA_OBJECT), r#"{"a":"#).await;
}

#[tokio::test]
async fn a_cut_object_in_a_json_fence_is_not_returned_as_its_first_key() {
    assert_blocking(CUT_OBJECT_IN_JSON_FENCE, Mode::JsonObject, CUT_OBJECT_TEXT).await;
}

#[tokio::test]
async fn a_cut_object_is_not_returned_as_a_closed_inner_object_or_key() {
    assert_blocking(
        CUT_AFTER_INNER_OBJECT,
        Mode::JsonObject,
        r#"{"a": {"b": 1}, "c":"#,
    )
    .await;
}

#[tokio::test]
async fn a_cut_array_is_not_returned_as_its_first_number() {
    assert_blocking(CUT_ARRAY, Mode::JsonObject, "[10, 20,").await;
}

#[tokio::test]
async fn a_number_in_the_prose_is_not_returned_in_place_of_the_object() {
    assert_blocking(PROSE_WITH_NUMBER, Mode::JsonObject, OBJECT_TEXT).await;
}

#[tokio::test]
async fn a_number_in_the_prose_is_not_returned_in_place_of_the_object_of_a_schema() {
    assert_blocking(PROSE_WITH_NUMBER, Mode::Schema(SCHEMA_OBJECT), OBJECT_TEXT).await;
}

#[tokio::test]
async fn a_quoted_word_in_the_prose_is_not_returned_in_place_of_the_object() {
    assert_blocking(PROSE_WITH_QUOTED_WORD, Mode::JsonObject, OBJECT_TEXT).await;
}

#[tokio::test]
async fn a_brace_in_the_prose_is_not_returned_in_place_of_the_array_of_a_schema() {
    assert_blocking(
        BRACE_PROSE_THEN_ARRAY,
        Mode::Schema(SCHEMA_ARRAY),
        "[10, 20]",
    )
    .await;
}

#[tokio::test]
async fn a_bracket_in_the_prose_is_not_returned_in_place_of_the_object_of_a_schema() {
    assert_blocking(
        BRACKET_PROSE_THEN_OBJECT,
        Mode::Schema(SCHEMA_OBJECT),
        OBJECT_TEXT,
    )
    .await;
}

/// The streamed reply holds this white space, so the non-streamed reply holds
/// it also.
#[tokio::test]
async fn white_space_after_the_closed_object_is_in_the_non_streamed_reply() {
    assert_blocking(OBJECT_THEN_NEWLINE, Mode::JsonObject, "{\"a\": 1}\n").await;
}

/// A stop string inside the JSON ends the non-streamed reply before it, as it
/// ends the streamed reply.
#[tokio::test]
async fn a_stop_string_inside_the_object_cuts_the_non_streamed_reply_there() {
    let stopped = Ask {
        stop: Some(","),
        ..ask(TWO_KEYS, Mode::JsonObject, false)
    };
    let (got, _) = reply(stopped).await;
    assert_eq!(got.finish, ["stop"], "harness: the stop string matched");
    assert_eq!(got.text(), r#"{"a": 1"#, "non-streamed content");
}

/// The generation ends at the stop string, before the grammar engaged.
#[tokio::test]
async fn a_stop_string_in_the_prose_makes_the_non_streamed_reply_a_refusal() {
    let stopped = Ask {
        stop: Some(" JSON"),
        ..ask(PROSE, Mode::JsonObject, false)
    };
    assert_blocking_refuses(stopped).await;
}
