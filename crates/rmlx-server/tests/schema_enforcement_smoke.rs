//! HTTP-surface tests for `json_schema` enforcement and the prompt-derived
//! reasoning channel — one surface, because the channel decides whether the
//! constraint ever engages.
//!
//! The value under test is `GenerationRequest::prompt_think_open`: the route
//! renders the chat template, reads off whether the assistant turn was left
//! inside an open `<think>` block, and threads the answer to the engine, which
//! uses it both for the `ThinkSplitter`'s initial channel and for the seed of
//! the JSON constraint's `is_thinking` gate. Getting it wrong latches the gate
//! and the constraint never engages — a request that answers HTTP 200 with the
//! schema never applied.
//!
//! A `RecordingGenerator` captures the request the route actually built, so
//! these exercise the real render → `GenerationRequest` path rather than
//! re-deriving the value in the test.

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
use parking_lot::Mutex;
use rmlx_server::{
    ApiErrorCounters, AppState, GenerationRequest, GenerationToken, Generator, ItlStore,
    LoadedModel, ModelLoader, ModelRegistry, SessionCache, TtftStore,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Chat templates under test ───────────────────────────────────────────────
//
// The three shapes real checkpoints ship, all within the set of
// thinking-capable architectures — which is exactly why the channel cannot be
// derived from the architecture.

/// Ternary-Bonsai's shape: the generation prompt closes the block itself, so
/// the model answers directly and never emits `</think>`.
const TPL_CLOSED_PREFILL: &str = "{% for m in messages %}{{ m['content'] }}{% endfor %}\
{% if add_generation_prompt %}<think>\n\n</think>\n\n{% endif %}";

/// Qwen3.6's thinking-on shape: the generation prompt leaves the block open.
const TPL_OPEN_PREFILL: &str = "{% for m in messages %}{{ m['content'] }}{% endfor %}\
{% if add_generation_prompt %}<think>\n{% endif %}";

/// Gemma-style: no think delimiters at all in the generation prompt.
const TPL_NO_PREFILL: &str = "{% for m in messages %}{{ m['content'] }}{% endfor %}\
{% if add_generation_prompt %}model\n{% endif %}";

// ── Recording generator ─────────────────────────────────────────────────────

/// Captures `prompt_think_open` from the request the route built, then emits a
/// single trivial token so the response completes normally.
#[derive(Clone)]
struct RecordingGenerator {
    seen: Arc<Mutex<Option<bool>>>,
}

impl Generator for RecordingGenerator {
    fn generate(
        &self,
        req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        *self.seen.lock() = Some(req.prompt_think_open);
        Box::pin(stream::iter(vec![
            Ok(GenerationToken {
                token_id: 1,
                piece: "ok".to_owned(),
                done: false,
                finish_reason: None,
                is_thinking: false,
                logprobs: None,
            }),
            Ok(GenerationToken {
                token_id: 0,
                piece: String::new(),
                done: true,
                finish_reason: Some("length".to_owned()),
                is_thinking: false,
                logprobs: None,
            }),
        ]))
    }
}

// ── AppState + server wiring ────────────────────────────────────────────────

fn recording_state(template: &str) -> (AppState, Arc<Mutex<Option<bool>>>, tempfile::TempDir) {
    let tmp = tempfile::tempdir().unwrap();
    let snap = tmp.path().join("recording");
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(
        snap.join("config.json"),
        r#"{"architectures":["Scripted"]}"#,
    )
    .unwrap();
    std::fs::write(snap.join("chat_template.jinja"), template).unwrap();
    std::fs::write(
        snap.join("tokenizer.json"),
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"x":0,"hi":1,"[UNK]":2},"unk_token":"[UNK]"}}"#,
    )
    .unwrap();

    let seen: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let loader_seen = Arc::clone(&seen);
    let loader: ModelLoader = Arc::new(move |_path, _id| {
        Ok(Box::new(RecordingGenerator {
            seen: Arc::clone(&loader_seen),
        }) as Box<dyn Generator>)
    });
    let state = AppState {
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
        id: "recording".to_owned(),
        model: Arc::new(RecordingGenerator {
            seen: Arc::clone(&seen),
        }),
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

async fn start_server(template: &str) -> (u16, Arc<Mutex<Option<bool>>>, tempfile::TempDir) {
    let (state, seen, tmp) = recording_state(template);
    let router = rmlx_server::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    (port, seen, tmp)
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

/// Send one chat request and return what the engine saw for
/// `prompt_think_open`.
async fn observed_channel(template: &str, user_content: &str) -> bool {
    let (port, seen, _tmp) = start_server(template).await;
    let body = serde_json::json!({
        "model": "recording",
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": 4
    })
    .to_string();
    let (status, b) = http(port, "/v1/chat/completions", &body).await;
    assert_eq!(status, 200, "body: {b}");
    let observed = seen.lock().expect("generator was invoked");
    observed
}

// ── The three template shapes reach the engine correctly ────────────────────

#[tokio::test]
async fn closed_prefill_reaches_the_engine_as_answer_mode() {
    assert!(
        !observed_channel(TPL_CLOSED_PREFILL, "hi").await,
        "a generation prompt that closes the think block must reach the engine \
         as answer-mode, or the constraint's is_thinking gate latches"
    );
}

#[tokio::test]
async fn open_prefill_reaches_the_engine_as_reasoning_mode() {
    assert!(
        observed_channel(TPL_OPEN_PREFILL, "hi").await,
        "a generation prompt that leaves the think block open must reach the \
         engine as reasoning-mode"
    );
}

#[tokio::test]
async fn absent_prefill_reaches_the_engine_as_answer_mode() {
    assert!(
        !observed_channel(TPL_NO_PREFILL, "hi").await,
        "no delimiters in the generation prompt means the model has not started \
         reasoning yet"
    );
}

// ── Message content must not decide the channel ─────────────────────────────

/// Message content is client-controlled. A user message carrying a bare
/// `<think>` must not flip the channel: it would latch `is_thinking`, route the
/// whole answer to `reasoning_content`, and leave a `json_schema` request
/// unenforced at HTTP 200 — reachable by anyone who can send a message.
#[tokio::test]
async fn a_think_tag_in_user_content_does_not_flip_the_channel() {
    assert!(
        !observed_channel(TPL_NO_PREFILL, "please explain <think> to me").await,
        "user content must not be able to open the reasoning channel"
    );
    assert!(
        !observed_channel(TPL_CLOSED_PREFILL, "please explain <think> to me").await,
        "user content must not override a template that closed the block"
    );
}

/// The inverse: content must not be able to *close* a block the template left
/// open either.
#[tokio::test]
async fn a_close_tag_in_user_content_does_not_flip_the_channel() {
    assert!(
        observed_channel(TPL_OPEN_PREFILL, "I said </think> earlier").await,
        "user content must not be able to close a block the template opened"
    );
}

// ── Degenerate delimiter overrides are refused at the boundary ──────────────

/// An empty delimiter matches at every offset: the prompt scan would report the
/// block open for any prompt, and the splitter's own scanner would never
/// advance past it. Both are reachable from one JSON field, so the route
/// refuses rather than letting either consumer inherit the value.
#[tokio::test]
async fn empty_think_delimiter_is_rejected() {
    for (field, value) in [
        ("thinking_start_token", ""),
        ("thinking_end_token", ""),
        ("thinking_start_token", "   "),
        ("thinking_end_token", "\t\n"),
    ] {
        let (port, _seen, _tmp) = start_server(TPL_CLOSED_PREFILL).await;
        let body = serde_json::json!({
            "model": "recording",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4,
            field: value,
        })
        .to_string();
        let (status, b) = http(port, "/v1/chat/completions", &body).await;
        assert_eq!(
            status, 400,
            "`{field}` = {value:?} must be refused, got {status}: {b}"
        );
    }
}

/// A non-empty override still works — the guard rejects only the degenerate
/// value, it does not disable the feature.
#[tokio::test]
async fn non_empty_think_delimiter_override_is_accepted() {
    let (port, seen, _tmp) = start_server(TPL_NO_PREFILL).await;
    let body = serde_json::json!({
        "model": "recording",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "thinking_start_token": "<|reason|>",
        "thinking_end_token": "<|/reason|>",
    })
    .to_string();
    let (status, b) = http(port, "/v1/chat/completions", &body).await;
    assert_eq!(status, 200, "body: {b}");
    let observed = seen.lock().expect("generator was invoked");
    assert!(
        !observed,
        "template emits neither custom delimiter — channel stays closed"
    );
}

// ── The Anthropic surface reads the same suffix ─────────────────────────────

/// `/v1/messages` renders the same template through its own pipeline. It builds
/// no constraint, so the blast radius is reasoning-vs-content routing rather
/// than schema enforcement — but the derivation must not diverge between the
/// two surfaces, and message content must not decide it there either.
#[tokio::test]
async fn anthropic_route_reads_the_generation_suffix_too() {
    for (template, content, want) in [
        (TPL_CLOSED_PREFILL, "hi", false),
        (TPL_OPEN_PREFILL, "hi", true),
        (TPL_NO_PREFILL, "explain <think> please", false),
    ] {
        let (port, seen, _tmp) = start_server(template).await;
        let body = serde_json::json!({
            "model": "recording",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 4
        })
        .to_string();
        let (status, b) = http(port, "/v1/messages", &body).await;
        assert_eq!(status, 200, "body: {b}");
        let observed = seen.lock().expect("generator was invoked");
        assert_eq!(
            observed, want,
            "template={template:?} content={content:?}: expected prompt_think_open={want}"
        );
    }
}

// ── A constraint that never engaged must not answer 200 ─────────────────────
//
// The recording generator ignores the constraint entirely, which is exactly the
// shape under test: the engine was built, moved into the decode path, and never
// applied to a single logit.

fn schema_request(stream: bool) -> String {
    serde_json::json!({
        "model": "recording",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
        "stream": stream,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "unit_answer",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": {"unit": {"type": "string", "enum": ["celsius"]}},
                    "required": ["unit"],
                    "additionalProperties": false
                }
            }
        }
    })
    .to_string()
}

/// Non-streaming: the whole stream is accumulated before any byte reaches the
/// client, so an unchecked body can still be refused. Returning it with HTTP
/// 200 would be indistinguishable from an enforced result.
#[tokio::test]
async fn nonstreaming_json_schema_that_never_engaged_is_refused() {
    let (port, _seen, _tmp) = start_server(TPL_NO_PREFILL).await;
    let (status, b) = http(port, "/v1/chat/completions", &schema_request(false)).await;
    assert_eq!(
        status, 502,
        "an unenforced response_format response must be refused, got {status}: {b}"
    );
    assert!(
        b.contains("constraint_not_engaged"),
        "the refusal must name its reason: {b}"
    );
}

/// Streaming cannot refuse: by the time the engine's terminal state is known,
/// the deltas are already on the wire. Pinned so the asymmetry is a decision on
/// record rather than an oversight — and so that a future buffering change has
/// to update this test deliberately.
#[tokio::test]
async fn streaming_json_schema_that_never_engaged_still_returns_200() {
    let (port, _seen, _tmp) = start_server(TPL_NO_PREFILL).await;
    let (status, _b) = http(port, "/v1/chat/completions", &schema_request(true)).await;
    assert_eq!(
        status, 200,
        "streaming has already emitted bytes; it cannot retract them"
    );
}

/// The refusal is scoped to `response_format`. A plain request through the same
/// generator — no constraint built at all — must be unaffected.
#[tokio::test]
async fn a_request_without_response_format_is_unaffected() {
    let (port, _seen, _tmp) = start_server(TPL_NO_PREFILL).await;
    let body = serde_json::json!({
        "model": "recording",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4
    })
    .to_string();
    let (status, b) = http(port, "/v1/chat/completions", &body).await;
    assert_eq!(status, 200, "body: {b}");
}
