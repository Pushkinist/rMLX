//! `chat_template_kwargs` and `reasoning_effort` on `/v1/chat/completions`.
//!
//! Each test sends a request over HTTP to a real router. A generator that
//! records the `GenerationRequest` stands in for the model. So the tests read
//! the rendered prompt and the inputs of the think splitter at the point where
//! the route hands the request to the engine, and the reply the route builds
//! from the generator's tokens.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    trivial_casts
)]

mod stub_server;

use std::pin::Pin;
use std::sync::{Arc, OnceLock};

use futures::stream::{self, Stream};
use parking_lot::Mutex;
use rmlx_server::{GenerationRequest, GenerationToken, Generator, LoadedModel, ModelRegistry};
use serde_json::{json, Value};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

const MODEL: &str = "kwmodel";

/// Every word the template can print. A word outside this list would decode
/// as `[UNK]` and hide what the template rendered.
const VOCAB: &[&str] = &[
    "hi",
    "evil",
    "BOS",
    "EOS",
    "foo_bar",
    "foo_1",
    "effort_xhigh",
    "effort_low",
    "effort_medium",
    "thinking_on",
    "thinking_off",
    "tools_present",
    "<",
    "></",
    "think",
    ">",
    "[UNK]",
];

/// Prints the variables a client can set, so a test reads them back from the
/// prompt. It refuses an effort outside xhigh, medium and low, as the
/// Qwen3.8 template does. It closes the think block for `enable_thinking`
/// equal to `false`, as that template does.
const TEMPLATE: &str = "\
{%- if messages[-1].content.startswith('refuse') %}{{ raise_exception('refused ' ~ messages[-1].content) }}{% endif %}\
{%- if messages[-1].content.startswith('fail') %}{% include messages[-1].content %}{% endif %}\
{{ bos_token }} \
{%- for m in messages %} {{ m.content }}{% endfor %} {{ eos_token }} \
{%- if tools %} tools_present{% endif %} \
{%- if foo is defined %} foo_{{ foo }}{% endif %} \
{%- if reasoning_effort is defined %}\
{%- if reasoning_effort not in ('xhigh', 'medium', 'low') %}\
{{- raise_exception('Unexpected reasoning effort ' ~ reasoning_effort ~ '. Supported types are xhigh, medium, and low.') }}\
{%- endif %} effort_{{ reasoning_effort }}{% endif %} \
{%- if enable_thinking is defined %} thinking_{{ 'on' if enable_thinking else 'off' }}{% endif %} \
{%- if add_generation_prompt %}\
{%- if enable_thinking is defined and enable_thinking is false %} <think></think>\
{%- else %} <think>{% endif %}{% endif %}";

/// Every tracing event of the test binary, as `name=value` text.
static EVENTS: OnceLock<Arc<Mutex<Vec<String>>>> = OnceLock::new();

struct FieldText(String);

impl tracing::field::Visit for FieldText {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        use std::fmt::Write;
        let _ = write!(self.0, "{}={value:?} ", field.name());
    }
}

struct Recorder(Arc<Mutex<Vec<String>>>);

impl tracing::Subscriber for Recorder {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        // axum's own trace event of a rejection holds the whole reply body.
        if event.metadata().target().starts_with("axum") {
            return;
        }
        let mut text = FieldText(String::new());
        event.record(&mut text);
        self.0.lock().push(text.0);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Install the recorder once and return what it has recorded.
fn events() -> &'static Arc<Mutex<Vec<String>>> {
    EVENTS.get_or_init(|| {
        let log = Arc::new(Mutex::new(Vec::new()));
        tracing::subscriber::set_global_default(Recorder(Arc::clone(&log))).unwrap();
        log
    })
}

/// What the generator saw of one request.
#[derive(Debug, Clone, PartialEq)]
struct Seen {
    prompt: String,
    prompt_think_open: bool,
    thinking_budget: Option<u32>,
    thinking_end_token_id: Option<u32>,
}

/// Records the request. Its reply puts the first piece in the reasoning
/// channel when the prompt left the think block open, as the think splitter
/// does at its start.
struct RecordingGenerator {
    tokenizer: tokenizers::Tokenizer,
    seen: Arc<Mutex<Vec<Seen>>>,
}

impl Generator for RecordingGenerator {
    fn generate(
        &self,
        req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        let prompt = self.tokenizer.decode(&req.prompt_tokens, false).unwrap();
        self.seen.lock().push(Seen {
            prompt,
            prompt_think_open: req.prompt_think_open,
            thinking_budget: req.thinking_budget,
            thinking_end_token_id: req.thinking_end_token_id,
        });
        let piece = |text: &str, is_thinking: bool| {
            Ok(GenerationToken {
                token_id: 1,
                piece: text.to_owned(),
                done: false,
                finish_reason: None,
                is_thinking,
                logprobs: None,
            })
        };
        let terminal = Ok(GenerationToken {
            token_id: 0,
            piece: String::new(),
            done: true,
            finish_reason: Some("stop".to_owned()),
            is_thinking: false,
            logprobs: None,
        });
        Box::pin(stream::iter(vec![
            piece("R", req.prompt_think_open),
            piece("A", false),
            terminal,
        ]))
    }
}

struct Server {
    port: u16,
    seen: Arc<Mutex<Vec<Seen>>>,
    snapshot: String,
    _tmp: tempfile::TempDir,
}

async fn start(default_enable_thinking: Option<bool>) -> Server {
    events();
    let tmp = tempfile::tempdir().unwrap();
    let snap = tmp.path().join(MODEL);
    std::fs::create_dir_all(&snap).unwrap();
    std::fs::write(
        snap.join("config.json"),
        r#"{"architectures":["Scripted"]}"#,
    )
    .unwrap();
    std::fs::write(
        snap.join("tokenizer_config.json"),
        r#"{"bos_token":"BOS","eos_token":"EOS"}"#,
    )
    .unwrap();
    std::fs::write(snap.join("chat_template.jinja"), TEMPLATE).unwrap();
    let vocab: serde_json::Map<String, Value> = VOCAB
        .iter()
        .enumerate()
        .map(|(i, w)| ((*w).to_owned(), json!(i)))
        .collect();
    let tokenizer_json = json!({
        "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
        "normalizer": null, "pre_tokenizer": {"type": "Whitespace"},
        "post_processor": null, "decoder": null,
        "model": {"type": "WordLevel", "vocab": vocab, "unk_token": "[UNK]"}
    });
    std::fs::write(snap.join("tokenizer.json"), tokenizer_json.to_string()).unwrap();

    let registry = ModelRegistry::from_paths(std::slice::from_ref(&snap));
    let mut state = stub_server::not_ready_state(registry);
    state.default_enable_thinking = default_enable_thinking;
    let seen = Arc::new(Mutex::new(Vec::new()));
    let generator = RecordingGenerator {
        tokenizer: tokenizers::Tokenizer::from_file(snap.join("tokenizer.json")).unwrap(),
        seen: Arc::clone(&seen),
    };
    let now = std::time::Instant::now();
    state.slots.write().push(LoadedModel {
        id: MODEL.to_owned(),
        model: Arc::new(generator),
        loaded_at: now,
        last_used: now,
        effective_max_ctx: usize::MAX,
        context_limits: None,
        decode_lease: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        unload_handle: Arc::new(Mutex::new(None)),
        keep_alive: rmlx_server::KeepAlivePolicy::Pin,
    });
    let port = stub_server::start_server_with_state(state).await;
    Server {
        port,
        seen,
        snapshot: snap.display().to_string(),
        _tmp: tmp,
    }
}

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
    let status = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    (status, text[body_start..].to_owned())
}

/// The outcome of one request on a fresh server.
struct Outcome {
    status: u16,
    reply: Value,
    seen: Option<Seen>,
    snapshot: String,
}

/// Send `{"model", "messages": [hi], "max_tokens": 8}` plus `extra` members.
async fn send(extra: Value, default_enable_thinking: Option<bool>) -> Outcome {
    let server = start(default_enable_thinking).await;
    let mut body = json!({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8
    });
    for (k, v) in extra.as_object().unwrap() {
        body[k] = v.clone();
    }
    let (status, text) = post(server.port, "/v1/chat/completions", &body.to_string()).await;
    let reply = serde_json::from_str(&text).unwrap_or(Value::String(text));
    let seen = server.seen.lock().first().cloned();
    Outcome {
        status,
        reply,
        seen,
        snapshot: server.snapshot.clone(),
    }
}

fn seen(o: &Outcome) -> &Seen {
    o.seen
        .as_ref()
        .unwrap_or_else(|| panic!("status {}, reply {}", o.status, o.reply))
}

fn message(o: &Outcome) -> &Value {
    &o.reply["choices"][0]["message"]
}

// ── chat_template_kwargs ────────────────────────────────────────────────────

/// A switch-off in `chat_template_kwargs` changes the rendered prompt, the
/// think splitter's start, the reply split and `usage` as the top-level field
/// does. The control request shows the pair differs from a request with no
/// switch.
#[tokio::test]
async fn kwargs_thinking_off_behaves_as_the_top_level_field() {
    let nested = send(
        json!({"chat_template_kwargs": {"enable_thinking": false}, "thinking_budget": 8}),
        None,
    )
    .await;
    let top = send(
        json!({"enable_thinking": false, "thinking_budget": 8}),
        None,
    )
    .await;
    let control = send(json!({"thinking_budget": 8}), None).await;

    assert_eq!(nested.status, 200, "{}", nested.reply);
    assert_eq!(seen(&nested), seen(&top));
    assert!(!seen(&nested).prompt_think_open);
    assert_eq!(seen(&nested).thinking_budget, Some(8));
    assert!(seen(&nested).prompt.contains("thinking_off"));
    assert_eq!(message(&nested), message(&top));
    assert_eq!(message(&nested)["content"], "RA");
    assert!(message(&nested).get("reasoning_content").is_none());
    assert_eq!(nested.reply["usage"], top.reply["usage"]);

    assert!(seen(&control).prompt_think_open);
    assert_ne!(seen(&control).prompt, seen(&nested).prompt);
    assert_eq!(message(&control)["reasoning_content"], "R");
    assert_eq!(message(&control)["content"], "A");
}

/// A top-level `enable_thinking` wins over the same key in
/// `chat_template_kwargs`, for every pair of values. Top-level `false` closes
/// the think block. Top-level `true` leaves the template to its default, so
/// the nested value is not passed either.
#[tokio::test]
async fn a_top_level_enable_thinking_wins_over_kwargs_for_every_value() {
    for (top, nested) in [(true, false), (false, true), (true, true), (false, false)] {
        let both = send(
            json!({"enable_thinking": top, "chat_template_kwargs": {"enable_thinking": nested}}),
            None,
        )
        .await;
        let alone = send(json!({"enable_thinking": top}), None).await;
        assert_eq!(seen(&both), seen(&alone), "top {top}, nested {nested}");
        assert_eq!(
            seen(&both).prompt_think_open,
            top,
            "top {top}, nested {nested}"
        );
        assert_eq!(
            seen(&both).prompt.contains("thinking_off"),
            !top,
            "top {top}, nested {nested}"
        );
        assert!(!seen(&both).prompt.contains("thinking_on"));
        assert_eq!(
            message(&both),
            message(&alone),
            "top {top}, nested {nested}"
        );
        assert_eq!(both.reply["usage"], alone.reply["usage"]);
    }
}

/// With no `chat_template_kwargs`, the prompt is what the release before
/// `chat_template_kwargs` rendered: it passed `enable_thinking` only when the
/// top-level field, else the server default, was `false`. Nine pairs of
/// (field, default).
#[tokio::test]
async fn without_kwargs_the_prompt_is_the_one_the_release_rendered() {
    for top in [None, Some(true), Some(false)] {
        for default in [None, Some(true), Some(false)] {
            let extra = top.map_or_else(|| json!({}), |on| json!({"enable_thinking": on}));
            let o = send(extra, default).await;
            let closed = top.or(default) == Some(false);
            let want = if closed {
                "BOS hi EOS thinking_off < think ></ think >"
            } else {
                "BOS hi EOS < think >"
            };
            let seen = seen(&o);
            assert_eq!(seen.prompt, want, "field {top:?}, default {default:?}");
            assert_eq!(seen.prompt_think_open, !closed);
        }
    }
}

/// `chat_template_kwargs` with `enable_thinking: true` and the server default
/// off: the request wins, over a bool and over a value that is not a bool.
#[tokio::test]
async fn kwargs_win_over_the_server_default() {
    let on = send(
        json!({"chat_template_kwargs": {"enable_thinking": true}}),
        Some(false),
    )
    .await;
    assert!(seen(&on).prompt_think_open);
    assert!(seen(&on).prompt.contains("thinking_on"));

    let default_only = send(json!({}), Some(false)).await;
    assert!(!seen(&default_only).prompt_think_open);

    let not_a_bool = send(
        json!({"chat_template_kwargs": {"enable_thinking": "no"}}),
        Some(false),
    )
    .await;
    assert!(seen(&not_a_bool).prompt.contains("thinking_on"));
}

/// Any key goes to the template. No key is refused.
#[tokio::test]
async fn every_kwargs_key_reaches_the_template() {
    let text = send(json!({"chat_template_kwargs": {"foo": "bar"}}), None).await;
    assert!(seen(&text).prompt.contains("foo_bar"), "{:?}", seen(&text));
    let number = send(
        json!({"chat_template_kwargs": {"foo": 1, "never_read": {"a": [1, 2]}}}),
        None,
    )
    .await;
    assert_eq!(number.status, 200, "{}", number.reply);
    assert!(seen(&number).prompt.contains("foo_1"));
}

/// The variables the server sets from the request cannot be replaced.
#[tokio::test]
async fn kwargs_cannot_replace_the_variables_the_server_sets() {
    let plain = send(json!({}), None).await;
    let evil = send(
        json!({"chat_template_kwargs": {
            "messages": [{"role": "user", "content": "evil"}],
            "tools": [{"type": "function", "function": {"name": "evil"}}],
            "add_generation_prompt": false,
            "bos_token": "evil",
            "eos_token": "evil"
        }}),
        None,
    )
    .await;
    assert_eq!(evil.status, 200, "{}", evil.reply);
    assert_eq!(seen(&evil), seen(&plain));
    assert!(!seen(&evil).prompt.contains("evil"));
    assert!(seen(&evil).prompt.contains("think"));
}

/// A kwargs value the template refuses is 400 with the template's message.
#[tokio::test]
async fn a_kwargs_effort_the_template_refuses_is_400_with_its_message() {
    let refused = send(
        json!({"chat_template_kwargs": {"reasoning_effort": "high"}}),
        None,
    )
    .await;
    assert_eq!(refused.status, 400, "{}", refused.reply);
    assert!(refused.reply["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Unexpected reasoning effort high. Supported types are xhigh, medium, and low."));
    assert!(refused.seen.is_none());
}

/// `chat_template_kwargs` that is not an object is a client error.
#[tokio::test]
async fn kwargs_that_is_not_an_object_is_a_client_error() {
    let o = send(json!({"chat_template_kwargs": "enable_thinking"}), None).await;
    assert!((400..500).contains(&o.status), "{} {}", o.status, o.reply);
    assert!(o.seen.is_none());
}

/// The Anthropic route has no template-variable fields. It renders with the
/// server default for `enable_thinking`.
#[tokio::test]
async fn the_anthropic_route_renders_with_the_server_default() {
    for (default, open) in [(Some(false), false), (Some(true), true), (None, true)] {
        let server = start(default).await;
        let body = json!({
            "model": MODEL, "max_tokens": 8,
            "messages": [{"role": "user", "content": "hi"}]
        });
        let (status, reply) = post(server.port, "/v1/messages", &body.to_string()).await;
        assert_eq!(status, 200, "default {default:?}: {reply}");
        let seen = server.seen.lock().first().cloned().unwrap();
        assert_eq!(seen.prompt_think_open, open, "default {default:?}");
        assert_eq!(seen.prompt.contains("thinking_off"), default == Some(false));
        assert!(!seen.prompt.contains("thinking_on"), "default {default:?}");
    }
}

// ── reasoning_effort ────────────────────────────────────────────────────────

/// An effort the template takes reaches it as sent, at the top level and in
/// `chat_template_kwargs`. A top-level effort wins over the nested one.
#[tokio::test]
async fn an_effort_reaches_the_template_as_sent() {
    for (extra, want) in [
        (json!({"reasoning_effort": "low"}), "effort_low"),
        (
            json!({"chat_template_kwargs": {"reasoning_effort": "medium"}}),
            "effort_medium",
        ),
        (
            json!({"reasoning_effort": "low",
                   "chat_template_kwargs": {"reasoning_effort": "xhigh"}}),
            "effort_low",
        ),
        (
            json!({"reasoning_effort": "xhigh",
                   "chat_template_kwargs": {"reasoning_effort": "high"}}),
            "effort_xhigh",
        ),
    ] {
        let o = send(extra.clone(), None).await;
        assert_eq!(o.status, 200, "{extra}: {}", o.reply);
        assert!(seen(&o).prompt.contains(want), "{extra}: {:?}", seen(&o));
    }
}

/// A value the template refuses is not rewritten and not pre-checked by the
/// server: the reply is 400 with the template's own message, for `high` and
/// for values with a capital, a dash, a digit or a trailing space.
#[tokio::test]
async fn an_effort_the_template_refuses_is_400_with_the_template_message() {
    for effort in ["high", "x-high", "High", "a1", "xhigh "] {
        let o = send(json!({"reasoning_effort": effort}), None).await;
        assert_eq!(o.status, 400, "{effort:?}: {}", o.reply);
        let msg = o.reply["error"]["message"].as_str().unwrap();
        assert!(
            msg.contains(&format!(
                "Unexpected reasoning effort {effort}. Supported types are xhigh, medium, and low."
            )),
            "{effort:?}: {msg}"
        );
        assert!(o.seen.is_none(), "{effort:?}");
    }
}

/// A 1 MiB effort that the template copies into its refusal gives a small 400
/// with no server path in it.
#[tokio::test]
async fn a_huge_effort_is_a_small_400_without_a_path() {
    let huge = "a".repeat(1 << 20);
    for extra in [
        json!({"reasoning_effort": huge}),
        json!({"chat_template_kwargs": {"reasoning_effort": huge}}),
    ] {
        let o = send(extra, None).await;
        let text = o.reply.to_string();
        assert_eq!(o.status, 400, "{}", &text[..text.len().min(200)]);
        assert!(text.len() < 1024, "{} bytes", text.len());
        assert!(text.contains("Unexpected reasoning effort"));
        assert!(!text.contains(&o.snapshot), "{text}");
    }
}

// ── log bounds ──────────────────────────────────────────────────────────────

/// No log event holds a client value whole. The error text of a template error
/// and of a rejected JSON body is cut to 256 characters; the 2 KiB body snippet
/// of a rejection is the one larger piece, so 4096 characters in one event is
/// the failure. The reply is asserted small only for a template error (a
/// refusal or a render failure); the reply of a rejected JSON body is not
/// covered. Cases: both routes, an effort at the top level and in
/// `chat_template_kwargs`, and a `chat_template_kwargs` of the wrong type.
#[tokio::test]
async fn a_client_value_in_an_error_is_cut_in_the_log_and_in_a_template_error_reply() {
    let huge = "Z".repeat(1 << 20);
    let chat = |extra: Value| {
        let mut body = json!({"model": MODEL, "max_tokens": 8,
                              "messages": [{"role": "user", "content": "hi"}]});
        for (k, v) in extra.as_object().unwrap() {
            body[k] = v.clone();
        }
        ("/v1/chat/completions", body)
    };
    let with_content = |route: &'static str, content: String| {
        (
            route,
            json!({"model": MODEL, "max_tokens": 8,
                   "messages": [{"role": "user", "content": content}]}),
        )
    };
    let cases = [
        chat(json!({"reasoning_effort": huge})),
        chat(json!({"chat_template_kwargs": {"reasoning_effort": huge}})),
        chat(json!({"chat_template_kwargs": huge})),
        chat(json!({"chat_template_kwargs": {
            "messages": huge, "tools": huge, "add_generation_prompt": huge,
            "bos_token": huge, "eos_token": huge
        }})),
        with_content("/v1/chat/completions", format!("refuse {huge}")),
        with_content("/v1/chat/completions", format!("fail{huge}")),
        with_content("/v1/messages", format!("refuse {huge}")),
        with_content("/v1/messages", format!("fail{huge}")),
    ];
    let server = start(None).await;
    for (route, body) in &cases {
        let (status, reply) = post(server.port, route, &body.to_string()).await;
        let shown = &reply[..reply.len().min(100)];
        let from_template = body["messages"][0]["content"]
            .as_str()
            .is_some_and(|c| c.starts_with("refuse") || c.starts_with("fail"))
            || body.get("reasoning_effort").is_some();
        if from_template {
            assert!(status >= 400, "{route}: status {status}");
            assert!(
                reply.len() < 2048,
                "{route}: reply of {} bytes: {shown}",
                reply.len()
            );
        }
    }
    let whole = &huge[..4096];
    for event in events().lock().iter() {
        assert!(
            !event.contains(whole),
            "a log event holds {} characters of a request value: {}",
            event.len(),
            &event[..event.len().min(120)]
        );
    }
}

// ── what a client key cannot replace ────────────────────────────────────────

/// A `chat_template_kwargs` key named like a function the server gives the
/// template replaces nothing: the template still refuses `high` with its own
/// message, as it does without the key.
#[tokio::test]
async fn a_kwargs_key_does_not_replace_a_function_of_the_template() {
    let o = send(
        json!({"reasoning_effort": "high",
               "chat_template_kwargs": {"raise_exception": 1}}),
        None,
    )
    .await;
    assert_eq!(o.status, 400, "{}", o.reply);
    assert!(o.reply["error"]["message"]
        .as_str()
        .unwrap()
        .contains("Unexpected reasoning effort high. Supported types are xhigh, medium, and low."));
}

/// One `warn` per dropped key per request, although the route renders twice.
#[tokio::test]
async fn a_dropped_key_is_logged_once_per_request() {
    let count = || {
        events()
            .lock()
            .iter()
            .filter(|e| e.contains("variable=\"range\""))
            .count()
    };
    let before = count();
    let o = send(
        json!({"chat_template_kwargs": {"range": 1, "foo": "bar"}}),
        None,
    )
    .await;
    assert_eq!(o.status, 200, "{}", o.reply);
    assert!(seen(&o).prompt.contains("foo_bar"));
    assert_eq!(count() - before, 1);
}

// ── the input bound ─────────────────────────────────────────────────────────

/// `chat_template_kwargs` counts against the bound on the text a request sends
/// to the tokeniser, the same bound as message text: alone, and added to the
/// message text.
#[tokio::test]
async fn kwargs_count_against_the_input_text_bound() {
    let big = "a".repeat(4 << 20);
    let half = "a".repeat(2 << 20);
    let refused =
        |o: &Outcome| o.status == 413 && o.reply["error"]["field"] == "total_input_tokens_estimate";

    let in_message = send(
        json!({"messages": [{"role": "user", "content": big}]}),
        None,
    )
    .await;
    assert!(refused(&in_message), "{}", in_message.status);
    let in_kwargs = send(json!({"chat_template_kwargs": {"foo": big}}), None).await;
    assert!(refused(&in_kwargs), "{}", in_kwargs.status);
    let in_both = send(
        json!({"messages": [{"role": "user", "content": half}],
               "chat_template_kwargs": {"foo": half}}),
        None,
    )
    .await;
    assert!(refused(&in_both), "{}", in_both.status);
    assert!(in_kwargs.seen.is_none());

    let under = send(json!({"chat_template_kwargs": {"foo": half}}), None).await;
    assert_eq!(under.status, 200, "{}", under.reply);
}
