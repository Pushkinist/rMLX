//! What the other work of the server sees while a model loads (no model, no
//! GPU: the loader is scripted and holds each load until the test releases
//! it).
//!
//! The routes are the routes of the server, under its layers and under
//! `serve_until`. A client is a blocking socket on the test thread with a
//! read timeout, and the runtime of the server is built by the test. So no
//! wait of a test needs a free async worker, and a server that stops serving
//! fails its test and does not hang the binary. A scripted load that is not
//! released returns an error at `LOAD_BOUND`.
//!
//! The server under test has one async worker where the test reads what a
//! load leaves for the other work: one load is then enough to show a worker
//! that is not free. `rmlx serve` has four, and four loads have the same
//! effect there.
//!
//! What cannot move, and its test:
//! - one load at a time, and no second load of a model that loads:
//!   `requests_for_a_model_that_loads_wait_and_use_that_load`,
//!   `a_load_after_a_load_evicts_by_the_slot_rule`;
//! - a request for a resident model does not wait for the load of another
//!   model: `a_request_for_a_resident_model_does_not_wait_for_a_load`;
//! - the cold flag, which is true for a call that found no resident model
//!   (it loaded, or it waited for a load) and false for a call that found
//!   one: `a_call_that_found_no_resident_model_is_cold`. The call that waits
//!   is known to wait: its first poll did not end it while the load was held;
//! - the load events of a chat or messages request are in its span:
//!   `the_load_events_of_a_request_carry_its_span`;
//! - a load that the runtime did not start is an error:
//!   `a_load_that_the_runtime_did_not_start_is_an_error`;
//! - the reply of a failed load: `a_failed_load_keeps_its_reply`,
//!   `a_load_that_panics_answers_500`, and
//!   `ensure_loaded_preserves_the_context_refusal_variant` of `http_smoke`
//!   for the error variant;
//! - the slot and keep-alive rules: the slot tests of `tests.rs` on the load
//!   itself, and `a_load_after_a_load_evicts_by_the_slot_rule` through the
//!   route;
//! - a client that leaves during a load:
//!   `a_client_that_leaves_during_a_load_leaves_one_resident_model`.
//!
//! Not covered here: the eager preload of `rmlx serve`. It calls the same
//! `ensure_loaded`, and the blocking load is not visible outside this crate,
//! so the compiler holds that it cannot load on its own thread again. A
//! start of a real server with a model is the proof of its log events.
//!
//! One check is not exact: that no second load started inside `SETTLE`. A
//! correct server always passes it. A server that starts a second load later
//! than `SETTLE` fails the count of loads that follows, not this check.

use std::io::{Read as _, Write as _};
use std::net::TcpStream;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::Request;
use axum::middleware::Next;
use parking_lot::{Condvar, Mutex};

use super::state::ModelLoader;
use super::tests::{slot_test_state, snapshot_registry};
use super::AppState;
use crate::engine::{Generator, NotReadyGenerator};
use crate::shutdown::{phase_channel, serve_until, Bounds, ExitDeadline};

/// The bound on every wait of a test.
const WAIT: Duration = Duration::from_secs(5);

/// A scripted load that no test released returns an error after this time.
const LOAD_BOUND: Duration = Duration::from_secs(20);

/// How long a test looks for a load that must not start.
const SETTLE: Duration = Duration::from_millis(500);

/// Shutdown bounds that no test reaches.
const FAR: Bounds = Bounds {
    reply: Duration::from_secs(120),
    exit: Duration::from_secs(180),
};

/// Shutdown bounds that a test reaches while its scripted load still runs.
const NEAR: Bounds = Bounds {
    reply: Duration::from_millis(100),
    exit: Duration::from_millis(200),
};

/// How a released load ends.
#[derive(Clone, Copy)]
enum Outcome {
    Loads,
    Fails,
    Panics,
}

/// The text of a scripted load failure.
const FAILURE: &str = "scripted load failure";

/// The loader of the server under test. Each load reports its model id, then
/// waits for one release.
struct Script {
    calls: AtomicUsize,
    entered: mpsc::Sender<String>,
    releases: Mutex<usize>,
    released: Condvar,
    outcome: Mutex<Outcome>,
}

impl Script {
    fn load(&self, id: &str) -> rmlx_core::Result<Box<dyn Generator>> {
        self.calls.fetch_add(1, Ordering::AcqRel);
        self.entered.send(id.to_owned()).ok();
        let deadline = Instant::now() + LOAD_BOUND;
        let mut releases = self.releases.lock();
        while *releases == 0 {
            if self
                .released
                .wait_until(&mut releases, deadline)
                .timed_out()
            {
                return Err(rmlx_core::Error::Model(
                    "the scripted load was not released".to_owned(),
                ));
            }
        }
        *releases -= 1;
        drop(releases);
        match *self.outcome.lock() {
            Outcome::Loads => Ok(Box::new(NotReadyGenerator)),
            Outcome::Fails => Err(rmlx_core::Error::Model(FAILURE.to_owned())),
            Outcome::Panics => panic!("scripted load panic"),
        }
    }

    fn release(&self, loads: usize) {
        *self.releases.lock() += loads;
        self.released.notify_all();
    }

    fn calls(&self) -> usize {
        self.calls.load(Ordering::Acquire)
    }
}

/// A server on its own runtime, with models `m0`, `m1`, … in its registry and
/// none loaded.
struct Server {
    port: u16,
    state: AppState,
    script: Arc<Script>,
    entered: mpsc::Receiver<String>,
    /// The events of the runtime threads of this server.
    events: Arc<Capture>,
    /// One message for each request that reached the routes.
    arrived: mpsc::Receiver<()>,
    signals: tokio::sync::mpsc::Sender<()>,
    exit: Option<tokio::task::JoinHandle<anyhow::Result<ExitDeadline>>>,
    runtime: Option<tokio::runtime::Runtime>,
    _snapshots: tempfile::TempDir,
}

impl Drop for Server {
    /// Every load that still waits returns, so the runtime can end.
    fn drop(&mut self) {
        self.script.release(usize::MAX / 2);
        drop(self.runtime.take());
    }
}

/// The files that let the chat routes render a prompt for a snapshot.
const PROMPT_FILES: [(&str, &str); 2] = [
    (
        "chat_template.jinja",
        "{% for m in messages %}{{ m['content'] }}{% endfor %}",
    ),
    (
        "tokenizer.json",
        r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"x":0,"hi":1,"[UNK]":2},"unk_token":"[UNK]"}}"#,
    ),
];

/// One event: its message, then the fields of the span it was emitted in.
type Event = (String, Vec<(String, String)>);

/// Records each event of the threads it is installed on, with the fields of
/// the span that the thread was inside.
#[derive(Default)]
struct Capture {
    next_span: AtomicUsize,
    spans: Mutex<std::collections::HashMap<u64, Vec<(String, String)>>>,
    events: Mutex<Vec<Event>>,
}

thread_local! {
    /// The spans this thread is inside, the innermost last.
    static ENTERED: std::cell::RefCell<Vec<u64>> = const { std::cell::RefCell::new(Vec::new()) };
}

#[derive(Default)]
struct Fields(Vec<(String, String)>);

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_owned(), format!("{value:?}")));
    }
}

impl tracing::Subscriber for Capture {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, attributes: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        let id = self.next_span.fetch_add(1, Ordering::AcqRel) as u64 + 1;
        let mut fields = Fields::default();
        attributes.record(&mut fields);
        self.spans.lock().insert(id, fields.0);
        tracing::span::Id::from_u64(id)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let span = match event.parent() {
            Some(parent) => Some(parent.into_u64()),
            None => ENTERED.with(|entered| entered.borrow().last().copied()),
        };
        let span_fields = span.and_then(|id| self.spans.lock().get(&id).cloned());
        let mut fields = Fields::default();
        event.record(&mut fields);
        let message = fields.0.into_iter().find(|(name, _)| name == "message");
        let message = message.map(|(_, text)| text).unwrap_or_default();
        self.events
            .lock()
            .push((message, span_fields.unwrap_or_default()));
    }
    fn enter(&self, span: &tracing::span::Id) {
        ENTERED.with(|entered| entered.borrow_mut().push(span.into_u64()));
    }
    fn exit(&self, _: &tracing::span::Id) {
        ENTERED.with(|entered| entered.borrow_mut().pop());
    }
}

thread_local! {
    /// Keeps the capture installed for the life of a runtime thread.
    static INSTALLED: std::cell::RefCell<Option<tracing::subscriber::DefaultGuard>> =
        const { std::cell::RefCell::new(None) };
}

/// A server with one slot.
fn start(workers: usize, models: usize, bounds: Bounds) -> Server {
    start_with_slots(workers, models, 1, bounds)
}

fn start_with_slots(workers: usize, models: usize, slots: usize, bounds: Bounds) -> Server {
    let (registry, snapshots) = snapshot_registry(models, &PROMPT_FILES);
    let (entered_tx, entered) = mpsc::channel();
    let script = Arc::new(Script {
        calls: AtomicUsize::new(0),
        entered: entered_tx,
        releases: Mutex::new(0),
        released: Condvar::new(),
        outcome: Mutex::new(Outcome::Loads),
    });
    let mut state = slot_test_state(registry, slots);
    let loader: ModelLoader = {
        let script = Arc::clone(&script);
        Arc::new(move |_path, id| script.load(id))
    };
    state.loader = loader;

    let events = Arc::new(Capture::default());
    let dispatch = tracing::Dispatch::new(Arc::clone(&events));
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(workers)
        .on_thread_start(move || {
            let guard = tracing::dispatcher::set_default(&dispatch);
            INSTALLED.with(|installed| *installed.borrow_mut() = Some(guard));
        })
        .enable_all()
        .build()
        .expect("a runtime");
    let listener = runtime
        .block_on(tokio::net::TcpListener::bind("127.0.0.1:0"))
        .expect("bind a local port");
    let port = listener.local_addr().expect("local address").port();
    let (signals, received) = tokio::sync::mpsc::channel(2);
    let (set_phase, phase) = phase_channel();
    let (arrived_tx, arrived) = mpsc::channel();
    let report_arrival = axum::middleware::from_fn(move |request: Request, next: Next| {
        arrived_tx.send(()).ok();
        next.run(request)
    });
    let exit = runtime.spawn(serve_until(
        listener,
        crate::routes(state.clone(), phase).layer(report_arrival),
        set_phase,
        received,
        bounds,
        Arc::new(AtomicUsize::new(0)),
    ));
    Server {
        port,
        state,
        script,
        entered,
        arrived,
        events,
        signals,
        exit: Some(exit),
        runtime: Some(runtime),
        _snapshots: snapshots,
    }
}

impl Server {
    /// Send one request. The reply is read by [`reply_of`].
    fn send(&self, method: &str, path: &str, headers: &str, body: &str) -> TcpStream {
        let mut client = TcpStream::connect(("127.0.0.1", self.port)).expect("connect");
        client
            .set_read_timeout(Some(WAIT))
            .expect("set the read timeout");
        let content_type = if body.is_empty() {
            ""
        } else {
            "Content-Type: application/json\r\n"
        };
        let head = format!(
            "{method} {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n\
             {content_type}{headers}Content-Length: {}\r\n\r\n{body}",
            body.len()
        );
        client.write_all(head.as_bytes()).expect("send");
        client
    }

    /// Send the request of `route` for `model`.
    fn send_to(&self, route: Route, model: &str, headers: &str) -> TcpStream {
        let (path, body) = route.request(model);
        self.send("POST", &path, headers, &body)
    }

    /// Wait until a load started, and return its model id.
    fn load_started(&self) -> String {
        self.entered
            .recv_timeout(WAIT)
            .expect("no request reached the load")
    }

    /// Wait until one more request reached the routes: a shutdown that
    /// starts after this finds the request in flight.
    fn request_arrived(&self) {
        self.arrived
            .recv_timeout(WAIT)
            .expect("the request did not reach the routes");
    }

    /// The span fields of the event `message`, which the server emitted
    /// once.
    fn span_of(&self, message: &str) -> Vec<(String, String)> {
        let events = self.events.events.lock();
        let mut found = events.iter().filter(|(text, _)| text == message);
        let (_, span) = found
            .next()
            .unwrap_or_else(|| panic!("no event `{message}`: {events:?}"));
        assert!(found.next().is_none(), "`{message}` twice: {events:?}");
        span.clone()
    }

    /// Call `ensure_loaded` for `m0` on the runtime of this server. The task
    /// reports on `pending` when its first poll did not end the call, and
    /// returns the cold flag.
    fn call(&self, pending: &mpsc::Sender<()>) -> tokio::task::JoinHandle<rmlx_core::Result<bool>> {
        let (state, pending) = (self.state.clone(), pending.clone());
        let runtime = self.runtime.as_ref().expect("the runtime still runs");
        runtime.spawn(async move {
            let mut call = std::pin::pin!(state.ensure_loaded("m0"));
            let loaded = match futures::poll!(&mut call) {
                std::task::Poll::Ready(loaded) => loaded,
                std::task::Poll::Pending => {
                    pending.send(()).ok();
                    call.await
                }
            };
            loaded.map(|(_generator, cold)| cold)
        })
    }

    /// The cold flag that `call` returned.
    fn cold_of(&self, call: tokio::task::JoinHandle<rmlx_core::Result<bool>>) -> bool {
        let runtime = self.runtime.as_ref().expect("the runtime still runs");
        let cold = runtime.block_on(async { tokio::time::timeout(WAIT, call).await });
        let cold = cold.expect("the call did not end");
        cold.expect("the call does not panic")
            .expect("the load does not fail")
    }

    fn resident(&self) -> Vec<String> {
        let slots = self.state.slots.read();
        slots.iter().map(|m| m.id.clone()).collect()
    }

    /// Send the shutdown signal.
    fn signal(&self) {
        self.signals
            .try_send(())
            .expect("the server reads its signals");
    }

    /// Wait for the serve future. Call it only after a reply showed that the
    /// runtime has a free worker.
    fn exit(&mut self) -> Option<ExitDeadline> {
        let exit = self.exit.take().expect("one exit per server");
        let runtime = self.runtime.as_ref().expect("the runtime still runs");
        let exit = runtime
            .block_on(async { tokio::time::timeout(WAIT, exit).await })
            .ok()?;
        let exit = exit.expect("the serve task does not panic");
        Some(exit.expect("the shutdown is not an error"))
    }
}

/// A route that loads the model of its request.
#[derive(Clone, Copy, Debug)]
enum Route {
    Load,
    Chat,
    Messages,
}

const ROUTES: [Route; 3] = [Route::Load, Route::Chat, Route::Messages];

impl Route {
    /// The error type of the 503 that a failed load gets.
    fn load_failure_type(self) -> &'static str {
        match self {
            Self::Load | Self::Chat => "service_unavailable",
            Self::Messages => "service_unavailable_error",
        }
    }

    /// The path and the JSON body of a request for `model`.
    fn request(self, model: &str) -> (String, String) {
        let messages = r#"[{"role":"user","content":"hi"}]"#;
        match self {
            Self::Load => (format!("/v1/models/{model}/load"), String::new()),
            Self::Chat => (
                "/v1/chat/completions".to_owned(),
                format!(r#"{{"model":"{model}","messages":{messages}}}"#),
            ),
            Self::Messages => (
                "/v1/messages".to_owned(),
                format!(r#"{{"model":"{model}","max_tokens":4,"messages":{messages}}}"#),
            ),
        }
    }
}

/// The whole reply. `None` when it did not end inside `WAIT`.
fn reply_of(mut client: TcpStream) -> Option<String> {
    let mut reply = String::new();
    client.read_to_string(&mut reply).ok()?;
    Some(reply)
}

fn body_of(reply: &str) -> serde_json::Value {
    let body = reply.split("\r\n\r\n").nth(1).expect("a body");
    serde_json::from_str(body).expect("a JSON body")
}

fn error_body(kind: &str, message: &str) -> serde_json::Value {
    serde_json::json!({ "error": { "message": message, "type": kind } })
}

/// While a request of each route is inside a load, the one async worker
/// answers a light request.
#[test]
fn a_light_request_is_answered_during_a_load() {
    for route in ROUTES {
        let server = start(1, 1, FAR);
        let loading = server.send_to(route, "m0", "");
        assert_eq!(server.load_started(), "m0", "{route:?}");

        let health = reply_of(server.send("GET", "/health", "", ""));

        let health = health.unwrap_or_else(|| panic!("{route:?}: no reply during the load"));
        assert!(health.starts_with("HTTP/1.1 200"), "{route:?}: {health}");
        server.script.release(1);
        assert!(reply_of(loading).is_some(), "{route:?}: no reply at all");
    }
}

/// The request timeout answers 408 to the request whose model loads. The load
/// runs on: the next request waits for it and uses that model.
#[test]
fn the_request_timeout_answers_408_during_a_load() {
    for route in ROUTES {
        let server = start(1, 1, FAR);
        let loading = server.send_to(route, "m0", "X-Request-Timeout-Seconds: 1\r\n");
        assert_eq!(server.load_started(), "m0", "{route:?}");

        let reply = reply_of(loading).unwrap_or_else(|| panic!("{route:?}: no reply"));

        assert!(reply.starts_with("HTTP/1.1 408"), "{route:?}: {reply}");
        assert_eq!(server.resident(), [""; 0], "{route:?}: resident too early");
        let again = server.send_to(Route::Load, "m0", "");
        assert!(
            server.entered.recv_timeout(SETTLE).is_err(),
            "{route:?}: a second load started after the 408"
        );
        server.script.release(1);
        let again = reply_of(again).expect("the next request got no reply");
        assert!(again.starts_with("HTTP/1.1 200"), "{route:?}: {again}");
        assert_eq!(server.resident(), ["m0"], "{route:?}");
        assert_eq!(server.script.calls(), 1, "{route:?}: a second load");
    }
}

/// The request timeout answers 408 to a request that waits for the load of
/// another request.
#[test]
fn the_request_timeout_answers_408_to_a_request_that_waits_for_a_load() {
    let server = start(1, 1, FAR);
    let loading = server.send_to(Route::Load, "m0", "");
    assert_eq!(server.load_started(), "m0");

    let waiting = server.send_to(Route::Chat, "m0", "X-Request-Timeout-Seconds: 1\r\n");

    let reply = reply_of(waiting).expect("the request that waits got no reply");
    assert!(reply.starts_with("HTTP/1.1 408"), "{reply}");
    server.script.release(1);
    let loaded = reply_of(loading).expect("the request that loads got no reply");
    assert!(loaded.starts_with("HTTP/1.1 200"), "{loaded}");
    assert_eq!(server.script.calls(), 1, "a second load");
}

/// At the reply bound of a shutdown, the request whose model loads and a
/// request that waits for that load each get the 503 of the shutdown. The
/// serve future returns with the load still in progress.
#[test]
fn a_shutdown_answers_503_to_the_requests_of_a_load() {
    for route in ROUTES {
        let mut server = start(1, 1, NEAR);
        let loading = server.send_to(route, "m0", "");
        server.request_arrived();
        assert_eq!(server.load_started(), "m0", "{route:?}");
        let waiting = server.send_to(route, "m0", "");
        server.request_arrived();

        server.signal();

        for (name, client) in [("loads", loading), ("waits", waiting)] {
            let reply = reply_of(client)
                .unwrap_or_else(|| panic!("{route:?}: the request that {name} got no reply"));
            assert!(
                reply.starts_with("HTTP/1.1 503"),
                "{route:?} {name}: {reply}"
            );
            assert_eq!(
                body_of(&reply),
                error_body("service_unavailable", "the server is shutting down"),
                "{route:?} {name}"
            );
        }
        assert!(
            server.exit().is_some(),
            "{route:?}: the serve future did not return"
        );
        assert_eq!(server.script.calls(), 1, "{route:?}: a second load");
    }
}

/// Requests for one model that is not loaded start one load. The others
/// wait, then use the model that this load made resident.
#[test]
fn requests_for_a_model_that_loads_wait_and_use_that_load() {
    for route in ROUTES {
        let server = start(4, 1, FAR);
        let clients = [Route::Load, route, route, route].map(|r| server.send_to(r, "m0", ""));
        assert_eq!(server.load_started(), "m0", "{route:?}");

        assert!(
            server.entered.recv_timeout(SETTLE).is_err(),
            "{route:?}: a second load of the same model started"
        );
        server.script.release(1);

        let replies = clients.map(reply_of);
        assert!(
            replies.iter().all(Option::is_some),
            "{route:?}: {replies:?}"
        );
        assert_eq!(server.script.calls(), 1, "{route:?}: loads");
        assert_eq!(server.resident(), ["m0"], "{route:?}");
    }
}

/// Two models, one slot: the load of the second model starts when the load
/// of the first ended, and it evicts the first.
#[test]
fn a_load_after_a_load_evicts_by_the_slot_rule() {
    let server = start(4, 2, FAR);
    let clients = ["m0", "m1"].map(|model| server.send_to(Route::Load, model, ""));
    let first = server.load_started();
    assert!(
        server.entered.recv_timeout(SETTLE).is_err(),
        "two loads at one time"
    );

    server.script.release(1);

    let second = server.load_started();
    assert_ne!(first, second, "a model loaded twice");
    server.script.release(1);
    for reply in clients.map(reply_of) {
        let reply = reply.expect("a load request got no reply");
        assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    }
    assert_eq!(server.resident(), [second]);
    assert_eq!(server.script.calls(), 2);
}

/// Two slots: while a second model loads, a request for the resident model
/// is answered.
#[test]
fn a_request_for_a_resident_model_does_not_wait_for_a_load() {
    let server = start_with_slots(4, 2, 2, FAR);
    server.script.release(1);
    let resident = reply_of(server.send_to(Route::Load, "m0", "")).expect("a reply");
    assert!(resident.starts_with("HTTP/1.1 200"), "{resident}");
    assert_eq!(server.load_started(), "m0");
    let loading = server.send_to(Route::Load, "m1", "");
    assert_eq!(server.load_started(), "m1");

    let reply = reply_of(server.send_to(Route::Load, "m0", ""));

    let reply = reply.expect("the request for the resident model got no reply");
    assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
    assert_eq!(server.resident(), ["m0"], "the second load ended early");
    server.script.release(1);
    assert!(reply_of(loading).is_some(), "the load request got no reply");
    assert_eq!(server.script.calls(), 2);
}

/// A client that closes its connection during the load leaves no half state:
/// the load ends, the model is resident once, and the next request uses it.
#[test]
fn a_client_that_leaves_during_a_load_leaves_one_resident_model() {
    for route in ROUTES {
        let server = start(1, 1, FAR);
        let gone = server.send_to(route, "m0", "");
        assert_eq!(server.load_started(), "m0", "{route:?}");

        drop(gone);

        let waits = server.send_to(Route::Load, "m0", "");
        assert!(
            server.entered.recv_timeout(SETTLE).is_err(),
            "{route:?}: a second load started when the client left"
        );
        server.script.release(1);
        let reply = reply_of(waits).expect("the next request got no reply");
        assert!(reply.starts_with("HTTP/1.1 200"), "{route:?}: {reply}");
        assert_eq!(server.resident(), ["m0"], "{route:?}");
        assert_eq!(server.script.calls(), 1, "{route:?}: a second load");
    }
}

/// A failed load answers 503 with the error type of its route and the text
/// of the error, leaves no model resident, and the next request
/// loads again.
#[test]
fn a_failed_load_keeps_its_reply() {
    let message = rmlx_core::Error::Model(FAILURE.to_owned()).to_string();
    for route in ROUTES {
        let server = start(1, 1, FAR);
        *server.script.outcome.lock() = Outcome::Fails;
        server.script.release(1);

        let reply = reply_of(server.send_to(route, "m0", "")).expect("a reply");

        assert!(reply.starts_with("HTTP/1.1 503"), "{route:?}: {reply}");
        assert_eq!(
            body_of(&reply),
            error_body(route.load_failure_type(), &message),
            "{route:?}"
        );
        assert_eq!(server.resident(), [""; 0], "{route:?}");
        *server.script.outcome.lock() = Outcome::Loads;
        server.script.release(1);
        let again = reply_of(server.send_to(Route::Load, "m0", "")).expect("a reply");
        assert!(again.starts_with("HTTP/1.1 200"), "{route:?}: {again}");
        assert_eq!(server.script.calls(), 2, "{route:?}");
    }
}

/// A load that panics answers 500, as a handler panic does, and the next
/// request loads.
#[test]
fn a_load_that_panics_answers_500() {
    for route in ROUTES {
        let server = start(1, 1, FAR);
        *server.script.outcome.lock() = Outcome::Panics;
        server.script.release(1);

        let reply = reply_of(server.send_to(route, "m0", "")).expect("a reply");

        assert!(reply.starts_with("HTTP/1.1 500"), "{route:?}: {reply}");
        *server.script.outcome.lock() = Outcome::Loads;
        server.script.release(1);
        let again = reply_of(server.send_to(Route::Load, "m0", "")).expect("a reply");
        assert!(again.starts_with("HTTP/1.1 200"), "{route:?}: {again}");
        assert_eq!(server.resident(), ["m0"], "{route:?}");
    }
}

/// A call that found no resident model is cold: the call that loaded, and
/// the call that waited for that load. A call that found the model resident
/// is warm.
#[test]
fn a_call_that_found_no_resident_model_is_cold() {
    let server = start(4, 1, FAR);
    let (pending_tx, pending) = mpsc::channel();

    let loads = server.call(&pending_tx);
    assert_eq!(server.load_started(), "m0");
    pending.recv_timeout(WAIT).expect("the first call ended");
    let waits = server.call(&pending_tx);
    pending
        .recv_timeout(WAIT)
        .expect("the second call did not wait for the load");
    server.script.release(1);

    assert!(server.cold_of(loads), "the call that loaded");
    assert!(server.cold_of(waits), "the call that waited");
    assert!(
        !server.cold_of(server.call(&pending_tx)),
        "the call that found the model resident"
    );
    assert!(pending.try_recv().is_err(), "the third call waited");
    assert_eq!(server.script.calls(), 1);
}

/// The load events of a chat or messages request are inside the span of
/// that request: they carry its request id.
#[test]
fn the_load_events_of_a_request_carry_its_span() {
    for route in [Route::Chat, Route::Messages] {
        for (outcome, events) in [
            (
                Outcome::Loads,
                ["slots: loading model", "slots: model loaded"],
            ),
            (
                Outcome::Fails,
                ["slots: loading model", "slots: model load failed"],
            ),
        ] {
            let server = start(1, 1, FAR);
            *server.script.outcome.lock() = outcome;
            server.script.release(1);
            let id = format!("load-span-{route:?}");

            let reply = server.send_to(route, "m0", &format!("X-Request-Id: {id}\r\n"));

            assert!(reply_of(reply).is_some(), "{route:?}: no reply");
            for event in events {
                let span = server.span_of(event);
                assert!(
                    span.contains(&("request_id".to_owned(), id.clone())),
                    "{route:?}: `{event}` is in the span {span:?}"
                );
            }
        }
    }
}

/// A load that the blocking pool refuses, because the runtime has started to
/// end, is a load error that names the model. The loader does not run.
#[test]
fn a_load_that_the_runtime_did_not_start_is_an_error() {
    let (registry, _snapshots) = snapshot_registry(1, &[]);
    let (loads_tx, loads) = mpsc::channel();
    let mut state = slot_test_state(registry, 1);
    state.loader = Arc::new(move |_path, id| {
        loads_tx.send(id.to_owned()).ok();
        Ok(Box::new(NotReadyGenerator) as Box<dyn Generator>)
    });
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .build()
        .expect("a runtime");
    let handle = runtime.handle().clone();
    runtime.shutdown_background();
    let (ended_tx, ended) = mpsc::channel();

    let caller = std::thread::spawn(move || {
        let loaded = handle.block_on(state.ensure_loaded("m0"));
        let loaded = loaded.map(|_| ()).map_err(|e| e.to_string());
        ended_tx.send(loaded).ok();
    });

    let ended = ended.recv_timeout(WAIT).expect("the call did not end");
    caller.join().expect("the caller does not panic");
    let error = ended.expect_err("a load that did not run is no success");
    assert!(
        error.starts_with("model: the load of model 'm0' did not run"),
        "{error}"
    );
    assert!(loads.try_recv().is_err(), "the loader ran");
}
