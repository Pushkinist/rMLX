//! The shutdown rule against scripted routes and scripted blocking jobs (no
//! model, no GPU, no real signal: a signal is a message the test sends).
//! Every scripted route is under the layers the server puts on its routes.
//!
//! Every wait has a bound, so a server that does not stop fails its test and
//! does not hang the binary. A scripted job ends when its test drops the
//! channel it waits on.
//!
//! Covered by a process, not here: the real signals, the bounds `serve`
//! passes, and the routes of the server under the layer. The test
//! `serve_shutdown` of `rmlx-cli` starts the binary for them. The Metal claim
//! of a shutdown is in `claim_tests.rs`.
//!
//! Not covered by any test: the exit of the process after `block_on_serve`
//! returns with a job still on the blocking pool. No test can leave a job
//! that evaluates on the GPU at the exit. Its proof is a run of a real server
//! with a model: a chat request in flight, one SIGTERM, then the exit code,
//! the events of `docs/SERVER_SHUTDOWN.md` and a restart with no refusal.

use std::convert::Infallible;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::routing::{get, post};
use axum::Router;
use futures::StreamExt;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::task::JoinHandle;

use super::{
    block_on_serve_within, phase_channel, serve_until, Bounds, ExitDeadline, FailedAfterSignal,
    Phase, PhaseWatch,
};
use crate::openai::tests::slot_test_state;
use crate::openai::AppState;
use crate::registry::ModelRegistry;

/// The bound on every wait of a test.
const WAIT: Duration = Duration::from_secs(5);

/// Shutdown bounds that no test reaches: what ends inside them ended for
/// another reason.
const FAR: Bounds = Bounds {
    reply: Duration::from_secs(120),
    exit: Duration::from_secs(180),
};

/// Shutdown bounds that a test reaches while its scripted job still runs.
const NEAR: Bounds = Bounds {
    reply: Duration::from_millis(100),
    exit: Duration::from_millis(200),
};

/// A server under `serve_until` and the channel that sends its signals.
struct Served {
    port: u16,
    signals: tokio::sync::mpsc::Sender<()>,
    phase: PhaseWatch,
    exit: JoinHandle<anyhow::Result<ExitDeadline>>,
}

impl Served {
    async fn signal(&self) {
        self.signals.send(()).await.expect("the server still runs");
    }

    /// Send a signal and wait until the server has taken it as its first.
    async fn signal_and_drain(&self) {
        self.signal().await;
        tokio::time::timeout(WAIT, self.phase.clone().reached(Phase::Draining))
            .await
            .expect("the shutdown did not start");
    }

    /// Wait for the serve future. `None` when it did not return inside `WAIT`.
    async fn exit(self) -> Option<ExitDeadline> {
        let exit = tokio::time::timeout(WAIT, self.exit).await.ok()?;
        let exit = exit.expect("the serve task does not panic");
        Some(exit.expect("the shutdown is not an error"))
    }
}

/// Serve `api` under the layers of the server.
async fn start(api: Router<AppState>, bounds: Bounds) -> Served {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a local port");
    let port = listener.local_addr().expect("local address").port();
    let (signals, received) = tokio::sync::mpsc::channel(2);
    let (set_phase, phase) = phase_channel();
    let state = slot_test_state(ModelRegistry::from_paths(&[]), 1);
    let exit = tokio::spawn(serve_until(
        listener,
        crate::with_layers(api, state, phase.clone()),
        set_phase,
        received,
        bounds,
        Arc::new(AtomicUsize::new(0)),
    ));
    Served {
        port,
        signals,
        phase,
        exit,
    }
}

async fn connect(port: u16) -> tokio::net::TcpStream {
    tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect")
}

/// The head of a request with no body, which asks the server to close the
/// connection after its reply.
fn head_of(path: &str, headers: &str) -> String {
    format!(
        "POST {path} HTTP/1.1\r\nHost: test\r\nConnection: close\r\n{headers}Content-Length: 0\r\n\r\n"
    )
}

/// Send one request and return the whole reply. `None` when the reply did not
/// end inside `WAIT`.
fn request(port: u16, path: &str, headers: &str) -> JoinHandle<Option<String>> {
    let head = head_of(path, headers);
    tokio::spawn(async move {
        let mut client = connect(port).await;
        client.write_all(head.as_bytes()).await.expect("send");
        let mut reply = String::new();
        tokio::time::timeout(WAIT, client.read_to_string(&mut reply))
            .await
            .ok()?
            .ok()?;
        Some(reply)
    })
}

async fn reply_of(reply: JoinHandle<Option<String>>) -> String {
    let reply = reply.await.expect("the client does not panic");
    reply.expect("the request got no whole reply")
}

/// The reply is the 503 of the shutdown, in the error envelope of the server.
fn assert_shutdown_503(reply: &str) {
    assert!(reply.starts_with("HTTP/1.1 503"), "{reply}");
    let body = reply.split("\r\n\r\n").nth(1).expect("a body");
    let body: serde_json::Value = serde_json::from_str(body).expect("a JSON body");
    assert_eq!(
        body,
        serde_json::json!({ "error": {
            "message": "the server is shutting down",
            "type": "service_unavailable",
        }}),
    );
}

/// Wait on a std channel from async code without blocking the runtime.
async fn report<T: Send + 'static>(rx: mpsc::Receiver<T>) -> Result<T, mpsc::RecvTimeoutError> {
    tokio::task::spawn_blocking(move || rx.recv_timeout(WAIT))
        .await
        .expect("the report task does not panic")
}

/// Sets its flag when it is dropped.
struct SetOnDrop(Arc<AtomicBool>);

impl Drop for SetOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

/// A scripted blocking job behind `POST /job`. It runs until the test sends on
/// `release` or drops it.
struct Job {
    api: Router<AppState>,
    started: mpsc::Receiver<()>,
    release: mpsc::Sender<()>,
    /// Whether the job stopped on the drop of its request, sent when the job
    /// returns.
    stopped_on_drop: mpsc::Receiver<bool>,
}

/// The reply of the scripted route.
const JOB_REPLY: &str = "job reply";

/// A job as a transcription runs it when `reads_the_drop`: it decodes
/// "windows" of 1 ms and stops when its request is dropped. Else a job as a
/// chat decode or a speech synthesis runs: it does not read the drop.
fn job(reads_the_drop: bool) -> Job {
    let (started_tx, started) = mpsc::channel();
    let (release, release_rx) = mpsc::channel::<()>();
    let release_rx = Arc::new(parking_lot::Mutex::new(Some(release_rx)));
    let (stopped_tx, stopped_on_drop) = mpsc::channel();
    let route = post(move || {
        let (started_tx, stopped_tx) = (started_tx.clone(), stopped_tx.clone());
        let release_rx = release_rx.lock().take().expect("one request for each job");
        let dropped = Arc::new(AtomicBool::new(false));
        let request = SetOnDrop(Arc::clone(&dropped));
        async move {
            let _request = request;
            tokio::task::spawn_blocking(move || {
                started_tx.send(()).ok();
                let stop = || reads_the_drop && dropped.load(Ordering::Acquire);
                while !stop() && release_rx.try_recv() == Err(mpsc::TryRecvError::Empty) {
                    std::thread::sleep(Duration::from_millis(1));
                }
                stopped_tx.send(stop()).ok();
            })
            .await
            .ok();
            JOB_REPLY
        }
    });
    Job {
        api: Router::new().route("/job", route),
        started,
        release,
        stopped_on_drop,
    }
}

/// A request that ends inside the reply bound gets its whole reply on both
/// job shapes, and its job is not stopped. The serve future returns when the
/// reply is sent.
#[tokio::test(flavor = "multi_thread")]
async fn a_reply_that_ends_inside_the_bound_is_complete() {
    for job in [job(false), job(true)] {
        let served = start(job.api, FAR).await;
        let reply = request(served.port, "/job", "");
        assert_eq!(report(job.started).await, Ok(()), "the job did not start");

        served.signal_and_drain().await;
        job.release.send(()).expect("the job still waits");

        let reply = reply_of(reply).await;
        assert!(reply.starts_with("HTTP/1.1 200"), "{reply}");
        assert!(reply.ends_with(JOB_REPLY), "{reply}");
        assert!(reply.contains("content-length: 9\r\n"), "{reply}");
        assert_eq!(report(job.stopped_on_drop).await, Ok(false), "stopped");
        let _deadline = served
            .exit()
            .await
            .expect("the serve future did not return after the reply");
    }
}

/// A request that is still open at the reply bound gets the 503 of the
/// shutdown on both job shapes, not before the bound, and the serve future
/// returns with the exit deadline. A job that reads the drop stops; a job
/// that does not still runs.
#[tokio::test(flavor = "multi_thread")]
async fn an_open_request_gets_503_and_its_stop_at_the_reply_bound() {
    for reads_the_drop in [false, true] {
        let job = job(reads_the_drop);
        let served = start(job.api, NEAR).await;
        let reply = request(served.port, "/job", "");
        assert_eq!(report(job.started).await, Ok(()), "the job did not start");

        let signalled = Instant::now();
        served.signal().await;

        assert_shutdown_503(&reply_of(reply).await);
        assert!(
            signalled.elapsed() >= NEAR.reply,
            "the 503 came before the reply bound"
        );
        let ExitDeadline(deadline) = served
            .exit()
            .await
            .expect("the serve future did not return after the reply bound");
        assert!(
            deadline >= signalled + NEAR.exit && deadline <= Instant::now() + NEAR.exit,
            "the exit deadline is not the signal plus the exit bound"
        );
        if reads_the_drop {
            assert_eq!(report(job.stopped_on_drop).await, Ok(true), "not stopped");
        } else {
            assert_eq!(
                job.stopped_on_drop.try_recv(),
                Err(mpsc::TryRecvError::Empty),
                "the job ended before the serve future returned"
            );
        }
    }
}

/// A second signal ends the wait for the open requests at once: the request
/// gets its 503, the serve future returns, and the exit deadline is counted
/// from the second signal.
#[tokio::test(flavor = "multi_thread")]
async fn a_second_signal_ends_the_wait_at_once() {
    let job = job(false);
    let served = start(job.api, FAR).await;
    let reply = request(served.port, "/job", "");
    assert_eq!(report(job.started).await, Ok(()), "the job did not start");
    served.signal_and_drain().await;

    served.signal().await;

    assert_shutdown_503(&reply_of(reply).await);
    let ExitDeadline(deadline) = served
        .exit()
        .await
        .expect("the serve future did not return after the second signal");
    assert!(
        deadline <= Instant::now() + FAR.exit.saturating_sub(FAR.reply),
        "the exit deadline still waits for the reply bound"
    );
}

/// Read from `client` until `seen` ends with `end`.
async fn read_until(client: &mut tokio::net::TcpStream, seen: &mut Vec<u8>, end: &[u8]) {
    while !seen.ends_with(end) {
        let read = tokio::time::timeout(WAIT, client.read_buf(seen)).await;
        assert!(
            matches!(read, Ok(Ok(n)) if n > 0),
            "the reply stopped before {:?}: {:?}",
            String::from_utf8_lossy(end),
            String::from_utf8_lossy(seen),
        );
    }
}

/// A reply that already streams ends when the open requests are dropped: the
/// client has its first chunk, then reads the end of a whole chunked body,
/// and the stream that fed the reply is dropped.
#[tokio::test(flavor = "multi_thread")]
async fn a_reply_that_streams_ends_when_the_requests_are_dropped() {
    let feed_dropped = Arc::new(AtomicBool::new(false));
    let route_feed_dropped = Arc::clone(&feed_dropped);
    let route = post(move || {
        let feed = SetOnDrop(Arc::clone(&route_feed_dropped));
        async move {
            let chunks = futures::stream::once(async { Ok::<_, Infallible>("first chunk") })
                .chain(futures::stream::pending())
                .inspect(move |_| {
                    let _held = &feed;
                });
            axum::body::Body::from_stream(chunks)
        }
    });
    let served = start(Router::new().route("/stream", route), NEAR).await;
    let mut client = connect(served.port).await;
    let head = head_of("/stream", "");
    client.write_all(head.as_bytes()).await.expect("send");
    let mut reply = Vec::new();
    read_until(&mut client, &mut reply, b"first chunk\r\n").await;
    assert!(
        !feed_dropped.load(Ordering::Acquire),
        "the feed ended early"
    );

    served.signal().await;

    read_until(&mut client, &mut reply, b"0\r\n\r\n").await;
    assert!(reply.starts_with(b"HTTP/1.1 200"));
    let _deadline = served
        .exit()
        .await
        .expect("the serve future did not return after the stream ended");
    assert!(feed_dropped.load(Ordering::Acquire), "the feed still runs");
}

/// A handler inside a synchronous call on its async worker, as a model load
/// runs, cannot be dropped: its connection stays open after the requests are
/// dropped. The serve future returns at the exit deadline, not before it and
/// not when the call returns.
#[tokio::test(flavor = "multi_thread")]
async fn a_handler_that_cannot_be_dropped_holds_the_exit_to_the_deadline_only() {
    let (entered_tx, entered) = mpsc::channel();
    let (release, release_rx) = mpsc::channel::<()>();
    let release_rx = Arc::new(parking_lot::Mutex::new(Some(release_rx)));
    let route = post(move || {
        let (entered_tx, release_rx) = (entered_tx.clone(), release_rx.lock().take());
        async move {
            entered_tx.send(()).ok();
            if let Some(release_rx) = release_rx {
                release_rx.recv().ok();
            }
            "late reply"
        }
    });
    let served = start(Router::new().route("/load", route), NEAR).await;
    let _reply = request(served.port, "/load", "");
    assert_eq!(report(entered).await, Ok(()), "no request");

    served.signal().await;

    let exit = served.exit().await;
    let returned = Instant::now();
    drop(release);
    let ExitDeadline(deadline) =
        exit.expect("the serve future did not return with a connection still open");
    assert!(
        returned >= deadline,
        "the serve future returned before the exit deadline"
    );
}

/// With no request in flight the serve future returns at the signal, with
/// the exit deadline of the first signal. An idle keep-alive connection does
/// not hold it.
#[tokio::test(flavor = "multi_thread")]
async fn a_shutdown_with_nothing_in_flight_returns_at_the_signal() {
    let api = Router::new().route("/ping", get(|| async { "pong" }));
    let served = start(api, FAR).await;
    let mut idle = connect(served.port).await;
    idle.write_all(b"GET /ping HTTP/1.1\r\nHost: test\r\n\r\n")
        .await
        .expect("send");
    read_until(&mut idle, &mut Vec::new(), b"pong").await;

    let signalled = Instant::now();
    served.signal().await;

    let ExitDeadline(deadline) = served
        .exit()
        .await
        .expect("the serve future did not return with nothing in flight");
    assert!(
        deadline >= signalled + FAR.exit && deadline <= Instant::now() + FAR.exit,
        "the exit deadline is not the signal plus the exit bound"
    );
}

/// The shutdown layer leaves the request timeout as it is: with no shutdown,
/// the 408 answers and its drop stops the job.
#[tokio::test(flavor = "multi_thread")]
async fn the_request_timeout_still_answers_408_under_the_layers() {
    let job = job(true);
    let served = start(job.api, FAR).await;

    let reply = request(served.port, "/job", "X-Request-Timeout-Seconds: 1\r\n");

    let reply = reply_of(reply).await;
    assert!(reply.starts_with("HTTP/1.1 408"), "{reply}");
    assert_eq!(report(job.stopped_on_drop).await, Ok(true), "not stopped");
    drop((served, job.release));
}

/// The two bounds of the server are the ones `docs/SERVER_SHUTDOWN.md` states.
#[test]
fn the_shutdown_doc_states_the_bounds_of_the_server() {
    let doc = include_str!("../../../docs/SERVER_SHUTDOWN.md");
    let (reply, exit) = (Bounds::SERVER.reply, Bounds::SERVER.exit);
    for sentence in [
        format!("has {} s to send its reply", reply.as_secs()),
        format!("within {} s of the first signal", exit.as_secs()),
        format!(
            "{} s after the second signal",
            exit.saturating_sub(reply).as_secs()
        ),
    ] {
        assert!(doc.contains(&sentence), "the doc does not say: {sentence}");
    }
}

/// The message of every `info!` and `warn!` call in `source`: the last string
/// literal of the call.
fn event_messages(source: &str) -> Vec<String> {
    let mut messages = Vec::new();
    for (start, _) in source
        .match_indices("info!(")
        .chain(source.match_indices("warn!("))
    {
        let (mut depth, mut literal, mut last) = (0_usize, None::<String>, None);
        let mut chars = source[start..].chars();
        while let Some(c) = chars.next() {
            match (&mut literal, c) {
                (Some(text), '\\') => text.extend(chars.next()),
                (Some(_), '"') => last = literal.take(),
                (Some(text), _) => text.push(c),
                (None, '"') => literal = Some(String::new()),
                (None, '(') => depth += 1,
                (None, ')') if depth == 1 => break,
                (None, ')') => depth -= 1,
                (None, _) => {}
            }
        }
        messages.push(last.expect("an event has a message"));
    }
    messages
}

/// Every event of the shutdown is a row of the log table of
/// `docs/SERVER_SHUTDOWN.md`, and the table has no other row.
#[test]
fn the_shutdown_doc_lists_the_events_of_the_shutdown() {
    let doc = include_str!("../../../docs/SERVER_SHUTDOWN.md");
    let server = include_str!("lib.rs");
    let signals = &server[server.find("fn shutdown_signals").expect("the signal fn")..];
    let signals = &signals[..signals.find("\n}\n").expect("the end of the signal fn")];
    let mut events = event_messages(include_str!("shutdown.rs"));
    events.extend(event_messages(signals));
    assert_eq!(events.len(), 12, "{events:?}");

    let rows: Vec<&str> = doc
        .lines()
        .filter(|line| line.starts_with("| `info` |") || line.starts_with("| `warn` |"))
        .collect();
    for event in &events {
        let cell = format!("| `{event}` |");
        assert_eq!(
            rows.iter().filter(|row| row.contains(&cell)).count(),
            1,
            "the log table does not have one row for: {event}"
        );
    }
    assert_eq!(
        rows.len(),
        events.len(),
        "the log table has a row for no event"
    );
}

/// A tokio runtime as `rmlx serve` builds it, small.
pub(crate) fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .expect("build a runtime")
}

/// A blocking job for a runtime that ends: it runs until the test sends on
/// `release` or drops it, and no longer than twice `WAIT`, so a runtime that
/// waits for it fails its test and does not hang the binary.
pub(crate) struct HeldJob {
    pub(crate) run: Box<dyn FnOnce() + Send>,
    pub(crate) started: mpsc::Receiver<()>,
    pub(crate) release: mpsc::Sender<()>,
    pub(crate) finished: Arc<AtomicBool>,
}

pub(crate) fn held_job() -> HeldJob {
    let (started_tx, started) = mpsc::channel();
    let (release, release_rx) = mpsc::channel::<()>();
    let finished = Arc::new(AtomicBool::new(false));
    let job_finished = Arc::clone(&finished);
    let run = Box::new(move || {
        started_tx.send(()).ok();
        release_rx.recv_timeout(WAIT + WAIT).ok();
        job_finished.store(true, Ordering::Release);
    });
    HeldJob {
        run,
        started,
        release,
        finished,
    }
}

/// A serve future that starts `job` on the blocking pool, waits until it runs
/// and returns `outcome`.
pub(crate) async fn serving(
    job: Box<dyn FnOnce() + Send>,
    started: mpsc::Receiver<()>,
    outcome: impl FnOnce() -> anyhow::Result<ExitDeadline>,
) -> anyhow::Result<ExitDeadline> {
    tokio::task::spawn_blocking(job);
    report(started).await?;
    outcome()
}

/// Run `end` on its own thread and return its result. `None` when it did not
/// return inside `WAIT`.
fn within_the_wait<T: Send + 'static>(end: impl FnOnce() -> T + Send + 'static) -> Option<T> {
    let (returned_tx, returned) = mpsc::channel();
    std::thread::spawn(move || returned_tx.send(end()).ok());
    returned.recv_timeout(WAIT).ok()
}

/// A blocking job that cannot stop does not hold the exit past the deadline:
/// `block_on_serve` returns, and the job still runs.
#[test]
fn the_runtime_ends_at_the_exit_deadline_while_a_job_runs() {
    let job = held_job();
    let (run, started) = (job.run, job.started);

    let returned = within_the_wait(move || {
        let deadline = || Ok(ExitDeadline(Instant::now() + NEAR.exit));
        block_on_serve_within(runtime(), None, serving(run, started, deadline), FAR.exit)
    });

    let job_ran_on = !job.finished.load(Ordering::Acquire);
    drop(job.release);
    returned
        .expect("block_on_serve did not return while the job ran")
        .expect("the shutdown is not an error");
    assert!(job_ran_on, "the job ended before block_on_serve returned");
}

/// A serve future that fails returns no deadline. The runtime still ends in a
/// bounded time while a job runs, and the error reaches the caller.
#[test]
fn a_serve_error_ends_the_runtime_in_a_bounded_time() {
    let job = held_job();
    let (run, started) = (job.run, job.started);

    let returned = within_the_wait(move || {
        let failed = || Err(anyhow::anyhow!("scripted serve error"));
        block_on_serve_within(runtime(), None, serving(run, started, failed), NEAR.exit)
    });

    let job_ran_on = !job.finished.load(Ordering::Acquire);
    drop(job.release);
    let error = returned
        .expect("block_on_serve did not return while the job ran")
        .expect_err("the serve error is returned");
    assert_eq!(error.to_string(), "scripted serve error");
    assert!(job_ran_on, "the job ended before block_on_serve returned");
}

/// A serve error after the first signal keeps the exit deadline of that
/// signal: the failure does not start a new bound.
#[test]
fn a_serve_error_after_the_signal_keeps_the_deadline_of_the_signal() {
    let job = held_job();
    let (run, started) = (job.run, job.started);

    let returned = within_the_wait(move || {
        let failed = || {
            Err(anyhow::Error::new(FailedAfterSignal {
                exit_at: Instant::now() + NEAR.exit,
                source: std::io::Error::other("scripted accept error"),
            }))
        };
        block_on_serve_within(runtime(), None, serving(run, started, failed), FAR.exit)
    });

    let job_ran_on = !job.finished.load(Ordering::Acquire);
    drop(job.release);
    let error = returned
        .expect("block_on_serve did not return at the deadline of the signal")
        .expect_err("the serve error is returned");
    assert_eq!(error.to_string(), "serve: scripted accept error");
    assert!(job_ran_on, "the job ended before block_on_serve returned");
}

/// A serve future that panics does not leave the runtime to its drop, which
/// waits for every blocking job: the runtime ends in a bounded time while a
/// job runs, and the panic goes on to the caller.
#[test]
fn a_serve_panic_ends_the_runtime_in_a_bounded_time() {
    let job = held_job();
    let (run, started) = (job.run, job.started);

    let returned = within_the_wait(move || {
        let panics = || panic!("scripted serve panic");
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            block_on_serve_within(runtime(), None, serving(run, started, panics), NEAR.exit)
        }))
    });

    let job_ran_on = !job.finished.load(Ordering::Acquire);
    drop(job.release);
    let panic = returned
        .expect("block_on_serve did not return while the job ran")
        .expect_err("the panic reaches the caller");
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"scripted serve panic"));
    assert!(job_ran_on, "the job ended before block_on_serve returned");
}

/// A blocking job that stops inside the deadline is awaited: it is not cut
/// by the exit.
#[test]
fn the_runtime_waits_for_a_job_that_stops_inside_the_deadline() {
    let (started_tx, started) = mpsc::channel();
    let (stop, stop_rx) = mpsc::channel::<()>();
    let finished = Arc::new(AtomicBool::new(false));
    let job_finished = Arc::clone(&finished);
    let run = Box::new(move || {
        started_tx.send(()).ok();
        stop_rx.recv_timeout(WAIT).ok();
        std::thread::sleep(Duration::from_millis(50));
        job_finished.store(true, Ordering::Release);
    });

    let returned = within_the_wait(move || {
        let deadline = move || {
            stop.send(()).ok();
            Ok(ExitDeadline(Instant::now() + FAR.exit))
        };
        block_on_serve_within(runtime(), None, serving(run, started, deadline), FAR.exit)
    });

    returned
        .expect("block_on_serve did not return after the job stopped")
        .expect("the shutdown is not an error");
    assert!(finished.load(Ordering::Acquire), "the exit cut the job");
}
