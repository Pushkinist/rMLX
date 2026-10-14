//! The shutdown rule: what a SIGINT or SIGTERM stops, and how long the exit
//! can take. It is the same for every route.
//!
//! 1. At the first signal the server accepts no new connection
//!    ([`Phase::Draining`]). Every request in flight has [`REPLY_BOUND`] to
//!    send its reply.
//! 2. At the reply bound, or at a second signal, every request that is still
//!    open is dropped ([`Phase::Stopping`]). A request with no response yet
//!    gets 503; a response that streams ends there, with no terminal item.
//! 3. The time between the two bounds is for the 503 to reach its client and
//!    for a blocking job that reads the drop of its request to return. A
//!    transcription walk reads it, before each window. A chat decode and a
//!    speech synthesis are not known to return in that time.
//! 4. At [`EXIT_BOUND`], or that same time after a second signal, the process
//!    exits. A job that still runs ends with it, and the Metal claim stays
//!    held until the process is gone.
//!
//! The layer cannot drop a handler that is inside a synchronous call on its
//! async worker: such a request gets no 503.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::Context as _;
use axum::body::{Body, HttpBody};
use axum::extract::{Request, State};
use axum::http::{Method, Uri};
use axum::middleware::Next;
use axum::response::Response;
use axum::serve::ListenerExt;
use axum::Router;
use futures::StreamExt;
use tokio::sync::{mpsc, watch};
use tracing::{info, warn};

use crate::claim::MetalClaim;
use crate::openai::errors::service_unavailable;

/// How long the open requests have, from the first signal, to send their
/// reply.
const REPLY_BOUND: Duration = Duration::from_secs(8);

/// How long the process has, from the first signal, to exit. It is below the
/// 20 s that `launchd` gives a service before SIGKILL.
const EXIT_BOUND: Duration = Duration::from_secs(10);

/// The two bounds of one shutdown, each counted from the first signal.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bounds {
    pub(crate) reply: Duration,
    pub(crate) exit: Duration,
}

impl Bounds {
    pub(crate) const SERVER: Self = Self {
        reply: REPLY_BOUND,
        exit: EXIT_BOUND,
    };
}

/// The instant at which a server that got its shutdown signal must be gone.
/// [`block_on_serve`] ends the runtime at it.
#[must_use = "a runtime that is dropped waits for every blocking job, with no bound"]
#[derive(Clone, Copy, Debug)]
pub struct ExitDeadline(pub(crate) Instant);

/// Where the server is in its shutdown. The order is the order in time.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Phase {
    Serving,
    Draining,
    Stopping,
}

/// The accept loop failed after the first signal. It keeps the exit deadline
/// of that signal, so the failure does not start a new bound.
#[derive(Debug, thiserror::Error)]
#[error("serve: {source}")]
pub(crate) struct FailedAfterSignal {
    pub(crate) exit_at: Instant,
    #[source]
    pub(crate) source: std::io::Error,
}

/// Reads the [`Phase`] that `serve_until` sets.
#[derive(Clone, Debug)]
pub(crate) struct PhaseWatch(watch::Receiver<Phase>);

impl PhaseWatch {
    /// For a router that no signal reaches: it stays at [`Phase::Serving`].
    pub(crate) fn never() -> Self {
        phase_channel().1
    }

    /// Resolves when the server is at `phase` or after it.
    // cancel-safe: it only waits on the channel.
    pub(crate) async fn reached(mut self, phase: Phase) {
        if self.0.wait_for(|now| *now >= phase).await.is_err() {
            std::future::pending::<()>().await;
        }
    }
}

pub(crate) fn phase_channel() -> (watch::Sender<Phase>, PhaseWatch) {
    let (set, watch) = watch::channel(Phase::Serving);
    (set, PhaseWatch(watch))
}

/// The layer of every route. At [`Phase::Stopping`] it drops the request and
/// answers 503, or ends the body of a response that streams.
// cancel-safe: a drop of this future drops the handler, as the layer does.
pub(crate) async fn stop_at_shutdown(
    State(phase): State<PhaseWatch>,
    req: Request,
    next: Next,
) -> Response {
    let (method, uri) = (req.method().clone(), req.uri().clone());
    tokio::select! {
        response = next.run(req) => end_stream_at_stop(response, phase, method, uri),
        () = phase.clone().reached(Phase::Stopping) => {
            warn!(%method, path = uri.path(), "request answered 503 by the server shutdown");
            service_unavailable("the server is shutting down")
        }
    }
}

/// A body of a known length is sent as it is. A body that streams ends at
/// [`Phase::Stopping`], and the stream that fed it is dropped.
fn end_stream_at_stop(response: Response, phase: PhaseWatch, method: Method, uri: Uri) -> Response {
    if response.body().size_hint().exact().is_some() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let cut = Box::pin(async move {
        phase.reached(Phase::Stopping).await;
        warn!(
            %method,
            path = uri.path(),
            "streaming reply ended by the server shutdown; it can lack its terminal item"
        );
    });
    let body = Body::from_stream(body.into_data_stream().take_until(cut));
    Response::from_parts(parts, body)
}

/// Serve `router` until the first of `signals`, then follow the rule of this
/// module. The router must carry [`stop_at_shutdown`] on the watch of
/// `phase`. `gpu_permit_requests` is the gauge of the requests that hold or
/// wait for the GPU permit, read for the log only; an embeddings request is
/// not in it.
///
/// # Errors
/// The accept loop failed, or it ended with no signal.
// cancel-safe: a drop stops the accept loop; the connections that are open
// run on until the runtime ends.
pub(crate) async fn serve_until(
    listener: tokio::net::TcpListener,
    router: Router,
    phase: watch::Sender<Phase>,
    mut signals: mpsc::Receiver<()>,
    bounds: Bounds,
    gpu_permit_requests: Arc<AtomicUsize>,
) -> anyhow::Result<ExitDeadline> {
    // axum 0.8 removed implicit TCP_NODELAY. With Nagle's algorithm enabled by
    // default, small SSE frames (one chunked-encoding event per token, ~100
    // bytes) interact with macOS delayed-ACK and add per-token
    // server→client latency. Setting TCP_NODELAY on every accepted socket is
    // what oMLX/mlx-lm/uvicorn already do via Python's default.
    let listener = listener.tap_io(|tcp_stream| {
        if let Err(err) = tcp_stream.set_nodelay(true) {
            tracing::trace!("failed to set TCP_NODELAY on incoming connection: {err:#}");
        }
    });

    let draining = PhaseWatch(phase.subscribe()).reached(Phase::Draining);
    let mut server = std::pin::pin!(std::future::IntoFuture::into_future(
        axum::serve(listener, router).with_graceful_shutdown(draining)
    ));
    tokio::select! {
        ended = &mut server => {
            ended.context("serve")?;
            return Err(anyhow::anyhow!("serve: ended with no shutdown signal"));
        }
        Some(()) = signals.recv() => {}
    }
    drain_and_stop(server, &phase, signals, bounds, &gpu_permit_requests).await
}

/// The shutdown from its first signal: steps 1 and 2 of the rule.
// cancel-safe: as `serve_until`; the phase stays where the drop found it.
async fn drain_and_stop(
    mut server: std::pin::Pin<&mut impl Future<Output = std::io::Result<()>>>,
    phase: &watch::Sender<Phase>,
    mut signals: mpsc::Receiver<()>,
    bounds: Bounds,
    gpu_permit_requests: &AtomicUsize,
) -> anyhow::Result<ExitDeadline> {
    let signalled = Instant::now();
    let exit_at = signalled + bounds.exit;
    info!(
        gpu_permit_requests = gpu_permit_requests.load(Ordering::Acquire),
        reply_bound_ms = bounds.reply.as_millis() as u64,
        exit_bound_ms = bounds.exit.as_millis() as u64,
        "shutdown started"
    );
    phase.send_replace(Phase::Draining);

    let failed = |source| FailedAfterSignal { exit_at, source };
    let Some(reason) = end_of_the_reply_wait(server.as_mut(), &mut signals, bounds.reply)
        .await
        .map_err(failed)?
    else {
        return Ok(ExitDeadline(exit_at));
    };

    warn!(
        reason,
        gpu_permit_requests = gpu_permit_requests.load(Ordering::Acquire),
        since_signal_ms = signalled.elapsed().as_millis() as u64,
        "the open requests are dropped"
    );
    phase.send_replace(Phase::Stopping);
    let exit_at = exit_at.min(Instant::now() + bounds.exit.saturating_sub(bounds.reply));

    match tokio::time::timeout_at(exit_at.into(), server).await {
        Ok(ended) => ended.map_err(|source| FailedAfterSignal { exit_at, source })?,
        Err(_elapsed) => warn!(
            since_signal_ms = signalled.elapsed().as_millis() as u64,
            "exit deadline reached with a connection still open"
        ),
    }
    Ok(ExitDeadline(exit_at))
}

/// Why the wait for the replies ended with a request still open. `None` when
/// every connection closed first.
// cancel-safe: it only polls the server and waits on a timer and a channel.
async fn end_of_the_reply_wait(
    server: std::pin::Pin<&mut impl Future<Output = std::io::Result<()>>>,
    signals: &mut mpsc::Receiver<()>,
    reply: Duration,
) -> std::io::Result<Option<&'static str>> {
    tokio::select! {
        ended = server => {
            ended?;
            info!("every connection closed");
            Ok(None)
        }
        () = tokio::time::sleep(reply) => Ok(Some("reply bound reached")),
        Some(()) = signals.recv() => Ok(Some("second signal")),
    }
}

/// Run `serving` on `rt`, then end the runtime at the deadline `serving`
/// returned: the blocking jobs are awaited until then and no longer. When the
/// runtime does not end in time, `claim` stays held until the process is gone,
/// because a job can still use the GPU.
///
/// A `serving` that fails or panics returns no deadline. The runtime then ends
/// within the exit bound, counted from the first signal when the failure knows
/// it and from the failure otherwise.
///
/// # Errors
/// The error of `serving`.
///
/// # Panics
/// With the panic of `serving`, after the runtime ended.
pub fn block_on_serve<F>(
    rt: tokio::runtime::Runtime,
    claim: Option<&MetalClaim>,
    serving: F,
) -> anyhow::Result<()>
where
    F: Future<Output = anyhow::Result<ExitDeadline>>,
{
    block_on_serve_within(rt, claim, serving, EXIT_BOUND)
}

/// [`block_on_serve`], with the bound of a runtime whose `serving` returned no
/// deadline.
pub(crate) fn block_on_serve_within<F>(
    rt: tokio::runtime::Runtime,
    claim: Option<&MetalClaim>,
    serving: F,
    exit: Duration,
) -> anyhow::Result<()>
where
    F: Future<Output = anyhow::Result<ExitDeadline>>,
{
    let outcome = std::panic::catch_unwind(AssertUnwindSafe(|| rt.block_on(serving)));
    let deadline = match &outcome {
        Ok(Ok(ExitDeadline(deadline))) => *deadline,
        Ok(Err(e)) => e
            .downcast_ref::<FailedAfterSignal>()
            .map_or_else(|| Instant::now() + exit, |failed| failed.exit_at),
        Err(_panic) => Instant::now() + exit,
    };
    end_runtime(rt, deadline, claim);
    match outcome {
        Ok(served) => served.map(|_deadline| ()),
        Err(panic) => std::panic::resume_unwind(panic),
    }
}

/// End `rt` at `deadline`: its blocking jobs are awaited until then. When it
/// does not end in time, `claim` stays held until the process is gone.
fn end_runtime(rt: tokio::runtime::Runtime, deadline: Instant, claim: Option<&MetalClaim>) {
    let left = deadline.saturating_duration_since(Instant::now());
    let started = Instant::now();
    rt.shutdown_timeout(left);
    let waited = started.elapsed();
    let (left_ms, waited_ms) = (left.as_millis() as u64, waited.as_millis() as u64);
    if waited < left {
        info!(left_ms, waited_ms, "runtime ended before the exit deadline");
        return;
    }
    warn!(
        left_ms,
        waited_ms, "exit deadline reached before the runtime ended; a blocking job can still run"
    );
    if let Some(claim) = claim {
        hold_claim_until_exit(claim);
    }
}

fn hold_claim_until_exit(claim: &MetalClaim) {
    match claim.hold_until_process_exit() {
        Ok(()) => info!("Metal claim stays held until the process is gone"),
        Err(e) => warn!(error = %e, "Metal claim could not be held past its guard"),
    }
}

#[cfg(test)]
#[path = "shutdown_tests.rs"]
pub(crate) mod shutdown_tests;
