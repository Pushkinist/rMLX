//! Tower `Layer` + `Service` that wraps every HTTP handler
//! future with [`std::panic::catch_unwind`].
//!
//! # Why
//!
//! A panic in an async handler does not unwind cleanly through the Tokio
//! executor — depending on whether the code runs inline or in a spawned task,
//! the panic either propagates to the runtime (crashing the worker) or
//! terminates the process. Either way the caller receives a connection drop
//! rather than an HTTP 500.
//!
//! Wrapping each handler future with `catch_unwind` converts the panic into a
//! synthetic `500 Internal Server Error` response, emits a structured
//! `tracing::error!` event, and lets the worker continue serving subsequent
//! requests — the Cloudflare lesson (fail-fast; convert panics to 500).
//!
//! # Safety of `AssertUnwindSafe`
//!
//! Wrapping an axum handler future in `AssertUnwindSafe` is sound here because:
//! - Axum handler futures are one-shot: driven to completion (or dropped on
//!   panic) and never re-polled after a panic occurs.
//! - All `AppState` shared across requests uses `Arc`-wrapped interior
//!   mutability (`parking_lot::Mutex/RwLock`). Lock guards released by RAII
//!   even when a future is dropped mid-execution, so shared state is not
//!   corrupted by a per-request future being abandoned.
//! - We never reuse the panicking future; `catch_unwind` drops it and we
//!   construct a fresh 500 response.
//!
//! # Placement
//!
//! Applied once at the outermost router layer in [`crate::build_router`] so
//! every route — `/health`, `/v1/chat/completions`, `/v1/embeddings`,
//! `/v1/messages`, metrics — is protected without per-route opt-in.

use std::future::Future;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::task::{Context, Poll};

use axum::body::Body;
use axum::http::{Method, Request, Response, StatusCode, Uri};
use futures::FutureExt as _;
use tower::Layer;
use tower::Service;
use tracing::error;

// ── Layer ─────────────────────────────────────────────────────────────────────

/// Tower [`Layer`] that wraps every inner service call in `catch_unwind`.
///
/// Install on the router with `.layer(CatchUnwindLayer::new())`.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub struct CatchUnwindLayer;

impl CatchUnwindLayer {
    /// Create a new [`CatchUnwindLayer`].
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl Default for CatchUnwindLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> Layer<S> for CatchUnwindLayer {
    type Service = CatchUnwindService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        CatchUnwindService { inner }
    }
}

// ── Service ───────────────────────────────────────────────────────────────────

/// Tower [`Service`] produced by [`CatchUnwindLayer`].
#[derive(Clone, Debug)]
pub struct CatchUnwindService<S> {
    inner: S,
}

impl<S, B> Service<Request<B>> for CatchUnwindService<S>
where
    S: Service<Request<B>, Response = Response<Body>> + Send + 'static,
    S::Future: Send + 'static,
    S::Error: Send + 'static,
    B: Send + 'static,
{
    type Response = Response<Body>;
    type Error = S::Error;
    type Future = CatchUnwindFuture<S::Error>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: Request<B>) -> Self::Future {
        // Capture URI + method for the error log before the request is consumed.
        let method = req.method().clone();
        let uri = req.uri().clone();

        // SAFETY: see module-level doc — AssertUnwindSafe is sound here.
        // Box the future so CatchUnwindFuture is Unpin without adding the
        // pin-project crate dependency.
        let inner_fut = self.inner.call(req);
        let catch_fut = Box::pin(AssertUnwindSafe(inner_fut).catch_unwind());

        CatchUnwindFuture {
            inner: catch_fut,
            method,
            uri,
        }
    }
}

// ── Future ────────────────────────────────────────────────────────────────────

type CatchResult<E> = Result<Result<Response<Body>, E>, Box<dyn std::any::Any + Send>>;

/// Future returned by [`CatchUnwindService::call`].
///
/// Polls the boxed `catch_unwind` future; converts panics into `500` responses.
pub struct CatchUnwindFuture<E> {
    inner: Pin<Box<dyn Future<Output = CatchResult<E>> + Send>>,
    method: Method,
    uri: Uri,
}

impl<E> std::fmt::Debug for CatchUnwindFuture<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CatchUnwindFuture")
            .field("method", &self.method)
            .field("uri", &self.uri)
            .finish_non_exhaustive()
    }
}

// cancel-safe: dropping this future mid-poll drops the boxed inner future, which
// triggers RAII cleanup (parking_lot guards, Arc decrements) without additional
// side-effect. No partial writes occur — response is only emitted via Poll::Ready.
// Safe to cancel from timeout middleware or tokio::select!.
impl<E> Future for CatchUnwindFuture<E> {
    type Output = Result<Response<Body>, E>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.inner.as_mut().poll(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(Ok(result)) => Poll::Ready(result),
            Poll::Ready(Err(panic_payload)) => {
                // Extract a human-readable payload string without moving
                // the Box (downcast_ref borrows it).
                let msg = panic_payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| panic_payload.downcast_ref::<String>().map(String::as_str))
                    .unwrap_or("(non-string panic payload)");

                error!(
                    target: "panic",
                    method = %self.method,
                    uri = %self.uri,
                    payload = msg,
                    "handler panic caught by CatchUnwindLayer — returning 500",
                );

                // Synthetic 500: generic body, no internal detail leaked.
                //
                // Response::builder() with a known-good StatusCode constant and
                // a fixed &str body is structurally infallible (the builder only
                // fails on invalid status codes or header values, neither of
                // which applies here). The unwrap_or_else branch is unreachable
                // in practice but avoids an expect() that would re-panic inside
                // the already-recovering code path.
                let resp = Response::builder()
                    .status(StatusCode::INTERNAL_SERVER_ERROR)
                    .header("content-type", "application/json")
                    .header("x-content-type-options", "nosniff")
                    .body(Body::from(
                        r#"{"error":{"type":"internal_error","message":"internal server error"}}"#,
                    ))
                    .unwrap_or_else(|_| Response::new(Body::from("internal server error")));

                Poll::Ready(Ok(resp))
            }
        }
    }
}
