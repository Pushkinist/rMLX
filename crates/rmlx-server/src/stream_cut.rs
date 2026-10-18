//! The one cut of a streamed reply, for the request time cap and for the
//! shutdown.

use std::future::Future;

use axum::body::{Body, Bytes, HttpBody};
use axum::response::Response;
use futures::{Stream, StreamExt};
use tokio::sync::mpsc;

type Chunk = Result<Bytes, axum::Error>;

/// Send the body of `response` until `cut` resolves. At `cut` the stream that
/// feeds the body is dropped and the body ends, whether or not the client
/// reads. A body of a known length is sent as it is. Only the data frames of
/// a streamed body are sent; its trailers are dropped.
///
/// The feed runs on a task of its own: hyper polls a body only while the
/// socket takes bytes, so a cut that waits for that poll never fires for a
/// client that reads nothing. A feed that panics aborts the reply with an
/// error, as a body with no cut does.
pub(crate) fn end_stream_at(
    response: Response,
    cut: impl Future<Output = ()> + Send + 'static,
) -> Response {
    if response.body().size_hint().exact().is_some() {
        return response;
    }
    let (parts, body) = response.into_parts();
    let (tx, rx) = mpsc::channel(1);
    let feed = tokio::spawn(forward(body.into_data_stream(), tx, cut));
    let chunks = futures::stream::unfold(Some((rx, feed)), |state| async move {
        let (mut rx, feed) = state?;
        if let Some(chunk) = rx.recv().await {
            return Some((chunk, Some((rx, feed))));
        }
        // The end of the channel is the end of the task, which has dropped
        // the feed by then. Only a panic of the feed is an error.
        match feed.await {
            Err(panicked) if panicked.is_panic() => Some((Err(axum::Error::new(panicked)), None)),
            _ => None,
        }
    });
    Response::from_parts(parts, Body::from_stream(chunks))
}

/// Move the chunks of `feed` to `tx` until the feed ends, a chunk is an error,
/// `cut` resolves or the reader of `tx` is gone.
// cancel-safe: a drop drops the feed.
async fn forward(
    mut feed: impl Stream<Item = Chunk> + Unpin,
    tx: mpsc::Sender<Chunk>,
    cut: impl Future<Output = ()>,
) {
    tokio::pin!(cut);
    loop {
        let chunk = tokio::select! {
            biased;
            () = &mut cut => break,
            () = tx.closed() => break,
            next = feed.next() => match next {
                Some(chunk) => chunk,
                None => break,
            },
        };
        let failed = chunk.is_err();
        tokio::select! {
            biased;
            () = &mut cut => break,
            sent = tx.send(chunk) => if sent.is_err() || failed { break },
        }
    }
}

#[cfg(test)]
#[path = "stream_cut_tests.rs"]
mod stream_cut_tests;
