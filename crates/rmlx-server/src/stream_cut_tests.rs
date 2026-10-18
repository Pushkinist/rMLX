//! The cut of a streamed reply, without a socket: the body is held or dropped
//! by the test, as hyper holds it for a client that reads nothing or drops it
//! for a client that left.

use std::io;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, BodyDataStream, Bytes};
use axum::response::Response;
use futures::{Stream, StreamExt};

use super::end_stream_at;

/// Sets its flag when it is dropped, after a pause: a body that ends before
/// the drop is over shows a clear flag.
struct SetOnDrop(Arc<AtomicBool>);

impl Drop for SetOnDrop {
    fn drop(&mut self) {
        std::thread::sleep(Duration::from_millis(100));
        self.0.store(true, Ordering::Release);
    }
}

type Item = io::Result<&'static str>;

/// A response whose body is `items` and then `rest`. The flag is set when the
/// stream that feeds it is dropped.
fn response_of(
    items: Vec<Item>,
    rest: impl Stream<Item = Item> + Send + 'static,
    feed_dropped: &Arc<AtomicBool>,
) -> Response {
    let held = SetOnDrop(Arc::clone(feed_dropped));
    let feed = futures::stream::iter(items).chain(rest).inspect(move |_| {
        let _held = &held;
    });
    Response::new(Body::from_stream(feed))
}

/// One chunk, and then nothing for ever.
fn pending_response(feed_dropped: &Arc<AtomicBool>) -> Response {
    response_of(
        vec![Ok("first chunk")],
        futures::stream::pending(),
        feed_dropped,
    )
}

/// A cut that resolves when the test sends on the returned sender, and counts
/// the times it ran to its end.
fn counted_cut() -> (
    tokio::sync::oneshot::Sender<()>,
    Arc<AtomicUsize>,
    impl std::future::Future<Output = ()> + Send + 'static,
) {
    let (at_cut, cut) = tokio::sync::oneshot::channel::<()>();
    let cuts = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&cuts);
    let cut = async move {
        cut.await.ok();
        counted.fetch_add(1, Ordering::SeqCst);
    };
    (at_cut, cuts, cut)
}

async fn set_within(flag: &AtomicBool, bound: Duration) -> bool {
    let start = tokio::time::Instant::now();
    while !flag.load(Ordering::Acquire) {
        if start.elapsed() >= bound {
            return false;
        }
        tokio::time::sleep(Duration::from_millis(5)).await;
    }
    true
}

/// The next chunk of `body`, which must come in a bounded time.
async fn next_of(body: &mut BodyDataStream) -> Option<Result<Bytes, axum::Error>> {
    tokio::time::timeout(Duration::from_secs(2), body.next())
        .await
        .expect("the body gave no chunk and no end")
}

/// At the cut the feed is dropped although nobody polls the body.
#[tokio::test(flavor = "multi_thread")]
async fn the_cut_drops_the_feed_of_a_body_that_nobody_polls() {
    let feed_dropped = Arc::new(AtomicBool::new(false));
    let (at_cut, cuts, cut) = counted_cut();
    let response = end_stream_at(pending_response(&feed_dropped), cut);
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !feed_dropped.load(Ordering::Acquire),
        "the feed ended early"
    );

    at_cut.send(()).expect("the cut still waits");

    assert!(
        set_within(&feed_dropped, Duration::from_secs(2)).await,
        "the feed still runs"
    );
    assert_eq!(cuts.load(Ordering::SeqCst), 1);
    drop(response);
}

/// A body that is dropped drops its feed before the cut.
#[tokio::test(flavor = "multi_thread")]
async fn a_dropped_body_drops_the_feed_before_the_cut() {
    let feed_dropped = Arc::new(AtomicBool::new(false));
    let response = end_stream_at(
        pending_response(&feed_dropped),
        std::future::pending::<()>(),
    );
    tokio::time::sleep(Duration::from_millis(50)).await;
    assert!(
        !feed_dropped.load(Ordering::Acquire),
        "the feed ended early"
    );

    drop(response);

    assert!(
        set_within(&feed_dropped, Duration::from_secs(2)).await,
        "the feed still runs"
    );
}

/// The reader gets the chunks that came before the cut, then the end, and the
/// feed is gone by then.
#[tokio::test(flavor = "multi_thread")]
async fn the_reader_gets_the_chunks_before_the_cut_and_then_the_end() {
    let feed_dropped = Arc::new(AtomicBool::new(false));
    let (at_cut, cuts, cut) = counted_cut();
    let response = end_stream_at(pending_response(&feed_dropped), cut);
    let mut body = response.into_body().into_data_stream();
    let first = next_of(&mut body).await.expect("a chunk").expect("bytes");
    assert_eq!(&first[..], b"first chunk");

    at_cut.send(()).expect("the cut still waits");

    assert!(next_of(&mut body).await.is_none(), "the body did not end");
    assert!(
        feed_dropped.load(Ordering::Acquire),
        "the body ended before the feed was gone"
    );
    assert_eq!(cuts.load(Ordering::SeqCst), 1);
}

/// A feed that ends by itself ends the body with all its chunks, and its cut
/// never runs to its end.
#[tokio::test(flavor = "multi_thread")]
async fn a_feed_that_ends_by_itself_ends_the_body_whole() {
    let feed_dropped = Arc::new(AtomicBool::new(false));
    let (_at_cut, cuts, cut) = counted_cut();
    let response = response_of(
        vec![Ok("one"), Ok("two")],
        futures::stream::empty(),
        &feed_dropped,
    );
    let mut body = end_stream_at(response, cut).into_body().into_data_stream();

    assert_eq!(
        &next_of(&mut body).await.expect("a chunk").expect("bytes")[..],
        b"one"
    );
    assert_eq!(
        &next_of(&mut body).await.expect("a chunk").expect("bytes")[..],
        b"two"
    );
    assert!(next_of(&mut body).await.is_none(), "the body did not end");
    assert!(
        feed_dropped.load(Ordering::Acquire),
        "the body ended before the feed was gone"
    );
    assert_eq!(cuts.load(Ordering::SeqCst), 0);
}

/// A chunk that is an error reaches the reader, and the body ends after it
/// although the feed has more to give.
#[tokio::test(flavor = "multi_thread")]
async fn an_error_chunk_reaches_the_reader_and_ends_the_body() {
    let feed_dropped = Arc::new(AtomicBool::new(false));
    let response = response_of(
        vec![Ok("one"), Err(io::Error::other("scripted failure"))],
        futures::stream::pending(),
        &feed_dropped,
    );
    let mut body = end_stream_at(response, std::future::pending::<()>())
        .into_body()
        .into_data_stream();

    assert_eq!(
        &next_of(&mut body).await.expect("a chunk").expect("bytes")[..],
        b"one"
    );
    assert!(next_of(&mut body).await.expect("an item").is_err());
    assert!(next_of(&mut body).await.is_none(), "the body did not end");
    assert!(
        feed_dropped.load(Ordering::Acquire),
        "the body ended before the feed was gone"
    );
}

/// A feed that panics aborts the reply with an error. A clean end would pass
/// for a whole answer.
#[tokio::test(flavor = "multi_thread")]
#[allow(
    clippy::panic,
    reason = "the feed of this test panics on purpose, as a defective handler does"
)]
async fn a_feed_that_panics_aborts_the_reply() {
    let feed =
        futures::stream::once(async { Ok::<_, io::Error>("one") }).chain(futures::stream::poll_fn(
            |_| -> std::task::Poll<Option<_>> { panic!("scripted panic of the feed") },
        ));
    let response = Response::new(Body::from_stream(feed));
    let mut body = end_stream_at(response, std::future::pending::<()>())
        .into_body()
        .into_data_stream();

    assert_eq!(
        &next_of(&mut body).await.expect("a chunk").expect("bytes")[..],
        b"one"
    );
    assert!(
        next_of(&mut body).await.expect("an item").is_err(),
        "the panic did not reach the reader"
    );
    assert!(next_of(&mut body).await.is_none(), "the body did not end");
}
