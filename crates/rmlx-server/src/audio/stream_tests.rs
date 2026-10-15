//! The streamed reply of a transcription, with scripted jobs in place of a
//! Whisper walk (no model, no GPU): the event bytes, the wait for the job to be
//! ready, a failure after the reply started, the window sink, and the stop of a
//! job whose reader is gone.

use std::sync::atomic::Ordering;
use std::sync::mpsc as std_mpsc;

use axum::http::{header, StatusCode};
use axum::response::Response;
use futures::StreamExt;
use rmlx_audio::wav::WavError;
use rmlx_audio::whisper::WhisperError;
use tokio::sync::{mpsc, oneshot};

use super::{send_delta, sse_event, stream_reply, StreamEvent, STREAM_EVENTS};
use crate::audio::test_events::{events_of, field};
use crate::audio::{failed, job_refusal, JobError};
use crate::engine::run_admitted;
use crate::engine::tests::{admitted, gpu_queue, report, wait_for_flag, REPORT_WAIT};

/// The reply of a scripted job that reports `ready` and then sends `events`.
async fn reply_of(ready: Result<(), JobError>, events: Vec<StreamEvent>) -> Response {
    let (ready_tx, ready_rx) = oneshot::channel();
    let (events_tx, events_rx) = mpsc::channel(STREAM_EVENTS);
    let job = run_admitted(None, move |_cancel| {
        let _ = ready_tx.send(ready);
        for event in events {
            let _ = events_tx.blocking_send(event);
        }
    });
    stream_reply(ready_rx, events_rx, job).await
}

/// The whole body of `response` as text.
async fn body_text(response: Response) -> String {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    String::from_utf8(bytes.to_vec()).expect("the body is UTF-8")
}

fn content_type(response: &Response) -> &str {
    response
        .headers()
        .get(header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or_default()
}

/// A streamed reply is server-sent events: one `data:` line for each delta,
/// then the done event with the whole text, and nothing after it. No line has
/// an `event:` name and no `[DONE]` line follows.
#[tokio::test(flavor = "multi_thread")]
async fn a_streamed_reply_sends_each_delta_then_the_done_event() {
    let response = reply_of(
        Ok(()),
        vec![
            StreamEvent::Delta("The first sentence.".to_owned()),
            StreamEvent::Delta(" The second one.".to_owned()),
            StreamEvent::Done("The first sentence. The second one.".to_owned()),
        ],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(content_type(&response), "text/event-stream");
    assert_eq!(
        body_text(response).await,
        "data: {\"type\":\"transcript.text.delta\",\"delta\":\"The first sentence.\"}\n\n\
         data: {\"type\":\"transcript.text.delta\",\"delta\":\" The second one.\"}\n\n\
         data: {\"type\":\"transcript.text.done\",\"text\":\"The first sentence. The second one.\"}\n\n"
    );
}

/// A failure before the first window, such as a file the decoder refuses, is
/// the JSON error reply a request without `stream` gets, not a stream.
#[tokio::test(flavor = "multi_thread")]
async fn a_failure_before_the_job_is_ready_is_the_error_reply() {
    let response = reply_of(Err(JobError::Decode(WavError::Empty)), Vec::new()).await;
    assert_eq!(response.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(content_type(&response), "application/json");
    let body: serde_json::Value = serde_json::from_str(&body_text(response).await).expect("JSON");
    assert_eq!(body["error"]["param"], "file");
    assert_eq!(body["error"]["type"], "invalid_request_error");
}

/// A failure after the reply started is the error event of the chat routes,
/// with the message and the type of the error reply, and then the stream ends.
#[tokio::test(flavor = "multi_thread")]
async fn a_failure_after_the_reply_started_is_an_error_event_that_ends_the_stream() {
    let response = reply_of(
        Ok(()),
        vec![
            StreamEvent::Delta("One.".to_owned()),
            StreamEvent::Failed(job_refusal(&JobError::Transcribe(WhisperError::Config(
                "x".to_owned(),
            )))),
        ],
    )
    .await;
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(
        body_text(response).await,
        "data: {\"type\":\"transcript.text.delta\",\"delta\":\"One.\"}\n\n\
         data: {\"error\":{\"message\":\"transcribe: config parse error: x\",\"type\":\"service_unavailable\"}}\n\n"
    );
}

/// A job that panics before it is ready is a 500 in the error envelope.
#[tokio::test(flavor = "multi_thread")]
async fn a_job_that_panics_before_it_is_ready_is_a_500() {
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), JobError>>();
    let (_events_tx, events_rx) = mpsc::channel(STREAM_EVENTS);
    let job = run_admitted(None, move |_cancel| {
        let _ready = ready_tx;
        panic!("scripted panic");
    });
    let response = stream_reply(ready_rx, events_rx, job).await;
    assert_eq!(response.status(), StatusCode::INTERNAL_SERVER_ERROR);
}

/// A client that drops the streamed reply after its first event: the job sees
/// the flag, and the permit is free when the job returns.
#[tokio::test(flavor = "multi_thread")]
async fn dropping_the_streamed_reply_sets_the_flag_and_frees_the_permit() {
    let (queue, pending) = gpu_queue();
    let guard = admitted(&queue, &pending).await;
    let (ready_tx, ready_rx) = oneshot::channel();
    let (events_tx, events_rx) = mpsc::channel(STREAM_EVENTS);
    let (saw_tx, saw_rx) = std_mpsc::channel();
    let job = run_admitted(Some(guard), move |cancel| {
        let _ = ready_tx.send(Ok(()));
        let _ = events_tx.blocking_send(StreamEvent::Delta("One.".to_owned()));
        saw_tx.send(wait_for_flag(cancel)).ok();
    });
    let response = stream_reply(ready_rx, events_rx, job).await;
    let mut body = response.into_body().into_data_stream();
    let first = body.next().await.expect("a first event").expect("bytes");
    assert!(
        first.starts_with(b"data: {\"type\":\"transcript.text.delta\",\"delta\":\"One.\"}"),
        "{first:?}"
    );
    assert_eq!(queue.available_permits(), 0, "the job holds the permit");

    drop(body);
    assert_eq!(
        report(saw_rx).await,
        Ok(true),
        "the job did not see the flag"
    );
    let permit = tokio::time::timeout(REPORT_WAIT, queue.acquire()).await;
    assert!(
        matches!(permit, Ok(Ok(_))),
        "the permit is not free after the job returned"
    );
    assert_eq!(pending.load(Ordering::Acquire), 0);
}

/// The sink sends a delta when the reader is there.
#[test]
fn a_delta_is_sent_while_the_reader_is_there() {
    let (tx, mut rx) = mpsc::channel(STREAM_EVENTS);
    assert!(send_delta(&tx, "One.".to_owned()).is_ok());
    assert!(matches!(rx.try_recv(), Ok(StreamEvent::Delta(delta)) if delta == "One."));
}

/// A send that fails, because the reader is gone, is the stop: the walk opens
/// no further window.
#[test]
fn a_failed_send_is_the_stop() {
    let (tx, rx) = mpsc::channel(STREAM_EVENTS);
    drop(rx);
    assert!(matches!(
        send_delta(&tx, "One.".to_owned()),
        Err(rmlx_core::Error::Cancelled)
    ));
}

/// A failure after the reply started is logged once, by the job, in the span
/// of the request: with the status the client got, 200, and `streamed`, not
/// with the status of the error reply it did not get.
#[test]
fn a_failure_after_the_start_is_logged_with_the_status_the_client_got() {
    let events = events_of(|| {
        let span = tracing::info_span!("audio_transcriptions", request = "scripted");
        let _entered = span.enter();
        let refusal = failed(
            &JobError::Transcribe(WhisperError::Config("x".to_owned())),
            true,
        );
        assert_eq!(refusal.status, StatusCode::SERVICE_UNAVAILABLE);
    });
    assert_eq!(events.len(), 1, "{events:?}");
    let event = &events[0];
    assert_eq!(field(event, "level"), Some("WARN"));
    assert_eq!(field(event, "status"), Some("200"));
    assert_eq!(field(event, "streamed"), Some("true"));
    assert_eq!(field(event, "error_type"), Some("\"service_unavailable\""));
    assert_eq!(field(event, "span.request"), Some("\"scripted\""));
}

/// The error event of a failure only formats the refusal: the job logged it,
/// so the body stream, which runs outside the span of the request, logs
/// nothing.
#[test]
fn the_error_event_of_a_failure_logs_nothing() {
    let refusal = job_refusal(&JobError::Transcribe(WhisperError::Config("x".to_owned())));
    let events = events_of(|| {
        let _ = sse_event(StreamEvent::Failed(refusal));
    });
    assert_eq!(events, Vec::<Vec<(String, String)>>::new());
}
