//! The streamed reply of a transcription: server-sent events, one
//! `transcript.text.delta` for each closed window that adds text, then one
//! `transcript.text.done` with the whole text.

use std::convert::Infallible;
use std::future::Future;
use std::sync::atomic::AtomicBool;

use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use futures::StreamExt;
use rmlx_audio::transcribe::{Segment, TextDeltas};
use rmlx_audio::whisper::WhisperError;
use tokio::sync::{mpsc, oneshot};

use super::{failed, panicked, prepare, refused, JobError, JobInput, Refusal};
use crate::engine::job_stream;
use crate::openai::errors::internal_error;
use crate::openai::streaming::error_event;

/// The most events a streamed transcription holds before its job waits for
/// the client to read.
pub(super) const STREAM_EVENTS: usize = 4;

/// What a streamed transcription sends after its reply started.
#[derive(Debug)]
pub(super) enum StreamEvent {
    /// The text of a closed window: [`TextDeltas::delta`].
    Delta(String),
    /// The text of the whole transcription: the last event.
    Done(String),
    /// The walk failed, with the refusal the job logged: the last event.
    Failed(Refusal),
}

/// The job of a streamed transcription. It reports on `ready` whether the
/// reply can start; then it sends the text of each closed window and the
/// whole text on `events`. A stopped walk sends nothing more: the walk logs
/// the stop.
pub(super) fn stream_job(
    input: JobInput,
    cancel: &AtomicBool,
    ready: oneshot::Sender<Result<(), JobError>>,
    events: &mpsc::Sender<StreamEvent>,
) {
    let job = match prepare(input, cancel) {
        Ok(job) => job,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let _ = ready.send(Ok(()));
    let last = match job.run(cancel, &mut delta_sink(events)) {
        Ok(transcription) => StreamEvent::Done(transcription.text),
        Err(JobError::Transcribe(WhisperError::Cancelled { .. })) => return,
        Err(e) => StreamEvent::Failed(failed(&e, true)),
    };
    let _ = events.blocking_send(last);
}

/// The window sink of a streamed transcription: it sends the text delta of
/// each window that adds text.
fn delta_sink(
    events: &mpsc::Sender<StreamEvent>,
) -> impl FnMut(&[Segment]) -> rmlx_core::Result<()> + '_ {
    let mut deltas = TextDeltas::default();
    move |window| {
        deltas
            .delta(window)
            .map_or(Ok(()), |delta| send_delta(events, delta))
    }
}

/// Send `delta`. A send that fails means the reader is gone: that is the stop,
/// and the walk opens no further window.
fn send_delta(events: &mpsc::Sender<StreamEvent>, delta: String) -> rmlx_core::Result<()> {
    events
        .blocking_send(StreamEvent::Delta(delta))
        .map_err(|_| rmlx_core::Error::Cancelled)
}

/// The reply of a streamed transcription. It waits for `ready`, so a request
/// that fails before its first window is the same error reply as without
/// `stream`; then it sends each event of the job as server-sent events.
// cancel-safe: a drop drops `job`, which sets the stop flag of the job; the job
// frees the GPU permit when it returns.
pub(super) async fn stream_reply(
    ready: oneshot::Receiver<Result<(), JobError>>,
    events: mpsc::Receiver<StreamEvent>,
    job: impl Future<Output = Result<(), tokio::task::JoinError>> + Send + 'static,
) -> Response {
    match ready.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => return refused(failed(&e, false)),
        // The job dropped `ready` without a report: it panicked.
        Err(_) => {
            return match job.await {
                Err(e) => panicked(&e),
                Ok(()) => internal_error("internal error"),
            }
        }
    }
    let events = job_stream(events, job).map(|event| Ok::<_, Infallible>(sse_event(event)));
    Sse::new(events).into_response()
}

/// The server-sent event of `event`: a `data:` line with a JSON object that
/// names its `type`, as OpenAI's streamed transcription does, or the error
/// event of the chat routes.
fn sse_event(event: StreamEvent) -> Event {
    let body = match event {
        StreamEvent::Delta(delta) => {
            serde_json::json!({"type": "transcript.text.delta", "delta": delta})
        }
        StreamEvent::Done(text) => {
            serde_json::json!({"type": "transcript.text.done", "text": text})
        }
        StreamEvent::Failed(refusal) => {
            return error_event(&refusal.message, refusal.error_type);
        }
    };
    Event::default().data(body.to_string())
}

#[cfg(test)]
#[path = "stream_tests.rs"]
mod stream_tests;
