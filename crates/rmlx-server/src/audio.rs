//! Whisper routes: `POST /v1/audio/transcriptions` and `/v1/audio/translations`.
//! The fields, limits and statuses are in `docs/AUDIO.md`. The speech route is
//! in `speech`, the form reader in `form`.
//!
//! A transcription holds the GPU. It passes `admit_request`, then decodes the
//! upload under the permit and before the model cache is read: decodes run one
//! at a time, a waiting request holds only its upload, and a refused file loads
//! no model.

#![allow(
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    reason = "audio handler is a sequential pipeline; splitting would obscure the control flow"
)]

use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use parking_lot::RwLock as PLRwLock;
use rmlx_audio::tokenizer::{TokenizerError, WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{TranscribeOptions, Transcriber, Transcription, WindowSink};
use rmlx_audio::wav::{WavDecoder, WavError, MAX_AUDIO_SECS};
use rmlx_audio::whisper::{WhisperError, WhisperModel};
use rmlx_core::cancel::ensure_not_cancelled;
use rmlx_metrics::events::EventRecorder;
use rmlx_models::sampler::RowPicker;
use serde::Serialize;
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, instrument, warn};

mod form;
mod speech;
mod stream;

use form::{parse_multipart, rejected_form, ResponseFormat};
pub use speech::{audio_speech, LoadedTts, SpeechRequest};
use stream::{stream_job, stream_reply, STREAM_EVENTS};

use crate::engine::{admit_request, run_admitted, Admission};
use crate::openai::errors::{error_response, error_response_with_param, internal_error};
use crate::openai::state::ApiErrorCategory;
use crate::openai::AppState;

/// The largest audio file a transcription request takes: OpenAI's cap.
const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;

/// The largest request body the server reads: an audio file at the cap, and
/// 1 MiB for the other form fields and the multipart framing.
pub(crate) const MAX_REQUEST_BYTES: usize = MAX_AUDIO_BYTES + 1024 * 1024;

/// The most bytes the log keeps of a name the client gave: an ignored field's
/// name, or a voice.
const LOGGED_NAME_BYTES: usize = 64;

/// The first `max` bytes of `text`, cut at a character boundary.
fn cut(text: &str, max: usize) -> &str {
    text.get(..text.floor_char_boundary(max))
        .unwrap_or_default()
}

/// A refused request: its status, the `type` and `message` of its error
/// envelope, the request field at fault, and whether the fault stays until the
/// operator acts.
#[derive(Debug, PartialEq, Eq)]
struct Refusal {
    status: StatusCode,
    error_type: &'static str,
    message: String,
    /// The `param` of the envelope: the request field at fault, if one is.
    param: Option<String>,
    /// Sends `x-should-retry: false`, which OpenAI's SDKs obey before their
    /// own retry of a 5xx.
    permanent: bool,
}

/// A request that the route cannot take, with no one field at fault.
fn unprocessable(message: String) -> Refusal {
    Refusal {
        status: StatusCode::UNPROCESSABLE_ENTITY,
        error_type: "invalid_request_error",
        message,
        param: None,
        permanent: false,
    }
}

/// A request that the route cannot take because of the field `param`.
fn bad_field(param: &str, message: String) -> Refusal {
    Refusal {
        param: Some(param.to_owned()),
        ..unprocessable(message)
    }
}

fn too_large(message: String) -> Refusal {
    Refusal {
        status: StatusCode::PAYLOAD_TOO_LARGE,
        ..unprocessable(message)
    }
}

/// The server cannot serve the request. `permanent`: the fault stays until the
/// operator acts.
fn unavailable(message: String, permanent: bool) -> Refusal {
    Refusal {
        status: StatusCode::SERVICE_UNAVAILABLE,
        error_type: "service_unavailable",
        message,
        param: None,
        permanent,
    }
}

fn refused(refusal: Refusal) -> Response {
    let mut response = error_response_with_param(
        refusal.status,
        refusal.error_type,
        refusal.param.as_deref(),
        &refusal.message,
    );
    if refusal.permanent {
        response
            .headers_mut()
            .insert("x-should-retry", HeaderValue::from_static("false"));
    }
    response
}

/// Where a transcription job failed.
#[derive(Debug)]
enum JobError {
    Decode(WavError),
    /// The prompt holds the text of a special token.
    Prompt(TokenizerError),
    Tokenizer(TokenizerError),
    Model(WhisperError),
    Init(WhisperError),
    Transcribe(WhisperError),
}

/// The answer to a failed job. The file, the prompt and a request for word
/// times from a model with no alignment heads are the client's fault: 422. The
/// rest is the server's: 503, the status `docs/SERVER_AUDIO.md` gives a load failure and
/// every other engine error. A load or init failure stays until the operator
/// acts; an engine failure during the transcription may not recur.
fn job_refusal(e: &JobError) -> Refusal {
    match e {
        JobError::Decode(e) => bad_field("file", format!("audio decode: {e}")),
        JobError::Prompt(e) => bad_field("prompt", format!("invalid prompt: {e}")),
        JobError::Tokenizer(e) => unavailable(format!("tokenizer: {e}"), true),
        JobError::Model(e) => unavailable(format!("whisper load: {}", reply_text(e)), true),
        JobError::Init(e) => unavailable(format!("transcriber init: {}", reply_text(e)), true),
        JobError::Transcribe(e @ WhisperError::NoAlignmentHeads) => {
            bad_field("timestamp_granularities", e.to_string())
        }
        JobError::Transcribe(e) => unavailable(format!("transcribe: {}", reply_text(e)), false),
    }
}

/// What starts text from MLX in the message of a Whisper failure: the
/// `WhisperError::Mlx` message, and the `NpzError::Mlx` message inside a
/// `WhisperError::WeightFile` one.
const MLX_TEXT_MARKS: [&str; 2] = ["MLX op error: ", "MLX: "];

/// The text of a Whisper failure for a reply. An MLX error message can hold a
/// path of the server's MLX install. The reply keeps the message up to the
/// first MLX mark (the file or the step) and gets a fixed text in place of the
/// rest; the log keeps the cause.
fn reply_text(e: &WhisperError) -> String {
    let text = e.to_string();
    match MLX_TEXT_MARKS
        .iter()
        .filter_map(|mark| text.find(mark))
        .min()
    {
        Some(at) => format!(
            "{}MLX error; the server log has the cause",
            text.get(..at).unwrap_or_default()
        ),
        None => text,
    }
}

#[derive(Debug, Serialize)]
struct TranscriptionResponse {
    text: String,
}

/// `POST /v1/audio/transcriptions` — Whisper STT.
#[instrument(skip_all, name = "audio_transcriptions", level = "info")]
pub async fn audio_transcriptions(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    handle_audio(state, &headers, multipart, WhisperTask::Transcribe).await
}

/// `POST /v1/audio/translations` — Whisper STT with English output.
#[instrument(skip_all, name = "audio_translations", level = "info")]
pub async fn audio_translations(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
) -> Response {
    handle_audio(state, &headers, multipart, WhisperTask::Translate).await
}

async fn handle_audio(
    state: AppState,
    headers: &HeaderMap,
    multipart: Result<Multipart, MultipartRejection>,
    task: WhisperTask,
) -> Response {
    let body_len = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    let mut multipart = match multipart {
        Ok(multipart) => multipart,
        Err(rejection) => return refused(rejected_form(&rejection)),
    };
    let fields = match parse_multipart(&mut multipart, body_len).await {
        Ok(f) => f,
        Err(refusal) => return refused(refusal),
    };

    if fields.audio_bytes.len() > MAX_AUDIO_BYTES {
        return refused(Refusal {
            param: Some("file".to_owned()),
            ..too_large(format!(
                "audio file is {} bytes, over the cap of {MAX_AUDIO_BYTES} bytes (25 MiB)",
                fields.audio_bytes.len()
            ))
        });
    }

    info!(
        model = fields.model,
        lang = fields.language,
        audio_bytes = fields.audio_bytes.len(),
        prompt_bytes = fields.prompt.len(),
        task = ?task,
        "audio request"
    );

    let Some((model_path, tok_path)) = state.audio_paths() else {
        return refused(unavailable(
            "audio model not configured; set --whisper-model-path".to_owned(),
            true,
        ));
    };

    // Audio queues in the chat routes' FIFO, which also gives the 429 back-pressure.
    let guard =
        match admit_request(&state.gpu_queue, &state.gpu_pending, state.max_queue_depth).await {
            Admission::QueueFull => {
                state.error_counts.increment(ApiErrorCategory::RateLimit);
                return error_response(
                    StatusCode::TOO_MANY_REQUESTS,
                    "rate_limit_error",
                    "server queue full",
                );
            }
            Admission::Admitted {
                guard,
                depth,
                wait_ms,
            } => {
                debug!(depth, wait_ms, "audio request admitted");
                guard
            }
        };

    let input = JobInput {
        audio_model: Arc::clone(&state.audio_model),
        model_path,
        tok_path,
        audio_bytes: fields.audio_bytes,
        prompt: fields.prompt,
        language: fields.language,
        temperature: fields.temperature,
        words: fields.words,
        task,
        metrics: state.metrics.clone(),
        device: state.device,
    };

    if fields.stream {
        let (ready_tx, ready_rx) = oneshot::channel();
        let (events_tx, events_rx) = mpsc::channel(STREAM_EVENTS);
        let job = run_admitted(Some(guard), move |cancel| {
            stream_job(input, cancel, ready_tx, &events_tx);
        });
        return stream_reply(ready_rx, events_rx, job).await;
    }

    let response_format = fields.response_format;
    let result = run_admitted(Some(guard), move |cancel| {
        prepare(input, cancel)?.run(cancel, &mut |_| Ok(()))
    })
    .await;
    job_answer(result, task, response_format)
}

/// What a transcription job reads, moved into the job.
struct JobInput {
    audio_model: AudioModelCache,
    model_path: PathBuf,
    tok_path: PathBuf,
    audio_bytes: Vec<u8>,
    prompt: String,
    language: String,
    temperature: f32,
    words: bool,
    task: WhisperTask,
    metrics: Option<Arc<EventRecorder>>,
    device: rmlx_mlx::Device,
}

/// The Whisper model and tokenizer of the server, loaded by the first request.
type AudioModelCache = Arc<PLRwLock<Option<(Arc<WhisperModel>, Arc<WhisperTokenizer>)>>>;

/// A job whose prompt is encoded, whose audio is decoded and whose model is
/// loaded: what is left is the walk over the windows.
struct ReadyJob {
    transcriber: Transcriber,
    samples: Vec<f32>,
    opts: TranscribeOptions,
    device: rmlx_mlx::Device,
    metrics: Option<Arc<EventRecorder>>,
    model_path: String,
    load_ms: Option<f64>,
}

/// Encode the prompt, decode the audio and load the model. A failure here is
/// the error reply of the request, for a streamed request too.
fn prepare(input: JobInput, cancel: &AtomicBool) -> Result<ReadyJob, JobError> {
    let JobInput {
        audio_model,
        model_path,
        tok_path,
        audio_bytes,
        prompt,
        language,
        temperature,
        words,
        task,
        metrics,
        device,
    } = input;

    // The prompt first: a prompt that cannot be used is refused before the
    // audio is decoded and before the model loads.
    let cached_tokenizer = audio_model
        .read()
        .as_ref()
        .map(|(_, tokenizer)| Arc::clone(tokenizer));
    let (prompt_tokens, mut loaded_tokenizer) =
        encode_prompt(cached_tokenizer, &tok_path, &prompt)?;
    drop(prompt);

    // Decode under the permit, one at a time, and before the model cache:
    // a file the decoder refuses loads no model.
    let samples = decode_upload(&audio_bytes, cancel)?;
    drop(audio_bytes);

    let t_load_start = std::time::Instant::now();
    let mut load_ms: Option<f64> = None;

    let (model, tokenizer): (Arc<WhisperModel>, Arc<WhisperTokenizer>) = {
        let read_guard = audio_model.read();
        if let Some((ref m, ref t)) = *read_guard {
            (Arc::clone(m), Arc::clone(t))
        } else {
            drop(read_guard);
            let mut write_guard = audio_model.write();
            // Another request may have loaded the model while this one waited for the lock.
            if let Some((ref m, ref t)) = *write_guard {
                (Arc::clone(m), Arc::clone(t))
            } else {
                let t = match loaded_tokenizer.take() {
                    Some(t) => t,
                    None => Arc::new(load_tokenizer(&tok_path)?),
                };
                let m = WhisperModel::load(&model_path).map_err(|e| {
                    warn!(path = %model_path.display(), error = %e, "Whisper model load failed");
                    JobError::Model(e)
                })?;
                let m = Arc::new(m);
                *write_guard = Some((Arc::clone(&m), Arc::clone(&t)));
                load_ms = Some(t_load_start.elapsed().as_secs_f64() * 1_000.0);
                (m, t)
            }
        }
    };

    let transcriber = Transcriber::new(model, tokenizer).map_err(JobError::Init)?;

    Ok(ReadyJob {
        transcriber,
        samples,
        opts: TranscribeOptions {
            language,
            task,
            temperature,
            condition_on_previous_text: true,
            prompt_tokens,
            words,
        },
        device,
        metrics,
        model_path: model_path.to_string_lossy().into_owned(),
        load_ms,
    })
}

impl ReadyJob {
    /// Walk the windows, handing each closed window to `on_window`, and record
    /// the metrics of a walk that ran to its end.
    fn run(
        self,
        cancel: &AtomicBool,
        on_window: WindowSink<'_>,
    ) -> Result<Transcription, JobError> {
        let audio_dur_secs = self.samples.len() as f64 / f64::from(rmlx_audio::mel::SAMPLE_RATE);
        let t_decode_start = std::time::Instant::now();
        let transcription = transcribe_request(
            &self.transcriber,
            &self.samples,
            &self.opts,
            self.device,
            cancel,
            on_window,
        )
        .map_err(JobError::Transcribe)?;
        let decode_ms = t_decode_start.elapsed().as_secs_f64() * 1_000.0;

        let rtf = if audio_dur_secs > 0.0 {
            decode_ms / 1_000.0 / audio_dur_secs
        } else {
            0.0
        };

        info!(
            decode_ms = decode_ms as u64,
            audio_dur_secs,
            rtf,
            n_segments = transcription.segments.len(),
            chars = transcription.text.len(),
            "transcription complete"
        );

        if let Some(sink) = &self.metrics {
            let notes = format!("task={:?}", self.opts.task);
            let mut measurements = vec![
                ("audio_decode_ms", "ms", decode_ms),
                ("audio_rtf", "ratio", rtf),
            ];
            if let Some(load_ms) = self.load_ms {
                measurements.push(("audio_load_ms", "ms", load_ms));
            }
            for (op, unit, value) in measurements {
                if let Err(e) = sink.record(&rmlx_metrics::events::Measurement {
                    model_path: &self.model_path,
                    quant_mode: "none",
                    stage: "audio",
                    op,
                    value_unit: unit,
                    value,
                    notes: &notes,
                }) {
                    warn!(error = %e, op, "audio metrics record failed (non-fatal)");
                }
            }
        }

        Ok(transcription)
    }
}

/// The answer to a finished transcription job: the transcript, the refusal of
/// a job that failed, or the 500 of a job that panicked.
fn job_answer(
    result: Result<Result<Transcription, JobError>, tokio::task::JoinError>,
    task: WhisperTask,
    format: ResponseFormat,
) -> Response {
    match result {
        Ok(Ok(transcription)) => build_response(&transcription, task, format),
        Ok(Err(e)) => refused(failed(&e, false)),
        Err(e) => panicked(&e),
    }
}

/// The refusal of a failed job, logged with its cause. `streamed`: the reply
/// is a stream that started, so the client gets the refusal as an error event
/// in a reply of status 200, and the log gives that status.
fn failed(e: &JobError, streamed: bool) -> Refusal {
    let refusal = job_refusal(e);
    let cause: &dyn std::fmt::Display = match e {
        JobError::Decode(cause) => cause,
        JobError::Prompt(cause) | JobError::Tokenizer(cause) => cause,
        JobError::Model(cause) | JobError::Init(cause) | JobError::Transcribe(cause) => cause,
    };
    let status = if streamed {
        StatusCode::OK
    } else {
        refusal.status
    };
    warn!(
        status = status.as_u16(),
        streamed,
        error_type = refusal.error_type,
        error = %cause,
        reply = refusal.message.as_str(),
        "transcription failed"
    );
    refusal
}

fn panicked(e: &tokio::task::JoinError) -> Response {
    warn!(error = %e, "transcription task panicked");
    internal_error("internal error")
}

/// Transcribe the samples of one request, handing each closed window to
/// `on_window`. One [`RowPicker`] with the sampler's default seed picks every
/// token of the request, so a repeated request gives the same tokens also when
/// a decode above temperature 0 ran. The transcription routes and
/// `rmlx transcribe` both call this.
pub fn transcribe_request(
    transcriber: &Transcriber,
    samples: &[f32],
    opts: &TranscribeOptions,
    device: rmlx_mlx::Device,
    cancel: &AtomicBool,
    on_window: WindowSink<'_>,
) -> Result<Transcription, WhisperError> {
    let mut picker = RowPicker::default();
    transcriber.transcribe(
        samples,
        opts,
        &mut |row, temperature| picker.pick(row, temperature),
        device,
        cancel,
        on_window,
    )
}

/// Load the Whisper tokenizer; a failure stays until the operator acts.
fn load_tokenizer(tok_path: &std::path::Path) -> Result<WhisperTokenizer, JobError> {
    WhisperTokenizer::from_path(tok_path).map_err(|e| {
        warn!(path = %tok_path.display(), error = %e, "Whisper tokenizer load failed");
        JobError::Tokenizer(e)
    })
}

/// The tokens of the request's `prompt`, and the tokenizer when this call
/// loaded it, for the model load to keep. `cached` is the tokenizer of the
/// model cache, if the cache is filled. A blank prompt loads nothing.
fn encode_prompt(
    cached: Option<Arc<WhisperTokenizer>>,
    tok_path: &std::path::Path,
    prompt: &str,
) -> Result<(Vec<u32>, Option<Arc<WhisperTokenizer>>), JobError> {
    if prompt.trim().is_empty() {
        return Ok((Vec::new(), None));
    }
    let (tokenizer, loaded) = if let Some(tokenizer) = cached {
        (tokenizer, None)
    } else {
        let tokenizer = Arc::new(load_tokenizer(tok_path)?);
        (Arc::clone(&tokenizer), Some(tokenizer))
    };
    let tokens = tokenizer.encode_prompt(prompt).map_err(prompt_error)?;
    Ok((tokens, loaded))
}

/// Only the text of a special token is the client's fault; any other failure
/// of the tokenizer is the server's.
fn prompt_error(e: TokenizerError) -> JobError {
    if matches!(e, TokenizerError::SpecialTokenText) {
        JobError::Prompt(e)
    } else {
        JobError::Tokenizer(e)
    }
}

/// The 16 kHz samples of the uploaded file, up to `MAX_AUDIO_SECS`. The decode
/// stops before its next packet once `cancel` is set.
fn decode_upload(bytes: &[u8], cancel: &AtomicBool) -> Result<Vec<f32>, JobError> {
    WavDecoder::decode_resampled(bytes, rmlx_audio::mel::SAMPLE_RATE, MAX_AUDIO_SECS, &|| {
        ensure_not_cancelled(cancel).is_err()
    })
    .map(|audio| audio.samples)
    .map_err(JobError::Decode)
}

fn build_response(
    transcription: &Transcription,
    task: WhisperTask,
    format: ResponseFormat,
) -> Response {
    use rmlx_audio::transcribe::{render_srt, render_vtt};
    let text = transcription.text.trim();
    match format {
        ResponseFormat::Text => (StatusCode::OK, text.to_owned()).into_response(),
        ResponseFormat::VerboseJson => Json(transcription.record(Some(task))).into_response(),
        ResponseFormat::Srt => {
            (StatusCode::OK, render_srt(&transcription.segments)).into_response()
        }
        ResponseFormat::Vtt => {
            (StatusCode::OK, render_vtt(&transcription.segments)).into_response()
        }
        ResponseFormat::Json => Json(TranscriptionResponse {
            text: text.to_owned(),
        })
        .into_response(),
    }
}

#[cfg(test)]
#[path = "audio_tests.rs"]
mod audio_tests;

#[cfg(test)]
mod test_events;
