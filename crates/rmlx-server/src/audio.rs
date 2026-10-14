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
use rmlx_audio::tokenizer::{TokenizerError, WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{TranscribeOptions, Transcriber, Transcription};
use rmlx_audio::wav::{WavDecoder, WavError, MAX_AUDIO_SECS};
use rmlx_audio::whisper::{WhisperError, WhisperModel};
use rmlx_core::cancel::ensure_not_cancelled;
use rmlx_models::sampler::RowPicker;
use serde::Serialize;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

mod form;
mod speech;

use form::{parse_multipart, rejected_form, ResponseFormat};
pub use speech::{audio_speech, LoadedTts, SpeechRequest};

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
    Tokenizer(TokenizerError),
    Model(WhisperError),
    Init(WhisperError),
    Transcribe(WhisperError),
}

/// The answer to a failed job. The file is the client's fault: 422. The rest is
/// the server's: 503, the status `docs/SERVER_AUDIO.md` gives a load failure and
/// every other engine error. A load or init failure stays until the operator
/// acts; an engine failure during the transcription may not recur.
fn job_refusal(e: &JobError) -> Refusal {
    match e {
        JobError::Decode(e) => bad_field("file", format!("audio decode: {e}")),
        JobError::Tokenizer(e) => unavailable(format!("tokenizer: {e}"), true),
        JobError::Model(e) => unavailable(format!("whisper load: {}", reply_text(e)), true),
        JobError::Init(e) => unavailable(format!("transcriber init: {}", reply_text(e)), true),
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

/// One segment in a `verbose_json` response: the keys of the OpenAI
/// `TranscriptionSegment` schema that a [`Segment`] holds. `avg_logprob` and
/// `no_speech_prob` are left out: the engine reads both for its window
/// decision and puts neither on a segment, and a `null` would make a typed
/// client throw.
///
/// [`Segment`]: rmlx_audio::transcribe::Segment
#[derive(Debug, Serialize)]
struct SegmentJson {
    id: usize,
    seek: usize,
    start: f32,
    end: f32,
    text: String,
    tokens: Vec<u32>,
    /// Temperature of the decode the segment's window kept.
    temperature: f32,
    compression_ratio: f32,
}

#[derive(Debug, Serialize)]
struct VerboseTranscriptionResponse {
    task: String,
    language: String,
    duration: f32,
    text: String,
    segments: Vec<SegmentJson>,
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

#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "audio handler is a sequential pipeline: parse → validate → admit → spawn_blocking → metrics → build response; splitting would obscure the control flow"
)]
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

    let audio_model_cache = Arc::clone(&state.audio_model);

    let audio_bytes = fields.audio_bytes;
    let language = fields.language.clone();
    let temperature = fields.temperature;
    let response_format = fields.response_format;

    let metrics_arc = state.metrics.clone();
    let device = state.device;
    let model_path_str = model_path.to_string_lossy().into_owned();

    let result = run_admitted(Some(guard), move |cancel| {
        // Decode under the permit, one at a time, and before the model cache:
        // a file the decoder refuses loads no model.
        let samples = decode_upload(&audio_bytes, cancel)?;
        drop(audio_bytes);

        let t_load_start = std::time::Instant::now();
        let mut load_ms_opt: Option<f64> = None;

        let (model, tokenizer): (Arc<WhisperModel>, Arc<WhisperTokenizer>) = {
            let read_guard = audio_model_cache.read();
            if let Some((ref m, ref t)) = *read_guard {
                (Arc::clone(m), Arc::clone(t))
            } else {
                drop(read_guard);
                let mut write_guard = audio_model_cache.write();
                // Another request may have loaded the model while this one waited for the lock.
                if let Some((ref m, ref t)) = *write_guard {
                    (Arc::clone(m), Arc::clone(t))
                } else {
                    let t = WhisperTokenizer::from_path(&tok_path).map_err(|e| {
                        warn!(path = %tok_path.display(), error = %e, "Whisper tokenizer load failed");
                        JobError::Tokenizer(e)
                    })?;
                    let m = WhisperModel::load(&model_path).map_err(|e| {
                        warn!(path = %model_path.display(), error = %e, "Whisper model load failed");
                        JobError::Model(e)
                    })?;
                    let m = Arc::new(m);
                    let t = Arc::new(t);
                    *write_guard = Some((Arc::clone(&m), Arc::clone(&t)));
                    load_ms_opt = Some(t_load_start.elapsed().as_secs_f64() * 1_000.0);
                    (m, t)
                }
            }
        };

        let audio_dur_secs = samples.len() as f64 / f64::from(rmlx_audio::mel::SAMPLE_RATE);

        let transcriber = Transcriber::new(Arc::clone(&model), Arc::clone(&tokenizer))
            .map_err(JobError::Init)?;

        let opts = TranscribeOptions {
            language: language.clone(),
            task,
            temperature,
            condition_on_previous_text: true,
        };

        let t_decode_start = std::time::Instant::now();
        let transcription = transcribe_request(&transcriber, &samples, &opts, device, cancel)
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
            "audio inference timing"
        );

        if let Some(sink) = &metrics_arc {
            let notes = format!("task={task:?}");
            let metrics_to_record: &[(&str, &str, f64)] = &[
                ("audio_decode_ms", "ms", decode_ms),
                ("audio_rtf", "ratio", rtf),
            ];
            for (op, unit, value) in metrics_to_record {
                if let Err(e) = sink.record(&rmlx_metrics::events::Measurement {
                    model_path: &model_path_str,
                    quant_mode: "none",
                    stage: "audio",
                    op,
                    value_unit: unit,
                    value: *value,
                    notes: &notes,
                }) {
                    warn!(error = %e, op, "audio metrics record failed (non-fatal)");
                }
            }
            if let Some(load_ms) = load_ms_opt {
                if let Err(e) = sink.record(&rmlx_metrics::events::Measurement {
                    model_path: &model_path_str,
                    quant_mode: "none",
                    stage: "audio",
                    op: "audio_load_ms",
                    value_unit: "ms",
                    value: load_ms,
                    notes: &notes,
                }) {
                    warn!(error = %e, "audio_load_ms record failed (non-fatal)");
                }
            }
        }

        Ok::<Transcription, JobError>(transcription)
    })
    .await;

    job_answer(result, task, response_format)
}

/// The answer to a finished transcription job: the transcript, the refusal of
/// a job that failed, or the 500 of a job that panicked.
fn job_answer(
    result: Result<Result<Transcription, JobError>, tokio::task::JoinError>,
    task: WhisperTask,
    format: ResponseFormat,
) -> Response {
    match result {
        Ok(Ok(transcription)) => {
            info!(
                chars = transcription.text.len(),
                segments = transcription.segments.len(),
                "transcription complete"
            );
            build_response(&transcription, task, format)
        }
        Ok(Err(e)) => {
            let refusal = job_refusal(&e);
            let cause: &dyn std::fmt::Display = match &e {
                JobError::Decode(cause) => cause,
                JobError::Tokenizer(cause) => cause,
                JobError::Model(cause) | JobError::Init(cause) | JobError::Transcribe(cause) => {
                    cause
                }
            };
            warn!(
                status = refusal.status.as_u16(),
                error = %cause,
                reply = refusal.message.as_str(),
                "transcription failed"
            );
            refused(refusal)
        }
        Err(e) => {
            warn!(error = %e, "transcription task panicked");
            internal_error("internal error")
        }
    }
}

/// Transcribe the samples of one request. One [`RowPicker`] with the sampler's
/// default seed picks every token of the request, so a repeated request gives
/// the same tokens also when a decode above temperature 0 ran. The
/// transcription routes and `rmlx transcribe` both call this.
pub fn transcribe_request(
    transcriber: &Transcriber,
    samples: &[f32],
    opts: &TranscribeOptions,
    device: rmlx_mlx::Device,
    cancel: &AtomicBool,
) -> Result<Transcription, WhisperError> {
    let mut picker = RowPicker::default();
    transcriber.transcribe(
        samples,
        opts,
        &mut |row, temperature| picker.pick(row, temperature),
        device,
        cancel,
    )
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
    use rmlx_audio::transcribe::{render, OutputFormat};
    let text = transcription.text.trim();
    match format {
        ResponseFormat::Text => (StatusCode::OK, text.to_owned()).into_response(),
        ResponseFormat::VerboseJson => {
            let body = VerboseTranscriptionResponse {
                task: match task {
                    WhisperTask::Transcribe => "transcribe".to_owned(),
                    WhisperTask::Translate => "translate".to_owned(),
                },
                language: transcription.language.clone(),
                duration: transcription.duration,
                text: text.to_owned(),
                segments: transcription
                    .segments
                    .iter()
                    .enumerate()
                    .map(|(id, s)| SegmentJson {
                        id,
                        seek: s.seek,
                        start: s.start,
                        end: s.end,
                        text: s.text.clone(),
                        tokens: s.tokens.clone(),
                        temperature: s.temperature,
                        compression_ratio: s.compression_ratio,
                    })
                    .collect(),
            };
            Json(body).into_response()
        }
        ResponseFormat::Srt => {
            (StatusCode::OK, render(transcription, OutputFormat::Srt)).into_response()
        }
        ResponseFormat::Vtt => {
            (StatusCode::OK, render(transcription, OutputFormat::Vtt)).into_response()
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
