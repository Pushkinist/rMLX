//! Audio routes: `POST /v1/audio/transcriptions`, `/v1/audio/translations` and
//! `/v1/audio/speech`. The fields, limits and statuses are in `docs/AUDIO.md`.
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

use axum::extract::{Multipart, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use rmlx_audio::tokenizer::{TokenizerError, WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{TranscribeOptions, Transcriber};
use rmlx_audio::wav::{WavDecoder, WavError, MAX_AUDIO_SECS};
use rmlx_audio::whisper::{WhisperError, WhisperModel};
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

mod form;

use form::{parse_multipart, ResponseFormat};

use crate::engine::{admit_request, Admission, GpuAdmission};
use crate::openai::state::ApiErrorCategory;
use crate::openai::AppState;

/// The largest audio file a transcription request takes: OpenAI's cap.
const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;

/// The largest request body the server reads: an audio file at the cap, and
/// 1 MiB for the other form fields and the multipart framing.
pub(crate) const MAX_REQUEST_BYTES: usize = MAX_AUDIO_BYTES + 1024 * 1024;

/// A refused request: its status, the text of its `{"error": …}` body, and
/// whether the fault stays until the operator acts.
#[derive(Debug, PartialEq, Eq)]
struct Refusal {
    status: StatusCode,
    message: String,
    /// Sends `x-should-retry: false`, which OpenAI's SDKs obey before their
    /// own retry of a 5xx.
    permanent: bool,
}

fn unprocessable(message: String) -> Refusal {
    Refusal {
        status: StatusCode::UNPROCESSABLE_ENTITY,
        message,
        permanent: false,
    }
}

fn too_large(message: String) -> Refusal {
    Refusal {
        status: StatusCode::PAYLOAD_TOO_LARGE,
        message,
        permanent: false,
    }
}

fn refused(refusal: Refusal) -> Response {
    let body = Json(json!({ "error": refusal.message }));
    if refusal.permanent {
        (refusal.status, [("x-should-retry", "false")], body).into_response()
    } else {
        (refusal.status, body).into_response()
    }
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
/// the server's: 503, the status `docs/SERVER.md` gives a load failure and
/// every other engine error. A load or init failure stays until the operator
/// acts; an engine failure during the transcription may not recur.
fn job_refusal(e: &JobError) -> Refusal {
    let server = |message: String, permanent: bool| Refusal {
        status: StatusCode::SERVICE_UNAVAILABLE,
        message,
        permanent,
    };
    match e {
        JobError::Decode(e) => unprocessable(format!("audio decode: {e}")),
        JobError::Tokenizer(e) => server(format!("tokenizer: {e}"), true),
        JobError::Model(e) => server(format!("whisper load: {e}"), true),
        JobError::Init(e) => server(format!("transcriber init: {e}"), true),
        JobError::Transcribe(e) => server(format!("transcribe: {e}"), false),
    }
}

#[derive(Debug, Serialize)]
struct TranscriptionResponse {
    text: String,
}

/// One segment in a `verbose_json` response: the keys of the OpenAI
/// `TranscriptionSegment` schema the decode computes. `avg_logprob` and
/// `no_speech_prob` are left out: no log-probability is kept and no no-speech
/// gate runs, and a `null` would make a typed client throw.
#[derive(Debug, Serialize)]
struct SegmentJson {
    id: usize,
    seek: usize,
    start: f32,
    end: f32,
    text: String,
    tokens: Vec<u32>,
    /// Every decode is greedy, whatever the request's `temperature`.
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
    multipart: Multipart,
) -> Response {
    handle_audio(state, &headers, multipart, WhisperTask::Transcribe).await
}

/// `POST /v1/audio/translations` — Whisper STT with English output.
#[instrument(skip_all, name = "audio_translations", level = "info")]
pub async fn audio_translations(
    State(state): State<AppState>,
    headers: HeaderMap,
    multipart: Multipart,
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
    mut multipart: Multipart,
    task: WhisperTask,
) -> Response {
    let body_len = headers
        .get(header::CONTENT_LENGTH)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.parse::<usize>().ok());
    let fields = match parse_multipart(&mut multipart, body_len).await {
        Ok(f) => f,
        Err(refusal) => return refused(refusal),
    };

    if fields.audio_bytes.len() > MAX_AUDIO_BYTES {
        return refused(too_large(format!(
            "audio file is {} bytes, over the cap of {MAX_AUDIO_BYTES} bytes (25 MiB)",
            fields.audio_bytes.len()
        )));
    }

    info!(
        model = fields.model,
        lang = fields.language,
        audio_bytes = fields.audio_bytes.len(),
        task = ?task,
        "audio request"
    );

    let Some((model_path, tok_path)) = state.audio_paths() else {
        return refused(Refusal {
            status: StatusCode::SERVICE_UNAVAILABLE,
            message: "audio model not configured; set --whisper-model-path".to_owned(),
            permanent: true,
        });
    };

    // Audio queues in the chat routes' FIFO, which also gives the 429 back-pressure.
    let guard =
        match admit_request(&state.gpu_queue, &state.gpu_pending, state.max_queue_depth).await {
            Admission::QueueFull => {
                state.error_counts.increment(ApiErrorCategory::RateLimit);
                return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(
                    json!({"error": {"type": "rate_limit_error", "message": "server queue full"}}),
                ),
            )
                .into_response();
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

    let result = run_admitted(guard, move |cancel| {
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
        let transcription = transcriber
            .transcribe(&samples, &opts, device, cancel)
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

        Ok::<rmlx_audio::transcribe::Transcription, JobError>(transcription)
    })
    .await;

    match result {
        Ok(Ok(transcription)) => {
            info!(
                chars = transcription.text.len(),
                segments = transcription.segments.len(),
                "transcription complete"
            );
            build_response(&transcription, task, response_format)
        }
        Ok(Err(e)) => {
            let refusal = job_refusal(&e);
            warn!(
                status = refusal.status.as_u16(),
                error = refusal.message,
                "transcription failed"
            );
            refused(refusal)
        }
        Err(e) => {
            warn!(error = %e, "transcription task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal error"})),
            )
                .into_response()
        }
    }
}

/// The 16 kHz samples of the uploaded file, up to `MAX_AUDIO_SECS`. The decode
/// stops before its next packet once `cancel` is set.
fn decode_upload(bytes: &[u8], cancel: &AtomicBool) -> Result<Vec<f32>, JobError> {
    WavDecoder::decode_resampled(bytes, rmlx_audio::mel::SAMPLE_RATE, MAX_AUDIO_SECS, &|| {
        cancel.load(Ordering::Relaxed)
    })
    .map(|audio| audio.samples)
    .map_err(JobError::Decode)
}

/// Run `job` on the blocking pool, inside the caller's span, while `guard`
/// holds the GPU permit. The flag handed to `job` is set when the returned
/// future is dropped: the request timed out or the client went away. A job
/// that reads the flag can stop early. The permit is released when `job`
/// returns, so the next request never shares the GPU with a job that is still
/// stopping.
// cancel-safe: a drop sets the flag; the job keeps the permit until it returns.
// After a normal return the flag is set with no reader.
async fn run_admitted<T, F>(guard: GpuAdmission, job: F) -> Result<T, tokio::task::JoinError>
where
    T: Send + 'static,
    F: FnOnce(&AtomicBool) -> T + Send + 'static,
{
    let cancel_on_drop = CancelOnDrop(Arc::new(AtomicBool::new(false)));
    let flag = Arc::clone(&cancel_on_drop.0);
    let span = tracing::Span::current();
    tokio::task::spawn_blocking(move || {
        let _entered = span.enter();
        let _guard = guard;
        job(&flag)
    })
    .await
}

struct CancelOnDrop(Arc<AtomicBool>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Relaxed);
    }
}

fn build_response(
    transcription: &rmlx_audio::transcribe::Transcription,
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
                        temperature: 0.0,
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

// ── POST /v1/audio/speech (TTS) ───────────────────────────────────────────────
//
// Accepted body fields:
//   model          — model identifier (e.g. "qwen3-tts", "tts-1")
//   input          — text to synthesize
//   voice          — voice name (see talker_config.spk_id in config.json)
//                    Available: serena, vivian, ryan, aiden, eric, dylan, ono_anna, sohee, uncle_fu
//   response_format — "wav" | "pcm" (default: "wav")
//   speed           — 0.25–4.0 (default: 1.0; unused at v1 — codec speed is fixed)
//
// Output sample rate: 24 000 Hz (12.5 Hz token rate × 1920 upsample).

/// Request body for `POST /v1/audio/speech`.
#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    /// Model identifier (e.g. `"qwen3-tts"` or `"tts-1"`).
    model: String,
    /// Text to synthesize.
    input: String,
    /// Voice name. Available: serena, vivian, ryan, aiden, eric, dylan, ono_anna, sohee, uncle_fu.
    #[serde(default = "default_voice")]
    voice: String,
    /// Output format: `"wav"` (default) or `"pcm"`.
    #[serde(default = "default_response_format_speech")]
    response_format: String,
    /// Playback speed multiplier (0.25–4.0). Accepted but not applied at v1.
    #[serde(default = "default_speed")]
    #[allow(dead_code)]
    speed: f32,
}

fn default_voice() -> String {
    "serena".to_owned()
}

fn default_response_format_speech() -> String {
    "wav".to_owned()
}

fn default_speed() -> f32 {
    1.0
}

/// `POST /v1/audio/speech` — Qwen3-TTS speech synthesis.
///
/// Synthesizes mono 24 kHz PCM from text via the Qwen3-TTS talker + codec decoder.
/// Returns `audio/wav` bytes on success.
#[allow(clippy::cognitive_complexity)]
#[instrument(skip_all, name = "audio_speech", level = "info")]
pub async fn audio_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    info!(
        model = req.model,
        voice = req.voice,
        input_len = req.input.len(),
        "audio/speech request"
    );

    // 1. Check if TTS paths are configured at all.
    let (Some(tts_model_path), Some(tts_tok_path)) =
        (&state.tts_model_path, &state.tts_tokenizer_path)
    else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({
                "error": {
                    "type": "not_supported",
                    "message": "TTS not configured; set --tts-model-path and --tts-tokenizer-path"
                }
            })),
        )
            .into_response();
    };

    // 2. GPU admission — TTS holds Metal; go through the same FIFO gate as LLM/Whisper.
    let guard =
        match admit_request(&state.gpu_queue, &state.gpu_pending, state.max_queue_depth).await {
            Admission::QueueFull => {
                state.error_counts.increment(ApiErrorCategory::RateLimit);
                return (
                StatusCode::TOO_MANY_REQUESTS,
                Json(
                    json!({"error": {"type": "rate_limit_error", "message": "server queue full"}}),
                ),
            )
                .into_response();
            }
            Admission::Admitted {
                guard,
                depth,
                wait_ms,
            } => {
                debug!(depth, wait_ms, "TTS request admitted");
                guard
            }
        };

    // 3. Move fields into the blocking closure.
    let tts_model_cache = Arc::clone(&state.tts_model);
    let tts_model_path = tts_model_path.clone();
    let tts_tok_path = tts_tok_path.clone();
    let voice = req.voice.clone();
    let input = req.input.clone();
    let response_format = req.response_format.clone();
    let metrics_arc = state.metrics.clone();
    let device = state.device;
    let model_path_str = tts_model_path.to_string_lossy().into_owned();

    let result = run_admitted(guard, move |_cancel| {
        // Load or retrieve cached (model, tokenizer).
        let t_load_start = std::time::Instant::now();
        let mut load_ms_opt: Option<f64> = None;

        let (model_mutex, tokenizer): (
            Arc<parking_lot::Mutex<rmlx_audio::tts::TtsModel>>,
            Arc<rmlx_audio::tts::TtsTokenizer>,
        ) = {
            let read_guard = tts_model_cache.read();
            if let Some((ref m, ref t)) = *read_guard {
                (Arc::clone(m), Arc::clone(t))
            } else {
                drop(read_guard);
                let mut write_guard = tts_model_cache.write();
                if let Some((ref m, ref t)) = *write_guard {
                    (Arc::clone(m), Arc::clone(t))
                } else {
                    let m = rmlx_audio::tts::TtsModel::load_config(
                        &tts_model_path,
                        &tts_tok_path,
                        device,
                    )
                    .map_err(|e| format!("tts config load: {e}"))?;
                    // Text tokenizer lives in the talker model snapshot (vocab.json +
                    // merges.txt), not in the codec-decoder path.
                    let t = rmlx_audio::tts::TtsTokenizer::from_path(&tts_model_path)
                        .map_err(|e| format!("tts tokenizer: {e}"))?;
                    let m = Arc::new(parking_lot::Mutex::new(m));
                    let t = Arc::new(t);
                    *write_guard = Some((Arc::clone(&m), Arc::clone(&t)));
                    load_ms_opt = Some(t_load_start.elapsed().as_secs_f64() * 1_000.0);
                    (m, t)
                }
            }
        };

        // Synthesize.
        let t_synth_start = std::time::Instant::now();
        let (samples, sample_rate) = {
            let mut model = model_mutex.lock();
            rmlx_audio::tts::synthesize(&input, &voice, &mut model, &tokenizer)
                .map_err(|e| format!("tts synthesize: {e}"))?
        };
        let synth_ms = t_synth_start.elapsed().as_secs_f64() * 1_000.0;

        // Audio duration in seconds (for RTF).
        let audio_dur_secs = samples.len() as f64 / f64::from(sample_rate);
        let rtf = if audio_dur_secs > 0.0 {
            synth_ms / 1_000.0 / audio_dur_secs
        } else {
            0.0
        };

        info!(
            synth_ms = synth_ms as u64,
            samples = samples.len(),
            sample_rate,
            rtf,
            "TTS synthesis complete"
        );

        // Encode output.
        let output_bytes: Vec<u8> = if response_format == "pcm" {
            // Raw f32-LE PCM.
            samples.iter().flat_map(|s| s.to_le_bytes()).collect()
        } else {
            // WAV (default).
            rmlx_audio::wav::WavEncoder::encode(&samples, sample_rate, 1)
                .map_err(|e| format!("wav encode: {e}"))?
        };

        // Emit metrics.
        if let Some(sink) = &metrics_arc {
            let metrics_to_record: &[(&str, &str, f64)] = &[
                ("audio_synth_ms", "ms", synth_ms),
                ("audio_rtf", "ratio", rtf),
            ];
            for (op, unit, value) in metrics_to_record {
                if let Err(e) = sink.record(&rmlx_metrics::events::Measurement {
                    model_path: &model_path_str,
                    quant_mode: "none",
                    stage: "tts",
                    op,
                    value_unit: unit,
                    value: *value,
                    notes: "",
                }) {
                    warn!(error = %e, op, "TTS metrics record failed (non-fatal)");
                }
            }
            if let Some(load_ms) = load_ms_opt {
                if let Err(e) = sink.record(&rmlx_metrics::events::Measurement {
                    model_path: &model_path_str,
                    quant_mode: "none",
                    stage: "tts",
                    op: "audio_load_ms",
                    value_unit: "ms",
                    value: load_ms,
                    notes: "",
                }) {
                    warn!(error = %e, "TTS audio_load_ms record failed (non-fatal)");
                }
            }
        }

        Ok::<(Vec<u8>, String), String>((output_bytes, response_format))
    })
    .await;

    match result {
        Ok(Ok((bytes, fmt))) => {
            let content_type = if fmt == "pcm" {
                "audio/pcm"
            } else {
                "audio/wav"
            };
            (
                StatusCode::OK,
                [(header::CONTENT_TYPE, content_type)],
                bytes,
            )
                .into_response()
        }
        Ok(Err(msg)) => {
            warn!(error = msg, "TTS synthesis failed");
            let status = if msg.contains("unknown voice") || msg.contains("UnknownVoice") {
                StatusCode::UNPROCESSABLE_ENTITY
            } else {
                StatusCode::INTERNAL_SERVER_ERROR
            };
            (status, Json(json!({"error": msg}))).into_response()
        }
        Err(e) => {
            warn!(error = %e, "TTS task panicked");
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": "internal error"})),
            )
                .into_response()
        }
    }
}

#[cfg(test)]
#[path = "audio_tests.rs"]
mod audio_tests;
