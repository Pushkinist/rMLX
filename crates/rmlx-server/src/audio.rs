//! Audio API endpoints: POST /v1/audio/transcriptions, /v1/audio/translations,
//! and POST /v1/audio/speech (TTS — Phase 4b pending).
//!
//! ## API shape (OpenAI-compatible)
//!
//! `POST /v1/audio/transcriptions` — Multipart form with:
//! - `file` — audio file bytes (any Symphonia-supported container at 16 kHz mono).
//! - `model` — model identifier (e.g. `whisper-large-v3`).
//! - `language` — optional BCP-47 language code (default: `"en"`). Unknown codes
//!   receive HTTP 422.
//! - `response_format` — `json` | `text` | `verbose_json` | `srt` | `vtt` (default: `json`).
//! - `temperature` — sampling temperature in `[0.0, 1.0]` (default: `0.0`). Malformed
//!   or out-of-range values receive HTTP 422.
//! - `prompt` — optional text to guide the decoder (not yet implemented).
//!
//! `POST /v1/audio/translations` — same shape; forces English output (translate task).
//!
//! ## v1 constraints
//!
//! - Any Symphonia-supported container at 16 kHz mono is accepted (WAV, MP3, FLAC, …).
//!   The decoder rejects audio at a sample rate other than 16 kHz with 422.
//! - No streaming (SSE timestamps) — deferred to v2.
//! - 25 MiB audio file size cap (enforced by server body limit; additional
//!   per-field check here). The transport `DefaultBodyLimit` is set to 26 MiB
//!   (25 MiB payload + 1 MiB multipart framing slack).
//!
//! ## Model loading and caching
//!
//! The Whisper model and tokenizer are loaded on the **first** audio request
//! and cached in `AppState::audio_model` for the lifetime of the server.
//! Subsequent requests `read()` the cache and `Arc::clone` — no re-parse,
//! no re-upload. A server restart is required to change the snapshot path.
//!
//! ## Admission
//!
//! Audio decode holds the GPU; this handler goes through the same
//! `admit_request` → `gpu_queue` FIFO semaphore path as the LLM chat routes.
//! This ensures:
//! - Audio is counted toward `max_queue_depth` (HTTP 429 backpressure applies).
//! - Audio waits in FIFO order with chat requests — no queue-jumping.
//! - The admission guard (`GpuAdmission`) is moved into the blocking closure
//!   and dropped when the closure returns, releasing the semaphore permit.
//!
//! ## No-tokenizer mode
//!
//! The mlx-community Whisper snapshot does NOT ship a `tokenizer.json`. The
//! handler looks for a companion tokenizer directory (`whisper_tokenizer_path`
//! in `AppState`, set via `--whisper-tokenizer-path` CLI flag). If either path
//! is absent the handler returns 503 with a clear message.
//! For production use: place the `openai/whisper-large-v3` tokenizer files
//! alongside the snapshot, or set `RMLX_WHISPER_TOKENIZER_PATH`.

#![allow(
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::too_many_lines,
    reason = "audio handler is a sequential pipeline; splitting would obscure the control flow"
)]

use axum::extract::{Multipart, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use rmlx_audio::tokenizer::{WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{TranscribeOptions, Transcriber};
use rmlx_audio::wav::WavDecoder;
use rmlx_audio::whisper::WhisperModel;
use rmlx_mlx::Device;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, info, instrument, warn};

use crate::engine::{admit_request, Admission};
use crate::openai::state::ApiErrorCategory;
use crate::openai::AppState;

// ── Request body fields ───────────────────────────────────────────────────────

/// Parsed multipart form fields for both transcription and translation.
struct AudioFormFields {
    /// Raw audio bytes.
    audio_bytes: Vec<u8>,
    /// Model identifier (for logging / routing).
    model: String,
    /// Optional language code.
    language: String,
    /// Response format.
    response_format: ResponseFormat,
    /// Sampling temperature.
    temperature: f32,
}

/// Response format for audio transcriptions.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum ResponseFormat {
    /// JSON object with `{"text": "..."}`.
    #[default]
    Json,
    /// Plain text only.
    Text,
    /// JSON with segments, language, duration.
    VerboseJson,
    /// SRT subtitle format.
    Srt,
    /// WebVTT format.
    Vtt,
}

impl ResponseFormat {
    /// Parse a response format string; returns an error string on unrecognised values
    /// so callers can return HTTP 422 rather than silently defaulting.
    fn parse_or_default(s: &str) -> Result<Self, String> {
        match s {
            "json" => Ok(Self::Json),
            "text" => Ok(Self::Text),
            "verbose_json" => Ok(Self::VerboseJson),
            "srt" => Ok(Self::Srt),
            "vtt" => Ok(Self::Vtt),
            other => Err(format!(
                "unsupported response_format '{other}'; must be one of: json, text, verbose_json, srt, vtt"
            )),
        }
    }
}

// ── Response types ────────────────────────────────────────────────────────────

/// `json` response: `{"text": "..."}`.
#[derive(Debug, Serialize)]
struct TranscriptionResponse {
    text: String,
}

/// One segment in a `verbose_json` response.
#[derive(Debug, Serialize)]
struct SegmentJson {
    id: usize,
    start: f32,
    end: f32,
    text: String,
}

/// `verbose_json` response with metadata.
#[derive(Debug, Serialize)]
struct VerboseTranscriptionResponse {
    task: String,
    language: String,
    duration: f32,
    text: String,
    segments: Vec<SegmentJson>,
}

// ── Handlers ──────────────────────────────────────────────────────────────────

/// `POST /v1/audio/transcriptions` — Whisper STT.
#[instrument(skip_all, name = "audio_transcriptions", level = "info")]
pub async fn audio_transcriptions(State(state): State<AppState>, multipart: Multipart) -> Response {
    handle_audio(state, multipart, WhisperTask::Transcribe).await
}

/// `POST /v1/audio/translations` — Whisper STT with English output.
#[instrument(skip_all, name = "audio_translations", level = "info")]
pub async fn audio_translations(State(state): State<AppState>, multipart: Multipart) -> Response {
    handle_audio(state, multipart, WhisperTask::Translate).await
}

// ── Core handler ──────────────────────────────────────────────────────────────

#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "audio handler is a sequential pipeline: parse → validate → admit → spawn_blocking → metrics → build response; splitting would obscure the control flow"
)]
async fn handle_audio(state: AppState, mut multipart: Multipart, task: WhisperTask) -> Response {
    // 1. Parse multipart form.
    let fields = match parse_multipart(&mut multipart).await {
        Ok(f) => f,
        Err(msg) => {
            return (
                StatusCode::UNPROCESSABLE_ENTITY,
                Json(json!({"error": msg})),
            )
                .into_response();
        }
    };

    // 2. Validate audio size (25 MiB cap).
    const MAX_AUDIO_BYTES: usize = 25 * 1024 * 1024;
    if fields.audio_bytes.len() > MAX_AUDIO_BYTES {
        return (
            StatusCode::UNPROCESSABLE_ENTITY,
            Json(json!({"error": format!("audio file too large ({} bytes, max 25 MiB)", fields.audio_bytes.len())})),
        )
            .into_response();
    }

    info!(
        model = fields.model,
        lang = fields.language,
        audio_bytes = fields.audio_bytes.len(),
        task = ?task,
        "audio request"
    );

    // 3. Resolve model and tokenizer paths from AppState.
    let Some((model_path, tok_path)) = state.audio_paths() else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "audio model not configured; set --whisper-model-path"})),
        )
            .into_response();
    };

    // 4. C5 admission gate — audio holds the GPU; go through the same FIFO
    //    semaphore as LLM chat routes for fairness + 429 backpressure.
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

    // 5. Clone the audio_model cache Arc for the blocking closure.
    let audio_model_cache = Arc::clone(&state.audio_model);

    // 6. Move audio bytes (not clone) into the blocking closure.
    let audio_bytes = fields.audio_bytes;
    let language = fields.language.clone();
    let temperature = fields.temperature;
    let response_format = fields.response_format;

    // 7. Clone metrics sink for the blocking closure.
    let metrics_arc = state.metrics.clone();
    let model_path_str = model_path.to_string_lossy().into_owned();

    // NOTE: the per-request multimodal encoder cache (`state.mm_cache`) is not
    // used by the long-form path — the transcription engine re-encodes per 30 s
    // window internally, so a single full-file encoder-output cache entry no
    // longer applies. Per-window caching can be reintroduced inside the engine
    // if profiling shows it pays off.

    let result = tokio::task::spawn_blocking(move || {
        // Hold the GPU admission guard for the duration of Whisper decode.
        // Dropping it releases the semaphore permit and decrements gpu_pending.
        let _guard = guard;

        let device = Device::Gpu;

        // Registering a thread-local GPU stream + CommandEncoder once per thread entry point.
        // tokio blocking-pool threads start with no GPU stream context; MLX's array
        // materialisation then fails with "There is no Stream(gpu, 0) in current thread".
        // Mirrors the pattern used at the text and image generate entry points.
        // The CPU stream is registered unconditionally (thread-local since MLX
        // 0.31/0.32) so a CPU-scheduled op does not fault on this worker thread.
        rmlx_mlx::ensure_cpu_default_stream();
        if device == Device::Gpu {
            rmlx_mlx::ensure_gpu_default_stream();
        }

        // Resolve model + tokenizer from cache, loading on first call.
        //
        // Read path (fast): no allocation, just Arc::clone while holding the
        // read lock for as long as needed to extract the two Arcs.
        let t_load_start = std::time::Instant::now();
        let mut load_ms_opt: Option<f64> = None;

        let (model, tokenizer): (Arc<WhisperModel>, Arc<WhisperTokenizer>) = {
            // Try the read path first (no allocation on the fast path).
            let read_guard = audio_model_cache.read();
            if let Some((ref m, ref t)) = *read_guard {
                (Arc::clone(m), Arc::clone(t))
            } else {
                drop(read_guard);
                // Write path: load and populate the cache.
                let mut write_guard = audio_model_cache.write();
                // Double-checked: another thread may have loaded while we waited.
                if let Some((ref m, ref t)) = *write_guard {
                    (Arc::clone(m), Arc::clone(t))
                } else {
                    let m = WhisperModel::load(&model_path)
                        .map_err(|e| format!("whisper load: {e}"))?;
                    let t = WhisperTokenizer::from_path(&tok_path)
                        .map_err(|e| format!("tokenizer: {e}"))?;
                    let m = Arc::new(m);
                    let t = Arc::new(t);
                    *write_guard = Some((Arc::clone(&m), Arc::clone(&t)));
                    load_ms_opt = Some(t_load_start.elapsed().as_secs_f64() * 1_000.0);
                    (m, t)
                }
            }
        };

        // Decode audio → mono f32 at native rate, then resample to 16 kHz if needed.
        let (raw_samples, sample_rate) =
            WavDecoder::decode(&audio_bytes).map_err(|e| format!("audio decode: {e}"))?;
        let samples = rmlx_audio::transcribe::resample_to_16k(&raw_samples, sample_rate);

        // Audio duration in seconds (for RTF calculation).
        let audio_dur_secs = samples.len() as f64 / f64::from(rmlx_audio::mel::SAMPLE_RATE);

        // Build the long-form transcriber (shared engine with `rmlx transcribe`).
        let transcriber = Transcriber::new(Arc::clone(&model), Arc::clone(&tokenizer))
            .map_err(|e| format!("transcriber init: {e}"))?;

        let opts = TranscribeOptions {
            language: language.clone(),
            task,
            temperature,
            condition_on_previous_text: true,
        };

        let t_decode_start = std::time::Instant::now();
        let transcription = transcriber
            .transcribe(&samples, &opts, device)
            .map_err(|e| format!("transcribe: {e}"))?;
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

        // Emit metrics to the events DB.
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
            // Record load_ms only on first call (when we actually loaded the model).
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

        Ok::<rmlx_audio::transcribe::Transcription, String>(transcription)
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
        Ok(Err(msg)) => {
            warn!(error = msg, "transcription failed");
            if msg.contains("silence") {
                (StatusCode::OK, Json(json!({"text": ""}))).into_response()
            } else {
                (
                    StatusCode::UNPROCESSABLE_ENTITY,
                    Json(json!({"error": msg})),
                )
                    .into_response()
            }
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

// ── Multipart parser ──────────────────────────────────────────────────────────

async fn parse_multipart(multipart: &mut Multipart) -> Result<AudioFormFields, String> {
    let mut audio_bytes: Option<Vec<u8>> = None;
    let mut model = "whisper-large-v3".to_owned();
    // Default: "auto" triggers language detection via model.detect_language().
    // Explicit "en", "fr", etc. use that language directly.
    let mut language = "auto".to_owned();
    let mut response_format = ResponseFormat::Json;
    let mut temperature = 0.0_f32;

    while let Some(field) = multipart.next_field().await.map_err(|e| e.to_string())? {
        let name = field.name().unwrap_or("").to_owned();
        match name.as_str() {
            "file" => {
                let bytes = field.bytes().await.map_err(|e| e.to_string())?;
                audio_bytes = Some(bytes.to_vec());
            }
            "model" => {
                let text = field.text().await.map_err(|e| e.to_string())?;
                model = text;
            }
            "language" => {
                let text = field.text().await.map_err(|e| e.to_string())?;
                if text.is_empty() || text == "auto" {
                    language.clear();
                    language.push_str("auto");
                } else {
                    // Validate: must be a known language code.
                    let tok = rmlx_audio::tokenizer::language_token(&text);
                    // language_token falls back to English for unknown codes.
                    // Reject anything that resolves to EN when the caller passed
                    // a non-EN code (unknown language).
                    use rmlx_audio::tokenizer::TOK_EN;
                    if tok == TOK_EN && text != "en" {
                        return Err(format!(
                            "unknown language code '{text}'; pass 'auto' for language detection, \
                             or see Whisper language table for supported codes"
                        ));
                    }
                    language = text;
                }
            }
            "response_format" => {
                let text = field.text().await.map_err(|e| e.to_string())?;
                response_format = ResponseFormat::parse_or_default(&text)?;
            }
            "temperature" => {
                let text = field.text().await.map_err(|e| e.to_string())?;
                let t = text.parse::<f32>().map_err(|_| {
                    format!("invalid temperature '{text}'; must be a finite float in [0.0, 1.0]")
                })?;
                if !t.is_finite() || t < 0.0 {
                    return Err(format!(
                        "temperature {t} is out of range; must be a finite float in [0.0, 1.0]"
                    ));
                }
                temperature = t;
            }
            "prompt" => {
                // Accepted but ignored at v1.
                debug!("prompt field received (not yet implemented)");
                let _ = field.bytes().await;
            }
            other => {
                debug!(field = other, "unknown multipart field; ignoring");
                let _ = field.bytes().await;
            }
        }
    }

    let audio_bytes = audio_bytes.ok_or_else(|| "missing required field: file".to_owned())?;
    if audio_bytes.is_empty() {
        return Err("audio file is empty".to_owned());
    }

    Ok(AudioFormFields {
        audio_bytes,
        model,
        language,
        response_format,
        temperature,
    })
}

// ── Response builder ──────────────────────────────────────────────────────────

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
                        start: s.start,
                        end: s.end,
                        text: s.text.clone(),
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

// ── POST /v1/audio/speech (TTS — Phase 4b) ────────────────────────────────────
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
    let model_path_str = tts_model_path.to_string_lossy().into_owned();

    let result = tokio::task::spawn_blocking(move || {
        let _guard = guard; // hold GPU admission for the duration of synthesis

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
                    let m = rmlx_audio::tts::TtsModel::load_config(&tts_model_path, &tts_tok_path)
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
                [(axum::http::header::CONTENT_TYPE, content_type)],
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
