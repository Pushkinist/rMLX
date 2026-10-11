//! `POST /v1/audio/speech`: Qwen3-TTS speech synthesis.

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::Json;
use rmlx_audio::tts::{
    synthesize, Synthesis, TtsConfig, TtsError, TtsModel, TtsTokenizer, MAX_FRAMES,
};
use serde::Deserialize;
use serde_json::json;
use std::sync::Arc;
use tracing::{debug, error, info, instrument, warn};

use super::{cut, run_admitted, LOGGED_NAME_BYTES};
use crate::engine::{admit_request, Admission};
use crate::openai::state::ApiErrorCategory;
use crate::openai::AppState;
use crate::tts_sampling::{resolve_tts_sampling, TtsPicker, TtsSampling, TtsSamplingRequest};

/// Request body for `POST /v1/audio/speech`.
#[derive(Debug, Deserialize)]
pub struct SpeechRequest {
    model: String,
    input: String,
    #[serde(default = "default_voice")]
    voice: String,
    #[serde(default = "default_response_format_speech")]
    response_format: String,
    /// Accepted and ignored: the codec speed is fixed.
    #[serde(default = "default_speed")]
    #[allow(dead_code)]
    speed: f32,
    /// `temperature`, `top_k`, `repetition_penalty` and `seed`, as in `/v1/chat/completions`.
    #[serde(flatten)]
    sampling: TtsSamplingRequest,
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

/// Response headers that tell the client how the generation ended:
/// `x-rmlx-tts-frames` (codec frames generated) and `x-rmlx-tts-truncated`
/// (`true` when the loop stopped at the frame cap before the end token).
fn speech_outcome_headers(frames: usize, truncated: bool) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert("x-rmlx-tts-frames", axum::http::HeaderValue::from(frames));
    let truncated = if truncated { "true" } else { "false" };
    headers.insert(
        "x-rmlx-tts-truncated",
        axum::http::HeaderValue::from_static(truncated),
    );
    headers
}

/// The 200 response of `synthesis`: the audio as `response_format` (`pcm`: raw
/// f32 LE; else WAV) with the headers of [`speech_outcome_headers`]. A truncated
/// synthesis also logs a `warn` event.
fn speech_response(synthesis: &Synthesis, response_format: &str) -> Result<Response, String> {
    if synthesis.truncated {
        warn!(
            frames = synthesis.frames,
            cap = MAX_FRAMES,
            "TTS generation reached the frame cap before the end token; the audio is truncated"
        );
    }
    let (content_type, bytes): (_, Vec<u8>) = if response_format == "pcm" {
        let pcm = synthesis.samples.iter().flat_map(|s| s.to_le_bytes());
        ("audio/pcm", pcm.collect())
    } else {
        let wav = rmlx_audio::wav::WavEncoder::encode(&synthesis.samples, synthesis.sample_rate, 1)
            .map_err(|e| format!("wav encode: {e}"))?;
        ("audio/wav", wav)
    };
    Ok((
        StatusCode::OK,
        speech_outcome_headers(synthesis.frames, synthesis.truncated),
        [(header::CONTENT_TYPE, content_type)],
        bytes,
    )
        .into_response())
}

/// Why a speech request failed, as the client sees it.
#[derive(Debug)]
enum SpeechFailure {
    /// The model has no such voice. The message lists the voices of the model.
    UnknownVoice(String),
    /// A fixed message. The cause is in the log, because it can name a local path.
    Internal(&'static str),
}

impl SpeechFailure {
    /// Log `error` with the model path, and keep only `message` for the client.
    fn internal(
        message: &'static str,
        error: &dyn std::fmt::Display,
        model_path: &std::path::Path,
    ) -> Self {
        error!(
            error = %error,
            model_path = %model_path.display(),
            failure = message,
            "speech request failed"
        );
        Self::Internal(message)
    }

    /// The failure of a `synthesize` error. An unknown voice is the client's
    /// error: the reply lists the voices of `config` and does not repeat `voice`.
    fn from_synthesis(
        error: &TtsError,
        voice: &str,
        config: &TtsConfig,
        model_path: &std::path::Path,
    ) -> Self {
        match error {
            TtsError::UnknownVoice(_) => {
                info!(
                    voice_len = voice.len(),
                    voice = cut(voice, LOGGED_NAME_BYTES),
                    "speech request names an unknown voice"
                );
                let mut voices: Vec<&str> = config
                    .talker_config
                    .spk_id
                    .keys()
                    .map(String::as_str)
                    .collect();
                voices.sort_unstable();
                Self::UnknownVoice(format!(
                    "unknown voice; valid voices: {}",
                    voices.join(", ")
                ))
            }
            TtsError::Empty
            | TtsError::Load(_)
            | TtsError::Mlx(_)
            | TtsError::Tokenizer(_)
            | TtsError::Pick(_)
            | _ => Self::internal("speech synthesis failed", error, model_path),
        }
    }

    fn category(&self) -> ApiErrorCategory {
        match self {
            Self::UnknownVoice(_) => ApiErrorCategory::BadRequest,
            Self::Internal(_) => ApiErrorCategory::Internal,
        }
    }

    fn into_response(self) -> Response {
        match self {
            Self::UnknownVoice(message) => crate::openai::errors::error_response(
                StatusCode::UNPROCESSABLE_ENTITY,
                "invalid_request_error",
                &message,
            ),
            Self::Internal(message) => crate::openai::errors::internal_error(message),
        }
    }
}

/// The speech stack that the first `/v1/audio/speech` request loads.
#[derive(Debug)]
pub struct LoadedTts {
    /// `synthesize` takes `&mut TtsModel`: it loads the weights on its first call.
    model: parking_lot::Mutex<TtsModel>,
    tokenizer: TtsTokenizer,
    /// The snapshot's sampling defaults, from its `generation_config.json`.
    sampling: TtsSampling,
}

/// `POST /v1/audio/speech` — Qwen3-TTS speech synthesis.
///
/// Returns the audio (`audio/wav`, or `audio/pcm` for `response_format=pcm`)
/// with the headers of [`speech_outcome_headers`].
#[allow(clippy::cognitive_complexity)]
#[instrument(skip_all, name = "audio_speech", level = "info")]
pub async fn audio_speech(
    State(state): State<AppState>,
    Json(req): Json<SpeechRequest>,
) -> Response {
    info!(
        model = cut(&req.model, LOGGED_NAME_BYTES),
        model_len = req.model.len(),
        voice = cut(&req.voice, LOGGED_NAME_BYTES),
        voice_len = req.voice.len(),
        input_len = req.input.len(),
        "audio/speech request"
    );

    let refusal = if req.input.trim().is_empty() {
        Err("`input` must not be empty".to_owned())
    } else {
        req.sampling.check()
    };
    if let Err(message) = refusal {
        state.error_counts.increment(ApiErrorCategory::BadRequest);
        return crate::openai::errors::bad_request(&message);
    }

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

    // TTS holds the GPU: it queues in the FIFO of the chat routes.
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

    let tts_model_cache = Arc::clone(&state.tts_model);
    let tts_model_path = tts_model_path.clone();
    let tts_tok_path = tts_tok_path.clone();
    let voice = req.voice.clone();
    let input = req.input.clone();
    let response_format = req.response_format.clone();
    let request_sampling = req.sampling;
    let metrics_arc = state.metrics.clone();
    let device = state.device;
    let model_path_str = tts_model_path.to_string_lossy().into_owned();

    let result = run_admitted(guard, move |_cancel| {
        let t_load_start = std::time::Instant::now();
        let mut load_ms_opt: Option<f64> = None;

        let loaded: Arc<LoadedTts> = {
            let read_guard = tts_model_cache.read();
            if let Some(loaded) = read_guard.as_ref() {
                Arc::clone(loaded)
            } else {
                drop(read_guard);
                let mut write_guard = tts_model_cache.write();
                if let Some(loaded) = write_guard.as_ref() {
                    Arc::clone(loaded)
                } else {
                    let failed = |message, e: &dyn std::fmt::Display| {
                        SpeechFailure::internal(message, e, &tts_model_path)
                    };
                    let sampling =
                        resolve_tts_sampling(&tts_model_path, &TtsSamplingRequest::default())
                            .map_err(|e| {
                                failed("speech sampling config of the model could not be read", &e)
                            })?;
                    let model = TtsModel::load_config(&tts_model_path, &tts_tok_path, device)
                        .map_err(|e| failed("speech model config could not be read", &e))?;
                    // Text tokenizer lives in the talker model snapshot (vocab.json +
                    // merges.txt), not in the codec-decoder path.
                    let tokenizer = TtsTokenizer::from_path(&tts_model_path)
                        .map_err(|e| failed("speech tokenizer could not be read", &e))?;
                    let loaded = Arc::new(LoadedTts {
                        model: parking_lot::Mutex::new(model),
                        tokenizer,
                        sampling,
                    });
                    *write_guard = Some(Arc::clone(&loaded));
                    load_ms_opt = Some(t_load_start.elapsed().as_secs_f64() * 1_000.0);
                    loaded
                }
            }
        };

        let sampling = loaded.sampling.clone().with_request(&request_sampling);
        debug!(
            talker_temperature = sampling.talker.sampler.temperature,
            talker_top_k = sampling.talker.sampler.top_k,
            talker_repetition_penalty = sampling.talker.penalty.rep_penalty,
            code_predictor_temperature = sampling.code_predictor.sampler.temperature,
            code_predictor_top_k = sampling.code_predictor.sampler.top_k,
            seed = sampling.talker.sampler.seed_or_default(),
            "TTS sampling resolved"
        );
        let mut picker = TtsPicker::new(sampling, device);

        let t_synth_start = std::time::Instant::now();
        let synthesis = {
            let mut model = loaded.model.lock();
            synthesize(&input, &voice, &mut model, &loaded.tokenizer, &mut picker).map_err(|e| {
                SpeechFailure::from_synthesis(&e, &voice, &model.config, &tts_model_path)
            })?
        };
        let synth_ms = t_synth_start.elapsed().as_secs_f64() * 1_000.0;
        let samples = synthesis.samples.as_slice();
        let sample_rate = synthesis.sample_rate;

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

        let response = speech_response(&synthesis, &response_format).map_err(|e| {
            SpeechFailure::internal("speech audio could not be encoded", &e, &tts_model_path)
        })?;

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

        Ok::<_, SpeechFailure>(response)
    })
    .await;

    match result {
        Ok(Ok(response)) => response,
        Ok(Err(failure)) => {
            state.error_counts.increment(failure.category());
            failure.into_response()
        }
        Err(e) => {
            warn!(error = %e, "TTS task panicked");
            state.error_counts.increment(ApiErrorCategory::Internal);
            crate::openai::errors::internal_error("internal error")
        }
    }
}

#[cfg(test)]
#[path = "speech_tests.rs"]
mod speech_tests;
