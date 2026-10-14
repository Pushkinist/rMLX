//! The speech route: the request body, the refusals before a model is needed,
//! the failure answers and the response headers that report how the
//! generation ended.

use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::http::StatusCode;
use axum::Json;

use rmlx_audio::tts::{Synthesis, TtsConfig, TtsError, MAX_FRAMES};

use super::{audio_speech, speech_outcome_headers, speech_response, SpeechFailure, SpeechRequest};
use crate::openai::tests::slot_test_state;
use crate::registry::ModelRegistry;

fn header<'a>(h: &'a axum::http::HeaderMap, name: &str) -> &'a str {
    h.get(name)
        .unwrap_or_else(|| panic!("missing header {name}"))
        .to_str()
        .unwrap()
}

#[test]
fn a_truncated_generation_is_reported() {
    let h = speech_outcome_headers(2048, true);
    assert_eq!(header(&h, "x-rmlx-tts-frames"), "2048");
    assert_eq!(header(&h, "x-rmlx-tts-truncated"), "true");
}

#[test]
fn a_finished_generation_is_reported() {
    let h = speech_outcome_headers(84, false);
    assert_eq!(header(&h, "x-rmlx-tts-frames"), "84");
    assert_eq!(header(&h, "x-rmlx-tts-truncated"), "false");
}

#[test]
fn a_speech_request_carries_the_sampling_fields() {
    let req: SpeechRequest = serde_json::from_str(
        r#"{"model": "tts-1", "input": "hello", "voice": "ryan",
            "temperature": 0.3, "top_k": 10, "repetition_penalty": 1.2, "seed": 7}"#,
    )
    .unwrap();
    assert_eq!(req.input, "hello");
    assert_eq!(req.voice, "ryan");
    assert_eq!(req.sampling.temperature, Some(0.3));
    assert_eq!(req.sampling.top_k, Some(10));
    assert_eq!(req.sampling.repetition_penalty, Some(1.2));
    assert_eq!(req.sampling.seed, Some(7));
}

#[test]
fn a_speech_request_without_sampling_fields_leaves_them_unset() {
    let req: SpeechRequest =
        serde_json::from_str(r#"{"model": "tts-1", "input": "hello"}"#).unwrap();
    assert_eq!(req.sampling.temperature, None);
    assert_eq!(req.sampling.top_k, None);
    assert_eq!(req.sampling.repetition_penalty, None);
    assert_eq!(req.sampling.seed, None);
}

/// Status and body bytes of one speech request against `state`.
async fn speak_raw(state: crate::openai::AppState, body: &str) -> (StatusCode, Vec<u8>) {
    let req: SpeechRequest = serde_json::from_str(body).unwrap();
    let resp = audio_speech(State(state), Json(req)).await;
    let status = resp.status();
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    (status, bytes.to_vec())
}

/// Status and JSON body of one speech request against `state`.
async fn speak(state: crate::openai::AppState, body: &str) -> (StatusCode, serde_json::Value) {
    let (status, bytes) = speak_raw(state, body).await;
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn no_tts_state() -> crate::openai::AppState {
    slot_test_state(ModelRegistry::from_paths(&[]), 1)
}

/// A sampling field out of range is refused before the route looks for a
/// model; a valid request reaches the next check (no TTS configured: 503).
async fn assert_refused(field: &str, value: &str) {
    let body = format!(r#"{{"model":"tts","input":"hello","{field}":{value}}}"#);
    let state = no_tts_state();
    let (status, json) = speak(state.clone(), &body).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{field}={value}: {json}");
    assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
    let message = json["error"]["message"].as_str().unwrap();
    assert!(message.contains(field), "{field}={value}: {message}");
    assert_eq!(
        state.error_counts.bad_request.load(Ordering::Relaxed),
        1,
        "{field}={value}: the refusal is counted once"
    );
}

/// A request that passes validation reaches the next check: 503, no TTS
/// configured.
async fn assert_accepted(fields: &str) {
    let body = format!(r#"{{"model":"tts","input":"hello",{fields}}}"#);
    let state = no_tts_state();
    let (status, json) = speak(state.clone(), &body).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE, "{fields}: {json}");
    assert_eq!(state.error_counts.bad_request.load(Ordering::Relaxed), 0);
}

#[tokio::test]
async fn a_valid_sampling_request_passes_validation() {
    assert_accepted(r#""temperature":2.0,"top_k":0,"repetition_penalty":0.5,"seed":0"#).await;
}

/// `top_p` is not a request field: any value is ignored.
#[tokio::test]
async fn a_top_p_in_the_request_is_ignored() {
    assert_accepted(r#""top_p":9.0"#).await;
}

/// `-0.0` is 0: greedy decoding.
#[tokio::test]
async fn a_negative_zero_temperature_is_greedy() {
    assert_accepted(r#""temperature":-0.0"#).await;
}

#[tokio::test]
async fn a_temperature_just_outside_the_bounds_is_refused() {
    assert_refused("temperature", "-0.1").await;
    assert_refused("temperature", "2.1").await;
}

/// The floor of `temperature` and of `repetition_penalty` is 1e-6: a smaller
/// value takes a logit out of the f32 range long before it is subnormal.
#[tokio::test]
async fn a_value_below_the_floor_is_refused() {
    let smallest_normal = format!("{:e}", f32::MIN_POSITIVE);
    let largest_subnormal = format!("{:e}", f32::from_bits(0x007f_ffff));
    for field in ["temperature", "repetition_penalty"] {
        for value in [
            "9.9e-7",
            smallest_normal.as_str(),
            largest_subnormal.as_str(),
        ] {
            assert_refused(field, value).await;
        }
    }
}

#[tokio::test]
async fn a_refusal_states_the_range() {
    for (field, range) in [
        ("temperature", "[1e-6, 2.0]"),
        ("repetition_penalty", "at least 1e-6"),
    ] {
        let body = format!(r#"{{"model":"tts","input":"hello","{field}":9.9e-7}}"#);
        let (_, json) = speak(no_tts_state(), &body).await;
        let message = json["error"]["message"].as_str().unwrap();
        assert!(message.contains(range), "{field}: {message}");
    }
}

#[tokio::test]
async fn the_ends_of_the_bounds_are_accepted() {
    assert_accepted(r#""temperature":1e-6,"repetition_penalty":1e-6"#).await;
    assert_accepted(r#""temperature":2.0"#).await;
}

/// The next f32 above 2.0.
#[tokio::test]
async fn a_temperature_one_step_above_two_is_refused() {
    assert_refused(
        "temperature",
        &format!("{}", f32::from_bits(2.0f32.to_bits() + 1)),
    )
    .await;
}

/// A value that rounds to zero in f32 is zero: greedy decoding.
#[tokio::test]
async fn a_temperature_that_rounds_to_zero_is_greedy() {
    assert_accepted(r#""temperature":-1e-50"#).await;
    assert_accepted(r#""temperature":1e-50"#).await;
}

/// An empty input is refused before the route looks for a model.
#[tokio::test]
async fn an_empty_input_is_refused() {
    for input in ["", "   ", r"\n\t"] {
        let body = format!(r#"{{"model":"tts","input":"{input}"}}"#);
        let state = no_tts_state();
        let (status, json) = speak(state.clone(), &body).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{input:?}: {json}");
        assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
        assert_eq!(json["error"]["message"], "`input` must not be empty");
        assert_eq!(state.error_counts.bad_request.load(Ordering::Relaxed), 1);
    }
}

/// A subnormal temperature divides the logits into infinity.
#[tokio::test]
async fn a_subnormal_temperature_is_refused() {
    assert_refused("temperature", "1e-40").await;
    assert_refused("temperature", "1e-38").await;
}

/// A subnormal penalty multiplies a repeated id to the top of every row.
#[tokio::test]
async fn a_subnormal_repetition_penalty_is_refused() {
    assert_refused("repetition_penalty", "1e-40").await;
}

#[tokio::test]
async fn a_negative_temperature_is_refused() {
    assert_refused("temperature", "-1").await;
}

#[tokio::test]
async fn a_temperature_above_two_is_refused() {
    assert_refused("temperature", "2.5").await;
}

/// 1e39 is out of the f32 range and parses as infinity.
#[tokio::test]
async fn an_infinite_temperature_is_refused() {
    assert_refused("temperature", "1e39").await;
}

#[tokio::test]
async fn a_zero_repetition_penalty_is_refused() {
    assert_refused("repetition_penalty", "0").await;
}

#[tokio::test]
async fn a_negative_repetition_penalty_is_refused() {
    assert_refused("repetition_penalty", "-1").await;
}

#[tokio::test]
async fn an_infinite_repetition_penalty_is_refused() {
    assert_refused("repetition_penalty", "1e39").await;
}

/// `top_k` and `seed` are unsigned, as in `/v1/chat/completions`: the JSON parser
/// refuses a negative value.
#[test]
fn a_negative_top_k_or_seed_does_not_parse() {
    for field in ["top_k", "seed"] {
        let body = format!(r#"{{"model":"tts","input":"hello","{field}":-1}}"#);
        assert!(
            serde_json::from_str::<SpeechRequest>(&body).is_err(),
            "{field}"
        );
    }
}

/// A talker `config.json` that `TtsConfig` parses.
const TALKER_CONFIG: &str = r#"{
    "model_type": "qwen3_tts",
    "tts_bos_token_id": 151672,
    "tts_eos_token_id": 151673,
    "tts_pad_token_id": 151671,
    "talker_config": {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,
        "num_hidden_layers": 28,
        "num_code_groups": 16,
        "codec_bos_id": 2149,
        "codec_eos_token_id": 2150,
        "codec_language_id": {},
        "spk_id": {"serena": 3066}
    }
}"#;

/// An empty BPE `tokenizer.json`.
const TOKENIZER: &str = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{"type":"BPE","dropout":null,"unk_token":null,"continuing_subword_prefix":null,"end_of_word_suffix":null,"fuse_unk":false,"byte_fallback":false,"vocab":{},"merges":[]}}"#;

/// A snapshot directory that holds `files`, and a state whose TTS paths name it.
fn snapshot_state(files: &[(&str, &str)]) -> (tempfile::TempDir, crate::openai::AppState) {
    let snapshot = tempfile::tempdir().unwrap();
    for (name, body) in files {
        std::fs::write(snapshot.path().join(name), body).unwrap();
    }
    let mut state = no_tts_state();
    state.tts_model_path = Some(snapshot.path().to_path_buf());
    state.tts_tokenizer_path = Some(snapshot.path().to_path_buf());
    (snapshot, state)
}

/// The bounds check runs before admission: while another job holds the one
/// GPU permit, a request with a value out of bounds is 400 at once and takes
/// no slot of the queue.
#[tokio::test]
async fn a_value_out_of_bounds_is_refused_before_admission() {
    let (_snapshot, state) = snapshot_state(&[]);
    let _busy = Arc::clone(&state.gpu_queue)
        .acquire_owned()
        .await
        .expect("the queue is open");

    let body = r#"{"model":"tts","input":"hello","temperature":3.0}"#;
    let (status, json) = tokio::time::timeout(Duration::from_secs(2), speak(state.clone(), body))
        .await
        .expect("the refusal does not wait for the GPU permit");
    assert_eq!(status, StatusCode::BAD_REQUEST, "{json}");
    assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
    assert_eq!(
        state.gpu_pending.load(Ordering::Relaxed),
        0,
        "the refused request holds no slot of the queue"
    );
}

/// The first request against a snapshot directory that holds `files` fails
/// with 500, the fixed `message`, the error envelope, and no local detail.
async fn assert_internal_failure(files: &[(&str, &str)], message: &str) {
    let (snapshot, state) = snapshot_state(files);
    let (events, guards) = capture();
    let (status, json) = speak(state.clone(), r#"{"model":"tts","input":"hello"}"#).await;
    drop(guards);
    assert!(
        problems(&events).is_empty(),
        "the failure is logged once, on the thread that saw the cause: {:?}",
        problems(&events)
    );
    assert_eq!(status, StatusCode::INTERNAL_SERVER_ERROR, "{json}");
    assert_eq!(json["error"]["type"], "internal_error", "{json}");
    assert_eq!(json["error"]["message"], message, "{json}");
    let text = json.to_string();
    let dir = snapshot.path().to_string_lossy();
    for local in [dir.as_ref(), "/", ".json", "No such file", "os error"] {
        assert!(!text.contains(local), "the body has {local:?}: {text}");
    }
    assert_eq!(state.error_counts.internal.load(Ordering::Relaxed), 1);
    assert_eq!(state.error_counts.bad_request.load(Ordering::Relaxed), 0);
}

/// An unknown voice is 422 in the envelope. The body lists the voices of the
/// model and does not repeat the voice of the request, whatever its length.
#[tokio::test]
async fn an_unknown_voice_is_refused_without_an_echo() {
    for voice in ["nobody".to_owned(), "n".repeat(100_000)] {
        let (_snapshot, state) = snapshot_state(&[
            ("config.json", TALKER_CONFIG),
            ("tokenizer.json", TOKENIZER),
        ]);
        let body = format!(r#"{{"model":"tts","input":"hello","voice":"{voice}"}}"#);
        let (events, guards) = capture();
        let (status, json) = speak(state.clone(), &body).await;
        drop(guards);
        assert!(
            problems(&events).is_empty(),
            "an unknown voice is not a server problem: {:?}",
            problems(&events)
        );
        for event in events.0.lock().unwrap().iter() {
            let logged = field(event, "voice").unwrap_or_default();
            assert!(logged.len() <= 66, "a log field has {} bytes", logged.len());
        }
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{json}");
        assert_eq!(json["error"]["type"], "invalid_request_error", "{json}");
        let text = json.to_string();
        assert!(text.len() < 1024, "the body has {} bytes", text.len());
        assert!(!text.contains("nobody") && !text.contains("nnnn"), "{text}");
        assert_eq!(
            json["error"]["message"], "unknown voice; valid voices: serena",
            "{json}"
        );
        assert_eq!(state.error_counts.bad_request.load(Ordering::Relaxed), 1);
        assert_eq!(state.error_counts.internal.load(Ordering::Relaxed), 0);
    }
}

/// An unparseable `generation_config.json` refuses the request with a
/// message that names no file system path.
#[tokio::test]
async fn an_unreadable_sampling_config_is_refused_without_a_path() {
    assert_internal_failure(
        &[("generation_config.json", "{ not json }")],
        "speech sampling config of the model could not be read",
    )
    .await;
}

/// A value out of bounds in the file takes the same refusal.
#[tokio::test]
async fn a_sampling_config_out_of_bounds_is_refused_without_a_path() {
    assert_internal_failure(
        &[("generation_config.json", r#"{"temperature": -3.0}"#)],
        "speech sampling config of the model could not be read",
    )
    .await;
}

#[tokio::test]
async fn a_missing_model_config_is_refused_without_a_path() {
    assert_internal_failure(&[], "speech model config could not be read").await;
}

#[tokio::test]
async fn a_missing_tokenizer_is_refused_without_a_path() {
    assert_internal_failure(
        &[("config.json", TALKER_CONFIG)],
        "speech tokenizer could not be read",
    )
    .await;
}

/// The weights load inside the first synthesis; a snapshot without them
/// fails there.
#[tokio::test]
async fn missing_weights_are_refused_without_a_path() {
    assert_internal_failure(
        &[
            ("config.json", TALKER_CONFIG),
            ("tokenizer.json", TOKENIZER),
        ],
        "speech synthesis failed",
    )
    .await;
}

/// `frames` frames of samples that differ from each other and from zero.
fn synthesis(frames: usize, truncated: bool) -> Synthesis {
    Synthesis {
        samples: (0..frames * 1920)
            .map(|i| ((i % 201) as f32 - 100.0) / 100.0)
            .collect(),
        sample_rate: 24_000,
        frames,
        truncated,
    }
}

/// The 200 response carries the audio, its content type, the frame count and
/// the truncation flag of the synthesis.
#[tokio::test]
async fn the_response_carries_the_audio_and_the_synthesis_outcome() {
    for (format, content_type, truncated) in [
        ("wav", "audio/wav", true),
        ("wav", "audio/wav", false),
        ("pcm", "audio/pcm", true),
    ] {
        let source = synthesis(3, truncated);
        let resp = speech_response(&source, format).unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let h = resp.headers().clone();
        assert_eq!(header(&h, "content-type"), content_type);
        assert_eq!(header(&h, "x-rmlx-tts-frames"), "3");
        assert_eq!(
            header(&h, "x-rmlx-tts-truncated"),
            if truncated { "true" } else { "false" }
        );
        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let want: Vec<u8> = if format == "pcm" {
            source
                .samples
                .iter()
                .flat_map(|s| s.to_le_bytes())
                .collect()
        } else {
            wav_pcm16_mono_24k(&source.samples)
        };
        assert!(
            body == want,
            "{format}: the body differs from the expected bytes"
        );
    }
}

/// A 44-byte RIFF header for mono PCM-16 at 24 kHz, then each sample as
/// `round(s * 32767)` in little-endian `i16`.
fn wav_pcm16_mono_24k(samples: &[f32]) -> Vec<u8> {
    let data = (samples.len() * 2) as u32;
    let mut out = Vec::new();
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(data + 36).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&24_000u32.to_le_bytes());
    out.extend_from_slice(&48_000u32.to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&((s * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

/// The fields of every event that this thread emits while it is installed.
#[derive(Default)]
struct Events(std::sync::Mutex<Vec<Vec<(String, String)>>>);

struct Fields(Vec<(String, String)>);

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.push((field.name().to_owned(), format!("{value:?}")));
    }
}

impl tracing::Subscriber for Events {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut fields = Fields(vec![(
            "level".to_owned(),
            event.metadata().level().to_string(),
        )]);
        event.record(&mut fields);
        self.0.lock().unwrap().push(fields.0);
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// Capture the events of this thread until the returned guards drop.
fn capture() -> (
    Arc<Events>,
    (tracing::Dispatch, tracing::subscriber::DefaultGuard),
) {
    let events = Arc::new(Events::default());
    // A second live dispatcher keeps a callsite that another thread registers
    // now from being cached as always on, which would bypass the scoped one.
    let keep = tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default());
    let guard = tracing::subscriber::set_default(Arc::clone(&events));
    (events, (keep, guard))
}

/// The events of `f`, each as its level and its fields.
fn events_of(f: impl FnOnce()) -> Vec<Vec<(String, String)>> {
    let (events, guards) = capture();
    f();
    drop(guards);
    let captured = events.0.lock().unwrap().clone();
    captured
}

/// The events at warn or error among `events`.
fn problems(events: &Events) -> Vec<Vec<(String, String)>> {
    let all = events.0.lock().unwrap();
    all.iter()
        .filter(|e| matches!(field(e, "level"), Some("WARN" | "ERROR")))
        .cloned()
        .collect()
}

fn field<'a>(event: &'a [(String, String)], name: &str) -> Option<&'a str> {
    event
        .iter()
        .find(|(key, _)| key == name)
        .map(|(_, value)| value.as_str())
}

#[test]
fn a_truncated_synthesis_logs_a_warn_event_with_the_frames_and_the_cap() {
    let events = events_of(|| {
        speech_response(&synthesis(5, true), "pcm").unwrap();
    });
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(field(&events[0], "level"), Some("WARN"));
    assert_eq!(field(&events[0], "frames"), Some("5"));
    assert_eq!(
        field(&events[0], "cap"),
        Some(MAX_FRAMES.to_string().as_str())
    );

    let finished = events_of(|| {
        speech_response(&synthesis(5, false), "pcm").unwrap();
    });
    assert!(finished.is_empty(), "{finished:?}");
}

/// The detail that the client does not get is in the log.
#[test]
fn an_internal_failure_logs_the_error_and_the_model_path() {
    let events = events_of(|| {
        let failure = SpeechFailure::internal(
            "speech model config could not be read",
            &"config.json: no such file",
            std::path::Path::new("snapshots/talker"),
        );
        assert!(matches!(
            failure,
            SpeechFailure::Internal("speech model config could not be read")
        ));
    });
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(field(&events[0], "level"), Some("ERROR"));
    assert_eq!(
        field(&events[0], "error"),
        Some("config.json: no such file")
    );
    assert_eq!(field(&events[0], "model_path"), Some("snapshots/talker"));
}

/// Every synthesis error but an unknown voice is one fixed message: a pick
/// error, an empty text and an MLX error carry detail that stays in the log.
#[test]
fn a_synthesis_error_keeps_its_detail_out_of_the_reply() {
    let config = TtsConfig::from_json(TALKER_CONFIG).unwrap();
    let path = std::path::Path::new("snapshots/talker");
    for error in [
        TtsError::Pick("the talker picked id 2500, which the mask blocks".to_owned()),
        TtsError::Mlx("quantized_matmul failed".to_owned()),
        TtsError::Load("snapshots/talker/model.safetensors: not found".to_owned()),
        TtsError::Empty,
    ] {
        let events = events_of(|| {
            let failure = SpeechFailure::from_synthesis(&error, "serena", &config, path);
            assert!(
                matches!(failure, SpeechFailure::Internal("speech synthesis failed")),
                "{error:?}: {failure:?}"
            );
        });
        assert_eq!(events.len(), 1, "{error:?}: one event, {events:?}");
        assert_eq!(field(&events[0], "level"), Some("ERROR"));
        assert_eq!(field(&events[0], "error"), Some(error.to_string().as_str()));
    }
}

/// An unknown voice is a client error: one info event with the length of the
/// voice and at most its first 64 bytes, cut at a character boundary.
#[test]
fn an_unknown_voice_logs_one_bounded_info_event() {
    let config = TtsConfig::from_json(TALKER_CONFIG).unwrap();
    let voice = "é".repeat(1000);
    let events = events_of(|| {
        let failure = SpeechFailure::from_synthesis(
            &TtsError::UnknownVoice(voice.clone()),
            &voice,
            &config,
            std::path::Path::new("snapshots/talker"),
        );
        match failure {
            SpeechFailure::UnknownVoice(message) => {
                assert_eq!(message, "unknown voice; valid voices: serena");
            }
            SpeechFailure::Internal(message) => panic!("expected an unknown voice: {message}"),
            SpeechFailure::Stopped => panic!("expected an unknown voice, got a stop"),
        }
    });
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(field(&events[0], "level"), Some("INFO"));
    assert_eq!(field(&events[0], "voice_len"), Some("2000"));
    let logged = field(&events[0], "voice").unwrap();
    assert_eq!(
        logged.trim_matches('"'),
        "é".repeat(32),
        "64 bytes of a 2-byte character"
    );
}

/// The request event keeps the length of the model name and at most its first
/// 64 bytes, as the transcription route does: a client can send a megabyte.
#[tokio::test]
async fn a_long_model_name_is_logged_cut_to_64_bytes() {
    let (events, guards) = capture();
    let body = format!(
        r#"{{"model": "{}", "input": "hello"}}"#,
        "m".repeat(1 << 20)
    );
    let (status, _) = speak(no_tts_state(), &body).await;
    drop(guards);
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    let captured = events.0.lock().unwrap().clone();
    let request = captured
        .iter()
        .find(|e| field(e, "message") == Some("audio/speech request"))
        .expect("the request event");
    assert_eq!(
        field(request, "model"),
        Some(format!("{:?}", "m".repeat(64)).as_str())
    );
    assert_eq!(field(request, "model_len"), Some("1048576"));
}

/// Without `--tts-model-path` the route is 503 `not_supported`, in the error
/// envelope with `message` before `type`.
#[tokio::test]
async fn an_unconfigured_route_is_503_not_supported() {
    let (status, bytes) = speak_raw(no_tts_state(), r#"{"model":"tts","input":"hello"}"#).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        bytes,
        br#"{"error":{"message":"TTS not configured; set --tts-model-path and --tts-tokenizer-path","type":"not_supported"}}"#
    );
}

/// A full queue is 429 `rate_limit_error`, in the error envelope with
/// `message` before `type`.
#[tokio::test]
async fn a_full_queue_is_429_rate_limit_error() {
    let (_snapshot, mut state) = snapshot_state(&[]);
    state.max_queue_depth = 1;
    state.gpu_pending.store(1, Ordering::Release);
    let (status, bytes) = speak_raw(state, r#"{"model":"tts","input":"hello"}"#).await;
    assert_eq!(status, StatusCode::TOO_MANY_REQUESTS);
    assert_eq!(
        bytes,
        br#"{"error":{"message":"server queue full","type":"rate_limit_error"}}"#
    );
}

/// The error of a synthesis that stopped because its request is gone: the
/// error that the stop helper returns for a set flag, as a synthesis hands it on.
fn the_stop_of_a_synthesis() -> Option<TtsError> {
    rmlx_core::cancel::ensure_not_cancelled(&std::sync::atomic::AtomicBool::new(true))
        .err()
        .map(TtsError::from)
}

/// A stopped synthesis is not a failure of the server: the route logs no
/// error for it and does not count it as an internal error.
#[test]
fn a_stopped_synthesis_is_not_a_failure() {
    let Some(stop) = the_stop_of_a_synthesis() else {
        panic!("a synthesis has no stop outcome");
    };
    let config = TtsConfig::from_json(TALKER_CONFIG).unwrap();
    let path = std::path::Path::new("snapshots/talker");
    let events = events_of(|| {
        let failure = SpeechFailure::from_synthesis(&stop, "serena", &config, path);
        assert!(matches!(failure, SpeechFailure::Stopped), "{failure:?}");
        assert!(
            failure.category().is_none(),
            "a stop is counted as an error"
        );
    });
    assert_eq!(events.len(), 1, "{events:?}");
    assert_eq!(field(&events[0], "level"), Some("INFO"));
    assert_eq!(field(&events[0], "emitted"), Some("0"));
}
