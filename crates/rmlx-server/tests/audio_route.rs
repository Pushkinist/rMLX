//! The transcription route's answer to each form field and each upload size,
//! on a server whose Whisper snapshot paths do not exist: a request the route
//! takes ends at the tokenizer load, after the audio decode.
//!
//! Oracle: the OpenAI `POST /v1/audio/transcriptions` reference
//! (`CreateTranscriptionRequest`): the field names and values a client sends,
//! and the 413 OpenAI answers for a file over its 26214400-byte cap.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stderr,
    reason = "test assertions, and each case prints the status and body it got"
)]

mod stub_server;
#[allow(
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    reason = "the helpers of the model tests, written for test files that allow these"
)]
mod thread_boundary;

use rmlx_audio::tokenizer::{WhisperTask, WhisperTokenizer};
use rmlx_audio::transcribe::{TranscribeOptions, Transcriber};
use rmlx_audio::wav::{WavDecoder, WavEncoder, MAX_AUDIO_SECS};
use rmlx_audio::whisper::WhisperModel;
use rmlx_mlx::Device;
use rmlx_models::sampler::RowPicker;
use rmlx_server::{AppState, ModelRegistry};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use stub_server::{not_ready_state, start_server_with_state};
use thread_boundary::{say_voice, snapshot, WHISPER, WHISPER_TOKENIZER};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

/// The OpenAI audio file cap, 25 MiB.
const CAP: usize = 26_214_400;

/// A state whose Whisper snapshot and tokenizer directories do not exist, and
/// the directory that holds them.
fn missing_whisper_state() -> (AppState, tempfile::TempDir) {
    let dir = tempfile::tempdir().unwrap();
    let mut state = not_ready_state(ModelRegistry::default());
    state.whisper_model_path = Some(dir.path().join("no-model"));
    state.whisper_tokenizer_path = Some(dir.path().join("no-tokenizer"));
    (state, dir)
}

/// One second of 16 kHz mono silence: a file the decoder takes.
fn short_wav() -> Vec<u8> {
    WavEncoder::encode(&vec![0.0_f32; 16_000], 16_000, 1).unwrap()
}

/// A multipart form of `parts`, each a field name and its bytes; the `file`
/// part carries a file name.
fn form(parts: &[(&str, &[u8])]) -> Vec<u8> {
    let mut body = Vec::new();
    for (name, value) in parts {
        let filename = if *name == "file" {
            "; filename=\"a.wav\""
        } else {
            ""
        };
        body.extend_from_slice(
            format!(
                "--XBOUNDARY\r\nContent-Disposition: form-data; name=\"{name}\"{filename}\r\n\r\n"
            )
            .as_bytes(),
        );
        body.extend_from_slice(value);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(b"--XBOUNDARY--\r\n");
    body
}

/// The `Content-Type` line of every form `form` builds.
const FORM_TYPE: &str = "Content-Type: multipart/form-data; boundary=XBOUNDARY\r\n";

const TRANSCRIPTIONS: &str = "/v1/audio/transcriptions";
const TRANSLATIONS: &str = "/v1/audio/translations";

/// What the server answered.
struct Answer {
    status: u16,
    /// The status line and the header lines.
    head: String,
    body: String,
}

/// Send a transcription request with the header lines `headers` (each ending
/// in CRLF) and `body`. The body is written while the answer is read: the
/// server may answer before it has read the whole upload, and a write it
/// refuses is not an error here.
async fn exchange(port: u16, headers: &str, body: Vec<u8>) -> Answer {
    exchange_to(port, TRANSCRIPTIONS, headers, body).await
}

/// `exchange` to the route `path`.
async fn exchange_to(port: u16, path: &str, headers: &str, body: Vec<u8>) -> Answer {
    let head = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
         {headers}\r\n"
    );
    let stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let (mut reader, mut writer) = stream.into_split();
    // The task hands the write half back: dropping it shuts the write side,
    // and the server drops a request whose client half-closed.
    let write = tokio::spawn(async move {
        if writer.write_all(head.as_bytes()).await.is_ok() {
            writer.write_all(&body).await.ok();
        }
        writer
    });
    let mut response = Vec::new();
    reader.read_to_end(&mut response).await.ok();
    drop(write.await.unwrap());
    let text = String::from_utf8_lossy(&response).into_owned();
    let status = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let head_end = text.find("\r\n\r\n").unwrap_or(text.len());
    let body_start = (head_end + 4).min(text.len());
    Answer {
        status,
        head: text[..head_end].to_owned(),
        body: text[body_start..].to_owned(),
    }
}

/// POST the form `parts` with its `Content-Length`.
async fn post_form(port: u16, parts: &[(&str, &[u8])]) -> Answer {
    let body = form(parts);
    let headers = format!("{FORM_TYPE}Content-Length: {}\r\n", body.len());
    exchange(port, &headers, body).await
}

/// POST the form `parts` to the route `path`.
async fn post_form_to(port: u16, path: &str, parts: &[(&str, &[u8])]) -> Answer {
    let body = form(parts);
    let headers = format!("{FORM_TYPE}Content-Length: {}\r\n", body.len());
    exchange_to(port, path, &headers, body).await
}

/// The `error` object of an answer, which must be the OpenAI envelope: JSON,
/// an `error` object with a string `message`, the string `type` given, a
/// `param` equal to the one given (absent or null when `None`), and no other
/// key. Returns the message.
fn envelope(answer: &Answer, status: u16, error_type: &str, param: Option<&str>) -> String {
    assert_eq!(answer.status, status, "{}", answer.body);
    let json: serde_json::Value = serde_json::from_str(&answer.body)
        .unwrap_or_else(|e| panic!("not JSON ({e}): {}", answer.body));
    let error = json
        .get("error")
        .and_then(serde_json::Value::as_object)
        .unwrap_or_else(|| panic!("`error` is not an object: {}", answer.body));
    assert_eq!(json.as_object().map(serde_json::Map::len), Some(1));
    assert_eq!(
        error.get("type").and_then(serde_json::Value::as_str),
        Some(error_type),
        "{}",
        answer.body
    );
    assert_eq!(
        error.get("param").and_then(serde_json::Value::as_str),
        param,
        "{}",
        answer.body
    );
    for key in error.keys() {
        assert!(
            ["message", "type", "param"].contains(&key.as_str()),
            "unexpected key {key}: {}",
            answer.body
        );
    }
    assert!(
        answer
            .head
            .to_ascii_lowercase()
            .contains("\r\ncontent-type: application/json"),
        "{}",
        answer.head
    );
    error
        .get("message")
        .and_then(serde_json::Value::as_str)
        .unwrap_or_else(|| panic!("no string `message`: {}", answer.body))
        .to_owned()
}

/// POST `audio` as the only field of the form.
async fn post_audio(port: u16, audio: &[u8]) -> Answer {
    post_form(port, &[("file", audio)]).await
}

/// An audio file the decoder refuses is 422 with the decoder's fact, before
/// the Whisper model loads: the configured snapshot paths do not exist.
#[tokio::test]
async fn a_refused_audio_file_is_422_before_the_model_loads() {
    // ADTS (AAC LC, 16 kHz, mono): 20 silent frames, a header with the
    // forbidden sample-rate index, 20 silent frames.
    const SILENT: [u8; 11] = [
        0xFF, 0xF1, 0x60, 0x40, 0x01, 0x7F, 0xFC, 0x00, 0xC8, 0x00, 0x07,
    ];
    let mut bad_header = SILENT;
    bad_header[2] = 0x7C;
    let mut aac = SILENT.repeat(20);
    aac.extend_from_slice(&bad_header);
    aac.extend(SILENT.repeat(20));

    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;

    let answer = post_audio(port, &aac).await;
    let message = envelope(&answer, 422, "invalid_request_error", Some("file"));
    assert!(
        message.starts_with("audio decode: audio read failed at 1.28 s"),
        "{message}"
    );
}

/// The request is admitted before its audio is decoded: with the queue full, a
/// file the decoder would refuse is 429, so no decode ran outside the queue.
#[tokio::test]
async fn a_full_queue_answers_429_before_the_audio_is_decoded() {
    const SILENT: [u8; 11] = [
        0xFF, 0xF1, 0x60, 0x40, 0x01, 0x7F, 0xFC, 0x00, 0xC8, 0x00, 0x07,
    ];
    let mut aac = SILENT.repeat(20);
    aac.extend_from_slice(&[0xFF, 0xF1, 0x7C, 0x40, 0x01, 0x7F, 0xFC]);

    let (mut state, _dir) = missing_whisper_state();
    state.max_queue_depth = 1;
    state
        .gpu_pending
        .store(1, std::sync::atomic::Ordering::Release);
    let port = start_server_with_state(state).await;

    let Answer { status, body, .. } = post_audio(port, &aac).await;
    assert_eq!(status, 429, "{body}");
}

/// A file of exactly the cap passes the size check: it reaches the decoder,
/// which refuses bytes that are no audio.
#[tokio::test]
async fn a_file_of_exactly_the_cap_reaches_the_decoder() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let Answer { status, body, .. } = post_audio(port, &vec![0x55; CAP]).await;
    eprintln!("CASE file {CAP} B: {status} {body}");
    assert_eq!(status, 422, "{body}");
    assert!(body.contains("audio decode:"), "{body}");
}

/// A file over the cap is 413, and the body names the cap and the file's
/// size: OpenAI answers 413 `Maximum content size limit (26214400) exceeded
/// (26258790 bytes read)`.
#[tokio::test]
async fn a_file_over_the_cap_is_413_naming_the_cap_and_its_size() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    for size in [CAP + 1, 26_720_078] {
        let answer = post_audio(port, &vec![0x55; size]).await;
        eprintln!("CASE file {size} B: {} {}", answer.status, answer.body);
        let message = envelope(&answer, 413, "invalid_request_error", Some("file"));
        assert!(message.contains(&CAP.to_string()), "{size}: {message}");
        assert!(message.contains(&size.to_string()), "{size}: {message}");
    }
}

/// A request body over the server's transport limit (a 15 min 16 kHz mono
/// WAV is 28,800,078 bytes) is 413 naming the cap and the size the request
/// declared.
#[tokio::test]
async fn a_body_over_the_transport_limit_is_413_naming_the_cap_and_its_size() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let body = form(&[("file", &vec![0x55; 28_800_078])]);
    let declared = body.len();
    let answer = exchange(
        port,
        &format!("{FORM_TYPE}Content-Length: {declared}\r\n"),
        body,
    )
    .await;
    eprintln!("CASE body {declared} B: {} {}", answer.status, answer.body);
    let message = envelope(&answer, 413, "invalid_request_error", None);
    assert!(message.contains(&CAP.to_string()), "{message}");
    assert!(message.contains(&declared.to_string()), "{message}");
}

/// A chunked body states no length; over the transport limit it is 413
/// naming the cap.
#[tokio::test]
async fn a_chunked_body_over_the_transport_limit_is_413_naming_the_cap() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let form = form(&[("file", &vec![0x55; 28_800_078])]);
    let mut chunked = Vec::with_capacity(form.len() + 1024);
    for chunk in form.chunks(1 << 20) {
        chunked.extend_from_slice(format!("{:x}\r\n", chunk.len()).as_bytes());
        chunked.extend_from_slice(chunk);
        chunked.extend_from_slice(b"\r\n");
    }
    chunked.extend_from_slice(b"0\r\n\r\n");
    let answer = exchange(
        port,
        &format!("{FORM_TYPE}Transfer-Encoding: chunked\r\n"),
        chunked,
    )
    .await;
    eprintln!("CASE chunked body: {} {}", answer.status, answer.body);
    let message = envelope(&answer, 413, "invalid_request_error", None);
    assert!(message.contains(&CAP.to_string()), "{message}");
}

/// A `Content-Length` far over the cap on a short body is 4xx, and the server
/// answers the next request: the file's buffer is sized from the header only
/// up to the cap. hyper answers 431, with no body, to `u64::MAX`; the two
/// others reach the route, which refuses the file in the error envelope.
#[tokio::test]
async fn a_content_length_far_over_the_cap_is_4xx_and_the_server_lives() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let body = form(&[("file", &[0x55; 1000])]);
    for declared in [u64::MAX, 1 << 63, 4 << 30] {
        let answer = exchange(
            port,
            &format!("{FORM_TYPE}Content-Length: {declared}\r\n"),
            body.clone(),
        )
        .await;
        eprintln!(
            "CASE Content-Length {declared}: {} {}",
            answer.status, answer.body
        );
        if declared == u64::MAX {
            assert_eq!((answer.status, answer.body.as_str()), (431, ""));
        } else {
            envelope(&answer, 422, "invalid_request_error", Some("file"));
        }
    }
    let Answer {
        status,
        body: answer,
        ..
    } = post_audio(port, &short_wav()).await;
    assert!(answer.contains("tokenizer"), "{status} {answer}");
}

/// A request for word times needs `verbose_json`: with any other format it is
/// 422 naming `timestamp_granularities`, under the array name OpenAI's SDKs
/// send and under the bare name. With `verbose_json`, before or after the
/// granularity and beside `segment`, it reaches the tokenizer load.
#[tokio::test]
async fn word_timestamps_need_verbose_json() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    for name in ["timestamp_granularities[]", "timestamp_granularities"] {
        for format in [None, Some("json"), Some("text"), Some("srt"), Some("vtt")] {
            let mut parts = vec![(name, b"word".as_slice()), ("file", wav.as_slice())];
            if let Some(format) = format {
                parts.push(("response_format", format.as_bytes()));
            }
            let answer = post_form(port, &parts).await;
            eprintln!(
                "CASE {name}=word, {format:?}: {} {}",
                answer.status, answer.body
            );
            let message = envelope(
                &answer,
                422,
                "invalid_request_error",
                Some("timestamp_granularities"),
            );
            assert!(message.contains("verbose_json"), "{message}");
        }
        for parts in [
            vec![
                ("response_format", b"verbose_json".as_slice()),
                (name, b"word".as_slice()),
                ("file", wav.as_slice()),
            ],
            vec![
                (name, b"segment".as_slice()),
                (name, b"word".as_slice()),
                ("file", wav.as_slice()),
                ("response_format", b"verbose_json".as_slice()),
            ],
        ] {
            let Answer { status, body, .. } = post_form(port, &parts).await;
            eprintln!("CASE {name}=word, verbose_json: {status} {body}");
            assert!(body.contains("tokenizer"), "{name}: {status} {body}");
        }
    }
}

/// Only `json` streams: `stream=true` with any other format is 422 naming
/// `stream`, whichever field comes first. With `json`, named or by default, the
/// request reaches the tokenizer load, and its failure there is the JSON error
/// reply, not a stream.
#[tokio::test]
async fn only_json_streams() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    for format in ["text", "srt", "vtt", "verbose_json"] {
        for parts in [
            [
                ("stream", b"true".as_slice()),
                ("response_format", format.as_bytes()),
                ("file", wav.as_slice()),
            ],
            [
                ("file", wav.as_slice()),
                ("response_format", format.as_bytes()),
                ("stream", b"true".as_slice()),
            ],
        ] {
            let answer = post_form(port, &parts).await;
            eprintln!(
                "CASE stream=true, {format}: {} {}",
                answer.status, answer.body
            );
            let message = envelope(&answer, 422, "invalid_request_error", Some("stream"));
            assert!(message.contains("json"), "{format}: {message}");
        }
    }
    for parts in [
        vec![("file", wav.as_slice()), ("stream", b"true".as_slice())],
        vec![
            ("stream", b"true".as_slice()),
            ("response_format", b"json".as_slice()),
            ("file", wav.as_slice()),
        ],
    ] {
        let answer = post_form(port, &parts).await;
        eprintln!("CASE stream=true, json: {} {}", answer.status, answer.body);
        let message = envelope(&answer, 503, "service_unavailable", None);
        assert!(message.contains("tokenizer"), "{message}");
    }
}

/// A file the decoder refuses is a 422 JSON reply under `stream=true` too: the
/// reply waits for the audio decode and the model load before it starts.
#[tokio::test]
async fn a_refused_audio_file_under_stream_is_a_422_reply() {
    const SILENT: [u8; 11] = [
        0xFF, 0xF1, 0x60, 0x40, 0x01, 0x7F, 0xFC, 0x00, 0xC8, 0x00, 0x07,
    ];
    let mut aac = SILENT.repeat(20);
    aac.extend_from_slice(&[0xFF, 0xF1, 0x7C, 0x40, 0x01, 0x7F, 0xFC]);
    aac.extend(SILENT.repeat(20));

    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let answer = post_form(port, &[("file", &aac), ("stream", b"true")]).await;
    eprintln!(
        "CASE refused file, stream=true: {} {}",
        answer.status, answer.body
    );
    let message = envelope(&answer, 422, "invalid_request_error", Some("file"));
    assert!(message.starts_with("audio decode: "), "{message}");
}

/// A field the route serves or ignores does not refuse the request: each of
/// these reaches the tokenizer load.
#[tokio::test]
async fn served_and_ignored_fields_reach_the_model_load() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    let fields: [(&str, &[u8]); 20] = [
        ("model", b"whisper-1"),
        ("language", b"ru"),
        ("language", b"Russian"),
        ("language", b"EN"),
        ("language", b"auto"),
        ("response_format", b"json"),
        ("response_format", b"text"),
        ("response_format", b"srt"),
        ("response_format", b"vtt"),
        ("response_format", b"verbose_json"),
        ("temperature", b"0"),
        ("temperature", b"0.000001"),
        ("temperature", b"0.2"),
        ("temperature", b"1.5"),
        ("prompt", b"Glossary: rMLX, Whisper."),
        ("timestamp_granularities[]", b"segment"),
        ("timestamp_granularities", b"segment"),
        ("stream", b"false"),
        ("include[]", b"logprobs"),
        ("chunking_strategy", b"auto"),
    ];
    for (name, value) in fields {
        let Answer { status, body, .. } = post_form(port, &[("file", &wav), (name, value)]).await;
        eprintln!(
            "CASE {name}={}: {status} {body}",
            String::from_utf8_lossy(value)
        );
        assert!(body.contains("tokenizer"), "{name}: {status} {body}");
    }
}

/// Each of these refused field values answers 422 with a fixed envelope that
/// names the field in `param`.
#[tokio::test]
async fn a_refused_field_value_answers_422_with_its_fixed_body() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    let cases: [(&str, &[u8], &str); 5] = [
        (
            "language",
            b"xx",
            "unknown language 'xx'; use a Whisper language code, an English language name or 'auto'",
        ),
        (
            "response_format",
            b"diarized_json",
            "unsupported response_format 'diarized_json'; must be one of: json, text, verbose_json, srt, vtt",
        ),
        (
            "temperature",
            b"-1",
            "temperature -1 is out of range; must be 0 or a finite float >= 1e-6",
        ),
        (
            "temperature",
            b"0.0000001",
            "temperature 0.0000001 is out of range; must be 0 or a finite float >= 1e-6",
        ),
        (
            "temperature",
            b"hot",
            "invalid temperature 'hot'; must be 0 or a finite float >= 1e-6",
        ),
    ];
    for (name, value, expected) in cases {
        let answer = post_form(port, &[("file", &wav), (name, value)]).await;
        eprintln!(
            "CASE {name}={}: {} {}",
            String::from_utf8_lossy(value),
            answer.status,
            answer.body
        );
        let message = envelope(&answer, 422, "invalid_request_error", Some(name));
        assert_eq!(message, expected, "{name}");
    }
    let answer = post_form(port, &[("model", b"whisper-1")]).await;
    let message = envelope(&answer, 422, "invalid_request_error", Some("file"));
    assert_eq!(message, "missing required field: file");
    let answer = post_audio(port, b"").await;
    let message = envelope(&answer, 422, "invalid_request_error", Some("file"));
    assert_eq!(message, "audio file is empty");
}

/// A Whisper load failure is the server's, not the request's: 503, and the
/// body names the file and the fact, never a directory of the server.
#[tokio::test]
async fn a_whisper_load_failure_is_503_and_names_no_directory() {
    let (state, dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let answer = post_audio(port, &short_wav()).await;
    eprintln!("CASE load failure: {} {}", answer.status, answer.body);
    let message = envelope(&answer, 503, "service_unavailable", None);
    assert!(message.contains("tokenizer.json"), "{message}");
    assert!(message.contains("No such file"), "{message}");
    let dir = dir.path().display().to_string();
    assert!(!answer.body.contains(&dir), "{}", answer.body);
    assert!(!answer.body.contains('/'), "{}", answer.body);
}

/// The same for the model: with the local large-v3 tokenizer and a snapshot
/// directory that holds nothing, the request ends at `config.json`. Runs on
/// the CPU when `RMLX_O_MODELS_ROOT` holds `openai__whisper-large-v3-tokenizer`;
/// no weight is read.
#[tokio::test]
async fn a_whisper_model_load_failure_is_503_and_names_no_directory() {
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT") else {
        eprintln!(
            "SKIP a_whisper_model_load_failure_is_503_and_names_no_directory: RMLX_O_MODELS_ROOT is unset"
        );
        return;
    };
    let tokenizer = std::path::Path::new(&root).join("openai__whisper-large-v3-tokenizer");
    if !tokenizer.join("tokenizer.json").exists() {
        eprintln!(
            "SKIP a_whisper_model_load_failure_is_503_and_names_no_directory: no openai__whisper-large-v3-tokenizer under the root"
        );
        return;
    }
    let (mut state, dir) = missing_whisper_state();
    state.whisper_tokenizer_path = Some(tokenizer);
    let port = start_server_with_state(state).await;
    let answer = post_audio(port, &short_wav()).await;
    eprintln!("CASE model load failure: {} {}", answer.status, answer.body);
    let message = envelope(&answer, 503, "service_unavailable", None);
    assert!(message.contains("config.json"), "{message}");
    let dir = dir.path().display().to_string();
    assert!(!answer.body.contains(&dir), "{}", answer.body);
    assert!(!answer.body.contains('/'), "{}", answer.body);
}

/// A prompt that holds the text of a special token is 422 naming `prompt`,
/// before the audio is decoded and before the model loads: the file is not a
/// sound and the snapshot directory does not exist, and neither is read. A plain
/// prompt passes and the request ends at the model load. Runs on the CPU when
/// `RMLX_O_MODELS_ROOT` holds `openai__whisper-large-v3-tokenizer`; no weight is
/// read.
#[tokio::test]
async fn a_prompt_with_special_token_text_is_422_before_the_audio_and_the_model() {
    const TEST: &str = "a_prompt_with_special_token_text_is_422_before_the_audio_and_the_model";
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT") else {
        eprintln!(
            "SKIP a_prompt_with_special_token_text_is_422_before_the_audio_and_the_model: RMLX_O_MODELS_ROOT is unset"
        );
        return;
    };
    let tokenizer = std::path::Path::new(&root).join("openai__whisper-large-v3-tokenizer");
    if !tokenizer.join("tokenizer.json").exists() {
        eprintln!(
            "SKIP a_prompt_with_special_token_text_is_422_before_the_audio_and_the_model: no openai__whisper-large-v3-tokenizer under the root"
        );
        return;
    }
    let (mut state, _dir) = missing_whisper_state();
    state.whisper_tokenizer_path = Some(tokenizer);
    let port = start_server_with_state(state).await;

    let not_a_sound: &[u8] = b"this is not audio";
    for prompt in ["Names: <|endoftext|>", "<|en|>", "a <|0.00|> b"] {
        for file in [not_a_sound, &short_wav()] {
            let answer = post_form(port, &[("file", file), ("prompt", prompt.as_bytes())]).await;
            eprintln!("CASE {TEST} {prompt:?}: {} {}", answer.status, answer.body);
            let message = envelope(&answer, 422, "invalid_request_error", Some("prompt"));
            assert!(message.starts_with("invalid prompt: "), "{message}");
        }
    }

    let answer = post_form(
        port,
        &[("file", &short_wav()), ("prompt", b"Glossary: rMLX.")],
    )
    .await;
    let message = envelope(&answer, 503, "service_unavailable", None);
    assert!(message.contains("config.json"), "{message}");
}

/// A load failure is permanent until the operator acts: the 503 carries
/// `x-should-retry: false`, which OpenAI's SDKs obey before their own retry of
/// a 5xx.
#[tokio::test]
async fn a_load_failure_tells_the_client_not_to_retry() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let answer = post_audio(port, &short_wav()).await;
    eprintln!("CASE load failure head: {}", answer.head);
    assert_eq!(answer.status, 503, "{}", answer.body);
    assert!(
        answer
            .head
            .to_ascii_lowercase()
            .contains("\r\nx-should-retry: false"),
        "{}",
        answer.head
    );
}

/// `stream` is read as a boolean whatever its case: `False` (Python's
/// `str(False)`), `0` and an empty value are false, and `True` and `1` are
/// true; each reaches the tokenizer load. Any other value is 422 naming the
/// field.
#[tokio::test]
async fn stream_is_read_as_a_boolean_in_any_case() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    for (value, refused) in [
        ("false", false),
        ("False", false),
        ("0", false),
        ("", false),
        ("true", false),
        ("True", false),
        ("1", false),
        ("maybe", true),
    ] {
        let answer = post_form(port, &[("file", &wav), ("stream", value.as_bytes())]).await;
        eprintln!("CASE stream={value}: {} {}", answer.status, answer.body);
        if refused {
            let message = envelope(&answer, 422, "invalid_request_error", Some("stream"));
            assert!(message.contains("stream"), "{value}: {message}");
        } else {
            assert!(
                answer.body.contains("tokenizer"),
                "{value}: {}",
                answer.body
            );
        }
    }
}

/// A form of at most 64 parts is read; one more is 422 naming the limit.
#[tokio::test]
async fn a_form_of_more_than_64_parts_is_422_naming_the_limit() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    for (parts, refused) in [(64, false), (65, true)] {
        let mut form: Vec<(&str, &[u8])> = vec![("file", &wav)];
        form.resize(parts, ("x", b"".as_slice()));
        let answer = post_form(port, &form).await;
        eprintln!("CASE {parts} parts: {} {}", answer.status, answer.body);
        if refused {
            let message = envelope(&answer, 422, "invalid_request_error", None);
            assert_eq!(message, "the form has more than 64 parts", "{parts}");
        } else {
            assert!(
                answer.body.contains("tokenizer"),
                "{parts}: {}",
                answer.body
            );
        }
    }
}

/// An ignored field with a 1 MiB name is read like any other.
#[tokio::test]
async fn an_ignored_field_with_a_long_name_is_read() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let name = "n".repeat(1 << 20);
    let Answer { status, body, .. } =
        post_form(port, &[(name.as_str(), b"x"), ("file", &short_wav())]).await;
    eprintln!("CASE 1 MiB field name: {status} {body}");
    assert!(body.contains("tokenizer"), "{status} {body}");
}

/// A body over the transport limit is 413 naming the cap when the excess is
/// in a field the route ignores, not in the file.
#[tokio::test]
async fn an_ignored_field_over_the_transport_limit_is_413_naming_the_cap() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let answer = post_form(
        port,
        &[("prompt", &vec![b'x'; 28_800_078]), ("file", &short_wav())],
    )
    .await;
    eprintln!(
        "CASE 28.8 MB ignored field: {} {}",
        answer.status, answer.body
    );
    let message = envelope(&answer, 413, "invalid_request_error", None);
    assert!(message.contains(&CAP.to_string()), "{message}");
}

/// A multipart request with no boundary is refused by axum's extractor before
/// the handler's body runs: 400 in the error envelope, on both routes.
#[tokio::test]
async fn a_form_with_no_boundary_is_400() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let body = form(&[("file", &short_wav())]);
    let headers = format!(
        "Content-Type: multipart/form-data\r\nContent-Length: {}\r\n",
        body.len()
    );
    for path in [TRANSCRIPTIONS, TRANSLATIONS] {
        let answer = exchange_to(port, path, &headers, body.clone()).await;
        eprintln!("CASE no boundary {path}: {} {}", answer.status, answer.body);
        let message = envelope(&answer, 400, "invalid_request_error", None);
        assert!(message.contains("boundary"), "{path}: {message}");
    }
}

/// A server with no Whisper paths answers every request 503 and tells the
/// client not to retry.
#[tokio::test]
async fn an_unconfigured_route_is_503_and_says_no_retry() {
    let state = not_ready_state(ModelRegistry::default());
    let port = start_server_with_state(state).await;
    let answer = post_audio(port, &short_wav()).await;
    let message = envelope(&answer, 503, "service_unavailable", None);
    assert_eq!(
        message,
        "audio model not configured; set --whisper-model-path"
    );
    assert!(
        answer
            .head
            .to_ascii_lowercase()
            .contains("\r\nx-should-retry: false"),
        "{}",
        answer.head
    );
}

/// A long value of a field the route reads is refused without its echo: a
/// 1 MiB value of control bytes (six bytes each once JSON-escaped) gives a
/// short 422 that names the field.
#[tokio::test]
async fn a_long_value_is_refused_without_its_echo() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    let long = vec![1_u8; 1 << 20];
    for name in [
        "language",
        "temperature",
        "response_format",
        "timestamp_granularities[]",
        "stream",
    ] {
        let answer = post_form(port, &[(name, long.as_slice()), ("file", &wav)]).await;
        let body = &answer.body;
        eprintln!(
            "CASE 1 MiB {name}: {}, {} B: {}",
            answer.status,
            body.len(),
            body.get(..120).unwrap_or(body)
        );
        let message = envelope(&answer, 422, "invalid_request_error", Some(name));
        assert!(body.len() < 200, "{name}: a {} B body", body.len());
        assert!(message.contains(name), "{name}: {message}");
    }
}

/// The limit of a short field is 64 bytes: an unknown language of 64 bytes is
/// echoed in its refusal; one of 65 bytes is refused for its length.
#[tokio::test]
async fn a_short_field_holds_at_most_64_bytes() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    let at_limit = "q".repeat(64);
    let answer = post_form(port, &[("language", at_limit.as_bytes()), ("file", &wav)]).await;
    let message = envelope(&answer, 422, "invalid_request_error", Some("language"));
    assert!(message.contains(&at_limit), "{message}");
    let over = "q".repeat(65);
    let answer = post_form(port, &[("language", over.as_bytes()), ("file", &wav)]).await;
    eprintln!("CASE 65-byte language: {} {}", answer.status, answer.body);
    let message = envelope(&answer, 422, "invalid_request_error", Some("language"));
    assert!(!message.contains(&at_limit), "{message}");
    assert!(message.contains("64 bytes"), "{message}");
}

/// The refusals the other tests do not reach, on both routes: a bad field
/// value, a missing file, the 503 of a Whisper load.
#[tokio::test]
async fn every_refusal_is_the_error_envelope_on_both_routes() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    for path in [TRANSCRIPTIONS, TRANSLATIONS] {
        let answer = post_form_to(port, path, &[("file", &wav), ("language", b"xx")]).await;
        envelope(&answer, 422, "invalid_request_error", Some("language"));

        let answer = post_form_to(
            port,
            path,
            &[("file", &wav), ("timestamp_granularities[]", b"phrase")],
        )
        .await;
        let message = envelope(
            &answer,
            422,
            "invalid_request_error",
            Some("timestamp_granularities[]"),
        );
        assert!(message.contains("phrase"), "{message}");

        let answer = post_form_to(
            port,
            path,
            &[("file", &wav), ("timestamp_granularities", b"phrase")],
        )
        .await;
        let message = envelope(
            &answer,
            422,
            "invalid_request_error",
            Some("timestamp_granularities"),
        );
        assert!(message.contains("phrase"), "{message}");

        let answer = post_form_to(port, path, &[("model", b"whisper-1")]).await;
        envelope(&answer, 422, "invalid_request_error", Some("file"));

        let answer = post_form_to(port, path, &[("file", &wav)]).await;
        envelope(&answer, 503, "service_unavailable", None);
    }
}

/// A form the multipart reader cannot parse is 422 in the error envelope, with
/// no `param`: no one request field is at fault.
#[tokio::test]
async fn an_unreadable_form_is_422_in_the_error_envelope() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let body =
        b"--XBOUNDARY\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1".to_vec();
    let headers = format!("{FORM_TYPE}Content-Length: {}\r\n", body.len());
    let answer = exchange(port, &headers, body).await;
    eprintln!("CASE unreadable form: {} {}", answer.status, answer.body);
    let message = envelope(&answer, 422, "invalid_request_error", None);
    assert_eq!(message, "Error parsing `multipart/form-data` request");
}

/// The 429 of a full queue keeps its envelope and its text.
#[tokio::test]
async fn a_full_queue_answer_is_the_rate_limit_envelope() {
    let (mut state, _dir) = missing_whisper_state();
    state.max_queue_depth = 1;
    state
        .gpu_pending
        .store(1, std::sync::atomic::Ordering::Release);
    let port = start_server_with_state(state).await;
    let answer = post_audio(port, &short_wav()).await;
    let message = envelope(&answer, 429, "rate_limit_error", None);
    assert_eq!(message, "server queue full");
}

/// The 408 of the request timeout keeps its envelope: a client that sends the
/// head of a form and stalls is cut at the deadline of its own header.
#[tokio::test]
async fn a_stalled_upload_is_the_timeout_envelope() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let head = format!(
        "POST {TRANSCRIPTIONS} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
         X-Request-Timeout-Seconds: 1\r\n{FORM_TYPE}Content-Length: 100000\r\n\r\n\
         --XBOUNDARY\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF"
    );
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    stream.write_all(head.as_bytes()).await.unwrap();
    let mut response = Vec::new();
    let read = tokio::time::timeout(
        std::time::Duration::from_secs(10),
        stream.read_to_end(&mut response),
    )
    .await;
    assert!(read.is_ok(), "no answer within 10 s");
    let text = String::from_utf8_lossy(&response).into_owned();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    let status = head
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let answer = Answer {
        status,
        head: head.to_owned(),
        body: body.to_owned(),
    };
    envelope(&answer, 408, "timeout", None);
}

/// Replies made before the route runs have no body: the 405 of the router for
/// a method other than POST and the 400 of the HTTP layer for a request head
/// it cannot parse. The 431 for an absurd `Content-Length` is held by the
/// Content-Length test above.
#[tokio::test]
async fn a_reply_made_before_the_route_has_no_body() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    for (request, status) in [
        (
            format!(
                "GET {TRANSCRIPTIONS} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\r\n"
            ),
            405,
        ),
        ("NOT AN HTTP REQUEST\r\n\r\n".to_owned(), 400),
    ] {
        let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
            .await
            .unwrap();
        stream.write_all(request.as_bytes()).await.unwrap();
        let mut response = Vec::new();
        let read = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            stream.read_to_end(&mut response),
        )
        .await;
        assert!(read.is_ok(), "no answer within 10 s to {request:?}");
        let text = String::from_utf8_lossy(&response).into_owned();
        eprintln!("CASE {request:?}: {text:?}");
        let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
        let got = head
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok());
        assert_eq!(got, Some(status), "{text:?}");
        assert_eq!(body, "", "{text:?}");
    }
}

// ── The token pick of a request, on the model ────────────────────────────────

/// A clip of two windows, as a 16 kHz mono WAV file: four words spoken 12
/// times by `say` with the English test voice, 20 s of zeros, and the same
/// speech again. Each window holds one true repetition.
fn repetition_wav() -> Option<Vec<u8>> {
    let dir = tempfile::tempdir().ok()?;
    let wav = dir.path().join("repetition.wav");
    if !say_voice::say_english_wav(&wav, &"one two three four ".repeat(12)) {
        return None;
    }
    let speech = WavDecoder::decode(&std::fs::read(&wav).ok()?).ok()?.samples;
    let pause = vec![0.0; 20 * 16_000];
    let samples = [speech.as_slice(), pause.as_slice(), speech.as_slice()].concat();
    WavEncoder::encode(&samples, 16_000, 1).ok()
}

/// The window (`seek`), the tokens and the temperature of each segment.
type Segments = Vec<(usize, Vec<u32>, f32)>;

/// The keys of a `verbose_json` reply that the test reads.
#[derive(serde::Deserialize)]
struct Reply {
    segments: Vec<ReplySegment>,
}

#[derive(serde::Deserialize)]
struct ReplySegment {
    seek: usize,
    tokens: Vec<u32>,
    temperature: f32,
}

/// How many windows have a segment whose decode was drawn.
fn windows_with_a_drawn_decode(segments: &Segments) -> usize {
    let mut seeks: Vec<usize> = segments
        .iter()
        .filter(|(_, _, temperature)| *temperature > 0.0)
        .map(|(seek, _, _)| *seek)
        .collect();
    seeks.dedup();
    seeks.len()
}

/// A voice that `say` does not have is not available: no clip is spoken, so a
/// model test stands down and does not run on the default voice of the host.
#[test]
fn a_voice_that_is_not_installed_gives_no_clip() {
    const NO_VOICE: &str = "NoSuchVoiceXyz";
    if !say_voice::voice_is_installed(say_voice::ENGLISH_VOICE) {
        eprintln!(
            "SKIP a_voice_that_is_not_installed_gives_no_clip: say or its voice is unavailable"
        );
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("no_voice.wav");
    assert!(!say_voice::voice_is_installed(NO_VOICE));
    // The name of a voice is the whole name: a prefix of one is not a voice.
    assert!(!say_voice::voice_is_installed("Eddy"));
    assert!(!say_voice::say_wav(NO_VOICE, &path, "hello"));
    assert!(!path.exists(), "a clip was spoken with another voice");
}

/// The route draws every token of a request from ONE generator with the
/// sampler's default seed. The clip has a spoken repetition in each of two
/// windows: the greedy decode of each is a loop, so each goes up the
/// temperature ladder and its kept decode is drawn. The tokens that the route answers are those of
/// `Transcriber::transcribe` with one `RowPicker::default()` for the whole
/// call. A picker made for each pick, for each decode or for each window
/// starts its stream again and gives other tokens in the second window;
/// another seed gives other tokens in both. `rmlx transcribe` calls the same
/// `transcribe_request`.
///
/// The first assertion is the precondition, not the rule: for this `say`
/// voice and this model, two windows keep a drawn decode.
#[tokio::test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-server --test audio_route the_route_draws -- --ignored --test-threads=1"]
async fn the_route_draws_a_request_from_one_seeded_generator() {
    const TEST: &str = "the_route_draws_a_request_from_one_seeded_generator";
    let Some(model_path) = snapshot(TEST, WHISPER.0, WHISPER.1) else {
        return;
    };
    let Some(tokenizer_path) = snapshot(TEST, WHISPER_TOKENIZER.0, WHISPER_TOKENIZER.1) else {
        return;
    };
    let Some(wav) = repetition_wav() else {
        eprintln!("SKIP the_route_draws_a_request_from_one_seeded_generator: say or its voice is unavailable");
        return;
    };

    let reference: Segments = {
        let samples = WavDecoder::decode_resampled(&wav, 16_000, MAX_AUDIO_SECS, &|| false)
            .expect("decode the clip")
            .samples;
        let model = WhisperModel::load(&model_path).expect("load whisper model");
        let tokenizer = WhisperTokenizer::from_path(&tokenizer_path).expect("load tokenizer");
        let transcriber =
            Transcriber::new(Arc::new(model), Arc::new(tokenizer)).expect("transcriber");
        let opts = TranscribeOptions {
            language: "en".to_owned(),
            task: WhisperTask::Transcribe,
            temperature: 0.0,
            condition_on_previous_text: true,
            prompt_tokens: Vec::new(),
            words: false,
        };
        let mut picker = RowPicker::default();
        let transcription = transcriber
            .transcribe(
                &samples,
                &opts,
                &mut |row, temperature| picker.pick(row, temperature),
                Device::Gpu,
                &AtomicBool::new(false),
                &mut |_| Ok(()),
            )
            .expect("transcribe");
        transcription
            .segments
            .into_iter()
            .map(|s| (s.seek, s.tokens, s.temperature))
            .collect()
    };
    assert!(
        windows_with_a_drawn_decode(&reference) >= 2,
        "precondition: two windows of the clip keep a decode above temperature 0; \
         windows and temperatures {:?}",
        reference
            .iter()
            .map(|(seek, _, t)| (*seek, *t))
            .collect::<Vec<_>>()
    );

    let mut state = not_ready_state(ModelRegistry::default());
    state.whisper_model_path = Some(model_path);
    state.whisper_tokenizer_path = Some(tokenizer_path);
    state.device = Device::Gpu;
    let port = start_server_with_state(state).await;
    let Answer { status, body, .. } = post_form(
        port,
        &[
            ("file", &wav),
            ("language", b"en"),
            ("response_format", b"verbose_json"),
        ],
    )
    .await;
    assert_eq!(status, 200, "{body}");
    let reply: Reply = serde_json::from_str(&body).expect("a verbose_json reply");
    let from_the_route: Segments = reply
        .segments
        .into_iter()
        .map(|segment| (segment.seek, segment.tokens, segment.temperature))
        .collect();
    assert_eq!(from_the_route, reference);
}

// ── A streamed transcription, on the model ──────────────────────────────────

/// The sentences of [`windows_wav`], one for each window.
const WINDOW_SENTENCES: [&str; 8] = [
    "Apples grow on the old trees in the orchard.",
    "The train leaves the station at noon.",
    "A small boat crossed the quiet lake.",
    "She painted the garden fence a bright blue.",
    "The library opens early on Mondays.",
    "Rain fell softly on the tin roof.",
    "He fixed the bicycle with a wrench.",
    "The market sells fresh bread every day.",
];

/// The 16 kHz samples of `sentence` spoken by `say` with the English test
/// voice; `None` when `say` or the voice is not available.
fn spoken(sentence: &str) -> Option<Vec<f32>> {
    let dir = tempfile::tempdir().ok()?;
    let wav = dir.path().join("sentence.wav");
    if !say_voice::say_english_wav(&wav, sentence) {
        return None;
    }
    Some(WavDecoder::decode(&std::fs::read(&wav).ok()?).ok()?.samples)
}

/// A clip of `sentences.len()` windows as a 16 kHz mono WAV file: each
/// sentence, then zeros to 29 s.
fn windows_wav(sentences: &[&str]) -> Option<Vec<u8>> {
    const SLOT: usize = 29 * 16_000;
    let mut samples = Vec::new();
    for sentence in sentences {
        let speech = spoken(sentence)?;
        let pause = SLOT.saturating_sub(speech.len());
        samples.extend(speech);
        samples.extend(std::iter::repeat_n(0.0, pause));
    }
    WavEncoder::encode(&samples, 16_000, 1).ok()
}

/// The longest wait for the next bytes of a streamed reply.
const STREAM_WAIT: std::time::Duration = std::time::Duration::from_secs(120);

/// A streamed reply as the client read it: the status, the header lines, and
/// each event with the time its bytes arrived.
struct Streamed {
    status: u16,
    head: String,
    events: Vec<(std::time::Instant, serde_json::Value)>,
}

impl Streamed {
    fn deltas(&self) -> Vec<&str> {
        self.events
            .iter()
            .filter(|(_, event)| event["type"] == "transcript.text.delta")
            .filter_map(|(_, event)| event["delta"].as_str())
            .collect()
    }
}

/// POST `wav` with `stream=true` and read the events as they arrive. With
/// `close_after_first_delta`, the client closes the connection once a delta
/// has arrived. Every `data:` line is one JSON event; the chunk-size lines of
/// the transfer encoding are not `data:` lines.
async fn stream_transcription(port: u16, wav: &[u8], close_after_first_delta: bool) -> Streamed {
    let body = form(&[("file", wav), ("language", b"en"), ("stream", b"true")]);
    let head = format!(
        "POST {TRANSCRIPTIONS} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
         {FORM_TYPE}Content-Length: {}\r\n\r\n",
        body.len()
    );
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    stream.write_all(head.as_bytes()).await.unwrap();
    stream.write_all(&body).await.unwrap();

    let mut raw: Vec<u8> = Vec::new();
    let mut read_to = 0;
    let mut events = Vec::new();
    let mut buf = vec![0_u8; 1 << 16];
    loop {
        let n = tokio::time::timeout(STREAM_WAIT, stream.read(&mut buf))
            .await
            .expect("the reply stalls")
            .expect("the reply reads");
        if n == 0 {
            break;
        }
        let now = std::time::Instant::now();
        raw.extend_from_slice(buf.get(..n).unwrap_or_default());
        while let Some(end) = raw
            .get(read_to..)
            .and_then(|rest| rest.iter().position(|&b| b == b'\n'))
        {
            let line = String::from_utf8_lossy(raw.get(read_to..read_to + end).unwrap_or_default())
                .into_owned();
            read_to += end + 1;
            if let Some(data) = line.trim_end_matches('\r').strip_prefix("data: ") {
                let event: serde_json::Value = serde_json::from_str(data)
                    .unwrap_or_else(|e| panic!("an event that is not JSON ({e}): {data}"));
                events.push((now, event));
            }
        }
        if close_after_first_delta
            && events
                .iter()
                .any(|(_, event)| event["type"] == "transcript.text.delta")
        {
            break;
        }
    }
    let text = String::from_utf8_lossy(&raw);
    let head_end = text.find("\r\n\r\n").unwrap_or(text.len());
    let status = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    Streamed {
        status,
        head: text[..head_end].to_owned(),
        events,
    }
}

/// The `text` of a `json` reply to `wav`, and how long the reply took.
async fn plain_text(port: u16, wav: &[u8]) -> (String, std::time::Duration) {
    let started = std::time::Instant::now();
    let Answer { status, body, .. } = post_form(port, &[("file", wav), ("language", b"en")]).await;
    let took = started.elapsed();
    assert_eq!(status, 200, "{body}");
    let reply: serde_json::Value = serde_json::from_str(&body).expect("a json reply");
    let text = reply.get("text").and_then(serde_json::Value::as_str);
    (text.expect("a text").to_owned(), took)
}

/// A streamed transcription of a clip of eight windows, through the route on
/// the model. The first delta arrives before half the time of the whole
/// transcription has passed; the deltas, joined with nothing, are the text of
/// the done event and the text of the same clip without `stream`; the done
/// event is the last event. A client that closes the connection after the
/// first delta stops the job: the next request, of one sentence, has its reply
/// before half the time from the first delta to the done event of a whole
/// stream has passed, so the walk did not go on to the end of the clip.
#[tokio::test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-server --test audio_route a_streamed_transcription -- --ignored --test-threads=1"]
async fn a_streamed_transcription_sends_each_window_as_it_closes() {
    const TEST: &str = "a_streamed_transcription_sends_each_window_as_it_closes";
    let Some(model_path) = snapshot(TEST, WHISPER.0, WHISPER.1) else {
        return;
    };
    let Some(tokenizer_path) = snapshot(TEST, WHISPER_TOKENIZER.0, WHISPER_TOKENIZER.1) else {
        return;
    };
    let (Some(clip), Some(sentence)) = (
        windows_wav(&WINDOW_SENTENCES),
        windows_wav(&WINDOW_SENTENCES[..1]),
    ) else {
        eprintln!(
            "SKIP a_streamed_transcription_sends_each_window_as_it_closes: say or its voice is unavailable"
        );
        return;
    };

    let mut state = not_ready_state(ModelRegistry::default());
    state.whisper_model_path = Some(model_path);
    state.whisper_tokenizer_path = Some(tokenizer_path);
    state.device = Device::Gpu;
    let port = start_server_with_state(state).await;

    // The first request loads the model; the timed ones find it loaded.
    plain_text(port, &sentence).await;
    let (text, plain_took) = plain_text(port, &clip).await;

    let started = std::time::Instant::now();
    let whole = stream_transcription(port, &clip, false).await;
    assert_eq!(whole.status, 200, "{}", whole.head);
    assert!(
        whole
            .head
            .to_ascii_lowercase()
            .contains("\r\ncontent-type: text/event-stream"),
        "{}",
        whole.head
    );
    let deltas = whole.deltas();
    eprintln!(
        "streamed: {} events, {} deltas, plain reply {plain_took:?}",
        whole.events.len(),
        deltas.len()
    );
    assert!(deltas.len() >= 2, "one delta for each window: {deltas:?}");
    let (first_at, _) = whole
        .events
        .iter()
        .find(|(_, event)| event["type"] == "transcript.text.delta")
        .expect("a delta");
    let (done_at, done) = whole.events.last().expect("an event");
    assert_eq!(done["type"], "transcript.text.done", "{done}");
    assert_eq!(
        whole
            .events
            .iter()
            .filter(|(_, event)| event["type"] == "transcript.text.done")
            .count(),
        1
    );
    assert!(
        first_at.duration_since(started) < plain_took / 2,
        "the first delta came {:?} after the request; the whole transcription takes {plain_took:?}",
        first_at.duration_since(started)
    );
    assert_eq!(deltas.concat(), done["text"].as_str().expect("a text"));
    assert_eq!(deltas.concat(), text);
    let span = done_at.duration_since(*first_at);

    let gone = stream_transcription(port, &clip, true).await;
    assert_eq!(gone.deltas().len(), 1, "the client closes after one delta");
    let closed = std::time::Instant::now();
    plain_text(port, &sentence).await;
    let next_after = closed.elapsed();
    eprintln!(
        "first delta to done {span:?}; the next request ended {next_after:?} after the close"
    );
    assert!(
        next_after < span / 2,
        "the next request ended {next_after:?} after the close; a whole stream runs {span:?} \
         from its first delta to its done event"
    );
}
