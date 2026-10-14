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
    let head = format!(
        "POST /v1/audio/transcriptions HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
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

    let Answer { status, body, .. } = post_audio(port, &aac).await;
    assert_eq!(status, 422, "{body}");
    assert!(
        body.contains("audio decode: audio read failed at 1.28 s"),
        "{body}"
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
        let Answer { status, body, .. } = post_audio(port, &vec![0x55; size]).await;
        eprintln!("CASE file {size} B: {status} {body}");
        assert_eq!(status, 413, "{size}: {body}");
        assert!(body.contains(&CAP.to_string()), "{size}: {body}");
        assert!(body.contains(&size.to_string()), "{size}: {body}");
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
    let Answer {
        status,
        body: answer,
        ..
    } = exchange(
        port,
        &format!("{FORM_TYPE}Content-Length: {declared}\r\n"),
        body,
    )
    .await;
    eprintln!("CASE body {declared} B: {status} {answer}");
    assert_eq!(status, 413, "{answer}");
    assert!(answer.contains(&CAP.to_string()), "{answer}");
    assert!(answer.contains(&declared.to_string()), "{answer}");
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
    let Answer {
        status,
        body: answer,
        ..
    } = exchange(
        port,
        &format!("{FORM_TYPE}Transfer-Encoding: chunked\r\n"),
        chunked,
    )
    .await;
    eprintln!("CASE chunked body: {status} {answer}");
    assert_eq!(status, 413, "{answer}");
    assert!(answer.contains(&CAP.to_string()), "{answer}");
}

/// A `Content-Length` far over the cap on a short body is 4xx, and the server
/// answers the next request: the file's buffer is sized from the header only
/// up to the cap. hyper answers 431 to `u64::MAX`; the two others reach the
/// route.
#[tokio::test]
async fn a_content_length_far_over_the_cap_is_4xx_and_the_server_lives() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let body = form(&[("file", &[0x55; 1000])]);
    for declared in [u64::MAX, 1 << 63, 4 << 30] {
        let Answer {
            status,
            body: answer,
            ..
        } = exchange(
            port,
            &format!("{FORM_TYPE}Content-Length: {declared}\r\n"),
            body.clone(),
        )
        .await;
        eprintln!("CASE Content-Length {declared}: {status} {answer}");
        assert!(
            (400..500).contains(&status),
            "{declared}: {status} {answer}"
        );
    }
    let Answer {
        status,
        body: answer,
        ..
    } = post_audio(port, &short_wav()).await;
    assert!(answer.contains("tokenizer"), "{status} {answer}");
}

/// Word timestamps are not built: a request for them is 422 naming the field,
/// under the array name OpenAI's SDKs send and under the bare name.
#[tokio::test]
async fn word_timestamps_are_422_naming_the_field() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    for parts in [
        vec![
            ("file", wav.as_slice()),
            ("timestamp_granularities[]", b"word".as_slice()),
        ],
        vec![
            ("file", wav.as_slice()),
            ("timestamp_granularities", b"word".as_slice()),
        ],
        vec![
            ("timestamp_granularities[]", b"segment".as_slice()),
            ("timestamp_granularities[]", b"word".as_slice()),
            ("file", wav.as_slice()),
        ],
    ] {
        let Answer { status, body, .. } = post_form(port, &parts).await;
        let named: Vec<String> = parts
            .iter()
            .filter(|(name, _)| *name != "file")
            .map(|(name, value)| format!("{name}={}", String::from_utf8_lossy(value)))
            .collect();
        eprintln!("CASE {named:?}: {status} {body}");
        assert_eq!(status, 422, "{body}");
        assert!(body.contains("timestamp_granularities"), "{body}");
        assert!(body.contains("word"), "{body}");
    }
}

/// The transcription is returned whole when it ends: `stream=true` is 422
/// naming the field.
#[tokio::test]
async fn stream_true_is_422_naming_the_field() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let Answer { status, body, .. } =
        post_form(port, &[("file", &short_wav()), ("stream", b"true")]).await;
    eprintln!("CASE stream=true: {status} {body}");
    assert_eq!(status, 422, "{body}");
    assert!(body.contains("stream"), "{body}");
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

/// Each of these refused field values answers 422 with a fixed body.
#[tokio::test]
async fn a_refused_field_value_answers_422_with_its_fixed_body() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let wav = short_wav();
    let cases: [(&str, &[u8], &str); 5] = [
        (
            "language",
            b"xx",
            r#"{"error":"unknown language 'xx'; use a Whisper language code, an English language name or 'auto'"}"#,
        ),
        (
            "response_format",
            b"diarized_json",
            r#"{"error":"unsupported response_format 'diarized_json'; must be one of: json, text, verbose_json, srt, vtt"}"#,
        ),
        (
            "temperature",
            b"-1",
            r#"{"error":"temperature -1 is out of range; must be 0 or a finite float >= 1e-6"}"#,
        ),
        (
            "temperature",
            b"0.0000001",
            r#"{"error":"temperature 0.0000001 is out of range; must be 0 or a finite float >= 1e-6"}"#,
        ),
        (
            "temperature",
            b"hot",
            r#"{"error":"invalid temperature 'hot'; must be 0 or a finite float >= 1e-6"}"#,
        ),
    ];
    for (name, value, expected) in cases {
        let Answer { status, body, .. } = post_form(port, &[("file", &wav), (name, value)]).await;
        eprintln!(
            "CASE {name}={}: {status} {body}",
            String::from_utf8_lossy(value)
        );
        assert_eq!((status, body.as_str()), (422, expected), "{name}");
    }
    let Answer { status, body, .. } = post_form(port, &[("model", b"whisper-1")]).await;
    assert_eq!(
        (status, body.as_str()),
        (422, r#"{"error":"missing required field: file"}"#)
    );
    let Answer { status, body, .. } = post_audio(port, b"").await;
    assert_eq!(
        (status, body.as_str()),
        (422, r#"{"error":"audio file is empty"}"#)
    );
}

/// A Whisper load failure is the server's, not the request's: 503, and the
/// body names the file and the fact, never a directory of the server.
#[tokio::test]
async fn a_whisper_load_failure_is_503_and_names_no_directory() {
    let (state, dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let Answer { status, body, .. } = post_audio(port, &short_wav()).await;
    eprintln!("CASE load failure: {status} {body}");
    assert_eq!(status, 503, "{body}");
    assert!(body.contains("tokenizer.json"), "{body}");
    assert!(body.contains("No such file"), "{body}");
    let dir = dir.path().display().to_string();
    assert!(!body.contains(&dir), "{body}");
    assert!(!body.contains('/'), "{body}");
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
    let Answer { status, body, .. } = post_audio(port, &short_wav()).await;
    eprintln!("CASE model load failure: {status} {body}");
    assert_eq!(status, 503, "{body}");
    assert!(body.contains("config.json"), "{body}");
    let dir = dir.path().display().to_string();
    assert!(!body.contains(&dir), "{body}");
    assert!(!body.contains('/'), "{body}");
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
/// `str(False)`), `0` and an empty value are false; `True` and `1` are true and
/// refused; any other value is 422 naming the field.
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
        ("true", true),
        ("True", true),
        ("1", true),
        ("maybe", true),
    ] {
        let Answer { status, body, .. } =
            post_form(port, &[("file", &wav), ("stream", value.as_bytes())]).await;
        eprintln!("CASE stream={value}: {status} {body}");
        if refused {
            assert_eq!(status, 422, "{value}: {body}");
            assert!(body.contains("stream"), "{value}: {body}");
        } else {
            assert!(body.contains("tokenizer"), "{value}: {status} {body}");
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
        let Answer { status, body, .. } = post_form(port, &form).await;
        eprintln!("CASE {parts} parts: {status} {body}");
        if refused {
            assert_eq!(status, 422, "{parts}: {body}");
            assert!(body.contains("64"), "{parts}: {body}");
        } else {
            assert!(body.contains("tokenizer"), "{parts}: {status} {body}");
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
    let Answer { status, body, .. } = post_form(
        port,
        &[("prompt", &vec![b'x'; 28_800_078]), ("file", &short_wav())],
    )
    .await;
    eprintln!("CASE 28.8 MB ignored field: {status} {body}");
    assert_eq!(status, 413, "{body}");
    assert!(body.contains(&CAP.to_string()), "{body}");
}

/// A multipart request with no boundary is refused by axum's extractor before
/// the route runs: 400, plain text.
#[tokio::test]
async fn a_form_with_no_boundary_is_400() {
    let (state, _dir) = missing_whisper_state();
    let port = start_server_with_state(state).await;
    let body = form(&[("file", &short_wav())]);
    let headers = format!(
        "Content-Type: multipart/form-data\r\nContent-Length: {}\r\n",
        body.len()
    );
    let answer = exchange(port, &headers, body).await;
    eprintln!("CASE no boundary: {} {}", answer.status, answer.body);
    assert_eq!(answer.status, 400, "{}", answer.body);
    assert!(answer.body.contains("boundary"), "{}", answer.body);
}

/// A server with no Whisper paths answers every request 503 and tells the
/// client not to retry.
#[tokio::test]
async fn an_unconfigured_route_is_503_and_says_no_retry() {
    let state = not_ready_state(ModelRegistry::default());
    let port = start_server_with_state(state).await;
    let answer = post_audio(port, &short_wav()).await;
    assert_eq!(
        (answer.status, answer.body.as_str()),
        (
            503,
            r#"{"error":"audio model not configured; set --whisper-model-path"}"#
        )
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
        let Answer { status, body, .. } =
            post_form(port, &[(name, long.as_slice()), ("file", &wav)]).await;
        eprintln!(
            "CASE 1 MiB {name}: {status}, {} B: {}",
            body.len(),
            body.get(..120).unwrap_or(&body)
        );
        assert_eq!(status, 422, "{name}");
        assert!(body.len() < 200, "{name}: a {} B body", body.len());
        assert!(body.contains(name), "{name}: {body}");
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
    let Answer { status, body, .. } =
        post_form(port, &[("language", at_limit.as_bytes()), ("file", &wav)]).await;
    assert_eq!(status, 422, "{body}");
    assert!(body.contains(&at_limit), "{body}");
    let over = "q".repeat(65);
    let Answer { status, body, .. } =
        post_form(port, &[("language", over.as_bytes()), ("file", &wav)]).await;
    eprintln!("CASE 65-byte language: {status} {body}");
    assert_eq!(status, 422, "{body}");
    assert!(!body.contains(&at_limit), "{body}");
    assert!(body.contains("64 bytes"), "{body}");
}

// ── The token pick of a request, on the model ────────────────────────────────

/// A clip of two windows, as a 16 kHz mono WAV file: four words spoken 12
/// times by `say` with the English test voice, 20 s of zeros, and the same
/// speech again. Each window holds one true repetition.
fn repetition_wav() -> Option<Vec<u8>> {
    let wav = std::env::temp_dir().join("rmlx_route_repetition.wav");
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
        };
        let mut picker = RowPicker::default();
        let transcription = transcriber
            .transcribe(
                &samples,
                &opts,
                &mut |row, temperature| picker.pick(row, temperature),
                Device::Gpu,
                &AtomicBool::new(false),
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
