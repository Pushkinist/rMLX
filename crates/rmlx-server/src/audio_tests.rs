//! The stop of a blocking audio job: an upload decode that reads the flag of
//! its request, and a client that disconnects while the job runs (no model, no
//! GPU). The flag and the permit of a job are held in `engine/tests.rs`.
//!
//! Not covered here: `handle_audio` handing the flag to `decode_upload` and to
//! `Transcriber::transcribe`. That needs a loaded Whisper model; only a
//! real-server run covers it.

use std::sync::atomic::Ordering;
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncWriteExt;

use rmlx_audio::tokenizer::WhisperTask;
use rmlx_audio::transcribe::{render, OutputFormat, Segment, Transcription, Word};
use rmlx_audio::wav::WavError;

use super::form::ResponseFormat;
use super::{
    build_response, cut, job_answer, job_refusal, prompt_error, refused, run_admitted, JobError,
};
use crate::engine::tests::{admitted, gpu_queue, report, wait_for_flag, REPORT_WAIT};

/// A request that is gone while its upload decodes: `decode_upload` under
/// `run_admitted`'s flag reads it before each packet and stops, and the permit
/// is free.
/// The job is held before its decode until the request future is dropped, so
/// the order does not depend on how fast the decoder runs.
#[tokio::test(flavor = "multi_thread")]
async fn a_request_gone_during_the_decode_stops_it_and_frees_the_permit() {
    // 313 silent ADTS frames of 1024 samples at 16 kHz: 20 s.
    const SILENT_ADTS: [u8; 11] = [
        0xFF, 0xF1, 0x60, 0x40, 0x01, 0x7F, 0xFC, 0x00, 0xC8, 0x00, 0x07,
    ];
    let upload = SILENT_ADTS.repeat(313);
    let (queue, pending) = gpu_queue();
    let guard = admitted(&queue, &pending).await;
    let (go_tx, go_rx) = mpsc::channel::<()>();
    let (done_tx, done_rx) = mpsc::channel();
    let request = run_admitted(Some(guard), move |cancel| {
        go_rx.recv_timeout(REPORT_WAIT).ok();
        done_tx.send(super::decode_upload(&upload, cancel)).ok();
    });
    // The timeout drops the request future, as the 408 does: the flag is set.
    let timed_out = tokio::time::timeout(Duration::from_millis(50), request).await;
    assert!(timed_out.is_err(), "the job is held before its decode");
    go_tx.send(()).expect("the job waits for go");

    let decoded = report(done_rx).await.expect("the job reports");
    assert!(
        matches!(decoded, Err(JobError::Decode(WavError::Cancelled { .. }))),
        "{decoded:?}"
    );
    let permit = tokio::time::timeout(REPORT_WAIT, queue.acquire()).await;
    assert!(
        matches!(permit, Ok(Ok(_))),
        "the permit is not free after the decode stopped"
    );
    assert_eq!(pending.load(Ordering::Acquire), 0);
}

/// A client that closes its connection while the job runs: the server drops
/// the request future, and the job sees the flag.
#[tokio::test(flavor = "multi_thread")]
async fn a_client_that_disconnects_sets_the_flag() {
    let (queue, pending) = gpu_queue();
    let (running_tx, running_rx) = mpsc::channel();
    let (saw_tx, saw_rx) = mpsc::channel();
    let route = axum::routing::post(move |body: axum::body::Bytes| {
        let (queue, pending) = (Arc::clone(&queue), Arc::clone(&pending));
        let (running_tx, saw_tx) = (running_tx.clone(), saw_tx.clone());
        async move {
            let guard = admitted(&queue, &pending).await;
            let result = run_admitted(Some(guard), move |cancel| {
                running_tx.send(()).ok();
                saw_tx.send(wait_for_flag(cancel)).ok();
            })
            .await;
            format!("{} bytes, {result:?}", body.len())
        }
    });
    let router = axum::Router::new().route("/job", route);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind a local port");
    let port = listener.local_addr().expect("local address").port();
    tokio::spawn(async move { axum::serve(listener, router).await });

    let mut client = tokio::net::TcpStream::connect(("127.0.0.1", port))
        .await
        .expect("connect");
    client
        .write_all(b"POST /job HTTP/1.1\r\nHost: test\r\nContent-Length: 3\r\n\r\nabc")
        .await
        .expect("send the request");
    assert_eq!(report(running_rx).await, Ok(()), "the job did not start");
    drop(client);
    assert_eq!(
        report(saw_rx).await,
        Ok(true),
        "the job did not see the flag"
    );
}

/// Two segments, as a transcription returns them.
fn two_segments() -> Transcription {
    Transcription {
        text: " The first sentence. The second one.".to_owned(),
        segments: vec![
            Segment {
                seek: 0,
                start: 0.0,
                end: 2.5,
                text: "The first sentence.".to_owned(),
                tokens: vec![50_365, 440, 700, 13, 50_490],
                temperature: 0.0,
                avg_logprob: -0.25,
                compression_ratio: 0.9,
                no_speech_prob: 0.03,
            },
            Segment {
                seek: 3_000,
                start: 2.5,
                end: 5.04,
                text: "The second one.".to_owned(),
                tokens: vec![50_490, 440, 1_150, 13, 50_617],
                temperature: 0.5,
                avg_logprob: -0.75,
                compression_ratio: 1.1,
                no_speech_prob: 0.125,
            },
        ],
        language: "english".to_owned(),
        duration: 5.04,
        words: None,
    }
}

/// [`two_segments`] with the words of a request for word times.
fn two_segments_with_words() -> Transcription {
    let word = |word: &str, start: f32, end: f32, segment: usize| Word {
        word: word.to_owned(),
        start,
        end,
        segment,
    };
    Transcription {
        words: Some(vec![
            word(" The", 0.0, 0.32, 0),
            word(" first", 0.32, 0.9, 0),
            word(" sentence.", 0.9, 2.5, 0),
            word(" The", 2.5, 2.88, 1),
            word(" second", 2.88, 3.4, 1),
            word(" one.", 3.4, 5.04, 1),
        ]),
        ..two_segments()
    }
}

async fn body_of(format: ResponseFormat) -> String {
    body_of_transcription(&two_segments(), format).await
}

async fn body_of_transcription(transcription: &Transcription, format: ResponseFormat) -> String {
    let response = build_response(transcription, WhisperTask::Transcribe, format);
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    String::from_utf8(bytes.to_vec()).expect("the body is UTF-8")
}

/// `json`, `text`, `srt` and `vtt` answer fixed bytes.
#[tokio::test]
async fn the_plain_formats_answer_unchanged_bytes() {
    for (format, expected) in [
        (
            ResponseFormat::Json,
            r#"{"text":"The first sentence. The second one."}"#,
        ),
        (ResponseFormat::Text, "The first sentence. The second one."),
        (
            ResponseFormat::Srt,
            "1\n00:00:00,000 --> 00:00:02,500\nThe first sentence.\n\n2\n00:00:02,500 --> 00:00:05,040\nThe second one.\n\n",
        ),
        (
            ResponseFormat::Vtt,
            "WEBVTT\n\n00:00:00.000 --> 00:00:02.500\nThe first sentence.\n\n00:00:02.500 --> 00:00:05.040\nThe second one.\n\n",
        ),
    ] {
        assert_eq!(body_of(format).await, expected, "{format:?}");
    }
}

/// A `verbose_json` segment holds the ten keys of the OpenAI
/// `TranscriptionSegment` schema, each with the value of the engine's segment
/// and the type of the schema. A `null` would make a typed client throw
/// (openai-dotnet reads them with `GetSingle`). The temperature, the average
/// log-probability and the no-speech probability are those of the segment, not
/// constants.
#[tokio::test]
async fn verbose_json_segments_carry_the_ten_openai_keys() {
    let body: serde_json::Value =
        serde_json::from_str(&body_of(ResponseFormat::VerboseJson).await).expect("JSON");
    assert_eq!(body["task"], "transcribe");
    assert_eq!(body["language"], "english");
    assert_eq!(body["duration"], 5.04);
    assert_eq!(body["text"], "The first sentence. The second one.");
    let segments = body["segments"].as_array().expect("segments");
    assert_eq!(segments.len(), 2);
    let segment = &segments[1];
    let mut keys: Vec<&str> = segment
        .as_object()
        .expect("a segment is an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(
        keys,
        [
            "avg_logprob",
            "compression_ratio",
            "end",
            "id",
            "no_speech_prob",
            "seek",
            "start",
            "temperature",
            "text",
            "tokens",
        ]
    );
    assert_eq!(segment["id"], 1);
    assert_eq!(segment["start"], 2.5);
    assert_eq!(segment["end"], 5.04);
    assert_eq!(segment["text"], "The second one.");
    assert_eq!(segment["temperature"], 0.5);
    assert_eq!(segment["avg_logprob"], -0.75);
    assert_eq!(segment["seek"], 3_000);
    assert_eq!(segment["compression_ratio"], 1.1);
    assert_eq!(segment["no_speech_prob"], 0.125);
    assert_eq!(
        segment["tokens"],
        serde_json::json!([50_490, 440, 1_150, 13, 50_617])
    );
}

/// `rmlx transcribe --format json` writes the segments that the route writes
/// in `verbose_json`, key for key and value for value: one producer builds
/// both.
#[tokio::test]
async fn the_cli_json_segments_are_the_verbose_json_segments() {
    let transcription = two_segments();
    let cli: serde_json::Value =
        serde_json::from_str(&render(&transcription, OutputFormat::Json).expect("renders"))
            .expect("JSON");
    let route: serde_json::Value =
        serde_json::from_str(&body_of(ResponseFormat::VerboseJson).await).expect("JSON");
    assert_eq!(cli["segments"], route["segments"]);
}

/// A `verbose_json` reply with no request for word times has the five keys
/// it had before word times were read: no `words`.
#[tokio::test]
async fn verbose_json_without_words_has_no_words_key() {
    let body: serde_json::Value =
        serde_json::from_str(&body_of(ResponseFormat::VerboseJson).await).expect("JSON");
    let mut keys: Vec<&str> = body
        .as_object()
        .expect("the reply is an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    assert_eq!(keys, ["duration", "language", "segments", "task", "text"]);
}

/// A `verbose_json` reply with word times holds a top-level `words` array
/// beside `segments`. Each word has the three keys of the OpenAI
/// `TranscriptionWord` schema, and its text has no leading space.
#[tokio::test]
async fn verbose_json_with_words_holds_the_words_beside_the_segments() {
    let body: serde_json::Value = serde_json::from_str(
        &body_of_transcription(&two_segments_with_words(), ResponseFormat::VerboseJson).await,
    )
    .expect("JSON");
    assert_eq!(body["segments"].as_array().expect("segments").len(), 2);
    let words = body["words"].as_array().expect("words");
    let texts: Vec<&str> = words.iter().filter_map(|w| w["word"].as_str()).collect();
    assert_eq!(
        texts,
        ["The", "first", "sentence.", "The", "second", "one."]
    );
    for word in words {
        let mut keys: Vec<&str> = word
            .as_object()
            .expect("a word is an object")
            .keys()
            .map(String::as_str)
            .collect();
        keys.sort_unstable();
        assert_eq!(keys, ["end", "start", "word"]);
    }
    assert_eq!(words[1]["start"], 0.32);
    assert_eq!(words[5]["end"], 5.04);
}

/// `rmlx transcribe --format json` writes the words that the route writes in
/// `verbose_json`: one producer builds both.
#[tokio::test]
async fn the_cli_json_words_are_the_verbose_json_words() {
    let transcription = two_segments_with_words();
    let cli: serde_json::Value =
        serde_json::from_str(&render(&transcription, OutputFormat::Json).expect("renders"))
            .expect("JSON");
    let route: serde_json::Value = serde_json::from_str(
        &body_of_transcription(&transcription, ResponseFormat::VerboseJson).await,
    )
    .expect("JSON");
    assert_eq!(cli["words"], route["words"]);
    assert!(cli["words"].is_array());
}

/// Each job failure and its answer: the file is the client's (422); a load,
/// a refused vocabulary or the transcriber's set-up is the server's until the
/// operator acts (503, no retry); an engine failure during the transcription
/// is the server's and may pass (503, retry allowed).
#[test]
fn each_job_failure_has_its_status_and_retry() {
    use axum::http::StatusCode;
    use rmlx_audio::tokenizer::TokenizerError;
    use rmlx_audio::wav::WavError;
    use rmlx_audio::whisper::WhisperError;

    let mlx = || WhisperError::Mlx("op failed".to_owned());
    let cases = [
        (
            JobError::Decode(WavError::Empty),
            StatusCode::UNPROCESSABLE_ENTITY,
            false,
            "audio decode: ",
        ),
        (
            JobError::Tokenizer(TokenizerError::Load("tokenizer.json: gone".to_owned())),
            StatusCode::SERVICE_UNAVAILABLE,
            true,
            "tokenizer: ",
        ),
        (
            JobError::Model(mlx()),
            StatusCode::SERVICE_UNAVAILABLE,
            true,
            "whisper load: ",
        ),
        (
            JobError::Init(mlx()),
            StatusCode::SERVICE_UNAVAILABLE,
            true,
            "transcriber init: ",
        ),
        (
            JobError::Transcribe(mlx()),
            StatusCode::SERVICE_UNAVAILABLE,
            false,
            "transcribe: ",
        ),
    ];
    for (error, status, permanent, prefix) in cases {
        let refusal = job_refusal(&error);
        assert_eq!(refusal.status, status, "{error:?}");
        assert_eq!(refusal.permanent, permanent, "{error:?}");
        assert!(refusal.message.starts_with(prefix), "{refusal:?}");
    }
}

/// The text of a special token in the prompt is the client's fault: 422 naming
/// the field. Any other failure of the tokenizer on the prompt is the server's:
/// 503, as for each other tokenizer failure.
#[test]
fn only_special_token_text_in_a_prompt_is_a_422() {
    use axum::http::StatusCode;
    use rmlx_audio::tokenizer::TokenizerError;

    let refusal = job_refusal(&prompt_error(TokenizerError::SpecialTokenText));
    assert_eq!(refusal.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(refusal.param.as_deref(), Some("prompt"));
    assert!(!refusal.permanent);
    assert!(
        refusal.message.starts_with("invalid prompt: "),
        "{refusal:?}"
    );

    let refusal = job_refusal(&prompt_error(TokenizerError::Encode("gone".to_owned())));
    assert_eq!(refusal.status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(refusal.param, None);
    assert!(refusal.message.starts_with("tokenizer: "), "{refusal:?}");
}

/// A request for word times on a model with no alignment heads is the
/// client's fault: 422 naming `timestamp_granularities`, with a message that
/// says why, and no retry header.
#[test]
fn words_from_a_model_with_no_alignment_heads_are_a_422() {
    use axum::http::StatusCode;
    use rmlx_audio::whisper::WhisperError;

    let refusal = job_refusal(&JobError::Transcribe(WhisperError::NoAlignmentHeads));
    assert_eq!(refusal.status, StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(refusal.param.as_deref(), Some("timestamp_granularities"));
    assert!(!refusal.permanent);
    assert!(
        refusal.message.contains("no alignment heads"),
        "{refusal:?}"
    );
}

/// A permanent refusal carries `x-should-retry: false`; another carries no
/// such header.
#[test]
fn only_a_permanent_refusal_says_no_retry() {
    let permanent = refused(job_refusal(&JobError::Init(
        rmlx_audio::whisper::WhisperError::Mlx("x".to_owned()),
    )));
    assert_eq!(
        permanent
            .headers()
            .get("x-should-retry")
            .map(axum::http::HeaderValue::as_bytes),
        Some(b"false".as_slice())
    );
    let passing = refused(job_refusal(&JobError::Transcribe(
        rmlx_audio::whisper::WhisperError::Mlx("x".to_owned()),
    )));
    assert!(passing.headers().get("x-should-retry").is_none());
}

#[test]
fn cut_keeps_whole_characters() {
    assert_eq!(cut("abc", 64), "abc");
    assert_eq!(cut("aé", 2), "a");
    assert_eq!(cut("", 4), "");
}

/// The JSON of a response's body.
async fn json_of(response: axum::response::Response) -> serde_json::Value {
    let bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .expect("the body reads");
    serde_json::from_slice(&bytes).expect("the body is JSON")
}

/// Each job failure is answered in the error envelope: the file is a bad
/// request that names `file`; every other fault is the server's, and names no
/// request field.
#[tokio::test]
async fn each_job_failure_is_answered_in_the_error_envelope() {
    use rmlx_audio::tokenizer::TokenizerError;
    use rmlx_audio::whisper::WhisperError;

    let mlx = || WhisperError::Mlx("op failed".to_owned());
    let cases = [
        (
            JobError::Decode(WavError::Empty),
            "invalid_request_error",
            Some("file"),
        ),
        (
            JobError::Tokenizer(TokenizerError::Load("tokenizer.json: gone".to_owned())),
            "service_unavailable",
            None,
        ),
        (JobError::Model(mlx()), "service_unavailable", None),
        (JobError::Init(mlx()), "service_unavailable", None),
        (JobError::Transcribe(mlx()), "service_unavailable", None),
    ];
    for (error, error_type, param) in cases {
        let refusal = job_refusal(&error);
        let message = refusal.message.clone();
        let json = json_of(refused(refusal)).await;
        let expected = match param {
            Some(param) => serde_json::json!({
                "error": {"message": message, "type": error_type, "param": param}
            }),
            None => serde_json::json!({"error": {"message": message, "type": error_type}}),
        };
        assert_eq!(json, expected, "{error:?}");
    }
}

/// A job that panics on the blocking pool is answered as a 500 in the error
/// envelope, and the text names no cause. The panic is real: `run_admitted`
/// returns the `JoinError` that `job_answer` gets in the handler.
#[tokio::test(flavor = "multi_thread")]
async fn a_panicked_job_is_answered_in_the_error_envelope() {
    let (queue, pending) = gpu_queue();
    let guard = admitted(&queue, &pending).await;
    let result = run_admitted(Some(guard), |_| -> Result<Transcription, JobError> {
        panic!("scripted panic, secret cause");
    })
    .await;
    assert!(result.is_err(), "the scripted job panics");
    let response = job_answer(result, WhisperTask::Transcribe, ResponseFormat::Json);
    assert_eq!(
        response.status(),
        axum::http::StatusCode::INTERNAL_SERVER_ERROR
    );
    assert_eq!(
        json_of(response).await,
        serde_json::json!({"error": {"message": "internal error", "type": "internal_error"}})
    );
    assert_eq!(queue.available_permits(), 1, "the panic frees the permit");
}

/// An MLX error message can hold a path of the server's MLX install (the
/// metallib load error does), so no reply carries it, whichever Whisper error
/// carries it. The weight-file case is built from the audio crate's own
/// `NpzError` message, so a change of that message turns the test red. The
/// file and the step stay in the reply. Another Whisper error text is sent as
/// it is.
#[test]
fn an_mlx_error_text_is_not_sent_in_a_reply() {
    use rmlx_audio::npz::NpzError;
    use rmlx_audio::whisper::WhisperError;

    let secret = "eval: Failed to load the default metallib. /opt/lib/mlx/mlx.metallib";
    let mlx = || WhisperError::Mlx(secret.to_owned());
    let weight_file =
        || WhisperError::WeightFile(format!("weights.npz: {}", NpzError::Mlx(secret.to_owned())));
    let cases = [
        (JobError::Model(mlx()), "whisper load: "),
        (JobError::Init(mlx()), "transcriber init: "),
        (JobError::Transcribe(mlx()), "transcribe: "),
        (
            JobError::Model(weight_file()),
            "whisper load: weight file error: weights.npz: ",
        ),
        (
            JobError::Transcribe(weight_file()),
            "transcribe: weight file error: weights.npz: ",
        ),
    ];
    for (error, kept) in cases {
        let message = job_refusal(&error).message;
        assert!(message.starts_with(kept), "{message}");
        assert!(!message.contains('/'), "{message}");
        assert!(!message.contains("metallib"), "{message}");
        assert!(!message.contains("eval"), "{message}");
        assert!(message.contains("MLX error"), "{message}");
    }
    let message = job_refusal(&JobError::Transcribe(WhisperError::Config("x".to_owned()))).message;
    assert_eq!(message, "transcribe: config parse error: x");
    let message = job_refusal(&JobError::Model(WhisperError::WeightFile(
        "weights.npz: file I/O: No such file or directory (os error 2)".to_owned(),
    )))
    .message;
    assert_eq!(
        message,
        "whisper load: weight file error: weights.npz: file I/O: No such file or directory (os error 2)"
    );
}
