//! Unit tests for `generate_blocking` and `generate_streaming` behaviour
//! that does not require a real model or HTTP server.
//!
//! Convention: `#[cfg(test)] #[path = "generate_tests.rs"] mod generate_tests;`
//! in `generate.rs` wires this file. No inline test blocks elsewhere.

#![allow(
    clippy::unwrap_used,
    reason = "test-only: panics surface the root cause clearly"
)]
#![allow(
    clippy::expect_used,
    reason = "test-only: panics surface the root cause clearly"
)]

use std::pin::Pin;
use std::sync::Arc;
use std::time::Instant;

use futures::stream::{self, Stream};
use rmlx_core::Error;

use crate::engine::{GenerationRequest, GenerationToken, Generator};
use crate::openai::state::{ApiErrorCounters, TtftStore};
use crate::openai::TTFT_RING_CAPACITY;

// ── Minimal stub generator ────────────────────────────────────────────────────

/// Emits a fixed sequence of tokens then signals done.
struct FixedTokenGenerator {
    pieces: Vec<&'static str>,
}

impl Generator for FixedTokenGenerator {
    fn generate(
        &self,
        _req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        let pieces = self.pieces.clone();
        let n = pieces.len();
        let tokens: Vec<rmlx_core::Result<GenerationToken>> = pieces
            .into_iter()
            .enumerate()
            .map(|(i, piece)| {
                let done = i + 1 == n;
                Ok(GenerationToken {
                    token_id: i as u32,
                    piece: piece.to_owned(),
                    done,
                    finish_reason: if done { Some("stop".to_owned()) } else { None },
                    is_thinking: false,
                    logprobs: None,
                })
            })
            .collect();
        Box::pin(stream::iter(tokens))
    }
}

/// Emits a single error item — exercises the early-error path.
struct ErrorGenerator;

impl Generator for ErrorGenerator {
    fn generate(
        &self,
        _req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        Box::pin(stream::once(async {
            Err(Error::Other("stub engine error".to_owned()))
        }))
    }
}

// ── Helper: minimal GenerationRequest ────────────────────────────────────────

fn minimal_request(model_id: &str) -> GenerationRequest {
    GenerationRequest {
        model_id: model_id.to_owned(),
        prompt_tokens: vec![1, 2, 3],
        max_tokens: 16,
        sampling: crate::engine::types::SamplingParams::default(),
        stop: vec![],
        stream: false,
        system: None,
        session_id: None,
        effective_prompt_cache_slots: None,
        metrics_drainer: None,
        itl_store: None,
        event_recorder: None,
        tools: None,
        tool_choice: None,
        response_format: None,
        constraint: None,
        is_thinking_handle: None,
        thinking_budget: None,
        thinking_end_token_id: None,
        prompt_think_open: false,
        emit_tool_markers: false,
        thinking_start_token: None,
        thinking_end_token: None,
        gpu_admission: None,
        kv_quant_override: None,
        max_ctx_override: None,
        images: vec![],
        audio_b64: vec![],
        image_max_tokens: None,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Non-streaming `generate_blocking` must populate `ttft_store` with exactly
/// one entry per completed request, using the correct `model_id`.
#[tokio::test]
async fn blocking_ttft_ring_populated() {
    let ttft_store = TtftStore::default();
    let tokens_in = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tokens_out = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let error_counts = ApiErrorCounters::new();
    let requests_completed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let requests_failed = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let generator: Arc<dyn Generator> = Arc::new(FixedTokenGenerator {
        pieces: vec!["hello", " world"],
    });

    let _ = super::generate_blocking(
        generator,
        minimal_request("test-model"),
        None,
        "test-model",
        None,
        None,
        false,
        Instant::now(),
        None,
        0,
        &tokens_in,
        &tokens_out,
        "req-001",
        &error_counts,
        &requests_completed,
        &requests_failed,
        None,
        false,
        None,
        &ttft_store,
    )
    .await;

    let ring = ttft_store.lock();
    assert_eq!(ring.len(), 1, "one completed request → one TTFT sample");
    assert_eq!(
        ring[0].model_id, "test-model",
        "TTFT sample must carry the correct model_id"
    );
    assert!(
        ring[0].ttft_ms < 5_000,
        "TTFT must be a plausible wall-clock value"
    );
}

/// When the engine returns an error on the first token, `ttft_store` must
/// remain empty (no TTFT is recorded for failed requests).
#[tokio::test]
async fn blocking_ttft_ring_empty_on_engine_error() {
    let ttft_store = TtftStore::default();
    let tokens_in = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let tokens_out = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let error_counts = ApiErrorCounters::new();
    let requests_completed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let requests_failed = Arc::new(std::sync::atomic::AtomicU64::new(0));

    let generator: Arc<dyn Generator> = Arc::new(ErrorGenerator);

    let _ = super::generate_blocking(
        generator,
        minimal_request("test-model"),
        None,
        "test-model",
        None,
        None,
        false,
        Instant::now(),
        None,
        0,
        &tokens_in,
        &tokens_out,
        "req-002",
        &error_counts,
        &requests_completed,
        &requests_failed,
        None,
        false,
        None,
        &ttft_store,
    )
    .await;

    let ring = ttft_store.lock();
    assert_eq!(
        ring.len(),
        0,
        "engine error on first token → no TTFT sample (first token never arrived)"
    );
}

/// TTFT ring evicts the oldest entry once `TTFT_RING_CAPACITY` is reached.
#[tokio::test]
async fn blocking_ttft_ring_respects_capacity() {
    let ttft_store = TtftStore::default();
    let error_counts = ApiErrorCounters::new();
    let requests_completed = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let requests_failed = Arc::new(std::sync::atomic::AtomicU64::new(0));

    // Run CAPACITY + 2 requests to verify oldest entries are evicted.
    for i in 0..(TTFT_RING_CAPACITY + 2) {
        let generator: Arc<dyn Generator> = Arc::new(FixedTokenGenerator {
            pieces: vec!["tok"],
        });
        let tokens_in = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let tokens_out = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let model = format!("model-{i}");
        let _ = super::generate_blocking(
            generator,
            minimal_request(&model),
            None,
            &model,
            None,
            None,
            false,
            Instant::now(),
            None,
            0,
            &tokens_in,
            &tokens_out,
            "req-cap",
            &error_counts,
            &requests_completed,
            &requests_failed,
            None,
            false,
            None,
            &ttft_store,
        )
        .await;
    }

    let ring = ttft_store.lock();
    assert_eq!(
        ring.len(),
        TTFT_RING_CAPACITY,
        "ring must not exceed TTFT_RING_CAPACITY after overflow"
    );
    // The two oldest entries (model-0, model-1) should have been evicted.
    assert!(
        ring.iter().all(|s| s.model_id != "model-0"),
        "model-0 must have been evicted"
    );
    assert!(
        ring.iter().all(|s| s.model_id != "model-1"),
        "model-1 must have been evicted"
    );
}

// ── `response_format` reply: logprobs ─────────────────────────────────────────

/// Emits one token for each `(piece, is_thinking)` with a logprob record whose
/// token surface is the piece, then the terminal token with the text `tail`.
struct LogprobGenerator {
    pieces: Vec<(&'static str, bool)>,
    tail: &'static str,
}

impl Generator for LogprobGenerator {
    fn generate(
        &self,
        _req: GenerationRequest,
    ) -> Pin<Box<dyn Stream<Item = rmlx_core::Result<GenerationToken>> + Send>> {
        let token = |piece: &str, is_thinking: bool, done: bool| {
            Ok(GenerationToken {
                token_id: 1,
                piece: piece.to_owned(),
                done,
                finish_reason: done.then(|| "stop".to_owned()),
                is_thinking,
                logprobs: (!done).then(|| crate::openai::ChatLogprobContent {
                    token: piece.to_owned(),
                    logprob: -0.5,
                    bytes: None,
                    top_logprobs: vec![],
                }),
            })
        };
        let mut tokens: Vec<_> = self
            .pieces
            .iter()
            .map(|(piece, is_thinking)| token(piece, *is_thinking, false))
            .collect();
        tokens.push(token(self.tail, false, true));
        Box::pin(stream::iter(tokens))
    }
}

/// The JSON body of one `generate_blocking` reply whose constraint reported
/// `engagement`.
async fn blocking_reply(
    generator: LogprobGenerator,
    engagement: rmlx_models::Engagement,
) -> serde_json::Value {
    let ttft_store = TtftStore::default();
    let counter = || Arc::new(std::sync::atomic::AtomicU64::new(0));
    let (tokens_in, tokens_out) = (counter(), counter());
    let (requests_completed, requests_failed) = (counter(), counter());
    let error_counts = ApiErrorCounters::new();
    let response = super::generate_blocking(
        Arc::new(generator),
        minimal_request("test-model"),
        None,
        "test-model",
        None,
        Some(Arc::new(engagement)),
        false,
        Instant::now(),
        None,
        0,
        &tokens_in,
        &tokens_out,
        "req-json",
        &error_counts,
        &requests_completed,
        &requests_failed,
        None,
        false,
        None,
        &ttft_store,
    )
    .await;
    let body = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    serde_json::from_slice(&body).unwrap()
}

/// The non-streamed `logprobs` cover the reasoning tokens and the answer
/// tokens whose bytes are in the content: the stream sends the same set.
#[tokio::test]
async fn blocking_json_mode_logprobs_cover_the_tokens_of_the_content() {
    let generator = LogprobGenerator {
        pieces: vec![("hm", true), ("```", false), ("\n{", false), ("}", false)],
        tail: "",
    };
    // The grammar engaged at the brace of the third token.
    let engagement = rmlx_models::Engagement::default();
    engagement.report(3, 1);

    let body = blocking_reply(generator, engagement).await;
    let choice = &body["choices"][0];
    assert_eq!(choice["message"]["content"], "{}");
    let surfaces: Vec<&str> = choice["logprobs"]["content"]
        .as_array()
        .expect("the reply has logprobs")
        .iter()
        .map(|entry| entry["token"].as_str().unwrap())
        .collect();
    assert_eq!(surfaces, ["hm", "\n{", "}"]);
}

/// The terminal token can carry text (the end of a character that the
/// detokenizer held). A reply can start in it.
#[tokio::test]
async fn blocking_json_mode_reply_can_start_in_the_terminal_token() {
    let generator = LogprobGenerator {
        pieces: vec![("```", false)],
        tail: "\n{",
    };
    let engagement = rmlx_models::Engagement::default();
    engagement.report(2, 1);

    let body = blocking_reply(generator, engagement).await;
    assert_eq!(body["choices"][0]["message"]["content"], "{");
}
