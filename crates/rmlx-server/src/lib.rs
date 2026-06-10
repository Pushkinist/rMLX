//! HTTP server with two chat-compatible surfaces.
//!
//! Both paths share one token stream. Schemas differ only in field names.
//! Stage 3.5: multi-model registry, load/unload/swap, idle eviction.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::float_cmp,
        clippy::ignore_without_reason,
        clippy::duration_suboptimal_units,
        // disallowed_methods is a separate lint from unwrap_used;
        // test code (bucket-B) is already exempted for unwrap_used, extend here.
        clippy::disallowed_methods,
    )
)]

pub mod admission;
pub mod anthropic;
pub mod audio;
pub mod bounds;
pub mod catch_unwind;
pub mod chat_template;
pub mod claim;
pub mod constraint_json;
pub mod detokenizer;
pub mod embeddings;
pub mod engine;
pub mod generation_config_io;
pub mod image_io;
pub mod keep_alive;
pub mod logged_json;
pub mod metrics_drainer;
pub mod openai;
pub mod registry;
pub mod retry;
pub mod session_cache;
pub mod stop_matcher;
pub mod tokenizer_io;
pub mod tool_parser;

pub use claim::{try_claim, ClaimError, MetalClaim, SENTINEL_PORT};

use std::net::SocketAddr;

use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use axum::serve::ListenerExt;
use axum::Router;
use serde_json::json;
use tracing::info;

pub use admission::{
    spawn_controller_task, AdmissionHandle, ControllerConfig, ControllerHandle, DecisionReason,
    StepMetrics,
};
pub use engine::{
    ArchGenerator, GenerationRequest, GenerationToken, Generator, ModelLoadConfig,
    NotReadyGenerator, SpeculativeGenerator,
};
pub use keep_alive::{
    parse_duration_spec, policy_from_request_field, DecodeLease, DecodeLeaseGuard, KeepAlivePolicy,
};
pub use metrics_drainer::{spawn_drainer, DrainerHandle, MetricEvent, MetricKind};
pub use openai::{
    compute_effective_timeout, register_ssd_prom_hooks, timeout_mw, ApiErrorCategory,
    ApiErrorCounters, AppState, ItlSample, ItlStore, LoadedModel, ModelLoader, TtftSample,
    TtftStore, UnloadReason, ITL_RING_CAPACITY, TTFT_RING_CAPACITY,
};
pub use registry::{ModelEntry, ModelRegistry, RegistryConfig, RegistryConfigEntry};
pub use session_cache::{SessionCache, SessionKey};

/// Build the axum `Router` for the full API surface.
///
/// OpenAI routes:
/// `POST /v1/chat/completions`
/// `GET /v1/models`
/// `POST /v1/models/:id/load` — G3: blocks until resident; 200 on ready (synchronous, no 202/poll)
/// `POST /v1/models/:id/unload`
/// `GET /v1/models/:id/status`
///
/// Anthropic routes:
/// `POST /v1/messages`
///
/// Health:
/// `GET /health`
///
/// Metrics:
/// `GET /metrics/cache` — prompt-cache hit/miss/bytes (N19) + TTFT ring-buffer (L6) — JSON
/// `GET /metrics` — Prometheus text exposition v0.0.4 (F5) — same data as /metrics/cache
/// `GET /v1/metrics` — rolling request-level JSON summary — mlx-vlm compatible
pub fn build_router(state: AppState) -> Router {
    Router::new()
        // Liveness probe.
        .route("/health", get(health))
        // OpenAI chat.
        .route("/v1/chat/completions", post(openai::chat_completions))
        // Model lifecycle.
        .route("/v1/models", get(openai::list_models))
        .route("/v1/models/{id}/load", post(openai::load_model))
        .route("/v1/models/{id}/unload", post(openai::unload_model))
        .route("/v1/models/{id}/status", get(openai::model_status))
        // Embeddings (jina-embeddings-v4 — text + image).
        .route("/v1/embeddings", post(embeddings::embeddings))
        // Audio STT: Whisper transcription + translation.
        .route(
            "/v1/audio/transcriptions",
            post(audio::audio_transcriptions),
        )
        .route("/v1/audio/translations", post(audio::audio_translations))
        // Audio TTS: Qwen3-TTS speech synthesis.
        .route("/v1/audio/speech", post(audio::audio_speech))
        // Anthropic API.
        .route("/v1/messages", post(anthropic::messages))
        // Metrics (N19): JSON.
        .route("/metrics/cache", get(openai::metrics_cache))
        // Metrics (F5): Prometheus text exposition v0.0.4.
        .route("/metrics", get(openai::metrics_prometheus))
        // Metrics: rolling request-level JSON summary (mlx-vlm compatible).
        .route("/v1/metrics", get(openai::metrics_v1_summary))
        // A8: per-request HTTP timeout middleware.
        // Applied after routing so it sees every handler (including /health).
        // Reads X-Request-Timeout-Seconds header; caps at AppState::max_timeout_secs.
        .layer(middleware::from_fn_with_state(state.clone(), timeout_mw))
        // Catches any handler panic and returns 500 instead of
        // dropping the connection or crashing the worker.
        .layer(catch_unwind::CatchUnwindLayer::new())
        // Transport-level body limit.
        //
        // Gate ordering:
        //   1. This transport limit (outermost axum layer) rejects bodies before
        //      they reach any handler or inner middleware.
        //   2. The audio handler has a per-field 25 MiB check (MAX_AUDIO_BYTES).
        //   3. Bounds checks in bounds.rs apply per-field caps for chat/embed routes.
        //
        // Sizing: the audio route advertises a 25 MiB audio payload cap. Multipart
        // framing (boundary markers, field headers) and the base64 overhead for
        // chat/embed image inputs both consume additional bytes.  26 MiB = 25 MiB
        // audio cap + 1 MiB framing slack — the tightest limit that guarantees a
        // valid 25 MiB audio upload is never dropped by the transport gate.
        .layer(DefaultBodyLimit::max(26 * 1024 * 1024))
        .with_state(state)
}

/// Bind to `host:port` and serve until the process is interrupted.
///
/// Caller is responsible for owning the tokio runtime.
pub async fn serve(state: AppState, host: &str, port: u16) -> anyhow::Result<()> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid address {host}:{port}: {e}"))?;

    let router = build_router(state);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;

    // axum 0.8 removed implicit TCP_NODELAY. With Nagle's algorithm enabled by
    // default, small SSE frames (one chunked-encoding event per token, ~100
    // bytes) interact with macOS delayed-ACK to add ~10–12 ms/token of
    // server→client latency. Setting TCP_NODELAY on every accepted socket
    // is what oMLX/mlx-lm/uvicorn already do via Python's default — closing
    // this gap was the dominant contributor to the 26b-a4b bench-vs-internal
    // 36→60 TPS gap.
    let listener = listener.tap_io(|tcp_stream| {
        if let Err(err) = tcp_stream.set_nodelay(true) {
            tracing::trace!("failed to set TCP_NODELAY on incoming connection: {err:#}");
        }
    });

    info!(address = %addr, "rmlx-server listening");

    axum::serve(listener, router)
        .await
        .map_err(|e| anyhow::anyhow!("serve: {e}"))
}

// ── Health handler ────────────────────────────────────────────────────────────

async fn health() -> axum::Json<serde_json::Value> {
    axum::Json(json!({"ok": true}))
}
