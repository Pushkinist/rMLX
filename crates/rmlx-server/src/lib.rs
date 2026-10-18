//! HTTP server with two chat-compatible surfaces.
//!
//! Both paths share one token stream. Schemas differ only in field names.
//! Multi-model registry, load/unload/swap, idle eviction.

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
pub mod image_io;
pub mod keep_alive;
pub mod logged_json;
pub mod metrics_drainer;
pub mod openai;
pub mod registry;
pub mod retry;
pub mod session_cache;
pub mod shutdown;
pub mod stop_matcher;
mod stream_cut;
pub mod tokenizer_io;
pub mod tool_parser;
pub(crate) mod tts_sampling;

pub use claim::{probe_claim, try_claim, ClaimError, MetalClaim};

use std::net::SocketAddr;

use axum::extract::DefaultBodyLimit;
use axum::middleware;
use axum::routing::{get, post};
use axum::Router;
use serde_json::json;
use tracing::{info, warn};

use shutdown::PhaseWatch;

pub use admission::{
    spawn_controller_task, AdmissionHandle, ControllerConfig, ControllerHandle, DecisionReason,
    StepMetrics,
};
pub use engine::{
    ArchGenerator, GenerationRequest, GenerationToken, Generator, ModelLoadConfig,
    NotReadyGenerator, SpeculativeGenerator, MIN_DRAFT_BLOCK_SIZE,
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
pub use session_cache::{effective_prompt_cache_slots, SessionCache, SessionKey};
pub use shutdown::{block_on_serve, ExitDeadline};

/// Build the axum `Router` for the full API surface.
///
/// OpenAI routes:
/// `POST /v1/chat/completions`
/// `GET /v1/models`
/// `POST /v1/models/:id/load` — blocks until resident; 200 on ready (synchronous, no 202/poll)
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
/// `GET /metrics/cache` — prompt-cache hit/miss/bytes + TTFT ring-buffer — JSON
/// `GET /metrics` — Prometheus text exposition v0.0.4 — same data as /metrics/cache
/// `GET /v1/metrics` — rolling request-level JSON summary — mlx-vlm compatible
pub fn build_router(state: AppState) -> Router {
    routes(state, PhaseWatch::never())
}

/// The API surface under the layers of the server.
fn routes(state: AppState, phase: PhaseWatch) -> Router {
    with_layers(api(), state, phase)
}

/// Every route of the server. A route is added here and nowhere else, so that
/// `with_layers` covers it.
fn api() -> Router<AppState> {
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
        // Metrics: JSON.
        .route("/metrics/cache", get(openai::metrics_cache))
        // Metrics: Prometheus text exposition v0.0.4.
        .route("/metrics", get(openai::metrics_prometheus))
        // Metrics: rolling request-level JSON summary (mlx-vlm compatible).
        .route("/v1/metrics", get(openai::metrics_v1_summary))
}

/// The layers every route carries.
fn with_layers(api: Router<AppState>, state: AppState, phase: PhaseWatch) -> Router {
    api
        // Inside the timeout, so the 408 drops a request as before.
        .layer(middleware::from_fn_with_state(
            phase,
            shutdown::stop_at_shutdown,
        ))
        // Per-request HTTP timeout middleware.
        // Applied after routing so it sees every handler (including /health).
        // Reads X-Request-Timeout-Seconds header; caps at AppState::max_timeout_secs.
        .layer(middleware::from_fn_with_state(state.clone(), timeout_mw))
        // Catches any handler panic and returns 500 instead of
        // dropping the connection or crashing the worker.
        .layer(catch_unwind::CatchUnwindLayer::new())
        // Transport-level body limit, sized for one audio upload at its cap.
        // A JSON body over it is refused by its extractor; a multipart body
        // over it fails while the audio handler reads it, and the handler
        // answers 413. bounds.rs then applies the per-field caps of the chat
        // and embeddings routes.
        .layer(DefaultBodyLimit::max(audio::MAX_REQUEST_BYTES))
        .with_state(state)
}

/// Bind to `host:port` and serve until SIGINT or SIGTERM, then stop within the
/// bounds of [`shutdown`]. Run it through [`block_on_serve`], which ends the
/// runtime at the deadline this returns.
///
/// # Errors
/// The address is not valid, the bind failed, or the accept loop failed.
pub async fn serve(state: AppState, host: &str, port: u16) -> anyhow::Result<ExitDeadline> {
    let addr: SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| anyhow::anyhow!("invalid address {host}:{port}: {e}"))?;

    let gpu_pending = std::sync::Arc::clone(&state.gpu_pending);
    let (phase, phase_watch) = shutdown::phase_channel();
    let router = routes(state, phase_watch);
    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .map_err(|e| anyhow::anyhow!("bind {addr}: {e}"))?;

    info!(address = %addr, "rmlx-server listening");

    shutdown::serve_until(
        listener,
        router,
        phase,
        shutdown_signals(),
        shutdown::Bounds::SERVER,
        gpu_pending,
    )
    .await
}

/// One message for each SIGINT (Ctrl-C) and SIGTERM (`kill`) the process
/// receives. The first starts the shutdown of [`shutdown`]; the second ends
/// its wait for the open requests.
fn shutdown_signals() -> tokio::sync::mpsc::Receiver<()> {
    use tokio::signal::unix::{signal, SignalKind};

    // A signal whose handler cannot be installed keeps its default
    // disposition: the process ends at once, and the kernel frees the claim.
    let listen = |kind: SignalKind, name: &'static str| match signal(kind) {
        Ok(listener) => Some(listener),
        Err(e) => {
            warn!(error = %e, signal = name, "failed to install the signal handler");
            None
        }
    };
    let mut sigint = listen(SignalKind::interrupt(), "SIGINT");
    let mut sigterm = listen(SignalKind::terminate(), "SIGTERM");
    let (received, signals) = tokio::sync::mpsc::channel(2);
    tokio::spawn(async move {
        loop {
            let name = tokio::select! {
                Some(()) = next_signal(&mut sigint) => "SIGINT",
                Some(()) = next_signal(&mut sigterm) => "SIGTERM",
                else => return,
            };
            info!(signal = name, "received a shutdown signal");
            if received.send(()).await.is_err() {
                return;
            }
        }
    });
    signals
}

// cancel-safe: it only awaits the signal listener.
async fn next_signal(listener: &mut Option<tokio::signal::unix::Signal>) -> Option<()> {
    listener.as_mut()?.recv().await
}

// ── Health handler ────────────────────────────────────────────────────────────

async fn health() -> axum::Json<serde_json::Value> {
    axum::Json(json!({"ok": true}))
}
