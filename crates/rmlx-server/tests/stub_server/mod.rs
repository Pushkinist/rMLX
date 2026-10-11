//! A server with no model behind it, for the HTTP tests.

#![allow(trivial_casts, reason = "the loader returns a boxed trait object")]

use std::sync::Arc;
use std::time::Duration;

use rmlx_server::{
    ApiErrorCounters, AppState, Generator, ItlStore, ModelLoader, ModelRegistry, NotReadyGenerator,
    SessionCache, TtftStore,
};

/// Build an `AppState` whose loader always returns `NotReadyGenerator`.
///
/// This matches the old test behaviour: model lookup in registry works (404
/// path tested), but generation returns 503 (NotReadyGenerator).
pub(crate) fn not_ready_state(registry: ModelRegistry) -> AppState {
    let loader: ModelLoader =
        Arc::new(|_path, _id| Ok(Box::new(NotReadyGenerator) as Box<dyn Generator>));
    AppState {
        device: rmlx_mlx::Device::Cpu,
        registry: Arc::new(registry),
        slots: Arc::new(parking_lot::RwLock::new(Vec::new())),
        embed_slot: Arc::new(parking_lot::RwLock::new(None)),
        mm_cache: Arc::new(rmlx_models::multimodal_cache::MultimodalCache::new(0)),
        gpu_gate: Arc::new(parking_lot::Mutex::new(())),
        gpu_queue: Arc::new(tokio::sync::Semaphore::new(1)),
        gpu_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_queue_depth: 64,
        max_loaded_models: 1,
        loader,
        metrics: None,
        idle_policy: rmlx_server::KeepAlivePolicy::Pin,
        max_tokens_cap: u32::MAX,
        max_timeout_secs: 600,
        session_cache: Arc::new(parking_lot::Mutex::new(SessionCache::new(4))),
        prompt_cache_slots: 4,
        ttft_store: TtftStore::default(),
        itl_store: ItlStore::default(),
        metrics_drainer: None,
        require_smoke_probe: false,
        default_temperature: None,
        default_enable_thinking: None,
        default_image_max_tokens: None,
        tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        started_at: std::time::Instant::now(),
        requests_started: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        admission_controller: None,
        admission_handle: None,
        whisper_model_path: None,
        whisper_tokenizer_path: None,
        audio_model: Arc::new(parking_lot::RwLock::new(None)),
        tts_model_path: None,
        tts_tokenizer_path: None,
        tts_model: Arc::new(parking_lot::RwLock::new(None)),
    }
}

/// Start a test server using the provided `AppState`.
#[allow(
    clippy::unwrap_used,
    reason = "a test server that cannot bind fails the test"
)]
pub(crate) async fn start_server_with_state(state: AppState) -> u16 {
    let router = rmlx_server::build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    port
}
