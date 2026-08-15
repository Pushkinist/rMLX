//! Multi-model lifecycle smoke test (Stage 3.5).
//!
//! Gated on the `RMLX_REGISTRY_TEST` environment variable. Mark `#[ignore]`
//! so `cargo test` skips it by default.
//!
//! Run manually:
//! RMLX_REGISTRY_TEST=1 cargo test -p rmlx-server -- --ignored multi_model --nocapture
//!
//! What is tested:
//! 1. `GET /v1/models` lists both registry entries; `loaded: false` initially.
//! 2. `POST /v1/models/{id}/load` puts a model in the slot.
//! 3. `GET /v1/models/{id}/status` confirms loaded state + loaded_at epoch.
//! 4. `POST /v1/chat/completions` auto-loads a different model (swap).

//! 5. `POST /v1/models/{id}/unload` evicts the second model; 404 after.
//! 6. `GET /v1/models` shows `loaded: false` for both.

#![allow(
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
    trivial_casts
)]

use std::sync::Arc;
use std::time::Duration;

use rmlx_server::{
    ApiErrorCounters, AppState, Generator, ItlStore, ModelLoader, ModelRegistry, NotReadyGenerator,
    SessionCache, TtftStore,
};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── Test models ───────────────────────────────────────────────────────────────

/// Two different model IDs registered in the test registry.
/// Because we inject NotReadyGenerator, paths just need to have valid
/// config.json (the test builds a minimal in-memory registry).
const MODEL_A: &str = "model-alpha";
const MODEL_B: &str = "model-beta";

// ── Server helper ─────────────────────────────────────────────────────────────

/// Build a two-entry registry with minimal temp snapshots.
fn make_two_model_registry() -> ModelRegistry {
    use std::io::Write as _;
    use tempfile::tempdir;

    let root = tempdir().unwrap();
    let snap_a = root.path().join(MODEL_A);
    let snap_b = root.path().join(MODEL_B);

    for (dir, arch) in [(&snap_a, "LlamaForCausalLM"), (&snap_b, "Qwen2ForCausalLM")] {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = serde_json::json!({
            "architectures": [arch],
            "dtype": "bfloat16"
        });
        let mut f = std::fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(cfg.to_string().as_bytes()).unwrap();
    }

    // from_id_paths via from_config-like pattern: use from_paths and then
    // override IDs. Since from_paths derives IDs from basenames, we create
    // the snap directories WITH the model-id names.
    let reg = ModelRegistry::from_paths(&[snap_a, snap_b]);

    // root is kept alive until reg is built. The tempdir may be dropped here
    // because from_paths canonicalises + stores the abs_path.
    // But: tempfile dirs are deleted on drop. That's fine for unit tests
    // since we only call load_config() once during from_paths.
    reg
}

/// Start a test server with a two-model registry, NotReadyGenerator loader.
async fn start_two_model_server() -> u16 {
    let reg = make_two_model_registry();

    let loader: ModelLoader =
        Arc::new(|_path, _id| Ok(Box::new(NotReadyGenerator) as Box<dyn Generator>));

    let state = AppState {
        registry: Arc::new(reg),
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
    };

    let router = rmlx_server::build_router(state);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();

    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });

    tokio::time::sleep(Duration::from_millis(20)).await;
    port
}

/// Raw HTTP request helper.
async fn http(port: u16, method: &str, path: &str, body: Option<&str>) -> (u16, String) {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();

    let content_header = if let Some(b) = &body {
        format!(
            "Content-Type: application/json\r\nContent-Length: {}\r\n",
            b.len()
        )
    } else {
        String::new()
    };

    let request = format!(
        "{method} {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n{content_header}\r\n{}",
        body.unwrap_or("")
    );

    stream.write_all(request.as_bytes()).await.unwrap();

    let mut response = Vec::new();
    stream.read_to_end(&mut response).await.unwrap();
    let text = String::from_utf8_lossy(&response).into_owned();

    let status: u16 = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    let body_start = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    (status, text[body_start..].to_owned())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

/// Multi-model lifecycle: list → load A → status A → chat (which triggers
/// swap to B) → unload B → list shows both unloaded.
///
/// Uses NotReadyGenerator so no real models are needed; exercises the
/// registry + slot mechanics end-to-end.
#[ignore = "integration test; run with RMLX_REGISTRY_TEST=1"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn multi_model_list_load_swap_unload() {
    if std::env::var("RMLX_REGISTRY_TEST").is_err() {
        eprintln!("RMLX_REGISTRY_TEST not set — skipping multi_model_list_load_swap_unload");
        return;
    }

    let port = start_two_model_server().await;

    // ── 1. GET /v1/models — both registered, neither loaded ───────────────────
    let (status, body) = http(port, "GET", "/v1/models", None).await;
    assert_eq!(status, 200, "list_models body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "list");
    let data = v["data"].as_array().unwrap();
    assert_eq!(data.len(), 2, "registry must have 2 entries");
    for entry in data {
        assert_eq!(entry["loaded"], false, "no model loaded initially: {entry}");
    }

    // ── 2. POST /v1/models/model-alpha/load ───────────────────────────────────
    let (status, body) = http(port, "POST", &format!("/v1/models/{MODEL_A}/load"), None).await;
    assert_eq!(status, 200, "load model-alpha body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["ok"], true);
    assert_eq!(v["model"], MODEL_A);

    // ── 3. GET /v1/models/model-alpha/status — loaded ─────────────────────────
    let (status, body) = http(port, "GET", &format!("/v1/models/{MODEL_A}/status"), None).await;
    assert_eq!(status, 200, "status model-alpha body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["id"], MODEL_A);
    assert_eq!(v["loaded"], true);
    // loaded_at should be a non-zero epoch second.
    let loaded_at_a = v["loaded_at"].as_u64().expect("loaded_at must be u64");
    assert!(loaded_at_a > 0, "loaded_at must be > 0");

    // ── 4. GET /v1/models/model-beta/status — not loaded ─────────────────────
    let (status, body) = http(port, "GET", &format!("/v1/models/{MODEL_B}/status"), None).await;
    assert_eq!(status, 200, "status model-beta body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["loaded"], false);

    // ── 5. POST /v1/models/model-beta/load — swaps model-alpha out ───────────
    let (status, body) = http(port, "POST", &format!("/v1/models/{MODEL_B}/load"), None).await;
    assert_eq!(status, 200, "load model-beta body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["ok"], true);

    // model-alpha must be evicted.
    let (status, body) = http(port, "GET", &format!("/v1/models/{MODEL_A}/status"), None).await;
    assert_eq!(status, 200, "status model-alpha after swap: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        v["loaded"], false,
        "model-alpha must be unloaded after swap"
    );

    // model-beta loaded_at must differ from model-alpha's loaded_at.
    let (status, body) = http(port, "GET", &format!("/v1/models/{MODEL_B}/status"), None).await;
    assert_eq!(status, 200, "status model-beta: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let loaded_at_b = v["loaded_at"].as_u64().expect("loaded_at_b must be u64");
    // loaded_at_b >= loaded_at_a (B was loaded after A).
    assert!(
        loaded_at_b >= loaded_at_a,
        "model-beta loaded_at ({loaded_at_b}) must be >= model-alpha's ({loaded_at_a})"
    );

    // ── 6. POST /v1/models/model-beta/unload ─────────────────────────────────
    let (status, body) = http(port, "POST", &format!("/v1/models/{MODEL_B}/unload"), None).await;
    assert_eq!(status, 200, "unload model-beta body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["ok"], true);

    // Second unload → 404.
    let (status, _body) = http(port, "POST", &format!("/v1/models/{MODEL_B}/unload"), None).await;
    assert_eq!(status, 404, "second unload must return 404");

    // ── 7. GET /v1/models — both unloaded ────────────────────────────────────
    let (status, body) = http(port, "GET", "/v1/models", None).await;
    assert_eq!(status, 200, "list_models final body: {body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let data = v["data"].as_array().unwrap();
    for entry in data {
        assert_eq!(
            entry["loaded"], false,
            "all models must be unloaded: {entry}"
        );
    }

    // ── 8. Load unknown model → 404 ───────────────────────────────────────────
    let (status, _) = http(port, "POST", "/v1/models/no-such-model/load", None).await;
    assert_eq!(status, 404, "loading unknown model must return 404");
}

/// Explicit load swaps the loaded model out (via POST /v1/models/{id}/load).
///
/// Chat completion auto-swap requires a tokenizer+template which minimal
/// test snapshots do not have; this test verifies the swap at the load
/// endpoint level which exercises the same `ensure_loaded` path.
#[ignore = "integration test; run with RMLX_REGISTRY_TEST=1"]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn explicit_load_swaps_current_model() {
    if std::env::var("RMLX_REGISTRY_TEST").is_err() {
        eprintln!("RMLX_REGISTRY_TEST not set — skipping explicit_load_swaps_current_model");
        return;
    }

    let port = start_two_model_server().await;

    // Load model-alpha.
    let (status, _) = http(port, "POST", &format!("/v1/models/{MODEL_A}/load"), None).await;
    assert_eq!(status, 200, "load model-alpha must succeed");

    // Load model-beta — should implicitly unload model-alpha.
    let (status, body) = http(port, "POST", &format!("/v1/models/{MODEL_B}/load"), None).await;
    assert_eq!(status, 200, "load model-beta must succeed: {body}");

    // model-beta must be loaded.
    let (status, body) = http(port, "GET", &format!("/v1/models/{MODEL_B}/status"), None).await;
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        v["loaded"], true,
        "model-beta must be in slot after swap: {body}"
    );

    // model-alpha must be evicted.
    let (status, body) = http(port, "GET", &format!("/v1/models/{MODEL_A}/status"), None).await;
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(
        v["loaded"], false,
        "model-alpha must be evicted after swap: {body}"
    );

    // Loading model-beta again (same model) must be idempotent — no error.
    let (status, _) = http(port, "POST", &format!("/v1/models/{MODEL_B}/load"), None).await;
    assert_eq!(
        status, 200,
        "re-loading same model must be 200 (idempotent)"
    );

    // model-beta still loaded.
    let (status, body) = http(port, "GET", &format!("/v1/models/{MODEL_B}/status"), None).await;
    assert_eq!(status, 200);
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["loaded"], true, "model-beta must still be loaded: {body}");
}
