//! Endpoint-shape smoke tests for `POST /v1/embeddings` (jina-embeddings-v4).
//!
//! Mirrors `http_smoke.rs`: a real `TcpListener` on port 0, router in a
//! background task, raw HTTP/1.1 over `TcpStream`.
//!
//! No-GPU validation tests run always. The 200/shape tests require the jina
//! snapshot + Metal and are `#[ignore]` (single-MLX-process rule — run in
//! isolation: `cargo test --test embeddings_smoke -- --ignored
//! --test-threads=1`).

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

const JINA_ID: &str = "jinaai__jina-embeddings-v4";

fn jina_snapshot_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_JINA_V4").map(std::path::PathBuf::from)
}

fn gemma4_e4b_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

fn state(registry: ModelRegistry) -> AppState {
    let loader: ModelLoader =
        Arc::new(|_p, _i| Ok(Box::new(NotReadyGenerator) as Box<dyn Generator>));
    AppState {
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

async fn start(state: AppState) -> u16 {
    let router = rmlx_server::build_router(state);
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    port
}

async fn post(port: u16, path: &str, body: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(format!("127.0.0.1:{port}"))
        .await
        .unwrap();
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nConnection: close\r\n\
         Content-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    stream.write_all(request.as_bytes()).await.unwrap();
    let mut resp = Vec::new();
    stream.read_to_end(&mut resp).await.unwrap();
    let text = String::from_utf8_lossy(&resp).into_owned();
    let status: u16 = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let bs = text.find("\r\n\r\n").map_or(text.len(), |i| i + 4);
    (status, text[bs..].to_owned())
}

fn jina_registry() -> Option<ModelRegistry> {
    let p = jina_snapshot_dir()?;
    Some(ModelRegistry::from_paths(std::slice::from_ref(&p)))
}

// ── No-GPU validation tests (always run) ──────────────────────────────────────

/// Malformed JSON body (missing required `input`) → 422 (axum serde reject).
#[tokio::test]
async fn malformed_body_is_422() {
    let port = start(state(ModelRegistry::default())).await;
    let (status, _b) = post(port, "/v1/embeddings", r#"{"model":"x"}"#).await;
    assert_eq!(status, 422, "missing `input` must be a 422 serde rejection");
}

/// Unknown model id → 404.
#[tokio::test]
async fn unknown_model_is_404() {
    let port = start(state(ModelRegistry::default())).await;
    let (status, body) = post(
        port,
        "/v1/embeddings",
        r#"{"model":"nope","input":"hello"}"#,
    )
    .await;
    assert_eq!(status, 404, "body: {body}");
}

/// `encoding_format` outside {float,base64} → 400.
#[tokio::test]
async fn invalid_encoding_format_is_400() {
    let Some(reg) = jina_registry() else {
        eprintln!("[SKIP] invalid_encoding_format_is_400: RMLX_TEST_MODEL_JINA_V4 not set");
        return;
    };
    let port = start(state(reg)).await;
    let body = format!(r#"{{"model":"{JINA_ID}","input":"hi","encoding_format":"weird"}}"#);
    let (status, b) = post(port, "/v1/embeddings", &body).await;
    assert_eq!(status, 400, "body: {b}");
    assert!(b.contains("encoding_format"), "body: {b}");
}

/// A non-jina (causal-LM) registry entry rejected on /v1/embeddings → 400.
#[tokio::test]
async fn non_embedding_model_is_400() {
    let Some(primary_buf) = gemma4_e4b_dir() else {
        eprintln!("[SKIP] non_embedding_model_is_400: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let reg = ModelRegistry::from_paths(std::slice::from_ref(&primary_buf));
    let port = start(state(reg)).await;
    let (status, b) = post(
        port,
        "/v1/embeddings",
        r#"{"model":"mlx-community__gemma-4-e4b-it-mxfp8","input":"hi"}"#,
    )
    .await;
    assert_eq!(status, 400, "body: {b}");
    assert!(b.contains("not an embedding model"), "body: {b}");
}

// ── GPU shape tests (ignored — single-MLX-process; run in isolation) ──────────
//
// Each of these posts to `/v1/embeddings`, and the handler loads the jina
// encoder and runs the forward under `rmlx_mlx::Device::Gpu` — in THIS process,
// on a `spawn_blocking` worker, but on the far side of an axum routing table.
// No source shape in this file names the device and no call graph links the
// `post(port, "/v1/embeddings", ..)` here to `embeddings()` there, so the
// `#[ignore]` gate cannot infer the Metal context and each carries the
// `metal-unscanned` marker instead. See docs/TESTING.md.
//
// The marker deliberately does NOT put them in `scripts/run_gpu_tests.sh`:
// every one is gated on `RMLX_TEST_MODEL_JINA_V4` and returns early without it,
// so on a machine with no jina snapshot the runner would execute five no-ops,
// see no Metal validation banner for rmlx-server, and fail the suite over a
// missing model. They run by hand:
//
//   RMLX_TEST_MODEL_JINA_V4=/abs/path/to/jinaai__jina-embeddings-v4 \
//     cargo test -p rmlx-server --test embeddings_smoke -- --ignored --test-threads=1

/// Valid single-vector request → 200 + OpenAI embeddings shape.
// gpu-test-gate: metal-unscanned  Metal is entered inside the handler.
#[tokio::test]
#[ignore = "GPU Metal: cargo test --test embeddings_smoke valid_single_vector -- --ignored --test-threads=1"]
async fn valid_single_vector_200_shape() {
    let Some(reg) = jina_registry() else {
        eprintln!("[SKIP] valid_single_vector_200_shape: RMLX_TEST_MODEL_JINA_V4 not set");
        return;
    };
    let port = start(state(reg)).await;
    let body = format!(r#"{{"model":"{JINA_ID}","input":"hello world"}}"#);
    let (status, b) = post(port, "/v1/embeddings", &body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["object"], "embedding");
    assert_eq!(v["data"][0]["index"], 0);
    let emb = v["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(emb.len(), 2048, "full single-vector dim == 2048");
    assert!(emb[0].is_f64(), "single-vector elements are floats");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap() > 0);
}

/// `return_multivector:true` toggles the embedding to `[[f32;128];seq]`.
// gpu-test-gate: metal-unscanned  Metal is entered inside the handler.
#[tokio::test]
#[ignore = "GPU Metal: cargo test --test embeddings_smoke return_multivector -- --ignored --test-threads=1"]
async fn return_multivector_toggles_shape() {
    let Some(reg) = jina_registry() else {
        eprintln!("[SKIP] return_multivector_toggles_shape: RMLX_TEST_MODEL_JINA_V4 not set");
        return;
    };
    let port = start(state(reg)).await;
    let body =
        format!(r#"{{"model":"{JINA_ID}","input":"hello world","return_multivector":true}}"#);
    let (status, b) = post(port, "/v1/embeddings", &body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    let mv = v["data"][0]["embedding"].as_array().unwrap();
    assert!(!mv.is_empty(), "multi-vector has >=1 token row");
    let row0 = mv[0].as_array().unwrap();
    assert_eq!(row0.len(), 128, "each token row width == 128");
}

/// Invalid matryoshka `dimensions` (not in {128,256,512,1024,2048}) → 400.
///
/// Unlike the other 400s in this file, this one is NOT a request-validation
/// rejection: the handler defers `dimensions` to the model's matryoshka set, so
/// the check runs in `pooling::single_vector` — after the encoder is loaded and
/// after a full `Device::Gpu` forward. The 400 is the tail of a GPU round trip,
/// which is why it needs the snapshot and the Metal context.
// gpu-test-gate: metal-unscanned  Metal is entered inside the handler.
#[tokio::test]
#[ignore = "GPU Metal: cargo test --test embeddings_smoke invalid_dimensions -- --ignored --test-threads=1"]
async fn invalid_dimensions_is_400() {
    let Some(reg) = jina_registry() else {
        eprintln!("[SKIP] invalid_dimensions_is_400: RMLX_TEST_MODEL_JINA_V4 not set");
        return;
    };
    let port = start(state(reg)).await;
    let body = format!(r#"{{"model":"{JINA_ID}","input":"hi","dimensions":384}}"#);
    let (status, b) = post(port, "/v1/embeddings", &body).await;
    assert_eq!(status, 400, "body: {b}");
    assert!(b.contains("invalid truncate_dim"), "body: {b}");
}

/// A deterministic 64x48 synthetic-gradient PNG, base64 — exactly the image
/// the parity probe and `scripts/parity/jina_v4_parity.py` synthesize, so the
/// endpoint exercises the same `grid_thw=[1,4,4]` path the parity gate covers.
const TEST_IMG_B64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwCAIAAAAuKetIAAAA5UlEQVR4nO1ZWwrDMAyTg+5/px5lR9hnGMODjo527BXWLvYqMMEIYwiJLDsxACdYXiMGAxwoSVeOGyjAGbDRkvmcTiDryukErtuqk3PbZXScSw7k8ykOeCgO2MyOj8C7u9g/ns91IAdOccBDcaC+rbvRcEoHinohX5EDn9ihJfhFXV8nP9vngVjxlA54KA7U7nUdjbh0ANIBaB5ApP4ejbh0ANIBSAfQu5bjC//f3kZtAxu2TG4/+B/YNj81D3goDtTlDUuAU+9CRTrgK3JgbvofgOYB7K8Xqt3rOqQDJU6vjz1w4AIqTWNtn1ky5AAAAABJRU5ErkJggg==";

/// Image input (single-vector): `{"input":{"image":"data:...;base64,..."}}`
/// → 200 + 2048-d float vector. End-to-end exercise of the M-RoPE + merge +
/// image-span pooling path (GPU; single-MLX-process).
// gpu-test-gate: metal-unscanned  Metal is entered inside the handler.
#[tokio::test]
#[ignore = "GPU Metal: cargo test --test embeddings_smoke image_single_vector -- --ignored --test-threads=1"]
async fn image_single_vector_200_shape() {
    let Some(reg) = jina_registry() else {
        eprintln!("[SKIP] image_single_vector_200_shape: RMLX_TEST_MODEL_JINA_V4 not set");
        return;
    };
    let port = start(state(reg)).await;
    let body = format!(
        r#"{{"model":"{JINA_ID}","input":{{"image":"data:image/png;base64,{TEST_IMG_B64}"}}}}"#
    );
    let (status, b) = post(port, "/v1/embeddings", &body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["object"], "embedding");
    let emb = v["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(emb.len(), 2048, "image single-vector dim == 2048");
    assert!(emb[0].is_f64(), "elements are floats");
}

/// Image input with `return_multivector:true` → `[[f32;128];seq]` (one row
/// per token of the expanded image sequence).
// gpu-test-gate: metal-unscanned  Metal is entered inside the handler.
#[tokio::test]
#[ignore = "GPU Metal: cargo test --test embeddings_smoke image_multivector -- --ignored --test-threads=1"]
async fn image_multivector_toggles_shape() {
    let Some(reg) = jina_registry() else {
        eprintln!("[SKIP] image_multivector_toggles_shape: RMLX_TEST_MODEL_JINA_V4 not set");
        return;
    };
    let port = start(state(reg)).await;
    let body = format!(
        r#"{{"model":"{JINA_ID}","input":[{{"image":"data:image/png;base64,{TEST_IMG_B64}"}}],"return_multivector":true}}"#
    );
    let (status, b) = post(port, "/v1/embeddings", &body).await;
    assert_eq!(status, 200, "body: {b}");
    let v: serde_json::Value = serde_json::from_str(&b).unwrap();
    let mv = v["data"][0]["embedding"].as_array().unwrap();
    assert!(!mv.is_empty(), "image multi-vector has >=1 token row");
    assert_eq!(
        mv[0].as_array().unwrap().len(),
        128,
        "each token row width == 128"
    );
}
