//! Shared harness for the thread-boundary tests: model state built on one
//! thread and first used on another.
//!
//! # The defect class
//!
//! MLX binds a lazy op to the default stream of the thread that built it. From
//! MLX 0.32 the CPU command-encoder map is thread-local, so the first
//! evaluation of that op on another thread fails with "There is no Stream(cpu,
//! N) in current thread.". The GPU map is thread-local on every supported MLX,
//! so a lazy GPU op fails the same way with "There is no Stream(gpu, N)".
//!
//! `rmlx serve` builds model state on one thread and uses it on another:
//!
//! - A chat model loads on the eager-preload thread, on an async worker (the
//!   on-demand load and the load route), or inside a request. It generates on
//!   a blocking-pool thread. Its vision tower, audio tower and speculative
//!   drafter load with it.
//! - The embedding, Whisper and TTS models load inside a request and serve
//!   later requests on the blocking-pool thread that tokio picks. A request
//!   that fails after the load leaves a model that no forward has evaluated.
//! - A prompt-cache or encoder-cache entry is written by one request and read
//!   by a later one. A request that fails after it wrote an entry leaves an
//!   entry that no forward has evaluated.
//!
//! # The seam
//!
//! The object that crosses the thread boundary in production: the model
//! `arch::load_model` returns, the `ArchGenerator` or `SpeculativeGenerator`
//! the server builds, the jina-v4, Whisper or TTS model the server caches
//! between requests, and a prompt-cache or encoder-cache entry. Every array
//! such an object holds must be evaluated before the object crosses.
//!
//! # The observable and what cannot move
//!
//! Each test runs one production hand-over twice with the same binary and the
//! same inputs. The oracle runs every step on one thread. The second arm runs
//! the steps after the hand-over on a second thread while the first thread
//! stays alive and idle, as a blocking-pool worker does. The second arm must
//! succeed and return exactly what the oracle returns: the greedy token ids,
//! the embedding bits, the transcript or the audio samples. A cache test also
//! asserts that the second request hit the entry the first one wrote, and a
//! drafter test that the second thread ran draft rounds.
//!
//! The oracle cannot see a change that moves the one-thread output: both arms
//! run the changed binary. The golden-token suites hold the one-thread tokens.
//!
//! # Mutations and what catches them
//!
//! Measured on mlx 0.32.1 / mlx-c 0.6.0_4 unless stated:
//!
//! - The text path of the loader left lazy (the current tree): the Qwen3 and
//!   Qwen3.5 text cells are red. A fix of the text path alone (evaluate the
//!   bf16 casts, the fused QKV and the YARN scale): the Qwen3 text cell turns
//!   green, the tower cells and the jina image cell stay red.
//! - A tower loader left lazy (the current tree): that tower's cell is red.
//! - Every safetensors tensor made a lazy CPU op at load: every drafter cell,
//!   the Qwen2, BitNet and unified-audio cells turn red. The per-request cells
//!   of the in-request loaders stay green, which is why each in-request loader
//!   also has a load-only cell.
//! - A lazy op in the Whisper loader: the load-only Whisper cell turns red, the
//!   per-request Whisper cell stays green.
//! - `TtsModel::load` leaves lazy GPU ops (the current tree): the load-only TTS
//!   cell is red on every MLX pair.
//! - The encoder cache stores a node the writing request never evaluates: the
//!   encoder-cache hit cell turns red. Publishing without an evaluation (the
//!   current tree): the failed-request gemma4 cell and the `multimodal_cache`
//!   publish tests are red on every MLX pair.
//! - The Qwen3-VL tower returns unevaluated embeds: nothing turns red, because
//!   the image decode evaluates them before its context check. Remove that
//!   evaluation too: the failed-request Qwen3-VL cell turns red.
//! - A prompt-cache store without its evaluation: nothing turns red at the
//!   default codec, because the decode evaluates the stored arrays. Make the
//!   stored copies lazy as well: the prompt-cache cell of that architecture
//!   turns red.
//! - The drafter bypassed (plain verifier decode): the drafter cell turns red
//!   on its round count, not on its tokens.
//!
//! Not caught by any cell: a lazy op on a load path no local snapshot reaches,
//! such as the zero router bias a Laguna checkpoint without
//! `e_score_correction_bias` gets.

#![allow(dead_code, unreachable_pub)]

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::Duration;

use futures::StreamExt;
use parking_lot::Mutex;
use rmlx_mlx::Device;
use rmlx_models::multimodal_cache::MultimodalCache;
use rmlx_models::{arch, Pcg32, PenaltyConfig, SamplerConfig};
use rmlx_server::engine::SamplingParams;
use rmlx_server::{
    ArchGenerator, GenerationRequest, Generator, ModelLoadConfig, SpeculativeGenerator,
};

#[path = "../../../rmlx-models/tests/common/round_stream.rs"]
pub mod round_stream;

use round_stream::RoundStreamRecorder;

/// A 64x64 solid-red PNG.
pub const RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElFTkSuQmCC";

pub const SPOKEN_TEXT: &str = "The capital of France is Paris.";

pub const N_TOKENS: u32 = 24;

// ── Snapshots ───────────────────────────────────────────────────────────────

/// The snapshot `slug` names under `RMLX_O_MODELS_ROOT`, or `None` after a
/// named `SKIP`.
///
/// `declares` is what the snapshot's `config.json` must name:
/// `architectures[0]`, or `model_type` when it names no architecture. An empty
/// `declares` is a directory without `config.json`, such as a tokenizer.
///
/// - Root unset, or the root does not hold the slug: `SKIP`.
/// - A snapshot with no weight file is a half-written download: `SKIP`.
/// - Root set but not a directory, or a snapshot that declares something else:
///   a failure, because configuration is present and wrong.
pub fn snapshot(test: &str, slug: &str, declares: &str) -> Option<PathBuf> {
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT").filter(|r| !r.is_empty()) else {
        println!("SKIP {test}: RMLX_O_MODELS_ROOT is not set");
        return None;
    };
    let root = PathBuf::from(root);
    assert!(
        root.is_dir(),
        "{test}: RMLX_O_MODELS_ROOT={} is set but is not a directory",
        root.display()
    );
    let dir = root.join(slug);
    if !dir.is_dir() {
        println!("SKIP {test}: RMLX_O_MODELS_ROOT does not hold {slug}");
        return None;
    }
    if declares.is_empty() {
        return Some(dir);
    }
    let has_weights = std::fs::read_dir(&dir)
        .expect("read snapshot dir")
        .filter_map(Result::ok)
        .any(|e| {
            let name = e.file_name();
            let name = name.to_string_lossy();
            name.ends_with(".safetensors") || name == "weights.npz"
        });
    if !has_weights {
        println!("SKIP {test}: {slug} has no weight file (a half-written download)");
        return None;
    }
    let config: serde_json::Value = serde_json::from_slice(
        &std::fs::read(dir.join("config.json")).expect("snapshot config.json"),
    )
    .expect("snapshot config.json is JSON");
    let found = config
        .pointer("/architectures/0")
        .or_else(|| config.get("model_type"))
        .and_then(serde_json::Value::as_str)
        .unwrap_or_default();
    assert_eq!(
        found, declares,
        "{test}: RMLX_O_MODELS_ROOT holds {slug}, but it declares {found:?}, not {declares:?}"
    );
    Some(dir)
}

// ── Inputs ──────────────────────────────────────────────────────────────────

/// The smoke prompt rendered through the snapshot's chat template.
pub fn chat_prompt(dir: &Path) -> Vec<u32> {
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    rmlx_server::chat_template::smoke_prompt_ids(dir, &tokenizer)
        .expect("the snapshot renders the smoke prompt through its chat template")
}

/// A prompt of several 256-token prompt-cache blocks.
pub fn long_prompt(dir: &Path) -> Vec<u32> {
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let text = [SPOKEN_TEXT; 100].join(" ");
    let ids = tokenizer
        .encode(text, true)
        .expect("encode")
        .get_ids()
        .to_vec();
    assert!(ids.len() >= 512, "the prompt must span two cache blocks");
    ids
}

/// `SPOKEN_TEXT` spoken by the system voice, as a 16 kHz mono 16-bit WAV.
pub fn spoken_wav() -> Vec<u8> {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("spoken.wav");
    let status = std::process::Command::new("say")
        .arg("-o")
        .arg(&path)
        .args(["--data-format=LEI16@16000", SPOKEN_TEXT])
        .status()
        .expect("run say");
    assert!(status.success(), "say failed: {status}");
    std::fs::read(&path).expect("read spoken.wav")
}

/// Standard base64. The server's own encoders are private to the crate, and
/// an integration test reaches only its public API.
pub fn base64(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let bits =
            chunk.iter().fold(0u32, |acc, &b| acc << 8 | u32::from(b)) << (8 * (3 - chunk.len()));
        for i in 0..=chunk.len() {
            out.push(char::from(ALPHABET[(bits >> (18 - 6 * i) & 63) as usize]));
        }
        for _ in chunk.len()..3 {
            out.push('=');
        }
    }
    out
}

pub fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ── Where the thread boundary falls ─────────────────────────────────────────

/// Where the thread boundary falls between the steps of one hand-over.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Every step on one thread: the oracle.
    None,
    /// The load on one thread, every later step on a second.
    AfterLoad,
    /// The load and every request but the last on one thread, the last
    /// request on a second.
    BeforeLastRequest,
}

/// Run `first`, then `second` on the state `first` returns.
///
/// Unless `split` is [`Split::None`], `second` runs on a new thread while the
/// thread that ran `first` stays alive and idle. `first` holds the load for
/// [`Split::AfterLoad`], and the load and the earlier requests for
/// [`Split::BeforeLastRequest`].
pub fn hand_over<S, A, B>(
    split: Split,
    first: impl FnOnce() -> (S, A) + Send + 'static,
    second: impl FnOnce(S) -> B + Send + 'static,
) -> (A, B)
where
    S: Send + 'static,
    A: Send + 'static,
    B: Send + 'static,
{
    if split == Split::None {
        return std::thread::spawn(move || {
            let (state, a) = first();
            (a, second(state))
        })
        .join()
        .expect("thread panicked");
    }
    let (state_tx, state_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let first_thread = std::thread::spawn(move || {
        state_tx.send(first()).expect("send state");
        release_rx.recv().ok();
    });
    let (state, a) = state_rx.recv().expect("state from the first thread");
    let b = std::thread::spawn(move || second(state))
        .join()
        .expect("second thread panicked");
    release_tx.send(()).ok();
    first_thread.join().expect("first thread panicked");
    (a, b)
}

// ── Text through `load_model` and `generate_greedy` ────────────────────────

pub fn load_model(dir: &Path, device: Device) -> arch::Architecture {
    arch::load_model(dir, device, &arch::LoadOpts::default()).expect("load_model")
}

pub fn greedy(
    model: &arch::Architecture,
    dir: &Path,
    prompt: &[u32],
    slots: usize,
    device: Device,
) -> rmlx_core::error::Result<Vec<u32>> {
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let mut token_history = Vec::new();
    let steps = model.generate_greedy(
        &tokenizer,
        prompt,
        N_TOKENS as usize,
        device,
        None,
        None,
        slots,
        &[],
        &mut |_| None,
        None,
        &sampler_cfg,
        &mut rng,
        &PenaltyConfig::default(),
        &mut token_history,
    )?;
    Ok(steps.iter().map(|s| s.token_id).collect())
}

// ── Through the server's generators ─────────────────────────────────────────

/// A runtime whose blocking pool holds one thread, so every blocking task, the
/// generator's own decode closure included, runs on that thread.
pub fn one_thread_pool() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(1)
        .thread_keep_alive(Duration::from_secs(3600))
        .enable_all()
        .build()
        .expect("tokio runtime")
}

/// Make `recorder` the tracing dispatcher of `pool`'s one blocking thread for
/// the rest of that thread's life, so it sees every event the decode emits
/// there and nothing from any other thread.
fn record_on(pool: &tokio::runtime::Runtime, recorder: &Arc<RoundStreamRecorder>) {
    let recorder = Arc::clone(recorder);
    pool.block_on(pool.spawn_blocking(move || {
        std::mem::forget(tracing::subscriber::set_default(recorder));
    }))
    .expect("install the recorder");
}

/// A thread that built a generator and now waits, alive, until dropped.
struct Parked {
    release: Option<mpsc::Sender<()>>,
    thread: Option<JoinHandle<()>>,
}

impl Drop for Parked {
    fn drop(&mut self) {
        drop(self.release.take());
        if let Some(t) = self.thread.take() {
            t.join().ok();
        }
    }
}

pub type Load = Arc<dyn Fn() -> Arc<dyn Generator> + Send + Sync>;

/// Build a generator with `load` and serve `requests` in order.
///
/// A `recorder` sees every event of the thread that serves the last request.
pub fn serve(
    split: Split,
    load: &Load,
    requests: Vec<GenerationRequest>,
    recorder: Option<&Arc<RoundStreamRecorder>>,
) -> Vec<Result<Vec<u32>, String>> {
    let pool = one_thread_pool();
    let (generator, parked) = if split == Split::AfterLoad {
        let (gen_tx, gen_rx) = mpsc::channel();
        let (release, release_rx) = mpsc::channel::<()>();
        let load = Arc::clone(load);
        let thread = std::thread::spawn(move || {
            gen_tx.send(load()).ok();
            release_rx.recv().ok();
        });
        let generator = gen_rx.recv().expect("generator from the load thread");
        let parked = Parked {
            release: Some(release),
            thread: Some(thread),
        };
        (generator, Some(parked))
    } else {
        let load = Arc::clone(load);
        let generator = pool
            .block_on(pool.spawn_blocking(move || load()))
            .expect("load task panicked");
        (generator, None)
    };
    let second_pool = (split == Split::BeforeLastRequest).then(one_thread_pool);
    if let Some(recorder) = recorder {
        record_on(second_pool.as_ref().unwrap_or(&pool), recorder);
    }
    let last = requests.len() - 1;
    let out = requests
        .into_iter()
        .enumerate()
        .map(|(i, req)| {
            let rt = match &second_pool {
                Some(second) if i == last => second,
                _ => &pool,
            };
            rt.block_on(async {
                let mut stream = generator.generate(req);
                let mut ids = Vec::new();
                while let Some(item) = stream.next().await {
                    ids.push(item.map_err(|e| e.to_string())?.token_id);
                }
                Ok::<_, String>(ids)
            })
        })
        .collect();
    drop(generator);
    drop(parked);
    out
}

/// Serve `requests` on one thread, then across `split`; the second arm must
/// return the oracle's tokens for every request.
pub fn assert_hand_over(
    test: &str,
    split: Split,
    load: &Load,
    requests: impl Fn() -> Vec<GenerationRequest>,
) {
    let oracle = serve(Split::None, load, requests(), None);
    assert_oracle(test, &oracle);
    let crossed = serve(split, load, requests(), None);
    assert_crossed(test, &crossed, &oracle);
}

pub fn assert_oracle(test: &str, oracle: &[Result<Vec<u32>, String>]) {
    for (i, r) in oracle.iter().enumerate() {
        assert!(
            r.as_ref().is_ok_and(|ids| !ids.is_empty()),
            "{test}: the oracle is every step on one thread and must decode; request {i}: {r:?}"
        );
    }
}

pub fn assert_crossed(
    test: &str,
    crossed: &[Result<Vec<u32>, String>],
    oracle: &[Result<Vec<u32>, String>],
) {
    for (i, (c, o)) in crossed.iter().zip(oracle).enumerate() {
        let c = c.as_ref().unwrap_or_else(|e| {
            panic!("{test}: request {i} failed on a thread that did not build its state: {e}")
        });
        assert_eq!(
            Some(c),
            o.as_ref().ok(),
            "{test}: request {i} emitted different tokens than the one-thread oracle"
        );
    }
}

pub fn load_config(device: Device, mm_cache: Option<Arc<MultimodalCache>>) -> ModelLoadConfig {
    ModelLoadConfig {
        device,
        kv_quant: None,
        max_ctx: None,
        prompt_cache_slots: 0,
        mm_cache,
        calibration: None,
        yarn: None,
        image_max_tokens: None,
    }
}

pub fn arch_generator(
    dir: PathBuf,
    device: Device,
    mm_cache: Option<Arc<MultimodalCache>>,
) -> Load {
    Arc::new(move || {
        let generator: Arc<dyn Generator> = Arc::new(
            ArchGenerator::from_snapshot(
                &dir,
                &load_config(device, mm_cache.clone()),
                Arc::new(Mutex::new(())),
            )
            .expect("ArchGenerator::from_snapshot"),
        );
        generator
    })
}

pub fn speculative_generator(verifier: PathBuf, drafter: PathBuf, device: Device) -> Load {
    Arc::new(move || {
        let generator: Arc<dyn Generator> = Arc::new(
            SpeculativeGenerator::from_snapshots_with_id(
                &verifier,
                &drafter,
                None,
                &load_config(device, None),
                Arc::new(Mutex::new(())),
                None,
                None,
            )
            .expect("SpeculativeGenerator::from_snapshots_with_id"),
        );
        generator
    })
}

/// A greedy request for the chat prompt of the generator loaded from `dir`.
pub fn request(dir: &Path, images: Vec<String>, audio_b64: Vec<String>) -> GenerationRequest {
    GenerationRequest {
        model_id: dir
            .file_name()
            .and_then(|n| n.to_str())
            .expect("snapshot dir name")
            .to_owned(),
        prompt_tokens: chat_prompt(dir),
        max_tokens: N_TOKENS,
        sampling: SamplingParams {
            temperature: 0.0,
            seed: Some(0),
            ..SamplingParams::default()
        },
        stop: Vec::new(),
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
        images,
        audio_b64,
        image_max_tokens: None,
    }
}

pub fn image_request(dir: &Path) -> GenerationRequest {
    let png = format!("data:image/png;base64,{RED_PNG_BASE64}");
    request(dir, vec![png], Vec::new())
}

pub fn audio_request(dir: &Path, wav: &[u8]) -> GenerationRequest {
    request(dir, Vec::new(), vec![base64(wav)])
}

pub fn text_request(dir: &Path) -> GenerationRequest {
    request(dir, Vec::new(), Vec::new())
}
