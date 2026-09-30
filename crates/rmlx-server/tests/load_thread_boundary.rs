//! Model state built on one thread and first used on another.
//!
//! `rmlx serve` builds model state on one thread and uses it on another. A
//! chat model loads on the eager-preload thread or, on demand, on an async
//! worker, and generates on a blocking-pool thread. Its vision tower, audio
//! tower and speculative drafter load with it. The embedding, Whisper and TTS
//! models load inside their first request and serve later requests on the
//! blocking-pool thread that tokio picks. A prompt-cache or encoder-cache entry
//! is written by one request and read by the next one.
//!
//! MLX binds a lazy op to the default stream of the thread that built it. From
//! MLX 0.32 the CPU command-encoder map is thread-local, so the first
//! evaluation of that op on another thread fails with "There is no Stream(cpu,
//! N) in current thread.". State that crosses a thread boundary must be
//! evaluated before it crosses.
//!
//! Each test runs one production hand-over twice. The oracle runs every step
//! on one thread. The second arm runs the steps after the hand-over on a second
//! thread while the first thread stays alive. The second arm must succeed and
//! return exactly what the oracle returns: same binary, same inputs, greedy
//! decode.
//!
//! Run: `make gpu-test CRATE=rmlx-server`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::cast_possible_truncation
)]

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

const BONSAI: &str = "prism-ml__Ternary-Bonsai-8B-mlx-2bit";
const GEMMA4_E2B: &str = "mlx-community__gemma-4-e2b-it-mxfp8";
const GEMMA4_E2B_ASSISTANT: &str = "mlx-community__gemma-4-E2B-it-assistant-bf16";
const GEMMA4_UNIFIED_12B: &str = "mlx-community__gemma-4-12B-it-mxfp8";
const MEDGEMMA: &str = "mlx-community__medgemma-1.5-4b-it-8bit";
const QWEN3_VL: &str = "mlx-community__Qwen3-VL-30B-A3B-Instruct-4bit";
const QWEN36_MOE: &str = "mlx-community__Qwen3.6-35B-A3B-8bit";
const QWEN36_DFLASH: &str = "z-lab__Qwen3.6-35B-A3B-DFlash";
const QWEN36_EAGLE3: &str = "Dogacel__specdrift-qwen3.6-35b-a3b-eagle3";
const QWEN38_4BIT: &str = "mlx-community__Qwen3.8-27B-4bit";
const QWEN38_MTP_4BIT: &str = "mlx-community__Qwen3.8-27B-MTP-4bit";
const QWEN38_DFLASH2: &str = "z-lab__Qwen3.8-27B-DFlash2";
const QWEN38_MXFP8: &str = "mlx-community__Qwen3.8-27B-mxfp8";
const ORNITH_9B: &str = "sahilchachra__ornith-1.0-9b-mxfp8-mlx";
const JINA_V4: &str = "jinaai__jina-embeddings-v4";
const WHISPER: &str = "mlx-community__whisper-large-v3-mlx";
const WHISPER_TOKENIZER: &str = "openai__whisper-large-v3-tokenizer";
const TTS: &str = "mlx-community__Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit";
const TTS_CODEC: &str = "Qwen__Qwen3-TTS-Tokenizer-12Hz";

/// A 64x64 solid-red PNG.
const RED_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAIAAAAlC+aJAAAAb0lEQVR4nO3PAQkAAAyEwO9feoshgnABdLep8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3I8QUNyPEFDcjxBQ3IPanc8OLDQitxAAAAAElFTkSuQmCC";

const SPOKEN_TEXT: &str = "The capital of France is Paris.";

const N_TOKENS: u32 = 24;

// ── Snapshots and inputs ────────────────────────────────────────────────────

/// Every snapshot `slugs` names under `RMLX_O_MODELS_ROOT`, or `None` after
/// printing why this cell stood down.
fn snapshots<const N: usize>(test: &str, slugs: [&str; N]) -> Option<[PathBuf; N]> {
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT").filter(|r| !r.is_empty()) else {
        println!("SKIP {test}: RMLX_O_MODELS_ROOT is not set");
        return None;
    };
    let dirs = slugs.map(|slug| Path::new(&root).join(slug));
    if let Some(missing) = dirs.iter().find(|d| !d.is_dir()) {
        println!(
            "SKIP {test}: RMLX_O_MODELS_ROOT does not hold {}",
            missing.display()
        );
        return None;
    }
    Some(dirs)
}

/// The smoke prompt rendered through the snapshot's chat template.
fn chat_prompt(dir: &Path) -> Vec<u32> {
    let tokenizer =
        tokenizers::Tokenizer::from_file(dir.join("tokenizer.json")).expect("tokenizer.json");
    rmlx_server::chat_template::smoke_prompt_ids(dir, &tokenizer)
        .expect("the snapshot renders the smoke prompt through its chat template")
}

/// A prompt of several 256-token prompt-cache blocks.
fn long_prompt(dir: &Path) -> Vec<u32> {
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
fn spoken_wav() -> Vec<u8> {
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

fn base64(bytes: &[u8]) -> String {
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

fn f32_bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

// ── Hand-over on plain threads ──────────────────────────────────────────────

/// Run `first`, then `second` on the state `first` returns.
///
/// With `split`, `second` runs on a new thread while the thread that ran
/// `first` stays alive and idle, as a blocking-pool worker does between
/// requests. Without it, both run on one thread.
fn hand_over<S, A, B>(
    split: bool,
    first: impl FnOnce() -> (S, A) + Send + 'static,
    second: impl FnOnce(S) -> B + Send + 'static,
) -> (A, B)
where
    S: Send + 'static,
    A: Send + 'static,
    B: Send + 'static,
{
    if !split {
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

// ── Hand-over through the server's generators ──────────────────────────────

/// A runtime whose blocking pool holds one thread, so every blocking task, the
/// generator's own decode closure included, runs on that thread.
fn one_thread_pool() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .max_blocking_threads(1)
        .thread_keep_alive(Duration::from_secs(3600))
        .enable_all()
        .build()
        .expect("tokio runtime")
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

/// Where the thread boundary falls in [`serve`].
#[derive(Clone, Copy)]
enum Split {
    /// Load and every request on the pool's one thread (the oracle).
    None,
    /// Load on another thread, requests on the pool's thread.
    AfterLoad,
    /// Load and every request but the last on one pool, the last on a second.
    BeforeLastRequest,
}

type Load = Arc<dyn Fn() -> Arc<dyn Generator> + Send + Sync>;

/// Build a generator with `load` and serve `requests` in order.
fn serve(
    split: Split,
    load: &Load,
    requests: Vec<GenerationRequest>,
) -> Vec<Result<Vec<u32>, String>> {
    let pool = one_thread_pool();
    let (generator, parked) = if matches!(split, Split::AfterLoad) {
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
    let second_pool = matches!(split, Split::BeforeLastRequest).then(one_thread_pool);
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
fn assert_hand_over(
    test: &str,
    split: Split,
    load: &Load,
    requests: impl Fn() -> Vec<GenerationRequest>,
) {
    let oracle = serve(Split::None, load, requests());
    for (i, r) in oracle.iter().enumerate() {
        assert!(
            r.as_ref().is_ok_and(|ids| !ids.is_empty()),
            "{test}: the oracle is every step on one thread and must decode; request {i}: {r:?}"
        );
    }
    let crossed = serve(split, load, requests());
    for (i, (c, o)) in crossed.iter().zip(&oracle).enumerate() {
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

fn load_config(mm_cache: Option<Arc<MultimodalCache>>) -> ModelLoadConfig {
    ModelLoadConfig {
        device: Device::Gpu,
        kv_quant: None,
        max_ctx: None,
        prompt_cache_slots: 0,
        mm_cache,
        calibration: None,
        yarn: None,
        image_max_tokens: None,
    }
}

fn arch_generator(dir: PathBuf, mm_cache: Option<Arc<MultimodalCache>>) -> Load {
    Arc::new(move || {
        let generator: Arc<dyn Generator> = Arc::new(
            ArchGenerator::from_snapshot(
                &dir,
                &load_config(mm_cache.clone()),
                Arc::new(Mutex::new(())),
            )
            .expect("ArchGenerator::from_snapshot"),
        );
        generator
    })
}

fn speculative_generator(verifier: PathBuf, drafter: PathBuf) -> Load {
    Arc::new(move || {
        let generator: Arc<dyn Generator> = Arc::new(
            SpeculativeGenerator::from_snapshots_with_id(
                &verifier,
                &drafter,
                None,
                &load_config(None),
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
fn request(dir: &Path, images: Vec<String>, audio_b64: Vec<String>) -> GenerationRequest {
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

fn image_request(dir: &Path) -> GenerationRequest {
    let png = format!("data:image/png;base64,{RED_PNG_BASE64}");
    request(dir, vec![png], Vec::new())
}

fn audio_request(dir: &Path, wav: &[u8]) -> GenerationRequest {
    request(dir, Vec::new(), vec![base64(wav)])
}

fn text_request(dir: &Path) -> GenerationRequest {
    request(dir, Vec::new(), Vec::new())
}

// ── Text: `load_model` and `generate_greedy` ────────────────────────────────

fn greedy(
    model: &arch::Architecture,
    dir: &Path,
    prompt: &[u32],
    slots: usize,
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
        Device::Gpu,
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

fn load_model(dir: &Path) -> arch::Architecture {
    arch::load_model(dir, Device::Gpu, &arch::LoadOpts::default()).expect("load_model")
}

/// Bonsai ships F16 scales, biases and norms, so its loader leaves lazy CPU
/// `astype` ops (and the fused QKV `concatenate`) in the model it returns.
#[test]
#[ignore = "requires Metal GPU and the Bonsai snapshot; run with `make gpu-test`"]
fn text_decodes_on_a_thread_that_did_not_load_the_model() {
    let test = "text_decodes_on_a_thread_that_did_not_load_the_model";
    let Some([dir]) = snapshots(test, [BONSAI]) else {
        return;
    };
    let run = |split: bool| {
        let (d1, d2) = (dir.clone(), dir.clone());
        hand_over(
            split,
            move || (load_model(&d1), ()),
            move |m| greedy(&m, &d2, &chat_prompt(&d2), 0),
        )
        .1
    };
    let oracle = run(false).expect("the one-thread oracle must decode");
    let crossed = run(true).unwrap_or_else(|e| {
        panic!("{test}: decode failed on a thread that did not load the model: {e}")
    });
    assert_eq!(
        crossed, oracle,
        "{test}: different tokens than the one-thread oracle"
    );
}

/// A prompt-cache entry written by one request and hit by the next request on
/// another thread.
#[test]
#[ignore = "requires Metal GPU and the Bonsai snapshot; run with `make gpu-test`"]
fn a_prompt_cache_hit_decodes_on_another_thread() {
    let test = "a_prompt_cache_hit_decodes_on_another_thread";
    let Some([dir]) = snapshots(test, [BONSAI]) else {
        return;
    };
    let run = |split: bool| {
        let (d1, d2) = (dir.clone(), dir.clone());
        let first = move || {
            let model = load_model(&d1);
            model.clear_prompt_cache();
            let ids = greedy(&model, &d1, &long_prompt(&d1), 1).expect("first request");
            let hits = model.cache_stats().map_or(0, |s| s.hits);
            (model, (ids, hits))
        };
        let second = move |model: arch::Architecture| {
            let ids = greedy(&model, &d2, &long_prompt(&d2), 1);
            (ids, model.cache_stats().map_or(0, |s| s.hits))
        };
        hand_over(split, first, second)
    };
    let ((oracle_first, _), (oracle_second, oracle_hits)) = run(false);
    let oracle_second = oracle_second.expect("the one-thread oracle must decode");
    let ((crossed_first, first_hits), (crossed_second, hits)) = run(true);
    assert_eq!(
        first_hits, 0,
        "{test}: the first request must miss the cleared cache"
    );
    let crossed_second = crossed_second.unwrap_or_else(|e| {
        panic!("{test}: a prompt-cache hit failed on a thread that did not write it: {e}")
    });
    assert_eq!(
        hits, oracle_hits,
        "{test}: the second request must hit as the oracle's did"
    );
    assert!(
        hits >= 1,
        "{test}: the second request did not hit the prompt cache"
    );
    assert_eq!(
        crossed_first, oracle_first,
        "{test}: first request differs from the oracle"
    );
    assert_eq!(
        crossed_second, oracle_second,
        "{test}: second request differs from the oracle"
    );
}

// ── Vision and audio towers: `ArchGenerator` ────────────────────────────────

fn assert_tower(test: &str, slug: &str, make: fn(&Path) -> GenerationRequest) {
    let Some([dir]) = snapshots(test, [slug]) else {
        return;
    };
    let load = arch_generator(dir.clone(), None);
    assert_hand_over(test, Split::AfterLoad, &load, || vec![make(&dir)]);
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn gemma4_vision_tower_runs_on_a_thread_that_did_not_load_it() {
    assert_tower(
        "gemma4_vision_tower_runs_on_a_thread_that_did_not_load_it",
        GEMMA4_E2B,
        image_request,
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-12B snapshot; run with `make gpu-test`"]
fn gemma4_unified_vision_embedder_runs_on_a_thread_that_did_not_load_it() {
    assert_tower(
        "gemma4_unified_vision_embedder_runs_on_a_thread_that_did_not_load_it",
        GEMMA4_UNIFIED_12B,
        image_request,
    );
}

#[test]
#[ignore = "requires Metal GPU and the medgemma snapshot; run with `make gpu-test`"]
fn gemma3_vision_tower_runs_on_a_thread_that_did_not_load_it() {
    assert_tower(
        "gemma3_vision_tower_runs_on_a_thread_that_did_not_load_it",
        MEDGEMMA,
        image_request,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3-VL snapshot; run with `make gpu-test`"]
fn qwen3_vl_vision_tower_runs_on_a_thread_that_did_not_load_it() {
    assert_tower(
        "qwen3_vl_vision_tower_runs_on_a_thread_that_did_not_load_it",
        QWEN3_VL,
        image_request,
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn gemma4_audio_tower_runs_on_a_thread_that_did_not_load_it() {
    let wav = spoken_wav();
    let test = "gemma4_audio_tower_runs_on_a_thread_that_did_not_load_it";
    let Some([dir]) = snapshots(test, [GEMMA4_E2B]) else {
        return;
    };
    let load = arch_generator(dir.clone(), None);
    assert_hand_over(test, Split::AfterLoad, &load, || {
        vec![audio_request(&dir, &wav)]
    });
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-12B snapshot; run with `make gpu-test`"]
fn gemma4_unified_audio_embedder_runs_on_a_thread_that_did_not_load_it() {
    let wav = spoken_wav();
    let test = "gemma4_unified_audio_embedder_runs_on_a_thread_that_did_not_load_it";
    let Some([dir]) = snapshots(test, [GEMMA4_UNIFIED_12B]) else {
        return;
    };
    let load = arch_generator(dir.clone(), None);
    assert_hand_over(test, Split::AfterLoad, &load, || {
        vec![audio_request(&dir, &wav)]
    });
}

/// The encoder-output cache: the first image request writes the entry on one
/// thread, the second request hits it on another.
#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn an_encoder_cache_hit_decodes_on_another_thread() {
    let test = "an_encoder_cache_hit_decodes_on_another_thread";
    let Some([dir]) = snapshots(test, [GEMMA4_E2B]) else {
        return;
    };
    let run = |split: Split| {
        let cache = Arc::new(MultimodalCache::new(256 << 20));
        let load = arch_generator(dir.clone(), Some(Arc::clone(&cache)));
        let out = serve(split, &load, vec![image_request(&dir), image_request(&dir)]);
        (out, cache.stats().hits)
    };
    let (oracle, oracle_hits) = run(Split::None);
    assert!(
        oracle
            .iter()
            .all(|r| r.as_ref().is_ok_and(|ids| !ids.is_empty())),
        "{test}: the one-thread oracle must decode: {oracle:?}"
    );
    let (crossed, hits) = run(Split::BeforeLastRequest);
    let second = crossed[1].as_ref().unwrap_or_else(|e| {
        panic!("{test}: an encoder-cache hit failed on a thread that did not write it: {e}")
    });
    assert_eq!(
        hits, 1,
        "{test}: the second request must hit the encoder cache once"
    );
    assert_eq!(
        oracle_hits, 1,
        "{test}: the oracle's second request must hit once"
    );
    assert_eq!(
        Some(second),
        oracle[1].as_ref().ok(),
        "{test}: different tokens than the oracle"
    );
    assert_eq!(
        crossed[0], oracle[0],
        "{test}: first request differs from the oracle"
    );
}

// ── Speculative drafters: `SpeculativeGenerator` ────────────────────────────

fn assert_drafter(test: &str, verifier: &str, drafter: &str) {
    let Some([v, d]) = snapshots(test, [verifier, drafter]) else {
        return;
    };
    let load = speculative_generator(v.clone(), d);
    assert_hand_over(test, Split::AfterLoad, &load, || vec![text_request(&v)]);
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b + assistant snapshots; run with `make gpu-test`"]
fn gemma4_mtp_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "gemma4_mtp_drafter_runs_on_a_thread_that_did_not_load_it",
        GEMMA4_E2B,
        GEMMA4_E2B_ASSISTANT,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B 4-bit + MTP snapshots; run with `make gpu-test`"]
fn qwen_mtp_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "qwen_mtp_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN38_4BIT,
        QWEN38_MTP_4BIT,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.6-35B + DFlash snapshots; run with `make gpu-test`"]
fn dflash_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "dflash_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN36_MOE,
        QWEN36_DFLASH,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B 4-bit + DFlash2 snapshots; run with `make gpu-test`"]
fn dflash2_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "dflash2_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN38_4BIT,
        QWEN38_DFLASH2,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.6-35B + EAGLE-3 snapshots; run with `make gpu-test`"]
fn eagle3_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "eagle3_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN36_MOE,
        QWEN36_EAGLE3,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B mxfp8 + ornith-9b snapshots; run with `make gpu-test`"]
fn two_model_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "two_model_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN38_MXFP8,
        ORNITH_9B,
    );
}

// ── Models the server loads inside their first request ──────────────────────

/// The first request loads jina-v4 and embeds text; the second embeds an image
/// on another thread, the first use of the vision weights.
#[test]
#[ignore = "requires Metal GPU and the jina-v4 snapshot; run with `make gpu-test`"]
fn jina_image_embedding_runs_on_a_thread_that_did_not_load_it() {
    use rmlx_models::jina_v4;
    let test = "jina_image_embedding_runs_on_a_thread_that_did_not_load_it";
    let Some([dir]) = snapshots(test, [JINA_V4]) else {
        return;
    };
    let run = |split: bool| {
        let (d1, d2) = (dir.clone(), dir.clone());
        let first = move || {
            let model = jina_v4::load_from_path(&d1).expect("jina_v4::load_from_path");
            let tokenizer = tokenizers::Tokenizer::from_file(d1.join("tokenizer.json"))
                .expect("tokenizer.json");
            let ids: Vec<i64> = tokenizer
                .encode(SPOKEN_TEXT, true)
                .expect("encode")
                .get_ids()
                .iter()
                .map(|&id| i64::from(id))
                .collect();
            let text = model
                .embed_single(&ids, Device::Gpu, None)
                .map(|v| f32_bits(&v));
            ((model, tokenizer), text)
        };
        let second = move |(model, tokenizer): (jina_v4::JinaV4, tokenizers::Tokenizer)| {
            let png = rmlx_server::image_io::load_image(
                &format!("data:image/png;base64,{RED_PNG_BASE64}"),
                rmlx_server::image_io::DEFAULT_HTTP_TIMEOUT,
            )
            .expect("png");
            let pcfg = jina_v4::ImagePreprocessConfig::from_model_dir(&d2).expect("preprocessor");
            let pv = jina_v4::preprocess_image_bytes(&png, &pcfg).expect("preprocess");
            let prompt: Vec<i64> = tokenizer
                .encode(jina_v4::image_prompt(), false)
                .expect("encode")
                .get_ids()
                .iter()
                .map(|&id| i64::from(id))
                .collect();
            model
                .embed_image_single(&prompt, &pv, Device::Gpu, None, None, 0)
                .map(|v| f32_bits(&v))
        };
        hand_over(split, first, second)
    };
    let (oracle_text, oracle_image) = run(false);
    let oracle_text = oracle_text.expect("the one-thread oracle must embed text");
    let oracle_image = oracle_image.expect("the one-thread oracle must embed the image");
    let (text, image) = run(true);
    let image = image.unwrap_or_else(|e| {
        panic!("{test}: the image embedding failed on a thread that did not load the model: {e}")
    });
    assert_eq!(
        text.ok(),
        Some(oracle_text),
        "{test}: text embedding differs from the oracle"
    );
    assert_eq!(
        image, oracle_image,
        "{test}: image embedding differs from the oracle"
    );
}

/// The first request loads Whisper and transcribes; the second transcribes on
/// another thread.
#[test]
#[ignore = "requires Metal GPU and the Whisper snapshots; run with `make gpu-test`"]
fn whisper_transcribes_on_a_thread_that_did_not_load_it() {
    use rmlx_audio::tokenizer::{WhisperTask, WhisperTokenizer};
    use rmlx_audio::transcribe::{TranscribeOptions, Transcriber};
    use rmlx_audio::whisper::WhisperModel;
    let test = "whisper_transcribes_on_a_thread_that_did_not_load_it";
    let Some([model_dir, tok_dir]) = snapshots(test, [WHISPER, WHISPER_TOKENIZER]) else {
        return;
    };
    let (raw, rate) = rmlx_audio::wav::WavDecoder::decode(&spoken_wav()).expect("decode wav");
    let samples = Arc::new(rmlx_audio::transcribe::resample_to_16k(&raw, rate));
    let transcribe = |model: &Arc<WhisperModel>, tok: &Arc<WhisperTokenizer>, samples: &[f32]| {
        let opts = TranscribeOptions {
            language: "en".to_owned(),
            task: WhisperTask::Transcribe,
            temperature: 0.0,
            condition_on_previous_text: true,
        };
        Transcriber::new(Arc::clone(model), Arc::clone(tok))
            .and_then(|t| t.transcribe(samples, &opts, Device::Gpu))
            .map(|t| t.text)
            .map_err(|e| e.to_string())
    };
    let run = |split: bool| {
        let (m, t, s1, s2) = (
            model_dir.clone(),
            tok_dir.clone(),
            Arc::clone(&samples),
            Arc::clone(&samples),
        );
        let first = move || {
            let model = Arc::new(WhisperModel::load(&m).expect("WhisperModel::load"));
            let tok = Arc::new(WhisperTokenizer::from_path(&t).expect("WhisperTokenizer"));
            let text = transcribe(&model, &tok, &s1);
            ((model, tok), text)
        };
        let second = move |(model, tok): (Arc<WhisperModel>, Arc<WhisperTokenizer>)| {
            transcribe(&model, &tok, &s2)
        };
        hand_over(split, first, second)
    };
    let (oracle_first, oracle_second) = run(false);
    let oracle_second = oracle_second.expect("the one-thread oracle must transcribe");
    assert!(
        !oracle_second.trim().is_empty(),
        "{test}: the oracle transcribed nothing"
    );
    let (first, second) = run(true);
    let second = second.unwrap_or_else(|e| {
        panic!("{test}: transcription failed on a thread that did not load the model: {e}")
    });
    assert_eq!(
        first, oracle_first,
        "{test}: first transcription differs from the oracle"
    );
    assert_eq!(
        second, oracle_second,
        "{test}: second transcription differs from the oracle"
    );
}

/// The first request loads the TTS weights and synthesizes; the second
/// synthesizes on another thread.
#[test]
#[ignore = "requires Metal GPU and the Qwen3-TTS snapshots; run with `make gpu-test`"]
fn tts_synthesizes_on_a_thread_that_did_not_load_it() {
    use rmlx_audio::tts::{synthesize, TtsModel, TtsTokenizer};
    let test = "tts_synthesizes_on_a_thread_that_did_not_load_it";
    let Some([model_dir, codec_dir]) = snapshots(test, [TTS, TTS_CODEC]) else {
        return;
    };
    let speak = |model: &mut TtsModel, tok: &TtsTokenizer| {
        synthesize(SPOKEN_TEXT, "serena", model, tok)
            .map(|(samples, _rate)| f32_bits(&samples))
            .map_err(|e| e.to_string())
    };
    let run = |split: bool| {
        let (m, c) = (model_dir.clone(), codec_dir.clone());
        let first = move || {
            let mut model =
                TtsModel::load_config(&m, &c, Device::Gpu).expect("TtsModel::load_config");
            let tok = TtsTokenizer::from_path(&m).expect("TtsTokenizer");
            let audio = speak(&mut model, &tok);
            ((model, tok), audio)
        };
        let second = move |(mut model, tok): (TtsModel, TtsTokenizer)| speak(&mut model, &tok);
        hand_over(split, first, second)
    };
    let (oracle_first, oracle_second) = run(false);
    let oracle_second = oracle_second.expect("the one-thread oracle must synthesize");
    assert!(
        !oracle_second.is_empty(),
        "{test}: the oracle synthesized no samples"
    );
    let (first, second) = run(true);
    let second = second.unwrap_or_else(|e| {
        panic!("{test}: synthesis failed on a thread that did not load the weights: {e}")
    });
    assert_eq!(
        first, oracle_first,
        "{test}: first synthesis differs from the oracle"
    );
    assert_eq!(
        second, oracle_second,
        "{test}: second synthesis differs from the oracle"
    );
}
