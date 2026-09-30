//! A loaded model must generate on a thread that did not load it.
//!
//! `rmlx serve` never generates on the thread that loaded the model: the eager
//! preload and the on-demand load run on one thread, and generation runs on a
//! tokio blocking-pool thread. MLX binds every lazy op to the default stream of
//! the thread that built it. From MLX 0.32 the CPU command-encoder map is
//! thread-local, so the first evaluation of such an op on another thread fails
//! with "There is no Stream(cpu, N) in current thread.". A loader that leaves a
//! lazy op in the model it returns (the Qwen3 fused QKV `concatenate`, for
//! example) therefore makes generation fail on any thread but the load thread.
//!
//! The oracle is the same checkpoint, loaded and generated on one thread. The
//! cross-thread arm must succeed and emit the same greedy token ids.
//!
//! Run:
//! cargo test -p rmlx-models --test load_thread_boundary -- --ignored --nocapture

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr
)]

mod common;

use std::path::{Path, PathBuf};
use std::sync::mpsc;

use rmlx_mlx::Device;
use rmlx_models::{arch, Pcg32, PenaltyConfig, SamplerConfig};

/// A dense Qwen3 checkpoint with quantized Q/K/V, so the loader builds the
/// fused QKV projection as a lazy CPU `concatenate`.
const BONSAI: common::GoldenModel = common::GoldenModel {
    slug: "prism-ml__Ternary-Bonsai-8B-mlx-2bit",
    archs: &["Qwen3ForCausalLM"],
};

const N_TOKENS: usize = 32;

/// Greedy-decode `N_TOKENS` ids from `model` on the calling thread, with the
/// prompt cache disabled so that no arm can read another arm's entry.
fn decode(model: &arch::Architecture, model_path: &Path) -> rmlx_core::error::Result<Vec<u32>> {
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt_ids: Vec<u32> = tokenizer
        .encode(common::GOLDEN_PROMPT, true)
        .expect("tokenize")
        .get_ids()
        .to_vec();
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
        &prompt_ids,
        N_TOKENS,
        Device::Gpu,
        None,
        None,
        0,
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

fn load(model_path: &Path) -> arch::Architecture {
    arch::load_model(model_path, Device::Gpu, &arch::LoadOpts::default()).expect("load_model")
}

/// Load on one thread and decode on a second thread, while the load thread
/// stays alive and idle, as the eager-preload thread does in `rmlx serve`.
fn load_here_decode_there(model_path: PathBuf) -> rmlx_core::error::Result<Vec<u32>> {
    let (model_tx, model_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let loader_path = model_path.clone();
    let loader = std::thread::spawn(move || {
        model_tx.send(load(&loader_path)).expect("send model");
        release_rx.recv().ok();
    });
    let model = model_rx.recv().expect("model from the load thread");
    let result = std::thread::spawn(move || decode(&model, &model_path))
        .join()
        .expect("decode thread panicked");
    release_tx.send(()).ok();
    loader.join().expect("load thread panicked");
    result
}

#[test]
#[ignore = "requires Metal GPU and the Bonsai snapshot; run with `make gpu-test`"]
fn a_model_loaded_on_one_thread_generates_on_another() {
    let Some(model_path) =
        common::model_for(&BONSAI, "a_model_loaded_on_one_thread_generates_on_another")
    else {
        return;
    };

    let same_thread = {
        let path = model_path.clone();
        std::thread::spawn(move || decode(&load(&path), &path))
            .join()
            .expect("same-thread arm panicked")
            .expect("the same-thread arm is the oracle and must decode")
    };
    assert_eq!(
        same_thread.len(),
        N_TOKENS,
        "oracle decoded a short sequence"
    );

    let cross_thread = load_here_decode_there(model_path).unwrap_or_else(|e| {
        panic!(
            "decode failed on a thread that did not load the model: {e}\n\
             The loader returned a lazy array bound to the load thread's stream."
        )
    });
    assert_eq!(
        cross_thread, same_thread,
        "the cross-thread decode emitted different ids than the same-thread oracle"
    );
}
