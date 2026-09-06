//! The two-model round loop under stochastic acceptance actually runs.
//!
//! `SpeculativeDispatcher::spec_generate_greedy` routes a request with
//! `temperature > 0` to `spec_generate_stochastic_cached` — the Leviathan
//! acceptance loop, the only round loop whose correctness argument differs
//! from greedy. Nothing else in the tree drives it: every alignment suite runs
//! at `temperature == 0`, so a change that broke the stochastic loop, or
//! silently routed it back to greedy, failed no gate.
//!
//! What this pins, on a real pair:
//!
//! - the loop emits an answer (not an error, not zero tokens);
//! - one seed reproduces one sequence — the request's `Pcg32` is threaded
//!   through draft sampling, the acceptance draws and the residual resamples
//!   in a fixed order, and a loop that reseeded or fell back to an unseeded
//!   draw would not;
//! - a different seed gives a different sequence, and so does `temperature ==
//!   0` — the loop is sampling, not argmaxing under another name.
//!
//! The pair is a Gemma4 verifier with the smaller Gemma4 as its full draft
//! model — the classic two-model form — resolved by slug from
//! `RMLX_O_MODELS_ROOT` so `make gpu-test` runs it wherever the snapshots are.
//! Two runs reaching Metal per assertion, so the test is `#[ignore]`d and
//! serialised by that target.
//!
//! Run:
//! RMLX_O_MODELS_ROOT=<models-root> \
//! cargo test -p rmlx-models --test two_model_stochastic -- --ignored --nocapture

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::too_many_lines
)]

mod common;

use std::path::{Path, PathBuf};

use rmlx_mlx::Device;
use rmlx_models::sampler::SamplerConfig;
use rmlx_models::speculative::SpeculativeDispatcher;

const VERIFIER_SLUG: &str = "mlx-community__gemma-4-e4b-it-mxfp8";
const DRAFT_SLUG: &str = "mlx-community__gemma-4-e2b-it-mxfp8";

/// Chat-templated the way Gemma4 expects, so the model answers rather than
/// continues. Open-ended on purpose: a prompt with one right answer leaves
/// sampling nothing to vary.
const PROMPT: &str =
    "<bos><start_of_turn>user\nWrite three sentences about the sea.<end_of_turn>\n<start_of_turn>model\n";

const N_TOKENS: usize = 64;
const K: usize = 4;

/// A snapshot by slug, or the reason this test stands down. A misconfigured
/// root is a failure — see `tests/common/mod.rs`.
fn snapshot(slug: &str) -> Result<PathBuf, String> {
    let root = std::env::var(common::MODELS_ROOT_VAR).ok();
    match common::slug_snapshot(root.as_deref(), slug, common::Role::Standalone) {
        common::Snapshot::Found { path, .. } => Ok(path),
        common::Snapshot::Absent(why) => Err(why),
        common::Snapshot::Misconfigured(why) => panic!("{why}"),
    }
}

fn eos_ids(model_path: &Path) -> Vec<u32> {
    let raw = std::fs::read(model_path.join("config.json")).expect("config.json");
    let cfg: serde_json::Value = serde_json::from_slice(&raw).expect("config.json parses");
    match cfg.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => Vec::new(),
    }
}

fn sampler(temperature: f32, seed: u64) -> SamplerConfig {
    SamplerConfig {
        temperature,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(seed),
        top_logprobs_k: 0,
    }
}

fn generate(
    dispatcher: &SpeculativeDispatcher,
    tk: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    eos: &[u32],
    cfg: &SamplerConfig,
) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    let mut step_fn = |s: &rmlx_models::ProbeStep| {
        ids.push(s.token_id);
        None
    };
    dispatcher
        .spec_generate_greedy(
            tk,
            prompt_ids,
            N_TOKENS,
            K,
            Some(rmlx_kv_quant::KvQuant::None),
            None,
            0,
            eos,
            &mut step_fn,
            None,
            cfg,
        )
        .expect("speculative generate");
    ids
}

#[ignore]
#[test]
fn stochastic_two_model_loop_samples_reproducibly() {
    let (verifier_path, draft_path) = match (snapshot(VERIFIER_SLUG), snapshot(DRAFT_SLUG)) {
        (Ok(v), Ok(d)) => (v, d),
        (Err(why), _) | (_, Err(why)) => {
            eprintln!("SKIP stochastic_two_model_loop_samples_reproducibly: {why}");
            return;
        }
    };
    let device = Device::Gpu;
    let dispatcher = SpeculativeDispatcher::load_speculative(&verifier_path, &draft_path, device)
        .expect("load verifier + draft");
    let tk =
        tokenizers::Tokenizer::from_file(verifier_path.join("tokenizer.json")).expect("tokenizer");
    let prompt_ids: Vec<u32> = tk.encode(PROMPT, false).expect("encode").get_ids().to_vec();
    let eos = eos_ids(&verifier_path);
    assert!(
        !eos.is_empty(),
        "the verifier config must name its stop ids"
    );

    let sampled = generate(&dispatcher, &tk, &prompt_ids, &eos, &sampler(1.0, 7));
    let again = generate(&dispatcher, &tk, &prompt_ids, &eos, &sampler(1.0, 7));
    let other_seed = generate(&dispatcher, &tk, &prompt_ids, &eos, &sampler(1.0, 8));
    let greedy = generate(&dispatcher, &tk, &prompt_ids, &eos, &sampler(0.0, 7));

    eprintln!(
        "[two_model_stochastic] seed 7 = {:?}\n  seed 8 = {:?}\n  greedy = {:?}",
        tk.decode(&sampled, false).unwrap_or_default(),
        tk.decode(&other_seed, false).unwrap_or_default(),
        tk.decode(&greedy, false).unwrap_or_default(),
    );

    assert!(
        sampled.len() >= 12,
        "the stochastic loop must emit a real answer, got {} tokens",
        sampled.len()
    );
    assert_eq!(
        sampled, again,
        "one seed must reproduce one sequence — the request RNG is not threaded through the loop"
    );
    assert_ne!(
        sampled, other_seed,
        "two seeds gave one sequence over {N_TOKENS} tokens at temperature 1.0 — the seed is not reaching the draws"
    );
    assert_ne!(
        sampled, greedy,
        "temperature 1.0 reproduced the greedy sequence over {N_TOKENS} tokens — the request was routed to the greedy loop"
    );
}
