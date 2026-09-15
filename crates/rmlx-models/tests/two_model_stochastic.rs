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
//! **What it cannot see, and what does.** Every assertion here is
//! self-consistency *within one build*, so a change that **reorders** the
//! request's draws passes it: the reordered stream is as reproducible under its
//! seed as the old one was, as different from a second seed, and as far from
//! greedy. Nothing else reaches it either — the pinned round stream and the
//! equivalence pairs run at temperature 0, where the verifier's tokens come off
//! an argmax that never reaches a draw. The only reading of a moved draw order
//! is **the same seed at two commits**, which is what the control below prints.
//! The recipe in full, because two of its steps are where it silently stops
//! being a comparison:
//!
//! 1. Check the other commit out into its own worktree and build it with its
//!    own `CARGO_TARGET_DIR`. **Never a shared `target/`**: one binary
//!    overwrites the other and the run compares a commit against itself.
//! 2. **Copy this file unchanged into that worktree** — at an older commit the
//!    control does not exist, and a re-typed harness is a second variable — and
//!    record one digest over both copies to say they are the same file.
//! 3. Before trusting either run, confirm each binary carries a string only its
//!    own side has (`strings <binary>`): a build that silently reused the other
//!    side's artefacts reads as agreement.
//! 4. Run this test with `--nocapture` in both and diff the `CELL` lines. They
//!    must be identical id for id; a difference is a moved stream and not
//!    something to re-bless.
//!
//! The pair is a Gemma4 verifier with the smaller Gemma4 as its full draft
//! model — the classic two-model form — resolved by slug from
//! `RMLX_O_MODELS_ROOT` so `make gpu-test` runs it wherever the snapshots are.
//! Several runs reaching Metal, so the test is `#[ignore]`d and serialised by
//! that target.
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

/// The control's own prompts, one that stops on an EOS well inside the budget
/// and one that runs the whole of it, so the cross-commit diff covers both a
/// request that ends itself and a request the budget ends.
const CONTROL_PROMPTS: [(&str, &str); 2] = [
    ("sea", PROMPT),
    (
        "cache",
        "<bos><start_of_turn>user\nExplain what a KV cache is and why it helps.<end_of_turn>\n<start_of_turn>model\n",
    ),
];

/// The control's cells: every temperature against every seed. A reordered draw
/// stream moves every one of them; a filter that stopped being applied moves the
/// two temperatures differently.
const CONTROL_TEMPERATURES: [f32; 2] = [0.7, 1.0];
const CONTROL_SEEDS: [u64; 2] = [7, 8];

/// The control's budget. Long enough that a moved draw order cannot hide in a
/// prefix the verifier is confident about.
const CONTROL_TOKENS: usize = 256;

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
    generate_n(dispatcher, tk, prompt_ids, eos, cfg, N_TOKENS)
}

fn generate_n(
    dispatcher: &SpeculativeDispatcher,
    tk: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    eos: &[u32],
    cfg: &SamplerConfig,
    n_tokens: usize,
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
            n_tokens,
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

/// Print the token stream each seeded cell produces, for a diff against another
/// commit.
///
/// **Not a gate, and its own assertions are not the point.** What it asserts is
/// that every cell emitted and that two seeds part at each temperature; what it
/// is *for* is the eight `CELL` lines it prints, which are the only reading in
/// this tree of a change that reordered the request's draws. The recipe is in
/// this file's own doc: the same test, at two commits, in two trees with their
/// own target directories, diffed.
///
/// It sits in this file because it is the gate above's complement, on the same
/// pair and one of the same prompts: what that one reads within a build, this
/// one reads across two. It does **not** share that test's model load — each
/// builds its own dispatcher in its own body — and it earns its place in
/// `make gpu-test` anyway, at about a minute on a gate of twenty-one, because
/// it is the only reading in the tree with power over the draw stream.
#[ignore]
#[test]
fn print_the_seeded_stochastic_streams_for_a_cross_commit_diff() {
    let (verifier_path, draft_path) = match (snapshot(VERIFIER_SLUG), snapshot(DRAFT_SLUG)) {
        (Ok(v), Ok(d)) => (v, d),
        (Err(why), _) | (_, Err(why)) => {
            eprintln!("SKIP print_the_seeded_stochastic_streams_for_a_cross_commit_diff: {why}");
            return;
        }
    };
    let device = Device::Gpu;
    let dispatcher = SpeculativeDispatcher::load_speculative(&verifier_path, &draft_path, device)
        .expect("load verifier + draft");
    let tk =
        tokenizers::Tokenizer::from_file(verifier_path.join("tokenizer.json")).expect("tokenizer");
    let eos = eos_ids(&verifier_path);
    assert!(
        !eos.is_empty(),
        "the verifier config must name its stop ids"
    );

    for (prompt_name, prompt) in CONTROL_PROMPTS {
        let prompt_ids: Vec<u32> = tk.encode(prompt, false).expect("encode").get_ids().to_vec();
        for temperature in CONTROL_TEMPERATURES {
            let mut first: Option<Vec<u32>> = None;
            for seed in CONTROL_SEEDS {
                let ids = generate_n(
                    &dispatcher,
                    &tk,
                    &prompt_ids,
                    &eos,
                    &sampler(temperature, seed),
                    CONTROL_TOKENS,
                );
                assert!(
                    !ids.is_empty(),
                    "the {prompt_name} cell at temperature {temperature} seed {seed} emitted nothing"
                );
                let joined: Vec<String> = ids.iter().map(u32::to_string).collect();
                println!(
                    "CELL prompt={prompt_name} temp={temperature} seed={seed} n={} ids={}",
                    ids.len(),
                    joined.join(",")
                );
                println!(
                    "TEXT prompt={prompt_name} temp={temperature} seed={seed} {}",
                    tk.decode(&ids, false)
                        .unwrap_or_default()
                        .replace('\n', "\\n")
                );
                match first {
                    Some(ref prev) => assert_ne!(
                        *prev, ids,
                        "the {prompt_name} cells at temperature {temperature} gave one stream \
                         for two seeds, so the seed is not reaching the draws"
                    ),
                    None => first = Some(ids),
                }
            }
        }
    }
}
