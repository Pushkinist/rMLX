//! Integration test: the prompt cache never serves one model's K/V to another.
//!
//! The prompt cache is **one static per architecture**, so every model of that
//! arch the process has resident shares it — the multi-model registry keeps
//! several loaded, and a speculative draft/verifier pair is two models by
//! construction. The `Exact` arm matches on token-id equality, which cannot
//! separate two models asking the same question. Only the `model_sig` term in
//! the cache key can.
//!
//! Serving that hit is not a metrics problem: model B would replay model A's
//! post-prefill K/V *and* A's first decode token through B's weights, and emit
//! wrong output with no error anywhere.
//!
//! Needs two snapshots of the **same architecture** with the **same KV shape**
//! (so a cross-served snapshot is structurally acceptable and the test is
//! actually exercising the key, not a shape mismatch) and **different weights**
//! (so the wrong answer is visible). The gemma-4-E2B mxfp8 / QAT-4bit pair fits:
//! both `Gemma4ForConditionalGeneration`, both 35 layers × 1 KV head × head_dim
//! 256.
//!
//!   RMLX_PROMPT_CACHE_TEST_MODEL_A=/path/to/mlx-community__gemma-4-e2b-it-mxfp8 \
//!   RMLX_PROMPT_CACHE_TEST_MODEL_B=/path/to/mlx-community__gemma-4-E2B-it-qat-4bit \
//!     cargo test -p rmlx-models --test prompt_cache_cross_model \
//!       -- --ignored --test-threads=1 --nocapture
//!
//! `#[ignore]` so plain `cargo test` skips it (needs real models + GPU).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::indexing_slicing,
    clippy::ignore_without_reason
)]

use std::path::PathBuf;

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

/// Slots for every generation here. Non-zero on purpose: a zero-slot cache
/// stores nothing and would make this test pass without testing anything.
const SLOTS: usize = 4;

const CODEC: KvQuant = KvQuant::K8V8;

fn load(var: &str) -> Option<(arch::Architecture, tokenizers::Tokenizer)> {
    let Ok(p) = std::env::var(var) else {
        eprintln!("{var} not set — skipping prompt_cache_cross_model");
        return None;
    };
    let path = PathBuf::from(p);
    let model =
        arch::load_model(&path, Device::Gpu, &arch::LoadOpts::default()).expect("arch::load_model");
    let tokenizer =
        tokenizers::Tokenizer::from_file(path.join("tokenizer.json")).expect("tokenizer.json");
    Some((model, tokenizer))
}

/// One greedy generation; returns the decoded token ids.
fn gen(
    model: &arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    n_tokens: usize,
) -> Vec<u32> {
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();

    model
        .generate_greedy(
            tokenizer,
            prompt_ids,
            n_tokens,
            Device::Gpu,
            Some(CODEC),
            None,
            SLOTS,
            &[], // no EOS stop — every run emits the same token count
            &mut |_| None,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("generate_greedy")
        .iter()
        .map(|s| s.token_id)
        .collect()
}

fn hits_misses(model: &arch::Architecture) -> (u64, u64) {
    model.cache_stats().map_or((0, 0), |s| (s.hits, s.misses))
}

#[ignore]
#[test]
fn second_model_of_the_same_arch_is_not_served_the_first_models_kv() {
    let Some((model_a, tok_a)) = load("RMLX_PROMPT_CACHE_TEST_MODEL_A") else {
        return;
    };
    let Some((model_b, _tok_b)) = load("RMLX_PROMPT_CACHE_TEST_MODEL_B") else {
        return;
    };
    assert_eq!(
        model_a.arch_class(),
        model_b.arch_class(),
        "both snapshots must be the same architecture — that is the whole point: \
         they share one static prompt cache"
    );

    // Chat-templated so the models emit real content: fed raw prose with no
    // turn markers they both degenerate to the same end-of-turn token, and the
    // vacuity guard below would (correctly) refuse the comparison.
    //
    // >= 256 tokens so a full block is stored and matchable; below one block
    // `find_best_prefix` returns None for everyone and nothing is proven.
    let body = "A cartographer walked the ridge at dawn, tracing every river and \
                switchback onto oiled linen. "
        .repeat(24);
    let text = format!(
        "<start_of_turn>user\n{body}\nSummarise the passage above, then explain what \
         the cartographer was most likely trying to achieve.<end_of_turn>\n\
         <start_of_turn>model\n"
    );
    let prompt: Vec<u32> = tok_a
        .encode(text, true)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    assert!(prompt.len() >= 256, "prompt must exceed one block");
    let n = 24;

    // B's own answer, from a cold cache.
    model_b.clear_prompt_cache();
    let b_alone = gen(&model_b, &tok_a, &prompt, n);

    // A now owns a slot for this exact prompt.
    model_a.clear_prompt_cache();
    let a_first = gen(&model_a, &tok_a, &prompt, n);

    // Control: A's own repeat must be an Exact hit. Without this, a Miss for B
    // below could just mean the slot was never stored, and the test would pass
    // for the wrong reason.
    let (h0, m0) = hits_misses(&model_a);
    let a_repeat = gen(&model_a, &tok_a, &prompt, n);
    let (h1, m1) = hits_misses(&model_a);
    assert_eq!(
        h1 - h0,
        1,
        "A's identical repeat must hit its own slot — otherwise B's miss proves nothing"
    );
    assert_eq!(a_first, a_repeat, "A's cache hit must reproduce A's output");

    // The two models must genuinely disagree, or a leak would be invisible.
    assert_ne!(
        a_first, b_alone,
        "the two snapshots decode the same prompt identically, so this pair cannot \
         discriminate — pick two models whose outputs differ"
    );

    // The load-bearing call: B asks the identical question while A's slot is live.
    let b_after_a = gen(&model_b, &tok_a, &prompt, n);
    let (h2, m2) = hits_misses(&model_b);

    println!(
        "[cross_model] arch={} prompt_len={} a_first={:?} b_alone={:?} b_after_a={:?} \
         A(h,m)=({h0},{m0})->({h1},{m1}) B_after(h,m)=({h2},{m2})",
        model_a.arch_class(),
        prompt.len(),
        &a_first[..4.min(a_first.len())],
        &b_alone[..4.min(b_alone.len())],
        &b_after_a[..4.min(b_after_a.len())],
    );

    assert_eq!(
        h2 - h1,
        0,
        "B must not hit a slot stored by A — the cache key has to carry model identity"
    );
    assert_eq!(m2 - m1, 1, "B's request must be recorded as a miss");
    assert_eq!(
        b_after_a, b_alone,
        "B decoded different tokens with A's snapshot resident than it did from a cold \
         cache — it was served another model's K/V and produced wrong output"
    );
}
