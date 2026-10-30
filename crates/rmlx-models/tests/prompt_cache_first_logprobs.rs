//! An exact prompt-cache hit replays the first token's logprobs, so a request
//! asking for logprobs gets one record per emitted token whether or not its
//! prompt was cached.
//!
//! One body, run over each architecture that emits logprobs and keeps a prompt
//! cache. Qwen3.5 and BitNet also have CPU cases beside their generate paths,
//! and Qwen3 is held by `bonsai_golden_tokens`. Gemma3 has no case here. Laguna,
//! Qwen2 and Qwen3-VL-MoE emit no logprobs on any path.
//!
//! Models resolve from `RMLX_O_MODELS_ROOT` by slug, or `RMLX_KV_TEST_MODEL` by
//! path.
//!
//! ```text
//! cargo test -p rmlx-models --test prompt_cache_first_logprobs -- --ignored --test-threads=1
//! ```

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "real-model test: a missing snapshot, step or record aborts the test loudly"
)]

mod common;

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, ProbeStep, SamplerConfig};

const GEMMA4: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__gemma-4-e2b-it-mxfp8",
    archs: &["Gemma4ForConditionalGeneration"],
};
const BITNET: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__bitnet-b1.58-2B-4T",
    archs: &["BitNetForCausalLM"],
};
/// The smallest local Qwen3.5-family snapshot with gated-delta-net layers.
const QWEN35: common::GoldenModel = common::GoldenModel {
    slug: "sahilchachra__ornith-1.0-9b-mxfp8-mlx",
    archs: &["Qwen3_5ForConditionalGeneration"],
};

/// Slots for every generation here. Non-zero on purpose: a zero-slot cache
/// stores nothing and the test would pass without testing anything.
const SLOTS: usize = 4;

const N_TOKENS: usize = 4;

fn generate(
    model: &arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    prompt_ids: &[u32],
    top_logprobs_k: u32,
) -> Vec<ProbeStep> {
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();
    model
        .generate_greedy(
            tokenizer,
            prompt_ids,
            N_TOKENS,
            Device::Gpu,
            Some(KvQuant::K8V8),
            None,
            SLOTS,
            &[],
            &mut |_| rmlx_models::StepVerdict::Continue,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
            &std::sync::atomic::AtomicBool::new(false),
        )
        .expect("generate_greedy")
}

fn hits(model: &arch::Architecture) -> u64 {
    model.cache_stats().map_or(0, |s| s.hits)
}

/// The model of `model` and its tokenizer, or `None` after a named SKIP.
fn load(
    model: &common::GoldenModel,
    test: &str,
) -> Option<(arch::Architecture, tokenizers::Tokenizer)> {
    let path = common::model_for(model, test)?;
    let loaded =
        arch::load_model(&path, Device::Gpu, &arch::LoadOpts::default()).expect("arch::load_model");
    let tokenizer =
        tokenizers::Tokenizer::from_file(path.join("tokenizer.json")).expect("load tokenizer.json");
    Some((loaded, tokenizer))
}

/// A prompt of at least one full block, or the cache never finds it again.
fn prompt(tokenizer: &tokenizers::Tokenizer) -> Vec<u32> {
    let text = "A cartographer walked the ridge at dawn, tracing every river and \
                switchback onto oiled linen. "
        .repeat(24);
    let ids = tokenizer
        .encode(text, true)
        .expect("tokenize")
        .get_ids()
        .to_vec();
    assert!(ids.len() >= 256, "prompt must exceed one block");
    ids
}

/// `prompt` followed by more text, so a request for it resumes from the entry
/// the shorter one left.
fn extended(tokenizer: &tokenizers::Tokenizer, prompt: &[u32]) -> Vec<u32> {
    let more = tokenizer
        .encode(" The map was finished by nightfall.", false)
        .expect("tokenize");
    [prompt, more.get_ids()].concat()
}

/// A miss and an exact hit with the same `top_logprobs_k` carry the same
/// first-token record and one record per step; a hit asking for none carries
/// none.
fn miss_then_hit(model: &common::GoldenModel, test: &str) {
    let Some((loaded, tokenizer)) = load(model, test) else {
        return;
    };
    let prompt = prompt(&tokenizer);

    loaded.clear_prompt_cache();
    let miss = generate(&loaded, &tokenizer, &prompt, 3);
    assert_eq!(hits(&loaded), 0, "the first generation is a miss");
    assert!(
        miss.iter().all(|s| s.logprobs.is_some()),
        "a miss carries one record per step"
    );

    let hit = generate(&loaded, &tokenizer, &prompt, 3);
    assert_eq!(hits(&loaded), 1, "the second generation is an exact hit");
    assert_eq!(hit[0].token_id, miss[0].token_id, "an exact hit replays");
    assert!(
        hit.iter().all(|s| s.logprobs.is_some()),
        "an exact hit carries one record per step"
    );
    let (replayed, missed) = (hit[0].logprobs.as_ref(), miss[0].logprobs.as_ref());
    let (replayed, missed) = (replayed.unwrap(), missed.unwrap());
    assert_eq!(replayed.token_id, missed.token_id);
    assert_eq!(replayed.token_logprob, missed.token_logprob);
    assert_eq!(replayed.top, missed.top);
    assert_eq!(replayed.top.len(), 3);

    let off = generate(&loaded, &tokenizer, &prompt, 0);
    assert_eq!(hits(&loaded), 2);
    assert!(
        off.iter().all(|s| s.logprobs.is_none()),
        "logprobs off replays none"
    );
}

/// A request that resumes from a shorter prompt's entry stores its own entry,
/// and the exact hit on that one replays the record the resumed request
/// computed: the same logits through the same function, so bit for bit.
fn resume_then_hit(model: &common::GoldenModel, test: &str) {
    let Some((loaded, tokenizer)) = load(model, test) else {
        return;
    };
    let short = prompt(&tokenizer);
    let long = extended(&tokenizer, &short);

    loaded.clear_prompt_cache();
    generate(&loaded, &tokenizer, &short, 0);
    let resumed = generate(&loaded, &tokenizer, &long, 3);
    assert_eq!(
        hits(&loaded),
        1,
        "the longer prompt resumes from the saved one"
    );

    let hit = generate(&loaded, &tokenizer, &long, 3);
    assert_eq!(
        hits(&loaded),
        2,
        "the repeated longer prompt is an exact hit"
    );
    assert!(
        hit.iter().all(|s| s.logprobs.is_some()),
        "an exact hit carries one record per step"
    );
    assert_eq!(hit[0].token_id, resumed[0].token_id);
    let (replayed, own) = (hit[0].logprobs.as_ref(), resumed[0].logprobs.as_ref());
    let (replayed, own) = (replayed.unwrap(), own.unwrap());
    assert_eq!(replayed.token_id, own.token_id);
    assert_eq!(replayed.token_logprob, own.token_logprob);
    assert_eq!(replayed.top, own.top);
}

#[ignore = "loads a model and drives the GPU"]
#[test]
fn gemma4_exact_hit_replays_first_logprobs() {
    miss_then_hit(&GEMMA4, "gemma4_exact_hit_replays_first_logprobs");
}

#[ignore = "loads a model and drives the GPU"]
#[test]
fn bitnet_exact_hit_replays_first_logprobs() {
    miss_then_hit(&BITNET, "bitnet_exact_hit_replays_first_logprobs");
}

#[ignore = "loads a model and drives the GPU"]
#[test]
fn gemma4_resumed_prompt_replays_first_logprobs() {
    resume_then_hit(&GEMMA4, "gemma4_resumed_prompt_replays_first_logprobs");
}

#[ignore = "loads a model and drives the GPU"]
#[test]
fn qwen35_cached_prompt_replays_first_logprobs() {
    miss_then_hit(&QWEN35, "qwen35_cached_prompt_replays_first_logprobs");
}

#[ignore = "loads a model and drives the GPU"]
#[test]
fn qwen35_resumed_prompt_replays_first_logprobs() {
    resume_then_hit(&QWEN35, "qwen35_resumed_prompt_replays_first_logprobs");
}
