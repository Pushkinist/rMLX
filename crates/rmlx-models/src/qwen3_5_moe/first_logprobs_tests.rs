//! The first token's logprobs across a prompt-cache exact hit, on the
//! attention-only synthetic snapshot (`TinyPack`) served on the CPU.

use std::sync::atomic::AtomicBool;

use super::*;
use crate::decode_loop::decode_loop_tests::tiny_tokenizer;
use crate::decode_loop::ProbeStep;
use crate::prompt_cache::BLOCK_TOKENS;
use crate::qwen3_5_moe::generate_greedy;
use crate::qwen3_5_moe::prompt_cache::{ensure_prompt_cache, PROMPT_CACHE};
use crate::sampler::{Pcg32, PenaltyConfig, SamplerConfig};
use rmlx_kv_quant::KvQuant;

/// One greedy generation of two tokens over a prompt of one block and `tail`
/// more tokens, asking for `top_k` logprobs. The cache keys on block digests,
/// so a prompt shorter than a block is never found again.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a failed generation aborts the test loudly"
)]
fn generate(model: &Qwen3_5MoeText, tail: usize, top_k: u32) -> Vec<ProbeStep> {
    let tokenizer = tiny_tokenizer(TINY_VOCAB as u32);
    let prompt: Vec<u32> = (0..(BLOCK_TOKENS + tail) as u32)
        .map(|i| i % TINY_VOCAB as u32)
        .collect();
    let sampler = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: top_k,
    };
    let mut rng = Pcg32::new(sampler.seed_or_default());
    let mut history = Vec::new();
    let mut step_fn = |_: &ProbeStep| -> Option<u32> { None };
    generate_greedy(
        model,
        &tokenizer,
        &prompt,
        2,
        rmlx_mlx::Device::Cpu,
        KvQuant::None,
        None,
        4,
        &[],
        &mut step_fn,
        None,
        &sampler,
        &mut rng,
        &PenaltyConfig::default(),
        &mut history,
        &AtomicBool::new(false),
    )
    .expect("generate_greedy")
}

fn clear_prompt_cache() {
    ensure_prompt_cache(4);
    PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
        }
    });
}

#[test]
fn an_exact_hit_replays_the_first_token_logprobs_a_miss_computed() {
    crate::test_child::run_child(
        "qwen3_5_moe::loader::pack_loader_tests::first_logprobs_tests::\
         an_exact_hit_replays_the_first_token_logprobs_a_miss_computed_child",
    );
}

fn cache_hits() -> u64 {
    PROMPT_CACHE.with_inner_mut(|guard| guard.as_ref().map_or(0, |c| c.stats().hits))
}

/// A miss with `top_logprobs_k = 3` and an exact hit with the same value emit
/// the same first-token record. A hit that asks for none emits none, though
/// the entry was stored by a request that asked for three.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test fixture: a missing step or record aborts the test loudly"
)]
fn an_exact_hit_replays_the_first_token_logprobs_a_miss_computed_child() -> Result<()> {
    if !crate::test_child::started_by_parent() {
        return Ok(());
    }
    rmlx_mlx::forbid_gpu()?;
    let model = TinyPack {
        gdn: false,
        ..PLAIN
    }
    .load()?;

    clear_prompt_cache();
    let miss = generate(&model, 0, 3);
    let missed = miss[0].logprobs.as_ref().expect("a miss computes logprobs");
    assert_eq!(missed.token_id, miss[0].token_id);
    assert_eq!(missed.top.len(), 3);

    assert_eq!(cache_hits(), 0, "the first generation is a miss");
    let hit = generate(&model, 0, 3);
    assert_eq!(cache_hits(), 1, "the second generation is a hit");
    assert_eq!(hit[0].token_id, miss[0].token_id, "an exact hit replays");
    let replayed = hit[0]
        .logprobs
        .as_ref()
        .expect("an exact hit replays the stored logprobs");
    assert_eq!(replayed.token_id, missed.token_id);
    assert_eq!(replayed.token_logprob, missed.token_logprob);
    assert_eq!(replayed.top, missed.top);

    let off = generate(&model, 0, 0);
    assert!(off[0].logprobs.is_none(), "logprobs off replays none");

    clear_prompt_cache();
    let stored_with_none = generate(&model, 0, 0);
    assert!(stored_with_none[0].logprobs.is_none());
    let hit = generate(&model, 0, 2);
    let replayed = hit[0]
        .logprobs
        .as_ref()
        .expect("an entry stored with logprobs off still replays them");
    assert_eq!(replayed.top.len(), 2);
    assert_eq!(replayed.token_logprob, missed.token_logprob);

    println!("{}", crate::test_child::CHILD_DONE);
    Ok(())
}

#[test]
fn a_prompt_resumed_from_a_saved_prefix_replays_its_first_token_logprobs() {
    crate::test_child::run_child(
        "qwen3_5_moe::loader::pack_loader_tests::first_logprobs_tests::\
         a_prompt_resumed_from_a_saved_prefix_replays_its_first_token_logprobs_child",
    );
}

/// The entry a resumed request stores carries the logprobs of the token the
/// tail forward produced.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test fixture: a missing step or record aborts the test loudly"
)]
fn a_prompt_resumed_from_a_saved_prefix_replays_its_first_token_logprobs_child() -> Result<()> {
    if !crate::test_child::started_by_parent() {
        return Ok(());
    }
    rmlx_mlx::forbid_gpu()?;
    let model = TinyPack {
        gdn: false,
        ..PLAIN
    }
    .load()?;

    clear_prompt_cache();
    generate(&model, 0, 0);
    let resumed = generate(&model, 3, 3);
    assert_eq!(
        cache_hits(),
        1,
        "the longer prompt resumes from the saved one"
    );

    let hit = generate(&model, 3, 3);
    assert_eq!(
        cache_hits(),
        2,
        "the repeated longer prompt is an exact hit"
    );
    assert_eq!(hit[0].token_id, resumed[0].token_id);
    let replayed = hit[0]
        .logprobs
        .as_ref()
        .expect("a resumed entry replays the logprobs of its first token");
    let emitted = resumed[0]
        .logprobs
        .as_ref()
        .expect("the resumed request computed logprobs");
    assert_eq!(replayed.token_id, emitted.token_id);
    assert_eq!(replayed.token_logprob, emitted.token_logprob);
    assert_eq!(replayed.top, emitted.top);

    println!("{}", crate::test_child::CHILD_DONE);
    Ok(())
}
