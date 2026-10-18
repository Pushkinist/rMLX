//! The first token's logprobs across a prompt-cache exact hit, on the one-layer
//! synthetic snapshot of `device_tests` served on the CPU. The GPU latch is
//! one-way, so the check runs in a child of this test binary.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test assertions: a failed setup step or a missing record is a test failure"
)]

use std::sync::atomic::AtomicBool;

use super::device_tests::write_snapshot;
use super::prompt_cache::{ensure_prompt_cache, PROMPT_CACHE};
use crate::decode_loop::decode_loop_tests::tiny_tokenizer;
use crate::decode_loop::ProbeStep;
use crate::prompt_cache::BLOCK_TOKENS;
use crate::sampler::{Pcg32, PenaltyConfig, SamplerConfig};
use crate::test_child::{run_child, started_by_parent, CHILD_DONE};
use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;

/// The vocabulary of the synthetic snapshot.
const VOCAB: u32 = 8;

/// Slots of the prompt cache for every generation but the zero-slot case.
const SLOTS: usize = 4;

fn generate(model: &super::BitNetText, top_k: u32) -> Vec<ProbeStep> {
    generate_with_slots(model, SLOTS, top_k)
}

/// One greedy generation of two tokens over a one-block prompt, asking for
/// `top_k` logprobs. The cache keys on block digests, so a prompt shorter than
/// a block is never found again.
fn generate_with_slots(model: &super::BitNetText, slots: usize, top_k: u32) -> Vec<ProbeStep> {
    let tokenizer = tiny_tokenizer(VOCAB);
    let prompt: Vec<u32> = (0..BLOCK_TOKENS as u32).map(|i| i % VOCAB).collect();
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
    let mut step_fn = |_: &ProbeStep| -> crate::decode_loop::StepVerdict {
        crate::decode_loop::StepVerdict::Continue
    };
    super::generate_greedy(
        model,
        &tokenizer,
        &prompt,
        2,
        Device::Cpu,
        KvQuant::None,
        None,
        slots,
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

fn cache_hits() -> u64 {
    PROMPT_CACHE.with_inner_mut(|guard| guard.as_ref().map_or(0, |c| c.stats().hits))
}

#[test]
fn an_exact_hit_replays_the_first_token_logprobs_a_miss_computed() {
    run_child(
        "bitnet::first_logprobs_tests::\
         an_exact_hit_replays_the_first_token_logprobs_a_miss_computed_child",
    );
}

/// A miss with `top_logprobs_k = 3` and an exact hit with the same value emit
/// the same first-token record. A hit that asks for none emits none, and an
/// entry stored by a request that asked for none still replays on demand.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn an_exact_hit_replays_the_first_token_logprobs_a_miss_computed_child() {
    if !started_by_parent() {
        return;
    }
    rmlx_mlx::forbid_gpu().unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path());
    let model = super::load_from_path(dir.path(), Device::Cpu).expect("CPU load");

    clear_prompt_cache();
    let miss = generate(&model, 3);
    let missed = miss[0].logprobs.as_ref().expect("a miss computes logprobs");
    assert_eq!(missed.token_id, miss[0].token_id);
    assert_eq!(missed.top.len(), 3);
    assert_eq!(cache_hits(), 0, "the first generation is a miss");

    let hit = generate(&model, 3);
    assert_eq!(cache_hits(), 1, "the second generation is a hit");
    assert_eq!(hit[0].token_id, miss[0].token_id, "an exact hit replays");
    let replayed = hit[0]
        .logprobs
        .as_ref()
        .expect("an exact hit replays the stored logprobs");
    assert_eq!(replayed.token_id, missed.token_id);
    assert_eq!(replayed.token_logprob, missed.token_logprob);
    assert_eq!(replayed.top, missed.top);

    let off = generate(&model, 0);
    assert_eq!(cache_hits(), 2);
    assert!(off[0].logprobs.is_none(), "logprobs off replays none");

    clear_prompt_cache();
    assert!(generate(&model, 0)[0].logprobs.is_none());
    let hit = generate(&model, 2);
    assert_eq!(cache_hits(), 1);
    let replayed = hit[0]
        .logprobs
        .as_ref()
        .expect("an entry stored with logprobs off still replays them");
    assert_eq!(replayed.top.len(), 2);
    assert_eq!(replayed.token_logprob, missed.token_logprob);

    println!("{CHILD_DONE}");
}

/// Collects the message of every event emitted on the thread it is installed on.
struct MessageLog(std::sync::Arc<std::sync::Mutex<Vec<String>>>);

struct MessageVisitor<'a>(&'a mut Vec<String>);

impl tracing::field::Visit for MessageVisitor<'_> {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        if field.name() == "message" {
            self.0.push(format!("{value:?}"));
        }
    }
}

impl tracing::Subscriber for MessageLog {
    fn enabled(&self, _: &tracing::Metadata<'_>) -> bool {
        true
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, event: &tracing::Event<'_>) {
        let mut log = self.0.lock().unwrap();
        event.record(&mut MessageVisitor(&mut log));
    }
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

/// The messages `f` emits.
fn messages_of(f: impl FnOnce()) -> Vec<String> {
    let log = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    tracing::subscriber::with_default(MessageLog(std::sync::Arc::clone(&log)), f);
    let messages = log.lock().unwrap().clone();
    messages
}

#[test]
fn a_cache_of_no_slots_takes_no_snapshot() {
    run_child("bitnet::first_logprobs_tests::a_cache_of_no_slots_takes_no_snapshot_child");
}

/// With no slots a Miss stores nothing, so it must not build the entry either:
/// the store reaches `push` only when a snapshot was taken, and `push` of a
/// zero-capacity cache says so. With slots the same Miss does store.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn a_cache_of_no_slots_takes_no_snapshot_child() {
    if !started_by_parent() {
        return;
    }
    rmlx_mlx::forbid_gpu().unwrap();
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path());
    let model = super::load_from_path(dir.path(), Device::Cpu).expect("CPU load");

    clear_prompt_cache();
    let stored = messages_of(|| {
        generate(&model, 3);
    });
    assert!(
        stored.iter().any(|m| m.contains("pushed snapshot")),
        "control: a Miss with slots stores a snapshot, got {stored:?}"
    );

    let none = messages_of(|| {
        generate_with_slots(&model, 0, 3);
    });
    assert!(
        !none.iter().any(|m| m.contains("zero slots configured")),
        "a Miss with no slots still built and offered an entry: {none:?}"
    );

    println!("{CHILD_DONE}");
}
