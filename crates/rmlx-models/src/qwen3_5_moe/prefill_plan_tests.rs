//! The forwards a cold Qwen3.5 prefill makes, read at the attention calls, on
//! the attention-only synthetic snapshot (`TinyPack`) served on the CPU.
//!
//! A cold prefill of this architecture takes one plan whatever the request
//! stores: a request with prompt-cache slots, one with none and the
//! speculative prefill cut the same prompt at the same places.
//!
//! The pieces of a run are read by the key rows of its attention calls: a call
//! whose key rows end within the forwarded tokens is a piece of the prefill, of
//! any length, and a decode step is not.

use std::sync::atomic::AtomicBool;

use super::*;
use crate::arch::Architecture;
use crate::decode_loop::decode_loop_tests::tiny_tokenizer;
use crate::decode_loop::{ProbeStep, StepVerdict};
use crate::qwen3_5_moe::generate_greedy;
use crate::qwen3_5_moe::prompt_cache::{ensure_prompt_cache, PROMPT_CACHE};
use crate::qwen3_5_moe::tests::anchor_resume_tests::stored_anchor_pos;
use crate::qwen3_5_moe::tests::attention_calls::{
    assert_the_recorder_is_live, recorded, AttentionCall, ChunkOverride,
};
use crate::qwen3_5_moe::tests::round_stream::RoundStreamRecorder;
use crate::sampler::{Pcg32, PenaltyConfig, SamplerConfig};
use crate::speculative::round_common::{cache_stack, lin_cache_stack};
use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;

/// The prefill chunk the run installs. A prompt of 700 or 769 tokens fits one
/// chunk.
const CHUNK: usize = 1024;
const MAX_SEQ: i32 = 2048;
/// The target of the event a capture of the recurrent state logs.
const CAPTURE_TARGET: &str = "rmlx_models::qwen3_5_moe::anchor_capture";

/// The first `len` ids of one sequence, so a longer prompt extends a shorter.
fn prompt(len: usize) -> Vec<u32> {
    (0..len as u32).map(|i| i % TINY_VOCAB as u32).collect()
}

/// The query rows of every forward over the first `forwarded` tokens: the
/// prefill pieces, in order.
fn pieces(calls: &[AttentionCall], forwarded: usize) -> Vec<i64> {
    calls
        .iter()
        .filter(|c| usize::try_from(c.k_rows).is_ok_and(|k| k <= forwarded))
        .map(|c| c.caller_q_rows)
        .collect()
}

fn cache_hits() -> u64 {
    PROMPT_CACHE.with_inner_mut(|guard| guard.as_ref().map_or(0, |c| c.stats().hits))
}

/// A cache of `slots` slots with nothing in it.
fn empty_cache(slots: usize) {
    ensure_prompt_cache(slots);
    PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
        }
    });
}

/// One greedy generation over a prompt of `len` tokens, from the cache as it
/// stands.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a failed generation aborts the test loudly"
)]
fn generate(model: &Qwen3_5MoeText, slots: usize, len: usize) {
    ensure_prompt_cache(slots);
    let tokenizer = tiny_tokenizer(TINY_VOCAB as u32);
    let sampler = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let mut rng = Pcg32::new(sampler.seed_or_default());
    let mut history = Vec::new();
    let mut step_fn = |_: &ProbeStep| StepVerdict::Continue;
    generate_greedy(
        model,
        &tokenizer,
        &prompt(len),
        2,
        Device::Cpu,
        KvQuant::None,
        Some(MAX_SEQ),
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
    .expect("generate_greedy");
}

/// The prefill pieces of one generation.
fn generate_pieces(model: &Qwen3_5MoeText, slots: usize, len: usize) -> Vec<i64> {
    let ((), calls) = recorded(|| generate(model, slots, len));
    pieces(&calls, len)
}

/// How many times one generation captured the recurrent state.
fn captures(model: &Qwen3_5MoeText, slots: usize, len: usize) -> usize {
    let recorder = RoundStreamRecorder::for_target(CAPTURE_TARGET);
    tracing::subscriber::with_default(std::sync::Arc::clone(&recorder), || {
        generate(model, slots, len);
    });
    recorder
        .events()
        .iter()
        .filter(|e| {
            e.message
                .contains("saved the recurrent state at the anchor")
        })
        .count()
}

/// The pieces of the speculative prefill of a prompt of `len` tokens: all of it
/// but the last token, which the round loop feeds on its own.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a failed prefill aborts the test loudly"
)]
fn speculative_pieces(verifier: &Architecture, len: usize, recurrent: bool) -> Vec<i64> {
    let ids = prompt(len);
    let (_, head) = ids.split_last().expect("a prompt has a last token");
    let mut kv = cache_stack(verifier, KvQuant::None, MAX_SEQ);
    let mut lin = lin_cache_stack(verifier);
    let ((), calls) = recorded(|| {
        crate::speculative::prefill_chunked(
            verifier,
            head,
            len,
            &mut kv,
            recurrent.then_some(lin.as_mut_slice()),
            Device::Cpu,
            &AtomicBool::new(false),
        )
        .expect("prefill_chunked");
    });
    pieces(&calls, head.len())
}

/// The pieces of the verifier prefill that captures hidden rows (DFlash 2,
/// EAGLE-3): the whole prompt, in chunks of its own.
#[allow(
    clippy::expect_used,
    reason = "test fixture: a failed prefill aborts the test loudly"
)]
fn capturing_pieces(verifier: &Architecture, len: usize) -> Vec<i64> {
    let ids = prompt(len);
    let mut kv = cache_stack(verifier, KvQuant::None, MAX_SEQ);
    let mut lin = lin_cache_stack(verifier);
    let ((), calls) = recorded(|| {
        verifier
            .forward_verify_capture_chunked(
                &ids,
                &[0],
                &mut kv,
                Some(&mut lin),
                CHUNK,
                None,
                Device::Cpu,
                &AtomicBool::new(false),
            )
            .expect("forward_verify_capture_chunked");
    });
    pieces(&calls, len)
}

#[test]
fn every_cold_prefill_cuts_the_prompt_where_the_anchor_is() {
    crate::test_child::run_child(
        "qwen3_5_moe::loader::pack_loader_tests::prefill_plan_tests::\
         every_cold_prefill_cuts_the_prompt_where_the_anchor_is_child",
    );
}

/// A request with slots, one with none, the speculative prefill and the
/// capturing prefill forward the same pieces. The prompt of 700 tokens has its
/// anchor inside the chunk. The prompt of 769 tokens has its anchor one token
/// from its end: the cut falls where the speculative head ends, and the
/// prompt's last piece is one token long.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn every_cold_prefill_cuts_the_prompt_where_the_anchor_is_child() -> Result<()> {
    if !crate::test_child::started_by_parent() {
        return Ok(());
    }
    rmlx_mlx::forbid_gpu()?;
    assert_the_recorder_is_live(Device::Cpu);
    let _chunk = ChunkOverride::install(crate::prefill_chunk::set_prefill_chunk, CHUNK);
    let tiny = TinyPack {
        gdn: false,
        ..PLAIN
    };
    let model = tiny.load()?;
    let verifier = Architecture::Qwen3_5Moe(tiny.load()?);

    // (prompt length, plain prefill, speculative head, capturing prefill, anchor)
    let cases: [(usize, [i64; 2], &[i64], &[i64], usize); 2] = [
        (700, [512, 188], &[512, 187], &[512, 188], 512),
        (769, [768, 1], &[768], &[768, 1], 768),
    ];
    for (len, plain, speculative, capturing, anchor) in cases {
        empty_cache(4);
        let with_slots = generate_pieces(&model, 4, len);
        assert_eq!(cache_hits(), 0, "{len}: the run is a miss");
        assert_eq!(with_slots, plain, "{len}: a request that stores an entry");
        assert_eq!(
            stored_anchor_pos(len),
            Some(anchor),
            "{len}: the entry keeps the state at the anchor"
        );

        empty_cache(0);
        let without_slots = generate_pieces(&model, 0, len);
        assert_eq!(without_slots, plain, "{len}: a request that stores none");
        assert_eq!(stored_anchor_pos(len), None, "{len}: no entry is stored");

        assert_eq!(
            speculative_pieces(&verifier, len, true),
            speculative,
            "{len}: the speculative head"
        );
        assert_eq!(
            capturing_pieces(&verifier, len),
            capturing,
            "{len}: the prefill that captures hidden rows"
        );
    }

    assert_eq!(
        speculative_pieces(&verifier, 700, false),
        vec![699],
        "a verifier with no recurrent state is not cut"
    );

    empty_cache(4);
    assert_eq!(
        captures(&model, 4, 700),
        1,
        "control: a stored entry captures"
    );
    empty_cache(0);
    assert_eq!(
        captures(&model, 0, 700),
        0,
        "a request that stores nothing captures nothing"
    );

    let head = prompt(699);
    for wrong in [698, 701] {
        let refused = crate::speculative::prefill_chunked(
            &verifier,
            &head,
            wrong,
            &mut cache_stack(&verifier, KvQuant::None, MAX_SEQ),
            None,
            Device::Cpu,
            &AtomicBool::new(false),
        );
        assert!(
            matches!(&refused, Err(Error::Model(m)) if m.contains("cannot have a head")),
            "a prompt of {wrong} tokens with a head of 699: {refused:?}"
        );
    }

    println!("{}", crate::test_child::CHILD_DONE);
    Ok(())
}

#[test]
fn a_resumed_request_takes_the_state_at_its_anchor() {
    crate::test_child::run_child(
        "qwen3_5_moe::loader::pack_loader_tests::prefill_plan_tests::\
         a_resumed_request_takes_the_state_at_its_anchor_child",
    );
}

/// A request that extends a stored prompt by one token resumes at its end,
/// which is the anchor of the longer prompt. No piece ends there: the state is
/// taken before the first piece. A request that resumes from 300 tokens has its
/// anchor inside the tail, which is split there.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn a_resumed_request_takes_the_state_at_its_anchor_child() -> Result<()> {
    if !crate::test_child::started_by_parent() {
        return Ok(());
    }
    rmlx_mlx::forbid_gpu()?;
    let _chunk = ChunkOverride::install(crate::prefill_chunk::set_prefill_chunk, CHUNK);
    let model = TinyPack {
        gdn: false,
        ..PLAIN
    }
    .load()?;

    empty_cache(4);
    generate(&model, 4, 768);
    assert_eq!(stored_anchor_pos(768), Some(512), "the stored prompt");
    generate(&model, 4, 769);
    assert_eq!(cache_hits(), 1, "the longer prompt resumes from the stored");
    assert_eq!(stored_anchor_pos(769), Some(768), "at the resume point");

    empty_cache(4);
    generate(&model, 4, 300);
    let hits = cache_hits();
    let resumed = generate_pieces(&model, 4, 700);
    assert_eq!(cache_hits(), hits + 1, "the longer prompt resumes from 300");
    assert_eq!(resumed, vec![212, 188], "the tail is split at the anchor");
    assert_eq!(stored_anchor_pos(700), Some(512), "inside the tail");

    println!("{}", crate::test_child::CHILD_DONE);
    Ok(())
}
