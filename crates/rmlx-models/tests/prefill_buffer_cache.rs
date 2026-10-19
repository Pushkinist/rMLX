//! The buffer-cache decision of a long chunked prefill.
//!
//! Each chunked prefill loop calls `release_chunk_buffers` after the state of
//! a chunk that is not the last one is evaluated. The call emits one event
//! with the live bytes and the cached bytes it read, the cached bytes it read
//! after its decision, and whether it released the cache; it trims the cache
//! to the live memory when the cache holds more than the live memory. A case
//! prefills a prompt of [`CHUNKS`] chunks from an empty cache and checks the
//! events of the request: there are exactly the expected number of them (so a
//! loop that stops calling the function is red), and each one released if and
//! only if its cache was larger than its live memory.
//!
//! The short cases (`*_reaches_every_chunk_boundary`) prove only that and the
//! release rule. The two long cases (`*_long_fresh_prefill_bounds_the_buffer_cache`)
//! prefill [`LONG_CHUNKS`] chunks, enough for the cache to pass the live
//! memory. They check that at least one release is followed by a later chunk
//! boundary, and that after every release the cache holds at most the live
//! memory.
//!
//! This does not bound the footprint on every model: a KV ring that grows
//! leaves its old ring in the cache beside the transient buffers of the chunk,
//! so no per-event bound on the live peak holds. The end-to-end check is the
//! long-prefix case in `crates/rmlx-server/tests/gone_request.rs`.
//!
//! Snapshots resolve from `RMLX_O_MODELS_ROOT` by slug (see
//! `tests/common/mod.rs`).
//!
//! Run:
//! cargo test -p rmlx-models --test prefill_buffer_cache -- --ignored --test-threads=1

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

mod common;

use std::sync::atomic::AtomicBool;

use rmlx_mlx::Device;
use rmlx_models::prefill_chunk::prefill_pieces;
use rmlx_models::prompt_cache::anchor_target;
use rmlx_models::speculative::eagle3::{
    eagle3_generate, Eagle3Drafter, DRAFTER_PREFILL_CHUNK, VERIFIER_PREFILL_CHUNK,
};
use rmlx_models::speculative::mtp::{mtp_generate, MtpDrafter};
use rmlx_models::speculative::VerifierTrunk;
use rmlx_models::{arch, Pcg32, PenaltyConfig, SamplerConfig};

const GEMMA4_E2B: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__gemma-4-e2b-it-mxfp8",
    archs: &["Gemma4ForConditionalGeneration"],
};

const BONSAI_8B: common::GoldenModel = common::GoldenModel {
    slug: "prism-ml__Ternary-Bonsai-8B-mlx-2bit",
    archs: &["Qwen3ForCausalLM"],
};

const READER_LM: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__jinaai-ReaderLM-v2",
    archs: &["Qwen2ForCausalLM"],
};

const BITNET: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__bitnet-b1.58-2B-4T",
    archs: &["BitNetForCausalLM"],
};

const QWEN38_VERIFIER: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__Qwen3.8-27B-mxfp8",
    archs: &["Qwen3_5ForConditionalGeneration"],
};

const QWEN38_MTP_SLUG: &str = "mlx-community__Qwen3.8-27B-MTP-mxfp8";

const QWEN36_VERIFIER: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__Qwen3.6-35B-A3B-8bit",
    archs: &["Qwen3_5MoeForConditionalGeneration"],
};

const QWEN36_EAGLE3_SLUG: &str = "Dogacel__specdrift-qwen3.6-35b-a3b-eagle3";

/// Draft-model override, for a models root that does not hold a drafter slug.
const DRAFT_VAR: &str = "RMLX_DRAFT_TEST_MODEL";

/// Chunks of the measured prefill.
const CHUNKS: usize = 16;

/// Chunks of a long prefill: 32 768 tokens at a chunk of 1024, where the cache
/// of an unreleased prefill is larger than the live memory on gemma-4-e2b and
/// on Ternary-Bonsai-8B.
const LONG_CHUNKS: usize = 32;

/// Tokens of the prompt that a prefix-tail case caches first: one prompt-cache
/// block, shorter than the sliding window of gemma-4-e2b.
const PREFIX_TOKENS: usize = 256;

const DECODE_TOKENS: usize = 2;

/// Tokens a speculative case asks for.
const SPEC_TOKENS: usize = 8;

/// The message of the event that decides on the cache after a chunk.
const BOUNDARY_EVENT: &str = "chunk boundary, MLX buffer cache";

/// The module that emits [`BOUNDARY_EVENT`]. The recorder accepts this target
/// and no other, so the capture changes no other callsite.
const BOUNDARY_TARGET: &str = "rmlx_models::decode_loop";

/// A prompt of at least `len` tokens. `case` leads the text, so the first
/// prompt-cache block differs between cases and no case finds the entry of
/// another.
fn prompt_ids(tokenizer: &tokenizers::Tokenizer, case: &str, len: usize) -> Vec<u32> {
    const SENTENCE: &str = " The quick brown fox jumps over the lazy dog near the river bank.";
    let mut repeats = 512_usize;
    loop {
        let text = format!("Case {case}.{}", SENTENCE.repeat(repeats));
        let ids = tokenizer.encode(text, true).expect("tokenize");
        if ids.get_ids().len() >= len {
            return ids.get_ids()[..len].to_vec();
        }
        repeats *= 2;
    }
}

fn ctx_for(prompt_len: usize, tokens: usize) -> i32 {
    i32::try_from(prompt_len + tokens + 64).expect("context fits i32")
}

fn sampler() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    }
}

fn eos_ids(model_path: &std::path::Path) -> Vec<u32> {
    let raw = std::fs::read(model_path.join("config.json")).expect("config.json");
    let cfg: serde_json::Value = serde_json::from_slice(&raw).expect("config.json is JSON");
    match cfg.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => Vec::new(),
    }
}

fn tokenizer_of(model_path: &std::path::Path) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer")
}

fn load(model_path: &std::path::Path) -> arch::Architecture {
    arch::load_model(model_path, Device::Gpu, &arch::LoadOpts::default()).expect("load model")
}

fn generate(
    model: &arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    ids: &[u32],
    max_ctx: i32,
) -> rmlx_core::error::Result<()> {
    let sampler_cfg = sampler();
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();
    let steps = model.generate_greedy(
        tokenizer,
        ids,
        DECODE_TOKENS,
        Device::Gpu,
        None,
        Some(max_ctx),
        1,
        &[],
        &mut |_| rmlx_models::StepVerdict::Continue,
        None,
        &sampler_cfg,
        &mut rng,
        &penalty_cfg,
        &mut token_history,
        &AtomicBool::new(false),
    )?;
    assert!(!steps.is_empty(), "generate_greedy returned no token");
    Ok(())
}

/// Run `run` from an empty cache and return the chunk-boundary events it
/// emitted.
fn boundary_events(run: impl FnOnce()) -> Vec<common::round_stream::CapturedEvent> {
    common::release_cached_buffers();
    let recorder = common::round_stream::RoundStreamRecorder::for_target(BOUNDARY_TARGET);
    tracing::subscriber::with_default(std::sync::Arc::clone(&recorder), run);
    recorder
        .events()
        .into_iter()
        .filter(|e| e.message.contains(BOUNDARY_EVENT))
        .collect()
}

fn event_bytes(event: &common::round_stream::CapturedEvent, name: &str) -> u64 {
    event
        .field(name)
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("a boundary event carries no numeric `{name}`: {event:?}"))
}

fn event_released(event: &common::round_stream::CapturedEvent) -> bool {
    event
        .field("released")
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| panic!("a boundary event carries no boolean `released`: {event:?}"))
}

/// The chunk boundaries a prefill of `head` tokens makes when the full prompt has
/// `prompt_len` tokens: the plan the engine cuts the prompt by.
///
/// The engine cuts at the prompt-cache anchor only when the model carries
/// recurrent state (`has_recurrent_state`); without it the plan is the chunk
/// edges alone.
fn planned_boundaries(
    head: usize,
    prompt_len: usize,
    chunk: usize,
    has_recurrent_state: bool,
) -> usize {
    let cut = if has_recurrent_state {
        anchor_target(prompt_len, 0)
    } else {
        None
    };
    prefill_pieces(0, head, chunk, cut).len() - 1
}

/// Hold the boundary events of a request to the rules of this file.
///
/// Each release must leave the cache at or below the live memory it read. With
/// `require_release`, the request must also have released the cache at least
/// once, and a chunk boundary must follow one of the releases. The events are
/// in order only for a request with one prefill loop.
fn assert_boundary_events(
    test: &str,
    expected: usize,
    require_release: bool,
    events: &[common::round_stream::CapturedEvent],
) {
    assert_eq!(
        events.len(),
        expected,
        "{test}: the request reached {} chunk boundaries; the loop under test reaches {expected}",
        events.len()
    );
    for event in events {
        let chunk_idx = event.field("chunk_idx").unwrap_or("?");
        let over_bound = event_bytes(event, "cached_bytes") > event_bytes(event, "active_bytes");
        assert_eq!(
            event_released(event),
            over_bound,
            "{test}: after chunk {chunk_idx} the cache is {event:?}; the cache is released if \
             and only if it holds more than the live memory"
        );
        if over_bound {
            // A buffer freed between the two after-reads is counted once or in
            // neither, so the sum is at or below the live plus cached total.
            let live_and_cached_after = event_bytes(event, "active_after_bytes")
                .checked_add(event_bytes(event, "cached_after_bytes"));
            let bound = event_bytes(event, "active_bytes").checked_mul(2);
            assert!(
                matches!((live_and_cached_after, bound), (Some(sum), Some(bound)) if sum <= bound),
                "{test}: after chunk {chunk_idx} the release left live plus cached bytes above \
                 twice the live bytes before it: {event:?}"
            );
        }
    }
    if !require_release {
        return;
    }
    assert!(
        events.iter().any(event_released),
        "{test}: the cache never passed the live memory in {expected} chunk boundaries, so the \
         release is not exercised"
    );
    let a_release_has_a_later_boundary = events.windows(2).any(|pair| event_released(&pair[0]));
    assert!(
        a_release_has_a_later_boundary,
        "{test}: no release was followed by a later chunk boundary, so the prefill after a \
         release was not exercised"
    );
}

/// A plain `generate_greedy` prefill of `chunks` chunks of `chunk_key`'s
/// prefill chunk.
///
/// With `cached_prefix`, a first request caches the first [`PREFIX_TOKENS`]
/// tokens, and the measured request extends that prompt, so the prefix-tail
/// loop prefills the rest. The first request has the small ceiling its own
/// prompt needs. The measured request has a ceiling far above it, so the
/// restored cache must take the ceiling of the request that uses it.
fn assert_plain_prefill_releases(
    test: &str,
    model_spec: &common::GoldenModel,
    chunk_key: &str,
    cached_prefix: bool,
    chunks: usize,
    require_release: bool,
) {
    let Some(model_path) = common::model_for(model_spec, test) else {
        return;
    };
    let model = load(&model_path);
    let tokenizer = tokenizer_of(&model_path);

    let chunk = rmlx_models::prefill_chunk::prefill_chunk_for(chunk_key);
    let measured_len = if cached_prefix {
        PREFIX_TOKENS + chunks * chunk
    } else {
        chunks * chunk
    };
    let ids = prompt_ids(&tokenizer, test, measured_len);
    let max_ctx = ctx_for(measured_len, DECODE_TOKENS);

    if cached_prefix {
        let prefix_ctx = ctx_for(PREFIX_TOKENS, DECODE_TOKENS);
        let first_tail_chunk_need = PREFIX_TOKENS + chunk;
        assert!(
            usize::try_from(prefix_ctx).is_ok_and(|ceiling| ceiling < first_tail_chunk_need),
            "{test}: the stored ceiling {prefix_ctx} must be below {first_tail_chunk_need}, the \
             first tail chunk's need, or a reused cache that keeps it never fails and the test \
             cannot catch the defect"
        );
        generate(&model, &tokenizer, &ids[..PREFIX_TOKENS], prefix_ctx)
            .expect("the request that stores the prefix");
    }
    let events = boundary_events(|| {
        let outcome = generate(&model, &tokenizer, &ids, max_ctx);
        assert!(
            outcome.is_ok(),
            "{test}: the request under ceiling {max_ctx} failed, as when a reused cache keeps the \
             ceiling of the request that stored it: {outcome:?}"
        );
    });
    assert_boundary_events(test, chunks - 1, require_release, &events);
}

#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
#[test]
fn gemma4_e2b_fresh_prefill_reaches_every_chunk_boundary() {
    assert_plain_prefill_releases(
        "gemma4_e2b_fresh_prefill_reaches_every_chunk_boundary",
        &GEMMA4_E2B,
        "gemma4",
        false,
        CHUNKS,
        false,
    );
}

#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
#[test]
fn gemma4_e2b_prefix_tail_reaches_every_chunk_boundary() {
    assert_plain_prefill_releases(
        "gemma4_e2b_prefix_tail_reaches_every_chunk_boundary",
        &GEMMA4_E2B,
        "gemma4",
        true,
        CHUNKS,
        false,
    );
}

#[ignore = "requires Metal GPU and the Ternary-Bonsai-8B snapshot; run with `make gpu-test`"]
#[test]
fn bonsai_8b_fresh_prefill_reaches_every_chunk_boundary() {
    assert_plain_prefill_releases(
        "bonsai_8b_fresh_prefill_reaches_every_chunk_boundary",
        &BONSAI_8B,
        "qwen3",
        false,
        CHUNKS,
        false,
    );
}

#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
#[test]
fn gemma4_e2b_long_fresh_prefill_bounds_the_buffer_cache() {
    assert_plain_prefill_releases(
        "gemma4_e2b_long_fresh_prefill_bounds_the_buffer_cache",
        &GEMMA4_E2B,
        "gemma4",
        false,
        LONG_CHUNKS,
        true,
    );
}

#[ignore = "requires Metal GPU and the Ternary-Bonsai-8B snapshot; run with `make gpu-test`"]
#[test]
fn bonsai_8b_long_fresh_prefill_bounds_the_buffer_cache() {
    assert_plain_prefill_releases(
        "bonsai_8b_long_fresh_prefill_bounds_the_buffer_cache",
        &BONSAI_8B,
        "qwen3",
        false,
        LONG_CHUNKS,
        true,
    );
}

#[ignore = "requires Metal GPU and the ReaderLM-v2 snapshot; run with `make gpu-test`"]
#[test]
fn reader_lm_v2_fresh_prefill_reaches_every_chunk_boundary() {
    assert_plain_prefill_releases(
        "reader_lm_v2_fresh_prefill_reaches_every_chunk_boundary",
        &READER_LM,
        "qwen2",
        false,
        CHUNKS,
        false,
    );
}

#[ignore = "requires Metal GPU and the BitNet b1.58 2B snapshot; run with `make gpu-test`"]
#[test]
fn bitnet_2b_fresh_prefill_reaches_every_chunk_boundary() {
    assert_plain_prefill_releases(
        "bitnet_2b_fresh_prefill_reaches_every_chunk_boundary",
        &BITNET,
        "bitnet",
        false,
        CHUNKS,
        false,
    );
}

/// The MTP round loop prefills the prompt less its last token through
/// `prefill_chunked`.
#[ignore = "requires Metal GPU and the Qwen3.8-27B and Qwen3.8-27B-MTP snapshots; run with `make gpu-test`"]
#[test]
fn qwen38_mtp_prefill_reaches_every_chunk_boundary() {
    let test = "qwen38_mtp_prefill_reaches_every_chunk_boundary";
    let Some(model_path) = common::model_for(&QWEN38_VERIFIER, test) else {
        return;
    };
    let Some(draft_path) = common::apply(
        common::slug_or_override(DRAFT_VAR, QWEN38_MTP_SLUG, common::Role::Sidecar),
        test,
    ) else {
        return;
    };
    let verifier = load(&model_path);
    let mut drafter = MtpDrafter::load(&draft_path, VerifierTrunk::of(&verifier), Device::Gpu)
        .expect("load sidecar");
    let tokenizer = tokenizer_of(&model_path);
    let eos = eos_ids(&model_path);

    let chunk = rmlx_models::prefill_chunk::prefill_chunk_for("qwen3_5_moe");
    let prompt_len = CHUNKS * chunk + 1;
    let ids = prompt_ids(&tokenizer, test, prompt_len);
    let max_ctx = ctx_for(prompt_len, SPEC_TOKENS);

    let events = boundary_events(|| {
        mtp_generate(
            &verifier,
            &mut drafter,
            &tokenizer,
            &ids,
            SPEC_TOKENS,
            5,
            None,
            Some(max_ctx),
            &eos,
            &mut |_| rmlx_models::StepVerdict::Continue,
            &sampler(),
            Device::Gpu,
            &AtomicBool::new(false),
        )
        .expect("mtp generate");
    });
    assert_boundary_events(
        test,
        planned_boundaries(prompt_len - 1, prompt_len, chunk, true),
        false,
        &events,
    );
}

/// The EAGLE-3 round loop prefills the verifier through
/// `forward_verify_capture_chunked` and then its own drafter in chunks.
#[ignore = "requires Metal GPU and the Qwen3.6-35B-A3B and EAGLE-3 drafter snapshots; run with `make gpu-test`"]
#[test]
fn qwen36_eagle3_prefill_reaches_every_chunk_boundary() {
    let test = "qwen36_eagle3_prefill_reaches_every_chunk_boundary";
    let Some(model_path) = common::model_for(&QWEN36_VERIFIER, test) else {
        return;
    };
    let Some(draft_path) = common::apply(
        common::slug_or_override(DRAFT_VAR, QWEN36_EAGLE3_SLUG, common::Role::Sidecar),
        test,
    ) else {
        return;
    };
    let verifier = load(&model_path);
    let eos = eos_ids(&model_path);
    let mut drafter =
        Eagle3Drafter::load(&draft_path, &verifier, &eos, Device::Gpu).expect("load drafter");
    let tokenizer = tokenizer_of(&model_path);

    let prompt_len = CHUNKS * VERIFIER_PREFILL_CHUNK;
    let ids = prompt_ids(&tokenizer, test, prompt_len);
    let max_ctx = ctx_for(prompt_len, SPEC_TOKENS);

    let events = boundary_events(|| {
        let mut decided_by = Vec::new();
        eagle3_generate(
            &verifier,
            &mut drafter,
            &tokenizer,
            &ids,
            SPEC_TOKENS,
            5,
            None,
            Some(max_ctx),
            &eos,
            &mut |_| rmlx_models::StepVerdict::Continue,
            &mut decided_by,
            &sampler(),
            Device::Gpu,
            &AtomicBool::new(false),
        )
        .expect("eagle3 generate");
    });
    // Qwen3.6-35B-A3B carries recurrent state, so the plan cuts at the anchor
    // (16128 of 16384 tokens) inside the last chunk: 15 chunk edges and the cut.
    let verifier_boundaries =
        planned_boundaries(prompt_len, prompt_len, VERIFIER_PREFILL_CHUNK, true);
    assert_eq!(
        verifier_boundaries, 16,
        "{test}: the plan for {prompt_len} tokens at chunk {VERIFIER_PREFILL_CHUNK} moved"
    );
    let drafter_boundaries = prompt_len.div_ceil(DRAFTER_PREFILL_CHUNK) - 1;
    assert_boundary_events(
        test,
        verifier_boundaries + drafter_boundaries,
        false,
        &events,
    );
}
