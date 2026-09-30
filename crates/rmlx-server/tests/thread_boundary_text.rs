//! Thread-boundary tests for the text path: a model `arch::load_model`
//! returned on one thread decodes on another, and a prompt-cache entry one
//! request wrote is hit by the next request on another thread.
//!
//! The seam, the observable, what cannot move and the mutation list are in the
//! module docs of `thread_boundary/mod.rs`.
//!
//! Run: `make gpu-test CRATE=rmlx-server FILTER=thread`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr
)]

mod thread_boundary;

use rmlx_mlx::Device;
use thread_boundary::{chat_prompt, greedy, hand_over, load_model, long_prompt, snapshot, Split};

const BONSAI_8B: (&str, &str) = ("prism-ml__Ternary-Bonsai-8B-mlx-2bit", "Qwen3ForCausalLM");
const TERNARY_BONSAI_27B: (&str, &str) = (
    "prism-ml__Ternary-Bonsai-27B-mlx-2bit",
    "Qwen3_5ForConditionalGeneration",
);
const READERLM_V2: (&str, &str) = ("mlx-community__jinaai-ReaderLM-v2", "Qwen2ForCausalLM");
const BITNET: (&str, &str) = ("mlx-community__bitnet-b1.58-2B-4T", "BitNetForCausalLM");
const GEMMA4_E2B: (&str, &str) = (
    "mlx-community__gemma-4-e2b-it-mxfp8",
    "Gemma4ForConditionalGeneration",
);
const QWEN36_MOE: (&str, &str) = (
    "mlx-community__Qwen3.6-35B-A3B-8bit",
    "Qwen3_5MoeForConditionalGeneration",
);

// ── A loaded model decodes on another thread ────────────────────────────────

fn assert_text(test: &str, (slug, declares): (&str, &str), device: Device) {
    let Some(dir) = snapshot(test, slug, declares) else {
        return;
    };
    let run = |split: Split| {
        let (d1, d2) = (dir.clone(), dir.clone());
        hand_over(
            split,
            move || (load_model(&d1, device), ()),
            move |m| greedy(&m, &d2, &chat_prompt(&d2), 0, device),
        )
        .1
    };
    let oracle = run(Split::None).expect("the one-thread oracle must decode");
    let crossed = run(Split::AfterLoad).unwrap_or_else(|e| {
        panic!("{test}: decode failed on a thread that did not load the model: {e}")
    });
    assert_eq!(
        crossed, oracle,
        "{test}: different tokens than the one-thread oracle"
    );
}

/// Bonsai-8B ships F16 scales, biases and norms, and a YARN scale. Its loader
/// also builds the fused QKV `concatenate`.
#[test]
#[ignore = "requires Metal GPU and the Bonsai-8B snapshot; run with `make gpu-test`"]
fn qwen3_text_decodes_on_a_thread_that_did_not_load_the_model() {
    assert_text(
        "qwen3_text_decodes_on_a_thread_that_did_not_load_the_model",
        BONSAI_8B,
        Device::Gpu,
    );
}

/// A dense Qwen3.5 checkpoint with F16 scales, biases and norms.
#[test]
#[ignore = "requires Metal GPU and the Ternary-Bonsai-27B snapshot; run with `make gpu-test`"]
fn qwen3_5_text_decodes_on_a_thread_that_did_not_load_the_model() {
    assert_text(
        "qwen3_5_text_decodes_on_a_thread_that_did_not_load_the_model",
        TERNARY_BONSAI_27B,
        Device::Gpu,
    );
}

/// A Qwen2 checkpoint with F16 scales, biases and norms.
#[test]
#[ignore = "requires Metal GPU and the ReaderLM-v2 snapshot; run with `make gpu-test`"]
fn qwen2_text_decodes_on_a_thread_that_did_not_load_the_model() {
    assert_text(
        "qwen2_text_decodes_on_a_thread_that_did_not_load_the_model",
        READERLM_V2,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the BitNet snapshot; run with `make gpu-test`"]
fn bitnet_text_decodes_on_a_thread_that_did_not_load_the_model() {
    assert_text(
        "bitnet_text_decodes_on_a_thread_that_did_not_load_the_model",
        BITNET,
        Device::Gpu,
    );
}

// ── A prompt-cache entry is hit on another thread ───────────────────────────

/// The first request loads the model, clears the prompt cache and writes one
/// entry. The second request, on another thread, must hit it.
fn assert_prompt_cache(test: &str, (slug, declares): (&str, &str), device: Device) {
    let Some(dir) = snapshot(test, slug, declares) else {
        return;
    };
    let run = |split: Split| {
        let (d1, d2) = (dir.clone(), dir.clone());
        let first = move || {
            let model = load_model(&d1, device);
            model.clear_prompt_cache();
            let ids = greedy(&model, &d1, &long_prompt(&d1), 1, device).expect("first request");
            let hits = model.cache_stats().map_or(0, |s| s.hits);
            (model, (ids, hits))
        };
        let second = move |model: rmlx_models::arch::Architecture| {
            let ids = greedy(&model, &d2, &long_prompt(&d2), 1, device);
            (ids, model.cache_stats().map_or(0, |s| s.hits))
        };
        hand_over(split, first, second)
    };
    let ((oracle_first, _), (oracle_second, oracle_hits)) = run(Split::None);
    let oracle_second = oracle_second.expect("the one-thread oracle must decode");
    assert!(
        oracle_hits >= 1,
        "{test}: the oracle's second request did not hit the prompt cache"
    );
    let ((crossed_first, first_hits), (crossed_second, hits)) = run(Split::BeforeLastRequest);
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
    assert_eq!(
        crossed_first, oracle_first,
        "{test}: first request differs from the oracle"
    );
    assert_eq!(
        crossed_second, oracle_second,
        "{test}: second request differs from the oracle"
    );
}

#[test]
#[ignore = "requires Metal GPU and the Bonsai-8B snapshot; run with `make gpu-test`"]
fn a_prompt_cache_hit_decodes_on_another_thread() {
    assert_prompt_cache(
        "a_prompt_cache_hit_decodes_on_another_thread",
        BONSAI_8B,
        Device::Gpu,
    );
}

/// Gemma4 keeps its sliding-window layers in a rotating ring, which the store
/// clones by snapshot and restore.
#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn a_gemma4_prompt_cache_hit_decodes_on_another_thread() {
    assert_prompt_cache(
        "a_gemma4_prompt_cache_hit_decodes_on_another_thread",
        GEMMA4_E2B,
        Device::Gpu,
    );
}

/// The Qwen3.5 MoE entry also carries the recurrent state of its linear
/// attention layers.
#[test]
#[ignore = "requires Metal GPU and the Qwen3.6-35B snapshot; run with `make gpu-test`"]
fn a_qwen3_5_moe_prompt_cache_hit_decodes_on_another_thread() {
    assert_prompt_cache(
        "a_qwen3_5_moe_prompt_cache_hit_decodes_on_another_thread",
        QWEN36_MOE,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the ReaderLM-v2 snapshot; run with `make gpu-test`"]
fn a_qwen2_prompt_cache_hit_decodes_on_another_thread() {
    assert_prompt_cache(
        "a_qwen2_prompt_cache_hit_decodes_on_another_thread",
        READERLM_V2,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the BitNet snapshot; run with `make gpu-test`"]
fn a_bitnet_prompt_cache_hit_decodes_on_another_thread() {
    assert_prompt_cache(
        "a_bitnet_prompt_cache_hit_decodes_on_another_thread",
        BITNET,
        Device::Gpu,
    );
}
