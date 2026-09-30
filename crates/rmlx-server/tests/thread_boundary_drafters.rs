//! Thread-boundary tests for the speculative drafters, through
//! `SpeculativeGenerator`: a verifier and drafter loaded on one thread serve a
//! request on another.
//!
//! Greedy verification emits the verifier's own argmax whatever the drafter
//! proposed, so equal tokens do not show that the drafter ran. Each cell also
//! reads the round events the serving thread emitted and requires at least one
//! round that drafted a token.
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
use thread_boundary::round_stream::{round_events, RoundStreamRecorder};
use thread_boundary::{
    assert_crossed, assert_oracle, serve, snapshot, speculative_generator, text_request, Split,
};

const GEMMA4_E2B: (&str, &str) = (
    "mlx-community__gemma-4-e2b-it-mxfp8",
    "Gemma4ForConditionalGeneration",
);
const GEMMA4_E2B_ASSISTANT: (&str, &str) = (
    "mlx-community__gemma-4-E2B-it-assistant-bf16",
    "Gemma4AssistantForCausalLM",
);
const QWEN36_MOE: (&str, &str) = (
    "mlx-community__Qwen3.6-35B-A3B-8bit",
    "Qwen3_5MoeForConditionalGeneration",
);
const QWEN36_DFLASH: (&str, &str) = ("z-lab__Qwen3.6-35B-A3B-DFlash", "DFlashDraftModel");
const QWEN36_EAGLE3: (&str, &str) = (
    "Dogacel__specdrift-qwen3.6-35b-a3b-eagle3",
    "LlamaForCausalLMEagle3",
);
const QWEN38_4BIT: (&str, &str) = (
    "mlx-community__Qwen3.8-27B-4bit",
    "Qwen3_5ForConditionalGeneration",
);
const QWEN38_MTP_4BIT: (&str, &str) = ("mlx-community__Qwen3.8-27B-MTP-4bit", "qwen3_5_mtp");
const QWEN38_DFLASH2: (&str, &str) = ("z-lab__Qwen3.8-27B-DFlash2", "DFlash2DraftModel");
const QWEN38_MXFP8: (&str, &str) = (
    "mlx-community__Qwen3.8-27B-mxfp8",
    "Qwen3_5ForConditionalGeneration",
);
const ORNITH_9B: (&str, &str) = (
    "sahilchachra__ornith-1.0-9b-mxfp8-mlx",
    "Qwen3_5ForConditionalGeneration",
);

/// Tokens drafted over the round events `recorder` captured.
fn drafted(recorder: &RoundStreamRecorder) -> u64 {
    round_events(&recorder.events())
        .iter()
        .map(|e| {
            e.field("num_draft")
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(0)
        })
        .sum()
}

fn assert_drafter(test: &str, verifier: (&str, &str), drafter: (&str, &str), device: Device) {
    let Some(v) = snapshot(test, verifier.0, verifier.1) else {
        return;
    };
    let Some(d) = snapshot(test, drafter.0, drafter.1) else {
        return;
    };
    let load = speculative_generator(v.clone(), d, device);

    let oracle_rounds = RoundStreamRecorder::new();
    let oracle = serve(
        Split::None,
        &load,
        vec![text_request(&v)],
        Some(&oracle_rounds),
    );
    assert_oracle(test, &oracle);
    assert!(
        drafted(&oracle_rounds) > 0,
        "{test}: the one-thread oracle drafted no token, so this cell cannot tell a run \
         that drafts from one that does not"
    );

    let crossed_rounds = RoundStreamRecorder::new();
    let crossed = serve(
        Split::AfterLoad,
        &load,
        vec![text_request(&v)],
        Some(&crossed_rounds),
    );
    assert_crossed(test, &crossed, &oracle);
    assert!(
        drafted(&crossed_rounds) > 0,
        "{test}: the thread that did not load the drafter ran no draft round"
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b + assistant snapshots; run with `make gpu-test`"]
fn gemma4_mtp_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "gemma4_mtp_drafter_runs_on_a_thread_that_did_not_load_it",
        GEMMA4_E2B,
        GEMMA4_E2B_ASSISTANT,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B 4-bit + MTP snapshots; run with `make gpu-test`"]
fn qwen_mtp_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "qwen_mtp_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN38_4BIT,
        QWEN38_MTP_4BIT,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.6-35B + DFlash snapshots; run with `make gpu-test`"]
fn dflash_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "dflash_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN36_MOE,
        QWEN36_DFLASH,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B 4-bit + DFlash2 snapshots; run with `make gpu-test`"]
fn dflash2_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "dflash2_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN38_4BIT,
        QWEN38_DFLASH2,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.6-35B + EAGLE-3 snapshots; run with `make gpu-test`"]
fn eagle3_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "eagle3_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN36_MOE,
        QWEN36_EAGLE3,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3.8-27B mxfp8 + ornith-9b snapshots; run with `make gpu-test`"]
fn two_model_drafter_runs_on_a_thread_that_did_not_load_it() {
    assert_drafter(
        "two_model_drafter_runs_on_a_thread_that_did_not_load_it",
        QWEN38_MXFP8,
        ORNITH_9B,
        Device::Gpu,
    );
}
