//! Thread-boundary tests for the vision and audio towers and the encoder-output
//! cache, through `ArchGenerator`: a generator loaded on one thread serves an
//! image or audio request on another, and an encoder-cache entry one request
//! wrote is hit by the next request on another thread.
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

use std::path::Path;
use std::sync::Arc;

use rmlx_mlx::Device;
use rmlx_models::multimodal_cache::MultimodalCache;
use rmlx_server::GenerationRequest;
use thread_boundary::{
    arch_generator, assert_hand_over, audio_request, image_request, serve, snapshot, spoken_wav,
    Split,
};

const GEMMA4_E2B: (&str, &str) = (
    "mlx-community__gemma-4-e2b-it-mxfp8",
    "Gemma4ForConditionalGeneration",
);
const GEMMA4_UNIFIED_12B: (&str, &str) = (
    "mlx-community__gemma-4-12B-it-mxfp8",
    "Gemma4UnifiedForConditionalGeneration",
);
const MEDGEMMA: (&str, &str) = (
    "mlx-community__medgemma-1.5-4b-it-8bit",
    "Gemma3ForConditionalGeneration",
);
const QWEN3_VL: (&str, &str) = (
    "mlx-community__Qwen3-VL-30B-A3B-Instruct-4bit",
    "Qwen3VLMoeForConditionalGeneration",
);

// ── A tower loaded with the generator runs on another thread ────────────────

fn assert_tower(
    test: &str,
    (slug, declares): (&str, &str),
    device: Device,
    make: impl Fn(&Path) -> GenerationRequest,
) {
    let Some(dir) = snapshot(test, slug, declares) else {
        return;
    };
    let load = arch_generator(dir.clone(), device, None);
    assert_hand_over(test, Split::AfterLoad, &load, || vec![make(&dir)]);
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn gemma4_vision_tower_runs_on_a_thread_that_did_not_load_it() {
    assert_tower(
        "gemma4_vision_tower_runs_on_a_thread_that_did_not_load_it",
        GEMMA4_E2B,
        Device::Gpu,
        image_request,
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-12B snapshot; run with `make gpu-test`"]
fn gemma4_unified_vision_embedder_runs_on_a_thread_that_did_not_load_it() {
    assert_tower(
        "gemma4_unified_vision_embedder_runs_on_a_thread_that_did_not_load_it",
        GEMMA4_UNIFIED_12B,
        Device::Gpu,
        image_request,
    );
}

#[test]
#[ignore = "requires Metal GPU and the medgemma snapshot; run with `make gpu-test`"]
fn gemma3_vision_tower_runs_on_a_thread_that_did_not_load_it() {
    assert_tower(
        "gemma3_vision_tower_runs_on_a_thread_that_did_not_load_it",
        MEDGEMMA,
        Device::Gpu,
        image_request,
    );
}

#[test]
#[ignore = "requires Metal GPU and the Qwen3-VL snapshot; run with `make gpu-test`"]
fn qwen3_vl_vision_tower_runs_on_a_thread_that_did_not_load_it() {
    assert_tower(
        "qwen3_vl_vision_tower_runs_on_a_thread_that_did_not_load_it",
        QWEN3_VL,
        Device::Gpu,
        image_request,
    );
}

/// The spoken clip is made after the snapshot check, so a host without the
/// snapshot stands down with a named `SKIP` before it needs the system voice.
fn assert_audio_tower(test: &str, snapshot_of: (&str, &str), device: Device) {
    let Some(dir) = snapshot(test, snapshot_of.0, snapshot_of.1) else {
        return;
    };
    let wav = spoken_wav();
    let load = arch_generator(dir.clone(), device, None);
    assert_hand_over(test, Split::AfterLoad, &load, || {
        vec![audio_request(&dir, &wav)]
    });
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn gemma4_audio_tower_runs_on_a_thread_that_did_not_load_it() {
    assert_audio_tower(
        "gemma4_audio_tower_runs_on_a_thread_that_did_not_load_it",
        GEMMA4_E2B,
        Device::Gpu,
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-12B snapshot; run with `make gpu-test`"]
fn gemma4_unified_audio_embedder_runs_on_a_thread_that_did_not_load_it() {
    assert_audio_tower(
        "gemma4_unified_audio_embedder_runs_on_a_thread_that_did_not_load_it",
        GEMMA4_UNIFIED_12B,
        Device::Gpu,
    );
}

// ── An encoder-cache entry is hit on another thread ─────────────────────────

/// The first image request writes the entry on one thread, the second request
/// hits it on another.
#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn an_encoder_cache_hit_decodes_on_another_thread() {
    let test = "an_encoder_cache_hit_decodes_on_another_thread";
    let Some(dir) = snapshot(test, GEMMA4_E2B.0, GEMMA4_E2B.1) else {
        return;
    };
    let run = |split: Split| {
        let cache = Arc::new(MultimodalCache::new(256 << 20));
        let load = arch_generator(dir.clone(), Device::Gpu, Some(Arc::clone(&cache)));
        let out = serve(
            split,
            &load,
            vec![image_request(&dir), image_request(&dir)],
            None,
        );
        (out, cache.stats().hits)
    };
    let (oracle, oracle_hits) = run(Split::None);
    assert!(
        oracle
            .iter()
            .all(|r| r.as_ref().is_ok_and(|ids| !ids.is_empty())),
        "{test}: the one-thread oracle must decode: {oracle:?}"
    );
    let (crossed, hits) = run(Split::BeforeLastRequest);
    let second = crossed[1].as_ref().unwrap_or_else(|e| {
        panic!("{test}: an encoder-cache hit failed on a thread that did not write it: {e}")
    });
    assert_eq!(
        hits, 1,
        "{test}: the second request must hit the encoder cache once"
    );
    assert_eq!(
        oracle_hits, 1,
        "{test}: the oracle's second request must hit once"
    );
    assert_eq!(
        Some(second),
        oracle[1].as_ref().ok(),
        "{test}: different tokens than the oracle"
    );
    assert_eq!(
        crossed[0], oracle[0],
        "{test}: first request differs from the oracle"
    );
}

/// A request that fails after its vision tower ran leaves the encoder output
/// in the cache. The next request for the same image hits that entry on
/// another thread.
fn assert_failed_request_cache_entry(
    test: &str,
    (slug, declares): (&str, &str),
    max_ctx: i32,
    device: Device,
) {
    let Some(dir) = snapshot(test, slug, declares) else {
        return;
    };
    let run = |split: Split| {
        let cache = Arc::new(MultimodalCache::new(256 << 20));
        let load = arch_generator(dir.clone(), device, Some(Arc::clone(&cache)));
        let mut too_long = image_request(&dir);
        too_long.max_ctx_override = Some(max_ctx);
        let out = serve(split, &load, vec![too_long, image_request(&dir)], None);
        (out, cache.stats().hits)
    };
    let (oracle, oracle_hits) = run(Split::None);
    assert!(
        oracle[0].is_err(),
        "{test}: the first request must fail after the vision tower ran: {:?}",
        oracle[0]
    );
    assert!(
        oracle[1].as_ref().is_ok_and(|ids| !ids.is_empty()),
        "{test}: the one-thread oracle must decode the second request: {:?}",
        oracle[1]
    );
    assert_eq!(
        oracle_hits, 1,
        "{test}: the oracle's second request must hit the failed request's entry"
    );
    let (crossed, hits) = run(Split::BeforeLastRequest);
    assert!(crossed[0].is_err(), "{test}: the first request must fail");
    let second = crossed[1].as_ref().unwrap_or_else(|e| {
        panic!(
            "{test}: the encoder-cache entry of a failed request cannot be read on another \
             thread: {e}"
        )
    });
    assert_eq!(
        hits, 1,
        "{test}: the second request must hit the failed request's entry"
    );
    assert_eq!(
        Some(second),
        oracle[1].as_ref().ok(),
        "{test}: different tokens than the one-thread oracle"
    );
}

#[test]
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
fn a_failed_image_request_leaves_a_gemma4_encoder_cache_entry_another_thread_can_read() {
    assert_failed_request_cache_entry(
        "a_failed_image_request_leaves_a_gemma4_encoder_cache_entry_another_thread_can_read",
        GEMMA4_E2B,
        256,
        Device::Gpu,
    );
}

/// The Qwen3-VL path publishes its encoder output through `put_many`, not
/// `get_or_compute`.
#[test]
#[ignore = "requires Metal GPU and the Qwen3-VL snapshot; run with `make gpu-test`"]
fn a_failed_image_request_leaves_a_qwen3_vl_encoder_cache_entry_another_thread_can_read() {
    assert_failed_request_cache_entry(
        "a_failed_image_request_leaves_a_qwen3_vl_encoder_cache_entry_another_thread_can_read",
        QWEN3_VL,
        16,
        Device::Gpu,
    );
}
