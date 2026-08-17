//! BitNet emits per-token logprobs when `top_logprobs_k > 0`.
//!
//! BitNet drives its own decode loop rather than the shared `pipelined_decode`,
//! so it does not inherit logprob capture from it. It once hardcoded
//! `ProbeStep.logprobs = None` at every construction site, which is invisible
//! at runtime — the payload is an `Option`, so an arch that never fills it
//! looks exactly like a request that never asked for one. The first thing that
//! noticed was the golden-token gate: its tie-margin probe reads the diverging
//! step's logprobs, so on BitNet it could never measure a margin and refused
//! every regeneration.
//!
//! A source-level gate (`every_arch_generate_path_honours_top_logprobs_k`)
//! pins the wiring without weights. This one is the live counterpart: real
//! weights, real Metal, real decode.
//!
//! Model: `mlx-community__bitnet-b1.58-2B-4T` (BitNetForCausalLM), resolved
//! from `RMLX_O_MODELS_ROOT` by slug or `RMLX_KV_TEST_MODEL` by path.
//!
//! ```text
//! cargo test -p rmlx-models --test bitnet_logprobs -- --ignored --test-threads=1
//! ```

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
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

mod common;

use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

const MODEL: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__bitnet-b1.58-2B-4T",
    archs: &["BitNetForCausalLM"],
};

/// `top_logprobs_k = 2` populates `ProbeStep.logprobs` on BitNet's decode steps.
///
/// Asserts the shape the tie-margin probe depends on: at least two candidates,
/// ranked descending, and — because this is temp=0 greedy — a rank-0 id equal
/// to the token actually emitted. A run that emits tokens but no logprobs is
/// the exact pre-fix behaviour and fails here.
#[ignore]
#[test]
fn bitnet_emits_logprobs_when_requested() {
    let Some(model_path) = common::model_for(&MODEL, "bitnet_emits_logprobs_when_requested") else {
        return;
    };

    let device = Device::Gpu;
    let model = arch::load_model(&model_path, device, &arch::LoadOpts::default())
        .expect("arch::load_model");
    let tokenizer = tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json"))
        .expect("load tokenizer.json");

    let prompt_ids: Vec<u32> = tokenizer
        .encode(common::GOLDEN_PROMPT, true)
        .expect("tokenize prompt")
        .get_ids()
        .to_vec();

    // Same deterministic configuration the golden gate decodes under, except
    // that this one asks for logprobs.
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 2,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();

    let steps = model
        .generate_greedy(
            &tokenizer,
            &prompt_ids,
            common::N_GOLDEN_TOKENS,
            device,
            Some(KvQuant::K8V8),
            None, // max_ctx: arch default
            1,    // single-slot prompt cache
            &[],  // no EOS stop — force the full N tokens
            &mut |_| None,
            None, // no sampler constraint
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("generate_greedy");

    assert!(!steps.is_empty(), "decode produced no steps at all");

    let with_lp = steps.iter().filter(|s| s.logprobs.is_some()).count();
    assert!(
        with_lp > 0,
        "top_logprobs_k=2 produced {} steps but not one carried logprobs — the decode path \
         is ignoring the field, which is what leaves the golden tie-margin probe unable to \
         measure anything",
        steps.len()
    );

    // The tie-margin probe reads a diverging step, which is a decode step, not
    // the prefill token. Require coverage past index 0 so a fix that wires only
    // the prefill tail does not read as success.
    let decode_with_lp = steps
        .iter()
        .skip(1)
        .filter(|s| s.logprobs.is_some())
        .count();
    assert!(
        decode_with_lp > 0,
        "only the prefill token carried logprobs; the tie-margin probe reads a decode step"
    );

    for (i, step) in steps.iter().enumerate() {
        let Some(lp) = step.logprobs.as_ref() else {
            continue;
        };
        assert!(
            lp.top.len() >= 2,
            "step {i}: top_logprobs_k=2 must yield two candidates, got {}",
            lp.top.len()
        );
        assert!(
            lp.top[0].1 >= lp.top[1].1,
            "step {i}: candidates must be ranked descending, got {:?}",
            lp.top
        );
        assert_eq!(
            lp.top[0].0, step.token_id,
            "step {i}: at temp=0 the emitted token must be the top-ranked candidate; \
             a mismatch means the logprobs describe a different step's distribution"
        );
        assert_eq!(
            lp.token_id, step.token_id,
            "step {i}: logprob payload names a different token than the step emitted"
        );
    }

    // The number the gate actually consumes.
    let margins: Vec<f32> = steps
        .iter()
        .filter_map(|s| s.logprobs.as_ref())
        .map(|lp| lp.top[0].1 - lp.top[1].1)
        .collect();
    println!(
        "bitnet logprobs: {}/{} steps carry a payload; top-2 margins = {:?}",
        with_lp,
        steps.len(),
        margins
    );
}
