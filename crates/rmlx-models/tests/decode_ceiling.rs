//! A generation that reaches the context ceiling stops there and keeps its
//! output.
//!
//! Each case prefills a prompt and asks for [`ASKED_TOKENS`] tokens at
//! temperature 0 with no stop token, so the only thing that ends a generation
//! is its budget. Under a ceiling that leaves [`ROOM_TOKENS`] tokens after the
//! prompt, the request must return `Ok` with exactly [`ROOM_TOKENS`] steps.
//! Under a ceiling that leaves [`ASKED_TOKENS`] tokens, the same request must
//! return all of them: this control is what shows the first result comes from
//! the ceiling and not from the prompt or the model. A prompt that fills the
//! ceiling is refused with the typed error.
//!
//! The cases cover a single-KV-head model (gemma-4-e2b) on its text path and
//! its image path, a multi-KV-head hybrid (Qwen3.8-27B-4bit) on its text path
//! and its MTP round loop, and Qwen3-VL on its text path. The image prompt is
//! counted after the image tokens are spliced in.
//!
//! Snapshots resolve from `RMLX_O_MODELS_ROOT` by slug (see
//! `tests/common/mod.rs`).
//!
//! Run:
//! cargo test -p rmlx-models --test decode_ceiling -- --ignored --test-threads=1

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stderr,
    reason = "real-model test: a missing snapshot, token or step aborts the test loudly, and a stood-down case prints its SKIP notice"
)]

mod common;

use std::sync::atomic::AtomicBool;

use rmlx_core::error::{Error, Result};
use rmlx_mlx::Device;
use rmlx_models::gemma4::{
    build_inputs_embeds, load_vision_tower, Gemma4ImageProcessor, Gemma4VisionConfig, BOI_TOKEN_ID,
    EOI_TOKEN_ID, IMAGE_TOKEN_ID,
};
use rmlx_models::speculative::mtp::{mtp_generate, MtpDrafter};
use rmlx_models::speculative::VerifierTrunk;
use rmlx_models::{arch, Pcg32, PenaltyConfig, SamplerConfig};

const GEMMA4_E2B: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__gemma-4-e2b-it-mxfp8",
    archs: &["Gemma4ForConditionalGeneration"],
};

const QWEN38_4BIT: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__Qwen3.8-27B-4bit",
    archs: &[
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ],
};

const QWEN3_VL: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__Qwen3-VL-30B-A3B-Instruct-4bit",
    archs: &["Qwen3VLMoeForConditionalGeneration"],
};

const QWEN38_MTP_4BIT_SLUG: &str = "mlx-community__Qwen3.8-27B-MTP-4bit";

/// Draft-model override, for a models root that does not hold a drafter slug.
const DRAFT_VAR: &str = "RMLX_DRAFT_TEST_MODEL";

/// Tokens of the text prompt of every text case.
const PROMPT_TOKENS: usize = 128;

/// Tokens a request asks for.
const ASKED_TOKENS: usize = 64;

/// Tokens the ceiling of the short cases leaves after the prompt.
const ROOM_TOKENS: usize = 8;

/// The block a speculative case asks for.
const SPEC_BLOCK: usize = 5;

/// A prompt of `len` tokens. `case` leads the text, so no case finds the
/// prompt-cache entry of another.
fn prompt_ids(tokenizer: &tokenizers::Tokenizer, case: &str, len: usize) -> Vec<u32> {
    const SENTENCE: &str = " The quick brown fox jumps over the lazy dog near the river bank.";
    let text = format!("Case {case}.{}", SENTENCE.repeat(len / 4 + 1));
    let ids = tokenizer.encode(text, true).expect("tokenize");
    assert!(ids.get_ids().len() >= len, "the prompt text is too short");
    ids.get_ids()[..len].to_vec()
}

fn ceiling(prompt_len: usize, room: usize) -> i32 {
    i32::try_from(prompt_len + room).expect("context fits i32")
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

fn tokenizer_of(model_path: &std::path::Path) -> tokenizers::Tokenizer {
    tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer")
}

fn load(model_path: &std::path::Path) -> arch::Architecture {
    arch::load_model(model_path, Device::Gpu, &arch::LoadOpts::default()).expect("load model")
}

/// Hold one request to the rules of this file. `run` generates [`ASKED_TOKENS`]
/// tokens under the ceiling it is given and returns the number of steps.
fn assert_stops_at_the_ceiling(
    test: &str,
    prompt_len: usize,
    mut run: impl FnMut(i32) -> Result<usize>,
) {
    let cut = run(ceiling(prompt_len, ROOM_TOKENS)).unwrap_or_else(|e| {
        panic!("{test}: a generation at the ceiling must keep its output: {e}")
    });
    assert_eq!(
        cut, ROOM_TOKENS,
        "{test}: the ceiling leaves {ROOM_TOKENS} tokens after the prompt"
    );

    let control = run(ceiling(prompt_len, ASKED_TOKENS))
        .unwrap_or_else(|e| panic!("{test}: the control generation failed: {e}"));
    assert_eq!(
        control, ASKED_TOKENS,
        "{test}: a ceiling with room for the whole request must not cut it"
    );

    let full = run(ceiling(prompt_len, 0));
    assert!(
        matches!(
            full,
            Err(Error::KvCeilingExceeded { requested, ceiling: c })
                if requested == ceiling(prompt_len, 1) && c == ceiling(prompt_len, 0)
        ),
        "{test}: a prompt that fills the ceiling must be refused with its values: {full:?}"
    );
}

/// Steps of one plain generation of [`ASKED_TOKENS`] under `max_ctx`.
fn generate(
    model: &arch::Architecture,
    tokenizer: &tokenizers::Tokenizer,
    ids: &[u32],
    max_ctx: i32,
) -> Result<usize> {
    let sampler_cfg = sampler();
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();
    model
        .generate_greedy(
            tokenizer,
            ids,
            ASKED_TOKENS,
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
        )
        .map(|steps| steps.len())
}

fn assert_plain_generation_stops_at_the_ceiling(test: &str, model_spec: &common::GoldenModel) {
    let Some(model_path) = common::model_for(model_spec, test) else {
        return;
    };
    let model = load(&model_path);
    let tokenizer = tokenizer_of(&model_path);
    let ids = prompt_ids(&tokenizer, test, PROMPT_TOKENS);
    assert_stops_at_the_ceiling(test, PROMPT_TOKENS, |max_ctx| {
        generate(&model, &tokenizer, &ids, max_ctx)
    });
}

#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
#[test]
fn gemma4_e2b_generation_stops_at_the_ceiling_and_keeps_its_output() {
    assert_plain_generation_stops_at_the_ceiling(
        "gemma4_e2b_generation_stops_at_the_ceiling_and_keeps_its_output",
        &GEMMA4_E2B,
    );
}

#[ignore = "requires Metal GPU and the Qwen3.8-27B-4bit snapshot; run with `make gpu-test`"]
#[test]
fn qwen38_4bit_generation_stops_at_the_ceiling_and_keeps_its_output() {
    assert_plain_generation_stops_at_the_ceiling(
        "qwen38_4bit_generation_stops_at_the_ceiling_and_keeps_its_output",
        &QWEN38_4BIT,
    );
}

#[ignore = "requires Metal GPU and the Qwen3-VL-30B snapshot; run with `make gpu-test`"]
#[test]
fn qwen3_vl_text_generation_stops_at_the_ceiling_and_keeps_its_output() {
    assert_plain_generation_stops_at_the_ceiling(
        "qwen3_vl_text_generation_stops_at_the_ceiling_and_keeps_its_output",
        &QWEN3_VL,
    );
}

/// The MTP round loop widens its last round to the budget, so its writes reach
/// the ceiling exactly; the case would fail at the ceiling if a round wrote one
/// position more.
#[ignore = "requires Metal GPU and the Qwen3.8-27B-4bit and Qwen3.8-27B-MTP-4bit snapshots; run with `make gpu-test`"]
#[test]
fn qwen38_4bit_mtp_generation_stops_at_the_ceiling_and_keeps_its_output() {
    let test = "qwen38_4bit_mtp_generation_stops_at_the_ceiling_and_keeps_its_output";
    let Some(model_path) = common::model_for(&QWEN38_4BIT, test) else {
        return;
    };
    let Some(draft_path) = common::apply(
        common::slug_or_override(DRAFT_VAR, QWEN38_MTP_4BIT_SLUG, common::Role::Sidecar),
        test,
    ) else {
        return;
    };
    let verifier = load(&model_path);
    let mut drafter = MtpDrafter::load(&draft_path, VerifierTrunk::of(&verifier), Device::Gpu)
        .expect("load sidecar");
    let tokenizer = tokenizer_of(&model_path);
    let ids = prompt_ids(&tokenizer, test, PROMPT_TOKENS);

    assert_stops_at_the_ceiling(test, PROMPT_TOKENS, |max_ctx| {
        mtp_generate(
            &verifier,
            &mut drafter,
            &tokenizer,
            &ids,
            ASKED_TOKENS,
            SPEC_BLOCK,
            None,
            Some(max_ctx),
            &[],
            &mut |_| rmlx_models::StepVerdict::Continue,
            &sampler(),
            Device::Gpu,
            &AtomicBool::new(false),
        )
        .map(|(steps, _widest_block)| steps.len())
    });
}

/// The image path counts the prompt after its image tokens are spliced in: the
/// ceiling is set from that length, so a budget taken from the text prompt
/// alone would let the generation cross it.
#[ignore = "requires Metal GPU and the gemma-4-e2b snapshot; run with `make gpu-test`"]
#[test]
fn gemma4_e2b_image_generation_stops_at_the_ceiling_and_keeps_its_output() {
    let test = "gemma4_e2b_image_generation_stops_at_the_ceiling_and_keeps_its_output";
    let Some(model_path) = common::model_for(&GEMMA4_E2B, test) else {
        return;
    };
    let Some(vision_cfg) = Gemma4VisionConfig::from_model_dir(&model_path).expect("vision_config")
    else {
        eprintln!("SKIP {test}: the snapshot ships no vision_config");
        return;
    };
    let model = load(&model_path);
    let tokenizer = tokenizer_of(&model_path);
    let text_model = model.as_gemma4().expect("a Gemma4 snapshot");
    let (vision, embedder) = load_vision_tower(&model_path, &vision_cfg).expect("vision tower");
    let processor = Gemma4ImageProcessor::from_model_dir(&model_path).expect("image processor");

    let rgb = image::RgbImage::from_pixel(64, 64, image::Rgb([220, 30, 30]));
    let mut png: Vec<u8> = Vec::new();
    image::DynamicImage::ImageRgb8(rgb)
        .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encode png");
    let n_soft = processor
        .preprocess(&png)
        .expect("preprocess")
        .num_soft_tokens;
    let mut ids: Vec<u32> = vec![2, BOI_TOKEN_ID];
    ids.extend(std::iter::repeat_n(IMAGE_TOKEN_ID, n_soft));
    ids.push(EOI_TOKEN_ID);
    ids.extend_from_slice(&[105, 2364, 107]);

    assert_stops_at_the_ceiling(test, ids.len(), |max_ctx| {
        let pixels = processor.preprocess(&png)?;
        let (embeds, masked_ids) = build_inputs_embeds(
            text_model,
            &vision,
            &embedder,
            &[pixels],
            &ids,
            Device::Gpu,
            None,
            0,
        )?;
        let sampler_cfg = sampler();
        let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
        let penalty_cfg = PenaltyConfig::default();
        let mut token_history: Vec<u32> = Vec::new();
        model
            .generate_image(
                &tokenizer,
                &ids,
                embeds,
                masked_ids,
                ASKED_TOKENS,
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
            )
            .map(|steps| steps.len())
    });
}
