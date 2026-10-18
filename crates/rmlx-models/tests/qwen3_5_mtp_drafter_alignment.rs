//! Qwen3.5-family MTP sidecar <-> verifier alignment gate.
//!
//! Pins two properties of the `--draft-kind mtp` path against real checkpoints:
//!
//!   1. **Dense sidecar acceptance.** An MTP sidecar whose `layers.0` FFN is a
//!      plain SwiGLU (`mlp.{gate,up,down}_proj`, no router, no `switch_mlp`)
//!      loads. Its `text_config` carries no `num_experts` /
//!      `num_experts_per_tok` at all, so both the config read and the layer
//!      build must key off facts rather than demand MoE keys. The Qwen3.8-27B
//!      sidecar is exactly this shape; the Qwen3.6-35B-A3B one is the MoE
//!      shape, and both must load through the same code.
//!
//!   2. **Rollback fidelity.** Greedy MTP is greedy decoding with the verifier
//!      in the loop: every emitted token is the verifier's own argmax at that
//!      position. So the round loop's output must track a plain greedy run for
//!      a long prefix. It cannot be asserted bit-identical — the verify pass
//!      scores a whole draft block in one forward while plain decode steps one
//!      token at a time, and on this hybrid stack that is a different reduction
//!      order through 48 GDN layers — but a *short* common prefix means the
//!      verifier state itself diverged, not that a near-tie flipped.
//!
//!      The failure this pins: a partial acceptance that leaves the verifier in
//!      a state a sequential decode would not reach. The rollback truncates the
//!      attention layers' KV to the accepted prefix and refolds the GDN layers'
//!      recurrent state from the round tape; a rollback that keeps a rejected
//!      draft, or refolds the wrong prefix, makes every later token read a wrong
//!      state and the run diverges within a few tokens.
//!
//! The second check's threshold is one rule, half of the shorter arm, checked
//! against each pair's two readings (see [`greedy_tracks_plain_greedy`]) — its
//! prompt is the Qwen chat template written out by hand, and the number of
//! tokens two arms share before an ordinary near-tie flip is a property of the
//! checkpoint. Point it at a different pair and re-measure both arms before
//! reading a failure as a regression. The first check is pair-independent.
//!
//! Server-free. Both halves resolve by slug under `RMLX_O_MODELS_ROOT`, so a
//! machine holding the snapshots runs this with nothing exported. Run:
//! cargo test -p rmlx-models --test qwen3_5_mtp_drafter_alignment -- --ignored --nocapture

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::too_many_lines
)]

use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::speculative::mtp::{mtp_generate, MtpDrafter};
use rmlx_models::speculative::VerifierTrunk;

mod common;

/// The verifier the pair-independent cells run on.
const VERIFIER: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__Qwen3.8-27B-mxfp8",
    archs: &["Qwen3_5ForConditionalGeneration"],
};

/// The drafter the pair-independent cells run on.
const DRAFT_SLUG: &str = "mlx-community__Qwen3.8-27B-MTP-mxfp8";

/// A verifier and the sidecar trained against it. A sidecar quantized unlike
/// its verifier loads and drafts fluently, so the two slugs travel together.
struct Pair {
    verifier: common::GoldenModel,
    draft_slug: &'static str,
}

const MXFP8_PAIR: Pair = Pair {
    verifier: VERIFIER,
    draft_slug: DRAFT_SLUG,
};

const FOUR_BIT_PAIR: Pair = Pair {
    verifier: common::GoldenModel {
        slug: "mlx-community__Qwen3.8-27B-4bit",
        archs: &["Qwen3_5ForConditionalGeneration"],
    },
    draft_slug: "mlx-community__Qwen3.8-27B-MTP-4bit",
};

/// Draft-model override, for a models root that does not hold [`DRAFT_SLUG`].
const DRAFT_VAR: &str = "RMLX_DRAFT_TEST_MODEL";

/// The verifier's stop ids, read from its `config.json` the same way the server
/// reads them (`eos_token_id` is a scalar on some checkpoints, an array on
/// others). Both arms must stop at the same place: past the natural end this
/// model emits maximum-entropy filler, where a near-tie flip says nothing about
/// the round loop.
fn eos_ids(model_path: &std::path::Path) -> Vec<u32> {
    let Ok(raw) = std::fs::read(model_path.join("config.json")) else {
        return Vec::new();
    };
    let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&raw) else {
        return Vec::new();
    };
    match cfg.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => Vec::new(),
    }
}

/// The prompt both arms decode. Chat-templated the way the Qwen3.5 template
/// does, so the model is in its normal answering regime rather than the
/// degenerate bare-string one.
const PROMPT: &str = "<|im_start|>user\nWhat is the capital of France? Answer in one short sentence.<|im_end|>\n<|im_start|>assistant\n";

/// The sidecar loads whatever FFN its `layers.0` actually carries.
///
/// A dense sidecar has no `num_experts` in `text_config` and no
/// `mlp.switch_mlp.*` tensors. Reading the expert counts as required keys
/// rejects it at config parse; building `MlpBlock::Moe` unconditionally rejects
/// it at tensor load. Either way the failure is a hard startup error, which is
/// why this asserts the load itself rather than anything downstream.
#[ignore]
#[test]
fn mtp_sidecar_loads_whatever_ffn_it_carries() {
    let test = "mtp_sidecar_loads_whatever_ffn_it_carries";
    let Some(model_path) = common::model_for(&VERIFIER, test) else {
        return;
    };
    let Some(draft_path) = common::apply(
        common::slug_or_override(DRAFT_VAR, DRAFT_SLUG, common::Role::Sidecar),
        test,
    ) else {
        return;
    };
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let hidden = verifier.hidden_size();

    let drafter = MtpDrafter::load(&draft_path, VerifierTrunk::of(&verifier), device)
        .expect("MTP sidecar must load regardless of whether its layers.0 FFN is dense or MoE");

    assert!(
        drafter.block_size().is_none_or(|b| b >= 2),
        "a declared block_size {:?} must leave room for at least one draft token",
        drafter.block_size()
    );
    assert_eq!(
        drafter.hidden_size(),
        hidden,
        "sidecar fc must project to the verifier width"
    );
    eprintln!(
        "[qwen35_mtp_align] sidecar loaded: hidden={hidden} block_size={:?}",
        drafter.block_size()
    );
}

/// The MTP round loop tracks plain greedy decoding for a long prefix.
///
/// Both arms run the same verifier at temp=0 on the same prompt. A rollback
/// that corrupts the verifier's state shows up here as an early divergence — a
/// rollback that keeps one rejected draft diverges within two tokens on both
/// pairs.
#[ignore]
#[test]
fn mtp_greedy_tracks_plain_greedy_for_a_long_prefix() {
    greedy_tracks_plain_greedy(
        "mtp_greedy_tracks_plain_greedy_for_a_long_prefix",
        &MXFP8_PAIR,
    );
}

/// The same check on the 4-bit pair, against its own measured readings.
#[ignore]
#[test]
fn mtp_greedy_tracks_plain_greedy_for_a_long_prefix_on_the_4bit_pair() {
    greedy_tracks_plain_greedy(
        "mtp_greedy_tracks_plain_greedy_for_a_long_prefix_on_the_4bit_pair",
        &FOUR_BIT_PAIR,
    );
}

fn greedy_tracks_plain_greedy(test: &str, pair: &Pair) {
    let Some(model_path) = common::model_for(&pair.verifier, test) else {
        return;
    };
    let Some(draft_path) = common::apply(
        common::slug_or_override(DRAFT_VAR, pair.draft_slug, common::Role::Sidecar),
        test,
    ) else {
        return;
    };
    if let (Some(verifier_mode), Some(draft_mode)) = (
        common::declared_quant_mode(&model_path),
        common::declared_quant_mode(&draft_path),
    ) {
        if verifier_mode != draft_mode {
            eprintln!(
                "SKIP {test}: the verifier is quantized {verifier_mode} and the sidecar \
                 {draft_mode}, so the two are not this pair"
            );
            return;
        }
    }
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let mut drafter =
        MtpDrafter::load(&draft_path, VerifierTrunk::of(&verifier), device).expect("load sidecar");
    let block_size = rmlx_models::speculative::default_block_for(drafter.block_size());

    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt_ids: Vec<u32> = tk.encode(PROMPT, false).expect("encode").get_ids().to_vec();

    const N_TOKENS: usize = 48;
    // Both arms must see the same KV codec, or the comparison measures the
    // codec rather than the rollback.
    let kv_quant = Some(rmlx_kv_quant::KvQuant::None);
    let eos = eos_ids(&model_path);
    assert!(
        !eos.is_empty(),
        "the verifier config must name its stop ids — without them both arms run \
         past the answer into maximum-entropy filler and the comparison is noise"
    );

    let sampler_cfg = rmlx_models::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };

    let mut spec_ids: Vec<u32> = Vec::new();
    let recorder = common::round_stream::RoundStreamRecorder::new();
    {
        let mut step_fn = |s: &rmlx_models::ProbeStep| {
            spec_ids.push(s.token_id);
            None
        };
        let (_steps, ran) =
            tracing::subscriber::with_default(std::sync::Arc::clone(&recorder), || {
                mtp_generate(
                    &verifier,
                    &mut drafter,
                    &tk,
                    &prompt_ids,
                    N_TOKENS,
                    block_size,
                    kv_quant,
                    None,
                    &eos,
                    &mut step_fn,
                    &sampler_cfg,
                    device,
                    &std::sync::atomic::AtomicBool::new(false),
                )
            })
            .expect("mtp generate");
        assert_eq!(ran, block_size, "the loop must run the block it was handed");
    }
    // A round that accepted fewer drafts than it proposed is the one that rolls
    // the verifier back. A run with none exercises no rollback, so tracking
    // plain greedy would say nothing about it.
    let count = |round: &common::round_stream::CapturedEvent, field: &str| -> usize {
        round
            .field(field)
            .and_then(|v| v.parse().ok())
            .unwrap_or_else(|| panic!("a round event carries a numeric {field}: {round:?}"))
    };
    let rounds = common::round_stream::round_events(&recorder.events());
    let partial_rounds = rounds
        .iter()
        .filter(|r| count(r, "accept") < count(r, "num_draft"))
        .count();
    eprintln!(
        "[qwen35_mtp_align] rounds={} partial_rounds={partial_rounds}",
        rounds.len()
    );
    assert!(
        partial_rounds > 0,
        "none of {} rounds was partially accepted, so no rollback ran and the common \
         prefix below says nothing about one",
        rounds.len()
    );

    let mut plain_ids: Vec<u32> = Vec::new();
    let mut rng = rmlx_models::sampler::Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = rmlx_models::sampler::PenaltyConfig::default();
    let mut history: Vec<u32> = Vec::new();
    {
        let mut plain_step = |s: &rmlx_models::ProbeStep| {
            plain_ids.push(s.token_id);
            None
        };
        verifier
            .generate_greedy(
                &tk,
                &prompt_ids,
                N_TOKENS,
                device,
                kv_quant,
                None,
                1,
                &eos,
                &mut plain_step,
                None,
                &sampler_cfg,
                &mut rng,
                &penalty_cfg,
                &mut history,
                &std::sync::atomic::AtomicBool::new(false),
            )
            .expect("plain greedy generate");
    }

    let common = spec_ids
        .iter()
        .zip(plain_ids.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "[qwen35_mtp_align] common prefix {common}/{} (spec={} plain={})\n  spec  = {:?}\n  plain = {:?}",
        N_TOKENS,
        spec_ids.len(),
        plain_ids.len(),
        tk.decode(&spec_ids, false).unwrap_or_default(),
        tk.decode(&plain_ids, false).unwrap_or_default(),
    );

    let shorter = spec_ids.len().min(plain_ids.len());
    assert!(
        shorter >= 12,
        "both arms must emit a real answer; got spec={} plain={}",
        spec_ids.len(),
        plain_ids.len()
    );
    // The oracle is the module doc's: a *short* common prefix means the
    // verifier state itself diverged. It is deliberately not bit-identity —
    // the verify pass scores a whole block in one forward while plain decode
    // steps one token at a time, and on this stack that is a different
    // reduction order through 48 GDN layers, so one legitimate near-tie flip
    // is expected behaviour and must not fail the gate.
    //
    // One rule, `shorter / 2`, checked against each pair's two readings: the
    // shipped round loop, and the same loop with a rollback that keeps one
    // rejected draft on every partial round (its rollback stops one position
    // short; `scripts/spec_broken_engine.sh` names the edit).
    //
    //   pair  | shipped              | rollback keeps a draft | floor
    //   mxfp8 | 31 of 31 shared      |  2 of 12 shared        | 6
    //   4-bit | 31 of 31 shared      |  2 of 31 shared        | 15
    //
    // The corrupted mxfp8 run also stopped early (12 tokens against 31), which
    // is why its floor is lower. Half leaves each pair's regimes a wide margin
    // — the corrupted arm is 3x (mxfp8) and 7.5x (4-bit) below it and the
    // correct arm 2.1x above — which is what keeps one legitimate near-tie flip
    // from failing the gate.
    //
    // The shipped run has 3 partial rounds of 11 on each pair.
    //
    // The limit: the run is 31 tokens, so a rollback defect whose first effect
    // lands after the floor passes. The partial-round assertion above shows the
    // run reached a rollback; it does not show the defect bites early.
    let floor = shorter / 2;
    assert!(
        common >= floor,
        "MTP tracked plain greedy for only {common} of {shorter} tokens (floor {floor}) — \
         a prefix this short means the verifier state the round loop leaves behind does \
         not match a sequential decode, not that a near-tie flipped"
    );
}

/// The round loop runs the block it was handed, past the sidecar's declaration.
///
/// The block a round ran is not otherwise observable from outside the loop, and
/// the value the shipped sidecars declare is small enough that a clamp back to
/// it would still produce fluent output at a plausible accept rate. This drives
/// a block above every shipped declaration and reads the loop's own answer for
/// what it ran.
#[ignore]
#[test]
fn the_round_loop_runs_a_block_deeper_than_the_sidecar_declares() {
    let test = "the_round_loop_runs_a_block_deeper_than_the_sidecar_declares";
    let Some(model_path) = common::model_for(&VERIFIER, test) else {
        return;
    };
    let Some(draft_path) = common::apply(
        common::slug_or_override(DRAFT_VAR, DRAFT_SLUG, common::Role::Sidecar),
        test,
    ) else {
        return;
    };
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let mut drafter =
        MtpDrafter::load(&draft_path, VerifierTrunk::of(&verifier), device).expect("load sidecar");
    let declared = drafter.block_size();
    let requested = declared.map_or(8, |d| d + 5);

    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt_ids: Vec<u32> = tk.encode(PROMPT, false).expect("encode").get_ids().to_vec();
    let eos = eos_ids(&model_path);
    let sampler_cfg = rmlx_models::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };

    let mut emitted = 0usize;
    let mut step_fn = |_: &rmlx_models::ProbeStep| {
        emitted += 1;
        None
    };
    let (_steps, ran) = mtp_generate(
        &verifier,
        &mut drafter,
        &tk,
        &prompt_ids,
        16,
        requested,
        Some(rmlx_kv_quant::KvQuant::None),
        None,
        &eos,
        &mut step_fn,
        &sampler_cfg,
        device,
        &std::sync::atomic::AtomicBool::new(false),
    )
    .expect("mtp generate");

    assert_eq!(
        ran, requested,
        "asked for {requested} against a sidecar declaring {declared:?}, and the rounds \
         ran {ran}"
    );
    assert!(
        declared.is_none_or(|d| requested > d),
        "this cell only says anything if the request is past the declaration: \
         declared={declared:?} requested={requested}"
    );
}
