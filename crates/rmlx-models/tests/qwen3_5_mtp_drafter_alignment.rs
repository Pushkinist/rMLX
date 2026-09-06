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
//!      The failure this pins: on partial acceptance the GDN recurrent state
//!      is restored from a pre-round snapshot and replayed over the kept
//!      prefix. That replay runs the whole layer stack, and the full-attention
//!      layers interleaved between the GDN layers feed them. Replaying through
//!      a fresh scratch KV stack makes those FA layers attend a `v_kept`-token
//!      prefix at positions `0..v_kept`, so every downstream GDN layer advances
//!      on a wrong hidden and the run degenerates within a few tokens.
//!
//! The second check's threshold is calibrated against the pair named below —
//! its prompt is that model's chat template written out by hand, and the number
//! of tokens two arms share before an ordinary near-tie flip is a property of
//! the checkpoint. Point it at a different pair and re-measure both arms before
//! reading a failure as a regression. The first check is pair-independent.
//!
//! Server-free. Run (verifier + sidecar present):
//! RMLX_KV_TEST_MODEL=<path-to>/mlx-community__Qwen3.8-27B-mxfp8 \
//! RMLX_DRAFT_TEST_MODEL=<path-to>/mlx-community__Qwen3.8-27B-MTP-mxfp8 \
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

use std::path::PathBuf;

use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::speculative::mtp::{mtp_generate_greedy, MtpDrafter};

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

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
    let (Some(model_path), Some(draft_path)) = (
        env_path("RMLX_KV_TEST_MODEL"),
        env_path("RMLX_DRAFT_TEST_MODEL"),
    ) else {
        eprintln!(
            "SKIP mtp_sidecar_loads_whatever_ffn_it_carries: RMLX_KV_TEST_MODEL and \
             RMLX_DRAFT_TEST_MODEL must both name an existing snapshot directory"
        );
        return;
    };
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let hidden = verifier.hidden_size();

    let drafter = MtpDrafter::load(&draft_path, hidden, device)
        .expect("MTP sidecar must load regardless of whether its layers.0 FFN is dense or MoE");

    assert!(
        drafter.block_size() >= 2,
        "block_size {} must leave room for at least one draft token",
        drafter.block_size()
    );
    assert_eq!(
        drafter.hidden_size(),
        hidden,
        "sidecar fc must project to the verifier width"
    );
    eprintln!(
        "[qwen35_mtp_align] sidecar loaded: hidden={hidden} block_size={}",
        drafter.block_size()
    );
}

/// The MTP round loop tracks plain greedy decoding for a long prefix.
///
/// Both arms run the same verifier at temp=0 on the same prompt. A rollback
/// that corrupts the verifier's recurrent state shows up here as an early
/// divergence — the pre-fix scratch-cache replay diverged within a handful of
/// tokens and then degenerated into a repetition loop.
#[ignore]
#[test]
fn mtp_greedy_tracks_plain_greedy_for_a_long_prefix() {
    let (Some(model_path), Some(draft_path)) = (
        env_path("RMLX_KV_TEST_MODEL"),
        env_path("RMLX_DRAFT_TEST_MODEL"),
    ) else {
        eprintln!(
            "SKIP mtp_greedy_tracks_plain_greedy_for_a_long_prefix: RMLX_KV_TEST_MODEL and \
             RMLX_DRAFT_TEST_MODEL must both name an existing snapshot directory"
        );
        return;
    };
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let hidden = verifier.hidden_size();
    let mut drafter = MtpDrafter::load(&draft_path, hidden, device).expect("load sidecar");
    let block_size = drafter.block_size();

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

    let mut spec_ids: Vec<u32> = Vec::new();
    {
        let mut step_fn = |s: &rmlx_models::ProbeStep| {
            spec_ids.push(s.token_id);
            None
        };
        mtp_generate_greedy(
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
            device,
        )
        .expect("mtp generate");
    }

    let mut plain_ids: Vec<u32> = Vec::new();
    let sampler_cfg = rmlx_models::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
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
    // The bound sits in the gap between the two measured regimes on the
    // documented pair: 31 of 31 shared tokens with the partial-accept rollback
    // replaying through the real KV caches, 4 of 31 replaying through a fresh
    // scratch stack. Half leaves both regimes a wide margin — the corrupted
    // arm is 3.8x below it and the correct arm 2.1x above — which is what
    // keeps one legitimate near-tie flip from failing the gate.
    let floor = shorter / 2;
    assert!(
        common >= floor,
        "MTP tracked plain greedy for only {common} of {shorter} tokens (floor {floor}) — \
         a prefix this short means the verifier state the round loop leaves behind does \
         not match a sequential decode, not that a near-tie flipped"
    );
}
