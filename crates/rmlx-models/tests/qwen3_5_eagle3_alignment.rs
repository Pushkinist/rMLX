//! Qwen3.5-family EAGLE-3 drafter <-> verifier alignment gate.
//!
//! Same oracle as `qwen3_5_mtp_drafter_alignment.rs`, pointed at the other
//! round loop. Greedy EAGLE-3 is greedy decoding with the verifier in the
//! loop: every emitted token is the verifier's own argmax at that position, so
//! the round loop's output must track a plain greedy run for a long prefix.
//! It cannot be asserted bit-identical — the verify pass scores a whole draft
//! block in one forward while plain decode steps one token at a time, and on
//! this hybrid stack that is a different reduction order through the GDN
//! layers — but a *short* common prefix means the verifier state itself
//! diverged, not that a near-tie flipped.
//!
//! The failure this pins: on partial acceptance the GDN recurrent state is
//! restored from a pre-round snapshot and replayed over the kept prefix. That
//! replay runs the whole layer stack, and the full-attention layers interleaved
//! between the GDN layers feed them. Replaying through a fresh scratch KV stack
//! makes those FA layers attend a `v_kept`-token prefix at positions
//! `0..v_kept`, so every downstream GDN layer advances on a wrong hidden.
//! `eagle3_generate` is GDN-only by construction (it refuses a verifier
//! without `needs_lin_caches`), so on this path the defect is always live on a
//! partial accept.
//!
//! The threshold is calibrated against the pair named below, and so is the
//! prompt: a fixture whose answer is full of near-ties spends the gate's whole
//! margin on legitimate flips (the same pair on an open-ended "describe five
//! cities" prompt reads 76/96 correct against 14/96 corrupted, where this one
//! reads 93/96 against 13/96). Point it at a different pair or a different
//! prompt and re-measure both arms before reading a failure as a regression.
//!
//! Server-free. Run:
//! RMLX_KV_TEST_MODEL=<path-to>/mlx-community__Qwen3.6-35B-A3B-8bit \
//! RMLX_DRAFT_TEST_MODEL=<path-to>/Dogacel__specdrift-qwen3.6-35b-a3b-eagle3 \
//! cargo test -p rmlx-models --test qwen3_5_eagle3_alignment -- --ignored --nocapture

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
use rmlx_models::speculative::eagle3::{eagle3_generate, Eagle3Drafter};

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
}

fn config_json(model_path: &std::path::Path) -> Option<serde_json::Value> {
    let raw = std::fs::read(model_path.join("config.json")).ok()?;
    serde_json::from_slice(&raw).ok()
}

/// The verifier's stop ids, read from its `config.json` the same way the server
/// reads them. Both arms must stop at the same place: past the natural end this
/// model emits maximum-entropy filler, where a near-tie flip says nothing about
/// the round loop.
fn eos_ids(model_path: &std::path::Path) -> Vec<u32> {
    let Some(cfg) = config_json(model_path) else {
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

/// An EAGLE-3 drafter snapshot declares a reduced draft vocabulary and the aux
/// hidden-state layer ids it reads. A drafter of any other kind pointed at this
/// suite is not a failure — it is the wrong pair, and the test says so.
fn is_eagle3_drafter(draft_path: &std::path::Path) -> bool {
    config_json(draft_path)
        .is_some_and(|c| c.get("draft_vocab_size").is_some() && c.get("eagle_config").is_some())
}

/// The prompt both arms decode, chat-templated the way the Qwen3.5 template
/// does, so the model is in its normal answering regime.
const PROMPT: &str = "<|im_start|>user\nList the first twelve prime numbers, separated by commas. Answer with the list only.<|im_end|>\n<|im_start|>assistant\n";

/// The EAGLE-3 round loop tracks plain greedy decoding for a long prefix.
#[ignore]
#[test]
fn eagle3_greedy_tracks_plain_greedy_for_a_long_prefix() {
    let (Some(model_path), Some(draft_path)) = (
        env_path("RMLX_KV_TEST_MODEL"),
        env_path("RMLX_DRAFT_TEST_MODEL"),
    ) else {
        eprintln!("[qwen35_eagle3_align] verifier/drafter unset or absent - skipping");
        return;
    };
    if !is_eagle3_drafter(&draft_path) {
        eprintln!(
            "[qwen35_eagle3_align] RMLX_DRAFT_TEST_MODEL is not an EAGLE-3 drafter - skipping"
        );
        return;
    }
    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let hidden = verifier.hidden_size();
    let vocab = verifier.vocab_size();
    let eos = eos_ids(&model_path);
    assert!(
        !eos.is_empty(),
        "the verifier config must name its stop ids — without them both arms run \
         past the answer into maximum-entropy filler and the comparison is noise"
    );
    let mut drafter =
        Eagle3Drafter::load(&draft_path, hidden, vocab, &eos, device).expect("load drafter");

    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt_ids: Vec<u32> = tk.encode(PROMPT, false).expect("encode").get_ids().to_vec();

    const N_TOKENS: usize = 96;
    const BLOCK_SIZE: usize = 5;
    // Both arms must see the same KV codec, or the comparison measures the
    // codec rather than the rollback.
    let kv_quant = Some(rmlx_kv_quant::KvQuant::None);

    let sampler_cfg = rmlx_models::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };

    let mut spec_ids: Vec<u32> = Vec::new();
    {
        let mut step_fn = |s: &rmlx_models::ProbeStep| {
            spec_ids.push(s.token_id);
            None
        };
        eagle3_generate(
            &verifier,
            &mut drafter,
            &tk,
            &prompt_ids,
            N_TOKENS,
            BLOCK_SIZE,
            kv_quant,
            None,
            &eos,
            &mut step_fn,
            &sampler_cfg,
            device,
        )
        .expect("eagle3 generate");
    }

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
            )
            .expect("plain greedy generate");
    }

    let common = spec_ids
        .iter()
        .zip(plain_ids.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "[qwen35_eagle3_align] common prefix {common}/{} (spec={} plain={})\n  spec  = {:?}\n  plain = {:?}",
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
    // Same bound as the MTP gate, in the gap between the two measured
    // regimes on the documented pair: 93 of 96 shared tokens with the
    // partial-accept rollback replaying through the real KV caches (the flip
    // at 93 is an ordinary near-tie, and the gate must tolerate it), 13 of 96
    // replaying through a fresh scratch stack. Half leaves the correct arm a
    // 1.9x margin and the corrupted arm a 3.7x one.
    let floor = shorter / 2;
    assert!(
        common >= floor,
        "EAGLE-3 tracked plain greedy for only {common} of {shorter} tokens (floor {floor}) — \
         a prefix this short means the verifier state the round loop leaves behind does \
         not match a sequential decode, not that a near-tie flipped"
    );
}
