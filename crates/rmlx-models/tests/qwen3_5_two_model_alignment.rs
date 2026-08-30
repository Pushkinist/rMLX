//! Classic two-model speculative decoding <-> verifier alignment gate, on a
//! GDN-hybrid pair.
//!
//! Same oracle as the MTP and EAGLE-3 alignment suites, pointed at
//! `SpeculativeDispatcher::spec_generate_greedy` — the two-model round loop
//! that runs a full draft model against a full verifier. Greedy speculative
//! decoding emits the verifier's own argmax at every position, so the round
//! loop must track a plain greedy run for a long prefix. Not bit-identical:
//! the verify pass scores a whole draft block in one forward while plain decode
//! steps one token at a time, and on a GDN hybrid that is a different reduction
//! order — but a *short* common prefix means the verifier state itself diverged.
//!
//! The failure this pins: on partial acceptance the GDN recurrent state has no
//! sequence axis, so it is restored from a pre-round snapshot and replayed over
//! the kept prefix. That replay runs the whole layer stack, and the
//! full-attention layers interleaved between the GDN layers feed them.
//! Replaying through a fresh scratch KV stack makes those FA layers attend a
//! `kept`-token prefix at positions `0..kept`, so every downstream GDN layer
//! advances on a wrong hidden. This path is where the shared rollback has
//! **two** callers per round — verifier and drafter — so a GDN pair exercises
//! both.
//!
//! This suite needs both halves of the pair to be GDN hybrids and to share a
//! vocabulary; it skips otherwise rather than failing, because a full-attention
//! pair measures nothing about the GDN rollback.
//!
//! The prompt is part of the calibration. A fixture whose answer is full of
//! near-ties spends the whole margin on legitimate flips: this pair reads 96/96
//! correct against 17/96 corrupted on the prompt below, and 32/96 against 11/96
//! on an open-ended "describe five cities" one — where the correct arm would
//! fail a gate the corrupted arm also fails.
//!
//! Server-free. Run:
//! RMLX_KV_TEST_MODEL=<path-to>/mlx-community__Qwen3.8-27B-mxfp8 \
//! RMLX_DRAFT_TEST_MODEL=<path-to>/sahilchachra__ornith-1.0-9b-mxfp8-mlx \
//! cargo test -p rmlx-models --test qwen3_5_two_model_alignment -- --ignored --nocapture

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
use rmlx_models::speculative::SpeculativeDispatcher;

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
/// reads them. Both arms must stop at the same place.
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

/// A hybrid stack declares its per-layer types; a `linear_attention` entry is
/// the GDN recurrence whose rollback this suite exercises. A full-attention
/// pair would pass trivially and prove nothing.
fn is_gdn_hybrid(model_path: &std::path::Path) -> bool {
    let Some(cfg) = config_json(model_path) else {
        return false;
    };
    let tc = cfg.get("text_config").unwrap_or(&cfg);
    tc.get("layer_types")
        .and_then(serde_json::Value::as_array)
        .is_some_and(|types| types.iter().any(|t| t.as_str() == Some("linear_attention")))
}

/// The prompt both arms decode, chat-templated the way the Qwen3.5 template
/// does, so the model is in its normal answering regime.
const PROMPT: &str = "<|im_start|>user\nList the first twelve prime numbers, separated by commas. Answer with the list only.<|im_end|>\n<|im_start|>assistant\n";

/// The two-model round loop tracks plain greedy decoding for a long prefix.
#[ignore]
#[test]
fn two_model_greedy_tracks_plain_greedy_for_a_long_prefix() {
    let (Some(model_path), Some(draft_path)) = (
        env_path("RMLX_KV_TEST_MODEL"),
        env_path("RMLX_DRAFT_TEST_MODEL"),
    ) else {
        eprintln!("[qwen35_two_model_align] verifier/draft unset or absent - skipping");
        return;
    };
    if !is_gdn_hybrid(&model_path) || !is_gdn_hybrid(&draft_path) {
        eprintln!(
            "[qwen35_two_model_align] pair is not a GDN hybrid on both sides - skipping \
             (a full-attention pair exercises no GDN rollback)"
        );
        return;
    }
    if model_path.canonicalize().ok() == draft_path.canonicalize().ok() {
        eprintln!(
            "[qwen35_two_model_align] verifier and draft name one snapshot - skipping \
             (load_speculative refuses that pair; point the two vars at different models)"
        );
        return;
    }
    let device = Device::Gpu;
    let dispatcher = SpeculativeDispatcher::load_speculative(&model_path, &draft_path, device)
        .expect("load verifier + draft");

    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt_ids: Vec<u32> = tk.encode(PROMPT, false).expect("encode").get_ids().to_vec();

    const N_TOKENS: usize = 96;
    // `k` = draft tokens per round. Large enough that partial accepts dominate
    // — a round that accepts everything never reaches the rollback.
    const K: usize = 4;
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
    {
        let mut step_fn = |s: &rmlx_models::ProbeStep| {
            spec_ids.push(s.token_id);
            None
        };
        dispatcher
            .spec_generate_greedy(
                &tk,
                &prompt_ids,
                N_TOKENS,
                K,
                kv_quant,
                None,
                0,
                &eos,
                &mut step_fn,
                None,
                &sampler_cfg,
            )
            .expect("speculative generate");
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
        dispatcher
            .verifier
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
        "[qwen35_two_model_align] common prefix {common}/{} (spec={} plain={})\n  spec  = {:?}\n  plain = {:?}",
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
    // Same bound as the MTP and EAGLE-3 gates, in the gap between the two
    // measured regimes on the documented pair: 96 of 96 shared tokens with the
    // partial-accept rollback replaying through the real KV caches, 17 of 96
    // replaying through a fresh scratch stack.
    let floor = shorter / 2;
    assert!(
        common >= floor,
        "two-model speculative tracked plain greedy for only {common} of {shorter} tokens \
         (floor {floor}) — a prefix this short means the verifier state the round loop \
         leaves behind does not match a sequential decode, not that a near-tie flipped"
    );
}
