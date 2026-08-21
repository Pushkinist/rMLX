//! Integration test for `--probe-smoke` against the primary test snapshot.
//!
//! Skips gracefully if the snapshot is absent — never fails CI on a developer
//! who doesn't have the model locally.
//!
//! The snapshot (gemma-4-e4b-it-mxfp8) is known-clean: prior decode-probe and
//! forward-probe runs showed no NaN and non-garbage tokens. We assert the
//! verdict is Ok or Inconclusive — NOT BrokenPunctLoop or BrokenNan.
//!
//! **Timeout note**: 8-token generation re-encodes the full prefix each step
//! (no KV cache). On the 4B Gemma4 on CPU this takes several minutes. The test
//! is marked `#[ignore]` to keep `cargo test` fast; run explicitly with:
//!
//! cargo test -p rmlx-cli --test smoke_probe -- --ignored --nocapture
//!
//! Or via the integration smoke run in the DoD verification step.

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
    clippy::ignore_without_reason
)]

fn primary_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

/// Full smoke probe integration test — marked `#[ignore]` due to multi-minute
/// CPU runtime. Run with `-- --ignored`.
///
/// Routes through `arch::load_model` (dispatch layer) instead of calling
/// `gemma4::load_from_path` directly.
#[test]
#[ignore]
fn smoke_probe_gemma4_mxfp8_is_clean() {
    let Some(model_path_buf) = primary_model_dir() else {
        eprintln!("[smoke_probe] skipping: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    let model_path = model_path_buf.as_path();
    if !model_path.exists() {
        eprintln!("[smoke_probe] primary snapshot absent at {model_path:?} — skipping");
        return;
    }

    // Load tokenizer + resolve BOS id.
    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path).expect("load tokenizer.json");

    // Extract BOS from tokenizer_config.json.
    let cfg_path = model_path.join("tokenizer_config.json");
    let data = std::fs::read(&cfg_path).expect("read tokenizer_config.json");
    let v: serde_json::Value = serde_json::from_slice(&data).expect("parse tokenizer_config.json");
    let bos_str = match v.get("bos_token") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(map)) => map
            .get("content")
            .and_then(|c| c.as_str())
            .unwrap_or("<bos>")
            .to_owned(),
        _ => "<bos>".to_owned(),
    };
    let bos_id = tokenizer
        .token_to_id(&bos_str)
        .expect("BOS token not in vocab");

    eprintln!("[smoke_probe] bos='{bos_str}' id={bos_id}");

    // Load model via arch dispatch (not gemma4::load_from_path directly).
    let model = rmlx_models::arch::load_model(
        model_path,
        rmlx_mlx::Device::Cpu,
        &rmlx_models::arch::LoadOpts::default(),
    )
    .expect("arch::load_model should succeed for Gemma4ForConditionalGeneration");

    // CPU: generate_greedy re-encodes the full prefix each step (O(N²), no KV cache).
    // kv_quant_override = None: use the engine default (`DEFAULT_KV_QUANT`).
    // max_ctx_override = None: derive from model.max_position_embeddings.
    // A7.2: smoke probe is greedy (temperature 0.0).
    let smoke_sampler_cfg = rmlx_models::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    };
    let mut smoke_rng = rmlx_models::Pcg32::new(smoke_sampler_cfg.seed_or_default());
    let smoke_penalty_cfg = rmlx_models::PenaltyConfig::default();
    let mut smoke_token_history: Vec<u32> = Vec::new();
    let steps = model
        .generate_greedy(
            &tokenizer,
            &[bos_id],
            8,
            rmlx_mlx::Device::Cpu,
            None,
            None,
            1,   // smoke probe: single-slot cache
            &[], // smoke probe: no EOS-stop, force full 8 steps
            &mut |_| None,
            None, // A6.2: no sampler constraint in smoke probe.
            &smoke_sampler_cfg,
            &mut smoke_rng,
            &smoke_penalty_cfg,
            &mut smoke_token_history,
        )
        .expect("generate_greedy should not error");

    assert!(!steps.is_empty(), "expected at least 1 step");

    for (i, s) in steps.iter().enumerate() {
        eprintln!(
            "[smoke_probe] step {i}: id={} piece='{}' |max|={:.4} nan={}",
            s.token_id, s.piece, s.max_abs_logit, s.nan_count
        );
    }

    let verdict = rmlx_models::arch::Architecture::classify_smoke(&steps);
    eprintln!("[smoke_probe] verdict: {verdict:?}");

    // The Gemma-4 mxfp8 snapshot is expected to be clean.
    assert!(
        matches!(
            verdict,
            rmlx_models::SmokeVerdict::Ok | rmlx_models::SmokeVerdict::Inconclusive { .. }
        ),
        "expected Ok or Inconclusive verdict for known-clean snapshot, got {verdict:?}"
    );
}
