// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
//! Diagnostic harness for the Qwen3.6 graph bug.
//!
//! Two modes:
//! - `qwen36_diag <model-dir> [cpu|gpu]` — single forward, prints argmax+max_abs.
//! - `qwen36_diag <model-dir> <device> <N>` — greedy generates N tokens, prints decoded text.

use std::path::PathBuf;

use rmlx_mlx::{argmax, max_axis, Device};

const CANONICAL_IDS: &[u32] = &[
    248045, 8678, 198, 2523, 513, 10631, 13, 248046, 198, 248045, 846, 198, 12675, 248046, 198,
    248045, 74455, 198, 248068, 198,
];

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex::lock() cannot poison; Option/Result unwrap on values established by construction in this fn"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported variants; exhaustive expansion would require updating on every new variant"
)]
fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let model_dir: PathBuf = args
        .next()
        .ok_or_else(|| anyhow::anyhow!("usage: qwen36_diag <model-dir> [cpu|gpu] [N_GEN]"))?
        .into();
    let dev_str = args.next().unwrap_or_else(|| "cpu".to_owned());
    let device = match dev_str.as_str() {
        "gpu" => Device::Gpu,
        _ => Device::Cpu,
    };
    let n_gen: usize = args
        .next()
        .map_or(0, |s| s.parse().expect("N_GEN must be integer"));

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();

    eprintln!("loading model from {}", model_dir.display());
    let model = rmlx_models::qwen3_5_moe::load_from_path(&model_dir)
        .map_err(|e| anyhow::anyhow!("load: {e}"))?;

    if n_gen == 0 {
        eprintln!(
            "running forward_seq with {} prompt tokens on {:?}...",
            CANONICAL_IDS.len(),
            device
        );
        let logits = model
            .forward_seq(CANONICAL_IDS, device)
            .map_err(|e| anyhow::anyhow!("forward_seq: {e}"))?;

        let vocab = model.cfg.vocab_size as i32;
        let logits_flat = logits
            .reshape(&[1, vocab], device)
            .map_err(|e| anyhow::anyhow!("reshape: {e}"))?;
        logits_flat
            .eval()
            .map_err(|e| anyhow::anyhow!("eval: {e}"))?;

        let top = argmax(&logits_flat, -1, device).map_err(|e| anyhow::anyhow!("argmax: {e}"))?;
        top.eval().map_err(|e| anyhow::anyhow!("eval top: {e}"))?;
        let top_id = i32::from_le_bytes(
            top.to_bytes()
                .map_err(|e| anyhow::anyhow!("to_bytes top: {e}"))?[..4]
                .try_into()
                .unwrap(),
        ) as u32;

        let max_v = max_axis(&logits_flat, -1, device).map_err(|e| anyhow::anyhow!("max: {e}"))?;
        max_v.eval().map_err(|e| anyhow::anyhow!("eval max: {e}"))?;
        let max_bytes = max_v
            .to_bytes()
            .map_err(|e| anyhow::anyhow!("max bytes: {e}"))?;
        let max_f32 = match logits_flat.dtype() {
            rmlx_mlx::Dtype::F32 => f32::from_le_bytes(max_bytes[..4].try_into().unwrap()),
            rmlx_mlx::Dtype::Bf16 => {
                let raw = u16::from_le_bytes(max_bytes[..2].try_into().unwrap());
                f32::from_bits(u32::from(raw) << 16)
            }
            _ => 0.0_f32,
        };

        println!("argmax_id     = {top_id}");
        println!("max_abs_logit = {max_f32:.4}");
        println!("(mlx-lm baseline: argmax_id=8160 \"Here\", max_abs=29.75)");
        return Ok(());
    }

    // Greedy generation mode.
    eprintln!("greedy generating {n_gen} tokens from canonical 20-token prompt on {device:?}...");
    let tk_path = model_dir.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer: {e}"))?;

    let mut ids: Vec<u32> = Vec::with_capacity(n_gen);
    let mut step_fn = |s: &rmlx_models::gemma4::ProbeStep| -> Option<u32> {
        ids.push(s.token_id);
        None
    };
    // A7.2: diag binary is greedy (temperature 0.0).
    let diag_sampler_cfg = rmlx_models::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    };
    let mut diag_rng = rmlx_models::Pcg32::new(diag_sampler_cfg.seed_or_default());
    let diag_penalty_cfg = rmlx_models::PenaltyConfig::default();
    let mut diag_token_history: Vec<u32> = Vec::new();
    let _steps = rmlx_models::qwen3_5_moe::generate_greedy(
        &model,
        &tokenizer,
        CANONICAL_IDS,
        n_gen,
        device,
        rmlx_kv_quant::KvQuant::K8V8,
        None,
        1,   // diag binary: single-slot cache
        &[], // diag binary: no EOS-stop, force full n_gen steps
        &mut step_fn,
        None, // A6.2: no sampler constraint in diag.
        &diag_sampler_cfg,
        &mut diag_rng,
        &diag_penalty_cfg,
        &mut diag_token_history,
    )
    .map_err(|e| anyhow::anyhow!("generate_greedy: {e}"))?;

    println!("generated {} tokens: {:?}", ids.len(), ids);
    let decoded = tokenizer
        .decode(&ids, false)
        .map_err(|e| anyhow::anyhow!("decode: {e}"))?;
    println!("decoded text:\n---\n{decoded}\n---");

    Ok(())
}
