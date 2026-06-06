// Fused-QK real-model proof probe.
//
// Usage:
//   RMLX_FUSED_QK=1 cargo run --profile release-perf -p rmlx-cli \
//     --example fused_qk_realmodel_probe -- <model_path> <codec> "<prompt>"
//
// Prints:
//   pre=N post=M delta=D  (fused_qk_total_dispatch_count before/after)
//   tps=X.XX              (decode-only tokens per second)
//   output: "<first ~10 tokens>"
//
// This is a CLI user-facing tool — `println!` is permitted per CLAUDE.md
// tracing rules (user-facing CLI output exemption).

#![allow(clippy::print_stdout, clippy::print_stderr, missing_docs)]

use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use rmlx_kv_quant::kvcache::fused_qk_total_dispatch_count;
use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

fn main() -> anyhow::Result<()> {
    // -- Tracing: stderr at debug level so dispatch + kvcache events are visible --
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "info,rmlx_kv_quant=debug,rmlx_models=debug".to_owned());
    fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_writer(std::io::stderr)
        .init();

    // -- Parse argv --
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: fused_qk_realmodel_probe <model_path> <codec> <prompt>");
        eprintln!("  codec: k8v4 | k8v8 | tsym3 | tsym4 | ...");
        std::process::exit(1);
    }
    let model_path = PathBuf::from(
        args.get(1)
            .ok_or_else(|| anyhow::anyhow!("missing argv[1]: model_path"))?,
    );
    let codec_str = args
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("missing argv[2]: codec"))?;
    let prompt_text = args
        .get(3)
        .ok_or_else(|| anyhow::anyhow!("missing argv[3]: prompt"))?;

    let kv_quant = KvQuant::from_str(codec_str)
        .map_err(|e| anyhow::anyhow!("bad codec '{codec_str}': {e}"))?;

    println!("model:  {}", model_path.display());
    println!("codec:  {kv_quant}");
    println!("prompt: {prompt_text}");
    println!();

    // -- Ensure RMLX_FUSED_QK is set (warn if not) --
    let fused_qk_on = rmlx_kv_quant::fused_qk_enabled();
    if !fused_qk_on {
        eprintln!("WARN: RMLX_FUSED_QK is not set to 1 — dispatch counter will stay 0 for q8/turbo codecs");
    }

    // -- Ensure GPU stream is registered on this thread --
    rmlx_mlx::ensure_gpu_default_stream();

    // -- Load model --
    eprintln!("loading model...");
    let model = arch::load_model(&model_path, Device::Gpu)
        .map_err(|e| anyhow::anyhow!("load_model: {e}"))?;
    eprintln!("model loaded.");

    // -- Load tokenizer --
    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    // -- Tokenize prompt --
    let encoding = tokenizer
        .encode(prompt_text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("prompt_tokens={}", prompt_ids.len());

    // -- Sampling config (greedy, temp=0) --
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();

    // -- Snapshot dispatch count BEFORE generation --
    let pre = fused_qk_total_dispatch_count();

    // -- Generate 64 tokens, capture per-token timing --
    let max_tokens: usize = 64;
    let ts_start = Instant::now();
    let mut first_cb_s: Option<f64> = None;
    let mut last_cb_s: f64 = 0.0;

    let mut step_fn = |_step: &rmlx_models::gemma4::ProbeStep| -> Option<u32> {
        let elapsed = ts_start.elapsed().as_secs_f64();
        if first_cb_s.is_none() {
            first_cb_s = Some(elapsed);
        }
        last_cb_s = elapsed;
        None // no early-stop signal
    };

    let steps = model
        .generate_greedy(
            &tokenizer,
            &prompt_ids,
            max_tokens,
            Device::Gpu,
            Some(kv_quant),
            None, // max_ctx: use default
            1,    // prompt_cache_slots: minimal
            &[],  // eos_ids: no early stop
            &mut step_fn,
            None, // constraint: none
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .map_err(|e| anyhow::anyhow!("generate_greedy: {e}"))?;

    // -- Snapshot dispatch count AFTER generation --
    let post = fused_qk_total_dispatch_count();

    // -- Compute decode-only TPS --
    let n_generated = steps.len();
    let decode_tps = if n_generated >= 2 {
        let decode_window = last_cb_s - first_cb_s.unwrap_or(0.0);
        if decode_window > 0.0 {
            (n_generated as f64 - 1.0) / decode_window
        } else {
            0.0
        }
    } else {
        0.0
    };

    // -- Build output snippet (first ~10 tokens) --
    let first_10: String = steps
        .iter()
        .take(10)
        .map(|s| s.piece.as_ref())
        .collect::<Vec<_>>()
        .join("");

    // -- Print structured results --
    println!("pre={pre} post={post} delta={}", post.wrapping_sub(pre));
    println!("tps={decode_tps:.2}");
    println!("n_generated={n_generated}");
    println!("output: \"{first_10}\"");

    Ok(())
}
