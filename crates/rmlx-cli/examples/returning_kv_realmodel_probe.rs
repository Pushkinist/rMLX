// Returning-KV real-model proof probe.
//
// Drives a model through prefill + 64-token decode and reports BOTH the
// TurboFlash dispatch counter and the fused-QK dispatch counter
// delta plus decode-only TPS. Use with Gemma4 to verify the returning-KV
// routing extension reaches the dispatch chain.
//
// Usage:
//   RMLX_TURBO_FLASH=1 cargo run --profile release-perf -p rmlx-cli \
//     --example returning_kv_realmodel_probe -- <model_path> <codec> <kernel> "<prompt>"
//
//   <kernel> : turbo_flash | fused_qk
//
// Both kernels can also be exercised simultaneously by setting both env
// vars; this binary uses <kernel> only as the "primary counter" reported
// in the delta line.
//
// Prints:
//   pre=N post=M delta=D  (selected kernel's counter)
//   tf_delta=X fqk_delta=Y (both counters' deltas)
//   tps=X.XX               (decode-only tokens per second)
//   output: "<first ~10 tokens>"

#![allow(clippy::print_stdout, clippy::print_stderr, missing_docs)]

use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use rmlx_kv_quant::kvcache::fused_qk_total_dispatch_count;
use rmlx_kv_quant::turbo_flash_msl::turbo_flash_dispatch_count;
use rmlx_kv_quant::KvQuant;
use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

fn main() -> anyhow::Result<()> {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = std::env::var("RUST_LOG")
        .unwrap_or_else(|_| "info,rmlx_kv_quant=debug,rmlx_models=debug".to_owned());
    fmt()
        .with_env_filter(EnvFilter::new(filter))
        .with_writer(std::io::stderr)
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.len() < 5 {
        eprintln!(
            "usage: returning_kv_realmodel_probe <model_path> <codec> <kernel> <prompt>\n  \
             codec : k8v4 | k8v8 | tsym3 | tsym4 | ...\n  \
             kernel: turbo_flash | fused_qk"
        );
        std::process::exit(1);
    }
    let model_path = PathBuf::from(
        args.get(1)
            .ok_or_else(|| anyhow::anyhow!("missing argv[1]: model_path"))?,
    );
    let codec_str = args
        .get(2)
        .ok_or_else(|| anyhow::anyhow!("missing argv[2]: codec"))?;
    let kernel_str = args
        .get(3)
        .ok_or_else(|| anyhow::anyhow!("missing argv[3]: kernel"))?;
    let prompt_text = args
        .get(4)
        .ok_or_else(|| anyhow::anyhow!("missing argv[4]: prompt"))?;

    let kv_quant = KvQuant::from_str(codec_str)
        .map_err(|e| anyhow::anyhow!("bad codec '{codec_str}': {e}"))?;

    println!("model:  {}", model_path.display());
    println!("codec:  {kv_quant}");
    println!("kernel: {kernel_str}");
    println!("prompt: {prompt_text}");
    println!();

    let tf_on = rmlx_kv_quant::turbo_flash_msl::turbo_flash_enabled();
    let fqk_on = rmlx_kv_quant::fused_qk_enabled();
    eprintln!("env: RMLX_TURBO_FLASH={tf_on} RMLX_FUSED_QK={fqk_on}");

    rmlx_mlx::ensure_gpu_default_stream();

    eprintln!("loading model...");
    let model = arch::load_model(&model_path, Device::Gpu, &arch::LoadOpts::default())
        .map_err(|e| anyhow::anyhow!("load_model: {e}"))?;
    eprintln!("model loaded.");

    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path)
        .map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;

    let encoding = tokenizer
        .encode(prompt_text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("encode: {e}"))?;
    let prompt_ids: Vec<u32> = encoding.get_ids().to_vec();
    eprintln!("prompt_tokens={}", prompt_ids.len());

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

    let tf_pre = turbo_flash_dispatch_count();
    let fqk_pre = fused_qk_total_dispatch_count();

    let max_tokens: usize = 64;
    let ts_start = Instant::now();
    let mut first_cb_s: Option<f64> = None;
    let mut last_cb_s: f64 = 0.0;

    let mut step_fn = |_step: &rmlx_models::ProbeStep| -> Option<u32> {
        let elapsed = ts_start.elapsed().as_secs_f64();
        if first_cb_s.is_none() {
            first_cb_s = Some(elapsed);
        }
        last_cb_s = elapsed;
        None
    };

    let steps = model
        .generate_greedy(
            &tokenizer,
            &prompt_ids,
            max_tokens,
            Device::Gpu,
            Some(kv_quant),
            None,
            1,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .map_err(|e| anyhow::anyhow!("generate_greedy: {e}"))?;

    let tf_post = turbo_flash_dispatch_count();
    let fqk_post = fused_qk_total_dispatch_count();
    let tf_delta = tf_post.wrapping_sub(tf_pre);
    let fqk_delta = fqk_post.wrapping_sub(fqk_pre);

    let (pre, post, delta) = match kernel_str.as_str() {
        "turbo_flash" => (tf_pre, tf_post, tf_delta),
        "fused_qk" => (fqk_pre, fqk_post, fqk_delta),
        other => {
            eprintln!("WARN: unknown kernel '{other}' — reporting turbo_flash by default");
            (tf_pre, tf_post, tf_delta)
        }
    };

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

    let first_10: String = steps
        .iter()
        .take(10)
        .map(|s| s.piece.as_ref())
        .collect::<Vec<_>>()
        .join("");

    println!("pre={pre} post={post} delta={delta}");
    println!("tf_delta={tf_delta} fqk_delta={fqk_delta}");
    println!("tps={decode_tps:.2}");
    println!("n_generated={n_generated}");
    println!("output: \"{first_10}\"");

    Ok(())
}
