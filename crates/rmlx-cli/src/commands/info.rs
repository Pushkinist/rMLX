// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
//! `rmlx info` — dump architecture and quantization metadata for a model snapshot.
//!
//! [`run_info`] inspects a model directory without loading it into the MLX
//! runtime (no Metal context required). It parses `config.json`, resolves
//! the shard index, and prints:
//! - Architecture family and config fields.
//! - Per-tensor quantization kind (bf16, fp8, affine-4bit, TurboQuant, …).
//! - KV-quant capability matrix for the snapshot.
//! - Optionally runs a forward + decode smoke probe (`--probe`) to verify
//!   the snapshot produces coherent output.
//!
//! # Public API
//!
//! - [`run_info`] — main entry point.
//! - [`load_bos_id`] — resolve the BOS token id from tokenizer metadata.
//! - [`opt_u32`] — format an `Option<u32>` as `"N"` or `"—"` for display.

#![allow(clippy::cognitive_complexity, clippy::too_many_lines)]
use std::io::Write as _;
use std::path::Path;

use rmlx_loader::{
    count_tensors_per_shard, load_config, load_shard_index, resolve, resolve_paro, view, ShardSet,
    TensorKind,
};
use rmlx_metrics::events::{EventRecorder, Measurement};
use rmlx_mlx::{argmax, max_axis, Device, Dtype};
use rmlx_models::arch;
use rmlx_models::read_load_phases;
use rmlx_quant::{dequant_to_f32, MxFamily, MxParams};
use tracing::{info, warn};

/// Print arch + quant info for `model_path` -- no inference, no MLX runtime.
///
/// `device` is used for forward and smoke probes when enabled.
/// `kv_quant_override` is forwarded to `generate_greedy` when `probe_smoke` is true.
/// `None` = auto (arch default from `KvCacheBuilder`); `Some(q)` = explicit override.
/// `max_ctx_override` is forwarded to `generate_greedy` when `probe_smoke` is true.
/// `None` = derive from mpe (capped at 4096); `Some(n)` = use `n` directly.
///
/// Returns `true` when the smoke probe detects a broken snapshot (caller exits 1).
#[allow(
    clippy::unwrap_used,
    reason = "Mutex::lock() cannot poison; Option/Result unwrap on values established by construction in this fn"
)]
pub(crate) fn run_info(
    model_path: &Path,
    probe_forward: bool,
    probe_smoke: bool,
    device: Device,
    kv_quant_override: Option<rmlx_kv_quant::KvQuant>,
    max_ctx_override: Option<i32>,
    sink: &EventRecorder,
) -> anyhow::Result<bool> {
    let cfg = load_config(model_path).map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
    let idx = load_shard_index(model_path).map_err(|e| anyhow::anyhow!("load_shard_index: {e}"))?;

    // -- derived values --------------------------------------------------------
    let basename = model_path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("(unknown)");
    let abs_path = model_path
        .canonicalize()
        .unwrap_or_else(|_| model_path.to_path_buf());
    let abs_path_str = abs_path.to_string_lossy();

    let arch = cfg.architectures.join(", ");
    let dtype = cfg.dtype.as_deref().unwrap_or("—");

    // quant_mode for metrics: "<mode> g<group_size>" or "none".
    let quant_mode = match &cfg.quantization {
        Some(q) => format!("{} g{}", q.mode_or_default(), q.group_size),
        None => "none".to_owned(),
    };

    // ParoQuant detection: check `quantization_config.quant_method == "paroquant"`.
    let is_paro = cfg.is_paroquant();

    let quant_str = if is_paro {
        // For PARO checkpoints, derive quant string from `quantization_config`.
        let qc = cfg.quantization_config.as_ref().unwrap();
        let bits = qc.bits.unwrap_or(4);
        let group_size = qc.group_size.unwrap_or(128);
        let krot = qc.krot.unwrap_or(0);
        format!("paroquant int4 bits={bits} group_size={group_size} krot={krot}")
    } else {
        match &cfg.quantization {
            Some(q) => {
                let mode = q.mode_or_default();
                // Annotate when the "affine" default was applied (mode absent in JSON).
                if q.mode.is_none() {
                    format!(
                        "affine(default) bits={} group_size={}",
                        q.bits, q.group_size
                    )
                } else {
                    format!("{mode} bits={} group_size={}", q.bits, q.group_size)
                }
            }
            None => "—".to_owned(),
        }
    };

    // Resolve ParoQuant rotation state when the checkpoint is PARO.
    // krot_hint comes from `quantization_config.krot` to avoid shard header I/O.
    let paro_state = if is_paro {
        let krot_hint = cfg.quantization_config.as_ref().and_then(|qc| qc.krot);
        match resolve_paro(&idx, model_path, krot_hint) {
            Ok(state) => {
                info!(
                    paro_layers = state.layer_count(),
                    krot_max = state.krot_max(),
                    "run_info: ParoQuant state resolved"
                );
                Some(state)
            }
            Err(e) => {
                warn!(error = %e, "run_info: resolve_paro failed");
                None
            }
        }
    } else {
        None
    };

    let counts = count_tensors_per_shard(&idx);
    let total_tensors: usize = counts.values().sum();

    // -- sibling resolver ------------------------------------------------------
    let resolved = resolve(&idx).map_err(|e| anyhow::anyhow!("resolve: {e}"))?;

    // Determine whether mxfp-tagged tensors should be shown as nvfp4 based on config.
    let is_nvfp4 = cfg
        .quantization
        .as_ref()
        .is_some_and(|q| q.mode_or_default() == "nvfp4");

    let mut n_plain: usize = 0;
    let mut n_affine: usize = 0;
    let mut n_mxfp: usize = 0;
    let mut n_nvfp4: usize = 0;
    let mut n_paroquant: usize = 0;
    let mut n_unknown: usize = 0;

    for t in &resolved {
        match &t.kind {
            TensorKind::Plain => n_plain += 1,
            TensorKind::Affine => n_affine += 1,
            TensorKind::Mxfp => {
                if is_nvfp4 {
                    n_nvfp4 += 1;
                } else {
                    n_mxfp += 1;
                }
            }
            TensorKind::Nvfp4 => n_nvfp4 += 1,
            TensorKind::ParoQuant => n_paroquant += 1,
            TensorKind::Unknown => n_unknown += 1,
        }
    }

    // -- stdout summary --------------------------------------------------------
    // Acquire the lock once for the whole summary block — avoids ~25 independent
    // lock()+unlock() calls that `println!` would issue (perf-book ch 13 §locked-stdout).
    {
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        writeln!(out, "model: {basename}")?;
        writeln!(out, "path:  {}", abs_path.display())?;
        writeln!(out, "arch:  {arch}")?;
        writeln!(out, "dtype: {dtype}")?;
        writeln!(out, "quant: {quant_str}")?;

        // -- ParoQuant section (only when PARO tensors detected) ------------------
        if let Some(ref paro) = paro_state {
            let paro_layers = paro.layer_count();
            let krot_max = paro.krot_max();
            // Approximate storage: pairs (I16) + theta (F16) + channel_scales (F16)
            // We can only count tensor slots here; byte count needs shape info.
            let rotation_slots = paro.rotation_tensor_slots();
            writeln!(out, "paroquant_layers: {paro_layers}")?;
            writeln!(out, "paroquant_krot_max: {krot_max}")?;
            writeln!(out, "paroquant_rotation_tensor_slots: {rotation_slots}")?;
        }

        writeln!(out, "text:")?;
        if let Some(tc) = &cfg.text_config {
            let layers_str = opt_u32(tc.num_hidden_layers);
            let hidden_str = opt_u32(tc.hidden_size);
            let q_heads_str = opt_u32(tc.num_attention_heads);
            let kv_heads_str = opt_u32(tc.num_key_value_heads);
            let sw_str = opt_u32(tc.sliding_window);
            let max_seq_str = opt_u32(tc.max_position_embeddings);

            let gqa_str = match (tc.num_attention_heads, tc.num_key_value_heads) {
                (Some(q), Some(kv)) if kv > 0 => format!("{}:1", q / kv),
                _ => "—".to_owned(),
            };

            let head_dim_opt = cfg.head_dim();
            let head_dim_str = head_dim_opt.map_or_else(|| "—".to_owned(), |d| d.to_string());

            writeln!(out, "  layers:        {layers_str}")?;
            writeln!(out, "  hidden_size:   {hidden_str}")?;
            writeln!(out, "  q_heads:       {q_heads_str}")?;
            writeln!(out, "  kv_heads:      {kv_heads_str}")?;
            writeln!(out, "  gqa_ratio:     {gqa_str}")?;
            writeln!(out, "  head_dim:      {head_dim_str}")?;
            writeln!(out, "  sliding_window:{sw_str}")?;
            writeln!(out, "  max_seq_len:   {max_seq_str}")?;

            // structured tracing
            info!(
                model = basename,
                arch = %arch,
                dtype,
                quant = %quant_str,
                num_hidden_layers = ?tc.num_hidden_layers,
                hidden_size = ?tc.hidden_size,
                num_attention_heads = ?tc.num_attention_heads,
                num_key_value_heads = ?tc.num_key_value_heads,
                head_dim = ?head_dim_opt,
                sliding_window = ?tc.sliding_window,
                max_position_embeddings = ?tc.max_position_embeddings,
                total_tensors,
                "info summary"
            );
        } else {
            let head_dim_opt = cfg.head_dim();
            let head_dim_str = head_dim_opt.map_or_else(|| "—".to_owned(), |d| d.to_string());
            writeln!(out, "  (no text_config)")?;
            writeln!(out, "  head_dim:      {head_dim_str}")?;
            info!(
                model = basename,
                arch = %arch,
                dtype,
                quant = %quant_str,
                head_dim = ?head_dim_opt,
                total_tensors,
                "info summary"
            );
        }

        writeln!(out, "shards:")?;
        writeln!(out, "  count: {}", counts.len())?;
        for (shard, count) in &counts {
            writeln!(out, "  {shard} \u{2014} {count} tensors")?;
        }

        writeln!(out, "resolved:")?;
        writeln!(out, "  plain:     {n_plain}")?;
        writeln!(out, "  affine:    {n_affine}")?;
        writeln!(out, "  mxfp:      {n_mxfp}")?;
        if is_nvfp4 {
            writeln!(out, "  nvfp4:     {n_nvfp4}")?;
        }
        if n_paroquant > 0 {
            writeln!(out, "  paroquant: {n_paroquant}")?;
        }
        if n_unknown > 0 {
            writeln!(out, "  unknown:   {n_unknown}")?;
        }

        writeln!(out, "total_tensors: {total_tensors}")?;
    } // lock released here

    // -- metrics ---------------------------------------------------------------
    sink.record(&Measurement {
        model_path: &abs_path_str,
        quant_mode: &quant_mode,
        stage: "stage0",
        op: "total_tensors",
        value_unit: "count",
        value: total_tensors as f64,
        notes: "",
    })
    .map_err(|e| anyhow::anyhow!("metrics record: {e}"))?;

    for (shard, count) in &counts {
        sink.record(&Measurement {
            model_path: &abs_path_str,
            quant_mode: &quant_mode,
            stage: "stage0",
            op: "shard_tensors",
            value_unit: "count",
            value: *count as f64,
            notes: shard,
        })
        .map_err(|e| anyhow::anyhow!("metrics record: {e}"))?;
    }

    let resolved_metrics: &[(&str, usize)] = &[
        ("resolved_plain", n_plain),
        ("resolved_affine", n_affine),
        ("resolved_mxfp", n_mxfp),
        ("resolved_nvfp4", n_nvfp4),
        ("resolved_paroquant", n_paroquant),
        ("resolved_unknown", n_unknown),
    ];
    for (op, count) in resolved_metrics {
        sink.record(&Measurement {
            model_path: &abs_path_str,
            quant_mode: &quant_mode,
            stage: "stage0",
            op,
            value_unit: "count",
            value: *count as f64,
            notes: "",
        })
        .map_err(|e| anyhow::anyhow!("metrics record: {e}"))?;
    }

    // -- mxfp8 decode probe ----------------------------------------------------
    // Run only when config mode is mxfp8. Picks the first 3 Mxfp tensors
    // (alphabetically -- already ordered by BTreeMap), decodes the first row
    // of each, counts NaN cells, and emits one metric record per tensor.
    let is_mxfp8_model = cfg
        .quantization
        .as_ref()
        .is_some_and(|q| q.mode_or_default() == "mxfp8");

    if is_mxfp8_model {
        let probe_result = run_decode_probe(
            model_path,
            &idx,
            &resolved,
            &abs_path_str,
            &quant_mode,
            sink,
        );
        match probe_result {
            Ok(probe_status) => {
                println!("decode_probe: {probe_status}");
            }
            Err(e) => {
                // Probe failure is non-fatal for info -- report and continue.
                warn!(error = %e, "decode probe failed");
                println!("decode_probe: error ({e})");
            }
        }
    }

    // -- forward probe ---------------------------------------------------------
    if probe_forward {
        info!(arch = %arch, ?device, "forward_probe: loading model via arch::load_model");
        match arch::load_model(model_path, device, &arch::LoadOpts::default()) {
            Err(e) => {
                warn!(error = %e, "forward_probe: arch not supported or load failed");
                println!("forward_probe: skipped ({e})");
            }
            Ok(model) => {
                print_load_phases_json();
                // Use BOS token id=2 (Gemma4). Other arch BOS tokens resolved at load time.
                let bos_token = 2u32;
                info!("forward_probe: running single-token pass token={bos_token}");
                match run_forward_probe(&model, bos_token, device) {
                    Ok((top_id, max_logit)) => {
                        println!("forward_probe: ok  top_token={top_id}  max_logit={max_logit:.4}");
                        info!(top_id, max_logit, "forward_probe complete");

                        sink.record(&Measurement {
                            model_path: &abs_path_str,
                            quant_mode: &quant_mode,
                            stage: "stage1",
                            op: "forward_probe_top_token",
                            value_unit: "token_id",
                            value: f64::from(top_id),
                            notes: "",
                        })
                        .map_err(|e| anyhow::anyhow!("metrics record forward probe: {e}"))?;
                    }
                    Err(e) => {
                        warn!(error = %e, "forward_probe failed");
                        println!("forward_probe: error ({e})");
                    }
                }
            }
        }
    }

    // -- smoke probe ----------------------------------------------------------
    if probe_smoke {
        info!(arch = %arch, ?device, "smoke_probe: loading model via arch::load_model");

        // Load model via architecture dispatch -- returns Error::Model for unsupported arches.
        let model = match arch::load_model(model_path, device, &arch::LoadOpts::default()) {
            Ok(m) => m,
            Err(e) => {
                warn!(error = %e, "smoke_probe: arch not supported or load failed");
                println!("smoke_probe: skipped ({e})");
                return Ok(false);
            }
        };
        print_load_phases_json();

        // Load tokenizer.
        let bos_id = load_bos_id(model_path)?;
        info!(bos_id, "smoke_probe: resolved BOS token id");

        let tokenizer_path = model_path.join("tokenizer.json");
        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("smoke_probe: load tokenizer: {e}"))?;

        // Build the smoke prompt through the model's real chat template when one
        // exists (production-shaped, turn-structured input), falling back to the
        // shared bare seed otherwise. This keeps the probe verdict consistent
        // with how the model is actually served — a bare instruction can make a
        // healthy instruction-tuned snapshot degenerate into a filler loop (the
        // reference loader does the same), which previously raised false
        // Broken* verdicts. Single source of truth lives in chat_template.
        let prompt_ids =
            rmlx_server::chat_template::smoke_prompt_ids(model_path, &tokenizer, bos_id)
                .map_err(|e| anyhow::anyhow!("smoke_probe: build seed prompt: {e}"))?;

        info!(
            ?kv_quant_override,
            ?max_ctx_override,
            prompt_len = prompt_ids.len(),
            seed = arch::SMOKE_PROMPT,
            "smoke_probe: running 8-token greedy generation from seeded prompt"
        );
        // A7.2: smoke probe is always greedy (temperature 0.0).
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
                &prompt_ids,
                8,
                device,
                kv_quant_override,
                max_ctx_override,
                1,   // smoke probe uses single-slot cache (no multi-slot needed)
                &[], // smoke probe: no EOS-stop, force full 8 steps
                &mut |_| None,
                None, // A6.2: smoke probe never uses sampler constraints.
                &smoke_sampler_cfg,
                &mut smoke_rng,
                &smoke_penalty_cfg,
                &mut smoke_token_history,
            )
            .map_err(|e| anyhow::anyhow!("smoke_probe: generate_greedy: {e}"))?;

        let verdict = arch::Architecture::classify_smoke(&steps);

        // -- stdout output -------------------------------------------------------
        let verdict_str = match &verdict {
            rmlx_models::SmokeVerdict::Ok => "ok",
            rmlx_models::SmokeVerdict::BrokenPunctLoop { .. } => "broken_punct_loop",
            rmlx_models::SmokeVerdict::BrokenNan { .. } => "broken_nan",
            rmlx_models::SmokeVerdict::Inconclusive { .. } => "inconclusive",
        };

        println!("smoke_probe: {verdict_str}");
        println!(
            "  prompt: bos={bos_id} seed={:?} ({} tokens)",
            arch::SMOKE_PROMPT,
            prompt_ids.len()
        );
        println!("  generated:");
        for (i, s) in steps.iter().enumerate() {
            println!(
                "    step {i}: {} '{}'  |max|={:.4}  nan={}",
                s.token_id, s.piece, s.max_abs_logit, s.nan_count
            );
        }
        println!("  verdict: {verdict:?}");

        // -- metrics -----------------------------------------------------------
        let n_steps = steps.len();
        let max_abs_overall = steps
            .iter()
            .map(|s| s.max_abs_logit)
            .fold(0.0_f32, f32::max);
        let verdict_code: f64 = match &verdict {
            rmlx_models::SmokeVerdict::Ok => 0.0,
            rmlx_models::SmokeVerdict::BrokenPunctLoop { .. } => 1.0,
            rmlx_models::SmokeVerdict::BrokenNan { .. } => 2.0,
            rmlx_models::SmokeVerdict::Inconclusive { .. } => 3.0,
        };

        sink.record(&Measurement {
            model_path: &abs_path_str,
            quant_mode: &quant_mode,
            stage: "stage1",
            op: "smoke_steps",
            value_unit: "count",
            value: n_steps as f64,
            notes: "",
        })
        .map_err(|e| anyhow::anyhow!("metrics record smoke_steps: {e}"))?;

        sink.record(&Measurement {
            model_path: &abs_path_str,
            quant_mode: &quant_mode,
            stage: "stage1",
            op: "smoke_verdict",
            value_unit: "enum",
            value: verdict_code,
            notes: verdict_str,
        })
        .map_err(|e| anyhow::anyhow!("metrics record smoke_verdict: {e}"))?;

        sink.record(&Measurement {
            model_path: &abs_path_str,
            quant_mode: &quant_mode,
            stage: "stage1",
            op: "smoke_max_abs_logit",
            value_unit: "abs",
            value: f64::from(max_abs_overall),
            notes: "",
        })
        .map_err(|e| anyhow::anyhow!("metrics record smoke_max_abs_logit: {e}"))?;

        info!(verdict = %verdict_str, n_steps, max_abs_overall, "smoke_probe complete");

        // Exit 1 on broken snapshot.
        let is_broken = matches!(
            verdict,
            rmlx_models::SmokeVerdict::BrokenPunctLoop { .. }
                | rmlx_models::SmokeVerdict::BrokenNan { .. }
        );
        return Ok(is_broken);
    }

    Ok(false)
}

/// Run a single-token forward pass and return `(top_token_id, max_logit)`.
///
/// Uses `Architecture::forward_seq` to dispatch through the arch enum.
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
fn run_forward_probe(
    model: &arch::Architecture,
    token_id: u32,
    device: Device,
) -> anyhow::Result<(u32, f32)> {
    let vocab = model.vocab_size() as i32;
    let logits = model
        .forward_seq(&[token_id], device)
        .map_err(|e| anyhow::anyhow!("forward_seq: {e}"))?;

    let logits_flat = logits
        .reshape(&[1, vocab], device)
        .map_err(|e| anyhow::anyhow!("reshape logits: {e}"))?;
    logits_flat
        .eval()
        .map_err(|e| anyhow::anyhow!("eval logits: {e}"))?;

    let top = argmax(&logits_flat, -1, device).map_err(|e| anyhow::anyhow!("argmax: {e}"))?;
    top.eval().map_err(|e| anyhow::anyhow!("eval top: {e}"))?;

    let max_val =
        max_axis(&logits_flat, -1, device).map_err(|e| anyhow::anyhow!("max_axis: {e}"))?;
    max_val
        .eval()
        .map_err(|e| anyhow::anyhow!("eval max: {e}"))?;

    let top_bytes = top
        .to_bytes()
        .map_err(|e| anyhow::anyhow!("to_bytes top: {e}"))?;
    let top_id = i32::from_le_bytes(top_bytes[..4].try_into().unwrap()) as u32;

    let max_bytes = max_val
        .to_bytes()
        .map_err(|e| anyhow::anyhow!("to_bytes max: {e}"))?;
    let max_f32 = match logits_flat.dtype() {
        Dtype::F32 => f32::from_le_bytes(max_bytes[..4].try_into().unwrap()),
        Dtype::Bf16 => {
            let raw = u16::from_le_bytes(max_bytes[..2].try_into().unwrap());
            f32::from_bits(u32::from(raw) << 16)
        }
        _ => {
            warn!("unexpected logits dtype {:?}", logits_flat.dtype());
            0.0
        }
    };

    Ok((top_id, max_f32))
}

/// Extract the BOS token id from `tokenizer_config.json` + `tokenizer.json`.
///
/// Duplicates the minimal logic from `rmlx_server::tokenizer_io` to avoid
/// adding rmlx-server as a dependency of the CLI probe path.
pub(crate) fn load_bos_id(model_dir: &Path) -> anyhow::Result<u32> {
    // Read tokenizer_config.json and extract bos_token string.
    let cfg_path = model_dir.join("tokenizer_config.json");
    let data = std::fs::read(&cfg_path)
        .map_err(|e| anyhow::anyhow!("cannot read tokenizer_config.json: {e}"))?;
    let v: serde_json::Value = serde_json::from_slice(&data)
        .map_err(|e| anyhow::anyhow!("malformed tokenizer_config.json: {e}"))?;

    let extract = |key: &str| -> Option<String> {
        match v.get(key) {
            Some(serde_json::Value::String(s)) => Some(s.clone()),
            Some(serde_json::Value::Object(map)) => map
                .get("content")
                .and_then(|c| c.as_str())
                .map(str::to_owned),
            _ => None,
        }
    };

    let tk_path = model_dir.join("tokenizer.json");
    let tk = tokenizers::Tokenizer::from_file(&tk_path)
        .map_err(|e| anyhow::anyhow!("cannot load tokenizer.json: {e}"))?;

    // Fallback chain for tokenizers that don't define a BOS (e.g. Qwen3.6).
    // Smoke probe just needs any valid starting token.
    let candidates: Vec<String> = [
        extract("bos_token"),
        Some("<bos>".to_owned()),
        Some("<|im_start|>".to_owned()),
        extract("eos_token"),
        Some("<|endoftext|>".to_owned()),
    ]
    .into_iter()
    .flatten()
    .collect();

    for cand in &candidates {
        if let Some(id) = tk.token_to_id(cand) {
            tracing::debug!(token = %cand, id, "load_bos_id: resolved");
            return Ok(id);
        }
    }

    Err(anyhow::anyhow!(
        "no BOS/EOS-like token found in tokenizer vocab; tried: {candidates:?}"
    ))
}

/// Run the mxfp8 decode probe: decode the first row of up to 3 Mxfp tensors,
/// count NaN cells, emit metrics, return status string.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
fn run_decode_probe(
    model_path: &Path,
    idx: &rmlx_loader::ShardIndex,
    resolved: &[rmlx_loader::ResolvedTensor],
    abs_path_str: &str,
    quant_mode: &str,
    sink: &EventRecorder,
) -> anyhow::Result<String> {
    // Open all shards (mmap, lazy page-fault).
    let shards =
        ShardSet::open(model_path, idx).map_err(|e| anyhow::anyhow!("ShardSet::open: {e}"))?;

    // Collect first 3 Mxfp tensors (already sorted by BTreeMap/resolve).
    let mxfp_tensors: Vec<&rmlx_loader::ResolvedTensor> = resolved
        .iter()
        .filter(|t| t.kind == TensorKind::Mxfp)
        .take(3)
        .collect();

    if mxfp_tensors.is_empty() {
        return Ok("skipped (no mxfp tensors found)".to_owned());
    }

    let mut any_nan = false;
    let probed = mxfp_tensors.len();

    for tensor in &mxfp_tensors {
        let weight_name = format!("{}.weight", tensor.base_name);
        let scales_name = format!("{}.scales", tensor.base_name);

        let weight_view = view(&shards, idx, &weight_name)
            .map_err(|e| anyhow::anyhow!("view {weight_name}: {e}"))?;
        let scales_view = view(&shards, idx, &scales_name)
            .map_err(|e| anyhow::anyhow!("view {scales_name}: {e}"))?;

        // weight shape: [rows, cols] (mxfp8: 1 byte per element, dtype U8)
        // scales shape: [rows, cols/32]
        if weight_view.shape.len() != 2 || scales_view.shape.len() != 2 {
            warn!(
                name = tensor.base_name.as_str(),
                weight_shape = ?weight_view.shape,
                scales_shape = ?scales_view.shape,
                "unexpected tensor rank in decode probe -- skipping"
            );
            continue;
        }

        let rows = weight_view.shape[0];
        let cols = weight_view.shape[1];
        let group_size: usize = 32; // mxfp8 fixed
        let groups_per_row = cols / group_size;

        if rows == 0 || cols == 0 || cols % group_size != 0 {
            warn!(
                name = tensor.base_name.as_str(),
                rows, cols, "degenerate shape in decode probe -- skipping"
            );
            continue;
        }

        // Probe: decode first row only.
        // Slice: weight first row = cols bytes, scales first row = groups_per_row bytes.
        let weight_row = &weight_view.bytes[..cols];
        let scales_row = &scales_view.bytes[..groups_per_row];

        let params = MxParams {
            family: MxFamily::Mxfp8,
            rows: 1,
            cols,
        };
        let mut out = vec![0.0_f32; cols];
        dequant_to_f32(&params, weight_row, scales_row, &mut out)
            .map_err(|e| anyhow::anyhow!("dequant {}: {e}", tensor.base_name))?;

        let nan_count = out.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            warn!(
                tensor = tensor.base_name.as_str(),
                nan_count, "decode probe: NaN values detected"
            );
            any_nan = true;
        }

        sink.record(&Measurement {
            model_path: abs_path_str,
            quant_mode,
            stage: "stage1",
            op: "probe_decode_nan",
            value_unit: "count",
            value: nan_count as f64,
            notes: &tensor.base_name,
        })
        .map_err(|e| anyhow::anyhow!("metrics record probe: {e}"))?;
    }

    let status = if any_nan {
        format!("nan_detected ({probed} mxfp tensors, first row each)")
    } else {
        format!("ok ({probed} mxfp tensors, first row each)")
    };

    Ok(status)
}

pub(crate) fn opt_u32(v: Option<u32>) -> String {
    match v {
        Some(n) => n.to_string(),
        None => "—".to_owned(),
    }
}

/// Print a JSON load-phase timing line if `read_load_phases()` returns `Some`.
///
/// Output format (single line):
/// `{"load":{"mmap_ms":N,"dequant_ms":N,"gpu_residency_ms":N,"first_kernel_ready_ms":N,"total_load_ms":N}}`
fn print_load_phases_json() {
    if let Some(p) = read_load_phases() {
        println!(
            r#"{{"load":{{"mmap_ms":{},"dequant_ms":{},"gpu_residency_ms":{},"first_kernel_ready_ms":{},"total_load_ms":{}}}}}"#,
            p.mmap_ms, p.dequant_ms, p.gpu_residency_ms, p.first_kernel_ready_ms, p.total_load_ms
        );
    }
}
