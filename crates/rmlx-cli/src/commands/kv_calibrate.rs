// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::map_unwrap_or,
    clippy::needless_range_loop,
    clippy::manual_is_multiple_of
)]
//! `rmlx kv-calibrate` — KV calibration passes.
//!
//! Two recipes:
//!
//! * **Weight-norm recipes** (`turbo2`, `turbo2_tcq`, `turbo3`, `turbo3_tcq`,
//!   `turbo4`) — CPU-only walks safetensors weight shards to compute per-head
//!   L2 norms for `k_proj.weight` and `v_proj.weight`, then selects the top-K
//!   high-precision indices per head. Writes `kv_calib.json`. No MLX
//!   allocation, no Metal claim. Safe to run alongside a live `rmlx serve`.
//!
//!   Works on both float (F32 / BF16 / F16) **and quantized** snapshots. For
//!   quantized models the K/V projection weights are U32-packed; the
//!   weight-norm pass dequantizes them to f32 first via the shared
//!   `rmlx_quant` codecs (affine + mxfp8/mxfp4/nvfp4) before computing norms.
//!   Detection is structural — the sibling `<base>.scales` tensor is present —
//!   so no flag is required.
//!
//! * **Head-budget recipe** (`head_budget`) — loads the model on GPU, prefills
//!   each calibration prompt, walks each layer's KV cache for the accumulated
//!   bf16 K buffer, computes per-(layer, head) k-budgets that cover the
//!   requested cumulative softmax-mass threshold under a K-norm² proxy, and
//!   writes `head_budgets.json` per the `rmlx-loader::head_budgets` schema.
//!   Requires the Metal claim (single-MLX rule).
//!
//! # Public API
//!
//! - [`run_kv_calibrate`] — main entry point.

use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Context;
use rmlx_loader::{
    calibrate_model, load_config, recipe_to_internal, write_head_budgets, write_kv_calibration,
    HeadBudgetCalibration, HeadBudgets,
};
use sha2::{Digest, Sha256};
use tracing::{info, instrument, warn};

use crate::commands::calibration_softmax::SoftmaxMassSink;

/// Default mass coverage threshold for the head_budget recipe.
const DEFAULT_HEAD_BUDGET_MASS_THRESHOLD: f32 = 0.95;

/// Default per-(layer, head) floor for the `softmax_mass` recipe. Must match
/// the `default_value_t` for `KvCalibrate::target_mass_budget_floor` in
/// `crates/rmlx-cli/src/main.rs`; the LOW-2 warn compares against this so the
/// silent-no-op gate only fires when the operator explicitly overrode it.
const DEFAULT_TARGET_MASS_BUDGET_FLOOR: u32 = 16;

/// Per-prompt token cap. Heuristic: keeps calibration under ~2 min on 7B-class
/// models even with long-context prompts (8k).
const HEAD_BUDGET_MAX_TOKENS_PER_PROMPT: usize = 768;

/// Wider per-prompt cap for the true softmax-mass recipe. Long-context
/// prompts cap at 8k tokens; budgets reflect realistic NIAH retrieval depth.
const SOFTMAX_MASS_MAX_TOKENS_PER_PROMPT: usize = 8192;

/// Run the KV calibration pass.
///
/// - `model_dir`: path to the MLX model snapshot (contains `config.json` + safetensors).
/// - `recipe`: one of `turbo2`, `turbo2_tcq`, `turbo3`, `turbo3_tcq`, `turbo4`,
///   or `head_budget` (softmax-mass per-(layer, head) budgets).
/// - `out`: output path for `kv_calib.json` (or `head_budgets.json` when the
///   `head_budget` recipe is selected); defaults to `<model_dir>/<filename>`.
/// - `prompts`: optional override prompt set for the `head_budget` recipe.
///   Ignored for weight-norm recipes. When `None`, defaults to
///   `prompts/calibration_default.json` from the workspace.
/// - `mass_threshold`: softmax-mass coverage target for the `head_budget`
///   recipe (default 0.95). Ignored for weight-norm recipes.
pub(crate) fn run_kv_calibrate(
    model_dir: &Path,
    recipe: &str,
    out: Option<&Path>,
    prompts: Option<&Path>,
    mass_threshold: Option<f32>,
    target_mass_budget_floor: u32,
) -> anyhow::Result<()> {
    // `--target-mass-budget-floor` is only honoured by the `softmax_mass`
    // recipe. Warn loudly when an operator supplies a non-default value
    // alongside a recipe that drops it silently.
    if recipe != "softmax_mass" && target_mass_budget_floor != DEFAULT_TARGET_MASS_BUDGET_FLOOR {
        warn!(
            value = target_mass_budget_floor,
            recipe = recipe,
            "RMLX_TARGET_MASS_BUDGET_FLOOR: ignored for non-softmax_mass recipe"
        );
    }
    match recipe {
        "head_budget" | "k_norm_proxy" => {
            return run_head_budget(
                model_dir,
                out,
                prompts,
                mass_threshold.unwrap_or(DEFAULT_HEAD_BUDGET_MASS_THRESHOLD),
                recipe,
            );
        }
        "softmax_mass" => {
            return run_softmax_mass(
                model_dir,
                out,
                prompts,
                mass_threshold.unwrap_or(DEFAULT_HEAD_BUDGET_MASS_THRESHOLD),
                target_mass_budget_floor,
            );
        }
        _ => {}
    }

    let t_start = Instant::now();

    let internal_recipe =
        recipe_to_internal(recipe).map_err(|e| anyhow::anyhow!("kv-calibrate: {e}"))?;

    println!(
        "rmlx kv-calibrate: loading config from {}",
        model_dir.display()
    );

    let calib = calibrate_model(model_dir, recipe, internal_recipe)
        .map_err(|e| anyhow::anyhow!("kv-calibrate: {e}"))?;

    let out_path: PathBuf = match out {
        Some(p) => p.to_path_buf(),
        None => model_dir.join("kv_calib.json"),
    };

    write_kv_calibration(&out_path, &calib).map_err(|e| anyhow::anyhow!("kv-calibrate: {e}"))?;

    let elapsed = t_start.elapsed();
    let size_kb = std::fs::metadata(&out_path).map_or(0.0, |m| m.len() as f64 / 1024.0);

    info!(
        path = %out_path.display(),
        size_kb,
        elapsed_secs = elapsed.as_secs_f64(),
        layer_count = calib.layers.len(),
        "kv-calibrate: output written"
    );
    println!(
        "\nSaved {} ({size_kb:.1} KB, {elapsed:.1?})",
        out_path.display()
    );
    println!("  {} layers calibrated for {recipe}", calib.layers.len());

    Ok(())
}

// ── head_budget recipe ────────────────────────────────────────────────────────

/// Default prompt-set path. Walks up from cwd to find `<repo>/prompts/<basename>`;
/// falls back to `<model_dir>/<basename>`.
fn resolve_prompts_path(model_dir: &Path, basename: &str) -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let mut probe: &Path = &cwd;
    loop {
        let candidate = probe.join("prompts").join(basename);
        if candidate.is_file() {
            return Some(candidate);
        }
        match probe.parent() {
            Some(p) => probe = p,
            None => break,
        }
    }
    let beside = model_dir.join(basename);
    if beside.is_file() {
        return Some(beside);
    }
    None
}

fn default_prompts_path(model_dir: &Path) -> Option<PathBuf> {
    resolve_prompts_path(model_dir, "calibration_default.json")
}

/// Long-context calibration prompt set; falls back to the legacy
/// `calibration_default.json` when the long-context file is absent.
fn softmax_mass_default_prompts_path(model_dir: &Path) -> Option<PathBuf> {
    resolve_prompts_path(model_dir, "calibration_long_context.json")
        .or_else(|| resolve_prompts_path(model_dir, "calibration_default.json"))
}

/// Schema for the calibration prompt JSON file.
#[derive(serde::Deserialize)]
struct PromptSet {
    #[allow(dead_code)]
    version: u32,
    #[serde(default)]
    description: String,
    prompts: Vec<String>,
}

#[instrument(skip_all, fields(model = %model_dir.display(), mass_threshold, recipe_label))]
fn run_head_budget(
    model_dir: &Path,
    out: Option<&Path>,
    prompts_path: Option<&Path>,
    mass_threshold: f32,
    recipe_label: &str,
) -> anyhow::Result<()> {
    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        anyhow::bail!(
            "kv-calibrate --recipe head_budget: {} is missing config.json",
            model_dir.display()
        );
    }
    if !(0.50..=1.00).contains(&mass_threshold) {
        anyhow::bail!(
            "kv-calibrate --recipe head_budget: --mass-threshold must be in [0.50, 1.00], got {mass_threshold}"
        );
    }

    let prompts_pb: PathBuf = match prompts_path {
        Some(p) => p.to_path_buf(),
        None => default_prompts_path(model_dir).ok_or_else(|| {
            anyhow::anyhow!(
                "kv-calibrate --recipe head_budget: no --prompts supplied and \
                 prompts/calibration_default.json not found by walking up from cwd"
            )
        })?,
    };
    let prompts_bytes = std::fs::read(&prompts_pb)
        .with_context(|| format!("read prompts file {}", prompts_pb.display()))?;
    let prompt_set: PromptSet = serde_json::from_slice(&prompts_bytes)
        .with_context(|| format!("parse prompts file {}", prompts_pb.display()))?;
    if prompt_set.prompts.is_empty() {
        anyhow::bail!("kv-calibrate --recipe head_budget: prompt set is empty");
    }
    let prompt_set_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(&prompts_bytes);
        let digest = hasher.finalize();
        digest.iter().fold(String::with_capacity(64), |mut s, b| {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
            s
        })
    };

    let out_path: PathBuf = match out {
        Some(p) => p.to_path_buf(),
        None => model_dir.join("head_budgets.json"),
    };

    println!(
        "rmlx kv-calibrate --recipe head_budget:\n  model:     {}\n  prompts:   {} ({} prompts)\n  threshold: {mass_threshold}\n  output:    {}\n",
        model_dir.display(),
        prompts_pb.display(),
        prompt_set.prompts.len(),
        out_path.display(),
    );
    if !prompt_set.description.is_empty() {
        println!("  description: {}\n", prompt_set.description);
    }

    // Architecture gate: only Qwen3 (Bonsai is the smoke target) is currently wired.
    let cfg = load_config(model_dir).map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
    let arch_name = cfg.architectures.first().cloned().unwrap_or_default();
    if arch_name != "Qwen3ForCausalLM" {
        anyhow::bail!(
            "kv-calibrate --recipe head_budget: architecture '{arch_name}' is not yet wired \
             (Qwen3ForCausalLM only — Bonsai smoke target). \
             Adding Gemma4 / Qwen3.5MoE / Qwen3VL is follow-up work."
        );
    }

    // Tokenise prompts up-front (cheap; before the heavy model load).
    let tokenizer = load_tokenizer(model_dir)?;
    let mut tokenised: Vec<Vec<u32>> = Vec::with_capacity(prompt_set.prompts.len());
    for (i, p) in prompt_set.prompts.iter().enumerate() {
        let enc = tokenizer
            .encode(p.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenize prompt {i}: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.len() > HEAD_BUDGET_MAX_TOKENS_PER_PROMPT {
            warn!(
                prompt_idx = i,
                original = ids.len(),
                cap = HEAD_BUDGET_MAX_TOKENS_PER_PROMPT,
                "head_budget: prompt longer than per-prompt token cap; truncating"
            );
            ids.truncate(HEAD_BUDGET_MAX_TOKENS_PER_PROMPT);
        }
        if ids.len() < 2 {
            warn!(prompt_idx = i, "head_budget: prompt <2 tokens; skipping");
            continue;
        }
        tokenised.push(ids);
    }
    if tokenised.is_empty() {
        anyhow::bail!("kv-calibrate --recipe head_budget: no usable prompts after tokenisation");
    }

    let t_load = Instant::now();
    let model = load_qwen3_for_calibration(model_dir)?;
    let load_secs = t_load.elapsed().as_secs_f64();
    info!(
        load_secs,
        num_layers = model.cfg.num_hidden_layers,
        num_heads = model.cfg.num_attention_heads,
        num_kv_heads = model.cfg.num_key_value_heads,
        "head_budget: model loaded"
    );
    println!(
        "model loaded ({load_secs:.1}s): layers={} q_heads={} kv_heads={} head_dim={}",
        model.cfg.num_hidden_layers,
        model.cfg.num_attention_heads,
        model.cfg.num_key_value_heads,
        model.cfg.head_dim,
    );

    let t_measure = Instant::now();
    let measurement = measure_head_budgets_qwen3(&model, &tokenised, mass_threshold)
        .map_err(|e| anyhow::anyhow!("measure_head_budgets_qwen3: {e}"))?;
    let measure_secs = t_measure.elapsed().as_secs_f64();
    info!(
        measure_secs,
        prompts = tokenised.len(),
        max_seq_len = measurement.max_seq_len,
        "head_budget: measurement complete"
    );

    let model_name = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_dir.display().to_string());
    let num_layers = measurement.per_layer_per_head_budget.len();
    let num_heads = measurement
        .per_layer_per_head_budget
        .first()
        .map_or(0, Vec::len);
    // Schema label identifies the *concept* (per-(layer, head) cumulative mass
    // coverage); the K-norm² proxy lives behind this recipe arm. See
    // `docs/KV_QUANT.md` "Sparse attention" for the gap discussion.
    let calibration = HeadBudgetCalibration::new(
        "softmax_mass".to_string(),
        prompt_set_sha256,
        tokenised.len() as u32,
        measurement.max_seq_len,
        mass_threshold,
    );
    let _ = recipe_label; // v1 file: no `recipe` field on disk.
    let budgets = HeadBudgets::new(
        model_name,
        num_layers,
        num_heads,
        calibration,
        measurement.per_layer_per_head_budget,
    );

    write_head_budgets(&out_path, &budgets).map_err(|e| anyhow::anyhow!("write: {e}"))?;
    let size_kb = std::fs::metadata(&out_path).map_or(0.0, |m| m.len() as f64 / 1024.0);

    let stats = summarise_budget_table(&budgets);
    info!(
        path = %out_path.display(),
        size_kb,
        load_secs,
        measure_secs,
        num_layers = budgets.num_layers,
        num_heads = budgets.num_heads,
        median_budget = stats.median,
        min_budget = stats.min,
        max_budget = stats.max,
        "head_budget: head_budgets.json written"
    );
    println!(
        "\nSaved {} ({size_kb:.1} KB)\n  num_layers={}, num_heads={}\n  budget stats: median={}, min={}, max={}\n  measurement secs: {measure_secs:.1}",
        out_path.display(),
        budgets.num_layers,
        budgets.num_heads,
        stats.median,
        stats.min,
        stats.max,
    );

    Ok(())
}

// ── softmax_mass recipe ───────────────────────────────────────────────────────

#[instrument(skip_all, fields(model = %model_dir.display(), mass_threshold, floor))]
fn run_softmax_mass(
    model_dir: &Path,
    out: Option<&Path>,
    prompts_path: Option<&Path>,
    mass_threshold: f32,
    floor: u32,
) -> anyhow::Result<()> {
    let config_path = model_dir.join("config.json");
    if !config_path.is_file() {
        anyhow::bail!(
            "kv-calibrate --recipe softmax_mass: {} is missing config.json",
            model_dir.display()
        );
    }
    if !(0.50..=1.00).contains(&mass_threshold) {
        anyhow::bail!(
            "kv-calibrate --recipe softmax_mass: --mass-threshold must be in [0.50, 1.00], got {mass_threshold}"
        );
    }
    if floor == 0 {
        anyhow::bail!(
            "kv-calibrate --recipe softmax_mass: --target-mass-budget-floor must be >= 1"
        );
    }

    let prompts_pb: PathBuf = match prompts_path {
        Some(p) => p.to_path_buf(),
        None => softmax_mass_default_prompts_path(model_dir).ok_or_else(|| {
            anyhow::anyhow!(
                "kv-calibrate --recipe softmax_mass: no --prompts supplied and \
                 prompts/calibration_long_context.json (or calibration_default.json) \
                 not found by walking up from cwd"
            )
        })?,
    };
    let prompts_bytes = std::fs::read(&prompts_pb)
        .with_context(|| format!("read prompts file {}", prompts_pb.display()))?;
    let prompt_set: PromptSet = serde_json::from_slice(&prompts_bytes)
        .with_context(|| format!("parse prompts file {}", prompts_pb.display()))?;
    if prompt_set.prompts.is_empty() {
        anyhow::bail!("kv-calibrate --recipe softmax_mass: prompt set is empty");
    }
    let prompt_set_sha256 = {
        let mut hasher = Sha256::new();
        hasher.update(&prompts_bytes);
        let digest = hasher.finalize();
        digest.iter().fold(String::with_capacity(64), |mut s, b| {
            use std::fmt::Write as _;
            let _ = write!(s, "{b:02x}");
            s
        })
    };
    let prompts_basename: String = prompts_pb
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let out_path: PathBuf = match out {
        Some(p) => p.to_path_buf(),
        None => model_dir.join("head_budgets.json"),
    };

    println!(
        "rmlx kv-calibrate --recipe softmax_mass:\n  model:     {}\n  prompts:   {} ({} prompts)\n  target:    {mass_threshold}\n  floor:     {floor}\n  output:    {}\n",
        model_dir.display(),
        prompts_pb.display(),
        prompt_set.prompts.len(),
        out_path.display(),
    );
    if !prompt_set.description.is_empty() {
        println!("  description: {}\n", prompt_set.description);
    }

    let cfg = load_config(model_dir).map_err(|e| anyhow::anyhow!("load_config: {e}"))?;
    let arch_name = cfg.architectures.first().cloned().unwrap_or_default();
    if arch_name != "Qwen3ForCausalLM" {
        anyhow::bail!(
            "kv-calibrate --recipe softmax_mass: architecture '{arch_name}' is not yet wired \
             (Qwen3ForCausalLM only — Bonsai smoke target). \
             Adding Gemma4 / Qwen3.5MoE / Qwen3VL is follow-up work."
        );
    }

    // Tokenise prompts up-front.
    let tokenizer = load_tokenizer(model_dir)?;
    let mut tokenised: Vec<Vec<u32>> = Vec::with_capacity(prompt_set.prompts.len());
    for (i, p) in prompt_set.prompts.iter().enumerate() {
        let enc = tokenizer
            .encode(p.as_str(), true)
            .map_err(|e| anyhow::anyhow!("tokenize prompt {i}: {e}"))?;
        let mut ids: Vec<u32> = enc.get_ids().to_vec();
        if ids.len() > SOFTMAX_MASS_MAX_TOKENS_PER_PROMPT {
            warn!(
                prompt_idx = i,
                original = ids.len(),
                cap = SOFTMAX_MASS_MAX_TOKENS_PER_PROMPT,
                "softmax_mass: prompt longer than per-prompt token cap; truncating"
            );
            ids.truncate(SOFTMAX_MASS_MAX_TOKENS_PER_PROMPT);
        }
        if ids.len() < 2 {
            warn!(prompt_idx = i, "softmax_mass: prompt <2 tokens; skipping");
            continue;
        }
        tokenised.push(ids);
    }
    if tokenised.is_empty() {
        anyhow::bail!("kv-calibrate --recipe softmax_mass: no usable prompts after tokenisation");
    }

    let t_load = Instant::now();
    let model = load_qwen3_for_calibration(model_dir)?;
    let load_secs = t_load.elapsed().as_secs_f64();
    let num_layers = model.cfg.num_hidden_layers;
    let n_q_heads = model.cfg.num_attention_heads;
    let n_kv_heads = model.cfg.num_key_value_heads;
    let head_dim = model.cfg.head_dim;
    if n_q_heads % n_kv_heads != 0 {
        anyhow::bail!(
            "kv-calibrate --recipe softmax_mass: GQA mismatch n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}"
        );
    }
    info!(
        load_secs,
        num_layers, n_q_heads, n_kv_heads, head_dim, "softmax_mass: model loaded"
    );
    println!(
        "model loaded ({load_secs:.1}s): layers={num_layers} q_heads={n_q_heads} kv_heads={n_kv_heads} head_dim={head_dim}"
    );

    let mut sink = SoftmaxMassSink::new(
        num_layers,
        n_q_heads,
        n_kv_heads,
        head_dim,
        mass_threshold,
        floor,
    );

    let t_measure = Instant::now();
    let device = Device::Gpu;
    for (prompt_idx, ids) in tokenised.iter().enumerate() {
        let seq = ids.len();
        let mut caches: Vec<KvCache> = (0..num_layers)
            .map(|_| KvCache::with_quant_max_seq(KvQuant::None, seq as i32))
            .collect();
        let t_prefill = Instant::now();
        let _logits = model
            .forward_seq_with_cache_calibrated(ids, Some(&mut caches), &mut sink, device)
            .map_err(|e| anyhow::anyhow!("softmax_mass prefill prompt {prompt_idx}: {e}"))?;
        let prefill_secs = t_prefill.elapsed().as_secs_f64();
        info!(
            prompt_idx,
            seq, prefill_secs, "softmax_mass: prompt prefill complete"
        );
    }
    let measure_secs = t_measure.elapsed().as_secs_f64();
    let max_seq_len = sink.max_seq_len;

    let per_layer_per_head_budget = sink.expand_to_q_heads();
    let num_heads = per_layer_per_head_budget.first().map_or(0, Vec::len);
    let model_name = model_dir
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model_dir.display().to_string());
    let calibration = HeadBudgetCalibration::new_v2(
        "softmax_mass".to_string(),
        prompt_set_sha256,
        tokenised.len() as u32,
        max_seq_len,
        mass_threshold,
        "softmax_mass".to_string(),
        mass_threshold,
        floor,
        vec![prompts_basename],
    );
    let budgets = HeadBudgets::new_v2(
        model_name,
        num_layers,
        num_heads,
        calibration,
        per_layer_per_head_budget,
    );

    write_head_budgets(&out_path, &budgets).map_err(|e| anyhow::anyhow!("write: {e}"))?;
    let size_kb = std::fs::metadata(&out_path).map_or(0.0, |m| m.len() as f64 / 1024.0);

    let stats = summarise_budget_table(&budgets);
    info!(
        path = %out_path.display(),
        size_kb,
        load_secs,
        measure_secs,
        num_layers = budgets.num_layers,
        num_heads = budgets.num_heads,
        median_budget = stats.median,
        min_budget = stats.min,
        max_budget = stats.max,
        "softmax_mass: head_budgets.json (v2) written"
    );
    println!(
        "\nSaved {} ({size_kb:.1} KB)\n  schema v2 (recipe=softmax_mass, target_mass={mass_threshold}, floor={floor})\n  num_layers={}, num_heads={}\n  budget stats: median={}, min={}, max={}\n  measurement secs: {measure_secs:.1}",
        out_path.display(),
        budgets.num_layers,
        budgets.num_heads,
        stats.median,
        stats.min,
        stats.max,
    );

    Ok(())
}

struct BudgetStats {
    median: u32,
    min: u32,
    max: u32,
}

fn summarise_budget_table(b: &HeadBudgets) -> BudgetStats {
    let mut all: Vec<u32> = b
        .per_layer_per_head_budget
        .iter()
        .flat_map(|row| row.iter().copied())
        .collect();
    if all.is_empty() {
        return BudgetStats {
            median: 0,
            min: 0,
            max: 0,
        };
    }
    all.sort_unstable();
    let len = all.len();
    BudgetStats {
        median: all.get(len / 2).copied().unwrap_or(0),
        min: all.first().copied().unwrap_or(0),
        max: all.last().copied().unwrap_or(0),
    }
}

fn load_tokenizer(model_dir: &Path) -> anyhow::Result<tokenizers::Tokenizer> {
    let tk_path = model_dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&tk_path)
        .map_err(|e| anyhow::anyhow!("load tokenizer.json: {e}"))
}

// ── Model load + softmax-mass measurement ────────────────────────────────────

use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device, Dtype};
use rmlx_models::arch;
use rmlx_models::arch::Architecture;
use rmlx_models::qwen3::Qwen3Text;

/// Load the Qwen3 model on GPU. Acquires the single-MLX claim transiently.
fn load_qwen3_for_calibration(model_dir: &Path) -> anyhow::Result<Qwen3Text> {
    // Single-MLX claim — port 0 = default kv-calibrate runtime slot.
    // Re-uses the runtime claim that `serve` would own; if a serve is live
    // the calibration must wait or the operator must stop it (hard rule 8).
    let device = Device::Gpu;
    let _claim = crate::commands::parse::acquire_claim_for_device(device, 0)?;

    let arch = arch::load_model(model_dir, device, &arch::LoadOpts::default())
        .map_err(|e| anyhow::anyhow!("load_model: {e}"))?;
    let arch_name = arch.arch_class();
    if let Architecture::Qwen3(m) = arch {
        Ok(m)
    } else {
        anyhow::bail!("head_budget: expected Qwen3 architecture; got {arch_name}")
    }
}

/// Per-(layer, head) budget table built up across all calibration prompts.
struct HeadBudgetMeasurement {
    /// `[num_layers][num_heads]` table (Q-heads — GQA-expanded from KV-heads).
    per_layer_per_head_budget: Vec<Vec<u32>>,
    /// Maximum prompt sequence length observed (post-tokenisation).
    max_seq_len: u32,
}

/// Run prefill + per-(layer, head) softmax-mass measurement for Qwen3.
///
/// Each prompt is tokenised, a fresh bf16 KV cache is allocated (`KvQuant::None`
/// so the K accumulator is exposed via `KvCache::calibration_k_bf16`), and the
/// model runs one prefill. After prefill, we read each layer's K tensor of
/// shape `[1, n_kv_heads, S, head_dim]`, compute per-position K-norm² as the
/// proxy for softmax mass weight under representative queries, then for each
/// (kv-head, query-position) sort the per-key scores descending and find the
/// smallest top-K such that the cumulative covers `mass_threshold`.
///
/// Per-head (in the Q-head sense) budget = max over all calibration query
/// positions of the per-(kv-head) top-K (after GQA expansion the same KV slot
/// is shared by multiple Q-heads, so each Q-head row of the schema gets the
/// kv-head's budget).
///
/// Proxy method note: the K-norm² ranking is a well-known stand-in for softmax
/// mass under randomly-projected queries (H2O, StreamingLLM). The production
/// sparse-attn dispatch is HOLD, so the proxy vs true-softmax delta has no
/// live operational impact — see KV_QUANT.md "Sparse attention".
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: row_off + head_dim is within the n_kv_heads * S_buf * head_dim slab; budgets indices match the table shape allocated above"
)]
fn measure_head_budgets_qwen3(
    model: &Qwen3Text,
    tokenised_prompts: &[Vec<u32>],
    mass_threshold: f32,
) -> anyhow::Result<HeadBudgetMeasurement> {
    let device = Device::Gpu;
    let num_layers = model.cfg.num_hidden_layers;
    let n_q_heads = model.cfg.num_attention_heads;
    let n_kv_heads = model.cfg.num_key_value_heads;
    let head_dim = model.cfg.head_dim;
    if n_q_heads % n_kv_heads != 0 {
        anyhow::bail!("head_budget: GQA mismatch — n_q_heads={n_q_heads}, n_kv_heads={n_kv_heads}");
    }
    let q_per_kv = n_q_heads / n_kv_heads;

    let mut per_layer_kv_budget: Vec<Vec<u32>> = vec![vec![1_u32; n_kv_heads]; num_layers];
    let mut max_seq_len: u32 = 0;

    for (prompt_idx, ids) in tokenised_prompts.iter().enumerate() {
        let seq = ids.len();
        if seq < 2 {
            continue;
        }
        max_seq_len = max_seq_len.max(seq as u32);

        let mut caches: Vec<KvCache> = (0..num_layers)
            .map(|_| KvCache::with_quant_max_seq(KvQuant::None, seq as i32))
            .collect();

        let t_prefill = Instant::now();
        let _logits = model
            .forward_seq_with_cache(ids, Some(&mut caches), device)
            .map_err(|e| anyhow::anyhow!("prefill prompt {prompt_idx}: {e}"))?;
        let prefill_secs = t_prefill.elapsed().as_secs_f64();
        info!(
            prompt_idx,
            seq, prefill_secs, "head_budget: prompt prefill complete"
        );

        for (layer_idx, cache) in caches.iter().enumerate() {
            let k = cache.calibration_k_bf16().ok_or_else(|| {
                anyhow::anyhow!(
                    "head_budget: layer {layer_idx} has no bf16 K accumulator (KvQuant::None expected)"
                )
            })?;
            let offset = cache.offset() as usize;
            if offset == 0 {
                continue;
            }
            let s_buf = k_buf_seq(k) as usize;
            if s_buf == 0 {
                continue;
            }
            let k_host = k_to_host_f32(k, n_kv_heads, head_dim)?;
            for kvh in 0..n_kv_heads {
                let kvh_off = kvh * s_buf * head_dim;
                let per_key_score: Vec<f32> = (0..offset)
                    .map(|pos| {
                        let row_off = kvh_off + pos * head_dim;
                        let row = &k_host[row_off..row_off + head_dim];
                        row.iter().map(|x| x * x).sum::<f32>()
                    })
                    .collect();
                let mut max_budget: u32 = 1;
                for q_pos in 0..offset {
                    let mut visible: Vec<f32> = per_key_score[..=q_pos].to_vec();
                    visible.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    let total: f32 = visible.iter().sum();
                    if total <= 0.0 || !total.is_finite() {
                        continue;
                    }
                    let mut acc: f32 = 0.0;
                    let mut budget: u32 = visible.len() as u32;
                    for (i, &v) in visible.iter().enumerate() {
                        acc += v;
                        if acc / total >= mass_threshold {
                            budget = (i + 1) as u32;
                            break;
                        }
                    }
                    if budget > max_budget {
                        max_budget = budget;
                    }
                }
                if max_budget > per_layer_kv_budget[layer_idx][kvh] {
                    per_layer_kv_budget[layer_idx][kvh] = max_budget;
                }
            }
        }
    }

    let per_layer_per_head_budget: Vec<Vec<u32>> = per_layer_kv_budget
        .iter()
        .map(|row| {
            let mut out = Vec::with_capacity(n_q_heads);
            for &b in row {
                for _ in 0..q_per_kv {
                    out.push(b);
                }
            }
            out
        })
        .collect();

    Ok(HeadBudgetMeasurement {
        per_layer_per_head_budget,
        max_seq_len,
    })
}

/// Return the third dimension (S_buf) of a `[1, n_kv_heads, S_buf, head_dim]`
/// K cache tensor.
fn k_buf_seq(k: &Array) -> i32 {
    let s = k.shape();
    *s.get(2).unwrap_or(&0)
}

/// Copy a `[1, n_kv_heads, S_buf, head_dim]` K tensor to a host f32 vec of
/// length `n_kv_heads * S_buf * head_dim`. Supports F32 / BF16.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds checked via explicit length-of-bytes guard above each branch"
)]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "K cache is always F32 or BF16 by KvCache::None bf16 path; \
              future dtype additions should surface as the explicit bail"
)]
fn k_to_host_f32(k: &Array, n_kv_heads: usize, head_dim: usize) -> anyhow::Result<Vec<f32>> {
    k.eval().map_err(|e| anyhow::anyhow!("k.eval: {e}"))?;
    let s_buf = k_buf_seq(k) as usize;
    let total = n_kv_heads * s_buf * head_dim;
    let bytes = k
        .to_bytes()
        .map_err(|e| anyhow::anyhow!("k.to_bytes: {e}"))?;
    match k.dtype() {
        Dtype::F32 => {
            if bytes.len() < total * 4 {
                anyhow::bail!(
                    "head_budget: F32 K buffer too small ({} < {})",
                    bytes.len(),
                    total * 4
                );
            }
            let out: Vec<f32> = bytes[..total * 4]
                .chunks_exact(4)
                .map(|c| {
                    let mut arr = [0u8; 4];
                    arr.copy_from_slice(c);
                    f32::from_le_bytes(arr)
                })
                .collect();
            Ok(out)
        }
        Dtype::Bf16 => {
            if bytes.len() < total * 2 {
                anyhow::bail!(
                    "head_budget: BF16 K buffer too small ({} < {})",
                    bytes.len(),
                    total * 2
                );
            }
            let mut out: Vec<f32> = Vec::with_capacity(total);
            for i in 0..total {
                let o = i * 2;
                let mut arr = [0u8; 2];
                arr.copy_from_slice(&bytes[o..o + 2]);
                let raw = u16::from_le_bytes(arr);
                out.push(f32::from_bits(u32::from(raw) << 16));
            }
            Ok(out)
        }
        other => anyhow::bail!("head_budget: unsupported K dtype {other:?}"),
    }
}

#[cfg(test)]
#[path = "kv_calibrate_tests.rs"]
mod kv_calibrate_tests;
