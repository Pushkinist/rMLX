//! `rmlx eval` command implementations.
//!
//! Currently exposes a single subcommand:
//!
//! * `rmlx eval ppl` -- sliding-window perplexity scorer. Loads a model,
//!   tokenizes a text file, runs [`rmlx_models::ppl::compute_ppl`], prints one
//!   JSON line to stdout, and (when `--corpus` is non-empty) emits one §8.5
//!   universal `RunRecord` to `<RMLX_HOME>/metrics/runs.db` under op
//!   `ppl_wikitext2`.
//!
//! Option A (HTTP `echo`+`logprobs`) was rejected because it required exposing
//! per-position logits across every supported architecture's forward path —
//! not surgical. Option B (this CLI subcommand) was chosen instead.

#![allow(clippy::cognitive_complexity)]
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::Result;
use rmlx_metrics::identity::RunIdentity;
use rmlx_mlx::Device;
use rmlx_models::{arch, ppl};
use tracing::{info, instrument, warn};

/// Run `rmlx eval ppl`.
///
/// Steps:
/// 1. Validate `--device`, open model snapshot, tokenize `--text-file`.
/// 2. Apply `--max-tokens` cap (0 = use whole corpus).
/// 3. Invoke `rmlx_models::ppl::compute_ppl` with the requested window/stride.
/// 4. Print one JSON line to stdout.
/// 5. When `--corpus` is non-empty, write a §8.5 universal `RunRecord` to
///    the metrics buffer and ingest it inline.
#[instrument(skip_all, fields(
    model = %model_path.display(),
    text_file = %text_file.display(),
    ctx_window,
    stride,
    corpus,
    device = %device_str,
))]
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported variants; exhaustive expansion would require updating on every new variant"
)]
pub(crate) fn run_ppl(
    model_path: &Path,
    text_file: &Path,
    ctx_window: usize,
    stride: usize,
    corpus: &str,
    device_str: &str,
    max_tokens: usize,
    run_id: &str,
) -> Result<()> {
    let device = match device_str {
        "cpu" => Device::Cpu,
        "gpu" => Device::Gpu,
        other => {
            return Err(anyhow::anyhow!(
                "--device must be 'cpu' or 'gpu', got '{other}'"
            ));
        }
    };

    // -- Read corpus + tokenize ------------------------------------------------
    let corpus_text = std::fs::read_to_string(text_file)
        .map_err(|e| anyhow::anyhow!("cannot read text file {}: {e}", text_file.display()))?;
    let tk_path = model_path.join("tokenizer.json");
    let tokenizer = tokenizers::Tokenizer::from_file(&tk_path)
        .map_err(|e| anyhow::anyhow!("cannot load tokenizer.json: {e}"))?;
    // `add_special_tokens=true` so BOS is prepended for the Qwen3-family
    // tokenizer. This matches what `mlx-lm`'s perplexity script does.
    let encoding = tokenizer
        .encode(corpus_text.as_str(), true)
        .map_err(|e| anyhow::anyhow!("tokenize corpus: {e}"))?;
    let mut tokens: Vec<u32> = encoding.get_ids().to_vec();
    if max_tokens > 0 && tokens.len() > max_tokens {
        info!(
            original = tokens.len(),
            cap = max_tokens,
            "ppl: corpus token count capped by --max-tokens"
        );
        tokens.truncate(max_tokens);
    }
    let n_tokens = tokens.len();
    if n_tokens < 2 {
        return Err(anyhow::anyhow!(
            "corpus has only {n_tokens} tokens after tokenization -- need >=2"
        ));
    }
    info!(
        n_tokens,
        ctx_window, stride, "ppl: tokenization complete; loading model"
    );

    // -- Load model -------------------------------------------------------------
    let ts_load = Instant::now();
    let model = arch::load_model(model_path, device, &arch::LoadOpts::default())
        .map_err(|e| anyhow::anyhow!("arch::load_model: {e}"))?;
    let load_ms = ts_load.elapsed().as_millis() as f64;
    info!(load_ms, arch = model.arch_class(), "ppl: model loaded");

    // -- Score ----------------------------------------------------------------
    let ts_score = Instant::now();
    let report =
        ppl::compute_ppl(&model, &tokens, ctx_window, stride, device).map_err(|e| match e {
            ppl::PplError::ArchUnsupported { arch } => anyhow::anyhow!(
                "ppl: arch '{arch}' is not yet supported by the perplexity scorer \
                 (Qwen3 only; see crates/rmlx-models/src/ppl.rs)"
            ),
            other => anyhow::anyhow!("ppl: {other}"),
        })?;
    let score_ms = ts_score.elapsed().as_millis() as f64;
    info!(
        ppl = report.ppl,
        mean_nll = report.mean_nll,
        scored_tokens = report.scored_tokens,
        windows = report.windows,
        score_ms,
        "ppl: scoring complete"
    );

    // -- Stdout JSON line ------------------------------------------------------
    let out = serde_json::json!({
        "ppl": report.ppl,
        "mean_nll": report.mean_nll,
        "scored_tokens": report.scored_tokens,
        "windows": report.windows,
        "ctx_window": ctx_window,
        "stride": stride,
        "n_tokens": n_tokens,
        "corpus": corpus,
        "model": model_path.display().to_string(),
        "arch": model.arch_class(),
        "load_ms": load_ms,
        "score_ms": score_ms,
    });
    println!("{}", serde_json::to_string(&out)?);

    // -- §8.5 universal record -------------------------------------------------
    if !corpus.is_empty() {
        // Derive a metrics-DB-accepted `weight_quant` tag from the snapshot
        // config. Mirrors the mapping in `run_baseline`.
        let cfg = rmlx_loader::load_config(model_path).ok();
        let weight_quant = cfg
            .as_ref()
            .and_then(|c| c.quantization.as_ref())
            .map_or_else(
                || "bf16".to_string(),
                |q| match q.mode_or_default() {
                    "mxfp8" | "mxfp4" | "nvfp4" => q.mode_or_default().to_string(),
                    _ => format!("{}bit", q.bits),
                },
            );

        let record = build_ppl_run_record(
            run_id,
            model_path,
            corpus,
            ctx_window,
            stride,
            n_tokens as i64,
            &report,
            load_ms,
            score_ms,
            &weight_quant,
        )?;
        if !rmlx_metrics::mode::observations_enabled() {
            info!("ppl: observations disabled, no record written");
            return Ok(());
        }

        let buf_path = write_buffer_record(&record)?;
        info!(path = %buf_path.display(), "ppl: wrote §8.5 ingest record");

        let db_path = rmlx_core::paths::metrics_db_path();
        match rmlx_metrics::schema::open(&db_path) {
            Ok(mut conn) => {
                let inserted_by = RunIdentity::rmlx().inserted_by("rmlx-cli");
                let mut rec_inst = rmlx_metrics::recorder::Recorder::new(&mut conn, inserted_by);
                let run: rmlx_metrics::ingest::RunRecord = serde_json::from_value(record)
                    .map_err(|e| anyhow::anyhow!("deserialize RunRecord: {e}"))?;
                match rec_inst.record_run(&run) {
                    Ok(outcome) => {
                        info!(
                            run_id = %outcome.run_id,
                            inserted = outcome.observation_ids.len(),
                            "ppl: ingested record into runs.db"
                        );
                        let _ = std::fs::remove_file(&buf_path);
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            buffer = %buf_path.display(),
                            "ppl: inline ingest failed; leaving buffer file for replay"
                        );
                    }
                }
            }
            Err(e) => {
                warn!(
                    error = %e,
                    buffer = %buf_path.display(),
                    "ppl: cannot open metrics DB; leaving buffer file for replay"
                );
            }
        }
    }

    Ok(())
}

/// Build a §8.5 universal `RunRecord` JSON for a completed PPL run.
///
/// Op family is `ppl`; the single metric is `ppl_<corpus>` (operator-friendly).
/// Other audit fields go into `mean_nll`, `scored_tokens`, `windows`.
fn build_ppl_run_record(
    run_id: &str,
    model_path: &Path,
    corpus: &str,
    ctx_window: usize,
    stride: usize,
    n_tokens: i64,
    report: &ppl::PplReport,
    load_ms: f64,
    score_ms: f64,
    weight_quant: &str,
) -> Result<serde_json::Value> {
    let snapshot = model_path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("canonicalize {}: {e}", model_path.display()))?;
    let snapshot_str = snapshot.to_string_lossy().to_string();
    let (ns, model_name) = rmlx_metrics::identity::split_model_path(&snapshot_str)
        .map_err(|e| anyhow::anyhow!("split_model_path({snapshot_str}): {e}"))?;
    let ts_utc = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

    // Op-name pattern: `ppl_<corpus_id>` so multiple corpora can coexist in
    // the same table (`ppl_wikitext2`, future `ppl_c4`, ...).
    let op_name = format!("ppl_{}", corpus.replace('-', ""));

    let metrics = serde_json::json!([
        { "name": op_name,         "value": report.ppl },
        { "name": "ppl_mean_nll",  "value": report.mean_nll },
        { "name": "ppl_scored_tokens", "value": report.scored_tokens as f64 },
        { "name": "ppl_windows",   "value": report.windows as f64 },
        { "name": "ppl_score_ms",  "value": score_ms },
        { "name": "model_load_ms", "value": load_ms },
    ]);

    let prompt_name = format!("{corpus}_ctx{ctx_window}_stride{stride}");

    let mut record = serde_json::json!({
        "schema_version": rmlx_metrics::ingest::RECORD_SCHEMA_VERSION,
        "model_namespace": ns,
        "model": model_name,
        // `weight_quant` is whitelist-validated by `rmlx_metrics::identity`;
        // PPL is independent of the KV cache so we report `none` (== bf16).
        "weight_quant": weight_quant,
        "kv_quant": "none",
        "ctx_max": ctx_window as i64,
        "prompt": {
            "name": prompt_name,
            // PPL bodies are corpora referenced by name — embedding the
            // full text would bloat the DB. We carry a deterministic
            // descriptor so the ingest's prompt-body invariant is satisfied.
            "body": format!("ppl-corpus:{corpus}:ctx={ctx_window}:stride={stride}"),
        },
        "ts_utc": ts_utc,
        "prompt_tokens": n_tokens,
        "max_tokens": 0,
        "temperature": 0.0,
        "n_warmups": 0,
        "n_measure": 1,
        "output_first_64": "",
        "notes": format!("corpus={corpus} ctx_window={ctx_window} stride={stride}"),
        "description": format!("ppl {run_id}"),
        "metrics": metrics,
    });

    // Identity comes from the single Rust source — eval does not assemble it.
    RunIdentity::rmlx()
        .stamp_json(&mut record)
        .map_err(|e| anyhow::anyhow!("stamp run identity: {e}"))?;

    Ok(record)
}

/// Write a `RunRecord` JSON to `<RMLX_HOME>/metrics/buffer/pending/<ts>-<uuid>.json`.
///
/// Filename matches the §8.5 universal-shape convention (`docs/METRICS_DB.md`).
fn write_buffer_record(rec: &serde_json::Value) -> Result<PathBuf> {
    let dir = rmlx_core::paths::ingest_buffer_dir();
    std::fs::create_dir_all(&dir).map_err(|e| anyhow::anyhow!("create {}: {e}", dir.display()))?;
    let ts = chrono::Utc::now().format("%Y%m%dT%H%M%S%.3f");
    let path = dir.join(format!("{ts}-{}.json", uuid::Uuid::new_v4()));
    std::fs::write(&path, serde_json::to_string_pretty(rec)?)
        .map_err(|e| anyhow::anyhow!("write {}: {e}", path.display()))?;
    Ok(path)
}

#[cfg(test)]
#[path = "eval_tests.rs"]
mod tests;
