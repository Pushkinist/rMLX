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
    ?kv_quant,
))]
#[allow(clippy::too_many_arguments)]
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
    git_sha: Option<&str>,
    kv_quant: Option<rmlx_kv_quant::KvQuant>,
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
    let report = ppl::compute_ppl(&model, &tokens, ctx_window, stride, device, kv_quant).map_err(
        |e| match e {
            ppl::PplError::ArchUnsupported { arch } => anyhow::anyhow!(
                "ppl: arch '{arch}' is not yet supported by the perplexity scorer \
                 (Qwen3, Gemma4, Qwen3.5; see crates/rmlx-models/src/ppl.rs)"
            ),
            other => anyhow::anyhow!("ppl: {other}"),
        },
    )?;
    let score_ms = ts_score.elapsed().as_millis() as f64;
    // `decode_config` carries the boundary and nothing else. Which scorer ran
    // is in the metric name — see `ppl_metric_name`.
    let decode_config =
        kv_quant.and_then(|_| rmlx_models::kv_cache::active_kv_boundary().decode_config());
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
        "kv_quant": kv_quant.map(|q| q.to_string()),
        "decode_config": decode_config,
    });
    println!("{}", serde_json::to_string(&out)?);

    // -- §8.5 universal record -------------------------------------------------
    if corpus.is_empty() {
        // A run that measured something and stored nothing should say so. The
        // number is on stdout either way, but a sweep whose rows never reach
        // the DB looks exactly like a sweep that did until someone queries for
        // them.
        warn!(
            ppl = report.ppl,
            "ppl: no --corpus given, so this measurement is printed and not recorded; \
             pass --corpus <name> to write it to runs.db"
        );
        return Ok(());
    }
    record_ppl_run(&PplRecordArgs {
        run_id,
        model_path,
        corpus,
        ctx_window,
        stride,
        n_tokens: n_tokens as i64,
        report: &report,
        load_ms,
        score_ms,
        git_sha,
        kv_quant,
        decode_config: decode_config.as_deref(),
    })
}

/// The metric name for a PPL run, which says which scorer produced the number.
///
/// The two scorers do not measure the same quantity. The default forwards each
/// window once with no cache at all; asking for a codec teacher-forces the
/// window through a real per-layer cache, one forward per scored token. That is
/// a change to what the number *means*, not an engine setting a run moved off
/// its default — so it belongs in the metric name and not in `decode_config`,
/// which stays the boundary's alone. Keeping one name for both would rank a
/// cacheless number against a cached one in `bests` and in
/// `metrics export --markdown`, and against `mlx_lm` rows that can never carry
/// a term this engine invented.
///
/// `ppl_<corpus_id>` so multiple corpora coexist (`ppl_wikitext2`, future
/// `ppl_c4`); `_cached` suffixed for the cache-bearing scorer.
fn ppl_metric_name(corpus: &str, kv_quant: Option<rmlx_kv_quant::KvQuant>) -> String {
    let base = format!("ppl_{}", corpus.replace('-', ""));
    if kv_quant.is_some() {
        format!("{base}_cached")
    } else {
        base
    }
}

/// Everything [`record_ppl_run`] needs to emit one §8.5 record.
struct PplRecordArgs<'a> {
    run_id: &'a str,
    model_path: &'a Path,
    corpus: &'a str,
    ctx_window: usize,
    stride: usize,
    n_tokens: i64,
    report: &'a ppl::PplReport,
    load_ms: f64,
    score_ms: f64,
    git_sha: Option<&'a str>,
    kv_quant: Option<rmlx_kv_quant::KvQuant>,
    decode_config: Option<&'a str>,
}

/// Write one completed PPL run to the ingest buffer and ingest it inline.
///
/// A failed ingest leaves the buffer file behind for a later replay sweep
/// rather than losing the measurement.
fn record_ppl_run(args: &PplRecordArgs<'_>) -> Result<()> {
    // Checked before building anything: `--metrics off` means a no-op at
    // the producer, not "build the record, then throw it away".
    if !rmlx_metrics::mode::observations_enabled() {
        info!("ppl: observations disabled, no record written");
        return Ok(());
    }

    // Derive a metrics-DB-accepted `weight_quant` tag from the snapshot
    // config. Mirrors the mapping in `run_baseline`.
    let cfg = rmlx_loader::load_config(args.model_path).ok();
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
        args.run_id,
        args.model_path,
        args.corpus,
        args.ctx_window,
        args.stride,
        args.n_tokens,
        args.report,
        args.load_ms,
        args.score_ms,
        &weight_quant,
        args.git_sha,
        args.kv_quant,
        args.decode_config,
    )?;

    let buf_path = write_buffer_record(&record)?;
    info!(path = %buf_path.display(), "ppl: wrote §8.5 ingest record");

    let db_path = rmlx_core::paths::metrics_db_path();
    match rmlx_metrics::schema::open(&db_path) {
        Ok(mut conn) => {
            let inserted_by = RunIdentity::get().inserted_by("rmlx-cli");
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
    Ok(())
}

/// Build a §8.5 universal `RunRecord` JSON for a completed PPL run.
///
/// Op family is `ppl`; the single metric is `ppl_<corpus>` (operator-friendly).
/// Other audit fields go into `mean_nll`, `scored_tokens`, `windows`.
#[allow(clippy::too_many_arguments)]
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
    git_sha: Option<&str>,
    kv_quant: Option<rmlx_kv_quant::KvQuant>,
    decode_config: Option<&str>,
) -> Result<serde_json::Value> {
    let snapshot = model_path
        .canonicalize()
        .map_err(|e| anyhow::anyhow!("canonicalize {}: {e}", model_path.display()))?;
    let snapshot_str = snapshot.to_string_lossy().to_string();
    let (ns, model_name) = rmlx_metrics::identity::split_model_path(&snapshot_str)
        .map_err(|e| anyhow::anyhow!("split_model_path({snapshot_str}): {e}"))?;
    let ts_utc = chrono::Utc::now().format("%Y-%m-%dT%H:%M:%SZ").to_string();

    let op_name = ppl_metric_name(corpus, kv_quant);

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
        "weight_quant": weight_quant,
        // The codec the scorer actually ran the cache at. A cacheless run
        // stores nothing, so `none` is the truth for it and not a placeholder.
        "kv_quant": kv_quant.map_or_else(|| "none".to_string(), |q| q.to_string()),
        "decode_config": decode_config,
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
    // `stamp_json` deliberately does not touch `git_sha`: that field is
    // caller-supplied provenance (see `RunIdentity`'s doc), not something
    // this binary derives. `--git-sha` is the only source for it here.
    RunIdentity::get()
        .stamp_json(&mut record)
        .map_err(|e| anyhow::anyhow!("stamp run identity: {e}"))?;
    // Blank-string `--git-sha ""` is not provenance either — normalize it to
    // the same `None` a caller who omitted the flag gets.
    let git_sha = git_sha.filter(|s| !s.trim().is_empty());
    record
        .as_object_mut()
        .ok_or_else(|| anyhow::anyhow!("record is not a JSON object"))?
        .insert("git_sha".to_string(), serde_json::Value::from(git_sha));

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
