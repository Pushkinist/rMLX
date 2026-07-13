// ── §2 Legacy-data ingester (docs/METRICS_DB.md §7) ─────────────────────────

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use rusqlite::{params, Connection};
use serde::Deserialize;
use serde_json::Value as JsonValue;
use sha2::{Digest, Sha256};
use tracing::{info, warn};

use crate::error::{Error, Result};
use crate::identity;
use crate::ingest::{MetricEntry, PromptRef, RunRecord, RECORD_SCHEMA_VERSION};
use crate::recorder::Recorder;

// ── Public options + report ───────────────────────────────────────────────────

/// Options for [`migrate_all`].
#[derive(Debug, Clone)]
#[allow(
    clippy::exhaustive_structs,
    reason = "closed options struct — fields are the complete migration-pass configuration; constructed with struct-literal from rmlx-cli; adding a field requires updating all MigrateOptions construction sites"
)]
pub struct MigrateOptions {
    /// Glob pattern for rMLX JSONL files, e.g. `"metrics/**/*.jsonl"`.
    /// When `None` the JSONL pass is skipped.
    pub rmlx_glob: Option<String>,
    /// Path to `Cross-Backend-Bench/metrics/summary.csv`.
    /// When `None` the CSV pass is skipped.
    pub cbb_csv: Option<PathBuf>,
    /// Path to `BENCHMARK_RECORDS.md` fallback table.
    /// When `None` the MD fallback pass is skipped.
    pub records_md: Option<PathBuf>,
    /// Hardware tag to stamp on every migrated observation.
    /// Defaults to `"m5_max_128gb"`.
    pub hardware_tag: String,
    /// Directory containing `longctx_4k.json` and siblings.
    pub prompts_dir: PathBuf,
    /// `inserted_by` audit field, e.g. `"migrate@0.0.1"`.
    pub inserted_by: String,
}

impl Default for MigrateOptions {
    fn default() -> Self {
        Self {
            rmlx_glob: None,
            cbb_csv: None,
            records_md: None,
            hardware_tag: "m5_max_128gb".to_string(),
            prompts_dir: PathBuf::from("prompts"),
            inserted_by: format!("migrate@{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

/// Summary of what [`migrate_all`] did.
#[derive(Debug, Default, Clone)]
#[non_exhaustive]
pub struct MigrateReport {
    /// Number of JSONL files matched by `rmlx_glob` and read.
    pub rmlx_jsonl_files_read: usize,
    /// Total JSONL rows (lines) encountered across all files.
    pub rmlx_jsonl_rows_total: usize,
    /// JSONL rows successfully inserted into `observations`.
    pub rmlx_jsonl_rows_inserted: usize,
    /// JSONL rows skipped (already present, de-duplicated by legacy key).
    pub rmlx_jsonl_rows_skipped: usize,
    /// `(file_path, error_message)` pairs for rows that could not be parsed
    /// or whose namespace was unknown.
    pub rmlx_jsonl_parse_failures: Vec<(String, String)>,

    /// Total rows read from the CBB CSV.
    pub cbb_csv_rows_total: usize,
    /// CSV rows successfully inserted into `observations`.
    pub cbb_csv_rows_inserted: usize,
    /// CSV rows skipped (already present or missing required fields).
    pub cbb_csv_rows_skipped: usize,
    /// `(line_number, error_message)` pairs for CSV rows that failed.
    pub cbb_csv_parse_failures: Vec<(usize, String)>,

    /// Number of `BENCHMARK_RECORDS.md` table cells successfully ingested.
    pub records_md_cells_added: usize,
}

// ── Entry point ───────────────────────────────────────────────────────────────

/// Replay legacy JSONL + CSV + optional MD records into `conn`.
///
/// Per docs/METRICS_DB.md §7. Idempotent: rows already present (identified
/// by the `legacy_run_key=<hex>` prefix in `observations.notes`) are skipped.
pub fn migrate_all(conn: &mut Connection, opts: &MigrateOptions) -> Result<MigrateReport> {
    let mut report = MigrateReport::default();

    // Load the canonical prompt once; it is shared by all rMLX JSONL rows and
    // used as the fallback for CBB CSV rows that had prompt_tokens=4096.
    let longctx_prompt = load_prompt_file(&opts.prompts_dir.join("longctx_4k.json"))?;

    // ── JSONL pass ────────────────────────────────────────────────────────────
    if let Some(glob_pat) = &opts.rmlx_glob {
        migrate_jsonl_files(conn, glob_pat, &longctx_prompt, opts, &mut report)?;
    }

    // ── CSV pass ──────────────────────────────────────────────────────────────
    if let Some(csv_path) = &opts.cbb_csv {
        migrate_cbb_csv(conn, csv_path, &longctx_prompt, opts, &mut report)?;
    }

    // ── MD fallback ───────────────────────────────────────────────────────────
    if let Some(md_path) = &opts.records_md {
        migrate_records_md(conn, md_path, &longctx_prompt, opts, &mut report)?;
    }

    Ok(report)
}

// ── Shared prompt type ────────────────────────────────────────────────────────

struct LoadedPrompt {
    name: String,
    body: JsonValue,
    tokens_approx: Option<i64>,
    /// Loaded from the prompt JSON file; carried through for future use.
    #[allow(dead_code)]
    notes: Option<String>,
}

/// Read a `prompts/<name>.json` file into a `LoadedPrompt`.
fn load_prompt_file(path: &Path) -> Result<LoadedPrompt> {
    let raw = std::fs::read_to_string(path).map_err(|e| {
        Error::Io(std::io::Error::new(
            e.kind(),
            format!("reading prompt file {}: {e}", path.display()),
        ))
    })?;
    let obj: JsonValue = serde_json::from_str(&raw).map_err(|e| {
        Error::Schema(format!(
            "prompt file {} is not valid JSON: {e}",
            path.display()
        ))
    })?;

    // serde_json's Index<&str> on Value returns Value::Null for missing keys
    // when the value is an Object; only panics on non-object types. `obj` is
    // guaranteed to be a JSON object because it was parsed from a prompt .json
    // file that must be a JSON object (non-object would produce unusable data
    // caught downstream by schema validation).
    #[allow(
        clippy::indexing_slicing,
        reason = "serde_json Index<&str> on Value returns Null for missing keys; obj is a parsed JSON object from a prompt file — non-object type cannot reach this point"
    )]
    let name = obj["name"].as_str().unwrap_or("longctx_4k").to_string();
    #[allow(
        clippy::indexing_slicing,
        reason = "serde_json Index<&str> on Value returns Null for missing keys; obj is a parsed JSON object from a prompt file"
    )]
    let tokens_approx = obj["tokens_approx"].as_i64();
    #[allow(
        clippy::indexing_slicing,
        reason = "serde_json Index<&str> on Value returns Null for missing keys; obj is a parsed JSON object from a prompt file"
    )]
    let notes = obj["notes"].as_str().map(str::to_string);

    // Prefer the `messages` array (CBB format); fall back to full object.
    #[allow(
        clippy::indexing_slicing,
        reason = "serde_json Index<&str> on Value returns Null for missing keys; obj is a parsed JSON object from a prompt file"
    )]
    let body = if obj["messages"].is_array() {
        #[allow(
            clippy::indexing_slicing,
            reason = "serde_json Index<&str> on Value returns Null for missing keys; obj is a parsed JSON object from a prompt file"
        )]
        obj["messages"].clone()
    } else {
        obj
    };

    Ok(LoadedPrompt {
        name,
        body,
        tokens_approx,
        notes,
    })
}

// ── JSONL ingester ────────────────────────────────────────────────────────────

/// Raw JSONL row shape from `metrics/perf-iter/*.jsonl` and siblings.
#[derive(Debug, Deserialize)]
struct RmlxJsonlRow {
    /// Source run_id from the JSONL file; discarded at DB insert (re-minted).
    #[allow(dead_code)]
    run_id: Option<String>,
    ts_utc: String,
    model_path: String,
    kv_quant: String,
    decode_tps_mean: f64,
    decode_tps_stddev: Option<f64>,
    step_ms_mean: Option<f64>,
    first_32_words: Option<Vec<String>>,
    git_sha: Option<String>,
    build_profile: Option<String>,
    notes: Option<String>,
}

fn migrate_jsonl_files(
    conn: &mut Connection,
    glob_pat: &str,
    prompt: &LoadedPrompt,
    opts: &MigrateOptions,
    report: &mut MigrateReport,
) -> Result<()> {
    let paths = glob_jsonl(glob_pat)?;
    report.rmlx_jsonl_files_read = paths.len();

    for path in paths {
        let path_str = path.display().to_string();
        let content = match std::fs::read_to_string(&path) {
            Ok(c) => c,
            Err(e) => {
                report
                    .rmlx_jsonl_parse_failures
                    .push((path_str.clone(), e.to_string()));
                continue;
            }
        };

        // Is this a dirty-suffixed file? Used to append `-dirty` to git_sha.
        let is_dirty_file = path_str.contains("-dirty");

        for line in content.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with("//") {
                continue;
            }
            report.rmlx_jsonl_rows_total += 1;

            match ingest_jsonl_row(conn, line, is_dirty_file, prompt, opts) {
                Ok(inserted) => {
                    if inserted {
                        report.rmlx_jsonl_rows_inserted += 1;
                    } else {
                        report.rmlx_jsonl_rows_skipped += 1;
                    }
                }
                Err(e) => {
                    report
                        .rmlx_jsonl_parse_failures
                        .push((path_str.clone(), e.to_string()));
                    report.rmlx_jsonl_rows_skipped += 1;
                }
            }
        }
    }

    info!(
        files = report.rmlx_jsonl_files_read,
        inserted = report.rmlx_jsonl_rows_inserted,
        skipped = report.rmlx_jsonl_rows_skipped,
        failures = report.rmlx_jsonl_parse_failures.len(),
        "JSONL migration done"
    );
    Ok(())
}

/// Returns `true` if a new observation was inserted, `false` if skipped.
fn ingest_jsonl_row(
    conn: &mut Connection,
    line: &str,
    is_dirty_file: bool,
    prompt: &LoadedPrompt,
    opts: &MigrateOptions,
) -> Result<bool> {
    let row: RmlxJsonlRow =
        serde_json::from_str(line).map_err(|e| Error::Schema(format!("JSONL parse error: {e}")))?;

    // Resolve namespace + model.
    let (model_namespace, model) = identity::split_model_path(&row.model_path)?;

    // Weight quant from model name suffix.
    let weight_quant = infer_weight_quant_from_model(&model).to_string();

    // KV quant canonicalization (parser-based, accepts `mixed_*`).
    let kv_quant = identity::canonicalize_kv_quant(&row.kv_quant)?;

    // Build legacy_run_key for idempotency (§7.5).
    let legacy_key = legacy_run_key_jsonl(
        &row.model_path,
        &kv_quant,
        &row.ts_utc,
        row.git_sha.as_deref().unwrap_or(""),
    );

    if observation_exists_by_legacy_key(conn, &legacy_key)? {
        return Ok(false);
    }

    // Git sha — append -dirty if the source file is dirty-marked.
    let git_sha = match (row.git_sha, is_dirty_file) {
        (Some(sha), true) if !sha.ends_with("-dirty") => Some(format!("{sha}-dirty")),
        (sha, _) => sha,
    };

    // first_32_words → output_first_64 (joined, truncated at 64 chars).
    let output_first_64 = row.first_32_words.map(|words| {
        let joined = words.join(" ");
        joined.chars().take(64).collect::<String>()
    });

    // Notes — prepend legacy_run_key for idempotency probe on next run.
    let notes = Some(format!(
        "legacy_run_key={legacy_key}{}",
        row.notes
            .as_deref()
            .map(|n| format!("; {n}"))
            .unwrap_or_default()
    ));

    // Notes for prompt ref.
    let prompt_notes =
        Some("legacy-import: prompt_tokens=4096 → longctx_4k.json per §7.3".to_string());

    let mut metrics = vec![MetricEntry {
        name: "decode_tps_warm".to_string(),
        value: Some(row.decode_tps_mean),
        stddev: row.decode_tps_stddev,
    }];
    if let Some(step) = row.step_ms_mean {
        metrics.push(MetricEntry {
            name: "step_ms_mean".to_string(),
            value: Some(step),
            stddev: None,
        });
    }

    let run = RunRecord {
        schema_version: RECORD_SCHEMA_VERSION,
        backend: "rmlx".to_string(),
        backend_version: None,
        model_namespace,
        model,
        weight_quant,
        kv_quant,
        ctx_max: 8192,
        prompt: PromptRef::ByBody {
            name: prompt.name.clone(),
            body: prompt.body.clone(),
            tokens_approx: prompt.tokens_approx,
            notes: prompt_notes,
        },
        ts_utc: row.ts_utc,
        git_sha,
        build_profile: row.build_profile,
        hardware_tag: opts.hardware_tag.clone(),
        prompt_tokens: Some(4096),
        max_tokens: Some(32),
        temperature: Some(0.0),
        seed: Some(0),
        n_warmups: Some(1),
        n_measure: Some(3),
        output_first_64,
        notes,
        description: None,
        metrics,
    };

    let mut rec = Recorder::legacy_archive(conn, &opts.inserted_by);
    rec.record_run(&run)?;
    Ok(true)
}

// ── CBB CSV ingester ──────────────────────────────────────────────────────────

#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    reason = "CSV migration path: sequential field parsing, validation, idempotency checks, and insertion in one natural unit; splitting across helpers would fragment the linear parse→validate→insert flow"
)]
fn migrate_cbb_csv(
    conn: &mut Connection,
    csv_path: &Path,
    prompt: &LoadedPrompt,
    opts: &MigrateOptions,
    report: &mut MigrateReport,
) -> Result<()> {
    let file = std::fs::File::open(csv_path)?;
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(file);

    let headers: Vec<String> = rdr
        .headers()
        .map_err(|e| Error::Schema(format!("CSV header error: {e}")))?
        .iter()
        .map(str::to_string)
        .collect();

    // Map header name → column index for cheap lookup.
    let col = |name: &str| -> Option<usize> { headers.iter().position(|h| h == name) };

    let ci_run_id = col("run_id");
    let ci_ts =
        col("timestamp_utc").ok_or_else(|| Error::Schema("CSV missing timestamp_utc".into()))?;
    let ci_backend = col("backend").ok_or_else(|| Error::Schema("CSV missing backend".into()))?;
    let ci_backend_ver = col("backend_version");
    let ci_model = col("model_id").ok_or_else(|| Error::Schema("CSV missing model_id".into()))?;
    let ci_quant = col("quant_signature")
        .ok_or_else(|| Error::Schema("CSV missing quant_signature".into()))?;
    let ci_device = col("device");
    let ci_prompt_tokens = col("prompt_tokens");
    let ci_max_tokens = col("max_tokens");
    let ci_ttft = col("ttft_ms");
    let ci_itl_p50 = col("itl_p50_ms");
    let ci_itl_p95 = col("itl_p95_ms");
    let ci_decode_tps = col("decode_tps");
    let ci_overall_tps = col("overall_tps");
    let ci_peak_rss = col("peak_rss_mb");
    let ci_task_pass = col("task_pass_at_1");
    let ci_success = col("success");
    let ci_output = col("output_first_64");

    for (idx, result) in rdr.records().enumerate() {
        let line_no = idx + 2; // 1-indexed, +1 for header
        report.cbb_csv_rows_total += 1;

        let record = match result {
            Ok(r) => r,
            Err(e) => {
                report
                    .cbb_csv_parse_failures
                    .push((line_no, format!("CSV read error: {e}")));
                report.cbb_csv_rows_skipped += 1;
                continue;
            }
        };

        let field = |idx: usize| record.get(idx).unwrap_or("").trim().to_string();
        let field_opt = |idx: Option<usize>| {
            idx.and_then(|i| record.get(i))
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
        };

        // Skip failed runs.
        let success_val = field_opt(ci_success).unwrap_or_default().to_lowercase();
        if success_val == "false" || success_val == "0" {
            report.cbb_csv_rows_skipped += 1;
            continue;
        }

        let ts_utc = field(ci_ts);
        let backend_raw = field(ci_backend);
        let model_id = field(ci_model);
        let quant_sig = field(ci_quant);
        let csv_run_id = field_opt(ci_run_id).unwrap_or_default();

        // Idempotency key.
        let legacy_key =
            legacy_run_key_csv(&backend_raw, &model_id, &quant_sig, &ts_utc, &csv_run_id);
        match observation_exists_by_legacy_key(conn, &legacy_key) {
            Ok(true) => {
                report.cbb_csv_rows_skipped += 1;
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                report.cbb_csv_parse_failures.push((line_no, e.to_string()));
                report.cbb_csv_rows_skipped += 1;
                continue;
            }
        }

        // Parse identity fields.
        let Some(backend) = normalize_csv_backend(&backend_raw) else {
            report
                .cbb_csv_parse_failures
                .push((line_no, format!("unknown backend: {backend_raw:?}")));
            report.cbb_csv_rows_skipped += 1;
            continue;
        };

        let (model_namespace, model) = match identity::split_model_path(&model_id) {
            Ok(pair) => pair,
            Err(e) => {
                report
                    .cbb_csv_parse_failures
                    .push((line_no, format!("model_id parse: {e}")));
                report.cbb_csv_rows_skipped += 1;
                continue;
            }
        };

        // quant_signature → (weight_quant, kv_quant). Split on '/'.
        let (weight_quant_raw, kv_quant_raw) = split_quant_signature(&quant_sig);
        let weight_quant = normalize_weight_quant(&weight_quant_raw);
        let kv_quant = normalize_kv_quant(&kv_quant_raw);

        // Validate against whitelists.
        if identity::canonicalize(
            "weight_quant",
            &weight_quant,
            identity::WEIGHT_QUANT_WHITELIST,
        )
        .is_err()
        {
            warn!(
                line = line_no,
                weight_quant, "unknown weight_quant; skipping CSV row"
            );
            report.cbb_csv_rows_skipped += 1;
            continue;
        }
        if identity::canonicalize_kv_quant(&kv_quant).is_err() {
            warn!(
                line = line_no,
                kv_quant, "unknown kv_quant; skipping CSV row"
            );
            report.cbb_csv_rows_skipped += 1;
            continue;
        }

        // Hardware tag: map known device strings.
        let hardware_tag = field_opt(ci_device).map_or_else(
            || opts.hardware_tag.clone(),
            |d| map_device_to_hardware_tag(&d),
        );

        // Backend version: strip backend name prefix if present.
        let backend_version = field_opt(ci_backend_ver).map(|v| strip_backend_prefix(&v));

        // Prompt: resolve by prompt_tokens.
        let prompt_tokens_raw = field_opt(ci_prompt_tokens).and_then(|s| s.parse::<i64>().ok());
        let prompt_ref = resolve_prompt_ref(prompt_tokens_raw, prompt);

        let max_tokens = field_opt(ci_max_tokens).and_then(|s| s.parse::<i64>().ok());
        let output_first_64 = field_opt(ci_output);

        // Build metrics array from available columns.
        let mut metrics: Vec<MetricEntry> = Vec::new();
        let parse_f64 = |idx: Option<usize>| -> Option<f64> {
            idx?.let_(|i| record.get(i)?.trim().parse().ok())
        };

        if let Some(v) = parse_f64(ci_decode_tps) {
            metrics.push(MetricEntry {
                name: "decode_tps_warm".into(),
                value: Some(v),
                stddev: None,
            });
        }
        if let Some(v) = parse_f64(ci_overall_tps) {
            metrics.push(MetricEntry {
                name: "overall_tps".into(),
                value: Some(v),
                stddev: None,
            });
        }
        if let Some(v) = parse_f64(ci_ttft) {
            metrics.push(MetricEntry {
                name: "ttft_warm_ms".into(),
                value: Some(v),
                stddev: None,
            });
        }
        if let Some(v) = parse_f64(ci_itl_p50) {
            metrics.push(MetricEntry {
                name: "itl_p50_ms".into(),
                value: Some(v),
                stddev: None,
            });
        }
        if let Some(v) = parse_f64(ci_itl_p95) {
            metrics.push(MetricEntry {
                name: "itl_p95_ms".into(),
                value: Some(v),
                stddev: None,
            });
        }
        if let Some(v) = parse_f64(ci_peak_rss) {
            metrics.push(MetricEntry {
                name: "peak_rss_mb".into(),
                value: Some(v),
                stddev: None,
            });
        }
        if let Some(v) = parse_f64(ci_task_pass) {
            metrics.push(MetricEntry {
                name: "task_pass_at_1".into(),
                value: Some(v),
                stddev: None,
            });
        }

        // Filter out zero-value task_pass_at_1 that CBB emits when not measured.
        let metrics: Vec<MetricEntry> = metrics
            .into_iter()
            .filter(|m| {
                m.value
                    .is_some_and(|v| v != 0.0 || m.name != "task_pass_at_1")
            })
            .collect();

        if metrics.is_empty() {
            report.cbb_csv_rows_skipped += 1;
            continue;
        }

        let notes = Some(format!(
            "legacy_run_key={legacy_key}; migrated from CBB summary.csv"
        ));

        let run = RunRecord {
            schema_version: RECORD_SCHEMA_VERSION,
            backend,
            backend_version,
            model_namespace,
            model,
            weight_quant,
            kv_quant,
            ctx_max: 8192,
            prompt: prompt_ref,
            ts_utc,
            git_sha: None,
            build_profile: None,
            hardware_tag,
            prompt_tokens: prompt_tokens_raw,
            max_tokens,
            temperature: Some(0.0),
            seed: Some(0),
            n_warmups: None,
            n_measure: None,
            output_first_64,
            notes,
            description: None,
            metrics,
        };

        match Recorder::legacy_archive(conn, &opts.inserted_by).record_run(&run) {
            Ok(_) => report.cbb_csv_rows_inserted += 1,
            Err(e) => {
                report
                    .cbb_csv_parse_failures
                    .push((line_no, format!("record_run: {e}")));
                report.cbb_csv_rows_skipped += 1;
            }
        }
    }

    info!(
        inserted = report.cbb_csv_rows_inserted,
        skipped = report.cbb_csv_rows_skipped,
        failures = report.cbb_csv_parse_failures.len(),
        "CBB CSV migration done"
    );
    Ok(())
}

// ── BENCHMARK_RECORDS.md fallback ────────────────────────────────────────────

#[allow(
    clippy::cognitive_complexity,
    clippy::too_many_lines,
    clippy::unnecessary_wraps,
    reason = "markdown fallback migration: sequential line-by-line parse→validate→insert; Result<()> return kept for consistency with other migrate_* functions and future Err propagation"
)]
fn migrate_records_md(
    conn: &mut Connection,
    md_path: &Path,
    prompt: &LoadedPrompt,
    opts: &MigrateOptions,
    report: &mut MigrateReport,
) -> Result<()> {
    let content = match std::fs::read_to_string(md_path) {
        Ok(c) => c,
        Err(e) => {
            warn!("BENCHMARK_RECORDS.md not readable: {e}; skipping fallback");
            return Ok(());
        }
    };

    // Parse model headings and table rows.
    // Heading format: `### `<namespace>__<model>`` (or bare model name).
    // Table row format: `| <backend> | <kv_quant> | <decode_tps> | ...`
    let mut current_model_path: Option<String> = None;

    for line in content.lines() {
        let line = line.trim();

        // Detect model heading: `### `mlx-community__gemma-4-e2b-it-mxfp8``
        if let Some(rest) = line.strip_prefix("### `") {
            if let Some(inner) = rest.strip_suffix('`') {
                // Exact backtick-quoted heading.
                current_model_path = Some(inner.to_string());
            } else {
                // Handle `### `text (desc, weight-quant)`` format used in the file.
                let trimmed = rest.split('`').next().unwrap_or("").trim();
                if !trimmed.is_empty() {
                    current_model_path = Some(trimmed.to_string());
                }
            }
            continue;
        }

        let model_path = match &current_model_path {
            Some(p) => p.clone(),
            None => continue,
        };

        // Data rows: `| backend | kv | decode_tps | ...`
        // Skip header rows (`---`) and rows without enough `|`.
        if !line.starts_with('|') || line.contains("---") || line.contains("Backend") {
            continue;
        }

        let cells: Vec<&str> = line
            .split('|')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .collect();
        if cells.len() < 3 {
            continue;
        }

        // cells.len() >= 3 is guaranteed by the `cells.len() < 3` guard above.
        #[allow(
            clippy::indexing_slicing,
            reason = "cells.len() < 3 continues above, so indices 0/1/2 are always valid here"
        )]
        let backend_raw = cells[0];
        #[allow(
            clippy::indexing_slicing,
            reason = "cells.len() < 3 continues above, so indices 0/1/2 are always valid here"
        )]
        let kv_raw = cells[1];
        // decode_tps is the third column (index 2).
        #[allow(
            clippy::indexing_slicing,
            reason = "cells.len() < 3 continues above, so indices 0/1/2 are always valid here"
        )]
        let decode_tps_raw = cells[2];

        // Skip non-numeric, N/A, or 'x' cells.
        let Some(decode_tps): Option<f64> = decode_tps_raw
            .split_whitespace()
            .next()
            .and_then(|s| s.parse().ok())
        else {
            continue;
        };

        let Some(backend) = normalize_csv_backend(backend_raw) else {
            continue;
        };

        // Resolve model identity. The heading uses the bare model name (no path).
        // Reconstruct a synthetic path so split_model_path handles the `__` convention.
        let synthetic_path = format!("/legacy/{model_path}");
        let Ok((model_namespace, model)) = identity::split_model_path(&synthetic_path) else {
            continue;
        };

        // weight_quant from model name; kv_quant from column.
        let weight_quant = infer_weight_quant_from_model(&model).to_string();
        let kv_quant = normalize_kv_quant(kv_raw);
        if identity::canonicalize_kv_quant(&kv_quant).is_err() {
            continue;
        }

        // Idempotency: key by (backend, model_namespace, model, kv_quant, "records_md").
        let legacy_key = hex_sha256(&format!(
            "records_md|{backend}|{model_namespace}|{model}|{weight_quant}|{kv_quant}"
        ));

        if observation_exists_by_legacy_key(conn, &legacy_key).unwrap_or(true) {
            continue;
        }

        let notes = Some(format!(
            "legacy_run_key={legacy_key}; migrated from BENCHMARK_RECORDS.md"
        ));

        let run = RunRecord {
            schema_version: RECORD_SCHEMA_VERSION,
            backend,
            backend_version: None,
            model_namespace,
            model,
            weight_quant,
            kv_quant,
            ctx_max: 8192,
            prompt: PromptRef::ByBody {
                name: prompt.name.clone(),
                body: prompt.body.clone(),
                tokens_approx: prompt.tokens_approx,
                notes: Some("migrated from BENCHMARK_RECORDS.md".to_string()),
            },
            ts_utc: "2026-01-01T00:00:00Z".to_string(),
            git_sha: Some("imported-from-records-md".to_string()),
            build_profile: None,
            hardware_tag: opts.hardware_tag.clone(),
            prompt_tokens: Some(4096),
            max_tokens: Some(32),
            temperature: Some(0.0),
            seed: Some(0),
            n_warmups: None,
            n_measure: None,
            output_first_64: None,
            notes,
            description: Some("migrated from BENCHMARK_RECORDS.md".to_string()),
            metrics: vec![MetricEntry {
                name: "decode_tps_warm".to_string(),
                value: Some(decode_tps),
                stddev: None,
            }],
        };

        match Recorder::legacy_archive(conn, &opts.inserted_by).record_run(&run) {
            Ok(_) => report.records_md_cells_added += 1,
            Err(e) => {
                warn!(legacy_key, error = %e, "MD fallback row rejected");
            }
        }
    }

    info!(
        added = report.records_md_cells_added,
        "MD fallback migration done"
    );
    Ok(())
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Infer weight quantization from model name suffix (§5.2 + §7.2 defaults).
///
/// Suffix-matching order matters — check longer/more-specific suffixes first.
pub fn infer_weight_quant_from_model(model: &str) -> &'static str {
    let lower = model.to_lowercase();
    // Microscaling FP families first (most specific).
    if lower.contains("mxfp8") {
        return "mxfp8";
    }
    if lower.contains("mxfp4") {
        return "mxfp4";
    }
    if lower.contains("nvfp4") {
        return "nvfp4";
    }
    // Bit-width suffixes — check longer strings before shorter ones.
    if lower.ends_with("-2bit") || lower.ends_with("_2bit") || lower.contains("-mlx-2bit") {
        return "2bit";
    }
    if lower.ends_with("-3bit") || lower.ends_with("_3bit") {
        return "3bit";
    }
    if lower.ends_with("-4bit") || lower.ends_with("_4bit") {
        return "4bit";
    }
    if lower.ends_with("-5bit") || lower.ends_with("_5bit") {
        return "5bit";
    }
    if lower.ends_with("-6bit") || lower.ends_with("_6bit") {
        return "6bit";
    }
    if lower.ends_with("-8bit") || lower.ends_with("_8bit") {
        return "8bit";
    }
    if lower.ends_with("-q8_0") || lower.ends_with("_q8_0") || lower.contains("-q8_0-") {
        return "q8_0";
    }
    if lower.ends_with("-q4_k_m") || lower.ends_with("_q4_k_m") {
        return "q4_k_m";
    }
    // PARO / paroquant rotation.
    if lower.ends_with("-paro") || lower.ends_with("_paro") || lower.contains("paro") {
        return "paro";
    }
    // Default: bf16 (unquantized baseline).
    "bf16"
}

/// Map known CBB `backend` strings to the §5.4 whitelist.
fn normalize_csv_backend(raw: &str) -> Option<String> {
    let lower = raw.trim().to_lowercase();
    match lower.as_str() {
        "rmlx" => Some("rmlx".into()),
        "mlx_lm" | "mlx-lm" | "mlxlm" => Some("mlx_lm".into()),
        "mlx_lm_tq" | "mlx-lm-tq" | "mlx-lm-turboquant" | "mlx_lm_turboquant" => {
            Some("mlx_lm_tq".into())
        }
        "paroquant" | "paro_quant" | "paro" => Some("paroquant".into()),
        "omlx" | "o_mlx" => Some("omlx".into()),
        "ollama" => Some("ollama".into()),
        "vllm" | "vllm-mlx" => Some("vllm".into()),
        _ => None,
    }
}

/// Split `quant_signature` on `/` into `(weight_quant, kv_quant)`.
///
/// - `"mxfp8/k8v8"` → `("mxfp8", "k8v8")`
/// - `"mxfp8"` (no slash) → `("mxfp8", "none")`
/// - `"mxfp8 (mlx via ollama)"` → treat the part before `(` as weight_quant
fn split_quant_signature(sig: &str) -> (String, String) {
    // Strip parenthetical annotations like "(mlx via ollama)".
    let clean = sig.split('(').next().unwrap_or(sig).trim();
    if let Some(slash) = clean.find('/') {
        (
            clean[..slash].trim().to_string(),
            clean[slash + 1..].trim().to_string(),
        )
    } else {
        (clean.to_string(), "none".to_string())
    }
}

/// Normalize a weight_quant string from CSV to a canonical §5.2 value.
fn normalize_weight_quant(raw: &str) -> String {
    let lower = raw.trim().to_lowercase();
    // Map known aliases.
    match lower.as_str() {
        "bf16 kv" | "bf16" | "(full ctx cache)" | "full ctx cache" => "bf16".into(),
        "k8v4" | "k8v8" | "planar" | "turbo4" | "turbo8" | "none" => {
            // These look like kv_quant values smuggled into weight_quant — treat as bf16.
            "bf16".into()
        }
        other => {
            // Try the suffix-based inference as fallback.
            infer_weight_quant_from_model(other).to_string()
        }
    }
}

/// Normalize kv_quant strings from CSV / MD to §5.3 canonical values.
fn normalize_kv_quant(raw: &str) -> String {
    let lower = raw.trim().to_lowercase();
    match lower.as_str() {
        "k8v4" => "k8v4".into(),
        "k8v8" => "k8v8".into(),
        "planar" => "planar".into(),
        "turbo4" => "turbo4".into(),
        "turbo8" => "turbo8".into(),
        "none" | "" | "bf16 kv" | "bf16" | "(full ctx cache)" | "full ctx cache" | "-" | "–" => {
            "none".into()
        }
        _ => "none".into(),
    }
}

/// Map CBB `device` column to `hardware_tag`.
fn map_device_to_hardware_tag(device: &str) -> String {
    match device.trim().to_lowercase().as_str() {
        "m5_max" | "gpu" => "m5_max_128gb".to_string(),
        other => other.to_string(),
    }
}

/// Strip a backend name prefix from a version string.
///
/// E.g. `"rmlx-0.0.1"` → `"0.0.1"`, `"mlx-lm-0.21.0"` → `"0.21.0"`.
fn strip_backend_prefix(version: &str) -> String {
    // If the string contains a digit-starting segment after a '-', that's the semver.
    let v = version.trim();
    // Try splitting on '-' and find first purely version-like segment.
    if let Some(pos) = v.find(|c: char| c.is_ascii_digit()) {
        v[pos..].to_string()
    } else {
        v.to_string()
    }
}

/// Resolve the prompt reference for a legacy row based on `prompt_tokens`.
fn resolve_prompt_ref(prompt_tokens: Option<i64>, canonical: &LoadedPrompt) -> PromptRef {
    match prompt_tokens {
        // Known mapping: 4096 → longctx_4k.
        Some(4096) | None => PromptRef::ByBody {
            name: canonical.name.clone(),
            body: canonical.body.clone(),
            tokens_approx: canonical.tokens_approx,
            notes: Some("legacy-import: prompt_tokens=4096 → longctx_4k.json per §7.3".to_string()),
        },
        // Unknown prompt_tokens: synthesize a sentinel prompt.
        Some(n) => PromptRef::ByBody {
            name: format!("legacy_unknown_{n}"),
            body: serde_json::json!(format!("<UNKNOWN BODY: legacy CBB row, prompt_tokens={n}>")),
            tokens_approx: Some(n),
            notes: Some(format!(
                "prompt_tokens={n} from row; unknown prompt; flag for cleanup"
            )),
        },
    }
}

/// Hex-SHA256 of the given string. Used for idempotency keys.
fn hex_sha256(s: &str) -> String {
    let digest = Sha256::digest(s.as_bytes());
    // write!(String) is infallible — let _ discards the unit Ok.
    digest.iter().fold(String::new(), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

/// Compute the idempotency key for an rMLX JSONL row.
fn legacy_run_key_jsonl(model_path: &str, kv_quant: &str, ts_utc: &str, git_sha: &str) -> String {
    hex_sha256(&format!("jsonl|{model_path}|{kv_quant}|{ts_utc}|{git_sha}"))
}

/// Compute the idempotency key for a CBB CSV row.
fn legacy_run_key_csv(
    backend: &str,
    model_id: &str,
    quant_sig: &str,
    ts_utc: &str,
    csv_run_id: &str,
) -> String {
    hex_sha256(&format!(
        "csv|{backend}|{model_id}|{quant_sig}|{ts_utc}|{csv_run_id}"
    ))
}

/// Return `true` if any observation row's `notes` starts with `legacy_run_key=<key>`.
fn observation_exists_by_legacy_key(conn: &Connection, key: &str) -> Result<bool> {
    use rusqlite::OptionalExtension;
    let prefix = format!("legacy_run_key={key}");
    let found: Option<i64> = conn
        .query_row(
            "SELECT 1 FROM observations WHERE notes LIKE ?1 || '%' LIMIT 1",
            params![prefix],
            |r| r.get(0),
        )
        .optional()?;
    Ok(found.is_some())
}

/// Manual glob expansion for `**/*.jsonl` — walks up to 3 dir levels deep.
///
/// We avoid adding the `glob` crate for a one-shot migration walker; the
/// metrics/ tree is shallow (depth ≤ 3), so a hand-rolled walker is simpler.
fn glob_jsonl(pattern: &str) -> Result<Vec<PathBuf>> {
    // Extract the root directory from the pattern (everything before `**` or `*`).
    let root = if let Some(pos) = pattern.find("**") {
        PathBuf::from(&pattern[..pos].trim_end_matches('/'))
    } else if let Some(pos) = pattern.rfind('/') {
        PathBuf::from(&pattern[..pos])
    } else {
        PathBuf::from(".")
    };

    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut results = Vec::new();
    walk_jsonl(&root, 0, 3, &mut results)?;
    results.sort();
    Ok(results)
}

fn walk_jsonl(dir: &Path, depth: usize, max_depth: usize, out: &mut Vec<PathBuf>) -> Result<()> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() && depth < max_depth {
            walk_jsonl(&path, depth + 1, max_depth, out)?;
        } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            out.push(path);
        }
    }
    Ok(())
}

// Tiny extension trait so `Option<usize>` can call `.let_()` for the closure
// idiom used in `parse_f64` inside the CSV ingester.
trait LetExt: Sized {
    fn let_<R>(self, f: impl FnOnce(Self) -> R) -> R;
}
impl<T> LetExt for T {
    fn let_<R>(self, f: impl FnOnce(Self) -> R) -> R {
        f(self)
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "legacy_tests.rs"]
mod tests;
