//! Export API per `docs/METRICS_DB.md` §9 and §8.2.
//!
//! All functions take a `&Connection` and return a `String`; the caller
//! decides where to write the output (file, stdout, etc.).
//!
//! # Public API
//!
//! - [`export_markdown`] — render `BENCHMARK_CHAMPIONS.md` from the current
//!   bests view; optionally scoped by a [`ScopeFile`] filter.
//! - [`export_json`] — full champion data as a JSON array.
//! - [`export_csv`] — champion data as CSV.
//! - [`export_jsonl`] — champion data as newline-delimited JSON.
//!
//! # See also
//!
//! - `docs/METRICS_DB.md` §9 — export contract and `BENCHMARK_CHAMPIONS.md` format.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use rusqlite::Connection;
use serde::Serialize;

use crate::error::Result;
use crate::query::BestRow;
use crate::scope::{ScopeFile, ScopeModel};

// ── Static header (lifted from BENCHMARK_RECORDS(old).md, scope section + methodology + backend + quant tables). ──

const HEADER_TEMPLATE: &str = include_str!("templates/header.md");

// ── Display ordering for backends (matches old file row order). ───────────────

const BACKEND_DISPLAY_ORDER: &[&str] =
    &["rmlx", "mlx_lm", "mlx_lm_tq", "omlx", "ollama", "paroquant"];

/// Map DB backend id → human display label (old-file conventions).
fn backend_display(backend: &str) -> &str {
    match backend {
        "mlx_lm" => "mlx-lm",
        "mlx_lm_tq" => "mlx-lm-tq",
        "omlx" => "oMLX",
        other => other,
    }
}

/// Map DB `kv_quant` → display label (old-file conventions).
fn kv_quant_display(kv_quant: &str) -> &str {
    match kv_quant {
        "none" => "bf16 KV",
        other => other,
    }
}

// ── Metric columns rendered in the per-model table (in column order). ────────

/// One rendered metric column: the row it reads and the heading it sits under.
///
/// The heading is a field rather than a parallel array. Two index-coupled
/// arrays render the header from one and the body from the other, so a reorder
/// in either puts every value under the wrong heading and nothing says so.
#[derive(Clone, Copy)]
struct MetricCol {
    db_name: &'static str,
    /// Column heading.
    label: &'static str,
    /// Decimal places for formatting. 0 = integer.
    decimals: usize,
}

const METRIC_COLUMNS: &[MetricCol] = &[
    MetricCol {
        db_name: "decode_tps_warm",
        label: "Decode TPS warm",
        decimals: 2,
    },
    MetricCol {
        db_name: "prefill_tps",
        label: "Prefill TPS",
        decimals: 0,
    },
    MetricCol {
        db_name: "ttft_cold_ms",
        label: "TTFT cold (ms)",
        decimals: 0,
    },
    MetricCol {
        db_name: "ttft_warm_ms",
        label: "TTFT warm (ms)",
        decimals: 0,
    },
    MetricCol {
        db_name: "peak_rss_mb",
        label: "Peak RSS (MB)",
        decimals: 0,
    },
];

// ── Speculative round-loop columns (in column order). ────────────────────────
//
// These belong to a section of their own rather than to the per-model table:
// every one of them is `-` on a plain-decode row, and there are far more of
// those. The metric set comes from `registry::SPEC_METRICS`, which is where a
// new speculative metric is declared; a test pins that this renders all of it.

const SPEC_METRIC_COLUMNS: &[MetricCol] = &[
    MetricCol {
        db_name: "decode_tps_warm",
        label: "Decode TPS warm",
        decimals: 2,
    },
    MetricCol {
        db_name: "accept_rate",
        label: "Accept rate",
        decimals: 3,
    },
    MetricCol {
        db_name: "tokens_per_round",
        label: "Tokens/round",
        decimals: 2,
    },
    MetricCol {
        db_name: "accepted_per_step",
        label: "Accepted/step",
        decimals: 2,
    },
    MetricCol {
        db_name: "draft_ms_per_round",
        label: "Draft ms/round",
        decimals: 2,
    },
    MetricCol {
        db_name: "verify_ms_per_round",
        label: "Verify ms/round",
        decimals: 2,
    },
    MetricCol {
        db_name: "loop_ms_per_round",
        label: "Loop ms/round",
        decimals: 2,
    },
];

// ── KV memory columns ────────────────────────────────────────────────────────

/// Key: `(model_namespace, model, ctx_max, kv_quant)` → best `kv_cache_bytes` value.
///
/// Built once per export and threaded into `render_model_table` so the table
/// can show `kv_gb` and `reduction_vs_bf16` without additional DB round trips.
type KvBytesMap = BTreeMap<(String, String, i64, String), f64>;

/// Build the KV-bytes lookup from the already-fetched `bests` rows.
///
/// Keeps the minimum (best, since `lower_better`) `kv_cache_bytes` value per
/// `(namespace, model, ctx_max, kv_quant)` cell across all backends and
/// prompt_ids. rMLX is the only backend expected to emit this metric, but the
/// code is generic.
fn build_kv_bytes_map(all: &[BestRow]) -> KvBytesMap {
    let mut map: KvBytesMap = BTreeMap::new();
    for r in all {
        if r.metric != "kv_cache_bytes" {
            continue;
        }
        let key = (
            r.cell.model_namespace.clone(),
            r.cell.model.clone(),
            r.cell.ctx_max,
            r.cell.kv_quant.clone(),
        );
        map.entry(key)
            .and_modify(|existing| {
                if r.value < *existing {
                    *existing = r.value;
                }
            })
            .or_insert(r.value);
    }
    map
}

/// Bytes → GiB (1024^3).
fn bytes_to_gib(bytes: f64) -> f64 {
    bytes / (1024.0 * 1024.0 * 1024.0)
}

/// Render `kv_gb` cell: `"N.NNN GB"` or `"-"` if no data.
fn fmt_kv_gb(kv_bytes: Option<f64>) -> String {
    match kv_bytes {
        Some(b) => format!("{:.3} GB", bytes_to_gib(b)),
        None => "-".to_owned(),
    }
}

/// Render `reduction_vs_bf16` cell: ratio rounded to 2 decimal places,
/// or `"-"` when no bf16 baseline exists for this `(model, ctx_max)`.
fn fmt_reduction(quant_bytes: Option<f64>, bf16_bytes: Option<f64>) -> String {
    match (quant_bytes, bf16_bytes) {
        (Some(q), Some(b)) if b > 0.0 => format!("{:.2}x", q / b),
        _ => "-".to_owned(),
    }
}

// ── Public API ───────────────────────────────────────────────────────────────

/// Render `BENCHMARK_CHAMPIONS.md` in the per-model pivoted layout.
///
/// When `scope` is `Some`, only models listed in the scope file appear, in
/// scope.order order, with their arch + weight_quant_display headings and
/// `N/A` cells for `unsupported = [...]` backends.
///
/// When `scope` is `None`, every observed (namespace, model) gets its own
/// section with `arch = unknown`. Useful for ad-hoc exports.
pub fn export_markdown(conn: &Connection, scope: Option<&ScopeFile>) -> Result<String> {
    let all = all_bests(conn)?;
    let exported_at = crate::time_util::now_iso8601()?;

    let mut out = String::with_capacity(16 * 1024);
    out.push_str(HEADER_TEMPLATE);
    out.push('\n');

    if all.is_empty() {
        out.push_str("*No observations in database.*\n\n");
        out.push_str(&provenance_section(&exported_at, 0, 0));
        return Ok(out);
    }

    // Build KV-bytes lookup once; threaded into each per-model table render.
    let kv_map = build_kv_bytes_map(&all);

    // ── Resolve scope models in render order ─────────────────────────────────
    let scope_models: Vec<ResolvedModel> = match scope {
        Some(s) => s
            .models
            .iter()
            .map(|m| ResolvedModel::from_scope(m.clone()))
            .collect(),
        None => resolve_from_db(&all),
    };

    out.push_str("## Records\n\n");

    let mut total_cells: usize = 0;

    for rm in &scope_models {
        // write!(String) is infallible — let _ discards the unit Ok.
        let _ = write!(
            out,
            "### `{}` ({}, {})\n\n",
            rm.display_id, rm.arch, rm.weight_quant_display
        );

        let rows = render_model_table(rm, &all, &kv_map, &mut total_cells);
        out.push_str(&rows);
        out.push('\n');
    }

    out.push_str(&render_speculative_table(&all));

    // ── Champion summary ─────────────────────────────────────────────────────
    out.push_str("---\n\n## Champion summary (auto-generated)\n\n");
    out.push_str("Per metric × model: which backend holds the record, with the rMLX gap.\n\n");
    out.push_str("| Model | Best decode TPS | Backend | rMLX best | Gap | rMLX best KV |\n");
    out.push_str("|---|---:|---|---:|---|---|\n");
    for rm in &scope_models {
        out.push_str(&champion_summary_row(rm, &all));
    }
    out.push('\n');

    // ── Provenance ───────────────────────────────────────────────────────────
    out.push_str(&provenance_section(&exported_at, all.len(), total_cells));

    Ok(out)
}

/// One row of the speculative table: **the whole cell key**, in schema order.
///
/// Every column of `cell::CELL_COLUMNS` is here — namespace and model share the
/// first element — because each metric in the row
/// is resolved independently against `bests`. Drop one — `ctx_max`, say — and a
/// row takes its higher-better token rate from a 4k run and its lower-better
/// milliseconds from a 128k one, under a heading that says the three
/// millisecond columns partition one round.
type SpecRowKey = (String, String, String, String, String, i64, i64);

/// Render the speculative section: one row per drafter arm, with the round-loop
/// figures beside the throughput they explain.
///
/// Empty string when no row in the DB names a drafter — a heading over an empty
/// table would read as "no drafter pays" rather than "nothing was measured".
fn render_speculative_table(all: &[BestRow]) -> String {
    let mut cells: BTreeMap<SpecRowKey, BTreeMap<String, &BestRow>> = BTreeMap::new();
    for r in all {
        let Some(config) = r.cell.decode_config.as_deref() else {
            continue;
        };
        if !crate::cell::decode_config_names_a_drafter(config) {
            continue;
        }
        cells
            .entry((
                format!("{}__{}", r.cell.model_namespace, r.cell.model),
                config.to_owned(),
                r.cell.backend.clone(),
                r.cell.kv_quant.clone(),
                r.cell.weight_quant.clone(),
                r.cell.ctx_max,
                r.cell.prompt_id,
            ))
            .or_default()
            .entry(r.metric.clone())
            .and_modify(|existing| {
                if best_of(existing, r) {
                    *existing = r;
                }
            })
            .or_insert(r);
    }
    if cells.is_empty() {
        return String::new();
    }

    let mut out = String::from("---\n\n## Speculative decoding (auto-generated)\n\n");
    out.push_str(
        "Tokens per verify round — accepted drafts plus the verifier's own token — is what a \
         drafter is read with. It equals `1 + accept_rate x (block - 1)` only while every round \
         drafts the configured block, which an adaptive drafter does not, so it is recorded \
         rather than derived here. The three `ms/round` columns partition one round's wall \
         clock; the last is the round loop's own overhead. One row is one cell, so the \
         context and prompt every figure was measured at are columns of it.\n\n",
    );
    out.push_str("| Model | Decode | Backend | KV-quant | Weight | ctx_max | prompt ");
    for col in SPEC_METRIC_COLUMNS {
        // write!(String) is infallible — let _ discards the unit Ok.
        let _ = write!(out, "| {} ", col.label);
    }
    out.push_str("| Updated |\n|---|---|---|---|---|---:|---:");
    for _ in SPEC_METRIC_COLUMNS {
        out.push_str("|---:");
    }
    out.push_str("|---|\n");

    for ((model, config, backend, kv_quant, weight_quant, ctx_max, prompt_id), metric_map) in &cells
    {
        let mut row = format!(
            "| `{model}` | {config} | {} | {} | {weight_quant} | {ctx_max} | {prompt_id} ",
            backend_display(backend),
            kv_quant_display(kv_quant)
        );
        for col in SPEC_METRIC_COLUMNS {
            match metric_map.get(col.db_name) {
                Some(r) => {
                    let _ = write!(row, "| {} ", fmt_value(r.value, col.decimals));
                }
                None => row.push_str("| - "),
            }
        }
        let _ = writeln!(row, "| {} |", updated_summary(metric_map));
        out.push_str(&row);
    }
    out.push('\n');
    out
}

/// Full bests dump as a compact JSON array (`Vec<BestRow>`).
pub fn export_json(conn: &Connection) -> Result<String> {
    let all = all_bests(conn)?;
    let serializable: Vec<SerBestRow> = all.iter().map(SerBestRow::from_best).collect();
    serde_json::to_string(&serializable)
        .map_err(|e| crate::error::Error::Schema(format!("json serialize: {e}")))
}

/// Bests as CSV with a header row. No external crate — manual escape.
pub fn export_csv(conn: &Connection) -> Result<String> {
    let all = all_bests(conn)?;

    let mut out = String::with_capacity(4 * 1024);
    out.push_str(
        "backend,model_namespace,model,weight_quant,kv_quant,ctx_max,prompt_id,\
         decode_config,metric,value,unit,direction,run_id,ts_utc,git_sha,backend_version,\
         hardware_tag,description,notes,inserted_by\n",
    );

    let sorted = sorted_rows(all);
    for r in &sorted {
        out.push_str(&csv_row(r));
        out.push('\n');
    }

    Ok(out)
}

/// Bests as JSONL — one compact JSON object per line.
pub fn export_jsonl(conn: &Connection) -> Result<String> {
    let all = all_bests(conn)?;
    let mut out = String::with_capacity(4 * 1024);
    for r in &all {
        let s = SerBestRow::from_best(r);
        let line = serde_json::to_string(&s)
            .map_err(|e| crate::error::Error::Schema(format!("json serialize: {e}")))?;
        out.push_str(&line);
        out.push('\n');
    }
    Ok(out)
}

// ── Markdown rendering helpers ───────────────────────────────────────────────

/// Either a real ScopeModel or a fallback derived from DB rows.
struct ResolvedModel {
    display_id: String,
    arch: String,
    weight_quant_display: String,
    /// Predicates: `(namespace, model)` pairs that map to this row.
    matchers: Vec<(String, String)>,
    /// Unsupported backends → reason.
    unsupported: Vec<(String, Option<String>)>,
}

impl ResolvedModel {
    fn from_scope(m: ScopeModel) -> Self {
        let mut matchers = vec![(m.namespace.clone(), m.name.clone())];
        for a in &m.aliases {
            matchers.push((a.namespace.clone(), a.name.clone()));
        }
        let unsupported = m
            .unsupported
            .into_iter()
            .map(|u| (u.backend, u.reason))
            .collect();
        Self {
            display_id: format!("{}__{}", m.namespace, m.name),
            arch: m.arch,
            weight_quant_display: m.weight_quant_display,
            matchers,
            unsupported,
        }
    }

    fn matches(&self, namespace: &str, model: &str) -> bool {
        self.matchers
            .iter()
            .any(|(ns, n)| ns == namespace && n == model)
    }
}

/// Build ResolvedModel list from the DB when no scope file is supplied.
fn resolve_from_db(all: &[BestRow]) -> Vec<ResolvedModel> {
    let mut seen: Vec<(String, String)> = Vec::new();
    for r in all {
        let key = (r.cell.model_namespace.clone(), r.cell.model.clone());
        if !seen.contains(&key) {
            seen.push(key);
        }
    }
    seen.sort();
    seen.into_iter()
        .map(|(ns, m)| ResolvedModel {
            display_id: format!("{ns}__{m}"),
            arch: "unknown".into(),
            weight_quant_display: "unknown".into(),
            matchers: vec![(ns, m)],
            unsupported: Vec::new(),
        })
        .collect()
}

/// One row of the per-model table: backend, KV codec, and how the tokens were
/// produced. All three are cell identity, so all three have to be in the key —
/// grouping without the last one merges two configurations under one label.
type RowKey = (String, String, Option<String>);

/// Render the per-model pivoted table.
///
/// `kv_map` is the pre-built `(namespace, model, ctx_max, kv_quant) → bytes`
/// lookup used to derive `kv_gb` and `reduction_vs_bf16` columns.
fn render_model_table(
    rm: &ResolvedModel,
    all: &[BestRow],
    kv_map: &KvBytesMap,
    total_cells: &mut usize,
) -> String {
    // Group bests for this model by (backend, kv_quant, decode_config) →
    // metric → row. `decode_config` is in the key because it is in the cell
    // key: without it a speculative arm and a plain one become one row whose
    // label describes only one of them, and `best_of` keeps the drafter's
    // number. When alias namespaces produce multiple champions for the same
    // group and metric, keep the better one per `direction`.
    let mut cells: BTreeMap<RowKey, BTreeMap<String, &BestRow>> = BTreeMap::new();
    for r in all {
        if !rm.matches(&r.cell.model_namespace, &r.cell.model) {
            continue;
        }
        let entry = cells
            .entry((
                r.cell.backend.clone(),
                r.cell.kv_quant.clone(),
                r.cell.decode_config.clone(),
            ))
            .or_default();
        match entry.get(&r.metric) {
            None => {
                entry.insert(r.metric.clone(), r);
            }
            Some(existing) => {
                if best_of(existing, r) {
                    entry.insert(r.metric.clone(), r);
                }
            }
        }
    }

    let mut out = String::new();
    out.push_str("| Backend | KV-quant | Decode ");
    for col in METRIC_COLUMNS {
        // write!(String) is infallible — let _ discards the unit Ok.
        let _ = write!(out, "| {} ", col.label);
    }
    out.push_str("| KV GB | reduction vs bf16 | Updated |\n");
    out.push_str("|---|---|---");
    for _ in METRIC_COLUMNS {
        out.push_str("|---:");
    }
    out.push_str("|---:|---:|---|\n");

    // Order rows: BACKEND_DISPLAY_ORDER first, then alphabetic for unknown
    // backends. Within a backend, sort kv_quant alphabetically.
    let mut row_keys: Vec<RowKey> = cells.keys().cloned().collect();
    row_keys.sort_by(|a, b| {
        let aa = BACKEND_DISPLAY_ORDER
            .iter()
            .position(|x| *x == a.0)
            .unwrap_or(usize::MAX);
        let bb = BACKEND_DISPLAY_ORDER
            .iter()
            .position(|x| *x == b.0)
            .unwrap_or(usize::MAX);
        aa.cmp(&bb)
            .then_with(|| a.0.cmp(&b.0))
            .then_with(|| a.1.cmp(&b.1))
            .then_with(|| a.2.cmp(&b.2))
    });

    for key in &row_keys {
        let (backend, kv_quant, decode_config) = key;
        let Some(metric_map) = cells.get(key) else {
            continue;
        };

        let backend_label = backend_display(backend);
        let kv_label = kv_quant_display(kv_quant);
        let decode_label = decode_config.as_deref().unwrap_or("plain");

        // Derive ctx_max from any row in this (backend, kv_quant) cell.
        let ctx_max = metric_map.values().next().map_or(8192, |r| r.cell.ctx_max);

        // Look up kv_cache_bytes for this (model, ctx_max, kv_quant) across
        // all matchers (namespace aliases included).
        let kv_bytes_quant: Option<f64> = rm
            .matchers
            .iter()
            .filter_map(|(ns, m)| kv_map.get(&(ns.clone(), m.clone(), ctx_max, kv_quant.clone())))
            .copied()
            .reduce(f64::min);

        // bf16 baseline: kv_quant = "none" at the same (model, ctx_max).
        let kv_bytes_bf16: Option<f64> = rm
            .matchers
            .iter()
            .filter_map(|(ns, m)| kv_map.get(&(ns.clone(), m.clone(), ctx_max, "none".to_owned())))
            .copied()
            .reduce(f64::min);

        let mut row = format!("| {backend_label} | {kv_label} | {decode_label} ");
        for col in METRIC_COLUMNS {
            match metric_map.get(col.db_name) {
                Some(r) => {
                    let _ = write!(row, "| {} ", fmt_value(r.value, col.decimals));
                    *total_cells += 1;
                }
                None => row.push_str("| - "),
            }
        }
        let _ = write!(row, "| {} ", fmt_kv_gb(kv_bytes_quant));
        let _ = write!(row, "| {} ", fmt_reduction(kv_bytes_quant, kv_bytes_bf16));
        let _ = writeln!(row, "| {} |", updated_summary(metric_map));
        out.push_str(&row);
    }

    // Append N/A rows for unsupported backends not already present.
    for (backend, reason) in &rm.unsupported {
        let already_present = row_keys.iter().any(|(b, _, _)| b == backend);
        if already_present {
            continue;
        }
        let label = backend_display(backend);
        let kv_col = "–";
        let reason_str = reason.as_deref().unwrap_or("not applicable");
        let mut row = format!("| {label} | {kv_col} ");
        for _ in METRIC_COLUMNS {
            row.push_str("| N/A ");
        }
        // KV GB and reduction columns also get N/A for unsupported backends.
        row.push_str("| N/A | N/A ");
        let _ = writeln!(row, "| {reason_str} |");
        out.push_str(&row);
    }

    out
}

/// Returns true if `candidate` is strictly better than `existing` given the
/// row's direction (higher_better / lower_better). Tie → keep existing.
fn best_of(existing: &BestRow, candidate: &BestRow) -> bool {
    match candidate.direction.as_str() {
        "lower_better" => candidate.value < existing.value,
        _ => candidate.value > existing.value,
    }
}

/// Pick the most recent ts_utc + run_id + (truncated) notes among the cell's metrics.
fn updated_summary(metric_map: &BTreeMap<String, &BestRow>) -> String {
    let mut newest: Option<&BestRow> = None;
    for r in metric_map.values() {
        match newest {
            None => newest = Some(r),
            Some(n) if r.ts_utc > n.ts_utc => newest = Some(r),
            _ => {}
        }
    }
    let Some(r) = newest else {
        return "-".into();
    };
    let date = r.ts_utc.split('T').next().unwrap_or(&r.ts_utc);
    let run = &r.run_id;
    let mut note_part = String::new();
    if let Some(n) = r.notes.as_deref().filter(|s| !s.is_empty()) {
        // `legacy_run_key=...` is migration bookkeeping; skip those notes.
        if !n.starts_with("legacy_run_key=") {
            let snippet: String = n.chars().take(80).collect();
            let trimmed = snippet.trim().trim_end_matches(';').trim();
            if !trimmed.is_empty() {
                note_part = format!(" — {trimmed}");
            }
        }
    }
    format!("{date} `{run}`{note_part}")
}

/// One row of the champion summary table.
fn champion_summary_row(rm: &ResolvedModel, all: &[BestRow]) -> String {
    // Find best decode_tps_warm across all backends for this model.
    let mut best_all: Option<&BestRow> = None;
    let mut best_rmlx: Option<&BestRow> = None;
    for r in all {
        if r.metric != "decode_tps_warm" {
            continue;
        }
        if !rm.matches(&r.cell.model_namespace, &r.cell.model) {
            continue;
        }
        match best_all {
            None => best_all = Some(r),
            Some(c) if r.value > c.value => best_all = Some(r),
            _ => {}
        }
        if r.cell.backend == "rmlx" {
            match best_rmlx {
                None => best_rmlx = Some(r),
                Some(c) if r.value > c.value => best_rmlx = Some(r),
                _ => {}
            }
        }
    }

    let model_col = &rm.display_id;

    let (best_val, best_be) = match best_all {
        Some(r) => (
            fmt_value(r.value, 2),
            format!(
                "{} ({})",
                backend_display(&r.cell.backend),
                kv_quant_display(&r.cell.kv_quant)
            ),
        ),
        None => ("-".into(), "-".into()),
    };

    let (rmlx_val, rmlx_kv) = match best_rmlx {
        Some(r) => (
            fmt_value(r.value, 2),
            kv_quant_display(&r.cell.kv_quant).to_string(),
        ),
        None => ("-".into(), "-".into()),
    };

    let gap = match (best_all, best_rmlx) {
        (Some(a), Some(b)) if a.value > 0.0 => {
            let pct = (b.value - a.value) / a.value * 100.0;
            if (pct.abs()) < 0.005 {
                "0%".to_string()
            } else {
                format!("{pct:+.1}%")
            }
        }
        _ => "-".into(),
    };

    format!("| {model_col} | {best_val} | {best_be} | {rmlx_val} | {gap} | {rmlx_kv} |\n")
}

fn fmt_value(v: f64, decimals: usize) -> String {
    if decimals == 0 {
        format!("{v:.0}")
    } else {
        format!("{v:.decimals$}")
    }
}

// ── DB + misc helpers ────────────────────────────────────────────────────────

/// Pull every row from the `bests` VIEW.
fn all_bests(conn: &Connection) -> Result<Vec<BestRow>> {
    let mut stmt = conn.prepare(
        "SELECT
             id, backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
             decode_config, metric, value, unit, direction,
             run_id, ts_utc, git_sha, backend_version, hardware_tag,
             description, notes, inserted_by
         FROM bests
         ORDER BY backend, model_namespace, model, weight_quant, kv_quant, decode_config, metric",
    )?;

    let rows = stmt
        .query_map([], crate::query::read::row_to_best)?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    Ok(rows)
}

/// Sort order for CSV/JSONL: the cell key, then metric.
fn sorted_rows(mut rows: Vec<BestRow>) -> Vec<BestRow> {
    rows.sort_by(|a, b| {
        a.cell
            .backend
            .cmp(&b.cell.backend)
            .then_with(|| a.cell.model_namespace.cmp(&b.cell.model_namespace))
            .then_with(|| a.cell.model.cmp(&b.cell.model))
            .then_with(|| a.cell.weight_quant.cmp(&b.cell.weight_quant))
            .then_with(|| a.cell.kv_quant.cmp(&b.cell.kv_quant))
            .then_with(|| a.cell.decode_config.cmp(&b.cell.decode_config))
            .then_with(|| a.metric.cmp(&b.metric))
    });
    rows
}

fn provenance_section(exported_at: &str, distinct_champions: usize, table_cells: usize) -> String {
    format!(
        "## Provenance\n\n\
         - Exported at: `{exported_at}`\n\
         - Distinct champions (bests rows): {distinct_champions}\n\
         - Table cells rendered: {table_cells}\n"
    )
}

/// Escape a string value for CSV per RFC 4180.
fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_owned()
    }
}

fn csv_opt(opt: Option<&str>) -> String {
    opt.map(csv_escape).unwrap_or_default()
}

fn csv_row(r: &BestRow) -> String {
    format!(
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        csv_escape(&r.cell.backend),
        csv_escape(&r.cell.model_namespace),
        csv_escape(&r.cell.model),
        csv_escape(&r.cell.weight_quant),
        csv_escape(&r.cell.kv_quant),
        r.cell.ctx_max,
        r.cell.prompt_id,
        csv_opt(r.cell.decode_config.as_deref()),
        csv_escape(&r.metric),
        r.value,
        csv_escape(&r.unit),
        csv_escape(&r.direction),
        csv_escape(&r.run_id),
        csv_escape(&r.ts_utc),
        csv_opt(r.git_sha.as_deref()),
        csv_opt(r.backend_version.as_deref()),
        csv_escape(&r.hardware_tag),
        csv_opt(r.description.as_deref()),
        csv_opt(r.notes.as_deref()),
        csv_escape(&r.inserted_by),
    )
}

// ── Serializable mirror of BestRow (avoids adding Serialize to query.rs types) ─

#[derive(Serialize)]
struct SerCell {
    backend: String,
    model_namespace: String,
    model: String,
    weight_quant: String,
    kv_quant: String,
    ctx_max: i64,
    prompt_id: i64,
    decode_config: Option<String>,
}

#[derive(Serialize)]
struct SerBestRow {
    observation_id: i64,
    cell: SerCell,
    metric: String,
    value: f64,
    unit: String,
    direction: String,
    run_id: String,
    ts_utc: String,
    git_sha: Option<String>,
    backend_version: Option<String>,
    hardware_tag: String,
    description: Option<String>,
    notes: Option<String>,
    inserted_by: String,
}

impl SerBestRow {
    fn from_best(r: &BestRow) -> Self {
        Self {
            observation_id: r.observation_id,
            cell: SerCell {
                backend: r.cell.backend.clone(),
                model_namespace: r.cell.model_namespace.clone(),
                model: r.cell.model.clone(),
                weight_quant: r.cell.weight_quant.clone(),
                kv_quant: r.cell.kv_quant.clone(),
                ctx_max: r.cell.ctx_max,
                prompt_id: r.cell.prompt_id,
                decode_config: r.cell.decode_config.clone(),
            },
            metric: r.metric.clone(),
            value: r.value,
            unit: r.unit.clone(),
            direction: r.direction.clone(),
            run_id: r.run_id.clone(),
            ts_utc: r.ts_utc.clone(),
            git_sha: r.git_sha.clone(),
            backend_version: r.backend_version.clone(),
            hardware_tag: r.hardware_tag.clone(),
            description: r.description.clone(),
            notes: r.notes.clone(),
            inserted_by: r.inserted_by.clone(),
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "export_tests.rs"]
mod export_tests;
