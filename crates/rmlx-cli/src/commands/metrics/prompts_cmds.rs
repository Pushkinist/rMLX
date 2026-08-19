// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::Path;

use anyhow::Context as _;
use rmlx_metrics::{prompts, query, schema};

use super::{repo_root, PromptsAction};

// ---------------------------------------------------------------------------
// prompts dispatch
// ---------------------------------------------------------------------------

pub(super) fn cmd_prompts(db_path: &Path, action: PromptsAction) -> anyhow::Result<()> {
    match action {
        PromptsAction::List => cmd_prompts_list(db_path),
        PromptsAction::Get { name } => cmd_prompts_get(db_path, &name),
        PromptsAction::Add { name, file, notes } => {
            cmd_prompts_add(db_path, name.as_deref(), &file, notes.as_deref())
        }
        PromptsAction::Sync => cmd_prompts_sync(db_path),
    }
}

fn cmd_prompts_list(db_path: &Path) -> anyhow::Result<()> {
    let conn = schema::open_checked(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let store = prompts::PromptStore::new(&conn);
    let rows = store.list().map_err(|e| anyhow::anyhow!("{e}"))?;
    if rows.is_empty() {
        println!("(no prompts registered)");
        return Ok(());
    }
    println!(
        "{:<6}  {:<20}  {:<10}  {:<14}  first_seen_utc",
        "id", "name", "sha256[:8]", "tokens_approx"
    );
    println!("{}", "-".repeat(80));
    for r in &rows {
        let sha_short = if r.sha256.len() >= 8 {
            &r.sha256[..8]
        } else {
            &r.sha256
        };
        let tokens = r
            .tokens_approx
            .map_or_else(|| "-".to_string(), |n| n.to_string());
        println!(
            "{:<6}  {:<20}  {:<10}  {:<14}  {}",
            r.id, r.name, sha_short, tokens, r.first_seen_utc
        );
    }
    Ok(())
}

fn cmd_prompts_get(db_path: &Path, name: &str) -> anyhow::Result<()> {
    let conn = schema::open_checked(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let store = prompts::PromptStore::new(&conn);
    let row = store
        .find_latest_by_name(name)
        .map_err(|e| anyhow::anyhow!("{e}"))?
        .ok_or_else(|| anyhow::anyhow!("no prompt found with name '{name}'"))?;
    println!("{}", serde_json::to_string_pretty(&row.body)?);
    Ok(())
}

fn cmd_prompts_add(
    db_path: &Path,
    name_override: Option<&str>,
    file: &Path,
    notes_override: Option<&str>,
) -> anyhow::Result<()> {
    let mut pf = prompts::parse_prompt_file(file)
        .map_err(|e| anyhow::anyhow!("parse prompt file {}: {e}", file.display()))?;

    // Apply overrides.
    if let Some(n) = name_override {
        pf.name = n.to_owned();
    }
    if let Some(notes) = notes_override {
        pf.notes = Some(notes.to_owned());
    }

    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let store = prompts::PromptStore::new(&conn);
    let id = store
        .get_or_insert(&pf.name, &pf.body, pf.tokens_approx, pf.notes.as_deref())
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("prompt id={id} name='{}' registered", pf.name);
    Ok(())
}

fn cmd_prompts_sync(db_path: &Path) -> anyhow::Result<()> {
    let root = repo_root();
    let prompts_dir = root.join("prompts");
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let (inserted, total) =
        prompts::sync_dir(&conn, &prompts_dir).map_err(|e| anyhow::anyhow!("{e}"))?;
    println!("sync: inserted={inserted}, total={total}");
    Ok(())
}

// ---------------------------------------------------------------------------
// champions
// ---------------------------------------------------------------------------

pub(super) fn cmd_champions(
    db_path: &Path,
    backend_filter: Option<&str>,
    jsonl: bool,
) -> anyhow::Result<()> {
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let rows = query::champions(&conn, backend_filter).map_err(|e| anyhow::anyhow!("{e}"))?;

    if jsonl {
        for r in &rows {
            println!("{}", serde_json::to_string(r)?);
        }
        return Ok(());
    }

    // ── Markdown render ───────────────────────────────────────────────────────
    // Preferred column ordering: decode_tps_warm, prefill_tps, ttft_warm_ms, peak_rss_mb first,
    // then the rest alpha-sorted.
    let priority = [
        "decode_tps_warm",
        "prefill_tps",
        "ttft_warm_ms",
        "peak_rss_mb",
    ];

    // Collect all metric names actually present in at least one row.
    let mut present_metrics: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for row in &rows {
        for k in row.metrics.keys() {
            present_metrics.insert(k.clone());
        }
    }

    let mut col_order: Vec<String> = Vec::with_capacity(present_metrics.len());
    for p in &priority {
        if present_metrics.contains(*p) {
            col_order.push(p.to_string());
        }
    }
    for m in &present_metrics {
        if !priority.contains(&m.as_str()) {
            col_order.push(m.clone());
        }
    }

    let scope = match backend_filter {
        None => "all backends".to_owned(),
        Some(b) => format!("backend={b}"),
    };
    println!("# Champion records — {scope}");
    println!(
        "<!-- Generated by `rmlx metrics champions{}`. Source: metrics/runs.db. -->",
        backend_filter
            .map(|b| format!(" --backend {b}"))
            .unwrap_or_default()
    );
    println!();
    println!(
        "_Generated from `metrics/runs.db`. \
         Spec: [`docs/METRICS_DB.md`](docs/METRICS_DB.md). \
         Methodology / hardware / scope: see `CLAUDE.md`._"
    );
    println!();

    if rows.is_empty() {
        println!("_(no data)_");
        return Ok(());
    }

    // Build header.
    let mut header_cells: Vec<String> =
        vec!["model_namespace", "model", "weight_quant", "kv_quant"]
            .into_iter()
            .map(str::to_owned)
            .collect();
    for m in &col_order {
        header_cells.push(m.clone());
        if backend_filter.is_none() {
            header_cells.push(format!("{m} backend"));
        }
    }
    println!("| {} |", header_cells.join(" | "));

    // Separator row.
    let sep: Vec<&str> = header_cells
        .iter()
        .map(|h| {
            // right-align numeric columns
            if h.contains("backend")
                || ["model_namespace", "model", "weight_quant", "kv_quant"].contains(&h.as_str())
            {
                "---"
            } else {
                "---:"
            }
        })
        .collect();
    println!(
        "|{}|",
        sep.iter()
            .map(|s| format!(" {s} "))
            .collect::<Vec<_>>()
            .join("|")
    );

    // Track backend distribution for footer.
    let mut backend_counts: std::collections::BTreeMap<String, usize> =
        std::collections::BTreeMap::new();

    // Data rows.
    for row in &rows {
        let mut cells: Vec<String> = vec![
            row.model_namespace.clone(),
            row.model.clone(),
            row.weight_quant.clone(),
            row.kv_quant.clone(),
        ];
        for m in &col_order {
            match row.metrics.get(m) {
                None => {
                    cells.push(String::new());
                    if backend_filter.is_none() {
                        cells.push(String::new());
                    }
                }
                Some(c) => {
                    // Format value: integers for ms/mb/bytes, 2-decimal for tps/ratio.
                    let formatted = if c.unit == "ms" || c.unit == "mb" || c.unit == "bytes" {
                        format!("{}", c.value.round() as i64)
                    } else {
                        format!("{:.2}", c.value)
                    };
                    cells.push(formatted);
                    if backend_filter.is_none() {
                        cells.push(c.backend.clone());
                        *backend_counts.entry(c.backend.clone()).or_insert(0) += 1;
                    } else {
                        *backend_counts.entry(c.backend.clone()).or_insert(0) += 1;
                    }
                }
            }
        }
        println!("| {} |", cells.join(" | "));
    }

    println!();
    let ts_now = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    let backend_dist = backend_counts
        .iter()
        .map(|(b, n)| format!("{b}={n}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "_Generated {ts_now}; {} cells; champion_backend_distribution: {}_",
        rows.len(),
        if backend_dist.is_empty() {
            "-".to_owned()
        } else {
            backend_dist
        }
    );

    Ok(())
}
