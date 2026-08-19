// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rmlx_metrics::{prompts, query, schema};
use rusqlite::params;

// ---------------------------------------------------------------------------
// Helper: resolve prompt_id from optional prompt_id or prompt_name
// ---------------------------------------------------------------------------

fn resolve_prompt_id(
    conn: &rusqlite::Connection,
    prompt_id: Option<i64>,
    prompt_name: Option<&str>,
) -> anyhow::Result<i64> {
    if let Some(id) = prompt_id {
        return Ok(id);
    }
    if let Some(name) = prompt_name {
        let store = prompts::PromptStore::new(conn);
        let row = store
            .find_latest_by_name(name)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        return row
            .map(|r| r.id)
            .ok_or_else(|| anyhow::anyhow!("no prompt found with name '{name}'"));
    }
    anyhow::bail!("either --prompt-id or --prompt-name is required")
}

// ---------------------------------------------------------------------------
// best
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn cmd_best(
    db_path: &Path,
    backend: &str,
    namespace: &str,
    model: &str,
    weight_quant: &str,
    kv_quant: &str,
    ctx_max: i64,
    prompt_id: Option<i64>,
    prompt_name: Option<&str>,
    metric: &str,
) -> anyhow::Result<()> {
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let pid = resolve_prompt_id(&conn, prompt_id, prompt_name)?;
    let cell = query::Cell {
        backend: backend.to_owned(),
        model_namespace: namespace.to_owned(),
        model: model.to_owned(),
        weight_quant: weight_quant.to_owned(),
        kv_quant: kv_quant.to_owned(),
        ctx_max,
        prompt_id: pid,
    };
    let row = query::best(&conn, &cell, metric).map_err(|e| anyhow::anyhow!("{e}"))?;
    match row {
        None => {
            eprintln!("no champion found for the given cell + metric");
            std::process::exit(1);
        }
        Some(r) => {
            println!("{}", serde_json::to_string(&r)?);
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// rank
// ---------------------------------------------------------------------------

pub(super) fn cmd_rank(
    db_path: &Path,
    metric: &str,
    backend: Option<&str>,
    limit: usize,
) -> anyhow::Result<()> {
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let rows = query::rank(&conn, metric, backend, limit).map_err(|e| anyhow::anyhow!("{e}"))?;
    for r in &rows {
        println!("{}", serde_json::to_string(r)?);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn cmd_compare(
    db_path: &Path,
    backends_csv: &str,
    metric: &str,
    _namespace: Option<&str>,
    _model: Option<&str>,
    _weight_quant: Option<&str>,
    _kv_quant: Option<&str>,
) -> anyhow::Result<()> {
    let backends: Vec<&str> = backends_csv.split(',').map(str::trim).collect();
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let rows = query::compare(&conn, &backends, metric).map_err(|e| anyhow::anyhow!("{e}"))?;
    for r in &rows {
        println!("{}", serde_json::to_string(r)?);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// history
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn cmd_history(
    db_path: &Path,
    backend: &str,
    namespace: &str,
    model: &str,
    weight_quant: &str,
    kv_quant: &str,
    ctx_max: i64,
    prompt_id: Option<i64>,
    prompt_name: Option<&str>,
    metric: Option<&str>,
    since: Option<&str>,
) -> anyhow::Result<()> {
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let pid = resolve_prompt_id(&conn, prompt_id, prompt_name)?;
    let cell = query::Cell {
        backend: backend.to_owned(),
        model_namespace: namespace.to_owned(),
        model: model.to_owned(),
        weight_quant: weight_quant.to_owned(),
        kv_quant: kv_quant.to_owned(),
        ctx_max,
        prompt_id: pid,
    };
    let rows = query::history(&conn, &cell, metric, since).map_err(|e| anyhow::anyhow!("{e}"))?;
    for r in &rows {
        println!("{}", serde_json::to_string(r)?);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// timeseries
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(super) fn cmd_timeseries(
    db_path: &Path,
    backend: &str,
    namespace: &str,
    model: &str,
    weight_quant: &str,
    kv_quant: &str,
    ctx_max: i64,
    prompt_id: Option<i64>,
    prompt_name: Option<&str>,
    metric: &str,
    since: Option<&str>,
    bucket_str: &str,
) -> anyhow::Result<()> {
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let pid = resolve_prompt_id(&conn, prompt_id, prompt_name)?;
    let cell = query::Cell {
        backend: backend.to_owned(),
        model_namespace: namespace.to_owned(),
        model: model.to_owned(),
        weight_quant: weight_quant.to_owned(),
        kv_quant: kv_quant.to_owned(),
        ctx_max,
        prompt_id: pid,
    };
    let bucket = match bucket_str.to_lowercase().as_str() {
        "week" => query::Bucket::Week,
        _ => query::Bucket::Day,
    };
    let rows = query::timeseries(&conn, &cell, metric, since, bucket)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    for r in &rows {
        println!("{}", serde_json::to_string(r)?);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// regress
// ---------------------------------------------------------------------------

pub(super) fn cmd_regress(
    db_path: &Path,
    model: &str,
    metric: &str,
    kv: Option<&str>,
    threshold_pct: f64,
) -> anyhow::Result<()> {
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;

    let result = query::regress(&conn, model, metric, kv, threshold_pct)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // Print the machine-readable JSON row and the human-readable summary.
    println!("{}", serde_json::to_string(&result)?);
    eprintln!("{}", result.message);

    // Exit-code gate (mirrors regression_gate.sh idiom):
    // 125 = no champion or no data → bisect skip
    // 1 = regressed beyond threshold
    // 0 = within tolerance
    if result.champion_value.is_none() || result.latest_value.is_none() {
        std::process::exit(125);
    }
    if result.regressed {
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// deltas
// ---------------------------------------------------------------------------

pub(super) fn cmd_deltas(
    db_path: &Path,
    since_sha: &str,
    threshold_pct: f64,
    exit_code: bool,
) -> anyhow::Result<()> {
    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let rows =
        query::deltas(&conn, since_sha, Some(threshold_pct)).map_err(|e| anyhow::anyhow!("{e}"))?;
    for r in &rows {
        println!("{}", serde_json::to_string(r)?);
    }
    // No comparable rows (all baselines missing) → skip, not a false regression.
    // Exit 125 matches the regression_gate.sh "git bisect skip" idiom.
    if exit_code {
        let has_baseline = rows.iter().any(|r| r.baseline_value.is_some());
        if has_baseline {
            let any_regressed = rows.iter().any(|r| r.regressed);
            if any_regressed {
                std::process::exit(1);
            }
        } else {
            // All rows lack a baseline (new cells never measured before the SHA).
            // Treat as "no comparison possible" → skip (exit 125, bisect-safe).
            if !rows.is_empty() {
                std::process::exit(125);
            }
            // Zero rows with a known SHA means the SHA exists but every cell
            // was within threshold — clean, exit 0.
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// describe
// ---------------------------------------------------------------------------

#[allow(
    clippy::unwrap_used,
    reason = "Mutex::lock() cannot poison; Option/Result unwrap on values established by construction in this fn"
)]
pub(super) fn cmd_describe(
    db_path: &Path,
    observation_id: Option<i64>,
    run_id: Option<&str>,
    text: &str,
) -> anyhow::Result<()> {
    if observation_id.is_none() && run_id.is_none() {
        anyhow::bail!("either --observation-id or --run-id is required");
    }

    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;

    let updated = if let Some(oid) = observation_id {
        conn.execute(
            "UPDATE observations SET description = ?1 WHERE id = ?2",
            params![text, oid],
        )
        .context("update observation description")?
    } else {
        let rid = run_id.unwrap();
        conn.execute(
            "UPDATE observations SET description = ?1 WHERE run_id = ?2",
            params![text, rid],
        )
        .context("update observations description by run_id")?
    };

    println!("updated {updated} row(s)");
    Ok(())
}

// ---------------------------------------------------------------------------
// query (SELECT-only raw SQL)
// ---------------------------------------------------------------------------

pub(super) fn cmd_query(db_path: &Path, sql: &str) -> anyhow::Result<()> {
    // Guard: first non-whitespace word must be SELECT (case-insensitive).
    let first_word = sql.split_whitespace().next().unwrap_or("").to_uppercase();
    if first_word != "SELECT" {
        anyhow::bail!(
            "only SELECT statements are allowed (got '{first_word}'); \
             use 'rmlx metrics open' for interactive access"
        );
    }

    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let mut stmt = conn.prepare(sql).context("prepare SQL")?;

    // Column names for header row.
    let column_names: Vec<String> = stmt
        .column_names()
        .iter()
        .map(ToString::to_string)
        .collect();
    println!("{}", column_names.join("\t"));

    // Rows as TSV.
    let mut rows = stmt.query([]).context("execute query")?;
    while let Some(row) = rows.next().context("fetch row")? {
        let mut fields: Vec<String> = Vec::with_capacity(column_names.len());
        for i in 0..column_names.len() {
            let val: rusqlite::types::Value = row.get(i).context("get column")?;
            let s = match val {
                rusqlite::types::Value::Null => String::new(),
                rusqlite::types::Value::Integer(n) => n.to_string(),
                rusqlite::types::Value::Real(f) => f.to_string(),
                rusqlite::types::Value::Text(t) => t,
                rusqlite::types::Value::Blob(b) => format!("<blob:{}>", b.len()),
            };
            fields.push(s);
        }
        println!("{}", fields.join("\t"));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// open
// ---------------------------------------------------------------------------

pub(super) fn cmd_open(db_path: &Path, readonly: bool) -> anyhow::Result<()> {
    // Find sqlite3 on PATH.
    let sqlite3 = which_sqlite3().ok_or_else(|| {
        anyhow::anyhow!(
            "'sqlite3' not found on PATH; install it with 'brew install sqlite' \
             or 'apt-get install sqlite3'"
        )
    })?;

    let mut cmd = std::process::Command::new(&sqlite3);
    if readonly {
        cmd.arg("-readonly");
    }
    cmd.arg(db_path);

    // Inherit all stdio so the interactive session works.
    let status = cmd.status().context("launch sqlite3")?;
    if !status.success() {
        std::process::exit(status.code().unwrap_or(1));
    }
    Ok(())
}

fn which_sqlite3() -> Option<PathBuf> {
    std::env::var_os("PATH").as_ref().and_then(|path| {
        std::env::split_paths(path).find_map(|dir| {
            let candidate = dir.join("sqlite3");
            if candidate.is_file() {
                Some(candidate)
            } else {
                None
            }
        })
    })
}
