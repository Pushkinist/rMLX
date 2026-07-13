//! Integration tests for `rmlx metrics` query/export/prompts/migrate subcommands.
//!
//! Covers: best, rank, compare, history, describe, query, export, prompts, migrate.
//! Uses `std::process::Command` against the compiled binary (same pattern as
//! the existing metrics_lifecycle.rs tests).
//!
//! All tests are fast (no model load) and run without `#[ignore]`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp
)]

use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn rmlx_bin() -> PathBuf {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo");
    let workspace_root = PathBuf::from(manifest_dir)
        .parent() // crates/
        .and_then(|p| p.parent()) // workspace root
        .expect("workspace root from CARGO_MANIFEST_DIR")
        .to_path_buf();
    let debug = workspace_root.join("target/debug/rmlx");
    assert!(
        debug.exists(),
        "target/debug/rmlx not found at {} — run `cargo build -p rmlx-cli` first",
        debug.display()
    );
    debug
}

/// Initialize a fresh DB and return its path.
fn init_db(td: &tempfile::TempDir) -> PathBuf {
    let db = td.path().join("runs.db");
    let out = Command::new(rmlx_bin())
        .args(["metrics", "--db"])
        .arg(&db)
        .arg("init")
        .output()
        .expect("launch rmlx metrics init");
    assert!(
        out.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    db
}

/// Run `rmlx metrics --db <db> <subargs...>` and return Output.
fn run_metrics(db: &Path, subargs: &[&str]) -> std::process::Output {
    Command::new(rmlx_bin())
        .arg("metrics")
        .arg("--db")
        .arg(db)
        .args(subargs)
        .output()
        .expect("launch rmlx")
}

/// Filter tracing log lines out of stdout. The binary emits INFO lines like
/// `<ESC>[2m<timestamp>...rmlx start...` to stdout. We keep only lines that
/// look like JSON (starting with `{`) or TSV/text data.
fn json_lines(stdout: &str) -> Vec<&str> {
    stdout.lines().filter(|l| l.starts_with('{')).collect()
}

/// Insert a single observation into the DB using `record --inline`.
fn record_one(db: &Path, ts: &str, value: f64, backend: &str) {
    let json = serde_json::json!({
        "backend": backend,
        "backend_version": "0.2.8",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "test_prompt", "body": "Hello world" },
        "ts_utc": ts,
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": "decode_tps_warm", "value": value }]
    })
    .to_string();

    let out = run_metrics(db, &["record", "--inline", &json]);
    assert!(
        out.status.success(),
        "record failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Resolve the prompt_id for a named prompt via raw rusqlite.
fn prompt_id_for(db: &Path, name: &str) -> i64 {
    let conn = rusqlite::Connection::open(db).unwrap();
    conn.query_row(
        "SELECT id FROM prompts WHERE name = ?1 ORDER BY first_seen_utc DESC LIMIT 1",
        rusqlite::params![name],
        |r| r.get(0),
    )
    .expect("prompt not found")
}

// ---------------------------------------------------------------------------
// best
// ---------------------------------------------------------------------------

/// `best` returns the champion (higher of two decode_tps_warm values).
#[test]
fn best_returns_champion_json() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // Insert lower value first, then higher value.
    record_one(&db, "2026-01-01T00:00:00Z", 80.0, "rmlx");
    record_one(&db, "2026-01-02T00:00:00Z", 120.0, "rmlx");

    let pid = prompt_id_for(&db, "test_prompt");

    let out = run_metrics(
        &db,
        &[
            "best",
            "--backend",
            "rmlx",
            "--namespace",
            "mlx-community",
            "--model",
            "gemma-4-e4b-it-mxfp8",
            "--weight-quant",
            "mxfp8",
            "--kv-quant",
            "k8v8",
            "--ctx-max",
            "8192",
            "--prompt-id",
            &pid.to_string(),
            "--metric",
            "decode_tps_warm",
        ],
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "best failed: stderr={stderr} stdout={stdout}"
    );

    let lines = json_lines(&stdout);
    assert_eq!(lines.len(), 1, "expected 1 JSON line: {stdout}");
    let v: serde_json::Value = serde_json::from_str(lines[0]).expect("parse JSON");
    let value = v["value"].as_f64().expect("value field");
    assert!(
        (value - 120.0).abs() < 0.01,
        "expected champion=120.0, got {value}"
    );
}

// ---------------------------------------------------------------------------
// rank
// ---------------------------------------------------------------------------

/// `rank` orders rows best-first and returns correct count.
#[test]
fn rank_orders_by_metric() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // Three different cells (different kv_quant) with different TPS.
    for (kv, tps, ts) in &[
        ("k8v4", 50.0_f64, "2026-01-01T00:00:00Z"),
        ("k8v8", 100.0_f64, "2026-01-02T00:00:00Z"),
        ("none", 75.0_f64, "2026-01-03T00:00:00Z"),
    ] {
        let json = serde_json::json!({
            "backend": "rmlx",
            "backend_version": "0.2.8",
            "model_namespace": "mlx-community",
            "model": "gemma-4-e4b-it-mxfp8",
            "weight_quant": "mxfp8",
            "kv_quant": kv,
            "ctx_max": 8192,
            "prompt": { "name": "test_prompt", "body": "Hello world" },
            "ts_utc": ts,
            "hardware_tag": "m5_max_128gb",
            "metrics": [{ "name": "decode_tps_warm", "value": tps }]
        })
        .to_string();
        let out = run_metrics(&db, &["record", "--inline", &json]);
        assert!(out.status.success(), "record failed");
    }

    let out = run_metrics(
        &db,
        &["rank", "--metric", "decode_tps_warm", "--limit", "10"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "rank failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let lines = json_lines(&stdout);
    assert_eq!(lines.len(), 3, "expected 3 JSON lines, got: {stdout}");

    // First line should have highest value (100.0).
    let first: serde_json::Value = serde_json::from_str(lines[0]).expect("parse first line");
    let first_val = first["value"].as_f64().unwrap();
    assert!(
        (first_val - 100.0).abs() < 0.01,
        "expected first=100.0, got {first_val}"
    );

    // Last line should have lowest (50.0).
    let last: serde_json::Value = serde_json::from_str(lines[2]).expect("parse last line");
    let last_val = last["value"].as_f64().unwrap();
    assert!(
        (last_val - 50.0).abs() < 0.01,
        "expected last=50.0, got {last_val}"
    );
}

// ---------------------------------------------------------------------------
// history
// ---------------------------------------------------------------------------

/// `history` returns observations in chronological order.
#[test]
fn history_returns_chronological() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // Insert 3 obs with out-of-order timestamps.
    record_one(&db, "2026-03-01T00:00:00Z", 90.0, "rmlx");
    record_one(&db, "2026-01-01T00:00:00Z", 70.0, "rmlx");
    record_one(&db, "2026-02-01T00:00:00Z", 80.0, "rmlx");

    let pid = prompt_id_for(&db, "test_prompt");

    let out = run_metrics(
        &db,
        &[
            "history",
            "--backend",
            "rmlx",
            "--namespace",
            "mlx-community",
            "--model",
            "gemma-4-e4b-it-mxfp8",
            "--weight-quant",
            "mxfp8",
            "--kv-quant",
            "k8v8",
            "--ctx-max",
            "8192",
            "--prompt-id",
            &pid.to_string(),
        ],
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "history failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let lines = json_lines(&stdout);
    assert_eq!(lines.len(), 3, "expected 3 JSON lines: {stdout}");

    // Values should be ascending in chronological order: 70, 80, 90.
    let vals: Vec<f64> = lines
        .iter()
        .map(|l| {
            let v: serde_json::Value = serde_json::from_str(l).unwrap();
            v["value"].as_f64().unwrap()
        })
        .collect();
    assert!(
        vals[0] < vals[1] && vals[1] < vals[2],
        "not ordered: {vals:?}"
    );
}

// ---------------------------------------------------------------------------
// compare
// ---------------------------------------------------------------------------

/// `compare` returns one row with both backends' champions for same cell.
#[test]
fn compare_two_backends() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // rmlx observation.
    record_one(&db, "2026-01-01T00:00:00Z", 100.0, "rmlx");

    // mlx_lm observation for same model/prompt/cell.
    let json = serde_json::json!({
        "backend": "mlx_lm",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "test_prompt", "body": "Hello world" },
        "ts_utc": "2026-01-02T00:00:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": "decode_tps_warm", "value": 85.0 }]
    })
    .to_string();
    let out = run_metrics(&db, &["record", "--inline", &json]);
    assert!(out.status.success(), "record mlx_lm failed");

    let out = run_metrics(
        &db,
        &[
            "compare",
            "--backends",
            "rmlx,mlx_lm",
            "--metric",
            "decode_tps_warm",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "compare failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let lines = json_lines(&stdout);
    assert_eq!(lines.len(), 1, "expected 1 JSON line: {stdout}");

    let v: serde_json::Value = serde_json::from_str(lines[0]).unwrap();
    // per_backend should have 2 entries.
    let per_backend = v["per_backend"].as_array().expect("per_backend array");
    assert_eq!(per_backend.len(), 2, "expected 2 backends in compare row");
}

// ---------------------------------------------------------------------------
// describe
// ---------------------------------------------------------------------------

/// `describe --observation-id` sets description on one row.
#[test]
fn describe_by_observation_id_updates_row() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);
    record_one(&db, "2026-01-01T00:00:00Z", 100.0, "rmlx");

    // Get the observation id.
    let conn = rusqlite::Connection::open(&db).unwrap();
    let oid: i64 = conn
        .query_row("SELECT id FROM observations LIMIT 1", [], |r| r.get(0))
        .unwrap();
    drop(conn);

    let out = run_metrics(
        &db,
        &[
            "describe",
            "--observation-id",
            &oid.to_string(),
            "--text",
            "test description",
        ],
    );
    assert!(
        out.status.success(),
        "describe failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("updated 1 row"),
        "expected 'updated 1 row': {stdout}"
    );

    // Verify DB.
    let conn = rusqlite::Connection::open(&db).unwrap();
    let desc: Option<String> = conn
        .query_row(
            "SELECT description FROM observations WHERE id = ?1",
            rusqlite::params![oid],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        desc.as_deref(),
        Some("test description"),
        "description not set"
    );
}

/// `describe --run-id` updates all observations in that run.
#[test]
fn describe_by_run_id_updates_all_rows_in_run() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // Insert a record with two metrics (two observation rows, same run_id).
    let json = serde_json::json!({
        "backend": "rmlx",
        "backend_version": "0.2.8",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "test_prompt", "body": "Hello world" },
        "ts_utc": "2026-01-01T00:00:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [
            { "name": "decode_tps_warm", "value": 100.0 },
            { "name": "prefill_tps", "value": 200.0 }
        ]
    })
    .to_string();
    let out = run_metrics(&db, &["record", "--inline", &json]);
    assert!(out.status.success(), "record failed");

    // Get the run_id.
    let conn = rusqlite::Connection::open(&db).unwrap();
    let rid: String = conn
        .query_row("SELECT run_id FROM observations LIMIT 1", [], |r| r.get(0))
        .unwrap();
    drop(conn);

    let out = run_metrics(
        &db,
        &["describe", "--run-id", &rid, "--text", "run-level note"],
    );
    assert!(
        out.status.success(),
        "describe by run_id failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("updated 2 row"),
        "expected 'updated 2 rows': {stdout}"
    );
}

// ---------------------------------------------------------------------------
// query
// ---------------------------------------------------------------------------

/// `query "SELECT ..."` works and emits TSV with header.
#[test]
fn query_select_works() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);
    record_one(&db, "2026-01-01T00:00:00Z", 100.0, "rmlx");

    let out = run_metrics(
        &db,
        &["query", "SELECT backend, metric, value FROM observations"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "query failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Strip tracing lines. TSV lines don't start with `{` but the header contains "backend".
    let tsv_lines: Vec<&str> = stdout
        .lines()
        .filter(|l| !l.contains("rmlx start") && !l.is_empty())
        .filter(|l| {
            // Keep lines that look like TSV (contain tabs or known column names).
            !l.trim_start().starts_with('\x1b') && !l.contains("INFO") && !l.contains("[2m")
        })
        .collect();

    assert!(
        tsv_lines.len() >= 2,
        "expected header + data row, got: {stdout}"
    );
    assert!(
        tsv_lines[0].contains("backend"),
        "header missing 'backend': {stdout}"
    );
    // Data row should contain rmlx.
    assert!(
        tsv_lines.iter().any(|l| l.contains("rmlx")),
        "data row missing 'rmlx': {stdout}"
    );
}

/// `query "DROP TABLE ..."` must be refused with non-zero exit.
#[test]
fn query_refuses_non_select() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let out = run_metrics(&db, &["query", "DROP TABLE prompts"]);
    assert!(
        !out.status.success(),
        "expected non-zero exit for DROP TABLE"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("SELECT") || stderr.contains("only"),
        "expected helpful error message: {stderr}"
    );
}

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

/// `export --markdown` emits a markdown file with a known header.
#[test]
fn export_markdown_contains_header() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);
    record_one(&db, "2026-01-01T00:00:00Z", 100.0, "rmlx");

    let out = run_metrics(&db, &["export", "--markdown"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "export --markdown failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        stdout.contains("# BENCHMARK_CHAMPIONS"),
        "expected '# BENCHMARK_CHAMPIONS' header: {stdout}"
    );
}

/// `export --csv` output contains the CSV header columns.
#[test]
fn export_csv_first_line_is_header() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);
    record_one(&db, "2026-01-01T00:00:00Z", 100.0, "rmlx");

    let out = run_metrics(&db, &["export", "--csv"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "export --csv failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // The CSV header row contains "backend" and "metric". Find it.
    let header_line = stdout
        .lines()
        .find(|l| l.contains("backend") && l.contains("metric"));
    assert!(
        header_line.is_some(),
        "no CSV header line found in: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// prompts
// ---------------------------------------------------------------------------

/// `prompts sync` against the real rMLX/prompts/ dir, then `prompts list` shows all 4 prompts.
#[test]
fn prompts_list_after_sync_shows_real_files() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // Locate the repo root (workspace_root from manifest_dir).
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let workspace_root = PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();
    let prompts_dir = workspace_root.join("prompts");

    // We need RMLX_REPO_ROOT so `prompts sync` finds the prompts/ dir.
    let out = Command::new(rmlx_bin())
        .env("RMLX_REPO_ROOT", &workspace_root)
        .arg("metrics")
        .arg("--db")
        .arg(&db)
        .arg("prompts")
        .arg("sync")
        .output()
        .expect("launch rmlx prompts sync");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "prompts sync failed: stderr={stderr} stdout={stdout}"
    );

    // Count JSON files in the real prompts/ dir that `prompts sync` would
    // actually ingest. `parse_prompt_file` (rmlx-metrics::prompts) requires
    // a top-level `messages` or `body` key; calibration-style files (e.g.
    // `calibration_default.json` for head-budget calibration)
    // have a `prompts` array instead and are skipped with a `tracing::warn!`.
    // Count only the syncable files so this test stays accurate when new
    // calibration-only JSON files are added under prompts/.
    let expected_count = std::fs::read_dir(&prompts_dir)
        .expect("read prompts dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("json"))
        .filter(|e| {
            let Ok(raw) = std::fs::read_to_string(e.path()) else {
                return false;
            };
            let Ok(obj) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(&raw)
            else {
                return false;
            };
            obj.contains_key("messages") || obj.contains_key("body")
        })
        .count();

    let out = Command::new(rmlx_bin())
        .env("RMLX_REPO_ROOT", &workspace_root)
        .arg("metrics")
        .arg("--db")
        .arg(&db)
        .arg("prompts")
        .arg("list")
        .output()
        .expect("launch rmlx prompts list");
    let list_stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "prompts list failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // Count data lines: lines that don't start with 'id' (header), '-' (separator),
    // are not empty, and don't contain tracing markers.
    let data_lines = list_stdout
        .lines()
        .filter(|l| {
            !l.is_empty()
                && !l.starts_with("id")
                && !l.starts_with('-')
                && !l.contains("no prompts")
                && !l.contains("INFO")
                && !l.contains("[2m")
                && !l.contains("rmlx start")
        })
        .count();

    assert_eq!(
        data_lines, expected_count,
        "expected {expected_count} prompts after sync, got {data_lines}: {list_stdout}"
    );
}

/// `prompts get --name <n>` prints the body.
#[test]
fn prompts_get_returns_body() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // Add a prompt directly via record.
    record_one(&db, "2026-01-01T00:00:00Z", 100.0, "rmlx");

    let out = run_metrics(&db, &["prompts", "get", "--name", "test_prompt"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "prompts get failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Body was "Hello world" → JSON string.
    assert!(
        stdout.contains("Hello world"),
        "expected body in output: {stdout}"
    );
}

// ---------------------------------------------------------------------------
// migrate (dryish smoke)
// ---------------------------------------------------------------------------

/// `migrate --rmlx-glob <tmpdir>/*.jsonl` ingests one valid JSONL file.
#[test]
fn migrate_dryish_smoke() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // Write a minimal perf-iter JSONL file.
    let jsonl_row = serde_json::json!({
        "ts_utc": "2026-01-01T00:00:00Z",
        "model_path": "/opt/open-models/mlx-community__gemma-4-e4b-it-mxfp8",
        "kv_quant": "k8v8",
        "decode_tps_mean": 100.0,
        "decode_tps_stddev": 1.5,
        "step_ms_mean": 10.0,
        "git_sha": "abc1234",
        "build_profile": "release",
        "notes": "smoke test row"
    });
    let jsonl_path = td.path().join("smoke.jsonl");
    std::fs::write(&jsonl_path, format!("{jsonl_row}\n")).unwrap();

    // Point RMLX_REPO_ROOT to the real workspace so migrate finds prompts/longctx_4k.json.
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    let workspace_root = PathBuf::from(manifest_dir)
        .parent()
        .and_then(|p| p.parent())
        .expect("workspace root")
        .to_path_buf();

    let glob = format!("{}/*.jsonl", td.path().display());
    let out = Command::new(rmlx_bin())
        .env("RMLX_REPO_ROOT", &workspace_root)
        .arg("metrics")
        .arg("--db")
        .arg(&db)
        .arg("migrate")
        .arg("--rmlx-glob")
        .arg(&glob)
        .output()
        .expect("launch rmlx migrate");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "migrate failed: stderr={stderr} stdout={stdout}"
    );

    // Extract the JSON report line (first line starting with '{').
    let json_line = stdout
        .lines()
        .find(|l| l.starts_with('{'))
        .unwrap_or_else(|| panic!("no JSON line in migrate output: {stdout}"));

    let report: serde_json::Value =
        serde_json::from_str(json_line).expect("parse migrate JSON report");
    let inserted = report["rmlx_jsonl_rows_inserted"]
        .as_u64()
        .expect("rmlx_jsonl_rows_inserted field");
    assert!(
        inserted > 0,
        "expected at least 1 row inserted, got report: {report}"
    );
}
