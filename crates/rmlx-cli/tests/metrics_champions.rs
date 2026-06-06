//! Integration tests for `rmlx metrics champions`.
//!
//! Uses `std::process::Command` against the compiled binary (same pattern as
//! metrics_query_export.rs). All tests are fast (no model load).

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

fn run_metrics(db: &Path, subargs: &[&str]) -> std::process::Output {
    Command::new(rmlx_bin())
        .arg("metrics")
        .arg("--db")
        .arg(db)
        .args(subargs)
        .output()
        .expect("launch rmlx")
}

/// Insert one observation via `record --inline`.
fn record_one(db: &Path, backend: &str, model: &str, metric: &str, value: f64, ts: &str) {
    let json = serde_json::json!({
        "backend": backend,
        "model_namespace": "mlx-community",
        "model": model,
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "test_prompt", "body": "Hello world" },
        "ts_utc": ts,
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": metric, "value": value }]
    })
    .to_string();

    let out = run_metrics(db, &["record", "--inline", &json]);
    assert!(
        out.status.success(),
        "record failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Default (no filter) render must:
/// - exit 0
/// - start with "# Champion records — all backends"
/// - contain a markdown table header line with `model_namespace`
/// - contain at least one data row after the separator
#[test]
fn champions_default_renders_markdown_with_scope_header() {
    let td = tempfile::TempDir::new().unwrap();
    let db = init_db(&td);

    record_one(
        &db,
        "rmlx",
        "gemma-4-e2b-it-mxfp8",
        "decode_tps_warm",
        95.0,
        "2026-05-01T10:00:00Z",
    );

    let out = run_metrics(&db, &["champions"]);
    assert!(
        out.status.success(),
        "exit non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("# Champion records — all backends"),
        "missing scope header. stdout:\n{stdout}"
    );
    assert!(
        stdout.contains("model_namespace"),
        "missing table header. stdout:\n{stdout}"
    );
    // At least one data row exists (contains the model name).
    assert!(
        stdout.contains("gemma-4-e2b-it-mxfp8"),
        "no data row. stdout:\n{stdout}"
    );
}

/// With `--backend rmlx` the scope header must say `backend=rmlx` and
/// the table must not include a per-metric backend column.
#[test]
fn champions_with_backend_filter_renders_per_backend_view() {
    let td = tempfile::TempDir::new().unwrap();
    let db = init_db(&td);

    record_one(
        &db,
        "rmlx",
        "gemma-4-e2b-it-mxfp8",
        "decode_tps_warm",
        95.0,
        "2026-05-01T10:00:00Z",
    );
    record_one(
        &db,
        "mlx_lm",
        "gemma-4-e2b-it-mxfp8",
        "decode_tps_warm",
        110.0,
        "2026-05-01T11:00:00Z",
    );

    let out = run_metrics(&db, &["champions", "--backend", "rmlx"]);
    assert!(
        out.status.success(),
        "exit non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("backend=rmlx"),
        "missing per-backend scope. stdout:\n{stdout}"
    );
    // When --backend is set the header must NOT have a "decode_tps_warm backend" column.
    assert!(
        !stdout.contains("decode_tps_warm backend"),
        "unexpected backend column when --backend filter is active. stdout:\n{stdout}"
    );
}

/// `--jsonl` must emit one JSON object per line, each parseable with a `model` field.
#[test]
fn champions_jsonl_emits_one_row_per_line() {
    let td = tempfile::TempDir::new().unwrap();
    let db = init_db(&td);

    record_one(
        &db,
        "rmlx",
        "gemma-4-e2b-it-mxfp8",
        "decode_tps_warm",
        95.0,
        "2026-05-01T10:00:00Z",
    );
    record_one(
        &db,
        "rmlx",
        "qwen3-8b-4bit",
        "decode_tps_warm",
        80.0,
        "2026-05-01T10:00:00Z",
    );

    let out = run_metrics(&db, &["champions", "--jsonl"]);
    assert!(
        out.status.success(),
        "exit non-zero: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    let json_rows: Vec<&str> = stdout.lines().filter(|l| l.starts_with('{')).collect();
    assert_eq!(
        json_rows.len(),
        2,
        "expected 2 JSONL rows, got {}. stdout:\n{stdout}",
        json_rows.len()
    );

    for line in &json_rows {
        let v: serde_json::Value = serde_json::from_str(line).expect("line must be valid JSON");
        assert!(v.get("model").is_some(), "missing 'model' field in: {line}");
        assert!(
            v.get("metrics").is_some(),
            "missing 'metrics' field in: {line}"
        );
    }
}

/// Against the live `metrics/runs.db`, `metrics champions` must exit 0 and
/// produce at least one champion row (or gracefully emit an empty table).
/// Skipped if DB is absent.
#[test]
fn champions_real_db_runs() {
    let real_db = PathBuf::from(
        std::env::var("RMLX_METRICS_DB").unwrap_or_else(|_| "metrics/runs.db".to_owned()),
    );

    if !real_db.exists() {
        eprintln!(
            "skipping champions_real_db_runs: {} not found",
            real_db.display()
        );
        return;
    }

    let out = Command::new(rmlx_bin())
        .args(["metrics", "--db"])
        .arg(&real_db)
        .arg("champions")
        .output()
        .expect("launch rmlx");

    assert!(
        out.status.success(),
        "exit non-zero against real DB: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("# Champion records"),
        "missing header in real-DB output. stdout:\n{}",
        &stdout[..stdout.len().min(500)]
    );
}
