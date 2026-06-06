//! Integration tests for `rmlx metrics record` subcommand variants.
//!
//! Covers: --inline, --file, --stdin, --dry-run, --replay-pending.
//! Tests are fast (no model load). All tempdir paths are absolute.

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
    clippy::float_cmp,
    clippy::items_after_statements
)]

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

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

/// Initialize a fresh DB in `td` and return its absolute path.
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

/// Run `rmlx metrics --db <db> record <extra_args...>` and return Output.
fn run_record(db: &Path, extra: &[&str]) -> std::process::Output {
    Command::new(rmlx_bin())
        .arg("metrics")
        .arg("--db")
        .arg(db)
        .arg("record")
        .args(extra)
        .output()
        .expect("launch rmlx metrics record")
}

/// Count rows in the `observations` table.
fn obs_count(db: &Path) -> i64 {
    use rusqlite::Connection;
    let conn = Connection::open(db).unwrap();
    conn.query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap()
}

/// A valid §8.5 RunRecord with one decode_tps_warm observation.
fn valid_record_json() -> String {
    serde_json::json!({
        "backend": "rmlx",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": {
            "name": "test_prompt",
            "body": "Hello world"
        },
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [
            { "name": "decode_tps_warm", "value": 100.0 }
        ]
    })
    .to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// --inline inserts one observation.
#[test]
fn record_inline_inserts_observation() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let json = valid_record_json();
    let out = run_record(&db, &["--inline", &json]);
    assert!(
        out.status.success(),
        "record --inline failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        obs_count(&db),
        1,
        "expected 1 observation after inline insert"
    );
}

/// --file inserts and then deletes the source file on success.
#[test]
fn record_file_consumed_on_success() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let buf = td.path().join("record.json");
    std::fs::write(&buf, valid_record_json()).unwrap();

    let out = run_record(&db, &["--file", buf.to_str().unwrap()]);
    assert!(
        out.status.success(),
        "record --file failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(obs_count(&db), 1);
    assert!(
        !buf.exists(),
        "buffer file should have been deleted on success"
    );
}

/// --file with invalid JSON (unknown backend) leaves the file intact and exits non-zero.
#[test]
fn record_file_kept_on_failure() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let bad = serde_json::json!({
        "backend": "pytorch",           // not in whitelist
        "model_namespace": "mlx-community",
        "model": "whatever",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "p", "body": "x" },
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": "decode_tps_warm", "value": 1.0 }]
    })
    .to_string();

    let buf = td.path().join("bad.json");
    std::fs::write(&buf, &bad).unwrap();

    let out = run_record(&db, &["--file", buf.to_str().unwrap()]);
    assert!(
        !out.status.success(),
        "expected non-zero exit for invalid backend"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("pytorch")
            || stderr.contains("not a valid backend")
            || stderr.contains("whitelist"),
        "expected 'pytorch' or 'whitelist' in stderr, got: {stderr}"
    );
    assert!(buf.exists(), "buffer file should remain after failure");
    assert_eq!(obs_count(&db), 0);
}

/// --stdin inserts one observation.
#[test]
fn record_stdin_inserts_observation() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let json = valid_record_json();
    let mut child = Command::new(rmlx_bin())
        .arg("metrics")
        .arg("--db")
        .arg(&db)
        .arg("record")
        .arg("--stdin")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rmlx metrics record --stdin");

    use std::io::Write as _;
    child
        .stdin
        .take()
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();

    let out = child.wait_with_output().expect("wait for child");
    assert!(
        out.status.success(),
        "record --stdin failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(obs_count(&db), 1);
}

/// --dry-run validates but inserts nothing.
#[test]
fn record_dry_run_does_not_insert() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let json = valid_record_json();
    let out = run_record(&db, &["--inline", &json, "--dry-run"]);
    assert!(
        out.status.success(),
        "dry-run failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(
        obs_count(&db),
        0,
        "--dry-run must not insert any observations"
    );
}

/// --replay-pending processes all valid files and removes them.
#[test]
fn replay_pending_consumes_pending_dir() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // Use RMLX_HOME so replay looks in our tempdir via paths::ingest_buffer_dir().
    let pending = td.path().join("metrics/buffer/pending");
    std::fs::create_dir_all(&pending).unwrap();

    for i in 0..3 {
        std::fs::write(pending.join(format!("run{i}.json")), valid_record_json()).unwrap();
    }

    let out = Command::new(rmlx_bin())
        .env("RMLX_HOME", td.path())
        .arg("metrics")
        .arg("--db")
        .arg(&db)
        .arg("record")
        .arg("--replay-pending")
        .output()
        .expect("launch replay");

    assert!(
        out.status.success(),
        "replay failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(obs_count(&db), 3, "expected 3 observations after replay");

    let remaining: Vec<_> = std::fs::read_dir(&pending)
        .unwrap()
        .filter_map(Result::ok)
        .collect();
    assert!(remaining.is_empty(), "all pending files should be removed");
}

/// --replay-pending moves invalid files to failed/ and exits 2; valid ones are consumed.
#[test]
fn replay_pending_moves_invalid_to_failed() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let pending = td.path().join("metrics/buffer/pending");
    let failed = td.path().join("metrics/buffer/failed");
    std::fs::create_dir_all(&pending).unwrap();

    // 1 valid
    std::fs::write(pending.join("valid.json"), valid_record_json()).unwrap();

    // 1 invalid (unknown backend)
    let bad = serde_json::json!({
        "backend": "pytorch",
        "model_namespace": "mlx-community",
        "model": "whatever",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "p", "body": "x" },
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": "decode_tps_warm", "value": 1.0 }]
    })
    .to_string();
    std::fs::write(pending.join("bad.json"), &bad).unwrap();

    let out = Command::new(rmlx_bin())
        .env("RMLX_HOME", td.path())
        .arg("metrics")
        .arg("--db")
        .arg(&db)
        .arg("record")
        .arg("--replay-pending")
        .output()
        .expect("launch replay");

    // exit code must be 2
    assert_eq!(
        out.status.code(),
        Some(2),
        "expected exit 2; stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // valid inserted, invalid moved
    assert_eq!(
        obs_count(&db),
        1,
        "only the valid record should be inserted"
    );
    assert!(
        !pending.join("bad.json").exists(),
        "bad.json must leave pending/"
    );
    assert!(
        failed.join("bad.json").exists(),
        "bad.json must be in failed/"
    );
    assert!(
        !pending.join("valid.json").exists(),
        "valid.json must leave pending/"
    );
}
