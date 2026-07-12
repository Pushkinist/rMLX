//! Integration tests for `rmlx metrics deltas` exit-code behaviour.
//!
//! Covers:
//! - exit 1 when any `DeltaRow.regressed == true` (default `--exit-code`)
//! - exit 0 when no regression even though a delta row exists (improvement)
//! - `--exit-code=false` always exits 0 regardless of regressions
//! - no-baseline case (all rows have `baseline_value == null`) exits 125, not 1
//! - zero rows (every cell within threshold) exits 0
//!
//! All tests seed an in-memory DB via `rusqlite` + `rmlx_metrics`, then write
//! it to a tempfile before spawning the rmlx binary.

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

use rmlx_metrics::{ingest::RunRecord, migrate, recorder::Recorder};
use rusqlite::Connection;
use serde_json::json;

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

/// Open an in-memory SQLite DB with the rmlx-metrics schema applied.
fn open_mem() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    migrate::run_pending(&mut conn).unwrap();
    conn
}

/// Persist an in-memory connection to a file path via VACUUM INTO.
fn persist_to(conn: &Connection, path: &Path) {
    conn.execute_batch(&format!("VACUUM INTO '{}'", path.display()))
        .unwrap();
}

/// Minimal RunRecord fixture.
///
/// `RunRecord` is `#[non_exhaustive]`, so an out-of-crate struct literal is a
/// compile error by design. External construction goes through either
/// `RunRecordBuilder` (rMLX's own emitters) or the §8.5 wire shape, as here —
/// this fixture needs to mint arbitrary backends and git SHAs, which the
/// builder deliberately does not allow.
fn make_run(
    backend: &str,
    model: &str,
    metric: &str,
    value: f64,
    ts: &str,
    git_sha: Option<&str>,
) -> RunRecord {
    serde_json::from_value(json!({
        "schema_version": rmlx_metrics::ingest::RECORD_SCHEMA_VERSION,
        "backend": backend,
        "backend_version": "0.0.1",
        "model_namespace": "mlx-community",
        "model": model,
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": {
            "name": "test_prompt",
            "body": "the quick brown fox",
            "tokens_approx": 4,
        },
        "ts_utc": ts,
        "git_sha": git_sha,
        "build_profile": "release",
        "hardware_tag": "m5_max_128gb",
        "prompt_tokens": 4,
        "max_tokens": 32,
        "temperature": 0.0,
        "seed": 0,
        "n_warmups": 1,
        "n_measure": 3,
        "metrics": [{ "name": metric, "value": value }],
    }))
    .expect("valid §8.5 record")
}

/// Seed a DB with a baseline observation at `sha_base`, then a regressed
/// post-baseline observation. Returns the DB file path.
fn seed_regressed_db(td: &tempfile::TempDir) -> PathBuf {
    let mut conn = open_mem();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");

    // Baseline: decode_tps_warm = 100 at sha_base.
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e4b-it-mxfp8",
        "decode_tps_warm",
        100.0,
        "2026-05-01T10:00:00Z",
        Some("sha_base"),
    ))
    .unwrap();
    // Regressed: decode_tps_warm = 50 after sha_base (>5% drop).
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e4b-it-mxfp8",
        "decode_tps_warm",
        50.0,
        "2026-05-10T10:00:00Z",
        Some("sha_after"),
    ))
    .unwrap();

    let db = td.path().join("regressed.db");
    persist_to(&conn, &db);
    db
}

/// Seed a DB with a baseline and an *improved* post-baseline observation
/// (no regression). Returns the DB file path.
fn seed_improved_db(td: &tempfile::TempDir) -> PathBuf {
    let mut conn = open_mem();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");

    // Baseline: decode_tps_warm = 100 at sha_base.
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e4b-it-mxfp8",
        "decode_tps_warm",
        100.0,
        "2026-05-01T10:00:00Z",
        Some("sha_base"),
    ))
    .unwrap();
    // Improved: decode_tps_warm = 120 after sha_base (improvement, not regression).
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e4b-it-mxfp8",
        "decode_tps_warm",
        120.0,
        "2026-05-10T10:00:00Z",
        Some("sha_after"),
    ))
    .unwrap();

    let db = td.path().join("improved.db");
    persist_to(&conn, &db);
    db
}

/// Seed a DB where the cell's observations are ALL newer than the baseline
/// SHA's timestamp, so `baseline_value` is `None` for every delta row.
///
/// Layout:
/// sha_anchor → ts 2026-05-01 (anchor so sha_anchor resolves)
/// new_cell → ts 2026-05-10 (after the anchor ts, no pre-anchor best)
///
/// The `deltas` query finds MIN(ts_utc) for sha_anchor = 2026-05-01.
/// The new_cell observation at 2026-05-10 is entirely post-baseline:
/// - pre-baseline best (ts <= 2026-05-01) = None → baseline_value = None
/// - delta row emitted because baseline_value.is_none() → row included
///
/// All rows in output have baseline_value = None → exit 125 path triggered.
fn seed_no_baseline_db(td: &tempfile::TempDir) -> PathBuf {
    let mut conn = open_mem();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");

    // Anchor observation: needed so sha_anchor resolves to a timestamp.
    rec.record_run(&make_run(
        "rmlx",
        "anchor-model",
        "decode_tps_warm",
        50.0,
        "2026-05-01T10:00:00Z",
        Some("sha_anchor"),
    ))
    .unwrap();

    // New cell — observations exist ONLY after the anchor timestamp.
    // baseline_value will be None for this cell (no pre-anchor data).
    rec.record_run(&make_run(
        "rmlx",
        "new-model",
        "decode_tps_warm",
        80.0,
        "2026-05-10T10:00:00Z",
        Some("sha_after"),
    ))
    .unwrap();

    let db = td.path().join("no_baseline.db");
    persist_to(&conn, &db);
    db
}

/// Seed a DB where all cells are within threshold of the baseline
/// (zero delta rows above threshold). Returns the DB file path.
fn seed_clean_db(td: &tempfile::TempDir) -> PathBuf {
    let mut conn = open_mem();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");

    // Baseline: 100 at sha_base.
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e4b-it-mxfp8",
        "decode_tps_warm",
        100.0,
        "2026-05-01T10:00:00Z",
        Some("sha_base"),
    ))
    .unwrap();
    // Within 5% threshold: decode_tps_warm = 99 (delta = -1%).
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e4b-it-mxfp8",
        "decode_tps_warm",
        99.0,
        "2026-05-10T10:00:00Z",
        Some("sha_after"),
    ))
    .unwrap();

    let db = td.path().join("clean.db");
    persist_to(&conn, &db);
    db
}

fn run_deltas(db: &Path, extra: &[&str]) -> std::process::Output {
    Command::new(rmlx_bin())
        .arg("metrics")
        .arg("--db")
        .arg(db)
        .arg("deltas")
        .args(extra)
        .output()
        .expect("failed to launch rmlx")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// Core DoD: exit 1 on regression, default `--exit-code`.
#[test]
fn deltas_exit_code_one_on_regression() {
    let td = tempfile::tempdir().unwrap();
    let db = seed_regressed_db(&td);
    let out = run_deltas(&db, &["--since-sha", "sha_base"]);
    assert_eq!(
        out.status.code(),
        Some(1),
        "expected exit 1 on regression; stdout={}  stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// No regression → exit 0 (improvement only).
#[test]
fn deltas_exit_zero_on_improvement() {
    let td = tempfile::tempdir().unwrap();
    let db = seed_improved_db(&td);
    let out = run_deltas(&db, &["--since-sha", "sha_base"]);
    // An improvement row exists but regressed == false → exit 0.
    assert_eq!(
        out.status.code(),
        Some(0),
        "expected exit 0 on improvement; stdout={}  stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// `--exit-code=false` must always exit 0, even with regressions.
#[test]
fn deltas_exit_code_false_always_zero() {
    let td = tempfile::tempdir().unwrap();
    let db = seed_regressed_db(&td);
    let out = run_deltas(&db, &["--since-sha", "sha_base", "--exit-code=false"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "--exit-code=false must suppress non-zero exit; stdout={}  stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// No-baseline case: all delta rows have `baseline_value == None` → exit 125,
/// not 1. This matches the regression_gate.sh "git bisect skip" idiom.
///
/// The DB has an "anchor" observation at sha_anchor's timestamp and a "new-model"
/// cell whose only observations are AFTER that timestamp. The `new-model` row
/// therefore has `baseline_value = None` (no pre-anchor best). Since all emitted
/// rows lack a baseline, the command exits 125 (bisect-skip), not 1 (regression).
#[test]
fn deltas_no_baseline_exits_125_not_1() {
    let td = tempfile::tempdir().unwrap();
    let db = seed_no_baseline_db(&td);
    let out = run_deltas(&db, &["--since-sha", "sha_anchor"]);
    // The anchor-model cell: baseline = 50, post-baseline best = 50 (no post-anchor obs) → delta 0% → not emitted.
    // The new-model cell: baseline_value = None (emitted because baseline_value.is_none()).
    // All emitted rows have baseline_value = None → exit 125.
    assert_eq!(
        out.status.code(),
        Some(125),
        "all-None-baseline should exit 125 (bisect skip), not 1; stdout={}  stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Zero rows within threshold → exit 0.
#[test]
fn deltas_within_threshold_exits_zero() {
    let td = tempfile::tempdir().unwrap();
    let db = seed_clean_db(&td);
    let out = run_deltas(&db, &["--since-sha", "sha_base"]);
    assert_eq!(
        out.status.code(),
        Some(0),
        "within-threshold should exit 0; stdout={}  stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}

/// Unknown SHA → non-zero exit (DB error, not regression).
#[test]
fn deltas_unknown_sha_exits_nonzero() {
    let td = tempfile::tempdir().unwrap();
    let db = seed_clean_db(&td);
    let out = run_deltas(&db, &["--since-sha", "sha_does_not_exist"]);
    assert_ne!(
        out.status.code(),
        Some(0),
        "unknown SHA should exit non-zero; stdout={}  stderr={}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr),
    );
}
