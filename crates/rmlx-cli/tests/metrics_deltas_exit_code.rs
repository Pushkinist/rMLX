//! Integration tests for `rmlx metrics deltas` exit-code behaviour.
//!
//! Covers:
//! - exit 1 when any `DeltaRow.regressed == true` (default `--exit-code`)
//! - exit 0 when no regression even though a delta row exists (improvement)
//! - `--exit-code=false` always exits 0 regardless of regressions
//! - no-baseline case (all rows have `baseline_value == null`) exits 125, not 1
//! - zero rows (every cell within threshold) exits 0
//! - `--prompt-prefix` compares only the cells whose prompt name carries it
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

use rmlx_metrics::{
    ingest::{PromptRef, RunRecord},
    migrate,
    recorder::Recorder,
};
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

/// One observation of `decode_tps_warm` on a cell keyed by its own prompt.
/// Prompts are content-addressed, so each name carries its own body.
fn make_prompt_run(prompt_name: &str, value: f64, ts: &str, git_sha: &str) -> RunRecord {
    let mut run = make_run(
        "rmlx",
        "gemma-4-e4b-it-mxfp8",
        "decode_tps_warm",
        value,
        ts,
        Some(git_sha),
    );
    run.prompt = PromptRef::ByBody {
        name: prompt_name.into(),
        body: json!(format!("body of {prompt_name}")),
        notes: None,
        tokens_approx: Some(4),
    };
    run
}

/// Two cells measured before and after `sha_base`: one under an
/// `ssd-canary-` prompt, one under an unrelated prompt. Each argument is that
/// cell's value after the baseline of 100.
fn seed_two_prompt_db(td: &tempfile::TempDir, canary_after: f64, other_after: f64) -> PathBuf {
    let mut conn = open_mem();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    for (prompt, after) in [
        ("ssd-canary-populate", canary_after),
        ("perf-canary-4096", other_after),
    ] {
        rec.record_run(&make_prompt_run(
            prompt,
            100.0,
            "2026-05-01T10:00:00Z",
            "sha_base",
        ))
        .unwrap();
        rec.record_run(&make_prompt_run(
            prompt,
            after,
            "2026-05-10T10:00:00Z",
            "sha_after",
        ))
        .unwrap();
    }
    let db = td.path().join("two_prompt.db");
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

/// A regression in a cell outside the prefix does not fail a gate scoped to
/// the prefix, and the same DB fails the unscoped gate.
#[test]
fn deltas_prompt_prefix_ignores_an_unrelated_regression() {
    let td = tempfile::tempdir().unwrap();
    let db = seed_two_prompt_db(&td, 99.0, 50.0);
    let scoped = run_deltas(
        &db,
        &["--since-sha", "sha_base", "--prompt-prefix", "ssd-canary-"],
    );
    assert_eq!(
        scoped.status.code(),
        Some(0),
        "scoped gate failed on an unrelated cell; stdout={}  stderr={}",
        String::from_utf8_lossy(&scoped.stdout),
        String::from_utf8_lossy(&scoped.stderr),
    );
    let unscoped = run_deltas(&db, &["--since-sha", "sha_base"]);
    assert_eq!(
        unscoped.status.code(),
        Some(1),
        "the unrelated regression must still fail the unscoped gate"
    );
}

/// A regression in a cell under the prefix fails the scoped gate, and only
/// that cell is printed.
#[test]
fn deltas_prompt_prefix_fails_on_its_own_regression() {
    let td = tempfile::tempdir().unwrap();
    let db = seed_two_prompt_db(&td, 50.0, 40.0);
    let out = run_deltas(
        &db,
        &["--since-sha", "sha_base", "--prompt-prefix", "ssd-canary-"],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.code(),
        Some(1),
        "scoped gate missed its own cell's regression; stdout={stdout}  stderr={}",
        String::from_utf8_lossy(&out.stderr),
    );
    assert_eq!(
        stdout.lines().count(),
        1,
        "only the canary cell may be printed: {stdout}"
    );
    assert!(stdout.contains("\"current_value\":50.0"), "{stdout}");
}
