//! Integration tests for the §8.5 run-identity contract (docs/METRICS_DB.md §8.5.1).
//!
//! Covers the three CLI ingest paths (`record --file`, `record --inline`,
//! `record --replay-pending`) plus `metrics identity` / `metrics validate`.
//! No model load — fast.

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
use std::process::Command;

/// Locate the cargo-built `rmlx` binary.
///
/// `CARGO_BIN_EXE_rmlx` is injected by Cargo's integration-test runner and
/// resolves to whatever profile/target dir this test binary was itself built
/// under — unlike a hard-coded `target/debug/rmlx`, it is never absent under
/// `--profile release-perf` or a custom `CARGO_TARGET_DIR`, and never
/// silently stale (exercising a different binary than the one under test).
fn rmlx_bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_rmlx"))
}

fn init_db(td: &tempfile::TempDir) -> PathBuf {
    let db = td.path().join("runs.db");
    let out = Command::new(rmlx_bin())
        .args(["metrics", "--db"])
        .arg(&db)
        .arg("init")
        .output()
        .expect("launch init");
    assert!(out.status.success());
    db
}

/// A §8.5 record for `backend` with the given `backend_version` (None ⇒ key absent).
fn record_json(backend: &str, version: Option<&str>) -> String {
    let mut obj = serde_json::json!({
        "backend": backend,
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "t", "body": "hi" },
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": "decode_tps_warm", "value": 100.0 }]
    });
    if let Some(v) = version {
        obj["backend_version"] = serde_json::Value::String(v.to_string());
    }
    obj.to_string()
}

fn record_inline(db: &Path, json: &str) -> std::process::Output {
    Command::new(rmlx_bin())
        .arg("metrics")
        .arg("--db")
        .arg(db)
        .args(["record", "--inline", json])
        .output()
        .expect("launch record")
}

fn obs_count(db: &Path) -> i64 {
    let conn = rusqlite::Connection::open(db).unwrap();
    conn.query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap()
}

// ── metrics identity ──────────────────────────────────────────────────────────

#[test]
fn identity_json_carries_semver_and_real_build_profile() {
    let out = Command::new(rmlx_bin())
        .args(["metrics", "identity", "--json"])
        .output()
        .expect("launch identity");
    assert!(out.status.success());

    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout.lines().last().expect("one JSON line on stdout");
    let v: serde_json::Value = serde_json::from_str(line).expect("stdout is clean JSON");

    assert_eq!(v["backend"], "rmlx");

    // backend_version is the workspace semver, not a literal or a git sha.
    let ver = v["backend_version"].as_str().expect("backend_version");
    assert_eq!(ver, env!("CARGO_PKG_VERSION"));
    assert_eq!(ver.split('.').count(), 3, "not semver: {ver}");

    // build_profile is a real Cargo profile dir name, never a debug_assertions guess.
    let profile = v["build_profile"].as_str().expect("build_profile");
    assert!(
        !profile.is_empty() && profile != "unknown",
        "got {profile:?}"
    );

    assert!(v.get("git_sha").is_some());
    assert!(v["hardware_tag"].as_str().is_some_and(|s| !s.is_empty()));
}

// ── Ingest path: record --inline ──────────────────────────────────────────────

#[test]
fn inline_rmlx_without_version_is_rejected() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let out = record_inline(&db, &record_json("rmlx", None));
    assert!(!out.status.success(), "missing version must be rejected");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("backend_version"), "unhelpful error: {err}");
    assert_eq!(obs_count(&db), 0, "nothing may be written");
}

#[test]
fn inline_rmlx_with_non_semver_version_is_rejected() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    // The exact junk classes found in the live DB.
    for junk in ["head", "379dcea", "1257883"] {
        let out = record_inline(&db, &record_json("rmlx", Some(junk)));
        assert!(!out.status.success(), "{junk:?} must be rejected");
    }
    assert_eq!(obs_count(&db), 0);
}

#[test]
fn inline_rmlx_with_semver_version_is_accepted() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let out = record_inline(&db, &record_json("rmlx", Some("0.2.8")));
    assert!(
        out.status.success(),
        "rejected: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(obs_count(&db), 1);
}

/// Cross-backend: llama.cpp has no semver (it emits a build_commit). It must
/// keep ingesting, or every non-rMLX bench breaks.
#[test]
fn inline_non_rmlx_backend_without_version_still_ingests() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);

    let out = record_inline(&db, &record_json("llama_cpp", None));
    assert!(
        out.status.success(),
        "cross-backend ingest broke: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(obs_count(&db), 1);
}

// ── Ingest path: record --file ────────────────────────────────────────────────

#[test]
fn file_rmlx_without_version_is_rejected_and_buffer_kept() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(&td);
    let buf = td.path().join("run.json");
    std::fs::write(&buf, record_json("rmlx", None)).unwrap();

    let out = Command::new(rmlx_bin())
        .arg("metrics")
        .arg("--db")
        .arg(&db)
        .arg("record")
        .arg("--file")
        .arg(&buf)
        .output()
        .expect("launch record");

    assert!(!out.status.success());
    assert_eq!(obs_count(&db), 0);
    assert!(buf.exists(), "rejected buffer file must be kept for triage");
}

// ── Ingest path: record --replay-pending ──────────────────────────────────────

/// A buffer emitted by an older build keeps ITS identity — replay never
/// re-stamps with the replaying binary's version.
#[test]
fn replay_pending_does_not_restamp_identity() {
    let td = tempfile::tempdir().unwrap();
    let home = td.path();
    let pending = home.join("metrics/buffer/pending");
    std::fs::create_dir_all(&pending).unwrap();

    let db = home.join("metrics/runs.db");
    let init = Command::new(rmlx_bin())
        .args(["metrics", "--db"])
        .arg(&db)
        .arg("init")
        .output()
        .expect("init");
    assert!(init.status.success());

    let mut rec: serde_json::Value =
        serde_json::from_str(&record_json("rmlx", Some("0.1.0"))).unwrap();
    rec["git_sha"] = serde_json::Value::String("deadbee".into());
    rec["build_profile"] = serde_json::Value::String("release-perf".into());
    std::fs::write(pending.join("old.json"), rec.to_string()).unwrap();

    let out = Command::new(rmlx_bin())
        .env("RMLX_HOME", home)
        .args(["metrics", "--db"])
        .arg(&db)
        .args(["record", "--replay-pending"])
        .output()
        .expect("launch replay");
    assert!(
        out.status.success(),
        "replay failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let conn = rusqlite::Connection::open(&db).unwrap();
    let (ver, sha, profile): (String, String, String) = conn
        .query_row(
            "SELECT backend_version, git_sha, build_profile FROM observations",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
        )
        .unwrap();

    assert_eq!(ver, "0.1.0", "replay must not re-stamp the version");
    assert_eq!(sha, "deadbee");
    assert_eq!(profile, "release-perf");
}

/// A pre-contract rMLX buffer (no version) is rejected into `failed/` instead of
/// silently adding another NULL-version row.
#[test]
fn replay_pending_moves_identity_less_rmlx_record_to_failed() {
    let td = tempfile::tempdir().unwrap();
    let home = td.path();
    let pending = home.join("metrics/buffer/pending");
    let failed = home.join("metrics/buffer/failed");
    std::fs::create_dir_all(&pending).unwrap();

    let db = home.join("metrics/runs.db");
    let init = Command::new(rmlx_bin())
        .args(["metrics", "--db"])
        .arg(&db)
        .arg("init")
        .output()
        .expect("init");
    assert!(init.status.success());

    std::fs::write(pending.join("bad.json"), record_json("rmlx", None)).unwrap();

    let out = Command::new(rmlx_bin())
        .env("RMLX_HOME", home)
        .args(["metrics", "--db"])
        .arg(&db)
        .args(["record", "--replay-pending"])
        .output()
        .expect("launch replay");

    assert!(!out.status.success(), "replay must report the rejection");
    assert_eq!(obs_count(&db), 0, "no NULL-version row may be written");
    assert!(failed.join("bad.json").exists(), "must land in failed/");
}

// ── metrics validate ──────────────────────────────────────────────────────────

#[test]
fn validate_accepts_good_and_rejects_bad_without_writing() {
    let td = tempfile::tempdir().unwrap();
    let good = td.path().join("good.json");
    let bad = td.path().join("bad.json");
    std::fs::write(&good, record_json("rmlx", Some("0.2.8"))).unwrap();
    std::fs::write(&bad, record_json("rmlx", None)).unwrap();

    let ok = Command::new(rmlx_bin())
        .args(["metrics", "validate", "--file"])
        .arg(&good)
        .output()
        .expect("launch validate");
    assert!(ok.status.success());

    let err = Command::new(rmlx_bin())
        .args(["metrics", "validate", "--file"])
        .arg(&bad)
        .output()
        .expect("launch validate");
    assert!(!err.status.success());
    assert!(String::from_utf8_lossy(&err.stderr).contains("backend_version"));
}

// ── --metrics off: producer-side no-op, proven not just documented ───────────

/// `rmlx --metrics off <cmd>` must never open, or create, `runs.db` — the
/// claim made in three doc comments (`mode.rs`, `events.rs`, `metrics_drainer.rs`)
/// but, until this test, never actually checked.
///
/// Uses `serve` with a nonexistent model path rather than a real model load:
/// `EventRecorder::open` (the thing under test) runs in `main.rs` BEFORE any
/// command-specific model loading, so the process reaches and exercises the
/// `--metrics off` branch and then fails fast on the bad model path — no GPU,
/// no model snapshot, no server needed for this specific property.
#[test]
fn metrics_off_never_creates_runs_db() {
    let td = tempfile::tempdir().unwrap();
    let rmlx_home = td.path();

    let out = Command::new(rmlx_bin())
        .env("RMLX_HOME", rmlx_home)
        .args([
            "--metrics",
            "off",
            "serve",
            "--model",
            "/nonexistent/rmlx-metrics-off-probe",
            "--port",
            "0",
        ])
        .output()
        .expect("launch serve");

    // Must fail (bad model path) — this test is not asserting serve succeeds.
    assert!(
        !out.status.success(),
        "expected a fast failure on the bogus model path"
    );

    // `rmlx_core::paths::metrics_dir()` eagerly `create_dir_all`s the bare
    // `metrics/` directory as a path-resolution side effect regardless of
    // `--metrics` mode (pre-existing behavior, out of scope here) — so the
    // directory existing is not the claim under test. The claim is
    // specifically that `runs.db` is never opened or created.
    assert!(
        !rmlx_home.join("metrics/runs.db").exists(),
        "--metrics off must not create runs.db: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Sibling positive control: WITHOUT `--metrics off` (default `full`), the
/// same fast-failing invocation still reaches `EventRecorder::open` and DOES
/// create `runs.db` — proving the previous test's absence is caused by the
/// flag, not by the command failing before reaching that code at all.
#[test]
fn default_metrics_mode_does_create_runs_db_on_the_same_fast_fail_path() {
    let td = tempfile::tempdir().unwrap();
    let rmlx_home = td.path();

    let out = Command::new(rmlx_bin())
        .env("RMLX_HOME", rmlx_home)
        .args([
            "serve",
            "--model",
            "/nonexistent/rmlx-metrics-off-probe",
            "--port",
            "0",
        ])
        .output()
        .expect("launch serve");

    assert!(!out.status.success());
    assert!(
        rmlx_home.join("metrics/runs.db").exists(),
        "default --metrics full should have created runs.db on this same \
         fast-fail path (control for metrics_off_never_creates_runs_db): {}",
        String::from_utf8_lossy(&out.stderr)
    );
}
