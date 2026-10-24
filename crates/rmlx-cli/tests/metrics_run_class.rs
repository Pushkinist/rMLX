//! The run class on the command line: `metrics record` refuses a record that
//! states no class, `baseline --record` and `eval ppl --corpus` refuse to run
//! without one, the history queries show it, and `metrics import` brings the
//! rows of another DB in under a stated class, once.
//!
//! No model is loaded: every refusal here happens before a load.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::path::{Path, PathBuf};
use std::process::Command;

use rusqlite::Connection;

const MODEL: &str = "gemma-4-e4b-it-mxfp8";
const PROMPT_BODY: &str = "Hello world";

fn rmlx_bin() -> PathBuf {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo");
    let debug = PathBuf::from(manifest_dir)
        .parent()
        .and_then(Path::parent)
        .expect("workspace root from CARGO_MANIFEST_DIR")
        .join("target/debug/rmlx");
    assert!(debug.exists(), "{} not built", debug.display());
    debug
}

/// `rmlx` with a scratch `RMLX_HOME`, so no command here can reach a real DB.
fn rmlx(home: &Path) -> Command {
    let mut cmd = Command::new(rmlx_bin());
    cmd.env("RMLX_HOME", home).env_remove("RMLX_METRICS_DB");
    cmd
}

fn metrics(home: &Path, db: &Path, args: &[&str]) -> std::process::Output {
    rmlx(home)
        .arg("metrics")
        .arg("--db")
        .arg(db)
        .args(args)
        .output()
        .expect("launch rmlx metrics")
}

fn init_db(home: &Path, name: &str) -> PathBuf {
    let db = home.join(name);
    let out = metrics(home, &db, &["init"]);
    assert!(out.status.success(), "init: {}", stderr(&out));
    db
}

fn stderr(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn stdout(out: &std::process::Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn record_json(class: Option<&str>, value: f64, ts: &str) -> String {
    let mut rec = serde_json::json!({
        "backend": "rmlx",
        "backend_version": "0.4.5",
        "model_namespace": "mlx-community",
        "model": MODEL,
        "weight_quant": "mxfp8",
        "kv_quant": "none",
        "ctx_max": 8192,
        "prompt": { "name": "test_prompt", "body": PROMPT_BODY },
        "ts_utc": ts,
        "hardware_tag": "m5_max_128gb",
        "metrics": [ { "name": "decode_tps_warm", "value": value } ],
    });
    if let Some(class) = class {
        rec["run_class"] = serde_json::json!(class);
    }
    rec.to_string()
}

fn count(db: &Path, sql: &str) -> i64 {
    Connection::open(db)
        .unwrap()
        .query_row(sql, [], |r| r.get(0))
        .unwrap()
}

// ── metrics record ──────────────────────────────────────────────────────────

#[test]
fn record_refuses_a_record_that_states_no_class() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");

    let out = metrics(
        td.path(),
        &db,
        &[
            "record",
            "--inline",
            &record_json(None, 100.0, "2026-10-01T00:00:00Z"),
        ],
    );
    assert!(!out.status.success(), "a classless record was accepted");
    assert!(stderr(&out).contains("run_class"), "{}", stderr(&out));
    assert_eq!(count(&db, "SELECT COUNT(*) FROM observations"), 0);
}

#[test]
fn record_takes_the_class_from_the_flag() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");

    let out = metrics(
        td.path(),
        &db,
        &[
            "record",
            "--run-class",
            "ab_arm",
            "--inline",
            &record_json(None, 100.0, "2026-10-01T00:00:00Z"),
        ],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM observations WHERE run_class = 'ab_arm'"
        ),
        1
    );
}

#[test]
fn record_refuses_a_flag_that_contradicts_the_record() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");

    let out = metrics(
        td.path(),
        &db,
        &[
            "record",
            "--run-class",
            "gate",
            "--inline",
            &record_json(Some("ab_arm"), 100.0, "2026-10-01T00:00:00Z"),
        ],
    );
    assert!(!out.status.success(), "a contradicting class was accepted");
    assert!(stderr(&out).contains("ab_arm"), "{}", stderr(&out));
    assert_eq!(count(&db, "SELECT COUNT(*) FROM observations"), 0);
}

// ── producers that build the record in-process ──────────────────────────────

#[test]
fn baseline_record_refuses_to_run_without_a_class() {
    let td = tempfile::tempdir().unwrap();
    let out = rmlx(td.path())
        .args(["baseline", "--model"])
        .arg(td.path().join("no-such-snapshot"))
        .args(["--prompt", "hi", "--device", "cpu", "--record"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(stderr(&out).contains("--run-class"), "{}", stderr(&out));
}

/// `--record` defaults to off, and the default is not a request to record.
#[test]
fn baseline_without_record_needs_no_class() {
    let td = tempfile::tempdir().unwrap();
    let out = rmlx(td.path())
        .args(["baseline", "--model"])
        .arg(td.path().join("no-such-snapshot"))
        .args(["--prompt", "hi", "--device", "cpu"])
        .output()
        .unwrap();
    assert!(!stderr(&out).contains("--run-class"), "{}", stderr(&out));
}

#[test]
fn eval_ppl_with_a_corpus_refuses_to_run_without_a_class() {
    let td = tempfile::tempdir().unwrap();
    let text = td.path().join("corpus.txt");
    std::fs::write(&text, "the quick brown fox").unwrap();
    let out = rmlx(td.path())
        .args(["eval", "ppl", "--model"])
        .arg(td.path().join("no-such-snapshot"))
        .arg("--text-file")
        .arg(&text)
        .args(["--corpus", "wikitext-2", "--device", "cpu"])
        .output()
        .unwrap();
    assert!(!out.status.success());
    assert!(stderr(&out).contains("--run-class"), "{}", stderr(&out));
}

// ── history queries show the class ──────────────────────────────────────────

#[test]
fn history_and_timeseries_show_every_class() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");
    for (class, value) in [("gate", 100.0), ("ab_arm", 120.0)] {
        let out = metrics(
            td.path(),
            &db,
            &[
                "record",
                "--inline",
                &record_json(Some(class), value, "2026-10-01T00:00:00Z"),
            ],
        );
        assert!(out.status.success(), "{}", stderr(&out));
    }
    let cell = [
        "--backend",
        "rmlx",
        "--namespace",
        "mlx-community",
        "--model",
        MODEL,
        "--weight-quant",
        "mxfp8",
        "--kv-quant",
        "none",
        "--prompt-name",
        "test_prompt",
    ];

    let mut args = vec!["history"];
    args.extend(cell);
    let history = stdout(&metrics(td.path(), &db, &args));
    assert!(history.contains(r#""run_class":"gate""#), "{history}");
    assert!(history.contains(r#""run_class":"ab_arm""#), "{history}");

    let mut args = vec!["timeseries"];
    args.extend(cell);
    args.extend(["--metric", "decode_tps_warm"]);
    let series = stdout(&metrics(td.path(), &db, &args));
    assert_eq!(series.lines().count(), 2, "one point per class: {series}");
    assert!(series.contains(r#""run_class":"ab_arm""#), "{series}");
}

// ── metrics import ──────────────────────────────────────────────────────────

/// A DB at schema version 8, the last one without `run_class`, holding two
/// runs under one prompt.
fn source_at_version(home: &Path, version: u32) -> PathBuf {
    let path = home.join(format!("source-v{version}.db"));
    let conn = Connection::open(&path).unwrap();
    for &(v, sql) in rmlx_metrics::schema::MIGRATIONS {
        if v > version {
            break;
        }
        conn.execute_batch(sql).unwrap();
        conn.execute_batch(&format!("PRAGMA user_version = {v};"))
            .unwrap();
    }
    let body = serde_json::json!(PROMPT_BODY);
    conn.execute(
        "INSERT INTO prompts (id, sha256, name, body, tokens_approx, first_seen_utc)
         VALUES (7, ?1, 'test_prompt', ?2, 2, '2026-09-01T00:00:00Z')",
        rusqlite::params![
            rmlx_metrics::ingest::prompt_body_sha256(&body),
            body.to_string()
        ],
    )
    .unwrap();
    for (run_id, value) in [
        ("20260901000000-aaaaaa", 90.0),
        ("20260902000000-bbbbbb", 95.0),
    ] {
        conn.execute(
            "INSERT INTO observations (
                 backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
                 metric, value, unit, direction, run_id, ts_utc, backend_version,
                 hardware_tag, inserted_utc, inserted_by
             ) VALUES (
                 'rmlx', 'mlx-community', ?1, 'mxfp8', 'none', 8192, 7,
                 'decode_tps_warm', ?2, 'tps', 'higher_better', ?3, '2026-09-01T00:00:00Z',
                 '0.4.4', 'm5_max_128gb', '2026-09-01T00:00:01Z', 'rmlx-cli@0.4.4'
             )",
            rusqlite::params![MODEL, value, run_id],
        )
        .unwrap();
    }
    path
}

#[test]
fn import_stamps_the_stated_class_and_adds_nothing_on_a_second_run() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");
    let out = metrics(
        td.path(),
        &db,
        &[
            "record",
            "--inline",
            &record_json(Some("gate"), 100.0, "2026-10-01T00:00:00Z"),
        ],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let source = source_at_version(td.path(), 8);
    let from = source.to_str().unwrap();

    for _ in 0..2 {
        let out = metrics(
            td.path(),
            &db,
            &["import", "--from", from, "--run-class", "ab_arm"],
        );
        assert!(out.status.success(), "{}", stderr(&out));
    }

    assert_eq!(count(&db, "SELECT COUNT(*) FROM observations"), 3);
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM observations WHERE run_class = 'ab_arm'"
        ),
        2
    );
    assert_eq!(
        count(&db, "SELECT COUNT(*) FROM prompts"),
        1,
        "prompts are content-addressed"
    );
    assert_eq!(
        count(
            &db,
            "SELECT CAST(value AS INTEGER) FROM bests WHERE metric = 'decode_tps_warm'"
        ),
        100,
        "an imported ab_arm row reached bests"
    );
}

#[test]
fn import_from_a_db_without_the_column_needs_a_stated_class() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");
    let source = source_at_version(td.path(), 8);

    let out = metrics(
        td.path(),
        &db,
        &["import", "--from", source.to_str().unwrap()],
    );
    assert!(!out.status.success(), "imported with no class");
    assert!(stderr(&out).contains("--run-class"), "{}", stderr(&out));
    assert_eq!(count(&db, "SELECT COUNT(*) FROM observations"), 0);
}

#[test]
fn import_keeps_the_class_a_source_row_states() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");
    let source = init_db(td.path(), "source.db");
    for (class, ts) in [
        ("gate", "2026-10-01T00:00:00Z"),
        ("diagnostic", "2026-10-02T00:00:00Z"),
    ] {
        let out = metrics(
            td.path(),
            &source,
            &["record", "--inline", &record_json(Some(class), 100.0, ts)],
        );
        assert!(out.status.success(), "{}", stderr(&out));
    }
    let from = source.to_str().unwrap();

    let out = metrics(
        td.path(),
        &db,
        &["import", "--from", from, "--run-class", "ab_arm"],
    );
    assert!(
        !out.status.success(),
        "a stated class overwrote the source's own"
    );
    assert_eq!(count(&db, "SELECT COUNT(*) FROM observations"), 0);

    let out = metrics(td.path(), &db, &["import", "--from", from]);
    assert!(out.status.success(), "{}", stderr(&out));
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM observations WHERE run_class = 'diagnostic'"
        ),
        1
    );
    assert_eq!(
        count(
            &db,
            "SELECT COUNT(*) FROM observations WHERE run_class = 'gate'"
        ),
        1
    );
}

#[test]
fn import_refuses_a_source_at_an_older_schema() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");
    let source = source_at_version(td.path(), 7);

    let out = metrics(
        td.path(),
        &db,
        &[
            "import",
            "--from",
            source.to_str().unwrap(),
            "--run-class",
            "ab_arm",
        ],
    );
    assert!(!out.status.success(), "imported from schema version 7");
    assert!(stderr(&out).contains("version 7"), "{}", stderr(&out));
    assert_eq!(count(&db, "SELECT COUNT(*) FROM observations"), 0);
}

#[test]
fn import_leaves_the_source_untouched() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");
    let source = source_at_version(td.path(), 8);
    let before = std::fs::read(&source).unwrap();

    let out = metrics(
        td.path(),
        &db,
        &[
            "import",
            "--from",
            source.to_str().unwrap(),
            "--run-class",
            "diagnostic",
        ],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let version: u32 = Connection::open(&source)
        .unwrap()
        .query_row("PRAGMA user_version", [], |r| r.get(0))
        .unwrap();
    assert_eq!(version, 8, "the import migrated its source");
    assert_eq!(std::fs::read(&source).unwrap(), before);
}

// ── a DB a newer binary migrated ────────────────────────────────────────────

#[test]
fn a_db_newer_than_this_binary_is_refused_by_readers_writers_and_doctor() {
    let td = tempfile::tempdir().unwrap();
    let db = init_db(td.path(), "runs.db");
    let last = rmlx_metrics::schema::MIGRATIONS
        .last()
        .map_or(0, |&(v, _)| v);
    Connection::open(&db)
        .unwrap()
        .execute_batch(&format!("PRAGMA user_version = {};", last + 1))
        .unwrap();

    for args in [
        vec!["rank", "--metric", "decode_tps_warm"],
        vec!["doctor", "--fix"],
        vec![
            "record",
            "--inline",
            &record_json(Some("gate"), 100.0, "2026-10-01T00:00:00Z"),
        ],
    ] {
        let out = metrics(td.path(), &db, &args);
        assert!(!out.status.success(), "{args:?} opened a newer DB");
        let err = stderr(&out);
        assert!(
            err.contains(&format!("version {}", last + 1)),
            "{args:?}: {err}"
        );
    }
}
