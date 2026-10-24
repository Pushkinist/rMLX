//! The run-class contract: only `gate` rows reach `bests` and the gate
//! queries, a record must state its class, and the migration files every row
//! that predates the column as `gate`.

use rusqlite::Connection;
use serde_json::json;

use crate::ingest::RunRecord;
use crate::query::{self, Cell};
use crate::recorder::Recorder;
use crate::schema::MIGRATIONS;

const MODEL: &str = "gemma-4-e4b-it-mxfp8";

fn test_conn() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::migrate::run_pending(&mut conn).unwrap();
    conn
}

/// A §8.5 record for one `decode_tps_warm` value, with `run_class` set to
/// `class` or left out when `None`.
fn record_json(
    class: Option<&str>,
    value: f64,
    ts: &str,
    git_sha: Option<&str>,
) -> serde_json::Value {
    let mut rec = json!({
        "backend": "rmlx",
        "backend_version": "0.4.5",
        "model_namespace": "mlx-community",
        "model": MODEL,
        "weight_quant": "mxfp8",
        "kv_quant": "none",
        "ctx_max": 8192,
        "prompt": { "name": "longctx_4k", "body": "the quick brown fox" },
        "ts_utc": ts,
        "git_sha": git_sha,
        "hardware_tag": "m5_max_128gb",
        "metrics": [ { "name": "decode_tps_warm", "value": value } ],
    });
    if let Some(class) = class {
        rec["run_class"] = json!(class);
    }
    rec
}

fn record(conn: &mut Connection, class: &str, value: f64, ts: &str, git_sha: Option<&str>) {
    let run: RunRecord =
        serde_json::from_value(record_json(Some(class), value, ts, git_sha)).unwrap();
    Recorder::new(conn, "test@0.0.1").record_run(&run).unwrap();
}

fn cell(conn: &Connection) -> Cell {
    let prompt_id: i64 = conn
        .query_row("SELECT id FROM prompts LIMIT 1", [], |r| r.get(0))
        .unwrap();
    Cell {
        backend: "rmlx".into(),
        model_namespace: "mlx-community".into(),
        model: MODEL.into(),
        weight_quant: "mxfp8".into(),
        kv_quant: "none".into(),
        ctx_max: 8192,
        prompt_id,
        decode_config: None,
    }
}

fn champion(conn: &Connection) -> Option<f64> {
    query::best(conn, &cell(conn), "decode_tps_warm")
        .unwrap()
        .map(|b| b.value)
}

// ── bests and the gate queries ──────────────────────────────────────────────

#[test]
fn an_ab_arm_faster_than_every_gate_row_is_not_the_champion() {
    let mut conn = test_conn();
    record(&mut conn, "gate", 100.0, "2026-10-01T00:00:00Z", None);
    record(&mut conn, "gate", 105.0, "2026-10-02T00:00:00Z", None);
    record(&mut conn, "ab_arm", 500.0, "2026-10-03T00:00:00Z", None);
    record(&mut conn, "diagnostic", 400.0, "2026-10-04T00:00:00Z", None);

    assert_eq!(champion(&conn), Some(105.0));
    let ranked = query::rank(&conn, "decode_tps_warm", None, 10).unwrap();
    assert_eq!(
        ranked.iter().map(|b| b.value).collect::<Vec<_>>(),
        vec![105.0]
    );
}

#[test]
fn an_ab_arm_row_does_not_move_deltas() {
    let mut conn = test_conn();
    record(
        &mut conn,
        "gate",
        100.0,
        "2026-10-01T00:00:00Z",
        Some("aaa1111"),
    );
    record(
        &mut conn,
        "gate",
        101.0,
        "2026-10-03T00:00:00Z",
        Some("bbb2222"),
    );
    // Half the gate rate, after the SHA: a regression if a gate read it.
    record(
        &mut conn,
        "ab_arm",
        50.0,
        "2026-10-04T00:00:00Z",
        Some("ccc3333"),
    );
    // Above every gate row, before the SHA: a baseline no gate row reaches.
    record(&mut conn, "ab_arm", 900.0, "2026-09-30T00:00:00Z", None);

    let rows = query::deltas(&conn, "aaa1111", Some(5.0), None).unwrap();
    assert!(
        rows.iter().all(|r| !r.regressed),
        "a non-gate row moved the gate: {rows:?}"
    );
}

#[test]
fn a_sha_that_only_an_ab_arm_row_carries_is_not_a_gate_baseline() {
    let mut conn = test_conn();
    record(
        &mut conn,
        "gate",
        100.0,
        "2026-10-02T00:00:00Z",
        Some("aaa1111"),
    );
    record(
        &mut conn,
        "ab_arm",
        90.0,
        "2026-10-01T00:00:00Z",
        Some("abab999"),
    );

    let err = query::deltas(&conn, "abab999", Some(5.0), None).unwrap_err();
    assert!(
        err.to_string().contains("abab999"),
        "unexpected error: {err}"
    );
}

#[test]
fn regress_compares_the_latest_gate_row() {
    let mut conn = test_conn();
    record(&mut conn, "gate", 100.0, "2026-10-01T00:00:00Z", None);
    record(&mut conn, "ab_arm", 10.0, "2026-10-02T00:00:00Z", None);

    let result = query::regress(&conn, "gemma-4-e4b", "decode_tps_warm", None, 1.0).unwrap();
    assert_eq!(result.latest_value, Some(100.0));
    assert!(!result.regressed, "{}", result.message);
}

#[test]
fn champions_list_no_cell_that_only_non_gate_rows_measured() {
    let mut conn = test_conn();
    record(&mut conn, "ab_arm", 100.0, "2026-10-01T00:00:00Z", None);
    record(&mut conn, "diagnostic", 90.0, "2026-10-02T00:00:00Z", None);

    assert!(query::champions(&conn, None).unwrap().is_empty());
    assert!(query::champions(&conn, Some("rmlx")).unwrap().is_empty());
}

// ── A record states its class ───────────────────────────────────────────────

#[test]
fn a_record_without_a_class_is_refused() {
    let run: RunRecord =
        serde_json::from_value(record_json(None, 100.0, "2026-10-01T00:00:00Z", None)).unwrap();
    let err = run.validate().unwrap_err();
    assert!(
        err.to_string().contains("run_class"),
        "unexpected error: {err}"
    );

    let mut conn = test_conn();
    assert!(Recorder::new(&mut conn, "test@0.0.1")
        .record_run(&run)
        .is_err());
    let rows: i64 = conn
        .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(rows, 0);
}

#[test]
fn a_misspelled_class_is_refused() {
    let parsed = serde_json::from_value::<RunRecord>(record_json(
        Some("gat"),
        100.0,
        "2026-10-01T00:00:00Z",
        None,
    ));
    let refused = match parsed {
        Err(_) => true,
        Ok(run) => run.validate().is_err(),
    };
    assert!(refused, "`gat` was accepted as a run class");
}

/// A row written around the recorder carries no class, and no class is not
/// `gate`.
#[test]
fn a_row_with_no_class_is_not_a_champion() {
    let mut conn = test_conn();
    record(&mut conn, "gate", 100.0, "2026-10-01T00:00:00Z", None);
    conn.execute(
        "INSERT INTO observations (
             backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
             metric, value, unit, direction, run_id, ts_utc, hardware_tag,
             inserted_utc, inserted_by
         )
         SELECT backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
                metric, 999.0, unit, direction, run_id, ts_utc, hardware_tag,
                inserted_utc, inserted_by
           FROM observations LIMIT 1",
        [],
    )
    .unwrap();

    assert_eq!(champion(&conn), Some(100.0));
}

#[test]
fn the_column_holds_the_three_classes_and_nothing_else() {
    let mut conn = test_conn();
    record(&mut conn, "gate", 100.0, "2026-10-01T00:00:00Z", None);

    for class in ["gate", "ab_arm", "diagnostic"] {
        conn.execute("UPDATE observations SET run_class = ?1", [class])
            .unwrap();
    }
    assert!(conn
        .execute("UPDATE observations SET run_class = 'gat'", [])
        .is_err());
}

// ── Migration ───────────────────────────────────────────────────────────────

#[test]
fn migrating_files_every_existing_row_as_gate() {
    let mut conn = Connection::open_in_memory().unwrap();
    for &(version, sql) in MIGRATIONS {
        if version > 8 {
            break;
        }
        conn.execute_batch(sql).unwrap();
        conn.execute_batch(&format!("PRAGMA user_version = {version};"))
            .unwrap();
    }
    conn.execute(
        "INSERT INTO prompts (id, sha256, name, body, tokens_approx, first_seen_utc)
         VALUES (1, 'sha', 'p', 'hi', 4, '2026-05-10T07:30:00Z')",
        [],
    )
    .unwrap();
    for (id, value) in [(1, 90.0), (2, 110.0)] {
        conn.execute(
            "INSERT INTO observations (
                 id, backend, model_namespace, model, weight_quant, kv_quant, ctx_max,
                 prompt_id, metric, value, unit, direction, run_id, ts_utc, hardware_tag,
                 inserted_utc, inserted_by
             ) VALUES (
                 ?1, 'rmlx', 'mlx-community', 'm', 'mxfp8', 'none', 8192, 1,
                 'decode_tps_warm', ?2, 'tps', 'higher_better', 'run', '2026-05-10T07:30:00Z',
                 'm5_max_128gb', '2026-05-10T07:30:00Z', 'test'
             )",
            rusqlite::params![id, value],
        )
        .unwrap();
    }

    crate::migrate::run_pending(&mut conn).unwrap();

    let classes: Vec<String> = conn
        .prepare("SELECT run_class FROM observations ORDER BY id")
        .unwrap()
        .query_map([], |r| r.get(0))
        .unwrap()
        .map(Result::unwrap)
        .collect();
    assert_eq!(classes, vec!["gate", "gate"]);
    let champion: f64 = conn
        .query_row("SELECT value FROM bests", [], |r| r.get(0))
        .unwrap();
    assert_eq!(champion, 110.0);
}
