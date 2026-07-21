use super::*;
use crate::migrate;

fn init_fresh() -> Connection {
    let mut conn = open_memory().unwrap();
    migrate::run_pending(&mut conn).unwrap();
    conn
}

// -----------------------------------------------------------------------
// Idempotency
// -----------------------------------------------------------------------

#[test]
fn init_schema_idempotent() {
    let mut conn = open_memory().unwrap();
    let applied_first = migrate::run_pending(&mut conn).unwrap();
    assert_eq!(
        applied_first as usize,
        MIGRATIONS.len(),
        "first run should apply every embedded migration"
    );

    let applied_second = migrate::run_pending(&mut conn).unwrap();
    assert_eq!(applied_second, 0, "second run must apply 0 migrations");
}

// -----------------------------------------------------------------------
// WAL PRAGMA (requires a file-backed DB — :memory: reports "memory")
// -----------------------------------------------------------------------

#[test]
fn pragma_wal_set() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("test.db");
    let conn = open(&path).unwrap();
    let mode: String = conn
        .query_row("PRAGMA journal_mode", [], |r| r.get(0))
        .unwrap();
    assert_eq!(mode, "wal");
}

// -----------------------------------------------------------------------
// bests VIEW — higher_better picks max
// -----------------------------------------------------------------------

#[test]
fn bests_view_picks_higher_better() {
    let conn = init_fresh();

    // Insert a sentinel prompt row to satisfy the FK.
    conn.execute(
        "INSERT INTO prompts(sha256, name, body, first_seen_utc)
         VALUES ('aaa', 'test', 'body', '2026-01-01T00:00:00Z')",
        [],
    )
    .unwrap();

    let prompt_id: i64 = conn
        .query_row("SELECT id FROM prompts WHERE sha256='aaa'", [], |r| {
            r.get(0)
        })
        .unwrap();

    // Three observations for the same cell, metric=decode_tps_warm, values 100/110/105.
    for (val, ts) in [
        (100.0, "2026-01-01T00:00:00Z"),
        (110.0, "2026-01-02T00:00:00Z"),
        (105.0, "2026-01-03T00:00:00Z"),
    ] {
        conn.execute(
            "INSERT INTO observations(
                 backend, model_namespace, model, weight_quant, kv_quant,
                 ctx_max, prompt_id, metric,
                 value, unit, direction,
                 run_id, ts_utc, hardware_tag, inserted_utc, inserted_by)
             VALUES ('rmlx','mlx-community','test-model','mxfp8','none',
                     8192, ?1, 'decode_tps_warm',
                     ?2, 'tps', 'higher_better',
                     'run0', ?3, 'm5_max_128gb',
                     '2026-01-01T00:00:00Z', 'test@0.0.1')",
            rusqlite::params![prompt_id, val, ts],
        )
        .unwrap();
    }

    let best_val: f64 = conn
        .query_row(
            "SELECT value FROM bests
              WHERE backend='rmlx' AND metric='decode_tps_warm'",
            [],
            |r| r.get(0),
        )
        .unwrap();

    assert!(
        (best_val - 110.0).abs() < f64::EPSILON,
        "expected 110, got {best_val}"
    );
}

// -----------------------------------------------------------------------
// bests VIEW — lower_better picks min
// -----------------------------------------------------------------------

#[test]
fn bests_view_picks_lower_better() {
    let conn = init_fresh();

    conn.execute(
        "INSERT INTO prompts(sha256, name, body, first_seen_utc)
         VALUES ('bbb', 'test2', 'body2', '2026-01-01T00:00:00Z')",
        [],
    )
    .unwrap();
    let prompt_id: i64 = conn
        .query_row("SELECT id FROM prompts WHERE sha256='bbb'", [], |r| {
            r.get(0)
        })
        .unwrap();

    // Values 10/8/9; lowest (8) should win.
    for (val, ts) in [
        (10.0_f64, "2026-01-01T00:00:00Z"),
        (8.0_f64, "2026-01-02T00:00:00Z"),
        (9.0_f64, "2026-01-03T00:00:00Z"),
    ] {
        conn.execute(
            "INSERT INTO observations(
                 backend, model_namespace, model, weight_quant, kv_quant,
                 ctx_max, prompt_id, metric,
                 value, unit, direction,
                 run_id, ts_utc, hardware_tag, inserted_utc, inserted_by)
             VALUES ('rmlx','mlx-community','test-model2','mxfp8','none',
                     8192, ?1, 'step_ms_mean',
                     ?2, 'ms', 'lower_better',
                     'run1', ?3, 'm5_max_128gb',
                     '2026-01-01T00:00:00Z', 'test@0.0.1')",
            rusqlite::params![prompt_id, val, ts],
        )
        .unwrap();
    }

    let best_val: f64 = conn
        .query_row(
            "SELECT value FROM bests
              WHERE backend='rmlx' AND metric='step_ms_mean'",
            [],
            |r| r.get(0),
        )
        .unwrap();

    assert!(
        (best_val - 8.0).abs() < f64::EPSILON,
        "expected 8.0, got {best_val}"
    );
}

// -----------------------------------------------------------------------
// bests VIEW — tie-break: newer ts_utc wins
// -----------------------------------------------------------------------

#[test]
fn bests_view_tiebreak_newer_wins() {
    let conn = init_fresh();

    conn.execute(
        "INSERT INTO prompts(sha256, name, body, first_seen_utc)
         VALUES ('ccc', 'test3', 'body3', '2026-01-01T00:00:00Z')",
        [],
    )
    .unwrap();
    let prompt_id: i64 = conn
        .query_row("SELECT id FROM prompts WHERE sha256='ccc'", [], |r| {
            r.get(0)
        })
        .unwrap();

    // Two observations with the same value; the newer one should be the champion.
    for ts in ["2026-01-01T00:00:00Z", "2026-01-05T00:00:00Z"] {
        conn.execute(
            "INSERT INTO observations(
                 backend, model_namespace, model, weight_quant, kv_quant,
                 ctx_max, prompt_id, metric,
                 value, unit, direction,
                 run_id, ts_utc, hardware_tag, inserted_utc, inserted_by)
             VALUES ('rmlx','mlx-community','test-model3','mxfp8','none',
                     8192, ?1, 'decode_tps_warm',
                     100.0, 'tps', 'higher_better',
                     'run2', ?2, 'm5_max_128gb',
                     '2026-01-01T00:00:00Z', 'test@0.0.1')",
            rusqlite::params![prompt_id, ts],
        )
        .unwrap();
    }

    let winner_ts: String = conn
        .query_row(
            "SELECT ts_utc FROM bests
              WHERE backend='rmlx' AND model='test-model3' AND metric='decode_tps_warm'",
            [],
            |r| r.get(0),
        )
        .unwrap();

    assert_eq!(
        winner_ts, "2026-01-05T00:00:00Z",
        "newer ts_utc must win the tie"
    );
}

// -----------------------------------------------------------------------
// prompts — UNIQUE sha256 constraint
// -----------------------------------------------------------------------

#[test]
fn prompts_unique_sha256() {
    let conn = init_fresh();

    conn.execute(
        "INSERT INTO prompts(sha256, name, body, first_seen_utc)
         VALUES ('deadbeef', 'p1', 'body', '2026-01-01T00:00:00Z')",
        [],
    )
    .unwrap();

    let result = conn.execute(
        "INSERT INTO prompts(sha256, name, body, first_seen_utc)
         VALUES ('deadbeef', 'p2', 'different body', '2026-01-02T00:00:00Z')",
        [],
    );

    assert!(result.is_err(), "second insert with same sha256 must fail");
}

// -----------------------------------------------------------------------
// observations — two rows with identical cell columns both survive
// -----------------------------------------------------------------------

#[test]
fn observations_no_pk_collision() {
    let conn = init_fresh();

    conn.execute(
        "INSERT INTO prompts(sha256, name, body, first_seen_utc)
         VALUES ('eee', 'p3', 'body3', '2026-01-01T00:00:00Z')",
        [],
    )
    .unwrap();
    let prompt_id: i64 = conn
        .query_row("SELECT id FROM prompts WHERE sha256='eee'", [], |r| {
            r.get(0)
        })
        .unwrap();

    for ts in ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z"] {
        conn.execute(
            "INSERT INTO observations(
                 backend, model_namespace, model, weight_quant, kv_quant,
                 ctx_max, prompt_id, metric,
                 value, unit, direction,
                 run_id, ts_utc, hardware_tag, inserted_utc, inserted_by)
             VALUES ('rmlx','mlx-community','dupe-model','mxfp8','none',
                     8192, ?1, 'decode_tps_warm',
                     99.0, 'tps', 'higher_better',
                     'run3', ?2, 'm5_max_128gb',
                     '2026-01-01T00:00:00Z', 'test@0.0.1')",
            rusqlite::params![prompt_id, ts],
        )
        .unwrap();
    }

    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM observations WHERE model='dupe-model'",
            [],
            |r| r.get(0),
        )
        .unwrap();

    assert_eq!(
        count, 2,
        "both identical-cell rows must survive (no PK collision)"
    );
}

// -----------------------------------------------------------------------
// events — migration 004 adds mlx_nax
// -----------------------------------------------------------------------

#[test]
fn events_table_has_mlx_nax_column() {
    let conn = init_fresh();

    let has_mlx_nax: bool = conn
        .prepare("SELECT 1 FROM pragma_table_info('events') WHERE name = 'mlx_nax'")
        .unwrap()
        .exists([])
        .unwrap();
    assert!(
        has_mlx_nax,
        "a fresh DB must carry migration 004's mlx_nax column on events"
    );
}

// -----------------------------------------------------------------------
// schema_meta — seed rows present after init
// -----------------------------------------------------------------------

#[test]
fn schema_meta_seeded() {
    let conn = init_fresh();

    let version: String = conn
        .query_row(
            "SELECT value FROM schema_meta WHERE key='schema_version'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(version, "1");

    let created_utc: String = conn
        .query_row(
            "SELECT value FROM schema_meta WHERE key='created_utc'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    // Must be parseable as ISO-8601 (contains 'T' separator and ends with 'Z').
    assert!(
        created_utc.contains('T') && created_utc.ends_with('Z'),
        "created_utc must be ISO-8601 UTC, got: {created_utc}"
    );
}
