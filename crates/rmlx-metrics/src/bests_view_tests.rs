use rusqlite::Connection;
use serde_json::json;

use crate::{
    ingest::{MetricEntry, PromptRef, RunRecord},
    recorder::Recorder,
};

// ── Helpers ───────────────────────────────────────────────────────────────

fn test_conn() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::migrate::run_pending(&mut conn).unwrap();
    conn
}

fn make_run(metric: &str, value: f64, kv_quant: &str) -> RunRecord {
    RunRecord {
        schema_version: crate::ingest::RECORD_SCHEMA_VERSION,
        backend: "rmlx".into(),
        backend_version: Some("0.1.0".into()),
        model_namespace: "mlx-community".into(),
        model: "Qwen3.6-35B-A3B-8bit".into(),
        weight_quant: "q8_0".into(),
        kv_quant: kv_quant.into(),
        ctx_max: 131_072,
        prompt: PromptRef::ByBody {
            name: "longctx_128k".into(),
            body: json!("the quick brown fox"),
            notes: None,
            tokens_approx: Some(4),
        },
        ts_utc: "2026-05-11T16:06:44Z".into(),
        git_sha: None,
        build_profile: Some("release".into()),
        hardware_tag: "m5_max_128gb".into(),
        prompt_tokens: Some(131_052),
        max_tokens: Some(32),
        temperature: Some(0.0),
        seed: Some(0),
        n_warmups: Some(1),
        n_measure: Some(3),
        output_first_64: None,
        notes: None,
        description: None,
        metrics: vec![MetricEntry {
            name: metric.into(),
            value: Some(value),
            stddev: None,
        }],
    }
}

/// Clone an existing observation with a new value / kv_quant, bypassing the
/// ingest validator — the historical rows this view has to survive were
/// written before any value gate existed, and cannot be re-created through it.
fn clone_row_with(conn: &Connection, metric: &str, value: f64, kv_quant: &str) {
    conn.execute(
        "INSERT INTO observations (
             backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
             metric, value, unit, direction, run_id, ts_utc, hardware_tag,
             inserted_utc, inserted_by
         )
         SELECT backend, model_namespace, model, weight_quant, ?1, ctx_max, prompt_id,
                metric, ?2, unit, direction, run_id, ts_utc, hardware_tag,
                inserted_utc, inserted_by
           FROM observations WHERE metric = ?3 LIMIT 1",
        rusqlite::params![kv_quant, value, metric],
    )
    .unwrap();
}

fn bests_values(conn: &Connection, metric: &str) -> Vec<(String, f64)> {
    let mut stmt = conn
        .prepare("SELECT kv_quant, value FROM bests WHERE metric = ?1 ORDER BY kv_quant")
        .unwrap();
    let rows = stmt
        .query_map(rusqlite::params![metric], |r| Ok((r.get(0)?, r.get(1)?)))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    rows
}

// ── Tests ─────────────────────────────────────────────────────────────────

/// A 998×-inflated `prefill_tps` row sharing a cell with a real measurement
/// must not become that cell's champion.
#[test]
fn implausible_prefill_row_does_not_win_its_cell() {
    let mut conn = test_conn();
    {
        let mut rec = Recorder::new(&mut conn, "test@0.1.0");
        rec.record_run(&make_run("prefill_tps", 494.7, "k8v4"))
            .unwrap();
    }
    // (prompt_tokens - 242) * 1000 for a 131_052-token prompt.
    clone_row_with(&conn, "prefill_tps", 130_810_000.0, "k8v4");

    assert_eq!(
        bests_values(&conn, "prefill_tps"),
        vec![("k8v4".to_string(), 494.7)],
        "the inflated row won the cell"
    );
}

/// A cell whose only `prefill_tps` row is `0.0` must leave `bests` entirely —
/// an upper bound alone would promote the zero into the champion table.
#[test]
fn a_cell_whose_only_rate_is_zero_leaves_bests() {
    let mut conn = test_conn();
    {
        let mut rec = Recorder::new(&mut conn, "test@0.1.0");
        rec.record_run(&make_run("prefill_tps", 494.7, "k8v4"))
            .unwrap();
    }
    clone_row_with(&conn, "prefill_tps", 0.0, "planar");

    let bests = bests_values(&conn, "prefill_tps");
    assert!(
        bests.iter().all(|(kv, _)| kv != "planar"),
        "a zero-rate-only cell published a champion: {bests:?}"
    );
    assert_eq!(bests.len(), 1, "the plausible cell must survive: {bests:?}");
}

/// The bound is per metric, not per number: `0` cache hits is a measurement
/// and must keep winning its cell. Guards the filter against over-reach.
#[test]
fn a_zero_counter_still_wins_its_cell() {
    let mut conn = test_conn();
    {
        let mut rec = Recorder::new(&mut conn, "test@0.1.0");
        rec.record_run(&make_run("prompt_cache_hits", 0.0, "k8v4"))
            .unwrap();
    }

    assert_eq!(
        bests_values(&conn, "prompt_cache_hits"),
        vec![("k8v4".to_string(), 0.0)],
        "a legitimate zero counter was filtered out of bests"
    );
}

/// A DB carrying an older `bests` definition is brought in line with the
/// registry on the next migration run, and left alone once it matches.
#[test]
fn ensure_rebuilds_a_stale_definition_exactly_once() {
    let conn = test_conn();

    conn.execute_batch(
        "DROP VIEW bests;
         CREATE VIEW bests AS SELECT * FROM observations",
    )
    .unwrap();

    assert!(
        super::ensure(&conn).unwrap(),
        "a stale view definition was left in place"
    );
    assert!(
        !super::ensure(&conn).unwrap(),
        "a current view definition was needlessly rebuilt"
    );

    let stored: String = conn
        .query_row(
            "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'bests'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(stored, super::create_sql());
}

/// Every registry metric contributes a branch, so no metric can be silently
/// unbounded in the view.
#[test]
fn generated_predicate_covers_every_registry_metric() {
    let sql = super::create_sql();
    for (name, _, _, _) in crate::registry::METRICS {
        assert!(
            sql.contains(&format!("WHEN '{name}' THEN")),
            "metric '{name}' has no branch in the bests predicate"
        );
    }
}
