use super::*;
use crate::{
    error::{Error, Result},
    ingest::{MetricEntry, PromptRef, RunRecord},
    recorder::Recorder,
};
use rusqlite::Connection;
use serde_json::json;

// ── Seed helpers ──────────────────────────────────────────────────────────

fn test_conn() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::migrate::run_pending(&mut conn).unwrap();
    conn
}

/// Minimal RunRecord builder.
fn make_run(
    backend: &str,
    model: &str,
    metric: &str,
    value: f64,
    ts: &str,
    git_sha: Option<&str>,
) -> RunRecord {
    RunRecord {
        schema_version: crate::ingest::RECORD_SCHEMA_VERSION,
        backend: backend.into(),
        backend_version: Some("0.0.1".into()),
        model_namespace: "mlx-community".into(),
        model: model.into(),
        weight_quant: "mxfp8".into(),
        kv_quant: "k8v8".into(),
        ctx_max: 8192,
        prompt: PromptRef::ByBody {
            name: "test_prompt".into(),
            body: json!("the quick brown fox"),
            notes: None,
            tokens_approx: Some(4),
        },
        ts_utc: ts.into(),
        git_sha: git_sha.map(str::to_owned),
        build_profile: Some("release".into()),
        hardware_tag: "m5_max_128gb".into(),
        prompt_tokens: Some(4),
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

fn make_run_lower(
    backend: &str,
    model: &str,
    metric: &str,
    value: f64,
    ts: &str,
    git_sha: Option<&str>,
) -> RunRecord {
    let mut r = make_run(backend, model, metric, value, ts, git_sha);
    // peak_rss_mb is lower_better
    r.metrics[0].name = metric.into();
    r
}

/// Insert a seeded set of observations covering 2 backends × 2 models × 2 metrics × multiple timestamps.
fn seed_observations(conn: &mut Connection) -> Result<i64> {
    let mut rec = Recorder::new(conn, "test@0.0.1");

    // rmlx / gemma-4-e2b / decode_tps_warm — 3 obs (95, 100, 90) → best = 100
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e2b",
        "decode_tps_warm",
        95.0,
        "2026-05-01T10:00:00Z",
        Some("sha001"),
    ))?;
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e2b",
        "decode_tps_warm",
        100.0,
        "2026-05-05T10:00:00Z",
        Some("sha002"),
    ))?;
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e2b",
        "decode_tps_warm",
        90.0,
        "2026-05-08T10:00:00Z",
        Some("sha003"),
    ))?;

    // rmlx / gemma-4-e2b / peak_rss_mb (lower_better) — 2 obs (2000, 1800) → best = 1800
    rec.record_run(&make_run_lower(
        "rmlx",
        "gemma-4-e2b",
        "peak_rss_mb",
        2000.0,
        "2026-05-01T10:00:00Z",
        Some("sha001"),
    ))?;
    rec.record_run(&make_run_lower(
        "rmlx",
        "gemma-4-e2b",
        "peak_rss_mb",
        1800.0,
        "2026-05-05T10:00:00Z",
        Some("sha002"),
    ))?;

    // mlx_lm / gemma-4-e2b / decode_tps_warm — 1 obs (88)
    rec.record_run(&make_run(
        "mlx_lm",
        "gemma-4-e2b",
        "decode_tps_warm",
        88.0,
        "2026-05-03T10:00:00Z",
        Some("sha001"),
    ))?;

    // rmlx / qwen3 / decode_tps_warm — 1 obs (75)
    rec.record_run(&make_run(
        "rmlx",
        "qwen3-8b",
        "decode_tps_warm",
        75.0,
        "2026-05-04T10:00:00Z",
        Some("sha001"),
    ))?;

    // mlx_lm / qwen3 / decode_tps_warm — 1 obs (70)
    rec.record_run(&make_run(
        "mlx_lm",
        "qwen3-8b",
        "decode_tps_warm",
        70.0,
        "2026-05-04T10:00:00Z",
        Some("sha001"),
    ))?;

    // return prompt_id=1 (all share the same prompt body)
    Ok(1)
}

fn default_cell(backend: &str, model: &str) -> Cell {
    Cell {
        backend: backend.into(),
        model_namespace: "mlx-community".into(),
        model: model.into(),
        weight_quant: "mxfp8".into(),
        kv_quant: "k8v8".into(),
        ctx_max: 8192,
        prompt_id: 1,
    }
}

// ── best ──────────────────────────────────────────────────────────────────

#[test]
fn best_returns_champion() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let cell = default_cell("rmlx", "gemma-4-e2b");
    let row = best(&conn, &cell, "decode_tps_warm").unwrap().unwrap();
    // 3 obs: 95, 100, 90 → champion = 100
    assert_eq!(row.value, 100.0);
}

#[test]
fn best_returns_none_when_empty() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let cell = default_cell("rmlx", "no-such-model");
    let row = best(&conn, &cell, "decode_tps_warm").unwrap();
    assert!(row.is_none());
}

// ── rank ──────────────────────────────────────────────────────────────────

#[test]
fn rank_orders_by_value_desc_for_higher_better() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let rows = rank(&conn, "decode_tps_warm", None, 10).unwrap();
    // Expected order: rmlx/gemma-4-e2b=100, rmlx/qwen3-8b=75, mlx_lm/gemma-4-e2b=88, mlx_lm/qwen3-8b=70
    // Sorted desc: 100, 88, 75, 70
    assert_eq!(rows.len(), 4);
    assert_eq!(rows[0].value, 100.0);
    assert_eq!(rows[1].value, 88.0);
    assert_eq!(rows[2].value, 75.0);
    assert_eq!(rows[3].value, 70.0);
}

#[test]
fn rank_orders_by_value_asc_for_lower_better() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let rows = rank(&conn, "peak_rss_mb", None, 10).unwrap();
    // Only one cell: rmlx/gemma-4-e2b, best = 1800
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].value, 1800.0);
    assert_eq!(rows[0].direction, "lower_better");
}

#[test]
fn rank_filters_by_backend() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let rows = rank(&conn, "decode_tps_warm", Some("mlx_lm"), 10).unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows.iter().all(|r| r.cell.backend == "mlx_lm"));
}

#[test]
fn rank_respects_limit() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let rows = rank(&conn, "decode_tps_warm", None, 2).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].value, 100.0);
}

// ── compare ───────────────────────────────────────────────────────────────

#[test]
fn compare_pairs_two_backends_per_cell() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let rows = compare(&conn, &["rmlx", "mlx_lm"], "decode_tps_warm").unwrap();
    // Two models → 2 CompareRows
    assert_eq!(rows.len(), 2);
    for row in &rows {
        assert_eq!(row.per_backend.len(), 2);
        // Both backends present for both models
        assert!(row.per_backend[0].1.is_some());
        assert!(row.per_backend[1].1.is_some());
    }
}

#[test]
fn compare_marks_missing_backend_as_none() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    // "paroquant" has no observations at all.
    let rows = compare(&conn, &["rmlx", "paroquant"], "decode_tps_warm").unwrap();
    for row in &rows {
        let rmlx_entry = row.per_backend.iter().find(|(b, _)| b == "rmlx");
        let paro_entry = row.per_backend.iter().find(|(b, _)| b == "paroquant");
        assert!(rmlx_entry.unwrap().1.is_some());
        assert!(paro_entry.unwrap().1.is_none());
    }
}

// ── history ───────────────────────────────────────────────────────────────

#[test]
fn history_returns_all_observations_for_cell() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let cell = default_cell("rmlx", "gemma-4-e2b");
    // 3 decode_tps_warm + 2 peak_rss_mb = 5 total
    let rows = history(&conn, &cell, None, None).unwrap();
    assert_eq!(rows.len(), 5);
    // Ordered by ts_utc ASC
    for w in rows.windows(2) {
        assert!(w[0].ts_utc <= w[1].ts_utc);
    }
}

#[test]
fn history_filtered_by_metric() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let cell = default_cell("rmlx", "gemma-4-e2b");
    let rows = history(&conn, &cell, Some("decode_tps_warm"), None).unwrap();
    assert_eq!(rows.len(), 3);
    assert!(rows.iter().all(|r| r.metric == "decode_tps_warm"));
}

#[test]
fn history_filtered_by_since_ts() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let cell = default_cell("rmlx", "gemma-4-e2b");
    // since 2026-05-05 → should include the 05-05 and 05-08 rows for decode_tps_warm
    let rows = history(
        &conn,
        &cell,
        Some("decode_tps_warm"),
        Some("2026-05-05T00:00:00Z"),
    )
    .unwrap();
    assert_eq!(rows.len(), 2);
    assert!(rows
        .iter()
        .all(|r| r.ts_utc.as_str() >= "2026-05-05T00:00:00Z"));
}

// ── timeseries ────────────────────────────────────────────────────────────

#[test]
fn timeseries_buckets_by_day() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let cell = default_cell("rmlx", "gemma-4-e2b");
    let pts = timeseries(&conn, &cell, "decode_tps_warm", None, Bucket::Day).unwrap();
    // 3 observations on 3 different days → 3 buckets
    assert_eq!(pts.len(), 3);
    // Bucket labels are date strings like "2026-05-01"
    assert_eq!(pts[0].bucket_start_utc, "2026-05-01");
    assert_eq!(pts[0].mean_value, 95.0);
    assert_eq!(pts[0].n, 1);
}

#[test]
fn timeseries_buckets_by_week() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let cell = default_cell("rmlx", "gemma-4-e2b");
    let pts = timeseries(&conn, &cell, "decode_tps_warm", None, Bucket::Week).unwrap();
    // 2026-05-01 is Friday → Monday = 2026-04-27
    // 2026-05-05 is Tuesday → Monday = 2026-05-04
    // 2026-05-08 is Friday → Monday = 2026-05-04
    // → 2 distinct Monday buckets: 2026-04-27 (1 obs) and 2026-05-04 (2 obs)
    assert_eq!(pts.len(), 2);
    assert_eq!(pts[0].bucket_start_utc, "2026-04-27");
    assert_eq!(pts[0].n, 1);
    assert_eq!(pts[1].bucket_start_utc, "2026-05-04");
    assert_eq!(pts[1].n, 2);
}

// ── deltas ────────────────────────────────────────────────────────────────

#[test]
fn deltas_unknown_sha_errors() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    let err = deltas(&conn, "sha_does_not_exist", None).unwrap_err();
    assert!(matches!(err, Error::Query(_)));
}

#[test]
fn deltas_no_change_returns_zero_delta() {
    let mut conn = test_conn();
    // Insert a single observation with sha_base, then ANOTHER with the same value and sha_after.
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e2b",
        "decode_tps_warm",
        95.0,
        "2026-05-01T10:00:00Z",
        Some("sha_base"),
    ))
    .unwrap();
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e2b",
        "decode_tps_warm",
        95.0, // exactly equal value after baseline
        "2026-05-05T10:00:00Z",
        Some("sha_after"),
    ))
    .unwrap();

    // sha_base baseline ts = 2026-05-01. Post-baseline best = 95.0 = baseline best = 95.0.
    // delta = 0%. With threshold=5%, delta=0 < 5 → NOT included (abs(0) > 5 is false).
    let rows = deltas(&conn, "sha_base", Some(5.0)).unwrap();
    // Row should NOT be in results because 0% delta is below 5% threshold.
    let row = rows.iter().find(|r| {
        r.cell.model == "gemma-4-e2b" && r.metric == "decode_tps_warm" && r.cell.backend == "rmlx"
    });
    assert!(
        row.is_none(),
        "zero-delta row should be excluded by threshold"
    );
}

// ── champions ─────────────────────────────────────────────────────────────

/// Seed a minimal two-backend scenario for champion tests:
/// rmlx=100, mlx_lm=110 for decode_tps_warm (higher_better).
fn seed_two_backend_one_metric(conn: &mut Connection) {
    let mut rec = Recorder::new(conn, "test@0.0.1");
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e2b",
        "decode_tps_warm",
        100.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
    rec.record_run(&make_run(
        "mlx_lm",
        "gemma-4-e2b",
        "decode_tps_warm",
        110.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
}

#[test]
fn champions_all_backends_picks_global_best() {
    let mut conn = test_conn();
    seed_two_backend_one_metric(&mut conn);
    let rows = champions(&conn, None).unwrap();
    assert_eq!(rows.len(), 1);
    let cell = rows[0].metrics.get("decode_tps_warm").unwrap();
    assert_eq!(cell.value, 110.0);
    assert_eq!(cell.backend, "mlx_lm");
}

#[test]
fn champions_with_backend_filter_picks_per_backend_best() {
    let mut conn = test_conn();
    seed_two_backend_one_metric(&mut conn);
    let rows = champions(&conn, Some("rmlx")).unwrap();
    assert_eq!(rows.len(), 1);
    let cell = rows[0].metrics.get("decode_tps_warm").unwrap();
    assert_eq!(cell.value, 100.0);
    assert_eq!(cell.backend, "rmlx");
}

#[test]
fn champions_groups_metrics_under_one_cell() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e2b",
        "decode_tps_warm",
        95.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
    rec.record_run(&make_run_lower(
        "rmlx",
        "gemma-4-e2b",
        "peak_rss_mb",
        2000.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
    let rows = champions(&conn, None).unwrap();
    assert_eq!(rows.len(), 1, "should be one cell-row");
    assert!(rows[0].metrics.contains_key("decode_tps_warm"));
    assert!(rows[0].metrics.contains_key("peak_rss_mb"));
    assert_eq!(rows[0].metrics.len(), 2);
}

#[test]
fn champions_sparse_metrics_omitted() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    // Only decode_tps_warm — step_ms_mean absent.
    rec.record_run(&make_run(
        "rmlx",
        "gemma-4-e2b",
        "decode_tps_warm",
        95.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
    let rows = champions(&conn, None).unwrap();
    assert_eq!(rows.len(), 1);
    assert!(rows[0].metrics.contains_key("decode_tps_warm"));
    assert!(!rows[0].metrics.contains_key("step_ms_mean"));
}

#[test]
fn champions_sorted_deterministic() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    // Insert in reverse alpha order.
    rec.record_run(&make_run(
        "rmlx",
        "zz-model",
        "decode_tps_warm",
        50.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
    rec.record_run(&make_run(
        "rmlx",
        "aa-model",
        "decode_tps_warm",
        60.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
    let rows = champions(&conn, None).unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].model, "aa-model");
    assert_eq!(rows[1].model, "zz-model");
}

#[test]
fn champions_lower_better_picks_min() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    let timestamps = [
        "2026-05-01T10:00:00Z",
        "2026-05-02T10:00:00Z",
        "2026-05-03T10:00:00Z",
    ];
    for (v, ts) in [10.0_f64, 5.0, 8.0].iter().zip(timestamps.iter()) {
        rec.record_run(&make_run_lower(
            "rmlx",
            "gemma-4-e2b",
            "step_ms_mean",
            *v,
            ts,
            None,
        ))
        .unwrap();
    }
    let rows = champions(&conn, None).unwrap();
    assert_eq!(rows.len(), 1);
    let cell = rows[0].metrics.get("step_ms_mean").unwrap();
    assert_eq!(cell.value, 5.0);
}

// ── regress ───────────────────────────────────────────────────────────────

/// Seed: champion = 97.0, latest = 92.15 (5% drop). Should exit 1 with 1% threshold.
fn seed_regress_scenario(conn: &mut Connection, champion_tps: f64, latest_tps: f64) {
    let mut rec = Recorder::new(conn, "test@0.0.1");
    // Champion row (earlier timestamp → will be the bests record since it's the max).
    rec.record_run(&make_run(
        "rmlx",
        "bonsai-8b",
        "decode_tps_warm",
        champion_tps,
        "2026-05-01T10:00:00Z",
        Some("sha_champ"),
    ))
    .unwrap();
    // Latest row (more recent but lower value → regression).
    rec.record_run(&make_run(
        "rmlx",
        "bonsai-8b",
        "decode_tps_warm",
        latest_tps,
        "2026-05-10T10:00:00Z",
        Some("sha_latest"),
    ))
    .unwrap();
}

#[test]
fn regress_flags_5pct_drop_on_decode_tps_warm() {
    let mut conn = test_conn();
    // champion = 97.0, latest = 92.15 → delta = (92.15-97)/97 ≈ -5.0%
    // With threshold=1% this should be a regression.
    let champion = 97.0_f64;
    let latest = champion * 0.95; // 5% drop
    seed_regress_scenario(&mut conn, champion, latest);

    let result = regress(&conn, "bonsai", "decode_tps_warm", None, 1.0).unwrap();
    assert!(result.regressed, "5% drop vs 1% threshold must be flagged");
    assert!(result.delta_pct.unwrap() < -1.0, "delta should be negative");
    assert!(
        result.message.contains("REGRESSED"),
        "message must say REGRESSED: {}",
        result.message
    );
    assert!(
        result.champion_value.is_some(),
        "champion_value must be present"
    );
    assert_eq!(result.champion_value.unwrap(), champion);
}

#[test]
fn regress_within_tolerance_returns_ok() {
    let mut conn = test_conn();
    // champion = 97.0, latest = 96.5 → delta ≈ -0.5%, threshold = 1% → within tolerance.
    let champion = 97.0_f64;
    let latest = 96.5_f64;
    seed_regress_scenario(&mut conn, champion, latest);

    let result = regress(&conn, "bonsai", "decode_tps_warm", None, 1.0).unwrap();
    assert!(!result.regressed, "0.5% drop vs 1% threshold must be ok");
    assert!(
        result.message.contains("ok"),
        "message must say ok: {}",
        result.message
    );
}

#[test]
fn regress_no_champion_returns_skip() {
    let conn = test_conn(); // empty DB — no observations at all
    let result = regress(&conn, "no-such-model", "decode_tps_warm", None, 1.0).unwrap();
    assert!(!result.regressed, "no champion is not a regression");
    assert!(result.champion_value.is_none());
    assert!(result.latest_value.is_none());
    assert!(
        result.message.contains("no champion") || result.message.contains("no observations"),
        "message: {}",
        result.message
    );
}

#[test]
fn regress_lower_better_regression_flagged() {
    // peak_rss_mb is lower_better: a rise is a regression.
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    // champion = 1800 MB (best = lowest)
    rec.record_run(&make_run_lower(
        "rmlx",
        "bonsai-8b",
        "peak_rss_mb",
        1800.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
    // latest = 1900 MB → 5.6% higher → regression for lower_better
    rec.record_run(&make_run_lower(
        "rmlx",
        "bonsai-8b",
        "peak_rss_mb",
        1900.0,
        "2026-05-10T10:00:00Z",
        None,
    ))
    .unwrap();

    let result = regress(&conn, "bonsai", "peak_rss_mb", None, 1.0).unwrap();
    assert!(
        result.regressed,
        "5.6% increase in lower_better must be flagged"
    );
    assert_eq!(result.direction, "lower_better");
}

#[test]
fn regress_lower_better_improvement_not_flagged() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    // champion = 1800 MB
    rec.record_run(&make_run_lower(
        "rmlx",
        "bonsai-8b",
        "peak_rss_mb",
        1800.0,
        "2026-05-01T10:00:00Z",
        None,
    ))
    .unwrap();
    // latest = 1750 MB → improvement (lower), not a regression
    rec.record_run(&make_run_lower(
        "rmlx",
        "bonsai-8b",
        "peak_rss_mb",
        1750.0,
        "2026-05-10T10:00:00Z",
        None,
    ))
    .unwrap();

    let result = regress(&conn, "bonsai", "peak_rss_mb", None, 1.0).unwrap();
    assert!(
        !result.regressed,
        "improvement in lower_better must not be flagged"
    );
}

#[test]
fn deltas_regression_flagged() {
    let mut conn = test_conn();
    seed_observations(&mut conn).unwrap();
    // sha002 earliest ts = 2026-05-05T10:00:00Z.
    // Observations at or before that ts for rmlx/qwen3-8b/decode_tps_warm:
    // 2026-05-04 value=75.0 (sha001) → baseline best = 75.0
    // Now insert a worse value after that baseline timestamp:
    let mut rec = Recorder::new(&mut conn, "test@0.0.1");
    rec.record_run(&make_run(
        "rmlx",
        "qwen3-8b",
        "decode_tps_warm",
        40.0, // worse than 75.0 → current best becomes 40.0
        "2026-05-09T10:00:00Z",
        Some("sha003"),
    ))
    .unwrap();

    // sha002 baseline → rmlx/qwen3-8b/decode_tps_warm baseline = 75.0, current = 40.0
    // delta = (40-75)/75*100 ≈ -46.7% → regressed (higher_better, delta < -5%)
    let rows = deltas(&conn, "sha002", Some(5.0)).unwrap();
    let reg_row = rows.iter().find(|r| {
        r.cell.model == "qwen3-8b" && r.metric == "decode_tps_warm" && r.cell.backend == "rmlx"
    });
    assert!(reg_row.is_some(), "should find qwen3-8b regression row");
    let reg = reg_row.unwrap();
    assert!(reg.regressed, "should be flagged as regressed");
    assert!(reg.delta_pct.unwrap() < -5.0);
}
