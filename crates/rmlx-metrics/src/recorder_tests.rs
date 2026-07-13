use super::*;
use crate::ingest::{MetricEntry, PromptRef, RunRecord};
use rusqlite::Connection;
use serde_json::json;

// ── Helpers ───────────────────────────────────────────────────────────────

fn test_conn() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::migrate::run_pending(&mut conn).unwrap();
    conn
}

fn base_run() -> RunRecord {
    RunRecord {
        schema_version: crate::ingest::RECORD_SCHEMA_VERSION,
        backend: "rmlx".into(),
        backend_version: Some("0.0.1".into()),
        model_namespace: "mlx-community".into(),
        model: "gemma-4-e4b-it-mxfp8".into(),
        weight_quant: "mxfp8".into(),
        kv_quant: "k8v8".into(),
        ctx_max: 8192,
        prompt: PromptRef::ByBody {
            name: "test_prompt".into(),
            body: json!("the quick brown fox"),
            notes: None,
            tokens_approx: Some(4),
        },
        ts_utc: "2026-05-10T07:30:00Z".into(),
        git_sha: Some("abc1234".into()),
        build_profile: Some("release".into()),
        hardware_tag: "m5_max_128gb".into(),
        prompt_tokens: Some(4),
        max_tokens: Some(32),
        temperature: Some(0.0),
        seed: Some(0),
        n_warmups: Some(1),
        n_measure: Some(3),
        output_first_64: Some("hello world".into()),
        notes: Some("auto-summary".into()),
        description: Some("abc1234: test run".into()),
        metrics: vec![
            MetricEntry {
                name: "decode_tps_warm".into(),
                value: Some(95.0),
                stddev: Some(1.5),
            },
            MetricEntry {
                name: "prefill_tps".into(),
                value: Some(500.0),
                stddev: None,
            },
            MetricEntry {
                name: "step_ms_mean".into(),
                value: Some(10.5),
                stddev: None,
            },
            MetricEntry {
                name: "peak_rss_mb".into(),
                value: None,
                stddev: None,
            },
        ],
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[test]
fn record_one_run_inserts_n_observations() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "rmlx-test@0.0.1");
    let outcome = rec.record_run(&base_run()).unwrap();

    assert_eq!(outcome.observation_ids.len(), 3, "3 non-null metrics");
    assert_eq!(outcome.skipped_metrics.len(), 1, "peak_rss_mb was null");
    assert_eq!(outcome.skipped_metrics[0], "peak_rss_mb");
}

#[test]
fn record_run_atomic_on_validation_failure() {
    let mut conn = test_conn();
    let mut run = base_run();
    run.metrics[0].name = "fake_metric".into();

    let mut rec = Recorder::new(&mut conn, "rmlx-test@0.0.1");
    assert!(rec.record_run(&run).is_err());

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn record_run_atomic_on_unknown_prompt_sha() {
    let mut conn = test_conn();
    let mut run = base_run();
    run.prompt = PromptRef::BySha256 {
        sha256: "a".repeat(64),
    };

    let mut rec = Recorder::new(&mut conn, "rmlx-test@0.0.1");
    let err = rec.record_run(&run).unwrap_err();
    assert!(matches!(err, Error::InvalidPrompt(_)));

    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 0);
}

#[test]
fn record_run_resolves_existing_prompt_by_body() {
    let mut conn = test_conn();

    // Pre-insert via PromptStore.
    let body = json!("the quick brown fox");
    let pre_id = {
        let store = PromptStore::new(&conn);
        store
            .get_or_insert("test_prompt", &body, Some(4), None)
            .unwrap()
    };

    // Recorder uses ByBody — must resolve to the SAME id.
    let mut rec = Recorder::new(&mut conn, "rmlx-test@0.0.1");
    let outcome = rec.record_run(&base_run()).unwrap();

    assert_eq!(outcome.prompt_id, pre_id, "no duplicate prompts row");

    let prompt_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM prompts", [], |r| r.get(0))
        .unwrap();
    assert_eq!(prompt_count, 1, "exactly one prompts row");
}

#[test]
fn record_run_uses_minted_run_id_format() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "rmlx-test@0.0.1");
    let outcome = rec.record_run(&base_run()).unwrap();

    let re = regex_lite::Regex::new(r"^\d{14}-[0-9a-f]{6}$").unwrap();
    assert!(
        re.is_match(&outcome.run_id),
        "run_id {:?} did not match pattern",
        outcome.run_id
    );
}

#[test]
fn record_run_inserted_by_field_populated() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "rmlx-cli@0.0.1");
    let outcome = rec.record_run(&base_run()).unwrap();

    let id = outcome.observation_ids[0];
    let inserted_by: String = conn
        .query_row(
            "SELECT inserted_by FROM observations WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap();

    assert_eq!(inserted_by, "rmlx-cli@0.0.1");
}

#[test]
fn record_run_inserted_utc_recent() {
    let mut conn = test_conn();
    let mut rec = Recorder::new(&mut conn, "rmlx-test@0.0.1");
    let outcome = rec.record_run(&base_run()).unwrap();

    let id = outcome.observation_ids[0];
    let inserted_utc: String = conn
        .query_row(
            "SELECT inserted_utc FROM observations WHERE id = ?1",
            params![id],
            |r| r.get(0),
        )
        .unwrap();

    // Parse and assert within last 5 seconds.
    let written = time::OffsetDateTime::parse(
        &inserted_utc,
        &time::format_description::well_known::Iso8601::DEFAULT,
    )
    .expect("inserted_utc must be ISO-8601");
    let now = time::OffsetDateTime::now_utc();
    let delta = (now - written).whole_seconds().abs();
    assert!(delta < 5, "inserted_utc too far from now: delta={delta}s");
}

#[test]
fn record_run_decode_stddev_only_on_decode_metrics() {
    let mut conn = test_conn();
    let mut run = base_run();
    // Set stddev on decode and step metrics explicitly.
    run.metrics = vec![
        MetricEntry {
            name: "decode_tps_warm".into(),
            value: Some(95.0),
            stddev: Some(0.5),
        },
        MetricEntry {
            name: "step_ms_mean".into(),
            value: Some(10.0),
            stddev: Some(2.1),
        },
    ];

    let mut rec = Recorder::new(&mut conn, "rmlx-test@0.0.1");
    let outcome = rec.record_run(&run).unwrap();
    assert_eq!(outcome.observation_ids.len(), 2);

    let (decode_std, step_std): (Option<f64>, Option<f64>) = conn
        .query_row(
            "SELECT
                MAX(CASE WHEN metric = 'decode_tps_warm' THEN decode_stddev END),
                MAX(CASE WHEN metric = 'step_ms_mean'    THEN decode_stddev END)
             FROM observations WHERE run_id = ?1",
            params![outcome.run_id],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();

    assert!((decode_std.unwrap() - 0.5).abs() < 1e-9, "decode stddev");
    assert!((step_std.unwrap() - 2.1).abs() < 1e-9, "step stddev");
}

#[test]
fn record_run_observation_columns_match_input() {
    let mut conn = test_conn();
    let run = base_run();

    let mut rec = Recorder::new(&mut conn, "rmlx-test@0.0.1");
    let outcome = rec.record_run(&run).unwrap();

    // Pick the decode_tps_warm row.
    let id = outcome.observation_ids[0];

    #[allow(clippy::type_complexity)]
    let row: (
        String,
        String,
        String,
        String,
        String,
        i64,
        i64,
        String,
        f64,
        String,
        String,
        String,
        String,
        Option<String>,
        Option<String>,
        Option<String>,
        String,
        Option<i64>,
        Option<i64>,
        Option<f64>,
        Option<i64>,
        Option<i64>,
        Option<i64>,
        Option<String>,
        Option<f64>,
        Option<String>,
        Option<String>,
    ) = conn
        .query_row(
            "SELECT
                backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
                metric, value, unit, direction,
                run_id, ts_utc, git_sha, build_profile, backend_version, hardware_tag,
                prompt_tokens, max_tokens, temperature, seed, n_warmups, n_measure,
                output_first_64, decode_stddev, notes, description
             FROM observations WHERE id = ?1",
            params![id],
            |r| {
                Ok((
                    r.get(0)?,
                    r.get(1)?,
                    r.get(2)?,
                    r.get(3)?,
                    r.get(4)?,
                    r.get(5)?,
                    r.get(6)?,
                    r.get(7)?,
                    r.get(8)?,
                    r.get(9)?,
                    r.get(10)?,
                    r.get(11)?,
                    r.get(12)?,
                    r.get(13)?,
                    r.get(14)?,
                    r.get(15)?,
                    r.get(16)?,
                    r.get(17)?,
                    r.get(18)?,
                    r.get(19)?,
                    r.get(20)?,
                    r.get(21)?,
                    r.get(22)?,
                    r.get(23)?,
                    r.get(24)?,
                    r.get(25)?,
                    r.get(26)?,
                ))
            },
        )
        .unwrap();

    assert_eq!(row.0, "rmlx");
    assert_eq!(row.1, "mlx-community");
    assert_eq!(row.2, "gemma-4-e4b-it-mxfp8");
    assert_eq!(row.3, "mxfp8");
    assert_eq!(row.4, "k8v8");
    assert_eq!(row.5, 8192_i64);
    assert_eq!(row.6, outcome.prompt_id);
    assert_eq!(row.7, "decode_tps_warm");
    assert!((row.8 - 95.0).abs() < 1e-9);
    assert_eq!(row.9, "tps");
    assert_eq!(row.10, "higher_better");
    assert_eq!(row.11, outcome.run_id);
    assert_eq!(row.12, "2026-05-10T07:30:00Z");
    assert_eq!(row.13.as_deref(), Some("abc1234"));
    assert_eq!(row.14.as_deref(), Some("release"));
    assert_eq!(row.15.as_deref(), Some("0.0.1"));
    assert_eq!(row.16, "m5_max_128gb");
    assert_eq!(row.17, Some(4));
    assert_eq!(row.18, Some(32));
    assert!((row.19.unwrap() - 0.0).abs() < 1e-9);
    assert_eq!(row.20, Some(0));
    assert_eq!(row.21, Some(1));
    assert_eq!(row.22, Some(3));
    assert_eq!(row.23.as_deref(), Some("hello world"));
    assert!((row.24.unwrap() - 1.5).abs() < 1e-9, "decode_stddev");
    assert_eq!(row.25.as_deref(), Some("auto-summary"));
    assert_eq!(row.26.as_deref(), Some("abc1234: test run"));
}
