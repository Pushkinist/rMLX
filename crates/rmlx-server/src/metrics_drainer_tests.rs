use super::*;

// ── Helper: build a timestamped event ────────────────────────────────────

fn make_event(kind: MetricKind) -> MetricEvent {
    MetricEvent {
        model_id: "mlx-community__gemma-4-e2b-it-mxfp8".into(),
        kv_quant: "k8v8".into(),
        ts_utc: "2026-05-11T00:00:00Z".into(),
        ctx_max: 8192,
        kind,
    }
}

// ── Producer → consumer round-trip ───────────────────────────────────────

/// Emit 5 events through the channel and verify all reach the consumer.
#[tokio::test]
async fn round_trip_five_events() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("runs.db");

    let handle = spawn_drainer(db_path.clone());

    for i in 0u64..5 {
        let ok = handle.try_emit(make_event(MetricKind::KvCacheBytes(1024 * i)));
        assert!(ok, "event {i} should enqueue successfully");
    }
    assert_eq!(handle.dropped_count(), 0, "no drops under low load");

    // Allow the drainer task to flush (>100 ms deadline).
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    // Verify rows in SQLite.
    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM observations WHERE metric = 'kv_cache_bytes'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(count, 5, "all 5 events should produce 1 observation each");
}

// ── Backpressure drop behavior ────────────────────────────────────────────

/// Flood the channel beyond capacity and verify the dropped counter rises.
///
/// We use a small custom channel so the test completes quickly.
#[test]
fn backpressure_increments_dropped_counter() {
    let (tx, _rx) = mpsc::channel::<MetricEvent>(2);
    let dropped = Arc::new(AtomicU64::new(0));
    let handle = DrainerHandle {
        tx: Some(tx),
        dropped: Arc::clone(&dropped),
    };

    // First 2 sends succeed (capacity = 2).
    assert!(handle.try_emit(make_event(MetricKind::TtftMs(10))));
    assert!(handle.try_emit(make_event(MetricKind::TtftMs(20))));

    // Third send should be dropped.
    assert!(!handle.try_emit(make_event(MetricKind::TtftMs(30))));
    assert_eq!(
        handle.dropped_count(),
        1,
        "one event should be counted as dropped"
    );

    // Fourth send also dropped.
    assert!(!handle.try_emit(make_event(MetricKind::TtftMs(40))));
    assert_eq!(handle.dropped_count(), 2);
}

// ── split_model_id ───────────────────────────────────────────────────────

#[test]
fn split_model_id_with_separator() {
    let (ns, name) = rmlx_metrics::identity::split_model_id("mlx-community__gemma-4-e2b-it-mxfp8");
    assert_eq!(ns, "mlx-community"); // hyphens preserved for whitelist match
    assert_eq!(name, "gemma-4-e2b-it-mxfp8");
}

#[test]
fn split_model_id_no_separator() {
    let (ns, name) = rmlx_metrics::identity::split_model_id("my-model");
    assert_eq!(ns, "local"); // "local" is always in NAMESPACE_WHITELIST
    assert_eq!(name, "my-model");
}

// ── event_kind_to_metrics ────────────────────────────────────────────────

#[test]
fn kv_cache_bytes_produces_one_metric() {
    let m = event_kind_to_metrics(&MetricKind::KvCacheBytes(12345));
    assert_eq!(m.len(), 1);
    assert_eq!(m[0].name, "kv_cache_bytes");
    assert!((m[0].value.unwrap() - 12345.0).abs() < 1.0);
}

#[test]
fn load_phases_produces_five_metrics() {
    let m = event_kind_to_metrics(&MetricKind::LoadPhases {
        mmap_ms: 1.0,
        dequant_ms: 2.0,
        gpu_residency_ms: 3.0,
        first_kernel_ready_ms: 4.0,
        total_ms: 10.0,
    });
    assert_eq!(m.len(), 5);
    let names: Vec<&str> = m.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"load_mmap_ms"));
    assert!(names.contains(&"load_total_ms"));
}

// ── infer_weight_quant ───────────────────────────────────────────────────

#[test]
fn infer_weight_quant_mxfp8() {
    assert_eq!(
        rmlx_metrics::identity::infer_weight_quant("mlx-community__gemma-4-e2b-it-mxfp8"),
        "mxfp8"
    );
}

#[test]
fn infer_weight_quant_fallback() {
    // No token matches → "bf16" (always-valid whitelist fallback)
    assert_eq!(
        rmlx_metrics::identity::infer_weight_quant("some-obscure-model"),
        "bf16"
    );
}

// ── F1b: PromptTokens / CompletionTokens mapping ────────────────────────

/// `PromptTokens` maps to one `MetricEntry` row with name
/// `"prompt_tokens_live"`.
#[test]
fn f1b_prompt_tokens_produces_one_metric() {
    let m = event_kind_to_metrics(&MetricKind::PromptTokens(4096));
    assert_eq!(m.len(), 1, "PromptTokens must emit exactly 1 MetricEntry");
    assert_eq!(m[0].name, "prompt_tokens_live");
    assert!(
        (m[0].value.unwrap() - 4096.0).abs() < 1.0,
        "value must match"
    );
}

/// `CompletionTokens` maps to one `MetricEntry` row with name
/// `"completion_tokens_live"`.
#[test]
fn f1b_completion_tokens_produces_one_metric() {
    let m = event_kind_to_metrics(&MetricKind::CompletionTokens(32));
    assert_eq!(
        m.len(),
        1,
        "CompletionTokens must emit exactly 1 MetricEntry"
    );
    assert_eq!(m[0].name, "completion_tokens_live");
    assert!((m[0].value.unwrap() - 32.0).abs() < 1.0, "value must match");
}

// ── F2: ctx_max threading ────────────────────────────────────────────────

/// `MetricEvent.ctx_max` is forwarded to `RunRecord.ctx_max` (not
/// the old sentinel `1`).
#[tokio::test]
async fn f2_ctx_max_is_real_value_not_sentinel() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("runs.db");

    let handle = spawn_drainer(db_path.clone());

    // Emit an event with a known ctx_max.
    let ev = MetricEvent {
        model_id: "mlx-community__gemma-4-e2b-it-mxfp8".into(),
        kv_quant: "k8v8".into(),
        ts_utc: "2026-05-18T00:00:00Z".into(),
        ctx_max: 8192,
        kind: MetricKind::KvCacheBytes(512),
    };
    assert!(handle.try_emit(ev));

    // Allow the drainer to flush.
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let stored_ctx_max: i64 = conn
        .query_row(
            "SELECT ctx_max FROM observations WHERE metric = 'kv_cache_bytes' LIMIT 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        stored_ctx_max, 8192,
        "ctx_max should be 8192, not sentinel 1"
    );
}

// ── F1b: registry coverage check ────────────────────────────────────────

/// The two new metric names must be present in the canonical registry
/// (no WARN-drop on record_run path).
#[test]
fn f1b_new_metrics_registered() {
    use rmlx_metrics::registry::lookup;
    let (unit, dir) =
        lookup("prompt_tokens_live").expect("prompt_tokens_live must be in the metric registry");
    assert_eq!(unit, "count");
    assert_eq!(dir, rmlx_metrics::registry::Direction::LowerBetter);

    let (unit2, dir2) = lookup("completion_tokens_live")
        .expect("completion_tokens_live must be in the metric registry");
    assert_eq!(unit2, "count");
    assert_eq!(dir2, rmlx_metrics::registry::Direction::LowerBetter);
}

// ── F9: ItlP99Ms / ItlSpikes metric mapping ─────────────────────────────

/// `ItlP99Ms` maps to one `MetricEntry` row with name `"itl_p99_ms"`.
#[test]
fn f9_itl_p99_produces_one_metric() {
    let m = event_kind_to_metrics(&MetricKind::ItlP99Ms(13.7));
    assert_eq!(m.len(), 1, "ItlP99Ms must emit exactly 1 MetricEntry");
    assert_eq!(m[0].name, "itl_p99_ms");
    assert!(
        (m[0].value.unwrap() - 13.7).abs() < 0.01,
        "value must match"
    );
}

/// `ItlSpikes` maps to one `MetricEntry` row with name `"itl_spikes"`.
#[test]
fn f9_itl_spikes_produces_one_metric() {
    let m = event_kind_to_metrics(&MetricKind::ItlSpikes(3));
    assert_eq!(m.len(), 1, "ItlSpikes must emit exactly 1 MetricEntry");
    assert_eq!(m[0].name, "itl_spikes");
    assert!((m[0].value.unwrap() - 3.0).abs() < 0.01, "value must match");
}

/// Both `itl_p99_ms` and `itl_spikes` must be in the registry (no WARN-drop).
#[test]
fn f9_new_metrics_registered() {
    use rmlx_metrics::registry::lookup;
    let (unit, dir) = lookup("itl_p99_ms").expect("itl_p99_ms must be in the metric registry");
    assert_eq!(unit, "ms");
    assert_eq!(dir, rmlx_metrics::registry::Direction::LowerBetter);

    let (unit2, dir2) = lookup("itl_spikes").expect("itl_spikes must be in the metric registry");
    assert_eq!(unit2, "count");
    assert_eq!(dir2, rmlx_metrics::registry::Direction::LowerBetter);
}

/// Round-trip: emit `ItlP99Ms` and `ItlSpikes` through the drainer and
/// verify rows appear in the `observations` table.
#[tokio::test]
async fn f9_itl_p99_and_spikes_round_trip() {
    let dir = tempfile::tempdir().unwrap();
    let db_path = dir.path().join("runs.db");

    let handle = spawn_drainer(db_path.clone());

    handle.try_emit(make_event(MetricKind::ItlP99Ms(18.4)));
    handle.try_emit(make_event(MetricKind::ItlSpikes(2)));

    tokio::time::sleep(std::time::Duration::from_millis(250)).await;

    let conn = rusqlite::Connection::open(&db_path).unwrap();
    let p99_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM observations WHERE metric = 'itl_p99_ms'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(p99_count, 1, "itl_p99_ms must produce 1 observation row");

    let spikes_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM observations WHERE metric = 'itl_spikes'",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(spikes_count, 1, "itl_spikes must produce 1 observation row");
}

// ── ItlStats metric mapping ──────────────────────────────────────────────

/// `ItlStats` must produce exactly three `MetricEntry` rows with the
/// correct names and values (M30 — itl_p50_ms, itl_p95_ms, step_ms_mean).
#[test]
fn itl_stats_produces_three_metrics_with_correct_names() {
    let m = event_kind_to_metrics(&MetricKind::ItlStats {
        p50_ms: 8.3,
        p95_ms: 11.2,
        mean_ms: 8.5,
        step_count: 128,
    });
    assert_eq!(m.len(), 3, "ItlStats must emit exactly 3 metric entries");

    let names: Vec<&str> = m.iter().map(|e| e.name.as_str()).collect();
    assert!(names.contains(&"itl_p50_ms"), "must include itl_p50_ms");
    assert!(names.contains(&"itl_p95_ms"), "must include itl_p95_ms");
    assert!(names.contains(&"step_ms_mean"), "must include step_ms_mean");

    let p50 = m.iter().find(|e| e.name == "itl_p50_ms").unwrap();
    let p95 = m.iter().find(|e| e.name == "itl_p95_ms").unwrap();
    let mean = m.iter().find(|e| e.name == "step_ms_mean").unwrap();

    assert!(
        (p50.value.unwrap() - 8.3).abs() < 0.01,
        "p50 value mismatch"
    );
    assert!(
        (p95.value.unwrap() - 11.2).abs() < 0.01,
        "p95 value mismatch"
    );
    assert!(
        (mean.value.unwrap() - 8.5).abs() < 0.01,
        "mean value mismatch"
    );
}
