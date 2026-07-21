use super::*;

#[test]
fn record_inserts_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "test-run-001").expect("open");
    rec.record(&Measurement {
        model_path: "/m",
        quant_mode: "mxfp8 g32",
        stage: "stage0",
        op: "total_tensors",
        value_unit: "count",
        value: 42.0,
        notes: "",
    })
    .expect("record");

    let conn = schema::open(&db).expect("reopen");
    let cnt: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE run_id = ?1",
            params!["test-run-001"],
            |r| r.get(0),
        )
        .expect("count");
    assert_eq!(cnt, 1, "exactly one row written");

    let op: String = conn
        .query_row(
            "SELECT op FROM events WHERE run_id = ?1",
            params!["test-run-001"],
            |r| r.get(0),
        )
        .expect("op");
    assert_eq!(op, "total_tensors");
}

#[test]
fn two_records_two_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "two").expect("open");
    for op in ["a", "b"] {
        rec.record(&Measurement {
            model_path: "/m",
            quant_mode: "none",
            stage: "stage0",
            op,
            value_unit: "count",
            value: 1.0,
            notes: "",
        })
        .expect("record");
    }
    let conn = schema::open(&db).expect("reopen");
    let cnt: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE run_id = ?1",
            params!["two"],
            |r| r.get(0),
        )
        .expect("count");
    assert_eq!(cnt, 2);
}

// ── per-request metric writes ─────────────────────────────────────

/// writing ttft_warm_ms and ttft_cold_ms each produce exactly one row
/// with the correct op and value.
#[test]
fn ttft_warm_and_cold_produce_distinct_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "test-ttft").expect("open");

    // Warm TTFT.
    rec.record(&Measurement {
        model_path: "mlx-community__bonsai",
        quant_mode: "n/a",
        stage: "request",
        op: "ttft_warm_ms",
        value_unit: "ms",
        value: 42.0,
        notes: "",
    })
    .expect("record warm");

    // Cold TTFT.
    rec.record(&Measurement {
        model_path: "mlx-community__bonsai",
        quant_mode: "n/a",
        stage: "request",
        op: "ttft_cold_ms",
        value_unit: "ms",
        value: 380.0,
        notes: "",
    })
    .expect("record cold");

    let conn = schema::open(&db).expect("reopen");

    let warm_val: f64 = conn
        .query_row(
            "SELECT value FROM events WHERE op = 'ttft_warm_ms' LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("ttft_warm_ms row");
    assert!((warm_val - 42.0).abs() < 1.0, "warm TTFT value must match");

    let cold_val: f64 = conn
        .query_row(
            "SELECT value FROM events WHERE op = 'ttft_cold_ms' LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("ttft_cold_ms row");
    assert!((cold_val - 380.0).abs() < 1.0, "cold TTFT value must match");
}

/// itl_p50_ms / itl_p95_ms / itl_p99_ms each produce one row with
/// the correct value and value_unit = "ms".
#[test]
fn itl_percentiles_produce_three_rows() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "test-itl").expect("open");

    for (op, val) in [
        ("itl_p50_ms", 8.3),
        ("itl_p95_ms", 14.1),
        ("itl_p99_ms", 20.5),
    ] {
        rec.record(&Measurement {
            model_path: "mlx-community__bonsai",
            quant_mode: "k8v8",
            stage: "request",
            op,
            value_unit: "ms",
            value: val,
            notes: "",
        })
        .expect("record itl");
    }

    let conn = schema::open(&db).expect("reopen");
    let cnt: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE run_id = 'test-itl' AND value_unit = 'ms'",
            [],
            |r| r.get(0),
        )
        .expect("count");
    assert_eq!(cnt, 3, "must produce 3 ITL rows");

    let p50: f64 = conn
        .query_row(
            "SELECT value FROM events WHERE op = 'itl_p50_ms' LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("p50 row");
    assert!((p50 - 8.3).abs() < 0.01, "p50 value mismatch");

    let p95: f64 = conn
        .query_row(
            "SELECT value FROM events WHERE op = 'itl_p95_ms' LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("p95 row");
    assert!((p95 - 14.1).abs() < 0.01, "p95 value mismatch");

    let p99: f64 = conn
        .query_row(
            "SELECT value FROM events WHERE op = 'itl_p99_ms' LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("p99 row");
    assert!((p99 - 20.5).abs() < 0.01, "p99 value mismatch");
}

/// when no ITL percentiles are written (simulating a 1-token
/// completion where the percentile guard fires), no itl_p*_ms rows appear.
///
/// This is a unit test of the events table itself — the guard logic lives
/// in engine.rs (`compute_itl_stats` returning `None` for < 2 tokens).
/// We verify here that NOT writing produces zero rows, i.e. the table state
/// is consistent even when the engine path is a no-op.
#[test]
fn no_itl_emit_for_single_token_completion() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "test-single").expect("open");

    // Write only a TTFT — what the engine writes for a 1-token completion.
    rec.record(&Measurement {
        model_path: "mlx-community__bonsai",
        quant_mode: "n/a",
        stage: "request",
        op: "ttft_warm_ms",
        value_unit: "ms",
        value: 55.0,
        notes: "",
    })
    .expect("record ttft");

    // Assert no ITL percentile rows were written.
    let conn = schema::open(&db).expect("reopen");
    let itl_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM events WHERE run_id = 'test-single' AND op LIKE 'itl_p%_ms'",
            [],
            |r| r.get(0),
        )
        .expect("itl count");
    assert_eq!(
        itl_count, 0,
        "no ITL rows expected for single-token completion"
    );
}

/// kv_cache_bytes produces one row with value_unit = "bytes".
#[test]
fn kv_cache_bytes_produces_one_row() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "test-kv").expect("open");

    rec.record(&Measurement {
        model_path: "mlx-community__bonsai",
        quant_mode: "k8v8",
        stage: "request",
        op: "kv_cache_bytes",
        value_unit: "bytes",
        value: 1_073_741_824.0, // 1 GiB
        notes: "",
    })
    .expect("record kv");

    let conn = schema::open(&db).expect("reopen");
    let val: f64 = conn
        .query_row(
            "SELECT value FROM events WHERE op = 'kv_cache_bytes' LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("kv_cache_bytes row");
    assert!(
        (val - 1_073_741_824.0).abs() < 1.0,
        "kv_cache_bytes value mismatch"
    );

    let unit: String = conn
        .query_row(
            "SELECT value_unit FROM events WHERE op = 'kv_cache_bytes' LIMIT 1",
            [],
            |r| r.get(0),
        )
        .expect("value_unit row");
    assert_eq!(unit, "bytes", "value_unit must be 'bytes'");
}

// ── `events` schema-shape tolerance ─────────────────────────────────────────
//
// Migration `003_events_identity.sql` briefly added `events.git_sha` before
// this contract settled on "events has no git_sha column" (the binary has no
// caller-supplied provenance for it — see `events.rs`'s module doc). A DB
// already migrated under the pre-fix 003 (the real `.rmlx/metrics/runs.db`
// among them) keeps that stray, permanently-`NULL` column: migrations are
// version-gated, so a DB already at `user_version=3` never re-runs 003. Both
// shapes must record cleanly, since `EventRecorder::record`'s `INSERT` names
// its columns explicitly either way.

/// Fresh DB, amended 003 applied: `events` has no `git_sha` column at all.
#[test]
fn record_on_fresh_db_has_no_git_sha_column() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "fresh-shape").expect("open");
    rec.record(&Measurement {
        model_path: "/m",
        quant_mode: "none",
        stage: "stage0",
        op: "total_tensors",
        value_unit: "count",
        value: 1.0,
        notes: "",
    })
    .expect("record on fresh (amended-003) schema");

    let conn = schema::open(&db).expect("reopen");
    let has_git_sha: bool = conn
        .prepare("SELECT 1 FROM pragma_table_info('events') WHERE name = 'git_sha'")
        .expect("prepare pragma_table_info")
        .exists([])
        .expect("query pragma_table_info");
    assert!(
        !has_git_sha,
        "fresh DB must not carry a git_sha column on events"
    );

    let backend_version: String = conn
        .query_row(
            "SELECT backend_version FROM events WHERE run_id = ?1",
            params!["fresh-shape"],
            |r| r.get(0),
        )
        .expect("backend_version row");
    assert!(
        !backend_version.is_empty(),
        "backend_version must still be stamped"
    );
}

/// Legacy-shape DB: `events` already carries the stray `git_sha` column from
/// a pre-fix migration 003 (simulates the real, already-migrated database).
/// `record()` must still succeed — the `INSERT`'s explicit column list never
/// mentions `git_sha`, so an extra nullable column on the table is inert.
#[test]
fn record_on_legacy_db_with_stray_git_sha_column_still_works() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    // Build the DB by hand up through the pre-fix shape of migration 003,
    // then mark it as already-migrated (`user_version = 3`) so
    // `migrate::run_pending` skips re-applying 003 — exactly how a real,
    // already-migrated database behaves.
    {
        let conn = schema::open(&db).expect("create");
        conn.execute_batch(schema::MIGRATIONS[0].1)
            .expect("apply 001_init");
        conn.execute_batch(schema::MIGRATIONS[1].1)
            .expect("apply 002_events");
        conn.execute_batch(
            "ALTER TABLE events ADD COLUMN backend_version TEXT;
             ALTER TABLE events ADD COLUMN git_sha         TEXT;
             ALTER TABLE events ADD COLUMN build_profile   TEXT;",
        )
        .expect("apply pre-fix 003 shape (with the stray git_sha column)");
        conn.execute_batch("PRAGMA user_version = 3;")
            .expect("mark schema as already migrated to 3");
    }

    let rec = EventRecorder::open_at(&db, "legacy-shape").expect("open");
    rec.record(&Measurement {
        model_path: "/m",
        quant_mode: "none",
        stage: "stage0",
        op: "total_tensors",
        value_unit: "count",
        value: 1.0,
        notes: "",
    })
    .expect("record on legacy (stray-git_sha-column) schema");

    let conn = schema::open(&db).expect("reopen");
    let has_git_sha: bool = conn
        .prepare("SELECT 1 FROM pragma_table_info('events') WHERE name = 'git_sha'")
        .expect("prepare pragma_table_info")
        .exists([])
        .expect("query pragma_table_info");
    assert!(has_git_sha, "test setup must preserve the stray column");

    let (backend_version, git_sha): (String, Option<String>) = conn
        .query_row(
            "SELECT backend_version, git_sha FROM events WHERE run_id = ?1",
            params!["legacy-shape"],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .expect("row");
    assert!(
        !backend_version.is_empty(),
        "backend_version must still be stamped"
    );
    assert!(
        git_sha.is_none(),
        "the stray git_sha column must stay NULL — nothing ever writes it"
    );
}

/// `mlx_nax` is populated on every insert, exactly like `backend_version` /
/// `build_profile` — sourced from the same process-wide [`crate::identity::RunIdentity`]
/// [`EventRecorder::record`] already reads for those two columns. The exact
/// string ("present" / "absent" / "unknown") depends on whether
/// `crate::identity::set_mlx_nax` was ever called in this process (only
/// `rmlx-cli::main()` does, in production) — this test asserts the row is
/// never NULL and always matches whatever the cached identity reports at
/// insert time, not a specific value.
#[test]
fn record_populates_mlx_nax() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let rec = EventRecorder::open_at(&db, "mlx-nax-run").expect("open");
    rec.record(&Measurement {
        model_path: "/m",
        quant_mode: "none",
        stage: "stage0",
        op: "total_tensors",
        value_unit: "count",
        value: 1.0,
        notes: "",
    })
    .expect("record");

    let conn = schema::open(&db).expect("reopen");
    let mlx_nax: Option<String> = conn
        .query_row(
            "SELECT mlx_nax FROM events WHERE run_id = ?1",
            params!["mlx-nax-run"],
            |r| r.get(0),
        )
        .expect("mlx_nax row");
    assert_eq!(
        mlx_nax.as_deref(),
        Some(RunIdentity::get().mlx_nax()),
        "mlx_nax must be populated from the same identity backend_version/build_profile use"
    );
}
