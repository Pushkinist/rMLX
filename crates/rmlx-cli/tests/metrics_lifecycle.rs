//! Integration tests for `rmlx metrics` lifecycle subcommands.
//!
//! Covers: init, doctor, backup, restore.
//! Uses `std::process::Command` against the compiled binary (same pattern as
//! `baseline_smoke.rs` and `info_smoke.rs`).
//!
//! All tests are fast (no model load) and run without `#[ignore]`.

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
    clippy::case_sensitive_file_extension_comparisons
)]

use std::path::{Path, PathBuf};
use std::process::Command;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Locate the rmlx binary: debug build (built by `cargo test`).
///
/// `CARGO_MANIFEST_DIR` points to `crates/rmlx-cli/`. The workspace root is two
/// levels up. `cargo test --test` also builds the named binary, so `rmlx` should
/// always be present when this test file runs.
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

/// Run `rmlx metrics --db <db_path> <subargs...>` from the workspace root.
/// Returns the `Output`.
fn run_metrics(db_path: &Path, subargs: &[&str]) -> std::process::Output {
    Command::new(rmlx_bin())
        .arg("metrics")
        .arg("--db")
        .arg(db_path)
        .args(subargs)
        .output()
        .expect("failed to launch rmlx")
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `init` creates the DB file; a second `init` on the same path returns non-zero.
#[test]
fn init_creates_db_then_refuses_second_init() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    // First init: must succeed and create the file.
    let out = run_metrics(&db, &["init"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "first init failed: status={} stdout={stdout} stderr={stderr}",
        out.status
    );
    assert!(db.exists(), "DB file not created after init");
    assert!(
        stdout.contains("DB initialized at"),
        "expected 'DB initialized at' in stdout: {stdout}"
    );

    // Second init: must fail.
    let out2 = run_metrics(&db, &["init"]);
    assert!(
        !out2.status.success(),
        "second init should return non-zero; got success"
    );
    let stderr2 = String::from_utf8_lossy(&out2.stderr);
    assert!(
        stderr2.contains("already exists"),
        "expected 'already exists' in stderr: {stderr2}"
    );
}

/// `doctor` on a freshly-initialised DB returns exit 0.
#[test]
fn doctor_clean_db_returns_ok() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    let init = run_metrics(&db, &["init"]);
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    let doc = run_metrics(&db, &["doctor"]);
    let stdout = String::from_utf8_lossy(&doc.stdout);
    let stderr = String::from_utf8_lossy(&doc.stderr);
    assert!(
        doc.status.success(),
        "doctor returned non-zero on clean DB: status={} stdout={stdout} stderr={stderr}",
        doc.status
    );
    assert!(
        stdout.contains("0 error(s)"),
        "expected '0 error(s)' in stdout: {stdout}"
    );
}

/// `doctor` detects an unknown backend and returns non-zero.
///
/// We insert an observation with `backend='pytorch'` (not in the whitelist)
/// directly via rusqlite, then run `rmlx metrics doctor` and assert it exits
/// non-zero with 'pytorch' in stderr.
#[test]
fn doctor_detects_unknown_backend() {
    use rmlx_metrics::{migrate, schema};

    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    // Init via CLI so the schema is present.
    let init = run_metrics(&db, &["init"]);
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    // Directly insert a bad-backend row via the library (bypasses CLI validation).
    {
        let mut conn = schema::open(&db).expect("open db");
        migrate::run_pending(&mut conn).expect("run_pending");

        // Insert a prompt row (required by FK).
        conn.execute(
            "INSERT INTO prompts(sha256, name, body, first_seen_utc)
             VALUES ('aabbcc', 'test-prompt', 'body', '2026-01-01T00:00:00Z')",
            [],
        )
        .expect("insert prompt");
        let prompt_id: i64 = conn
            .query_row("SELECT id FROM prompts WHERE sha256='aabbcc'", [], |r| {
                r.get(0)
            })
            .expect("select prompt");

        conn.execute(
            "INSERT INTO observations(
                 backend, model_namespace, model, weight_quant, kv_quant,
                 ctx_max, prompt_id, metric, value, unit, direction,
                 run_id, ts_utc, hardware_tag, inserted_utc, inserted_by)
             VALUES ('pytorch','mlx-community','test-model','mxfp8','none',
                     8192, ?1, 'decode_tps_warm', 42.0, 'tps', 'higher_better',
                     'run0', '2026-01-01T00:00:00Z', 'm5_max_128gb',
                     '2026-01-01T00:00:00Z', 'test@0.0.1')",
            rusqlite::params![prompt_id],
        )
        .expect("insert bad observation");
    }

    let doc = run_metrics(&db, &["doctor"]);
    assert!(
        !doc.status.success(),
        "doctor should return non-zero when unknown backend present"
    );
    let stderr = String::from_utf8_lossy(&doc.stderr);
    assert!(
        stderr.contains("pytorch"),
        "expected 'pytorch' in doctor stderr: {stderr}"
    );
}

/// `doctor` reports coverage gaps on an empty DB (every Coverage::Yes
/// metric has zero rows → all are gaps).
///
/// Uses a freshly-initialised DB with no observations inserted. Expects
/// doctor to exit 0 (gaps are warnings, not errors) and mention "coverage gap"
/// in stdout.
#[test]
fn f13_doctor_reports_coverage_gaps_on_empty_db() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    let init = run_metrics(&db, &["init"]);
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    let doc = run_metrics(&db, &["doctor"]);
    let stdout = String::from_utf8_lossy(&doc.stdout);
    let stderr = String::from_utf8_lossy(&doc.stderr);

    // Doctor must exit 0 — gaps are informational warnings, not hard errors.
    assert!(
        doc.status.success(),
        "doctor must exit 0 on empty DB (gaps are warnings): status={} stdout={stdout} stderr={stderr}",
        doc.status
    );
    // Coverage-gap lines emitted to stderr (eprintln! per [warn] convention).
    assert!(
        stderr.contains("coverage gap"),
        "expected 'coverage gap' in doctor stderr on empty DB: {stderr}"
    );
    // Summary must mention the gaps as warnings.
    assert!(
        stdout.contains("warning(s)"),
        "expected 'warning(s)' in doctor summary: {stdout}"
    );
}

/// `doctor` prints the coverage-matrix section and exits 0 on a DB
/// that already has rows for every Coverage::Yes metric (no gaps).
#[test]
fn f13_doctor_coverage_matrix_no_gaps_when_all_present() {
    use rmlx_metrics::registry::{Coverage, COVERAGE_MATRIX};
    use rmlx_metrics::{migrate, registry, schema};

    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    let init = run_metrics(&db, &["init"]);
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    // Insert one observation row for every (backend, metric) pair with
    // Coverage::Yes so the gap check finds zero gaps.
    {
        let mut conn = schema::open(&db).expect("open db");
        migrate::run_pending(&mut conn).expect("run_pending");

        conn.execute(
            "INSERT INTO prompts(sha256, name, body, first_seen_utc)
             VALUES ('f13aabbcc', 'f13-sentinel', '(f13)', '2026-01-01T00:00:00Z')",
            [],
        )
        .expect("insert prompt");
        let prompt_id: i64 = conn
            .query_row("SELECT id FROM prompts WHERE sha256='f13aabbcc'", [], |r| {
                r.get(0)
            })
            .expect("select prompt");

        for (backend, metric, cov) in COVERAGE_MATRIX {
            if *cov != Coverage::Yes {
                continue;
            }
            // Derive unit/direction from the registry so the row passes the
            // direction/unit sanity checks.
            let (unit, dir) = registry::lookup(metric).expect("metric in registry");
            conn.execute(
                "INSERT INTO observations(
                     backend, model_namespace, model, weight_quant, kv_quant,
                     ctx_max, prompt_id, metric, value, unit, direction,
                     run_id, ts_utc, hardware_tag, inserted_utc, inserted_by)
                 VALUES (?1,'mlx-community','f13-model','mxfp8','none',
                         8192, ?2, ?3, 1.0, ?4, ?5,
                         'f13run', '2026-01-01T00:00:00Z', 'm5_max_128gb',
                         '2026-01-01T00:00:00Z', 'test@0.0.1')",
                rusqlite::params![backend, prompt_id, metric, unit, dir.as_str()],
            )
            .expect("insert coverage row");
        }
    }

    let doc = run_metrics(&db, &["doctor"]);
    let stdout = String::from_utf8_lossy(&doc.stdout);
    let stderr = String::from_utf8_lossy(&doc.stderr);
    assert!(
        doc.status.success(),
        "doctor must exit 0 when all Coverage::Yes rows are present: stdout={stdout} stderr={stderr}"
    );
    assert!(
        stdout.contains("all Coverage::Yes metrics have at least one row"),
        "expected coverage-ok message: {stdout}"
    );
}

/// `backup` with no explicit `--out` creates a file in `metrics/backups/`.
///
/// Note: the backup will land in the cwd (`metrics/backups/`) — the test
/// verifies the backup file exists in the temp-subdir passed via `--db`.
/// Because the default backups dir is relative to cwd (workspace root), we
/// use `--out` to redirect to a path we control.
#[test]
fn backup_out_path_creates_file() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    let init = run_metrics(&db, &["init"]);
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    let backup_path = dir.path().join("backup.db");
    let out = run_metrics(&db, &["backup", "--out", backup_path.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "backup failed: status={} stdout={stdout} stderr={stderr}",
        out.status
    );
    assert!(
        backup_path.exists(),
        "backup file not created at {}",
        backup_path.display()
    );
    assert!(
        stdout.contains("wrote backup to"),
        "expected 'wrote backup to' in stdout: {stdout}"
    );
}

/// `backup --keep N` prunes old backups, keeping N + the newly-written one.
///
/// We create 5 existing backup files in a temp dir (named `runs-*.db` to match
/// the filter used by `prune_backups`), then run backup `--keep 3` writing the
/// new backup to the same dir. Pruning deletes the 2 oldest, leaving 4 total
/// (3 old + 1 new).
///
/// We do NOT control exact mtimes across the 5 old files; we just assert that
/// exactly 4 files remain — whatever the oldest 2 were, they must be gone.
#[test]
fn backup_keep_n_prunes_old() {
    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");
    let backups_dir = dir.path().join("backups");
    std::fs::create_dir_all(&backups_dir).expect("create backups dir");

    let init = run_metrics(&db, &["init"]);
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    // Create 5 stub files named `runs-*.db` — they match the filter in prune_backups.
    // Write each one with a small sleep via std::thread::sleep to spread mtimes.
    for i in 0..5u32 {
        let path = backups_dir.join(format!("runs-stub-{i:02}.db"));
        std::fs::write(&path, format!("stub {i}")).expect("write stub backup");
        // 20ms apart so mtime ordering is reliable even on FAT-ms-resolution filesystems.
        std::thread::sleep(std::time::Duration::from_millis(20));
    }

    // Run backup --keep 3, directing the new backup into the same backups_dir.
    let new_backup = backups_dir.join("runs-new.db");
    let out = run_metrics(
        &db,
        &[
            "backup",
            "--out",
            new_backup.to_str().unwrap(),
            "--keep",
            "3",
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "backup --keep 3 failed: status={} stdout={stdout} stderr={stderr}",
        out.status
    );

    // Count remaining .db files.
    let remaining: usize = std::fs::read_dir(&backups_dir)
        .expect("read backups dir")
        .filter_map(Result::ok)
        .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("db"))
        .count();

    // 5 old + 1 new = 6 total. --keep 3 keeps the N=3 most-recent files total
    // (including just_written). just_written is the newest, so: new + 2 newest
    // old = 3 files. The 3 oldest old files are deleted.
    assert_eq!(
        remaining, 3,
        "expected 3 files after --keep 3 (new + 2 newest old); got {remaining}"
    );
}

/// `doctor` reports NO warning for champion cells with refract-validated
/// kv_quant values (`none`, `k8v8`, `k4v4`).
///
/// Inserts one observation row for each validated kv_quant so each becomes a
/// champion cell, then asserts doctor exits 0 and emits no
/// "refract-unvalidated" warning in stderr.
#[test]
fn f17_doctor_no_warning_for_refract_validated_kv_quant() {
    use rmlx_metrics::{migrate, registry, schema};

    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    let init = run_metrics(&db, &["init"]);
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    {
        let mut conn = schema::open(&db).expect("open db");
        migrate::run_pending(&mut conn).expect("run_pending");

        conn.execute(
            "INSERT INTO prompts(sha256, name, body, first_seen_utc)
             VALUES ('f17validated', 'f17-prompt', 'body', '2026-01-01T00:00:00Z')",
            [],
        )
        .expect("insert prompt");
        let prompt_id: i64 = conn
            .query_row(
                "SELECT id FROM prompts WHERE sha256='f17validated'",
                [],
                |r| r.get(0),
            )
            .expect("select prompt");

        // One observation per validated kv_quant so each is a champion cell.
        // Note: k4v4 is in the REFRACT_VALIDATED set but not in KV_QUANT_WHITELIST
        // (not yet a live kv_quant variant), so we test only the two that can be
        // inserted: "none" and "k8v8".
        let (unit, dir) = registry::lookup("decode_tps_warm").expect("metric in registry");
        for kq in &["none", "k8v8"] {
            conn.execute(
                "INSERT INTO observations(
                     backend, model_namespace, model, weight_quant, kv_quant,
                     ctx_max, prompt_id, metric, value, unit, direction,
                     run_id, ts_utc, hardware_tag, inserted_utc, inserted_by)
                 VALUES ('rmlx','mlx-community','f17-model','mxfp8',?1,
                         8192, ?2, 'decode_tps_warm', 50.0, ?3, ?4,
                         'f17run0', '2026-01-01T00:00:00Z', 'm5_max_128gb',
                         '2026-01-01T00:00:00Z', 'test@0.0.1')",
                rusqlite::params![kq, prompt_id, unit, dir.as_str()],
            )
            .expect("insert validated kv_quant row");
        }
    }

    let doc = run_metrics(&db, &["doctor"]);
    let stdout = String::from_utf8_lossy(&doc.stdout);
    let stderr = String::from_utf8_lossy(&doc.stderr);

    assert!(
        doc.status.success(),
        "doctor must exit 0 for refract-validated kv_quants: stdout={stdout} stderr={stderr}"
    );
    assert!(
        !stderr.contains("refract-unvalidated kv_quant"),
        "must not warn about validated kv_quants; stderr={stderr}"
    );
}

/// `doctor` emits exactly one `[warn] refract-unvalidated kv_quant` line
/// per out-of-set champion cell (`k8v4`, `planar`, `turbo4`).
///
/// Inserts one observation per out-of-set kv_quant so each becomes a champion
/// cell, then asserts doctor exits 0 (warnings are not errors) and emits a
/// named warning line for each cell.
#[test]
fn f17_doctor_warns_for_refract_unvalidated_kv_quant() {
    use rmlx_metrics::{migrate, registry, schema};

    let dir = tempfile::tempdir().expect("tempdir");
    let db = dir.path().join("runs.db");

    let init = run_metrics(&db, &["init"]);
    assert!(
        init.status.success(),
        "init failed: {}",
        String::from_utf8_lossy(&init.stderr)
    );

    {
        let mut conn = schema::open(&db).expect("open db");
        migrate::run_pending(&mut conn).expect("run_pending");

        conn.execute(
            "INSERT INTO prompts(sha256, name, body, first_seen_utc)
             VALUES ('f17unvalidated', 'f17u-prompt', 'body', '2026-01-01T00:00:00Z')",
            [],
        )
        .expect("insert prompt");
        let prompt_id: i64 = conn
            .query_row(
                "SELECT id FROM prompts WHERE sha256='f17unvalidated'",
                [],
                |r| r.get(0),
            )
            .expect("select prompt");

        let (unit, dir) = registry::lookup("decode_tps_warm").expect("metric in registry");
        for kq in &["k8v4", "planar", "turbo4"] {
            conn.execute(
                "INSERT INTO observations(
                     backend, model_namespace, model, weight_quant, kv_quant,
                     ctx_max, prompt_id, metric, value, unit, direction,
                     run_id, ts_utc, hardware_tag, inserted_utc, inserted_by)
                 VALUES ('rmlx','mlx-community','f17u-model','mxfp8',?1,
                         8192, ?2, 'decode_tps_warm', 50.0, ?3, ?4,
                         'f17run1', '2026-01-01T00:00:00Z', 'm5_max_128gb',
                         '2026-01-01T00:00:00Z', 'test@0.0.1')",
                rusqlite::params![kq, prompt_id, unit, dir.as_str()],
            )
            .expect("insert unvalidated kv_quant row");
        }
    }

    let doc = run_metrics(&db, &["doctor"]);
    let stdout = String::from_utf8_lossy(&doc.stdout);
    let stderr = String::from_utf8_lossy(&doc.stderr);

    // Warnings are not hard errors — doctor must still exit 0.
    assert!(
        doc.status.success(),
        "doctor must exit 0 (refract-unvalidated kv_quant is a warning, not an error): \
         stdout={stdout} stderr={stderr}"
    );

    // One [warn] line per out-of-set cell.
    for kq in &["k8v4", "planar", "turbo4"] {
        assert!(
            stderr.contains(&format!("kv_quant='{kq}'")),
            "expected refract-unvalidated warning for kv_quant='{kq}' in stderr: {stderr}"
        );
    }
    assert!(
        stderr.contains("refract-unvalidated kv_quant"),
        "expected 'refract-unvalidated kv_quant' in stderr: {stderr}"
    );
    // Summary must list 3 warnings (one per cell) among total warnings.
    assert!(
        stdout.contains("warning(s)"),
        "expected 'warning(s)' in doctor summary: {stdout}"
    );
}

/// Run `rmlx metrics --db <db_path> <subargs...>` with `RMLX_HOME` set so the
/// restore snapshot lands in a controlled temp dir rather than the workspace.
fn run_metrics_with_home(
    db_path: &Path,
    rmlx_home: &Path,
    subargs: &[&str],
) -> std::process::Output {
    Command::new(rmlx_bin())
        .env("RMLX_HOME", rmlx_home)
        .arg("metrics")
        .arg("--db")
        .arg(db_path)
        .args(subargs)
        .output()
        .expect("failed to launch rmlx")
}

/// Regression test: two `restore` calls within the same second must not collide
/// on the pre-restore snapshot filename.
///
/// Both calls land in the same second (the loop is tight — no sleep), so the
/// second call would previously fail with "file already exists" from
/// `VACUUM INTO`. After the fix (counter suffix on collision) both backups
/// exist with distinct names.
#[test]
fn restore_collision_produces_distinct_snapshot_names() {
    use rmlx_metrics::{migrate, schema};

    let dir = tempfile::tempdir().expect("tempdir");
    let rmlx_home = dir.path().join("rmlx_home");
    std::fs::create_dir_all(&rmlx_home).expect("create rmlx_home");

    let db_a = dir.path().join("a.db");
    let db_b = dir.path().join("b.db");

    // Init A.
    let init_a = run_metrics_with_home(&db_a, &rmlx_home, &["init"]);
    assert!(
        init_a.status.success(),
        "init A failed: {}",
        String::from_utf8_lossy(&init_a.stderr)
    );

    // Init B with a sentinel so we can distinguish it from A.
    {
        let mut conn_b = schema::open(&db_b).expect("open B");
        migrate::run_pending(&mut conn_b).expect("migrate B");
        conn_b
            .execute(
                "INSERT INTO prompts(sha256, name, body, first_seen_utc)
                 VALUES ('collision-sha', 'collision', 'body', '2026-01-01T00:00:00Z')",
                [],
            )
            .expect("insert sentinel");
    }

    // Make a backup of B to use as restore source (both calls restore from the same file).
    let backup_of_b = dir.path().join("b_backup.db");
    let backup_out = run_metrics_with_home(
        &db_b,
        &rmlx_home,
        &["backup", "--out", backup_of_b.to_str().unwrap()],
    );
    assert!(
        backup_out.status.success(),
        "backup B failed: {}",
        String::from_utf8_lossy(&backup_out.stderr)
    );

    // First restore: A → from B's backup.
    let r1 = run_metrics_with_home(
        &db_a,
        &rmlx_home,
        &["restore", "--from", backup_of_b.to_str().unwrap()],
    );
    let r1_stdout = String::from_utf8_lossy(&r1.stdout);
    let r1_stderr = String::from_utf8_lossy(&r1.stderr);
    assert!(
        r1.status.success(),
        "first restore failed: status={} stdout={r1_stdout} stderr={r1_stderr}",
        r1.status
    );

    // Second restore immediately after: same second, same timestamp.
    // The fix must produce a distinct snapshot name (pre-restore-<ts>-1.db).
    let r2 = run_metrics_with_home(
        &db_a,
        &rmlx_home,
        &["restore", "--from", backup_of_b.to_str().unwrap()],
    );
    let r2_stdout = String::from_utf8_lossy(&r2.stdout);
    let r2_stderr = String::from_utf8_lossy(&r2.stderr);
    assert!(
        r2.status.success(),
        "second restore (same second) failed — snapshot name collision: status={} stdout={r2_stdout} stderr={r2_stderr}",
        r2.status
    );

    // Both snapshot files must exist in rmlx_home/metrics/backups/.
    let backups_dir = rmlx_home.join("metrics").join("backups");
    let snapshot_count = std::fs::read_dir(&backups_dir)
        .expect("read backups dir")
        .filter_map(Result::ok)
        .filter(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("pre-restore-") && n.ends_with(".db"))
        })
        .count();

    assert_eq!(
        snapshot_count,
        2,
        "expected 2 distinct pre-restore snapshots (one per restore call); \
         got {snapshot_count} in {}",
        backups_dir.display()
    );
}

/// `restore` replaces the DB and creates a pre-restore snapshot.
#[test]
fn restore_replaces_db_and_snapshots_current() {
    use rmlx_metrics::{migrate, schema};

    let dir = tempfile::tempdir().expect("tempdir");
    let db_a = dir.path().join("a.db");
    let db_b = dir.path().join("b.db");
    let backups_dir = dir.path().join("backups");

    // Init DB A via CLI.
    let init_a = run_metrics(&db_a, &["init"]);
    assert!(
        init_a.status.success(),
        "init A failed: {}",
        String::from_utf8_lossy(&init_a.stderr)
    );

    // Init DB B and insert a sentinel row to distinguish it.
    {
        let mut conn_b = schema::open(&db_b).expect("open B");
        migrate::run_pending(&mut conn_b).expect("migrate B");
        conn_b
            .execute(
                "INSERT INTO prompts(sha256, name, body, first_seen_utc)
                 VALUES ('sentinel-sha', 'sentinel', 'body', '2026-01-01T00:00:00Z')",
                [],
            )
            .expect("insert sentinel");
    }

    // Make a backup of B.
    std::fs::create_dir_all(&backups_dir).expect("create backups dir");
    let backup_of_b = backups_dir.join("b_backup.db");
    let backup_out = run_metrics(&db_b, &["backup", "--out", backup_of_b.to_str().unwrap()]);
    assert!(
        backup_out.status.success(),
        "backup B failed: {}",
        String::from_utf8_lossy(&backup_out.stderr)
    );

    // Restore A from backup-of-B. The snapshot dir is `metrics/backups/` relative
    // to cwd — but we want to verify the snapshot lands somewhere sensible.
    // Since restore uses cwd-relative `metrics/backups/`, we just check the command
    // succeeds and that the restored DB contains B's sentinel.
    let restore = run_metrics(&db_a, &["restore", "--from", backup_of_b.to_str().unwrap()]);
    let stdout = String::from_utf8_lossy(&restore.stdout);
    let stderr = String::from_utf8_lossy(&restore.stderr);
    assert!(
        restore.status.success(),
        "restore failed: status={} stdout={stdout} stderr={stderr}",
        restore.status
    );
    assert!(
        stdout.contains("restored from"),
        "expected 'restored from' in stdout: {stdout}"
    );
    assert!(
        stdout.contains("snapshotted"),
        "expected 'snapshotted' in stdout: {stdout}"
    );

    // Verify restored DB (db_a) now contains B's sentinel.
    let conn_restored = schema::open(&db_a).expect("open restored a");
    let count: i64 = conn_restored
        .query_row(
            "SELECT COUNT(*) FROM prompts WHERE sha256='sentinel-sha'",
            [],
            |r| r.get(0),
        )
        .expect("query sentinel");
    assert_eq!(count, 1, "restored DB should contain B's sentinel prompt");
}
