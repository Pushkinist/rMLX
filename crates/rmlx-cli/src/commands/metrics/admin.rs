// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rmlx_metrics::{bests_view, identity, migrate, registry, schema};
use rusqlite::params;

// ---------------------------------------------------------------------------
// init
// ---------------------------------------------------------------------------

pub(super) fn cmd_init(db_path: &Path) -> anyhow::Result<()> {
    if db_path.exists() {
        anyhow::bail!(
            "metrics DB already exists at {}; use 'doctor' to migrate schema or delete the file first",
            db_path.display()
        );
    }

    let mut conn =
        schema::open(db_path).with_context(|| format!("open DB at {}", db_path.display()))?;

    let applied = migrate::run_pending(&mut conn).with_context(|| "run_pending migrations")?;

    println!("applied {applied} migration(s)");
    println!("DB initialized at {}", db_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// doctor
// ---------------------------------------------------------------------------

pub(super) fn cmd_doctor(db_path: &Path, fix: bool) -> anyhow::Result<()> {
    let mut conn =
        schema::open(db_path).with_context(|| format!("open DB at {}", db_path.display()))?;

    let mut errors: u32 = 0;
    let mut warnings: u32 = 0;

    // ── Check 1: integrity_check ──────────────────────────────────────────────
    {
        let result: String = conn
            .query_row("PRAGMA integrity_check", [], |r| r.get(0))
            .context("PRAGMA integrity_check")?;
        if result == "ok" {
            println!("[ok] integrity_check: ok");
        } else {
            eprintln!("[FAIL] integrity_check returned: {result}");
            errors += 1;
        }
    }

    // ── Check 2: foreign_key_check ────────────────────────────────────────────
    {
        let mut stmt = conn
            .prepare("PRAGMA foreign_key_check")
            .context("prepare foreign_key_check")?;
        let rows: Vec<String> = stmt
            .query_map([], |r| {
                let table: String = r.get(0)?;
                let rowid: i64 = r.get(1)?;
                Ok(format!("{table}:{rowid}"))
            })
            .context("query foreign_key_check")?
            .collect::<Result<_, _>>()
            .context("collect foreign_key_check")?;

        if rows.is_empty() {
            println!("[ok] foreign_key_check: no violations");
        } else {
            eprintln!("[FAIL] foreign_key_check violations: {}", rows.join(", "));
            errors += 1;
        }
    }

    // ── Check 3: schema version / pending migrations ───────────────────────────
    {
        let user_version: u32 = conn
            .query_row("PRAGMA user_version", [], |r| r.get(0))
            .context("PRAGMA user_version")?;
        let latest = schema::MIGRATIONS.len() as u32;

        if user_version < latest {
            println!(
                "[migrate] schema at v{user_version}, latest v{latest} — applying pending migrations"
            );
            let applied = migrate::run_pending(&mut conn).context("run_pending migrations")?;
            println!("[ok] applied {applied} migration(s); schema now at v{latest}");
        } else {
            println!("[ok] schema version: v{user_version} (current)");
        }
    }

    // ── Check 3b: bests view matches the registry ─────────────────────────────
    //
    // The view is generated from the §4 registry, not pinned to a migration
    // number, so a DB already at the latest schema version can still carry a
    // definition built from an older registry — including one with no
    // plausibility filter at all. Checking `user_version` cannot see that.
    {
        if bests_view::ensure(&conn).context("ensure bests view")? {
            println!("[fix] bests view: rebuilt from the §4 metric registry");
        } else {
            println!("[ok] bests view: definition matches the §4 metric registry");
        }
    }

    // ── Check 4: whitelist sweep ──────────────────────────────────────────────
    {
        let checks: &[(&str, &str, &[&str])] = &[
            ("backend", "backend", identity::BACKEND_WHITELIST),
            (
                "model_namespace",
                "model_namespace",
                identity::NAMESPACE_WHITELIST,
            ),
            (
                "weight_quant",
                "weight_quant",
                identity::WEIGHT_QUANT_WHITELIST,
            ),
        ];

        for (col, field, whitelist) in checks {
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT DISTINCT {col}, id FROM observations WHERE {col} NOT IN ({placeholders})",
                    placeholders = whitelist
                        .iter()
                        .map(|v| format!("'{v}'"))
                        .collect::<Vec<_>>()
                        .join(","),
                ))
                .with_context(|| format!("prepare whitelist sweep for {col}"))?;

            let bad: Vec<(String, i64)> = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .with_context(|| format!("query whitelist sweep for {col}"))?
                .collect::<Result<_, _>>()
                .with_context(|| format!("collect whitelist sweep for {col}"))?;

            if bad.is_empty() {
                println!("[ok] whitelist {field}: all values recognized");
            } else {
                for (val, pk) in &bad {
                    eprintln!(
                        "[FAIL] whitelist {field}: unknown value '{val}' (observation id={pk})"
                    );
                }
                errors += 1;
            }
        }

        // kv_quant uses parser-based validation: the canonical form
        // includes the long `mixed_k<kb>g<kg>_v<vb>g<vg>` shape which cannot
        // fit a fixed IN-list. Pull the distinct values and validate each.
        {
            let mut stmt = conn
                .prepare("SELECT DISTINCT kv_quant, MIN(id) FROM observations GROUP BY kv_quant")
                .context("prepare kv_quant validation sweep")?;
            let rows: Vec<(String, i64)> = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .context("query kv_quant validation sweep")?
                .collect::<Result<_, _>>()
                .context("collect kv_quant validation sweep")?;
            let mut any_bad = false;
            for (val, pk) in &rows {
                if identity::canonicalize_kv_quant(val).is_err() {
                    eprintln!(
                        "[FAIL] whitelist kv_quant: unknown value '{val}' (observation id={pk})"
                    );
                    any_bad = true;
                }
            }
            if any_bad {
                errors += 1;
            } else {
                println!("[ok] whitelist kv_quant: all values recognized");
            }
        }

        // Metric column whitelist check
        {
            let metric_names: Vec<&str> = registry::METRICS.iter().map(|(n, _, _, _)| *n).collect();
            let placeholders = metric_names
                .iter()
                .map(|v| format!("'{v}'"))
                .collect::<Vec<_>>()
                .join(",");
            let mut stmt = conn
                .prepare(&format!(
                    "SELECT DISTINCT metric, id FROM observations WHERE metric NOT IN ({placeholders})"
                ))
                .context("prepare metric whitelist sweep")?;

            let bad: Vec<(String, i64)> = stmt
                .query_map([], |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)))
                .context("query metric whitelist sweep")?
                .collect::<Result<_, _>>()
                .context("collect metric whitelist sweep")?;

            if bad.is_empty() {
                println!("[ok] whitelist metric: all values recognized");
            } else {
                for (val, pk) in &bad {
                    eprintln!(
                        "[FAIL] whitelist metric: unknown value '{val}' (observation id={pk})"
                    );
                }
                errors += 1;
            }
        }
    }

    // ── Check 5: direction sanity ──────────────────────────────────────────────
    {
        let mut stmt = conn
            .prepare(
                "SELECT metric, direction, COUNT(*) FROM observations GROUP BY metric, direction",
            )
            .context("prepare direction check")?;

        let rows: Vec<(String, String, i64)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .context("query direction check")?
            .collect::<Result<_, _>>()
            .context("collect direction check")?;

        for (metric, direction, count) in &rows {
            if let Ok((_, expected_dir)) = registry::lookup(metric) {
                let expected_str = expected_dir.as_str();
                if direction != expected_str {
                    eprintln!(
                        "[FAIL] direction mismatch for metric '{metric}': got '{direction}', expected '{expected_str}' ({count} row(s))"
                    );
                    if fix {
                        conn.execute(
                            "UPDATE observations SET direction = ?1 WHERE metric = ?2 AND direction = ?3",
                            params![expected_str, metric, direction],
                        )
                        .with_context(|| format!("fix direction for metric '{metric}'"))?;
                        println!("[fix] corrected direction for '{metric}' ({count} row(s))");
                    }
                    errors += 1;
                }
            }
        }
        if errors == 0 || rows.is_empty() {
            println!("[ok] direction sanity: all directions match registry");
        }
    }

    // ── Check 6: unit sanity ──────────────────────────────────────────────────
    {
        let mut stmt = conn
            .prepare("SELECT metric, unit, COUNT(*) FROM observations GROUP BY metric, unit")
            .context("prepare unit check")?;

        let rows: Vec<(String, String, i64)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .context("query unit check")?
            .collect::<Result<_, _>>()
            .context("collect unit check")?;

        let mut unit_errors: u32 = 0;
        for (metric, unit, count) in &rows {
            if let Ok((expected_unit, _)) = registry::lookup(metric) {
                if unit != expected_unit {
                    eprintln!(
                        "[FAIL] unit mismatch for metric '{metric}': got '{unit}', expected '{expected_unit}' ({count} row(s))"
                    );
                    if fix {
                        conn.execute(
                            "UPDATE observations SET unit = ?1 WHERE metric = ?2 AND unit = ?3",
                            params![expected_unit, metric, unit],
                        )
                        .with_context(|| format!("fix unit for metric '{metric}'"))?;
                        println!("[fix] corrected unit for '{metric}' ({count} row(s))");
                    }
                    unit_errors += 1;
                }
            }
        }
        if unit_errors == 0 {
            println!("[ok] unit sanity: all units match registry");
        } else {
            errors += unit_errors;
        }
    }

    // ── Check 6b: value plausibility ──────────────────────────────────────────
    //
    // Unit sanity above compares the unit *label* against the registry; it
    // cannot see a number that is not in that unit at all. This check compares
    // the number against the registry's plausible window: a rate of exactly
    // 0.0 (a missing field ingested as a measurement) or a value orders of
    // magnitude past the hardware (an arithmetic accident) is reported here.
    //
    // A warning, not an error, and deliberately so. `observations` is
    // append-only: the rows this finds predate the ingest gate and cannot be
    // deleted or corrected — the true value is not recoverable from the row,
    // only re-measurable. An error here would make every `make ci` on a
    // machine with history permanently red, which is not a gate but a stuck
    // light. The gate that can actually fail is `RunRecord::validate`, which
    // refuses such a value at the door; this check is the report of what it
    // would now refuse, and the `bests` view already excludes the rows.
    {
        let mut plausibility_warnings: u32 = 0;
        for (metric, _, _, bounds) in registry::METRICS {
            let (bad, min_bad, max_bad, first_id): (i64, Option<f64>, Option<f64>, Option<i64>) =
                conn.query_row(
                    &format!(
                        "SELECT COUNT(*), MIN(value), MAX(value), MIN(id)
                           FROM observations
                          WHERE metric = ?1 AND NOT ({})",
                        bounds.sql("value")
                    ),
                    params![metric],
                    |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
                )
                .with_context(|| format!("value-plausibility query for metric '{metric}'"))?;

            if bad > 0 {
                let (lo, hi, id) = (
                    min_bad.unwrap_or(f64::NAN),
                    max_bad.unwrap_or(f64::NAN),
                    first_id.unwrap_or(-1),
                );
                eprintln!(
                    "[WARN] value plausibility: metric '{metric}' has {bad} row(s) outside {} \
                     (range {lo} .. {hi}, first observation id={id})",
                    bounds.describe()
                );
                plausibility_warnings += 1;
            }
        }
        if plausibility_warnings == 0 {
            println!("[ok] value plausibility: all values inside registry bounds");
        } else {
            println!(
                "[warn] {plausibility_warnings} metric(s) carry implausible values — historical \
                 rows, excluded from the `bests` view, refused by ingest today; re-measure, \
                 do not repair"
            );
            warnings += plausibility_warnings;
        }
    }

    // ── Check 7: stale-prompt warn ────────────────────────────────────────────
    {
        // Load SHA256 of all current prompt files from rMLX/prompts/*.json
        let prompt_dir = PathBuf::from("prompts");
        let mut current_sha256s: std::collections::HashSet<String> =
            std::collections::HashSet::new();

        if prompt_dir.is_dir() {
            for entry in std::fs::read_dir(&prompt_dir)
                .with_context(|| format!("read dir {}", prompt_dir.display()))?
            {
                let entry = entry.context("read dir entry")?;
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) == Some("json") {
                    let bytes = std::fs::read(&path)
                        .with_context(|| format!("read prompt file {}", path.display()))?;
                    if let Some(digest) = prompt_file_sha256(&bytes) {
                        current_sha256s.insert(digest);
                    }
                }
            }
        }

        // Find prompts referenced from observations whose sha256 isn't in current files.
        let mut stmt = conn
            .prepare(
                "SELECT p.id, p.name, p.sha256
                   FROM prompts p
                  WHERE p.id IN (SELECT DISTINCT prompt_id FROM observations)",
            )
            .context("prepare stale-prompt check")?;

        let rows: Vec<(i64, String, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))
            .context("query stale-prompt check")?
            .collect::<Result<_, _>>()
            .context("collect stale-prompt check")?;

        let mut stale_count = 0u32;
        for (id, name, sha) in &rows {
            if !current_sha256s.contains(sha.as_str()) {
                eprintln!(
                    "[WARN] stale prompt: id={id} name='{name}' sha256={sha} not found in prompts/*.json"
                );
                stale_count += 1;
            }
        }

        if stale_count == 0 {
            println!("[ok] stale-prompt check: no stale prompts detected");
        } else {
            println!("[warn] {stale_count} stale prompt(s) referenced from observations");
            warnings += stale_count;
        }
    }

    // ── Check 8: COVERAGE_MATRIX gap check ───────────────────────────────────
    //
    // For each (backend, metric) pair in COVERAGE_MATRIX with Coverage::Yes,
    // verify the `observations` table has at least one row for that (backend,
    // metric) combination. A missing row is informational — it means the
    // metric is expected but has never been recorded. Reported as [warn] to
    // distinguish from hard structural errors above.
    {
        use rmlx_metrics::registry::{Coverage, COVERAGE_MATRIX};

        let mut gap_count: u32 = 0;
        for (backend, metric, coverage) in COVERAGE_MATRIX {
            if *coverage != Coverage::Yes {
                continue;
            }
            let row_count: i64 = conn
                .query_row(
                    "SELECT COUNT(*) FROM observations WHERE backend = ?1 AND metric = ?2",
                    rusqlite::params![backend, metric],
                    |r| r.get(0),
                )
                .with_context(|| {
                    format!("coverage-gap query for backend='{backend}' metric='{metric}'")
                })?;

            if row_count == 0 {
                eprintln!(
                    "[warn] coverage gap: backend='{backend}' metric='{metric}' expected (Coverage::Yes) but 0 rows in observations"
                );
                gap_count += 1;
            }
        }
        if gap_count == 0 {
            println!("[ok] coverage matrix: all Coverage::Yes metrics have at least one row");
        } else {
            println!("[warn] {gap_count} coverage gap(s) — metrics expected but not yet recorded");
            warnings += gap_count;
        }
    }

    // ── Check 9: refract-unvalidated kv_quant champion warning ───────────────
    //
    // The refract CI gate validates token-fidelity only for the
    // "plain" KV-quant set: none, k8v8, k4v4. Rotation-based and exotic
    // families (k8v4, turbo4, turbo8, planar, …) have no defined PPL
    // semantics under refract — the gate does not cover them. A champion row
    // whose kv_quant falls outside this validated set is NOT wrong, but its
    // fidelity/PPL is unverified by the CI gate. Emit one [warn] line per
    // distinct (model_namespace, model, weight_quant, kv_quant) cell so the
    // operator knows to hand-validate before trusting that cell as a champion.
    //
    // Query path: reuses the same `bests` VIEW that `query::champions` reads.
    // No new SQL schema.
    {
        const REFRACT_VALIDATED: &[&str] = &["none", "k8v8", "k4v4"];

        let placeholders = REFRACT_VALIDATED
            .iter()
            .map(|v| format!("'{v}'"))
            .collect::<Vec<_>>()
            .join(", ");

        let sql = format!(
            "SELECT DISTINCT model_namespace, model, weight_quant, kv_quant
               FROM bests
              WHERE kv_quant NOT IN ({placeholders})
              ORDER BY model_namespace, model, weight_quant, kv_quant"
        );

        let mut stmt = conn
            .prepare(&sql)
            .context("prepare refract-kv-quant check")?;

        let cells: Vec<(String, String, String, String)> = stmt
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)))
            .context("query refract-kv-quant check")?
            .collect::<Result<_, _>>()
            .context("collect refract-kv-quant check")?;

        if cells.is_empty() {
            println!("[ok] refract-kv-quant: all champion cells use refract-validated kv_quant");
        } else {
            for (ns, model, wq, kq) in &cells {
                eprintln!(
                    "[warn] refract-unvalidated kv_quant: model_namespace='{ns}' model='{model}' \
                     weight_quant='{wq}' kv_quant='{kq}' — fidelity/PPL unverified by refract CI; \
                     hand-validate before trusting as champion (see docs/research/F17-refract-fidelity-scope.md)"
                );
            }
            let n = cells.len() as u32;
            println!(
                "[warn] {n} champion cell(s) use refract-unvalidated kv_quant (not an error — hand-validate)"
            );
            warnings += n;
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    println!("\ndoctor summary: {errors} error(s), {warnings} warning(s)");

    if errors > 0 {
        anyhow::bail!("doctor found {errors} error(s)");
    }

    Ok(())
}

/// Compute hex-encoded SHA-256 of the canonical JSON of the `body` field in a
/// prompt file, matching the algorithm used by `rmlx_metrics::ingest::prompt_body_sha256`.
///
/// Returns `None` if the file cannot be parsed as JSON or has no `body` field.
fn prompt_file_sha256(file_bytes: &[u8]) -> Option<String> {
    use sha2::{Digest, Sha256};

    let json: serde_json::Value = serde_json::from_slice(file_bytes).ok()?;
    let body = json.get("body")?;
    let canonical = serde_json::to_vec(body).ok()?;
    let digest = Sha256::digest(&canonical);
    // write!(String) is infallible — let _ discards the unit Ok.
    Some(digest.iter().fold(String::new(), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    }))
}

// ---------------------------------------------------------------------------
// backup
// ---------------------------------------------------------------------------

pub(super) fn cmd_backup(
    db_path: &Path,
    out: Option<PathBuf>,
    keep: Option<usize>,
) -> anyhow::Result<()> {
    let backup_path = if let Some(p) = out {
        p
    } else {
        let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S");
        let backups_dir = rmlx_core::paths::metrics_dir().join("backups");
        std::fs::create_dir_all(&backups_dir).context("create metrics/backups")?;
        backups_dir.join(format!("runs-{ts}.db"))
    };

    if let Some(parent) = backup_path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create backup parent dir {}", parent.display()))?;
        }
    }

    let conn =
        schema::open(db_path).with_context(|| format!("open DB at {}", db_path.display()))?;

    // VACUUM INTO produces a consistent, WAL-checkpointed copy.
    conn.execute_batch(&format!("VACUUM INTO '{}'", backup_path.display()))
        .with_context(|| format!("VACUUM INTO {}", backup_path.display()))?;

    println!("wrote backup to {}", backup_path.display());

    // Prune old backups if --keep N specified.
    // Pruning scans the same directory the backup was written to.
    if let Some(n) = keep {
        let prune_dir = backup_path
            .parent()
            .map_or_else(|| PathBuf::from("."), Path::to_path_buf);
        prune_backups(&prune_dir, n, &backup_path)?;
    }

    Ok(())
}

/// Keep the `n` most-recent backups in `dir` (by mtime), deleting older ones.
/// `just_written` is excluded from pruning regardless of mtime.
fn prune_backups(dir: &Path, keep: usize, just_written: &Path) -> anyhow::Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }

    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = std::fs::read_dir(dir)
        .with_context(|| format!("read backups dir {}", dir.display()))?
        .filter_map(Result::ok)
        .filter(|e| {
            let p = e.path();
            p.extension().and_then(|x| x.to_str()) == Some("db")
                && p.file_name()
                    .and_then(|n| n.to_str())
                    .is_some_and(|n| n.starts_with("runs-"))
        })
        .filter_map(|e| {
            let p = e.path();
            let mtime = e.metadata().ok()?.modified().ok()?;
            Some((mtime, p))
        })
        .collect();

    // Sort newest-first.
    candidates.sort_by_key(|entry| std::cmp::Reverse(entry.0));

    // Delete anything beyond the `keep` limit, skipping just_written.
    let mut kept = 0usize;
    for (_, path) in &candidates {
        // The just-written backup is always counted as kept.
        if path == just_written {
            kept += 1;
            continue;
        }
        if kept < keep {
            kept += 1;
        } else {
            std::fs::remove_file(path)
                .with_context(|| format!("delete old backup {}", path.display()))?;
            println!("pruned old backup {}", path.display());
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// restore
// ---------------------------------------------------------------------------

pub(super) fn cmd_restore(db_path: &Path, from: &Path) -> anyhow::Result<()> {
    if !from.exists() {
        anyhow::bail!("restore source not found: {}", from.display());
    }

    // Snapshot current DB first.
    let ts = chrono::Utc::now().format("%Y%m%d-%H%M%S");
    let snapshot_dir = rmlx_core::paths::metrics_dir().join("backups");
    std::fs::create_dir_all(&snapshot_dir).context("create metrics/backups")?;
    // Use a collision-free path: append a counter suffix when the timestamped
    // name already exists (can happen when two restores land in the same second).
    let snapshot_path = {
        let base = snapshot_dir.join(format!("pre-restore-{ts}.db"));
        if base.exists() {
            let mut n = 1u32;
            loop {
                let candidate = snapshot_dir.join(format!("pre-restore-{ts}-{n}.db"));
                if !candidate.exists() {
                    break candidate;
                }
                n += 1;
            }
        } else {
            base
        }
    };

    if db_path.exists() {
        let conn = schema::open(db_path)
            .with_context(|| format!("open current DB at {}", db_path.display()))?;
        conn.execute_batch(&format!("VACUUM INTO '{}'", snapshot_path.display()))
            .with_context(|| format!("snapshot current DB to {}", snapshot_path.display()))?;
        println!("snapshotted current DB to {}", snapshot_path.display());
    } else {
        println!(
            "no current DB at {} — nothing to snapshot",
            db_path.display()
        );
    }

    // Atomic rename: copy source to tmp, then rename into place.
    let tmp_path = db_path.with_extension("tmp");
    std::fs::copy(from, &tmp_path)
        .with_context(|| format!("copy {} → {}", from.display(), tmp_path.display()))?;
    std::fs::rename(&tmp_path, db_path)
        .with_context(|| format!("rename {} → {}", tmp_path.display(), db_path.display()))?;

    println!(
        "restored from {}; previous DB snapshotted to {}",
        from.display(),
        snapshot_path.display()
    );

    Ok(())
}
