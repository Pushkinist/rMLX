//! Schema migration runner.
//!
//! Applies pending SQL migrations from [`crate::schema::MIGRATIONS`] to a
//! live SQLite connection. One migration per transaction; `user_version`
//! PRAGMA tracks the applied version in the DB header.

use rusqlite::Connection;

use crate::{bests_view, error::Result, schema::MIGRATIONS, time_util::now_iso8601};

/// Apply all pending migrations from [`MIGRATIONS`] to `conn`.
///
/// Reads `PRAGMA user_version` to find the current schema version, then runs
/// every migration whose target version exceeds it, in order. Each migration
/// executes inside its own transaction. After migration 001 is applied, the
/// `schema_meta` seed rows are inserted (idempotent via `INSERT OR IGNORE`).
///
/// Finally, [`bests_view::ensure`] brings the `bests` view in line with the §4
/// registry — see that module for why the view is generated rather than
/// pinned to a migration.
///
/// Returns the number of migrations applied (0 when already up to date).
pub fn run_pending(conn: &mut Connection) -> Result<u32> {
    let current_version: u32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;

    let mut applied: u32 = 0;

    for &(target_version, sql) in MIGRATIONS {
        if target_version <= current_version {
            continue;
        }

        // Each migration runs in its own transaction.
        let tx = conn.transaction()?;
        tx.execute_batch(sql)?;
        // Mirror schema_meta.schema_version in the SQLite header.
        tx.execute_batch(&format!("PRAGMA user_version = {target_version};"))?;
        tx.commit()?;

        // Post-migration hooks.
        if target_version == 1 {
            seed_schema_meta(conn)?;
        }
        if target_version == 6 {
            backfill_decode_config(conn)?;
        }
        if target_version == 7 {
            null_default_decode_config(conn)?;
        }

        applied += 1;
    }

    // The `bests` view is generated from the §4 registry, not pinned to a
    // migration number: a bounds change has to reach existing DBs too, and a
    // view carries no data to migrate. Cheap no-op when already current.
    bests_view::ensure(conn)?;

    Ok(applied)
}

/// Fill `decode_config` for rows whose own `notes` say what they were.
///
/// Migration 005 added the column and left every existing row NULL, which is
/// what ordinary decode carries — so a speculative row written before the
/// column kept sharing a cell with the plain row it should rank apart from, and
/// kept winning it. The bench scripts recorded the drafter in `notes` long
/// before there was a column for it, so most of those rows can say what they
/// were.
///
/// This writes no measurement: it classifies a row from that row's own fields
/// into a column that was NULL for want of existing. `notes` that say nothing
/// either way stay NULL and are named in docs/METRICS_DB.md.
///
/// The rule itself is [`crate::cell::decode_config_from_notes`] — one parser,
/// not a second spelling in SQL.
///
/// Returns how many rows were classified as speculative.
fn backfill_decode_config(conn: &Connection) -> Result<usize> {
    use crate::cell::{decode_config_from_notes, NotesVerdict};

    let pending: Vec<(i64, String)> = {
        let mut stmt = conn.prepare(
            "SELECT id, notes FROM observations
             WHERE decode_config IS NULL AND notes IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get::<_, String>(1)?)))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };

    let mut filled = 0_usize;
    for (id, notes) in pending {
        if let NotesVerdict::Speculative(config) = decode_config_from_notes(&notes) {
            conn.execute(
                "UPDATE observations SET decode_config = ?1 WHERE id = ?2",
                rusqlite::params![config, id],
            )?;
            filled += 1;
        }
    }

    if filled > 0 {
        tracing::info!(
            rows = filled,
            "migrate: classified pre-existing rows by the drafter their notes name"
        );
    }

    Ok(filled)
}

/// Replace a `decode_config` that spells the engine's own defaults with NULL.
///
/// §3.2 makes `NULL` the engine at its defaults, so a row spelling them out is
/// a second spelling of one configuration — and two spellings are two cells
/// that never rank against each other. `RunRecord::validate` refuses such a
/// record now; this brings the rows written before it did into the same cell
/// they always belonged in.
///
/// Writes no measurement: it rewrites how a row says the engine was configured,
/// on rows whose configuration was the default. The predicate is
/// [`crate::cell::decode_config_is_all_defaults`], reading the engine's own
/// constants — not a SQL literal, which would be a second copy of the values
/// this column exists to keep honest.
///
/// Returns how many rows were moved to the default cell.
fn null_default_decode_config(conn: &Connection) -> Result<usize> {
    let spellings: Vec<String> = {
        let mut stmt = conn.prepare(
            "SELECT DISTINCT decode_config FROM observations WHERE decode_config IS NOT NULL",
        )?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };

    let mut moved = 0_usize;
    for spelling in spellings {
        if !crate::cell::decode_config_is_all_defaults(&spelling) {
            continue;
        }
        let n = conn.execute(
            "UPDATE observations SET decode_config = NULL WHERE decode_config = ?1",
            rusqlite::params![spelling],
        )?;
        tracing::info!(
            rows = n,
            %spelling,
            "migrate: decode_config spelled the engine defaults; moved to the default cell"
        );
        moved += n;
    }
    Ok(moved)
}

/// Insert the well-known `schema_meta` seed rows.
///
/// Uses `INSERT OR IGNORE` so re-running (e.g. on a DB that already has the
/// rows from a previous init) is a no-op rather than an error.
fn seed_schema_meta(conn: &Connection) -> Result<()> {
    let now = now_iso8601()?;
    let created_by = format!("rmlx-metrics@{}", env!("CARGO_PKG_VERSION"));

    conn.execute_batch(&format!(
        "INSERT OR IGNORE INTO schema_meta(key, value) VALUES
             ('schema_version',    '1'),
             ('created_utc',       '{now}'),
             ('created_by',        '{created_by}'),
             ('hardware_tag',      'm5_max_128gb'),
             ('default_namespace', 'mlx-community');",
    ))?;

    Ok(())
}

#[cfg(test)]
#[path = "schema_runner_tests.rs"]
mod tests;
