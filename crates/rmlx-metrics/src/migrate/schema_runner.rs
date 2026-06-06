//! Schema migration runner.
//!
//! Applies pending SQL migrations from [`crate::schema::MIGRATIONS`] to a
//! live SQLite connection. One migration per transaction; `user_version`
//! PRAGMA tracks the applied version in the DB header.

use rusqlite::Connection;

use crate::{error::Result, schema::MIGRATIONS, time_util::now_iso8601};

/// Apply all pending migrations from [`MIGRATIONS`] to `conn`.
///
/// Reads `PRAGMA user_version` to find the current schema version, then runs
/// every migration whose target version exceeds it, in order. Each migration
/// executes inside its own transaction. After migration 001 is applied, the
/// `schema_meta` seed rows are inserted (idempotent via `INSERT OR IGNORE`).
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

        applied += 1;
    }

    Ok(applied)
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
