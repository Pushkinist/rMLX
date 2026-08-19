//! DB connection factory with mandatory PRAGMAs.
//!
//! Every open path (`open`, `open_memory`, `open_readonly`) applies the same
//! PRAGMA set from docs/METRICS_DB.md §8.2 / §10.5.

use rusqlite::{Connection, OpenFlags};

use crate::error::Result;

// ---------------------------------------------------------------------------
// Embedded migrations (binary-shipped; no filesystem access at runtime).
// ---------------------------------------------------------------------------

/// All schema migrations in ascending version order.
/// `run_pending` in `migrate.rs` walks this slice.
pub static MIGRATIONS: &[(u32, &str)] = &[
    (1, include_str!("migrations/001_init.sql")),
    (2, include_str!("migrations/002_events.sql")),
    (3, include_str!("migrations/003_events_identity.sql")),
    (4, include_str!("migrations/004_events_mlx_nax.sql")),
];

// ---------------------------------------------------------------------------
// Connection helpers
// ---------------------------------------------------------------------------

/// Apply the mandatory connection-level PRAGMAs.
///
/// `journal_mode=WAL` persists in the DB header; the others are per-connection
/// and must be re-applied on every open (SQLite resets them on close).
fn apply_pragmas(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        "PRAGMA journal_mode = WAL;
         PRAGMA synchronous  = NORMAL;
         PRAGMA foreign_keys = ON;
         PRAGMA busy_timeout = 5000;",
    )?;
    Ok(())
}

/// Open (or create) the DB at `path` for read-write access.
///
/// Returns a connection with all mandatory PRAGMAs applied.
/// Use `migrate::run_pending` immediately after to bring the schema up to date.
pub fn open(path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open(path)?;
    apply_pragmas(&conn)?;
    Ok(conn)
}

/// Open (or create) the DB at `path` read-write with the schema brought up to
/// date, including the registry-generated `bests` view.
///
/// This is what a command that *reads* should use. `open` alone leaves the
/// caller reading whatever schema the last writer happened to leave behind —
/// a query command run against a DB older than the binary silently reads a
/// stale `bests` definition, which is how a champion view keeps publishing
/// rows the current registry rejects.
pub fn open_migrated(path: &std::path::Path) -> Result<Connection> {
    let mut conn = open(path)?;
    crate::migrate::run_pending(&mut conn)?;
    Ok(conn)
}

/// Open an in-memory DB for tests.
///
/// `:memory:` does not persist WAL to disk (journal_mode stays `memory`),
/// but all other PRAGMAs apply normally.
pub fn open_memory() -> Result<Connection> {
    let conn = Connection::open_in_memory()?;
    apply_pragmas(&conn)?;
    Ok(conn)
}

/// Open an existing DB at `path` in read-only mode.
///
/// Used by CLI read commands (`best`, `rank`, `query`, `export`) so they
/// cannot accidentally mutate the DB while a writer is active.
pub fn open_readonly(path: &std::path::Path) -> Result<Connection> {
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)?;
    apply_pragmas(&conn)?;
    Ok(conn)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "schema_tests.rs"]
mod tests;
