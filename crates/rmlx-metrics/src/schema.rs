//! DB connection factory with mandatory PRAGMAs.
//!
//! Every open path (`open`, `open_memory`, `open_readonly`) applies the same
//! PRAGMA set from docs/METRICS_DB.md §8.2 / §10.5.

use rusqlite::{Connection, OpenFlags};

use crate::error::{Error, Result};

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
    (
        5,
        include_str!("migrations/005_observations_decode_config.sql"),
    ),
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

/// Open the DB at `path` for a command that *writes*, bringing the schema up
/// to date first (including the registry-generated `bests` view).
///
/// Only for writers. A read command must not migrate — see [`open_checked`].
pub fn open_migrated(path: &std::path::Path) -> Result<Connection> {
    let mut conn = open(path)?;
    crate::migrate::run_pending(&mut conn)?;
    Ok(conn)
}

/// Open an existing DB at `path` for a command that only reads, refusing to
/// run if what it would read is stale.
///
/// Two things a read command must not do, both of which `open` alone allows:
///
/// * **Create anything.** `Connection::open` creates an empty file for a
///   mistyped path, and migrating it would then hand back a fully-formed,
///   empty metrics DB instead of an error. This refuses a path that does not
///   already exist.
/// * **Migrate.** `run_pending` ends in a `DROP VIEW` / `CREATE VIEW`, which
///   would change what `bests` — and therefore `BENCHMARK_CHAMPIONS.md` —
///   publishes, as a side effect of a query. `<RMLX_HOME>/metrics/legacy/` is
///   declared read-only archive material and `backups/` holds snapshots; both
///   are reachable through `--db`.
///
/// So staleness is reported instead of silently repaired: if the stored
/// `bests` definition does not match the §4 registry, the caller is told to
/// run `rmlx metrics doctor --fix`.
///
/// The connection is opened read-write rather than with `open_readonly` on
/// purpose. A read-only connection to a DB with a non-empty `-wal` must create
/// the `-shm` file, so it fails outright ("unable to open database file") when
/// the containing directory is not writable — trading a mutation hazard for an
/// availability one. Nothing on this path issues a write.
pub fn open_checked(path: &std::path::Path) -> Result<Connection> {
    if !path.exists() {
        return Err(Error::Schema(format!(
            "no metrics DB at {} — check --db / RMLX_METRICS_DB, or run `rmlx metrics init`",
            path.display()
        )));
    }

    let conn = open(path)?;

    if crate::bests_view::is_stale(&conn)? {
        return Err(Error::Schema(format!(
            "the `bests` view in {} was built from a different metric registry than this \
             binary's, so a champion read here would not match §4.1 — run \
             `rmlx metrics doctor --fix` to rebuild it",
            path.display()
        )));
    }

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
