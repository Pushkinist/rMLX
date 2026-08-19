//! The `bests` view — champion per cell (docs/METRICS_DB.md §3.3).
//!
//! The view ranks `observations` within each cell partition, so whatever the
//! largest `higher_better` value in a partition is becomes that cell's
//! published champion. That is the right rule *for measurements*, and the
//! wrong rule for anything else: a fabricated stand-in — a rate of `0.0` from
//! a run that generated no tokens, a prefill "rate" that is really
//! `prompt_tokens × 1000` — beats every real row it shares a partition with
//! and then publishes into `BENCHMARK_CHAMPIONS.md`.
//!
//! So the view only ranks rows the §4 registry can call a measurement, and the
//! predicate is *generated from that registry* rather than retyped in SQL:
//! one definition of "plausible", enforced at ingest by
//! [`crate::ingest::RunRecord::validate`] and here by the same [`Bounds`] the
//! validator uses. [`ensure`] rebuilds the view whenever the stored definition
//! and the registry disagree, so a bounds change propagates on the next open
//! instead of silently applying to new rows only.
//!
//! [`Bounds`]: crate::registry::Bounds

use std::fmt::Write as _;

use rusqlite::Connection;

use crate::error::Result;
use crate::registry;

/// Renders the §4.1 plausibility predicate as a SQLite boolean expression over
/// `column`, keyed on the row's `metric`.
///
/// Every consumer that ranks or aggregates `observations` must `AND` this in.
/// `bests` does it once, but `deltas`, `regress` and `timeseries` each run
/// their own SQL over the base table — and a gate that ranks the rows the
/// champion view refuses is the drift this predicate exists to prevent.
///
/// No parameterization: every fragment comes from the compile-time registry.
pub fn plausible_sql(column: &str) -> String {
    let mut sql = String::from("CASE metric\n");
    for (name, _, _, bounds) in registry::METRICS {
        // write! to a String is infallible — the unit Ok is discarded.
        let _ = writeln!(
            sql,
            "                WHEN '{name}' THEN ({})",
            bounds.sql(column)
        );
    }
    // A metric with no registry entry cannot be bounded, only reported —
    // `rmlx metrics doctor` fails on it under the metric-whitelist check.
    sql.push_str("                ELSE 1\n            END");
    sql
}

/// Renders the `CREATE VIEW` statement for `bests` from the §4 registry.
pub fn create_sql() -> String {
    let plausible = plausible_sql("value");

    format!(
        "CREATE VIEW bests AS
WITH ranked AS (
    SELECT
        o.*,
        ROW_NUMBER() OVER (
            PARTITION BY backend, model_namespace, model, weight_quant, kv_quant,
                         ctx_max, prompt_id, metric
            ORDER BY
                CASE WHEN direction = 'higher_better' THEN  value END DESC,
                CASE WHEN direction = 'lower_better'  THEN -value END DESC,
                ts_utc DESC
        ) AS rn
    FROM observations o
    WHERE {plausible}
)
SELECT * FROM ranked WHERE rn = 1"
    )
}

/// Reads the stored `CREATE VIEW` text for `bests`, if the view exists.
///
/// `None` means *absent*, and nothing else: a locked, damaged or unreadable DB
/// propagates its error rather than being reported as "stale" and then dropped.
fn stored_sql(conn: &Connection) -> Result<Option<String>> {
    match conn.query_row(
        "SELECT sql FROM sqlite_master WHERE type = 'view' AND name = 'bests'",
        [],
        |r| r.get::<_, String>(0),
    ) {
        Ok(sql) => Ok(Some(sql)),
        Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
        Err(e) => Err(e.into()),
    }
}

/// Whether the stored `bests` definition differs from the registry's.
///
/// A read command calls this instead of [`ensure`]: it must know the view is
/// current, and must not be the thing that changes it.
pub fn is_stale(conn: &Connection) -> Result<bool> {
    Ok(stored_sql(conn)?.as_deref() != Some(create_sql().as_str()))
}

/// Recreates `bests` when the stored definition does not match [`create_sql`].
///
/// Returns `true` when the view was rebuilt. A view holds no data, so dropping
/// and recreating it mutates no observation — the append-only contract is
/// untouched.
pub fn ensure(conn: &Connection) -> Result<bool> {
    let want = create_sql();

    if stored_sql(conn)?.as_deref() == Some(want.as_str()) {
        return Ok(false);
    }

    conn.execute_batch(&format!("DROP VIEW IF EXISTS bests;\n{want};"))?;
    Ok(true)
}

#[cfg(test)]
#[path = "bests_view_tests.rs"]
mod tests;
