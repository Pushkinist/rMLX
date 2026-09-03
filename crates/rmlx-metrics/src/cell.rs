//! What identifies a metrics cell — in one place, for everything that keys on it.
//!
//! A cell is the set of rows that are alternative measurements of the same
//! thing, so the champion rule ("largest `higher_better` wins") only means
//! something within one. Adding a column to that key is therefore not a change
//! to `bests` alone: `rmlx metrics best`, `compare`, `history`, `timeseries`
//! and the `deltas --exit-code` gate all select or group by the same key, and
//! any one of them left behind reads two configurations as one and reports the
//! larger. `decode_config` was added to the view first and to none of them, and
//! nothing failed — which is why the key lives here now and every consumer
//! renders its `WHERE` from [`predicate`] rather than retyping the columns.
//!
//! [`crate::query::read::cell_keyed_sql`] enumerates every SQL body built from
//! this module so a test can assert the columns actually reached it.

use std::fmt::Write as _;

/// One column of the cell key.
#[derive(Debug, Clone, Copy)]
#[allow(
    clippy::exhaustive_structs,
    reason = "two fields describing a column name and its nullability; a third would be a different abstraction"
)]
pub struct CellColumn {
    /// Column name in `observations` / `bests`.
    pub name: &'static str,
    /// Whether the column admits NULL, which decides `IS` versus `=`.
    pub nullable: bool,
}

/// The columns that identify a cell, in schema order. `metric` is not one of
/// them — it selects *which* measurement, not which configuration — but every
/// partition and lookup pairs this list with it.
pub const CELL_COLUMNS: &[CellColumn] = &[
    CellColumn {
        name: "backend",
        nullable: false,
    },
    CellColumn {
        name: "model_namespace",
        nullable: false,
    },
    CellColumn {
        name: "model",
        nullable: false,
    },
    CellColumn {
        name: "weight_quant",
        nullable: false,
    },
    CellColumn {
        name: "kv_quant",
        nullable: false,
    },
    CellColumn {
        name: "ctx_max",
        nullable: false,
    },
    CellColumn {
        name: "prompt_id",
        nullable: false,
    },
    CellColumn {
        name: "decode_config",
        nullable: true,
    },
];

/// `PARTITION BY` / `GROUP BY` column list: the cell key plus `metric`.
pub fn partition_columns() -> String {
    let mut out = String::new();
    for col in CELL_COLUMNS {
        if !out.is_empty() {
            out.push_str(", ");
        }
        out.push_str(col.name);
    }
    out.push_str(", metric");
    out
}

/// A `WHERE`-clause fragment binding every cell column, `?first` upward.
///
/// The nullable column is compared with `IS`, not `=`: `decode_config = NULL`
/// is never true, so an ordinary-decode cell would match no row and the caller
/// would read "no champion" where there is one.
///
/// Returns the fragment and the next free parameter index, so a caller that
/// appends `metric` or a timestamp does not have to count.
pub fn predicate(first_param: usize) -> (String, usize) {
    let mut sql = String::new();
    for (offset, col) in CELL_COLUMNS.iter().enumerate() {
        let idx = first_param + offset;
        let op = if col.nullable { "IS" } else { "=" };
        if !sql.is_empty() {
            sql.push_str("\n           AND ");
        }
        // write! to a String is infallible — the unit Ok is discarded.
        let _ = write!(sql, "{} {op} ?{idx}", col.name);
    }
    (sql, first_param + CELL_COLUMNS.len())
}

#[cfg(test)]
#[path = "cell_tests.rs"]
mod tests;
