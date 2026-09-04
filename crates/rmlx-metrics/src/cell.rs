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

// ── Deriving `decode_config` from a row's own fields ──────────────────────────

/// What a row's `notes` say about how its tokens were produced.
#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed classification — a row's notes either name a drafter, say there was none, or say nothing"
)]
pub enum NotesVerdict {
    /// The notes name a drafter and its block size.
    Speculative(String),
    /// The notes say there was no drafter, or name the plain arm of a bench.
    Plain,
    /// The notes carry no statement either way.
    Silent,
}

/// The canonical `decode_config` for a speculative arm.
///
/// `block_size` is the block the run was *configured* with. `depth_policy`
/// names how the round loop picks each round's block when it does not simply
/// take that one — `None` for a loop that drafts a fixed block every round.
///
/// The policy is part of cell identity because the block is not observable
/// from the configured ceiling once a loop moves off it: DFlash already halves
/// and grows its block from the recent accept rate, so an adaptive arm at
/// ceiling 16 and a fixed arm at block 16 are different configurations that
/// would otherwise rank against each other under one label. Absence of the
/// term is the fixed block, which is what keeps every row recorded before the
/// term existed in the cell it has always been in.
///
/// This is the only place the format is written. The engine composes the
/// string here and logs it on its `done` line, and `scripts/spec_bench.sh`
/// records what the engine said rather than spelling it a second time.
pub fn decode_config(draft_kind: &str, block_size: u32, depth_policy: Option<&str>) -> String {
    match depth_policy {
        Some(policy) => {
            format!("{draft_kind}/block={block_size},{draft_kind}/depth={policy}")
        }
        None => format!("{draft_kind}/block={block_size}"),
    }
}

/// Whether `value` is a well-formed `decode_config` — see `docs/METRICS_DB.md`
/// §3.2 for the grammar and why it is a grammar and not a free-form label.
///
/// One or more `key=value` terms joined by `,`, no whitespace, terms strictly
/// ordered by key. A key is one or more lower-case `[a-z0-9_]` segments joined
/// by `/`, which is what makes the speculative arm's `mtp/block=5` and a
/// prefill setting's `prefill_chunk=1024` terms of one shape. A value is a
/// non-empty run of `[A-Za-z0-9_.+-]`.
///
/// The ordering requirement is the part that carries weight: `decode_config`
/// is cell identity, so two emitters describing the same engine configuration
/// in different term orders would split one cell in two and rank neither
/// against the other. Absence (`NULL`) is the engine at its defaults and is
/// not spelled here.
pub fn decode_config_is_well_formed(value: &str) -> bool {
    let key_ok = |key: &str| {
        !key.is_empty()
            && key.split('/').all(|segment| {
                !segment.is_empty()
                    && segment
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
            })
    };
    let value_ok = |v: &str| {
        !v.is_empty()
            && v.chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '+' | '-'))
    };

    let mut previous_key: Option<&str> = None;
    for term in value.split(',') {
        let Some((key, term_value)) = term.split_once('=') else {
            return false;
        };
        if !key_ok(key) || !value_ok(term_value) {
            return false;
        }
        if previous_key.is_some_and(|previous| previous >= key) {
            return false;
        }
        previous_key = Some(key);
    }
    previous_key.is_some()
}

/// Whether every term of `value` spells a setting's own shipped default.
///
/// `NULL` is the engine at its defaults (`docs/METRICS_DB.md` §3.2), so a
/// string that says the same thing is a second spelling of one configuration —
/// and two spellings are two cells that never rank against each other. This is
/// how a whole campaign's rows can sit beside the rows they were meant to be
/// compared with and win nothing.
///
/// The defaults come from [`rmlx_core::kv_boundary::DECODE_CONFIG_NUMERIC_DEFAULTS`],
/// the same constants the engine applies. A key that is not in that table has
/// no single default — `mtp/block` has none (absence means no drafter at all)
/// and `prefill_chunk`'s is per-architecture — so a `decode_config` containing
/// one is never all-defaults, whatever its other terms say.
///
/// Returns `false` for a malformed value: this answers "is it redundant", not
/// "is it legal", and [`decode_config_is_well_formed`] is the one that answers
/// the second.
pub fn decode_config_is_all_defaults(value: &str) -> bool {
    if !decode_config_is_well_formed(value) {
        return false;
    }
    value.split(',').all(|term| {
        let Some((key, term_value)) = term.split_once('=') else {
            return false;
        };
        rmlx_core::kv_boundary::DECODE_CONFIG_NUMERIC_DEFAULTS
            .iter()
            .any(|&(default_key, default)| {
                default_key == key && term_value.parse::<u64>() == Ok(default)
            })
    })
}

/// Read the value of `key=` out of a `notes` string, up to the next space.
fn note_value<'a>(notes: &'a str, key: &str) -> Option<&'a str> {
    let rest = notes.split(key).nth(1)?;
    let value = rest.split_whitespace().next()?;
    (!value.is_empty()).then_some(value)
}

/// Classify a row by what its own `notes` say.
///
/// Bench scripts have recorded the drafter in `notes` since long before there
/// was a column for it, so most rows written before that column existed can say
/// what they were. The order matters: an early `spec_bench.sh` put the run's
/// drafter flags on *both* arms, so a row saying `config=normal` alongside
/// `draft_kind=mtp` is the no-drafter arm of a speculative bench — and its own
/// `draft_tokens_total` is zero. "It says there was no drafter" therefore wins
/// over "it names one".
pub fn decode_config_from_notes(notes: &str) -> NotesVerdict {
    if notes.contains("draft_kind=none")
        || notes.contains("config=normal")
        || notes.contains("config=base")
    {
        return NotesVerdict::Plain;
    }

    let kind = note_value(notes, "draft_kind=");
    let block = note_value(notes, "block_size=");
    match (kind, block) {
        (Some(kind), Some(block)) if kind != "none" => {
            match block.parse::<u32>() {
                // The drafter name is lifted verbatim out of free-form notes,
                // so it can be anything a bench script wrote — `Eagle3`, a
                // typo, a word with a space in it. Composing that into a
                // `decode_config` and storing it would put an out-of-grammar
                // string into the append-only table, where nothing can take it
                // back out. A row whose notes do not compose to a legal
                // configuration does not say what it was.
                Ok(block) => {
                    // Notes record the configured block and nothing about
                    // how the loop chose each round's, so a backfilled row can
                    // only claim the fixed spelling.
                    let config = decode_config(kind, block, None);
                    if decode_config_is_well_formed(&config) {
                        NotesVerdict::Speculative(config)
                    } else {
                        NotesVerdict::Silent
                    }
                }
                // A block size that is not a number is not a block size; the
                // row does not say what it was.
                Err(_) => NotesVerdict::Silent,
            }
        }
        _ => NotesVerdict::Silent,
    }
}

#[cfg(test)]
#[path = "cell_tests.rs"]
mod tests;
