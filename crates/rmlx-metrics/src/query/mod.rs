//! Read-only query API per `docs/METRICS_DB.md` §8.2.
//!
//! All functions take a `&Connection` and return plain data structs. No writes
//! are performed here — mutating ops live in [`crate::recorder`].
//!
//! # Public API
//!
//! - [`best`] — best-known measurement for a `(model, backend, kv_quant)` cell.
//! - [`rank`] — leaderboard: all cells ranked by a given metric.
//! - [`compare`] — pairwise backend diff for a metric across all cells.
//! - [`history`] — all observations for a cell, oldest first.
//! - [`timeseries`] — observations bucketed into time intervals.
//! - [`deltas`] — detect regressions by comparing sequential observations.
//! - [`regress`] — statistical regression detection (mean ± N·σ).
//! - [`champions`] — best row per metric × model, used for export.
//! - Key structs: [`Cell`], [`BestRow`], [`ObservationRow`], [`CompareRow`],
//!   [`DeltaRow`], [`ChampionRow`], [`RegressResult`].
//!
//! # See also
//!
//! - `docs/METRICS_DB.md` — schema, query API contract, §8.2.

mod read;
mod types;

pub use read::{best, champions, compare, deltas, history, rank, regress, timeseries};
pub use types::{
    BestRow, Bucket, Cell, ChampionCell, ChampionRow, CompareRow, DeltaRow, ObservationRow,
    RegressResult, TimeseriesPoint,
};

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod query_tests;
