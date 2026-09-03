//! The head/tail layer counts held at the KV boundary floor, and the
//! `decode_config` terms that name them.
//!
//! These live here rather than beside the policy in `rmlx_models::kv_cache`
//! because two crates that cannot see each other need the same numbers: the
//! engine, which applies them, and `rmlx-metrics`, which has to recognise a
//! `decode_config` that spells them out and refuse it (`NULL` is the engine at
//! its defaults — `docs/METRICS_DB.md` §3.2). A second copy in the metrics
//! crate would be a table that drifts silently the day the counts move, which
//! is exactly the failure the cell-identity grammar exists to prevent.
//!
//! Only the *values* are here. The policy that reads them — which layers are
//! promoted, and to which codec — stays in `rmlx_models::kv_cache`.

/// Default number of leading layers held at the boundary floor.
pub const DEFAULT_BOUNDARY_HEAD_N: usize = 2;

/// Default number of trailing layers held at the boundary floor.
pub const DEFAULT_BOUNDARY_TAIL_N: usize = 8;

/// `decode_config` key for the head count (`docs/METRICS_DB.md` §3.2).
pub const BOUNDARY_HEAD_KEY: &str = "kv_boundary/head";

/// `decode_config` key for the tail count.
pub const BOUNDARY_TAIL_KEY: &str = "kv_boundary/tail";

/// Every `decode_config` key whose setting has a single process-wide default,
/// paired with that default.
///
/// A term that matches its entry here says nothing: it describes the engine as
/// shipped, which `NULL` already says. Keys absent from this table either have
/// no default (`mtp/block` — absence means no drafter at all) or have a
/// per-architecture one (`prefill_chunk`), and neither can be recognised from
/// the term alone.
///
/// Values are numeric so each entry *is* the constant above rather than a
/// string spelling of it.
pub const DECODE_CONFIG_NUMERIC_DEFAULTS: &[(&str, u64)] = &[
    (BOUNDARY_HEAD_KEY, DEFAULT_BOUNDARY_HEAD_N as u64),
    (BOUNDARY_TAIL_KEY, DEFAULT_BOUNDARY_TAIL_N as u64),
];
