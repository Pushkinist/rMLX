//! Runtime per-event recorder — DB-direct replacement for the legacy
//! `rmlx_core::metrics::MetricsSink` that used to write
//! `metrics/<run-id>.jsonl` and `metrics/summary.csv` from every call site.
//!
//! Design:
//! * Open the DB once per run (lazily by call site) via [`EventRecorder::open`].
//! * [`EventRecorder::record`] does ONE `INSERT INTO events …` per
//!   measurement. No buffering, no jsonl writer, no csv writer. Crash-safe
//!   by SQLite WAL (the same PRAGMA stack other recorders use).
//! * Schema lives in migration `002_events.sql`. Run `migrate::run_pending`
//!   on first open.
//!
//! The legacy CSV mirror (`metrics/summary.csv`) and per-run jsonl
//! (`metrics/<run-id>.jsonl`) writers are gone. The DB is the single
//! source-of-truth, see CLAUDE.md "Metrics retention".
//!
//! `EventRecorder` is `Send + Sync`: the inner `rusqlite::Connection` is
//! guarded by a `Mutex` so callers can clone the `Arc<EventRecorder>` across
//! threads / async tasks.

use std::path::Path;
use std::sync::Mutex;

use rusqlite::{params, Connection};
use serde::Serialize;

use crate::error::{Error, Result};
use crate::identity::RunIdentity;
use crate::time_util::now_iso8601;
use crate::{migrate, schema};

// ── SSD-tier per-block event kinds ───────────────────────────────────────────

/// Payload for one SSD-tier spill event (one `.kvb` write).
///
/// All durations in microseconds. The sub-fields break the total `dur_us`
/// into three non-overlapping phases (serialize → fs write → index record).
/// The sum of the three sub-fields ≈ `dur_us` (within a few µs of overhead
/// between phase boundaries).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed event struct — fields are the complete SSD-spill timing payload; constructed with struct-literal from rmlx-models::kv_cache::spill; adding a field requires updating all construction sites"
)]
#[derive(Debug, Clone)]
pub struct SsdSpillEvent {
    /// Namespace (model_id) this block was spilled under.
    pub namespace: String,
    /// Raw bytes written to the `.kvb` file.
    pub bytes: u64,
    /// Wall time for the entire `drain_one` call (µs).
    pub dur_us: u64,
    /// Phase 1 — `write_caches` serialize-only (tensor eval + safetensors
    /// layout build, before `serialize_to_file` touches the FS).
    pub dur_serialize_us: u64,
    /// Phase 2 — `serialize_to_file` FS write + fsync (µs).
    pub dur_write_us: u64,
    /// Phase 3 — `SsdKvIndex::record` SQLite INSERT (µs).
    pub dur_index_us: u64,
}

/// Payload for one SSD-tier hydrate event (one `.kvb` read-back on RAM miss).
///
/// All durations in microseconds. Sub-fields cover five sequential phases.
/// Their sum ≈ `dur_us` (a few µs of boundary overhead is expected).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed event struct — fields are the complete SSD-hydrate timing payload; constructed with struct-literal from rmlx-models::kv_cache::hydrate; adding a field requires updating all construction sites"
)]
#[derive(Debug, Clone)]
pub struct SsdHydrateEvent {
    /// Namespace (model_id) this block was hydrated under.
    pub namespace: String,
    /// Raw bytes read from the `.kvb` file.
    pub bytes: u64,
    /// Wall time for the entire `lookup` call (µs).
    pub dur_us: u64,
    /// Phase 1 — `SsdKvIndex::lookup_longest_prefix` SQLite read (µs).
    pub dur_lookup_us: u64,
    /// Phase 2 — `block_io::read_caches` mmap + safetensors parse (µs).
    pub dur_read_us: u64,
    /// Phase 3 — `KvBlockReader::hydrate` CPU-side dequant / storage rebuild (µs).
    pub dur_dequant_us: u64,
    /// Phase 4 — time to wrap each reconstructed [`KvStorage`] into a
    /// decode-ready [`KvCache`]; CPU-only struct construction, not a GPU upload.
    pub dur_finalize_us: u64,
    /// Phase 5 — `SsdKvIndex::touch` SQLite UPDATE (µs).
    pub dur_touch_us: u64,
    /// Number of KV-cache blocks reconstructed.
    pub block_count: u64,
}

/// One measurement event. All string fields are borrowed to avoid heap
/// allocation on the hot path. Schema mirrors the legacy `Measurement`
/// shape so call sites need not change beyond the type-import line.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed measurement struct — fields are the complete metric-record contract; constructed with struct-literal from rmlx-server; adding a field requires updating all Measurement construction sites"
)]
#[derive(Debug, Clone, Serialize)]
pub struct Measurement<'a> {
    /// Filesystem path or identifier of the model being measured.
    pub model_path: &'a str,
    /// KV-quantization mode string (e.g. `"k8v8"`, `"none"`).
    pub quant_mode: &'a str,
    /// Logical stage of the measurement (e.g. `"request"`, `"ssd_tier"`).
    pub stage: &'a str,
    /// Operation name matching a registry metric (e.g. `"decode_tps_warm"`).
    pub op: &'a str,
    /// Unit of the measured value (e.g. `"ms"`, `"bytes"`, `"tps"`).
    pub value_unit: &'a str,
    /// Numeric value of the measurement.
    pub value: f64,
    /// Optional free-form notes attached to this event row.
    pub notes: &'a str,
}

/// DB-direct per-event recorder. Replaces `MetricsSink`. Open once per run;
/// share via `Arc` across threads.
///
/// Every row is stamped with the same [`RunIdentity`] that `observations` rows
/// carry, so the two tables agree on who produced a run without having to
/// reverse-engineer a semver out of the SHA embedded in `run_id`.
///
/// Under `--metrics off` the recorder holds no connection: the SQLite file is
/// never opened or created, and [`EventRecorder::record`] is an immediate
/// `Ok(())`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal recorder — fields are private impl detail; public API is the record() method, not struct literal construction"
)]
pub struct EventRecorder {
    run_id: String,
    /// `None` under `--metrics off` — no DB is opened.
    conn: Option<Mutex<Connection>>,
}

#[allow(
    clippy::missing_fields_in_debug,
    reason = "conn (Mutex<Connection>) is an opaque implementation detail; exposing it in Debug would dump internal SQLite state and add no diagnostic value"
)]
impl std::fmt::Debug for EventRecorder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EventRecorder")
            .field("run_id", &self.run_id)
            .finish()
    }
}

impl EventRecorder {
    /// Open the canonical runs DB (resolved via `rmlx_core::paths`) and
    /// ensure all migrations have run. Cheap on subsequent calls (the
    /// PRAGMA stack is per-connection but the schema is persistent).
    ///
    /// Under `--metrics off` this never even resolves the DB path — checked
    /// here, before [`rmlx_core::paths::metrics_db_path`], not left to
    /// [`open_at`](Self::open_at)'s own gate — because path resolution itself
    /// is not free: `metrics_db_path` walks through `metrics_dir`, which
    /// unconditionally `create_dir_all`s `<RMLX_HOME>/metrics/`. Calling it
    /// and THEN checking the mode would still leave an empty `metrics/`
    /// directory behind under `off`.
    pub fn open(run_id: &str) -> Result<Self> {
        if !crate::mode::events_enabled() {
            tracing::debug!(run_id, "events: --metrics off, no DB opened");
            return Ok(Self {
                run_id: run_id.to_owned(),
                conn: None,
            });
        }
        let path = rmlx_core::paths::metrics_db_path();
        Self::open_at(&path, run_id)
    }

    /// Same as [`open`] but lets the caller specify the DB path. Used by
    /// unit tests with a `tempdir`.
    ///
    /// Under `--metrics off` this opens nothing — not the file, not the parent
    /// directory, not even [`RunIdentity`] — and returns a recorder whose
    /// `record` is a no-op. Identity is resolved lazily inside [`record`](Self::record),
    /// only on the path that actually writes a row, so the off-mode early
    /// return does no work at all to fill a field it will never read. Carries
    /// its own copy of the mode check (rather than relying solely on
    /// [`open`](Self::open)'s) because this function is `pub` and callable
    /// directly with an arbitrary path, bypassing `open`'s gate entirely.
    pub fn open_at(db_path: &Path, run_id: &str) -> Result<Self> {
        if !crate::mode::events_enabled() {
            tracing::debug!(run_id, "events: --metrics off, no DB opened");
            return Ok(Self {
                run_id: run_id.to_owned(),
                conn: None,
            });
        }

        if let Some(parent) = db_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| {
                    Error::Schema(format!("events: create dir {}: {e}", parent.display()))
                })?;
            }
        }
        let mut conn = schema::open(db_path)?;
        migrate::run_pending(&mut conn)?;
        Ok(Self {
            run_id: run_id.to_owned(),
            conn: Some(Mutex::new(conn)),
        })
    }

    /// Append one measurement event to the `events` table.
    ///
    /// Returns `Err` on any SQLite failure — no retry. The hot path is one
    /// prepared `INSERT` per call; SQLite WAL absorbs concurrent writers.
    ///
    /// No-op under `--metrics off`.
    pub fn record(&self, m: &Measurement<'_>) -> Result<()> {
        let Some(conn) = self.conn.as_ref() else {
            return Ok(());
        };
        // Resolved here, not stored on `self`: cheap after the first call
        // anywhere in the process (a cached `&'static` read), and this way
        // the `--metrics off` early return above never touches identity at all.
        let identity = RunIdentity::get();
        let ts = now_iso8601()?;
        let guard = conn
            .lock()
            .map_err(|_| Error::Schema("events: connection mutex poisoned".to_owned()))?;
        guard.execute(
            "INSERT INTO events (
                run_id, ts_utc, model_path, quant_mode,
                stage, op, value_unit, value, notes,
                backend_version, build_profile, mlx_nax
             ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                &self.run_id,
                ts,
                m.model_path,
                m.quant_mode,
                m.stage,
                m.op,
                m.value_unit,
                m.value,
                m.notes,
                &identity.backend_version,
                &identity.build_profile,
                &identity.mlx_nax,
            ],
        )?;

        tracing::debug!(
            run_id = %self.run_id,
            op = m.op,
            value = m.value,
            value_unit = m.value_unit,
            "events::record"
        );
        Ok(())
    }

    /// Append one SSD-spill event to the `events` table.
    ///
    /// Stored as a single row: `op = "ssd_spill"`, `value = dur_us`,
    /// `value_unit = "us"`, and all sub-fields packed into `notes` as a
    /// compact JSON string so the full timing breakdown is queryable.
    pub fn record_ssd_spill(&self, ev: &SsdSpillEvent) -> Result<()> {
        let notes = format!(
            r#"{{"bytes":{},"dur_serialize_us":{},"dur_write_us":{},"dur_index_us":{}}}"#,
            ev.bytes, ev.dur_serialize_us, ev.dur_write_us, ev.dur_index_us,
        );
        self.record(&Measurement {
            model_path: &ev.namespace,
            quant_mode: "",
            stage: "ssd_tier",
            op: "ssd_spill",
            value_unit: "us",
            value: ev.dur_us as f64,
            notes: &notes,
        })
    }

    /// Append one SSD-hydrate event to the `events` table.
    ///
    /// Stored as a single row: `op = "ssd_hydrate"`, `value = dur_us`,
    /// `value_unit = "us"`, sub-fields in `notes` JSON.
    pub fn record_ssd_hydrate(&self, ev: &SsdHydrateEvent) -> Result<()> {
        let notes = format!(
            r#"{{"bytes":{},"dur_lookup_us":{},"dur_read_us":{},"dur_dequant_us":{},"dur_finalize_us":{},"dur_touch_us":{},"block_count":{}}}"#,
            ev.bytes,
            ev.dur_lookup_us,
            ev.dur_read_us,
            ev.dur_dequant_us,
            ev.dur_finalize_us,
            ev.dur_touch_us,
            ev.block_count,
        );
        self.record(&Measurement {
            model_path: &ev.namespace,
            quant_mode: "",
            stage: "ssd_tier",
            op: "ssd_hydrate",
            value_unit: "us",
            value: ev.dur_us as f64,
            notes: &notes,
        })
    }

    /// Run identifier this recorder was opened with.
    pub fn run_id(&self) -> &str {
        &self.run_id
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "events_tests.rs"]
mod tests;
