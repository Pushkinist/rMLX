//! Atomic per-run recorder per `docs/METRICS_DB.md` §8.2.1.
//!
//! [`Recorder`] ingests one [`crate::ingest::RunRecord`] into the
//! `observations` SQLite table in a single transaction. Duplicate runs
//! (same identity × prompt × metric) are skipped with a warning rather
//! than overwritten.
//!
//! # Public API
//!
//! - [`Recorder`] — stateless ingestion driver; constructed from a `&Connection`.
//! - [`RecordOutcome`] — summary returned after `record()`: rows inserted,
//!   skipped, and any per-row errors.
//!
//! # See also
//!
//! - `docs/METRICS_DB.md` §8.2.1 — recorder API contract and duplicate policy.

use rusqlite::params;
use rusqlite::Connection;

use crate::error::{Error, Result};
use crate::ingest::{prompt_body_sha256, IdentityPolicy, PromptRef, RunRecord};
use crate::prompts::{PromptId, PromptStore};
use crate::registry;
use crate::time_util::now_iso8601;

// ── Public types ──────────────────────────────────────────────────────────────

/// DB-direct run recorder. Validates, mints a run_id, and inserts observations in one transaction.
#[derive(Debug)]
#[allow(
    clippy::exhaustive_structs,
    reason = "internal DB accessor — fields are private; public API is record_run(), not struct literal construction"
)]
pub struct Recorder<'a> {
    conn: &'a mut Connection,
    inserted_by: String,
    policy: IdentityPolicy,
}

impl<'a> Recorder<'a> {
    /// `inserted_by` should look like `"rmlx-cli@0.2.8"` — see §10.3 audit.
    /// Build it with `RunIdentity::inserted_by` rather than a literal.
    ///
    /// Enforces the §8.5 run-identity contract: an `rmlx` record without a
    /// semver `backend_version` is rejected.
    pub fn new(conn: &'a mut Connection, inserted_by: impl Into<String>) -> Self {
        Self {
            conn,
            inserted_by: inserted_by.into(),
            policy: IdentityPolicy::Enforce,
        }
    }

    /// Recorder for the one-shot import of pre-contract archives
    /// (`rmlx metrics migrate`): legacy JSONL / CBB CSV / records-MD rows that
    /// predate the identity contract and have no version to state.
    ///
    /// NEVER use this to record a new measurement — it is the one door around
    /// the identity check, and it exists only because fabricating a semver for
    /// a 2026-01 archive row would be exactly the bug the check prevents.
    ///
    /// `pub(crate)`, not `pub`: the only legitimate caller is
    /// `migrate::legacy`, in this same crate. Exposing this to `rmlx-cli` /
    /// `rmlx-server` would hand the next emitter the exact escape hatch this
    /// whole contract exists to close.
    pub(crate) fn legacy_archive(conn: &'a mut Connection, inserted_by: impl Into<String>) -> Self {
        Self {
            conn,
            inserted_by: inserted_by.into(),
            policy: IdentityPolicy::LegacyArchive,
        }
    }

    /// Validates the run, mints a run_id, resolves the prompt, then inserts
    /// one observations row per non-null metric. ALL inside one transaction.
    ///
    /// Returns the minted run_id and the count of observations inserted.
    ///
    /// # Atomicity
    ///
    /// Per §8.2.1: validation failure returns `Err` before any transaction is
    /// opened. Any other failure inside the transaction causes a rollback — the
    /// DB is left unchanged.
    pub fn record_run(&mut self, run: &RunRecord) -> Result<RecordOutcome> {
        // Step 1: validate before opening the transaction. Every ingest path in
        // the workspace funnels through here, so this is the single point that
        // enforces the §8.5 contract.
        run.validate_with(self.policy)?;

        // Step 2: begin transaction.
        let tx = self.conn.transaction()?;

        // Step 3: resolve prompt.
        // Note: PromptStore::get_or_insert opens its own sub-transaction which
        // SQLite rejects inside an active transaction. We inline the equivalent
        // logic here, running directly on the outer transaction `tx`.
        let prompt_id = match &run.prompt {
            PromptRef::ByBody {
                name,
                body,
                tokens_approx,
                notes,
            } => prompt_get_or_insert_in_tx(&tx, name, body, *tokens_approx, notes.as_deref())?,

            PromptRef::BySha256 { sha256 } => {
                // find_by_sha256 is a read-only query; PromptStore is fine for that.
                let store = PromptStore::new(&tx);
                match store.find_by_sha256(sha256)? {
                    Some(row) => row.id,
                    None => {
                        return Err(Error::InvalidPrompt(format!(
                            "sha256 {sha256} not registered; pass full body in 'prompt'"
                        )));
                    }
                }
            }
        };

        // Step 4: mint run_id: <YYYYMMDDHHMMSS>-<6hex>.
        let inserted_utc = now_iso8601()?;
        // Derive the timestamp prefix from inserted_utc (already YYYY-MM-DDTHH:MM:SSZ).
        // Strip separators: "2026-05-10T07:30:00Z" → "20260510073000".
        let ts_digits: String = inserted_utc
            .chars()
            .filter(char::is_ascii_digit)
            .take(14)
            .collect();
        let hex6 = &uuid::Uuid::new_v4().simple().to_string()[..6];
        let run_id = format!("{ts_digits}-{hex6}");

        // Steps 5 + 6: for each metric entry, insert a row or skip if null.
        let mut observation_ids: Vec<i64> = Vec::new();
        let mut skipped_metrics: Vec<String> = Vec::new();

        for entry in &run.metrics {
            let Some(value) = entry.value else {
                skipped_metrics.push(entry.name.clone());
                continue;
            };

            // validate() already checked registry membership; re-lookup for canonical strings.
            let (unit, direction) = registry::lookup(&entry.name)?;

            tx.execute(
                "INSERT INTO observations (
                    backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id,
                    metric, value, unit, direction,
                    run_id, ts_utc, git_sha, build_profile, backend_version, hardware_tag,
                    prompt_tokens, max_tokens, temperature, seed, n_warmups, n_measure,
                    output_first_64, decode_stddev, notes, description,
                    inserted_utc, inserted_by
                ) VALUES (
                    ?1,  ?2,  ?3,  ?4,  ?5,  ?6,  ?7,
                    ?8,  ?9,  ?10, ?11,
                    ?12, ?13, ?14, ?15, ?16, ?17,
                    ?18, ?19, ?20, ?21, ?22, ?23,
                    ?24, ?25, ?26, ?27,
                    ?28, ?29
                )",
                params![
                    // cell identity
                    run.backend,
                    run.model_namespace,
                    run.model,
                    run.weight_quant,
                    run.kv_quant,
                    run.ctx_max,
                    prompt_id,
                    // metric
                    entry.name,
                    value,
                    unit,
                    direction.as_str(),
                    // run context
                    run_id,
                    run.ts_utc,
                    run.git_sha,
                    run.build_profile,
                    run.backend_version,
                    run.hardware_tag,
                    // bench config
                    run.prompt_tokens,
                    run.max_tokens,
                    run.temperature,
                    run.seed,
                    run.n_warmups,
                    run.n_measure,
                    // side data
                    run.output_first_64,
                    // decode_stddev: the spec says "only meaningful for decode_tps_*" but we
                    // store whatever is passed — recorder doesn't filter on metric name.
                    // Callers that care about this looseness should validate before passing.
                    entry.stddev,
                    run.notes,
                    run.description,
                    // bookkeeping
                    inserted_utc,
                    self.inserted_by,
                ],
            )?;

            observation_ids.push(tx.last_insert_rowid());
        }

        // Step 7: commit.
        tx.commit()?;

        // Step 8: return outcome.
        Ok(RecordOutcome {
            run_id,
            observation_ids,
            prompt_id,
            skipped_metrics,
        })
    }
}

// ── Private helpers ───────────────────────────────────────────────────────────

/// Prompt get-or-insert that runs on an **already-open** connection or
/// transaction, without opening a nested sub-transaction.
///
/// `PromptStore::get_or_insert` opens its own `unchecked_transaction` which
/// SQLite rejects when a transaction is already active. This function performs
/// the identical logic (sha256 dedup → INSERT if absent) directly on `conn`.
fn prompt_get_or_insert_in_tx(
    conn: &Connection,
    name: &str,
    body: &serde_json::Value,
    tokens_approx: Option<i64>,
    notes: Option<&str>,
) -> Result<PromptId> {
    use rusqlite::OptionalExtension;

    let sha = prompt_body_sha256(body);

    // Fast path: already present.
    if let Some(id) = conn
        .query_row(
            "SELECT id FROM prompts WHERE sha256 = ?1",
            params![sha],
            |r| r.get::<_, PromptId>(0),
        )
        .optional()?
    {
        return Ok(id);
    }

    // Slow path: insert into the caller's transaction.
    let body_text = serde_json::to_string(body)
        .map_err(|e| Error::Schema(format!("body serialization failed: {e}")))?;
    let now = now_iso8601()?;
    conn.execute(
        "INSERT INTO prompts (sha256, name, body, tokens_approx, first_seen_utc, notes)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![sha, name, body_text, tokens_approx, now, notes],
    )?;
    Ok(conn.last_insert_rowid())
}

// ── Outcome ───────────────────────────────────────────────────────────────────

/// Outcome of a successful [`Recorder::record_run`] call.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct RecordOutcome {
    /// Minted run_id in format `<YYYYMMDDHHMMSS>-<6hex>`.
    pub run_id: String,
    /// Row ids of inserted observations (one per non-null metric).
    pub observation_ids: Vec<i64>,
    /// Resolved prompt id from the `prompts` table.
    pub prompt_id: i64,
    /// Metric names whose value was `None` — no row written for these.
    pub skipped_metrics: Vec<String>,
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "recorder_tests.rs"]
mod tests;
