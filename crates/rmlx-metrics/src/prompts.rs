//! Prompt registry per `docs/METRICS_DB.md` §3.1 + §8.7.
//!
//! Prompts are content-addressed by SHA-256 over the canonical JSON of the
//! prompt body. This allows observations to reference a prompt by hash
//! without duplicating the body in every row.
//!
//! # Public API
//!
//! - [`PromptStore`] — DB-backed prompt insert/lookup; wraps `&Connection`.
//! - [`PromptRow`] — one row from the `prompts` table.
//! - [`PromptFile`] — deserialized prompt JSON file from `prompts/`.
//! - [`parse_prompt_file`] — parse a prompt JSON file from disk.
//! - [`sync_dir`] — upsert all prompt files in a directory into the registry.

use rusqlite::{params, Connection, OptionalExtension};
use serde_json::Value;
use std::path::Path;

use crate::error::{Error, Result};
use crate::ingest::prompt_body_sha256;
use crate::time_util::now_iso8601;

// ── Types ─────────────────────────────────────────────────────────────────────

/// SQLite row-id type for the `prompts` table.
pub type PromptId = i64;

/// A row from the `prompts` table.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PromptRow {
    /// Row ID in the `prompts` table.
    pub id: PromptId,
    /// SHA-256 hex of the prompt body (64 lowercase chars); content-addresses the row.
    pub sha256: String,
    /// Display name for this prompt (e.g. `"longctx_4k"`).
    pub name: String,
    /// Deserialized from the TEXT (JSON-encoded) `body` column.
    pub body: Value,
    /// Approximate token count for the body, if known.
    pub tokens_approx: Option<i64>,
    /// ISO-8601 UTC timestamp of the first time this prompt was seen.
    pub first_seen_utc: String,
    /// Optional notes about this prompt.
    pub notes: Option<String>,
}

// ── Store ─────────────────────────────────────────────────────────────────────

/// Typed access to the `prompts` table.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal DB accessor — fields are private; public API is the get_or_insert/lookup methods, not struct literal construction"
)]
pub struct PromptStore<'a> {
    conn: &'a Connection,
}

impl std::fmt::Debug for PromptStore<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PromptStore").finish_non_exhaustive()
    }
}

impl<'a> PromptStore<'a> {
    /// Wrap a borrowed connection. The connection must outlive the store.
    pub fn new(conn: &'a Connection) -> Self {
        Self { conn }
    }

    /// Insert if new sha256, else return existing id.
    ///
    /// `body` is the prompt content (string, array, or any JSON value).
    /// Stored as JSON-encoded TEXT. The sha256 is computed via
    /// [`prompt_body_sha256`] — identical body = identical hash = no new row.
    pub fn get_or_insert(
        &self,
        name: &str,
        body: &Value,
        tokens_approx: Option<i64>,
        notes: Option<&str>,
    ) -> Result<PromptId> {
        let sha = prompt_body_sha256(body);

        // Fast path: already present.
        if let Some(id) = self
            .conn
            .query_row(
                "SELECT id FROM prompts WHERE sha256 = ?1",
                params![sha],
                |r| r.get::<_, PromptId>(0),
            )
            .optional()?
        {
            return Ok(id);
        }

        // Slow path: insert inside a transaction.
        let body_text = serde_json::to_string(body)
            .map_err(|e| Error::Schema(format!("body serialization failed: {e}")))?;
        let now = now_iso8601()?;

        let tx = self.conn.unchecked_transaction()?;
        tx.execute(
            "INSERT INTO prompts (sha256, name, body, tokens_approx, first_seen_utc, notes)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![sha, name, body_text, tokens_approx, now, notes],
        )?;
        let id = tx.last_insert_rowid();
        tx.commit()?;

        Ok(id)
    }

    /// Look up by sha256. `None` if not present.
    pub fn find_by_sha256(&self, sha256: &str) -> Result<Option<PromptRow>> {
        self.conn
            .query_row(
                "SELECT id, sha256, name, body, tokens_approx, first_seen_utc, notes
                 FROM prompts WHERE sha256 = ?1",
                params![sha256],
                row_to_prompt,
            )
            .optional()
            .map_err(Error::from)
    }

    /// Look up by id. `None` if not present.
    pub fn find_by_id(&self, id: PromptId) -> Result<Option<PromptRow>> {
        self.conn
            .query_row(
                "SELECT id, sha256, name, body, tokens_approx, first_seen_utc, notes
                 FROM prompts WHERE id = ?1",
                params![id],
                row_to_prompt,
            )
            .optional()
            .map_err(Error::from)
    }

    /// Find the latest (by `first_seen_utc`) row with the given `name`.
    ///
    /// Multiple revisions with the same name are normal — each body change
    /// creates a new row. Returns `None` if no prompt with that name exists.
    pub fn find_latest_by_name(&self, name: &str) -> Result<Option<PromptRow>> {
        self.conn
            .query_row(
                "SELECT id, sha256, name, body, tokens_approx, first_seen_utc, notes
                 FROM prompts WHERE name = ?1
                 ORDER BY first_seen_utc DESC
                 LIMIT 1",
                params![name],
                row_to_prompt,
            )
            .optional()
            .map_err(Error::from)
    }

    /// Return all rows ordered by `id`.
    pub fn list(&self) -> Result<Vec<PromptRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, sha256, name, body, tokens_approx, first_seen_utc, notes
             FROM prompts ORDER BY id",
        )?;
        let rows = stmt
            .query_map([], row_to_prompt)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(rows)
    }
}

// ── Row helper ────────────────────────────────────────────────────────────────

fn row_to_prompt(r: &rusqlite::Row<'_>) -> rusqlite::Result<PromptRow> {
    let body_text: String = r.get(3)?;
    let body: Value = serde_json::from_str(&body_text).map_err(|e| {
        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(e))
    })?;
    Ok(PromptRow {
        id: r.get(0)?,
        sha256: r.get(1)?,
        name: r.get(2)?,
        body,
        tokens_approx: r.get(4)?,
        first_seen_utc: r.get(5)?,
        notes: r.get(6)?,
    })
}

// ── Prompt file ───────────────────────────────────────────────────────────────

/// Parsed representation of a single `rMLX/prompts/*.json` file.
#[derive(Debug)]
#[non_exhaustive]
pub struct PromptFile {
    /// Display name for the prompt (from JSON `name` field, or file stem fallback).
    pub name: String,
    /// Parsed prompt body (messages array or string).
    pub body: Value,
    /// Approximate token count, if present in the JSON file.
    pub tokens_approx: Option<i64>,
    /// Optional notes from the JSON file.
    pub notes: Option<String>,
}

/// Parse a single prompt JSON file into a [`PromptFile`].
///
/// Body selection priority:
/// 1. JSON key `messages` (whole array treated as body).
/// 2. JSON key `body` (string or any value).
/// 3. Error [`Error::InvalidPrompt`] if neither key is present.
///
/// `name` is taken from the JSON `name` field; if absent, derived from the
/// file stem with a `tracing::warn!`.
pub fn parse_prompt_file(path: &Path) -> Result<PromptFile> {
    let raw = std::fs::read_to_string(path)?;
    let mut obj: serde_json::Map<String, Value> = serde_json::from_str(&raw)
        .map_err(|e| Error::InvalidPrompt(format!("{}: JSON parse error: {e}", path.display())))?;

    // Name field.
    let name = if let Some(Value::String(n)) = obj.remove("name") {
        n
    } else {
        let stem = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        tracing::warn!(
            path = %path.display(),
            stem = %stem,
            "prompt file has no 'name' field — using file stem"
        );
        stem
    };

    // Body field (messages > body).
    let body = if let Some(messages) = obj.remove("messages") {
        messages
    } else if let Some(b) = obj.remove("body") {
        b
    } else {
        return Err(Error::InvalidPrompt(format!(
            "{}: file has neither 'messages' nor 'body' key",
            path.display()
        )));
    };

    // Optional fields.
    let tokens_approx = obj.get("tokens_approx").and_then(Value::as_i64);
    let notes = obj.get("notes").and_then(Value::as_str).map(str::to_string);

    Ok(PromptFile {
        name,
        body,
        tokens_approx,
        notes,
    })
}

// ── Directory sync ────────────────────────────────────────────────────────────

/// Sync every `*.json` file under `dir` (one level deep) into the `prompts`
/// table.
///
/// Returns `(inserted_count, total_files)`. Non-JSON files are skipped
/// silently. Files that fail to parse emit a `tracing::warn!` and are skipped
/// without aborting the sync.
pub fn sync_dir(conn: &Connection, dir: &Path) -> Result<(usize, usize)> {
    let store = PromptStore::new(conn);
    let mut total = 0usize;
    let mut inserted = 0usize;

    let entries = std::fs::read_dir(dir)?;

    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // One level deep, files only.
        if !path.is_file() {
            continue;
        }
        // Skip non-JSON.
        match path.extension().and_then(|e| e.to_str()) {
            Some("json") => {}
            _ => continue,
        }

        total += 1;

        let pf = match parse_prompt_file(&path) {
            Ok(pf) => pf,
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "skipping prompt file due to parse error");
                continue;
            }
        };

        // Check current sha before insert to track "inserted vs already-present".
        let sha = prompt_body_sha256(&pf.body);
        let already_present = conn
            .query_row(
                "SELECT id FROM prompts WHERE sha256 = ?1",
                params![sha],
                |r| r.get::<_, PromptId>(0),
            )
            .optional()?
            .is_some();

        store.get_or_insert(&pf.name, &pf.body, pf.tokens_approx, pf.notes.as_deref())?;

        if !already_present {
            inserted += 1;
        }
    }

    Ok((inserted, total))
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "prompts_tests.rs"]
mod prompts_tests;
