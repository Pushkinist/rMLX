//! Universal §8.5 ingest contract — the JSON shape every backend emits.
//!
//! Every benchmark backend (rMLX, mlx_lm, paroquant, omlx, ollama) writes one
//! JSON file per run in this shape, which [`crate::recorder`] then ingests
//! into the `observations` SQLite table.
//!
//! # Public API
//!
//! - [`RunRecord`] — top-level envelope: identity fields + metric entries.
//! - [`PromptRef`] — either an inline prompt body or a SHA-256 reference to
//!   a prompt already registered in the `prompts` table.
//! - [`MetricEntry`] — one measurement: name, value, unit, direction.
//! - [`prompt_body_sha256`] — canonical SHA-256 of a JSON prompt body,
//!   used to content-address the `prompts` table.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::identity;
use crate::registry;

// ── Core record ───────────────────────────────────────────────────────────────

#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed record struct — fields are the complete universal §8.5 run-record contract; constructed with struct-literal from rmlx-server; adding a field requires updating all RunRecord construction sites"
)]
/// Top-level §8.5 run record emitted by every benchmark backend (see docs/METRICS_DB.md §8.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Canonical backend identifier (e.g. `"rmlx"`, `"mlx_lm"`).
    pub backend: String,
    /// Semver string of the backend binary, if known.
    #[serde(default)]
    pub backend_version: Option<String>,
    /// Model namespace from the whitelist (e.g. `"mlx-community"`).
    pub model_namespace: String,
    /// Model repository name within the namespace.
    pub model: String,
    /// Canonical weight quantization string (e.g. `"mxfp8"`, `"8bit"`).
    pub weight_quant: String,
    /// Canonical KV-cache quantization string (e.g. `"k8v8"`, `"none"`).
    pub kv_quant: String,
    /// Maximum context length used during this bench run (tokens).
    pub ctx_max: i64,
    /// Prompt used for this run — inline body or SHA-256 reference.
    pub prompt: PromptRef,
    /// ISO-8601 UTC timestamp, validated as parseable.
    pub ts_utc: String,
    /// Git commit SHA of the backend binary, if known.
    #[serde(default)]
    pub git_sha: Option<String>,
    /// Cargo build profile (e.g. `"release"`, `"release-perf"`).
    #[serde(default)]
    pub build_profile: Option<String>,
    /// Hardware tag identifying the test machine (e.g. `"m5_max_128gb"`).
    pub hardware_tag: String,
    /// Number of tokens in the prompt, as counted by the bench harness.
    #[serde(default)]
    pub prompt_tokens: Option<i64>,
    /// Maximum number of tokens generated per measurement call.
    #[serde(default)]
    pub max_tokens: Option<i64>,
    /// Sampling temperature used; `0.0` for deterministic greedy decoding.
    #[serde(default)]
    pub temperature: Option<f64>,
    /// Random seed for the sampler, if set.
    #[serde(default)]
    pub seed: Option<i64>,
    /// Number of warmup calls before timed measurements.
    #[serde(default)]
    pub n_warmups: Option<i64>,
    /// Number of timed measurement calls.
    #[serde(default)]
    pub n_measure: Option<i64>,
    /// First ≤64 characters of the model's output, for coherence checks.
    #[serde(default)]
    pub output_first_64: Option<String>,
    /// Free-form run notes (auto-summary, legacy keys, etc.).
    #[serde(default)]
    pub notes: Option<String>,
    /// Human-readable description of the run (e.g. `"sha1234: add KV quant"`).
    #[serde(default)]
    pub description: Option<String>,
    /// One entry per measured metric; `value = None` entries are skipped.
    pub metrics: Vec<MetricEntry>,
}

// ── Prompt ref ────────────────────────────────────────────────────────────────

/// Either a full prompt body (with optional name + notes) or a sha256-only
/// reference to an already-registered prompt.
///
/// Body forms accepted: a JSON string (flat body) or any JSON value (e.g. a
/// messages array) — both are content-addressed via [`prompt_body_sha256`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
#[allow(
    clippy::exhaustive_enums,
    reason = "ingest wire enum — exactly two prompt-reference forms; adding a form requires updating the §8.5 ingest contract and all bench scripts"
)]
pub enum PromptRef {
    /// Inline prompt: the bench harness provides the full body.
    ByBody {
        /// Display name for this prompt (e.g. `"longctx_4k"`).
        name: String,
        /// Prompt body — a JSON string or messages array.
        body: serde_json::Value,
        /// Optional free-form notes about this prompt.
        #[serde(default)]
        notes: Option<String>,
        /// Approximate token count for the body, if pre-counted.
        #[serde(default)]
        tokens_approx: Option<i64>,
    },
    /// Reference by SHA-256 to a prompt already registered in the `prompts` table.
    BySha256 {
        /// Hex SHA-256 of the prompt body (64 lowercase hex chars).
        sha256: String,
    },
}

// ── Metric entry ──────────────────────────────────────────────────────────────

#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed metric-entry struct — three fields are the complete §8.5 metric-entry contract; constructed with struct-literal from rmlx-server; adding a field requires updating all MetricEntry construction sites"
)]
#[derive(Debug, Clone, Serialize, Deserialize)]
/// One §8.5 metric measurement: a registry name, an optional value, and an optional stddev.
pub struct MetricEntry {
    /// Metric name matching a registry entry (see docs/METRICS_DB.md §4).
    pub name: String,
    /// `None` → skipped (sparse). Recorder writes no row for null entries.
    pub value: Option<f64>,
    /// Optional standard deviation across `n_measure` calls.
    #[serde(default)]
    pub stddev: Option<f64>,
}

// ── RunRecord impl ────────────────────────────────────────────────────────────

impl RunRecord {
    /// Validates per §8.5 required fields + §4 metric registry + §5 whitelists.
    ///
    /// Does NOT touch the DB. Returns `Ok(())` if the record is structurally
    /// accepted; returns the first specific error encountered otherwise.
    pub fn validate(&self) -> Result<()> {
        // backend
        identity::canonicalize("backend", &self.backend, identity::BACKEND_WHITELIST)?;

        // model_namespace
        identity::canonicalize(
            "model_namespace",
            &self.model_namespace,
            identity::NAMESPACE_WHITELIST,
        )?;

        // weight_quant
        identity::canonicalize(
            "weight_quant",
            &self.weight_quant,
            identity::WEIGHT_QUANT_WHITELIST,
        )?;

        // kv_quant: parser-based validation accepts `mixed_k<>g<>_v<>g<>`.
        identity::canonicalize_kv_quant(&self.kv_quant)?;

        // ctx_max
        if self.ctx_max <= 0 {
            return Err(Error::InvalidIngestField {
                field: "ctx_max".to_string(),
                message: format!("must be > 0, got {}", self.ctx_max),
            });
        }

        // ts_utc — parseable as ISO-8601
        time::OffsetDateTime::parse(
            &self.ts_utc,
            &time::format_description::well_known::Iso8601::DEFAULT,
        )
        .map_err(|_| Error::InvalidTimestamp(self.ts_utc.clone()))?;

        // hardware_tag
        if self.hardware_tag.is_empty() {
            return Err(Error::InvalidIngestField {
                field: "hardware_tag".to_string(),
                message: "must not be empty".to_string(),
            });
        }

        // prompt
        match &self.prompt {
            PromptRef::ByBody { name, body, .. } => {
                if name.is_empty() {
                    return Err(Error::InvalidPrompt("name must not be empty".to_string()));
                }
                if body.is_null() {
                    return Err(Error::InvalidPrompt("body must not be null".to_string()));
                }
            }
            PromptRef::BySha256 { sha256 } => {
                if sha256.len() != 64 || !sha256.chars().all(|c| c.is_ascii_hexdigit()) {
                    return Err(Error::InvalidPrompt(format!(
                        "sha256 must be a 64-character hex string, got {sha256:?}"
                    )));
                }
            }
        }

        // temperature range (strict)
        if let Some(t) = self.temperature {
            if !(0.0..=2.0).contains(&t) {
                return Err(Error::InvalidIngestField {
                    field: "temperature".to_string(),
                    message: format!("must be in 0.0..=2.0, got {t}"),
                });
            }
        }

        // metrics non-empty
        if self.metrics.is_empty() {
            return Err(Error::NoMeasurements);
        }

        // at least one non-null value
        let has_measurement = self.metrics.iter().any(|m| m.value.is_some());
        if !has_measurement {
            return Err(Error::NoMeasurements);
        }

        // every metric name in registry
        for entry in &self.metrics {
            registry::lookup(&entry.name)?;
        }

        Ok(())
    }

    /// Returns the subset of metrics with non-null values, in insertion order.
    ///
    /// Used by the recorder to know what observations to write.
    pub fn measured_metrics(&self) -> impl Iterator<Item = &MetricEntry> {
        self.metrics.iter().filter(|m| m.value.is_some())
    }
}

// ── Prompt hashing ────────────────────────────────────────────────────────────

/// SHA-256 of the canonical JSON serialization of `body`.
///
/// If `body` is a JSON string, serde_json serializes it as `"<content>"` (with
/// quotes) — so `prompt_body_sha256(json!("foo"))` and
/// `prompt_body_sha256(json!(["foo"]))` produce different hashes, as expected.
/// The sha256 is stable across runs for identical input values.
pub fn prompt_body_sha256(body: &serde_json::Value) -> String {
    use sha2::{Digest, Sha256};
    #[allow(
        clippy::expect_used,
        reason = "serde_json::Value always serializes to valid JSON — no custom Serialize impls, no IO, infallible"
    )]
    let canonical = serde_json::to_vec(body).expect("serde_json::Value always serializes");
    let digest = Sha256::digest(&canonical);
    // write!(String) is infallible — let _ discards the unit Ok.
    digest.iter().fold(String::new(), |mut acc, b| {
        let _ = write!(acc, "{b:02x}");
        acc
    })
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "ingest_tests.rs"]
mod tests;
