//! Universal §8.5 ingest contract — the JSON shape every backend emits.
//!
//! Every benchmark backend (rMLX, mlx_lm, paroquant, omlx, ollama) writes one
//! JSON file per run in this shape, which [`crate::recorder`] then ingests
//! into the `observations` SQLite table.
//!
//! # Public API
//!
//! - [`RunRecord`] — top-level envelope: identity fields + metric entries.
//! - [`RunRecordBuilder`] — the only way to *construct* a `RunRecord` in Rust
//!   from outside this crate. See below for the other, deliberately open, door.
//! - [`PromptRef`] — either an inline prompt body or a SHA-256 reference to
//!   a prompt already registered in the `prompts` table.
//! - [`MetricEntry`] — one measurement: name, value, unit, direction.
//! - [`prompt_body_sha256`] — canonical SHA-256 of a JSON prompt body,
//!   used to content-address the `prompts` table.
//!
//! # Two ways to get a `RunRecord`, one gate
//!
//! `RunRecord` is `#[non_exhaustive]`, so a struct literal outside this crate
//! is a compile error — Rust emitters go through [`RunRecordBuilder::rmlx`],
//! which fills identity and canonicalization itself and leaves the caller
//! only the measurement. Twelve independent emitters each hand-rolling the
//! identity block is what let `backend_version` rot into NULLs, `'0.0.1'`
//! literals and raw git SHAs.
//!
//! But `RunRecord` also derives `Deserialize`, and that door is deliberately
//! open: shell/Python emitters, foreign backends (`llama_cpp`, `mlx_lm`, …),
//! and `--replay-pending` all produce a `RunRecord` by parsing §8.5 JSON, with
//! a caller-chosen identity block — replaying a buffer file must reproduce
//! the *emitting* build's identity, not re-stamp whoever replays it (see
//! `docs/METRICS_DB.md` §8.5.1). What makes this safe is not that the
//! identity is unforgeable — a hand-edited buffer file can claim anything
//! semver-shaped — it is that [`RunRecord::validate`] runs on every ingest
//! path and is the actual gate. The builder is the *convenient*, correct-by
//! -construction way to get one in Rust; `Deserialize` is the *open*,
//! validated-on-the-way-in way everyone else gets one.

use std::fmt::Write as _;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::identity::{self, RunIdentity};
use crate::registry;
use crate::time_util::now_iso8601;

/// Wire-format version of the §8.5 record shape understood by this binary.
///
/// Bump when the shape changes incompatibly. A record declaring a *higher*
/// version is rejected loudly rather than silently mis-parsed; a record with
/// no `schema_version` at all is assumed to be v1 (the shape that predates
/// this field), so archived buffer files still replay.
pub const RECORD_SCHEMA_VERSION: u32 = 1;

fn default_schema_version() -> u32 {
    1
}

// ── Core record ───────────────────────────────────────────────────────────────

/// Top-level §8.5 run record emitted by every benchmark backend (see docs/METRICS_DB.md §8.5).
///
/// `#[non_exhaustive]`: construct via [`RunRecordBuilder`] (Rust) or deserialize
/// from the §8.5 JSON (scripts, other backends). A struct literal outside
/// `rmlx-metrics` is a compile error, by design.
///
/// The five identity/provenance fields (`backend`, `backend_version`,
/// `git_sha`, `build_profile`, `hardware_tag`) are `pub(crate)`, not `pub`:
/// `#[non_exhaustive]`
/// blocks a struct *literal* from outside this crate, but every field was
/// still individually mutable — `let mut r = builder.build()?; r.backend_version
/// = Some("0.0.1".into());` compiled fine and bypassed the validator entirely,
/// since `build()` validates once, at construction, not on every field write.
/// Read them via the getters below; write them only through
/// [`RunRecordBuilder`] or (in-crate) direct construction.
///
/// This does **not** mean every historical failure mode is now structurally
/// impossible: [`RunRecord::validate`] checks that `backend_version` is
/// semver-*shaped*, not that it is authentic. A fabricated-but-well-formed
/// `"0.0.1"` from a hand-written JSON buffer file still ingests — the fields
/// being non-`pub` only closes the in-Rust mutation hole, it is not a
/// cryptographic attestation of provenance.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    /// Wire-format version of this record. Defaults to 1 when absent.
    #[serde(default = "default_schema_version")]
    pub schema_version: u32,
    /// Canonical backend identifier (e.g. `"rmlx"`, `"mlx_lm"`). Read via [`RunRecord::backend`].
    pub(crate) backend: String,
    /// Semver string of the backend binary, if known. Read via [`RunRecord::backend_version`].
    #[serde(default)]
    pub(crate) backend_version: Option<String>,
    /// Model namespace from the whitelist (e.g. `"mlx-community"`).
    pub model_namespace: String,
    /// Model repository name within the namespace.
    ///
    /// `model_id` is accepted as a deserialize alias for this field: the
    /// canonical §8.5 wire key is `model` (see `docs/METRICS_DB.md` §8.5),
    /// but the recorder tolerates a `model_id`-keyed record too, so a
    /// differently-named emitter does not silently drop into
    /// `metrics/buffer/failed/`.
    #[serde(alias = "model_id")]
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
    /// Commit SHA the caller attributes this run to, if supplied. The binary
    /// never derives this itself (see `rmlx_metrics::identity::RunIdentity`'s
    /// doc); it is ordinary caller-supplied provenance, exactly like
    /// `hardware_tag`. Read via [`RunRecord::git_sha`].
    #[serde(default)]
    pub(crate) git_sha: Option<String>,
    /// Cargo build profile (e.g. `"release"`, `"release-perf"`). Read via [`RunRecord::build_profile`].
    #[serde(default)]
    pub(crate) build_profile: Option<String>,
    /// Hardware tag identifying the test machine (e.g. `"m5_max_128gb"`). Read via [`RunRecord::hardware_tag`].
    pub(crate) hardware_tag: String,
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
    /// The engine settings this run moved off their defaults.
    ///
    /// Part of the `bests` cell key: a run at a non-default configuration
    /// answers a different question from one at the defaults for the same
    /// model, quant and prompt, so ranking their rates against each other
    /// publishes the one as the other's. A speculative arm was the first
    /// setting to need this and is not the only one — a swept prefill chunk
    /// files here too, e.g. `"mtp/block=5"` or `"prefill_chunk=1024"`.
    ///
    /// `None` is every setting at its default. A value is validated against
    /// the §3.2 grammar by
    /// [`crate::cell::decode_config_is_well_formed`] in [`RunRecord::validate`]:
    /// the column is cell identity, so two spellings of one configuration
    /// would split its measurements into two cells that never rank against
    /// each other.
    #[serde(default)]
    pub decode_config: Option<String>,
    /// Free-form run notes (auto-summary, legacy keys, etc.).
    #[serde(default)]
    pub notes: Option<String>,
    /// Human-readable description of the run (e.g. `"sha1234: add KV quant"`).
    #[serde(default)]
    pub description: Option<String>,
    /// One entry per measured metric; `value = None` entries are skipped.
    pub metrics: Vec<MetricEntry>,
}

/// The marker a record carries to say it is not a measurement.
///
/// A probe that exercises the ingest path — "does `validate` still refuse a
/// malformed `decode_config`?" — needs a record to hand it, and the shortest
/// way to build one is to copy a real record and change a field. That record
/// still carries real identity, so if the probe's expectation is wrong the row
/// lands in a live cell under a placeholder value and wins it. Two such rows
/// reached this DB that way; they are named in docs/METRICS_DB.md and cannot be
/// taken back out, because the table is append-only.
///
/// So a record may declare itself. Put this anywhere in `notes` or
/// `description` and [`RunRecord::validate`] refuses it, whatever else is
/// right about it — the refusal is the point, and it happens before any
/// transaction opens.
///
/// The other route, and the better one when it fits, is `rmlx metrics record
/// --dry-run`: it runs the whole of `validate` and returns before the commit,
/// so a probe that only needs to know *whether* a record is accepted never
/// builds a writing path at all.
pub const SYNTHETIC_MARKER: &str = "synthetic=true";

impl RunRecord {
    /// Canonical backend identifier (e.g. `"rmlx"`, `"mlx_lm"`).
    pub fn backend(&self) -> &str {
        &self.backend
    }

    /// Semver string of the backend binary, if known.
    pub fn backend_version(&self) -> Option<&str> {
        self.backend_version.as_deref()
    }

    /// Commit SHA the caller attributes this run to, if supplied.
    pub fn git_sha(&self) -> Option<&str> {
        self.git_sha.as_deref()
    }

    /// Cargo build profile (e.g. `"release"`, `"release-perf"`), if known.
    pub fn build_profile(&self) -> Option<&str> {
        self.build_profile.as_deref()
    }

    /// Hardware tag identifying the test machine (e.g. `"m5_max_128gb"`).
    pub fn hardware_tag(&self) -> &str {
        &self.hardware_tag
    }
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

// ── Identity policy ───────────────────────────────────────────────────────────

/// How strictly the §8.5 run-identity fields are enforced.
///
/// The strict rule exists because `backend_version` is `Option` with
/// `#[serde(default)]`: an emitter that forgets the key deserializes to `None`
/// and the row lands with a silent NULL. Enforcing at ingest — the one place
/// every writer funnels through — is what stops the next emitter regressing.
///
/// `pub(crate)`: the `LegacyArchive` variant is the one door around the
/// identity check (see [`crate::recorder::Recorder::legacy_archive`]), and
/// keeping the whole enum crate-private means no crate outside `rmlx-metrics`
/// can even name it, let alone select it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed policy — live ingest vs. one-shot legacy import; a third mode would be a contract change"
)]
pub(crate) enum IdentityPolicy {
    /// Live ingest. An `rmlx` record MUST carry a semver `backend_version`.
    /// Every path that records a new measurement uses this.
    #[default]
    Enforce,
    /// One-shot import of pre-contract archives (`rmlx metrics migrate`).
    ///
    /// Those rows predate the identity contract and genuinely have no version
    /// to state; fabricating one is exactly the bug this policy exists to
    /// prevent. NEVER use for a new measurement.
    LegacyArchive,
}

// ── RunRecord impl ────────────────────────────────────────────────────────────

impl RunRecord {
    /// Drops metric entries the §4 registry cannot read as a measurement,
    /// returning how many were dropped.
    ///
    /// For the *archive* converters only (`migrate::legacy`). Those tools'
    /// CSV/JSONL exports write `0.0` in a column they did not measure, and a
    /// whole historical run should not be lost because one column carries
    /// that placeholder. Live ingest gets no such courtesy: [`Self::validate`]
    /// rejects the record so the emitter is fixed rather than the number
    /// quietly dropped.
    pub(crate) fn drop_implausible_metrics(&mut self) -> usize {
        let before = self.metrics.len();
        self.metrics.retain(|entry| match entry.value {
            Some(value) => match registry::bounds(&entry.name) {
                Ok(bounds) => bounds.contains(value),
                // An unregistered name is not this function's business — keep
                // it so `validate` still rejects it by name.
                Err(_) => true,
            },
            None => true,
        });
        before - self.metrics.len()
    }

    /// Validates per §8.5 required fields + §4 metric registry + §5
    /// whitelists (`backend`, `weight_quant`), enforcing the run-identity
    /// contract ([`IdentityPolicy::Enforce`]). `model_namespace`, `model`,
    /// and `kv_quant` are free-form recorded labels, not whitelisted — see
    /// `identity::canonicalize_kv_quant`.
    ///
    /// Does NOT touch the DB. Returns `Ok(())` if the record is structurally
    /// accepted; returns the first specific error encountered otherwise.
    pub fn validate(&self) -> Result<()> {
        self.validate_with(IdentityPolicy::Enforce)
    }

    /// [`RunRecord::validate`] with an explicit identity policy.
    ///
    /// `pub(crate)`: the policy choice is not a caller-facing knob. Every
    /// external caller gets [`RunRecord::validate`] (always [`IdentityPolicy::Enforce`]).
    pub(crate) fn validate_with(&self, policy: IdentityPolicy) -> Result<()> {
        // schema_version: a record from the future is a loud failure, never a
        // silent mis-parse of fields this binary does not know about.
        if self.schema_version > RECORD_SCHEMA_VERSION {
            return Err(Error::InvalidIngestField {
                field: "schema_version".to_string(),
                message: format!(
                    "record declares v{}, this binary understands up to v{RECORD_SCHEMA_VERSION} — upgrade rmlx",
                    self.schema_version
                ),
            });
        }

        // A record that says it is not a measurement is not stored, however
        // well-formed the rest of it is. Checked first: every other rule below
        // asks whether the record describes a real run correctly, and this one
        // asks whether it claims to describe a run at all.
        for (field, value) in [
            ("notes", self.notes.as_deref()),
            ("description", self.description.as_deref()),
        ] {
            if value.is_some_and(|v| v.contains(SYNTHETIC_MARKER)) {
                return Err(Error::InvalidIngestField {
                    field: field.to_string(),
                    message: format!(
                        "record is marked `{SYNTHETIC_MARKER}`, so it is not a measurement and \
                         is not stored. `observations` is append-only: a placeholder value that \
                         reaches a live cell wins it and cannot be removed. To check whether a \
                         record would be accepted, use `rmlx metrics record --dry-run`, which \
                         runs this whole validation and returns before the commit."
                    ),
                });
            }
        }

        // backend
        identity::canonicalize("backend", &self.backend, identity::BACKEND_WHITELIST)?;

        // backend_version: rMLX is our own binary, so it always knows its own
        // semver. Other backends legitimately have none (llama.cpp emits a
        // build_commit) — leave the field free-form and optional for them.
        if policy == IdentityPolicy::Enforce && self.backend == "rmlx" {
            match self.backend_version.as_deref() {
                Some(v) if identity::is_semver(v) => {}
                Some(v) => {
                    return Err(Error::MissingBackendVersion {
                        got: format!("{v:?}"),
                    })
                }
                None => {
                    return Err(Error::MissingBackendVersion {
                        got: "<null>".to_string(),
                    })
                }
            }
        }

        // model_namespace / model: free-form recorded labels, same as
        // kv_quant (see `identity::canonicalize_kv_quant`) — not validated
        // against `NAMESPACE_WHITELIST`. An unrecognized namespace (a new
        // model host, a typo, a local finetune) must still record, not
        // silently drop into `metrics/buffer/failed/`. `model` was already
        // unchecked here; `model_namespace` now matches it.

        // weight_quant
        identity::canonicalize(
            "weight_quant",
            &self.weight_quant,
            identity::WEIGHT_QUANT_WHITELIST,
        )?;

        // kv_quant: free-form recorded label, not validated — see
        // `identity::canonicalize_kv_quant`. Always `Ok`; called for the
        // alias normalization (`bf16`/`f16` → `none`), not as a gate.
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

        // decode_config — cell identity, so its spelling is a contract and not
        // a label. Two emitters describing one engine configuration in two
        // spellings put its measurements in two cells, where neither ranks
        // against the other and both look like champions.
        if let Some(config) = self.decode_config.as_deref() {
            if !crate::cell::decode_config_is_well_formed(config) {
                return Err(Error::InvalidIngestField {
                    field: "decode_config".to_string(),
                    message: format!(
                        "must be `key=value` terms joined by `,` and ordered by key, got {config:?}"
                    ),
                });
            }
            // Refused rather than normalised, and loudly: a caller that spells
            // out the shipped default has misunderstood what the column is
            // for, and silently rewriting the value to NULL would let the next
            // campaign make the same mistake at scale. The message names the
            // defaults so the fix is obvious.
            if crate::cell::decode_config_is_all_defaults(config) {
                return Err(Error::InvalidIngestField {
                    field: "decode_config".to_string(),
                    message: format!(
                        "{config:?} spells the engine's own defaults ({}); NULL is how a \
                         run at the defaults is recorded, and a second spelling of one \
                         configuration puts its measurements in a cell that ranks against \
                         nothing. Omit the field.",
                        rmlx_core::kv_boundary::DECODE_CONFIG_NUMERIC_DEFAULTS
                            .iter()
                            .map(|&(key, value)| format!("{key}={value}"))
                            .collect::<Vec<_>>()
                            .join(","),
                    ),
                });
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

        // every metric name in registry, and every value a possible
        // measurement of it. A null value is a deliberate "not measured" and
        // is skipped by the recorder; a fabricated stand-in (a zero rate, an
        // arithmetic accident orders of magnitude out of range) is not, and it
        // outranks every real row in `bests` the moment it lands.
        for entry in &self.metrics {
            registry::lookup(&entry.name)?;
            if let Some(value) = entry.value {
                let bounds = registry::bounds(&entry.name)?;
                if !bounds.contains(value) {
                    return Err(Error::ImplausibleValue {
                        metric: entry.name.clone(),
                        value,
                        bounds: bounds.describe(),
                    });
                }
            }
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

// ── Builder ───────────────────────────────────────────────────────────────────

/// The one way to build a §8.5 record in Rust.
///
/// [`RunRecordBuilder::rmlx`] fills everything the caller has no business
/// choosing:
///
/// * the whole identity block, from [`RunIdentity::rmlx`] (`backend`,
///   `backend_version`, `build_profile`, `hardware_tag` — NOT `git_sha`,
///   which the binary cannot honestly derive; it stays `None` unless a
///   future caller adds a `.git_sha(...)` builder method for it);
/// * `model_namespace` / `model` / `weight_quant`, parsed and inferred from the
///   model id ([`identity::split_model_id`], [`identity::infer_weight_quant`]);
/// * `kv_quant`, canonicalized ([`identity::canonicalize_kv_quant`]);
/// * `ts_utc`, as an ISO-8601 UTC stamp of now;
/// * `schema_version`.
///
/// The caller supplies the measurement and nothing else. Adding a new metric is
/// therefore a one-liner — `.metric("decode_tps_warm", Some(119.1), None)` — and
/// there is no identity code to get wrong.
#[derive(Debug, Clone)]
pub struct RunRecordBuilder {
    rec: RunRecord,
}

impl RunRecordBuilder {
    /// Begin a record produced by *this* rMLX binary.
    ///
    /// `model_id` is a snapshot id or path (`"mlx-community__gemma-4-e2b-it-mxfp8"`,
    /// or an absolute snapshot path); `kv_quant` is any form
    /// [`identity::canonicalize_kv_quant`] accepts.
    ///
    /// Returns `Err` when `kv_quant` does not canonicalize — silently falling
    /// back to `"none"` would file an unrecognised KV-quant label (a typo, a
    /// future codec) under the champion cell for *unquantised* KV. The caller
    /// (server drainer, `baseline`, `eval`) already handles a builder `Err` by
    /// skipping the record with a warning; propagate rather than mislabel.
    pub fn rmlx(model_id: &str, kv_quant: &str, ctx_max: i64, prompt: PromptRef) -> Result<Self> {
        let ident = RunIdentity::rmlx();
        let (model_namespace, model) = identity::split_model_id(model_id);
        Ok(Self {
            rec: RunRecord {
                schema_version: RECORD_SCHEMA_VERSION,
                backend: ident.backend,
                backend_version: Some(ident.backend_version),
                model_namespace,
                model,
                weight_quant: identity::infer_weight_quant(model_id),
                kv_quant: identity::canonicalize_kv_quant(kv_quant)?,
                ctx_max,
                prompt,
                ts_utc: now_iso8601()?,
                // Caller-supplied provenance, not something the binary
                // derives (see `RunIdentity`'s doc). No caller of this
                // builder has a git-sha input today (the server drainer has
                // no `--git-sha` flag), so this is always `None` — a future
                // caller that gains one should add a `.git_sha(...)` builder
                // method rather than reach into `rec` directly.
                git_sha: None,
                build_profile: Some(ident.build_profile),
                hardware_tag: ident.hardware_tag,
                prompt_tokens: None,
                max_tokens: None,
                temperature: None,
                seed: None,
                n_warmups: None,
                n_measure: None,
                output_first_64: None,
                notes: None,
                description: None,
                decode_config: None,
                metrics: Vec::new(),
            },
        })
    }

    /// Override the weight quant inferred from the model id.
    #[must_use]
    pub fn weight_quant(mut self, wq: impl Into<String>) -> Self {
        self.rec.weight_quant = wq.into();
        self
    }

    /// Override the measurement timestamp (defaults to now).
    ///
    /// Used by the server drainer, where the event was measured earlier than
    /// the batch flush that records it.
    #[must_use]
    pub fn ts_utc(mut self, ts: impl Into<String>) -> Self {
        self.rec.ts_utc = ts.into();
        self
    }

    /// Bench-config counters: prompt/generated token counts.
    #[must_use]
    pub fn tokens(mut self, prompt_tokens: Option<i64>, max_tokens: Option<i64>) -> Self {
        self.rec.prompt_tokens = prompt_tokens;
        self.rec.max_tokens = max_tokens;
        self
    }

    /// Bench-config sampling parameters.
    #[must_use]
    pub fn sampling(mut self, temperature: Option<f64>, seed: Option<i64>) -> Self {
        self.rec.temperature = temperature;
        self.rec.seed = seed;
        self
    }

    /// Bench-config run counts: warmups discarded, measurements averaged.
    #[must_use]
    pub fn runs(mut self, n_warmups: Option<i64>, n_measure: Option<i64>) -> Self {
        self.rec.n_warmups = n_warmups;
        self.rec.n_measure = n_measure;
        self
    }

    /// First ≤64 chars of generated output, for temp=0 equality probes.
    #[must_use]
    pub fn output_first_64(mut self, s: impl Into<String>) -> Self {
        self.rec.output_first_64 = Some(s.into());
        self
    }

    /// Machine-written auto-summary.
    #[must_use]
    pub fn notes(mut self, s: impl Into<String>) -> Self {
        self.rec.notes = Some(s.into());
        self
    }

    /// Human/Claude-written analysis of the run.
    #[must_use]
    pub fn description(mut self, s: impl Into<String>) -> Self {
        self.rec.description = Some(s.into());
        self
    }

    /// Append one measurement. `value = None` is recorded as a sparse skip.
    #[must_use]
    pub fn metric(
        mut self,
        name: impl Into<String>,
        value: Option<f64>,
        stddev: Option<f64>,
    ) -> Self {
        self.rec.metrics.push(MetricEntry {
            name: name.into(),
            value,
            stddev,
        });
        self
    }

    /// Append many measurements at once.
    #[must_use]
    pub fn metrics(mut self, entries: impl IntoIterator<Item = MetricEntry>) -> Self {
        self.rec.metrics.extend(entries);
        self
    }

    /// Validate and finish. Fails exactly as ingest would — a record that
    /// cannot be built could not have been recorded either.
    pub fn build(self) -> Result<RunRecord> {
        self.rec.validate()?;
        Ok(self.rec)
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
