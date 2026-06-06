//! Legacy buffer-file schema support.
//!
//! Two converters live here:
//!
//! 1. [`LegacyRunRecord`] / [`try_parse_legacy`] — handles the legacy bench-script
//!    shape (t3_final_bench, final_matrix_bench, gemma_matrix_bench).
//!
//! 2. [`LegacyCbbRecord`] / [`try_parse_cbb`] — handles the May-10 CBB runner
//!    shape which is structurally §8.5 but has compound `weight_quant` strings
//!    (e.g. `"mxfp8 g32 + kv-k8v8"`) and non-canonical backend names
//!    (e.g. `"mlx-lm-turboquant"`).
//!
//! The dispatcher in `crates/rmlx-cli/src/commands/metrics.rs` tries parsers in
//! order:
//! canonical §8.5 → try_parse_legacy → try_parse_cbb (CBB May-10)
//!
//! # Legacy bench-script key differences vs canonical §8.5
//!
//! - `model_name` instead of `model`
//! - `max_ctx` instead of `ctx_max`
//! - `observations[].metric` uses short names: `decode_tps`, `ttft_ms`
//! - `observations[].direction` uses `higher_is_better` / `lower_is_better`
//! - `prompt_body` (raw JSON value) instead of `prompt: {name, body}`
//! - Missing: `hardware_tag`, `build_profile`, `temperature`, `seed`,
//!   `n_warmups`, `n_measure`, `output_first_64`
//!
//! Defaults applied during conversion:
//! - `hardware_tag` → `"m5_max_128gb"`
//! - `build_profile` → `None`
//! - `prompt.name` → `"longctx_4k"` (all legacy bench runs used this prompt)
//! - `output_first_64` → from `first_32_tokens` list (joined, truncated)
//!
//! Metric name mapping:
//! - `decode_tps` → `decode_tps_warm`
//! - `ttft_ms` → `ttft_warm_ms`
//! - `prefill_tps` → `prefill_tps` (unchanged)

use serde::Deserialize;
use serde_json::Value;

use crate::ingest::{MetricEntry, PromptRef, RunRecord};

// ── Legacy observation entry ──────────────────────────────────────────────────

/// One metric measurement from a pre-§8.5 bench-script buffer file.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub struct LegacyObservation {
    /// Metric name (may be a short alias; converted via `map_metric_name`).
    pub metric: String,
    /// Measured value (unit/direction inferred from the registry).
    pub value: f64,
    // unit, direction, run_type, notes — all ignored; registry provides unit/direction
}

// ── Legacy run record ────────────────────────────────────────────────────────

/// Subset of fields actually present in legacy bench-script buffer files.
#[derive(Debug, Deserialize)]
#[non_exhaustive]
pub struct LegacyRunRecord {
    /// Canonical backend identifier (e.g. `"rmlx"`, `"mlx_lm"`).
    pub backend: String,
    /// Semver string of the backend binary, if present.
    #[serde(default)]
    pub backend_version: Option<String>,
    /// Model namespace from the identity whitelist.
    pub model_namespace: String,
    /// Old field name. §8.5 uses `model`.
    pub model_name: String,
    /// Canonical weight quantization string.
    pub weight_quant: String,
    /// KV-cache quantization string (pre-canonical; may be `"bf16"` meaning `"none"`).
    pub kv_quant: String,
    /// Old field name. §8.5 uses `ctx_max`.
    pub max_ctx: i64,
    /// Optional: present in scripts that loaded a prompt file.
    #[serde(default)]
    pub prompt_body: Option<Value>,
    /// ISO-8601 UTC timestamp string.
    pub ts_utc: String,
    /// Short git SHA of the backend binary, if captured.
    #[serde(default)]
    pub git_sha: Option<String>,
    /// Number of prompt tokens, if counted by the bench harness.
    #[serde(default)]
    pub prompt_tokens: Option<i64>,
    /// Maximum output tokens requested.
    #[serde(default)]
    pub max_tokens: Option<i64>,
    /// Observation entries for this run.
    pub observations: Vec<LegacyObservation>,
    /// Optional list of tokens; joined to produce `output_first_64`.
    #[serde(default)]
    pub first_32_tokens: Vec<Value>,
}

// ── Conversion ───────────────────────────────────────────────────────────────

impl LegacyRunRecord {
    /// Convert to a canonical [`RunRecord`].
    ///
    /// Returns `None` only if `observations` is empty (nothing to record).
    pub fn to_run_record(self) -> Option<RunRecord> {
        // Normalize kv_quant: the legacy bench scripts used "bf16" to mean
        // "no KV quantization". The §5.3 canonical value is "none".
        let kv_quant = map_kv_quant(&self.kv_quant);

        // Map short metric names to §4 canonical names.
        let metrics: Vec<MetricEntry> = self
            .observations
            .into_iter()
            .map(|obs| MetricEntry {
                name: map_metric_name(&obs.metric),
                value: Some(obs.value),
                stddev: None,
            })
            .collect();

        if metrics.is_empty() {
            return None;
        }

        // Build PromptRef.
        let prompt = match self.prompt_body {
            Some(body) => PromptRef::ByBody {
                name: "longctx_4k".to_string(),
                body,
                notes: None,
                tokens_approx: self.prompt_tokens,
            },
            None => {
                // No prompt body recorded. Use a synthetic placeholder body
                // so the prompt registry can store it. All legacy bench runs
                // used the longctx_4k CBB prompt; we record the body as a
                // placeholder string to keep the FK happy.
                PromptRef::ByBody {
                    name: "longctx_4k".to_string(),
                    body: Value::String(
                        "<legacy buffer record — prompt body not captured>".to_string(),
                    ),
                    notes: Some("legacy: bench script did not capture prompt_body".to_string()),
                    tokens_approx: self.prompt_tokens,
                }
            }
        };

        // Build output_first_64 from first_32_tokens list.
        let output_first_64 = if self.first_32_tokens.is_empty() {
            None
        } else {
            let words: Vec<String> = self
                .first_32_tokens
                .iter()
                .filter_map(|v| v.as_str().map(str::to_owned))
                .collect();
            if words.is_empty() {
                None
            } else {
                Some(words.join(" ").chars().take(64).collect())
            }
        };

        Some(RunRecord {
            backend: self.backend,
            backend_version: self.backend_version,
            model_namespace: self.model_namespace,
            model: self.model_name,
            weight_quant: self.weight_quant,
            kv_quant,
            ctx_max: self.max_ctx,
            prompt,
            ts_utc: self.ts_utc,
            git_sha: self.git_sha,
            build_profile: None,
            hardware_tag: "m5_max_128gb".to_string(),
            prompt_tokens: self.prompt_tokens,
            max_tokens: self.max_tokens,
            temperature: None,
            seed: None,
            n_warmups: None,
            n_measure: None,
            output_first_64,
            notes: Some("ingested from legacy buffer (schema pre-§8.5)".to_string()),
            description: None,
            metrics,
        })
    }
}

/// Map legacy short metric names to §4 canonical names.
fn map_metric_name(name: &str) -> String {
    match name {
        "decode_tps" => "decode_tps_warm",
        "ttft_ms" => "ttft_warm_ms",
        // prefill_tps is already canonical; others pass through unchanged.
        other => other,
    }
    .to_string()
}

/// Normalize legacy kv_quant strings to §5.3 canonical values.
///
/// The legacy bench scripts used `"bf16"` to mean "no KV quantization".
/// The §5.3 canonical value for that is `"none"`.
fn map_kv_quant(kv: &str) -> String {
    match kv {
        "bf16" => "none",
        other => other,
    }
    .to_string()
}

// ── Try-parse helper (used by metrics.rs ingest_one) ─────────────────────────

/// Attempt to parse `json_text` as a [`LegacyRunRecord`] and convert it.
///
/// Returns `None` if the JSON doesn't look like a legacy record (i.e. the
/// `model_name` field is absent) so the caller can surface the original §8.5
/// parse error instead.
pub fn try_parse_legacy(json_text: &str) -> Option<RunRecord> {
    // Quick pre-check: legacy records have `model_name`, canonical ones have `model`.
    // Avoid a full parse if this is clearly a canonical record.
    let raw: serde_json::Map<String, Value> = serde_json::from_str(json_text).ok()?;
    if !raw.contains_key("model_name") {
        return None;
    }
    let legacy: LegacyRunRecord = serde_json::from_str(json_text).ok()?;
    legacy.to_run_record()
}

// ── CBB runner record (May-10 epoch) ─────────────────────────────────────────

/// Buffer file shape emitted by the May-10 CBB runner.
///
/// The record is structurally almost identical to canonical §8.5 but:
/// - `weight_quant` is a compound string like `"mxfp8 g32 + kv-k8v8"` that
///   encodes both the base weight quantization and the KV quantization mode.
/// - `backend` uses human-readable names (`"mlx-lm-turboquant"`, `"mlx-lm"`)
///   rather than the canonical underscore form (`"mlx_lm_tq"`, `"mlx_lm"`).
/// - `kv_quant` is always `"none"` (KV info is already in `weight_quant`).
///
/// Conversion rules:
/// - [`parse_cbb_weight_quant`] splits the compound string and returns
///   `(canonical_weight_quant, canonical_kv_quant)`.
/// - [`map_cbb_backend`] maps the display name to the canonical identifier.
#[derive(Debug, serde::Deserialize)]
#[non_exhaustive]
pub struct LegacyCbbRecord {
    /// Human-readable backend name (e.g. `"mlx-lm-turboquant"`); converted by `map_cbb_backend`.
    pub backend: String,
    /// Backend version string, if present.
    #[serde(default)]
    pub backend_version: Option<String>,
    /// Model namespace from the identity whitelist.
    pub model_namespace: String,
    /// Model repository name within the namespace.
    pub model: String,
    /// Compound string, e.g. `"mxfp8 g32 + kv-k8v8"` or bare `"8bit"`.
    pub weight_quant: String,
    /// Maximum context length for this bench run (tokens).
    pub ctx_max: i64,
    /// Always `"none"` in CBB files; KV mode is encoded in `weight_quant`.
    #[serde(default)]
    pub kv_quant: Option<String>,
    /// Prompt object; extracted via §8.5 `{name, body}` keys.
    pub prompt: Value,
    /// ISO-8601 UTC timestamp string.
    pub ts_utc: String,
    /// Short git SHA of the backend binary, if captured.
    #[serde(default)]
    pub git_sha: Option<String>,
    /// Cargo build profile (e.g. `"release-perf"`), if captured.
    #[serde(default)]
    pub build_profile: Option<String>,
    /// Hardware tag identifying the test machine.
    pub hardware_tag: String,
    /// Number of prompt tokens, if counted.
    #[serde(default)]
    pub prompt_tokens: Option<i64>,
    /// Maximum output tokens requested.
    #[serde(default)]
    pub max_tokens: Option<i64>,
    /// Sampling temperature used.
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
    /// Free-form run notes.
    #[serde(default)]
    pub notes: Option<String>,
    /// Human-readable description of the run.
    #[serde(default)]
    pub description: Option<String>,
    /// Measurement entries — one per tracked metric.
    pub metrics: Vec<MetricEntry>,
}

impl LegacyCbbRecord {
    /// Convert to a canonical [`RunRecord`].
    ///
    /// Returns `None` if:
    /// - `weight_quant` cannot be parsed (unknown base or KV suffix).
    /// - `backend` is not in the CBB → canonical mapping.
    /// - `metrics` is empty.
    pub fn to_run_record(self) -> Option<RunRecord> {
        let backend = map_cbb_backend(&self.backend)?;

        let (weight_quant, kv_quant) = parse_cbb_weight_quant(&self.weight_quant)?;

        if self.metrics.is_empty() {
            return None;
        }

        // Re-use the prompt value as-is: CBB files emit the canonical
        // §8.5 `prompt: {name, body, tokens_approx}` shape.
        let prompt = PromptRef::ByBody {
            name: self
                .prompt
                .get("name")
                .and_then(|v| v.as_str())
                .unwrap_or("longctx_4k")
                .to_string(),
            body: self
                .prompt
                .get("body")
                .cloned()
                .unwrap_or(self.prompt.clone()),
            notes: None,
            tokens_approx: self.prompt_tokens,
        };

        Some(RunRecord {
            backend,
            backend_version: self.backend_version,
            model_namespace: self.model_namespace,
            model: self.model,
            weight_quant,
            kv_quant,
            ctx_max: self.ctx_max,
            prompt,
            ts_utc: self.ts_utc,
            git_sha: self.git_sha,
            build_profile: self.build_profile,
            hardware_tag: self.hardware_tag,
            prompt_tokens: self.prompt_tokens,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            seed: self.seed,
            n_warmups: self.n_warmups,
            n_measure: self.n_measure,
            output_first_64: self.output_first_64,
            notes: Some(match self.notes.as_deref() {
                Some(n) => format!("{n} (ingested from CBB May-10 buffer)"),
                None => "ingested from CBB May-10 buffer (compound weight_quant)".to_string(),
            }),
            description: self.description,
            metrics: self.metrics,
        })
    }
}

// ── CBB mapping helpers ───────────────────────────────────────────────────────

/// Map a CBB display backend name to the canonical §5 identifier.
///
/// Returns `None` for unrecognized names (record should be skipped).
fn map_cbb_backend(backend: &str) -> Option<String> {
    let canonical = match backend {
        "rmlx" => "rmlx",
        "mlx-lm" | "mlx_lm" => "mlx_lm",
        "mlx-lm-turboquant" | "mlx_lm_tq" => "mlx_lm_tq",
        "omlx" => "omlx",
        "paroquant" => "paroquant",
        "ollama" => "ollama",
        "vllm" => "vllm",
        _ => return None,
    };
    Some(canonical.to_string())
}

/// Parse a CBB compound `weight_quant` string into `(canonical_weight, canonical_kv)`.
///
/// Accepted forms:
/// - `"<base> [tokens...] + kv-<kv>"` — split on ` + `.
/// - Bare canonical weight (no ` + `) — treated as `kv_quant = "none"`.
///
/// Base → canonical mapping:
/// | CBB base | canonical |
/// |-------------------|-----------|
/// | `mxfp8 …` | `mxfp8` |
/// | `affine …` | `8bit` |
/// | `2-bit ternary` | `2bit` |
/// | `8bit` | `8bit` |
/// | `q8_0` | `q8_0` |
/// | `q4_k_m` | `q4_k_m` |
///
/// KV suffix → canonical mapping:
/// | CBB suffix | canonical |
/// |-------------------------------|-----------|
/// | `kv-bf16`, `kv-none` | `none` |
/// | `kv-k8v4`, `kv-k4v4` | `k8v4` |
/// | `kv-k8v8` | `k8v8` |
/// | `kv-planar` | `planar` |
/// | `kv-turbo3`, `kv-turbo3_v4`, | |
/// | `kv-turbo4`, `kv-turbo4_v4` | `turbo4` |
///
/// Returns `None` if the base or KV suffix is unrecognized.
pub fn parse_cbb_weight_quant(raw: &str) -> Option<(String, String)> {
    let raw = raw.trim();

    // Split on the canonical CBB separator ` + `.
    let (base_part, kv_part) = match raw.split_once(" + ") {
        Some((l, r)) => (l.trim(), r.trim()),
        None => (raw, ""),
    };

    let weight = map_cbb_base_weight(base_part)?;

    let kv = if kv_part.is_empty() {
        "none".to_string()
    } else {
        map_cbb_kv_suffix(kv_part)?
    };

    Some((weight, kv))
}

/// Map the base portion of a CBB compound weight string to a canonical value.
fn map_cbb_base_weight(base: &str) -> Option<String> {
    // Match on leading token only (the rest are modifiers like g32, b8, etc.).
    let leading = base.split_whitespace().next().unwrap_or("");
    let canonical = match leading {
        "mxfp8" => "mxfp8",
        "mxfp4" => "mxfp4",
        "affine" | "8bit" => "8bit", // both map to canonical "8bit" affine quant
        "2-bit" => "2bit",
        "3-bit" => "3bit",
        "4-bit" => "4bit",
        "q8_0" => "q8_0",
        "q4_k_m" => "q4_k_m",
        "bf16" => "bf16",
        "fp16" => "fp16",
        "paro" => "paro",
        _ => return None,
    };
    Some(canonical.to_string())
}

/// Map a CBB KV suffix token (the part after ` + `) to a canonical kv_quant.
fn map_cbb_kv_suffix(suffix: &str) -> Option<String> {
    let canonical = match suffix {
        "kv-bf16" | "kv-none" => "none",
        "kv-k8v4" | "kv-k4v4" => "k8v4",
        "kv-k8v8" => "k8v8",
        "kv-planar" => "planar",
        "kv-turbo3" | "kv-turbo3_v4" | "kv-turbo4" | "kv-turbo4_v4" => "turbo4",
        _ => return None,
    };
    Some(canonical.to_string())
}

// ── CBB try-parse helper ──────────────────────────────────────────────────────

/// Attempt to parse `json_text` as a [`LegacyCbbRecord`] and convert it.
///
/// Detection heuristic: CBB files have `hardware_tag` (canonical) AND `model`
/// (canonical, not `model_name`) AND `metrics` (not `observations`), AND a
/// compound `weight_quant` containing `" + "` OR a backend that needs mapping.
///
/// Returns `None` if the JSON doesn't match the CBB shape or conversion fails
/// (unknown base weight, unknown KV suffix, unknown backend).
pub fn try_parse_cbb(json_text: &str) -> Option<RunRecord> {
    // Quick pre-check: CBB files have `hardware_tag` + `model` + `metrics`.
    // If any sentinel is missing this isn't a CBB file.
    let raw: serde_json::Map<String, Value> = serde_json::from_str(json_text).ok()?;
    if !raw.contains_key("hardware_tag")
        || !raw.contains_key("model")
        || !raw.contains_key("metrics")
    {
        return None;
    }
    // Must have at least one field that needs mapping:
    // - compound weight_quant (contains " + ")
    // - non-canonical backend name (e.g. "mlx-lm-turboquant")
    // - legacy kv_quant value (e.g. "bf16" meaning "no KV quant")
    let wq = raw.get("weight_quant")?.as_str().unwrap_or("");
    let be = raw.get("backend")?.as_str().unwrap_or("");
    let kv = raw.get("kv_quant").and_then(|v| v.as_str()).unwrap_or("");
    let needs_weight_map = wq.contains(" + ");
    let needs_backend_map = ![
        "rmlx",
        "mlx_lm",
        "mlx_lm_tq",
        "omlx",
        "paroquant",
        "ollama",
        "vllm",
    ]
    .contains(&be);
    // "bf16" in kv_quant is a CBB-era alias for "none" (no KV quantization).
    let needs_kv_map = kv == "bf16";
    if !needs_weight_map && !needs_backend_map && !needs_kv_map {
        return None;
    }

    let cbb: LegacyCbbRecord = serde_json::from_str(json_text).ok()?;
    cbb.to_run_record()
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "legacy_ingest_tests.rs"]
mod tests;
