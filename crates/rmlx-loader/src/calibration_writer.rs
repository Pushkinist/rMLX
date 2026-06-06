//! KV calibration schema: `kv_calib.json` writer + reader.
//!
//! Defines the v1 schema produced by `rmlx kv-calibrate` and consumed by
//! the TurboQuant KV codec at runtime.
//!
//! ## Schema versioning
//!
//! The `version` field is always `1`. rMLX extends the schema additively:
//!
//! | Schema label | `version` | Added fields |
//! |---|---|---|
//! | mtq v1 | `1` | *(baseline)* |
//! | rMLX v1.1 | `1` | `LayerCalib::codebook` (per-layer codebook override, optional) |
//! | rMLX v1.2 | `1` | `KvCalibration::head_budgets` (populated post-hoc by the loader scanning a sibling `head_budgets.json`, never serialised back into `kv_calib.json`) |
//!
//! v1 files (no `codebook` field) parse cleanly via `#[serde(default)]` — full
//! backwards-compatibility with `multi-turboquant`'s `turboquant_kv.json` v1.
//! Writing a v1.1 file with `codebook = Some(...)` produces JSON that
//! a pure-v1 reader will silently ignore (forward-compatible in the `serde_json`
//! `deny_unknown_fields = false` default, which is the case here).
//!
//! # Public API
//!
//! - [`KvCalibration`] — top-level schema struct.
//! - [`LayerCalib`] — per-layer high-precision index lists.
//! - [`CalibrationMeta`] — provenance metadata.
//! - [`recipe_to_internal`] — recipe string → internal recipe name.
//! - [`write_kv_calibration`] — serialize to JSON file.
//! - [`read_kv_calibration`] — deserialize from JSON file.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use serde::{Deserialize, Serialize};

use rmlx_core::{Error, Result};

use crate::head_budgets::HeadBudgets;

// ── Recipe mapping ────────────────────────────────────────────────────────────

/// Map a user-facing recipe name to the internal recipe identifier.
///
/// | User recipe | Internal recipe |
/// |---|---|
/// | `turbo2` | `turboquant25` |
/// | `turbo2_tcq` | `turboquant25` |
/// | `turbo3` | `turboquant35` |
/// | `turbo3_tcq` | `turboquant35` |
/// | `turbo4` | `turboquant35` |
/// | `turboquant25` | `turboquant25` (identity passthrough) |
/// | `turboquant35` | `turboquant35` (identity passthrough) |
///
/// Returns `Err` for any unrecognised recipe name.
pub fn recipe_to_internal(recipe: &str) -> Result<&'static str> {
    match recipe {
        // User-facing aliases and identity passthrough (matches mtq's recipe_map lines 89-93).
        "turbo2" | "turbo2_tcq" | "turboquant25" => Ok("turboquant25"),
        "turbo3" | "turbo3_tcq" | "turbo4" | "turboquant35" => Ok("turboquant35"),
        other => Err(Error::Config(format!(
            "unknown recipe '{other}'; expected one of: turbo2, turbo2_tcq, turbo3, turbo3_tcq, turbo4, turboquant25, turboquant35"
        ))),
    }
}

/// Outlier count for a given `head_dim` and internal recipe.
///
/// Ported from `multi_turboquant/methods/turboquant.py::get_outlier_count`.
///
/// | Internal recipe | Ratio | Group alignment |
/// |---|---|---|
/// | `turboquant25` | 0.25 | 16 |
/// | `turboquant35` | 0.50 | 16 |
///
/// **Rounding convention**: uses `f64::round()` (round-half-away-from-zero), which
/// matches the result for all mainstream head_dims (64, 128, 256). mtq's Python
/// `round()` uses banker's rounding (round-half-to-even); these diverge only at
/// exact half-group boundaries with non-standard head_dims (e.g. `head_dim=80,
/// ratio=0.5` → Rust=48, Python=48; `head_dim=160, ratio=0.25` → Rust=48,
/// Python=48 — divergence requires a midpoint that is also non-zero mod the
/// group, which does not arise for standard powers-of-two head_dims).
pub fn outlier_count_for(head_dim: u32, internal_recipe: &str) -> Result<u32> {
    const GROUP_ALIGNMENT: u32 = 16;
    let ratio = match internal_recipe {
        "turboquant25" => 0.25_f64,
        "turboquant35" => 0.50_f64,
        other => return Err(Error::Config(format!("unknown internal recipe '{other}'"))),
    };
    let raw = f64::from(head_dim) * ratio;
    // round() is round-half-away-from-zero (Rust convention). For standard
    // head_dims (64/128/256) `raw / GROUP_ALIGNMENT` is never at an exact 0.5
    // midpoint, so this is identical to mtq's banker's round for all real models.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let aligned = ((raw / f64::from(GROUP_ALIGNMENT)).round() as u32) * GROUP_ALIGNMENT;
    if aligned == 0 || aligned >= head_dim {
        return Err(Error::Config(format!(
            "unsupported head_dim {head_dim} for {internal_recipe}: aligned outlier count \
             {aligned} is out of range (0, {head_dim})"
        )));
    }
    Ok(aligned)
}

// ── Schema ────────────────────────────────────────────────────────────────────

/// Per-layer codebook override (rMLX schema v1.1).
///
/// When present, the V-side TurboQuant CPU encoder uses these centroids instead
/// of the built-in Lloyd-Max N(0,1) codebook. The K side is unaffected.
///
/// ## Field semantics
///
/// - `value` — per-layer V codebook. Length **must** equal `2^bits` (e.g. 16
///   for 4-bit). Centroids must be in strictly ascending order (validation is
///   at encode time). An empty `Vec` (`[]`) deserializes cleanly but will
///   return `Error::Quant` at first encode for that layer. `None` (omitted)
///   means "use built-in".
///
/// ## GPU dispatch
///
/// When `value` is `Some`, the V-side encode is forced to the CPU scalar path
/// for that layer (the MSL kernel has the Lloyd-Max codebook hardwired). CPU
/// scalar encode is materially slower than the MSL kernel; T19b will close
/// the gap. Exact factor TBD pending benchmark.
// SCHEMA-COMPAT (rMLX v1.x): do NOT add #[serde(deny_unknown_fields)] — forward-compat
// between point versions depends on the serde default. The `key` field may be added
// additively in a future version; omit it here until it is consumed by a codec.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CodebookOverride {
    /// Per-layer V-side codebook (`2^bits` centroids in strictly ascending order).
    ///
    /// `None` = not present (omitted from JSON) → use built-in Lloyd-Max.
    /// `Some(vec)` = use these centroids; vec length must equal `2^bits`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value: Option<Vec<f32>>,
}

/// Per-layer calibration: high-precision index lists for K and V projections.
///
/// Each inner `Vec<u32>` is the sorted list of high-precision head-dimension
/// indices for one KV head. Length of the outer vec = `num_kv_heads`.
///
/// See `CodebookOverride` for schema-compat policy (no `deny_unknown_fields`).
// SCHEMA-COMPAT (rMLX v1.x): do NOT add #[serde(deny_unknown_fields)] — forward-compat
// between point versions depends on the serde default.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LayerCalib {
    /// Sorted top-K indices per KV head for the K projection.
    pub key_high_precision_indices: Vec<Vec<u32>>,
    /// Sorted top-K indices per KV head for the V projection.
    pub value_high_precision_indices: Vec<Vec<u32>>,
    /// Per-layer V-side codebook override (rMLX schema v1.1).
    ///
    /// `None` (absent from JSON, or deserialized from a v1 file) = use the
    /// built-in Lloyd-Max N(0,1) codebook. `Some(cb)` = use `cb.value` for
    /// the V encoder for this layer. K side always uses its built-in codebook.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub codebook: Option<CodebookOverride>,
}

/// Provenance metadata embedded in the calibration file.
///
/// All numeric fields are zero / empty-string for weight-norm calibration
/// (no real prompt data was consumed).
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMeta {
    /// Calibration method identifier (e.g. `"weight_norm"`).
    pub method: String,
    /// Loss objective (e.g. `"l2_norm"`).
    pub objective: String,
    /// Number of prompts observed (`0` for weight-norm path).
    pub num_prompts: u32,
    /// Maximum sequence length used (`0` for weight-norm path).
    pub max_seq_len: u32,
    /// Batch size used (`0` for weight-norm path).
    pub batch_size: u32,
    /// Total tokens observed (`0` for weight-norm path).
    pub num_observed_tokens: u64,
    /// Weight dtype as a string (e.g. `"bfloat16"`).
    pub dtype: String,
    /// Device used (`"cpu"` for weight-norm path).
    pub device: String,
    /// SHA-256 of the prompt corpus (`""` for weight-norm path).
    pub prompts_sha256: String,
}

/// Top-level `kv_calib.json` v1 schema.
///
/// Schema is byte-identical with `multi-turboquant`'s `turboquant_kv.json` v1.
/// The exact key set is:
/// `version`, `recipe`, `head_size`, `model_name`, `transform_version`,
/// `codebook_version`, `layers`, `calibration`.
///
/// See `CodebookOverride` for schema-compat policy (no `deny_unknown_fields`).
// SCHEMA-COMPAT (rMLX v1.x): do NOT add #[serde(deny_unknown_fields)] — forward-compat
// between point versions depends on the serde default.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct KvCalibration {
    /// Schema version, always `1`.
    pub version: u32,
    /// Internal recipe name: `"turboquant25"` or `"turboquant35"`.
    pub recipe: String,
    /// Head dimension used for calibration.
    pub head_size: u32,
    /// Model snapshot directory basename (informational).
    pub model_name: String,
    /// Transform version tag.
    pub transform_version: String,
    /// Codebook version tag.
    pub codebook_version: String,
    /// Per-layer calibration data, keyed by layer attention prefix
    /// (e.g. `"model.layers.0.self_attn"`).
    pub layers: BTreeMap<String, LayerCalib>,
    /// Provenance metadata.
    pub calibration: CalibrationMeta,
    /// Schema v1.2: per-layer-per-head sparse-attention budgets.
    ///
    /// Populated post-hoc by the loader scanning for a sibling
    /// `head_budgets.json` next to `kv_calib.json`. Always `None` when the
    /// `KvCalibration` is freshly read from `kv_calib.json` via
    /// [`read_kv_calibration`] — that reader treats `head_budgets.json` as a
    /// distinct on-disk artifact. The field is `#[serde(skip)]` so this
    /// runtime-only attachment never round-trips through `kv_calib.json`.
    #[serde(skip)]
    pub head_budgets: Option<HeadBudgets>,
}

// ── I/O ───────────────────────────────────────────────────────────────────────

/// Serialize `calib` to `path` as a pretty-printed JSON file.
///
/// Creates parent directories if they do not exist.
pub fn write_kv_calibration(path: &Path, calib: &KvCalibration) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            Error::Loader(format!(
                "cannot create parent dir {}: {e}",
                parent.display()
            ))
        })?;
    }
    let json = serde_json::to_string_pretty(calib)
        .map_err(|e| Error::Config(format!("kv_calib.json serialization failed: {e}")))?;
    std::fs::write(path, json.as_bytes())
        .map_err(|e| Error::Loader(format!("cannot write {}: {e}", path.display())))?;
    Ok(())
}

/// Deserialize `kv_calib.json` (or `turboquant_kv.json`) from `path`.
pub fn read_kv_calibration(path: &Path) -> Result<KvCalibration> {
    let data = std::fs::read(path)
        .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;
    let calib: KvCalibration = serde_json::from_slice(&data).map_err(|e| {
        Error::Config(format!(
            "malformed kv_calib.json at {}: {e}",
            path.display()
        ))
    })?;
    Ok(calib)
}

/// Probe `<model_dir>/kv_calib.json` and return the parsed [`KvCalibration`] if
/// the file is present and valid.
///
/// Returns `None` (without error) when:
/// - The file does not exist.
/// - The schema `version` field is not `1`.
/// - The `head_size` field does not match `expected_head_size`.
///
/// In the mismatch cases a `warn!` is emitted with details so operators can
/// diagnose calibration files generated for a different model.
///
/// Emits `info!` on success with the layer count and recipe name.
pub fn discover_kv_calibration(model_dir: &Path, expected_head_size: u32) -> Option<KvCalibration> {
    let path = model_dir.join("kv_calib.json");
    if !path.exists() {
        return None;
    }
    let calib = match read_kv_calibration(&path) {
        Err(e) => {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "discover_kv_calibration: failed to parse kv_calib.json — skipping"
            );
            return None;
        }
        Ok(c) => c,
    };
    if let Some(err) = validate_calib(&calib, expected_head_size) {
        warn_invalid_calib(&path, &err);
        return None;
    }
    tracing::info!(
        path = %path.display(),
        layers = calib.layers.len(),
        recipe = %calib.recipe,
        head_size = calib.head_size,
        "discover_kv_calibration: calibration loaded successfully"
    );
    Some(calib)
}

/// Emit a structured `warn!` for a failed `validate_calib` result.
///
/// Extracted so `discover_kv_calibration` stays below the cognitive-complexity
/// threshold. Structured fields vary by error variant so callers can filter logs
/// by `expected_head_size` / `actual_head_size` or `calib_version` independently.
fn warn_invalid_calib(path: &Path, err: &ValidationError) {
    match err {
        ValidationError::UnsupportedVersion(v) => {
            tracing::warn!(
                path = %path.display(),
                calib_version = v,
                reason = %err,
                "discover_kv_calibration: skipping invalid file"
            );
        }
        ValidationError::HeadSizeMismatch { expected, actual } => {
            tracing::warn!(
                path = %path.display(),
                expected_head_size = expected,
                actual_head_size = actual,
                reason = %err,
                "discover_kv_calibration: skipping invalid file"
            );
        }
    }
}

/// Structured validation error for `validate_calib`.
#[derive(Debug)]
enum ValidationError {
    UnsupportedVersion(u32),
    HeadSizeMismatch { expected: u32, actual: u32 },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(v) => {
                write!(f, "unsupported schema version {v} (expected 1)")
            }
            Self::HeadSizeMismatch { expected, actual } => {
                write!(
                    f,
                    "head_size mismatch: expected {expected}, got {actual} \
                     (calibration generated for a different model)"
                )
            }
        }
    }
}

/// Returns a `ValidationError` if `calib` fails validation, or `None` if OK.
fn validate_calib(calib: &KvCalibration, expected_head_size: u32) -> Option<ValidationError> {
    if calib.version != 1 {
        return Some(ValidationError::UnsupportedVersion(calib.version));
    }
    if calib.head_size != expected_head_size {
        return Some(ValidationError::HeadSizeMismatch {
            expected: expected_head_size,
            actual: calib.head_size,
        });
    }
    None
}

#[cfg(test)]
#[path = "calibration_writer_tests.rs"]
mod calibration_writer_tests;
