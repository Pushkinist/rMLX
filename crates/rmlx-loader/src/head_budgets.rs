//! `head_budgets.json` schema (v1 + v2) — reader, writer, validator.
//!
//! Per-(layer, head) k-budgets used by the two-phase sparse-attention dispatcher.
//! Produced by `rmlx kv-calibrate --recipe {head_budget|softmax_mass}` and
//! consumed at model load time alongside `kv_calib.json`.
//!
//! # Versions
//!
//! - **v1** — K-norm² proxy. `method == "softmax_mass"` in the schema label,
//!   but the implementation uses the H2O/StreamingLLM K-norm² stand-in.
//!   No `recipe` field, no provenance list. Files emitted by the v1 calibrator.
//! - **v2** — true softmax-mass calibration with a real Q@K^T → softmax
//!   pass. Adds:
//!     * `recipe: "softmax_mass" | "k_norm_proxy"` — concrete measurement
//!       recipe identifier (`softmax_mass` is the v2 default).
//!     * `target_mass: f32` — cumulative softmax-mass target (was unnamed at v1).
//!     * `target_mass_budget_floor: u32` — minimum budget per (layer, head)
//!       (guards against pathological single-mass peaks crashing the dispatcher).
//!     * `prompts_provenance: Vec<String>` — list of prompt-file basenames
//!       used in the calibration corpus (for reproducibility).
//!
//! Loading a v1 file is supported indefinitely (back-compat) and emits a
//! `tracing::warn!` advising re-calibration with the v2 softmax-mass recipe. The validator
//! enforces structural invariants (shape match, no zero budgets) for both.

use std::path::Path;

use serde::{Deserialize, Serialize};

use rmlx_core::{Error, Result};

// ── Schema ────────────────────────────────────────────────────────────────────

/// Provenance metadata for a `head_budgets.json` calibration pass.
///
/// v1 fields (always present): `method`, `prompt_set_sha256`, `num_prompts`,
/// `max_seq_len`, `mass_threshold`.
///
/// v2-only fields (`Option`, `None` on v1 load):
/// `recipe`, `target_mass`, `target_mass_budget_floor`, `prompts_provenance`.
// SCHEMA-COMPAT (rMLX v1.x): no `deny_unknown_fields` so future point versions
// can add fields additively.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadBudgetCalibration {
    /// Calibration method identifier (`"softmax_mass"` for both v1 and v2 —
    /// see `recipe` for the actual measurement variant).
    pub method: String,
    /// SHA-256 (lowercase hex) of the prompt corpus used for measurement.
    pub prompt_set_sha256: String,
    /// Number of prompts observed.
    pub num_prompts: u32,
    /// Maximum sequence length forwarded per prompt.
    pub max_seq_len: u32,
    /// Softmax-mass threshold the budget targets (e.g. `0.95`). Kept as a v1
    /// alias for `target_mass`; v2 writers set both to the same value.
    pub mass_threshold: f32,

    // ── v2 additions ─────────────────────────────────────────────────────────
    /// v2 — concrete measurement recipe identifier.
    ///
    /// - `"softmax_mass"` — true Q@K^T softmax cumulative-mass measurement
    ///   (v2 default).
    /// - `"k_norm_proxy"` — legacy K-norm² stand-in (v1 recipe; explicit
    ///   v2 caller asks for the proxy via `--recipe k-norm-proxy`).
    ///
    /// `None` on a v1 file load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub recipe: Option<String>,

    /// v2 — explicit target cumulative softmax-mass coverage.
    /// Should match `mass_threshold`. `None` on a v1 file load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_mass: Option<f32>,

    /// v2 — minimum k-budget per (layer, head). Guards against
    /// pathological prompts where a single key carries ≥`target_mass` of the
    /// softmax weight; without the floor the dispatcher would crash on
    /// 1-slot sparse decode. `None` on a v1 file load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_mass_budget_floor: Option<u32>,

    /// v2 — basenames of the calibration prompt files (relative paths
    /// or filenames, not absolute machine paths). `None` on a v1 file load.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompts_provenance: Option<Vec<String>>,
}

/// Top-level `head_budgets.json` schema.
///
/// Per-layer, per-head k-budget table for the two-phase sparse-attention
/// dispatcher. `per_layer_per_head_budget[layer][head]` is the integer count
/// of KV slots phase-2 must attend to for that (layer, head).
///
/// `version`: `1` for legacy K-norm² proxy files, `2` for true softmax-mass
/// files. Both are accepted by `load_head_budgets`.
// SCHEMA-COMPAT (rMLX v1.x): no `deny_unknown_fields` so future point versions
// can add fields additively.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadBudgets {
    /// Schema version: `1` (legacy K-norm² proxy) or `2` (true softmax-mass).
    pub version: u32,
    /// Model snapshot directory basename (informational).
    pub model_name: String,
    /// Number of attention layers covered by the table.
    pub num_layers: usize,
    /// Number of (query) attention heads per layer.
    pub num_heads: usize,
    /// Provenance metadata.
    pub calibration: HeadBudgetCalibration,
    /// `[num_layers][num_heads]` table of phase-2 KV budgets.
    pub per_layer_per_head_budget: Vec<Vec<u32>>,
}

impl HeadBudgetCalibration {
    /// Construct a v1 calibration metadata record. v2 fields default to `None`.
    /// Cross-crate constructor (struct is `#[non_exhaustive]`).
    #[must_use]
    pub fn new(
        method: String,
        prompt_set_sha256: String,
        num_prompts: u32,
        max_seq_len: u32,
        mass_threshold: f32,
    ) -> Self {
        Self {
            method,
            prompt_set_sha256,
            num_prompts,
            max_seq_len,
            mass_threshold,
            recipe: None,
            target_mass: None,
            target_mass_budget_floor: None,
            prompts_provenance: None,
        }
    }

    /// Construct a v2 calibration metadata record (all fields set).
    /// Cross-crate constructor (struct is `#[non_exhaustive]`).
    #[must_use]
    pub fn new_v2(
        method: String,
        prompt_set_sha256: String,
        num_prompts: u32,
        max_seq_len: u32,
        mass_threshold: f32,
        recipe: String,
        target_mass: f32,
        target_mass_budget_floor: u32,
        prompts_provenance: Vec<String>,
    ) -> Self {
        Self {
            method,
            prompt_set_sha256,
            num_prompts,
            max_seq_len,
            mass_threshold,
            recipe: Some(recipe),
            target_mass: Some(target_mass),
            target_mass_budget_floor: Some(target_mass_budget_floor),
            prompts_provenance: Some(prompts_provenance),
        }
    }
}

impl HeadBudgets {
    /// Construct a v1 `HeadBudgets` table. Cross-crate constructor (struct is
    /// `#[non_exhaustive]`). Shape is **not** validated by the constructor —
    /// pass through [`write_head_budgets`] / [`load_head_budgets`] for the
    /// structural check.
    #[must_use]
    pub fn new(
        model_name: String,
        num_layers: usize,
        num_heads: usize,
        calibration: HeadBudgetCalibration,
        per_layer_per_head_budget: Vec<Vec<u32>>,
    ) -> Self {
        Self {
            version: 1,
            model_name,
            num_layers,
            num_heads,
            calibration,
            per_layer_per_head_budget,
        }
    }

    /// Construct a v2 `HeadBudgets` table. Same shape contract as
    /// `new`; differs only in the recorded schema version.
    #[must_use]
    pub fn new_v2(
        model_name: String,
        num_layers: usize,
        num_heads: usize,
        calibration: HeadBudgetCalibration,
        per_layer_per_head_budget: Vec<Vec<u32>>,
    ) -> Self {
        Self {
            version: 2,
            model_name,
            num_layers,
            num_heads,
            calibration,
            per_layer_per_head_budget,
        }
    }
}

// ── I/O ───────────────────────────────────────────────────────────────────────

/// Write a [`HeadBudgets`] table to `path`.
///
/// Pretty-printed JSON; structural invariants checked first (matching shape,
/// no zero budgets, supported schema version). Both v1 and v2 tables write
/// the same way.
pub fn write_head_budgets<P: AsRef<Path>>(path: P, budgets: &HeadBudgets) -> Result<()> {
    validate(budgets)
        .map_err(|e| Error::Config(format!("write_head_budgets: structural check failed: {e}")))?;
    let json = serde_json::to_vec_pretty(budgets)
        .map_err(|e| Error::Loader(format!("serialize HeadBudgets: {e}")))?;
    std::fs::write(path.as_ref(), json).map_err(|e| {
        Error::Loader(format!(
            "cannot write head_budgets.json to {}: {e}",
            path.as_ref().display()
        ))
    })
}

/// Load + validate a `head_budgets.json` file. Returns `Ok(None)` when absent.
///
/// Accepts schema v1 and v2. v1 loads emit a `tracing::warn!` advising
/// re-calibration with the true softmax-mass recipe — the dispatcher
/// will still consume v1 budgets correctly but the budgets reflect a K-norm²
/// proxy, not real softmax mass.
pub fn load_head_budgets<P: AsRef<Path>>(path: P) -> Result<Option<HeadBudgets>> {
    let path = path.as_ref();
    if !path.exists() {
        return Ok(None);
    }
    let data = std::fs::read(path)
        .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;
    let budgets: HeadBudgets = serde_json::from_slice(&data).map_err(|e| {
        Error::Config(format!(
            "malformed head_budgets.json at {}: {e}",
            path.display()
        ))
    })?;
    validate(&budgets).map_err(|e| {
        Error::Config(format!(
            "head_budgets.json at {} failed validation: {e}",
            path.display()
        ))
    })?;
    if budgets.version == 1 {
        tracing::warn!(
            path = %path.display(),
            "loaded v1 head_budgets (K-norm² proxy); recipe drift may affect \
             retrieval quality — re-calibrate with \
             `rmlx kv-calibrate --recipe softmax-mass`"
        );
    }
    Ok(Some(budgets))
}

/// Structural validation: schema version + shape match + no zero budgets.
///
/// Accepts `version ∈ {1, 2}`. Both versions share the same shape contract;
/// v2-only fields are optional in the calibration metadata and not part of
/// the structural check.
fn validate(b: &HeadBudgets) -> std::result::Result<(), String> {
    if b.version != 1 && b.version != 2 {
        return Err(format!(
            "unsupported schema version {} (expected 1 or 2)",
            b.version
        ));
    }
    if b.per_layer_per_head_budget.len() != b.num_layers {
        return Err(format!(
            "per_layer_per_head_budget has {} rows, expected num_layers={}",
            b.per_layer_per_head_budget.len(),
            b.num_layers
        ));
    }
    for (layer_idx, row) in b.per_layer_per_head_budget.iter().enumerate() {
        if row.len() != b.num_heads {
            return Err(format!(
                "per_layer_per_head_budget[{layer_idx}] has {} columns, expected num_heads={}",
                row.len(),
                b.num_heads
            ));
        }
        for (head_idx, &budget) in row.iter().enumerate() {
            if budget == 0 {
                return Err(format!(
                    "per_layer_per_head_budget[{layer_idx}][{head_idx}] is zero \
                     (every (layer, head) must attend to at least one KV slot)"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "head_budgets_tests.rs"]
mod head_budgets_tests;

#[cfg(test)]
#[path = "head_budgets_v2_tests.rs"]
mod head_budgets_v2_tests;
