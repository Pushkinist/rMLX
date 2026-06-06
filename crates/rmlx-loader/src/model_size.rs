//! Model-size estimation from `config.json`.
//!
//! Provides [`estimate_params_billions`], a lightweight heuristic used by the
//! auto-selector to estimate the model's total parameter count without
//! loading any weights.
//!
//! # Approximation
//!
//! The estimate uses the transformer parameter-count heuristic:
//!
//! ```text
//! params ≈ hidden_size² × num_layers × 12 / 1e9  (billions)
//! ```
//!
//! The factor 12 accounts for the dominant parameter blocks in a standard
//! decoder layer (attention Q/K/V/O + two MLP weight matrices × 2 for
//! up/down + embedding contribution amortised).  Note that `hidden_size` is
//! squared — this is the standard transformer-parameter heuristic; a single
//! linear layer is `hidden_size × hidden_size`, and there are ~12 such layers
//! per decoder block. Example: a 7B model with `hidden_size=4096, layers=32`
//! gives `4096² × 32 × 12 / 1e9 ≈ 6.4 B`, matching the actual parameter count.
//!
//! The estimate is intentionally rough (±15–25% for modern architectures).
//! Its only purpose is to place the model in the right VRAM-pressure tier;
//! exact byte-counts are not required.
//!
//! # Resolution order
//!
//! 1. `text_config.hidden_size` + `text_config.num_hidden_layers` (typed
//!    fields on [`TextConfig`] — multimodal / Gemma4 layout).
//! 2. Top-level `extras["hidden_size"]` + `extras["num_hidden_layers"]`
//!    (Qwen3 / Bonsai flat layout with no `text_config`).
//!
//! Returns `None` when neither field pair is available (unknown arch).
//! Callers should fall back to a conservative default (e.g. 7.0 B).

use crate::config::ModelConfig;

/// Estimate the number of trainable parameters in billions for `config`.
///
/// Uses `hidden_size² × num_hidden_layers × 12 / 1e9`.  The factor 12 is a
/// transformer heuristic covering attention (Q/K/V/O) + MLP (gate/up/down)
/// weight matrices per layer, which dominate the total count for LLMs.  Each
/// layer contributes approximately `12 × hidden_size²` parameters (attention:
/// 4 matrices of `hidden_size × hidden_size`; MLP: 3 matrices of roughly
/// `hidden_size × 4×hidden_size`; residual norms: negligible).
///
/// Returns `None` when the required fields are absent from the config.
/// Callers should fall back to a default (e.g. `7.0`) rather than propagating
/// `None` to the auto-selector.
pub fn estimate_params_billions(config: &ModelConfig) -> Option<f32> {
    let (hidden_size, num_layers) = resolve_arch_fields(config)?;
    if hidden_size == 0 || num_layers == 0 {
        return None;
    }
    let hs = f64::from(hidden_size);
    let estimate = hs * hs * f64::from(num_layers) * 12.0 / 1e9;
    Some(estimate as f32)
}

/// Extract (hidden_size, num_hidden_layers) from the config using the
/// resolution order described in the module doc.
fn resolve_arch_fields(config: &ModelConfig) -> Option<(u32, u32)> {
    // 1. Typed text_config fields — multimodal / Gemma4 / nested-config layout.
    if let Some(tc) = &config.text_config {
        if let (Some(hs), Some(nl)) = (tc.hidden_size, tc.num_hidden_layers) {
            return Some((hs, nl));
        }
    }

    // 2. Top-level extras — Qwen3 / Bonsai / flat-config layout.
    let hs = config
        .extras
        .get("hidden_size")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())?;
    let nl = config
        .extras
        .get("num_hidden_layers")
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| u32::try_from(v).ok())?;
    Some((hs, nl))
}

#[cfg(test)]
#[path = "model_size_tests.rs"]
mod model_size_tests;
