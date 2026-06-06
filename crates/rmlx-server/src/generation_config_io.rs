//! `generation_config.json` parsing for mlx-community model snapshots.
//!
//! Optional file. Absent → `Ok(None)`. Present but unparseable → `Err`.
//! Field-level: only the keys we care about; unknown keys ignored.

use std::path::Path;

use serde::Deserialize;

use rmlx_core::{Error, Result};

// ── GenerationConfig ──────────────────────────────────────────────────────────

/// Sampling defaults parsed from `generation_config.json`.
///
/// All fields are optional: a key absent from the JSON file yields `None`.
/// Unknown keys in the file are silently ignored.
#[non_exhaustive]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerationConfig {
    /// Default sampling temperature from `generation_config.json`.
    pub temperature: Option<f32>,
    /// Default nucleus sampling probability.
    pub top_p: Option<f32>,
    /// Default top-k sampling cutoff.
    pub top_k: Option<u32>,
    /// Default repetition penalty multiplier.
    pub repetition_penalty: Option<f32>,
    /// Default maximum new tokens per request.
    pub max_new_tokens: Option<u32>,
}

/// Load and parse `<model_dir>/generation_config.json`.
///
/// Returns:
/// - `Ok(None)` — file is absent (not an error; treat as no defaults).
/// - `Ok(Some(cfg))` — file present and parsed successfully.
/// - `Err(_)` — file present but could not be read or parsed.
pub fn load_generation_config(model_dir: &Path) -> Result<Option<GenerationConfig>> {
    let path = model_dir.join("generation_config.json");
    if !path.exists() {
        return Ok(None);
    }
    let src = std::fs::read_to_string(&path)
        .map_err(|e| Error::Other(format!("read {}: {e}", path.display())))?;
    let cfg: GenerationConfig = serde_json::from_str(&src)
        .map_err(|e| Error::Other(format!("parse {}: {e}", path.display())))?;
    Ok(Some(cfg))
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "generation_config_io_tests.rs"]
mod generation_config_io_tests;
