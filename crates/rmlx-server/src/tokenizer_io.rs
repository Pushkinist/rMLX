//! Tokenizer loading and encoding for mlx-community model snapshots.
//!
//! Reads `tokenizer.json` (via the `tokenizers` crate) and
//! `tokenizer_config.json` (via serde_json) from a model directory.
//!
//! Stage 1.7 — prompt pipeline.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use rmlx_core::{Error, Result};

// ── TokenizerConfig ───────────────────────────────────────────────────────────

/// Parsed `tokenizer_config.json`.
///
/// `bos_token` and `eos_token` may be:
/// - a plain string: `"<bos>"`
/// - null
/// - an object: `{"content": "<bos>", "lstrip": false, ...}`
///
/// Use [`extract_token_str`] to normalise all three shapes.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerConfig {
    /// Beginning-of-sequence token string, if present in `tokenizer_config.json`.
    #[serde(default, deserialize_with = "de_token_field")]
    pub bos_token: Option<String>,

    /// End-of-sequence token string, if present in `tokenizer_config.json`.
    #[serde(default, deserialize_with = "de_token_field")]
    pub eos_token: Option<String>,

    /// All other keys (pad_token, unk_token, processor_class, etc.)
    #[serde(flatten)]
    pub extras: HashMap<String, Value>,
}

/// Custom deserializer that handles string | null | {"content": "..."}.
fn de_token_field<'de, D>(de: D) -> std::result::Result<Option<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let v: Option<Value> = Option::deserialize(de)?;
    Ok(v.as_ref().and_then(extract_token_str))
}

/// Extract a token string from any of the three shapes used in
/// `tokenizer_config.json`.
///
/// - `"<bos>"` → `Some("<bos>")`
/// - `null` / `undefined` → `None`
/// - `{"content": "<bos>", ...}` → `Some("<bos>")`
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
)]
pub fn extract_token_str(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Object(map) => map
            .get("content")
            .and_then(|c| c.as_str())
            .map(str::to_owned),
        _ => None,
    }
}

/// Load and parse `<model_dir>/tokenizer_config.json`.
pub fn load_tokenizer_config(model_dir: &Path) -> Result<TokenizerConfig> {
    let path = model_dir.join("tokenizer_config.json");
    let data = std::fs::read(&path)
        .map_err(|e| Error::Other(format!("cannot read {}: {e}", path.display())))?;
    serde_json::from_slice(&data).map_err(|e| {
        Error::Other(format!(
            "malformed tokenizer_config.json at {}: {e}",
            path.display()
        ))
    })
}

/// Load `<model_dir>/tokenizer.json` via the `tokenizers` crate.
///
/// This is a one-shot blocking read; do NOT call per request.
pub fn load_tokenizer(model_dir: &Path) -> Result<tokenizers::Tokenizer> {
    let path = model_dir.join("tokenizer.json");
    tokenizers::Tokenizer::from_file(&path).map_err(|e| {
        Error::Other(format!(
            "failed to load tokenizer from {}: {e}",
            path.display()
        ))
    })
}

/// Encode `text` to token IDs.
///
/// `add_special_tokens = false` because the chat template already inserts
/// BOS/EOS where the model expects them.
pub fn encode(tk: &tokenizers::Tokenizer, text: &str) -> Result<Vec<u32>> {
    let encoding = tk
        .encode(text, false)
        .map_err(|e| Error::Other(format!("tokenizer encode failed: {e}")))?;
    Ok(encoding.get_ids().to_vec())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tokenizer_io_tests.rs"]
mod tests;
