//! `config.json` parsing — model architecture config and per-tensor quantization metadata.
//!
//! Deserializes the HuggingFace `config.json` layout produced by mlx-community
//! model snapshots. Handles both the flat top-level layout (Qwen2) and the
//! nested `text_config` sub-object layout (Gemma4, vision-language models).
//!
//! # Public API
//!
//! - [`load_config`] — read and parse `<model_dir>/config.json`.
//! - [`ModelConfig`] — top-level envelope with arch string, quant metadata,
//!   and the `text_config` sub-object.
//! - [`TextConfig`] — architecture fields (layers, heads, hidden dim, …).
//! - [`QuantConfig`] — per-layer affine-4bit quant config.
//! - [`ParoQuantConfig`] — PARO rotation quant config.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::Map;
use tracing::error;

use rmlx_core::{Error, Result};

// Bounded against malformed/adversarial config.json tensor_overrides maps.
// Production models typically have <10 overrides; 4096 is a generous but defensive ceiling.
pub(crate) const MAX_OVERRIDES: usize = 4_096;

/// ParoQuant-style quantization configuration found in `quantization_config`
/// (as opposed to the MLX `quantization` field).
///
/// Canonical keys (from `z-lab/paroquant` HF checkpoints):
/// `quant_method = "paroquant"`, `bits`, `group_size`, `krot`.
///
/// Note: some non-PARO HF checkpoints (e.g. `mlx-community/gemma-4-e4b-it-mxfp8`)
/// also have a `quantization_config` field but with a different schema (no `quant_method`
/// or `krot`). All fields are therefore `Option` so that deserialization never fails
/// on unrecognised schemas. Use `ModelConfig::is_paroquant()` to confirm.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParoQuantConfig {
    /// `"paroquant"` for PARO checkpoints; may be absent in other schemas.
    pub quant_method: Option<String>,
    /// Bit width per packed code (typically 4 or 8 for PARO).
    pub bits: Option<u8>,
    /// Group size (elements per shared scale).
    pub group_size: Option<u32>,
    /// Number of rotation groups per linear layer. Present only in PARO checkpoints.
    pub krot: Option<u32>,
    /// Absorb any extra fields from non-PARO schemas without error.
    #[serde(flatten)]
    pub extras: Map<String, serde_json::Value>,
}

/// Global or per-tensor quantization parameters.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantConfig {
    /// Group size (elements per shared scale).
    pub group_size: u32,
    /// Bit width per packed code.
    pub bits: u8,
    /// Optional in some snapshots. When absent, convention (from mlx-lm) is `"affine"`.
    pub mode: Option<String>,
    /// Per-tensor overrides (key = tensor name prefix, value = override config).
    /// Not present in most snapshots but must round-trip without error.
    pub tensor_overrides: Option<HashMap<String, QuantConfig>>,
}

impl QuantConfig {
    /// Return the mode string, defaulting to `"affine"` when the field is absent.
    pub fn mode_or_default(&self) -> &str {
        self.mode.as_deref().unwrap_or("affine")
    }
}

/// Text-generation sub-config. All fields are `Option` because multimodal
/// configs nest this inside `text_config` and may omit some fields.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextConfig {
    /// Number of decoder layers.
    pub num_hidden_layers: Option<u32>,
    /// Residual hidden dimension.
    pub hidden_size: Option<u32>,
    /// Number of attention heads in the decoder.
    pub num_attention_heads: Option<u32>,
    /// Number of KV heads (GQA factor `num_attention_heads / num_key_value_heads`).
    pub num_key_value_heads: Option<u32>,
    /// Sliding-window attention size, when applicable.
    pub sliding_window: Option<u32>,
    /// Maximum position embedding span (raw, pre-YARN).
    pub max_position_embeddings: Option<u32>,
    /// Per-layer type names (`"full_attention"` / `"sliding_attention"` / etc.).
    pub layer_types: Option<Vec<String>>,
    /// All other text_config keys round-trip cleanly.
    #[serde(flatten)]
    pub extras: Map<String, serde_json::Value>,
}

/// Top-level `config.json`.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Architecture identifiers from the HuggingFace config (`["Gemma4ForConditionalGeneration"]`, etc.).
    ///
    /// Defaults to empty when absent: some standalone sidecar configs (e.g. the
    /// Qwen3.5 `qwen3_5_mtp` MTP-drafter split) carry only `model_type` and no
    /// `architectures` array. Callers that need an arch fall back to
    /// `model_type` / tensor-name detection.
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Default tensor dtype string (`"bfloat16"`, `"float16"`, etc.).
    pub dtype: Option<String>,
    /// MLX-native affine/mxfp quantization parameters (field name `quantization`).
    pub quantization: Option<QuantConfig>,
    /// ParoQuant-style quantization config (field name `quantization_config`).
    /// Present in `z-lab/*-PARO` checkpoints; absent in standard MLX snapshots.
    pub quantization_config: Option<ParoQuantConfig>,
    /// Nested text-generation sub-config (used by multimodal configs).
    pub text_config: Option<TextConfig>,
    /// All other top-level keys round-trip cleanly.
    #[serde(flatten)]
    pub extras: Map<String, serde_json::Value>,
}

impl ModelConfig {
    /// Returns `true` when the checkpoint was produced by ParoQuant.
    ///
    /// Detection rule: `quantization_config.quant_method == "paroquant"`.
    /// Does not require `quantization` to be absent (though in practice it is).
    pub fn is_paroquant(&self) -> bool {
        self.quantization_config
            .as_ref()
            .is_some_and(|qc| qc.quant_method.as_deref() == Some("paroquant"))
    }

    /// Resolve the **full-attention** head_dim from this config.
    ///
    /// Returns the value that downstream V-side codec validators (e.g. `tq4`
    /// requires head_dim ∈ {128, 256}) must check against. SWA-only fields
    /// (Gemma3/Gemma4 `head_dim` when `global_head_dim` is also present) are
    /// **never** returned — SWA layers stay bf16 per D6.7.
    ///
    /// Resolution order:
    /// 1. If `architectures[0]` is a Gemma3/Gemma4 variant, prefer
    ///    `text_config.global_head_dim` (the FA head_dim).
    /// 2. Otherwise read `text_config.head_dim` if present.
    /// 3. Otherwise read top-level `head_dim` (for Qwen3-style configs that
    ///    flatten all fields at the root and omit `text_config`).
    /// 4. Otherwise divide `hidden_size / num_attention_heads`, but only when
    ///    both are present AND the division has no remainder. Try the
    ///    `text_config`-nested values first, then the top-level extras.
    /// 5. Otherwise `None`.
    ///
    /// Returning `None` is a valid outcome — callers must not guess. The T3
    /// resolver explicitly maps `None` to `HeadDimUnknown`.
    pub fn head_dim(&self) -> Option<usize> {
        let arch = self.architectures.first().map_or("", String::as_str);
        let is_gemma = matches!(
            arch,
            "Gemma3ForCausalLM"
                | "Gemma3ForConditionalGeneration"
                | "Gemma4ForCausalLM"
                | "Gemma4ForConditionalGeneration"
                | "Gemma4UnifiedForConditionalGeneration"
        );

        // 1. Gemma3/Gemma4: global_head_dim (FA) — lives in text_config.extras.
        if is_gemma {
            if let Some(v) = self
                .text_config
                .as_ref()
                .and_then(|tc| tc.extras.get("global_head_dim"))
                .and_then(serde_json::Value::as_u64)
            {
                return Some(v as usize);
            }
        }

        // 2. text_config.head_dim (the typed-field path doesn't exist; read via extras).
        if let Some(v) = self
            .text_config
            .as_ref()
            .and_then(|tc| tc.extras.get("head_dim"))
            .and_then(serde_json::Value::as_u64)
        {
            return Some(v as usize);
        }

        // 3. Top-level head_dim (Qwen3/Bonsai-style: no text_config, all fields at root).
        if let Some(v) = self
            .extras
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
        {
            return Some(v as usize);
        }

        // 4. Divide-fallback: hidden_size / num_attention_heads, only when exact.
        let try_divide = |hs_opt: Option<u64>, nh_opt: Option<u64>| -> Option<usize> {
            let hs = hs_opt?;
            let nh = nh_opt?;
            if nh == 0 || hs % nh != 0 {
                return None;
            }
            Some((hs / nh) as usize)
        };

        // Prefer text_config (the multimodal nest) over top-level extras.
        if let Some(tc) = &self.text_config {
            if let Some(d) = try_divide(
                tc.hidden_size.map(u64::from),
                tc.num_attention_heads.map(u64::from),
            ) {
                return Some(d);
            }
        }
        if let Some(d) = try_divide(
            self.extras
                .get("hidden_size")
                .and_then(serde_json::Value::as_u64),
            self.extras
                .get("num_attention_heads")
                .and_then(serde_json::Value::as_u64),
        ) {
            return Some(d);
        }

        // 5. Unknown.
        None
    }

    /// Extract EOS token ids from the parsed config.
    ///
    /// HuggingFace `config.json` stores `eos_token_id` as either:
    /// - a single integer (e.g. Bonsai: `"eos_token_id": 151645`)
    /// - an array of integers (e.g. Gemma4: `"eos_token_id": [1, 106, 50]`,
    ///   Qwen3.5MoE: `"eos_token_id": [248046, 248044]`)
    ///
    /// Returns an empty `Vec` if the field is absent, null, or malformed —
    /// callers should treat empty as "no EOS-stop, run to max_tokens".
    pub fn eos_token_ids(&self) -> Vec<u32> {
        // Nested-config architectures (e.g. Qwen3-VL-MoE
        // `Qwen3VLMoeForConditionalGeneration`) carry `eos_token_id` inside
        // `text_config`, not at the top level. Fall back to that when the
        // top-level field is absent — additive: archs with a top-level
        // `eos_token_id` are unaffected.
        // Prefer the top-level `eos_token_id`. Nested-config architectures (e.g.
        // Qwen3-VL-MoE) set the top-level field to `null` and carry the real id
        // inside `text_config.eos_token_id`; fall back to that whenever the
        // top-level value is absent OR null. Additive: archs with a usable
        // top-level `eos_token_id` are unaffected.
        let top = self.extras.get("eos_token_id");
        let nested = self
            .text_config
            .as_ref()
            .and_then(|tc| tc.extras.get("eos_token_id"));
        let v = match top {
            Some(v) if !v.is_null() => v,
            _ => match nested {
                Some(v) if !v.is_null() => v,
                _ => return Vec::new(),
            },
        };
        #[allow(
            clippy::wildcard_enum_match_arm,
            reason = "serde_json::Value has many variants; all non-Number/Array forms (Null, Bool, String, Object) are invalid here and should return empty"
        )]
        match v {
            serde_json::Value::Number(n) => n
                .as_u64()
                .and_then(|u| u32::try_from(u).ok())
                .map_or_else(Vec::new, |id| vec![id]),
            serde_json::Value::Array(arr) => arr
                .iter()
                .filter_map(|x| x.as_u64().and_then(|u| u32::try_from(u).ok()))
                .collect(),
            _ => Vec::new(),
        }
    }
}

/// Load and parse `<model_dir>/config.json`.
pub fn load_config(model_dir: &Path) -> Result<ModelConfig> {
    let path = model_dir.join("config.json");
    let data = std::fs::read(&path)
        .map_err(|e| Error::Loader(format!("cannot read {}: {e}", path.display())))?;
    let cfg: ModelConfig = serde_json::from_slice(&data)
        .map_err(|e| Error::Loader(format!("malformed config.json at {}: {e}", path.display())))?;

    // Reject oversized tensor_overrides maps.
    if let Some(overrides) = cfg
        .quantization
        .as_ref()
        .and_then(|q| q.tensor_overrides.as_ref())
    {
        let n = overrides.len();
        if n > MAX_OVERRIDES {
            error!(
                got = n,
                max = MAX_OVERRIDES,
                file = "config.json",
                "tensor_overrides exceeds MAX_OVERRIDES bound — possible malformed or adversarial config"
            );
            return Err(Error::Loader(format!(
                "config.json tensor_overrides has {n} entries (max {MAX_OVERRIDES})"
            )));
        }
    }

    Ok(cfg)
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
