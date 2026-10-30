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

use crate::hadamard::HADAMARD_CONFIG;

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
    /// Read through [`ModelConfig::dispatch_arch`].
    ///
    /// Defaults to empty when absent: some standalone sidecar configs (e.g. the
    /// Qwen3.5 `qwen3_5_mtp` MTP-drafter split) carry only `model_type` and no
    /// `architectures` array, and a Prism Hadamard pack carries none either.
    #[serde(default)]
    architectures: Vec<String>,
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

    /// The `docs/METRICS_DB.md` §5.2 weight label this config describes, or
    /// `None` when it names no weight format.
    ///
    /// ParoQuant first, then the MLX `quantization` block — its `mode`
    /// lowercased, or `<bits>bit` when the mode is absent or `affine` — then an
    /// unquantized `dtype` / `torch_dtype`. No list of modes lives here: the
    /// recorder's §5.2 whitelist is the one check, and it refuses a mode it
    /// does not name. Kept in step with `weight_quant` in
    /// `scripts/lib/snapshot_identity.py` by one shared table,
    /// `tests/fixtures/weight_quant_labels.json`: the two disagreeing would
    /// split one model across two cells.
    pub fn weight_quant_label(&self) -> Option<String> {
        if self.is_paroquant() {
            return Some("paro".to_owned());
        }
        if let Some(quant) = &self.quantization {
            let mode = quant.mode_or_default().to_lowercase();
            return Some(match mode.as_str() {
                "" | "affine" => format!("{}bit", quant.bits),
                _ => mode,
            });
        }
        let dtype = self
            .dtype
            .as_deref()
            .filter(|d| !d.is_empty())
            .or_else(|| self.extras.get("torch_dtype").and_then(|v| v.as_str()));
        match dtype {
            Some("bfloat16") => Some("bf16".to_owned()),
            Some("float16") => Some("fp16".to_owned()),
            _ => None,
        }
    }

    /// Returns `true` when `config.json` names a Hadamard contract file
    /// (`hadamard_config`): the snapshot is a Prism Hadamard pack. Whether the
    /// contract holds is [`crate::HadamardContract::read`]'s question.
    pub fn is_hadamard_pack(&self) -> bool {
        self.extras.contains_key(HADAMARD_CONFIG)
    }

    /// The architecture class this snapshot dispatches on: `architectures[0]`
    /// when the config names one. A Prism Hadamard pack names none; it
    /// dispatches on the class of the backbone its `base_model_type` names.
    /// `None` when neither gives a class.
    pub fn dispatch_arch(&self) -> Option<&str> {
        if let Some(declared) = self.architectures.first() {
            return Some(declared);
        }
        if !self.is_hadamard_pack() {
            return None;
        }
        match self.extras.get("base_model_type")?.as_str()? {
            "qwen3_5" => Some("Qwen3_5ForConditionalGeneration"),
            _ => None,
        }
    }

    /// Resolve the **full-attention** head_dim from this config.
    ///
    /// Returns the value that downstream V-side codec validators (e.g. `tq4`
    /// requires head_dim ∈ {128, 256}) must check against. SWA-only fields
    /// (Gemma3/Gemma4 `head_dim` when `global_head_dim` is also present) are
    /// **never** returned — SWA layers stay bf16.
    ///
    /// Resolution order:
    /// 1. If [`Self::dispatch_arch`] is a Gemma3/Gemma4 variant, prefer
    ///    `text_config.global_head_dim` (the FA head_dim).
    /// 2. Otherwise read `text_config.head_dim` if present.
    /// 3. Otherwise read top-level `head_dim` (for Qwen3-style configs that
    ///    flatten all fields at the root and omit `text_config`).
    /// 4. Otherwise divide `hidden_size / num_attention_heads`, but only when
    ///    both are present AND the division has no remainder. Try the
    ///    `text_config`-nested values first, then the top-level extras.
    /// 5. Otherwise `None`.
    ///
    /// Returning `None` is a valid outcome — callers must not guess. The
    /// `--cache-type-*` resolver explicitly maps `None` to `HeadDimUnknown`.
    pub fn head_dim(&self) -> Option<usize> {
        let arch = self.dispatch_arch().unwrap_or("");
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

    /// The `eos_token_id` of `config.json`: the top-level value, or, when it
    /// is absent or null, the `text_config` value (Qwen3-VL-MoE sets the top
    /// level to null). The served stop set is [`ModelConfig::stop_token_ids`].
    fn eos_token_ids(&self) -> Vec<u32> {
        let top = self.extras.get("eos_token_id").filter(|v| !v.is_null());
        let nested = || {
            self.text_config
                .as_ref()
                .and_then(|tc| tc.extras.get("eos_token_id"))
        };
        top.or_else(nested).map_or_else(Vec::new, token_ids)
    }
}

/// The token ids an `eos_token_id` value names: one integer or an array of
/// them. An entry that is not a `u32` is skipped; any other value names none.
fn token_ids(value: &serde_json::Value) -> Vec<u32> {
    let id = |v: &serde_json::Value| v.as_u64().and_then(|u| u32::try_from(u).ok());
    match value.as_array() {
        Some(entries) => entries.iter().filter_map(id).collect(),
        None => id(value).into_iter().collect(),
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

/// Sampling defaults and end ids from `generation_config.json`. A key the file
/// does not hold is `None` (no ids for `eos_token_id`); other keys are ignored.
#[allow(
    clippy::exhaustive_structs,
    reason = "plain file record built by callers and tests; a new key is a new field at every site on purpose"
)]
#[derive(Debug, Clone, Default, Deserialize)]
pub struct GenerationConfig {
    /// Default sampling temperature.
    pub temperature: Option<f32>,
    /// Default nucleus sampling probability.
    pub top_p: Option<f32>,
    /// Default top-k sampling cutoff.
    pub top_k: Option<u32>,
    /// Default repetition penalty multiplier.
    pub repetition_penalty: Option<f32>,
    /// Default maximum new tokens per request.
    pub max_new_tokens: Option<u32>,
    /// `false` asks for greedy decoding. Only the speech route reads it.
    pub do_sample: Option<bool>,
    /// Qwen3-TTS code predictor: `false` asks for greedy decoding.
    pub subtalker_dosample: Option<bool>,
    /// Qwen3-TTS code predictor: sampling temperature.
    pub subtalker_temperature: Option<f32>,
    /// Qwen3-TTS code predictor: nucleus sampling probability.
    pub subtalker_top_p: Option<f32>,
    /// Qwen3-TTS code predictor: top-k sampling cutoff.
    pub subtalker_top_k: Option<u32>,
    /// The token ids `eos_token_id` names: one integer or an array.
    #[serde(default, rename = "eos_token_id", deserialize_with = "de_token_ids")]
    pub eos_token_ids: Vec<u32>,
}

fn de_token_ids<'de, D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Vec<u32>, D::Error> {
    serde_json::Value::deserialize(d).map(|v| token_ids(&v))
}

/// Load `<model_dir>/generation_config.json`. `Ok(None)` when the file does
/// not exist; an error naming it when it exists and cannot be read or parsed.
pub fn load_generation_config(model_dir: &Path) -> Result<Option<GenerationConfig>> {
    let path = model_dir.join("generation_config.json");
    let data = match std::fs::read(&path) {
        Ok(data) => data,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(Error::Loader(format!(
                "cannot read {}: {e}",
                path.display()
            )))
        }
    };
    serde_json::from_slice(&data)
        .map(Some)
        .map_err(|e| Error::Loader(format!("malformed {}: {e}", path.display())))
}

impl ModelConfig {
    /// The token ids that end a generation: the `config.json` ids in their
    /// order, then each `generation_config.json` id they lack. Empty when
    /// neither file names one.
    ///
    /// HuggingFace `generate` and mlx-lm stop at the `generation_config.json`
    /// ids and read `config.json` only when that file names none. The union
    /// keeps every id `config.json` names as well, so no stop is lost.
    pub fn stop_token_ids(&self, generation: Option<&GenerationConfig>) -> Vec<u32> {
        let mut ids = self.eos_token_ids();
        for &id in generation.map_or(&[][..], |g| &g.eos_token_ids) {
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        ids
    }
}

/// [`ModelConfig::stop_token_ids`] of the snapshot at `model_dir`, read from
/// its `config.json` and [`load_generation_config`].
pub fn stop_token_ids(model_dir: &Path) -> Result<Vec<u32>> {
    let generation = load_generation_config(model_dir)?;
    Ok(load_config(model_dir)?.stop_token_ids(generation.as_ref()))
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
