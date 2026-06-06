//! Gemma4 config parsing: text, vision, and audio sub-configs.
//!
//! Deserializes the `config.json` sub-objects for `Gemma4ForConditionalGeneration`.
//! The top-level model config is handled by [`rmlx_loader::config::ModelConfig`];
//! this module owns the Gemma4-specific fields within `text_config`,
//! `vision_config`, and `audio_config`.
//!
//! # Public API
//!
//! - [`Gemma4TextConfig`] — text-decoder architecture config (layers, heads,
//!   hidden dim, MoE params, altup, SWA window, etc.).
//! - [`LayerType`] — whether a decoder layer uses full or sliding-window attention.
//! - [`Gemma4VisionConfig`] — vision encoder config (patch size, channels, …).
//! - [`Gemma4AudioConfig`] — audio encoder config.

#![allow(clippy::too_many_lines)]
use std::collections::HashMap;

use rmlx_core::error::{Error, Result};

/// Subset of `text_config` needed for the text forward pass.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Gemma4 text-forward contract; adding a field requires updating Gemma4TextConfig::from_model_config and all Gemma4 layer constructors"
)]
#[derive(Debug, Clone)]
pub struct Gemma4TextConfig {
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads for sliding-attention layers (GQA).
    pub num_key_value_heads: usize,
    /// KV head count for full-attention layers when `attention_k_eq_v=true`.
    /// Set to `num_global_key_value_heads` from config when present (26B/31B),
    /// otherwise falls back to `num_key_value_heads`.
    pub num_global_key_value_heads: usize,
    /// Per-head dimension for sliding-attention layers.
    pub head_dim: usize,
    /// Per-head dimension for full-attention layers.
    pub global_head_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Sliding-window attention context length.
    pub sliding_window: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Whether lm_head shares weights with the embedding table.
    pub tie_word_embeddings: bool,
    /// Number of trailing layers sharing a single KV cache (AltUp).
    pub num_kv_shared_layers: usize,
    /// AltUp per-layer input hidden size; 0 if unused.
    pub hidden_size_per_layer_input: usize,
    /// Logit soft-capping value (30.0 on e4b).
    pub final_logit_softcapping: f32,
    /// Per-layer attention type (42 entries).
    pub layer_types: Vec<LayerType>,
    // Quant params (global defaults).
    /// Quantization group size.
    pub quant_group_size: i32,
    /// Quantization bit-width.
    pub quant_bits: i32,
    /// Quantization mode string (e.g. `"mxfp8"`, `"affine"`).
    pub quant_mode: String,
    /// Per-tensor quant overrides (keyed by tensor base path, e.g. "language_model.model.layers.0.router.proj").
    pub quant_overrides: HashMap<String, (i32, i32, String)>,
    // RoPE
    /// RoPE theta for sliding-attention layers.
    pub rope_sliding_theta: f32,
    /// RoPE theta for full-attention layers.
    pub rope_full_theta: f32,
    /// Rotated dims for full attention (partial_rotary_factor * global_head_dim).
    pub rope_full_dims: i32,
    /// When true (26B/31B), full-attention layers share K=V: the snapshot stores
    /// only k_proj and omits v_proj. Loader reuses k_proj weights as v_proj.
    /// Set from `text_config.attention_k_eq_v`. Reference: mlx-lm gemma4_text.py
    /// `self.use_k_eq_v = config.attention_k_eq_v and not self.is_sliding`.
    pub attention_k_eq_v: bool,
    /// MoE block enabled (26B model). When true every layer has both a dense MLP
    /// and a sparse MoE block whose outputs are summed (each post-normed separately).
    /// Set from `text_config.enable_moe_block`.
    pub enable_moe_block: bool,
    /// Total number of MoE experts per layer (0 if no MoE).
    pub num_experts: usize,
    /// Number of experts selected per token.
    pub top_k_experts: usize,
    /// MoE expert FFN intermediate dimension.
    pub moe_intermediate_size: usize,
    /// From `text_config.max_position_embeddings`. Used to size the pre-allocated
    /// KV buffer (Stage-3.1). Capped to `KV_MAX_SEQ_DEFAULT` at runtime if absent.
    pub max_position_embeddings: u32,
}

#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two Gemma4 attention-layer types (SlidingAttention/FullAttention); adding a type requires updating all layer_types match arms and the from_model_config parser"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Attention-layer type for one Gemma4 decoder layer.
pub enum LayerType {
    /// Local sliding-window attention.
    SlidingAttention,
    /// Global full-context attention.
    FullAttention,
}

/// Gemma4 SigLIP-style vision tower config (parsed from `config.json`
/// `vision_config`). `None` for text-only checkpoints.
///
/// Reference: `mlx_vlm/models/gemma4/config.py` `VisionConfig` and the e4b
/// snapshot `vision_config` block.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Gemma4 vision-tower contract; adding a field requires updating Gemma4VisionConfig::from_json and all vision-tower constructors"
)]
#[derive(Debug, Clone)]
pub struct Gemma4VisionConfig {
    /// Vision encoder hidden dimension.
    pub hidden_size: usize,
    /// Vision encoder FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Number of transformer layers in the vision encoder.
    pub num_hidden_layers: usize,
    /// Number of attention heads in the vision encoder.
    pub num_attention_heads: usize,
    /// Number of KV heads in the vision encoder.
    pub num_key_value_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Spatial patch size (16). `input_proj` in dim = `3 * patch_size^2`.
    pub patch_size: usize,
    /// One-hot position-embedding table length per axis (10240).
    pub position_embedding_size: usize,
    /// Spatial pooling kernel applied by `VisionPooler` (3).
    pub pooling_kernel_size: usize,
    /// Pooled output token budget (280). Drives `max_patches` padding.
    pub default_output_length: usize,
    /// Multidimensional-RoPE base frequency (`rope_parameters.rope_theta`, 100).
    pub rope_theta: f32,
    /// `use_clipped_linears` — when true every Linear is a `ClippableLinear`
    /// carrying `input_min/max` + `output_min/max` clamp buffers.
    pub use_clipped_linears: bool,
    /// `standardize` — apply `(h - std_bias) * std_scale` after pooling.
    pub standardize: bool,
}

impl Gemma4VisionConfig {
    /// `max_patches = default_output_length * pooling_kernel_size^2` — the
    /// padded patch count fed through the encoder.
    #[inline]
    pub fn max_patches(&self) -> usize {
        self.default_output_length * self.pooling_kernel_size * self.pooling_kernel_size
    }

    /// Parse from the `vision_config` JSON object. Missing keys fall back to
    /// the documented Gemma4 defaults (the e4b `gemma4_vision` values).
    pub fn from_json(v: &serde_json::Value) -> Self {
        let u = |key: &str, dflt: usize| -> usize {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(dflt, |x| x as usize)
        };
        let f = |key: &str, dflt: f32| -> f32 {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .map_or(dflt, |x| x as f32)
        };
        let b = |key: &str, dflt: bool| -> bool {
            v.get(key)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(dflt)
        };
        let num_attention_heads = u("num_attention_heads", 12);
        let rope_theta = v
            .get("rope_parameters")
            .and_then(|p| p.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .map_or(100.0, |x| x as f32);
        Self {
            hidden_size: u("hidden_size", 768),
            intermediate_size: u("intermediate_size", 3072),
            num_hidden_layers: u("num_hidden_layers", 16),
            num_attention_heads,
            num_key_value_heads: u("num_key_value_heads", num_attention_heads),
            head_dim: u("head_dim", 64),
            rms_norm_eps: f("rms_norm_eps", 1e-6),
            patch_size: u("patch_size", 16),
            position_embedding_size: u("position_embedding_size", 10240),
            pooling_kernel_size: u("pooling_kernel_size", 3),
            default_output_length: u("default_output_length", 280),
            rope_theta,
            use_clipped_linears: b("use_clipped_linears", false),
            standardize: b("standardize", false),
        }
    }

    /// Read `vision_config` from a model directory's `config.json`. Returns
    /// `None` for text-only checkpoints (no `vision_config` key).
    pub fn from_model_dir(model_dir: &std::path::Path) -> Result<Option<Self>> {
        let path = model_dir.join("config.json");
        let data = std::fs::read(&path)
            .map_err(|e| Error::Config(format!("gemma4: cannot read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "gemma4: malformed config.json at {}: {e}",
                path.display()
            ))
        })?;
        Ok(v.get("vision_config").map(Self::from_json))
    }
}

/// Gemma4 Conformer audio tower config (parsed from `config.json`
/// `audio_config`). `None` for checkpoints without an audio tower.
///
/// Reference: `mlx_vlm/models/gemma4/config.py` `AudioConfig` and the e4b
/// snapshot `audio_config` block.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Gemma4 audio-tower contract; adding a field requires updating Gemma4AudioConfig::from_json and all audio-tower constructors"
)]
#[derive(Debug, Clone)]
pub struct Gemma4AudioConfig {
    /// Audio encoder hidden dimension.
    pub hidden_size: usize,
    /// Number of Conformer layers in the audio encoder.
    pub num_hidden_layers: usize,
    /// Number of attention heads per Conformer block.
    pub num_attention_heads: usize,
    /// SSCP per-stage conv output channels (`[128, 32]` on e4b).
    pub subsampling_conv_channels: Vec<usize>,
    /// LightConv1d depthwise kernel size (5).
    pub conv_kernel_size: usize,
    /// Macaron FFW residual scaling (0.5).
    pub residual_weight: f32,
    /// Chunk size for chunked local attention.
    pub attention_chunk_size: usize,
    /// Left context tokens for chunked attention.
    pub attention_context_left: usize,
    /// Right context tokens for chunked attention.
    pub attention_context_right: usize,
    /// Soft logit cap for attention scores.
    pub attention_logit_cap: f32,
    /// Fill value for out-of-context attention positions.
    pub attention_invalid_logits_value: f32,
    /// When true every Linear is a `ClippableLinear` with clamp buffers.
    pub use_clipped_linears: bool,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Gradient clipping threshold (training artefact, informational).
    pub gradient_clipping: f32,
    /// Output projection target dim (1536). `None` → no `output_proj`.
    pub output_proj_dims: Option<usize>,
    /// `<audio_soft_token>` id (top-level `audio_token_id`, 258881 on e4b).
    pub audio_token_id: u32,
}

impl Gemma4AudioConfig {
    /// Parse from the `audio_config` JSON object. `audio_token_id` is the
    /// top-level config value. Missing keys fall back to the documented Gemma4
    /// defaults (the e4b values).
    pub fn from_json(v: &serde_json::Value, audio_token_id: u32) -> Self {
        let u = |key: &str, dflt: usize| -> usize {
            v.get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(dflt, |x| x as usize)
        };
        let f = |key: &str, dflt: f32| -> f32 {
            v.get(key)
                .and_then(serde_json::Value::as_f64)
                .map_or(dflt, |x| x as f32)
        };
        let b = |key: &str, dflt: bool| -> bool {
            v.get(key)
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(dflt)
        };
        let subsampling_conv_channels = v
            .get("subsampling_conv_channels")
            .and_then(serde_json::Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect::<Vec<_>>()
            })
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![128, 32]);
        let output_proj_dims = v
            .get("output_proj_dims")
            .and_then(serde_json::Value::as_u64)
            .map(|x| x as usize);
        Self {
            hidden_size: u("hidden_size", 1024),
            num_hidden_layers: u("num_hidden_layers", 12),
            num_attention_heads: u("num_attention_heads", 8),
            subsampling_conv_channels,
            conv_kernel_size: u("conv_kernel_size", 5),
            residual_weight: f("residual_weight", 0.5),
            attention_chunk_size: u("attention_chunk_size", 12),
            attention_context_left: u("attention_context_left", 13),
            attention_context_right: u("attention_context_right", 0),
            attention_logit_cap: f("attention_logit_cap", 50.0),
            attention_invalid_logits_value: f("attention_invalid_logits_value", -1e9),
            use_clipped_linears: b("use_clipped_linears", true),
            rms_norm_eps: f("rms_norm_eps", 1e-6),
            gradient_clipping: f("gradient_clipping", 1e10),
            output_proj_dims,
            audio_token_id,
        }
    }

    /// Read `audio_config` from a model directory's `config.json`. Returns
    /// `None` for checkpoints without an `audio_config` key.
    pub fn from_model_dir(model_dir: &std::path::Path) -> Result<Option<Self>> {
        let path = model_dir.join("config.json");
        let data = std::fs::read(&path)
            .map_err(|e| Error::Config(format!("gemma4: cannot read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "gemma4: malformed config.json at {}: {e}",
                path.display()
            ))
        })?;
        let audio_token_id = v
            .get("audio_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(258881) as u32;
        Ok(v.get("audio_config")
            .map(|ac| Self::from_json(ac, audio_token_id)))
    }
}

impl Gemma4TextConfig {
    /// Parse from a `ModelConfig` loaded by the loader crate.
    ///
    /// `raw_quant`: the raw JSON value of the `quantization` key, needed to extract
    /// inline per-tensor overrides that Serde drops when parsing into `QuantConfig`.
    pub fn from_model_config(
        cfg: &rmlx_loader::ModelConfig,
        raw_quant: Option<&serde_json::Value>,
    ) -> Result<Self> {
        let tc = cfg.text_config.as_ref().ok_or_else(|| {
            Error::Config("gemma4: missing text_config in config.json".to_owned())
        })?;

        // Quant defaults from top-level.
        let (qgs, qbits, qmode) = if let Some(q) = &cfg.quantization {
            (
                q.group_size as i32,
                i32::from(q.bits),
                q.mode_or_default().to_owned(),
            )
        } else {
            (32, 8, "mxfp8".to_owned())
        };

        let hidden_size = tc.hidden_size.unwrap_or(2560) as usize;
        let num_attention_heads = tc.num_attention_heads.unwrap_or(8) as usize;
        let num_key_value_heads = tc.num_key_value_heads.unwrap_or(2) as usize;
        let num_hidden_layers = tc.num_hidden_layers.unwrap_or(42) as usize;

        // Extra text_config fields live in tc.extras (via the flatten on TextConfig).
        let tc_extras = &tc.extras;

        let head_dim = tc_extras
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(256) as usize;
        let global_head_dim = tc_extras
            .get("global_head_dim")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(512) as usize;
        let intermediate_size = tc_extras
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10240) as usize;
        let vocab_size = tc_extras
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(262144) as usize;
        let sliding_window = tc.sliding_window.unwrap_or(512) as usize;
        let rms_norm_eps = tc_extras
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;
        let tie_word_embeddings = tc_extras
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);
        let num_kv_shared_layers = tc_extras
            .get("num_kv_shared_layers")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(18) as usize;
        let hidden_size_per_layer_input = tc_extras
            .get("hidden_size_per_layer_input")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let final_logit_softcapping = tc_extras
            .get("final_logit_softcapping")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(30.0) as f32;

        let attention_k_eq_v = tc_extras
            .get("attention_k_eq_v")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        // num_global_key_value_heads: used as KV head count for full-attention layers
        // when attention_k_eq_v=true (26B has 2, 31B has 4). Falls back to num_key_value_heads.
        let num_global_key_value_heads = tc_extras
            .get("num_global_key_value_heads")
            .and_then(serde_json::Value::as_u64)
            .map_or(num_key_value_heads, |v| v as usize);

        // layer_types: prefer TextConfig.layer_types (the typed field), fall back to extras.
        let layer_types: Vec<LayerType> = if let Some(lt_vec) = &tc.layer_types {
            lt_vec
                .iter()
                .map(|s| match s.as_str() {
                    "full_attention" => LayerType::FullAttention,
                    _ => LayerType::SlidingAttention,
                })
                .collect()
        } else if let Some(arr) = tc_extras.get("layer_types").and_then(|v| v.as_array()) {
            arr.iter()
                .map(|v| match v.as_str().unwrap_or("sliding_attention") {
                    "full_attention" => LayerType::FullAttention,
                    _ => LayerType::SlidingAttention,
                })
                .collect()
        } else {
            // Default: 5 sliding + 1 full, repeated 7×.
            (0..num_hidden_layers)
                .map(|i| {
                    if i % 6 == 5 {
                        LayerType::FullAttention
                    } else {
                        LayerType::SlidingAttention
                    }
                })
                .collect()
        };

        // RoPE parameters.
        let (rope_sliding_theta, rope_full_theta, rope_full_dims) = {
            let rope_params = tc_extras.get("rope_parameters");
            let sliding_theta = rope_params
                .and_then(|p| p.get("sliding_attention"))
                .and_then(|p| p.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(10_000.0) as f32;
            let full_theta = rope_params
                .and_then(|p| p.get("full_attention"))
                .and_then(|p| p.get("rope_theta"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(1_000_000.0) as f32;
            let partial_factor = rope_params
                .and_then(|p| p.get("full_attention"))
                .and_then(|p| p.get("partial_rotary_factor"))
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.25) as f32;
            let full_dims = (partial_factor * global_head_dim as f32) as i32;
            (sliding_theta, full_theta, full_dims)
        };

        let enable_moe_block = tc_extras
            .get("enable_moe_block")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let num_experts = tc_extras
            .get("num_experts")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let top_k_experts = tc_extras
            .get("top_k_experts")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;
        let moe_intermediate_size = tc_extras
            .get("moe_intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0) as usize;

        let max_position_embeddings = tc.max_position_embeddings.unwrap_or(0);

        // Parse per-tensor quant overrides from the raw quantization dict.
        // Keys like "language_model.model.layers.N.router.proj" carry {group_size, bits} overrides.
        let mut quant_overrides: HashMap<String, (i32, i32, String)> = HashMap::new();
        if let Some(quant_obj) = raw_quant.and_then(|v| v.as_object()) {
            for (key, val) in quant_obj {
                if matches!(
                    key.as_str(),
                    "group_size" | "bits" | "mode" | "tensor_overrides"
                ) {
                    continue;
                }
                if let Some(obj) = val.as_object() {
                    let ov_gs = obj
                        .get("group_size")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(qgs as u64) as i32;
                    let ov_bits = obj
                        .get("bits")
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(qbits as u64) as i32;
                    let ov_mode = obj
                        .get("mode")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_owned();
                    quant_overrides.insert(key.clone(), (ov_gs, ov_bits, ov_mode));
                }
            }
        }

        Ok(Gemma4TextConfig {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            num_global_key_value_heads,
            head_dim,
            global_head_dim,
            intermediate_size,
            vocab_size,
            sliding_window,
            rms_norm_eps,
            tie_word_embeddings,
            num_kv_shared_layers,
            hidden_size_per_layer_input,
            final_logit_softcapping,
            layer_types,
            quant_group_size: qgs,
            quant_bits: qbits,
            quant_mode: qmode,
            rope_sliding_theta,
            rope_full_theta,
            rope_full_dims,
            attention_k_eq_v,
            enable_moe_block,
            num_experts,
            top_k_experts,
            moe_intermediate_size,
            max_position_embeddings,
            quant_overrides,
        })
    }
}
