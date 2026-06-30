//! Config parser for Qwen3_5MoeForConditionalGeneration and dense variants.

#![allow(trivial_numeric_casts)]

use std::collections::HashMap;

use rmlx_core::error::{Error, Result};

use crate::layers::QuantParams;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Qwen3.5-MoE model contract; adding a field requires updating from_model_config and all Qwen3_5MoE layer constructors"
)]
#[derive(Debug, Clone)]
/// Parsed Qwen3_5MoeForConditionalGeneration config (GatedDeltaNet hybrid + full-attn MoE).
pub struct Qwen3_5MoeConfig {
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Whether lm_head shares weights with the embedding table.
    pub tie_word_embeddings: bool,
    /// Period at which a full-attention layer replaces a GatedDeltaNet layer.
    pub full_attention_interval: usize,
    /// Dense SwiGLU FFN intermediate dimension. Used by the dense MLP variant
    /// (`Qwen3_5ForConditionalGeneration`); 0 for pure-MoE checkpoints.
    pub intermediate_size: usize,
    /// Total number of MoE experts per layer. 0 marks a dense checkpoint.
    pub num_experts: usize,
    /// Number of experts selected per token.
    pub num_experts_per_tok: usize,
    /// MoE expert FFN intermediate dimension.
    pub moe_intermediate_size: usize,
    /// Shared (always-active) expert intermediate dimension.
    pub shared_expert_intermediate_size: usize,
    /// Normalize top-k routing weights to sum to 1.
    pub norm_topk_prob: bool,
    // GatedDeltaNet dims
    /// Number of value heads in the GatedDeltaNet linear-attention block.
    pub linear_num_value_heads: usize,
    /// Number of key heads in the GatedDeltaNet linear-attention block.
    pub linear_num_key_heads: usize,
    /// Key head dimension in the GatedDeltaNet block.
    pub linear_key_head_dim: usize,
    /// Value head dimension in the GatedDeltaNet block.
    pub linear_value_head_dim: usize,
    /// Depthwise conv kernel size in the GatedDeltaNet block.
    pub linear_conv_kernel_dim: usize,
    // RoPE
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Number of rotated dimensions (`partial_rotary_factor * head_dim`).
    pub rope_dims: usize, // partial_rotary_factor * head_dim
    // Quant
    /// Quantization group size.
    pub quant_group_size: i32,
    /// Quantization bit-width.
    pub quant_bits: i32,
    /// Quantization mode string.
    pub quant_mode: String,
    /// Per-tensor quantization overrides keyed by tensor base name.
    pub quant_overrides: HashMap<String, QuantParams>,
    /// From `text_config.max_position_embeddings`. Used to size the KV buffer.
    /// 0 if absent (falls back to `KV_MAX_SEQ_DEFAULT`).
    pub max_position_embeddings: u32,
}

impl Qwen3_5MoeConfig {
    /// `raw_text_config`: the raw JSON value of the `text_config` key (re-read before
    /// Serde struct-parsing). When present, all model fields are read from this map.
    /// When absent (pure-text snapshot), falls back to `cfg.extras`.
    pub fn from_model_config(
        cfg: &rmlx_loader::ModelConfig,
        raw_quant: Option<&serde_json::Value>,
        raw_text_config: Option<&serde_json::Value>,
    ) -> Result<Self> {
        // Qwen3_5MoeForConditionalGeneration wraps a vision/multimodal config: all
        // text-model fields live under `text_config`, not at the top level.
        // raw_text_config is the raw JSON object (before Serde field splitting).
        let e: &serde_json::Map<String, serde_json::Value> = raw_text_config
            .and_then(|v| v.as_object())
            .unwrap_or(&cfg.extras);

        macro_rules! req_u64 {
            ($key:expr) => {
                e.get($key)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| Error::Config(format!("qwen3_5_moe: missing {}", $key)))?
                    as usize
            };
        }
        macro_rules! opt_u64 {
            ($key:expr, $default:expr) => {
                e.get($key)
                    .and_then(|v| v.as_u64())
                    .unwrap_or($default as u64) as usize
            };
        }

        let hidden_size = req_u64!("hidden_size");
        let num_hidden_layers = req_u64!("num_hidden_layers");
        let num_attention_heads = req_u64!("num_attention_heads");
        let num_key_value_heads = req_u64!("num_key_value_heads");
        let head_dim = e
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .map_or(hidden_size / num_attention_heads, |v| v as usize);
        let vocab_size = req_u64!("vocab_size");
        let rms_norm_eps = e
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;
        let tie_word_embeddings = e
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let full_attention_interval = opt_u64!("full_attention_interval", 4);
        // MoE-specific fields are optional: dense Qwen3.5 checkpoints
        // (`Qwen3_5ForConditionalGeneration` with a plain SwiGLU MLP) omit them.
        // `num_experts == 0` is the canonical "dense, no experts" marker the
        // loader keys its per-layer MLP detection on. `intermediate_size` is the
        // dense FFN width; `moe_intermediate_size` falls back to it so a dense
        // config still yields a sane value.
        let num_experts = opt_u64!("num_experts", 0);
        let num_experts_per_tok = opt_u64!("num_experts_per_tok", 1);
        let intermediate_size = opt_u64!("intermediate_size", 0);
        let moe_intermediate_size = opt_u64!("moe_intermediate_size", intermediate_size as u64);
        let shared_expert_intermediate_size =
            opt_u64!("shared_expert_intermediate_size", intermediate_size as u64);
        let norm_topk_prob = e
            .get("norm_topk_prob")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true); // Python default = True

        let linear_num_value_heads = opt_u64!("linear_num_value_heads", 64);
        let linear_num_key_heads = opt_u64!("linear_num_key_heads", 16);
        let linear_key_head_dim = opt_u64!("linear_key_head_dim", 128);
        let linear_value_head_dim = opt_u64!("linear_value_head_dim", 128);
        let linear_conv_kernel_dim = opt_u64!("linear_conv_kernel_dim", 4);

        // RoPE: read from rope_parameters if present.
        let rp = e.get("rope_parameters").and_then(|v| v.as_object());
        let rope_theta = rp
            .and_then(|m| m.get("rope_theta"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(10_000_000.0) as f32;
        let partial_rotary_factor = rp
            .and_then(|m| m.get("partial_rotary_factor"))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.25);
        let rope_dims = ((head_dim as f64) * partial_rotary_factor).round() as usize;

        // Global quant.
        let (quant_group_size, quant_bits, quant_mode, quant_overrides) =
            extract_quant(cfg, raw_quant);

        let max_position_embeddings = e
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32)
            .or_else(|| cfg.text_config.as_ref()?.max_position_embeddings)
            .unwrap_or(0);

        Ok(Qwen3_5MoeConfig {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            tie_word_embeddings,
            full_attention_interval,
            intermediate_size,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            norm_topk_prob,
            linear_num_value_heads,
            linear_num_key_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            rope_theta,
            rope_dims,
            quant_group_size,
            quant_bits,
            quant_mode,
            quant_overrides,
            max_position_embeddings,
        })
    }
}

pub(super) fn extract_quant(
    cfg: &rmlx_loader::ModelConfig,
    raw_quant_val: Option<&serde_json::Value>,
) -> (i32, i32, String, HashMap<String, QuantParams>) {
    let (gs, bits, mode) = if let Some(q) = &cfg.quantization {
        (
            q.group_size as i32,
            i32::from(q.bits),
            q.mode_or_default().to_owned(),
        )
    } else {
        (64, 8, "affine".to_owned())
    };

    let mut overrides: HashMap<String, QuantParams> = HashMap::new();
    if let Some(quant_obj) = raw_quant_val.and_then(|v| v.as_object()) {
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
                    .unwrap_or(gs as u64) as i32;
                let ov_bits = obj
                    .get("bits")
                    .and_then(serde_json::Value::as_u64)
                    .unwrap_or(bits as u64) as i32;
                let ov_mode = obj
                    .get("mode")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_owned();
                overrides.insert(
                    key.clone(),
                    QuantParams {
                        group_size: ov_gs,
                        bits: ov_bits,
                        mode: ov_mode,
                    },
                );
            }
        }
    }
    (gs, bits, mode, overrides)
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod config_tests;
