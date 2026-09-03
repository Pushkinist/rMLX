//! Config parser for `Qwen3VLMoeForConditionalGeneration` (`qwen3_vl_moe`).
//!
//! The top-level config nests two sub-configs:
//! - `text_config` (`qwen3_vl_moe_text`) — a **plain Qwen3-MoE GQA** decoder
//!   (NOT the Qwen3-Next GatedDeltaNet hybrid in `qwen3_5_moe`): full rotary,
//!   per-head q_norm/k_norm, MoE every layer, no shared expert.
//! - `vision_config` (`qwen3_vl_moe`) — the Qwen3-VL ViT (LayerNorm blocks,
//!   GELU-tanh MLP, learned pos-embed interpolation, deepstack mergers).
//!
//! Faithful to `mlx-vlm/mlx_vlm/models/qwen3_vl_moe/config.py`.

#![allow(clippy::float_cmp, clippy::unnecessary_wraps)]
use std::collections::HashMap;

use rmlx_core::error::{Error, Result};

use crate::layers::QuantParams;

// ---------------------------------------------------------------------------
// Text sub-config (plain Qwen3-MoE)
// ---------------------------------------------------------------------------

/// `text_config` — plain Qwen3-MoE GQA decoder.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Qwen3-VL-MoE text-decoder contract; adding a field requires updating from_raw and all qwen3_vl_moe layer constructors"
)]
#[derive(Debug, Clone)]
pub struct Qwen3VlMoeTextConfig {
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
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
    /// Total number of MoE experts per layer.
    pub num_experts: usize,
    /// Number of experts selected per token.
    pub num_experts_per_tok: usize,
    /// Every N-th layer uses MoE; 1 means every layer.
    pub decoder_sparse_step: usize,
    /// MoE expert FFN intermediate dimension.
    pub moe_intermediate_size: usize,
    /// Normalize top-k routing weights to sum to 1.
    pub norm_topk_prob: bool,
    /// Layer indices forced to use a dense MLP instead of MoE (usually empty).
    pub mlp_only_layers: Vec<usize>,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// `rope_scaling.mrope_section` — the three (T, H, W) section widths.
    /// Sums to `head_dim / 2`.
    pub mrope_section: Vec<usize>,
    /// `rope_scaling.mrope_interleaved` — true for Qwen3-VL (interleaved layout).
    pub mrope_interleaved: bool,
    /// Maximum sequence length from config.
    pub max_position_embeddings: u32,
    /// Positional capacity of this checkpoint. The generate paths read this
    /// rather than `max_position_embeddings`, so the fold from raw field to
    /// context bound happens once, here.
    pub context: crate::context::ContextLimits,
    // Quant (shared global config).
    /// Quantization group size.
    pub quant_group_size: i32,
    /// Quantization bit-width.
    pub quant_bits: i32,
    /// Quantization mode string.
    pub quant_mode: String,
    /// Per-tensor quantization overrides.
    pub quant_overrides: HashMap<String, QuantParams>,
}

// ---------------------------------------------------------------------------
// Vision sub-config (Qwen3-VL ViT)
// ---------------------------------------------------------------------------

/// `vision_config` — Qwen3-VL vision transformer.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Qwen3-VL vision-tower contract; adding a field requires updating from_raw and all qwen3_vl_moe vision constructors"
)]
#[derive(Debug, Clone)]
pub struct Qwen3VlMoeVisionConfig {
    /// Number of transformer layers in the vision tower.
    pub depth: usize,
    /// Vision encoder hidden dimension.
    pub hidden_size: usize,
    /// Vision encoder FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Projection output dimension (must match text decoder hidden size).
    pub out_hidden_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of input image channels (3 = RGB).
    pub in_channels: usize,
    /// Spatial patch size in pixels.
    pub patch_size: usize,
    /// Spatial merge factor applied after the encoder (2 = 2×2 → 1 token).
    pub spatial_merge_size: usize,
    /// Temporal patch size for video inputs.
    pub temporal_patch_size: usize,
    /// Learned position embedding table length.
    pub num_position_embeddings: usize,
    /// Layer indices whose post-block hidden is captured + merged into a
    /// deepstack feature (additively injected into the matching decoder layer).
    pub deepstack_visual_indexes: Vec<usize>,
    /// LayerNorm epsilon for the vision encoder.
    pub layer_norm_eps: f32,
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Parsed `Qwen3VLMoeForConditionalGeneration` config.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Qwen3-VL-MoE model contract; adding a field requires updating from_raw and the Qwen3VlMoe loader"
)]
#[derive(Debug, Clone)]
pub struct Qwen3VlMoeConfig {
    /// Parsed text-decoder sub-config.
    pub text: Qwen3VlMoeTextConfig,
    /// Parsed vision-tower sub-config.
    pub vision: Qwen3VlMoeVisionConfig,
    /// Token id for `<image>` soft tokens.
    pub image_token_id: i64,
    /// Token id for `<video>` soft tokens.
    pub video_token_id: i64,
    /// Token id for `<vision_start>`.
    pub vision_start_token_id: i64,
    /// Token id for `<vision_end>`.
    pub vision_end_token_id: i64,
}

impl Qwen3VlMoeConfig {
    /// Parse from the raw top-level config JSON object plus the global
    /// quantization block (shared across text + vision projections).
    pub fn from_raw(
        raw: &serde_json::Map<String, serde_json::Value>,
        quant: (i32, i32, String, HashMap<String, QuantParams>),
    ) -> Result<Self> {
        let tc = raw
            .get("text_config")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Config("qwen3_vl_moe: missing text_config".into()))?;
        let vc = raw
            .get("vision_config")
            .and_then(|v| v.as_object())
            .ok_or_else(|| Error::Config("qwen3_vl_moe: missing vision_config".into()))?;

        let text = parse_text(tc, quant)?;
        let vision = parse_vision(vc)?;

        let image_token_id = raw
            .get("image_token_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(151655);
        let video_token_id = raw
            .get("video_token_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(151656);
        let vision_start_token_id = raw
            .get("vision_start_token_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(151652);
        let vision_end_token_id = raw
            .get("vision_end_token_id")
            .and_then(serde_json::Value::as_i64)
            .unwrap_or(151653);

        Ok(Qwen3VlMoeConfig {
            text,
            vision,
            image_token_id,
            video_token_id,
            vision_start_token_id,
            vision_end_token_id,
        })
    }
}

fn parse_text(
    e: &serde_json::Map<String, serde_json::Value>,
    quant: (i32, i32, String, HashMap<String, QuantParams>),
) -> Result<Qwen3VlMoeTextConfig> {
    macro_rules! req_u64 {
        ($key:expr) => {
            e.get($key)
                .and_then(|v| v.as_u64())
                .ok_or_else(|| Error::Config(format!("qwen3_vl_moe text: missing {}", $key)))?
                as usize
        };
    }

    let hidden_size = req_u64!("hidden_size");
    let num_attention_heads = req_u64!("num_attention_heads");
    let head_dim = e
        .get("head_dim")
        .and_then(serde_json::Value::as_u64)
        .map_or(hidden_size / num_attention_heads, |v| v as usize);
    let num_key_value_heads = e
        .get("num_key_value_heads")
        .and_then(serde_json::Value::as_u64)
        .map_or(num_attention_heads, |v| v as usize);

    let rope_scaling = e.get("rope_scaling").and_then(|v| v.as_object());
    let mrope_section: Vec<usize> = rope_scaling
        .and_then(|m| m.get("mrope_section"))
        .and_then(|v| v.as_array())
        .map_or_else(
            || vec![24, 20, 20],
            |a| {
                a.iter()
                    .filter_map(|x| x.as_u64().map(|n| n as usize))
                    .collect()
            },
        );
    let mrope_interleaved = rope_scaling
        .and_then(|m| m.get("mrope_interleaved"))
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(false);

    let mlp_only_layers: Vec<usize> = e
        .get("mlp_only_layers")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();

    let mpe = e
        .get("max_position_embeddings")
        .and_then(serde_json::Value::as_u64)
        .map_or(0, |v| v as u32);

    Ok(Qwen3VlMoeTextConfig {
        num_hidden_layers: req_u64!("num_hidden_layers"),
        hidden_size,
        intermediate_size: req_u64!("intermediate_size"),
        num_attention_heads,
        num_key_value_heads,
        head_dim,
        vocab_size: req_u64!("vocab_size"),
        rms_norm_eps: e
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32,
        tie_word_embeddings: e
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false),
        num_experts: req_u64!("num_experts"),
        num_experts_per_tok: req_u64!("num_experts_per_tok"),
        decoder_sparse_step: e
            .get("decoder_sparse_step")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1) as usize,
        moe_intermediate_size: req_u64!("moe_intermediate_size"),
        norm_topk_prob: e
            .get("norm_topk_prob")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true),
        mlp_only_layers,
        rope_theta: e
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(5_000_000.0) as f32,
        mrope_section,
        mrope_interleaved,
        max_position_embeddings: mpe,
        // Qwen3-VL-MoE implements no RoPE scaling: its trained window is the
        // limit.
        context: crate::context::ContextLimits::trained_only(mpe as i32),
        quant_group_size: quant.0,
        quant_bits: quant.1,
        quant_mode: quant.2,
        quant_overrides: quant.3,
    })
}

fn parse_vision(e: &serde_json::Map<String, serde_json::Value>) -> Result<Qwen3VlMoeVisionConfig> {
    macro_rules! opt_u64 {
        ($key:expr, $default:expr) => {
            e.get($key).and_then(|v| v.as_u64()).unwrap_or($default) as usize
        };
    }
    let deepstack_visual_indexes: Vec<usize> = e
        .get("deepstack_visual_indexes")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_u64().map(|n| n as usize))
                .collect()
        })
        .unwrap_or_default();

    Ok(Qwen3VlMoeVisionConfig {
        depth: opt_u64!("depth", 27),
        hidden_size: opt_u64!("hidden_size", 1152),
        intermediate_size: opt_u64!("intermediate_size", 4304),
        out_hidden_size: opt_u64!("out_hidden_size", 2048),
        num_heads: opt_u64!("num_heads", 16),
        in_channels: opt_u64!("in_channels", 3),
        patch_size: opt_u64!("patch_size", 16),
        spatial_merge_size: opt_u64!("spatial_merge_size", 2),
        temporal_patch_size: opt_u64!("temporal_patch_size", 2),
        num_position_embeddings: opt_u64!("num_position_embeddings", 2304),
        deepstack_visual_indexes,
        layer_norm_eps: e
            .get("layer_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32,
    })
}

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
