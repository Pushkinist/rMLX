//! jina-embeddings-v4 config parsing.
//!
//! Parses `/config.json` from a `jinaai/jina-embeddings-v4` snapshot.
//! Fields map to the nested `text_config` / `vision_config` sub-objects
//! plus the top-level projector / pooling / task metadata.

#![allow(clippy::cognitive_complexity, clippy::too_many_lines)]
use std::path::Path;

use rmlx_core::error::{Error, Result};

// ---------------------------------------------------------------------------
// Text sub-config
// ---------------------------------------------------------------------------

/// Fields from `text_config` (Qwen2.5-VL-3B backbone, plain bf16).
///
/// `head_dim` is inferred as `hidden_size / num_attention_heads` when absent
/// (the jina-v4 config.json does not carry an explicit `head_dim` field).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete JinaV4 text-backbone contract; adding a field requires updating JinaV4Config::from_model_dir and all jina-v4 layer constructors"
)]
#[derive(Debug, Clone)]
pub struct JinaV4TextConfig {
    /// Text backbone hidden dimension.
    pub hidden_size: usize,
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_key_value_heads: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency (f64 to match jina config.json).
    pub rope_theta: f64,
    /// Inferred: `hidden_size / num_attention_heads`.
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Raw `sliding_window` value from JSON (null → 0 → disabled).
    pub sliding_window: Option<u32>,
    /// Whether sliding window attention is active.
    pub use_sliding_window: bool,
    /// Maximum sequence length from config.
    pub max_position_embeddings: u32,
}

// ---------------------------------------------------------------------------
// Vision sub-config
// ---------------------------------------------------------------------------

/// Fields from `vision_config` (32-layer ViT, window + full-attn pattern).
///
/// Note: vision MLP and attn.proj use `bias=True` (jina-specific porting trap;
/// stock mlx_vlm uses `bias=False`). Recorded here as a doc reminder; actual
/// layer construction is a later subtask.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete JinaV4 vision-tower contract; adding a field requires updating JinaV4Config::from_model_dir and all jina-v4 vision constructors"
)]
#[derive(Debug, Clone)]
pub struct JinaV4VisionConfig {
    /// Number of transformer blocks.
    pub depth: usize,
    /// Vision encoder hidden dimension.
    pub hidden_size: usize,
    /// Vision encoder FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Output projection size (maps to text hidden_size).
    pub out_hidden_size: usize,
    /// Block indexes that use full (global) attention; all others use window.
    pub fullatt_block_indexes: Vec<usize>,
    /// Window attention size (blocks not in `fullatt_block_indexes` use this).
    pub window_size: usize,
    /// Logical patch size (derived from spatial/temporal).
    pub patch_size: usize,
    /// Spatial patch size in pixels.
    pub spatial_patch_size: usize,
    /// Temporal patch size for video.
    pub temporal_patch_size: usize,
    /// Spatial merge factor (2 = 2×2 → 1 token).
    pub spatial_merge_size: usize,
    /// Number of input image channels (3 = RGB).
    pub in_channels: usize,
}

// ---------------------------------------------------------------------------
// Top-level config
// ---------------------------------------------------------------------------

/// Parsed configuration for `jinaai/jina-embeddings-v4`.
///
/// Jina-v4 is a pure-bf16 multimodal embedding encoder (NOT a causal LM).
/// It is standalone — it does NOT go through the `Architecture` enum.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete JinaV4 top-level model contract; adding a field requires updating from_model_dir and all jina-v4 model constructors"
)]
#[derive(Debug, Clone)]
pub struct JinaV4Config {
    /// Parsed text-backbone sub-config.
    pub text_config: JinaV4TextConfig,
    /// Parsed vision-tower sub-config.
    pub vision_config: JinaV4VisionConfig,

    /// Pooling strategy for single-vector output: `"mean"`.
    pub single_vector_pool_strategy: String,
    /// Output dim of the multi-vector projector Linear (2048 → 128).
    pub multi_vector_projector_dim: usize,
    /// Matryoshka truncation dimensions (ascending).
    pub matryoshka_dims: Vec<usize>,
    /// Runtime LoRA task names (e.g. `["retrieval", "text-matching", "code"]`).
    pub task_names: Vec<String>,
    /// 3D M-RoPE channel split `text_config.rope_scaling.mrope_section`
    /// (jina-v4: `[16, 24, 24]`, sums to `head_dim/2 = 64`). `None` when the
    /// config omits it (text-only models); the image path falls back to the
    /// jina default. Only consumed by the image path.
    pub mrope_section: Option<Vec<usize>>,

    // Special token ids used for image-span detection and text encoding.
    /// Token id for `<|vision_start|>`.
    pub vision_start_token_id: u32,
    /// Token id for `<|vision_end|>`.
    pub vision_end_token_id: u32,
    /// Token id for `<|vision_token|>` / `<|image_pad|>`.
    pub vision_token_id: u32,
    /// Alias for `vision_token_id` (HF field `image_token_id`).
    pub image_token_id: u32,
    /// Token id for the beginning-of-sequence token.
    pub bos_token_id: u32,
    /// Token id for the end-of-sequence token.
    pub eos_token_id: u32,
}

impl JinaV4Config {
    /// Parse from a `config.json` `serde_json::Value` (full file).
    ///
    /// Unknown top-level / nested keys are silently ignored — forward-compat.
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let obj = v.as_object().ok_or_else(|| {
            Error::Config("jina-v4: config.json root is not a JSON object".to_owned())
        })?;

        // ---- text_config -----------------------------------------------
        let tc = obj.get("text_config").ok_or_else(|| {
            Error::Config("jina-v4: missing text_config in config.json".to_owned())
        })?;

        let hidden_size = tc
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2048) as usize;
        let num_hidden_layers = tc
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(36) as usize;
        let num_attention_heads = tc
            .get("num_attention_heads")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(16) as usize;
        let num_key_value_heads = tc
            .get("num_key_value_heads")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2) as usize;
        let intermediate_size = tc
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(11008) as usize;
        let rms_norm_eps = tc
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;
        let rope_theta = tc
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1_000_000.0);
        let head_dim = tc
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .map_or(hidden_size / num_attention_heads, |v| v as usize);
        let vocab_size = tc
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(151936) as usize;
        let sliding_window: Option<u32> = tc
            .get("sliding_window")
            .and_then(serde_json::Value::as_u64)
            .map(|v| v as u32);
        let use_sliding_window = tc
            .get("use_sliding_window")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let max_position_embeddings = tc
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(128000) as u32;
        // text_config.rope_scaling.mrope_section (3D M-RoPE channel split).
        let mrope_section: Option<Vec<usize>> = tc
            .get("rope_scaling")
            .and_then(|rs| rs.get("mrope_section"))
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(serde_json::Value::as_u64)
                    .map(|x| x as usize)
                    .collect()
            });

        let text_config = JinaV4TextConfig {
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            intermediate_size,
            rms_norm_eps,
            rope_theta,
            head_dim,
            vocab_size,
            sliding_window,
            use_sliding_window,
            max_position_embeddings,
        };

        // ---- vision_config ---------------------------------------------
        let vc = obj.get("vision_config").ok_or_else(|| {
            Error::Config("jina-v4: missing vision_config in config.json".to_owned())
        })?;

        let v_depth = vc
            .get("depth")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(32) as usize;
        let v_hidden_size = vc
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(1280) as usize;
        let v_intermediate_size = vc
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(3420) as usize;
        let v_num_heads = vc
            .get("num_heads")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(16) as usize;
        let v_out_hidden_size = vc
            .get("out_hidden_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2048) as usize;
        let v_fullatt_block_indexes: Vec<usize> = vc
            .get("fullatt_block_indexes")
            .and_then(|v| v.as_array())
            .map_or_else(
                || vec![7, 15, 23, 31],
                |arr| {
                    arr.iter()
                        .filter_map(serde_json::Value::as_u64)
                        .map(|x| x as usize)
                        .collect()
                },
            );
        let v_window_size = vc
            .get("window_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(112) as usize;
        let v_patch_size = vc
            .get("patch_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(14) as usize;
        let v_spatial_patch_size = vc
            .get("spatial_patch_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(14) as usize;
        let v_temporal_patch_size = vc
            .get("temporal_patch_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2) as usize;
        let v_spatial_merge_size = vc
            .get("spatial_merge_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(2) as usize;
        let v_in_channels = vc
            .get("in_channels")
            .and_then(serde_json::Value::as_u64)
            .or_else(|| vc.get("in_chans").and_then(serde_json::Value::as_u64))
            .unwrap_or(3) as usize;

        let vision_config = JinaV4VisionConfig {
            depth: v_depth,
            hidden_size: v_hidden_size,
            intermediate_size: v_intermediate_size,
            num_heads: v_num_heads,
            out_hidden_size: v_out_hidden_size,
            fullatt_block_indexes: v_fullatt_block_indexes,
            window_size: v_window_size,
            patch_size: v_patch_size,
            spatial_patch_size: v_spatial_patch_size,
            temporal_patch_size: v_temporal_patch_size,
            spatial_merge_size: v_spatial_merge_size,
            in_channels: v_in_channels,
        };

        // ---- top-level fields ------------------------------------------
        let single_vector_pool_strategy = obj
            .get("single_vector_pool_strategy")
            .and_then(|v| v.as_str())
            .unwrap_or("mean")
            .to_owned();
        let multi_vector_projector_dim = obj
            .get("multi_vector_projector_dim")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(128) as usize;
        let matryoshka_dims: Vec<usize> = obj
            .get("matryoshka_dims")
            .and_then(|v| v.as_array())
            .map_or_else(
                || vec![128, 256, 512, 1024, 2048],
                |arr| {
                    arr.iter()
                        .filter_map(serde_json::Value::as_u64)
                        .map(|x| x as usize)
                        .collect()
                },
            );
        let task_names: Vec<String> = obj
            .get("task_names")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|x| x.as_str())
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();

        // ---- special token ids -----------------------------------------
        let vision_start_token_id = obj
            .get("vision_start_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(151652) as u32;
        let vision_end_token_id = obj
            .get("vision_end_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(151653) as u32;
        // `vision_token_id` is the canonical field; `image_token_id` is the alias.
        let vision_token_id = obj
            .get("vision_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(151654) as u32;
        let image_token_id = obj
            .get("image_token_id")
            .and_then(serde_json::Value::as_u64)
            .map_or(vision_token_id, |v| v as u32);
        let bos_token_id = obj
            .get("bos_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(151643) as u32;
        let eos_token_id = obj
            .get("eos_token_id")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(151645) as u32;

        Ok(JinaV4Config {
            text_config,
            vision_config,
            single_vector_pool_strategy,
            multi_vector_projector_dim,
            matryoshka_dims,
            task_names,
            mrope_section,
            vision_start_token_id,
            vision_end_token_id,
            vision_token_id,
            image_token_id,
            bos_token_id,
            eos_token_id,
        })
    }

    /// Parse directly from a `config.json` file on disk.
    pub fn from_file(config_path: &Path) -> Result<Self> {
        let data = std::fs::read(config_path).map_err(|e| {
            Error::Config(format!(
                "jina-v4: cannot read {}: {e}",
                config_path.display()
            ))
        })?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "jina-v4: malformed config.json at {}: {e}",
                config_path.display()
            ))
        })?;
        Self::from_json(&v)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "config_tests.rs"]
mod tests;
