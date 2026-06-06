//! Gemma3 configuration types.

#![allow(clippy::manual_let_else)]
use rmlx_core::error::{Error, Result};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Subset of `text_config` needed for the Gemma3 text forward pass.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Gemma3 text-forward contract; adding a field requires updating Gemma3TextConfig::from_model_config and all Gemma3 layer constructors"
)]
#[derive(Debug, Clone)]
pub struct Gemma3TextConfig {
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Sliding-window attention context length.
    pub sliding_window: usize,
    /// Period of full-attention layers (`(layer_idx+1) % N == 0` means full).
    pub sliding_window_pattern: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// `query_pre_attn_scalar^-0.5` — pre-computed.
    pub attn_scale: f32,
    /// Whether lm_head shares weights with the embedding table.
    pub tie_word_embeddings: bool,
    /// `None` means no softcapping (medgemma config has `null`).
    pub final_logit_softcapping: Option<f32>,
    /// Per-layer attention type.
    pub layer_types: Vec<LayerType>,
    /// RoPE theta for sliding-attention layers (rope_local_base_freq).
    pub rope_local_theta: f32,
    /// RoPE theta for full-attention layers (rope_theta).
    pub rope_global_theta: f32,
    // Quant parameters.
    /// Quantization group size.
    pub quant_group_size: i32,
    /// Quantization bit-width.
    pub quant_bits: i32,
    /// Quantization mode string (e.g. `"affine"`).
    pub quant_mode: String,
}

#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two Gemma3 attention-layer types (SlidingAttention/FullAttention); adding a type requires updating all layer_types match arms and from_model_config"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Attention type for a Gemma3 decoder layer.
pub enum LayerType {
    /// Local sliding-window attention.
    SlidingAttention,
    /// Global full-context attention.
    FullAttention,
}

// ---------------------------------------------------------------------------
// Vision config (standard SigLIP, parsed from `vision_config`)
// ---------------------------------------------------------------------------

/// Gemma3 SigLIP vision tower config (`config.json` `vision_config`).
/// `None` for text-only checkpoints.
///
/// Reference: `mlx_vlm/models/gemma3/config.py` `VisionConfig` + the medgemma
/// `vision_config` block (`siglip_vision_model`, image 896, patch 14, 27 layers,
/// hidden 1152, 16 heads, intermediate 4304). Standard SigLIP — no
/// ClippableLinear, learned 1D position embeddings, full bidirectional MHA.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Gemma3 vision-tower contract; adding a field requires updating Gemma3VisionConfig::from_model_dir and all vision-tower constructors"
)]
#[derive(Debug, Clone)]
pub struct Gemma3VisionConfig {
    /// Vision encoder hidden dimension.
    pub hidden_size: usize,
    /// Vision encoder FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Number of transformer layers in the vision encoder.
    pub num_hidden_layers: usize,
    /// Number of MHA heads in the vision encoder.
    pub num_attention_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Spatial patch size in pixels (14 for medgemma).
    pub patch_size: usize,
    /// Input image size in pixels (896 for medgemma).
    pub image_size: usize,
    /// Number of input image channels (3 = RGB).
    pub num_channels: usize,
    /// LayerNorm epsilon for the vision encoder.
    pub layer_norm_eps: f32,
    /// Soft tokens emitted per image after AvgPool2d (`mm_tokens_per_image`, 256).
    pub mm_tokens_per_image: usize,
    /// Language-model hidden size (einsum projection target). Carried here so
    /// the projector + scatter scale (`1/sqrt(text_hidden)`) is self-contained.
    pub text_hidden_size: usize,
    /// `mm_soft_emb_norm` RMSNorm eps (`vision_config.layer_norm_eps`).
    pub mm_norm_eps: f32,
}

impl Gemma3VisionConfig {
    /// Patches along each spatial axis (`image_size / patch_size`, 64).
    #[inline]
    pub fn patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Soft tokens along each side after pooling (`sqrt(mm_tokens_per_image)`, 16).
    #[inline]
    pub fn tokens_per_side(&self) -> usize {
        (self.mm_tokens_per_image as f64).sqrt() as usize
    }

    /// AvgPool2d kernel/stride (`patches_per_side / tokens_per_side`, 4).
    #[inline]
    pub fn pool_kernel(&self) -> usize {
        self.patches_per_side() / self.tokens_per_side()
    }

    /// Read `vision_config` (+ `image_token_index`, `mm_tokens_per_image`, and
    /// `text_config.hidden_size`) from a model directory's `config.json`.
    /// Returns `None` for text-only checkpoints (no `vision_config` key).
    pub fn from_model_dir(model_dir: &std::path::Path) -> Result<Option<Self>> {
        let path = model_dir.join("config.json");
        let data = std::fs::read(&path)
            .map_err(|e| Error::Config(format!("gemma3: cannot read {}: {e}", path.display())))?;
        let v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| {
            Error::Config(format!(
                "gemma3: malformed config.json at {}: {e}",
                path.display()
            ))
        })?;
        let vc = match v.get("vision_config") {
            Some(vc) => vc,
            None => return Ok(None),
        };
        let u = |obj: &serde_json::Value, key: &str, dflt: usize| -> usize {
            obj.get(key)
                .and_then(serde_json::Value::as_u64)
                .map_or(dflt, |x| x as usize)
        };
        let f = |obj: &serde_json::Value, key: &str, dflt: f32| -> f32 {
            obj.get(key)
                .and_then(serde_json::Value::as_f64)
                .map_or(dflt, |x| x as f32)
        };
        let hidden_size = u(vc, "hidden_size", 1152);
        let num_attention_heads = u(vc, "num_attention_heads", 16);
        let head_dim = hidden_size / num_attention_heads;
        let mm_tokens_per_image = u(&v, "mm_tokens_per_image", 256);
        let text_hidden_size = v
            .get("text_config")
            .map_or(2560, |tc| u(tc, "hidden_size", 2560));
        let layer_norm_eps = f(vc, "layer_norm_eps", 1e-6);
        Ok(Some(Gemma3VisionConfig {
            hidden_size,
            intermediate_size: u(vc, "intermediate_size", 4304),
            num_hidden_layers: u(vc, "num_hidden_layers", 27),
            num_attention_heads,
            head_dim,
            patch_size: u(vc, "patch_size", 14),
            image_size: u(vc, "image_size", 896),
            num_channels: u(vc, "num_channels", 3),
            layer_norm_eps,
            mm_tokens_per_image,
            text_hidden_size,
            mm_norm_eps: layer_norm_eps,
        }))
    }
}

impl Gemma3TextConfig {
    /// Parse from a [`rmlx_loader::ModelConfig`] loaded from `config.json`.
    pub fn from_model_config(cfg: &rmlx_loader::ModelConfig) -> Result<Self> {
        let tc = cfg.text_config.as_ref().ok_or_else(|| {
            Error::Config("gemma3: missing text_config in config.json".to_owned())
        })?;

        let extras = &tc.extras;

        let num_hidden_layers = tc.num_hidden_layers.unwrap_or(34) as usize;
        let hidden_size = tc.hidden_size.unwrap_or(2560) as usize;
        let num_attention_heads = tc.num_attention_heads.unwrap_or(8) as usize;
        let num_key_value_heads = tc.num_key_value_heads.unwrap_or(4) as usize;
        let sliding_window = tc.sliding_window.unwrap_or(1024) as usize;

        let head_dim = extras
            .get("head_dim")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(256) as usize;
        let intermediate_size = extras
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(10240) as usize;
        let vocab_size = extras
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(262208) as usize;
        let rms_norm_eps = extras
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;

        // `_sliding_window_pattern` in this snapshot (prefixed underscore).
        let sliding_window_pattern = extras
            .get("_sliding_window_pattern")
            .or_else(|| extras.get("sliding_window_pattern"))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(6) as usize;

        let query_pre_attn_scalar = extras
            .get("query_pre_attn_scalar")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(256.0) as f32;
        let attn_scale = query_pre_attn_scalar.powf(-0.5);

        let tie_word_embeddings = extras
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);

        // `final_logit_softcapping` is null in medgemma → None.
        let final_logit_softcapping = extras
            .get("final_logit_softcapping")
            .and_then(serde_json::Value::as_f64)
            .filter(|&v| v > 0.0)
            .map(|v| v as f32);

        // layer_types from the typed TextConfig field (populated for medgemma).
        let layer_types: Vec<LayerType> = if let Some(lt_vec) = &tc.layer_types {
            lt_vec
                .iter()
                .map(|s| match s.as_str() {
                    "full_attention" => LayerType::FullAttention,
                    _ => LayerType::SlidingAttention,
                })
                .collect()
        } else {
            // Fallback: Gemma3 pattern `(layer_idx + 1) % sliding_window_pattern == 0`.
            // Reference: gemma3_text.py Attention.__init__ line 55:
            // `self.is_sliding = (layer_idx + 1) % args.sliding_window_pattern != 0`
            (0..num_hidden_layers)
                .map(|i| {
                    if (i + 1) % sliding_window_pattern == 0 {
                        LayerType::FullAttention
                    } else {
                        LayerType::SlidingAttention
                    }
                })
                .collect()
        };

        let rope_local_theta = extras
            .get("rope_local_base_freq")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(10_000.0) as f32;
        let rope_global_theta = extras
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1_000_000.0) as f32;

        let (quant_group_size, quant_bits, quant_mode) = if let Some(q) = &cfg.quantization {
            (
                q.group_size as i32,
                i32::from(q.bits),
                q.mode_or_default().to_owned(),
            )
        } else {
            (64, 8, "affine".to_owned())
        };

        Ok(Gemma3TextConfig {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            sliding_window,
            sliding_window_pattern,
            rms_norm_eps,
            attn_scale,
            tie_word_embeddings,
            final_logit_softcapping,
            layer_types,
            rope_local_theta,
            rope_global_theta,
            quant_group_size,
            quant_bits,
            quant_mode,
        })
    }
}
