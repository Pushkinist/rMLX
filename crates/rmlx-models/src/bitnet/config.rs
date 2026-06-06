//! BitNet configuration type.

use rmlx_core::error::{Error, Result};

/// Configuration for `BitNetForCausalLM`.
///
/// Parsed from `config.json` at load time. BitNet stores all config fields
/// at root level (same as Qwen2), not inside a nested `text_config`.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete BitNet forward-pass contract; adding a field requires updating BitNetConfig::from_model_config and all BitNet layer constructors"
)]
#[derive(Debug, Clone)]
pub struct BitNetConfig {
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of key/value heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension (`hidden_size / num_attention_heads`).
    pub head_dim: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base theta.
    pub rope_theta: f32,
    /// Whether lm_head shares weights with embed_tokens.
    pub tie_word_embeddings: bool,
    /// Maximum sequence length supported (from `max_position_embeddings`).
    pub max_position_embeddings: usize,
}

impl BitNetConfig {
    /// Parse from a [`rmlx_loader::ModelConfig`] loaded from `config.json`.
    ///
    /// BitNet's config.json is flat (same layout as Qwen2): all fields at root,
    /// no nested `text_config` sub-object.
    pub fn from_model_config(cfg: &rmlx_loader::ModelConfig) -> Result<Self> {
        // All fields live at root in extras.
        let e = &cfg.extras;

        let hidden_size = e
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("bitnet: missing hidden_size".to_owned()))?
            as usize;

        let num_hidden_layers = e
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("bitnet: missing num_hidden_layers".to_owned()))?
            as usize;

        let num_attention_heads = e
            .get("num_attention_heads")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("bitnet: missing num_attention_heads".to_owned()))?
            as usize;

        let num_key_value_heads = e
            .get("num_key_value_heads")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("bitnet: missing num_key_value_heads".to_owned()))?
            as usize;

        let vocab_size = e
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("bitnet: missing vocab_size".to_owned()))?
            as usize;

        let max_position_embeddings = e
            .get("max_position_embeddings")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("bitnet: missing max_position_embeddings".to_owned()))?
            as usize;

        let rms_norm_eps = {
            let default = 1e-5_f64;
            if let Some(v) = e.get("rms_norm_eps").and_then(serde_json::Value::as_f64) {
                v
            } else {
                tracing::warn!(field = "rms_norm_eps", default, "bitnet: using default");
                default
            }
        } as f32;

        let rope_theta = {
            let default = 500_000.0_f64;
            if let Some(v) = e.get("rope_theta").and_then(serde_json::Value::as_f64) {
                v
            } else {
                tracing::warn!(field = "rope_theta", default, "bitnet: using default");
                default
            }
        } as f32;

        let tie_word_embeddings = if let Some(v) = e
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
        {
            v
        } else {
            tracing::warn!(
                field = "tie_word_embeddings",
                default = true,
                "bitnet: using default"
            );
            true
        };

        // head_dim is not explicit in BitNet config; derive it.
        let head_dim = hidden_size / num_attention_heads;

        Ok(BitNetConfig {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            max_position_embeddings,
        })
    }
}
