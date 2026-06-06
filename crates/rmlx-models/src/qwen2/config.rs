// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

use rmlx_core::error::{Error, Result};

/// Subset of root-level config.json fields for the Qwen2 forward pass.
///
/// Qwen2 dense models store all fields at the root of config.json,
/// not inside a `text_config` sub-object.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Qwen2 model contract; adding a field requires updating from_model_config and all Qwen2 layer constructors"
)]
#[derive(Debug, Clone)]
/// Parsed Qwen2ForCausalLM config.
pub struct Qwen2Config {
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_key_value_heads: usize,
    /// head_dim = hidden_size / num_attention_heads (implicit in Qwen2).
    pub head_dim: usize,
    /// FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Whether lm_head shares weights with the embedding table.
    pub tie_word_embeddings: bool,
    /// Quantization group size.
    pub quant_group_size: i32,
    /// Quantization bit-width.
    pub quant_bits: i32,
    /// Quantization mode string.
    pub quant_mode: String,
}

impl Qwen2Config {
    /// Parse from a [`rmlx_loader::ModelConfig`] loaded from `config.json`.
    pub fn from_model_config(cfg: &rmlx_loader::ModelConfig) -> Result<Self> {
        // Qwen2 keeps everything at root — extras holds the non-standard keys.
        let e = &cfg.extras;

        let hidden_size = e
            .get("hidden_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen2: missing hidden_size".into()))?
            as usize;
        let num_hidden_layers = e
            .get("num_hidden_layers")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen2: missing num_hidden_layers".into()))?
            as usize;
        let intermediate_size = e
            .get("intermediate_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen2: missing intermediate_size".into()))?
            as usize;
        let num_attention_heads = e
            .get("num_attention_heads")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen2: missing num_attention_heads".into()))?
            as usize;
        let num_key_value_heads = e
            .get("num_key_value_heads")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen2: missing num_key_value_heads".into()))?
            as usize;
        let vocab_size = e
            .get("vocab_size")
            .and_then(serde_json::Value::as_u64)
            .ok_or_else(|| Error::Config("qwen2: missing vocab_size".into()))?
            as usize;
        let rms_norm_eps = e
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;
        let rope_theta = e
            .get("rope_theta")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1_000_000.0) as f32;
        let tie_word_embeddings = e
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(true);

        let head_dim = hidden_size / num_attention_heads;

        let (quant_group_size, quant_bits, quant_mode) = if let Some(q) = &cfg.quantization {
            (
                q.group_size as i32,
                i32::from(q.bits),
                q.mode_or_default().to_owned(),
            )
        } else {
            (64, 8, "affine".to_owned())
        };

        Ok(Qwen2Config {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            rope_theta,
            tie_word_embeddings,
            quant_group_size,
            quant_bits,
            quant_mode,
        })
    }
}
