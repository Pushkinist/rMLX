//! Laguna configuration types and parsing helpers.

use std::collections::HashMap;

use rmlx_core::error::{Error, Result};
use tracing::debug;

use crate::layers::QuantParams;

// ---------------------------------------------------------------------------
// Config types
// ---------------------------------------------------------------------------

#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two Laguna attention-layer kinds (FullAttention/SlidingAttention); adding a kind requires updating all layer_attn_kinds match arms and from_model_config"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// Attention-layer kind for one Laguna decoder layer.
pub enum LayerKind {
    /// Global full-context attention.
    FullAttention,
    /// Local sliding-window attention.
    SlidingAttention,
}

#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two Laguna MLP kinds (Dense/Sparse); adding a kind requires updating all layer_mlp_kinds match arms and from_model_config"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
/// FFN kind for one Laguna decoder layer.
pub enum MlpKind {
    /// Dense (shared) FFN.
    Dense,
    /// Sparse mixture-of-experts FFN.
    Sparse,
}

#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed config struct — all fields are the complete Laguna model contract; adding a field requires updating from_model_config and all Laguna layer constructors"
)]
#[derive(Debug, Clone)]
/// Parsed LagunaForCausalLM model configuration.
pub struct LagunaConfig {
    /// Number of decoder layers.
    pub num_hidden_layers: usize,
    /// Model hidden dimension.
    pub hidden_size: usize,
    /// Default number of query attention heads.
    pub num_attention_heads: usize,
    /// Number of KV heads (GQA).
    pub num_key_value_heads: usize,
    /// Per-head dimension.
    pub head_dim: usize,
    /// Shared-expert FFN intermediate dimension.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Sliding-window attention context length.
    pub sliding_window: usize,
    /// Whether lm_head shares weights with the embedding table.
    pub tie_word_embeddings: bool,
    /// Per-layer attention kind (full or sliding).
    pub layer_attn_kinds: Vec<LayerKind>,
    /// Per-layer MLP kind (dense or sparse MoE).
    pub layer_mlp_kinds: Vec<MlpKind>,
    /// Per-layer query head count (variable across Laguna layers).
    pub layer_num_heads: Vec<usize>,
    /// RoPE theta for full-attention layers.
    pub rope_theta_full: f32,
    /// RoPE theta for sliding-attention layers.
    pub rope_theta_sliding: f32,
    /// Number of dimensions rotated for full-attention layers (partial_rotary_factor * head_dim).
    pub rope_dims_full: usize,
    /// Total number of MoE experts.
    pub num_experts: usize,
    /// Number of experts selected per token.
    pub num_experts_per_tok: usize,
    /// MoE expert FFN intermediate dimension.
    pub moe_intermediate_size: usize,
    /// Shared (always-active) expert intermediate dimension.
    pub shared_expert_intermediate_size: usize,
    /// Routing score multiplier for load-balancing.
    pub moe_routed_scaling_factor: f32,
    /// Quantization group size.
    pub quant_group_size: i32,
    /// Quantization bit-width.
    pub quant_bits: i32,
    /// Quantization mode string.
    pub quant_mode: String,
    /// Per-tensor quant overrides extracted from the inline quantization dict.
    pub quant_overrides: HashMap<String, QuantParams>,
}

impl LagunaConfig {
    /// Parse config from a `ModelConfig` plus the raw JSON value of the
    /// `quantization` key (needed to extract inline per-tensor overrides that
    /// Serde drops when deserializing `QuantConfig`).
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn from_model_config(
        cfg: &rmlx_loader::ModelConfig,
        raw_quant: Option<&serde_json::Value>,
    ) -> Result<Self> {
        let e = &cfg.extras;

        macro_rules! req_u64 {
            ($key:expr) => {
                e.get($key)
                    .and_then(|v| v.as_u64())
                    .ok_or_else(|| Error::Config(format!("laguna: missing {}", $key)))? as usize
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
        let intermediate_size = req_u64!("intermediate_size");
        let vocab_size = req_u64!("vocab_size");
        let rms_norm_eps = e
            .get("rms_norm_eps")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1e-6) as f32;
        let sliding_window = e
            .get("sliding_window")
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(512) as usize;
        let tie_word_embeddings = e
            .get("tie_word_embeddings")
            .and_then(serde_json::Value::as_bool)
            .unwrap_or(false);
        let num_experts = req_u64!("num_experts");
        let num_experts_per_tok = req_u64!("num_experts_per_tok");
        let moe_intermediate_size = req_u64!("moe_intermediate_size");
        let shared_expert_intermediate_size = req_u64!("shared_expert_intermediate_size");
        let moe_routed_scaling_factor = e
            .get("moe_routed_scaling_factor")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0) as f32;

        let layer_attn_kinds: Vec<LayerKind> =
            if let Some(arr) = e.get("layer_types").and_then(|v| v.as_array()) {
                arr.iter()
                    .map(|v| match v.as_str().unwrap_or("full_attention") {
                        "sliding_attention" => LayerKind::SlidingAttention,
                        _ => LayerKind::FullAttention,
                    })
                    .collect()
            } else {
                vec![LayerKind::FullAttention; num_hidden_layers]
            };

        let layer_mlp_kinds: Vec<MlpKind> =
            if let Some(arr) = e.get("mlp_layer_types").and_then(|v| v.as_array()) {
                arr.iter()
                    .map(|v| match v.as_str().unwrap_or("sparse") {
                        "dense" => MlpKind::Dense,
                        _ => MlpKind::Sparse,
                    })
                    .collect()
            } else {
                let mut kinds = vec![MlpKind::Sparse; num_hidden_layers];
                if !kinds.is_empty() {
                    kinds[0] = MlpKind::Dense;
                }
                kinds
            };

        let layer_num_heads: Vec<usize> = if let Some(arr) = e
            .get("num_attention_heads_per_layer")
            .and_then(|v| v.as_array())
        {
            arr.iter()
                .map(|v| v.as_u64().unwrap_or(num_attention_heads as u64) as usize)
                .collect()
        } else {
            vec![num_attention_heads; num_hidden_layers]
        };

        let (rope_theta_full, rope_theta_sliding, rope_dims_full) = parse_rope_params(e, head_dim);

        let (quant_group_size, quant_bits, quant_mode, quant_overrides) =
            extract_quant(cfg, raw_quant);

        Ok(LagunaConfig {
            num_hidden_layers,
            hidden_size,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            intermediate_size,
            vocab_size,
            rms_norm_eps,
            sliding_window,
            tie_word_embeddings,
            layer_attn_kinds,
            layer_mlp_kinds,
            layer_num_heads,
            rope_theta_full,
            rope_theta_sliding,
            rope_dims_full,
            num_experts,
            num_experts_per_tok,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            moe_routed_scaling_factor,
            quant_group_size,
            quant_bits,
            quant_mode,
            quant_overrides,
        })
    }
}

pub(super) fn parse_rope_params(
    e: &serde_json::Map<String, serde_json::Value>,
    head_dim: usize,
) -> (f32, f32, usize) {
    let rp = e.get("rope_parameters").and_then(|v| v.as_object());

    let rope_theta_full = rp
        .and_then(|m| m.get("full_attention"))
        .and_then(|v| v.as_object())
        .and_then(|m| m.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(500_000.0) as f32;

    let rope_theta_sliding = rp
        .and_then(|m| m.get("sliding_attention"))
        .and_then(|v| v.as_object())
        .and_then(|m| m.get("rope_theta"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(10_000.0) as f32;

    let partial_full = rp
        .and_then(|m| m.get("full_attention"))
        .and_then(|v| v.as_object())
        .and_then(|m| m.get("partial_rotary_factor"))
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.5);

    let rope_dims_full = ((head_dim as f64) * partial_full).round() as usize;

    (rope_theta_full, rope_theta_sliding, rope_dims_full)
}

/// Extract global quant params plus inline per-tensor overrides.
///
/// Laguna stores per-tensor overrides directly in the `quantization` dict as
/// extra keys (e.g. `"model.layers.N.mlp.gate.proj": {group_size, bits}`).
/// Serde silently drops them when parsing into `QuantConfig` (no catch-all field),
/// so we re-read the raw config JSON to capture them.
///
/// `raw_quant_val`: the raw JSON value of the `quantization` key, re-read from
/// `config.json` before Serde struct-parsing.
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
                    .unwrap_or("") // empty => inherit global
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

    if !overrides.is_empty() {
        debug!(
            count = overrides.len(),
            "laguna: loaded quant tensor overrides"
        );
    }

    (gs, bits, mode, overrides)
}
