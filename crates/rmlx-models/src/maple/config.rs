//! Maple checkpoint `config.json`.

use serde::Deserialize;

/// `MapleForCausalLM` config as stored in the DeepGrove MLX snapshot.
#[derive(Debug, Clone, Deserialize)]
pub struct MapleConfig {
    /// Hidden width (2048).
    pub hidden_size: i32,
    /// Dense MLP width (unused when every layer is MoE).
    #[serde(default = "default_intermediate")]
    pub intermediate_size: i32,
    /// Expert FFN width (512).
    pub moe_intermediate_size: i32,
    /// Decoder depth (24).
    pub num_hidden_layers: i32,
    /// Query heads (16).
    pub num_attention_heads: i32,
    /// Key/value heads (4).
    pub num_key_value_heads: i32,
    /// Head dim (128).
    pub head_dim: i32,
    /// Routed experts (256).
    pub num_experts: i32,
    /// Top-k (8).
    pub num_experts_per_tok: i32,
    /// First N layers that are dense MLP. Maple-Preview is 0 (all MoE).
    #[serde(default)]
    pub first_k_dense_replace: i32,
    /// RMSNorm eps (1e-6).
    #[serde(default = "default_eps")]
    pub rms_norm_eps: f32,
    /// RoPE base (10000).
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,
    /// Fraction of head_dim that is rotated (0.5 → 64).
    #[serde(default = "default_partial_rotary")]
    pub partial_rotary_factor: f32,
    /// Sliding-window size for SWA layers (512).
    #[serde(default = "default_swa")]
    pub sliding_window: i32,
    /// Per-layer `"sliding_attention"` / `"full_attention"`.
    #[serde(default)]
    pub layer_types: Vec<String>,
    /// Vocab (151936).
    pub vocab_size: i32,
    /// Max context.
    #[serde(default = "default_max_pos")]
    pub max_position_embeddings: i32,
    /// Per-head Q/K RMSNorm.
    #[serde(default = "default_true")]
    pub use_qk_norm: bool,
    /// Projection bias (false).
    #[serde(default)]
    pub use_bias: bool,
    /// Tied embeddings (false).
    #[serde(default)]
    pub tie_word_embeddings: bool,
    /// Snapshot quantization block (`bits`, `group_size`, per-tensor overrides).
    #[serde(default)]
    pub quantization: Option<MapleQuantization>,
}

/// `quantization` / `quantization_config` object.
#[derive(Debug, Clone, Deserialize)]
pub struct MapleQuantization {
    /// Default packed bits (2 for Maple linears).
    #[serde(default = "default_bits")]
    pub bits: u8,
    /// Default group size (128).
    #[serde(default = "default_group")]
    pub group_size: i32,
}

fn default_intermediate() -> i32 {
    5120
}
fn default_eps() -> f32 {
    1e-6
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_partial_rotary() -> f32 {
    0.5
}
fn default_swa() -> i32 {
    512
}
fn default_max_pos() -> i32 {
    128_000
}
fn default_true() -> bool {
    true
}
fn default_bits() -> u8 {
    2
}
fn default_group() -> i32 {
    128
}

impl MapleConfig {
    /// Sliding-window layer? Full-attention layers are NoPE.
    #[must_use]
    pub fn is_swa_layer(&self, layer: usize) -> bool {
        self.layer_types
            .get(layer)
            .map(|t| t == "sliding_attention")
            .unwrap_or(true)
    }

    /// Rotary dims (`head_dim * partial_rotary_factor`).
    #[must_use]
    pub fn rope_dims(&self) -> i32 {
        (self.head_dim as f32 * self.partial_rotary_factor) as i32
    }

    /// MoE / most linear group size.
    #[must_use]
    pub fn moe_group_size(&self) -> i32 {
        self.quantization
            .as_ref()
            .map_or(128, |q| q.group_size)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_config_parses() {
        let raw = r#"{
            "hidden_size": 2048,
            "moe_intermediate_size": 512,
            "num_hidden_layers": 24,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "head_dim": 128,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "first_k_dense_replace": 0,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000,
            "partial_rotary_factor": 0.5,
            "sliding_window": 512,
            "layer_types": [
                "sliding_attention","sliding_attention","sliding_attention","full_attention"
            ],
            "vocab_size": 151936,
            "quantization": {"bits": 2, "group_size": 128}
        }"#;
        let cfg: MapleConfig = serde_json::from_str(raw).expect("parse maple config");
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_experts, 256);
        assert_eq!(cfg.rope_dims(), 64);
        assert!(cfg.is_swa_layer(0));
        assert!(!cfg.is_swa_layer(3));
        assert_eq!(cfg.moe_group_size(), 128);
    }
}
