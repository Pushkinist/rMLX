//! Config-parse tests for `Qwen3_5MoeConfig::from_model_config`.
//!
//! The same parser serves both Qwen3.5 arch strings. A dense
//! `Qwen3_5ForConditionalGeneration` checkpoint omits the MoE-only fields
//! (`num_experts`, `moe_intermediate_size`, ...) entirely; the parser must
//! treat those as optional and report `num_experts == 0` — the marker the
//! loader keys its per-layer dense-vs-MoE MLP selection on. A MoE checkpoint
//! must still parse its expert counts verbatim.

use super::Qwen3_5MoeConfig;

/// Parse helper: split the JSON into a typed `ModelConfig` plus the raw
/// `text_config` value, exactly as the loader feeds `from_model_config`.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: malformed inline JSON should abort the test loudly"
)]
fn parse(json: &str) -> Qwen3_5MoeConfig {
    let raw: rmlx_loader::ModelConfig = serde_json::from_str(json).unwrap();
    let raw_json: serde_json::Value = serde_json::from_str(json).unwrap();
    let raw_quant = raw_json.get("quantization");
    let raw_text_config = raw_json.get("text_config");
    Qwen3_5MoeConfig::from_model_config(&raw, raw_quant, raw_text_config).unwrap()
}

/// Dense Qwen3.5 mxfp8 (ornith-style): no MoE fields. `num_experts == 0`,
/// `intermediate_size` carried, `moe_intermediate_size` falls back to it.
#[test]
fn dense_config_parses_with_zero_experts() {
    let json = r#"{
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "dtype": "bfloat16",
        "quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
        "text_config": {
            "num_hidden_layers": 8,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "intermediate_size": 1024,
            "vocab_size": 2048,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "linear_num_value_heads": 8,
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
        }
    }"#;
    let cfg = parse(json);
    assert_eq!(cfg.num_hidden_layers, 8);
    assert_eq!(
        cfg.num_experts, 0,
        "dense checkpoint reports zero experts — the loader's dense marker"
    );
    assert_eq!(cfg.intermediate_size, 1024);
    assert_eq!(
        cfg.moe_intermediate_size, 1024,
        "moe_intermediate_size falls back to dense intermediate_size when absent"
    );
}

/// MoE Qwen3.5 (A3B-style): expert counts present and parsed verbatim.
#[test]
fn moe_config_parses_expert_counts() {
    let json = r#"{
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "dtype": "bfloat16",
        "quantization": {"group_size": 32, "bits": 8, "mode": "mxfp8"},
        "text_config": {
            "num_hidden_layers": 8,
            "hidden_size": 256,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 64,
            "vocab_size": 2048,
            "rms_norm_eps": 1e-6,
            "full_attention_interval": 4,
            "num_experts": 256,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 512,
            "shared_expert_intermediate_size": 512,
            "linear_num_value_heads": 8,
            "linear_num_key_heads": 4,
            "linear_key_head_dim": 32,
            "linear_value_head_dim": 32,
            "rope_parameters": {"rope_theta": 10000000.0, "partial_rotary_factor": 0.25}
        }
    }"#;
    let cfg = parse(json);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.num_experts_per_tok, 8);
    assert_eq!(cfg.moe_intermediate_size, 512);
    assert_eq!(cfg.shared_expert_intermediate_size, 512);
}
