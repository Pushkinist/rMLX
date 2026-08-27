//! CPU tests for Maple config parse and SWA / RoPE helpers.

#![allow(clippy::expect_used, clippy::unwrap_used)]

use super::*;

const SNAPSHOT_JSON: &str = r#"{
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

#[test]
fn snapshot_config_parses() {
    let cfg: MapleConfig = serde_json::from_str(SNAPSHOT_JSON).expect("parse maple config");
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.num_experts, 256);
    assert_eq!(cfg.rope_dims(), 64);
    assert!(cfg.is_swa_layer(0));
    assert!(!cfg.is_swa_layer(3));
    assert_eq!(cfg.moe_group_size(), 128);
}

#[test]
fn swa_pattern_cycles_when_layer_types_is_short() {
    let cfg: MapleConfig = serde_json::from_str(SNAPSHOT_JSON).expect("parse maple config");
    // 3 SWA + 1 full, repeating: layers 4 and 7 must not fall back to "SWA".
    assert!(cfg.is_swa_layer(4));
    assert!(!cfg.is_swa_layer(7));
}

#[test]
fn empty_layer_types_are_full_attention() {
    let mut cfg: MapleConfig = serde_json::from_str(SNAPSHOT_JSON).expect("parse maple config");
    cfg.layer_types.clear();
    assert!(!cfg.is_swa_layer(0));
    assert!(!cfg.is_swa_layer(3));
}

#[test]
fn rope_scaling_null_or_absent_is_plain_rope() {
    let cfg: MapleConfig = serde_json::from_str(SNAPSHOT_JSON).expect("parse maple config");
    assert!(!cfg.has_rope_scaling());
    let with_null = SNAPSHOT_JSON.trim_end_matches('}').to_owned() + r#", "rope_scaling": null}"#;
    let cfg: MapleConfig = serde_json::from_str(&with_null).expect("parse");
    assert!(!cfg.has_rope_scaling());
}

#[test]
fn rope_scaling_object_is_flagged() {
    let with = SNAPSHOT_JSON.trim_end_matches('}').to_owned()
        + r#", "rope_scaling": {"rope_type": "yarn"}}"#;
    let cfg: MapleConfig = serde_json::from_str(&with).expect("parse");
    assert!(cfg.has_rope_scaling());
}
