//! Laguna unit tests.

#![allow(clippy::float_cmp)]
use super::config::{LagunaConfig, LayerKind, MlpKind};

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn laguna_config_parses_basic_fields() {
    let raw_json = r#"{
        "architectures": ["LagunaForCausalLM"],
        "hidden_size": 2048,
        "num_hidden_layers": 4,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "intermediate_size": 4096,
        "vocab_size": 100352,
        "rms_norm_eps": 1e-6,
        "sliding_window": 512,
        "tie_word_embeddings": false,
        "num_experts": 16,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 256,
        "shared_expert_intermediate_size": 256,
        "moe_routed_scaling_factor": 2.5,
        "layer_types": ["full_attention", "sliding_attention", "sliding_attention", "full_attention"],
        "mlp_layer_types": ["dense", "sparse", "sparse", "sparse"],
        "num_attention_heads_per_layer": [16, 24, 24, 16],
        "rope_parameters": {
            "full_attention": {"rope_theta": 500000.0, "rope_type": "yarn", "partial_rotary_factor": 0.5},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default", "partial_rotary_factor": 1.0}
        },
        "quantization": {
            "group_size": 32,
            "bits": 8,
            "mode": "mxfp8",
            "model.layers.1.mlp.gate.proj": {"group_size": 64, "bits": 8}
        }
    }"#;

    let model_cfg: rmlx_loader::ModelConfig =
        serde_json::from_str(raw_json).expect("parse ModelConfig");
    let raw_val: serde_json::Value = serde_json::from_str(raw_json).expect("parse raw json");
    let raw_quant = raw_val.get("quantization");
    let cfg = LagunaConfig::from_model_config(&model_cfg, raw_quant).expect("from_model_config");

    assert_eq!(cfg.num_hidden_layers, 4);
    assert_eq!(cfg.hidden_size, 2048);
    assert_eq!(cfg.head_dim, 128);
    assert_eq!(cfg.num_experts, 16);
    assert_eq!(cfg.rope_theta_full, 500_000.0);
    assert_eq!(cfg.rope_theta_sliding, 10_000.0);
    assert_eq!(cfg.rope_dims_full, 64); // 0.5 * 128
    assert_eq!(cfg.quant_mode, "mxfp8");
    assert_eq!(cfg.quant_group_size, 32);
    assert_eq!(cfg.quant_overrides.len(), 1, "one inline override");
    let ov = cfg
        .quant_overrides
        .get("model.layers.1.mlp.gate.proj")
        .expect("override key present");
    assert_eq!(ov.group_size, 64);
    assert_eq!(ov.bits, 8);
    assert_eq!(cfg.layer_mlp_kinds[0], MlpKind::Dense);
    assert_eq!(cfg.layer_mlp_kinds[1], MlpKind::Sparse);
    assert_eq!(cfg.layer_attn_kinds[0], LayerKind::FullAttention);
    assert_eq!(cfg.layer_attn_kinds[1], LayerKind::SlidingAttention);
    assert_eq!(cfg.layer_num_heads[1], 24);
}

/// Integration test: load Laguna-XS.2-mxfp8 and run a forward pass.
///
/// Run explicitly:
/// cargo test -p rmlx-models integration_laguna_xs2 -- --ignored
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn integration_laguna_xs2() {
    use rmlx_mlx::Device;

    let Some(model_dir_buf) =
        std::env::var_os("RMLX_TEST_MODEL_LAGUNA").map(std::path::PathBuf::from)
    else {
        eprintln!("integration_laguna_xs2: skipping: RMLX_TEST_MODEL_LAGUNA not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        eprintln!("integration_laguna_xs2: snapshot absent, skipping");
        return;
    }

    let model = super::loader::load_from_path(model_dir).expect("load_from_path failed");

    let logits = model
        .forward_seq(&[1], Device::Gpu)
        .expect("forward_seq failed");
    logits.eval().expect("logits eval");

    let vocab = model.cfg.vocab_size as i32;
    let logits_flat = logits.reshape(&[1, vocab], Device::Gpu).expect("reshape");
    logits_flat.eval().expect("logits_flat eval");

    let max_abs = rmlx_mlx::max_axis(&logits_flat, -1, Device::Gpu).expect("max_axis");
    max_abs.eval().expect("max eval");
    let max_bytes = max_abs.to_bytes().expect("to_bytes");
    assert!(!max_bytes.is_empty(), "non-empty logits");
    assert_eq!(logits_flat.shape(), vec![1, vocab]);
    eprintln!("forward_probe: logits shape=[1, {vocab}] non-empty");
}
