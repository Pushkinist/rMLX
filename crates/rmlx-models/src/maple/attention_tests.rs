//! CPU tests for Maple attention construction and MapleRMSNorm dtype.

use super::super::config::MapleConfig;
use super::{MapleAttention, MapleRmsNorm};
use crate::layers::Linear;
use rmlx_mlx::{Array, Device, Dtype};

#[allow(
    clippy::unwrap_used,
    reason = "test fixture: malformed inline JSON should abort the test loudly"
)]
fn sample_cfg() -> MapleConfig {
    serde_json::from_str(
        r#"{
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
        }"#,
    )
    .unwrap()
}

#[allow(clippy::unwrap_used, reason = "tiny host arrays for constructor tests")]
fn dummy_lin() -> Linear {
    Linear::Plain {
        weight: Array::from_f32_slice(&[0.0], &[1, 1]).unwrap(),
    }
}

#[allow(clippy::unwrap_used, reason = "tiny host arrays for constructor tests")]
fn dummy_norm(eps: f32) -> MapleRmsNorm {
    MapleRmsNorm::new(Array::from_f32_slice(&[1.0], &[1]).unwrap(), eps)
}

fn dummy_attn(cfg: &MapleConfig, layer: usize) -> MapleAttention {
    MapleAttention::new(
        cfg,
        layer,
        dummy_lin(),
        dummy_lin(),
        dummy_lin(),
        dummy_lin(),
        dummy_norm(cfg.rms_norm_eps),
        dummy_norm(cfg.rms_norm_eps),
    )
}

#[test]
fn swa_layer_sets_use_rope_full_layer_is_nope() {
    let cfg = sample_cfg();
    let swa = dummy_attn(&cfg, 0);
    assert!(swa.use_rope);
    assert_eq!(swa.rope_dims, 64);
    assert_eq!(swa.n_q, 16);
    assert_eq!(swa.n_kv, 4);
    assert_eq!(swa.head_dim, 128);
    let full = dummy_attn(&cfg, 3);
    assert!(!full.use_rope);
}

#[test]
fn scale_is_inv_sqrt_head_dim() {
    let cfg = sample_cfg();
    let attn = dummy_attn(&cfg, 0);
    let want = (128.0_f32).sqrt().recip();
    assert!((attn.scale - want).abs() < 1e-6);
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "CPU MapleRMSNorm dtype probe; unwrap aborts the test on MLX failure"
)]
fn maple_rms_norm_returns_input_dtype() {
    let x = Array::from_f32_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]).unwrap();
    let x = x.astype(Dtype::Bf16, Device::Cpu).unwrap();
    let w = Array::from_f32_slice(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
    let w = w.astype(Dtype::Bf16, Device::Cpu).unwrap();
    let n = MapleRmsNorm::new(w, 1e-6);
    let out = n.forward(&x, Device::Cpu).unwrap();
    out.eval().unwrap();
    assert_eq!(out.dtype(), Dtype::Bf16);
    assert_eq!(out.shape(), vec![1, 1, 1, 4]);
}
