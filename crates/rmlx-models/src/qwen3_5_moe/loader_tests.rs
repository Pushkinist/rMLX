//! Host-only tests for the Qwen3.5 loader's per-layer MLP detection.
//!
//! `build_mlp` selects dense SwiGLU vs sparse MoE purely by which tensors a
//! layer carries (`mlp.switch_mlp.gate_proj.weight` ⇒ MoE, otherwise dense).
//! It only calls `Weights::has` plus a caller-supplied `lin` closure, so a fake
//! `lin` returning a tiny host-side `Linear::Plain` exercises the real probe
//! with no Metal claim. These cases drive `build_mlp` directly and assert the
//! returned `MlpBlock` variant — proving the loader's branch, not a hand-copied
//! witness string.

use std::path::Path;

use rmlx_loader::ShardSet;
use rmlx_mlx::Array;

use super::super::config::Qwen3_5MoeConfig;
use super::super::decoder_layer::MlpBlock;
use super::super::layers::Linear;
use super::build_mlp;
use crate::load_util::Weights;

/// Write a multi-tensor F32 `.safetensors` shard into `dir`. F32 is host-side
/// in `Array::from_bytes` (no Metal claim), so the loaded weights stay on host.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: temp-file I/O and serialization failures should abort the test loudly"
)]
fn write_shard(dir: &Path, filename: &str, tensors: &[(&str, usize)]) {
    let buffers: Vec<Vec<u8>> = tensors
        .iter()
        .map(|&(_, n)| {
            let mut bytes = Vec::with_capacity(n * 4);
            for _ in 0..n {
                bytes.extend_from_slice(&1.0f32.to_le_bytes());
            }
            bytes
        })
        .collect();

    let views: Vec<(String, safetensors::tensor::TensorView<'_>)> = tensors
        .iter()
        .zip(buffers.iter())
        .map(|(&(name, n), buf)| {
            let tv = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![n], buf)
                .unwrap();
            (name.to_owned(), tv)
        })
        .collect();

    let bytes = safetensors::serialize(views, None).unwrap();
    std::fs::write(dir.join(filename), bytes).unwrap();
}

/// Minimal MoE config — only `num_experts` / `num_experts_per_tok` /
/// `norm_topk_prob` are read by `build_mlp`'s MoE branch; the rest are inert.
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: malformed inline JSON should abort the test loudly"
)]
fn moe_cfg() -> Qwen3_5MoeConfig {
    let json = r#"{
        "architectures": ["Qwen3_5MoeForConditionalGeneration"],
        "dtype": "bfloat16",
        "text_config": {
            "num_hidden_layers": 4,
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
    let raw: rmlx_loader::ModelConfig = serde_json::from_str(json).unwrap();
    let raw_json: serde_json::Value = serde_json::from_str(json).unwrap();
    Qwen3_5MoeConfig::from_model_config(&raw, None, raw_json.get("text_config")).unwrap()
}

// `build_mlp` takes `lin: &impl Fn(&str) -> Result<Linear>`. The tests pass a
// local closure returning a tiny host-side `Linear::Plain` — `build_mlp` only
// stores whatever `lin` returns, so the dtype/shape is irrelevant to which
// `MlpBlock` variant it selects. A closure (not a free `fn`) keeps the `Result`
// return the seam requires without tripping clippy's `unnecessary_wraps` lint.

/// A layer carrying `mlp.switch_mlp.gate_proj.weight` (+ router + shared expert)
/// builds an `MlpBlock::Moe`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open / Array construction failures should abort the test loudly"
)]
fn build_mlp_detects_moe_layout() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "s.safetensors",
        &[
            ("mlp.gate.weight", 4),
            ("mlp.switch_mlp.gate_proj.weight", 8),
            ("mlp.switch_mlp.up_proj.weight", 8),
            ("mlp.switch_mlp.down_proj.weight", 8),
            ("mlp.shared_expert.gate_proj.weight", 8),
            ("mlp.shared_expert.up_proj.weight", 8),
            ("mlp.shared_expert.down_proj.weight", 8),
            ("mlp.shared_expert_gate.weight", 4),
        ],
    );
    let shards = ShardSet::open_dir(dir.path()).unwrap();
    let w = Weights::scan_only(&shards);

    let fake_lin = |_base: &str| {
        Ok(Linear::Plain {
            weight: Array::from_f32_slice(&[1.0], &[1]).unwrap(),
        })
    };
    let block = build_mlp(&w, "mlp", &moe_cfg(), &fake_lin).unwrap();
    assert!(
        matches!(block, MlpBlock::Moe(_)),
        "switch_mlp present ⇒ build_mlp must return MlpBlock::Moe"
    );
}

/// A layer carrying plain `mlp.{gate,up,down}_proj.weight` with no `switch_mlp`
/// router builds an `MlpBlock::Dense`.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test fixture: tempdir / ShardSet open / Array construction failures should abort the test loudly"
)]
fn build_mlp_detects_dense_layout() {
    let dir = tempfile::tempdir().unwrap();
    write_shard(
        dir.path(),
        "s.safetensors",
        &[
            ("mlp.gate_proj.weight", 8),
            ("mlp.up_proj.weight", 8),
            ("mlp.down_proj.weight", 8),
        ],
    );
    let shards = ShardSet::open_dir(dir.path()).unwrap();
    let w = Weights::scan_only(&shards);

    let fake_lin = |_base: &str| {
        Ok(Linear::Plain {
            weight: Array::from_f32_slice(&[1.0], &[1]).unwrap(),
        })
    };
    let block = build_mlp(&w, "mlp", &moe_cfg(), &fake_lin).unwrap();
    assert!(
        matches!(block, MlpBlock::Dense(_)),
        "no switch_mlp router ⇒ build_mlp must return MlpBlock::Dense"
    );
}
