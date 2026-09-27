//! BitNet loads and runs on the device `arch::load_model` was given: with the
//! GPU forbidden, a CPU load and forward of a one-layer synthetic snapshot
//! never asks for a GPU stream.
//!
//! The GPU latch is process-global and one-way, so the check runs in a child
//! of this test binary.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use crate::arch::{load_model, Architecture, LoadOpts};
use rmlx_mlx::Device;
use std::path::Path;
use std::process::{Command, Stdio};

const CHILD_MARKER: &str = "started-by-a-bitnet-device-parent-test";
const CHILD_DONE: &str = "bitnet-device-child done";
const HIDDEN: usize = 8;
const KV_OUT: usize = 4;

/// bf16 1.0, little-endian.
const BF16_ONE: [u8; 2] = [0x80, 0x3F];

struct Tensor {
    name: String,
    dtype: &'static str,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn bf16(name: &str, shape: &[usize]) -> Tensor {
    let n: usize = shape.iter().product();
    Tensor {
        name: name.to_owned(),
        dtype: "BF16",
        shape: shape.to_vec(),
        bytes: BF16_ONE.repeat(n),
    }
}

/// A ternary linear layer: `out` rows packed four to a byte, all trits 0,
/// and its bf16 scale.
fn bitlinear(base: &str, out: usize, cols: usize) -> [Tensor; 2] {
    [
        Tensor {
            name: format!("{base}.weight"),
            dtype: "U8",
            shape: vec![out / 4, cols],
            bytes: vec![0x55; out / 4 * cols],
        },
        bf16(&format!("{base}.weight_scale"), &[1]),
    ]
}

fn write_snapshot(dir: &Path) {
    let config = serde_json::json!({
        "architectures": ["BitNetForCausalLM"],
        "model_type": "bitnet",
        "hidden_size": HIDDEN,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "intermediate_size": HIDDEN,
        "vocab_size": HIDDEN,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "tie_word_embeddings": true,
        "max_position_embeddings": 64
    });
    std::fs::write(dir.join("config.json"), config.to_string()).unwrap();

    let l = "model.layers.0";
    let mut tensors = vec![
        bf16("model.embed_tokens.weight", &[HIDDEN, HIDDEN]),
        bf16("model.norm.weight", &[HIDDEN]),
        bf16(&format!("{l}.input_layernorm.weight"), &[HIDDEN]),
        bf16(&format!("{l}.post_attention_layernorm.weight"), &[HIDDEN]),
        bf16(&format!("{l}.self_attn.attn_sub_norm.weight"), &[HIDDEN]),
        bf16(&format!("{l}.mlp.ffn_sub_norm.weight"), &[HIDDEN]),
    ];
    for (proj, out) in [
        ("self_attn.q_proj", HIDDEN),
        ("self_attn.k_proj", KV_OUT),
        ("self_attn.v_proj", KV_OUT),
        ("self_attn.o_proj", HIDDEN),
        ("mlp.gate_proj", HIDDEN),
        ("mlp.up_proj", HIDDEN),
        ("mlp.down_proj", HIDDEN),
    ] {
        tensors.extend(bitlinear(&format!("{l}.{proj}"), out, HIDDEN));
    }

    let mut header = serde_json::Map::new();
    let mut offset = 0;
    for t in &tensors {
        header.insert(
            t.name.clone(),
            serde_json::json!({"dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes.len()]}),
        );
        offset += t.bytes.len();
    }
    let mut header = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    while !header.len().is_multiple_of(8) {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(&header);
    for t in &tensors {
        bytes.extend_from_slice(&t.bytes);
    }
    std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
}

#[test]
fn bitnet_load_and_forward_on_cpu_is_admitted() {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            "bitnet::device_tests::bitnet_load_and_forward_on_cpu_is_admitted_child",
            CHILD_MARKER,
            "--exact",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .stdin(Stdio::null())
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}\n{stderr}");
    assert!(
        stdout.contains(CHILD_DONE),
        "child did not run to its end:\n{stdout}"
    );
}

#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn bitnet_load_and_forward_on_cpu_is_admitted_child() {
    if !std::env::args().any(|arg| arg == CHILD_MARKER) {
        return;
    }
    rmlx_mlx::forbid_gpu();
    let dir = tempfile::tempdir().unwrap();
    write_snapshot(dir.path());
    let arch = load_model(dir.path(), Device::Cpu, &LoadOpts::default()).expect("CPU load");
    let Architecture::BitNet(model) = arch else {
        panic!("the snapshot is a BitNet");
    };
    let logits = model
        .forward_seq(&[1, 2], Device::Cpu)
        .expect("CPU forward");
    logits.eval().expect("CPU eval");
    println!("{CHILD_DONE}");
}
