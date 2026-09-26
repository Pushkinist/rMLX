//! Qwen3-TTS runs on the device its model was given: with the GPU forbidden,
//! a CPU model loads and synthesizes without asking for a GPU stream.
//!
//! The GPU latch is process-global and one-way, so each check runs in a child
//! of this test binary.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use super::{synthesize, TtsConfig, TtsError, TtsModel, TtsTokenizer};
use rmlx_mlx::Device;

use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

const CHILD_MARKER: &str = "started-by-a-tts-device-parent-test";
const CHILD_DONE: &str = "tts-device-child done";
/// Every key the loaders ask for is a one-element f32 tensor of this shape;
/// the load-time transposes and codebook arithmetic accept it.
const SHAPE: [usize; 3] = [1, 1, 1];

fn run_child(name: &str) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args([
            &format!("tts::tts_device_tests::{name}"),
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
    assert!(out.status.success(), "{name} failed:\n{stdout}\n{stderr}");
    assert!(
        stdout.contains(CHILD_DONE),
        "{name} did not run to its end:\n{stdout}"
    );
}

fn started_by_parent() -> bool {
    std::env::args().any(|arg| arg == CHILD_MARKER)
}

fn tiny_config() -> TtsConfig {
    TtsConfig::from_json(
        r#"{
        "model_type": "qwen3_tts",
        "tts_bos_token_id": 1,
        "tts_eos_token_id": 2,
        "tts_pad_token_id": 3,
        "talker_config": {
            "hidden_size": 1,
            "num_attention_heads": 1,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "num_code_groups": 16,
            "codec_bos_id": 4,
            "codec_eos_token_id": 5,
            "code_predictor_num_hidden_layers": 1,
            "spk_id": {"serena": 6}
        }
    }"#,
    )
    .unwrap()
}

/// A synthetic snapshot directory: one single-tensor shard per key, so the
/// loader's per-lookup header parse stays one entry long.
struct Snapshot {
    dir: PathBuf,
    weight_map: serde_json::Map<String, serde_json::Value>,
}

impl Snapshot {
    fn new(dir: PathBuf) -> Self {
        std::fs::create_dir_all(&dir).unwrap();
        let mut snapshot = Self {
            dir,
            weight_map: serde_json::Map::new(),
        };
        snapshot.add("unused");
        snapshot
    }

    fn contains(&self, key: &str) -> bool {
        self.weight_map.contains_key(key)
    }

    fn add(&mut self, key: &str) {
        let shard = format!("{key}.safetensors");
        let mut header = serde_json::to_vec(&serde_json::json!({
            key: {"dtype": "F32", "shape": SHAPE, "data_offsets": [0, 4]}
        }))
        .unwrap();
        while !header.len().is_multiple_of(8) {
            header.push(b' ');
        }
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(&header);
        bytes.extend_from_slice(&1.0f32.to_le_bytes());
        std::fs::write(self.dir.join(&shard), bytes).unwrap();
        self.weight_map.insert(key.to_owned(), shard.into());
        let index = serde_json::json!({ "weight_map": self.weight_map });
        std::fs::write(
            self.dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&index).unwrap(),
        )
        .unwrap();
    }
}

/// A CPU `TtsModel` over a synthetic talker + codec snapshot holding every
/// key the loaders ask for, found by loading until nothing is missing.
fn loaded_cpu_model(dir: &Path) -> TtsModel {
    let mut talker = Snapshot::new(dir.join("talker"));
    let mut codec = Snapshot::new(dir.join("codec"));
    for _ in 0..2000 {
        let mut model = TtsModel::new_for_test(
            tiny_config(),
            talker.dir.clone(),
            codec.dir.clone(),
            Device::Cpu,
        );
        match model.load() {
            Ok(()) => return model,
            Err(TtsError::Load(msg)) => {
                let (key, _) = msg.split_once(": ").unwrap();
                let snapshot = if model.talker.is_none() {
                    &mut talker
                } else {
                    &mut codec
                };
                assert!(!snapshot.contains(key), "{key} present but refused: {msg}");
                snapshot.add(key);
            }
            Err(other) => panic!("CPU load failed: {other:?}"),
        }
    }
    panic!("the synthetic snapshot never completed");
}

#[test]
fn tts_codec_load_on_cpu_is_admitted() {
    run_child("tts_codec_load_on_cpu_is_admitted_child");
}

#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn tts_codec_load_on_cpu_is_admitted_child() {
    if !started_by_parent() {
        return;
    }
    rmlx_mlx::forbid_gpu();
    let dir = tempfile::tempdir().unwrap();
    let model = loaded_cpu_model(dir.path());
    assert!(model.codec.is_some(), "the codec decoder loaded on the CPU");
    println!("{CHILD_DONE}");
}

#[test]
fn tts_synthesize_on_cpu_is_admitted() {
    run_child("tts_synthesize_on_cpu_is_admitted_child");
}

/// The synthetic weights cannot produce speech, so synthesis may fail on a
/// shape; what it must not do is ask for the GPU.
#[test]
#[ignore = "child process; its parent test starts it with a marker argument"]
fn tts_synthesize_on_cpu_is_admitted_child() {
    if !started_by_parent() {
        return;
    }
    rmlx_mlx::forbid_gpu();
    let dir = tempfile::tempdir().unwrap();
    let mut model = loaded_cpu_model(dir.path());
    let result = synthesize("hello", "serena", &mut model, &TtsTokenizer::stub());
    if let Err(e) = &result {
        assert!(
            !e.to_string().contains("GPU work is refused"),
            "synthesis asked for the GPU: {e}"
        );
    }
    println!("{CHILD_DONE}");
}
