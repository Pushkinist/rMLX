//! Unit tests for `rmlx transcribe` arch dispatch (no model load).

use super::{run_transcribe, TranscribeArgs};
use rmlx_mlx::Device;
use std::path::Path;

/// A non-Whisper config.json yields a clear "unsupported architecture" error
/// without attempting to load any model.
#[test]
fn rejects_non_whisper_arch() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]}"#,
    )
    .unwrap();

    let args = TranscribeArgs {
        audio: Path::new("/nonexistent/audio.wav"),
        model: dir.path(),
        tokenizer: None,
        format: "txt",
        language: "auto",
        translate: false,
    };
    let err = run_transcribe(&args, Device::Cpu).unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("unsupported ASR architecture") && msg.contains("qwen3"),
        "unexpected error: {msg}"
    );
}

/// A missing config.json is a clean error, not a panic.
#[test]
fn missing_config_is_error() {
    let dir = tempfile::tempdir().unwrap();
    let args = TranscribeArgs {
        audio: Path::new("/nonexistent/audio.wav"),
        model: dir.path(),
        tokenizer: None,
        format: "txt",
        language: "auto",
        translate: false,
    };
    assert!(run_transcribe(&args, Device::Cpu).is_err());
}
