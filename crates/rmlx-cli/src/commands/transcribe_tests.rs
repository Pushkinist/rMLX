//! Unit tests for `rmlx transcribe` arch dispatch (no model load).

use super::{run_transcribe, TranscribeArgs};
use crate::commands::parse::ClaimedDevice;
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
    let err = run_transcribe(&args, &ClaimedDevice::cpu()).unwrap_err();
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
    assert!(run_transcribe(&args, &ClaimedDevice::cpu()).is_err());
}

/// A file the decoder refuses is refused with the decoder's fact, before any
/// model load: the model directory holds a config and no weights.
#[test]
fn a_refused_audio_file_is_refused_before_the_model_loads() {
    // ADTS (AAC LC, 16 kHz, mono): 20 silent frames, then a header with the
    // forbidden sample-rate index.
    const SILENT: [u8; 11] = [
        0xFF, 0xF1, 0x60, 0x40, 0x01, 0x7F, 0xFC, 0x00, 0xC8, 0x00, 0x07,
    ];
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "whisper"}"#,
    )
    .unwrap();
    let mut aac = SILENT.repeat(20);
    aac.extend_from_slice(&[0xFF, 0xF1, 0x7C, 0x40, 0x01, 0x7F, 0xFC]);
    let audio = dir.path().join("cut.aac");
    std::fs::write(&audio, aac).unwrap();

    let args = TranscribeArgs {
        audio: &audio,
        model: dir.path(),
        tokenizer: None,
        format: "txt",
        language: "auto",
        translate: false,
    };
    let msg = format!(
        "{}",
        run_transcribe(&args, &ClaimedDevice::cpu()).unwrap_err()
    );
    assert_eq!(
        msg,
        "audio decode: audio read failed at 1.28 s: malformed stream: adts: forbidden sample rate"
    );
}

/// A language Whisper does not know is refused with its value, before the
/// audio is read: the audio path does not exist, so a late check would fail on
/// the read instead.
#[test]
fn an_unknown_language_is_refused_naming_the_value() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("config.json"),
        r#"{"model_type": "whisper"}"#,
    )
    .unwrap();
    let args = TranscribeArgs {
        audio: Path::new("/nonexistent/audio.wav"),
        model: dir.path(),
        tokenizer: None,
        format: "txt",
        language: "xx",
        translate: false,
    };
    let msg = format!(
        "{}",
        run_transcribe(&args, &ClaimedDevice::cpu()).unwrap_err()
    );
    assert_eq!(
        msg,
        "unknown language 'xx'; use a Whisper language code, an English language name or 'auto'"
    );
}
