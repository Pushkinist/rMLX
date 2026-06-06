//! Qwen3-TTS unit tests.

use super::{load_codec_decoder, synthesize, TtsConfig, TtsError, TtsModel, TtsTokenizer};
use rmlx_mlx::Array;

/// Config parses correctly for the CustomVoice snapshot.
#[test]
fn tts_config_parses() {
    let json = r#"{
        "model_type": "qwen3_tts",
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "tts_pad_token_id": 151671,
        "talker_config": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 28,
            "num_code_groups": 16,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_language_id": {"english": 2050},
            "spk_id": {"serena": 3066, "vivian": 3065}
        }
    }"#;
    let cfg = TtsConfig::from_json(json).expect("config should parse");
    assert_eq!(cfg.model_type, "qwen3_tts");
    assert_eq!(cfg.tts_bos_token_id, 151672);
    assert_eq!(cfg.talker_config.hidden_size, 2048);
    assert_eq!(cfg.talker_config.num_code_groups, 16);
    assert_eq!(cfg.speaker_id("serena"), Some(3066));
    assert_eq!(cfg.speaker_id("vivian"), Some(3065));
    assert!(cfg.speaker_id("unknown_voice").is_none());
}

/// Synthesize with an unknown voice returns the correct error.
#[test]
fn synthesize_unknown_voice_returns_error() {
    let json = r#"{
        "model_type": "qwen3_tts",
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "tts_pad_token_id": 151671,
        "talker_config": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 28,
            "num_code_groups": 16,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_language_id": {},
            "spk_id": {"serena": 3066}
        }
    }"#;
    let config = TtsConfig::from_json(json).unwrap();
    let mut model = TtsModel::new_for_test(
        config,
        std::path::PathBuf::from("/tmp"),
        std::path::PathBuf::from("/tmp"),
    );
    let tok = TtsTokenizer::stub();

    let result = synthesize("hello", "nonexistent_voice", &mut model, &tok);
    match result {
        Err(TtsError::UnknownVoice(v)) => assert_eq!(v, "nonexistent_voice"),
        other => panic!("expected UnknownVoice, got {other:?}"),
    }
}

/// Synthesize with a valid voice but missing weights returns a load error.
/// Phase 4b implementation is complete; this test exercises the lazy-load
/// error path that fires when the talker model directory is invalid.
#[test]
fn synthesize_valid_voice_returns_load_error() {
    let json = r#"{
        "model_type": "qwen3_tts",
        "tts_bos_token_id": 151672,
        "tts_eos_token_id": 151673,
        "tts_pad_token_id": 151671,
        "talker_config": {
            "hidden_size": 2048,
            "num_attention_heads": 16,
            "num_key_value_heads": 8,
            "num_hidden_layers": 28,
            "num_code_groups": 16,
            "codec_bos_id": 2149,
            "codec_eos_token_id": 2150,
            "codec_language_id": {},
            "spk_id": {"serena": 3066}
        }
    }"#;
    let config = TtsConfig::from_json(json).unwrap();
    let mut model = TtsModel::new_for_test(
        config,
        std::path::PathBuf::from("/tmp"),
        std::path::PathBuf::from("/tmp"),
    );
    let tok = TtsTokenizer::stub();

    let result = synthesize("hello", "serena", &mut model, &tok);
    assert!(
        matches!(result, Err(TtsError::Load(_))),
        "expected Load error when weights missing, got {result:?}"
    );
}

/// Debug codec decoder: loads real weights and runs with fixed codes.
/// Prints intermediate values so they can be compared with reference Python.
/// Run with: cargo test -p rmlx-audio codec_decoder_debug -- --nocapture 2>&1 | grep -v "^running\|^test "
#[test]
#[ignore = "requires real model weights at Open Models path; run manually for debugging"]
fn codec_decoder_debug() {
    let codec_path = std::path::Path::new(
        "models/mlx-community__Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit/speech_tokenizer",
    );
    if !codec_path.exists() {
        eprintln!("SKIP: codec path not found");
        return;
    }

    let decoder = load_codec_decoder(codec_path).expect("load codec decoder");

    // codes: [1, 16, 12] — code 100 for semantic, 50 for acoustic (same as Python reference)
    let t = 12i32;
    let mut codes_data = vec![0i32; 16 * 12];
    for item in codes_data.iter_mut().take(12) {
        *item = 100; // semantic code
    }
    for i in 1..16 {
        for j in 0..12 {
            codes_data[i * 12 + j] = 50; // acoustic codes
        }
    }
    let codes = Array::from_i32_slice(&codes_data, &[1, 16, t]).expect("build codes array");

    let d = rmlx_mlx::Device::Gpu;
    let samples = decoder.debug_decode(&codes, d).expect("debug decode");
    eprintln!(
        "samples len={} rms={:.6}",
        samples.len(),
        (samples.iter().map(|x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
    );
}
