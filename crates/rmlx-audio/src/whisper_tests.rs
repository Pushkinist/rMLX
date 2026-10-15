//! Whisper model unit tests.
//!
//! Full model tests (load + transcribe) are integration tests under
//! `crates/rmlx-audio/tests/transcribe.rs`. They resolve the Whisper snapshot
//! from `RMLX_O_MODELS_ROOT` auto-discovery (the same convention as
//! `make model-check-full`) and skip gracefully when the model or fixtures are
//! absent — no bespoke env var. The tests here verify config parsing, NPZ
//! parsing logic, and weight-map helpers.

use super::snapshot_fixture::{config_json, decoder_tensors, tiny_config_json, weight_map};
use super::{DecodeFilters, TextDecoder, WhisperConfig, WhisperError};
use crate::npz::extract_npy_shape;
use crate::tokenizer::{
    N_VOCAB, TOK_EN, TOK_EOT, TOK_NO_TIMESTAMPS, TOK_SOT, TOK_TIMESTAMP_BEGIN, TOK_TRANSCRIBE,
};
use rmlx_mlx::{Array, Device};

fn fresh_logits() -> Vec<f32> {
    // All zeros so we can see exactly which entries get masked to -inf.
    vec![0.0_f32; N_VOCAB]
}

/// SuppressBlank masks EOT + blank ids at the first step only.
#[test]
fn suppress_blank_first_step_only() {
    let blank = vec![TOK_EOT, 220];
    let f = DecodeFilters::new(vec![], blank, false);
    let mut l = fresh_logits();
    f.apply(&mut l, &[], true);
    assert!(l[TOK_EOT as usize].is_infinite());
    assert!(l[220].is_infinite());

    // Not the first step: blank ids are no longer suppressed (EOT allowed).
    let mut l2 = fresh_logits();
    f.apply(&mut l2, &[5_u32], false);
    assert!(l2[TOK_EOT as usize].is_finite());
}

/// no_timestamps mode masks every timestamp token and notimestamps.
#[test]
fn no_timestamps_mode_masks_all_timestamps() {
    let f = DecodeFilters::new(vec![], vec![TOK_EOT], false);
    let mut l = fresh_logits();
    f.apply(&mut l, &[5_u32], false);
    assert!(l[TOK_TIMESTAMP_BEGIN as usize].is_infinite());
    assert!(l[N_VOCAB - 1].is_infinite());
    // A plain text token stays finite.
    assert!(l[100].is_finite());
}

/// Timestamp mode: the first sampled token must be a timestamp (BOS rule).
#[test]
fn timestamp_mode_bos_forces_timestamp() {
    let f = DecodeFilters::new(vec![], vec![TOK_EOT], true);
    let mut l = fresh_logits();
    f.apply(&mut l, &[], true);
    // All text < timestamp_begin masked.
    assert!(l[100].is_infinite());
    assert!(l[TOK_NO_TIMESTAMPS as usize].is_infinite());
    // Timestamps remain available.
    assert!(l[TOK_TIMESTAMP_BEGIN as usize].is_finite());
}

/// Timestamp mode: right after a single opening timestamp the model must be
/// able to emit TEXT (the penultimate<2 ⇒ treated-as-timestamp branch forces a
/// non-timestamp next). This is the bug that produced empty transcripts.
#[test]
fn timestamp_mode_after_open_ts_allows_text() {
    let f = DecodeFilters::new(vec![], vec![TOK_EOT], true);
    let mut l = fresh_logits();
    // One opening timestamp sampled.
    f.apply(&mut l, &[TOK_TIMESTAMP_BEGIN], false);
    // Text tokens must remain selectable (NOT all masked → not forced to EOT).
    assert!(
        l[100].is_finite(),
        "after a single opening timestamp the decoder must be able to emit text"
    );
    // And further timestamps are masked (must pair with non-timestamp first).
    assert!(l[TOK_TIMESTAMP_BEGIN as usize].is_infinite());
}

/// Config JSON parses correctly for the large-v3 snapshot.
#[test]
fn parse_config_large_v3() {
    let json = r#"{
        "n_mels": 128,
        "n_audio_ctx": 1500,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_vocab": 51866,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": 32,
        "model_type": "whisper"
    }"#;
    let cfg = WhisperConfig::from_json(json).unwrap();
    assert_eq!(cfg.n_mels, 128);
    assert_eq!(cfg.n_audio_ctx, 1500);
    assert_eq!(cfg.n_audio_state, 1280);
    assert_eq!(cfg.n_vocab, 51866);
    assert_eq!(cfg.n_text_ctx, 448);
}

fn config_refusal(json: &str) -> String {
    let err = WhisperConfig::from_json(json)
        .expect_err("the config must be refused")
        .to_string();
    assert!(
        err.contains("config.json"),
        "the error names no file: {err}"
    );
    err
}

fn assert_names(err: &str, words: &[&str]) {
    for word in words {
        assert!(err.contains(word), "the error does not name {word}: {err}");
    }
}

/// large-v3, large-v3-turbo (4 decoder layers) and distil-large-v3 (2) share
/// the large-v3 vocabulary (mlx-community `config.json` of each).
#[test]
fn the_large_v3_family_configs_load() {
    for n_text_layer in [32, 4, 2] {
        WhisperConfig::from_json(&config_json(128, 51_866, 1500, n_text_layer).to_string())
            .unwrap_or_else(|e| panic!("n_text_layer {n_text_layer}: {e}"));
    }
}

/// tiny, base, small, medium, large-v1 and large-v2 state `n_vocab` 51865 and
/// 80 mel bins; their `.en` models state 51864 (mlx-community `config.json`).
#[test]
fn a_config_of_another_vocabulary_is_refused_naming_n_vocab() {
    for n_vocab in [51_865, 51_864] {
        let err = config_refusal(&config_json(80, n_vocab, 1500, 4).to_string());
        assert_names(&err, &["n_vocab", &n_vocab.to_string(), "51866"]);
    }
}

/// The decode's 30 s window is 3000 mel frames, which the encoder halves to
/// 1500 positions.
#[test]
fn an_audio_context_other_than_1500_is_refused() {
    let err = config_refusal(&config_json(128, 51_866, 750, 32).to_string());
    assert_names(&err, &["n_audio_ctx", "750", "1500"]);
}

/// A quantized snapshot stores packed weights that the load would read as
/// plain matrices, and the first window would fail in a matmul: the load
/// refuses the `quantization` key and names it.
#[test]
fn a_quantized_config_is_refused_naming_the_key() {
    let mut config = config_json(128, 51_866, 1500, 4);
    config["quantization"] = serde_json::json!({ "group_size": 64, "bits": 4 });
    let err = config_refusal(&config.to_string());
    assert_names(&err, &["quantization"]);
}

/// A `quantization` key of `null` states no quantization.
#[test]
fn a_null_quantization_key_loads() {
    let mut config = config_json(128, 51_866, 1500, 4);
    config["quantization"] = serde_json::Value::Null;
    WhisperConfig::from_json(&config.to_string()).expect("a null key states no quantization");
}

/// The refusal comes before the weight file is read: a directory with a
/// large-v2 or a quantized `config.json` and no weights fails on the config,
/// and one with a large-v3 `config.json` and no weights fails on the weight file.
#[test]
fn load_refuses_the_config_before_it_reads_the_weights() {
    let dir = tempfile::tempdir().unwrap();
    let load_error = |json: String| {
        std::fs::write(dir.path().join("config.json"), json).unwrap();
        match super::WhisperModel::load(dir.path()) {
            Ok(_) => panic!("a directory with no weights must not load"),
            Err(e) => e.to_string(),
        }
    };
    let refused = load_error(config_json(80, 51_865, 1500, 32).to_string());
    assert_names(&refused, &["config.json", "n_vocab", "51865"]);
    let mut quantized = config_json(128, 51_866, 1500, 32);
    quantized["quantization"] = serde_json::json!({ "group_size": 64, "bits": 8 });
    assert_names(&load_error(quantized.to_string()), &["quantization"]);
    let no_weights = load_error(config_json(128, 51_866, 1500, 32).to_string());
    assert_names(&no_weights, &["weight file error"]);
}

/// The server sends a load error to the client: it names the file, never the
/// directory the operator configured.
#[test]
fn a_load_error_names_the_file_and_no_directory() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().display().to_string();
    let load_error = || match super::WhisperModel::load(dir.path()) {
        Ok(_) => panic!("a directory with no weights must not load"),
        Err(e) => e.to_string(),
    };
    let no_config = load_error();
    assert_names(&no_config, &["config.json", "No such file"]);
    assert!(!no_config.contains(&path), "{no_config}");
    std::fs::write(
        dir.path().join("config.json"),
        config_json(128, 51_866, 1500, 32).to_string(),
    )
    .unwrap();
    let no_weights = load_error();
    assert_names(&no_weights, &["weights.safetensors", "weights.npz"]);
    assert!(!no_weights.contains(&path), "{no_weights}");
}

/// The local large-v3 `config.json` passes the check. Runs on the CPU when
/// `RMLX_O_MODELS_ROOT` holds `mlx-community__whisper-large-v3-mlx`; no weight
/// is read.
#[test]
fn the_local_large_v3_config_loads() {
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT") else {
        eprintln!("SKIP the_local_large_v3_config_loads: RMLX_O_MODELS_ROOT is unset");
        return;
    };
    let path = std::path::Path::new(&root)
        .join("mlx-community__whisper-large-v3-mlx")
        .join("config.json");
    let Ok(json) = std::fs::read_to_string(&path) else {
        eprintln!(
            "SKIP the_local_large_v3_config_loads: no mlx-community__whisper-large-v3-mlx config.json under the root"
        );
        return;
    };
    let cfg = WhisperConfig::from_json(&json).expect("the large-v3 config must load");
    assert_eq!(cfg.n_vocab, 51_866);
}

/// NPY shape extraction for 2D array.
#[test]
fn npy_shape_2d() {
    let header = "{'descr': '<f2', 'fortran_order': False, 'shape': (1280, 1280), }";
    let shape = extract_npy_shape(header).unwrap();
    assert_eq!(shape, vec![1280, 1280]);
}

/// NPY shape extraction for 1D array.
#[test]
fn npy_shape_1d() {
    let header = "{'descr': '<f2', 'fortran_order': False, 'shape': (1280,), }";
    let shape = extract_npy_shape(header).unwrap();
    assert_eq!(shape, vec![1280]);
}

/// NPY shape extraction for 3D array.
#[test]
fn npy_shape_3d() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (32, 400, 1280), }";
    let shape = extract_npy_shape(header).unwrap();
    assert_eq!(shape, vec![32, 400, 1280]);
}

/// NPY shape of a scalar (0-dimensional) is empty vec.
#[test]
fn npy_shape_scalar() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (), }";
    let shape = extract_npy_shape(header).unwrap();
    assert_eq!(shape, Vec::<usize>::new());
}

// NOTE: Full smoke-probe + long-form regression tests (WhisperModel::load →
// transcribe) live in crates/rmlx-audio/tests/transcribe.rs and resolve the
// snapshot from RMLX_O_MODELS_ROOT auto-discovery (skip-if-absent).

// ── The decoder of one window, on a decoder of two small blocks ──────────────

const TINY_STATE: usize = 4;
const TINY_BLOCKS: usize = 2;

/// An array of `shape` with small values that differ from one element to the
/// next and from one `seed` to the next.
fn values(shape: &[i32], seed: u32) -> Result<Array, WhisperError> {
    let len: i32 = shape.iter().product();
    let data: Vec<f32> = (0..len as u32)
        .map(|i| ((i.wrapping_mul(2_654_435_761).wrapping_add(seed) >> 20) as f32 / 4096.0) - 0.5)
        .collect();
    Ok(Array::from_f32_slice(&data, shape)?)
}

/// A text decoder of two blocks with a state of four, the large-v3 vocabulary
/// and a context of 16 positions.
fn tiny_decoder() -> Result<TextDecoder, WhisperError> {
    let cfg = WhisperConfig::from_json(&tiny_config_json(0, TINY_BLOCKS).to_string())?;
    TextDecoder::load(&weight_map(&decoder_tensors(TINY_BLOCKS)), &cfg)
}

fn first_largest(row: &[f32]) -> u32 {
    let mut best = 0;
    for (id, logit) in row.iter().enumerate() {
        if row.get(best).is_some_and(|largest| logit > largest) {
            best = id;
        }
    }
    best as u32
}

/// The cross-attention keys and values of a window are projected one time,
/// by the forward that reads the start of the window: one projection for each
/// decoder block. Each decode of the window reads them and projects nothing,
/// whatever the number of decodes. A second window projects its own.
#[test]
fn the_cross_attention_of_a_window_is_projected_one_time() -> Result<(), WhisperError> {
    let decoder = tiny_decoder()?;
    let encoder_out = values(&[1, 3, TINY_STATE as i32], 99)?;
    let filters = DecodeFilters::new(Vec::new(), Vec::new(), true);
    let prefix = [TOK_SOT, TOK_EN, TOK_TRANSCRIBE];

    assert_eq!(decoder.cross_projections(), 0);
    let (_, window) = decoder.open_window(&encoder_out, Device::Cpu)?;
    assert_eq!(decoder.cross_projections(), TINY_BLOCKS);
    let mut decodes = Vec::new();
    for decode in 1..=3 {
        decodes.push(window.decode(&prefix, 4, &filters, |row: &[f32]| Ok(first_largest(row)))?);
        assert_eq!(
            decoder.cross_projections(),
            TINY_BLOCKS,
            "after decode {decode} of the window"
        );
    }
    assert!(decodes.iter().all(|decoded| !decoded.tokens.is_empty()));
    assert!(decodes.windows(2).all(|pair| pair[0] == pair[1]));

    decoder.open_window(&encoder_out, Device::Cpu)?;
    assert_eq!(decoder.cross_projections(), 2 * TINY_BLOCKS);
    Ok(())
}
