//! Whisper model unit tests.
//!
//! Full model tests (load + transcribe) are integration tests under
//! `crates/rmlx-audio/tests/transcribe.rs`. They resolve the Whisper snapshot
//! from `RMLX_O_MODELS_ROOT` auto-discovery (the same convention as
//! `make model-check-full`) and skip gracefully when the model or fixtures are
//! absent — no bespoke env var. The tests here verify config parsing, NPZ
//! parsing logic, and weight-map helpers.

use super::{DecodeFilters, WhisperConfig};
use crate::npz::{extract_npy_dtype, extract_npy_shape};
use crate::tokenizer::{N_VOCAB, TOK_EOT, TOK_NO_TIMESTAMPS, TOK_TIMESTAMP_BEGIN};
use rmlx_mlx::Dtype;

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

/// A `config.json` with the large-v3 sizes except the four given.
fn config_json(n_mels: usize, n_vocab: usize, n_audio_ctx: usize, n_text_layer: usize) -> String {
    serde_json::json!({
        "n_mels": n_mels,
        "n_audio_ctx": n_audio_ctx,
        "n_audio_state": 1280,
        "n_audio_head": 20,
        "n_audio_layer": 32,
        "n_vocab": n_vocab,
        "n_text_ctx": 448,
        "n_text_state": 1280,
        "n_text_head": 20,
        "n_text_layer": n_text_layer,
    })
    .to_string()
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
        WhisperConfig::from_json(&config_json(128, 51_866, 1500, n_text_layer))
            .unwrap_or_else(|e| panic!("n_text_layer {n_text_layer}: {e}"));
    }
}

/// tiny, base, small, medium, large-v1 and large-v2 state `n_vocab` 51865 and
/// 80 mel bins; their `.en` models state 51864 (mlx-community `config.json`).
#[test]
fn a_config_of_another_vocabulary_is_refused_naming_n_vocab() {
    for n_vocab in [51_865, 51_864] {
        let err = config_refusal(&config_json(80, n_vocab, 1500, 4));
        assert_names(&err, &["n_vocab", &n_vocab.to_string(), "51866"]);
    }
}

/// The decode's 30 s window is 3000 mel frames, which the encoder halves to
/// 1500 positions.
#[test]
fn an_audio_context_other_than_1500_is_refused() {
    let err = config_refusal(&config_json(128, 51_866, 750, 32));
    assert_names(&err, &["n_audio_ctx", "750", "1500"]);
}

/// The refusal comes before `weights.npz` is read: a directory with a large-v2
/// `config.json` and no weights fails on `n_vocab`, and one with a large-v3
/// `config.json` and no weights fails on the weight file.
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
    let refused = load_error(config_json(80, 51_865, 1500, 32));
    assert_names(&refused, &["config.json", "n_vocab", "51865"]);
    let no_weights = load_error(config_json(128, 51_866, 1500, 32));
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
        config_json(128, 51_866, 1500, 32),
    )
    .unwrap();
    let no_weights = load_error();
    assert_names(&no_weights, &["weights.npz", "No such file"]);
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

/// NPY dtype extraction for f2 (float16).
#[test]
fn npy_dtype_f2() {
    let header = "{'descr': '<f2', 'fortran_order': False, 'shape': (1280, 1280), }";
    let dtype = extract_npy_dtype(header);
    assert_eq!(dtype, Some(Dtype::F16));
}

/// NPY dtype extraction for f4 (float32).
#[test]
fn npy_dtype_f4() {
    let header = "{'descr': '<f4', 'fortran_order': False, 'shape': (100,), }";
    let dtype = extract_npy_dtype(header);
    assert_eq!(dtype, Some(Dtype::F32));
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
