//! Whisper model unit tests.
//!
//! Full model tests (load + transcribe) are integration tests under `tests/`
//! and are gated by the `RMLX_TEST_MODEL_WHISPER` env var. The tests here
//! verify config parsing, NPZ parsing logic, and weight-map helpers.

use super::WhisperConfig;
use crate::npz::{extract_npy_dtype, extract_npy_shape};
use rmlx_mlx::Dtype;

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

// NOTE: Full smoke-probe test (WhisperModel::load → transcribe 1 s WAV) lives
// in crates/rmlx-audio/tests/smoke.rs and is gated by RMLX_TEST_MODEL_WHISPER.
