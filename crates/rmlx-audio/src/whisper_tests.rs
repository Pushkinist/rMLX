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
use crate::tokenizer::{TOK_EOT, TOK_NO_TIMESTAMPS, TOK_TIMESTAMP_BEGIN};
use rmlx_mlx::Dtype;

const VOCAB: usize = 51_866;

fn fresh_logits() -> Vec<f32> {
    // All zeros so we can see exactly which entries get masked to -inf.
    vec![0.0_f32; VOCAB]
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
    assert!(l[VOCAB - 1].is_infinite());
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
