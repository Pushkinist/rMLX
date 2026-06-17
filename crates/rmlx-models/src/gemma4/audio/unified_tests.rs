//! Model-free unit coverage for the gemma4_unified audio embedder front-end:
//! config parse, soft-token math, and the waveform-frame chunking/padding
//! plumbing (frame count, padding, layout).

use super::*;

fn cfg_12b() -> UnifiedAudioConfig {
    // Verified gemma-4-12B `gemma4_unified_audio` values.
    UnifiedAudioConfig {
        audio_embed_dim: 640,
        audio_samples_per_token: 640,
        output_proj_dims: 640,
        rms_norm_eps: 1e-6,
        audio_token_id: 258_881,
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "test asserts JSON parse succeeds on a known-good literal"
)]
fn config_parse_matches_snapshot() {
    let v: serde_json::Value = serde_json::from_str(
        r#"{
          "model_type": "gemma4_unified_audio",
          "audio_embed_dim": 640, "audio_samples_per_token": 640,
          "output_proj_dims": 640, "rms_norm_eps": 1e-06
        }"#,
    )
    .unwrap();
    let cfg = UnifiedAudioConfig::from_json(&v, 258_881);
    assert_eq!(cfg.audio_embed_dim, 640);
    assert_eq!(cfg.audio_samples_per_token, 640);
    assert_eq!(cfg.output_proj_dims, 640);
    assert_eq!(cfg.audio_token_id, 258_881);
}

#[test]
fn config_defaults_when_keys_missing() {
    let v: serde_json::Value = serde_json::from_str("{}").unwrap_or(serde_json::Value::Null);
    let cfg = UnifiedAudioConfig::from_json(&v, 1);
    assert_eq!(cfg.audio_embed_dim, 640);
    assert_eq!(cfg.audio_samples_per_token, 640);
    assert_eq!(cfg.output_proj_dims, 640);
    assert_eq!(cfg.audio_token_id, 1);
}

#[test]
fn soft_token_count_is_ceil_div_640() {
    let cfg = cfg_12b();
    // Exactly one frame.
    assert_eq!(unified_num_audio_soft_tokens(640, &cfg), 1);
    // One sample over → two frames (ceil).
    assert_eq!(unified_num_audio_soft_tokens(641, &cfg), 2);
    // One short of a frame → still one frame (ceil rounds up).
    assert_eq!(unified_num_audio_soft_tokens(639, &cfg), 1);
    // 16000 samples (1 s @ 16 kHz) → ceil(16000/640) = 25 frames.
    assert_eq!(unified_num_audio_soft_tokens(16_000, &cfg), 25);
}

#[test]
fn soft_token_count_zero_for_empty_clip() {
    let cfg = cfg_12b();
    assert_eq!(unified_num_audio_soft_tokens(0, &cfg), 0);
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test indices/slices bounded by the asserted frame length (700 < 1280)"
)]
fn extract_frames_pads_tail_to_full_frame() {
    // 700 samples → 2 frames (640 + 60), tail zero-padded to 1280.
    let samples: Vec<f32> = (0..700).map(|i| i as f32).collect();
    let (frames, num_tokens) = extract_waveform_frames(&samples, 640);
    assert_eq!(num_tokens, 2);
    assert_eq!(frames.len(), 2 * 640);
    // The original samples are preserved verbatim (no scaling/normalization).
    for (i, &v) in samples.iter().enumerate() {
        assert!((frames[i] - v).abs() < f32::EPSILON, "sample {i} mismatch");
    }
    // The padded tail (700..1280) is zero.
    for &v in &frames[700..] {
        assert!(v.abs() < f32::EPSILON, "tail padding not zero");
    }
}

#[test]
fn extract_frames_exact_multiple_no_extra_frame() {
    // 1280 = 2 * 640 exactly → 2 frames, no padding.
    let samples = vec![1.0_f32; 1280];
    let (frames, num_tokens) = extract_waveform_frames(&samples, 640);
    assert_eq!(num_tokens, 2);
    assert_eq!(frames.len(), 1280);
    assert!(frames.iter().all(|&v| (v - 1.0).abs() < f32::EPSILON));
}

#[test]
fn extract_frames_empty_clip_is_zero_frames() {
    let (frames, num_tokens) = extract_waveform_frames(&[], 640);
    assert_eq!(num_tokens, 0);
    assert!(frames.is_empty());
}

#[test]
fn frame_count_matches_soft_token_count() {
    // The host front-end frame count must equal the prompt-block soft-token
    // count for the scatter to align.
    let cfg = cfg_12b();
    for n in [0usize, 1, 639, 640, 641, 1500, 16_000, 48_000] {
        let samples = vec![0.5_f32; n];
        let (_frames, num_tokens) = extract_waveform_frames(&samples, cfg.audio_samples_per_token);
        assert_eq!(
            num_tokens,
            unified_num_audio_soft_tokens(n, &cfg),
            "frame/soft-token mismatch at n={n}"
        );
    }
}

// ---------------------------------------------------------------------------
// Model-gated integration: load the real 12B `embed_audio` projection and run
// a forward over synthetic frames, asserting the [1, num_tokens, 3840] output
// shape. Skips gracefully when the snapshot is not available.
// ---------------------------------------------------------------------------

/// Resolve the unified 12B snapshot dir from a dedicated env var, falling back
/// to `<RMLX_O_MODELS_ROOT>/mlx-community__gemma-4-12B-it-mxfp8`. Returns `None`
/// when neither is set / the dir is absent (test skips).
fn unified_12b_dir() -> Option<std::path::PathBuf> {
    if let Some(p) = std::env::var_os("RMLX_TEST_MODEL_GEMMA4_12B") {
        let p = std::path::PathBuf::from(p);
        return p.exists().then_some(p);
    }
    let root = std::env::var_os("RMLX_O_MODELS_ROOT")?;
    let p = std::path::PathBuf::from(root).join("mlx-community__gemma-4-12B-it-mxfp8");
    p.exists().then_some(p)
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "model-gated integration test: .expect() documents the load/forward invariant; failure here is a genuine test failure"
)]
fn unified_audio_embedder_forward_real_weights() {
    let Some(dir) = unified_12b_dir() else {
        eprintln!("SKIP: gemma-4-12B snapshot not available (set RMLX_TEST_MODEL_GEMMA4_12B or RMLX_O_MODELS_ROOT)");
        return;
    };
    let cfg = match UnifiedAudioConfig::from_model_dir(&dir) {
        Ok(Some(c)) => c,
        Ok(None) => {
            eprintln!("SKIP: no audio_config in {}", dir.display());
            return;
        }
        Err(e) => {
            eprintln!("SKIP: config read failed: {e}");
            return;
        }
    };
    let embedder = match load_unified_audio_embedder(&dir, &cfg) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("SKIP: embed_audio load failed: {e}");
            return;
        }
    };

    // 3 frames of synthetic raw waveform (1920 samples).
    let num_tokens = 3usize;
    let (frames, n) = extract_waveform_frames(&vec![0.01_f32; 1920], cfg.audio_samples_per_token);
    assert_eq!(n, num_tokens);

    let out = embedder
        .forward(&frames, num_tokens, Device::Cpu)
        .expect("unified audio forward");
    out.eval().expect("eval");
    let shp = out.shape();
    assert_eq!(shp.first().copied(), Some(1), "batch dim");
    assert_eq!(shp.get(1).copied(), Some(num_tokens as i32), "token dim");
    // Projected to text hidden (3840 on the 12B).
    assert_eq!(shp.get(2).copied(), Some(3840), "hidden dim");
}
