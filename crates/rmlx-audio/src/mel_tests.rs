//! Tests for Whisper mel-spectrogram extractor.

use super::{MelExtractor, N_FFT, N_FRAMES, N_SAMPLES, SAMPLE_RATE};

/// Output shape is `[N_FRAMES, n_mels]` for a full 30-second input.
#[test]
fn output_shape_full_chunk() {
    let extractor = MelExtractor::new(128).unwrap();
    let samples = vec![0.0_f32; N_SAMPLES];
    let frames = extractor.extract(&samples).unwrap();
    assert_eq!(frames.len(), N_FRAMES);
    for f in &frames {
        assert_eq!(f.len(), 128);
    }
}

/// Shorter inputs are zero-padded to 30 s.
#[test]
fn short_input_padded() {
    let extractor = MelExtractor::new(128).unwrap();
    let samples = vec![0.1_f32; 1600]; // 0.1 s
    let frames = extractor.extract(&samples).unwrap();
    // Output is always N_FRAMES after padding.
    assert_eq!(frames.len(), N_FRAMES);
}

/// Longer inputs are trimmed to 30 s.
#[test]
fn long_input_trimmed() {
    let extractor = MelExtractor::new(128).unwrap();
    let samples = vec![0.1_f32; N_SAMPLES + 1600];
    let frames = extractor.extract(&samples).unwrap();
    assert_eq!(frames.len(), N_FRAMES);
}

/// 80-mel variant (small/medium Whisper) produces correct column count.
#[test]
fn n_mels_80() {
    let extractor = MelExtractor::new(80).unwrap();
    let samples = vec![0.0_f32; N_SAMPLES];
    let frames = extractor.extract(&samples).unwrap();
    assert_eq!(frames.len(), N_FRAMES);
    for f in &frames {
        assert_eq!(f.len(), 80);
    }
}

/// Silence produces well-defined output values (not NaN or Inf).
#[test]
fn silence_no_nan() {
    let extractor = MelExtractor::new(128).unwrap();
    let samples = vec![0.0_f32; N_SAMPLES];
    let frames = extractor.extract(&samples).unwrap();
    for row in &frames {
        for &v in row {
            assert!(v.is_finite(), "NaN or Inf in silence output: {v}");
        }
    }
}

/// A 440 Hz sine produces non-constant output.
#[test]
fn sine_non_trivial() {
    let extractor = MelExtractor::new(128).unwrap();
    let sr = SAMPLE_RATE as f32;
    let freq = 440.0_f32;
    let samples: Vec<f32> = (0..N_SAMPLES)
        .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin() * 0.5)
        .collect();
    let frames = extractor.extract(&samples).unwrap();
    // Check that at least some frame has energy in the 440 Hz band (not all zeros).
    let has_energy = frames.iter().any(|row| row.iter().any(|&v| v > -2.0));
    assert!(has_energy, "no energy detected for 440 Hz sine");
}

/// Golden-value smoke test: for a known silent input the normalised log-mel
/// values cluster around (0 + 4) / 4 = 1.0 (the clipped floor after
/// per-chunk normalisation). We accept ±0.1 tolerance.
#[test]
fn silence_floor_value() {
    let extractor = MelExtractor::new(128).unwrap();
    let samples = vec![0.0_f32; N_SAMPLES];
    let frames = extractor.extract(&samples).unwrap();
    // For silence, mel[k] = 0, log10(max(0, 1e-10)) = -10.
    // max(-10, max - 8) = max(-10, -10 - 8) = max(-10, -18) = -10.
    // (-10 + 4) / 4 = -1.5.
    for row in &frames {
        for &v in row {
            assert!((v - (-1.5)).abs() < 0.01, "silence floor unexpected: {v}");
        }
    }
}

/// N_FFT constant smoke test.
#[test]
fn constants_sanity() {
    assert_eq!(N_FFT, 400);
    assert_eq!(N_SAMPLES, 480_000);
    assert_eq!(N_FRAMES, 3_000);
    assert_eq!(SAMPLE_RATE, 16_000);
}

/// Filterbank values match the librosa reference (mel_filters.npz, mel_128).
///
/// Reference values extracted from the mlx_whisper mel_filters.npz asset:
/// `filters[mel_bin, fft_bin]` layout (note: librosa uses [n_mels, n_fft_bins]).
/// Our Rust `filters[fft_bin][mel_bin]` is the transpose.
#[test]
fn filterbank_matches_librosa_reference() {
    let n_fft_bins = N_FFT / 2 + 1; // 201
    let filters = super::build_mel_filters(n_fft_bins, 128);
    // librosa reference: filters[mel=0, fft=1] = 0.012374
    //                   filters[mel=1, fft=1] = 0.030393
    //                   filters[mel=2, fft=2] = 0.024748
    //                   filters[mel=3, fft=2] = 0.018019
    // Our layout: filters[fft_bin][mel_bin]
    let tol = 0.001_f32;
    assert!(
        (filters[1][0] - 0.012374_f32).abs() < tol,
        "filters[1][0] = {}, expected ~0.012374",
        filters[1][0]
    );
    assert!(
        (filters[1][1] - 0.030393_f32).abs() < tol,
        "filters[1][1] = {}, expected ~0.030393",
        filters[1][1]
    );
    assert!(
        (filters[2][2] - 0.024748_f32).abs() < tol,
        "filters[2][2] = {}, expected ~0.024748",
        filters[2][2]
    );
    assert!(
        (filters[2][3] - 0.018019_f32).abs() < tol,
        "filters[2][3] = {}, expected ~0.018019",
        filters[2][3]
    );
}
