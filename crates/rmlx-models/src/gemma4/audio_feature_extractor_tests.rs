use super::*;
use std::f32::consts::PI;

/// Default extractor matching the e4b snapshot params.
fn default_extractor() -> Gemma4AudioFeatureExtractor {
    Gemma4AudioFeatureExtractor::new(
        128,    // feature_size
        16_000, // sampling_rate
        20.0,   // frame_length_ms → 320 samples
        10.0,   // hop_length_ms   → 160 samples
        0.0,    // min_frequency
        8000.0, // max_frequency
        0.0,    // preemphasis (off)
        true,   // preemphasis_htk_flavor
        false,  // fft_overdrive
        1e-3,   // mel_floor
        1.0,    // input_scale_factor
        None,   // per_bin_mean
        None,   // per_bin_stddev
    )
}

/// Generate a pure 1 kHz sine wave at 16 kHz sample rate.
fn sine_1khz(duration_secs: f32, sample_rate: u32) -> Vec<f32> {
    let n = (duration_secs * sample_rate as f32) as usize;
    (0..n)
        .map(|i| (2.0 * PI * 1000.0 * i as f32 / sample_rate as f32).sin())
        .collect()
}

// ── Shape test ────────────────────────────────────────────────────────────

/// DoD: output shape must be `[T_frames][feature_size]` with the
/// analytically correct frame count.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn gemma4_mel_shape() {
    let ext = default_extractor();
    let samples = sine_1khz(1.0, 16_000); // 16 000 samples
    let frames = ext.extract(&samples).expect("extract should succeed");

    // Analytic frame count (matches Python `_unfold` formula):
    // padded_len = n_samples + frame_length // 2 = 16000 + 160 = 16160
    // frame_size_for_unfold = frame_length + 1 = 321
    // T = (16160 - 321) / 160 + 1 = 15839 / 160 + 1 = 98 + 1 = 99
    let expected_frames = ext.num_frames(samples.len());
    assert_eq!(
        frames.len(),
        expected_frames,
        "frame count mismatch: got {}, expected {}",
        frames.len(),
        expected_frames
    );

    // Every frame must have exactly `feature_size` mel bins.
    for (t, frame) in frames.iter().enumerate() {
        assert_eq!(
            frame.len(),
            ext.feature_size,
            "frame[{t}] has wrong width: {} != {}",
            frame.len(),
            ext.feature_size
        );
    }

    // Sanity: feature_size is 128, frame count should be 99 for 1 s @ 16 kHz.
    assert_eq!(ext.feature_size, 128);
    assert_eq!(
        expected_frames, 99,
        "analytic frame count for 1 s @ 16 kHz should be 99, got {expected_frames}"
    );
}

// ── Peak-bin test ─────────────────────────────────────────────────────────

/// For a 1 kHz sine, the peak mel bin across all frames should correspond
/// to a frequency in the neighbourhood of 1 kHz.
///
/// The mel filterbank for params (min=0 Hz, max=8000 Hz, 128 bins,
/// sampling_rate=16 kHz, fft_length=512) maps mel bin `m` to
/// approximately:
/// mel_to_hz(mel_min + m * (mel_max - mel_min) / 128)
///
/// 1 kHz in HTK mel = 2595 * log10(1 + 1000/700) ≈ 999.985 mel.
/// mel_min = 0 (since min_frequency = 0).
/// mel_max = hz_to_mel(8000) ≈ 2840.02 mel.
/// Bin index at 1 kHz ≈ 1000 / 2840 * 128 ≈ 45.
///
/// We check the peak bin across the middle frames is in [30, 65] — a ±15
/// bin tolerance that covers filterbank quantisation and spectral leakage.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn gemma4_mel_peak_bin_at_1khz() {
    let ext = default_extractor();
    let samples = sine_1khz(1.0, 16_000);
    let frames = ext.extract(&samples).expect("extract should succeed");

    // Use the middle third of frames to avoid edge effects.
    let mid_start = frames.len() / 3;
    let mid_end = 2 * frames.len() / 3;
    let mid_frames = &frames[mid_start..mid_end];

    // Average energy across middle frames.
    let mut avg = vec![0.0_f32; ext.feature_size];
    for frame in mid_frames {
        for (m, &v) in frame.iter().enumerate() {
            avg[m] += v;
        }
    }
    let n = mid_frames.len() as f32;
    for v in &mut avg {
        *v /= n;
    }

    // Find peak bin.
    let peak_bin = avg
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .expect("avg must be non-empty");

    assert!(
        (30..=65).contains(&peak_bin),
        "peak mel bin {peak_bin} is outside [30, 65] for a 1 kHz sine"
    );
}

// ── Mel filterbank symmetry test ──────────────────────────────────────────

/// All filterbank weights must be non-negative; the vast majority of mel
/// bins must have non-zero support.
///
/// With `min_frequency = 0 Hz`, sampling_rate = 16 000 Hz, and
/// `fft_length = 512` the FFT bin spacing is 31.25 Hz. The first
/// mel filter spans [0, ~28 Hz] — narrower than one FFT bin — so it
/// legally produces all-zero weights. We skip that degenerate bin in
/// the non-zero assertion and verify the remaining 127 bins are all active.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn mel_filterbank_positive_weights() {
    let filters = build_mel_filterbank(257, 128, 0.0, 8000.0, 16_000);

    // 1. All weights must be non-negative.
    for (i, row) in filters.iter().enumerate() {
        for (j, &w) in row.iter().enumerate() {
            assert!(w >= 0.0, "filters[{i}][{j}] = {w} is negative");
        }
    }

    // 2. Mel bins 1..128 must each have at least one non-zero weight.
    // Mel bin 0 may be all-zero when min_frequency = 0 because the
    // filter's support (≈ 0–28 Hz) is narrower than the FFT bin
    // spacing (31.25 Hz for fft_length=512, sr=16000).
    for m in 1..128 {
        let total: f32 = filters.iter().map(|row| row[m]).sum();
        assert!(total > 0.0, "mel bin {m} has all-zero weights");
    }
}

// ── Config round-trip test ────────────────────────────────────────────────

/// Parse the e4b processor_config.json (feature_extractor block) and verify
/// the derived parameters match the expected values.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn config_parse_e4b() {
    // Matches the `feature_extractor` block in processor_config.json.
    let json = r#"{
        "feature_extractor_type": "Gemma4AudioFeatureExtractor",
        "sampling_rate": 16000,
        "num_mel_filters": 128,
        "fft_length": 512,
        "hop_length": 160
    }"#;

    let ext =
        Gemma4AudioFeatureExtractor::from_processor_config_str(json).expect("parse must succeed");

    assert_eq!(ext.feature_size, 128);
    assert_eq!(ext.sampling_rate, 16_000);
    assert_eq!(ext.fft_length(), 512);
    assert_eq!(ext.hop_length(), 160);
    // frame_length from default frame_length_ms=20 ms @ 16 kHz = 320.
    assert_eq!(ext.frame_length(), 320);
}

// ── Numeric parity dump (manual comparison against Python) ────────────────

/// Dump values that can be compared against the Python reference.
///
/// Expected Python output (run against the same 1 kHz / 1 s signal):
/// frame[50][:5]: [-6.908, 0.657, -0.527, 0.456, -0.052] (frame 0)
/// frame[50][:5]: [-6.908, -5.206, -6.045, -5.542, -5.894]
/// peak mel bin (middle frames): 44
///
/// Tolerance: max abs diff < 0.01 (f32 rounding vs f64 numpy).
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn gemma4_mel_numeric_parity() {
    let ext = default_extractor();
    let samples = sine_1khz(1.0, 16_000);
    let frames = ext.extract(&samples).unwrap();

    // Python reference values for frame[50][:5]:
    // [-6.9077554, -5.2064962, -6.0453873, -5.54231, -5.8942695]
    let py_frame50: [f32; 5] = [
        -6.907_755_4,
        -5.206_496_2,
        -6.045_387_3,
        -5.542_31,
        -5.894_269_5,
    ];
    let rust_frame50 = &frames[50];
    for (m, (&py, &ru)) in py_frame50.iter().zip(rust_frame50.iter()).enumerate() {
        let diff = (py - ru).abs();
        assert!(
            diff < 0.01,
            "frame[50][{m}]: Python={py:.6}, Rust={ru:.6}, diff={diff:.6}"
        );
    }

    // Peak mel bin should match Python's 44.
    let mid_frames = &frames[33..66];
    let mut avg = vec![0.0_f32; ext.feature_size];
    for frame in mid_frames {
        for (m, &v) in frame.iter().enumerate() {
            avg[m] += v;
        }
    }
    let n = mid_frames.len() as f32;
    for v in &mut avg {
        *v /= n;
    }
    let peak_bin = avg
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    assert_eq!(
        peak_bin, 44,
        "peak mel bin should match Python reference (44)"
    );
}

// ── HTK mel formula test ──────────────────────────────────────────────────

/// Verify the HTK mel conversion functions are inverses of each other.
#[test]
fn htk_mel_roundtrip() {
    for &hz in &[0.0_f64, 100.0, 1000.0, 4000.0, 8000.0] {
        let mel = hz_to_mel_htk(hz);
        let recovered = mel_to_hz_htk(mel);
        assert!(
            (recovered - hz).abs() < 1e-6,
            "roundtrip failed at {hz} Hz: mel={mel}, recovered={recovered}"
        );
    }
}
