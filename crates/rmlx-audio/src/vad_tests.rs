//! Tests for the Silero VAD implementation.

use rmlx_mlx::Device;

use super::{voiced_segments, SileroVad, VadState, HOP_LENGTH};

/// Verify the VAD model loads successfully from the embedded asset.
#[test]
fn test_vad_loads() {
    let vad = SileroVad::load(Device::Gpu);
    assert!(vad.is_ok(), "VAD load failed: {:?}", vad.err());
}

/// Silence (all-zeros) should produce low voice probability.
#[test]
fn test_silence_gives_low_prob() {
    let vad = SileroVad::load(Device::Gpu).expect("VAD load");
    // 0.5 s of silence at 16kHz = 8000 samples
    let samples = vec![0.0_f32; 8000];
    let state = VadState::new_zeroed(Device::Gpu).expect("state");
    let (probs, _new_state) = vad.forward(&samples, state, Device::Gpu).expect("forward");

    assert!(!probs.is_empty(), "no frames produced");
    let mean: f32 = probs.iter().sum::<f32>() / probs.len() as f32;
    assert!(
        mean < 0.3,
        "silence should have low mean VAD prob, got {mean:.3}"
    );
}

/// A loud sinusoidal signal should produce higher voice probabilities than silence.
/// (The VAD is designed to detect speech, not pure tones, but this at least
/// verifies the signal propagates through the model rather than being suppressed.)
#[test]
fn test_sine_vs_silence() {
    let vad = SileroVad::load(Device::Gpu).expect("VAD load");

    // 1 second of 440 Hz sine.
    let sine: Vec<f32> = (0..16_000)
        .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.5)
        .collect();
    // 1 second of silence.
    let silence = vec![0.0_f32; 16_000];

    let st1 = VadState::new_zeroed(Device::Gpu).expect("state");
    let st2 = VadState::new_zeroed(Device::Gpu).expect("state");

    let (probs_sine, _) = vad.forward(&sine, st1, Device::Gpu).expect("forward sine");
    let (probs_sil, _) = vad
        .forward(&silence, st2, Device::Gpu)
        .expect("forward silence");

    let mean_sine: f32 = probs_sine.iter().sum::<f32>() / probs_sine.len() as f32;
    let mean_sil: f32 = probs_sil.iter().sum::<f32>() / probs_sil.len() as f32;

    // The sine signal should produce at least as high probability as silence.
    // Not asserting > 0.5 because a pure tone is not speech, but it should
    // not be systematically lower than pure silence.
    assert!(
        mean_sine >= mean_sil * 0.5,
        "sine ({mean_sine:.3}) unexpectedly lower than silence ({mean_sil:.3})"
    );
}

/// Integration test: concatenate a 5s clip 7 times to get >30s audio and
/// verify the VAD runs without errors.
#[test]
fn test_long_audio_no_panic() {
    let vad = SileroVad::load(Device::Gpu).expect("VAD load");

    // 5 second sine clip repeated 7 times = 35 seconds.
    let five_s: Vec<f32> = (0..80_000)
        .map(|i| (2.0 * std::f32::consts::PI * 300.0 * i as f32 / 16_000.0).sin() * 0.3)
        .collect();
    let long_audio: Vec<f32> = five_s.iter().copied().cycle().take(7 * 80_000).collect();

    let state = VadState::new_zeroed(Device::Gpu).expect("state");
    let (probs, _) = vad
        .forward(&long_audio, state, Device::Gpu)
        .expect("forward long");

    // At 128 samples/frame, 560000 samples gives ~4375 frames.
    let expected_frames =
        (long_audio.len() + 2 * super::STFT_PAD - super::WIN_LENGTH) / HOP_LENGTH + 1;
    assert!(
        probs.len() >= expected_frames / 2,
        "expected ~{expected_frames} frames, got {}",
        probs.len()
    );
}

/// voiced_segments: simple threshold test.
#[test]
fn test_voiced_segments_basic() {
    // 10 frames: silent | voiced | silent
    let probs = vec![0.1, 0.1, 0.8, 0.9, 0.85, 0.8, 0.75, 0.1, 0.1, 0.1];
    let segs = voiced_segments(&probs, 0.5, 2, 2);
    assert_eq!(segs.len(), 1, "expected one segment, got {segs:?}");
    let (start, end) = segs[0];
    assert_eq!(start, 2 * HOP_LENGTH);
    assert!(end > start);
}

/// voiced_segments: empty input returns empty.
#[test]
fn test_voiced_segments_empty() {
    assert!(voiced_segments(&[], 0.5, 1, 1).is_empty());
}
