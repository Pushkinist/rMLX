//! Tests for WAV encode/decode round-trip.

use super::{WavDecoder, WavEncoder};

/// Encode a short 440 Hz sine wave and decode it back; verify round-trip.
#[test]
fn round_trip_440hz() {
    let sample_rate: u32 = 16_000;
    let freq = 440.0_f32;
    let duration_secs = 0.1_f32;
    let n = (sample_rate as f32 * duration_secs) as usize;
    let original: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect();

    let wav_bytes = WavEncoder::encode(&original, sample_rate, 1).unwrap();
    // Check WAV magic.
    assert_eq!(&wav_bytes[0..4], b"RIFF");
    assert_eq!(&wav_bytes[8..12], b"WAVE");

    let (decoded, rate) = WavDecoder::decode(&wav_bytes).unwrap();
    assert_eq!(rate, sample_rate);
    assert_eq!(decoded.len(), original.len());

    // Quantisation noise should be < 1/32768 ≈ 3e-5.
    for (orig, dec) in original.iter().zip(decoded.iter()) {
        let err = (orig - dec).abs();
        assert!(err < 1e-4, "sample error {err} too large");
    }
}

/// Silence encodes and decodes to zero.
#[test]
fn silence_round_trip() {
    let silence = vec![0.0_f32; 160];
    let bytes = WavEncoder::encode(&silence, 16_000, 1).unwrap();
    let (decoded, rate) = WavDecoder::decode(&bytes).unwrap();
    assert_eq!(rate, 16_000);
    for s in &decoded {
        assert!(s.abs() < 1e-6, "non-zero silence sample: {s}");
    }
}

/// WAV header is exactly 44 bytes for PCM.
#[test]
fn header_size() {
    let samples = vec![0.0_f32; 100];
    let bytes = WavEncoder::encode(&samples, 16_000, 1).unwrap();
    assert_eq!(bytes.len(), 44 + 100 * 2);
}
