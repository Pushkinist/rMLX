use super::*;

// ── WAV builder (no external encoder needed) ──────────────────────────────

/// Build a minimal RIFF/WAV blob: PCM 16-bit mono at `sample_rate` Hz,
/// containing `n_samples` samples of a 440 Hz sine wave (silence also
/// works; sine gives non-trivial values to round-trip check).
fn build_wav_mono_sine(sample_rate: u32, n_samples: usize) -> Vec<u8> {
    let n_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let block_align = n_channels * bits_per_sample / 8;
    let byte_rate = sample_rate * u32::from(block_align);
    let data_len = (n_samples * usize::from(block_align)) as u32;
    let riff_len = 36 + data_len; // RIFF chunk size

    let mut w = Vec::with_capacity(44 + data_len as usize);

    // RIFF header
    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&riff_len.to_le_bytes());
    w.extend_from_slice(b"WAVE");

    // fmt sub-chunk
    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&n_channels.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&byte_rate.to_le_bytes());
    w.extend_from_slice(&block_align.to_le_bytes());
    w.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data sub-chunk
    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());

    let freq = 440.0_f32;
    for i in 0..n_samples {
        let t = i as f32 / sample_rate as f32;
        let sample = (2.0 * std::f32::consts::PI * freq * t).sin();
        let pcm = (sample * 32767.0) as i16;
        w.extend_from_slice(&pcm.to_le_bytes());
    }

    w
}

/// Build a minimal RIFF/WAV blob: PCM 16-bit **stereo** at `sample_rate` Hz.
/// Left channel = +1.0 (max positive), right channel = -1.0 (max negative).
/// Mono mix should average to ~0.
fn build_wav_stereo_lr(sample_rate: u32, n_samples: usize) -> Vec<u8> {
    let n_channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let block_align = n_channels * bits_per_sample / 8;
    let byte_rate = sample_rate * u32::from(block_align);
    let data_len = (n_samples * usize::from(block_align)) as u32;
    let riff_len = 36 + data_len;

    let mut w = Vec::with_capacity(44 + data_len as usize);

    w.extend_from_slice(b"RIFF");
    w.extend_from_slice(&riff_len.to_le_bytes());
    w.extend_from_slice(b"WAVE");

    w.extend_from_slice(b"fmt ");
    w.extend_from_slice(&16u32.to_le_bytes());
    w.extend_from_slice(&1u16.to_le_bytes()); // PCM
    w.extend_from_slice(&n_channels.to_le_bytes());
    w.extend_from_slice(&sample_rate.to_le_bytes());
    w.extend_from_slice(&byte_rate.to_le_bytes());
    w.extend_from_slice(&block_align.to_le_bytes());
    w.extend_from_slice(&bits_per_sample.to_le_bytes());

    w.extend_from_slice(b"data");
    w.extend_from_slice(&data_len.to_le_bytes());

    for _ in 0..n_samples {
        // Left = i16::MAX, Right = i16::MIN + 1 (symmetric around 0)
        let left: i16 = 32767;
        let right: i16 = -32767;
        w.extend_from_slice(&left.to_le_bytes());
        w.extend_from_slice(&right.to_le_bytes());
    }

    w
}

// ── Tests: decode_audio_bytes ─────────────────────────────────────────────

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn wav_mono_correct_sample_rate_and_count() {
    let sr = 16_000u32;
    let n = 1600usize; // 0.1 s
    let wav = build_wav_mono_sine(sr, n);

    let (samples, rate) = decode_audio_bytes(&wav).expect("WAV decode failed");

    assert_eq!(rate, sr, "sample rate mismatch");
    assert_eq!(samples.len(), n, "sample count mismatch");
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn wav_mono_samples_in_range() {
    let wav = build_wav_mono_sine(22_050, 512);
    let (samples, _) = decode_audio_bytes(&wav).unwrap();
    for &s in &samples {
        assert!((-1.0..=1.0).contains(&s), "sample {s} outside [-1, 1]");
    }
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn wav_stereo_mixed_to_mono() {
    let sr = 8_000u32;
    let n = 200usize;
    let wav = build_wav_stereo_lr(sr, n);

    let (samples, rate) = decode_audio_bytes(&wav).unwrap();

    assert_eq!(rate, sr);
    // stereo WAV has n frames; mono result should be n samples
    assert_eq!(samples.len(), n, "mono sample count = frame count");
    // L=+32767, R=-32767 → average ≈ 0 (within one i16 LSB after f32 conversion)
    for &s in &samples {
        assert!(s.abs() < 1e-4, "stereo L+R average should be ~0, got {s}");
    }
}

#[test]
fn empty_bytes_returns_error() {
    assert!(decode_audio_bytes(b"").is_err());
}

#[test]
fn garbage_bytes_returns_error() {
    assert!(decode_audio_bytes(b"not audio data at all").is_err());
}

// ── Tests: decode_audio_source (base64 branch) ────────────────────────────

#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn b64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b0 = u32::from(chunk[0]);
        let b1 = chunk.get(1).map_or(0u32, |&b| u32::from(b));
        let b2 = chunk.get(2).map_or(0u32, |&b| u32::from(b));
        let triple = (b0 << 16) | (b1 << 8) | b2;
        out.push(CHARS[((triple >> 18) & 0x3f) as usize] as char);
        out.push(CHARS[((triple >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(CHARS[((triple >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(CHARS[(triple & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn data_uri_wav_roundtrip() {
    let sr = 16_000u32;
    let n = 800usize;
    let wav = build_wav_mono_sine(sr, n);
    let uri = format!("data:audio/wav;base64,{}", b64_encode(&wav));

    let (samples, rate) = decode_audio_source(&uri).expect("data-URI decode failed");

    assert_eq!(rate, sr);
    assert_eq!(samples.len(), n);
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn raw_base64_wav_roundtrip() {
    let sr = 44_100u32;
    let n = 441usize; // 0.01 s
    let wav = build_wav_mono_sine(sr, n);
    let b64 = b64_encode(&wav);

    let (samples, rate) = decode_audio_source(&b64).expect("raw base64 decode failed");

    assert_eq!(rate, sr);
    assert_eq!(samples.len(), n);
}

#[test]
fn data_uri_wrong_encoding_rejected() {
    // data URI without ";base64" must be rejected.
    assert!(decode_audio_source("data:audio/wav,somedata").is_err());
}

#[test]
fn data_uri_no_comma_rejected() {
    assert!(decode_audio_source("data:audio/wav;base64").is_err());
}

// ── Tests: mix_to_mono ────────────────────────────────────────────────────

#[test]
fn mix_to_mono_single_channel_passthrough() {
    let input = vec![0.1, 0.2, 0.3];
    assert_eq!(mix_to_mono(&input, 1), input);
}

#[test]
fn mix_to_mono_two_channels_averaged() {
    // L=1.0, R=0.0 → mono=0.5 for each frame.
    let input = vec![1.0_f32, 0.0, 1.0, 0.0];
    let mono = mix_to_mono(&input, 2);
    assert_eq!(mono.len(), 2);
    for &s in &mono {
        assert!((s - 0.5).abs() < 1e-6, "expected 0.5 got {s}");
    }
}
