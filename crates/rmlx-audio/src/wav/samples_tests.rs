//! Tests for the mono samples of a decode: the downmix, the resample fed in
//! parts, and the sample rates the decoder takes.

use super::MonoSamples;
use crate::wav::tests::summary;
use crate::wav::{WavDecoder, WavEncoder, WavError};

/// `wav` decoded to `rate` Hz under the production duration cap, never
/// cancelled.
fn resampled(wav: &[u8], rate: u32) -> Result<crate::wav::DecodedAudio, WavError> {
    WavDecoder::decode_resampled(wav, rate, crate::wav::MAX_AUDIO_SECS, &|| false)
}

/// The linear resample of a whole buffer that a resample fed in parts must
/// match bit for bit.
fn whole_buffer_resample(samples: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || samples.is_empty() {
        return samples.to_vec();
    }
    let ratio = f64::from(to) / f64::from(from);
    let out_len = ((samples.len() as f64) * ratio).round() as usize;
    let last = samples.len() - 1;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 / ratio;
            let idx = src_pos.floor() as usize;
            if idx >= last {
                return samples[last];
            }
            let frac = (src_pos - idx as f64) as f32;
            samples[idx] + (samples[idx + 1] - samples[idx]) * frac
        })
        .collect()
}

/// `n` samples in [-1, 1], with exact zeros of both signs among them.
fn noise(n: usize) -> Vec<f32> {
    let mut state: u32 = 0x1234_5678;
    (0..n)
        .map(|i| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            match i % 97 {
                0 => 0.0,
                1 => -0.0,
                _ => (state >> 8) as f32 / (1 << 23) as f32 - 1.0,
            }
        })
        .collect()
}

fn bits(samples: &[f32]) -> Vec<u32> {
    samples.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn a_resample_fed_in_parts_matches_the_whole_buffer_resample_bit_for_bit() {
    let rates = [
        8_000, 11_025, 16_000, 22_050, 24_000, 32_000, 44_100, 48_000, 96_000,
    ];
    let mut checked = 0;
    for from in rates {
        for len in [0, 1, 2, 3, 5, 17, 1_000, 4_801] {
            let samples = noise(len);
            for part in [1, 7, 1_152, len.max(1)] {
                for (hold, cut) in [(0, 0), (5, 0), (5, 5), (2_000, 1_105)] {
                    let mut mono =
                        MonoSamples::new(from, Some(16_000), hold, len, crate::wav::MAX_AUDIO_SECS);
                    for chunk in samples.chunks(part) {
                        mono.push_frames(chunk, 1).unwrap();
                    }
                    let got = mono.finish(cut);
                    let kept = &samples[..len.saturating_sub(cut)];
                    let want = whole_buffer_resample(kept, from, 16_000);
                    assert_eq!(
                        bits(&got),
                        bits(&want),
                        "from {from} Hz, {len} samples in parts of {part}, hold {hold}, cut {cut}"
                    );
                    checked += 1;
                }
            }
        }
    }
    assert_eq!(checked, 9 * 8 * 4 * 4);
}

#[test]
fn channels_are_averaged_and_silence_is_zero_frames() {
    let mut mono = MonoSamples::new(16_000, None, 0, 0, crate::wav::MAX_AUDIO_SECS);
    mono.push_frames(&[0.5, -0.25, 1.0, 0.0, -1.0, -0.5], 2)
        .unwrap();
    mono.push_silence(2).unwrap();
    mono.push_frames(&[0.25, 0.75], 2).unwrap();
    assert_eq!(mono.finish(0), [0.125, 0.5, -0.75, 0.0, 0.0, 0.5]);

    // The sum times the reciprocal of the count, as the whole-buffer
    // downmix did: 0.9 / 3 in f32 is 0.29999998.
    let mut three = MonoSamples::new(16_000, None, 0, 0, crate::wav::MAX_AUDIO_SECS);
    three.push_frames(&[0.9, 0.0, 0.0], 3).unwrap();
    assert_eq!(three.finish(0), [0.3]);
}

#[test]
fn a_resample_to_the_file_rate_returns_the_samples_unchanged() {
    let samples = noise(1_001);
    let wav = WavEncoder::encode(&samples, 16_000, 1).unwrap();
    let native = WavDecoder::decode(&wav).unwrap();
    let resampled = resampled(&wav, 16_000).unwrap();
    assert_eq!(bits(&resampled.samples), bits(&native.samples));
    assert_eq!(resampled.sample_rate, 16_000);
}

#[test]
fn a_48k_file_resampled_to_16k_holds_a_third_of_its_samples() {
    let ramp: Vec<f32> = (0..4_800).map(|i| i as f32 / 4_800.0).collect();
    let wav = WavEncoder::encode(&ramp, 48_000, 1).unwrap();
    let got = resampled(&wav, 16_000).unwrap();
    assert_eq!(got.samples.len(), 1_600);
    assert_eq!(got.sample_rate, 48_000, "the file's own rate");
    assert!(got.samples.first().is_some_and(|&x| x < 0.05));
    assert!(got.samples.last().is_some_and(|&x| x > 0.9));
}

/// `got` is the refusal of a sample rate of `hz`.
fn refused_for_rate(got: &Result<crate::wav::DecodedAudio, WavError>, hz: u32) -> bool {
    got.as_ref().is_err_and(|e| {
        e.to_string()
            == format!("sample rate of {hz} Hz is outside the 4000 to 768000 Hz the decoder takes")
    })
}

#[test]
fn a_file_rate_outside_4_khz_to_768_khz_is_refused() {
    for hz in [0, 1, 3_999, 768_001] {
        let wav = WavEncoder::encode(&[0.25; 100], hz, 1).unwrap();
        for got in [WavDecoder::decode(&wav), resampled(&wav, 16_000)] {
            assert!(refused_for_rate(&got, hz), "{hz} Hz: {}", summary(&got));
        }
    }
    for hz in [4_000, 768_000] {
        let wav = WavEncoder::encode(&[0.25; 100], hz, 1).unwrap();
        let got = resampled(&wav, 16_000);
        assert!(got.is_ok(), "{hz} Hz: {}", summary(&got));
    }
}

#[test]
fn a_resample_rate_outside_4_khz_to_768_khz_is_refused() {
    let wav = WavEncoder::encode(&[0.0; 160], 16_000, 1).unwrap();
    for hz in [0, 1, 3_999, 768_001] {
        let got = resampled(&wav, hz);
        assert!(refused_for_rate(&got, hz), "{hz} Hz: {}", summary(&got));
    }
}

#[test]
fn a_stream_too_short_for_one_output_sample_is_no_sample_not_an_error() {
    // One sample at 48 kHz is a third of a sample at 16 kHz, which rounds to
    // none, as the whole-buffer resample gives.
    let wav = WavEncoder::encode(&[0.5], 48_000, 1).unwrap();
    let got = resampled(&wav, 16_000);
    assert!(
        got.as_ref().is_ok_and(|d| d.samples.is_empty()),
        "{}",
        summary(&got)
    );
}
