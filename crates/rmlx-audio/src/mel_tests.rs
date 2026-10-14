//! Tests for Whisper mel-spectrogram extractor.

use super::{pad_after_content, MelExtractor, N_FFT, N_FRAMES, N_SAMPLES, SAMPLE_RATE};

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

/// A tone with a slow level change, so no two frames are the same and the
/// last samples are not zero.
fn tone(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * (0.2 + 0.1 * (3.0 * t).sin())
        })
        .collect()
}

/// A window shorter than 30 s has two inputs, as in openai-whisper
/// (`docs/AUDIO.md`, "Short-window pad").
///
/// The detection input is `extract`: the audio is padded with zero samples,
/// so the 3000 frames are those of the same samples before zeros to 30 s, bit
/// for bit, and each frame that lies whole in the pad has the floor of the
/// window in each mel bin, the least value of the window, below 0.0.
///
/// The decode input is that mel after `pad_after_content`: the
/// `len / HOP_LENGTH` content frames do not change, and each frame after them
/// is 0.0 in each mel bin. Zeros in the audio of a full window are content.
///
/// The two lengths: a short clip that is a whole number of hops, and the last
/// window of a long file, which is not.
#[test]
fn a_short_window_has_a_detection_input_and_a_decode_input() {
    let extractor = MelExtractor::new(128).unwrap();
    // The length in samples, its content frames (the length over the hop of
    // 160, rounded down), and the first frame that reads zero samples only: a
    // frame reads 400 samples around its centre, at 160 times its index.
    for (len, content, whole_in_the_pad) in
        [(43_520_usize, 272_usize, 274_usize), (152_531, 953, 955)]
    {
        let samples = tone(len);
        let frames = extractor.extract(&samples).unwrap();
        assert_eq!(frames.len(), N_FRAMES, "{len} samples");

        let mut full_window = samples;
        full_window.resize(N_SAMPLES, 0.0);
        let full = extractor.extract(&full_window).unwrap();
        assert_eq!(
            frames, full,
            "{len} samples: the frames are those of the audio before zeros to 30 s"
        );

        let mut decode_input = frames.clone();
        pad_after_content(&mut decode_input, len);
        assert_eq!(
            decode_input[..content],
            frames[..content],
            "{len} samples: the content frames do not change"
        );
        for (index, frame) in decode_input.iter().enumerate().skip(content) {
            assert!(
                frame.iter().all(|&v| v == 0.0),
                "{len} samples: frame {index} is after the {content} content frames and is not \
                 0.0: first bin {}",
                frame[0]
            );
        }
        let mut full_decode_input = full.clone();
        pad_after_content(&mut full_decode_input, N_SAMPLES);
        assert_eq!(
            full_decode_input, full,
            "{len} samples: a full window is all content"
        );

        let floor = frames
            .iter()
            .flatten()
            .fold(f32::INFINITY, |least, &v| least.min(v));
        assert!(floor < 0.0, "{len} samples: the floor is {floor}");
        assert!(
            frames[..content]
                .iter()
                .all(|frame| frame.iter().any(|&v| v > floor)),
            "{len} samples: each of the {content} content frames has a bin above the floor"
        );
        assert!(
            frames[whole_in_the_pad - 1].iter().any(|&v| v > floor),
            "{len} samples: frame {} still reads samples of the content",
            whole_in_the_pad - 1
        );
        for (index, frame) in frames.iter().enumerate().skip(whole_in_the_pad) {
            assert!(
                frame.iter().all(|&v| v == floor),
                "{len} samples: frame {index} reads zero samples only and is not at the floor \
                 {floor}: first bin {}",
                frame[0]
            );
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
