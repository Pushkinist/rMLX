//! Whisper log-mel spectrogram.
//!
//! Faithful port of the Python reference from `mlx-audio/mlx_audio/utils.py`
//! (`stft` + `mel_filters`) and `mlx-audio/mlx_audio/stt/models/whisper/audio.py`
//! (`log_mel_spectrogram`).
//!
//! ## Pipeline
//!
//! 1. Optional right-zero-padding to `N_SAMPLES` (480 000 frames = 30 s at 16 kHz).
//! 2. Hann-windowed STFT: frame length = `N_FFT` = 400, hop = 160, complex FFT.
//! 3. Power spectrum: `|STFT|²` for bins `[0, N_FFT/2]` (201 bins).
//! 4. Triangular mel filterbank (Slaney norm, librosa Slaney mel scale,
//!    `n_mels` = 128 for large-v3). Applied as matrix multiply → `[n_mels, T]`.
//! 5. Log-compression + clipping:
//!    ```text
//!    log_spec = max(mel, 1e-10).log10()
//!    log_spec = max(log_spec, log_spec.max() - 8.0)
//!    log_spec = (log_spec + 4.0) / 4.0
//!    ```
//!
//! ## Output
//!
//! `extract()` returns `[T_frames, n_mels]` (row = one time step, column = mel
//! bin), which is the shape the Whisper encoder expects after the Conv1d stem.
//!
//! ## Differences from Gemma4AudioFeatureExtractor
//!
//! Whisper uses a different mel scale (Slaney normalised), different FFT length
//! (N_FFT = 400 vs 512 for Gemma4), different normalisation scheme, and no
//! preemphasis. We do NOT re-use the Gemma4 extractor.

use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::{num_complex::Complex, FftPlanner};
use thiserror::Error;
use tracing::{debug, instrument};

// ── Whisper audio constants ───────────────────────────────────────────────────

/// Target sample rate (16 kHz).
pub const SAMPLE_RATE: u32 = 16_000;
/// STFT frame length in samples.
pub const N_FFT: usize = 400;
/// STFT hop length in samples (10 ms at 16 kHz).
pub const HOP_LENGTH: usize = 160;
/// Chunk length in seconds (Whisper processes 30 s chunks).
pub const CHUNK_LENGTH: usize = 30;
/// Total samples per 30-second chunk.
pub const N_SAMPLES: usize = CHUNK_LENGTH * SAMPLE_RATE as usize;
/// Total frames in a 30-second chunk.
pub const N_FRAMES: usize = N_SAMPLES / HOP_LENGTH; // 3000

// ── Errors ────────────────────────────────────────────────────────────────────

/// Errors from mel-spectrogram computation.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum MelError {
    /// Audio is empty after padding / trimming.
    #[error("audio is empty after processing")]
    Empty,
    /// Config parameter invalid.
    #[error("mel config error: {0}")]
    Config(String),
}

// ── MelExtractor ─────────────────────────────────────────────────────────────

/// Whisper log-mel feature extractor.
///
/// Construct once and reuse across transcription requests:
/// ```rust,no_run
/// use rmlx_audio::mel::MelExtractor;
/// let samples: Vec<f32> = vec![0.0; 480_000];
/// let extractor = MelExtractor::new(128).unwrap();
/// let frames = extractor.extract(&samples).unwrap();
/// ```
pub struct MelExtractor {
    n_mels: usize,
    /// Mel filterbank matrix, shape `[n_fft_bins][n_mels]`.
    /// n_fft_bins = N_FFT / 2 + 1 = 201.
    filters: Vec<Vec<f32>>,
    /// Hann window of length `N_FFT`.
    window: Vec<f32>,
    /// Cached RFFT plan of length `N_FFT`.
    fft: Arc<dyn rustfft::Fft<f32>>,
}

impl std::fmt::Debug for MelExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MelExtractor")
            .field("n_mels", &self.n_mels)
            .finish_non_exhaustive()
    }
}

impl MelExtractor {
    /// Create a new extractor.
    ///
    /// `n_mels` should be 80 (small/medium Whisper) or 128 (large-v2/v3).
    pub fn new(n_mels: usize) -> Result<Self, MelError> {
        if n_mels == 0 {
            return Err(MelError::Config("n_mels must be > 0".to_owned()));
        }
        let n_fft_bins = N_FFT / 2 + 1; // 201

        // Build Hann window of length N_FFT (periodic, matches torch.hann_window).
        let window: Vec<f32> = (0..N_FFT)
            .map(|n| {
                let angle = 2.0 * PI * n as f32 / N_FFT as f32;
                0.5f32.mul_add(-angle.cos(), 0.5)
            })
            .collect();

        // Mel filterbank: precomputed librosa Slaney filterbank (exact match to
        // mlx_whisper training). Loaded from embedded binary for n_mels ∈ {80,128}.
        let filters = build_mel_filters(n_fft_bins, n_mels);

        // FFT plan of length N_FFT.
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        debug!(n_mels, n_fft_bins, "MelExtractor built");

        Ok(Self {
            n_mels,
            filters,
            window,
            fft,
        })
    }

    /// Extract log-mel features from mono f32 samples.
    ///
    /// `samples` must be at `SAMPLE_RATE` (16 kHz). They are padded or trimmed
    /// to `N_SAMPLES` (30 s = 480 000 samples) before STFT.
    ///
    /// Returns `[T_frames][n_mels]` — one row per 10 ms hop.
    #[instrument(skip(self, samples), fields(n = samples.len()), level = "debug")]
    #[allow(
        clippy::indexing_slicing,
        reason = "indices bounded by construction: fft_buf indexed in [0, N_FFT), frames indexed in [0, num_frames)"
    )]
    pub fn extract(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, MelError> {
        if samples.is_empty() {
            return Err(MelError::Empty);
        }

        // Pad or trim to N_SAMPLES.
        let audio = pad_or_trim(samples);
        let n = audio.len(); // == N_SAMPLES

        let n_fft_bins = N_FFT / 2 + 1; // 201

        // Reflect-pad left by N_FFT/2 (= 200) so first frame is centred at t=0.
        // This matches the Python reference: np.pad(audio, N_FFT // 2, mode='reflect').
        let pad = N_FFT / 2;
        let mut padded = Vec::with_capacity(n + 2 * pad);
        // Reflect left: samples[pad..0] reversed.
        for i in (1..=pad).rev() {
            padded.push(*audio.get(i).unwrap_or(&0.0));
        }
        padded.extend_from_slice(&audio);
        // Reflect right: samples[n-2..n-pad-2] reversed.
        for i in (n.saturating_sub(pad + 1)..n.saturating_sub(1)).rev() {
            padded.push(*audio.get(i).unwrap_or(&0.0));
        }
        // Ensure padded length is at least N_SAMPLES + 2*pad.
        while padded.len() < n + 2 * pad {
            padded.push(0.0);
        }

        let padded_len = padded.len();
        let num_frames = if padded_len >= N_FFT {
            (padded_len - N_FFT) / HOP_LENGTH + 1
        } else {
            0
        };

        if num_frames == 0 {
            return Err(MelError::Empty);
        }

        // Complex FFT scratch buffer (reused per frame).
        let mut fft_buf: Vec<Complex<f32>> = vec![Complex::default(); N_FFT];

        // First pass: compute linear mel-power spectrogram for all frames.
        let mut mel_power: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_LENGTH;

            // Apply Hann window.
            for i in 0..N_FFT {
                let s = padded[start + i];
                fft_buf[i] = Complex::new(s * self.window[i], 0.0);
            }

            // In-place FFT.
            self.fft.process(&mut fft_buf);

            // Power spectrum |STFT|² for bins [0, N_FFT/2+1).
            // Match Python: freqs[:-1] i.e. bins [0, N_FFT/2] (201 values).
            // Multiply each bin's power by the mel filterbank row.
            let mut mel_frame = vec![0.0_f32; self.n_mels];
            for (k, bin) in fft_buf.iter().enumerate().take(n_fft_bins) {
                let power = bin.norm_sqr();
                let filter_row = &self.filters[k];
                for m in 0..self.n_mels {
                    mel_frame[m] += power * filter_row[m];
                }
            }

            mel_power.push(mel_frame);
        }

        // The Python reference does `stft[..., :-1]` to trim the last frame
        // (matches openai/whisper audio.py). Clamp to exactly N_FRAMES.
        mel_power.truncate(N_FRAMES);

        // Second pass: log-compress with global max normalization.
        // Python: log_spec = np.log10(np.maximum(mel, 1e-10))
        //         log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
        //         log_spec = (log_spec + 4.0) / 4.0
        // IMPORTANT: the clamp floor uses the GLOBAL max across all frames,
        // not a per-frame max. Using per-frame max produces incorrect activations.
        let mut global_log_max = f32::NEG_INFINITY;
        for frame in &mel_power {
            for &v in frame {
                let lv = v.max(1e-10_f32).log10();
                if lv > global_log_max {
                    global_log_max = lv;
                }
            }
        }
        let floor = global_log_max - 8.0;

        let log_mel: Vec<Vec<f32>> = mel_power
            .into_iter()
            .map(|frame| {
                frame
                    .into_iter()
                    .map(|v| {
                        let lv = v.max(1e-10_f32).log10();
                        (lv.max(floor) + 4.0) / 4.0
                    })
                    .collect()
            })
            .collect();

        debug!(
            n_frames = log_mel.len(),
            n_mels = self.n_mels,
            "mel extraction complete"
        );
        Ok(log_mel)
    }
}

// ── pad_or_trim ───────────────────────────────────────────────────────────────

/// Pad (zero) or trim `samples` to exactly `N_SAMPLES`.
#[allow(
    clippy::indexing_slicing,
    reason = "slice bounded: checked n > N_SAMPLES immediately before indexing [..N_SAMPLES]"
)]
pub fn pad_or_trim(samples: &[f32]) -> Vec<f32> {
    let n = samples.len();
    if n == N_SAMPLES {
        return samples.to_vec();
    }
    if n > N_SAMPLES {
        return samples[..N_SAMPLES].to_vec();
    }
    let mut out = samples.to_vec();
    out.resize(N_SAMPLES, 0.0_f32);
    out
}

// ── Mel filterbank (Slaney/librosa reference, embedded binary) ───────────────
//
// The filterbank is pre-computed from librosa.filters.mel(sr=16000, n_fft=400,
// n_mels={80,128}, htk=False, norm='slaney') — the exact filter used by
// mlx_whisper (mlx_whisper/assets/mel_filters.npz). Computing it from scratch
// with the HTK formula diverges from the librosa Slaney scale and causes
// incorrect mel spectrogram values, leading to wrong encoder output.
//
// Binary layout: flat row-major float32 array, shape [n_fft_bins][n_mels],
// i.e. filters[fft_bin * n_mels + mel_bin] = weight for that (bin, mel) pair.
// This matches our internal `Vec<Vec<f32>>` structure.

/// Load precomputed librosa mel filterbank for n_mels=128.
/// Shape: `[n_fft_bins=201][n_mels=128]`.
fn build_mel_filters_128() -> Vec<Vec<f32>> {
    const DATA: &[u8] = include_bytes!("mel_filters_128.bin");
    let n_fft_bins = 201_usize;
    let n_mels = 128_usize;
    let floats: Vec<f32> = DATA
        .chunks_exact(4)
        .map(|b| {
            // `chunks_exact(4)` guarantees b.len() == 4; try_into never fails here.
            #[allow(
                clippy::unwrap_used,
                reason = "chunks_exact(4) guarantees exactly 4 bytes; try_into cannot fail"
            )]
            f32::from_le_bytes(b.try_into().unwrap())
        })
        .collect();
    let mut filters = vec![vec![0.0_f32; n_mels]; n_fft_bins];
    for k in 0..n_fft_bins {
        for m in 0..n_mels {
            #[allow(
                clippy::indexing_slicing,
                reason = "k < n_fft_bins and m < n_mels are loop bounds; flat index k*n_mels+m < n_fft_bins*n_mels"
            )]
            {
                filters[k][m] = floats[k * n_mels + m];
            }
        }
    }
    filters
}

/// Load precomputed librosa mel filterbank for n_mels=80.
/// Shape: `[n_fft_bins=201][n_mels=80]`.
fn build_mel_filters_80() -> Vec<Vec<f32>> {
    const DATA: &[u8] = include_bytes!("mel_filters_80.bin");
    let n_fft_bins = 201_usize;
    let n_mels = 80_usize;
    let floats: Vec<f32> = DATA
        .chunks_exact(4)
        .map(|b| {
            // `chunks_exact(4)` guarantees b.len() == 4; try_into never fails here.
            #[allow(
                clippy::unwrap_used,
                reason = "chunks_exact(4) guarantees exactly 4 bytes; try_into cannot fail"
            )]
            f32::from_le_bytes(b.try_into().unwrap())
        })
        .collect();
    let mut filters = vec![vec![0.0_f32; n_mels]; n_fft_bins];
    for k in 0..n_fft_bins {
        for m in 0..n_mels {
            #[allow(
                clippy::indexing_slicing,
                reason = "k < n_fft_bins and m < n_mels are loop bounds; flat index k*n_mels+m < n_fft_bins*n_mels"
            )]
            {
                filters[k][m] = floats[k * n_mels + m];
            }
        }
    }
    filters
}

/// Build mel filterbank matrix of shape `[n_fft_bins][n_mels]`.
///
/// Returns the exact librosa filterbank for n_mels ∈ {80, 128}. For other
/// values, falls back to a runtime-computed Slaney HTK filterbank (not
/// compatible with Whisper models, which were trained on the librosa filter).
#[allow(
    clippy::indexing_slicing,
    reason = "all indices bounded by n_fft_bins and n_mels which are validated at construction"
)]
pub(crate) fn build_mel_filters(n_fft_bins: usize, n_mels: usize) -> Vec<Vec<f32>> {
    // Use the exact pre-computed librosa filterbank for the two standard sizes.
    if n_mels == 128 {
        return build_mel_filters_128();
    }
    if n_mels == 80 {
        return build_mel_filters_80();
    }

    // Fallback: runtime-computed Slaney HTK filterbank (not training-compatible
    // but supports arbitrary n_mels for non-Whisper use cases).
    let sr = f64::from(SAMPLE_RATE);
    let f_max = sr / 2.0;
    let mel_max = 2595.0 * (1.0 + f_max / 700.0).log10();
    let n_pts = n_mels + 2;
    let freq_pts: Vec<f64> = (0..n_pts)
        .map(|i| {
            let mel = mel_max * i as f64 / (n_pts - 1) as f64;
            700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
        })
        .collect();
    let fft_freqs: Vec<f64> = (0..n_fft_bins)
        .map(|k| k as f64 * sr / (2.0 * (n_fft_bins - 1) as f64))
        .collect();
    let mut filters = vec![vec![0.0_f32; n_mels]; n_fft_bins];
    for m in 0..n_mels {
        let lower = freq_pts[m];
        let center = freq_pts[m + 1];
        let upper = freq_pts[m + 2];
        let norm = 2.0 / (upper - lower).max(1e-12);
        let rise_denom = (center - lower).max(1e-12);
        let fall_denom = (upper - center).max(1e-12);
        for k in 0..n_fft_bins {
            let f = fft_freqs[k];
            let rising = (f - lower) / rise_denom;
            let falling = (upper - f) / fall_denom;
            let weight = rising.min(falling).max(0.0) * norm;
            filters[k][m] = weight as f32;
        }
    }
    filters
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "mel_tests.rs"]
mod tests;
