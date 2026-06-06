//! Gemma 4 mel-spectrogram CPU feature extractor.
//!
//! Pure Rust port of `mlx_vlm.models.gemma4.audio_feature_extractor`
//! (`Gemma4AudioFeatureExtractor`). No MLX ops — this is the host-side
//! preprocessing stage that runs before the audio encoder.
//!
//! ## Pipeline (matches upstream exactly)
//!
//! 1. Optional input scaling and dithering (disabled by default).
//! 2. Semicausal left-pad: prepend `frame_length // 2` zeros so the first
//!    frame is centred at t = 0 (matches HuggingFace Transformers behaviour).
//! 3. Frame the signal using a stride of `hop_length` and a frame window of
//!    `frame_length + 1` samples (the extra sample drives the HTK preemphasis
//!    difference equation and is dropped afterwards).
//! 4. HTK-flavour preemphasis (when `preemphasis > 0`):
//!    - first sample → `frame[0] * (1 - α)`
//!    - rest → `frame[1..N-1] - α * frame[0..N-2]`
//!
//!    When `preemphasis == 0` simply slice off the last sentinel sample.
//! 5. Multiply each frame by a **periodic** Hann window:
//!    `w[n] = 0.5 − 0.5 · cos(2π·n / frame_length)`.
//! 6. Real FFT of length `fft_length = 2^⌈log₂(frame_length)⌉`.
//! 7. Magnitude spectrum `|RFFT|`.
//! 8. Triangular mel filterbank (HTK mel scale, no Slaney normalisation):
//!    applied as matrix-multiply `[T, fft_length/2+1] × [fft_bins, num_mel]`.
//! 9. Log compression: `log(mel_spec + mel_floor)`.
//! 10. Optional per-bin mean / stddev normalisation.
//!
//! ## Output layout
//!
//! `extract()` returns `Vec<Vec<f32>>` of shape `[T_frames][feature_size]`
//! where `T_frames = (padded_length - (frame_length + 1)) / hop_length + 1`
//! and `padded_length = n_samples + frame_length // 2`.
//!
//! This is row-major frames-first order: each inner `Vec` is one mel frame of
//! length `feature_size`. The downstream audio encoder consumes it as
//! `[T_frames, feature_size]`.
//!
//! ## Mel scale
//!
//! HTK formula (as in the Python reference — not Slaney):
//! `hz_to_mel(f) = 2595 · log₁₀(1 + f / 700)`
//! `mel_to_hz(m) = 700 · (10^(m / 2595) − 1)`
//!
//! ## Config
//!
//! Parameters are loaded from the model's `processor_config.json`
//! `feature_extractor` block. The defaults below match the
//! `mlx-community__gemma-4-e4b-it-mxfp8` snapshot:
//!
//! | Field | Default | Derived |
//! |------------------|---------|---------|
//! | `sampling_rate` | 16 000 | |
//! | `num_mel_filters`| 128 | → `feature_size` |
//! | `fft_length` | 512 | verified `= 2^⌈log₂(frame_length)⌉` |
//! | `hop_length` | 160 | = round(16000 × 10 ms) |
//! | `frame_length` | 320 | = round(16000 × 20 ms) |
//! | `min_frequency` | 0.0 | |
//! | `max_frequency` | 8000.0 | |
//! | `preemphasis` | 0.0 | |
//! | `mel_floor` | 1e-3 | |

use std::f32::consts::PI;
use std::sync::Arc;

use rustfft::{num_complex::Complex, FftPlanner};
use serde::Deserialize;
use thiserror::Error;
use tracing::{debug, instrument};

// ── Error ────────────────────────────────────────────────────────────────────

/// Errors produced by [`Gemma4AudioFeatureExtractor`].
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum AudioFeatError {
    /// Config JSON could not be parsed.
    #[error("audio feature extractor config error: {0}")]
    Config(String),
    /// Input waveform is empty or shorter than one frame.
    #[error("audio input too short: {n_samples} samples, need at least {min_samples}")]
    TooShort {
        /// Number of samples in the provided input.
        n_samples: usize,
        /// Minimum number of samples required for one frame.
        min_samples: usize,
    },
}

// ── Config serde shape ───────────────────────────────────────────────────────

/// Subset of `processor_config.json` → `feature_extractor` block.
///
/// Fields absent in the JSON file fall back to the Python class defaults.
#[non_exhaustive]
#[derive(Debug, Deserialize)]
pub struct AudioFeatureExtractorConfig {
    /// Number of mel filter bins (= `feature_size` in the Python class).
    #[serde(default = "default_num_mel_filters", alias = "feature_size")]
    pub num_mel_filters: usize,

    /// Target sample rate in Hz.
    #[serde(default = "default_sampling_rate")]
    pub sampling_rate: u32,

    /// Frame length in milliseconds. Overridden by `fft_length` / `hop_length`
    /// when those are present and imply a different value.
    #[serde(default = "default_frame_length_ms")]
    pub frame_length_ms: f64,

    /// Hop (stride) in milliseconds.
    #[serde(default = "default_hop_length_ms")]
    pub hop_length_ms: f64,

    /// If present, the integer hop length in *samples* (overrides `hop_length_ms`).
    #[serde(default)]
    pub hop_length: Option<usize>,

    /// If present, the integer FFT length in *samples* (overrides the power-of-2
    /// rounding of `frame_length_ms`).
    #[serde(default)]
    pub fft_length: Option<usize>,

    /// Lower frequency bound for the mel filterbank (Hz).
    #[serde(default = "default_min_frequency")]
    pub min_frequency: f64,

    /// Upper frequency bound for the mel filterbank (Hz).
    #[serde(default = "default_max_frequency")]
    pub max_frequency: f64,

    /// HTK preemphasis coefficient α. 0.0 disables preemphasis.
    #[serde(default)]
    pub preemphasis: f64,

    /// Use HTK-flavour preemphasis (first sample scaled, rest differenced).
    #[serde(default = "default_preemphasis_htk")]
    pub preemphasis_htk_flavor: bool,

    /// Whether to double the FFT length (fft_overdrive).
    #[serde(default)]
    pub fft_overdrive: bool,

    /// Log-compression floor value.
    #[serde(default = "default_mel_floor")]
    pub mel_floor: f64,

    /// Input scale factor applied before framing.
    #[serde(default = "default_input_scale_factor")]
    pub input_scale_factor: f64,

    /// Per-bin mean (optional normalisation, length = num_mel_filters).
    #[serde(default)]
    pub per_bin_mean: Option<Vec<f32>>,

    /// Per-bin stddev (optional normalisation, length = num_mel_filters).
    #[serde(default)]
    pub per_bin_stddev: Option<Vec<f32>>,
}

fn default_num_mel_filters() -> usize {
    128
}
fn default_sampling_rate() -> u32 {
    16_000
}
fn default_frame_length_ms() -> f64 {
    20.0
}
fn default_hop_length_ms() -> f64 {
    10.0
}
fn default_min_frequency() -> f64 {
    0.0
}
fn default_max_frequency() -> f64 {
    8000.0
}
fn default_preemphasis_htk() -> bool {
    true
}
fn default_mel_floor() -> f64 {
    1e-3
}
fn default_input_scale_factor() -> f64 {
    1.0
}

// ── Feature extractor ────────────────────────────────────────────────────────

/// CPU log-mel feature extractor for Gemma 4 audio.
///
/// Construct via [`Gemma4AudioFeatureExtractor::new`] (with explicit params)
/// or [`Gemma4AudioFeatureExtractor::from_config`] (from JSON config).
///
/// Call [`Gemma4AudioFeatureExtractor::extract`] to process a mono f32
/// waveform into a `[T_frames][feature_size]` log-mel spectrogram.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed feature-extractor — private implementation fields; public API is new(), from_config(), and extract(); adding a field requires updating new() and from_config()"
)]
pub struct Gemma4AudioFeatureExtractor {
    /// Number of mel filter bins.
    pub feature_size: usize,

    /// Expected sample rate in Hz (for informational / resampling checks).
    pub sampling_rate: u32,

    /// Frame length in samples.
    frame_length: usize,

    /// Hop length (stride) in samples.
    hop_length: usize,

    /// FFT length (≥ frame_length, power of 2).
    fft_length: usize,

    /// HTK preemphasis coefficient (0 = off).
    preemphasis: f64,

    /// Use HTK-flavour preemphasis (first sample scaled, rest differenced).
    preemphasis_htk_flavor: bool,

    /// Log-compression floor (added before log to avoid log(0)).
    mel_floor: f32,

    /// Input scale factor applied to raw samples.
    input_scale_factor: f32,

    /// Periodic Hann window of length `frame_length`.
    window: Vec<f32>,

    /// Mel filterbank matrix, shape `[fft_length / 2 + 1][feature_size]`.
    ///
    /// Row = FFT bin, col = mel bin. Applied as:
    /// `mel_spec[t][m] = Σ_k magnitude[t][k] · mel_filters[k][m]`
    mel_filters: Vec<Vec<f32>>,

    /// Optional per-bin mean for normalisation, length = feature_size.
    per_bin_mean: Option<Vec<f32>>,

    /// Optional per-bin stddev for normalisation, length = feature_size.
    per_bin_stddev: Option<Vec<f32>>,

    /// Cached FFT plan for `fft_length`.
    fft: Arc<dyn rustfft::Fft<f32>>,
}

// `Arc<dyn Fft<f32>>` does not implement `Debug`, so we provide a manual impl.
impl std::fmt::Debug for Gemma4AudioFeatureExtractor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gemma4AudioFeatureExtractor")
            .field("feature_size", &self.feature_size)
            .field("sampling_rate", &self.sampling_rate)
            .field("frame_length", &self.frame_length)
            .field("hop_length", &self.hop_length)
            .field("fft_length", &self.fft_length)
            .field("preemphasis", &self.preemphasis)
            .field("mel_floor", &self.mel_floor)
            .finish_non_exhaustive()
    }
}

impl Gemma4AudioFeatureExtractor {
    /// Construct from explicit parameters (mirrors the Python `__init__`).
    ///
    /// # Parameters
    ///
    /// - `feature_size`: number of mel bins.
    /// - `sampling_rate`: expected audio sample rate in Hz.
    /// - `frame_length_ms`: frame duration in milliseconds.
    /// - `hop_length_ms`: hop duration in milliseconds.
    /// - `min_frequency`: lower bound of mel filterbank in Hz.
    /// - `max_frequency`: upper bound of mel filterbank in Hz.
    /// - `preemphasis`: HTK preemphasis coefficient (0 = off).
    /// - `preemphasis_htk_flavor`: if true, use HTK-style first-sample scaling.
    /// - `fft_overdrive`: if true, double the FFT length.
    /// - `mel_floor`: log-compression floor.
    /// - `input_scale_factor`: multiply samples by this before framing.
    /// - `per_bin_mean` / `per_bin_stddev`: optional per-bin normalisation.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        feature_size: usize,
        sampling_rate: u32,
        frame_length_ms: f64,
        hop_length_ms: f64,
        min_frequency: f64,
        max_frequency: f64,
        preemphasis: f64,
        preemphasis_htk_flavor: bool,
        fft_overdrive: bool,
        mel_floor: f64,
        input_scale_factor: f64,
        per_bin_mean: Option<Vec<f32>>,
        per_bin_stddev: Option<Vec<f32>>,
    ) -> Self {
        let frame_length = (f64::from(sampling_rate) * frame_length_ms / 1000.0).round() as usize;
        let hop_length = (f64::from(sampling_rate) * hop_length_ms / 1000.0).round() as usize;

        // Power-of-2 FFT length ≥ frame_length.
        let mut fft_length = frame_length.next_power_of_two();
        if fft_overdrive {
            fft_length *= 2;
        }

        Self::build(
            feature_size,
            sampling_rate,
            frame_length,
            hop_length,
            fft_length,
            min_frequency,
            max_frequency,
            preemphasis,
            preemphasis_htk_flavor,
            mel_floor,
            input_scale_factor,
            per_bin_mean,
            per_bin_stddev,
        )
    }

    /// Construct from a parsed [`AudioFeatureExtractorConfig`].
    pub fn from_config(cfg: AudioFeatureExtractorConfig) -> Self {
        let frame_length =
            (f64::from(cfg.sampling_rate) * cfg.frame_length_ms / 1000.0).round() as usize;
        let hop_length = cfg.hop_length.unwrap_or_else(|| {
            (f64::from(cfg.sampling_rate) * cfg.hop_length_ms / 1000.0).round() as usize
        });

        let mut fft_length = cfg
            .fft_length
            .unwrap_or_else(|| frame_length.next_power_of_two());
        if cfg.fft_overdrive {
            fft_length *= 2;
        }

        Self::build(
            cfg.num_mel_filters,
            cfg.sampling_rate,
            frame_length,
            hop_length,
            fft_length,
            cfg.min_frequency,
            cfg.max_frequency,
            cfg.preemphasis,
            cfg.preemphasis_htk_flavor,
            cfg.mel_floor,
            cfg.input_scale_factor,
            cfg.per_bin_mean,
            cfg.per_bin_stddev,
        )
    }

    /// Parse a `processor_config.json` string (or the `feature_extractor`
    /// sub-object as JSON) and construct the extractor.
    ///
    /// Accepts two shapes:
    /// - The full `processor_config.json` (outer object with a
    ///   `"feature_extractor"` key).
    /// - The `feature_extractor` block directly.
    pub fn from_processor_config_str(json: &str) -> Result<Self, AudioFeatError> {
        // Try outer wrapper first.
        let cfg: AudioFeatureExtractorConfig =
            if let Ok(outer) = serde_json::from_str::<serde_json::Value>(json) {
                if let Some(fe) = outer.get("feature_extractor") {
                    serde_json::from_value(fe.clone())
                        .map_err(|e| AudioFeatError::Config(e.to_string()))?
                } else {
                    serde_json::from_str(json).map_err(|e| AudioFeatError::Config(e.to_string()))?
                }
            } else {
                return Err(AudioFeatError::Config("invalid JSON".into()));
            };

        Ok(Self::from_config(cfg))
    }

    // ── Internal builder ─────────────────────────────────────────────────────

    #[allow(clippy::too_many_arguments)]
    fn build(
        feature_size: usize,
        sampling_rate: u32,
        frame_length: usize,
        hop_length: usize,
        fft_length: usize,
        min_frequency: f64,
        max_frequency: f64,
        preemphasis: f64,
        preemphasis_htk_flavor: bool,
        mel_floor: f64,
        input_scale_factor: f64,
        per_bin_mean: Option<Vec<f32>>,
        per_bin_stddev: Option<Vec<f32>>,
    ) -> Self {
        // Periodic Hann window: w[n] = 0.5 - 0.5 * cos(2π * n / frame_length).
        // This is `torch.hann_window(periodic=True)` / `np.hann` style used by
        // HuggingFace Transformers.
        let window: Vec<f32> = (0..frame_length)
            .map(|n| {
                let arg = 2.0 * PI * n as f32 / frame_length as f32;
                0.5f32.mul_add(-arg.cos(), 0.5)
            })
            .collect();

        let num_fft_bins = fft_length / 2 + 1;
        let mel_filters = build_mel_filterbank(
            num_fft_bins,
            feature_size,
            min_frequency,
            max_frequency,
            sampling_rate,
        );

        // Pre-build the RFFT plan. rustfft works on complex arrays even for
        // real FFTs; we zero-pad and use the forward complex FFT, then take
        // the first `fft_length/2 + 1` bins.
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(fft_length);

        debug!(
            feature_size,
            sampling_rate,
            frame_length,
            hop_length,
            fft_length,
            num_fft_bins,
            "Gemma4AudioFeatureExtractor built"
        );

        Gemma4AudioFeatureExtractor {
            feature_size,
            sampling_rate,
            frame_length,
            hop_length,
            fft_length,
            preemphasis,
            preemphasis_htk_flavor,
            mel_floor: mel_floor as f32,
            input_scale_factor: input_scale_factor as f32,
            window,
            mel_filters,
            per_bin_mean,
            per_bin_stddev,
            fft,
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Extract a log-mel spectrogram from a mono f32 waveform.
    ///
    /// `samples` should be at `self.sampling_rate` Hz, values in `[-1, 1]`.
    ///
    /// Returns `Ok(frames)` where `frames[t]` is the mel frame at time step `t`,
    /// length `self.feature_size`. The returned shape is `[T_frames][feature_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`AudioFeatError::TooShort`] if `samples` is shorter than one
    /// frame after padding.
    #[instrument(skip(self, samples), fields(n_samples = samples.len()), level = "debug")]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub fn extract(&self, samples: &[f32]) -> Result<Vec<Vec<f32>>, AudioFeatError> {
        let n_samples = samples.len();

        // Optional input scaling.
        let samples_owned: Vec<f32> = if (self.input_scale_factor - 1.0).abs() > f32::EPSILON {
            samples
                .iter()
                .map(|&s| s * self.input_scale_factor)
                .collect()
        } else {
            samples.to_vec()
        };

        // Semicausal left-pad: prepend `frame_length // 2` zeros so the first
        // frame is centred at sample 0 (matches HuggingFace Transformers).
        let pad_left = self.frame_length / 2;
        let mut padded = Vec::with_capacity(pad_left + n_samples);
        padded.resize(pad_left, 0.0_f32);
        padded.extend_from_slice(&samples_owned);

        let padded_len = padded.len();

        // Frame size for the unfold step includes one extra sentinel sample
        // that drives the preemphasis difference equation.
        let frame_size_for_unfold = self.frame_length + 1;

        if padded_len < frame_size_for_unfold {
            return Err(AudioFeatError::TooShort {
                n_samples,
                min_samples: frame_size_for_unfold - pad_left,
            });
        }

        // Number of frames: (padded_len - frame_size_for_unfold) / hop_length + 1
        let num_frames = (padded_len - frame_size_for_unfold) / self.hop_length + 1;

        let num_fft_bins = self.fft_length / 2 + 1;

        // Scratch buffer for the complex FFT (reused across frames).
        let mut fft_buf: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); self.fft_length];

        let mut output: Vec<Vec<f32>> = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * self.hop_length;
            // Safety: `num_frames` is computed so this never overruns `padded`.
            let raw_frame = &padded[start..start + frame_size_for_unfold];

            // Step 3 & 4: preemphasis then window.
            // Build the `frame_length`-sample analysis frame.
            let frame: Vec<f32> = if self.preemphasis > 0.0 {
                if self.preemphasis_htk_flavor {
                    // HTK flavour: first sample scaled, rest differenced.
                    let alpha = self.preemphasis as f32;
                    let mut f = Vec::with_capacity(self.frame_length);
                    f.push(raw_frame[0] * (1.0 - alpha));
                    for i in 1..self.frame_length {
                        f.push(alpha.mul_add(-raw_frame[i - 1], raw_frame[i]));
                    }
                    f
                } else {
                    // Standard causal preemphasis: x[n] - α·x[n-1].
                    let alpha = self.preemphasis as f32;
                    (1..=self.frame_length)
                        .map(|i| alpha.mul_add(-raw_frame[i - 1], raw_frame[i]))
                        .collect()
                }
            } else {
                // No preemphasis — just drop the sentinel last sample.
                raw_frame[..self.frame_length].to_vec()
            };

            // Apply Hann window.
            // Fill FFT buffer: windowed samples followed by zero-padding.
            for i in 0..self.fft_length {
                fft_buf[i] = if i < self.frame_length {
                    Complex::new(frame[i] * self.window[i], 0.0)
                } else {
                    Complex::new(0.0, 0.0)
                };
            }

            // In-place complex FFT (forward).
            self.fft.process(&mut fft_buf);

            // Magnitude spectrum: take first `fft_length/2 + 1` bins.
            let magnitudes: Vec<f32> = fft_buf[..num_fft_bins].iter().map(|c| c.norm()).collect();

            // Apply mel filterbank: mel_spec[m] = Σ_k magnitudes[k] * mel_filters[k][m].
            let mut mel_frame = vec![0.0_f32; self.feature_size];
            for (k, &mag) in magnitudes.iter().enumerate() {
                let filter_row = &self.mel_filters[k];
                for m in 0..self.feature_size {
                    mel_frame[m] += mag * filter_row[m];
                }
            }

            // Log compression: log(mel + mel_floor).
            for v in &mut mel_frame {
                *v = (*v + self.mel_floor).ln();
            }

            // Optional per-bin mean / stddev normalisation.
            if let Some(mean) = &self.per_bin_mean {
                for m in 0..self.feature_size {
                    mel_frame[m] -= mean[m];
                }
            }
            if let Some(stddev) = &self.per_bin_stddev {
                for m in 0..self.feature_size {
                    mel_frame[m] /= stddev[m];
                }
            }

            output.push(mel_frame);
        }

        debug!(num_frames, feature_size = self.feature_size, "extract done");
        Ok(output)
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Number of mel frames produced from `n_samples` of audio (after padding).
    pub fn num_frames(&self, n_samples: usize) -> usize {
        let padded = n_samples + self.frame_length / 2;
        let frame_size_for_unfold = self.frame_length + 1;
        if padded < frame_size_for_unfold {
            return 0;
        }
        (padded - frame_size_for_unfold) / self.hop_length + 1
    }

    /// The frame length in samples.
    pub fn frame_length(&self) -> usize {
        self.frame_length
    }

    /// The hop length in samples.
    pub fn hop_length(&self) -> usize {
        self.hop_length
    }

    /// The FFT length in samples.
    pub fn fft_length(&self) -> usize {
        self.fft_length
    }
}

// ── Mel filterbank ───────────────────────────────────────────────────────────

/// Build a triangular mel filterbank using the **HTK mel scale**.
///
/// Returns a matrix of shape `[num_fft_bins][num_mel_filters]`.
///
/// ## HTK mel scale
///
/// ```text
/// hz_to_mel(f) = 2595 · log₁₀(1 + f / 700)
/// mel_to_hz(m) = 700 · (10^(m / 2595) − 1)
/// ```
///
/// ## Filter construction
///
/// `num_mel_filters + 2` equally-spaced mel points are mapped back to Hz,
/// giving centre frequencies for each triangular filter plus the flanking
/// boundaries. For filter `i` (0-indexed):
/// - lower = freq_points[i]
/// - centre = freq_points[i+1]
/// - upper = freq_points[i+2]
///
/// The weight for FFT bin `k` (frequency `f_k`) is:
/// - rising slope : `(f_k − lower) / (centre − lower)`
/// - falling slope: `(upper − f_k) / (upper − centre)`
/// - clipped to `[0, 1]` and the minimum of rising/falling is taken.
///
/// No Slaney area-normalisation is applied (`norm = None` in the Python).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn build_mel_filterbank(
    num_fft_bins: usize,
    num_mel_filters: usize,
    min_frequency: f64,
    max_frequency: f64,
    sampling_rate: u32,
) -> Vec<Vec<f32>> {
    let mel_min = hz_to_mel_htk(min_frequency);
    let mel_max = hz_to_mel_htk(max_frequency);

    // `num_mel_filters + 2` mel-equally-spaced points mapped to Hz.
    let n_points = num_mel_filters + 2;
    let freq_points: Vec<f64> = (0..n_points)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64;
            mel_to_hz_htk(mel)
        })
        .collect();

    // FFT bin centre frequencies.
    // Bin k → freq = k * (sampling_rate / (2 * (num_fft_bins - 1))).
    let bin_to_hz = f64::from(sampling_rate) / (2.0 * (num_fft_bins - 1) as f64);
    let fft_freqs: Vec<f64> = (0..num_fft_bins).map(|k| k as f64 * bin_to_hz).collect();

    // Build [num_fft_bins][num_mel_filters] matrix.
    let mut filters = vec![vec![0.0_f32; num_mel_filters]; num_fft_bins];

    for i in 0..num_mel_filters {
        let lower = freq_points[i];
        let center = freq_points[i + 1];
        let upper = freq_points[i + 2];

        let rise_denom = (center - lower).max(1e-10);
        let fall_denom = (upper - center).max(1e-10);

        for k in 0..num_fft_bins {
            let f = fft_freqs[k];
            let rising = (f - lower) / rise_denom;
            let falling = (upper - f) / fall_denom;
            let weight = rising.min(falling).max(0.0);
            filters[k][i] = weight as f32;
        }
    }

    filters
}

/// HTK mel scale: hz → mel.
#[inline]
fn hz_to_mel_htk(hz: f64) -> f64 {
    2595.0 * (1.0 + hz / 700.0).log10()
}

/// HTK mel scale: mel → hz.
#[inline]
fn mel_to_hz_htk(mel: f64) -> f64 {
    700.0 * (10.0_f64.powf(mel / 2595.0) - 1.0)
}

// ── Unit tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "audio_feature_extractor_tests.rs"]
mod tests;
