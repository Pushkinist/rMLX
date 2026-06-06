//! High-level transcription and translation API.
//!
//! Orchestrates: audio decode → mel extraction → Whisper encode → decode
//! → tokenizer decode → text output.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use rmlx_audio::transcript::{TranscribeOptions, transcribe};
//! use rmlx_mlx::Device;
//!
//! // Load the model and tokenizer once (expensive), cache them.
//! let opts = TranscribeOptions {
//!     model_path: "/path/to/whisper-large-v3-mlx".into(),
//!     tokenizer_path: "/path/to/whisper-tokenizer-dir".into(),
//!     language: "en".into(),
//!     temperature: 0.0,
//!     max_tokens: 224,
//!     ..Default::default()
//! };
//! let wav_bytes = std::fs::read("audio.wav").unwrap();
//! let result = transcribe(&wav_bytes, opts, Device::Gpu).unwrap();
//! println!("{}", result.text);
//! ```

use std::path::PathBuf;

use rmlx_mlx::Device;
use thiserror::Error;
use tracing::{debug, instrument};

use crate::mel::MelExtractor;
use crate::tokenizer::{WhisperTask, WhisperTokenizer};
use crate::vad::{SileroVad, VadState, voiced_segments};
use crate::wav::WavDecoder;
use crate::whisper::WhisperModel;

// ── Long-audio chunking constants ─────────────────────────────────────────────

/// Whisper encoder window in samples (30 s × 16 kHz).
const WHISPER_WINDOW_SAMPLES: usize = 30 * 16_000; // 480 000

/// Overlap between consecutive Whisper chunks (seconds × sample rate).
/// Prevents clipping words at chunk boundaries.
const CHUNK_OVERLAP_SAMPLES: usize = 16_000; // 1 s overlap

/// VAD probability threshold for voiced/silence decision.
const VAD_THRESHOLD: f32 = 0.5;
/// Minimum voiced frames for a segment to be kept.
const VAD_MIN_SPEECH_FRAMES: usize = 4; // ~320 ms
/// Minimum silence frames to split a segment.
const VAD_MIN_SILENCE_FRAMES: usize = 8; // ~640 ms

// ── Error ─────────────────────────────────────────────────────────────────────

/// Transcription errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum TranscriptError {
    /// WAV decode error.
    #[error("audio decode: {0}")]
    Audio(String),
    /// Mel extraction error.
    #[error("mel extraction: {0}")]
    Mel(String),
    /// Whisper model error.
    #[error("whisper: {0}")]
    Whisper(String),
    /// Tokenizer error.
    #[error("tokenizer: {0}")]
    Tokenizer(String),
    /// Resampling required but not implemented.
    #[error("audio must be at 16 kHz (got {actual_hz} Hz); resample before passing")]
    WrongSampleRate {
        /// Actual sample rate.
        actual_hz: u32,
    },
}

impl From<crate::wav::WavError> for TranscriptError {
    fn from(e: crate::wav::WavError) -> Self {
        Self::Audio(e.to_string())
    }
}

impl From<crate::mel::MelError> for TranscriptError {
    fn from(e: crate::mel::MelError) -> Self {
        Self::Mel(e.to_string())
    }
}

impl From<crate::whisper::WhisperError> for TranscriptError {
    fn from(e: crate::whisper::WhisperError) -> Self {
        Self::Whisper(e.to_string())
    }
}

impl From<crate::tokenizer::TokenizerError> for TranscriptError {
    fn from(e: crate::tokenizer::TokenizerError) -> Self {
        Self::Tokenizer(e.to_string())
    }
}

// ── Options ───────────────────────────────────────────────────────────────────

/// Options for `transcribe()` / `translate()`.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TranscribeOptions {
    /// Path to the Whisper snapshot directory (contains `config.json` + `weights.npz`).
    pub model_path: PathBuf,
    /// Path to a directory containing `tokenizer.json` (e.g. `openai/whisper-large-v3`).
    pub tokenizer_path: PathBuf,
    /// Audio language code (e.g. `"en"`, `"fr"`). Used for the decoder SOT sequence.
    pub language: String,
    /// Decoding temperature (0 = greedy argmax).
    pub temperature: f32,
    /// Maximum tokens to generate (Whisper default = 224).
    pub max_tokens: usize,
}

impl Default for TranscribeOptions {
    fn default() -> Self {
        Self {
            model_path: PathBuf::new(),
            tokenizer_path: PathBuf::new(),
            language: "en".to_owned(),
            temperature: 0.0,
            max_tokens: 224,
        }
    }
}

// ── Result ────────────────────────────────────────────────────────────────────

/// Transcription output.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub struct TranscriptResult {
    /// Decoded text.
    pub text: String,
    /// Raw token IDs (excluding SOT prefix and EOT).
    pub tokens: Vec<u32>,
    /// Task that was run.
    pub task: WhisperTask,
}

// ── Transcribe ────────────────────────────────────────────────────────────────

/// Transcribe audio bytes to text using Whisper.
///
/// Loads the model and tokenizer on each call. For a server context, callers
/// should cache the loaded model and tokenizer and call the lower-level APIs
/// directly (see `WhisperModel`, `MelExtractor`, `WhisperTokenizer`).
#[instrument(skip(audio_bytes, opts), fields(audio_len = audio_bytes.len()), level = "debug")]
pub fn transcribe(
    audio_bytes: &[u8],
    opts: TranscribeOptions,
    device: Device,
) -> Result<TranscriptResult, TranscriptError> {
    run(audio_bytes, opts, WhisperTask::Transcribe, device)
}

/// Translate audio bytes to English text using Whisper.
///
/// Same as `transcribe()` but uses the `translate` task token so the output
/// is always in English regardless of the audio language.
#[instrument(skip(audio_bytes, opts), fields(audio_len = audio_bytes.len()), level = "debug")]
pub fn translate(
    audio_bytes: &[u8],
    opts: TranscribeOptions,
    device: Device,
) -> Result<TranscriptResult, TranscriptError> {
    run(audio_bytes, opts, WhisperTask::Translate, device)
}

#[allow(
    clippy::needless_pass_by_value,
    reason = "public transcribe/translate take opts by value for ergonomics; run is private and mirrors that signature"
)]
fn run(
    audio_bytes: &[u8],
    opts: TranscribeOptions,
    task: WhisperTask,
    device: Device,
) -> Result<TranscriptResult, TranscriptError> {
    // 1. Decode audio.
    let (samples, sample_rate) = WavDecoder::decode(audio_bytes)?;
    if sample_rate != crate::mel::SAMPLE_RATE {
        return Err(TranscriptError::WrongSampleRate {
            actual_hz: sample_rate,
        });
    }

    // 2. Load model and tokenizer once (shared across all chunks).
    let model = WhisperModel::load(&opts.model_path)?;
    let tokenizer = WhisperTokenizer::from_path(&opts.tokenizer_path)
        .map_err(|e| TranscriptError::Tokenizer(e.to_string()))?;

    // 3. For short audio (<= 30 s), use the direct path.
    //    For long audio (> 30 s), use VAD-guided chunking.
    let all_tokens: Vec<u32> = if samples.len() <= WHISPER_WINDOW_SAMPLES {
        transcribe_chunk(&samples, &model, &tokenizer, &opts, task, device)?
    } else {
        transcribe_long(samples, &model, &tokenizer, &opts, task, device)?
    };

    // 4. Decode tokens to text.
    let text = tokenizer
        .decode(&all_tokens)
        .map_err(|e| TranscriptError::Tokenizer(e.to_string()))?;

    Ok(TranscriptResult {
        text,
        tokens: all_tokens,
        task,
    })
}

/// Transcribe a single chunk (samples must fit in Whisper's 30 s window).
fn transcribe_chunk(
    samples: &[f32],
    model: &WhisperModel,
    tokenizer: &WhisperTokenizer,
    opts: &TranscribeOptions,
    task: WhisperTask,
    device: Device,
) -> Result<Vec<u32>, TranscriptError> {
    let extractor = MelExtractor::new(128).map_err(|e| TranscriptError::Mel(e.to_string()))?;
    let mel_frames = extractor.extract(samples)?;
    let encoder_out = model.encode_mel(&mel_frames, device)?;
    let sot = tokenizer.sot_sequence(&opts.language, task, false);
    let tokens = model
        .greedy_decode(&encoder_out, &sot, opts.max_tokens, opts.temperature, device)
        .map_err(TranscriptError::from)?;
    Ok(tokens)
}

/// Long-audio transcription via Silero VAD chunking.
///
/// ## Strategy
///
/// 1. Run Silero VAD on the full audio to get per-frame voice probabilities.
/// 2. Extract voiced segments (threshold = 0.5, minimum speech/silence gap).
/// 3. Build window-aligned chunks (max 30 s each) from the voiced segments,
///    plus 1 s overlap at each boundary to avoid clipping words.
/// 4. Transcribe each chunk independently; concatenate tokens.
///
/// If VAD fails to load (unlikely — weights are embedded), fall back to
/// naive sliding-window chunking without VAD guidance.
fn transcribe_long(
    samples: Vec<f32>,
    model: &WhisperModel,
    tokenizer: &WhisperTokenizer,
    opts: &TranscribeOptions,
    task: WhisperTask,
    device: Device,
) -> Result<Vec<u32>, TranscriptError> {
    debug!(
        n_samples = samples.len(),
        duration_s = samples.len() as f32 / 16_000.0,
        "long audio: using VAD-guided chunking"
    );

    // Attempt VAD-guided segmentation; fall back on any VAD error.
    let chunks: Vec<(usize, usize)> = match build_vad_chunks(&samples, device) {
        Ok(c) => {
            debug!(n_chunks = c.len(), "VAD chunks built");
            c
        }
        Err(e) => {
            debug!(error = %e, "VAD failed; using sliding window fallback");
            sliding_window_chunks(samples.len())
        }
    };

    let mut all_tokens: Vec<u32> = Vec::new();
    let extractor = MelExtractor::new(128).map_err(|e| TranscriptError::Mel(e.to_string()))?;

    for (chunk_idx, (start, end)) in chunks.iter().enumerate() {
        let chunk = &samples[*start..(*end).min(samples.len())];
        if chunk.is_empty() {
            continue;
        }
        debug!(
            chunk = chunk_idx,
            start_s = *start as f32 / 16_000.0,
            end_s = *end as f32 / 16_000.0,
            n_samples = chunk.len(),
            "transcribing chunk"
        );

        let mel_frames = extractor.extract(chunk)?;
        let encoder_out = model.encode_mel(&mel_frames, device)?;
        let sot = tokenizer.sot_sequence(&opts.language, task, false);
        let tokens = match model.greedy_decode(
            &encoder_out,
            &sot,
            opts.max_tokens,
            opts.temperature,
            device,
        ) {
            Ok(t) => t,
            Err(crate::whisper::WhisperError::Silence) => {
                debug!(chunk = chunk_idx, "chunk detected as silence; skipping");
                continue;
            }
            Err(e) => return Err(TranscriptError::from(e)),
        };
        all_tokens.extend_from_slice(&tokens);
    }

    Ok(all_tokens)
}

/// Build VAD-guided chunks (start_sample, end_sample) from the full audio.
fn build_vad_chunks(
    samples: &[f32],
    device: Device,
) -> Result<Vec<(usize, usize)>, String> {
    let vad = SileroVad::load(device).map_err(|e| e.to_string())?;
    let state = VadState::new_zeroed(device).map_err(|e| e.to_string())?;
    let (probs, _) = vad.forward(samples, state, device).map_err(|e| e.to_string())?;

    let segs = voiced_segments(
        &probs,
        VAD_THRESHOLD,
        VAD_MIN_SPEECH_FRAMES,
        VAD_MIN_SILENCE_FRAMES,
    );

    if segs.is_empty() {
        // No voiced segments found; treat entire audio as one window sequence.
        return Ok(sliding_window_chunks(samples.len()));
    }

    // Merge voiced segments into Whisper-window-sized chunks with overlap.
    let mut chunks: Vec<(usize, usize)> = Vec::new();
    let mut chunk_start = segs[0].0.saturating_sub(CHUNK_OVERLAP_SAMPLES);
    let mut chunk_end = segs[0].1 + CHUNK_OVERLAP_SAMPLES;

    for &(seg_start, seg_end) in &segs[1..] {
        let proposed_end = seg_end + CHUNK_OVERLAP_SAMPLES;
        if proposed_end - chunk_start > WHISPER_WINDOW_SAMPLES {
            // Current chunk is full; close it and start a new one.
            chunks.push((chunk_start, chunk_end.min(samples.len())));
            chunk_start = seg_start.saturating_sub(CHUNK_OVERLAP_SAMPLES);
            chunk_end = proposed_end;
        } else {
            chunk_end = proposed_end;
        }
    }
    // Flush last chunk.
    if chunk_end > chunk_start {
        chunks.push((chunk_start, chunk_end.min(samples.len())));
    }

    Ok(chunks)
}

/// Naive sliding-window chunking fallback (no VAD).
fn sliding_window_chunks(total_samples: usize) -> Vec<(usize, usize)> {
    let step = WHISPER_WINDOW_SAMPLES.saturating_sub(CHUNK_OVERLAP_SAMPLES);
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < total_samples {
        let end = (start + WHISPER_WINDOW_SAMPLES).min(total_samples);
        chunks.push((start, end));
        if end == total_samples {
            break;
        }
        start += step;
    }
    chunks
}
