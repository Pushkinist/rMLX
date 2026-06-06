//! rMLX audio subsystem: Whisper STT, Silero VAD, WAV I/O, mel-spectrogram.
//!
//! ## Crate layout
//!
//! - `wav`       — WAV/audio decoder (Symphonia-backed; returns mono f32 at native
//!   sample rate; callers MUST validate `sample_rate == 16_000`).
//!   Also: WAV encoder (PCM-16 LE mono).
//! - `mel`       — Whisper log-mel spectrogram (STFT + triangular mel filterbank).
//! - `tokenizer` — Whisper BPE tokenizer (GPT-2 vocabulary + special tokens).
//! - `whisper`   — Whisper encoder + decoder model layers and inference loop.
//! - `vad`       — Silero VAD v4 (16kHz LSTM-based voice activity detection).
//!   Used internally for long-audio chunking in the Whisper pipeline.
//!   Weights vendored in `assets/silero_vad_16k.safetensors` (MIT license).
//!
//! ## Design notes
//!
//! The crate boundary is justified because:
//! 1. Audio processing is a distinct subsystem with its own data types (PCM
//!    samples, mel frames, Whisper token sequences) that do not appear in the
//!    LLM text-generation path.
//! 2. The Whisper model lifecycle (load from `.npz`, encode, decode loop) is
//!    architecturally orthogonal to the chat-LLM `Generator` trait.
//! 3. Audio models (TTS, VAD) share the same WAV I/O and mel primitives.
//!
//! - `npz`       — ZIP/NPZ central-directory parser with ZIP64 support.
//!   Used by `whisper` to load `weights.npz` and by `tts` to load codec weights.
//! - `tts`       — Qwen3-TTS synthesis pipeline (Phase 4b).
//!
//! ## Out of scope (v1)
//!
//! - Diarization.
//! - Speech-to-speech.

#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::float_cmp,
        clippy::disallowed_methods,
        clippy::approx_constant,
        clippy::manual_div_ceil,
    )
)]

pub mod mel;
pub mod npz;
pub mod tokenizer;
pub mod tts;
pub mod vad;
pub mod wav;
pub mod whisper;

pub use vad::{voiced_segments, SileroVad, VadState};
pub use wav::{WavDecoder, WavEncoder};
