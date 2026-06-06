//! Audio encoder/decoder for the Whisper STT + TTS paths.
//!
//! ## Decode
//!
//! [`WavDecoder`] wraps `symphonia` and decodes **any Symphonia-supported
//! container** (WAV, MP3, FLAC, OGG, AAC, …). The output is always:
//!
//! - Mono f32 (multi-channel inputs are averaged to mono).
//! - Native sample rate of the file — **callers MUST validate that
//!   `sample_rate == 16_000`** before using the samples for Whisper; the
//!   decoder does not resample.
//!
//! ## Encode
//!
//! [`WavEncoder`] writes a standard 44-byte RIFF PCM-16 header followed by
//! raw little-endian s16 samples.
//!
//! ## Why hand-rolled encode instead of `hound`?
//!
//! The workspace already depends on `symphonia` for decode. Adding `hound` for
//! simple PCM-16 encode would be a redundant dep. The 44-byte RIFF header is
//! trivially correct to hand-roll.

use std::io::Cursor;

use symphonia::core::{
    audio::SampleBuffer,
    codecs::DecoderOptions,
    errors::Error as SymphoniaError,
    formats::FormatOptions,
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    probe::Hint,
};
use thiserror::Error;
use tracing::{debug, instrument};

// ── Errors ────────────────────────────────────────────────────────────────────

/// WAV encode / decode errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum WavError {
    /// Symphonia probe / format error.
    #[error("WAV probe failed: {0}")]
    Probe(String),
    /// No audio track found in the container.
    #[error("no audio track in WAV")]
    NoTrack,
    /// Codec error during decode.
    #[error("WAV decode error: {0}")]
    Decode(String),
    /// Zero samples decoded.
    #[error("WAV stream contained zero samples")]
    Empty,
    /// Invalid parameters passed to encoder.
    #[error("WAV encode error: {0}")]
    Encode(String),
}

// ── WavDecoder ────────────────────────────────────────────────────────────────

/// Decode WAV bytes into mono f32 samples.
///
/// Samples are mixed to mono (channel average) and normalised to `[-1.0, 1.0]`.
/// Sample rate is returned alongside the samples.
#[non_exhaustive]
pub struct WavDecoder;

impl std::fmt::Debug for WavDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WavDecoder").finish()
    }
}

impl WavDecoder {
    /// Decode `bytes` (WAV/MP3/any Symphonia-supported format) into mono f32.
    ///
    /// Returns `(samples_mono_f32, sample_rate_hz)`.
    #[instrument(skip(bytes), fields(len = bytes.len()), level = "debug")]
    #[allow(
        clippy::cognitive_complexity,
        reason = "decode loop is inherently stateful; splitting it would obscure the packet-decode-mix flow"
    )]
    pub fn decode(bytes: &[u8]) -> Result<(Vec<f32>, u32), WavError> {
        let cursor = Cursor::new(bytes.to_vec());
        let mss = MediaSourceStream::new(Box::new(cursor), MediaSourceStreamOptions::default());

        let probed = symphonia::default::get_probe()
            .format(
                &Hint::new(),
                mss,
                &FormatOptions::default(),
                &MetadataOptions::default(),
            )
            .map_err(|e| WavError::Probe(e.to_string()))?;

        let mut reader = probed.format;

        let track = reader.default_track().ok_or(WavError::NoTrack)?;

        let track_id = track.id;
        let codec_params = track.codec_params.clone();

        let sample_rate = codec_params
            .sample_rate
            .ok_or_else(|| WavError::Decode("missing sample_rate in codec params".to_owned()))?;

        let n_channels = codec_params
            .channels
            .map_or(1, symphonia::core::audio::Channels::count)
            .max(1);

        debug!(sample_rate, n_channels, "WAV track selected");

        let mut decoder = symphonia::default::get_codecs()
            .make(&codec_params, &DecoderOptions::default())
            .map_err(|e| WavError::Decode(e.to_string()))?;

        let mut interleaved: Vec<f32> = Vec::new();
        let mut sample_buf: Option<SampleBuffer<f32>> = None;

        loop {
            let packet = match reader.next_packet() {
                Ok(p) => p,
                Err(SymphoniaError::IoError(ref e))
                    if e.kind() == std::io::ErrorKind::UnexpectedEof =>
                {
                    break;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "WAV packet error; stopping decode");
                    break;
                }
            };

            if packet.track_id() != track_id {
                continue;
            }

            let decoded = match decoder.decode(&packet) {
                Ok(buf) => buf,
                Err(SymphoniaError::DecodeError(msg)) => {
                    tracing::warn!(msg, "skipping corrupt audio packet");
                    continue;
                }
                Err(e) => return Err(WavError::Decode(e.to_string())),
            };

            let spec = *decoded.spec();
            let sb = sample_buf
                .get_or_insert_with(|| SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));
            sb.copy_interleaved_ref(decoded);
            interleaved.extend_from_slice(sb.samples());
        }

        if interleaved.is_empty() {
            return Err(WavError::Empty);
        }

        let mono = mix_to_mono(&interleaved, n_channels);
        debug!(sample_rate, mono_samples = mono.len(), "WAV decode done");
        Ok((mono, sample_rate))
    }
}

/// Average interleaved multi-channel samples to mono.
#[allow(
    clippy::indexing_slicing,
    reason = "base + ch indices bounded by n_channels and n_frames derived from len / n_channels"
)]
fn mix_to_mono(interleaved: &[f32], n_channels: usize) -> Vec<f32> {
    if n_channels == 1 {
        return interleaved.to_vec();
    }
    let n_frames = interleaved.len() / n_channels;
    let inv = 1.0_f32 / n_channels as f32;
    (0..n_frames)
        .map(|f| {
            let base = f * n_channels;
            interleaved[base..base + n_channels].iter().sum::<f32>() * inv
        })
        .collect()
}

// ── WavEncoder ────────────────────────────────────────────────────────────────

/// Encode mono f32 samples into PCM-16 little-endian WAV bytes.
///
/// The output is a self-contained WAV file (44-byte RIFF header + raw samples).
/// Input samples are clamped to `[-1.0, 1.0]` and scaled to i16 range.
#[non_exhaustive]
pub struct WavEncoder;

impl std::fmt::Debug for WavEncoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WavEncoder").finish()
    }
}

impl WavEncoder {
    /// Encode `samples` (mono f32, `[-1, 1]`) at `sample_rate` Hz to WAV bytes.
    ///
    /// `num_channels` is always 1 for the STT/TTS paths; pass 1.
    pub fn encode(
        samples: &[f32],
        sample_rate: u32,
        num_channels: u16,
    ) -> Result<Vec<u8>, WavError> {
        if num_channels == 0 {
            return Err(WavError::Encode("num_channels must be ≥ 1".to_owned()));
        }

        // Convert f32 → i16 with clamping.
        let pcm: Vec<i16> = samples
            .iter()
            .map(|&s| {
                let clamped = s.clamp(-1.0, 1.0);
                // Scale to i16 range. Avoid i16::MIN edge case by using 32767.
                // .round() prevents silent truncation bias (N1).
                (clamped * 32767.0).round() as i16
            })
            .collect();

        let num_samples = pcm.len();
        // Each i16 = 2 bytes.
        let data_bytes = num_samples * 2;
        // RIFF chunk size = data_bytes + 36 (for the rest of the header after "RIFF????").
        let riff_chunk_size = (data_bytes + 36) as u32;
        let byte_rate = sample_rate * u32::from(num_channels) * 2; // 16-bit = 2 bytes per sample
        let block_align = num_channels * 2;

        let mut out = Vec::with_capacity(44 + data_bytes);

        // RIFF header (44 bytes total for PCM WAV).
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&riff_chunk_size.to_le_bytes());
        out.extend_from_slice(b"WAVE");

        // fmt  chunk.
        out.extend_from_slice(b"fmt ");
        out.extend_from_slice(&16u32.to_le_bytes()); // subchunk1Size = 16 for PCM
        out.extend_from_slice(&1u16.to_le_bytes()); // audioFormat = 1 (PCM)
        out.extend_from_slice(&num_channels.to_le_bytes());
        out.extend_from_slice(&sample_rate.to_le_bytes());
        out.extend_from_slice(&byte_rate.to_le_bytes());
        out.extend_from_slice(&block_align.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes()); // bitsPerSample = 16

        // data chunk.
        out.extend_from_slice(b"data");
        out.extend_from_slice(&(data_bytes as u32).to_le_bytes());
        for sample in &pcm {
            out.extend_from_slice(&sample.to_le_bytes());
        }

        debug!(
            sample_rate,
            num_channels,
            num_samples,
            output_bytes = out.len(),
            "WAV encode done"
        );

        Ok(out)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "wav_tests.rs"]
mod tests;
