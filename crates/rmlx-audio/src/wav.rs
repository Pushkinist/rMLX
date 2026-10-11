//! Audio encoder/decoder for the Whisper STT + TTS paths.
//!
//! ## Decode
//!
//! [`WavDecoder`] wraps `symphonia` and decodes the containers the workspace
//! enables: WAV (PCM), MP3, MP4/M4A (AAC) and ADTS (AAC). The output is always:
//!
//! - Mono f32 (multi-channel inputs are averaged to mono).
//! - The file's own sample rate from [`WavDecoder::decode`], or the rate asked
//!   of [`WavDecoder::decode_resampled`].
//!
//! A file rate outside [`SAMPLE_RATES`] is refused.
//!
//! A packet that does not decode is dropped. The audio lost to dropped
//! packets and to a stream that ends short of its declared length is
//! counted; above [`lost_limit_secs`] the decode is refused, below it the loss
//! is logged. A reader that fails is refused.
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
use std::num::NonZeroU32;
use std::ops::RangeInclusive;

use symphonia::core::{
    codecs::audio::{AudioCodecParameters, AudioDecoder, AudioDecoderOptions},
    errors::Error as SymphoniaError,
    formats::{
        probe::Hint,
        well_known::{
            FORMAT_ID_ISOMP4, FORMAT_ID_MP1, FORMAT_ID_MP2, FORMAT_ID_MP3, FORMAT_ID_WAVE,
        },
        FormatId, FormatOptions, FormatReader, Track, TrackType,
    },
    io::{MediaSourceStream, MediaSourceStreamOptions},
    meta::MetadataOptions,
    packet::Packet,
    units::TimeBase,
};
mod samples;

use samples::MonoSamples;
use thiserror::Error;
use tracing::{debug, info, instrument, warn};

/// The sample rates the decoder takes, of the file and of a resample, in Hz.
/// Below 4 kHz the band holds no speech (telephony is 8 kHz); the floor also
/// holds a resample to 16 kHz to four output samples an input sample. 768 kHz
/// is the highest PCM rate in use.
pub const SAMPLE_RATES: RangeInclusive<u32> = 4_000..=768_000;

/// The most audio a transcription decodes, in seconds: 3 hours. At 16 kHz mono
/// f32 that is 691.2 MB of samples. The 25 MiB upload cap is a cap on bytes,
/// and compressed audio of that size can hold more than 7 hours.
pub const MAX_AUDIO_SECS: u32 = 3 * 60 * 60;

/// The share of a stream's length a decode may lose before it is refused.
pub const LOST_FRACTION: f64 = 0.01;
/// The loss a stream of any length may have: more than one frame of any
/// compiled codec at 8 kHz (an AAC frame is 1024 / 8000 = 0.128 s).
pub const LOST_FLOOR_SECS: f64 = 0.2;
/// The most a stream of any length may lose.
pub const LOST_CAP_SECS: f64 = 5.0;

/// The most audio a decode of a `stream_secs` long stream may lose before it
/// is refused: the packets that do not decode plus the part of a declared
/// length the stream does not hold. A smaller loss is logged at `warn`.
#[must_use]
pub fn lost_limit_secs(stream_secs: f64) -> f64 {
    (stream_secs * LOST_FRACTION).clamp(LOST_FLOOR_SECS, LOST_CAP_SECS)
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Audio encode / decode errors.
#[non_exhaustive]
#[derive(Debug, Error)]
pub enum WavError {
    /// Symphonia probe / format error.
    #[error("audio probe failed: {0}")]
    Probe(String),
    /// No audio track found in the container.
    #[error("no audio track in the file")]
    NoTrack,
    /// The track's decoder cannot be set up.
    #[error("audio decoder setup failed: {0}")]
    Decode(String),
    /// Zero samples decoded.
    #[error("audio stream contained zero samples")]
    Empty,
    /// The file's sample rate, or the rate a resample is asked for, is outside
    /// [`SAMPLE_RATES`].
    #[error(
        "sample rate of {0} Hz is outside the {min} to {max} Hz the decoder takes",
        min = SAMPLE_RATES.start(),
        max = SAMPLE_RATES.end()
    )]
    SampleRate(u32),
    /// The container reader failed.
    #[error("audio read failed at {at_secs:.2} s: {error}")]
    ReadFailed {
        /// The end of the last packet read.
        at_secs: f64,
        /// The reader's error.
        error: String,
    },
    /// The decoder failed in a way a dropped packet does not explain: it
    /// needs a reset, or it hit a limit.
    #[error("audio decode failed at {at_secs:.2} s: {error}")]
    PacketFailed {
        /// The start of the packet.
        at_secs: f64,
        /// The decoder's error.
        error: String,
    },
    /// More audio lost than [`lost_limit_secs`] allows.
    #[error("{}", lost_message(.loss, *.limit_secs, *.reached_secs, *.declared_secs))]
    LostAudio {
        /// What was lost.
        loss: AudioLoss,
        /// The loss the stream's length allows.
        limit_secs: f64,
        /// The end of the last packet read.
        reached_secs: f64,
        /// The length the file declares.
        declared_secs: Option<f64>,
    },
    /// Invalid parameters passed to encoder.
    #[error("WAV encode error: {0}")]
    Encode(String),
    /// The file declares more audio than the duration cap, or the stream
    /// passed it.
    #[error("{}", too_long_message(*.declared_secs, *.max_secs))]
    TooLong {
        /// The length the file declares; `None` when the stream passed the
        /// cap as it decoded.
        declared_secs: Option<f64>,
        /// The duration cap.
        max_secs: u32,
    },
    /// The caller's cancel flag was set; the decode stopped before the next
    /// packet.
    #[error("audio decode cancelled at {at_secs:.2} s")]
    Cancelled {
        /// The end of the last packet read.
        at_secs: f64,
    },
}

fn too_long_message(declared_secs: Option<f64>, max_secs: u32) -> String {
    match declared_secs {
        Some(secs) => format!(
            "the file declares {secs:.3} s of audio, over the {max_secs} s the decoder takes"
        ),
        None => format!("the stream holds more than the {max_secs} s of audio the decoder takes"),
    }
}

/// The sample frames of `secs` seconds at `rate` Hz.
fn frames_in(secs: u32, rate: u32) -> usize {
    usize::try_from(u64::from(secs) * u64::from(rate)).unwrap_or(usize::MAX)
}

fn lost_message(
    loss: &AudioLoss,
    limit_secs: f64,
    reached_secs: f64,
    declared_secs: Option<f64>,
) -> String {
    let mut parts = Vec::new();
    if let (Some(first), Some(last)) = (loss.first_drop_at_secs, loss.last_drop_at_secs) {
        parts.push(format!(
            "{} packets did not decode ({:.2} s, from {first:.2} s to {last:.2} s; first error: {})",
            loss.dropped_packets,
            loss.dropped_secs,
            loss.first_error.as_deref().unwrap_or("none"),
        ));
    }
    if let (true, Some(declared)) = (loss.short_secs > 0.0, declared_secs) {
        parts.push(format!(
            "the stream ends at {reached_secs:.2} s of the {declared:.2} s the file declares"
        ));
    }
    format!(
        "audio lost: {:.2} s, over the {limit_secs:.2} s limit for this file: {}",
        loss.total_secs(),
        parts.join("; ")
    )
}

// ── WavDecoder ────────────────────────────────────────────────────────────────

/// Decode audio bytes into mono f32 samples.
///
/// Samples are mixed to mono (channel average) and normalised to `[-1.0, 1.0]`.
#[non_exhaustive]
pub struct WavDecoder;

impl std::fmt::Debug for WavDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WavDecoder").finish()
    }
}

/// The result of [`WavDecoder::decode`]: the samples, and what the decode
/// read, lost and where it stopped.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq)]
pub struct DecodedAudio {
    /// Mono samples, the channels averaged: at `sample_rate`, or at the rate
    /// asked of [`WavDecoder::decode_resampled`].
    pub samples: Vec<f32>,
    /// The file's own sample rate, in Hz.
    pub sample_rate: u32,
    /// The stream length the file declares, in seconds. `None` when the file
    /// declares none, only a length the reader guesses from the bitrate, a
    /// length of 0, or a placeholder WAV data length.
    pub declared_secs: Option<f64>,
    /// The end of the last packet read, in seconds.
    pub reached_secs: f64,
    /// How the reader stopped.
    pub end: StreamEnd,
    /// The audio lost, at most [`lost_limit_secs`] of the stream's length.
    pub loss: AudioLoss,
}

/// How the container reader stopped.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StreamEnd {
    /// The reader reported the end of the stream.
    EndOfStream,
    /// The bytes ran out inside a packet: the normal end of a stream that
    /// declares no length.
    EndOfBytes,
}

/// The audio a decode lost. Dropped packets are missing from the samples, so
/// the audio after them moves earlier by `dropped_secs`.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AudioLoss {
    /// The packets that did not decode and were dropped.
    pub dropped_packets: u64,
    /// The audio the dropped packets held, in seconds.
    pub dropped_secs: f64,
    /// The start of the first dropped packet, in seconds.
    pub first_drop_at_secs: Option<f64>,
    /// The start of the last dropped packet, in seconds.
    pub last_drop_at_secs: Option<f64>,
    /// The decoder's error on the first dropped packet.
    pub first_error: Option<String>,
    /// The part of the declared length the stream does not hold, in seconds.
    pub short_secs: f64,
}

impl AudioLoss {
    /// All the audio lost, in seconds.
    #[must_use]
    pub fn total_secs(&self) -> f64 {
        self.dropped_secs + self.short_secs
    }
}

impl WavDecoder {
    /// Decode `bytes` (WAV, MP3, M4A or ADTS) into mono f32 at the file's own
    /// sample rate.
    ///
    /// # Errors
    ///
    /// [`WavError::ReadFailed`] when the container reader fails,
    /// [`WavError::PacketFailed`] when the decoder fails for a reason other
    /// than a damaged packet, and [`WavError::LostAudio`] when the dropped
    /// packets and the part of the declared length the stream does not hold
    /// add up to more than [`lost_limit_secs`] of the stream's length. A
    /// stream of which no packet decodes is [`WavError::PacketFailed`]. More
    /// than [`MAX_AUDIO_SECS`] of audio is [`WavError::TooLong`].
    pub fn decode(bytes: &[u8]) -> Result<DecodedAudio, WavError> {
        decode_at(bytes, None, MAX_AUDIO_SECS, &|| false)
    }

    /// Decode `bytes` like [`WavDecoder::decode`], resampled to `rate` Hz by
    /// linear interpolation as the packets decode. The samples are the same,
    /// bit for bit, as a linear resample of the whole output of
    /// [`WavDecoder::decode`]; the memory beside `bytes` is the output and one
    /// packet. More than `max_secs` of audio is refused: up front when the
    /// file declares its length, and before the output passes it otherwise.
    /// `cancelled` is asked before each packet; when it says the caller is
    /// gone, the decode stops there.
    ///
    /// # Errors
    ///
    /// The errors of [`WavDecoder::decode`], [`WavError::SampleRate`] for a
    /// `rate` outside [`SAMPLE_RATES`], [`WavError::TooLong`] past `max_secs`
    /// and [`WavError::Cancelled`].
    pub fn decode_resampled(
        bytes: &[u8],
        rate: u32,
        max_secs: u32,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<DecodedAudio, WavError> {
        if !SAMPLE_RATES.contains(&rate) {
            return Err(WavError::SampleRate(rate));
        }
        decode_at(bytes, Some(rate), max_secs, cancelled)
    }
}

/// Decode `bytes` to mono at `rate`, or at the file's own rate for `None`.
#[instrument(skip_all, fields(len = bytes.len(), ?rate), level = "debug")]
fn decode_at(
    bytes: &[u8],
    rate: Option<u32>,
    max_secs: u32,
    cancelled: &dyn Fn() -> bool,
) -> Result<DecodedAudio, WavError> {
    let mss = MediaSourceStream::new(
        Box::new(Cursor::new(bytes)),
        MediaSourceStreamOptions::default(),
    );

    let mut reader = symphonia::default::get_probe()
        .probe(
            &Hint::new(),
            mss,
            FormatOptions::default(),
            MetadataOptions::default(),
        )
        .map_err(|e| WavError::Probe(e.to_string()))?;

    let format = reader.format_info().format;
    let track = reader
        .default_track(TrackType::Audio)
        .ok_or(WavError::NoTrack)?
        .clone();

    let audio_params = track
        .codec_params
        .as_ref()
        .and_then(|cp| cp.audio())
        .ok_or(WavError::NoTrack)?
        .clone();

    let sample_rate = audio_params
        .sample_rate
        .ok_or_else(|| WavError::Decode("missing sample_rate in codec params".to_owned()))?;
    let file_rate = NonZeroU32::new(sample_rate)
        .filter(|hz| SAMPLE_RATES.contains(&hz.get()))
        .ok_or(WavError::SampleRate(sample_rate))?;
    let timeline = Timeline {
        time_base: track
            .time_base
            .unwrap_or_else(|| TimeBase::from_recip(file_rate)),
        sample_rate: file_rate,
    };
    let n_channels = audio_params
        .channels
        .as_ref()
        .map_or(1, symphonia::core::audio::Channels::count)
        .max(1);

    debug!(sample_rate, n_channels, "audio track selected");

    // Symphonia's own trim cuts MPEG audio at the track length even where
    // more frames follow (two files joined); the delay and padding are
    // trimmed here instead.
    let decoder_options = AudioDecoderOptions::default().gapless(false);
    let reg_entry = symphonia::default::get_codecs()
        .get_audio_decoder(audio_params.codec)
        .ok_or_else(|| WavError::Decode("no decoder for audio codec".to_owned()))?;
    let mut decoder = (reg_entry.factory)(&audio_params, &decoder_options)
        .map_err(|e| WavError::Decode(e.to_string()))?;

    let declared = declared_length(format, &track, bytes);
    refuse_declared_past_cap(declared, timeline, max_secs)?;
    let most = most_frames(format, bytes.len(), &audio_params, n_channels);
    // A declared or guessed length sizes the output up to what the bytes can
    // hold; a WAV whose data length is a placeholder is sized from its bytes.
    let expected = match (format, declared) {
        (FORMAT_ID_WAVE, None) => most,
        (FORMAT_ID_WAVE, Some(ticks)) => timeline.frames(ticks).min(most),
        _ => track
            .duration
            .map_or(0, |ticks| timeline.frames(ticks.get()))
            .min(most),
    };
    let padding = track.padding.map_or(0, |p| timeline.frames(u64::from(p)));
    let mut mono = MonoSamples::new(sample_rate, rate, padding, expected, max_secs);
    let (log, end) = read_packets(
        reader.as_mut(),
        decoder.as_mut(),
        track.id,
        timeline,
        n_channels,
        declared,
        &mut mono,
        cancelled,
    )?;

    log.decoded_any(timeline)?;
    let short = declared.map_or(0, |length| length.saturating_sub(log.reached));
    let loss = log.loss(timeline, short);
    let declared_secs = declared.map(|ticks| timeline.secs(ticks));
    let reached_secs = timeline.secs(log.reached);
    debug!(?declared_secs, reached_secs, ?end, "audio stream read");
    warn_if_cut_inside_a_packet(format, &end, declared, reached_secs);

    hold_to_limit(&loss, declared_secs, reached_secs, &end)?;
    if mono.received == 0 {
        return Err(WavError::Empty);
    }

    let cut = log.padding_to_cut(declared, track.padding, timeline);
    mono.refuse_past_cap_after(cut, sample_rate)?;

    Ok(DecodedAudio {
        samples: mono.finish(cut),
        sample_rate,
        declared_secs,
        reached_secs,
        end,
        loss,
    })
}

/// Refuse a file that declares more than `max_secs` of audio, before any
/// packet is read.
fn refuse_declared_past_cap(
    declared: Option<u64>,
    timeline: Timeline,
    max_secs: u32,
) -> Result<(), WavError> {
    let max_frames = frames_in(max_secs, timeline.sample_rate.get());
    let Some(ticks) = declared.filter(|&ticks| timeline.frames(ticks) > max_frames) else {
        return Ok(());
    };
    let declared_secs = timeline.secs(ticks);
    warn!(
        declared_secs,
        max_secs, "the file declares more audio than the cap; decode refused"
    );
    Err(WavError::TooLong {
        declared_secs: Some(declared_secs),
        max_secs,
    })
}

/// Read and decode every packet of the track `track_id` into `mono` (frames
/// before time 0, the encoder delay, left out): what the loop saw, and how
/// the reader stopped. `cancelled` is asked before each packet.
fn read_packets(
    reader: &mut dyn FormatReader,
    decoder: &mut dyn AudioDecoder,
    track_id: u32,
    timeline: Timeline,
    n_channels: usize,
    declared: Option<u64>,
    mono: &mut MonoSamples,
    cancelled: &dyn Fn() -> bool,
) -> Result<(PacketLog, StreamEnd), WavError> {
    let mut frame_buf: Vec<f32> = Vec::new();
    let mut log = PacketLog::default();

    let end = loop {
        log.hold_to_stop(cancelled, timeline)?;
        let packet = match reader.next_packet() {
            Ok(Some(p)) => p,
            Ok(None) => break StreamEnd::EndOfStream,
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break StreamEnd::EndOfBytes;
            }
            Err(e) => {
                let at_secs = timeline.secs(log.reached);
                warn!(at_secs, error = %e, "audio reader failed; decode refused");
                return Err(WavError::ReadFailed {
                    at_secs,
                    error: e.to_string(),
                });
            }
        };

        if packet.track_id != track_id {
            continue;
        }

        match decoder.decode(&packet) {
            Ok(decoded) => {
                // copy_to_vec_interleaved resizes + overwrites frame_buf, so no clear() needed.
                decoded.copy_to_vec_interleaved::<f32>(&mut frame_buf);
                let delay = u64::try_from(packet.pts.get().saturating_neg()).unwrap_or(0);
                let skip = timeline.frames(delay).saturating_mul(n_channels);
                mono.push_frames(frame_buf.get(skip..).unwrap_or_default(), n_channels)?;
                log.reach(&packet, timeline.ticks(decoded.frames()));
                log.decoded += 1;
            }
            Err(e) if is_damaged_packet(&e) => {
                debug!(
                    at_secs = timeline.secs(start_ticks(&packet)),
                    error = %e,
                    "audio packet does not decode; dropped"
                );
                log.drop_packet(&packet, &e);
                log.hold_drops_to_cap(timeline, declared)?;
                mono.push_silence(timeline.frames(ticks_from_zero(&packet)))?;
            }
            Err(e) => {
                let at_secs = timeline.secs(start_ticks(&packet));
                warn!(at_secs, error = %e, "audio decoder failed; decode refused");
                return Err(WavError::PacketFailed {
                    at_secs,
                    error: e.to_string(),
                });
            }
        }
    };
    Ok((log, end))
}

/// Refuse a decode that lost more than [`lost_limit_secs`] of the stream's
/// length; log a smaller loss.
fn hold_to_limit(
    loss: &AudioLoss,
    declared_secs: Option<f64>,
    reached_secs: f64,
    end: &StreamEnd,
) -> Result<(), WavError> {
    let lost_secs = loss.total_secs();
    let limit_secs = lost_limit_secs(declared_secs.unwrap_or(0.0).max(reached_secs));
    if lost_secs > limit_secs {
        warn!(
            lost_secs,
            limit_secs,
            dropped_packets = loss.dropped_packets,
            dropped_secs = loss.dropped_secs,
            short_secs = loss.short_secs,
            first_drop_at_secs = ?loss.first_drop_at_secs,
            ?declared_secs,
            reached_secs,
            ?end,
            "audio lost over the limit; decode refused"
        );
        return Err(WavError::LostAudio {
            loss: loss.clone(),
            limit_secs,
            reached_secs,
            declared_secs,
        });
    }
    if lost_secs > 0.0 {
        warn!(
            lost_secs,
            limit_secs,
            dropped_packets = loss.dropped_packets,
            dropped_secs = loss.dropped_secs,
            short_secs = loss.short_secs,
            first_drop_at_secs = ?loss.first_drop_at_secs,
            ?declared_secs,
            reached_secs,
            "audio lost under the limit; the samples have a gap or end early"
        );
    }
    Ok(())
}

/// A stream that declares no length cannot be checked for a cut end, but bytes
/// that end inside a packet are a sign of one. A WAV is left out: a WAV
/// written to a pipe ends that way.
fn warn_if_cut_inside_a_packet(
    format: FormatId,
    end: &StreamEnd,
    declared: Option<u64>,
    reached_secs: f64,
) {
    if *end == StreamEnd::EndOfBytes && declared.is_none() && format != FORMAT_ID_WAVE {
        warn!(
            reached_secs,
            "the bytes end inside a packet; the file may be cut"
        );
    }
}

/// A damaged packet surfaces as any of these: the AAC decoder reads past the
/// end of a mangled packet (`IoError`) or meets a feature bit flipped on
/// (`Unsupported`).
fn is_damaged_packet(error: &SymphoniaError) -> bool {
    matches!(
        error,
        SymphoniaError::DecodeError(_)
            | SymphoniaError::IoError(_)
            | SymphoniaError::Unsupported(_)
    )
}

/// The lowest WAV data length taken as a placeholder, not a length. Writers
/// that cannot seek back to patch the header leave `0xFFFFFFFF` (ffmpeg),
/// `0x7FFFF000` rounded down to the block (sox 14.4.2) or `0x80000000`
/// (arecord); a real data chunk that long holds 2 GiB.
const WAV_PLACEHOLDER_MIN: u32 = 0x7FFF_0000;

/// The most sample frames `bytes` bytes can decode to. A WAV sample takes at
/// least one byte. A compressed frame of 1024 samples takes at least four (an
/// AAC-LC mono frame of silence), the densest of the compiled codecs.
fn most_frames(
    format: FormatId,
    bytes: usize,
    params: &AudioCodecParameters,
    n_channels: usize,
) -> usize {
    if format == FORMAT_ID_WAVE {
        let sample_bytes = params
            .bits_per_coded_sample
            .map_or(1, |bits| usize::try_from(bits / 8).unwrap_or(1).max(1));
        bytes / (n_channels * sample_bytes)
    } else {
        bytes.saturating_mul(1024 / 4)
    }
}

/// The stream length the file declares, in time-base ticks. Symphonia guesses
/// the length of ADTS, and of MPEG audio with no LAME tag (the tag sets
/// `delay`), from the bitrate; a guess is not a declared length. A Xing or
/// VBRI frame count is exact, but Symphonia does not say which of the two it
/// read, so it is not used either.
fn declared_length(format: FormatId, track: &Track, bytes: &[u8]) -> Option<u64> {
    let declared = match format {
        FORMAT_ID_WAVE => {
            let placeholder = wav_data_length(bytes).is_some_and(|len| len >= WAV_PLACEHOLDER_MIN);
            if placeholder {
                debug!("the WAV data length is a placeholder; the stream is not held to it");
            }
            !placeholder
        }
        FORMAT_ID_ISOMP4 => true,
        FORMAT_ID_MP1 | FORMAT_ID_MP2 | FORMAT_ID_MP3 => track.delay.is_some(),
        _ => false,
    };
    track
        .duration
        .filter(|_| declared)
        .map(symphonia::core::units::Duration::get)
        .filter(|&length| length > 0)
}

/// The data chunk length a RIFF/WAVE header states, read from the bytes;
/// `None` when the bytes do not hold one.
fn wav_data_length(bytes: &[u8]) -> Option<u32> {
    if bytes.get(..4)? != b"RIFF" || bytes.get(8..12)? != b"WAVE" {
        return None;
    }
    let mut at: usize = 12;
    loop {
        let (id, len) = bytes.get(at..)?.get(..8)?.split_at(4);
        let len = u32::from_le_bytes(<[u8; 4]>::try_from(len).ok()?);
        if id == b"data" {
            return Some(len);
        }
        let body = usize::try_from(len)
            .ok()?
            .checked_add(usize::from(len % 2 == 1))?;
        at = at.checked_add(8)?.checked_add(body)?;
    }
}

/// What the packet loop saw, in ticks of the track's time base.
#[derive(Default)]
struct PacketLog {
    /// The end of the last packet read: its decoded frames, or its nominal
    /// length when it was dropped.
    reached: u64,
    decoded: u64,
    dropped: u64,
    dropped_ticks: u64,
    first_drop: Option<u64>,
    last_drop: Option<u64>,
    first_error: Option<String>,
}

impl PacketLog {
    fn reach(&mut self, packet: &Packet, ticks: u64) {
        let end = packet.pts.get().saturating_add_unsigned(ticks);
        self.reached = self.reached.max(u64::try_from(end).unwrap_or(0));
    }

    /// A stream of which no packet decodes was not damaged, it was not
    /// decodable: the decoder's error on the first packet.
    fn decoded_any(&self, timeline: Timeline) -> Result<(), WavError> {
        if self.decoded > 0 || self.dropped == 0 {
            return Ok(());
        }
        let at_secs = timeline.secs(self.first_drop.unwrap_or(0));
        let error = self.first_error.clone().unwrap_or_default();
        warn!(at_secs, error, "no audio packet decodes; decode refused");
        Err(WavError::PacketFailed { at_secs, error })
    }

    /// Refuse once the dropped packets hold more audio than any stream may
    /// lose, before their silence is written: the silence that stands in for
    /// dropped packets is at most [`LOST_CAP_SECS`] long.
    fn hold_drops_to_cap(&self, timeline: Timeline, declared: Option<u64>) -> Result<(), WavError> {
        let loss = self.loss(timeline, 0);
        if loss.dropped_secs <= LOST_CAP_SECS {
            return Ok(());
        }
        let reached_secs = timeline.secs(self.reached);
        let declared_secs = declared.map(|ticks| timeline.secs(ticks));
        let limit_secs = lost_limit_secs(declared_secs.unwrap_or(0.0).max(reached_secs));
        warn!(
            lost_secs = loss.dropped_secs,
            limit_secs,
            dropped_packets = loss.dropped_packets,
            first_drop_at_secs = ?loss.first_drop_at_secs,
            ?declared_secs,
            reached_secs,
            "dropped packets over the largest loss limit; decode refused before the end"
        );
        Err(WavError::LostAudio {
            loss,
            limit_secs,
            reached_secs,
            declared_secs,
        })
    }

    /// Refuse when `cancelled` says the caller is gone.
    fn hold_to_stop(
        &self,
        cancelled: &dyn Fn() -> bool,
        timeline: Timeline,
    ) -> Result<(), WavError> {
        if !cancelled() {
            return Ok(());
        }
        let at_secs = timeline.secs(self.reached);
        info!(at_secs, "audio decode cancelled");
        Err(WavError::Cancelled { at_secs })
    }

    fn drop_packet(&mut self, packet: &Packet, error: &SymphoniaError) {
        let at = start_ticks(packet);
        self.reach(packet, packet.dur.get());
        self.dropped += 1;
        self.dropped_ticks = self.dropped_ticks.saturating_add(packet.dur.get());
        self.first_drop.get_or_insert(at);
        self.last_drop = Some(at);
        if self.first_error.is_none() {
            self.first_error = Some(error.to_string());
        }
    }

    /// The frames of the declared `padding` to cut: all of them when the
    /// stream ends where the declared length plus the padding says, none when
    /// it holds more.
    fn padding_to_cut(
        &self,
        declared: Option<u64>,
        padding: Option<u32>,
        timeline: Timeline,
    ) -> usize {
        let (Some(length), Some(padding)) = (declared, padding) else {
            return 0;
        };
        let stream_end = length.saturating_add(u64::from(padding));
        if self.reached == stream_end {
            return timeline.frames(u64::from(padding));
        }
        if self.reached > stream_end {
            warn!(
                declared_secs = timeline.secs(length),
                reached_secs = timeline.secs(self.reached),
                "the stream holds more audio than its LAME tag declares; all of it is kept"
            );
        }
        0
    }

    fn loss(&self, timeline: Timeline, short: u64) -> AudioLoss {
        AudioLoss {
            dropped_packets: self.dropped,
            dropped_secs: timeline.secs(self.dropped_ticks),
            first_drop_at_secs: self.first_drop.map(|t| timeline.secs(t)),
            last_drop_at_secs: self.last_drop.map(|t| timeline.secs(t)),
            first_error: self.first_error.clone(),
            short_secs: timeline.secs(short),
        }
    }
}

/// The ticks of `packet` at or after time 0: the part a dropped packet leaves
/// as silence, the encoder delay before 0 being cut anyway.
fn ticks_from_zero(packet: &Packet) -> u64 {
    let end = packet.pts.get().saturating_add_unsigned(packet.dur.get());
    u64::try_from(end)
        .unwrap_or(0)
        .saturating_sub(start_ticks(packet))
}

/// The start of a packet in ticks; encoder delay before 0 counts as 0.
fn start_ticks(packet: &Packet) -> u64 {
    u64::try_from(packet.pts.get()).unwrap_or(0)
}

/// Converts between the track's time-base ticks, sample frames and seconds.
#[derive(Clone, Copy)]
struct Timeline {
    time_base: TimeBase,
    sample_rate: NonZeroU32,
}

impl Timeline {
    #[allow(
        clippy::cast_precision_loss,
        reason = "a tick count above 2^53 is beyond any audio length"
    )]
    fn secs(self, ticks: u64) -> f64 {
        ticks as f64 * f64::from(self.time_base.numer.get()) / f64::from(self.time_base.denom.get())
    }

    /// Ticks of `frames` sample frames.
    fn ticks(self, frames: usize) -> u64 {
        let frames = u128::try_from(frames).unwrap_or(u128::MAX);
        let per_second = u128::from(self.time_base.denom.get());
        let ticks = frames.saturating_mul(per_second)
            / (u128::from(self.time_base.numer.get()) * u128::from(self.sample_rate.get()));
        u64::try_from(ticks).unwrap_or(u64::MAX)
    }

    /// Sample frames in `ticks` ticks.
    fn frames(self, ticks: u64) -> usize {
        let ticks = u128::from(ticks);
        let frames = ticks
            .saturating_mul(u128::from(self.time_base.numer.get()))
            .saturating_mul(u128::from(self.sample_rate.get()))
            / u128::from(self.time_base.denom.get());
        usize::try_from(frames).unwrap_or(usize::MAX)
    }
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
                // .round() prevents silent truncation bias.
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
