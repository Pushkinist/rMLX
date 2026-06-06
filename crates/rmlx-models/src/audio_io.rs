//! Shared audio-source loader and decoder.
//!
//! Covers WAV and MP3 via [Symphonia] (pure-Rust, no system libs).
//!
//! ## Input variants (auto-detected by [`decode_audio_source`])
//! 1. `data:audio/…;base64,<b64>` — inline data URI (RFC 2397).
//! 2. `/<path>` / `./…` / `~/…` or any path `Path::is_file()` accepts — read from disk.
//! 3. Anything else — treated as raw base64.
//!
//! ## Core primitive
//! [`decode_audio_bytes`] accepts raw audio bytes (WAV, MP3) and returns
//! `(samples_mono_f32, sample_rate_hz)`. Channels are averaged to mono.
//! Samples are in `[-1.0, 1.0]`.
//!
//! [Symphonia]: https://github.com/pdeljanov/Symphonia

#![allow(
    clippy::cognitive_complexity,
    clippy::default_trait_access,
    clippy::redundant_closure_for_method_calls
)]
use std::io::Cursor;

use symphonia::core::{
    audio::SampleBuffer, codecs::DecoderOptions, errors::Error as SymphoniaError,
    formats::FormatOptions, io::MediaSourceStream, meta::MetadataOptions, probe::Hint,
};
use tracing::{debug, instrument, warn};

// ── Public API ────────────────────────────────────────────────────────────────

/// Decode raw audio bytes (WAV or MP3) into a mono f32 sample vector.
///
/// Returns `(samples, sample_rate)` where `samples` are averaged across all
/// channels and normalised to `[-1.0, 1.0]`.
///
/// # Errors
/// Returns a human-readable `String` on format probe failure, codec error,
/// or if the stream contains no decodable audio data.
#[instrument(skip(bytes), fields(bytes = bytes.len()), level = "debug")]
pub fn decode_audio_bytes(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    // Wrap the slice in a seekable in-memory cursor — satisfies MediaSource.
    let cursor = Cursor::new(bytes.to_vec());
    let mss = MediaSourceStream::new(Box::new(cursor), Default::default());

    // Probe: let symphonia detect the container type.
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            mss,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(|e| format!("audio probe failed: {e}"))?;

    let mut reader = probed.format;

    // Pick the first default/best audio track.
    let track = reader
        .default_track()
        .ok_or_else(|| "no audio track found".to_owned())?;

    let track_id = track.id;
    let codec_params = track.codec_params.clone();

    let sample_rate = codec_params
        .sample_rate
        .ok_or_else(|| "codec params missing sample_rate".to_owned())?;

    let n_channels = codec_params.channels.map_or(1, |c| c.count()).max(1);

    debug!(
        sample_rate,
        n_channels, track_id, "audio track selected for decode"
    );

    let mut decoder = symphonia::default::get_codecs()
        .make(&codec_params, &DecoderOptions::default())
        .map_err(|e| format!("failed to create audio decoder: {e}"))?;

    // Accumulate interleaved f32 samples from all packets.
    let mut interleaved: Vec<f32> = Vec::new();
    let mut sample_buf: Option<SampleBuffer<f32>> = None;

    loop {
        let packet = match reader.next_packet() {
            Ok(p) => p,
            // End of stream is the normal termination signal.
            Err(SymphoniaError::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(e) => {
                warn!(error = %e, "audio packet read error; stopping decode");
                break;
            }
        };

        if packet.track_id() != track_id {
            continue;
        }

        let decoded = match decoder.decode(&packet) {
            Ok(buf) => buf,
            Err(SymphoniaError::DecodeError(msg)) => {
                warn!(msg, "skipping corrupt audio packet");
                continue;
            }
            Err(e) => return Err(format!("audio decode error: {e}")),
        };

        // Lazily allocate the sample buffer on the first decoded frame so we
        // know the exact capacity needed.
        let spec = *decoded.spec();
        let sb = sample_buf
            .get_or_insert_with(|| SampleBuffer::<f32>::new(decoded.capacity() as u64, spec));

        sb.copy_interleaved_ref(decoded);
        interleaved.extend_from_slice(sb.samples());
    }

    if interleaved.is_empty() {
        return Err("audio stream decoded zero samples".to_owned());
    }

    // Mix interleaved multi-channel samples to mono by averaging channels.
    let mono = mix_to_mono(&interleaved, n_channels);

    debug!(
        sample_rate,
        n_channels,
        mono_samples = mono.len(),
        "audio decode complete"
    );

    Ok((mono, sample_rate))
}

/// Decode an audio source string into mono f32 samples.
///
/// `src` is interpreted in this order:
/// 1. `data:audio/…;base64,<b64>` — data URI.
/// 2. A file path (`/…`, `./…`, `~/…`, or any path `Path::is_file()` accepts).
/// 3. Raw base64.
///
/// Returns `(samples, sample_rate)` — same contract as [`decode_audio_bytes`].
#[instrument(skip(src), fields(kind = %detect_source_kind(src)), level = "debug")]
pub fn decode_audio_source(src: &str) -> Result<(Vec<f32>, u32), String> {
    let bytes = load_audio_bytes(src)?;
    decode_audio_bytes(&bytes)
}

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Resolve `src` to raw audio bytes without decoding.
fn load_audio_bytes(src: &str) -> Result<Vec<u8>, String> {
    let s = src.trim();

    // ── 1. data: URI ──────────────────────────────────────────────────────────
    if let Some(rest) = s.strip_prefix("data:") {
        let comma = rest
            .find(',')
            .ok_or_else(|| "malformed data URI (no comma)".to_owned())?;
        let meta = &rest[..comma];
        let payload = &rest[comma + 1..];
        if !meta.contains("base64") {
            return Err("only base64 data URIs are supported".to_owned());
        }
        return base64_decode(payload).map_err(|e| format!("data URI base64 decode: {e}"));
    }

    // ── 2. File path ──────────────────────────────────────────────────────────
    let looks_pathish = s.starts_with('/')
        || s.starts_with("./")
        || s.starts_with("~/")
        || std::path::Path::new(s).is_file();
    if looks_pathish {
        let p = if let Some(home_rel) = s.strip_prefix("~/") {
            match std::env::var_os("HOME") {
                Some(h) => std::path::PathBuf::from(h).join(home_rel),
                None => std::path::PathBuf::from(s),
            }
        } else {
            std::path::PathBuf::from(s)
        };
        return std::fs::read(&p)
            .map_err(|e| format!("cannot read audio file {}: {e}", p.display()));
    }

    // ── 3. Raw base64 ─────────────────────────────────────────────────────────
    base64_decode(s).map_err(|e| format!("base64 decode: {e}"))
}

/// Average interleaved multi-channel samples to mono.
///
/// `n_channels == 1` is a no-op copy. The input length must be an exact
/// multiple of `n_channels`; any trailing incomplete frame is dropped with a
/// warning.
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn mix_to_mono(interleaved: &[f32], n_channels: usize) -> Vec<f32> {
    if n_channels == 1 {
        return interleaved.to_vec();
    }
    let n_frames = interleaved.len() / n_channels;
    let leftover = interleaved.len() % n_channels;
    if leftover != 0 {
        warn!(
            leftover,
            n_channels, "trailing incomplete audio frame dropped during mono mix"
        );
    }
    let inv = 1.0_f32 / n_channels as f32;
    (0..n_frames)
        .map(|f| {
            let base = f * n_channels;
            interleaved[base..base + n_channels].iter().sum::<f32>() * inv
        })
        .collect()
}

/// One-word label for the tracing `kind` field (no I/O).
fn detect_source_kind(s: &str) -> &'static str {
    let s = s.trim();
    if s.starts_with("data:") {
        "data_uri"
    } else if s.starts_with('/') || s.starts_with("./") || s.starts_with("~/") {
        "file"
    } else {
        "base64"
    }
}

// ── Internal base64 decoder ───────────────────────────────────────────────────
// Mirrors image_io.rs — RFC 4648 standard alphabet, optional padding.

fn base64_decode(input: &str) -> Result<Vec<u8>, String> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut bits = 0u32;
    let mut nbits = 0u32;
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    for &c in input.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        let v = val(c).ok_or_else(|| format!("invalid base64 char {:?}", c as char))?;
        bits = (bits << 6) | u32::from(v);
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((bits >> nbits) as u8);
        }
    }
    if out.is_empty() {
        return Err("empty base64 payload".to_owned());
    }
    Ok(out)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "audio_io_tests.rs"]
mod tests;
