//! The window plan and the window walk of the long-form transcription: which
//! segments a window's tokens make, how far the seek moves, and which windows
//! the walk opens. Pure functions, no model.

use std::sync::atomic::{AtomicBool, Ordering};

use super::{Segment, WindowDecode};
use crate::mel::{HOP_LENGTH, N_SAMPLES, SAMPLE_RATE};
use crate::tokenizer::TOK_TIMESTAMP_BEGIN;

/// Seconds of audio represented by one timestamp-token step (Whisper uses 0.02 s).
const TIME_PRECISION: f32 = 0.02;

/// Samples of audio represented by one timestamp-token step.
const SAMPLES_PER_TIMESTAMP: usize = SAMPLE_RATE as usize / 50;

/// How far past the real end of a window a segment may open before it counts as
/// zero-pad filler (0.5 s).
const PAD_TAIL_MARGIN_SAMPLES: usize = SAMPLE_RATE as usize / 2;

/// One segment cut from a window's token stream, before text decode. Times are
/// seconds from the window start; `tokens` is the segment's slice, its
/// timestamps included.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RawSegment {
    start: f32,
    end: f32,
    tokens: Vec<u32>,
}

/// Which rule set the seek advance of a window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SeekRule {
    /// A timestamp pair in the stream and one timestamp at the end: the whole
    /// window.
    SingleEnding,
    /// A timestamp pair and no single ending: the end of the last closed
    /// segment.
    LastClosed { end_samples: usize },
    /// No timestamp pair: one segment and the whole window.
    NoPair,
    /// The last closed segment ends at the window start, so the seek would not
    /// move: the whole window.
    ZeroFallback,
}

/// What one decoded window yields: its segments and how far the seek moves.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct WindowPlan {
    pub(super) segments: Vec<RawSegment>,
    pub(super) advance_samples: usize,
    pub(super) rule: SeekRule,
}

/// Split one window's sampled tokens into timestamped segments and decide the
/// seek advance. `tokens` are the tokens after the decoder prefix, with the end
/// token stripped. `window_samples` is the real (un-padded) length of the window.
///
/// A timestamp pair (two timestamps in a row) closes one segment and opens the
/// next. With a pair in the stream:
/// - a stream that ends in one timestamp has no speech after it, so the whole
///   window is consumed;
/// - otherwise the unfinished segment is dropped and the seek goes to the end
///   of the last closed segment.
///
/// With no pair the stream is one segment from the window start, and the whole
/// window is consumed.
pub(super) fn plan_window(tokens: &[u32], window_samples: usize) -> WindowPlan {
    let window_dur = window_samples as f32 / SAMPLE_RATE as f32;
    let is_timestamp = |tok: u32| tok >= TOK_TIMESTAMP_BEGIN;
    let ticks = |tok: u32| tok.saturating_sub(TOK_TIMESTAMP_BEGIN);
    let to_secs = |tok: u32| ticks(tok) as f32 * TIME_PRECISION;
    // A segment that opens more than 0.5 s past the real audio is in the
    // zero-pad tail, where Whisper prints filler ("you", "thank you").
    let opens_in_speech = |tok: u32| {
        ticks(tok) as usize * SAMPLES_PER_TIMESTAMP <= window_samples + PAD_TAIL_MARGIN_SAMPLES
    };
    let ends_in_one_timestamp = matches!(
        tokens,
        [.., before, last] if !is_timestamp(*before) && is_timestamp(*last)
    );
    // Index of the second token of every timestamp pair: where one slice ends
    // and the next begins.
    let mut cuts: Vec<usize> = tokens
        .windows(2)
        .enumerate()
        .filter(|(_, pair)| pair.iter().all(|&t| is_timestamp(t)))
        .map(|(i, _)| i + 1)
        .collect();

    let mut segments: Vec<RawSegment> = Vec::new();
    let rule = if cuts.is_empty() {
        let end = match tokens.iter().rev().find(|&&t| is_timestamp(t)) {
            Some(&last) if last != TOK_TIMESTAMP_BEGIN => to_secs(last),
            _ => window_dur,
        };
        segments.push(RawSegment {
            start: 0.0,
            end,
            tokens: tokens.to_vec(),
        });
        SeekRule::NoPair
    } else {
        let rule = if ends_in_one_timestamp {
            cuts.push(tokens.len());
            SeekRule::SingleEnding
        } else {
            let end_samples = cuts
                .last()
                .and_then(|&cut| tokens.get(cut - 1))
                .map_or(0, |&tok| ticks(tok) as usize * SAMPLES_PER_TIMESTAMP);
            if end_samples == 0 {
                SeekRule::ZeroFallback
            } else {
                SeekRule::LastClosed { end_samples }
            }
        };
        let mut from = 0;
        for &cut in &cuts {
            let slice = tokens.get(from..cut).unwrap_or_default();
            if let (Some(&first), Some(&last)) = (slice.first(), slice.last()) {
                if opens_in_speech(first) {
                    segments.push(RawSegment {
                        start: to_secs(first),
                        end: to_secs(last),
                        tokens: slice.to_vec(),
                    });
                }
            }
            from = cut;
        }
        rule
    };
    segments.retain(|s| s.tokens.iter().any(|&t| !is_timestamp(t)));

    let advance_samples = match rule {
        SeekRule::LastClosed { end_samples } => end_samples.min(window_samples),
        SeekRule::SingleEnding | SeekRule::NoPair | SeekRule::ZeroFallback => window_samples,
    };
    WindowPlan {
        segments,
        advance_samples,
        rule,
    }
}

/// The segments of the window that starts at sample `seek`, with times from
/// the start of the audio: the window start plus each raw time.
/// `decode_text` gives the text of a segment's text tokens; a segment whose
/// trimmed text is empty is dropped. Each segment's compression ratio is that
/// of the decode the window kept, as openai-whisper gives the ratio of the
/// result it kept after a fallback.
pub(super) fn window_segments(
    raw: &[RawSegment],
    seek: usize,
    decoded: &WindowDecode,
    decode_text: impl Fn(&[u32]) -> String,
) -> Vec<Segment> {
    let window_start_time = seek as f32 / SAMPLE_RATE as f32;
    raw.iter()
        .filter_map(|raw| {
            let text_tokens: Vec<u32> = raw
                .tokens
                .iter()
                .copied()
                .filter(|&t| t < TOK_TIMESTAMP_BEGIN)
                .collect();
            let text = decode_text(&text_tokens);
            let trimmed = text.trim();
            (!trimmed.is_empty()).then(|| Segment {
                seek: seek / HOP_LENGTH,
                start: window_start_time + raw.start,
                end: window_start_time + raw.end,
                text: trimmed.to_owned(),
                tokens: raw.tokens.clone(),
                compression_ratio: decoded.kept_ratio,
            })
        })
        .collect()
}

/// How a window walk ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum WalkEnd {
    /// No audio is left to open a window.
    Finished,
    /// The cancel flag was set: window number `window` (from 0), which starts
    /// at sample `seek`, was not opened.
    Cancelled { window: usize, seek: usize },
}

/// Walk the audio window by window. `on_window(seek, window_samples)` decodes
/// the window that starts at sample `seek` and returns how many samples to
/// advance; an advance of 0 consumes the whole window, so the walk always ends.
/// The walk stops when fewer than one mel hop of audio is left, as the
/// reference does at its 10 ms frame grid. `cancel` is read before each window
/// opens; once it is set, the walk opens no further window.
pub(super) fn walk_windows<E>(
    total_samples: usize,
    cancel: &AtomicBool,
    mut on_window: impl FnMut(usize, usize) -> Result<usize, E>,
) -> Result<WalkEnd, E> {
    let mut seek = 0;
    let mut window = 0;
    while total_samples.saturating_sub(seek) >= HOP_LENGTH {
        if cancel.load(Ordering::Relaxed) {
            return Ok(WalkEnd::Cancelled { window, seek });
        }
        let window_samples = (total_samples - seek).min(N_SAMPLES);
        seek += match on_window(seek, window_samples)? {
            0 => window_samples,
            advance => advance,
        };
        window += 1;
    }
    Ok(WalkEnd::Finished)
}

#[cfg(test)]
#[path = "seek_tests.rs"]
mod seek_tests;
