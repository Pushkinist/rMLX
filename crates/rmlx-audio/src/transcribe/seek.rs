//! The window plan and the window walk of the long-form transcription: which
//! segments a window's tokens make, how far the seek moves, and which windows
//! the walk opens. Pure functions, no model.

use std::sync::atomic::{AtomicBool, Ordering};

use super::fallback::Attempt;
use super::Segment;
use crate::mel::{HOP_LENGTH, N_SAMPLES, SAMPLE_RATE};
use crate::tokenizer::TOK_TIMESTAMP_BEGIN;

/// Samples of audio represented by one timestamp-token step (0.02 s).
const SAMPLES_PER_TIMESTAMP: usize = SAMPLE_RATE as usize / 50;

/// One segment cut from a window's token stream, before text decode. Its
/// start and its end are in samples from the window start; `tokens` is the
/// segment's slice, its timestamps included.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct RawSegment {
    start_samples: usize,
    end_samples: usize,
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
///
/// The start and the end of a segment are those of its timestamps, also when
/// the decoder put one in the zero pad after a short window:
/// [`window_segments`] bounds the times of the reply.
pub(super) fn plan_window(tokens: &[u32], window_samples: usize) -> WindowPlan {
    let is_timestamp = |tok: u32| tok >= TOK_TIMESTAMP_BEGIN;
    let ticks = |tok: u32| tok.saturating_sub(TOK_TIMESTAMP_BEGIN);
    let to_samples = |tok: u32| ticks(tok) as usize * SAMPLES_PER_TIMESTAMP;
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
        let end_samples = match tokens.iter().rev().find(|&&t| is_timestamp(t)) {
            Some(&last) if last != TOK_TIMESTAMP_BEGIN => to_samples(last),
            _ => window_samples,
        };
        segments.push(RawSegment {
            start_samples: 0,
            end_samples,
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
                .map_or(0, |&tok| to_samples(tok));
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
                segments.push(RawSegment {
                    start_samples: to_samples(first),
                    end_samples: to_samples(last),
                    tokens: slice.to_vec(),
                });
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

/// The segments of the reply for the window that starts at sample `seek`, in
/// a file of `total_samples`. The start and the end of a segment are sample
/// positions in the file, and each time of the reply is one such position in
/// seconds, so a later position is never an earlier time. An end is bounded
/// by the end of the file: a short window is the last one of a file, so a
/// position that the decoder put in its zero pad is the end of the file. A
/// segment whose start is not before its bounded end is dropped: it opens at
/// or after the end of the real audio, in the zero pad, where Whisper prints
/// filler ("you", "thank you"). So each segment of the reply starts before it
/// ends, and ends in the file.
///
/// `decode_text` gives the text of a segment's text tokens; a segment whose
/// trimmed text is empty is dropped. Each segment's temperature and compression
/// ratio are those of `kept`, the decode the window kept, as openai-whisper
/// gives the values of the result it kept after a fallback. Its tokens are its
/// slice of the window's tokens, with the timestamps as the decoder gave them.
pub(super) fn window_segments(
    raw: &[RawSegment],
    seek: usize,
    total_samples: usize,
    kept: &Attempt,
    decode_text: impl Fn(&[u32]) -> String,
) -> Vec<Segment> {
    let secs = |samples: usize| samples as f32 / SAMPLE_RATE as f32;
    raw.iter()
        .filter_map(|raw| {
            let start = secs(seek + raw.start_samples);
            let end = secs((seek + raw.end_samples).min(total_samples));
            // Also a segment that is shorter than one step of `f32` seconds.
            if start >= end {
                return None;
            }
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
                start,
                end,
                text: trimmed.to_owned(),
                tokens: raw.tokens.clone(),
                temperature: kept.temperature,
                compression_ratio: kept.compression_ratio,
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
