//! Unit tests for the window plan and the window walk (no model needed).

use super::{plan_window, walk_windows, window_segments, RawSegment, SeekRule, WalkEnd};
use crate::mel::HOP_LENGTH;
use crate::tokenizer::TOK_TIMESTAMP_BEGIN;
use crate::transcribe::fallback::{Attempt, Verdict};
use std::sync::atomic::{AtomicBool, Ordering};

// ── Window plan: segments and seek advance from a window's sampled tokens ────
//
// Oracle: the seek decision block of openai-whisper `transcribe()`
// (https://github.com/openai/whisper/blob/86098128c0b4f24f0e2aa2994de830614b474227/whisper/transcribe.py#L339-L399).
// The reference counts in 10 ms mel frames and rounds to whole frames; the
// sample counts below are the translation of its rule to samples at 16 kHz
// (one timestamp step is 320 samples). Rules:
//   - a stream that ends in ONE timestamp (`[text, ts]`) means no speech follows
//     the last timestamp: the whole real window is consumed;
//   - a stream with a timestamp pair that does not end in one timestamp drops
//     the unfinished segment and seeks to the end of the last closed segment;
//   - a stream with no timestamp pair is one segment that starts at the window
//     start (not at its first timestamp) and consumes the whole real window.
// Not part of the reference: the reply has no segment that opens at or after
// the end of the real audio (see "Segment times against the real audio").

const SAMPLES_PER_WINDOW: usize = 480_000;
const A: u32 = 1_000;
const B: u32 = 1_001;

fn ts(secs: f32) -> u32 {
    TOK_TIMESTAMP_BEGIN + (secs / 0.02).round() as u32
}

/// A segment from `start` to `end`, in seconds from the window start.
fn seg(start: f32, end: f32, tokens: &[u32]) -> RawSegment {
    RawSegment {
        start_samples: (start * 16_000.0).round() as usize,
        end_samples: (end * 16_000.0).round() as usize,
        tokens: tokens.to_vec(),
    }
}

#[track_caller]
fn assert_plan(
    tokens: &[u32],
    window_samples: usize,
    want_segments: &[RawSegment],
    want_advance: usize,
) {
    let plan = plan_window(tokens, window_samples);
    assert_eq!(
        plan.advance_samples, want_advance,
        "seek advance in samples for a {window_samples}-sample window"
    );
    assert_eq!(
        plan.segments.len(),
        want_segments.len(),
        "{:?}",
        plan.segments
    );
    for (got, want) in plan.segments.iter().zip(want_segments) {
        assert!(
            got.start_samples.abs_diff(want.start_samples) <= 16
                && got.end_samples.abs_diff(want.end_samples) <= 16,
            "segment times: got {got:?}, want {want:?}"
        );
        let text: Vec<u32> = got
            .tokens
            .iter()
            .copied()
            .filter(|&t| t < TOK_TIMESTAMP_BEGIN)
            .collect();
        assert_eq!(text, want.tokens, "the text tokens of {got:?}");
    }
}

/// Each segment carries its slice of the window's tokens with its
/// timestamps, as openai-whisper's `new_segment(tokens=sliced_tokens)`. A
/// window with no timestamp pair is one segment that holds every token.
#[test]
fn segments_carry_their_slice_with_its_timestamps() {
    let slices = |tokens: &[u32], window_samples: usize| -> Vec<Vec<u32>> {
        plan_window(tokens, window_samples)
            .segments
            .into_iter()
            .map(|s| s.tokens)
            .collect()
    };
    let single_ending = [ts(0.0), A, A, ts(2.28), ts(2.66), B, B, ts(5.32)];
    assert_eq!(
        slices(&single_ending, 86_551),
        [
            vec![ts(0.0), A, A, ts(2.28)],
            vec![ts(2.66), B, B, ts(5.32)]
        ]
    );
    let unfinished = [ts(0.0), A, ts(2.0), ts(2.5), B];
    assert_eq!(
        slices(&unfinished, SAMPLES_PER_WINDOW),
        [vec![ts(0.0), A, ts(2.0)]]
    );
    let no_pair = [ts(0.0), A, B];
    assert_eq!(slices(&no_pair, SAMPLES_PER_WINDOW), [no_pair.to_vec()]);
}

/// Two spoken sentences, 5.4 s of audio, one timestamp at the end: one pass
/// over the clip. Seeking to the last timestamp would open two tail windows
/// over the last 0.09 s.
#[test]
fn window_ending_in_one_timestamp_consumes_whole_short_window() {
    let tokens = [ts(0.0), A, A, ts(2.28), ts(2.66), B, B, ts(5.32)];
    assert_plan(
        &tokens,
        86_551,
        &[seg(0.0, 2.28, &[A, A]), seg(2.66, 5.32, &[B, B])],
        86_551,
    );
}

/// The same rule on a full window: a last timestamp at 29.40 s does not move the
/// next window to 29.40 s.
#[test]
fn window_ending_in_one_timestamp_consumes_whole_full_window() {
    assert_plan(
        &[ts(0.0), A, ts(29.40)],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 29.40, &[A])],
        SAMPLES_PER_WINDOW,
    );
    assert_plan(
        &[ts(0.0), A, ts(10.0), ts(10.5), B, ts(29.40)],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 10.0, &[A]), seg(10.5, 29.40, &[B])],
        SAMPLES_PER_WINDOW,
    );
}

#[test]
fn short_window_ending_in_one_timestamp_consumes_whole_window() {
    assert_plan(
        &[ts(0.0), A, ts(1.5)],
        32_000,
        &[seg(0.0, 1.5, &[A])],
        32_000,
    );
}

/// A stream that ends in a timestamp pair seeks to the end of the last closed
/// segment.
#[test]
fn window_ending_in_equal_timestamp_pair_seeks_to_the_pair() {
    assert_plan(
        &[ts(0.0), A, ts(12.0), ts(12.0)],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 12.0, &[A])],
        192_000,
    );
}

/// A pair whose second stamp is later: the seek goes to the first stamp (the
/// end of the closed segment), not to the second.
#[test]
fn window_ending_in_unequal_timestamp_pair_seeks_to_the_closed_segment() {
    assert_plan(
        &[ts(0.0), A, ts(12.0), ts(12.4)],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 12.0, &[A])],
        192_000,
    );
}

/// Text after the last closed segment is an unfinished segment: it is not
/// emitted, and the seek returns to the end of the closed segment so the next
/// window decodes that speech once.
#[test]
fn unfinished_tail_segment_is_dropped_and_seek_returns_to_closed_segment() {
    assert_plan(
        &[ts(0.0), A, ts(2.0), ts(2.5), B],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 2.0, &[A])],
        32_000,
    );
}

/// Text with no timestamp at all: one segment, the whole window consumed.
#[test]
fn text_without_timestamps_consumes_whole_window() {
    assert_plan(
        &[A, B],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 30.0, &[A, B])],
        SAMPLES_PER_WINDOW,
    );
    assert_plan(&[A, B], 48_000, &[seg(0.0, 3.0, &[A, B])], 48_000);
}

/// No text and no closing timestamp: nothing is emitted, the window is
/// consumed, and the seek always moves forward.
#[test]
fn empty_stream_and_lone_opening_timestamp_consume_whole_window() {
    assert_plan(&[], SAMPLES_PER_WINDOW, &[], SAMPLES_PER_WINDOW);
    assert_plan(&[ts(0.0)], SAMPLES_PER_WINDOW, &[], SAMPLES_PER_WINDOW);
    assert_plan(&[], 1_434, &[], 1_434);
}

/// Consuming the whole window must advance by exactly the window's sample
/// count for every length, on both whole-window branches. Windows under 160
/// samples are tested here through `plan_window`; the walk never opens them.
#[test]
fn whole_window_consumption_leaves_no_sample_behind() {
    for window_samples in 1..=SAMPLES_PER_WINDOW {
        let no_timestamps = plan_window(&[A, B], window_samples);
        assert_eq!(
            no_timestamps.advance_samples, window_samples,
            "no timestamps, window of {window_samples} samples"
        );
        let one_timestamp = plan_window(&[ts(0.0), A, ts(0.02)], window_samples);
        assert_eq!(
            one_timestamp.advance_samples, window_samples,
            "one closing timestamp, window of {window_samples} samples"
        );
        let pair_then_one_timestamp = plan_window(
            &[ts(0.0), A, ts(0.02), ts(0.04), B, ts(0.06)],
            window_samples,
        );
        assert_eq!(
            pair_then_one_timestamp.advance_samples, window_samples,
            "timestamp pair, then one closing timestamp, window of {window_samples} samples"
        );
    }
}

/// Several closed segments: the seek goes to the end of the LAST closed
/// segment, not the first, and the unfinished text after it is dropped.
#[test]
fn several_closed_segments_seek_to_the_last_one() {
    assert_plan(
        &[ts(0.0), A, ts(2.0), ts(2.5), B, ts(4.0), ts(4.5), A],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 2.0, &[A]), seg(2.5, 4.0, &[B])],
        64_000,
    );
}

/// One opening timestamp and text, nothing else: no timestamp pair, so one
/// segment over the whole window and the whole window consumed.
#[test]
fn opening_timestamp_and_text_without_closing_is_one_whole_window_segment() {
    assert_plan(
        &[ts(0.0), A, B],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 30.0, &[A, B])],
        SAMPLES_PER_WINDOW,
    );
}

/// A pair at the very start closes nothing, so the seek would not move: the
/// window is consumed instead of looping on it.
#[test]
fn seek_always_moves_forward() {
    assert_plan(
        &[ts(0.0), ts(0.0)],
        SAMPLES_PER_WINDOW,
        &[],
        SAMPLES_PER_WINDOW,
    );
    assert_plan(&[ts(0.0), ts(0.0)], 1_434, &[], 1_434);
}

/// With no timestamp pair the segment starts at the window start, as in the
/// reference, even when the first timestamp is later.
#[test]
fn single_segment_starts_at_the_window_start() {
    assert_plan(
        &[ts(1.0), A, ts(3.0)],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 3.0, &[A])],
        SAMPLES_PER_WINDOW,
    );
}

/// With a timestamp pair, the first segment starts at its own first timestamp
/// (the reference takes `sliced_tokens[0]`), not at the window start.
#[test]
fn pair_branch_first_segment_starts_at_its_timestamp() {
    assert_plan(
        &[ts(1.0), A, ts(2.0), ts(2.5), B, ts(4.0)],
        SAMPLES_PER_WINDOW,
        &[seg(1.0, 2.0, &[A]), seg(2.5, 4.0, &[B])],
        SAMPLES_PER_WINDOW,
    );
}

/// The last closed segment may end past the real audio of a short window; the
/// advance is clamped to the window.
#[test]
fn seek_never_passes_the_end_of_the_window() {
    assert_plan(
        &[ts(0.0), A, ts(2.0), ts(2.0)],
        16_000,
        &[seg(0.0, 2.0, &[A])],
        16_000,
    );
}

/// Text before the first timestamp is not a panic: its start is the window start.
#[test]
fn text_before_the_first_timestamp_starts_at_the_window_start() {
    assert_plan(
        &[A, ts(2.0), ts(2.5), B, ts(4.0)],
        SAMPLES_PER_WINDOW,
        &[seg(0.0, 2.0, &[A]), seg(2.5, 4.0, &[B])],
        SAMPLES_PER_WINDOW,
    );
}

/// Each branch names the rule that set the advance.
#[test]
fn plan_names_the_seek_rule() {
    let rule = |tokens: &[u32]| plan_window(tokens, SAMPLES_PER_WINDOW).rule;
    assert_eq!(
        rule(&[ts(0.0), A, ts(2.0), ts(2.5), B, ts(4.0)]),
        SeekRule::SingleEnding
    );
    assert_eq!(
        rule(&[ts(0.0), A, ts(12.0), ts(12.4)]),
        SeekRule::LastClosed {
            end_samples: 192_000
        }
    );
    assert_eq!(
        rule(&[ts(0.0), A, ts(0.02), ts(0.02)]),
        SeekRule::LastClosed { end_samples: 320 }
    );
    assert_eq!(rule(&[A, B]), SeekRule::NoPair);
    assert_eq!(rule(&[ts(0.0), ts(0.0)]), SeekRule::ZeroFallback);
}

// ── Segment times against the real audio ─────────────────────────────────────
//
// The decoder can close a segment after the real end of a short window: the
// timestamp is in the zero pad. A short window is the last window of a file,
// so the real end of its audio is the end of the file. openai-whisper does
// not bound the timestamp (`transcribe.py#L361-L362` and `#L379-L394` at the
// commit above give a segment `time_offset` plus the timestamp), so its
// segment can end after the audio. Here no segment of a reply starts or ends
// after the duration of the file, and a segment that is left with no length
// is not in the reply.

/// A tail window of 3.37 s that starts at 108.68 s of a file of 112.05 s.
const TAIL_SEEK: usize = 1_738_880;
const TAIL_SAMPLES: usize = 53_920;

fn accepted() -> Attempt {
    Attempt {
        temperature: 0.0,
        with_prompt: true,
        compression_ratio: 1.0,
        avg_logprob: -0.3,
        verdict: Verdict::Accepted,
    }
}

/// The duration of a file that ends with the window, and the start and the
/// end of each segment that the reply has for the window's `tokens`.
fn reply(tokens: &[u32], seek: usize, window_samples: usize) -> (f32, Vec<(f32, f32)>) {
    let duration = (seek + window_samples) as f32 / 16_000.0;
    let plan = plan_window(tokens, window_samples);
    let total_samples = seek + window_samples;
    let segments = window_segments(&plan.segments, seek, total_samples, &accepted(), |_| {
        "text".to_owned()
    });
    let times = segments.iter().map(|s| (s.start, s.end)).collect();
    (duration, times)
}

/// A closing timestamp of 4.00 s in a tail window of 3.37 s ends the segment
/// at the duration of the file, with a timestamp pair and with none. The seek
/// advance is the whole window, as before.
#[test]
#[allow(clippy::float_cmp, reason = "the bound is exact")]
fn a_closing_timestamp_after_the_real_audio_ends_the_segment_with_the_file() {
    let no_pair = [ts(0.0), A, ts(4.0)];
    let pair = [ts(0.0), A, ts(2.0), ts(2.0), B, ts(4.0)];
    for (tokens, segments) in [(&no_pair[..], 1), (&pair[..], 2)] {
        let (duration, times) = reply(tokens, TAIL_SEEK, TAIL_SAMPLES);
        assert_eq!(times.len(), segments, "{times:?}");
        assert_eq!(times.last().map(|t| t.1), Some(duration), "{times:?}");
        assert!(times.iter().all(|(start, end)| start < end), "{times:?}");
        assert_eq!(
            plan_window(tokens, TAIL_SAMPLES).advance_samples,
            TAIL_SAMPLES
        );
    }
}

/// A segment that opens exactly at the end of the real audio is not in the
/// reply, for each timestamp at which a tail window can end and at two seek
/// positions: the opening timestamp and the end of the file are the same
/// sample. A sum of seconds in `f32` can put that start one step before the
/// duration and keep a segment whose two times print the same millisecond.
#[test]
fn a_segment_that_opens_at_the_end_of_the_real_audio_is_not_in_the_reply() {
    let at_the_end = [ts(0.0), A, ts(0.3), ts(0.3), B, ts(1.0)];
    assert_eq!(reply(&at_the_end, 0, 4_800).1.len(), 1);

    for seek in [0, 1_738_880] {
        for tick in 1..=1_500_u32 {
            let opens = TOK_TIMESTAMP_BEGIN + tick;
            let tokens = [ts(0.0), A, opens, opens, B, opens + 35];
            let window_samples = tick as usize * 320;
            let (duration, times) = reply(&tokens, seek, window_samples);
            assert_eq!(times.len(), 1, "seek {seek}, tick {tick}: {times:?}");
            assert!(
                times
                    .iter()
                    .all(|(start, end)| start < end && *end <= duration),
                "seek {seek}, tick {tick}: {times:?} in {duration} s"
            );
        }
    }
}

/// Three hours into a file, one sample is less than one step of `f32`
/// seconds. A segment that opens one sample before the end of the file has
/// two equal times; it is not in the reply, and each segment that is in it
/// starts before it ends.
#[test]
fn a_segment_shorter_than_one_step_of_seconds_is_not_in_the_reply() {
    let three_hours = 3 * 3_600 * 16_000;
    for tick in 1..=1_500_u32 {
        let opens = TOK_TIMESTAMP_BEGIN + tick;
        let tokens = [ts(0.0), A, opens, opens, B, opens + 35];
        let (duration, times) = reply(&tokens, three_hours, tick as usize * 320 + 1);
        assert_eq!(times.len(), 1, "tick {tick}: {times:?}");
        assert!(
            times
                .iter()
                .all(|(start, end)| start < end && *end <= duration),
            "tick {tick}: {times:?} in {duration} s"
        );
    }
}

/// A segment that opens after the real audio is not in the reply: its text is
/// from the zero pad. This holds a little after the real end and far after
/// it. The seek
/// reads the tokens, not the reply: the advance is the whole window, and the
/// end of the last closed segment is still the uncut timestamp.
#[test]
#[allow(clippy::float_cmp, reason = "the bound is exact")]
fn a_segment_that_opens_after_the_real_audio_is_not_in_the_reply() {
    for opens in [3.6, 3.9, 3.38] {
        let tokens = [ts(0.0), A, ts(opens), ts(opens), B, ts(4.0)];
        let (duration, times) = reply(&tokens, TAIL_SEEK, TAIL_SAMPLES);
        assert_eq!(times.len(), 1, "opens at {opens}: {times:?}");
        assert_eq!(times[0].1, duration, "opens at {opens}");
        assert_eq!(
            plan_window(&tokens, TAIL_SAMPLES).advance_samples,
            TAIL_SAMPLES
        );
    }

    let closed_late = [ts(0.0), A, ts(4.0), ts(4.0), B];
    let plan = plan_window(&closed_late, TAIL_SAMPLES);
    assert_eq!(
        plan.rule,
        SeekRule::LastClosed {
            end_samples: 64_000
        }
    );
    assert_eq!(plan.advance_samples, TAIL_SAMPLES);
    let (duration, times) = reply(&closed_late, TAIL_SEEK, TAIL_SAMPLES);
    assert_eq!(times.len(), 1);
    assert_eq!(times[0].1, duration);
}

/// No segment of a reply starts or ends after the duration of the file, with
/// no tolerance. The window start and a time in the window are added in
/// `f32`, and for some windows that sum is above the duration of the file:
/// the inputs here hold such windows.
#[test]
fn no_segment_of_a_reply_ends_after_the_duration_of_the_file() {
    let mut sums_above_the_duration = 0;
    for seek in (0..40).map(|k| 480_000 * k + 7_919 * (k % 5)) {
        for window_samples in (1..60).map(|k| 160 + 7_993 * k) {
            let (duration, times) = reply(&[ts(0.0), A], seek, window_samples);
            let sum = seek as f32 / 16_000.0 + window_samples as f32 / 16_000.0;
            sums_above_the_duration += usize::from(sum > duration);
            assert_eq!(times.len(), 1);
            for (start, end) in times {
                assert!(
                    start < end && end <= duration,
                    "{start} to {end} in a file of {duration} s (seek {seek}, window {window_samples})"
                );
            }
        }
    }
    assert!(
        sums_above_the_duration > 0,
        "no input rounds above the duration"
    );
}

// ── Window walk: which windows are opened, from scripted token streams ───────

const MAX_SCRIPTED_WINDOWS: usize = 8;

/// Run `walk_windows` over `total_samples`, answering window `k` with
/// `scripts[k]`. Any window past the script gets a one-timestamp filler stream.
/// Returns every `(seek, window_samples)` the walk asked for.
fn walk(total_samples: usize, scripts: &[&[u32]]) -> Vec<(usize, usize)> {
    let filler = [ts(0.0), A, ts(0.04)];
    let mut calls: Vec<(usize, usize)> = Vec::new();
    let never = AtomicBool::new(false);
    let result = walk_windows(total_samples, &never, |seek, window_samples| {
        // No scripted walk opens this many windows: a walk that does not end
        // fails here and does not hang the test binary.
        if calls.len() >= MAX_SCRIPTED_WINDOWS {
            return Err(());
        }
        let tokens = scripts.get(calls.len()).copied().unwrap_or(&filler);
        calls.push((seek, window_samples));
        Ok(plan_window(tokens, window_samples).advance_samples)
    });
    assert_eq!(
        result,
        Ok(WalkEnd::Finished),
        "walk did not end; windows: {calls:?}"
    );
    calls
}

/// Two sentences in 5.4 s with one timestamp at the end: one window.
#[test]
fn clip_with_a_single_ending_is_one_window() {
    let clip = [ts(0.0), A, A, ts(2.28), ts(2.66), B, B, ts(5.32)];
    assert_eq!(walk(86_551, &[&clip]), [(0, 86_551)]);
}

/// A window that ends in a pair at 5.40 s leaves 151 samples, less than one mel
/// hop (the reference counts 540 whole frames in 86,551 samples): one window.
#[test]
fn pair_ending_less_than_a_hop_before_the_end_is_one_window() {
    let clip = [ts(0.0), A, ts(2.0), ts(2.5), B, ts(5.40), ts(5.40)];
    assert_eq!(walk(86_551, &[&clip]), [(0, 86_551)]);
}

/// 30 s and 5 ms: after the full window, 80 samples are left, less than one hop.
#[test]
fn input_of_a_window_plus_five_milliseconds_is_one_window() {
    let window = [ts(0.0), A, ts(29.40)];
    assert_eq!(walk(480_080, &[&window]), [(0, 480_000)]);
}

/// Exactly one hop of audio opens a window; one sample less does not.
#[test]
fn walk_opens_a_window_from_one_hop_of_audio() {
    assert_eq!(walk(159, &[]), []);
    assert_eq!(walk(160, &[]), [(0, 160)]);
    assert_eq!(walk(0, &[]), []);
}

/// 40 s, the first window ends in one timestamp at 29.40 s: the second window
/// starts at 30 s, not at 29.40 s.
#[test]
fn single_ending_at_29_4_starts_the_next_window_at_30_s() {
    let first = [ts(0.0), A, ts(29.40)];
    let second = [ts(0.0), B, ts(9.98)];
    assert_eq!(
        walk(640_000, &[&first, &second]),
        [(0, 480_000), (480_000, 160_000)]
    );
}

/// 40 s, the first window ends in a pair at 12.0 s: the second window starts
/// at 12.0 s.
#[test]
fn pair_at_12_s_starts_the_next_window_at_12_s() {
    let first = [ts(0.0), A, ts(12.0), ts(12.0)];
    let second = [ts(0.0), B, ts(27.9)];
    assert_eq!(
        walk(640_000, &[&first, &second]),
        [(0, 480_000), (192_000, 448_000)]
    );
}

/// A window that reports an advance of 0 is consumed whole, so the walk ends.
#[test]
fn walk_consumes_a_window_that_reports_no_advance() {
    let mut calls: Vec<(usize, usize)> = Vec::new();
    let never = AtomicBool::new(false);
    let result = walk_windows(640_000, &never, |seek, window_samples| {
        if calls.len() >= MAX_SCRIPTED_WINDOWS {
            return Err(());
        }
        calls.push((seek, window_samples));
        Ok(0)
    });
    assert_eq!(result, Ok(WalkEnd::Finished));
    assert_eq!(calls, [(0, 480_000), (480_000, 160_000)]);
}

/// An error from a window stops the walk and is returned.
#[test]
fn walk_stops_on_the_first_error() {
    let mut calls = 0;
    let never = AtomicBool::new(false);
    let result = walk_windows(640_000, &never, |_, _| {
        calls += 1;
        Err("decode failed")
    });
    assert_eq!(result, Err("decode failed"));
    assert_eq!(calls, 1);
}

/// One hour of audio, each window answered from a cycle of the seek rules: the
/// windows tile the audio with no gap and no overlap the rules did not ask for,
/// every window is full until the tail, and the walk ends.
#[test]
fn an_hour_of_windows_tiles_the_audio_and_ends() {
    const HOUR: usize = 3_600 * 16_000;
    let single_ending = [ts(0.0), A, ts(10.0), ts(10.5), B, ts(29.4)];
    let closed_at_12_s = [ts(0.0), A, ts(12.0), ts(12.4), B];
    let no_pair = [ts(0.0), A, B];
    let zero_fallback = [ts(0.0), ts(0.0)];
    let closed_at_one_step = [ts(0.0), A, ts(0.02), ts(0.02)];
    let cycle: [&[u32]; 5] = [
        &single_ending,
        &closed_at_12_s,
        &no_pair,
        &zero_fallback,
        &closed_at_one_step,
    ];
    // Every advance is at least one timestamp step (320 samples), so no walk
    // that ends needs more windows than this.
    let max_windows = HOUR / 320;
    let mut windows: Vec<(usize, usize, usize)> = Vec::new();
    let never = AtomicBool::new(false);
    let result = walk_windows(HOUR, &never, |seek, window_samples| {
        if windows.len() >= max_windows {
            return Err(());
        }
        let tokens = cycle
            .get(windows.len() % cycle.len())
            .copied()
            .unwrap_or(&[]);
        let advance = plan_window(tokens, window_samples).advance_samples;
        windows.push((seek, window_samples, advance));
        Ok(advance)
    });
    assert_eq!(
        result,
        Ok(WalkEnd::Finished),
        "walk did not end after {} windows",
        windows.len()
    );
    assert!(
        windows.len() > 100,
        "an hour is many windows: {}",
        windows.len()
    );
    assert_eq!(windows.first().map(|w| w.0), Some(0));
    for pair in windows.windows(2) {
        if let [(seek, _, advance), (next_seek, _, _)] = pair {
            assert_eq!(*next_seek, seek + advance, "window after {pair:?}");
        }
    }
    for &(seek, window_samples, advance) in &windows {
        assert_eq!(window_samples, (HOUR - seek).min(SAMPLES_PER_WINDOW));
        assert!(
            advance > 0 && advance <= window_samples,
            "{seek}: {advance}"
        );
    }
    if let Some(&(seek, _, advance)) = windows.last() {
        assert!(
            HOUR - (seek + advance) < HOP_LENGTH,
            "audio left after the walk: {}",
            HOUR - (seek + advance)
        );
    }
}

// ── Window segments: times from the start of the audio ───────────────────────

/// A window that starts at 30 s (sample 480000, mel frame 3000): each segment
/// time is 30 s plus its time in the window, the text is decoded from the
/// text tokens only and trimmed, a segment with blank text is dropped, and
/// each kept segment carries the window's frame, its own slice, and the
/// temperature and the compression ratio of the decode the window kept.
#[test]
fn window_segments_are_offset_by_the_window_start() {
    let raw = [
        seg(0.0, 2.28, &[ts(0.0), 1, ts(2.28)]),
        seg(2.66, 5.32, &[ts(2.66), 2, ts(5.32)]),
        seg(6.0, 7.0, &[ts(6.0), 3, ts(7.0)]),
    ];
    let text = |tokens: &[u32]| match tokens {
        [1] => " First sentence. ".to_owned(),
        [2] => "Second sentence.".to_owned(),
        [3] => "  ".to_owned(),
        other => panic!("decoded tokens other than one segment's text: {other:?}"),
    };
    let kept = Attempt {
        temperature: 0.4,
        with_prompt: false,
        compression_ratio: 1.25,
        avg_logprob: -0.3,
        verdict: Verdict::Accepted,
    };
    let segments = window_segments(&raw, 480_000, 960_000, &kept, text);
    for (segment, raw) in segments.iter().zip(&raw) {
        assert_eq!(segment.seek, 3000);
        assert_eq!(segment.tokens, raw.tokens);
        assert!((segment.temperature - 0.4).abs() < f32::EPSILON);
        assert!((segment.compression_ratio - 1.25).abs() < f32::EPSILON);
    }
    let got: Vec<(f32, f32, &str)> = segments
        .iter()
        .map(|s| (s.start, s.end, s.text.as_str()))
        .collect();
    let want = [
        (30.0, 32.28, "First sentence."),
        (32.66, 35.32, "Second sentence."),
    ];
    assert_eq!(got.len(), want.len(), "{got:?}");
    for ((start, end, text), (w_start, w_end, w_text)) in got.iter().zip(want) {
        assert!((start - w_start).abs() < 1e-4, "{got:?}");
        assert!((end - w_end).abs() < 1e-4, "{got:?}");
        assert_eq!(*text, w_text);
    }
}

// ── Window walk: the cancel flag, set from inside a scripted window ──────────

/// Four full windows of audio.
const FOUR_WINDOWS: usize = 4 * SAMPLES_PER_WINDOW;

/// Walk `total_samples`, each window advancing by `advance(window index,
/// window_samples)`, and set the cancel flag while window `cancel_in` decodes.
/// Returns the seeks of the windows the walk opened and how it ended.
fn walk_cancelled_in(
    total_samples: usize,
    cancel_in: usize,
    advance: impl Fn(usize, usize) -> usize,
) -> (Vec<usize>, Result<WalkEnd, ()>) {
    let cancel = AtomicBool::new(false);
    let mut seeks: Vec<usize> = Vec::new();
    let end = walk_windows(total_samples, &cancel, |seek, window_samples| {
        if seeks.len() >= MAX_SCRIPTED_WINDOWS {
            return Err(());
        }
        if seeks.len() == cancel_in {
            cancel.store(true, Ordering::Relaxed);
        }
        let step = advance(seeks.len(), window_samples);
        seeks.push(seek);
        Ok(step)
    });
    (seeks, end)
}

/// The flag set while window 1 decodes: window 1 ends, window 2 never opens.
#[test]
fn a_flag_set_in_window_1_stops_the_walk_before_window_2() {
    let (seeks, end) = walk_cancelled_in(FOUR_WINDOWS, 1, |_, whole| whole);
    assert_eq!(seeks, [0, SAMPLES_PER_WINDOW]);
    assert_eq!(
        end,
        Ok(WalkEnd::Cancelled {
            window: 2,
            seek: 2 * SAMPLES_PER_WINDOW
        })
    );
}

/// A flag that is set before the walk starts opens no window.
#[test]
fn a_flag_set_before_the_walk_opens_no_window() {
    let cancel = AtomicBool::new(true);
    let mut opened = 0;
    let end: Result<WalkEnd, ()> = walk_windows(FOUR_WINDOWS, &cancel, |_, whole| {
        opened += 1;
        Ok(whole)
    });
    assert_eq!(opened, 0);
    assert_eq!(end, Ok(WalkEnd::Cancelled { window: 0, seek: 0 }));
}

/// The flag set in the last window: no window is left to skip, so the walk
/// ends as finished.
#[test]
fn a_flag_set_in_the_last_window_ends_the_walk_finished() {
    let (seeks, end) = walk_cancelled_in(FOUR_WINDOWS, 3, |_, whole| whole);
    assert_eq!(seeks.len(), 4);
    assert_eq!(end, Ok(WalkEnd::Finished));
}

/// A window that seeks back to a closed segment at 12 s: the window that is
/// not opened starts at the planned seek, not at a whole-window step.
#[test]
fn a_cancelled_walk_reports_the_planned_seek() {
    let twelve_s = 12 * 16_000;
    let (seeks, end) = walk_cancelled_in(FOUR_WINDOWS, 0, |_, _| twelve_s);
    assert_eq!(seeks, [0]);
    assert_eq!(
        end,
        Ok(WalkEnd::Cancelled {
            window: 1,
            seek: twelve_s
        })
    );
}
