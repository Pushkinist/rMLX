//! Unit tests for the long-form transcription engine helpers (no model needed).

use super::{
    compression_ratio, decode_window, fmt_time, next_prompt, plan_window, previous_text_cap,
    render_srt, render_vtt, resample_to_16k, walk_windows, window_token_budget, OutputFormat,
    RawSegment, SeekRule, Segment, WindowDecode, COMPRESSION_RATIO_LOOP,
};
use crate::tokenizer::TOK_TIMESTAMP_BEGIN;

/// Tokens prepended by the seek loop before the previous-text prompt: the
/// `<|startofprev|>` marker (1) plus the timestamp-mode SOT prefix
/// `[<|sot|>, <lang>, <transcribe>]` (3).
const SOT_PREV_MARKER: usize = 1;
const SOT_PREFIX_LEN: usize = 3;

/// Regression for the positional-embedding overflow: a window with a FULL
/// previous-text prompt + the largest generation budget must never request a
/// decoder position `>= n_text_ctx`, otherwise the positional-embedding slice
/// `[offset, offset+seq)` runs off the `[n_text_ctx, n_state]` table and aborts
/// the whole transcription. Model-free: it exercises the runtime bound formulas
/// directly for every realistic `n_text_ctx`.
#[test]
fn decode_budget_never_overruns_positional_table() {
    // large-v3 is 448; tiny is 448 too (all Whisper variants share n_text_ctx=448),
    // but sweep a range of realistic context lengths to keep the bound general.
    // Values below ~8 are degenerate (the fixed 4-token SOT_PREV+SOT prefix alone
    // exceeds them) and no real Whisper config uses them.
    for &n_text_ctx in &[8usize, 16, 64, 256, 447, 448, 449, 1024] {
        // Worst case: the previous-text prompt is filled to its cap.
        let prompt_cap = previous_text_cap(n_text_ctx);
        let prefix_len = SOT_PREV_MARKER + prompt_cap + SOT_PREFIX_LEN;

        // The prefix (SOT_PREV + capped prompt + SOT prefix) must itself fit so the
        // prefill rows `[0, prefix_len)` stay on the positional table.
        assert!(
            prefix_len <= n_text_ctx,
            "n_text_ctx={n_text_ctx}: prefix_len={prefix_len} overruns the table on prefill"
        );

        // The decoder offset starts at `prefix_len` (prefill) and grows by one per
        // generated token. The single largest positional row ever requested is
        // `prefix_len + generated - 1`.
        let max_tokens = window_token_budget(n_text_ctx, prefix_len);
        let max_offset_row = prefix_len + max_tokens; // prefill rows [0,prefix_len) + max_tokens steps

        assert!(
            max_offset_row <= n_text_ctx,
            "n_text_ctx={n_text_ctx}: prefix_len={prefix_len} + max_tokens={max_tokens} \
             = {max_offset_row} exceeds n_text_ctx — positional-embedding overflow"
        );
        // The largest *row index* requested is `max_offset_row - 1` and must be a
        // valid index into the `[n_text_ctx, n_state]` table (i.e. `< n_text_ctx`),
        // whenever any token is generated.
        if max_tokens > 0 {
            assert!(
                max_offset_row - 1 < n_text_ctx,
                "n_text_ctx={n_text_ctx}: last positional row {} is out of bounds",
                max_offset_row - 1
            );
        }
    }
}

/// Even without a previous-text prompt (prefix = SOT prefix only), the generation
/// budget must keep the decoder position in bounds.
#[test]
fn decode_budget_no_prompt_in_bounds() {
    for &n_text_ctx in &[4usize, 448, 1024] {
        let prefix_len = SOT_PREFIX_LEN; // no SOT_PREV / prompt on the first window
        let max_tokens = window_token_budget(n_text_ctx, prefix_len);
        assert!(prefix_len + max_tokens <= n_text_ctx, "ctx={n_text_ctx}");
    }
}

/// The budget collapses to zero (rather than underflowing) when the prefix already
/// fills the context — the seek loop then emits nothing for that window instead of
/// crashing.
#[test]
fn decode_budget_saturates_when_prefix_full() {
    assert_eq!(window_token_budget(448, 448), 0);
    assert_eq!(window_token_budget(448, 1000), 0);
    assert_eq!(previous_text_cap(0), 0);
    assert_eq!(previous_text_cap(1), 0);
    assert_eq!(previous_text_cap(448), 223);
}

#[test]
fn output_format_parse() {
    assert_eq!(OutputFormat::parse("txt").unwrap(), OutputFormat::Txt);
    assert_eq!(OutputFormat::parse("text").unwrap(), OutputFormat::Txt);
    assert_eq!(OutputFormat::parse("json").unwrap(), OutputFormat::Json);
    assert_eq!(OutputFormat::parse("srt").unwrap(), OutputFormat::Srt);
    assert_eq!(OutputFormat::parse("vtt").unwrap(), OutputFormat::Vtt);
    assert!(OutputFormat::parse("flac").is_err());
}

#[test]
fn time_formatting_srt_vtt() {
    // 1h 2m 3.456s.
    let secs = 3600.0 + 120.0 + 3.456;
    assert_eq!(fmt_time(secs, true), "01:02:03,456");
    assert_eq!(fmt_time(secs, false), "01:02:03.456");
    // Zero.
    assert_eq!(fmt_time(0.0, false), "00:00:00.000");
    // Negative clamps to zero.
    assert_eq!(fmt_time(-5.0, false), "00:00:00.000");
}

#[test]
fn srt_vtt_multi_segment() {
    let segs = vec![
        Segment {
            start: 0.0,
            end: 2.5,
            text: "hello world".to_owned(),
        },
        Segment {
            start: 2.5,
            end: 5.0,
            text: "second line".to_owned(),
        },
    ];
    let srt = render_srt(&segs);
    assert!(srt.contains("1\n00:00:00,000 --> 00:00:02,500\nhello world"));
    assert!(srt.contains("2\n00:00:02,500 --> 00:00:05,000\nsecond line"));

    let vtt = render_vtt(&segs);
    assert!(vtt.starts_with("WEBVTT\n\n"));
    assert!(vtt.contains("00:00:00.000 --> 00:00:02.500\nhello world"));
    assert!(vtt.contains("00:00:02.500 --> 00:00:05.000\nsecond line"));
}

#[test]
fn resample_identity_when_already_16k() {
    let s = vec![0.1_f32, 0.2, 0.3, 0.4];
    assert_eq!(resample_to_16k(&s, 16_000), s);
}

#[test]
fn resample_downsamples_48k_to_16k() {
    // 48k -> 16k should produce ~1/3 the samples.
    let s: Vec<f32> = (0..4800).map(|i| i as f32 / 4800.0).collect();
    let out = resample_to_16k(&s, 48_000);
    // 4800 * 16000/48000 = 1600.
    assert!((out.len() as i64 - 1600).abs() <= 1, "len = {}", out.len());
    // Monotone ramp preserved roughly.
    assert!(out.first().copied().unwrap_or(1.0) < 0.05);
    assert!(out.last().copied().unwrap_or(0.0) > 0.9);
}

#[test]
fn resample_empty() {
    assert_eq!(resample_to_16k(&[], 48_000), Vec::<f32>::new());
}

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
// Not part of the reference: a segment that opens more than 0.5 s past the end
// of a short window is dropped (see `segment_opening_past_the_real_audio_is_dropped`).

const SAMPLES_PER_WINDOW: usize = 480_000;
const A: u32 = 1_000;
const B: u32 = 1_001;

fn ts(secs: f32) -> u32 {
    TOK_TIMESTAMP_BEGIN + (secs / 0.02).round() as u32
}

fn seg(start: f32, end: f32, tokens: &[u32]) -> RawSegment {
    RawSegment {
        start,
        end,
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
            (got.start - want.start).abs() < 1e-3 && (got.end - want.end).abs() < 1e-3,
            "segment times: got {got:?}, want {want:?}"
        );
        assert_eq!(got.tokens, want.tokens);
    }
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

/// Not part of the reference rule: a segment that opens more than 0.5 s past
/// the real end of a short window sits in the zero-pad tail and is dropped, to
/// stop filler there. A segment that opens inside that margin is kept.
#[test]
fn segment_opening_past_the_real_audio_is_dropped() {
    assert_plan(
        &[ts(0.0), A, ts(0.5), ts(2.0), B, ts(2.5)],
        16_000,
        &[seg(0.0, 0.5, &[A])],
        16_000,
    );
    assert_plan(
        &[ts(0.0), A, ts(0.5), ts(1.4), B, ts(1.6)],
        16_000,
        &[seg(0.0, 0.5, &[A]), seg(1.4, 1.6, &[B])],
        16_000,
    );
}

/// The margin is exactly 0.5 s: a segment opening at the real end plus 0.5 s is
/// kept, one step later is dropped.
#[test]
fn pad_tail_guard_margin_is_half_a_second() {
    assert_plan(
        &[ts(0.0), A, ts(0.5), ts(1.5), B, ts(1.8)],
        16_000,
        &[seg(0.0, 0.5, &[A]), seg(1.5, 1.8, &[B])],
        16_000,
    );
    assert_plan(
        &[ts(0.0), A, ts(0.5), ts(1.52), B, ts(1.8)],
        16_000,
        &[seg(0.0, 0.5, &[A])],
        16_000,
    );
    // 16,100 is not a multiple of 320: the limit is 24,100 samples, and a start
    // at 1.52 s is 24,320.
    assert_plan(
        &[ts(0.0), A, ts(0.5), ts(1.52), B, ts(1.8)],
        16_100,
        &[seg(0.0, 0.5, &[A])],
        16_100,
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

// ── Window walk: which windows are opened, from scripted token streams ───────

const MAX_SCRIPTED_WINDOWS: usize = 8;

/// Run `walk_windows` over `total_samples`, answering window `k` with
/// `scripts[k]`. Any window past the script gets a one-timestamp filler stream.
/// Returns every `(seek, window_samples)` the walk asked for.
fn walk(total_samples: usize, scripts: &[&[u32]]) -> Vec<(usize, usize)> {
    let filler = [ts(0.0), A, ts(0.04)];
    let mut calls: Vec<(usize, usize)> = Vec::new();
    let result: Result<(), ()> = walk_windows(total_samples, |seek, window_samples| {
        // No scripted walk opens this many windows: a walk that does not end
        // fails here and does not hang the test binary.
        if calls.len() >= MAX_SCRIPTED_WINDOWS {
            return Err(());
        }
        let tokens = scripts.get(calls.len()).copied().unwrap_or(&filler);
        calls.push((seek, window_samples));
        Ok(plan_window(tokens, window_samples).advance_samples)
    });
    assert_eq!(result, Ok(()), "walk did not end; windows: {calls:?}");
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
    let result: Result<(), ()> = walk_windows(640_000, |seek, window_samples| {
        if calls.len() >= MAX_SCRIPTED_WINDOWS {
            return Err(());
        }
        calls.push((seek, window_samples));
        Ok(0)
    });
    assert_eq!(result, Ok(()));
    assert_eq!(calls, [(0, 480_000), (480_000, 160_000)]);
}

/// An error from a window stops the walk and is returned.
#[test]
fn walk_stops_on_the_first_error() {
    let mut calls = 0;
    let result = walk_windows(640_000, |_, _| {
        calls += 1;
        Err("decode failed")
    });
    assert_eq!(result, Err("decode failed"));
    assert_eq!(calls, 1);
}

// ── Window decode: the repetition guard, from scripted decodes ───────────────

/// Sentence `k` is the text of token `k`; timestamps have no text.
const SENTENCES: [&str; 8] = [
    "Thanks for joining the review today.",
    "First we look at the release notes.",
    "The build passed on every machine last night.",
    "Next, the team checks the open questions.",
    "Two of them need a decision from the owner.",
    "We will meet again on Friday.",
    "Please send your notes before then.",
    "And then we write the summary for the next item.",
];
const REPEATED: u32 = 7;
const PROMPT: [u32; 2] = [3, 4];

fn sentences(tokens: &[u32]) -> String {
    tokens
        .iter()
        .filter_map(|&t| SENTENCES.get(t as usize).copied())
        .collect::<Vec<_>>()
        .join(" ")
}

/// Four varied sentences, then one sentence `repeats` times, with no closing
/// timestamp: the shape of a greedy repetition loop.
fn looping(repeats: usize) -> Vec<u32> {
    let mut tokens = vec![TOK_TIMESTAMP_BEGIN, 0, 1, 2, 3];
    tokens.extend(std::iter::repeat_n(REPEATED, repeats));
    tokens
}

fn clean() -> Vec<u32> {
    vec![
        TOK_TIMESTAMP_BEGIN,
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        TOK_TIMESTAMP_BEGIN + 1450,
    ]
}

/// Run `decode_window` with a decoder that returns `with_prompt` when it is
/// given a prompt and `without_prompt` when it is not. Returns the result and
/// every prompt the decoder was given.
fn guarded(
    prompt: &[u32],
    with_prompt: &[u32],
    without_prompt: &[u32],
) -> (WindowDecode, Vec<Vec<u32>>) {
    let mut calls: Vec<Vec<u32>> = Vec::new();
    let result: Result<WindowDecode, ()> = decode_window(
        prompt,
        |p: &[u32]| {
            calls.push(p.to_vec());
            Ok(if p.is_empty() {
                without_prompt
            } else {
                with_prompt
            }
            .to_vec())
        },
        sentences,
    );
    match result {
        Ok(decoded) => (decoded, calls),
        Err(()) => panic!("scripted decode failed; calls: {calls:?}"),
    }
}

/// The kept tokens, whether they may prompt the next window, and whether a
/// decode without the prompt ran.
fn kept(decoded: WindowDecode) -> (Vec<u32>, bool, bool) {
    (decoded.tokens, decoded.carry_text, decoded.retried)
}

/// A window that loops under the previous-text prompt is decoded again without
/// it; the clean result is kept and may prompt the next window.
#[test]
fn a_loop_under_the_prompt_is_decoded_again_without_it() {
    let (decoded, calls) = guarded(&PROMPT, &looping(8), &clean());
    assert_eq!(calls, [PROMPT.to_vec(), Vec::new()]);
    assert_eq!(kept(decoded), (clean(), true, true));
}

/// A window that does not loop is decoded once, with its prompt.
#[test]
fn a_clean_window_is_decoded_once() {
    let (decoded, calls) = guarded(&PROMPT, &clean(), &looping(8));
    assert_eq!(calls, [PROMPT.to_vec()]);
    assert_eq!(kept(decoded), (clean(), true, false));
}

/// A loop that remains without the prompt is kept, but it does not prompt the
/// next window.
#[test]
fn a_loop_that_remains_without_the_prompt_is_not_carried() {
    let (decoded, calls) = guarded(&PROMPT, &looping(8), &looping(6));
    assert_eq!(calls, [PROMPT.to_vec(), Vec::new()]);
    assert_eq!(kept(decoded), (looping(6), false, true));
}

/// A loop decoded with no prompt has no prompt to drop: it is decoded once and
/// not carried.
#[test]
fn a_loop_with_no_prompt_is_decoded_once_and_not_carried() {
    let (decoded, calls) = guarded(&[], &clean(), &looping(8));
    assert_eq!(calls, [Vec::<u32>::new()]);
    assert_eq!(kept(decoded), (looping(8), false, false));
}

/// The loop bound is the reference compression ratio, 2.4: one sentence four
/// times after four others compresses to about 2.3 and is kept; five times, to
/// about 2.6, and is decoded again.
#[test]
fn the_loop_bound_is_a_compression_ratio_of_2_4() {
    let (_, calls) = guarded(&PROMPT, &looping(4), &clean());
    assert_eq!(calls.len(), 1, "four repeats are not a loop");
    let (_, calls) = guarded(&PROMPT, &looping(5), &clean());
    assert_eq!(calls.len(), 2, "five repeats are a loop");
}

/// A decode error on the second decode is returned.
#[test]
fn an_error_on_the_decode_without_the_prompt_is_returned() {
    let mut calls = 0;
    let result = decode_window(
        &PROMPT,
        |p: &[u32]| {
            calls += 1;
            if p.is_empty() {
                Err("decode failed")
            } else {
                Ok(looping(8))
            }
        },
        sentences,
    );
    assert_eq!(result, Err("decode failed"));
    assert_eq!(calls, 2);
}

/// When both decodes loop, the one that compresses less is kept.
#[test]
fn when_both_decodes_loop_the_one_with_the_lower_ratio_is_kept() {
    let (decoded, calls) = guarded(&PROMPT, &looping(6), &looping(8));
    assert_eq!(calls, [PROMPT.to_vec(), Vec::new()]);
    assert_eq!(kept(decoded), (looping(6), false, true));
}

// ── The compression ratio at the loop bound ──────────────────────────────────

/// The first 372 bytes of the five-repeat text: 372 bytes over a 155-byte zlib
/// stream, a ratio of exactly 2.4.
fn text_at_the_bound() -> String {
    let text = sentences(&looping(5));
    text.get(..372).unwrap_or_default().to_owned()
}

#[test]
#[allow(clippy::float_cmp, reason = "372 / 155 is exactly the f32 nearest 2.4")]
fn compression_ratio_is_zlib_bytes_over_trimmed_text_bytes() {
    let at_bound = text_at_the_bound();
    assert_eq!(at_bound.len(), 372);
    assert_eq!(compression_ratio(&at_bound), 2.4);
    assert_eq!(compression_ratio(&format!("  {at_bound}\n")), 2.4);
    let one_more = sentences(&looping(5))
        .get(..373)
        .unwrap_or_default()
        .to_owned();
    assert!(compression_ratio(&one_more) > COMPRESSION_RATIO_LOOP);
    assert_eq!(compression_ratio(""), 0.0);
}

/// A ratio of exactly 2.4 is not a loop; the next byte makes one.
#[test]
fn a_ratio_of_exactly_2_4_is_not_a_loop() {
    let text = sentences(&looping(5));
    for (len, loops) in [(372, false), (373, true)] {
        let window = text.get(..len).unwrap_or_default().to_owned();
        let mut calls = 0;
        let result: Result<WindowDecode, ()> = decode_window(
            &PROMPT,
            |_: &[u32]| {
                calls += 1;
                Ok(vec![calls])
            },
            |tokens: &[u32]| {
                if tokens == [1] {
                    window.clone()
                } else {
                    sentences(&clean())
                }
            },
        );
        assert!(result.is_ok());
        assert_eq!(calls, if loops { 2 } else { 1 }, "{len} bytes");
    }
}

// ── The prompt for the next window ───────────────────────────────────────────

/// Encodes text as one token per byte, so the cap is easy to read.
fn bytes_as_tokens(text: &str) -> Vec<u32> {
    text.bytes().map(u32::from).collect()
}

/// A window whose text may not be carried clears the prompt.
#[test]
fn a_window_that_may_not_be_carried_clears_the_prompt() {
    let next = next_prompt(PROMPT.to_vec(), false, true, "loop", bytes_as_tokens, 10);
    assert_eq!(next, Vec::<u32>::new());
}

/// With conditioning on, the window's trimmed text replaces the prompt, cut to
/// its last `cap` tokens.
#[test]
fn a_carried_window_replaces_the_prompt_cut_to_the_cap() {
    let next = next_prompt(PROMPT.to_vec(), true, true, " abcdef ", bytes_as_tokens, 4);
    assert_eq!(next, bytes_as_tokens("cdef"));
    let next = next_prompt(PROMPT.to_vec(), true, true, "ab", bytes_as_tokens, 4);
    assert_eq!(next, bytes_as_tokens("ab"));
}

/// With conditioning off, or a window with no text, the prompt is kept.
#[test]
fn the_prompt_is_kept_with_conditioning_off_or_no_text() {
    let next = next_prompt(PROMPT.to_vec(), true, false, "abc", bytes_as_tokens, 4);
    assert_eq!(next, PROMPT.to_vec());
    let next = next_prompt(PROMPT.to_vec(), true, true, "  ", bytes_as_tokens, 4);
    assert_eq!(next, PROMPT.to_vec());
}

/// A retry whose ratio is exactly 2.4 is not a loop: it is kept and prompts the
/// next window.
#[test]
fn a_retry_at_a_ratio_of_exactly_2_4_is_carried() {
    let text = sentences(&looping(5));
    let at_bound = text.get(..372).unwrap_or_default().to_owned();
    let one_more = text.get(..373).unwrap_or_default().to_owned();
    let result: Result<WindowDecode, ()> = decode_window(
        &PROMPT,
        |p: &[u32]| Ok(vec![u32::from(p.is_empty())]),
        |tokens: &[u32]| {
            if tokens == [1] {
                at_bound.clone()
            } else {
                one_more.clone()
            }
        },
    );
    let Ok(decoded) = result else {
        panic!("scripted decode failed");
    };
    assert_eq!(kept(decoded), (vec![1], true, true));
}

/// When both decodes loop with the same ratio, the retry is kept.
#[test]
fn of_two_loops_with_the_same_ratio_the_retry_is_kept() {
    let mut retry = looping(8);
    retry.push(TOK_TIMESTAMP_BEGIN + 1450);
    let (decoded, _) = guarded(&PROMPT, &looping(8), &retry);
    assert_eq!(kept(decoded), (retry, false, true));
}
