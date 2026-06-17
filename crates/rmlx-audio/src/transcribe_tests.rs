//! Unit tests for the long-form transcription engine helpers (no model needed).

use super::{
    fmt_time, previous_text_cap, render_srt, render_vtt, resample_to_16k, window_token_budget,
    OutputFormat, Segment,
};

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
    assert!(resample_to_16k(&[], 48_000).is_empty());
}
