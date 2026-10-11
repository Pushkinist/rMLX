//! Unit tests for the long-form transcription engine helpers (no model needed).

use super::{
    compression_ratio, decode_window, fmt_time, next_prompt, previous_text_cap, render_srt,
    render_vtt, reported_language, requested_language, resolve_language, window_token_budget,
    OutputFormat, Segment, WindowDecode, COMPRESSION_RATIO_LOOP,
};
use crate::tokenizer::{TOK_EN, TOK_TIMESTAMP_BEGIN};
use crate::whisper::WhisperError;

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
            seek: 0,
            start: 0.0,
            end: 2.5,
            text: "hello world".to_owned(),
            tokens: Vec::new(),
            compression_ratio: 1.0,
        },
        Segment {
            seek: 0,
            start: 2.5,
            end: 5.0,
            text: "second line".to_owned(),
            tokens: Vec::new(),
            compression_ratio: 1.0,
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

/// `auto` asks for detection; a code or an English name, in any case, names a
/// token.
#[test]
fn a_requested_language_is_auto_or_a_token() {
    for auto in ["auto", "AUTO", "Auto"] {
        assert!(matches!(requested_language(auto), Ok(None)), "{auto}");
    }
    for (spelling, token) in [("en", TOK_EN), ("EN", TOK_EN), ("French", TOK_EN + 6)] {
        assert!(
            matches!(requested_language(spelling), Ok(Some(t)) if t == token),
            "{spelling}"
        );
    }
}

/// A language Whisper does not know is an error that names the value; it is
/// never English.
#[test]
fn an_unknown_requested_language_is_an_error_naming_it() {
    for unknown in ["xx", "klingon", "", "en-US"] {
        let error = requested_language(unknown).expect_err(unknown);
        assert!(
            matches!(&error, WhisperError::UnknownLanguage(value) if value == unknown),
            "{error:?}"
        );
        assert!(
            error.to_string().contains(&format!("'{unknown}'")),
            "{error}"
        );
    }
}

/// The language a transcription reports is the English name of the token the
/// decode used, whether the request named it or the model detected it. The
/// text `lang_tok=` is in no report.
#[test]
fn the_reported_language_is_the_english_name_of_the_token() {
    assert_eq!(reported_language(Some(TOK_EN)), "english");
    assert_eq!(reported_language(Some(TOK_EN + 6)), "french");
    assert_eq!(reported_language(Some(TOK_EN + 99)), "cantonese");
    for token in TOK_EN..=TOK_EN + 99 {
        assert!(!reported_language(Some(token)).contains("lang_tok"));
    }
}

/// The first window detects the language and stores it; a later window reads
/// the stored token and does not detect again.
#[test]
fn the_first_window_decides_the_language_of_the_file() {
    let mut resolved = requested_language("auto").unwrap();
    let mut detections = 0;
    let mut detect = |token| {
        detections += 1;
        Ok(token)
    };
    assert_eq!(
        resolve_language(&mut resolved, || detect(TOK_EN + 6)).unwrap(),
        TOK_EN + 6
    );
    assert_eq!(resolved, Some(TOK_EN + 6));
    assert_eq!(
        resolve_language(&mut resolved, || detect(TOK_EN + 1)).unwrap(),
        TOK_EN + 6
    );
    assert_eq!(detections, 1);
    assert_eq!(reported_language(resolved), "french");
}

/// A named language is never detected.
#[test]
fn a_named_language_is_not_detected() {
    let mut resolved = requested_language("Russian").unwrap();
    let got = resolve_language(&mut resolved, || panic!("a named language is not detected"));
    assert_eq!(got.unwrap(), TOK_EN + 4);
    assert_eq!(reported_language(resolved), "russian");
}

/// A failed detection fails the window; no English is stored.
#[test]
fn a_failed_detection_is_an_error_and_stores_nothing() {
    let mut resolved = None;
    let got = resolve_language(&mut resolved, || {
        Err(WhisperError::Mlx("op failed".to_owned()))
    });
    assert!(matches!(got, Err(WhisperError::Mlx(_))), "{got:?}");
    assert_eq!(resolved, None);
}

/// With `auto` and audio shorter than one mel hop, no window runs and no token
/// exists; the report names english.
#[test]
fn a_report_with_no_detected_language_names_english() {
    assert_eq!(reported_language(None), "english");
}
