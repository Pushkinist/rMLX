//! The word-timestamp steps on made inputs. Each expected value is worked out
//! by hand from openai-whisper's `whisper/timing.py` and `whisper/tokenizer.py`.

#![allow(
    clippy::indexing_slicing,
    clippy::float_cmp,
    reason = "test assertions on made inputs of known size and exact values"
)]

use super::*;
use crate::tokenizer::TOK_TIMESTAMP_BEGIN;

fn close(found: &[f32], expected: &[f32]) -> bool {
    found.len() == expected.len()
        && found
            .iter()
            .zip(expected)
            .all(|(f, e)| (f - e).abs() < 1e-5)
}

fn timing(word: &str, tokens: &[u32], start: f64, end: f64) -> WordTiming {
    WordTiming {
        word: word.to_owned(),
        tokens: tokens.to_vec(),
        start,
        end,
    }
}

fn texts(words: &[WordTiming]) -> Vec<&str> {
    words.iter().map(|w| w.word.as_str()).collect()
}

// ── DTW ─────────────────────────────────────────────────────────────────────

/// A ridge on the diagonal, with a cost of 1 off it, is walked on the
/// diagonal.
#[test]
fn a_diagonal_ridge_gives_the_diagonal_path() {
    let x = [-1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0];
    assert_eq!(dtw(&x, 3, 3), [(1, 1), (2, 2), (3, 3)]);
}

/// Two text rows, the first strong on frames 0 and 1, the second on frames 2
/// and 3: the path stays in row 1 for two frames, then steps down on the
/// diagonal into row 2.
#[test]
fn two_blocks_give_a_staircase() {
    let x = [-1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, -1.0];
    let path = dtw(&x, 2, 4);
    assert_eq!(path, [(1, 1), (1, 2), (2, 3), (2, 4)]);
    assert_eq!(jump_times(&path), [0.0, 0.04]);
}

/// At cell (2, 2) the diagonal and the cell above both cost 0 and the cell to
/// the left costs 1. No cost is strictly least, so the move comes from the
/// left, as openai-whisper's `else` branch takes it, and the path goes down
/// column 1 first.
#[test]
fn a_tie_goes_to_the_cell_on_the_left() {
    let x = [0.0, 0.0, 1.0, 0.0];
    assert_eq!(dtw(&x, 2, 2), [(1, 1), (2, 1), (2, 2)]);
}

/// Row 0 of the moves is read as moves from the left and column 0 as moves
/// from above, whatever the grid held there.
#[test]
fn the_backtrace_walks_left_on_row_0_and_up_on_column_0() {
    let (rows, cols) = (2, 3);
    let mut up_to_row_0 = vec![Step::Diagonal; (rows + 1) * (cols + 1)];
    up_to_row_0[2 * 4 + 3] = Step::Up;
    up_to_row_0[4 + 3] = Step::Up;
    assert_eq!(
        backtrace(&mut up_to_row_0, rows, cols),
        [(0, 1), (0, 2), (0, 3), (1, 3), (2, 3)]
    );

    let mut left_to_column_0 = vec![Step::Left; (rows + 1) * (cols + 1)];
    let path = backtrace(&mut left_to_column_0, rows, cols);
    assert_eq!(path, [(1, 0), (2, 0), (2, 1), (2, 2), (2, 3)]);
    assert_eq!(jump_times(&path), [-0.02, -0.02]);
}

// ── Times of the words ──────────────────────────────────────────────────────

/// A path enters rows 1, 2 and 3 at frames 1, 2 and 3, and row 4 at frame 6:
/// the jump times are (frame - 1) / 50. A word starts at the time of the row
/// of its first token and ends at the time of the row after its last token;
/// the end token gives no word.
#[test]
fn a_word_runs_from_the_row_of_its_first_token_to_the_row_after_its_last() {
    let path = [(1, 1), (1, 2), (2, 2), (3, 3), (3, 4), (3, 5), (4, 6)];
    let times = jump_times(&path);
    assert_eq!(times, [0.0, 0.02, 0.04, 0.1]);
    let words = word_times(
        &times,
        vec![
            (" Hi".to_owned(), vec![1]),
            (" there".to_owned(), vec![2, 3]),
            (String::new(), vec![TOK_EOT]),
        ],
    );
    assert_eq!(
        words,
        [
            timing(" Hi", &[1], 0.0, 0.02),
            timing(" there", &[2, 3], 0.02, 0.1)
        ]
    );
    assert_eq!(
        word_times(&times, vec![(String::new(), vec![TOK_EOT])]),
        Vec::<WordTiming>::new()
    );
}

// ── Median filter ───────────────────────────────────────────────────────────

/// One high frame among low ones is removed: each run of 7 holds at most two
/// high values, the reflected one included.
#[test]
fn the_median_filter_removes_an_impulse() {
    let row = [0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(median_filter(&row, MEDIAN_FILTER_WIDTH), [0.0; 8]);
}

/// A ramp is kept inside; at each end the reflected pad (3, 2, 1 before 0, and
/// 8, 7, 6 after 9) moves the median.
#[test]
fn the_median_filter_keeps_a_ramp_and_reflects_at_both_ends() {
    let row: Vec<f32> = (0..10).map(|v| v as f32).collect();
    assert_eq!(
        median_filter(&row, MEDIAN_FILTER_WIDTH),
        [2.0, 2.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.0, 7.0]
    );
}

/// A row of 3 values or fewer is returned as it is; a row of 4 is reflected by
/// 3 at each end, to 2 3 1 [4 1 3 2] 3 1 4.
#[test]
fn the_median_filter_of_a_short_row() {
    assert_eq!(
        median_filter(&[5.0, 1.0, 3.0], MEDIAN_FILTER_WIDTH),
        [5.0, 1.0, 3.0]
    );
    assert_eq!(
        median_filter(&[4.0, 1.0, 3.0, 2.0], MEDIAN_FILTER_WIDTH),
        [2.0, 3.0, 2.0, 3.0]
    );
}

// ── Standardisation and the alignment matrix ───────────────────────────────

/// Frame 0 holds 1, 2, 3 over three tokens: mean 2, population deviation
/// sqrt(2/3). Frame 1 holds 5 at each token: its deviation is 0 and it becomes
/// 0.0, not `NaN`.
#[test]
fn a_frame_is_standardised_over_the_tokens_and_a_constant_frame_is_zero() {
    let mut weights = [1.0, 5.0, 2.0, 5.0, 3.0, 5.0];
    standardize_frames(&mut weights, 3, 2);
    let z = 1.0 / (2.0_f32 / 3.0).sqrt();
    assert!(close(&weights, &[-z, 0.0, 0.0, 0.0, z, 0.0]), "{weights:?}");
}

/// Head 0: each token's two scores are (0, ln 3) or (ln 3, 0), a softmax of
/// (0.25, 0.75) or (0.75, 0.25). Frame 0 holds 0.25 at tokens 0, 1, 2, 4 and
/// 0.75 at tokens 3 and 5: mean 5/12, deviation 0.2357, so 0.25 becomes
/// -0.7071 and 0.75 becomes 1.4142; frame 1 is the negative. Head 1 has equal
/// scores, so each of its frames is constant and becomes 0.0. The mean over the
/// two heads is half of head 0. Two frames are fewer than the filter's half
/// width, so the filter keeps them. Of the six rows, the three of the start
/// sequence and the last (the end token) are cut: rows 3 and 4 are left.
#[test]
fn the_alignment_matrix_of_two_heads() {
    let ln3 = 3.0_f32.ln();
    let low = [0.0, ln3];
    let high = [ln3, 0.0];
    let head_0 = [low, low, low, high, low, high].concat();
    let head_1 = vec![0.0_f32; 12];
    let scores = HeadScores {
        heads: 2,
        tokens: 6,
        frames: 2,
        values: [head_0, head_1].concat(),
    };
    let matrix = alignment_matrix(&scores, 4);
    assert_eq!((matrix.rows, matrix.cols), (2, 2));
    let (a, b) = (2.0_f32.sqrt() / 2.0, 0.5_f32.sqrt() / 2.0);
    assert!(
        close(&matrix.values, &[a, -a, -b, b]),
        "{:?}",
        matrix.values
    );
}

// ── Words from tokens ───────────────────────────────────────────────────────

/// A fake tokenizer: the bytes of each token, decoded as UTF-8 with each
/// incomplete sequence replaced by U+FFFD, and nothing for the end token.
fn decode(tokens: &[u32]) -> String {
    let bytes: Vec<u8> = tokens
        .iter()
        .flat_map(|&token| match token {
            10 => b" Hel".to_vec(),
            11 => b"lo".to_vec(),
            12 => b",".to_vec(),
            13 => b" world".to_vec(),
            // The two parts of 日 (E6 97 A5), and 本 (E6 9C AC) whole.
            20 => vec![0xE6, 0x97],
            21 => vec![0xA5],
            22 => vec![0xE6, 0x9C, 0xAC],
            30 => vec![0xFF],
            _ => Vec::new(),
        })
        .collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// In a language with spaces, a piece with no space and no punctuation joins
/// the word before it; punctuation and the end token start words.
#[test]
fn a_space_language_joins_a_piece_with_no_space_to_the_word_before() {
    let words = split_to_word_tokens(&[10, 11, 12, 13, TOK_EOT], false, decode);
    assert_eq!(
        words,
        [
            (" Hello".to_owned(), vec![10, 11]),
            (",".to_owned(), vec![12]),
            (" world".to_owned(), vec![13]),
            (String::new(), vec![TOK_EOT]),
        ]
    );
}

/// In a language with no spaces, each word is one whole character: token 20
/// alone decodes to U+FFFD where the whole text has 日, so it waits for token
/// 21. A space language joins the two characters into one word.
#[test]
fn a_no_space_language_splits_at_whole_characters() {
    let tokens = [20, 21, 22, TOK_EOT];
    assert_eq!(
        split_to_word_tokens(&tokens, true, decode),
        [
            ("日".to_owned(), vec![20, 21]),
            ("本".to_owned(), vec![22]),
            (String::new(), vec![TOK_EOT]),
        ]
    );
    assert_eq!(
        split_to_word_tokens(&tokens, false, decode),
        [
            ("日本".to_owned(), vec![20, 21, 22]),
            (String::new(), vec![TOK_EOT]),
        ]
    );
}

/// A U+FFFD that the whole text also has at that place is a character of its
/// own.
#[test]
fn a_replacement_character_of_the_whole_text_is_a_word() {
    assert_eq!(
        split_to_word_tokens(&[30, 22, TOK_EOT], true, decode),
        [
            ("\u{FFFD}".to_owned(), vec![30]),
            ("本".to_owned(), vec![22]),
            (String::new(), vec![TOK_EOT]),
        ]
    );
}

// ── Punctuation ─────────────────────────────────────────────────────────────

/// A comma and a full stop join the word before them; the words they leave
/// have no text and no tokens and keep their times.
#[test]
fn appended_punctuation_joins_the_word_before() {
    let mut words = vec![
        timing(" Hello", &[1], 0.0, 0.5),
        timing(",", &[2], 0.5, 0.6),
        timing(" world", &[3], 0.6, 1.0),
        timing(".", &[4], 1.0, 1.1),
    ];
    merge_punctuations(&mut words);
    assert_eq!(texts(&words), [" Hello,", "", " world.", ""]);
    assert_eq!(words[0].tokens, [1, 2]);
    assert_eq!(words[2].tokens, [3, 4]);
    assert_eq!((words[1].start, words[1].end), (0.5, 0.6));
    assert_eq!(words[1].tokens, Vec::<u32>::new());
}

/// An opening quote after a space joins the next word, and an opening
/// bracket joins the word after it across a quote that was already joined.
#[test]
fn prepended_punctuation_joins_the_next_word() {
    let mut words = vec![timing(" \"", &[1], 0.0, 0.1), timing("Hi", &[2], 0.1, 0.4)];
    merge_punctuations(&mut words);
    assert_eq!(texts(&words), ["", " \"Hi"]);
    assert_eq!(words[1].tokens, [1, 2]);

    let mut words = vec![
        timing(" (", &[1], 0.0, 0.1),
        timing(" \"", &[2], 0.1, 0.2),
        timing("Hi", &[3], 0.2, 0.4),
    ];
    merge_punctuations(&mut words);
    assert_eq!(texts(&words), ["", "", " ( \"Hi"]);
    assert_eq!(words[2].tokens, [1, 2, 3]);
}

/// A word that ends in a space takes no appended mark, and a mark with a space
/// before it is not an appended mark.
#[test]
fn a_space_stops_the_append() {
    let mut words = vec![
        timing(" Hello ", &[1], 0.0, 0.5),
        timing(",", &[2], 0.5, 0.6),
    ];
    merge_punctuations(&mut words);
    assert_eq!(texts(&words), [" Hello ", ","]);

    let mut words = vec![
        timing(" Hello", &[1], 0.0, 0.5),
        timing(" ,", &[2], 0.5, 0.6),
    ];
    merge_punctuations(&mut words);
    assert_eq!(texts(&words), [" Hello", " ,"]);
}

// ── Durations ───────────────────────────────────────────────────────────────

/// The median of the durations that are not 0, the mean of the two middle ones
/// for an even count, at most 0.7 s.
#[test]
fn the_median_duration() {
    let of = |durations: &[f64]| {
        let words: Vec<WordTiming> = durations
            .iter()
            .map(|&d| timing(" a", &[1], 1.0, 1.0 + d))
            .collect();
        median_duration(&words)
    };
    assert!((of(&[0.1, 0.0, 0.3, 0.2]) - 0.2).abs() < 1e-12);
    assert!((of(&[0.1, 0.2, 0.3, 0.4]) - 0.25).abs() < 1e-12);
    assert_eq!(of(&[2.0, 3.0]), 0.7);
    assert_eq!(of(&[0.0, 0.0]), 0.0);
}

/// With a longest duration of 0.4 s: a mark longer than that ends 0.4 s after
/// it starts, and the word after a mark starts 0.4 s before it ends. The first
/// word, and a long word after a word that is not a mark, keep their times.
#[test]
fn a_long_word_at_a_sentence_mark_is_cut() {
    let mut words = vec![
        timing("A", &[1], 0.0, 0.9),
        timing(".", &[2], 0.2, 1.0),
        timing(" B", &[3], 1.0, 2.0),
        timing(" C", &[4], 2.0, 3.0),
    ];
    cut_at_sentence_marks(&mut words, 0.4);
    let times: Vec<(f64, f64)> = words.iter().map(|w| (w.start, w.end)).collect();
    assert!((times[1].1 - 0.6).abs() < 1e-12, "{times:?}");
    assert!((times[2].0 - 1.6).abs() < 1e-12, "{times:?}");
    assert_eq!(times[0], (0.0, 0.9));
    assert_eq!(times[3], (2.0, 3.0));
}

/// A median of 0.2 s: two medians are 0.4 s and four are 0.8 s.
#[test]
fn a_long_first_word_after_a_pause_is_cut() {
    let times = |words: &[WordTiming]| -> Vec<(f64, f64)> {
        words.iter().map(|w| (w.start, w.end)).collect()
    };
    let near = |found: &[(f64, f64)], expected: &[(f64, f64)]| {
        found
            .iter()
            .zip(expected)
            .all(|(f, e)| (f.0 - e.0).abs() < 1e-12 && (f.1 - e.1).abs() < 1e-12)
    };

    // The first word is 1.0 s long and the second 0.6 s: they meet at
    // max(2.6 / 2, 2.6 - 0.4) = 2.2, and the first starts at 2.2 - 0.4.
    let mut words = vec![
        timing(" A", &[1], 1.0, 2.0),
        timing(" B", &[2], 2.0, 2.6),
        timing(" C", &[3], 2.6, 2.8),
    ];
    cut_long_first_word(&mut words, 0.2, 0.0);
    assert!(
        near(&times(&words), &[(1.8, 2.2), (2.2, 2.6), (2.6, 2.8)]),
        "{:?}",
        times(&words)
    );

    // The same words 0.5 s after the last speech: no pause, no cut.
    let mut words = vec![timing(" A", &[1], 1.0, 2.0), timing(" B", &[2], 2.0, 2.6)];
    cut_long_first_word(&mut words, 0.2, 1.5);
    assert_eq!(times(&words), [(1.0, 2.0), (2.0, 2.6)]);

    // A second word of 0.3 s is not cut; the first starts at 2.0 - 0.4.
    let mut words = vec![timing(" A", &[1], 1.0, 2.0), timing(" B", &[2], 2.0, 2.3)];
    cut_long_first_word(&mut words, 0.2, 0.0);
    assert!(
        near(&times(&words), &[(1.6, 2.0), (2.0, 2.3)]),
        "{:?}",
        times(&words)
    );

    // A short first word with a long pair: 2.0 - 1.0 > 0.8. They meet at
    // max(1.0, 1.6) = 1.6.
    let mut words = vec![timing(" A", &[1], 1.0, 1.3), timing(" B", &[2], 1.3, 2.0)];
    cut_long_first_word(&mut words, 0.2, 0.0);
    assert!(
        near(&times(&words), &[(1.2, 1.6), (1.6, 2.0)]),
        "{:?}",
        times(&words)
    );
}

// ── Segments ────────────────────────────────────────────────────────────────

fn segment(tokens: &[u32], start: f32, end: f32) -> Segment {
    Segment {
        seek: 1_000,
        start,
        end,
        text: String::new(),
        tokens: tokens.to_vec(),
        temperature: 0.0,
        avg_logprob: -0.2,
        compression_ratio: 1.0,
        no_speech_prob: 0.01,
    }
}

/// Words go to the segments by the count of text tokens; a word with no text
/// is skipped and its tokens are not counted. The times are from the start of
/// the file (the window starts at 10 s), and a time outside the segment is
/// clamped into it: " world." ends at 11.2 in a segment that ends at 11.0.
/// " Next" comes 8 s after the end of the first segment and is 7.8 s long, so
/// the long-first-word rule makes it (18.0, 19.0), which the clamp puts at the
/// end of its segment. The last speech is the end of the last segment.
#[test]
fn words_go_to_their_segments_and_are_clamped_into_them() {
    let ts = TOK_TIMESTAMP_BEGIN;
    let segments = [
        segment(&[ts, 1, 2, 3, ts + 50], 10.0, 11.0),
        segment(&[ts + 50, 4, ts + 75], 11.0, 11.5),
    ];
    let alignment = [
        timing(" Hello", &[1], 0.0, 0.5),
        timing("", &[], 0.5, 0.5),
        timing(" world.", &[2, 3], 0.5, 1.2),
        timing(" Next", &[4], 1.2, 9.0),
    ];
    let mut last_speech = 9.9;
    let words = assign_to_segments(
        alignment.to_vec(),
        &segments,
        10.0,
        4,
        0.5,
        &mut last_speech,
    );
    let found: Vec<(&str, f32, f32, usize)> = words
        .iter()
        .map(|w| (w.word.as_str(), w.start, w.end, w.segment))
        .collect();
    assert_eq!(
        found,
        [
            (" Hello", 10.0, 10.5, 4),
            (" world.", 10.5, 11.0, 4),
            (" Next", 11.5, 11.5, 5),
        ]
    );
    assert_eq!(last_speech, 11.5);
}

/// All steps on one head of two frames. Rows 3 and 4 (`<|notimestamps|>` and
/// " Hel") are strong on frame 0, the rest on frame 1: frame 0 holds 0.75 at
/// those two of eight rows and 0.25 elsewhere, which standardises to sqrt(3)
/// and -1/sqrt(3). The cost of a row strong on frame 0 is (-sqrt(3), sqrt(3)),
/// of the others (1/sqrt(3), -1/sqrt(3)). The warping path is (1,1), (2,1),
/// (3,2), (4,2): rows 1 to 4 are entered at 0.0, 0.0, 0.02 and 0.02 s. " Hello"
/// runs from row 1 to row 3 and "," from row 3 to row 4. The comma joins
/// " Hello", and the times are from the window start at seek 1000 (10 s).
#[test]
fn the_words_of_a_window() {
    let ln3 = 3.0_f32.ln();
    let (low, high) = ([0.0, ln3], [ln3, 0.0]);
    let scores = HeadScores {
        heads: 1,
        tokens: 8,
        frames: 2,
        values: [low, low, low, high, high, low, low, low].concat(),
    };
    let ts = TOK_TIMESTAMP_BEGIN;
    let segments = [segment(&[ts, 10, 11, 12, ts + 100], 10.0, 12.0)];
    let mut last_speech = 0.0;
    let words = window_words(&scores, 4, &segments, "en", decode, 7, &mut last_speech);
    assert_eq!(
        words,
        Ok(vec![Word {
            word: " Hello,".to_owned(),
            start: 10.0,
            end: 10.02,
            segment: 7,
        }])
    );
    assert_eq!(last_speech, 12.0);
}

/// One score that is not a number makes the alignment matrix not finite: the
/// window gives an error, not words and not an empty list.
#[test]
fn a_score_that_is_not_a_number_is_an_error() {
    let ln3 = 3.0_f32.ln();
    let (low, high) = ([0.0, ln3], [ln3, 0.0]);
    let mut values = [low, low, low, high, high, low, low, low].concat();
    values[9] = f32::NAN;
    let scores = HeadScores {
        heads: 1,
        tokens: 8,
        frames: 2,
        values,
    };
    let ts = TOK_TIMESTAMP_BEGIN;
    let segments = [segment(&[ts, 10, 11, 12, ts + 100], 10.0, 12.0)];
    let mut last_speech = 0.0;
    assert_eq!(
        window_words(&scores, 4, &segments, "en", decode, 7, &mut last_speech),
        Err(NonFiniteAlignment)
    );
}

/// A segment whose start is after its end (made by hand) does not panic the
/// clamp: each time goes to the end of the segment.
#[test]
fn a_segment_that_ends_before_it_starts_does_not_panic_the_clamp() {
    let ts = TOK_TIMESTAMP_BEGIN;
    let segments = [segment(&[ts, 1, ts + 50], 2.0, 1.0)];
    let mut last_speech = 0.0;
    let words = assign_to_segments(
        vec![timing(" Hi", &[1], 0.5, 1.5)],
        &segments,
        0.0,
        0,
        0.5,
        &mut last_speech,
    );
    let times: Vec<(f32, f32)> = words.iter().map(|w| (w.start, w.end)).collect();
    assert_eq!(times, [(1.0, 1.0)]);
}

/// The times are rounded to 0.01 s after the window offset is added, a half
/// away from zero: 0.123 and 0.456 after 10 s give 10.12 and 10.46, and 0.125
/// (exact in binary) gives 0.13, where Python's `round(0.125, 2)` gives 0.12.
#[test]
fn word_times_are_rounded_to_hundredths() {
    let ts = TOK_TIMESTAMP_BEGIN;
    let times = |offset: f64, start: f64, end: f64| {
        let segments = [segment(&[ts, 1, ts + 1000], 0.0, 30.0)];
        let mut last_speech = offset + end;
        assign_to_segments(
            vec![timing(" Hi", &[1], start, end)],
            &segments,
            offset,
            0,
            1.0,
            &mut last_speech,
        )
        .iter()
        .map(|w| (w.start, w.end))
        .collect::<Vec<_>>()
    };
    assert_eq!(times(10.0, 0.123, 0.456), [(10.12, 10.46)]);
    assert_eq!(times(0.0, 0.125, 0.5), [(0.13, 0.5)]);
}

/// Eight frames, three tokens, one head; the first token is the prefix (its
/// row stays) and the last is the end token (cut). Each token has a score of
/// ln 3 (h) at four frames and 0 (l) at the others, so each softmax row holds
/// four h and four l of the same two values:
///
/// ```text
///   token 0  h h h h l l l l
///   token 1  l l l h h h h l
///   token 2  l l l l h h h h
/// ```
///
/// A frame with one h standardises its h to sqrt(2) (s) and its l to
/// -1/sqrt(2) (-a); a frame with two h gives a to an h and -s to its l. So
/// token 0 is s s s a -s -s -s -a and token 1 is -a -a -a a a a a -a: frame 7
/// of token 1 is one -a among a values. The median of 7 with a reflect pad
/// gives s s s a -a -s -s -s and -a -a -a a a a a a. A filter on the token
/// axis keeps the rows as they are, and a filter before the standardisation
/// moves the h of token 1 to frame 7 and gives -s at frame 4 of token 0.
#[test]
fn the_median_filter_runs_along_the_frames_after_the_standardisation() {
    let ln3 = 3.0_f32.ln();
    let row = |high: [usize; 4]| -> Vec<f32> {
        (0..8)
            .map(|frame| if high.contains(&frame) { ln3 } else { 0.0 })
            .collect()
    };
    let scores = HeadScores {
        heads: 1,
        tokens: 3,
        frames: 8,
        values: [row([0, 1, 2, 3]), row([3, 4, 5, 6]), row([4, 5, 6, 7])].concat(),
    };
    let matrix = alignment_matrix(&scores, 1);
    let (s, a) = (2.0_f32.sqrt(), 0.5_f32.sqrt());
    assert_eq!((matrix.rows, matrix.cols), (2, 8));
    assert!(
        close(
            &matrix.values,
            &[s, s, s, a, -a, -s, -s, -s, -a, -a, -a, a, a, a, a, a]
        ),
        "{:?}",
        matrix.values
    );
}

/// The text tokens of a window are those below the end token, segment after
/// segment.
#[test]
fn the_text_tokens_of_a_window() {
    let ts = TOK_TIMESTAMP_BEGIN;
    let segments = [
        segment(&[ts, 1, 2, ts + 50], 0.0, 1.0),
        segment(&[ts + 50, 3, ts + 75], 1.0, 1.5),
    ];
    assert_eq!(text_tokens(&segments), [1, 2, 3]);
}
