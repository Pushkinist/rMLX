//! The word timestamps of one decoded window, as openai-whisper finds them
//! (`whisper/timing.py`): the cross-attention of the alignment heads over the
//! text tokens, a median filter, dynamic time warping, and the rules that cut
//! the time of a long word. Pure functions, no model: the GPU gives the scores
//! of the heads before the softmax, and all steps after that run here.
//!
//! Words are added to the segments and do not change them: a word time outside
//! its segment is clamped into it. openai-whisper moves the segment bounds and
//! the seek from the words instead.

use super::{Segment, Word};
use crate::mel::{HOP_LENGTH, SAMPLE_RATE};
use crate::tokenizer::TOK_EOT;
use crate::whisper::{HeadScores, WhisperError};

/// Width of the median filter along the frames: openai-whisper's
/// `medfilt_width`.
const MEDIAN_FILTER_WIDTH: usize = 7;

/// Encoder frames in one second: one frame is 0.02 s (openai-whisper's
/// `TOKENS_PER_SECOND`).
const FRAMES_PER_SECOND: f64 = 50.0;

/// The median word duration that the duration rules use is at most this, in
/// seconds.
const MAX_MEDIAN_DURATION: f64 = 0.7;

/// A word that is one of these marks ends a sentence.
const SENTENCE_END_MARKS: &str = ".。!！?？";

/// openai-whisper's `prepend_punctuations`: such a word joins the next word.
const PREPENDED_PUNCTUATION: &str = "\"'“¿([{-";

/// openai-whisper's `append_punctuations`: such a word joins the word before it.
const APPENDED_PUNCTUATION: &str = "\"'.。,，!！?？:：”)]}、";

/// Python's `string.punctuation`.
const ASCII_PUNCTUATION: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";

/// Languages that do not put spaces between words: their words are cut at
/// each complete character.
const NO_SPACE_LANGUAGES: [&str; 6] = ["zh", "ja", "th", "lo", "my", "yue"];

/// A matrix of `rows` x `cols` values, row by row.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Matrix {
    pub(super) rows: usize,
    pub(super) cols: usize,
    pub(super) values: Vec<f32>,
}

/// One word of a window before the segments: its text with its spaces, its
/// tokens, and its start and end in seconds from the window start.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct WordTiming {
    pub(super) word: String,
    pub(super) tokens: Vec<u32>,
    pub(super) start: f64,
    pub(super) end: f64,
}

/// One move of the warping path into a cell of the cost grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Step {
    /// From the cell up and to the left.
    Diagonal,
    /// From the cell above: the next token, the same frame.
    Up,
    /// From the cell to the left: the same token, the next frame.
    Left,
}

/// The text tokens of the segments of a window, in order: the tokens that the
/// alignment pass forces.
pub(super) fn text_tokens(segments: &[Segment]) -> Vec<u32> {
    segments
        .iter()
        .flat_map(|segment| segment.tokens.iter().copied().filter(|&t| t < TOK_EOT))
        .collect()
}

/// An alignment matrix with a value that is not finite: the scores of the
/// alignment heads were not numbers, so no word can be timed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("word alignment: the alignment matrix holds a value that is not a number or is infinite")]
pub(crate) struct NonFiniteAlignment;

impl From<NonFiniteAlignment> for WhisperError {
    fn from(e: NonFiniteAlignment) -> Self {
        Self::Mlx(e.to_string())
    }
}

/// The words of the segments of one window. `scores` are those of the
/// alignment pass over a prefix of `prefix_len` tokens (the start sequence and
/// `<|notimestamps|>`), the [`text_tokens`] of `segments` and the end token.
/// `language` is the language code of the window and `decode` gives the text
/// of tokens. `first_segment` is the index of `segments[0]` in the
/// transcription. `last_speech` is the end of the last segment before this
/// window that has words; it moves to the end of each segment of this window
/// that has words. An alignment matrix with a value that is not finite is an
/// error.
pub(super) fn window_words(
    scores: &HeadScores,
    prefix_len: usize,
    segments: &[Segment],
    language: &str,
    decode: impl Fn(&[u32]) -> String,
    first_segment: usize,
    last_speech: &mut f64,
) -> Result<Vec<Word>, NonFiniteAlignment> {
    let Some(first) = segments.first() else {
        return Ok(Vec::new());
    };
    let mut tokens = text_tokens(segments);
    if tokens.is_empty() {
        return Ok(Vec::new());
    }
    tokens.push(TOK_EOT);
    let offset = (first.seek * HOP_LENGTH) as f64 / f64::from(SAMPLE_RATE);
    let split = split_to_word_tokens(&tokens, NO_SPACE_LANGUAGES.contains(&language), &decode);
    let matrix = alignment_matrix(scores, prefix_len);
    if matrix.values.iter().any(|value| !value.is_finite()) {
        return Err(NonFiniteAlignment);
    }
    let mut alignment = timed_words(&matrix, split);
    let median = median_duration(&alignment);
    let max_duration = median * 2.0;
    if alignment.iter().any(|word| word.end - word.start != 0.0) {
        cut_at_sentence_marks(&mut alignment, max_duration);
    }
    merge_punctuations(&mut alignment);
    Ok(assign_to_segments(
        alignment,
        segments,
        offset,
        first_segment,
        median,
        last_speech,
    ))
}

/// The alignment matrix of the text: for each head, the softmax of each
/// token's scores over the frames, each frame standardised over the tokens,
/// and a median filter along the frames; then the mean over the heads. The
/// rows are those of `<|notimestamps|>`, the last of the `prefix_len` tokens
/// before the text, and of the text tokens: the rows of the start sequence
/// before it and the row of the end token are cut.
pub(super) fn alignment_matrix(scores: &HeadScores, prefix_len: usize) -> Matrix {
    let HeadScores {
        heads,
        tokens,
        frames,
        ..
    } = *scores;
    let mut sum = vec![0.0_f32; tokens * frames];
    for head in scores
        .values
        .chunks_exact((tokens * frames).max(1))
        .take(heads)
    {
        let mut weights = head.to_vec();
        for row in weights.chunks_exact_mut(frames.max(1)) {
            softmax(row);
        }
        standardize_frames(&mut weights, tokens, frames);
        for row in weights.chunks_exact_mut(frames.max(1)) {
            let filtered = median_filter(row, MEDIAN_FILTER_WIDTH);
            row.copy_from_slice(&filtered);
        }
        for (total, value) in sum.iter_mut().zip(&weights) {
            *total += value;
        }
    }
    let start_sequence = prefix_len.saturating_sub(1);
    let rows = tokens.saturating_sub(start_sequence + 1);
    let values = sum
        .iter()
        .skip(start_sequence * frames)
        .take(rows * frames)
        .map(|total| total / heads as f32)
        .collect();
    Matrix {
        rows,
        cols: frames,
        values,
    }
}

/// The softmax of `row`, in place.
fn softmax(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0_f32;
    for value in row.iter_mut() {
        *value = (*value - max).exp();
        total += *value;
    }
    for value in row.iter_mut() {
        *value /= total;
    }
}

/// Standardise each frame of a `tokens` x `frames` matrix over the tokens:
/// subtract the mean and divide by the population standard deviation. A frame
/// whose deviation is 0 has the same value at each token and becomes 0.0 at
/// each token (openai-whisper divides by 0 there and gives `NaN`).
pub(super) fn standardize_frames(weights: &mut [f32], tokens: usize, frames: usize) {
    for frame in 0..frames {
        let column = || weights.iter().skip(frame).step_by(frames).take(tokens);
        let mean = column().map(|&v| f64::from(v)).sum::<f64>() / tokens as f64;
        let variance = column()
            .map(|&v| (f64::from(v) - mean).powi(2))
            .sum::<f64>()
            / tokens as f64;
        let deviation = variance.sqrt();
        for value in weights.iter_mut().skip(frame).step_by(frames).take(tokens) {
            *value = if deviation == 0.0 {
                0.0
            } else {
                ((f64::from(*value) - mean) / deviation) as f32
            };
        }
    }
}

/// The median of each run of `width` values of `row` (an odd width), with the
/// row reflected at both ends by `width / 2` values, as openai-whisper's
/// `median_filter`. A row of `width / 2` values or fewer is returned as it is.
pub(super) fn median_filter(row: &[f32], width: usize) -> Vec<f32> {
    let pad = width / 2;
    let len = row.len();
    if len <= pad {
        return row.to_vec();
    }
    let reflected = |at: usize| -> f32 {
        let index = if at < pad {
            pad - at
        } else if at - pad >= len {
            2 * (len - 1) - (at - pad)
        } else {
            at - pad
        };
        row.get(index).copied().unwrap_or_default()
    };
    let mut window = vec![0.0_f32; width];
    (0..len)
        .map(|start| {
            for (offset, slot) in window.iter_mut().enumerate() {
                *slot = reflected(start + offset);
            }
            window.sort_unstable_by(f32::total_cmp);
            window.get(pad).copied().unwrap_or_default()
        })
        .collect()
}

/// Dynamic time warping over `x`, a `rows` x `cols` cost matrix, as
/// openai-whisper's `dtw_cpu`: the cost of a cell is its value plus the
/// least cost of the cell up and to the left, the cell above and the cell to
/// the left. Of two equal least costs, the move from the left wins; so does
/// the move from the left when the diagonal and the cell above tie. Returns
/// the path from [`backtrace`].
#[allow(
    clippy::indexing_slicing,
    reason = "i in 1..=rows, j in 1..=cols: every index is inside the (rows + 1) x (cols + 1) grid or the rows x cols input"
)]
pub(super) fn dtw(x: &[f32], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let width = cols + 1;
    let mut cost = vec![f32::INFINITY; (rows + 1) * width];
    let mut trace = vec![Step::Left; (rows + 1) * width];
    cost[0] = 0.0;
    for j in 1..=cols {
        for i in 1..=rows {
            let diagonal = cost[(i - 1) * width + j - 1];
            let up = cost[(i - 1) * width + j];
            let left = cost[i * width + j - 1];
            let (least, step) = if diagonal < up && diagonal < left {
                (diagonal, Step::Diagonal)
            } else if up < diagonal && up < left {
                (up, Step::Up)
            } else {
                (left, Step::Left)
            };
            cost[i * width + j] = x[(i - 1) * cols + j - 1] + least;
            trace[i * width + j] = step;
        }
    }
    backtrace(&mut trace, rows, cols)
}

/// The path through a `(rows + 1)` x `(cols + 1)` grid of moves, from cell
/// `(rows, cols)` back to `(0, 0)`, as openai-whisper's `backtrace`. Before
/// the walk, row 0 is set to moves from the left and column 0 to moves from
/// above. The path is in order from the first cell after `(0, 0)` and holds
/// grid cells: openai-whisper reports each cell `(i, j)` as text index
/// `i - 1` and time index `j - 1`.
#[allow(
    clippy::indexing_slicing,
    reason = "the walk starts at (rows, cols) and each move lowers i or j while one is above 0"
)]
pub(super) fn backtrace(trace: &mut [Step], rows: usize, cols: usize) -> Vec<(usize, usize)> {
    let width = cols + 1;
    for step in trace.iter_mut().take(width) {
        *step = Step::Left;
    }
    for step in trace.iter_mut().step_by(width).take(rows + 1) {
        *step = Step::Up;
    }
    let (mut i, mut j) = (rows, cols);
    let mut path = Vec::with_capacity(rows + cols);
    while i > 0 || j > 0 {
        path.push((i, j));
        match trace[i * width + j] {
            Step::Diagonal => {
                i -= 1;
                j -= 1;
            }
            Step::Up => i -= 1,
            Step::Left => j -= 1,
        }
    }
    path.reverse();
    path
}

/// The time of each step of `path` into a new text row, in seconds: the time
/// at which the path enters the row.
pub(super) fn jump_times(path: &[(usize, usize)]) -> Vec<f64> {
    let mut row_before = None;
    path.iter()
        .filter(|&&(i, _)| row_before.replace(i) != Some(i))
        .map(|&(_, j)| (j as f64 - 1.0) / FRAMES_PER_SECOND)
        .collect()
}

/// The words of `tokens` and the tokens of each, as openai-whisper's
/// `split_to_word_tokens`. For a language with no spaces
/// (`no_space`), each word is the shortest run of tokens that decodes to whole
/// characters. Otherwise those runs join the word before them, and a run
/// starts a new word when its first token is special, its text starts with a
/// space or its trimmed text is in Python's `string.punctuation`.
pub(super) fn split_to_word_tokens(
    tokens: &[u32],
    no_space: bool,
    decode: impl Fn(&[u32]) -> String,
) -> Vec<(String, Vec<u32>)> {
    let pieces = split_on_characters(tokens, &decode);
    if no_space {
        return pieces;
    }
    let mut words: Vec<(String, Vec<u32>)> = Vec::new();
    for (piece, piece_tokens) in pieces {
        let special = piece_tokens.first().is_some_and(|&t| t >= TOK_EOT);
        let with_space = piece.starts_with(' ');
        let punctuation = ASCII_PUNCTUATION.contains(piece.trim());
        match words.last_mut() {
            Some((word, word_tokens)) if !(special || with_space || punctuation) => {
                word.push_str(&piece);
                word_tokens.extend(piece_tokens);
            }
            _ => words.push((piece, piece_tokens)),
        }
    }
    words
}

/// openai-whisper's `split_tokens_on_unicode`: grow a run of tokens until its
/// text has no U+FFFD, or its first U+FFFD is also in the text of all
/// `tokens` at the same place.
fn split_on_characters(
    tokens: &[u32],
    decode: &impl Fn(&[u32]) -> String,
) -> Vec<(String, Vec<u32>)> {
    const REPLACEMENT: char = '\u{FFFD}';
    let whole: Vec<char> = decode(tokens).chars().collect();
    let mut pieces = Vec::new();
    let mut run = Vec::new();
    let mut offset = 0;
    for &token in tokens {
        run.push(token);
        let text = decode(&run);
        let complete = text
            .chars()
            .position(|c| c == REPLACEMENT)
            .is_none_or(|at| whole.get(offset + at) == Some(&REPLACEMENT));
        if complete {
            offset += text.chars().count();
            pieces.push((text, std::mem::take(&mut run)));
        }
    }
    pieces
}

/// The words of `split` with their times along the warping path of `matrix`,
/// which [`dtw`] finds on the negative of the matrix.
fn timed_words(matrix: &Matrix, split: Vec<(String, Vec<u32>)>) -> Vec<WordTiming> {
    if split.len() <= 1 {
        return Vec::new();
    }
    let negated: Vec<f32> = matrix.values.iter().map(|v| -v).collect();
    let path = dtw(&negated, matrix.rows, matrix.cols);
    word_times(&jump_times(&path), split)
}

/// The words of `split` with their times: a word starts at the jump time of
/// the row of its first token and ends at that of the row after its last
/// token. The last word, the end token, is dropped. With one word or none
/// there are no words.
pub(super) fn word_times(times: &[f64], split: Vec<(String, Vec<u32>)>) -> Vec<WordTiming> {
    if split.len() <= 1 {
        return Vec::new();
    }
    // The path enters every row, so each word boundary has a time.
    let time_at = |row: usize| times.get(row).copied().unwrap_or_default();
    let n_words = split.len() - 1;
    let mut boundary = 0;
    split
        .into_iter()
        .take(n_words)
        .map(|(word, tokens)| {
            let start = time_at(boundary);
            boundary += tokens.len();
            WordTiming {
                word,
                tokens,
                start,
                end: time_at(boundary),
            }
        })
        .collect()
}

/// The median of the word durations that are not 0, at most 0.7 s; 0.0 when
/// every duration is 0.
pub(super) fn median_duration(words: &[WordTiming]) -> f64 {
    let mut durations: Vec<f64> = words
        .iter()
        .map(|word| word.end - word.start)
        .filter(|&duration| duration != 0.0)
        .collect();
    durations.sort_unstable_by(f64::total_cmp);
    let at = |index: usize| durations.get(index).copied().unwrap_or_default();
    let median = match durations.len() {
        0 => 0.0,
        n if n % 2 == 1 => at(n / 2),
        n => (at(n / 2 - 1) + at(n / 2)) / 2.0,
    };
    median.min(MAX_MEDIAN_DURATION)
}

/// A word after the first that is longer than `max_duration`: when it is a
/// sentence mark, its end moves to `max_duration` after its start; when the
/// word before it is a mark, its start moves to `max_duration` before its end.
pub(super) fn cut_at_sentence_marks(words: &mut [WordTiming], max_duration: f64) {
    for i in 1..words.len() {
        let previous_is_mark = words
            .get(i - 1)
            .is_some_and(|word| SENTENCE_END_MARKS.contains(word.word.as_str()));
        let Some(word) = words.get_mut(i) else {
            continue;
        };
        if word.end - word.start > max_duration {
            if SENTENCE_END_MARKS.contains(word.word.as_str()) {
                word.end = word.start + max_duration;
            } else if previous_is_mark {
                word.start = word.end - max_duration;
            }
        }
    }
}

/// openai-whisper's `merge_punctuations`. From the end, a word that starts with
/// a space and whose trimmed text is in [`PREPENDED_PUNCTUATION`] joins the
/// next word that is left. Then from the start, a word whose text is in
/// [`APPENDED_PUNCTUATION`] joins the word before it, when that word does not
/// end in a space. A word that joins another keeps its times and has no text
/// and no tokens. "In" is a substring test, as Python's `in` on strings: the
/// empty text is in each set.
pub(super) fn merge_punctuations(words: &mut [WordTiming]) {
    let mut into = words.len().saturating_sub(1);
    for at in (0..words.len().saturating_sub(1)).rev() {
        let joins = words.get(at).is_some_and(|word| {
            word.word.starts_with(' ') && PREPENDED_PUNCTUATION.contains(word.word.trim())
        });
        if joins {
            let (word, tokens) = take_text(words, at);
            if let Some(following) = words.get_mut(into) {
                following.word.insert_str(0, &word);
                following.tokens.splice(0..0, tokens);
            }
        } else {
            into = at;
        }
    }
    let mut into = 0;
    for at in 1..words.len() {
        let joins = words
            .get(into)
            .is_some_and(|word| !word.word.ends_with(' '))
            && words
                .get(at)
                .is_some_and(|word| APPENDED_PUNCTUATION.contains(word.word.as_str()));
        if joins {
            let (word, tokens) = take_text(words, at);
            if let Some(previous) = words.get_mut(into) {
                previous.word.push_str(&word);
                previous.tokens.extend(tokens);
            }
        } else {
            into = at;
        }
    }
}

/// The text and the tokens of `words[at]`, which keeps none.
fn take_text(words: &mut [WordTiming], at: usize) -> (String, Vec<u32>) {
    words.get_mut(at).map_or_else(Default::default, |word| {
        (
            std::mem::take(&mut word.word),
            std::mem::take(&mut word.tokens),
        )
    })
}

/// Give each segment its words, by the count of its text tokens, as
/// openai-whisper's `add_word_timestamps`: a word without text is skipped, the
/// times are rounded to 0.01 s from the start of the file, and the rule for a
/// long first word applies. Then each time is clamped into its segment.
fn assign_to_segments(
    alignment: Vec<WordTiming>,
    segments: &[Segment],
    offset: f64,
    first_segment: usize,
    median: f64,
    last_speech: &mut f64,
) -> Vec<Word> {
    let mut words = Vec::new();
    let mut next = alignment.into_iter();
    for (index, segment) in segments.iter().enumerate() {
        let n_tokens = segment.tokens.iter().filter(|&&t| t < TOK_EOT).count();
        let mut saved = 0;
        let mut timed: Vec<WordTiming> = Vec::new();
        while saved < n_tokens {
            let Some(word) = next.next() else {
                break;
            };
            saved += word.tokens.len();
            if !word.word.is_empty() {
                timed.push(WordTiming {
                    word: word.word,
                    tokens: Vec::new(),
                    start: round_to_hundredths(offset + word.start),
                    end: round_to_hundredths(offset + word.end),
                });
            }
        }
        if timed.is_empty() {
            continue;
        }
        cut_long_first_word(&mut timed, median, *last_speech);
        let (low, high) = (f64::from(segment.start), f64::from(segment.end));
        words.extend(timed.into_iter().map(|word| Word {
            word: word.word,
            start: word.start.max(low).min(high) as f32,
            end: word.end.max(low).min(high) as f32,
            segment: first_segment + index,
        }));
        *last_speech = high;
    }
    words
}

/// `value` rounded to two decimals.
fn round_to_hundredths(value: f64) -> f64 {
    (value * 100.0).round() / 100.0
}

/// openai-whisper's rule for the first word after a pause of more than four
/// median durations since `last_speech`, when that word is longer than two
/// median durations or the first two words together are longer than four: a
/// second word longer than two medians and the first word meet at a shared
/// bound, and the first word starts at most two medians before its end.
pub(super) fn cut_long_first_word(words: &mut [WordTiming], median: f64, last_speech: f64) {
    let max_duration = median * 2.0;
    let Some((first, rest)) = words.split_first_mut() else {
        return;
    };
    let second = rest.first_mut();
    let long = first.end - first.start > max_duration
        || second
            .as_ref()
            .is_some_and(|second| second.end - first.start > max_duration * 2.0);
    if first.end - last_speech <= median * 4.0 || !long {
        return;
    }
    if let Some(second) = second.filter(|second| second.end - second.start > max_duration) {
        let bound = (second.end / 2.0).max(second.end - max_duration);
        first.end = bound;
        second.start = bound;
    }
    first.start = (first.end - max_duration).max(0.0);
}

#[cfg(test)]
#[path = "words_tests.rs"]
mod words_tests;
