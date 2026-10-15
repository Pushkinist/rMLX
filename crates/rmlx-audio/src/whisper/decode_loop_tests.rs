//! Unit tests for the host side of the Whisper decoder, from scripted logit
//! rows (no model).

use std::convert::Infallible;

use super::{decode_loop, window_start, writes_at_most, Decoded, NonFiniteRow, Stop, WindowStart};
use crate::tokenizer::{
    N_VOCAB, TOK_EN, TOK_EOT, TOK_NOSPEECH, TOK_NO_TIMESTAMPS, TOK_SOT, TOK_TIMESTAMP_BEGIN,
    TOK_TRANSCRIBE, TOK_TRANSLATE,
};
use crate::whisper::DecodeFilters;

/// Two text tokens.
const A: u32 = 1_000;
const B: u32 = 1_001;
/// The blank token of the first-step rule.
const BLANK: u32 = 220;
const NO_SPEECH_AT_START: f32 = 0.9;

const PREFIX: [u32; 3] = [TOK_SOT, TOK_EN, TOK_TRANSCRIBE];

fn ts(ticks: u32) -> u32 {
    TOK_TIMESTAMP_BEGIN + ticks
}

fn filters() -> DecodeFilters {
    DecodeFilters::new(
        vec![TOK_SOT, TOK_TRANSCRIBE, TOK_TRANSLATE, TOK_NOSPEECH],
        vec![TOK_EOT, BLANK],
        true,
    )
}

/// A row of `-inf` with the given logits.
fn row(logits: &[(u32, f32)]) -> Vec<f32> {
    let mut row = vec![f32::NEG_INFINITY; N_VOCAB];
    for &(token, logit) in logits {
        row[token as usize] = logit;
    }
    row
}

/// A row of zeros in which `<|nospeech|>` has the softmax share `share`.
fn no_speech_row(share: f32) -> Vec<f32> {
    let mut row = vec![0.0_f32; N_VOCAB];
    row[TOK_NOSPEECH as usize] = (share / (1.0 - share) * (N_VOCAB as f32 - 1.0)).ln();
    row
}

/// The rows the scripted decoder gives for sampled step `step`, before any
/// filter. Each holds a large logit on `<|notimestamps|>`, which the filters
/// always mask, so a log-probability of an unfiltered row is far off.
///
/// After the filters, the finite logits are:
/// - step 0: `<|0.00|>` 3.0 and `<|0.10|>` 1.0 (text is masked at the start);
/// - step 1: `A` 2.0 and `B` 1.0 (a timestamp may not follow one timestamp);
/// - step 2: `<|0.20|>` 2.0 and `<|0.40|>` 1.5 (timestamps outweigh the text);
/// - step 3: the end token 1.0, and `<|0.20|>` 0.0 when the open timestamp is
///   `<|0.20|>`.
fn step_row(step: usize) -> Vec<f32> {
    let masked = (TOK_NO_TIMESTAMPS, 5.0);
    match step {
        0 => row(&[(ts(0), 3.0), (ts(5), 1.0), (A, 9.0), masked]),
        1 => row(&[(A, 2.0), (B, 1.0), (ts(7), 4.0), masked]),
        2 => row(&[(ts(10), 2.0), (ts(20), 1.5), (B, 0.0), masked]),
        _ => row(&[(TOK_EOT, 1.0), (ts(10), 0.0), (A, 7.0), masked]),
    }
}

/// `chosen - ln(sum(exp(finite)))`: the log-probability of a logit among the
/// finite logits of its row.
fn log_share(chosen: f32, finite: &[f32]) -> f32 {
    chosen - finite.iter().map(|x| x.exp()).sum::<f32>().ln()
}

fn close(got: f32, want: f32) -> bool {
    (got - want).abs() < 1e-3
}

/// What a scripted decode saw and returned.
struct Run {
    decoded: Decoded,
    /// The input of every decoder forward.
    inputs: Vec<Vec<u32>>,
    /// The filtered row of every pick.
    picked_from: Vec<Vec<f32>>,
}

/// Decode `prefix` against the scripted decoder: the last prefix position and
/// each later position give [`step_row`]. `pick(call, row)` chooses each token.
fn run(
    prefix: &[u32],
    max_tokens: usize,
    n_text_ctx: usize,
    mut pick: impl FnMut(usize, &[f32]) -> u32,
) -> Run {
    let mut inputs: Vec<Vec<u32>> = Vec::new();
    let mut picked_from: Vec<Vec<f32>> = Vec::new();
    let result: Result<Decoded, Infallible> = decode_loop(
        prefix,
        max_tokens,
        n_text_ctx,
        &filters(),
        |input: &[u32]| {
            inputs.push(input.to_vec());
            Ok(step_row(inputs.len() - 1))
        },
        |row: &[f32]| {
            picked_from.push(row.to_vec());
            Ok(pick(picked_from.len() - 1, row))
        },
    );
    match result {
        Ok(decoded) => Run {
            decoded,
            inputs,
            picked_from,
        },
        Err(never) => match never {},
    }
}

/// The first largest logit of `row`.
fn greedy(_: usize, row: &[f32]) -> u32 {
    let mut best = 0;
    for (id, logit) in row.iter().enumerate() {
        if logit > &row[best] {
            best = id;
        }
    }
    best as u32
}

/// A scripted pick. At the two steps where the filters leave a choice of
/// text or of a closing timestamp, it is not the largest logit of its row.
fn second_choice(call: usize, _: &[f32]) -> u32 {
    [ts(0), B, ts(20), TOK_EOT]
        .get(call)
        .copied()
        .unwrap_or(TOK_EOT)
}

// ── What must not move ───────────────────────────────────────────────────────

/// A greedy pick gives the largest filtered logit of each row; the end token
/// is not in the result, and each token is the next decoder input.
#[test]
fn a_greedy_pick_gives_the_largest_filtered_logit_of_each_row() {
    let got = run(&PREFIX, 224, 448, greedy);
    assert_eq!(got.decoded.tokens, [ts(0), A, ts(10)]);
    assert_eq!(got.decoded.stop, Stop::EndToken);
    assert_eq!(
        got.inputs,
        [PREFIX.to_vec(), vec![ts(0)], vec![A], vec![ts(10)]]
    );
}

/// The token budget and the decoder context both end a decode with no end
/// token, and the decode names which one ended it. The first token is picked
/// also with a budget of 0.
#[test]
fn the_budget_and_the_context_end_a_decode() {
    let at_the_budget = run(&PREFIX, 2, 448, greedy).decoded;
    assert_eq!(at_the_budget.tokens, [ts(0), A]);
    assert_eq!(at_the_budget.stop, Stop::TokenBudget);
    let context = PREFIX.len() + 1;
    let at_the_context = run(&PREFIX, 224, context, greedy);
    assert_eq!(at_the_context.decoded.tokens, [ts(0), A]);
    assert_eq!(at_the_context.decoded.stop, Stop::Context);
    assert_eq!(at_the_context.inputs, [PREFIX.to_vec(), vec![ts(0)]]);
    assert_eq!(run(&PREFIX, 0, 448, greedy).decoded.tokens, [ts(0)]);
}

/// An error of a decoder forward or of a pick ends the decode with that
/// error.
#[test]
fn an_error_of_a_forward_or_of_a_pick_is_returned() {
    let failed_forward: Result<Decoded, &str> = decode_loop(
        &PREFIX,
        224,
        448,
        &filters(),
        |input: &[u32]| {
            if input == PREFIX {
                Ok(step_row(0))
            } else {
                Err("forward failed")
            }
        },
        |row: &[f32]| Ok(greedy(0, row)),
    );
    assert_eq!(failed_forward, Err("forward failed"));
    let failed_pick: Result<Decoded, &str> = decode_loop(
        &PREFIX,
        224,
        448,
        &filters(),
        |_: &[u32]| Ok(step_row(0)),
        |_: &[f32]| Err("pick failed"),
    );
    assert_eq!(failed_pick, Err("pick failed"));
}

// ── The start of a window ────────────────────────────────────────────────────

/// A row in which `<|nospeech|>` has the softmax share `share` and `language`
/// has the largest logit of the language tokens. The largest logit of the row
/// is that of `<|nospeech|>`, which is not a language token.
fn start_row(share: f32, language: u32) -> Vec<f32> {
    let mut row = no_speech_row(share);
    row[language as usize] = 2.0;
    row
}

/// The failure of a scripted start read: only a row that is not finite.
#[derive(Debug, PartialEq, Eq)]
struct NotFinite(NonFiniteRow);

impl From<NonFiniteRow> for NotFinite {
    fn from(row: NonFiniteRow) -> Self {
        Self(row)
    }
}

/// What `window_start` read of `row`, or why it refused, and every input it
/// fed to the decoder.
fn read_start(row: &[f32]) -> (Result<WindowStart, NotFinite>, Vec<Vec<u32>>) {
    let mut inputs: Vec<Vec<u32>> = Vec::new();
    let result = window_start(|input: &[u32]| {
        inputs.push(input.to_vec());
        Ok(row.to_vec())
    });
    (result, inputs)
}

/// What `window_start` read and every input it fed to the decoder.
fn start_of(row: &[f32]) -> (WindowStart, Vec<Vec<u32>>) {
    match read_start(row) {
        (Ok(start), inputs) => (start, inputs),
        (Err(refused), _) => panic!("the start row was refused: {refused:?}"),
    }
}

/// A start row with a `NaN` or `+inf` logit is an error, wherever the logit is:
/// the no-speech probability would be `NaN` and the language would be the
/// first one with no message.
#[test]
fn a_start_row_that_is_not_finite_is_an_error() {
    for bad in [f32::NAN, f32::INFINITY] {
        for at in [0, TOK_EN as usize, TOK_NOSPEECH as usize, N_VOCAB - 1] {
            let mut row = start_row(NO_SPEECH_AT_START, TOK_EN);
            row[at] = bad;
            let (read, inputs) = read_start(&row);
            assert_eq!(
                read,
                Err(NotFinite(NonFiniteRow { len: row.len() })),
                "{bad} at {at}"
            );
            assert_eq!(inputs, [vec![TOK_SOT]]);
        }
    }
}

/// A masked logit is not a fault: the start row may hold `-inf`.
#[test]
fn a_start_row_with_a_masked_logit_is_read() {
    let mut row = start_row(NO_SPEECH_AT_START, TOK_EN);
    row[0] = f32::NEG_INFINITY;
    assert!(read_start(&row).0.is_ok());
}

/// The no-speech row is read with one forward whose whole input is the start
/// token: no previous-text prompt, and no other token, comes before it.
#[test]
fn the_no_speech_read_feeds_the_start_token_alone() {
    let (_, inputs) = start_of(&start_row(NO_SPEECH_AT_START, TOK_EN));
    assert_eq!(inputs, [vec![TOK_SOT]]);
}

/// The no-speech probability is the softmax share of `<|nospeech|>` in the
/// unfiltered row. The decode filters mask that token.
#[test]
fn the_no_speech_probability_is_the_share_of_the_unfiltered_start_row() {
    for share in [0.9, 0.61, 0.3, 0.012] {
        let got = start_of(&start_row(share, TOK_EN)).0.no_speech_prob;
        assert!(close(got, share), "share {share}: {got}");
    }
}

/// The same forward gives the language of the window: the row exists, so the
/// language detection of the first window adds no second forward.
#[test]
fn one_forward_gives_the_language_and_the_no_speech_probability() {
    let french = TOK_EN + 6;
    let (start, inputs) = start_of(&start_row(NO_SPEECH_AT_START, french));
    assert_eq!(start.language, french);
    assert!(close(start.no_speech_prob, NO_SPEECH_AT_START));
    assert_eq!(inputs.len(), 1, "one forward for the two values");
}

// ── The average log-probability ──────────────────────────────────────────────

/// The average log-probability sums the log-probability of every sampled token
/// and of the end token, each in its filtered row, and divides by the token
/// count plus one.
#[test]
fn the_average_log_probability_counts_the_sampled_tokens_and_the_end_token() {
    let sum = log_share(3.0, &[3.0, 1.0])
        + log_share(2.0, &[2.0, 1.0])
        + log_share(2.0, &[2.0, 1.5])
        + log_share(1.0, &[1.0, 0.0]);
    let got = run(&PREFIX, 224, 448, greedy).decoded.avg_logprob;
    assert!(close(got, sum / 4.0), "got {got}, want {}", sum / 4.0);
}

/// A decode that the budget ends has no end token to count; the divisor is
/// still the token count plus one, as in openai-whisper.
#[test]
fn a_decode_the_budget_ends_divides_by_the_token_count_plus_one() {
    let sum = log_share(3.0, &[3.0, 1.0]) + log_share(2.0, &[2.0, 1.0]);
    let got = run(&PREFIX, 2, 448, greedy).decoded.avg_logprob;
    assert!(close(got, sum / 3.0), "got {got}, want {}", sum / 3.0);
}

// ── The token pick ───────────────────────────────────────────────────────────

/// Every sampled token, the first and the end token included, is one pick from
/// its filtered row, in order: a seeded pick then draws the same stream on a
/// repeated decode. The row is not scaled: the pick owns the temperature.
#[test]
fn every_token_is_one_pick_from_its_filtered_row() {
    let got = run(&PREFIX, 224, 448, second_choice);
    assert_eq!(got.decoded.tokens, [ts(0), B, ts(20)]);
    assert_eq!(
        got.inputs,
        [PREFIX.to_vec(), vec![ts(0)], vec![B], vec![ts(20)]]
    );
    assert_eq!(
        got.picked_from.len(),
        4,
        "one pick for each token and the end"
    );
    for (call, row) in got.picked_from.iter().enumerate() {
        assert_eq!(
            row[TOK_NO_TIMESTAMPS as usize],
            f32::NEG_INFINITY,
            "pick {call} reads a filtered row"
        );
    }
    assert_eq!(got.picked_from[0][A as usize], f32::NEG_INFINITY);
    assert_eq!(got.picked_from[0][ts(0) as usize], 3.0);
    assert_eq!(got.picked_from[1][B as usize], 1.0);
}

/// The average log-probability is that of the picked tokens, not of the
/// largest logit of each row.
#[test]
fn the_average_log_probability_is_that_of_the_picked_tokens() {
    let sum = log_share(3.0, &[3.0, 1.0])
        + log_share(1.0, &[2.0, 1.0])
        + log_share(1.5, &[2.0, 1.5])
        + log_share(1.0, &[1.0]);
    let got = run(&PREFIX, 224, 448, second_choice).decoded.avg_logprob;
    assert!(close(got, sum / 4.0), "got {got}, want {}", sum / 4.0);
}

// ── The bounded decode of the no-speech gate ─────────────────────────────────

/// Filters with no timestamp tokens, so each scripted row is one text token
/// or the end token.
fn text_only_filters() -> DecodeFilters {
    DecodeFilters::new(Vec::new(), Vec::new(), false)
}

/// Whether a scripted decode that writes `text_tokens` text tokens and then
/// the end token is within `bound`, and the count of decoder forwards.
fn within(bound: usize, text_tokens: usize) -> (bool, usize) {
    let mut forwards = 0;
    let result: Result<bool, Infallible> = writes_at_most(
        bound,
        &PREFIX,
        224,
        448,
        &text_only_filters(),
        |_: &[u32]| {
            forwards += 1;
            Ok(if forwards <= text_tokens {
                row(&[(A, 1.0)])
            } else {
                row(&[(TOK_EOT, 1.0)])
            })
        },
        |row: &[f32]| Ok(greedy(0, row)),
    );
    match result {
        Ok(within) => (within, forwards),
        Err(never) => match never {},
    }
}

/// A text of exactly the bound is within it; one token more is not.
#[test]
fn a_text_of_the_bound_is_within_it_and_one_token_more_is_not() {
    for bound in [0, 1, 8] {
        assert!(within(bound, bound).0, "{bound} text tokens, bound {bound}");
        assert!(!within(bound, bound + 1).0, "one more than {bound}");
        assert!(within(bound, 0).0, "no text, bound {bound}");
    }
}

/// The decode ends when the text passes the bound: for a bound of 8 it makes
/// the forward of the prefix and one forward for each of 8 text tokens, and
/// no forward after the ninth text token, for a text of 9 tokens and for one
/// of 200. A text within the bound is decoded to its end token.
#[test]
fn the_bounded_decode_stops_when_the_text_passes_the_bound() {
    assert_eq!(within(8, 9), (false, 9));
    assert_eq!(within(8, 200), (false, 9));
    assert_eq!(within(8, 8), (true, 9));
    assert_eq!(within(8, 3), (true, 4));
    assert_eq!(within(8, 0), (true, 1));
}

/// A timestamp token and the end token are not text: the scripted window of
/// two timestamps and one text token is within a bound of 1, and not within
/// a bound of 0. The bounded decode reads filtered rows, as each decode does.
#[test]
fn a_timestamp_token_is_not_text() {
    let check = |bound: usize| -> bool {
        let mut step = 0;
        let result: Result<bool, Infallible> = writes_at_most(
            bound,
            &PREFIX,
            224,
            448,
            &filters(),
            |_: &[u32]| {
                step += 1;
                Ok(step_row(step - 1))
            },
            |row: &[f32]| Ok(greedy(0, row)),
        );
        match result {
            Ok(within) => within,
            Err(never) => match never {},
        }
    };
    assert!(check(1));
    assert!(!check(0));
}

/// An error of a forward or of a pick is the error of the bounded decode.
#[test]
fn an_error_in_the_bounded_decode_is_returned() {
    let failed_forward: Result<bool, &str> = writes_at_most(
        8,
        &PREFIX,
        224,
        448,
        &text_only_filters(),
        |_: &[u32]| Err("forward failed"),
        |row: &[f32]| Ok(greedy(0, row)),
    );
    assert_eq!(failed_forward, Err("forward failed"));
    let failed_pick: Result<bool, &str> = writes_at_most(
        8,
        &PREFIX,
        224,
        448,
        &text_only_filters(),
        |_: &[u32]| Ok(row(&[(A, 1.0)])),
        |_: &[f32]| Err("pick failed"),
    );
    assert_eq!(failed_pick, Err("pick failed"));
}
