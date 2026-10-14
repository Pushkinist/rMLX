//! Unit tests for the decision of one window, from scripted decodes (no model).

use std::convert::Infallible;

use super::{
    decode_with_fallback, temperature_ladder, Verdict, WindowOutcome, LOGPROB_THRESHOLD,
    NO_SPEECH_THRESHOLD, PROMPT_RESET_TEMPERATURE,
};
use crate::tokenizer::TOK_TIMESTAMP_BEGIN;
use crate::transcribe::seek::plan_window;
use crate::transcribe::tests::{clean, looping, sentences, PROMPT};
use crate::transcribe::{compression_ratio, next_prompt};
use crate::whisper::decode_loop::{Decoded, Stop};

/// The average log-probability of a sure decode.
const SURE: f32 = -0.3;
/// The no-speech probability of a spoken window.
const SPOKEN: f32 = 0.01;
const SAMPLES_PER_WINDOW: usize = 480_000;

fn decoded(tokens: Vec<u32>, avg_logprob: f32) -> Decoded {
    Decoded {
        tokens,
        avg_logprob,
        stop: Stop::EndToken,
    }
}

fn sure(tokens: Vec<u32>) -> Decoded {
    decoded(tokens, SURE)
}

/// A clean token stream that differs from every other `spoken(k)`.
fn spoken(k: u32) -> Vec<u32> {
    let mut tokens = clean();
    tokens.insert(1, k);
    tokens
}

/// A weak decode of a spoken window.
fn weak(k: u32, avg_logprob: f32) -> Decoded {
    decoded(spoken(k), avg_logprob)
}

/// One closed segment of one second, then an open one.
fn closed_then_open() -> Vec<u32> {
    let pair = TOK_TIMESTAMP_BEGIN + 50;
    vec![TOK_TIMESTAMP_BEGIN, 5, pair, pair, 6]
}

/// What a scripted window gave: the outcome and, for each decode that ran, its
/// prompt and its temperature in tenths.
struct Run {
    outcome: WindowOutcome,
    calls: Vec<(Vec<u32>, u32)>,
    /// The text bound of each run of the bounded decode of the gate.
    bounded_decodes: Vec<usize>,
}

/// Run `decode_with_fallback` for a spoken window from the temperature `start`
/// with a decoder that returns `script` in order. A call after the last
/// scripted decode gets the last one again, so a decode that should not run
/// shows in `calls`.
fn run(prompt: &[u32], start: f32, script: &[Decoded]) -> Run {
    run_window(SPOKEN, prompt, start, script)
}

/// [`run`] for a window with the no-speech probability `no_speech_prob`,
/// whose greedy decode with no prompt writes no more than a stock phrase.
fn run_window(no_speech_prob: f32, prompt: &[u32], start: f32, script: &[Decoded]) -> Run {
    run_gate(no_speech_prob, true, prompt, start, script)
}

/// [`run`] for a window with the no-speech probability `no_speech_prob`.
/// `short_text` is what the bounded decode of the gate answers: whether the
/// greedy decode with no prompt is within the text bound.
fn run_gate(
    no_speech_prob: f32,
    short_text: bool,
    prompt: &[u32],
    start: f32,
    script: &[Decoded],
) -> Run {
    let mut calls: Vec<(Vec<u32>, u32)> = Vec::new();
    let mut bounded_decodes: Vec<usize> = Vec::new();
    let result: Result<WindowOutcome, Infallible> = decode_with_fallback(
        no_speech_prob,
        |text_bound: usize| {
            bounded_decodes.push(text_bound);
            Ok(short_text)
        },
        prompt,
        start,
        |p: &[u32], temperature: f32| {
            let next = script.get(calls.len()).or(script.last()).cloned();
            calls.push((p.to_vec(), (temperature * 10.0).round() as u32));
            Ok(next.unwrap_or_else(|| sure(clean())))
        },
        sentences,
    );
    match result {
        Ok(outcome) => Run {
            outcome,
            calls,
            bounded_decodes,
        },
        Err(never) => match never {},
    }
}

fn with_prompt(tenths: u32) -> (Vec<u32>, u32) {
    (PROMPT.to_vec(), tenths)
}

fn without_prompt(tenths: u32) -> (Vec<u32>, u32) {
    (Vec::new(), tenths)
}

fn verdicts(outcome: &WindowOutcome) -> Vec<Verdict> {
    outcome.attempts.iter().map(|a| a.verdict).collect()
}

/// The temperature in tenths and the average log-probability of the kept
/// decode.
fn kept_scores(outcome: &WindowOutcome) -> Option<(u32, f32)> {
    outcome
        .kept()
        .map(|a| ((a.temperature * 10.0).round() as u32, a.avg_logprob))
}

fn tenths(start: f32) -> Vec<u32> {
    temperature_ladder(start)
        .iter()
        .map(|t| (t * 10.0).round() as u32)
        .collect()
}

// ── What must not move ───────────────────────────────────────────────────────

/// A sure, clean window is decoded once, at the first temperature, with its
/// prompt; it is kept, it is not skipped and it prompts the next window.
#[test]
fn a_sure_clean_window_is_decoded_once_at_the_first_temperature() {
    let got = run(&PROMPT, 0.0, &[sure(clean())]);
    assert_eq!(got.calls, [with_prompt(0)]);
    assert_eq!(got.outcome.tokens, clean());
    assert_eq!(verdicts(&got.outcome), [Verdict::Accepted]);
    assert!(got.outcome.carry_text);
    assert!(!got.outcome.skipped);
}

/// A loop under the prompt is first decoded again at the same temperature
/// without the prompt. A clean result ends the window there. Each decode is
/// recorded with its prompt flag and the compression ratio of its own text.
#[test]
fn a_loop_under_the_prompt_is_first_decoded_again_without_it() {
    let got = run(&PROMPT, 0.0, &[sure(looping(8)), sure(clean())]);
    assert_eq!(got.calls, [with_prompt(0), without_prompt(0)]);
    assert_eq!(got.outcome.tokens, clean());
    assert_eq!(verdicts(&got.outcome), [Verdict::Loop, Verdict::Accepted]);
    assert!(got.outcome.carry_text);
    let recorded: Vec<(bool, f32)> = got
        .outcome
        .attempts
        .iter()
        .map(|a| (a.with_prompt, a.compression_ratio))
        .collect();
    let ratio = |tokens: &[u32]| compression_ratio(&sentences(tokens));
    assert_eq!(
        recorded,
        [(true, ratio(&looping(8))), (false, ratio(&clean()))]
    );
}

/// A loop that is also weak is a loop: it is decoded again at the same
/// temperature without the prompt, not at the next temperature with it.
#[test]
fn a_weak_loop_under_the_prompt_is_a_loop() {
    let script = [decoded(looping(8), -1.2), sure(clean())];
    let got = run(&PROMPT, 0.0, &script);
    assert_eq!(got.calls, [with_prompt(0), without_prompt(0)]);
    assert_eq!(verdicts(&got.outcome), [Verdict::Loop, Verdict::Accepted]);
}

/// With one temperature, a loop that remains without the prompt is kept and
/// does not prompt the next window.
#[test]
fn with_one_temperature_a_loop_that_remains_is_kept_and_not_carried() {
    let got = run(&PROMPT, 1.0, &[sure(looping(8)), sure(looping(6))]);
    assert_eq!(got.calls, [with_prompt(10), without_prompt(10)]);
    assert_eq!(got.outcome.tokens, looping(6));
    assert!(!got.outcome.carry_text);
    assert!(!got.outcome.skipped);
}

/// Of two loops with the same compression ratio, the later one is kept.
#[test]
fn of_two_loops_with_the_same_ratio_the_later_one_is_kept() {
    let mut later = looping(8);
    later.push(TOK_TIMESTAMP_BEGIN + 1450);
    let got = run(&PROMPT, 1.0, &[sure(looping(8)), sure(later.clone())]);
    assert_eq!(got.outcome.kept, 1);
    assert_eq!(got.outcome.tokens, later);
}

/// A window that is not skipped keeps the plan of its tokens.
#[test]
fn a_spoken_window_keeps_the_plan_of_its_tokens() {
    let got = run(&PROMPT, 0.0, &[sure(closed_then_open())]);
    let plan = plan_window(&got.outcome.tokens, SAMPLES_PER_WINDOW);
    assert_eq!(plan.segments.len(), 1);
    assert_eq!(plan.advance_samples, 16_000);
}

/// A loop decoded with no prompt has no prompt to drop: the next decode is at
/// the next temperature. With one temperature it is decoded once, kept and
/// not carried.
#[test]
fn a_loop_with_no_prompt_goes_to_the_next_temperature() {
    let got = run(&[], 0.0, &[sure(looping(8)), sure(clean())]);
    assert_eq!(got.calls, [without_prompt(0), without_prompt(2)]);
    assert_eq!(got.outcome.tokens, clean());
    assert!(got.outcome.carry_text);

    let got = run(&[], 1.0, &[sure(looping(8))]);
    assert_eq!(got.calls, [without_prompt(10)]);
    assert_eq!(got.outcome.tokens, looping(8));
    assert!(!got.outcome.carry_text);
}

/// The loop bound is the reference compression ratio, 2.4: one sentence four
/// times after four others compresses to about 2.3 and is accepted; five
/// times, to about 2.6, and is a loop. A ratio of exactly 2.4 is not a loop;
/// the next byte makes one.
#[test]
fn the_loop_bound_is_a_compression_ratio_above_2_4() {
    let one_verdict = |tokens: Vec<u32>| verdicts(&run(&[], 1.0, &[sure(tokens)]).outcome);
    assert_eq!(one_verdict(looping(4)), [Verdict::Accepted]);
    assert_eq!(one_verdict(looping(5)), [Verdict::Loop]);

    let text = sentences(&looping(5));
    for (len, verdict) in [(372, Verdict::Accepted), (373, Verdict::Loop)] {
        let window = text.get(..len).unwrap_or_default().to_owned();
        let result: Result<WindowOutcome, Infallible> = decode_with_fallback(
            SPOKEN,
            |_: usize| Ok(true),
            &[],
            1.0,
            |_: &[u32], _: f32| Ok(sure(vec![1])),
            |_: &[u32]| window.clone(),
        );
        match result {
            Ok(outcome) => assert_eq!(verdicts(&outcome), [verdict], "{len} bytes"),
            Err(never) => match never {},
        }
    }
}

/// A decode error is returned, also from the decode without the prompt.
#[test]
fn an_error_of_the_decode_without_the_prompt_is_returned() {
    let mut calls = 0;
    let result = decode_with_fallback(
        SPOKEN,
        |_: usize| Ok(true),
        &PROMPT,
        0.0,
        |p: &[u32], _: f32| {
            calls += 1;
            if p.is_empty() {
                Err("decode failed")
            } else {
                Ok(sure(looping(8)))
            }
        },
        sentences,
    );
    assert_eq!(result, Err("decode failed"));
    assert_eq!(calls, 2);
}

// ── The temperature ladder ───────────────────────────────────────────────────

/// A request temperature of 0 gives 0.0 to 1.0 in steps of 0.2. A higher start
/// gives the steps from it that do not pass 1.0.
#[test]
fn the_ladder_starts_at_the_request_temperature_and_ends_at_one() {
    assert_eq!(tenths(0.0), [0, 2, 4, 6, 8, 10]);
    assert_eq!(tenths(0.5), [5, 7, 9]);
    assert_eq!(tenths(0.9), [9]);
    assert_eq!(tenths(1.0), [10]);
    let want = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0];
    for (got, want) in temperature_ladder(0.0).iter().zip(want) {
        assert!((got - want).abs() < 1e-6, "{got} for {want}");
    }
}

/// A start above 1.0 is a ladder of that one step, and a window then has one
/// decode at that temperature. No start gives an empty ladder.
#[test]
fn a_start_above_one_is_one_step_at_that_temperature() {
    assert_eq!(tenths(1.5), [15]);
    assert_eq!(tenths(2.0), [20]);
    for start in [0.0, 0.3, 1.0, 1.01, 1.5, 100.0] {
        assert!(!temperature_ladder(start).is_empty(), "start {start}");
    }
    let got = run(&[], 1.5, &[weak(0, -1.2)]);
    assert_eq!(got.calls, [without_prompt(15)]);
    assert_eq!(got.outcome.tokens, spoken(0));
}

/// A loop that remains without the prompt goes up the ladder, and the prompt
/// stays dropped. The first decode that is not a loop is kept.
#[test]
fn a_loop_that_remains_goes_up_the_ladder_without_the_prompt() {
    let script = [sure(looping(8)), sure(looping(6)), sure(clean())];
    let got = run(&PROMPT, 0.0, &script);
    assert_eq!(
        got.calls,
        [with_prompt(0), without_prompt(0), without_prompt(2)]
    );
    assert_eq!(got.outcome.tokens, clean());
    assert_eq!(
        verdicts(&got.outcome),
        [Verdict::Loop, Verdict::Loop, Verdict::Accepted]
    );
    assert_eq!(kept_scores(&got.outcome), Some((2, SURE)));
    assert!(got.outcome.carry_text);
}

/// A weak decode goes up the ladder with the prompt. The scores of the window
/// are those of the kept decode, not of the first one.
#[test]
fn a_weak_decode_goes_up_the_ladder_with_the_prompt() {
    let script = [decoded(spoken(0), -1.2), decoded(spoken(1), -0.4)];
    let got = run(&PROMPT, 0.0, &script);
    assert_eq!(got.calls, [with_prompt(0), with_prompt(2)]);
    assert_eq!(got.outcome.tokens, spoken(1));
    assert_eq!(
        verdicts(&got.outcome),
        [Verdict::WeakLogprob, Verdict::Accepted]
    );
    assert_eq!(kept_scores(&got.outcome), Some((2, -0.4)));
}

/// A loop that first shows under the prompt at a later temperature is decoded
/// again at that temperature without the prompt.
#[test]
fn a_loop_at_a_later_temperature_drops_the_prompt_there() {
    let script = [weak(0, -1.2), sure(looping(8)), sure(clean())];
    let got = run(&PROMPT, 0.0, &script);
    assert_eq!(
        got.calls,
        [with_prompt(0), with_prompt(2), without_prompt(2)]
    );
    assert_eq!(got.outcome.tokens, clean());
}

/// The weak bound is an average log-probability below -1.0: exactly -1.0 is
/// not weak, and a value above it is not weak.
#[test]
fn the_weak_bound_is_an_average_log_probability_below_minus_one() {
    for (avg_logprob, decodes) in [(LOGPROB_THRESHOLD, 1), (-1.001, 2), (-0.1, 1), (-0.999, 1)] {
        let got = run(&[], 0.0, &[weak(0, avg_logprob), sure(spoken(1))]);
        assert_eq!(
            got.calls.len(),
            decodes,
            "average log-probability {avg_logprob}"
        );
    }
}

/// An average log-probability that is not finite is weak.
#[test]
fn a_log_probability_that_is_not_finite_is_weak() {
    for avg_logprob in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY] {
        let got = run(&[], 0.0, &[weak(0, avg_logprob), sure(spoken(1))]);
        assert_eq!(
            verdicts(&got.outcome),
            [Verdict::WeakLogprob, Verdict::Accepted],
            "average log-probability {avg_logprob}"
        );
        assert_eq!(got.outcome.tokens, spoken(1));
    }
}

// ── The decode that is kept when none is accepted ────────────────────────────

/// When every decode is a loop, the one with the lowest compression ratio is
/// kept, whatever its place in the order, and it does not prompt the next
/// window.
#[test]
fn of_loops_only_the_one_with_the_lowest_ratio_is_kept() {
    let ratio = |repeats| compression_ratio(&sentences(&looping(repeats)));
    assert!(ratio(5) < ratio(7) && ratio(7) < ratio(8) && ratio(8) < ratio(9));
    let script = [
        sure(looping(8)),
        sure(looping(5)),
        sure(looping(9)),
        sure(looping(7)),
    ];
    let got = run(&PROMPT, 0.6, &script);
    assert_eq!(
        got.calls,
        [
            with_prompt(6),
            without_prompt(6),
            without_prompt(8),
            without_prompt(10)
        ]
    );
    assert_eq!(got.outcome.kept, 1);
    assert_eq!(got.outcome.tokens, looping(5));
    assert!(!got.outcome.carry_text);
}

/// When every decode is weak and none is a loop, the one with the highest
/// average log-probability is kept, whatever its place in the order.
#[test]
fn of_weak_decodes_the_one_with_the_highest_log_probability_is_kept() {
    let script = [weak(0, -1.5), weak(1, -1.1), weak(2, -1.3)];
    let got = run(&[], 0.6, &script);
    assert_eq!(got.calls.len(), 3);
    assert_eq!(verdicts(&got.outcome), [Verdict::WeakLogprob; 3]);
    assert_eq!(got.outcome.kept, 1);
    assert_eq!(got.outcome.tokens, spoken(1));
    assert_eq!(kept_scores(&got.outcome), Some((8, -1.1)));
}

/// The full ladder with every decode weak: each decode has the prompt, the
/// best one is kept, and it does not prompt the next window although its
/// temperature is below 0.5. openai-whisper keeps the last decode here.
#[test]
fn when_every_decode_of_the_ladder_is_weak_the_best_is_kept_and_not_carried() {
    let script = [
        weak(0, -1.5),
        weak(1, -1.05),
        weak(2, -1.3),
        weak(3, -1.2),
        weak(4, -1.4),
        weak(5, -1.6),
    ];
    let got = run(&PROMPT, 0.0, &script);
    let ladder: Vec<(Vec<u32>, u32)> = [0, 2, 4, 6, 8, 10].map(with_prompt).to_vec();
    assert_eq!(got.calls, ladder);
    assert_eq!(got.outcome.kept, 1);
    assert_eq!(got.outcome.tokens, spoken(1));
    assert_eq!(kept_scores(&got.outcome), Some((2, -1.05)));
    assert!(!got.outcome.carry_text);
    assert!(!got.outcome.skipped);
}

/// A weak decode that is not a loop is kept before a loop, even a sure one.
#[test]
fn a_weak_decode_is_kept_before_a_loop() {
    let script = [
        decoded(looping(8), -0.2),
        decoded(looping(6), -0.2),
        weak(0, -1.4),
        decoded(looping(7), -0.1),
    ];
    let got = run(&PROMPT, 0.6, &script);
    assert_eq!(got.calls.len(), 4);
    assert_eq!(got.outcome.kept, 2);
    assert_eq!(got.outcome.tokens, spoken(0));
}

/// Of two weak decodes with the same average log-probability, the later one
/// is kept.
#[test]
fn of_two_weak_decodes_with_the_same_log_probability_the_later_one_is_kept() {
    let got = run(&[], 0.8, &[weak(0, -1.2), weak(1, -1.2)]);
    assert_eq!(got.outcome.kept, 1);
    assert_eq!(got.outcome.tokens, spoken(1));
}

// ── The prompt for the next window ───────────────────────────────────────────

/// A decode accepted at a temperature above 0.5 does not prompt the next
/// window; one accepted at 0.5 or below does.
#[test]
fn a_decode_accepted_above_temperature_0_5_does_not_prompt_the_next_window() {
    let script = [weak(0, -1.2), weak(1, -1.2), weak(2, -1.2), sure(spoken(3))];
    let at_0_6 = run(&[], 0.0, &script);
    assert_eq!(kept_scores(&at_0_6.outcome), Some((6, SURE)));
    assert!(!at_0_6.outcome.carry_text, "accepted at 0.6");

    let script = [weak(0, -1.2), weak(1, -1.2), sure(spoken(2))];
    let at_0_4 = run(&[], 0.0, &script);
    assert_eq!(kept_scores(&at_0_4.outcome), Some((4, SURE)));
    assert!(at_0_4.outcome.carry_text, "accepted at 0.4");

    let at_bound = run(&[], PROMPT_RESET_TEMPERATURE, &[sure(clean())]);
    assert_eq!(at_bound.calls, [without_prompt(5)]);
    assert!(at_bound.outcome.carry_text, "accepted at exactly 0.5");
}

// ── The no-speech gate ───────────────────────────────────────────────────────

/// The second condition of the skip: a window above the no-speech bound is
/// skipped only when its greedy decode with no prompt writes at most 8 text
/// tokens. The bounded decode runs one time, with that bound.
#[test]
fn a_window_above_the_no_speech_bound_with_a_short_text_is_skipped() {
    let got = run_gate(0.9, true, &PROMPT, 0.0, &[sure(clean())]);
    assert_eq!(got.bounded_decodes, [8]);
    assert!(got.outcome.skipped);
    assert!(got.calls.is_empty(), "decodes {:?}", got.calls);
}

/// A window above the no-speech bound whose greedy decode with no prompt
/// writes more than the bound is not skipped: it is decoded as each window
/// is, with its prompt, the retry without the prompt and the ladder.
#[test]
fn a_window_above_the_no_speech_bound_with_a_long_text_is_decoded() {
    for no_speech_prob in [0.6001, 0.9193, 1.0] {
        let got = run_gate(no_speech_prob, false, &PROMPT, 0.0, &[sure(clean())]);
        assert_eq!(got.bounded_decodes, [8], "no-speech {no_speech_prob}");
        assert!(!got.outcome.skipped, "no-speech {no_speech_prob}");
        assert_eq!(got.calls, [with_prompt(0)], "no-speech {no_speech_prob}");
        assert_eq!(got.outcome.tokens, clean());
        assert!(got.outcome.carry_text);
    }
    let script = [sure(looping(8)), sure(looping(6)), sure(clean())];
    let got = run_gate(0.9, false, &PROMPT, 0.0, &script);
    assert_eq!(
        got.calls,
        [with_prompt(0), without_prompt(0), without_prompt(2)]
    );
}

/// A window at or below the no-speech bound is decoded, and the bounded
/// decode of the gate does not run for it, short text or long.
#[test]
fn the_bounded_decode_does_not_run_at_or_below_the_no_speech_bound() {
    for no_speech_prob in [NO_SPEECH_THRESHOLD, 0.5999, 0.53, 0.0, f32::NAN] {
        for short_text in [true, false] {
            let got = run_gate(no_speech_prob, short_text, &PROMPT, 0.0, &[sure(clean())]);
            assert!(got.bounded_decodes.is_empty(), "no-speech {no_speech_prob}");
            assert!(!got.outcome.skipped, "no-speech {no_speech_prob}");
            assert_eq!(got.calls, [with_prompt(0)], "no-speech {no_speech_prob}");
        }
    }
}

/// An error of the bounded decode of the gate is the error of the window.
#[test]
fn an_error_of_the_bounded_decode_is_returned() {
    let result = decode_with_fallback(
        0.9,
        |_: usize| Err("bounded decode failed"),
        &PROMPT,
        0.0,
        |_: &[u32], _: f32| Ok(sure(clean())),
        sentences,
    );
    assert_eq!(result, Err("bounded decode failed"));
}

/// A window above the no-speech bound whose text is short is skipped before
/// any decode of the ladder, at each request temperature: no fallback runs.
#[test]
fn a_window_above_the_no_speech_bound_is_skipped_before_any_decode() {
    for no_speech_prob in [0.6001, 0.61, 0.7076, 0.9, 1.0] {
        for start in [0.0, 0.4, 0.8, 1.5] {
            let got = run_window(no_speech_prob, &PROMPT, start, &[sure(clean())]);
            let case = format!("no-speech {no_speech_prob}, temperature {start}");
            assert!(got.calls.is_empty(), "{case}: decodes {:?}", got.calls);
            assert!(got.outcome.skipped, "{case}");
            assert!(got.outcome.tokens.is_empty(), "{case}");
            assert!(got.outcome.attempts.is_empty(), "{case}");
            assert_eq!(got.outcome.kept(), None, "{case}");
        }
    }
}

/// A no-speech probability of exactly 0.6 does not skip, and no value below
/// it does. A probability that is not a number is no sign of silence.
#[test]
fn a_no_speech_probability_of_exactly_the_bound_does_not_skip() {
    for no_speech_prob in [NO_SPEECH_THRESHOLD, 0.5999, 0.59, 0.31, 0.0, f32::NAN] {
        let got = run_window(no_speech_prob, &PROMPT, 0.0, &[sure(clean())]);
        assert_eq!(got.calls, [with_prompt(0)], "no-speech {no_speech_prob}");
        assert!(!got.outcome.skipped, "no-speech {no_speech_prob}");
        assert_eq!(got.outcome.tokens, clean(), "no-speech {no_speech_prob}");
    }
}

/// The skip does not read an average log-probability. A window above the
/// bound is skipped when its decode would be sure, and a window below the
/// bound is not skipped when each of its decodes is weak.
#[test]
fn the_skip_does_not_read_the_log_probability() {
    let sure_filler = decoded(clean(), -0.2);
    let got = run_window(0.9, &[], 0.0, &[sure_filler]);
    assert!(got.calls.is_empty(), "decodes {:?}", got.calls);
    assert!(got.outcome.skipped);

    for avg_logprob in [-1.5, -5.0, f32::NEG_INFINITY, f32::NAN] {
        let got = run_window(0.59, &[], 1.0, &[weak(0, avg_logprob)]);
        assert_eq!(got.calls.len(), 1, "log-probability {avg_logprob}");
        assert!(!got.outcome.skipped, "log-probability {avg_logprob}");
        assert_eq!(got.outcome.tokens, spoken(0));
    }
}

/// A skipped window gives no segment and consumes all its real samples, and
/// the previous-text prompt stays as it is.
#[test]
fn a_skipped_window_gives_no_segment_consumes_the_window_and_leaves_the_prompt() {
    let got = run_window(0.9, &PROMPT, 0.0, &[sure(closed_then_open())]);
    for window_samples in [SAMPLES_PER_WINDOW, 86_551, 1_600] {
        let plan = plan_window(&got.outcome.tokens, window_samples);
        assert!(plan.segments.is_empty(), "segments: {:?}", plan.segments);
        assert_eq!(plan.advance_samples, window_samples);
    }
    let next = next_prompt(
        PROMPT.to_vec(),
        &got.outcome,
        true,
        "",
        |_: &str| vec![99],
        10,
    );
    assert_eq!(next, PROMPT.to_vec());
}
