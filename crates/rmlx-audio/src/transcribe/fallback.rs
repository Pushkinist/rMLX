//! The decision for one window of the long-form transcription: whether the
//! window is skipped, which decodes run, at which temperature and with which
//! prompt, and which one is kept. Pure functions, no model.

use tracing::{debug, warn};

use super::{compression_ratio, COMPRESSION_RATIO_LOOP};
use crate::whisper::decode_loop::Decoded;

/// No-speech probability above which a window is skipped (the value of
/// openai-whisper's `no_speech_threshold`).
const NO_SPEECH_THRESHOLD: f32 = 0.6;

/// The most text tokens that the greedy decode of a window with no prompt can
/// write for the window to be skipped. Measured on large-v3: each window with
/// no speech wrote at most 3 (` you`, ` Thank you.`); the spoken window above
/// the no-speech bound that the model can transcribe wrote 223, the token
/// budget. 8 leaves room for a longer stock phrase, and it is the most text
/// that a skip can lose.
const STOCK_PHRASE_TEXT_TOKENS: usize = 8;

/// openai-whisper's `logprob_threshold`. A decode whose average log-probability
/// is below it, or is not finite, is weak.
const LOGPROB_THRESHOLD: f32 = -1.0;

/// Temperature above which an accepted decode does not prompt the next window,
/// as in openai-whisper.
const PROMPT_RESET_TEMPERATURE: f32 = 0.5;

/// The distance between two steps of the temperature ladder, its last step,
/// and how many steps of that distance go from 0 to the last step, as the
/// openai-whisper command line builds them.
const LADDER_STEP: f32 = 0.2;
const LADDER_END: f32 = 1.0;
const LADDER_STEPS: u8 = 5;

/// The temperatures of a window's decodes, in order: `start`, then steps of 0.2
/// that do not pass 1.0. A start above 1.0 gives that one step: the ladder is
/// never empty.
fn temperature_ladder(start: f32) -> Vec<f32> {
    let higher = (1..=LADDER_STEPS)
        .map(|step| start + LADDER_STEP * f32::from(step))
        .take_while(|&temperature| temperature <= LADDER_END + 1e-6);
    std::iter::once(start).chain(higher).collect()
}

/// How one decode of a window was judged. The tests run in this order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Verdict {
    /// The compression ratio of the text is above 2.4.
    Loop,
    /// The average log-probability is below -1.0 or is not finite.
    WeakLogprob,
    /// None of the above: no further decode runs.
    Accepted,
}

/// One decode of a window and the values its verdict was read from.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct Attempt {
    pub(super) temperature: f32,
    pub(super) with_prompt: bool,
    pub(super) compression_ratio: f32,
    pub(super) avg_logprob: f32,
    pub(super) verdict: Verdict,
}

impl Attempt {
    /// Whether this decode is kept before `earlier`: a decode that is not a
    /// loop before a loop, then the higher average log-probability, or the
    /// lower compression ratio of two loops. Of two equal decodes the later
    /// one is kept. An accepted decode has a higher average than a weak one.
    fn is_kept_before(&self, earlier: &Self) -> bool {
        let is_loop = |attempt: &Self| attempt.verdict == Verdict::Loop;
        let finite_or_lowest = |avg: f32| if avg.is_finite() { avg } else { f32::MIN };
        match (is_loop(self), is_loop(earlier)) {
            (true, true) => self.compression_ratio <= earlier.compression_ratio,
            (true, false) => false,
            (false, true) => true,
            (false, false) => {
                finite_or_lowest(self.avg_logprob) >= finite_or_lowest(earlier.avg_logprob)
            }
        }
    }
}

/// What the decision for one window gave.
#[derive(Debug, Clone, PartialEq)]
pub(super) struct WindowOutcome {
    /// The tokens of the kept decode. A skipped window has none, so its plan
    /// has no segment and consumes the whole window.
    pub(super) tokens: Vec<u32>,
    /// Every decode that ran, in order. A skipped window has none.
    pub(super) attempts: Vec<Attempt>,
    /// Index in `attempts` of the kept decode.
    pub(super) kept: usize,
    /// The no-speech probability of the window is above 0.6 and the model
    /// writes no more than a stock phrase for it: no decode of the ladder ran,
    /// the window gives no segment, all its samples are consumed and the
    /// prompt stays as it is.
    pub(super) skipped: bool,
    /// Whether the window leaves the previous-text prompt in use: the kept
    /// decode is accepted at a temperature of 0.5 or below and its text
    /// becomes the prompt, or the window is skipped and the prompt stays.
    pub(super) carry_text: bool,
}

impl WindowOutcome {
    /// The kept decode; `None` for a skipped window.
    pub(super) fn kept(&self) -> Option<&Attempt> {
        self.attempts.get(self.kept)
    }

    /// Record the decision for the window that starts at sample `seek`.
    pub(super) fn log(&self, seek: usize, no_speech_prob: f32) {
        let Some(kept) = self.kept() else {
            debug!(seek, no_speech_prob, "no speech in the window; skipped");
            return;
        };
        if kept.verdict != Verdict::Accepted {
            warn!(
                seek,
                decodes = self.attempts.len(),
                verdict = ?kept.verdict,
                temperature = kept.temperature,
                compression_ratio = kept.compression_ratio,
                avg_logprob = kept.avg_logprob,
                "no decode of the window was accepted; the best one is kept and does not prompt the next window"
            );
        } else if self.attempts.len() > 1 {
            debug!(
                seek,
                decodes = self.attempts.len(),
                temperature = kept.temperature,
                with_prompt = kept.with_prompt,
                "the first decode of the window was a loop or weak; a later decode is kept"
            );
        }
    }
}

/// Decide one window. `no_speech_prob` is the no-speech probability of the
/// window, read with no previous-text prompt. `writes_at_most(n)` says whether
/// the greedy decode of the window with no prompt writes at most `n` text
/// tokens. `decode(prompt, temperature)` runs the decoder with that
/// previous-text prompt (empty for none) at that temperature; `text_of` gives
/// the text of a token stream. The temperatures are
/// [`temperature_ladder`]`(start)`.
///
/// A window is skipped, before any decode of the ladder, when both hold: its
/// no-speech probability is above 0.6, and the model writes no more than a
/// stock phrase for it ([`STOCK_PHRASE_TEXT_TOKENS`]). The second condition
/// is read only for a window that has the first. It is there because the
/// probability alone is above 0.6 for 30 s of a spoken repetition, which the
/// model transcribes. The skip does not read an average log-probability:
/// openai-whisper reads the probability behind the prompt and skips only a
/// decode that is also weak, and that rule keeps the stock text of a window
/// with no speech.
///
/// The decodes run up the ladder and stop at the first one that is accepted,
/// as in openai-whisper. Two more rules are this project's and not the
/// reference's:
/// - A loop under the prompt is decoded again at the same temperature without
///   the prompt, and the prompt stays dropped for the rest of the window.
///   openai-whisper gives every decode of a window the same prompt.
/// - When no decode is accepted, the kept one is the decode with the highest
///   average log-probability among those that are not loops (faster-whisper's
///   `generate_with_fallback` has this rule), or the decode with the lowest
///   compression ratio when all are loops; of two equal decodes the later one
///   is kept. openai-whisper keeps the last decode.
pub(super) fn decode_with_fallback<E>(
    no_speech_prob: f32,
    writes_at_most: impl FnOnce(usize) -> Result<bool, E>,
    mut prompt: &[u32],
    start: f32,
    mut decode: impl FnMut(&[u32], f32) -> Result<Decoded, E>,
    text_of: impl Fn(&[u32]) -> String,
) -> Result<WindowOutcome, E> {
    let mut outcome = WindowOutcome {
        tokens: Vec::new(),
        attempts: Vec::new(),
        kept: 0,
        skipped: no_speech_prob > NO_SPEECH_THRESHOLD && writes_at_most(STOCK_PHRASE_TEXT_TOKENS)?,
        carry_text: true,
    };
    if outcome.skipped {
        return Ok(outcome);
    }
    let mut temperatures = temperature_ladder(start).into_iter().peekable();
    while let Some(&temperature) = temperatures.peek() {
        let decoded = decode(prompt, temperature)?;
        let ratio = compression_ratio(&text_of(&decoded.tokens));
        let weak = !decoded.avg_logprob.is_finite() || decoded.avg_logprob < LOGPROB_THRESHOLD;
        let attempt = Attempt {
            temperature,
            with_prompt: !prompt.is_empty(),
            compression_ratio: ratio,
            avg_logprob: decoded.avg_logprob,
            verdict: if ratio > COMPRESSION_RATIO_LOOP {
                Verdict::Loop
            } else if weak {
                Verdict::WeakLogprob
            } else {
                Verdict::Accepted
            },
        };
        debug!(
            temperature,
            with_prompt = attempt.with_prompt,
            compression_ratio = attempt.compression_ratio,
            avg_logprob = attempt.avg_logprob,
            verdict = ?attempt.verdict,
            "window decode judged"
        );
        if outcome
            .kept()
            .is_none_or(|kept| attempt.is_kept_before(kept))
        {
            outcome.kept = outcome.attempts.len();
            outcome.tokens = decoded.tokens;
        }
        let verdict = attempt.verdict;
        outcome.attempts.push(attempt);
        match verdict {
            Verdict::Accepted => break,
            Verdict::Loop if !prompt.is_empty() => prompt = &[],
            Verdict::Loop | Verdict::WeakLogprob => {
                temperatures.next();
            }
        }
    }
    outcome.carry_text = outcome.kept().is_some_and(|kept| {
        kept.verdict == Verdict::Accepted && kept.temperature <= PROMPT_RESET_TEMPERATURE
    });
    Ok(outcome)
}

#[cfg(test)]
#[path = "fallback_tests.rs"]
mod fallback_tests;
