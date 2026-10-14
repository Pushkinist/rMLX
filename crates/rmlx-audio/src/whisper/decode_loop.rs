//! The host side of the Whisper decoder for one window: the read of the start
//! of the window, which token is picked from each logit row of a decode, and
//! the average log-probability of the result. No model: the decoder forward
//! and the token pick are closures.
//!
//! `forward(input)` feeds `input` to the decoder after the tokens fed before
//! it and returns the logit row of the last position of `input`.

use super::{logsumexp, DecodeFilters};
use crate::tokenizer::{TOK_EN, TOK_EOT, TOK_LANG_LAST, TOK_NOSPEECH, TOK_SOT};

/// What the logit row of `<|startoftranscript|>` says about a window when no
/// token comes before it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct WindowStart {
    /// The language token with the largest logit.
    pub(crate) language: u32,
    /// The softmax share of `<|nospeech|>` in the unfiltered row. A
    /// previous-text prompt before the start token moves this value, so the
    /// row is read with none.
    pub(crate) no_speech_prob: f32,
}

/// Read the start of a window with one decoder forward of the start token
/// alone, on a decoder that was fed nothing.
pub(crate) fn window_start<E>(
    forward: impl FnOnce(&[u32]) -> Result<Vec<f32>, E>,
) -> Result<WindowStart, E> {
    let row = forward(&[TOK_SOT])?;
    let languages = row
        .get(TOK_EN as usize..=TOK_LANG_LAST as usize)
        .unwrap_or_default();
    let mut language = 0;
    for (index, logit) in languages.iter().enumerate() {
        if languages.get(language).is_some_and(|best| logit > best) {
            language = index;
        }
    }
    Ok(WindowStart {
        language: TOK_EN + language as u32,
        no_speech_prob: log_probability(&row, TOK_NOSPEECH).exp(),
    })
}

/// The result of one decode of a window.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Decoded {
    /// The sampled tokens after the prefix, with the end token stripped.
    pub(crate) tokens: Vec<u32>,
    /// openai-whisper's `avg_logprob`: the sum of the log-probability of each
    /// sampled token and of the end token, each under the log-softmax of its
    /// filtered row at temperature 1, over the token count plus one.
    pub(crate) avg_logprob: f32,
    /// What ended the decode.
    pub(crate) stop: Stop,
}

/// What ended a decode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stop {
    /// The end token was picked.
    EndToken,
    /// The token budget of the window was reached.
    TokenBudget,
    /// The next decoder position is past the decoder context.
    Context,
}

/// Decode one window from `prefix`. `pick(row)` chooses one token from a
/// filtered row: the caller owns the temperature and the generator, and every
/// sampled token, the end token included, is one call.
///
/// `max_tokens` bounds the sampled tokens and `n_text_ctx` the decoder
/// position; the first token is always picked.
pub(crate) fn decode_loop<E>(
    prefix: &[u32],
    max_tokens: usize,
    n_text_ctx: usize,
    filters: &DecodeFilters,
    mut forward: impl FnMut(&[u32]) -> Result<Vec<f32>, E>,
    mut pick: impl FnMut(&[f32]) -> Result<u32, E>,
) -> Result<Decoded, E> {
    let mut tokens: Vec<u32> = Vec::new();
    let mut sum_logprob = 0.0_f32;
    let mut row = forward(prefix)?;
    let mut fed = prefix.len();
    let stop = loop {
        filters.apply(&mut row, &tokens, tokens.is_empty());
        let next = pick(&row)?;
        sum_logprob += log_probability(&row, next);
        if next == TOK_EOT {
            break Stop::EndToken;
        }
        tokens.push(next);
        if tokens.len() >= max_tokens {
            break Stop::TokenBudget;
        }
        if fed >= n_text_ctx {
            break Stop::Context;
        }
        row = forward(&[next])?;
        fed += 1;
    };
    let avg_logprob = sum_logprob / (tokens.len() as f32 + 1.0);
    Ok(Decoded {
        tokens,
        avg_logprob,
        stop,
    })
}

/// Whether a decode of a window from `prefix` writes at most `text_bound` text
/// tokens. It is [`decode_loop`] with a pick that ends the decode when the text
/// passes the bound, so it decodes at most `text_bound + 1` text tokens. A
/// text token is an id below the end token: a timestamp token and the end
/// token are not text.
pub(crate) fn writes_at_most<E>(
    text_bound: usize,
    prefix: &[u32],
    max_tokens: usize,
    n_text_ctx: usize,
    filters: &DecodeFilters,
    mut forward: impl FnMut(&[u32]) -> Result<Vec<f32>, E>,
    mut pick: impl FnMut(&[f32]) -> Result<u32, E>,
) -> Result<bool, E> {
    /// Why the bounded decode ended before its end token.
    enum Ended<E> {
        PastTheBound,
        Failed(E),
    }
    let mut text_tokens = 0;
    let decoded = decode_loop(
        prefix,
        max_tokens,
        n_text_ctx,
        filters,
        |input: &[u32]| forward(input).map_err(Ended::Failed),
        |row: &[f32]| {
            let token = pick(row).map_err(Ended::Failed)?;
            text_tokens += usize::from(token < TOK_EOT);
            if text_tokens > text_bound {
                return Err(Ended::PastTheBound);
            }
            Ok(token)
        },
    );
    match decoded {
        Ok(_) => Ok(true),
        Err(Ended::PastTheBound) => Ok(false),
        Err(Ended::Failed(error)) => Err(error),
    }
}

/// The log-probability of `token` under the softmax of `row`. A token that is
/// not in the row has none: the result is `-inf`.
fn log_probability(row: &[f32], token: u32) -> f32 {
    row.get(token as usize)
        .map_or(f32::NEG_INFINITY, |logit| logit - logsumexp(row))
}

#[cfg(test)]
#[path = "decode_loop_tests.rs"]
mod decode_loop_tests;
