// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout)]
//! `rmlx eval spec-answer` — judge a speculative answer against a plain one.
//!
//! Reads two non-streamed `/v1/chat/completions` bodies carrying `logprobs`,
//! one from a no-drafter run and one from a speculative run of the same prompt
//! at temperature 0, and applies the first-divergence oracle of
//! [`rmlx_models::answer_equivalence`] to them. It loads no model, takes no
//! claim and writes nothing but its verdict.
//!
//! Positions are record indices. A record's `token` is the vocabulary piece of
//! the token the engine emitted, which is one to one with its id inside one
//! tokenizer, so interning the pieces to integers keeps equality of ids. The
//! engine sends no record for a forced token, which is why the record count is
//! held to `usage.completion_tokens`: a shifted position would read the wrong
//! margin.
//!
//! Stdout is `key=value` lines: `verdict=near_tie|refused|unjudgeable`, then
//! `position=`, `margin=` and `rank=` where they were computed, then
//! `reason=`. Exit code 0 is a near tie, 1 a divergence the oracle refuses, 2 an
//! input it cannot judge.

use std::collections::HashMap;
use std::path::Path;

use rmlx_models::answer_equivalence::{
    common_prefix_len, read_divergence, top_two_margin, Divergence, MAX_DIVERGENCE_CONFIDENCE,
    MIN_LENGTH_RATIO,
};
use serde::Deserialize;

/// What the oracle made of a pair of answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Verdict {
    /// The first divergence sits at a decision the verifier was nearly tied on.
    NearTie,
    /// The pair diverges where the verifier was not nearly tied, or one run
    /// ended or ran on where the other did not.
    Refused,
    /// The inputs cannot be read as a comparison.
    Unjudgeable,
}

impl Verdict {
    fn label(self) -> &'static str {
        match self {
            Self::NearTie => "near_tie",
            Self::Refused => "refused",
            Self::Unjudgeable => "unjudgeable",
        }
    }

    fn exit_code(self) -> i32 {
        match self {
            Self::NearTie => 0,
            Self::Refused => 1,
            Self::Unjudgeable => 2,
        }
    }
}

/// A verdict with the figures behind it.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Judgement {
    pub(crate) verdict: Verdict,
    pub(crate) position: Option<usize>,
    pub(crate) margin: Option<f32>,
    pub(crate) rank: Option<f64>,
    pub(crate) reason: String,
}

impl Judgement {
    fn unjudgeable(reason: String) -> Self {
        Self {
            verdict: Verdict::Unjudgeable,
            position: None,
            margin: None,
            rank: None,
            reason,
        }
    }

    fn lines(&self) -> Vec<String> {
        let mut lines = vec![format!("verdict={}", self.verdict.label())];
        if let Some(position) = self.position {
            lines.push(format!("position={position}"));
        }
        if let Some(margin) = self.margin {
            lines.push(format!("margin={margin:.4}"));
        }
        if let Some(rank) = self.rank {
            lines.push(format!("rank={rank:.4}"));
        }
        lines.push(format!("reason={}", self.reason));
        lines
    }
}

#[derive(Deserialize)]
struct Body {
    choices: Option<Vec<Choice>>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    logprobs: Option<Logprobs>,
}

#[derive(Deserialize)]
struct Logprobs {
    content: Option<Vec<Record>>,
}

#[derive(Deserialize)]
struct Record {
    token: String,
    #[serde(default)]
    top_logprobs: Vec<Candidate>,
}

#[derive(Deserialize)]
struct Candidate {
    logprob: f32,
}

#[derive(Deserialize)]
struct Usage {
    completion_tokens: Option<u64>,
}

/// One response, as far as the oracle reads it.
struct Answer {
    tokens: Vec<String>,
    top_logprobs: Vec<Vec<f32>>,
}

/// The records of one response, or why it cannot be read as an answer.
fn read_answer(name: &str, text: &str) -> Result<Answer, String> {
    let body: Body =
        serde_json::from_str(text).map_err(|e| format!("{name}: not a chat completion: {e}"))?;
    let records = body
        .choices
        .and_then(|choices| choices.into_iter().next())
        .and_then(|choice| choice.logprobs)
        .and_then(|logprobs| logprobs.content)
        .ok_or_else(|| format!("{name}: no choices[0].logprobs.content"))?;
    let completion_tokens = body
        .usage
        .and_then(|usage| usage.completion_tokens)
        .ok_or_else(|| format!("{name}: no usage.completion_tokens"))?;
    if u64::try_from(records.len()).ok() != Some(completion_tokens) {
        return Err(format!(
            "{name}: {} logprob records for {completion_tokens} completion tokens, so a \
             record index is not a token position",
            records.len()
        ));
    }
    Ok(Answer {
        tokens: records.iter().map(|r| r.token.clone()).collect(),
        top_logprobs: records
            .iter()
            .map(|r| r.top_logprobs.iter().map(|c| c.logprob).collect())
            .collect(),
    })
}

/// One integer per distinct piece, the same map for every stream it is given.
fn intern<'a>(map: &mut HashMap<&'a str, u32>, tokens: &'a [String]) -> Vec<u32> {
    tokens
        .iter()
        .map(|token| {
            let next = u32::try_from(map.len()).unwrap_or(u32::MAX);
            *map.entry(token.as_str()).or_insert(next)
        })
        .collect()
}

/// Judge the speculative response `spec_text` against the no-drafter response
/// `plain_text`. `plain_name` and `spec_name` label the inputs in a reason.
pub(crate) fn judge_bodies(
    plain_name: &str,
    plain_text: &str,
    spec_name: &str,
    spec_text: &str,
) -> Judgement {
    let (plain, spec) = match (
        read_answer(plain_name, plain_text),
        read_answer(spec_name, spec_text),
    ) {
        (Ok(plain), Ok(spec)) => (plain, spec),
        (Err(reason), _) | (_, Err(reason)) => return Judgement::unjudgeable(reason),
    };
    if let Some(at) = plain.top_logprobs.iter().position(|top| top.len() < 2) {
        return Judgement::unjudgeable(format!(
            "{plain_name}: record {at} carries fewer than two top_logprobs, so it has no \
             margin to read; ask for top_logprobs of at least 2"
        ));
    }
    if plain.tokens == spec.tokens {
        return Judgement::unjudgeable(format!(
            "{plain_name} and {spec_name} hold the same token pieces, so the oracle sees no \
             difference between them"
        ));
    }

    let mut ids = HashMap::new();
    let plain_ids = intern(&mut ids, &plain.tokens);
    let spec_ids = intern(&mut ids, &spec.tokens);
    let margins: Vec<f32> = plain
        .top_logprobs
        .iter()
        .map(|top| top_two_margin(top.iter().copied()))
        .collect();

    if plain_ids.is_empty() {
        return Judgement::unjudgeable(format!("{plain_name} holds no tokens to compare against"));
    }

    let refused = |position, rank, reason| Judgement {
        verdict: Verdict::Refused,
        position,
        margin: position.and_then(|p| margins.get(p).copied()),
        rank,
        reason,
    };
    match read_divergence(&spec_ids, &plain_ids, &margins) {
        Divergence::Unread => refused(
            Some(common_prefix_len(&spec_ids, &plain_ids)),
            None,
            format!(
                "the speculative answer reproduced the whole reference answer of {} tokens \
                 and ran on to {} — the round loop did not stop where the verifier stopped",
                plain_ids.len(),
                spec_ids.len()
            ),
        ),
        Divergence::ShortRun { ratio, at } => refused(
            at.map(|(position, _)| position),
            at.map(|(_, rank)| rank),
            format!(
                "the answers have {} and {} tokens, a ratio of {ratio:.4} (floor \
                 {MIN_LENGTH_RATIO}) — one stopped well before the other",
                spec_ids.len(),
                plain_ids.len()
            ),
        ),
        Divergence::Confident(position, rank) => refused(
            Some(position),
            Some(rank),
            format!(
                "first differs from the no-drafter arm at token {position}, where the \
                 verifier was surer than {rank:.4} of its own decisions on this answer \
                 (ceiling {MAX_DIVERGENCE_CONFIDENCE})"
            ),
        ),
        Divergence::NearTie(position, rank) => Judgement {
            verdict: Verdict::NearTie,
            position: Some(position),
            margin: margins.get(position).copied(),
            rank: Some(rank),
            reason: format!(
                "first differs from the no-drafter arm at token {position}, a decision the \
                 verifier was nearly tied on: it was surer than {rank:.4} of its own \
                 decisions (ceiling {MAX_DIVERGENCE_CONFIDENCE})"
            ),
        },
    }
}

/// Run `rmlx eval spec-answer` and return the exit code.
pub(crate) fn run_spec_answer(plain: &Path, spec: &Path) -> i32 {
    let judgement = match (
        std::fs::read_to_string(plain),
        std::fs::read_to_string(spec),
    ) {
        (Ok(plain_text), Ok(spec_text)) => judge_bodies(
            &plain.display().to_string(),
            &plain_text,
            &spec.display().to_string(),
            &spec_text,
        ),
        (Err(e), _) => Judgement::unjudgeable(format!("{}: {e}", plain.display())),
        (_, Err(e)) => Judgement::unjudgeable(format!("{}: {e}", spec.display())),
    };
    for line in judgement.lines() {
        println!("{line}");
    }
    judgement.verdict.exit_code()
}

#[cfg(test)]
#[path = "spec_answer_tests.rs"]
mod tests;
