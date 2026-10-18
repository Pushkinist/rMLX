//! The divergence oracle for two decodes of one prompt.
//!
//! Greedy speculative decoding emits the verifier's own argmax at every
//! position, so a speculative run and a plain run of the same verifier are two
//! ways of computing one answer. They are not bit-identical: the verify pass
//! scores a block in one forward where plain decode steps one token at a time,
//! and that is a different reduction order. A reduction-order difference can
//! only flip a decision the verifier was already nearly indifferent about.
//!
//! [`divergence_confidence`] reads the verifier's top-two logprob margin at the
//! position the two runs first differ and returns where that margin sits in the
//! reference run's own margin distribution. It is a rank, not a number of nats,
//! so one ceiling covers models whose logits are not on the same scale. See
//! `docs/SPEC_ANSWER_EQUIVALENCE.md`.

/// The shorter run over the longer. `lcs_ratio`-style agreement measures divide
/// by the shorter stream, so a run that stopped at a third of the other's
/// length and matched its prefix would otherwise score 1.0.
pub const MIN_LENGTH_RATIO: f64 = 0.60;

/// Where the first divergence's own confidence may sit in the reference run's
/// margin distribution.
///
/// **Both sides measured**, over six pairs and six prompts each, by running the
/// gate against the shipped engine and against a deliberately broken one per
/// loop — two on the block pair and two on the restricted-vocabulary one:
///
/// | engine | percentile of the reference arm's own margins |
/// |---|---|
/// | assistant pair, as shipped | 0.0000 to 0.0820 |
/// | recurrent pair, as shipped | 0.0000 to 0.0234 |
/// | block pair, as shipped | 0.0000 to 0.0273 |
/// | adaptive pair, as shipped, at the block it is served | 0.0000 to 0.0234 |
/// | adaptive pair, as shipped, over its whole schedule | 0.0000 to 0.0703 |
/// | restricted-vocabulary pair, as shipped | 0.0000 to 0.0703 |
/// | two-model pair, as shipped | 0.0000 to 0.0234 |
/// | assistant pair, SWA ring keeping its rejected block tail | 0.4219 to 0.9258 |
/// | recurrent pair, acceptance walk without the final norm | 0.0000 to 0.5000 |
/// | block pair, rejected tail never rolled off | 0.0117 to 0.9297 |
/// | block pair, one rejected draft kept every partial round | 0.1758 to 0.6406 |
/// | adaptive pair, one rejected draft kept every partial round | 0.0703 to 0.8320 |
/// | restricted-vocabulary pair, one rejected draft kept every round | 0.0000 to 0.8320 |
/// | two-model pair, one rejected draft kept every partial round | 0.0000 to 0.6680 |
///
/// The measurement leaves a band, and the value sits inside it: above the worst
/// correct cell (0.0820, 1.46x) and under the lowest broken cell above it
/// (0.1445, 1.20x). It clears every correct cell, and every broken engine here
/// is refused on at least four of the prompts the gate judged for it — which is
/// what running every prompt rather than one buys, since no broken engine is
/// refused on all of them by this oracle alone.
///
/// **The adaptive pair's two rows are the same engine at two blocks**, and the
/// wider one reads three times the narrower. Both clear the ceiling and both are
/// refused on none of the prompts the gate judges, against at least four for
/// every broken engine here — but 0.0703 is also where the broken adaptive row
/// starts, so the two populations touch on that pair at that block, and the
/// margin the ceiling has over a correct reading there is 1.71x rather than the
/// 1.46x the paragraph above quotes. The narrower row was measured before this
/// pair was split in two and its block is not recorded; the served block
/// reproduces it exactly.
///
/// One broken engine is outside that statement, and that is a property of the
/// defect rather than of the ceiling: leaving the correction on the restricted
/// argmax only changes an answer where that vocabulary falls short of the
/// verifier's, which the runs measure at one to five tokens per answer. It is
/// refused on one of the five prompts the gate judges, at a token the
/// drafter's vocabulary cannot name — and at the round's **correction**, which
/// is the only thing that keeps it refused now that the declared
/// restricted-vocabulary boundary at an accepted position is not.
pub const MAX_DIVERGENCE_CONFIDENCE: f64 = 0.12;

/// How many leading tokens two streams share.
pub fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Where the reference run's confidence at the first divergence sits in its own
/// distribution: the fraction of its decisions it was **less** sure about.
///
/// `margins[i]` is the verifier's top-two logprob gap at position `i` of the
/// reference run. The two runs share every position before the first
/// divergence, so `margins[d]` is the gap the speculative run faced as well.
/// Returns the position and the rank.
///
/// `None` when the runs never differ, or when they first differ past the end of
/// the reference run — the length guard owns that case.
pub fn divergence_confidence(spec: &[u32], plain: &[u32], margins: &[f32]) -> Option<(usize, f64)> {
    let d = common_prefix_len(spec, plain);
    if d >= spec.len() && d >= plain.len() {
        return None;
    }
    let at = *margins.get(d)?;
    let below = margins.iter().filter(|m| **m < at).count();
    Some((d, below as f64 / margins.len() as f64))
}

/// The verifier's top-two logprob gap at one position, from the logprobs of its
/// top candidates in any order.
///
/// A position that reported fewer than two candidates has no gap, and the
/// result is `f32::INFINITY`: a margin that was not reported must not read as a
/// tie.
pub fn top_two_margin(top: impl IntoIterator<Item = f32>) -> f32 {
    let (mut best, mut second): (Option<f32>, Option<f32>) = (None, None);
    for logprob in top {
        if best.is_none_or(|b| logprob.total_cmp(&b).is_gt()) {
            second = best;
            best = Some(logprob);
        } else if second.is_none_or(|s| logprob.total_cmp(&s).is_gt()) {
            second = Some(logprob);
        }
    }
    match (best, second) {
        (Some(best), Some(second)) => best - second,
        _ => f32::INFINITY,
    }
}

/// What the rule below made of a pair of runs.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed verdict of one rule: every caller matches all four outcomes, and a fifth must be handled at each"
)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Divergence {
    /// The runs never differ, or first differ past the end of the reference run.
    Unread,
    /// The shorter run over the longer is under [`MIN_LENGTH_RATIO`].
    ShortRun {
        /// The shorter run's length over the longer's.
        ratio: f64,
        /// Position and rank of the first divergence, when the oracle could
        /// read it.
        at: Option<(usize, f64)>,
    },
    /// The first divergence sits above [`MAX_DIVERGENCE_CONFIDENCE`]: the
    /// verifier was sure of its decision there. Position and rank.
    Confident(usize, f64),
    /// The first divergence sits at or under [`MAX_DIVERGENCE_CONFIDENCE`]: a
    /// decision the verifier was nearly tied on. Position and rank.
    NearTie(usize, f64),
}

/// The verdict rule every caller applies to two runs of one prompt: the length
/// ratio first, then the rank of the first divergence against the ceiling.
///
/// `margins` is the reference run's per-position top-two gap, as for
/// [`divergence_confidence`].
pub fn read_divergence(spec: &[u32], plain: &[u32], margins: &[f32]) -> Divergence {
    let shorter = spec.len().min(plain.len());
    let longer = spec.len().max(plain.len());
    let ratio = shorter as f64 / longer as f64;
    let at = divergence_confidence(spec, plain, margins);
    if ratio < MIN_LENGTH_RATIO {
        return Divergence::ShortRun { ratio, at };
    }
    match at {
        None => Divergence::Unread,
        Some((position, rank)) if rank > MAX_DIVERGENCE_CONFIDENCE => {
            Divergence::Confident(position, rank)
        }
        Some((position, rank)) => Divergence::NearTie(position, rank),
    }
}

#[cfg(test)]
#[path = "answer_equivalence_tests.rs"]
mod tests;
