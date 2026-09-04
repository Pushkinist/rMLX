//! What a speculative round loop counted, and the one `done` line that reports it.
//!
//! Every round loop in this module family — the two MTP paths, DFlash, EAGLE-3
//! and the two-model loop — keeps the same counters and closes a request with
//! the same record. Holding the counters, the derivation and the log site in
//! one place is what makes a row from one drafter comparable with a row from
//! another.
//!
//! The figure that carries the comparison is `tokens_per_round`. It equals
//! `1 + accept_rate * (block - 1)` only while the block is the configured one
//! every round, and DFlash's already is not: it halves and grows its block from
//! the recent accept rate, so its rows have never been derivable that way. That
//! is also why the block policy is part of [`Self::decode_config`] rather than
//! context — an adaptive arm at ceiling 16 and a fixed arm at block 16 are
//! different configurations.
//!
//! **The derivation does not change what it measures.** Every figure below is
//! arithmetic over counters the loops already keep; nothing here forces an
//! evaluation, allocates inside a round, or adds a clock read to one. The two
//! spans that are new — prefill and the round loop as a whole — are one
//! `Instant::now()` each per *request*.
//!
//! `draft_ms` and `verifier_ms` are the wall-clock spans of the two call sites,
//! not the cost of the work those calls issue: this engine evaluates lazily, so
//! work issued inside one span can be paid for in another. They are reported as
//! what they are. Inserting a blocking evaluation to make them attributable
//! would price the phases by changing them, and that blocking evaluation is
//! itself one of the costs the round loop is trying to shed.

use crate::speculative::DraftKind;

/// `decode_config` drafter name for the classic two-model round loop, which is
/// not one of the sidecar [`DraftKind`]s.
const TWO_MODEL_KIND: &str = "two_model";

/// `decode_config` depth policy for a loop that resizes its block from the
/// accept rate of the recent rounds.
const DEPTH_POLICY_ACCEPT_RATE: &str = "accept_rate";

/// Which round loop produced a request's tokens.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SpecLoop {
    /// Gemma4 shared-K/V assistant drafter.
    MtpAssistant,
    /// Qwen3.5-family MTP sidecar.
    MtpSidecar,
    /// Draft-Flash block drafter.
    DFlash,
    /// EAGLE-3 drafter.
    Eagle3,
    /// Two full models, greedy acceptance.
    TwoModelGreedy,
    /// Two full models, Leviathan stochastic acceptance.
    TwoModelStochastic,
}

impl SpecLoop {
    /// The event this loop closes every request with. One per request, whatever
    /// stopped it — a reader counting records against requests served needs the
    /// EOS case to leave one too.
    pub(crate) const fn done_event(self) -> &'static str {
        match self {
            Self::MtpAssistant => "mtp_assistant_generate_greedy: done",
            Self::MtpSidecar => "mtp_generate_greedy: done",
            Self::DFlash => "dflash_generate_greedy: done",
            Self::Eagle3 => "eagle3_generate_greedy: done",
            Self::TwoModelGreedy => "spec_generate_greedy_cached: done",
            Self::TwoModelStochastic => "spec_generate_stochastic_cached: done",
        }
    }

    /// The drafter name this loop's rows carry in `decode_config`.
    pub(crate) fn draft_kind(self) -> &'static str {
        match self {
            // Both MTP paths record as one drafter kind: the CLI selects them
            // with the same `--draft-kind mtp` and routes on the verifier's
            // architecture family.
            Self::MtpAssistant | Self::MtpSidecar => DraftKind::Mtp.as_str(),
            Self::DFlash => DraftKind::DFlash.as_str(),
            Self::Eagle3 => DraftKind::Eagle3.as_str(),
            Self::TwoModelGreedy | Self::TwoModelStochastic => TWO_MODEL_KIND,
        }
    }

    /// How this loop picks each round's block size, or `None` when it drafts
    /// the configured block every round.
    pub(crate) const fn depth_policy(self) -> Option<&'static str> {
        match self {
            Self::DFlash => Some(DEPTH_POLICY_ACCEPT_RATE),
            Self::MtpAssistant
            | Self::MtpSidecar
            | Self::Eagle3
            | Self::TwoModelGreedy
            | Self::TwoModelStochastic => None,
        }
    }
}

/// Nanoseconds to milliseconds.
fn ms(ns: u128) -> f64 {
    (ns as f64) / 1.0e6
}

/// One speculative request's counters, and the per-round figures they imply.
///
/// Built once, after the round loop, by the loop that ran it. The loops keep
/// the counters as locals while they run and hand them over here.
#[derive(Debug, Clone)]
pub(crate) struct RoundStats {
    /// Which loop produced the tokens.
    pub(crate) loop_kind: SpecLoop,
    /// The block size the request was configured with, verifier token included.
    pub(crate) block_size: usize,
    /// Rounds the loop entered.
    pub(crate) rounds: usize,
    /// Tokens handed to the sink.
    pub(crate) emitted: usize,
    /// Tokens the drafter proposed, over all rounds.
    pub(crate) total_draft: usize,
    /// Proposed tokens the verifier accepted, over all rounds.
    pub(crate) total_accept: usize,
    /// Prompt prefill span.
    pub(crate) prefill_ns: u128,
    /// Wall-clock spent inside the drafter call, over all rounds.
    pub(crate) draft_ns: u128,
    /// Wall-clock spent inside the verify forward, over all rounds.
    pub(crate) verifier_ns: u128,
    /// Wall-clock of the whole round loop, drafting and verifying included.
    pub(crate) round_loop_ns: u128,
    /// Wall-clock of the whole request, prefill included.
    pub(crate) elapsed_ns: u128,
    /// The decode rate over the first-token-to-last window, or `None` when the
    /// request emitted too few tokens to have an interval.
    pub(crate) decode_tps: Option<f64>,
}

impl RoundStats {
    /// Accepted proposals over proposals. Zero when nothing was drafted.
    pub(crate) fn accept_rate(&self) -> f64 {
        ratio(self.total_accept as f64, self.total_draft as f64)
    }

    /// Accepted proposals per verifier step.
    pub(crate) fn accepted_per_step(&self) -> f64 {
        ratio(self.total_accept as f64, self.rounds as f64)
    }

    /// Tokens emitted per verify round — accepted proposals plus the verifier's
    /// own token. The figure a speculative result is read with, and the one
    /// that stops being derivable the moment the block stops being fixed.
    pub(crate) fn tokens_per_round(&self) -> f64 {
        ratio(self.emitted as f64, self.rounds as f64)
    }

    /// Drafting wall-clock per round.
    pub(crate) fn draft_ms_per_round(&self) -> f64 {
        ratio(ms(self.draft_ns), self.rounds as f64)
    }

    /// Verify wall-clock per round.
    pub(crate) fn verify_ms_per_round(&self) -> f64 {
        ratio(ms(self.verifier_ns), self.rounds as f64)
    }

    /// Round-loop wall-clock per round that is neither drafting nor verifying:
    /// rollback, snapshot and restore, cache truncation, acceptance walks and
    /// sampling.
    ///
    /// The three spans partition the round loop by construction — drafting and
    /// verifying are disjoint sub-spans of it — so this is a residual, not a
    /// fourth measurement, and it cannot be negative unless a loop times a span
    /// that starts outside its own round loop.
    pub(crate) fn loop_ms_per_round(&self) -> f64 {
        let overhead = ms(self.round_loop_ns) - ms(self.draft_ns) - ms(self.verifier_ns);
        ratio(overhead, self.rounds as f64)
    }

    /// The cell this request's rows belong to.
    pub(crate) fn decode_config(&self) -> String {
        rmlx_metrics::cell::decode_config(
            self.loop_kind.draft_kind(),
            u32::try_from(self.block_size).unwrap_or(u32::MAX),
            self.loop_kind.depth_policy(),
        )
    }

    /// Close the request with the one record every reader of a speculative run
    /// takes its numbers from.
    ///
    /// `scripts/lib/spec_round_log.py` is the only thing that parses it, and
    /// `scripts/spec_bench.sh` records what it finds here — including
    /// `decode_config`, so a new drafter or a new depth policy reaches the
    /// metrics store without a bench script learning about it.
    pub(crate) fn log_done(&self) {
        tracing::info!(
            rounds = self.rounds,
            emitted = self.emitted,
            total_draft = self.total_draft,
            total_accept = self.total_accept,
            accept_rate = self.accept_rate(),
            accepted_per_step = self.accepted_per_step(),
            tokens_per_round = self.tokens_per_round(),
            decode_tps = ?self.decode_tps,
            elapsed_ms = ms(self.elapsed_ns),
            prefill_ms = ms(self.prefill_ns),
            round_ms = ms(self.round_loop_ns),
            draft_ms = ms(self.draft_ns),
            verifier_ms = ms(self.verifier_ns),
            draft_ms_per_round = self.draft_ms_per_round(),
            verify_ms_per_round = self.verify_ms_per_round(),
            loop_ms_per_round = self.loop_ms_per_round(),
            block_size = self.block_size,
            decode_config = %self.decode_config(),
            "{}",
            self.loop_kind.done_event()
        );
    }
}

/// `numerator / denominator`, or zero when there is no denominator.
///
/// Zero rather than NaN: a request that ran no round has no per-round figure,
/// and a NaN in that slot would reach the metrics store, where the plausibility
/// bounds reject it and the whole record is refused.
fn ratio(numerator: f64, denominator: f64) -> f64 {
    if denominator > 0.0 {
        numerator / denominator
    } else {
        0.0
    }
}

#[cfg(test)]
#[path = "round_stats_tests.rs"]
mod round_stats_tests;
