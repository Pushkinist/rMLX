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
//! is also why the block policy is part of [`RoundStats::decode_config`] rather
//! than context — an adaptive arm at ceiling 16 and a fixed arm at block 16 are
//! different configurations.
//!
//! **A round's tokens are the ones a round produced.** The four sidecar loops
//! argmax a bonus token out of the prefill forward and emit it before the round
//! loop starts, so `emitted` carries one token no verify round produced; the
//! two-model loops emit nothing outside their loop. Dividing `emitted` by
//! `rounds` would therefore read `+1/rounds` high on four of six loops —
//! measured at +1.35% and +0.98% on two of them — and make the one figure the
//! record exists to compare incomparable between them. Each loop counts what
//! its rounds emit at the emit site and reports it as
//! [`RoundStats::emitted_in_rounds`]; that count is the figure's numerator, and
//! `seed_emitted` beside it makes the three counts add up or say why not.
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
    /// Every variant, once — the population the per-loop tests sweep.
    ///
    /// Paired with [`Self::index`], whose match is exhaustive: a seventh
    /// variant does not compile until it has an index, and
    /// `every_variant_is_in_all_once` does not pass until it is in this list at
    /// that index. A hand-written list in a test file is forced by neither, and
    /// a new loop would silently join none of the six per-loop tests.
    ///
    /// It lives beside the enum because that is where the exhaustiveness comes
    /// from, and is compiled with the tests because that is its only caller —
    /// `make ci` builds them, so the guarantee holds where it is checked.
    #[cfg(test)]
    pub(crate) const ALL: &'static [Self] = &[
        Self::MtpAssistant,
        Self::MtpSidecar,
        Self::DFlash,
        Self::Eagle3,
        Self::TwoModelGreedy,
        Self::TwoModelStochastic,
    ];

    /// This variant's position in [`Self::ALL`].
    #[cfg(test)]
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::MtpAssistant => 0,
            Self::MtpSidecar => 1,
            Self::DFlash => 2,
            Self::Eagle3 => 3,
            Self::TwoModelGreedy => 4,
            Self::TwoModelStochastic => 5,
        }
    }

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
    ///
    /// Read from [`rmlx_metrics::cell::ADAPTIVE_DRAFTERS`] rather than declared
    /// here: the same fact decides what a row's cell is, what a row recovered
    /// from notes is classified as, and which historical rows migration 008
    /// rewrites. A second copy in the engine is the drift `decode_config`
    /// exists to prevent.
    pub(crate) fn depth_policy(self) -> Option<&'static str> {
        rmlx_metrics::cell::inherent_depth_policy(self.draft_kind())
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
#[derive(Debug)]
pub(crate) struct RoundStats {
    /// Which loop produced the tokens.
    pub(crate) loop_kind: SpecLoop,
    /// The block size the request was configured with, verifier token included.
    pub(crate) block_size: usize,
    /// Rounds the loop entered.
    pub(crate) rounds: usize,
    /// Tokens handed to the sink.
    pub(crate) emitted: usize,
    /// How many tokens the round loop emitted, counted at its emit site.
    ///
    /// The ground truth for [`Self::tokens_per_round`], and — paired with
    /// `emitted` and `seed_emitted`, which are read at two other points — an
    /// exact cross-check on the loop's own accounting. One integer increment per
    /// emitted token: no evaluation, no allocation, nothing that reaches a
    /// device.
    pub(crate) emitted_in_rounds: usize,
    /// How many of those the loop had already emitted when its round loop
    /// started.
    ///
    /// A measurement, not a constant: the sidecar loops argmax a bonus token
    /// out of the prefill forward and emit it before the first round, the
    /// two-model loops emit nothing until a round has run, and each loop reads
    /// its own `emitted.len()` at that point. A per-loop constant would say the
    /// same thing today and drift the day a loop stops emitting one — silently,
    /// into an append-only store.
    pub(crate) seed_emitted: usize,
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

    /// Tokens the rounds produced — accepted proposals plus the verifier's own
    /// token, per round.
    ///
    /// The figure a speculative result is read with, and the one that stops
    /// being derivable the moment the block stops being fixed. The loop's seed
    /// token is excluded: it comes out of the prefill forward, not out of a
    /// round, and counting it makes the sidecar loops incomparable with the
    /// two-model ones.
    pub(crate) fn tokens_per_round(&self) -> f64 {
        ratio(self.round_emitted() as f64, self.rounds as f64)
    }

    /// Tokens the round loop itself emitted, as the loop counted them.
    pub(crate) const fn round_emitted(&self) -> usize {
        self.emitted_in_rounds
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
    /// fourth measurement. [`Self::span_violation`] names the one way that can
    /// stop being true.
    pub(crate) fn loop_ms_per_round(&self) -> f64 {
        let overhead = ms(self.round_loop_ns) - ms(self.draft_ns) - ms(self.verifier_ns);
        ratio(overhead, self.rounds as f64)
    }

    /// Why the emitted counts do not add up, when they do not.
    ///
    /// The two counts come from different places — one is `emitted.len()` at the
    /// end, the other `emitted.len()` before the first round — so this is a
    /// cross-check on the loop's own accounting rather than a restatement of it.
    ///
    /// Three counts taken at three points in the loop: `seed_emitted` before the
    /// first round, `emitted_in_rounds` at the emit site inside it, `emitted` at
    /// the end. They must add up, exactly, on every request — which is what
    /// makes this a detector rather than a description.
    ///
    /// The drift it exists for is a seed captured one line *too early*, before
    /// the pre-round `emit_step`. That makes the sum disagree by one on every
    /// request that emitted anything, whatever the model, prompt or token
    /// budget. An earlier revision inferred the drift from an emission-budget
    /// inequality instead, which only bites when a request's rounds exactly
    /// saturate `total_accept + rounds`: measured on the four reachable loops at
    /// a fixed `--max-tokens`, that held for one of them, and on the other three
    /// a 0.5%-wrong `tokens_per_round` reached the store in silence.
    pub(crate) fn seed_violation(&self) -> Option<String> {
        let accounted = self.seed_emitted.saturating_add(self.emitted_in_rounds);
        if accounted != self.emitted {
            return Some(format!(
                "emitted {} but accounts for {accounted} — {} before the rounds and {} \
                 inside them. The three counts are taken at three points in the loop and \
                 a seed read before the pre-round emission is one of the ways they stop \
                 adding up",
                self.emitted, self.seed_emitted, self.emitted_in_rounds
            ));
        }
        // A second, independent invariant on the same counters: a round emits
        // the tokens it accepted plus the verifier's own and no more. It is not
        // the seed detector — the equality above is — but it is free here and it
        // is the one an acceptance-walk change would break.
        let budget = self.total_accept.saturating_add(self.rounds);
        (self.emitted_in_rounds > budget).then(|| {
            format!(
                "{} tokens credited to {} rounds that could have produced {budget} \
                 (accepted plus one verifier token each)",
                self.emitted_in_rounds, self.rounds
            )
        })
    }

    /// Why the three spans do not partition the round loop, when they do not.
    ///
    /// Drafting and verifying are sub-spans of the round loop, so their sum
    /// cannot exceed it. A loop that starts one of those timers before
    /// `round_loop_t0` makes the residual negative, and a negative duration is
    /// refused by the ingest bounds — taking the whole run's record with it,
    /// naming no field. Reported here so the field is named while the run is
    /// still in front of somebody.
    ///
    /// Reported and nothing else. This carried a `debug_assert!(false)` for one
    /// commit, which turned a report into a crate-wide abort under exactly the
    /// profile `make test` and `make gpu-test` run at, and compiled to nothing
    /// under `release-perf` where a real timing anomaly would be worth seeing.
    /// The property it guarded is pinned by
    /// `a_phase_span_reaching_outside_the_round_loop_is_named` instead.
    pub(crate) fn span_violation(&self) -> Option<String> {
        let inner = self.draft_ns.saturating_add(self.verifier_ns);
        (inner > self.round_loop_ns).then(|| {
            format!(
                "draft_ms {:.3} + verifier_ms {:.3} exceeds round_ms {:.3}: a timer starts \
                 outside the round loop it is attributed to, and loop_ms_per_round is negative",
                ms(self.draft_ns),
                ms(self.verifier_ns),
                ms(self.round_loop_ns)
            )
        })
    }

    /// The cell this request's rows belong to.
    pub(crate) fn decode_config(&self) -> String {
        rmlx_metrics::cell::decode_config(
            self.loop_kind.draft_kind(),
            self.block_size,
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
    ///
    /// The event's target is this module for every loop, not the loop's own —
    /// six callsites became one. `loop_kind` is a field so the loop is still
    /// selectable, by field rather than by module path.
    pub(crate) fn log_done(&self) {
        for reason in [self.span_violation(), self.seed_violation()]
            .into_iter()
            .flatten()
        {
            tracing::error!(
                loop_kind = ?self.loop_kind,
                "speculative round accounting is inconsistent: {reason}"
            );
        }
        tracing::info!(
            loop_kind = ?self.loop_kind,
            rounds = self.rounds,
            emitted = self.emitted,
            seed_emitted = self.seed_emitted,
            emitted_in_rounds = self.emitted_in_rounds,
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
