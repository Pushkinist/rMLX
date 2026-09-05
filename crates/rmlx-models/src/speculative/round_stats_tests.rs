//! Round-stat derivation: the formulas, the emission accounting each loop has,
//! and the cell every loop's rows land in.

use super::{RoundStats, SpecLoop};

/// What each loop emits before its round loop starts, as the loops themselves
/// measure it: the sidecar paths argmax a bonus token out of the prefill
/// forward, the two-model paths emit nothing until a round has run.
///
/// This is the fixture's model of the loops, not the engine's — the engine
/// measures it. It exists so the two accountings are both exercised here.
fn seed_of(loop_kind: SpecLoop) -> usize {
    match loop_kind {
        SpecLoop::MtpAssistant | SpecLoop::MtpSidecar | SpecLoop::DFlash | SpecLoop::Eagle3 => 1,
        SpecLoop::TwoModelGreedy | SpecLoop::TwoModelStochastic => 0,
    }
}

/// A request as `loop_kind` would really have accounted it: 8 rounds at block
/// 5, every round drafting 4 and the verifier accepting 12 of the 32, plus the
/// seed token the loop emits outside its rounds.
///
/// The three timings are pairwise distinct multiples of the round count, so
/// swapping any two of the three derivations fails rather than cancelling:
/// 200 / 400 / 1200 ms give 25 / 50 / 75 ms per round.
fn sample(loop_kind: SpecLoop) -> RoundStats {
    let rounds = 8usize;
    let total_accept = 12usize;
    let seed_emitted = seed_of(loop_kind);
    // One token under the emission budget, deliberately: an earlier revision
    // inferred the seed drift from that inequality, and its fixtures were built
    // to sit exactly at the budget — the only place it could fire. On real
    // requests it fired on one of the four reachable loops.
    let emitted_in_rounds = total_accept + rounds - 1;
    RoundStats {
        loop_kind,
        block_size: 5,
        rounds,
        emitted: seed_emitted + emitted_in_rounds,
        emitted_in_rounds,
        seed_emitted,
        total_draft: 32,
        total_accept,
        prefill_ns: 500_000_000,
        draft_ns: 200_000_000,
        verifier_ns: 400_000_000,
        round_loop_ns: 1_200_000_000,
        elapsed_ns: 1_800_000_000,
        charged: false,
        decode_tps: Some(20.0),
    }
}

#[test]
#[allow(
    clippy::float_cmp,
    reason = "the derivation is the same division as the expectation, so the two \
              are bit-identical and an epsilon band would hide a changed formula"
)]
fn derived_figures_are_the_documented_formulas() {
    let stats = sample(SpecLoop::MtpSidecar);
    assert_eq!(stats.emitted, 20, "1 seed + the 19 the rounds emitted");
    assert_eq!(stats.accept_rate(), 12.0 / 32.0);
    assert_eq!(stats.accepted_per_step(), 12.0 / 8.0);
    // The seed token is not a round's product and does not reach the figure.
    assert_eq!(stats.tokens_per_round(), 19.0 / 8.0);
    assert_eq!(stats.draft_ms_per_round(), 200.0 / 8.0);
    assert_eq!(stats.verify_ms_per_round(), 400.0 / 8.0);
    // 1200 ms in the loop, 200 drafting, 400 verifying: 600 ms of loop over 8
    // rounds. Prefill is outside the round loop and does not reach it.
    assert_eq!(stats.loop_ms_per_round(), 600.0 / 8.0);
}

/// Each of the three per-round timings is a different number, so a swap of any
/// two derivations changes an assertion rather than cancelling out.
#[test]
fn the_three_timings_are_pairwise_distinct() {
    let stats = sample(SpecLoop::DFlash);
    let figures = [
        stats.draft_ms_per_round(),
        stats.verify_ms_per_round(),
        stats.loop_ms_per_round(),
    ];
    for (i, a) in figures.iter().enumerate() {
        for b in figures.iter().skip(i + 1) {
            assert!(
                (a - b).abs() > 1.0,
                "two per-round timings read {a} and {b}; a fixture that cannot tell \
                 them apart cannot tell a swap of their derivations apart either"
            );
        }
    }
}

/// The three spans partition the round loop, so the per-round figures sum back
/// to it. A `loop_ms_per_round` derived from `elapsed` instead would carry the
/// prefill and this would not hold.
#[test]
fn the_per_round_split_sums_back_to_the_round_loop() {
    let stats = sample(SpecLoop::DFlash);
    let summed =
        (stats.draft_ms_per_round() + stats.verify_ms_per_round() + stats.loop_ms_per_round())
            * stats.rounds as f64;
    assert!(
        (summed - 1200.0).abs() < 1e-9,
        "per-round split sums to {summed} ms, round loop was 1200 ms"
    );
}

/// A request that emitted its stop token before entering a round still closes,
/// and every per-round figure is zero rather than NaN — a NaN would be refused
/// at ingest and take the whole record with it.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "the assertion is that the value is exactly zero rather than NaN or \
              near-zero; a band would accept both of the values it rules out"
)]
fn a_request_with_no_round_derives_zeros_not_nan() {
    let mut stats = sample(SpecLoop::Eagle3);
    stats.rounds = 0;
    stats.emitted = 1;
    stats.total_draft = 0;
    stats.total_accept = 0;
    for value in [
        stats.accept_rate(),
        stats.accepted_per_step(),
        stats.tokens_per_round(),
        stats.draft_ms_per_round(),
        stats.verify_ms_per_round(),
        stats.loop_ms_per_round(),
    ] {
        assert_eq!(value, 0.0, "no round means no per-round figure");
    }
}

// ── The emission accounting, per loop ────────────────────────────────────────

/// The identity `1 + accept_rate * (block - 1)` recovers `tokens_per_round`
/// while every round drafted the configured block — on **every** loop, which is
/// the whole point of subtracting the seed token. Counting the seed makes it
/// false on the four sidecar loops and true on the two-model ones, so a fixture
/// that models only one accounting cannot see the difference.
#[test]
fn the_fixed_block_identity_holds_on_every_loop() {
    for &loop_kind in SpecLoop::ALL {
        // The identity describes a run whose every round emitted `accept + 1`;
        // the sample is one token under that on purpose, so this case restores
        // it rather than the sample being built to satisfy it.
        let mut stats = sample(loop_kind);
        stats.emitted_in_rounds = stats.total_accept + stats.rounds;
        stats.emitted = stats.seed_emitted + stats.emitted_in_rounds;
        assert_eq!(
            stats.total_draft,
            stats.rounds * (stats.block_size - 1),
            "{loop_kind:?}: the sample must be a run that never resized its block"
        );
        let from_fixed_block = 1.0 + stats.accept_rate() * (stats.block_size as f64 - 1.0);
        assert!(
            (stats.tokens_per_round() - from_fixed_block).abs() < 1e-9,
            "{loop_kind:?}: the identity gives {from_fixed_block}, the loop measured {}",
            stats.tokens_per_round()
        );
    }
}

/// Two loops that ran the same rounds report the same figure whatever they
/// emitted before them — which is the whole reason the seed is subtracted.
#[test]
fn what_a_loop_emitted_before_its_rounds_does_not_reach_the_figure() {
    for &loop_kind in SpecLoop::ALL {
        let stats = sample(loop_kind);
        assert_eq!(stats.emitted - stats.round_emitted(), stats.seed_emitted);
        assert_eq!(stats.round_emitted(), 19, "{loop_kind:?}");
    }
}

/// The three counts must add up on every request, and this is the drift they
/// exist to catch: a seed captured one line before the pre-round emission.
///
/// The point of counting at the emit site is that this fires **whatever the
/// request looked like**. The predecessor inferred the drift from an
/// emission-budget inequality, which only bites when a request's rounds exactly
/// saturate `total_accept + rounds`; the sample here deliberately sits one token
/// under that, where the old check was blind and three of the four reachable
/// loops actually live.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion is that the drift is named; unwrapping the None case \
              here is the failure the test exists to report"
)]
fn a_seed_taken_before_the_pre_round_emission_is_named_on_any_request() {
    for &loop_kind in SpecLoop::ALL {
        let sound = sample(loop_kind);
        assert!(
            sound.seed_violation().is_none(),
            "{loop_kind:?} must be sound"
        );
        assert!(
            sound.emitted_in_rounds < sound.total_accept + sound.rounds,
            "{loop_kind:?}: the sample must sit under the emission budget, or this \
             would not distinguish the equality from the inequality it replaced"
        );

        // A loop that emits nothing outside its rounds has no earlier line to
        // take the seed from, which is why the two-model paths are exempt rather
        // than untested.
        if sound.seed_emitted == 0 {
            continue;
        }
        let mut drifted = sample(loop_kind);
        drifted.seed_emitted -= 1;
        let reason = drifted.seed_violation().expect("must be named");
        assert!(reason.contains("accounts for"), "{loop_kind:?}: {reason}");
    }
}

/// Any disagreement between the three counts is named, not only the seed drift:
/// a round loop that emitted more than it counted, or counted more than it
/// emitted, is the same inconsistency from the other side.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion is that the inconsistency is named; unwrapping the \
              None case here is the failure the test exists to report"
)]
fn any_disagreement_between_the_three_counts_is_named() {
    let mut miscounted = sample(SpecLoop::DFlash);
    miscounted.emitted_in_rounds += 1;
    let reason = miscounted.seed_violation().expect("must be named");
    assert!(reason.contains("accounts for"), "{reason}");

    let mut over_budget = sample(SpecLoop::MtpSidecar);
    over_budget.emitted_in_rounds = over_budget.total_accept + over_budget.rounds + 1;
    over_budget.emitted = over_budget.seed_emitted + over_budget.emitted_in_rounds;
    let reason = over_budget.seed_violation().expect("must be named");
    assert!(reason.contains("could have produced"), "{reason}");
}

/// `ALL` and `index` are two halves of one list and the compiler holds both: a
/// seventh variant does not compile until it has an index, and does not pass
/// here until it is in `ALL` at that index.
#[test]
fn every_variant_is_in_all_once() {
    for (position, &loop_kind) in SpecLoop::ALL.iter().enumerate() {
        assert_eq!(loop_kind.index(), position, "{loop_kind:?}");
    }
    let mut indices: Vec<usize> = SpecLoop::ALL.iter().map(|k| k.index()).collect();
    indices.sort_unstable();
    indices.dedup();
    assert_eq!(
        indices.len(),
        SpecLoop::ALL.len(),
        "two variants share an index"
    );
}

/// Counting the seed token would read `+1/rounds` high, and the two loop
/// families would stop being comparable. Stated as a number so a change back
/// fails here.
#[test]
fn counting_the_seed_token_would_bias_the_sidecar_loops() {
    let sidecar = sample(SpecLoop::MtpSidecar);
    let two_model = sample(SpecLoop::TwoModelGreedy);
    assert!(
        (sidecar.tokens_per_round() - two_model.tokens_per_round()).abs() < 1e-9,
        "two loops that produced the same rounds must report the same figure"
    );
    let naive = sidecar.emitted as f64 / sidecar.rounds as f64;
    assert!(
        (naive - sidecar.tokens_per_round() - 1.0 / sidecar.rounds as f64).abs() < 1e-9,
        "the bias is exactly one token per round"
    );
}

/// A loop that starts a phase timer before its round loop makes the residual
/// negative. The value is left as it is — a clamp would hide it — and the
/// violation is named so the field is identifiable before ingest refuses the
/// whole record over a negative duration.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion is that the violation is named; unwrapping the None \
              case here is the failure the test exists to report"
)]
fn a_phase_span_reaching_outside_the_round_loop_is_named() {
    let mut stats = sample(SpecLoop::MtpSidecar);
    assert!(stats.span_violation().is_none(), "the sample must be sound");

    stats.draft_ns = 1_000_000_000;
    let reason = stats.span_violation().expect("must be named");
    assert!(reason.contains("outside the round loop"), "{reason}");
    assert!(
        stats.loop_ms_per_round() < 0.0,
        "the residual must stay visible rather than be clamped"
    );
}

// ── Identity of the loops ────────────────────────────────────────────────────

#[test]
fn every_loop_names_a_distinct_done_event() {
    let mut seen: Vec<&str> = Vec::new();
    for &loop_kind in SpecLoop::ALL {
        let event = loop_kind.done_event();
        assert!(!seen.contains(&event), "{event} is claimed by two loops");
        assert!(
            event.ends_with(": done"),
            "{event} must end the request, and the log reader keys on that"
        );
        seen.push(event);
    }
}

/// Every loop composes a `decode_config` the ingest gate accepts, and the
/// adaptive loop's is a different cell from the fixed loops' at the same block.
#[test]
fn every_loop_composes_a_well_formed_cell() {
    let mut adaptive: Vec<String> = Vec::new();
    let mut fixed: Vec<String> = Vec::new();
    for &loop_kind in SpecLoop::ALL {
        let config = sample(loop_kind).decode_config();
        assert!(
            rmlx_metrics::cell::decode_config_is_well_formed(&config),
            "{loop_kind:?} composed {config}"
        );
        assert!(
            !rmlx_metrics::cell::decode_config_is_all_defaults(&config),
            "{loop_kind:?} composed {config}, which says the engine was at its \
             defaults — a drafter never is"
        );
        if loop_kind.depth_policy().is_some() {
            adaptive.push(config);
        } else {
            fixed.push(config);
        }
    }
    assert!(!adaptive.is_empty(), "no loop declares an adaptive block");
    for a in &adaptive {
        assert!(
            !fixed.contains(a),
            "{a} shares a cell with a fixed-block arm"
        );
    }
}

/// Every loop's depth policy, written down once per variant.
///
/// `every_loop_composes_a_well_formed_cell` only asserts the adaptive set is
/// non-empty, which one existing loop satisfies forever. This table has an entry
/// per variant and is checked against `SpecLoop::ALL`, so a seventh loop fails
/// here until someone records what its block policy is — and is checked against
/// `ADAPTIVE_DRAFTERS`, so the engine's match and the shared list cannot
/// disagree about it.
#[test]
fn every_loop_is_classified_against_the_shared_list() {
    let expected: &[(SpecLoop, Option<&str>)] = &[
        (SpecLoop::MtpAssistant, None),
        (SpecLoop::MtpSidecar, None),
        (SpecLoop::DFlash, Some("accept_rate")),
        (SpecLoop::Eagle3, None),
        (SpecLoop::TwoModelGreedy, None),
        (SpecLoop::TwoModelStochastic, None),
    ];
    assert_eq!(
        expected.len(),
        SpecLoop::ALL.len(),
        "a loop was added without recording whether its block is the configured one"
    );
    for &(loop_kind, want) in expected {
        assert!(SpecLoop::ALL.contains(&loop_kind), "{loop_kind:?}");
        assert_eq!(loop_kind.depth_policy(), want, "{loop_kind:?}");
        assert_eq!(
            loop_kind.depth_policy(),
            rmlx_metrics::cell::inherent_depth_policy(loop_kind.draft_kind()),
            "{loop_kind:?}: the loop's match and ADAPTIVE_DRAFTERS disagree"
        );
    }
}

/// Both MTP paths record as one drafter, and the two-model loops as one that is
/// neither.
#[test]
fn loops_that_are_one_drafter_share_a_kind() {
    assert_eq!(
        SpecLoop::MtpAssistant.draft_kind(),
        SpecLoop::MtpSidecar.draft_kind()
    );
    assert_eq!(
        SpecLoop::TwoModelGreedy.draft_kind(),
        SpecLoop::TwoModelStochastic.draft_kind()
    );
    assert_ne!(
        SpecLoop::TwoModelGreedy.draft_kind(),
        SpecLoop::MtpSidecar.draft_kind()
    );
}

// ---------------------------------------------------------------------------
// The charge switch
// ---------------------------------------------------------------------------

/// A subscriber that answers `enabled` from a fixed verdict and keeps every
/// question it was asked.
///
/// Not a filter. It exists to record *what* [`phases_charged`] asks about,
/// which is the part a mutation moves: a switch retargeted at another string,
/// or lowered from `TRACE` to `DEBUG` — which would charge every `--log debug`
/// run — changes the recorded question rather than the recorded answer, and a
/// test that only checked the answer would stay green through both.
struct AskRecorder {
    verdict: bool,
    asked: std::sync::Mutex<Vec<(String, tracing::Level)>>,
}

impl AskRecorder {
    fn new(verdict: bool) -> std::sync::Arc<Self> {
        std::sync::Arc::new(Self {
            verdict,
            asked: std::sync::Mutex::new(Vec::new()),
        })
    }

    fn questions(&self) -> Vec<(String, tracing::Level)> {
        // A poisoned lock still holds the questions; this fixture has no
        // invariant a panicking writer could have broken.
        self.asked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl tracing::Subscriber for AskRecorder {
    fn enabled(&self, meta: &tracing::Metadata<'_>) -> bool {
        self.asked
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((meta.target().to_owned(), *meta.level()));
        self.verdict
    }
    fn new_span(&self, _: &tracing::span::Attributes<'_>) -> tracing::span::Id {
        tracing::span::Id::from_u64(1)
    }
    fn record(&self, _: &tracing::span::Id, _: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _: &tracing::span::Id, _: &tracing::span::Id) {}
    fn event(&self, _: &tracing::Event<'_>) {}
    fn enter(&self, _: &tracing::span::Id) {}
    fn exit(&self, _: &tracing::span::Id) {}
}

#[test]
fn the_charge_switch_asks_about_the_phase_target_at_trace() {
    let rec = AskRecorder::new(true);
    let charged =
        tracing::subscriber::with_default(std::sync::Arc::clone(&rec), super::phases_charged);
    assert!(
        charged,
        "a subscriber that enables everything must leave the phases charged"
    );
    let asked = rec.questions();
    assert!(
        !asked.is_empty(),
        "the switch consulted no subscriber, so this test asserts nothing about it"
    );
    let want = (super::PHASE_TARGET.to_owned(), tracing::Level::TRACE);
    // Every question, not their number: the macro is free to ask more than
    // once, and pinning the count would pin its implementation rather than the
    // switch's.
    assert!(
        asked.iter().all(|q| *q == want),
        "the switch must ask about {} at TRACE and nothing else — at DEBUG it would \
         charge every `--log debug` run, and on another target it would answer to a \
         filter no documentation names. Asked: {asked:?}",
        super::PHASE_TARGET
    );
}

#[test]
fn a_subscriber_that_declines_the_phase_target_leaves_the_phases_uncharged() {
    let rec = AskRecorder::new(false);
    let charged = tracing::subscriber::with_default(rec, super::phases_charged);
    assert!(
        !charged,
        "the default schedule is the uncharged one: a filter that does not enable \
         the phase target must not make the engine drain its pipeline per round"
    );
}
