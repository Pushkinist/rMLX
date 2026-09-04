//! Round-stat derivation: the formulas, and the cell every loop's rows land in.

use super::{RoundStats, SpecLoop};

const ALL_LOOPS: &[SpecLoop] = &[
    SpecLoop::MtpAssistant,
    SpecLoop::MtpSidecar,
    SpecLoop::DFlash,
    SpecLoop::Eagle3,
    SpecLoop::TwoModelGreedy,
    SpecLoop::TwoModelStochastic,
];

/// A request with counters chosen so every derived figure is exact in binary.
fn sample(loop_kind: SpecLoop) -> RoundStats {
    RoundStats {
        loop_kind,
        block_size: 5,
        rounds: 8,
        emitted: 20,
        total_draft: 32,
        total_accept: 12,
        prefill_ns: 500_000_000,
        draft_ns: 200_000_000,
        verifier_ns: 600_000_000,
        round_loop_ns: 1_000_000_000,
        elapsed_ns: 1_600_000_000,
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
    assert_eq!(stats.accept_rate(), 12.0 / 32.0);
    assert_eq!(stats.accepted_per_step(), 12.0 / 8.0);
    assert_eq!(stats.tokens_per_round(), 20.0 / 8.0);
    assert_eq!(stats.draft_ms_per_round(), 200.0 / 8.0);
    assert_eq!(stats.verify_ms_per_round(), 600.0 / 8.0);
    // 1000 ms in the loop, 200 drafting, 600 verifying: 200 ms of loop over 8
    // rounds. Prefill is outside the round loop and does not reach it.
    assert_eq!(stats.loop_ms_per_round(), 200.0 / 8.0);
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
        (summed - 1000.0).abs() < 1e-9,
        "per-round split sums to {summed} ms, round loop was 1000 ms"
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

/// The identity `1 + accept_rate * (block - 1)` recovers `tokens_per_round`
/// only while every round drafted the configured block, which is what the
/// sample counters describe.
#[test]
fn the_fixed_block_identity_holds_when_every_round_drafted_the_block() {
    let stats = sample(SpecLoop::MtpSidecar);
    assert_eq!(
        stats.total_draft,
        stats.rounds * (stats.block_size - 1),
        "the sample must be a run that never resized its block"
    );
    let from_fixed_block = 1.0 + stats.accept_rate() * (stats.block_size as f64 - 1.0);
    assert!((stats.tokens_per_round() - from_fixed_block).abs() < 1e-9);
}

/// A loop that shrank its block drafted fewer tokens than the ceiling allows,
/// and the identity then over-predicts — which is the whole reason
/// `tokens_per_round` is recorded rather than derived at read time.
#[test]
fn the_fixed_block_identity_over_predicts_once_the_block_is_resized() {
    let mut stats = sample(SpecLoop::DFlash);
    // Eight rounds at ceiling 5 could draft 32; this run drafted 24, so it
    // spent some rounds on a smaller block.
    stats.total_draft = 24;
    assert!(stats.total_draft < stats.rounds * (stats.block_size - 1));

    let from_fixed_block = 1.0 + stats.accept_rate() * (stats.block_size as f64 - 1.0);
    assert!(
        from_fixed_block - stats.tokens_per_round() > 0.4,
        "the fixed-block identity gave {from_fixed_block}, the loop measured {}",
        stats.tokens_per_round()
    );
}

#[test]
fn every_loop_names_a_distinct_done_event() {
    let mut seen: Vec<&str> = Vec::new();
    for &loop_kind in ALL_LOOPS {
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
    for &loop_kind in ALL_LOOPS {
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
