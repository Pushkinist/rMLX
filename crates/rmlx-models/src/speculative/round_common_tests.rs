//! The recorder's mapping from a loop's totals to the record that is ingested.
//!
//! This is the one piece of `round_common` a test can reach: it takes no
//! device, no model and no cache, so it runs on the CPU. What it pins is the
//! wiring, and the wiring is exactly what nothing else here can see — a record
//! is a `done` line in a log, so a field written from the wrong place produces
//! a plausible row and no failure anywhere. Two fields make that concrete:
//! `charged` decides how a row's timings are read and is a literal `false` in
//! four loops and a switch in three; and `total_draft` / `total_accept`
//! swapped inverts every ingested `accept_rate` while every token, every count
//! and every gate stays as it was.

use std::time::Instant;

use super::{round_stats, RoundTotals};
use crate::decode_loop::ProbeStep;
use crate::speculative::{DecodeWindow, SpecLoop};

/// Totals whose every counter is a different number.
///
/// Distinct values are the point: two fields carrying the same figure agree
/// when they are cross-wired, so a fixture of zeros and ones would pass the
/// swap this file exists to catch.
fn totals(charged: bool) -> RoundTotals {
    RoundTotals {
        loop_kind: SpecLoop::MtpSidecar,
        block_size: 5,
        conditioned_rows: Some(7),
        charged,
        rounds: 11,
        emitted_in_rounds: 13,
        total_draft: 17,
        total_accept: 19,
        prefill_ns: 23,
        draft_ns: 29,
        verifier_ns: 31,
        round_loop_ns: 37,
        t_total: Instant::now(),
    }
}

/// `n` emitted steps, so the record's `emitted` has a length to read.
fn emitted(n: usize) -> Vec<ProbeStep> {
    (0..n)
        .map(|i| ProbeStep {
            token_id: i as u32,
            piece: String::from("x").into_boxed_str(),
            max_abs_logit: 0.0,
            nan_count: 0,
            logprobs: None,
        })
        .collect()
}

/// The record carries the charge decision the loop handed over, both ways.
///
/// A recorder that writes a constant here re-attributes the phase timings of
/// every loop that charges, with every token and every count identical, and
/// `check-spec-charge` cannot see it: that gate reads the token at the call
/// site, which is all a scan of the caller can do.
///
/// Mutation: replace `charged` in [`round_stats`] with `false`.
#[test]
fn the_recorder_writes_the_charge_decision_it_was_handed() {
    for charged in [false, true] {
        let record = round_stats(&totals(charged), &emitted(3), 2, &DecodeWindow::new());
        assert_eq!(
            record.charged, charged,
            "the record must carry the decision the loop made, not one of its own"
        );
    }
}

/// Every counter reaches the record under its own name.
///
/// The three figures the recorder derives are read here too: `emitted` off the
/// buffer's length, `seed_emitted` off its own argument — the two are different
/// numbers on purpose — and `decode_tps` off a window that has seen nothing,
/// which has no rate to report.
///
/// Mutation: swap `total_draft` and `total_accept` in [`round_stats`], which
/// inverts every ingested `accept_rate` and changes nothing else in the tree.
#[test]
fn every_counter_reaches_the_record_under_its_own_name() {
    let record = round_stats(&totals(true), &emitted(3), 2, &DecodeWindow::new());
    assert_eq!(record.loop_kind, SpecLoop::MtpSidecar);
    assert_eq!(record.block_size, 5);
    assert_eq!(record.conditioned_rows, Some(7));
    assert_eq!(record.rounds, 11);
    assert_eq!(record.emitted_in_rounds, 13);
    assert_eq!(record.total_draft, 17);
    assert_eq!(record.total_accept, 19);
    assert_eq!(record.prefill_ns, 23);
    assert_eq!(record.draft_ns, 29);
    assert_eq!(record.verifier_ns, 31);
    assert_eq!(record.round_loop_ns, 37);
    assert_eq!(record.emitted, 3, "the tokens handed to the sink");
    assert_eq!(record.seed_emitted, 2, "the count the caller passed");
    assert_eq!(
        record.decode_tps, None,
        "a window that has seen no token has no rate to report"
    );
    // The derived accept rate is what a swapped pair moves, and it is what an
    // ingested row is read with.
    assert!((record.accept_rate() - 19.0 / 17.0).abs() < 1e-12);
}
