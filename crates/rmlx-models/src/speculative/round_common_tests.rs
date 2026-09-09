//! What `round_common` can be read on the CPU: the recorder's mapping from a
//! loop's totals to the record that is ingested, and the rollback's own
//! contract on the offsets it is handed.
//!
//! The mapping takes no device, no model and no cache. What it pins is the
//! wiring, and the wiring is exactly what nothing else here can see — a record
//! is a `done` line in a log, so a field written from the wrong place produces
//! a plausible row and no failure anywhere. Two fields make that concrete:
//! `charged` decides how a row's timings are read and is a literal `false` in
//! four loops and a switch in three; and `total_draft` / `total_accept`
//! swapped inverts every ingested `accept_rate` while every token, every count
//! and every gate stays as it was.
//!
//! Three of the record's fields are derived here rather than carried on the
//! totals — `emitted` off the buffer's length, `elapsed_ns` off the request's
//! start, `decode_tps` off the window — and `seed_emitted` is a fourth that is
//! neither: it is passed straight through, because its source differs between
//! the recorder's two callers.

use std::time::{Duration, Instant};

use rmlx_kv_quant::LinearAttnCache;
use rmlx_mlx::Device;

use super::{rollback_round, rollback_round_caches, round_stats, RoundTotals};
use crate::decode_loop::ProbeStep;
use crate::speculative::{DecodeWindow, SpecLoop};

/// How far back the fixture's request started.
///
/// A second is far outside every counter the fixture carries, so a wall-clock
/// read replaced by any of them lands below this floor rather than inside a
/// plausible range. Backdating rather than sleeping keeps the bound exact: a
/// sleep-calibrated one is nondeterministic under a loaded
/// `cargo test --workspace`, which is the same reason `DecodeWindow`'s own
/// tests inject their instants.
const BACKDATE: Duration = Duration::from_secs(1);

/// Totals whose every counter is a different number.
///
/// Distinct values are the point: two fields carrying the same figure agree
/// when they are cross-wired, so a fixture of zeros and ones would pass the
/// swap this file exists to catch.
#[allow(
    clippy::expect_used,
    reason = "test-only: a monotonic clock that cannot be read one second back would make the wall-clock bound below vacuous, so failing here is the honest outcome"
)]
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
        t_total: Instant::now()
            .checked_sub(BACKDATE)
            .expect("a monotonic clock readable one second back"),
    }
}

/// A window that has seen two tokens 40 ms apart, which is 25 tok/s exactly.
///
/// Instants are injected, never slept for — the idiom
/// `decode_window_excludes_the_time_before_the_first_token` uses for the same
/// reason.
fn window_at_25_tps() -> DecodeWindow {
    let t0 = Instant::now();
    let mut w = DecodeWindow::new();
    w.mark_at(t0);
    w.mark_at(t0 + Duration::from_millis(40));
    w
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
/// every loop that charges, with every token and every count identical.
///
/// Three checks cover that between them, and each is blind to what the others
/// see. `check-spec-charge`'s RULE 5 refuses the *literal* form — any
/// `charged:` field written outside a round loop, whether or not its value is
/// legible. This test catches a value rebound before the literal, where the
/// field is still the shorthand and RULE 5 reads nothing wrong. And
/// `clippy::redundant_field_names` under `-D warnings` is what makes the
/// shorthand the only legal spelling of the correct form, so there is no third
/// way to write it that neither of the first two is looking at.
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

/// Every counter reaches the record under its own name, and so does each of
/// the three figures the recorder derives.
///
/// `emitted` comes off the buffer's length and `seed_emitted` off its own
/// argument — different numbers on purpose, since one is a pass-through and the
/// other is not. `elapsed_ns` is bracketed by two reads of the same clock
/// taken either side of the call, so it is pinned to the read that happened
/// rather than to a duration, and held above [`BACKDATE`] so that no counter in
/// the fixture can stand in for it. `decode_tps` is read from a window driven
/// to a known rate; the window that has seen nothing is the second case below,
/// and it must arrive as `None` rather than as a zero.
///
/// Mutation: swap `total_draft` and `total_accept` in [`round_stats`], which
/// inverts every ingested `accept_rate` and changes nothing else in the tree;
/// or write `elapsed_ns: prefill_ns`, which lands a wall-clock of 23 ns in an
/// append-only store.
#[test]
fn every_counter_reaches_the_record_under_its_own_name() {
    let totals = totals(true);
    let steps = emitted(3);
    let window = window_at_25_tps();
    let before = totals.t_total.elapsed().as_nanos();
    let record = round_stats(&totals, &steps, 2, &window);
    let after = totals.t_total.elapsed().as_nanos();
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
    assert!(
        (before..=after).contains(&record.elapsed_ns),
        "the request's wall-clock must be the clock read while the record was \
         built: {} is outside the {before}..={after} that read happened inside",
        record.elapsed_ns
    );
    assert!(
        record.elapsed_ns >= BACKDATE.as_nanos(),
        "the request started {BACKDATE:?} ago, so a wall-clock of {} is one of \
         the fixture's counters standing in for it",
        record.elapsed_ns
    );
    assert!(
        record
            .decode_tps
            .is_some_and(|tps| (tps - 25.0).abs() < 1e-9),
        "two tokens 40 ms apart is 25 tok/s, and the record read {:?}",
        record.decode_tps
    );
    // The derived accept rate is what a swapped pair moves, and it is what an
    // ingested row is read with.
    assert!((record.accept_rate() - 19.0 / 17.0).abs() < 1e-12);

    // A window that has seen no token has no rate, and the recorder passes that
    // through rather than turning it into a throughput of zero.
    let empty = round_stats(&totals, &steps, 2, &DecodeWindow::new());
    assert_eq!(empty.decode_tps, None);
}

/// The offsets the caller passes have to describe the round it is rolling back.
///
/// Unreachable through [`super::rollback_round`], which takes this arm only
/// when `target_offset` is inside the round — so this is the low-level
/// contract, read directly, and it is why the branch above it is not the only
/// thing standing between a caller's arithmetic and a wrong prefix. `lin` is
/// never touched: the refusal is before it.
#[test]
fn rollback_refuses_offsets_that_overrun_the_round() {
    let mut lin = vec![LinearAttnCache::new()];
    let err = rollback_round_caches(
        &mut [],
        Some(&mut lin),
        &[1, 2, 3],
        100,
        105,
        false,
        Device::Cpu,
    )
    .err()
    .map_or_else(String::new, |e| e.to_string());
    assert!(
        err.contains("retained prefix 5 exceeds the 3 tokens"),
        "offsets that overrun the round must be refused; got: {err:?}"
    );
}

/// Which arm the rollback takes is decided by the round's own extent, and it is
/// decidable without a device.
///
/// The caches stand at `pre_round_offset + round_tokens.len()`, so a target
/// below that dropped positions and one at or above it dropped none. A fresh
/// recurrent cache separates the two arms: the refold arm asks it for the tape
/// its forwards were supposed to record and refuses when there is none, and the
/// disarm arm takes whatever tape is there and returns.
///
/// Mutation: `<` to `<=` in [`rollback_round`]. A full accept then refolds a
/// prefix nothing rejected — which on a real round is a recurrent state rebuilt
/// from a tape the loop was about to drop.
#[test]
fn the_rollback_refolds_only_when_the_round_dropped_positions() {
    // (target offset, whether the round dropped anything)
    let cases = [(99, true), (102, true), (103, false), (104, false)];
    for (target, dropped) in cases {
        let mut lin = vec![LinearAttnCache::new()];
        let err = rollback_round(
            &mut [],
            Some(&mut lin),
            &[1, 2, 3],
            100,
            target,
            false,
            Device::Cpu,
        )
        .err()
        .map_or_else(String::new, |e| e.to_string());
        let refolded = err.contains("recurrent layer 0 has no round tape");
        assert_eq!(
            refolded,
            dropped,
            "the caches stand at 103 having consumed three tokens from 100, so a \
             target of {target} {} — got {err:?}",
            if dropped {
                "is a partial accept and must refold"
            } else {
                "dropped nothing and must only disarm"
            }
        );
    }
}
