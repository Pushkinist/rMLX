//! Which rows a round hands the drafter, by identity rather than by count.
//!
//! The round loop slices its conditioning rows out of the verify pass's capture
//! and projects them. Every wrong slice of the right length — the rejected tail,
//! a shifted window, the rows in another order — produces a buffer of the same
//! shape, and greedy verification then emits the verifier's own tokens whatever
//! the drafter was conditioned on. So the answer cannot say which rows were
//! taken, the row count cannot either, and this is what does: the capture's rows
//! carry their absolute position, and the carried projection is compared against
//! the projection of the rows the round is supposed to have kept.
//!
//! Runs on the CPU device against the fixture drafter; no snapshot, no verifier.

use rmlx_mlx::{concatenate, Array, Device};

use super::committed_rows;
use crate::speculative::dflash2::DFlash2Drafter;

/// The fixture drafter's width, matching `forward_tests`.
const SCALE_HIDDEN: usize = 64;

/// Largest difference two projections of the same rows may show — the same
/// bound `forward_tests` derives, and five orders under a projection of
/// different rows.
const PROJECTION_TOL: f32 = 1e-5;

#[allow(
    clippy::expect_used,
    reason = "test helper: a fixture that cannot be loaded is the assertion failing"
)]
fn fixture_drafter() -> DFlash2Drafter {
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dflash2_scale");
    DFlash2Drafter::load(std::path::Path::new(fixture), SCALE_HIDDEN, Device::Cpu)
        .expect("the fixture snapshot loads")
}

/// Rows that differ in direction, element `c` of absolute position `r` being
/// `sin(0.37 r + 0.11 c)` — the fixture `forward_tests` uses, for the reason it
/// gives there.
#[allow(
    clippy::expect_used,
    reason = "test helper: an array that cannot be built is the assertion failing"
)]
fn rows_at(first: i32, rows: i32, width: i32) -> Array {
    let data: Vec<f32> = (first..first + rows)
        .flat_map(|r| (0..width).map(move |c| (0.37 * r as f32 + 0.11 * c as f32).sin()))
        .collect();
    Array::from_f32_slice(&data, &[1, rows, width]).expect("build rows")
}

#[allow(
    clippy::expect_used,
    reason = "test assertion: an array that cannot be read back is the assertion failing"
)]
fn to_f32(a: &Array) -> Vec<f32> {
    let bytes = a.to_bytes().expect("read array bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().expect("4 bytes is an f32")))
        .collect()
}

fn max_abs_diff(a: &Array, b: &Array) -> f32 {
    let (x, y) = (to_f32(a), to_f32(b));
    assert_eq!(x.len(), y.len(), "compared arrays must have the same shape");
    x.iter()
        .zip(y.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0f32, f32::max)
}

/// The rows a round commits are the accepted prefix of its capture, and across
/// rounds they tile the token sequence once each.
///
/// The rounds here are a full accept, a partial one and a single-token one, over
/// a window the buffer saturates, so the tail being compared is a mixture of
/// rows committed by different rounds at different call shapes.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "test assertion: the rank is the one the drafter's projection returns"
)]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a projection that cannot be taken is the assertion failing"
)]
fn a_round_commits_the_accepted_prefix_of_its_capture() {
    let drafter = fixture_drafter();
    let width = (drafter.cfg.target_layer_ids.len() * SCALE_HIDDEN) as i32;
    let keep = drafter.conditioning_rows();

    // The prompt's rows, then rounds of block 5 accepting 4, 1 and 0 proposals.
    let prompt_rows = 6;
    let mut history = rows_at(0, prompt_rows, width);
    let mut carried = drafter
        .project_conditioning(
            &drafter
                .trim_conditioning(&history)
                .expect("trim the prompt"),
        )
        .expect("project the prompt");
    // The next position the verifier will score: the correction the last round
    // emitted, which is this round's carry token and its capture's row 0.
    let mut next = prompt_rows;

    for (round, accept) in [4usize, 1, 0].into_iter().enumerate() {
        // The verify pass scores the carry token and four proposals. Only the
        // first `accept + 1` rows survive the round's rollback.
        let v_hidden = rows_at(next, 5, width);
        let committed = committed_rows(&v_hidden, accept, width, Device::Cpu)
            .expect("slice the committed rows");

        assert_eq!(
            to_f32(&committed),
            to_f32(&rows_at(next, accept as i32 + 1, width)),
            "round {round}: the committed rows are not positions {next}..={} of the \
             capture",
            next + accept as i32
        );

        let (grown, projected) = drafter
            .advance_conditioning(&carried, &committed)
            .expect("advance the conditioning");
        carried = grown;
        assert_eq!(projected, accept as i32 + 1, "round {round}");

        history = concatenate(&[&history, &committed], 1, Device::Cpu).expect("extend the history");
        // Every position from the prompt through this round's accepted prefix,
        // each exactly once, trimmed to the window.
        let want = drafter
            .project_conditioning(
                &drafter
                    .trim_conditioning(&history)
                    .expect("trim the history"),
            )
            .expect("project the history");
        let diff = max_abs_diff(&carried, &want);
        assert!(
            diff < PROJECTION_TOL,
            "round {round}: the carried rows are not the accepted prefixes of every \
             round so far — they differ by {diff:e}"
        );

        // The correction the verifier emitted at the first rejected position is
        // the next round's carry token, and it is scored there for the first
        // time: no round commits it twice and none skips it.
        next += accept as i32 + 1;
        assert!(
            carried.shape()[1] <= keep,
            "round {round} carries {} rows past a bound of {keep}",
            carried.shape()[1]
        );
    }

    // The prompt's rows plus each round's carry token and accepted proposals.
    assert_eq!(
        next,
        prompt_rows + 5 + 2 + 1,
        "the rounds must have consumed each position exactly once"
    );
}

/// A capture shorter than the round accepted is refused rather than sliced past
/// its own end.
#[test]
fn a_capture_shorter_than_the_accepted_prefix_is_refused() {
    let width = 4;
    let v_hidden = rows_at(0, 3, width);
    let err = match committed_rows(&v_hidden, 5, width, Device::Cpu) {
        Ok(a) => panic!(
            "a capture of 3 rows accepted 5 proposals, got {:?}",
            a.shape()
        ),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("holds 3 positions"),
        "the refusal does not say what it had: {err}"
    );
}
