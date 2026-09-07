//! What a chunked capture keeps, against what it would have kept unbounded.
//!
//! The oracle is the unbounded accumulation itself: the same chunks are pushed
//! twice, once with a tail limit and once without, and the bounded answer has to
//! be the tail of the other one row for row. That is what makes a widened limit
//! visible — a bound that kept more rows, or the whole capture, still returns
//! rows the drafter can read, and only the count and the peak say so.
//!
//! The cases sweep the boundaries the published shape does not reach: a window
//! narrower than one chunk (in-tree drafter fixtures declare 8 and 64, and
//! `check_config` refuses only a window under 2), a prompt exactly at the
//! window and one row past it, and a window of zero rows.
//!
//! Everything here runs on the CPU device and needs no model snapshot.

use rmlx_mlx::{Array, Device};

use super::CaptureTail;

/// A stand-in for `len(capture_layer_ids) * hidden_size`, narrow enough that a
/// long prompt's rows are cheap to compare byte for byte.
const WIDTH: i32 = 4;

/// `(kept rows, prefill chunk, prompt positions)`.
///
/// 2047 is the published DFlash 2 pair's `sliding_window - 1` and 1024 is the
/// chunk both round loops pass.
const CASES: [(Option<usize>, usize, usize); 7] = [
    (Some(2047), 1024, 5000),
    (Some(63), 1024, 5000),
    (Some(2047), 1024, 2047),
    (Some(2047), 1024, 2048),
    (Some(2047), 1024, 1500),
    (Some(0), 1024, 5000),
    (None, 1024, 5000),
];

/// A chunk whose every row holds that row's own absolute prompt position, so a
/// kept row says where it came from.
#[allow(
    clippy::expect_used,
    reason = "test helper: an array that cannot be built is the assertion failing"
)]
fn chunk_at(first_row: usize, rows: usize) -> Array {
    let data: Vec<f32> = (first_row..first_row + rows)
        .flat_map(|r| std::iter::repeat_n(r as f32, WIDTH as usize))
        .collect();
    Array::from_f32_slice(&data, &[1, rows as i32, WIDTH]).expect("build chunk")
}

/// Push `n` prompt positions through a fresh accumulator in `chunk`-sized
/// chunks, and report the most rows it held at any point.
#[allow(
    clippy::expect_used,
    reason = "test helper: a chunk the accumulator refuses is the assertion failing"
)]
fn accumulate(keep: Option<usize>, chunk: usize, n: usize) -> (CaptureTail, usize) {
    let mut tail = CaptureTail::new(keep);
    let mut peak = 0usize;
    let mut pos = 0usize;
    while pos < n {
        let end = (pos + chunk).min(n);
        tail.push(chunk_at(pos, end - pos)).expect("push chunk");
        peak = peak.max(tail.retained_rows());
        pos = end;
    }
    (tail, peak)
}

#[allow(
    clippy::expect_used,
    reason = "test assertion: an array that cannot be read back is the assertion failing"
)]
fn bytes_of(a: &Array) -> Vec<u8> {
    a.to_bytes().expect("read capture bytes")
}

/// The first element of every row — that row's prompt position.
#[allow(
    clippy::indexing_slicing,
    reason = "test assertion: the rank is the one the accumulator returns"
)]
#[allow(
    clippy::expect_used,
    reason = "test assertion: an array that cannot be read back is the assertion failing"
)]
fn row_positions(a: &Array) -> Vec<usize> {
    let width = a.shape()[2] as usize;
    bytes_of(a)
        .chunks_exact(4)
        .step_by(width)
        .map(|b| {
            let v = f32::from_le_bytes(b.try_into().expect("4 bytes is an f32"));
            v as usize
        })
        .collect()
}

/// Across every shape of window, chunk and prompt: the kept rows are the newest
/// ones, byte for byte what the unbounded capture ends with, and nothing older
/// is still held.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: a capture that cannot be joined is the assertion failing"
)]
fn the_kept_rows_are_the_tail_the_unbounded_capture_would_have_returned() {
    for (keep, chunk, n) in CASES {
        let case = format!("keep={keep:?} chunk={chunk} n={n}");
        let want_rows = keep.map_or(n, |k| k.min(n));

        let (unbounded, unbounded_peak) = accumulate(None, chunk, n);
        assert_eq!(
            unbounded_peak, n,
            "{case}: an unbounded accumulation holds every position"
        );
        let whole = unbounded.finish(Device::Cpu).expect("join unbounded");
        let reference = whole
            .slice(
                &[0, (n - want_rows) as i32, 0],
                &[1, n as i32, WIDTH],
                &[1, 1, 1],
                Device::Cpu,
            )
            .expect("slice the unbounded tail");

        let (bounded, peak) = accumulate(keep, chunk, n);
        let kept = bounded.finish(Device::Cpu).expect("join bounded");

        assert_eq!(
            kept.shape(),
            vec![1, want_rows as i32, WIDTH],
            "{case}: the kept capture is not the rows the caller asked for"
        );
        assert_eq!(
            row_positions(&kept),
            row_positions(&reference),
            "{case}: the kept rows are not the newest ones"
        );
        assert_eq!(
            bytes_of(&kept),
            bytes_of(&reference),
            "{case}: the kept rows differ from the tail the unbounded capture would have handed back"
        );

        // A bound that held every chunk to the end returns the same rows, and
        // only this says so.
        if let Some(keep) = keep {
            assert!(
                peak <= keep + chunk,
                "{case}: held {peak} rows, more than the window plus one chunk"
            );
        }
    }
}

/// A capture of another rank is refused rather than joined along whichever axis
/// happens to be second.
#[test]
fn a_chunk_of_the_wrong_rank_is_refused() {
    let mut tail = CaptureTail::new(Some(2047));
    let flat = match Array::from_f32_slice(&[0.0, 1.0, 2.0, 3.0], &[1, 4]) {
        Ok(a) => a,
        Err(e) => panic!("build flat chunk: {e}"),
    };
    let err = match tail.push(flat) {
        Ok(()) => panic!("a rank-2 chunk was accepted"),
        Err(e) => e.to_string(),
    };
    assert!(
        err.contains("[1, rows, width]"),
        "the refusal does not name the shape it wanted: {err}"
    );
}

/// Nothing pushed is an error, not an empty answer a caller would condition on.
#[test]
fn an_empty_accumulation_is_refused() {
    let tail = CaptureTail::new(None);
    match tail.finish(Device::Cpu) {
        Ok(a) => panic!("an empty accumulation returned {:?}", a.shape()),
        Err(e) => assert!(
            e.to_string().contains("no capture chunk"),
            "unexpected refusal: {e}"
        ),
    }
}
