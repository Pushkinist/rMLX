//! The conditioning buffer's window: what the round loop keeps between rounds.
//!
//! The rest of the round loop is the shape every sidecar loop here has, and its
//! parts are covered where they live — the acceptance walk and the cache
//! rollback in `speculative/tests.rs`, the drafter forward and the selector in
//! `forward_tests.rs` and `selector_tests.rs`, and the loop end to end against
//! plain greedy in `tests/spec_greedy_equivalence.rs`. What is new here is the
//! bound on the buffer those rounds carry forward, and that is what this file
//! covers.

use rmlx_mlx::{Array, Device};

use super::keep_last_rows;

/// Rows of `[1, rows, width]` counting up from `first`, one distinct value per
/// row so a window that kept the wrong end is a different answer.
#[allow(
    clippy::expect_used,
    reason = "test fixture: an Array this small failing to build is the assertion failing"
)]
fn rows_of(first: i32, rows: i32, width: i32) -> Array {
    let data: Vec<f32> = (0..rows)
        .flat_map(|r| std::iter::repeat_n((first + r) as f32, width as usize))
        .collect();
    Array::from_f32_slice(&data, &[1, rows, width]).expect("fixture array builds")
}

/// The first element of every row, which is that row's own number.
#[allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "test assertions: an array the fixture built failing to evaluate is the assertion failing, and its shape is the one the fixture gave it"
)]
fn row_numbers(a: &Array) -> Vec<i32> {
    a.eval().expect("evaluates");
    let bytes = a.to_bytes().expect("reads back");
    let shape = a.shape();
    let width = shape[2] as usize;
    let floats: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    floats.iter().step_by(width).map(|v| *v as i32).collect()
}

/// A buffer inside the window is carried whole: nothing is dropped before the
/// drafter can read it.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: an Err from a well-formed buffer is the assertion failing"
)]
fn a_buffer_inside_the_window_is_kept_whole() {
    let kept = keep_last_rows(&rows_of(0, 5, 3), 8, 3, Device::Cpu).expect("kept");
    assert_eq!(kept.shape(), vec![1, 5, 3]);
    assert_eq!(row_numbers(&kept), vec![0, 1, 2, 3, 4]);
}

/// One row past the window drops exactly the oldest one.
///
/// The narrowest case the trim is observable at, and the one an off-by-one in
/// the start index moves: a buffer exactly the window wide is not, because the
/// slice that would be taken there is the whole buffer whichever way the
/// comparison goes.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: an Err from a well-formed buffer is the assertion failing"
)]
fn one_row_past_the_window_drops_exactly_the_oldest() {
    let kept = keep_last_rows(&rows_of(0, 5, 3), 4, 3, Device::Cpu).expect("kept");
    assert_eq!(kept.shape(), vec![1, 4, 3]);
    assert_eq!(row_numbers(&kept), vec![1, 2, 3, 4]);
}

/// Past the window the **newest** rows survive.
///
/// Which end is kept is the whole property: the drafter attends over the rows
/// nearest the block, so a window that kept the oldest ones would condition
/// every round on the start of the prompt and go on drafting without an error.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: an Err from a well-formed buffer is the assertion failing"
)]
fn past_the_window_the_newest_rows_survive() {
    let kept = keep_last_rows(&rows_of(0, 9, 3), 4, 3, Device::Cpu).expect("kept");
    assert_eq!(kept.shape(), vec![1, 4, 3]);
    assert_eq!(row_numbers(&kept), vec![5, 6, 7, 8]);
}

/// Trimming is idempotent, which is what makes the loop's grow-then-trim step
/// bounded rather than merely bounded on its first pass.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: an Err from a well-formed buffer is the assertion failing"
)]
fn trimming_an_already_trimmed_buffer_changes_nothing() {
    let once = keep_last_rows(&rows_of(0, 9, 3), 4, 3, Device::Cpu).expect("kept");
    let twice = keep_last_rows(&once, 4, 3, Device::Cpu).expect("kept again");
    assert_eq!(row_numbers(&twice), vec![5, 6, 7, 8]);
}

/// A buffer of another width is refused rather than trimmed.
///
/// The width is `len(target_layer_ids) * hidden_size`, and a capture taken at a
/// different set of target layers has the same rank and the same row count. The
/// drafter's `fc` would consume it, project it, and draft from it.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: the refusal is what is asserted, and unwrapping the Ok case here is that assertion failing"
)]
fn a_conditioning_buffer_of_the_wrong_width_is_refused() {
    let err = keep_last_rows(&rows_of(0, 4, 3), 8, 5, Device::Cpu)
        .expect_err("a width the config does not predict is refused")
        .to_string();
    assert!(err.contains("[1, 4, 3]"), "names what it got: {err}");
    assert!(err.contains("[1, rows, 5]"), "names what it wanted: {err}");
}

/// A buffer of the wrong rank is refused too — the rank is what makes "axis 1
/// is the rows" true.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test assertion: the refusal is what is asserted, and unwrapping the Ok case here is that assertion failing"
)]
fn a_conditioning_buffer_of_the_wrong_rank_is_refused() {
    let flat = Array::from_f32_slice(&[0.0, 1.0, 2.0], &[3]).expect("fixture array builds");
    let err = keep_last_rows(&flat, 8, 3, Device::Cpu)
        .expect_err("a rank the config does not predict is refused")
        .to_string();
    assert!(err.contains("[3]"), "names what it got: {err}");
}
