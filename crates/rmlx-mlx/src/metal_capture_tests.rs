//! Unit tests for the GPU-capture window policy and request validation.
//!
//! Both surfaces under test are pure: [`Window`] holds no Metal state and
//! [`validate`] takes the capture-layer flag as an argument instead of reading
//! the environment, so nothing here touches a GPU or the process env.
//!
//! Run them with the feature on — an ordinary `cargo test` compiles none of
//! this out (`make test-capture`):
//!
//! ```sh
//! cargo test -p rmlx-mlx --features metal-capture metal_capture
//! ```

use super::{min_tokens_for_window, validate, Action, Window};

/// Drive `n` boundaries through a window, returning the action at each.
fn drive(w: &mut Window, n: u32) -> Vec<Action> {
    (0..n).map(|_| w.tick()).collect()
}

// ---------------------------------------------------------------------------
// Window policy
// ---------------------------------------------------------------------------

#[test]
fn window_opens_after_the_skipped_steps() {
    let mut w = Window::new(4, 8);
    let acts = drive(&mut w, 5);
    assert_eq!(
        acts,
        vec![
            Action::Idle,
            Action::Idle,
            Action::Idle,
            Action::Idle,
            Action::Open
        ],
        "skip=4 must open at the 5th boundary, not earlier and not later"
    );
    assert!(w.opened());
    assert!(!w.closed());
}

#[test]
fn window_closes_after_the_step_budget() {
    let mut w = Window::new(4, 8);
    let acts = drive(&mut w, 13);
    // Open at boundary 5, close at boundary 13 → steps 5..=12 inside = 8.
    assert_eq!(acts.iter().filter(|a| **a == Action::Open).count(), 1);
    assert_eq!(acts.get(12), Some(&Action::Close), "closes at boundary 13");
    assert!(w.closed());
    assert_eq!(w.captured_steps_at_end(), 8);
}

#[test]
fn window_opens_and_closes_exactly_once() {
    let mut w = Window::new(2, 3);
    let acts = drive(&mut w, 40);
    assert_eq!(acts.iter().filter(|a| **a == Action::Open).count(), 1);
    assert_eq!(acts.iter().filter(|a| **a == Action::Close).count(), 1);
    assert_eq!(w.captured_steps_at_end(), 3, "budget is not exceeded");
}

#[test]
fn zero_skip_opens_at_the_first_boundary() {
    // Boundary 1 opens (step 1 runs inside), boundary 2 is step 2, boundary 3
    // closes — two whole steps captured.
    let mut w = Window::new(0, 2);
    assert_eq!(w.tick(), Action::Open);
    assert_eq!(w.tick(), Action::Idle);
    assert_eq!(w.tick(), Action::Close);
    assert_eq!(w.captured_steps_at_end(), 2);
}

#[test]
fn short_generation_reports_the_steps_it_actually_got() {
    // Opens at boundary 5; generation ends after boundary 9, so steps 5..=9 ran.
    let mut w = Window::new(4, 8);
    drive(&mut w, 9);
    assert!(w.opened());
    assert!(!w.closed(), "budget was not spent");
    assert_eq!(w.captured_steps_at_end(), 5);
}

#[test]
fn window_that_never_opens_reports_zero_and_what_it_needed() {
    let mut w = Window::new(10, 4);
    drive(&mut w, 6);
    assert!(!w.opened());
    assert_eq!(w.captured_steps_at_end(), 0);
    assert_eq!(w.seen(), 6);
    assert_eq!(w.steps_needed_to_open(), 11);
}

#[test]
fn min_tokens_covers_open_fill_and_close() {
    // The boundary count a generation reaches is what the loop drives; the
    // window needs one boundary past the closing one to exist.
    let (skip, steps) = (4, 8);
    let need = min_tokens_for_window(skip, steps);
    let mut w = Window::new(skip, steps);
    drive(&mut w, need - 1);
    assert!(
        w.closed(),
        "min_tokens_for_window must be enough boundaries for a clean close"
    );

    let mut short = Window::new(skip, steps);
    drive(&mut short, need - 2);
    assert!(
        !short.closed(),
        "one boundary fewer must NOT close — otherwise the minimum is overstated"
    );
}

// ---------------------------------------------------------------------------
// Request validation
// ---------------------------------------------------------------------------

#[test]
fn validate_rejects_a_missing_capture_layer_with_the_fix() {
    let dir = std::env::temp_dir();
    let err = validate(&dir.join("rmlx-vcl.gputrace"), false, 8)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("MTL_CAPTURE_ENABLED"),
        "the error must name the env var Metal needs at launch, got: {err}"
    );
}

#[test]
fn validate_accepts_a_fresh_path_when_the_layer_is_inserted() {
    let dir = std::env::temp_dir();
    let res = validate(
        &dir.join("rmlx-fresh-path-that-does-not-exist.gputrace"),
        true,
        8,
    );
    assert!(res.is_ok(), "fresh path must validate: {res:?}");
}

#[test]
fn validate_rejects_a_zero_step_window() {
    let dir = std::env::temp_dir();
    let err = validate(&dir.join("rmlx-zero.gputrace"), true, 0)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("zero steps"),
        "a zero-wide window must be rejected, got: {err}"
    );
}

#[test]
fn validate_rejects_an_existing_destination() {
    // Metal refuses to overwrite a bundle; catching it here costs seconds
    // instead of a full model load. Any existing path proves the branch.
    let existing = std::env::temp_dir();
    assert!(existing.is_dir(), "test fixture: temp dir must exist");
    let err = validate(&existing, true, 8)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("already exists"),
        "an occupied destination must be rejected, got: {err}"
    );
}

#[test]
fn validate_rejects_a_missing_parent_directory() {
    let missing = std::env::temp_dir()
        .join("rmlx-no-such-dir-8f3a1c")
        .join("t.gputrace");
    let err = validate(&missing, true, 8)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("does not exist"),
        "a missing parent dir must be rejected, got: {err}"
    );
}
