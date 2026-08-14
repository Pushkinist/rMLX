//! Tests for the `--gpu-capture` driver's loud-failure paths.
//!
//! These compile only under the `metal-capture` feature, like the module they
//! cover:
//!
//! ```sh
//! cargo test -p rmlx-cli --features metal-capture gpu_capture
//! ```
//!
//! None of them arms the process-global capture driver — every case here is
//! rejected, or observed, before any Metal state is touched.

use std::path::Path;

use rmlx_mlx::metal_capture::Outcome;

use super::{arm, bundle_bytes, describe, report};

#[test]
fn no_flag_means_no_capture_and_no_error() {
    let requested = arm(None, 4, 8, 32);
    assert!(matches!(requested, Ok(false)), "got {requested:?}");
}

#[test]
fn rejects_a_generation_too_short_to_close_the_window() {
    // skip 4 + steps 8 needs 14 tokens; 12 cannot close the window.
    let err = arm(Some(Path::new("/tmp/rmlx-short.gputrace")), 4, 8, 12)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("--max-tokens is 12") && err.contains("at least 14"),
        "the error must name both the requirement and what was passed, got: {err}"
    );
}

#[test]
fn accepts_the_exact_minimum_token_count() {
    // The boundary case must NOT be rejected, or the driver is overstating the
    // requirement. It fails later (no capture layer in a test process), but the
    // token-count gate is what is under test here.
    let err = arm(Some(Path::new("/tmp/rmlx-exact.gputrace")), 4, 8, 14)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        !err.contains("--max-tokens"),
        "14 tokens is exactly enough and must clear the token gate, got: {err}"
    );
}

#[test]
fn report_is_silent_when_nothing_was_requested() {
    assert!(report(false).is_ok());
}

#[test]
fn report_fails_loudly_when_an_armed_capture_produced_nothing() {
    let err = report(true)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("already disarmed"),
        "an armed capture that produced nothing must be an error, got: {err}"
    );
}

#[test]
fn zero_decode_steps_names_the_uncovered_arch_not_the_token_count() {
    // An arch that does not decode through the shared loop reaches the window
    // zero times. Telling the operator to raise --max-tokens there is a dead
    // end, so this case must point at the hook instead.
    let err = describe(Outcome::NeverOpened { seen: 0, needed: 5 }, true)
        .err()
        .map(|e| e.to_string())
        .unwrap_or_default();
    assert!(
        err.contains("pipelined_decode"),
        "zero steps must name the shared decode loop, got: {err}"
    );
    assert!(
        !err.contains("--max-tokens"),
        "zero steps must NOT send the operator after the token count, got: {err}"
    );
}

#[test]
fn a_window_that_never_opened_is_an_error_naming_what_it_needed() {
    // Reached when generation stops (EOS) before the window's first step. The
    // run "succeeded" and wrote no trace — exactly the silent-failure mode this
    // driver exists to prevent, so it must exit non-zero.
    let err = describe(
        Outcome::NeverOpened {
            seen: 6,
            needed: 31,
        },
        true,
    )
    .err()
    .map(|e| e.to_string())
    .unwrap_or_default();
    assert!(
        err.contains("6 decode steps") && err.contains("step 31"),
        "the error must report both what the run reached and what the window needed, got: {err}"
    );
}

#[test]
fn a_short_but_real_capture_is_reported_not_rejected() {
    // Fewer steps than asked for is still a usable trace; it must warn, not fail.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("short.gputrace");
    std::fs::create_dir(&path).expect("mkdir");
    let res = describe(
        Outcome::Captured {
            path,
            steps: 3,
            complete: false,
        },
        true,
    );
    assert!(
        res.is_ok(),
        "a short capture must not fail the run: {res:?}"
    );
}

#[test]
fn bundle_bytes_sums_a_directory_and_tolerates_a_missing_path() {
    let dir = tempfile::tempdir().expect("tempdir");
    let nested = dir.path().join("store0");
    std::fs::create_dir(&nested).expect("mkdir");
    std::fs::write(dir.path().join("index"), b"0123456789").expect("write");
    std::fs::write(nested.join("blob"), b"01234").expect("write");

    assert_eq!(bundle_bytes(dir.path()), 15, "bundle size is recursive");
    assert_eq!(
        bundle_bytes(&dir.path().join("nope")),
        0,
        "a missing bundle must report zero, not panic"
    );
}
