//! Model-free routing-guard tests for `ArchGenerator`.
//!
//! `reject_combined_image_audio` is the request-level guard that fires before
//! any tower access, so these tests need no model snapshot. They lock the
//! invariant that a request carrying BOTH an image and an audio clip is
//! rejected with a CLEAR error (never a silent drop), while the audio-only,
//! image-only, and text-only paths route unchanged (no rejection).

use super::reject_combined_image_audio;
use rmlx_core::Error;

#[test]
#[allow(
    clippy::expect_used,
    reason = "guard must return Some(err) for the both-present case; .expect() documents that invariant"
)]
fn combined_image_and_audio_is_rejected_with_clear_error() {
    let err = reject_combined_image_audio(true, true)
        .expect("combined image + audio must be rejected, not silently dropped");
    // Must surface as a request-level `Other` error (proper HTTP error, not a panic).
    assert!(
        matches!(err, Error::Other(_)),
        "expected Error::Other request-level rejection, got {err:?}"
    );
    let msg = err.to_string();
    assert!(
        msg.contains("combined image + audio input in one request is not supported"),
        "error message must clearly explain the unsupported combination: {msg}"
    );
    assert!(
        msg.contains("separate turns"),
        "error message must tell the caller to send them in separate turns: {msg}"
    );
}

#[test]
fn audio_only_request_is_not_rejected() {
    assert!(
        reject_combined_image_audio(false, true).is_none(),
        "audio-only request must route to the audio path, not be rejected"
    );
}

#[test]
fn image_only_request_is_not_rejected() {
    assert!(
        reject_combined_image_audio(true, false).is_none(),
        "image-only request must route to the image path, not be rejected"
    );
}

#[test]
fn text_only_request_is_not_rejected() {
    assert!(
        reject_combined_image_audio(false, false).is_none(),
        "text-only request must route to the plain path, not be rejected"
    );
}
