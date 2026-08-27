#![allow(
    clippy::indexing_slicing,
    reason = "test fixtures establish these fixed lengths immediately before indexing; the slices exercise identity behavior, not fallible input handling"
)]

use super::prompt_identity_matches;

#[test]
fn stored_prompt_longer_than_last_full_block_candidate_is_verified() {
    // A 1,118-token stored snapshot is discovered through its first four
    // full blocks (1,024 tokens), but identity verification covers the
    // non-block-aligned tail as well.
    let stored: Vec<u32> = (0..1_118).collect();
    let candidate: Vec<u32> = stored[..1_024].to_vec();
    assert_eq!(candidate.len(), 1_024);
    assert!(!prompt_identity_matches(&candidate, &stored));
    assert!(prompt_identity_matches(&stored, &stored));

    let mut extension = stored.clone();
    extension.extend([1_118, 1_119]);
    assert!(prompt_identity_matches(&extension, &stored));
}

#[test]
fn divergent_or_truncated_request_fails_closed() {
    let stored: Vec<u32> = (0..1_118).collect();

    let mut divergent = stored.clone();
    divergent[1_117] = u32::MAX;
    assert!(!prompt_identity_matches(&divergent, &stored));

    assert!(!prompt_identity_matches(&stored[..1_117], &stored));
}
