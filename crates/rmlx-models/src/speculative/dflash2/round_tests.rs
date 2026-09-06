//! The block a round runs at.
//!
//! The rest of the round loop needs a verifier and is covered where that is
//! available — the acceptance walk and the cache rollback in
//! `speculative/tests.rs`, the drafter forward and the selector in
//! `forward_tests.rs` and `selector_tests.rs`, and the loop end to end against
//! plain greedy in `tests/spec_greedy_equivalence.rs`. What is here is the one
//! decision the loop makes before any of that: how wide a block to run, which
//! sizes every per-round allocation it then makes.

use super::{round_block_total, MAX_BLOCK_SIZE};

/// The request wins when it asks for less than the checkpoint was trained at.
#[test]
fn a_request_narrower_than_the_checkpoint_runs_at_the_request() {
    assert_eq!(round_block_total(5, 8), 5);
}

/// The checkpoint wins when the request asks for more than it was trained at:
/// the selector's chain is defined over the trained block and no wider.
#[test]
fn a_request_wider_than_the_checkpoint_runs_at_the_checkpoint() {
    assert_eq!(round_block_total(8, 5), 5);
}

/// Two positions is the floor at both ends — a block of one is the seed alone
/// and drafts nothing, so a request or a checkpoint below it still runs a round
/// that proposes something.
#[test]
fn neither_side_can_take_the_block_below_a_seed_and_one_draft() {
    assert_eq!(round_block_total(0, 8), 2);
    assert_eq!(round_block_total(8, 1), 2);
    assert_eq!(round_block_total(0, 0), 2);
}

/// A checkpoint whose config never went through `check_config` is still bounded.
///
/// This is the case the clamp exists for, and the only one that separates it
/// from the loader's refusal: [`super::DFlash2Drafter`] is publicly
/// constructible with public fields, so `declared` here is whatever the caller
/// put in the struct. Without the clamp this returns that number, and the round
/// sizes its token buffer, its verify input and its selector chain from it.
#[test]
fn a_config_the_loader_never_saw_is_still_bounded_by_one_verify_forward() {
    assert_eq!(round_block_total(usize::MAX, usize::MAX), MAX_BLOCK_SIZE);
    assert_eq!(round_block_total(usize::MAX, 4_294_967_295), MAX_BLOCK_SIZE);
    // And the request alone cannot lift it past the checkpoint either.
    assert_eq!(round_block_total(usize::MAX, 8), 8);
}

/// The ceiling admits its own value and refuses the next one, so it cannot be
/// off by one in either direction.
#[test]
fn the_ceiling_admits_itself_and_nothing_above() {
    assert_eq!(
        round_block_total(MAX_BLOCK_SIZE, MAX_BLOCK_SIZE),
        MAX_BLOCK_SIZE
    );
    assert_eq!(
        round_block_total(MAX_BLOCK_SIZE + 1, MAX_BLOCK_SIZE + 1),
        MAX_BLOCK_SIZE
    );
    assert_eq!(
        round_block_total(MAX_BLOCK_SIZE - 1, MAX_BLOCK_SIZE),
        MAX_BLOCK_SIZE - 1
    );
}
