use super::*;

/// Compile-check: the public MTP surface exists with the expected sigs.
#[test]
fn mtp_module_compiles() {
    // Reference the items so the symbols are checked at compile time
    // without spelling out their (clippy-flagged complex) fn types.
    let _load = MtpDrafter::load;
    let _ = _load;
}

/// The sidecar's declared block is the depth it was trained at, and the loop
/// runs the depth the request asks for.
///
/// Every shipped Qwen3.5-family sidecar declares `block_size: 3`, so a request
/// clamped to the declaration could never run a deeper block on any checkpoint
/// that exists.
#[test]
fn a_request_deeper_than_the_declared_block_runs_at_the_request() {
    for requested in [4, 6, 8, 16] {
        assert_eq!(round_block_total(requested), requested);
    }
}

/// A request past what one verify forward can score is clamped to it.
#[test]
fn a_request_past_the_verify_ceiling_runs_at_the_ceiling() {
    assert_eq!(round_block_total(MAX_BLOCK_SIZE), MAX_BLOCK_SIZE);
    assert_eq!(round_block_total(MAX_BLOCK_SIZE + 1), MAX_BLOCK_SIZE);
    assert_eq!(round_block_total(usize::MAX), MAX_BLOCK_SIZE);
}

/// Below two there is a seed and no draft, so there is no round to run.
#[test]
fn a_request_below_two_runs_at_two() {
    assert_eq!(round_block_total(0), 2);
    assert_eq!(round_block_total(1), 2);
    assert_eq!(round_block_total(2), 2);
}
