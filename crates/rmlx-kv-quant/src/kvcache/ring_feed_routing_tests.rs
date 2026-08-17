//! Which append arm a packed-K chunk takes, and why the block arm is live.
//!
//! The packed rotor K append has two arms: a ring-only tail that skips the CPU
//! block push, and a block path that pushes a CPU block and first reconciles any
//! pre-existing ring-only tail (`materialize_*_ring_tail`). The block arm was
//! documented as unreachable at `b == 1` — "a fused multi-token append
//! transition does not occur" — which reads as "this code is dead" and invites
//! removing the reconcile.
//!
//! # What these tests can and cannot reach
//!
//! They assert the routing of the **named feed constants** the legacy rotor
//! `update_*` entries pass ([`LEGACY_ROTOR_SYM_FEED`],
//! [`LEGACY_ROTOR_K_ONLY_FEED`]). That is a CPU-only gate, so `make ci` runs it.
//!
//! It is **not** a proof that a given entry still passes those constants: the
//! append path they feed runs an MSL encode kernel, so nothing below `make
//! gpu-test` can observe the consequence. Those pins are four `#[ignore]` tests
//! in `rotor_flash_dispatch_tests`:
//!
//! * `rotor_k_only_{3,4}_multi_token_append_after_fused_decode_takes_the_block_path`
//!   — block pushed **and** ring kept, which admits only `Maintain`.
//! * `rotor_sym{3,4}_multi_token_append_after_fused_decode_drops_the_ring` —
//!   block pushed **and** ring dropped, which admits only `Skip`.
//!
//! If a call site is changed to bypass its constant, this file stays green and
//! the matching pair goes red.

use super::{is_ring_only_append, RingFeed, LEGACY_ROTOR_K_ONLY_FEED, LEGACY_ROTOR_SYM_FEED};

/// The arm is chosen by `feed` and `b` only — never by chunk length.
///
/// A ring-only feed carries a 5-token chunk as happily as a single decode step,
/// and a `Maintain` / `Skip` feed takes the block path even for a single token.
/// So "multi-token" is not what routes an append to the block arm; the caller's
/// feed is, and the fused decode entry is the only caller that passes
/// `MaintainRingOnly`.
#[test]
fn ring_only_arm_is_selected_by_feed_and_batch_not_by_chunk_length() {
    let single = [1_i32, 2, 1, 128];
    let multi = [1_i32, 2, 5, 128];

    for shape in [single, multi] {
        assert!(
            is_ring_only_append(RingFeed::MaintainRingOnly, &shape),
            "a ring-only feed must take the ring arm at seq={} — the ring stores a \
             multi-token chunk the same way it stores one decode step",
            shape[2]
        );
        assert!(
            !is_ring_only_append(RingFeed::Maintain, &shape),
            "a Maintain feed must take the block path at seq={}",
            shape[2]
        );
        assert!(
            !is_ring_only_append(RingFeed::Skip, &shape),
            "a Skip feed must take the block path at seq={}",
            shape[2]
        );
    }
}

/// Both legacy rotor feeds route a `b == 1` multi-token chunk to the block path.
///
/// This is the claim the old comment denied. The legacy entries are what the
/// SDPA dispatcher falls through to whenever its fused gate (`q_seq == 1`)
/// rejects the forward — a speculative verify chunk, or a continuation turn's
/// prompt tokens against a warm cache. `b > 1` is a *second*, and today
/// unreachable, way in; asserting only that one would leave the live way
/// untested.
///
/// Asserting the constants rather than bare `RingFeed` values is what ties this
/// to the call sites: changing what the legacy entries pass means changing a
/// constant this test reads.
#[test]
fn both_legacy_rotor_feeds_reach_the_block_path_at_batch_one() {
    let batch_one = [1_i32, 2, 5, 128];

    for (name, feed) in [
        ("LEGACY_ROTOR_SYM_FEED", LEGACY_ROTOR_SYM_FEED),
        ("LEGACY_ROTOR_K_ONLY_FEED", LEGACY_ROTOR_K_ONLY_FEED),
    ] {
        assert!(
            !is_ring_only_append(feed, &batch_one),
            "{name} must reach the block path without a b > 1 chunk"
        );
    }

    let batched = [2_i32, 2, 5, 128];
    assert!(
        !is_ring_only_append(RingFeed::MaintainRingOnly, &batched),
        "b > 1 is the second way into the block path, not the only one"
    );
}

/// The sym feed clears the ring; the K-only feed keeps it.
///
/// That difference is load-bearing for the truncation story: with the ring gone,
/// the CPU blocks are the only copy of the prefix, so a mid-block cut that drops
/// the trailing block is unrecoverable. The sym path is therefore where a
/// speculative partial accept bites first.
#[test]
fn only_the_k_only_legacy_feed_keeps_the_ring() {
    assert_eq!(
        LEGACY_ROTOR_SYM_FEED,
        RingFeed::Skip,
        "the sym/asym legacy entries dequantize the whole prefix, so they drop the ring \
         and make the CPU blocks the only copy"
    );
    assert_eq!(
        LEGACY_ROTOR_K_ONLY_FEED,
        RingFeed::Maintain,
        "the K-only legacy entry keeps the ring, so a ring-only tail survives a fallback step"
    );
}
