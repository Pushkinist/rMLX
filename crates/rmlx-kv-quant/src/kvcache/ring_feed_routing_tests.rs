//! Which append arm a packed-K chunk takes, and why the block arm is live.
//!
//! The packed rotor / iso K appends have two arms: a ring-only tail that skips
//! the CPU block push, and a block path that pushes a CPU block and first
//! reconciles any pre-existing ring-only tail (`materialize_*_ring_tail`). The
//! block arm was documented as unreachable at `b == 1` — "a fused multi-token
//! append transition does not occur" — which reads as "this code is dead" and
//! invites removing the reconcile. It is not dead: the arm is selected by the
//! caller's `RingFeed`, and the legacy `update_*` entries a `q_seq > 1` forward
//! falls through to pass `Maintain` / `Skip`.

use super::{is_ring_only_append, RingFeed};

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

/// The block arm at `b == 1` is live, which is the claim the old comment denied.
///
/// `update_rotor_k_only_{3,4}` passes `Maintain` and `update_rotor{3,4}_sym` /
/// the asym entries pass `Skip`. Those are the entries the SDPA dispatcher falls
/// through to whenever its fused gate (`q_seq == 1`) rejects the forward — a
/// speculative verify chunk, or a continuation turn's prompt tokens against a
/// warm cache. `b > 1` is a *second*, and today unreachable, way in; asserting
/// only that one would leave the live way untested.
#[test]
fn block_path_is_reachable_at_batch_one() {
    let batch_one = [1_i32, 2, 5, 128];
    let batched = [2_i32, 2, 5, 128];

    assert!(
        !is_ring_only_append(RingFeed::Maintain, &batch_one),
        "the block path must be reachable without a b > 1 chunk"
    );
    assert!(
        !is_ring_only_append(RingFeed::MaintainRingOnly, &batched),
        "b > 1 is the second way into the block path, not the only one"
    );
}
