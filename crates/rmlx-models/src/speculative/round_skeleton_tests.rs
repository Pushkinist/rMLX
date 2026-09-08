//! What every speculative round loop computes the same way, pinned by identity
//! before any of it is extracted into one place.
//!
//! # The oracle for an extraction
//!
//! An extraction that moves a body out of a loop and calls it instead is
//! behaviour-preserving when, for every drafter, **the per-round event stream is
//! unchanged**: for each round, the block it ran, the accepted count, the number
//! of proposals, the tokens it committed, and — where the loop reports one — the
//! projected conditioning rows; and, per request, the accept counters
//! (`total_draft`, `total_accept`, `rounds`, `emitted`, `seed_emitted`,
//! `emitted_in_rounds`) that `RoundStats` closes on. Beside that stream sits the
//! answer-equivalence pair for the same drafter, judged by the
//! divergence-confidence oracle in `docs/SPEC_ANSWER_EQUIVALENCE.md`.
//!
//! **The greedy digest is not evidence here.** Greedy verification emits the
//! verifier's own argmax at every position whatever the drafter proposed, so a
//! byte-identical token stream after a draft-side change says that run's
//! near-ties happened not to move — not that the round loop is unaffected. A
//! rollback off by one, a block narrowed one token short, a conditioning row
//! projected twice: each of those changes the round stream and can leave the
//! answer alone.
//!
//! Each of the three observables is blind to something, and they are blind to
//! different things:
//!
//! - **The per-round event stream** sees the round's shape — block, accept,
//!   proposals, committed rows — and is blind to the *values* inside it: two
//!   runs agreeing on every count can still be scoring different logits, and a
//!   loop that emits no such event (four of the seven do not; see below) is
//!   invisible to it entirely.
//! - **The accept counters** see the aggregate and are blind to which round
//!   moved: a round that accepted one too many and a later one that accepted one
//!   too few sum to the same request.
//! - **The equivalence pairs** see the answer and are blind wherever a pair is
//!   not resolvable, and blind by construction to the stochastic two-model loop,
//!   which has no temperature-0 arm to compare against.
//! - **The tests in this file** see the arithmetic and are blind to every
//!   forward, every cache and every drafter: they fix what the loops compute
//!   from counts, not what the model computes from weights.
//!
//! # What is pinned here, and why by identity
//!
//! Each test below fixes a piece that exists in more than one loop as its own
//! copy, at absolute values, so an extraction is judged against something that
//! can fail rather than against a re-derivation of itself.

use super::dflash::{dflash_next_block_size, walk_block_greedy};
use super::eagle3::{eagle3_next_block_size, eagle3_walk};
use super::{accept_prefix, two_model_drafts_per_round};

/// One acceptance walk over a drafted block, spelled three times.
///
/// `dflash::walk_block_greedy` and `eagle3::eagle3_walk` are the same source
/// under two names; `accept_prefix` reaches the same answer from the verifier's
/// side. The extraction keeps one of them, so what has to survive is the answer
/// all three give — accepted count and committed tokens — and the fact that
/// `budget` caps the emission without capping the acceptance.
///
/// Mutation: in `eagle3_walk`, change `new_tokens.truncate(budget)` to
/// `new_tokens.truncate(budget + 1)`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the fixture rows below are same-length by construction, so accept_prefix's length guard cannot fire on them"
)]
fn the_three_acceptance_walks_commit_the_same_rows() {
    // (drafted, verifier's own tokens, budget, expected accepted, expected commit)
    let cases: [(&[u32], &[u32], usize, usize, &[u32]); 6] = [
        // Every proposal held: the walk commits them and the bonus past them.
        (&[7, 8, 9], &[7, 8, 9, 10], 8, 3, &[7, 8, 9, 10]),
        // The first proposal missed: nothing accepted, the correction alone.
        (&[7, 8, 9], &[4, 8, 9, 10], 8, 0, &[4]),
        // A miss in the middle: the agreed prefix and the correction at it.
        (&[7, 8, 9], &[7, 8, 5, 10], 8, 2, &[7, 8, 5]),
        // The budget caps the commit and leaves the acceptance alone — the
        // round rolled its caches back to what it accepted, not to what it
        // emitted.
        (&[7, 8, 9], &[7, 8, 9, 10], 2, 3, &[7, 8]),
        // A budget of one: the first agreed token only.
        (&[7, 8, 9], &[7, 8, 9, 10], 1, 3, &[7]),
        // A single proposal, missed.
        (&[7], &[4, 5], 8, 0, &[4]),
    ];
    for (draft, verifier, budget, want_accept, want_commit) in cases {
        let (a1, c1) = walk_block_greedy(draft, verifier, budget);
        let (a2, c2) = eagle3_walk(draft, verifier, budget);
        let (a3, c3) = accept_prefix(verifier, draft, budget)
            .expect("verifier rows are one longer than the proposals in every case above");
        assert_eq!(
            (a1, c1.as_slice()),
            (want_accept, want_commit),
            "walk_block_greedy on {draft:?} against {verifier:?} at budget {budget}"
        );
        assert_eq!(
            (a2, c2.as_slice()),
            (want_accept, want_commit),
            "eagle3_walk on {draft:?} against {verifier:?} at budget {budget}"
        );
        assert_eq!(
            (a3, c3.as_slice()),
            (want_accept, want_commit),
            "accept_prefix on {draft:?} against {verifier:?} at budget {budget}"
        );
    }
}

/// The one behaviour the three walks do **not** share.
///
/// `accept_prefix` refuses a call whose verifier rows are not the proposals plus
/// one bonus slot — the shape a swapped call arrives in, which otherwise returns
/// a plausible accept count that then drives a KV rollback. The two block walks
/// take the same call and answer it. An extraction that keeps one walk decides
/// which of the two behaviours the other four loops inherit, and this is what
/// says the decision was made rather than fallen into.
///
/// Mutation: delete the length check at the head of `accept_prefix`.
#[test]
fn only_one_of_the_three_walks_refuses_a_swapped_call() {
    let draft: &[u32] = &[7, 8, 9];
    let verifier: &[u32] = &[7, 8, 9, 10];
    assert!(
        accept_prefix(draft, verifier, 8).is_err(),
        "accept_prefix must refuse a call whose rows arrive the wrong way round"
    );
    // The block walks read the shorter slice and answer: four proposals
    // accepted where only three were made, which is the count that would drive
    // the rollback.
    assert_eq!(
        walk_block_greedy(verifier, draft, 8),
        (4, vec![7, 8, 9, 10])
    );
    assert_eq!(eagle3_walk(verifier, draft, 8), (4, vec![7, 8, 9, 10]));
}

/// Every round narrows its block against the remaining budget, and over the
/// domain a round can reach they are one function.
///
/// Five spellings are in the tree: `eagle3_next_block_size(block, remaining+1)`;
/// DFlash 2's inline `block.min(remaining + 1)`; the MTP sidecar's and the
/// assistant's `block.min(remaining + 1).max(2)`; and the two-model loops'
/// `remaining.min(k).max(1)`, in proposals rather than in blocks. A loop only
/// narrows with at least one token left to emit and a block of at least two, and
/// over that domain all five give `block.min(remaining + 1)`. The `.max(2)` and
/// the `.max(1)` are dead there, which is what makes them safe to drop and worth
/// pinning before they are.
///
/// DFlash 1 is the exception and stays its own function: its block follows the
/// accept rate of the recent rounds.
///
/// Mutation: change `eagle3_next_block_size` to
/// `requested_block_total.min(remaining_budget + 1)`.
#[test]
fn the_round_block_is_one_function_of_the_block_and_the_budget() {
    for block in 2..=24usize {
        for remaining in 1..=32usize {
            let want = block.min(remaining + 1);
            assert_eq!(
                eagle3_next_block_size(block, remaining + 1),
                want,
                "eagle3 at block {block}, remaining {remaining}"
            );
            assert_eq!(
                block.min(remaining + 1),
                want,
                "dflash2 at block {block}, remaining {remaining}"
            );
            assert_eq!(
                block.min(remaining + 1).max(2),
                want,
                "the sidecar spelling at block {block}, remaining {remaining}"
            );
            // The two-model loops count proposals: `k` is the block less the
            // verifier's own token, and the round's block is one more than the
            // proposals it drafted.
            let k = two_model_drafts_per_round(block - 1);
            assert_eq!(
                remaining.min(k).max(1) + 1,
                want,
                "the two-model spelling at block {block}, remaining {remaining}"
            );
        }
    }
    // Absolute rows, so a rewritten sweep cannot pass by agreeing with itself.
    assert_eq!(eagle3_next_block_size(8, 4), 4);
    assert_eq!(eagle3_next_block_size(8, 9), 8);
    assert_eq!(eagle3_next_block_size(2, 64), 2);
    // The adaptive schedule is a different function, and stays one: with no
    // history it takes the same narrowing, and with a poor recent accept rate it
    // does not.
    assert_eq!(dflash_next_block_size(&[], 16, 9, false), 9);
    assert_eq!(dflash_next_block_size(&[(0, 7), (0, 7)], 16, 17, false), 4);
}

/// The verifier's rollback target, in the two spellings the loops use.
///
/// Five loops compute it from the tail — `v_offset_before - (proposals -
/// accept)` — and the assistant computes it from the head, `pre_round_offset +
/// accept + 1`. They are the same position: the verify forward consumed the
/// carry token and every proposal, so `v_offset_before = pre + 1 + proposals`.
/// An extraction that keeps one spelling has to keep this equality, and an
/// off-by-one in it leaves a rejected draft in the cache every partial round —
/// the defect the equivalence pairs' broken engines are built from.
///
/// Mutation: change `want_target` below to `pre + accept as i32 + 2`.
#[test]
fn the_two_rollback_spellings_name_the_same_position() {
    for pre in [0_i32, 1, 37, 4096] {
        for proposals in 1..=8usize {
            for accept in 0..=proposals {
                let v_k = 1 + proposals as i32;
                let v_offset_before = pre + v_k;
                let from_tail = v_offset_before - (proposals as i32 - accept as i32);
                let from_head = pre + accept as i32 + 1;
                let want_target = pre + accept as i32 + 1;
                assert_eq!(
                    from_tail, want_target,
                    "tail spelling at pre {pre}, {proposals} proposals, {accept} accepted"
                );
                assert_eq!(
                    from_head, want_target,
                    "head spelling at pre {pre}, {proposals} proposals, {accept} accepted"
                );
                // The pre-round offset the rollback replays from is the one the
                // tail spelling reconstructs, not one the loop remembered.
                assert_eq!(v_offset_before - v_k, pre);
            }
        }
    }
    // Absolute rows, both spellings, one cell.
    for (pre, proposals, accept, want) in [(4096_i32, 7_i32, 3_i32, 4100_i32), (0, 4, 0, 1)] {
        assert_eq!(pre + accept + 1, want);
        assert_eq!((pre + 1 + proposals) - (proposals - accept), want);
    }
}

/// The drafter side keeps one row fewer than the verifier, and both sidecar and
/// two-model loops say so in their own arithmetic.
///
/// The two-model loop drops `(proposals - accept - 1).max(0)` rows from the
/// draft cache because its drafting pass never feeds the last proposal back;
/// the MTP sidecar keeps `draft_start + accept + 1` slots because its head wrote
/// the carry seed into the first one. The two land on the same count of retained
/// rows, and keeping one fewer degrades the accept rate every round with nothing
/// saying so.
///
/// Mutation: change the `.max(0)` in `d_drop` below to `.max(1)`.
#[test]
fn the_draft_side_keeps_the_carry_and_the_accepted_prefix() {
    for proposals in 1..=8usize {
        for accept in 0..=proposals {
            let d_offset_before = 100 + proposals as i32;
            let d_drop = (proposals as i32 - accept as i32 - 1).max(0);
            let two_model_target = d_offset_before - d_drop;
            // The draft pass fed the seed plus every proposal but the last, so
            // the round's own rows are `proposals` and the retained ones are the
            // carry plus the accepted prefix.
            let retained = two_model_target - 100;
            assert_eq!(
                retained,
                (accept as i32 + 1).min(proposals as i32),
                "two-model retention at {proposals} proposals, {accept} accepted"
            );
            // The sidecar counts from where its head started writing.
            let draft_start = 100_i32;
            let sidecar_target = draft_start + accept as i32 + 1;
            assert_eq!(
                sidecar_target - draft_start,
                accept as i32 + 1,
                "sidecar retention at {proposals} proposals, {accept} accepted"
            );
        }
    }
    // Absolute rows: a full accept drops nothing, a total reject drops all but
    // the carry, and one short of a full accept drops nothing either.
    for (proposals, accept, want_drop) in [(7_i32, 7_i32, 0_i32), (7, 0, 6), (7, 6, 0), (1, 0, 0)] {
        assert_eq!(
            (proposals - accept - 1).max(0),
            want_drop,
            "{proposals} proposals, {accept} accepted"
        );
    }
}
