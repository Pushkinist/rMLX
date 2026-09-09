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
//! Two per-request facts belong in that list and are easy to leave out of it,
//! because neither is a token and neither is a count of one:
//!
//! - **Whether the request reported its verifier's resident KV at all.**
//!   `round_common::report_verifier_kv_bytes` is called on the normal exit and
//!   skipped on one EOS exit per loop — the seed EOS for the five sidecar
//!   loops, the in-round EOS for the two two-model ones. A loop that changed
//!   which exit it takes reports a different figure, or none, with every token
//!   identical. Nothing enforces this; pinned by review — a report added on a
//!   seed-EOS exit leaves every gate green.
//! - **`RoundStats::charged`**, which is `phases_charged()` in three loops and a
//!   literal `false` in four. It decides whether each round forces its carried
//!   arrays before its span closes, so it moves the phase timings and the work
//!   attributed to the drafter — and an extraction that hard-wires one value
//!   silently re-attributes four loops' time.
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
//!   not resolvable, and blind to **every sampled arm**. That is wider than the
//!   stochastic two-model loop: each sidecar loop's `VerifierDraw` draws from the
//!   verifier's post-sampling distribution above temperature 0, and none of those
//!   arms has a temperature-0 comparand either, because neither side is then a
//!   function of the model alone. The only coverage a sampled arm has is
//!   `make check-spec-sampling`, which reads that each driver takes the
//!   request's sampler and does not read whether the distribution is right, and
//!   `crates/rmlx-models/tests/spec_sampled_distribution.rs`, which reads that on
//!   one pair under `make gpu-test`.
//! - **The tests in this file** see the arithmetic and are blind to every
//!   forward, every cache and every drafter: they fix what the loops compute
//!   from counts, not what the model computes from weights. They are also blind
//!   to **which loop calls which helper**: every call site below is inside a
//!   round loop that needs a model, a device and a drafter to execute, so a
//!   call site wired to the wrong helper or the wrong argument passes every test
//!   in this crate. Measured, not assumed — `mtp_generate` rolling its verifier
//!   KV back one position short leaves the whole `rmlx-models` suite green.
//!
//! Two request shapes never reach any of this and the extraction must not make
//! either reachable: a request carrying a sampler constraint is refused at the
//! speculative entry (`spec_generate_greedy` returns `Error::Model`, and the
//! sidecar routes never build a constraint engine), and a prompt-cache hit
//! cannot occur because every loop allocates its own scratch cache stack per
//! request and never consults or publishes a prompt-cache slot. A shared round
//! loop that accepted a pre-filled cache, or that stopped refusing a
//! constraint, would open a path neither the pairs nor these tests can see.
//!
//! # What is pinned here, and why by identity
//!
//! Each test below fixes a piece that exists in more than one loop as its own
//! copy, at absolute values, so an extraction is judged against something that
//! can fail rather than against a re-derivation of itself.

use super::dflash::dflash_next_block_size;
use super::{
    accept_prefix, draft_rows_to_drop, rollback_target_from_head, rollback_target_from_tail,
    round_block, two_model_drafts_per_round, MAX_BLOCK_SIZE,
};

/// One acceptance walk over a drafted block, and the rows the three spellings
/// agreed on before it was one.
///
/// The two block walks are gone and [`accept_prefix`] is what every round loop
/// calls, so what these rows pin is the answer the collapse had to preserve —
/// accepted count and committed tokens — and the fact that `budget` caps the
/// emission without capping the acceptance. The empty-proposal row is the shape
/// the loops' own guard refuses before a walk ever sees it.
///
/// Mutation: change `new_tokens.truncate(budget)`'s equivalent in
/// [`accept_prefix`] — `if emit.len() < budget` — to `<= budget`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the fixture rows below are same-length by construction, so accept_prefix's length guard cannot fire on them"
)]
fn the_acceptance_walk_commits_the_rows_the_three_spellings_agreed_on() {
    // (drafted, verifier's own tokens, budget, expected accepted, expected commit)
    let cases: [(&[u32], &[u32], usize, usize, &[u32]); 4] = [
        // The budget caps the commit and leaves the acceptance alone — the
        // round rolled its caches back to what it accepted, not to what it
        // emitted. A budget of one is the narrowest a round can run at.
        (&[7, 8, 9], &[7, 8, 9, 10], 1, 3, &[7]),
        // A single proposal, missed.
        (&[7], &[4, 5], 8, 0, &[4]),
        // A single proposal, held.
        (&[7], &[7, 5], 8, 1, &[7, 5]),
        // No proposals at all: the shape the loops' empty-chain guard exists to
        // refuse before it reaches a walk. The walk answers it rather than
        // refusing, so the guard is the only thing standing between a broken
        // drafter and a round that silently emits one token.
        (&[], &[4], 8, 0, &[4]),
    ];
    for (draft, verifier, budget, want_accept, want_commit) in cases {
        let (accepted, commit) = accept_prefix(verifier, draft, budget)
            .expect("verifier rows are one longer than the proposals in every case above");
        assert_eq!(
            (accepted, commit.as_slice()),
            (want_accept, want_commit),
            "accept_prefix on {draft:?} against {verifier:?} at budget {budget}"
        );
    }
}

/// Every round narrows its block against the remaining budget, and over the
/// domain a round can reach that is one function.
///
/// [`round_block`] is now the only producer, called by the MTP sidecar, DFlash 2,
/// EAGLE-3 and the Gemma4 assistant. Three of those four spelled it with a
/// `.max(2)` that could not fire from inside a round — a loop only narrows with
/// at least one token left to emit, and every block resolver in the tree returns
/// at least 2 — and the shared producer keeps the floor anyway, for a caller
/// that arrives at `remaining` some other way. The two-model loops count
/// proposals rather than blocks and reach the same number one off, which is
/// checked here because it is the one place the two units meet.
///
/// DFlash 1 adapts on top of it rather than beside it: `dflash_next_block_size`
/// opens with [`round_block`] and then moves the result with the accept rate of
/// the recent rounds, so the narrowing has one producer and the schedule is the
/// only part that is DFlash 1's own.
///
/// Mutation: change `round_block` to `block_total.min(remaining)`.
#[test]
fn the_round_block_is_one_function_of_the_block_and_the_budget() {
    for block in 2..=24usize {
        for remaining in 1..=32usize {
            let want = round_block(block, remaining);
            assert!(
                (2..=block).contains(&want),
                "a round must run a block of at least two and never wider than \
                 the request: block {block}, remaining {remaining}, got {want}"
            );
            assert!(
                want <= remaining + 1,
                "a round cannot verify more positions than it may emit, plus its \
                 own token: block {block}, remaining {remaining}, got {want}"
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
    // Absolute rows.
    // A budget with nothing left to emit is outside the domain a round loop
    // reaches, and the floor is what answers there — a round of one drafts
    // nothing.
    assert_eq!(round_block(8, 0), 2);
    assert_eq!(round_block(8, 3), 4);
    assert_eq!(round_block(8, 8), 8);
    assert_eq!(round_block(8, 64), 8);
    assert_eq!(round_block(2, 64), 2);
    // The two-model draft count is clamped to what one verify forward scores.
    assert_eq!(two_model_drafts_per_round(4096), MAX_BLOCK_SIZE - 1);
    // The adaptive schedule is a different function, and stays one: with no
    // history it takes the same narrowing, and with a poor recent accept rate it
    // does not.
    assert_eq!(dflash_next_block_size(&[], 16, 8, false), 9);
    assert_eq!(dflash_next_block_size(&[(0, 7), (0, 7)], 16, 16, false), 4);
}

/// The verifier's rollback target, in the two spellings the loops use.
///
/// Five loops call [`rollback_target_from_tail`] and the assistant calls
/// [`rollback_target_from_head`], because it reads its offset before the verify
/// forward rather than after. They must name the same position: the forward
/// consumed the carry token and every proposal, so `v_offset_before = pre + 1 +
/// proposals`. An off-by-one either way leaves a rejected draft in the cache
/// every partial round — the defect the equivalence pairs' broken engines are
/// built from.
///
/// Mutation: change `rollback_target_from_head` to `pre_round_offset + accept as
/// i32 + 2`, or `rollback_target_from_tail` to drop `proposals - accept + 1`.
#[test]
fn the_two_rollback_spellings_name_the_same_position() {
    for pre in [0_i32, 1, 37, 4096] {
        for proposals in 1..=8usize {
            for accept in 0..=proposals {
                let v_k = 1 + proposals as i32;
                let v_offset_before = pre + v_k;
                let from_tail = rollback_target_from_tail(v_offset_before, proposals, accept);
                let from_head = rollback_target_from_head(pre, accept);
                assert_eq!(
                    from_tail, from_head,
                    "the two spellings part at pre {pre}, {proposals} proposals, {accept} accepted"
                );
                // The retained positions are the carry and the accepted prefix,
                // and nothing else: the correction the round emits past them is
                // a prediction the verifier has not processed.
                assert_eq!(
                    from_tail - pre,
                    accept as i32 + 1,
                    "retained rows at pre {pre}, {proposals} proposals, {accept} accepted"
                );
                // A full acceptance drops nothing.
                if accept == proposals {
                    assert_eq!(from_tail, v_offset_before);
                }
            }
        }
    }
    // Absolute rows, both spellings, one cell each.
    assert_eq!(rollback_target_from_tail(4104, 7, 3), 4100);
    assert_eq!(rollback_target_from_head(4096, 3), 4100);
    assert_eq!(rollback_target_from_tail(5, 4, 0), 1);
    assert_eq!(rollback_target_from_head(0, 0), 1);
}

/// The drafter side keeps one row more than the verifier drops.
///
/// The two-model loop calls [`draft_rows_to_drop`] because its drafting pass
/// never feeds the last proposal back, so the draft cache is one row shorter
/// than the verifier's for the same round. Dropping one too many discards the
/// last accepted draft's K/V every partial round, which shows up only as a
/// falling accept rate.
///
/// Mutation: change `draft_rows_to_drop` to `(proposals - accept).max(0)`.
#[test]
fn the_draft_side_keeps_the_carry_and_the_accepted_prefix() {
    for proposals in 1..=8usize {
        for accept in 0..=proposals {
            let d_offset_before = 100 + proposals as i32;
            let retained = d_offset_before - draft_rows_to_drop(proposals, accept) - 100;
            assert_eq!(
                retained,
                (accept as i32 + 1).min(proposals as i32),
                "two-model retention at {proposals} proposals, {accept} accepted"
            );
        }
    }
    // Absolute rows: a full accept drops nothing, one short of a full accept
    // drops nothing either, and a total reject drops all but the carry.
    assert_eq!(draft_rows_to_drop(7, 7), 0);
    assert_eq!(draft_rows_to_drop(7, 6), 0);
    assert_eq!(draft_rows_to_drop(7, 0), 6);
    assert_eq!(draft_rows_to_drop(1, 0), 0);
}
