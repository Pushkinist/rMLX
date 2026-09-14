//! How the adaptive schedule is wired, pinned by text.
//!
//! [`super::dflash_next_block_size`] is a pure function and is pinned by value
//! in `tests.rs` beside this file and in `round_skeleton_tests.rs`. What is not
//! a value is the **wiring**: the history the schedule adapts on is written at
//! the end of `verify`, where the round's acceptance is known, and read by
//! `block` at the head of the round after it. That order is the whole of what
//! this chunk moved out of a loop body, and nothing else in `make ci` sees it.
//!
//! Two mutations say why it needs a reading of its own. Swap the tuple —
//! `(proposals.len(), accept)` — and the schedule adapts on the drafted count
//! against the accepted one, so it grows where it should shrink. Move the push
//! into `condition` and a round that stopped on an EOS no longer records, and
//! the last round of every request is missing from the history. Both change
//! every later round's block, and so `num_draft` on every later round line —
//! which only the snapshot-gated equivalence pair and its pinned round stream
//! can see. Neither is a compile error and neither moves a count this crate's
//! own suite reads.
//!
//! **It reads text and is blind past that.** A push at the position below,
//! inside a branch that never runs, reads identical; so does a `block` that
//! names the history and then ignores what it read. What it holds is that there
//! is exactly one writer, that it is in `verify`, that its tuple is in the
//! order the schedule reads, and that `block` is what reads it. Driving the
//! real methods instead would need a verifier, which is a model load — the same
//! reason `round_loop.rs` carries no unit test.

use crate::speculative::text_scan::lines_in_fns;

/// The drafter's own source, read as text.
const ROUND_SRC: &str = include_str!("round.rs");

/// The schedule's history has one writer, it is `verify`, and its tuple is in
/// the order the schedule reads it.
///
/// Mutation: swap the tuple to `(proposals.len(), accept)`; move the statement
/// into `condition`; write it a second time anywhere.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly one element, so the read cannot fail"
)]
fn the_verify_writes_the_history_the_block_schedule_reads() {
    let writes = lines_in_fns(ROUND_SRC, "self.recent.push(");
    assert_eq!(
        writes.len(),
        1,
        "the schedule's history has one writer and this drafter states {} — a second \
         push double-counts a round and a lost one drops it, and every later round's \
         block moves either way: {writes:?}",
        writes.len()
    );
    let (line, owner) = writes.first().expect("one writer, asserted above");
    assert_eq!(
        *owner, "verify",
        "the history is written in `verify`, where the round's acceptance is known, \
         and this one is written in `{owner}` — a round that stops on an EOS never \
         reaches `condition`, so the last round of every request would go unrecorded"
    );
    assert_eq!(
        *line, "self.recent.push((accept, proposals.len()));",
        "the schedule reads `(accepted, drafted)` and this round writes `{line}` — a \
         swapped pair grows the block where the accept rate says to shrink it, with \
         every token still the verifier's own"
    );
}

/// `block` is what reads that history, through the schedule and not beside it.
///
/// Mutation: drop `&self.recent` from the call, or call the schedule from
/// somewhere else.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly one element, so the read cannot fail"
)]
fn the_block_method_reads_the_history_through_the_schedule() {
    let reads = lines_in_fns(ROUND_SRC, "dflash_next_block_size(");
    assert_eq!(
        reads.len(),
        1,
        "the schedule has one caller in this drafter and it states {}: {reads:?}",
        reads.len()
    );
    let (line, owner) = reads.first().expect("one caller, asserted above");
    assert_eq!(
        *owner, "block",
        "the schedule is the `block` method's body and this call sits in `{owner}`"
    );
    assert!(
        line.contains("&self.recent"),
        "the schedule adapts on the history `verify` wrote and this call hands it \
         `{line}` — a call that passes an empty slice runs the loop's own narrowing \
         under the schedule's name"
    );
}
