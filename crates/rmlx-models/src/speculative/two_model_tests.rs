//! Six wirings this drafter owns, pinned by text.
//!
//! The round loop is `round_loop.rs`'s and its own gates read it. What moved out
//! of the two-model greedy loop body and into this file is wiring — an order, a
//! condition and a read — and nothing else in `make ci` sees any of it:
//!
//! 1. **What the drafting pass tapes.** The pass feeds its seed and every
//!    proposal but the last, so that is the prefix its rollback keeps and the
//!    length the recurrent refold refuses a tape for. Tape the whole proposal
//!    chain instead and the refold is refused on every recurrent pair; tape only
//!    the seed and it keeps a prefix one round too short, silently, on a pair
//!    that has no recurrent state at all.
//! 2. **The full-accept resync.** A round that accepted every proposal left the
//!    draft cache one token behind the verifier's, so the next pass feeds that
//!    token ahead of the correction. Drop the condition and every round feeds a
//!    token the cache already holds; drop the prepend and the draft model
//!    silently skips a position of its own context on the rounds it did best on.
//! 3. **The drafter-side rollback and the target it reports.** The cache is
//!    rolled back once per round, in `rollback`, and the round line's target is
//!    read back off the cache afterwards — computing it instead makes the line
//!    restate this side's own arithmetic and stop being the cross-check the
//!    round calls it.
//! 4. **Where the next round's carry comes from.** The round decides it and
//!    hands it over; this drafter reads it and derives nothing. Derived a second
//!    time off the same `Verdict`, a drift feeds the draft model a seed the
//!    verifier is not scoring against — which moves the accept rate and nothing
//!    else, so no pair and no pinned cell can see it.
//! 5. **Which read of the draft cache the round line's `before` is.** It is the
//!    one taken before the rollback; a second read taken after it reports the
//!    span as empty on every partial round, and the target beside it still
//!    agrees with the verifier's.
//! 6. **Where the round's tokens come from.** They are drawn through the round's
//!    own `VerifierDraw`, and this drafter builds no read-back of its own. That
//!    one is not merely unseen by `make ci` — it is unseen by everything, since
//!    this path runs at temperature 0 alone, where the draw is the argmax it
//!    would be replaced by.
//!
//! **It reads text and is blind past that.** A call at the position below inside
//! a branch that never runs reads identical, and so does one whose arguments are
//! wrong. Driving the real methods would need two models, which is a model load
//! — the same reason `round_loop.rs` carries no unit test. What sees the rest is
//! the equivalence pair `the_two_model_round_loop_reproduces_plain_greedy` and
//! the per-round stream it writes.

use crate::speculative::text_scan::{is_code, lines_in_fns};

/// The drafter's own source, read as text.
const ROUND_SRC: &str = include_str!("two_model.rs");

/// The drafting pass tapes what it fed, which is its seed and every proposal but
/// the last.
///
/// Mutation: tape `&proposed`; tape `&[]`; build the tape in `rollback`, where
/// the seed has already moved on to the next round.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly one element, so the read cannot fail"
)]
fn the_drafting_pass_tapes_its_seed_and_every_proposal_but_the_last() {
    let tapes = lines_in_fns(ROUND_SRC, "fill_fed(");
    assert_eq!(
        tapes.len(),
        1,
        "the round tapes what its drafting pass fed once and this drafter tapes it {} \
         time(s): {tapes:?}",
        tapes.len()
    );
    let (_, owner) = tapes.first().expect("one tape, asserted above");
    assert_eq!(
        *owner, "propose",
        "the tape is built where the pass ran and this one is built in `{owner}` — the \
         seed it opens with is overwritten by the resync, so a tape built after that \
         describes the next round"
    );
    let head: Vec<&str> = ROUND_SRC
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("proposed.split_last()"))
        .collect();
    assert_eq!(
        head,
        vec!["proposed.split_last().map_or(&[], |(_, head)| head),"],
        "the pass never feeds its last output back, so the tape holds every proposal \
         but the last and this drafter tapes {head:?} — a tape of the whole chain is \
         refused by the recurrent refold, and one of the seed alone is not"
    );
}

/// The next drafting pass opens on the last proposal only when the round
/// accepted every one of them.
///
/// Mutation: drop the condition; compare against `verdict.commit.len()`; push
/// the correction before the proposal rather than after it.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertions above establish exactly one element each, so the reads cannot fail"
)]
fn the_resync_prepends_the_last_proposal_on_a_full_acceptance_alone() {
    let guards = lines_in_fns(ROUND_SRC, "if verdict.accept ==");
    assert_eq!(
        guards.len(),
        1,
        "the resync is decided once and this drafter decides it {} time(s): {guards:?}",
        guards.len()
    );
    let (line, owner) = guards.first().expect("one decision, asserted above");
    assert_eq!(
        *owner, "rollback",
        "the resync is a function of the round's acceptance and this one is decided in \
         `{owner}`"
    );
    assert_eq!(
        *line, "if verdict.accept == self.proposals {",
        "a round that accepted every proposal is the one that left the draft cache a \
         token behind, and this drafter decides it by `{line}` — the committed count in \
         its place resyncs on a budget-cut round that did not accept everything"
    );
    let resync: Vec<&str> = ROUND_SRC
        .lines()
        .map(str::trim)
        .filter(|l| l.starts_with("self.seed."))
        .collect();
    assert_eq!(
        resync,
        vec![
            "self.seed.clear();",
            "self.seed.extend(self.last_proposal);",
            "self.seed.push(correction);",
        ],
        "the pass feeds the held-back proposal and then the correction, in that order, \
         and this drafter writes {resync:?} — a reversed pair feeds the draft model two \
         tokens out of order and the answer is still the verifier's"
    );
}

/// The draft model's caches are rolled back once per round, in `rollback`, and
/// the span's target is read back off them rather than computed.
///
/// Mutation: roll back from `verify`; roll back twice; report
/// `outcome.verifier_target`, or the figure the rollback was handed, as the
/// target.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertions above establish exactly one element each, so the reads cannot fail"
)]
fn the_drafter_rolls_its_own_caches_back_and_reads_its_target_back() {
    let rollbacks = lines_in_fns(ROUND_SRC, "rollback_round(");
    assert_eq!(
        rollbacks.len(),
        1,
        "the draft side rolls back once per round and this drafter rolls back {} \
         time(s): {rollbacks:?} — a second call drops the rejected tail twice, and a \
         lost one leaves the draft cache holding proposals the verifier refused",
        rollbacks.len()
    );
    let (_, owner) = rollbacks.first().expect("one rollback, asserted above");
    assert_eq!(
        *owner, "rollback",
        "the draft-side rollback is the round's and this one sits in `{owner}` — the \
         loop times its rollback across the call, and one inside `verify` runs before \
         the round's acceptance is committed"
    );
    let targets = lines_in_fns(ROUND_SRC, "target: self.cache_offset(),");
    assert_eq!(
        targets.len(),
        1,
        "the span's target is read back off the cache the rollback left, and this \
         drafter reads it back {} time(s): {targets:?} — a computed target restates \
         this side's own arithmetic and stops being a cross-check on the verifier's",
        targets.len()
    );
    let (_, owner) = targets.first().expect("one target, asserted above");
    assert_eq!(
        *owner, "rollback",
        "the target is read back where the rollback left the cache and this one is read \
         in `{owner}`"
    );
}

/// The verify pass draws the verifier's tokens through the round's own draw,
/// and this drafter reads its logits back no other way.
///
/// At temperature 0 the draw *is* the device argmax, and this path runs at
/// temperature 0 alone — the entry guard routes every sampled request to the
/// stochastic loop — so a raw argmax read-back here decodes the same tokens
/// today and nothing at runtime parts from it. `make check-spec-sampling` reads
/// the loops and the entries, not a drafter's `verify`, and the one pair
/// `crates/rmlx-models/tests/spec_sampled_distribution.rs` drives is the
/// assistant's. This reading is what stands in their place, and what it is for
/// is the day that routing changes.
///
/// Mutation: read the tokens back through `argmax(&v_logits, -1, device)` and
/// `argmax_tokens`; move the draw into `propose`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly one element, so the read cannot fail"
)]
fn the_verify_pass_draws_through_the_rounds_own_draw() {
    let draws = lines_in_fns(ROUND_SRC, "ctx.draw");
    assert_eq!(
        draws.len(),
        1,
        "the round reads the verifier's own tokens once, through the draw the loop \
         built, and this drafter reads it {} time(s): {draws:?}",
        draws.len()
    );
    let (line, owner) = draws.first().expect("one draw, asserted above");
    assert_eq!(
        *owner, "verify",
        "the draw is the verify pass's and this one sits in `{owner}`"
    );
    assert_eq!(
        *line, "let v_tokens = ctx.draw.block_tokens(&v_logits, fed.len(), device)?;",
        "the round's tokens come off the request's draw at every verified position \
         and this drafter reads `{line}`"
    );
    let raw: Vec<&str> = ROUND_SRC
        .lines()
        .map(str::trim)
        .filter(|l| is_code(l) && l.contains("argmax"))
        .collect();
    assert!(
        raw.is_empty(),
        "a drafter that argmaxes the verifier's logits itself has a second read-back \
         beside the request's draw, and this one has {raw:?}"
    );
}

/// The next round's carry is the round's, read off the outcome and derived
/// nowhere in this drafter.
///
/// The loop takes the verifier's own token at the accepted position off the
/// commit and hands it over on `RoundOutcome`. A drafter that reads the same
/// `Verdict` for itself is a second producer of one token: the two agree until
/// one of them changes, and then the drafting pass opens on a token the verify
/// input does not carry. Every other observable is blind to that — the answer is
/// the verifier's argmax either way, and the round line carries no seed — so the
/// accept rate is all that moves.
///
/// Mutation: `verdict.commit.last().copied()` in place of `outcome.carry`;
/// `verdict.commit.first()`, which no other reading here parts from.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly one element, so the read cannot fail"
)]
fn the_next_rounds_carry_is_read_off_the_round_and_not_derived_again() {
    let reads = lines_in_fns(ROUND_SRC, "outcome.carry");
    assert_eq!(
        reads.len(),
        1,
        "the drafter reads the round's carry once and this one reads it {} time(s): \
         {reads:?}",
        reads.len()
    );
    let (line, owner) = reads.first().expect("one read, asserted above");
    assert_eq!(
        *owner, "rollback",
        "the carry is read where the next round's seed is built and this one is read \
         in `{owner}`"
    );
    assert_eq!(
        *line, "let correction = outcome.carry;",
        "the carry is taken whole off the outcome and this drafter takes `{line}`"
    );
    let derived: Vec<&str> = ROUND_SRC
        .lines()
        .map(str::trim)
        .filter(|l| is_code(l) && l.contains("verdict.commit"))
        .collect();
    assert!(
        derived.is_empty(),
        "a drafter that reads the round's commit for itself is a second producer of \
         the carry, and this one reads {derived:?}"
    );
}

/// The round line's `before` is the read taken before the rollback, and the
/// target the read taken after it.
///
/// Both are `self.cache_offset()` and the difference is only where they sit, so
/// a second read shadowing the first reports a span of zero on every partial
/// round while the target still agrees with the verifier's — and the answer, the
/// accept counters and every other field of the line are unmoved.
///
/// Mutation: rebind `before` immediately before the span is built; swap the two
/// reads.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly two elements, so the reads cannot fail"
)]
fn the_span_opens_on_the_read_taken_before_the_rollback() {
    let reads = lines_in_fns(ROUND_SRC, "self.cache_offset()");
    assert_eq!(
        reads.len(),
        2,
        "the round reads the draft cache twice, once each side of its rollback, and \
         this drafter reads it {} time(s): {reads:?}",
        reads.len()
    );
    assert!(
        reads.iter().all(|(_, owner)| owner == "rollback"),
        "both reads belong to the round's rollback and these sit in {reads:?}"
    );
    let (first, _) = reads.first().expect("two reads, asserted above");
    assert_eq!(
        *first, "let before = self.cache_offset();",
        "the first read is the one the span opens on, taken before anything is \
         dropped, and this drafter takes `{first}` — a read taken after the rollback \
         reports a span of nothing on every partial round"
    );
}
