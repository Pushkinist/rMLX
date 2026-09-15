//! Four wirings this drafter owns, pinned by text.
//!
//! The round loop is `round_loop.rs`'s and its own gates read it. What moved out
//! of the EAGLE-3 loop body and into this file is wiring — an order and a
//! choice, not an arithmetic — and nothing else in `make ci` sees any of it:
//!
//! 1. **Which read-back the verify pass takes, and the branch that takes it.**
//!    The reduced one is an argmax over a subset of the verifier's row and is
//!    sound at temperature 0 only, so the request's own draw decides it beside
//!    the drafter's offer. Drop the `!` on the decision and a sampled request
//!    takes a reduction its distribution cannot survive, with fluent text on the
//!    other side; negate the branch that consumes the decision and the same
//!    thing happens with the decision itself still reading correctly.
//! 2. **The restricted prefix the loop attributes tokens from.** It is the
//!    round's acceptance under that same condition and zero otherwise. Report it
//!    unconditionally and a full-vocabulary request claims a restriction it did
//!    not take; report zero always and the boundary rule in
//!    `docs/SPEC_ANSWER_EQUIVALENCE.md` waives nothing it should.
//! 3. **Where the re-run's correction comes from.** The round decides the next
//!    round's carry and hands it over; this drafter reads it and derives
//!    nothing. Derived a second time off the same `Verdict`, a drift re-runs the
//!    drafter over a correction the verifier is not scoring against — which
//!    moves the accept rate and nothing else, so no pair and no pinned cell can
//!    see it.
//! 4. **The drafter-side rollback and the target it reports.** The cache is
//!    rolled back by re-running it, exactly once per round, and the round line's
//!    target is read back off the cache afterwards — computing it instead makes
//!    the line restate the verifier's own number and stop being the cross-check
//!    the round calls it.
//!
//! **It reads text and is blind past that.** A call at the position below inside
//! a branch that never runs reads identical, and so does one whose arguments are
//! wrong. Driving the real methods would need a verifier, which is a model load
//! — the same reason `round_loop.rs` carries no unit test. What sees the rest is
//! the equivalence pair `the_restricted_vocab_round_loop_reproduces_plain_greedy`
//! and the per-round stream it writes.

use crate::speculative::text_scan::{is_code, lines_in_fns};

/// The drafter's own source, read as text.
const ROUND_SRC: &str = include_str!("round.rs");

/// The request's draw decides the read-back beside the drafter's offer, and the
/// branch that consumes that decision reads it in the same sense.
///
/// Two readings, because either alone leaves a hole. The declaration alone is
/// blind to a negated branch, which takes the reduced read-back on exactly the
/// requests that may not have it while the decision line still reads correctly;
/// the branch alone is blind to a decision that stopped consulting the draw.
///
/// Mutation: drop the `!` in the decision; drop its `&& !ctx.draw.sampling()`
/// term; negate the branch — `if !self.restricted_read_back`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertions above establish exactly one element each, so the reads cannot fail"
)]
fn the_request_draw_decides_the_read_back_and_the_branch_reads_it() {
    let reads = lines_in_fns(ROUND_SRC, "hot_path_active()");
    assert_eq!(
        reads.len(),
        1,
        "the read-back is decided once per request and this drafter decides it {} \
         time(s): {reads:?}",
        reads.len()
    );
    let (line, owner) = reads.first().expect("one decision, asserted above");
    assert_eq!(
        *owner, "prefill",
        "the decision is the request's and is taken before its first round, and \
         this one is taken in `{owner}` — the loop's own per-request line reports \
         what `prefill` declared"
    );
    assert_eq!(
        *line,
        "self.restricted_read_back = self.drafter.hot_path_active() && !ctx.draw.sampling();",
        "a sampled request cannot take the reduced read-back, and this request \
         decides it by `{line}` — a dropped negation hands a sampled request an \
         argmax over a subset of the row, and the answer is still fluent"
    );
    let branches = lines_in_fns(ROUND_SRC, "= if self.restricted_read_back {");
    assert_eq!(
        branches.len(),
        1,
        "one branch consumes the decision and this drafter has {}: {branches:?} — a \
         negated branch runs the reduced read-back on the requests that may not \
         have it, with the decision line above still reading correctly",
        branches.len()
    );
    let (line, owner) = branches.first().expect("one branch, asserted above");
    assert_eq!(
        *owner, "verify",
        "the branch is the verify pass's and this one sits in `{owner}`"
    );
    assert_eq!(
        *line,
        "let (v_tokens, v_hidden, scored_over_reduced_vocab) = if self.restricted_read_back {",
        "the arm that runs is what yields the round's reduced-prefix flag, and this \
         branch reads `{line}` — a flag produced beside the branch rather than by \
         it can name a read-back the round did not take"
    );
    // The two arm literals. They are free constants: the branch decides which
    // one is returned and neither is derived from anything, so a flipped one
    // reports a read-back its own arm did not take while both readings above
    // still pass. `guard_restricted_prefix` refuses that at runtime on any
    // request that runs the arm; this sees it on the arm no pair here executes,
    // because every EAGLE-3 pair in the tree runs at temperature 0 with the
    // reduced ids present and never takes the full-vocabulary branch.
    let arms: Vec<&str> = ROUND_SRC
        .lines()
        .map(str::trim)
        .filter(|l| is_code(l) && l.starts_with("(v_tokens, v_hidden, "))
        .collect();
    assert_eq!(
        arms,
        vec!["(v_tokens, v_hidden, true)", "(v_tokens, v_hidden, false)"],
        "the reduced arm yields `true` and the full-vocabulary arm `false`, in that \
         order, and this drafter yields {arms:?}"
    );
}

/// The restricted prefix the loop attributes from is that same condition's
/// acceptance, and nothing else.
///
/// Mutation: report `accept` unconditionally, or `0` unconditionally, or the
/// committed count instead of the acceptance.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly one element, so the read cannot fail"
)]
fn the_restricted_prefix_is_the_acceptance_of_a_restricted_round() {
    let writes = lines_in_fns(ROUND_SRC, "restricted:");
    assert_eq!(
        writes.len(),
        1,
        "the round states its restricted prefix once and this drafter states it {} \
         time(s): {writes:?}",
        writes.len()
    );
    let (line, owner) = writes.first().expect("one statement, asserted above");
    assert_eq!(
        *owner, "verify",
        "the prefix is known where the acceptance is, and this one is stated in \
         `{owner}`"
    );
    assert_eq!(
        *line, "restricted: if scored_over_reduced_vocab { accept } else { 0 },",
        "the prefix is the acceptance of a round that took the reduced read-back \
         and zero otherwise, and this round states `{line}` — either constant \
         makes the answer-equivalence boundary waive the wrong positions"
    );
}

/// The drafter's cache is rolled back by one re-run per round, in `rollback`,
/// and the span's target is read back off the cache rather than computed.
///
/// Mutation: call `accept_and_reseed` twice, or from `verify`; report
/// `outcome.verifier_target` or `d_offset_before + accept + 1` as the target.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertions above establish exactly one element each, so the reads cannot fail"
)]
fn the_drafter_rolls_back_by_re_running_and_reads_its_target_back() {
    let reseeds = lines_in_fns(ROUND_SRC, ".accept_and_reseed(");
    assert_eq!(
        reseeds.len(),
        1,
        "the drafter re-runs its cache once per round and this one re-runs it {} \
         time(s): {reseeds:?} — a second call re-runs the accepted prefix on top \
         of itself, and a lost one leaves the cache holding the rejected tail",
        reseeds.len()
    );
    let (_, owner) = reseeds.first().expect("one re-run, asserted above");
    assert_eq!(
        *owner, "rollback",
        "the re-run is the round's drafter-side rollback and this one sits in \
         `{owner}` — the loop times its rollback across that call, and a re-run \
         inside `verify` runs before the round's acceptance is committed"
    );
    let spans = lines_in_fns(ROUND_SRC, "CacheSpan {");
    assert_eq!(
        spans.len(),
        1,
        "the round reports one drafter-side cache span and this drafter builds {}: \
         {spans:?}",
        spans.len()
    );
    let (_, owner) = spans.first().expect("one span, asserted above");
    assert_eq!(
        *owner, "rollback",
        "the span says where the rollback left the cache and this one is built in \
         `{owner}`"
    );
    let targets = lines_in_fns(ROUND_SRC, "target: self.drafter.cache_offset(),");
    assert_eq!(
        targets.len(),
        1,
        "the span's target is read back off the cache the re-run left, and this \
         drafter reads it back {} time(s): {targets:?} — a computed target \
         restates the verifier's own number and stops being a cross-check on it",
        targets.len()
    );
    let (_, owner) = targets.first().expect("one target, asserted above");
    assert_eq!(
        *owner, "rollback",
        "the target is read back where the rollback left the cache and this one is \
         read in `{owner}`"
    );
}

/// The correction the re-run conditions on is the round's carry, read off the
/// outcome and derived nowhere in this drafter.
///
/// The loop takes the verifier's own token at the accepted position off the
/// commit and hands it over on `RoundOutcome`. A drafter that reads the same
/// `Verdict` for itself is a second producer of one token: the two agree until
/// one of them changes, and then the re-run conditions on a token the next
/// round's verify input does not carry. Every other observable is blind to that
/// — the answer is the verifier's own token either way, and the round line
/// carries no correction — so the accept rate is all that moves.
///
/// `carry_tok` is not that second producer and stays: it is the token `propose`
/// was handed, copied for the per-position trace, and the trace names the round
/// it opened rather than the one after it.
///
/// Mutation: `verdict.commit.last().copied().unwrap_or(self.carry_tok)` in
/// place of `outcome.carry`; `verdict.commit.first()`, which no other reading
/// here parts from.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly one element, so the read cannot fail"
)]
fn the_re_runs_correction_is_read_off_the_round_and_not_derived_again() {
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
        "the carry is read where the re-run conditions on it and this one is read in \
         `{owner}`"
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
