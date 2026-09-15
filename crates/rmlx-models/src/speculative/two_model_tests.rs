//! What this drafter owns: six wirings pinned by text, and the one acceptance
//! rule that is not the shared one, driven.
//!
//! The round loop is `round_loop.rs`'s and its own gates read it. What moved out
//! of the two-model loop bodies and into this file is wiring — an order, a
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
//! 6. **Where the round's tokens come from.** Every one of them — the verifier's
//!    tokens, the verifier's distributions, the drafted proposals and the
//!    acceptance coins — is drawn through the round's own `VerifierDraw`, and
//!    this drafter builds no read-back and no generator of its own. That is the
//!    request's one seeded stream: a second generator seeded from the same value
//!    is a second stream correlated with the first, and both look reproducible.
//!
//! The acceptance rule is not in that list, because it is not a wiring: it is
//! arithmetic over two distributions and a seeded draw, and
//! [`the_stochastic_rule_commits_the_prefix_and_one_token_the_verifier_stands_behind`]
//! below drives it on the CPU rather than reading it.
//!
//! **It reads text and is blind past that.** A call at the position below inside
//! a branch that never runs reads identical, and so does one whose arguments are
//! wrong. Driving the real methods would need two models, which is a model load
//! — the same reason `round_loop.rs` carries no unit test. What sees the rest is
//! the equivalence pair `the_two_model_round_loop_reproduces_plain_greedy` and
//! the per-round stream it writes.

use super::stochastic_prefix;
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

/// Every token of a round comes off the round's own draw, and this drafter
/// reads its logits back no other way.
///
/// Four reads, two rules: the greedy rule takes the verifier's own token at each
/// position, the stochastic one takes the verifier's whole distribution there
/// and the coin it tests each proposal with, and its drafting pass draws the
/// proposals through the same stream. What is refused is a second read-back —
/// a raw `argmax` beside the draw — and a second generator, which is what a
/// drafter seeding its own `Pcg32` from the request's seed would be: two
/// correlated streams, each looking reproducible on its own.
///
/// At temperature 0 the draw *is* the device argmax, and the greedy rule runs at
/// temperature 0 alone — the entry guard routes every sampled request to the
/// stochastic rule — so a raw argmax read-back there decodes the same tokens
/// today and nothing at runtime parts from it. `make check-spec-sampling` reads
/// the loops and the entries, not a drafter's `verify`, and the one pair
/// `crates/rmlx-models/tests/spec_sampled_distribution.rs` drives is the
/// assistant's. This reading is what stands in their place, and what it is for
/// is the day that routing changes.
///
/// Mutation: read the tokens back through `argmax(&v_logits, -1, device)` and
/// `argmax_tokens`; move a draw into `propose`; seed a `Pcg32` here and test the
/// acceptance coins against it.
#[test]
fn every_token_of_a_round_comes_off_the_rounds_own_draw() {
    let draws = lines_in_fns(ROUND_SRC, "ctx.draw");
    let read: Vec<(&str, &str)> = draws.iter().map(|(l, o)| (*l, o.as_str())).collect();
    assert_eq!(
        read,
        vec![
            ("&mut ctx.draw,", "propose"),
            (
                "let v_tokens = ctx.draw.block_tokens(&v_logits, fed.len(), device)?;",
                "verify"
            ),
            (
                "let p = ctx.draw.block_distributions(&v_logits, fed.len(), device)?;",
                "verify"
            ),
            (
                "let (accept, commit) = stochastic_prefix(&p, q, proposed, ctx.draw.rng())?;",
                "verify"
            ),
        ],
        "the round draws its proposals in `propose` and reads the verifier and its \
         coins in `verify`, all four through the draw the loop built"
    );
    let raw: Vec<&str> = ROUND_SRC
        .lines()
        .map(str::trim)
        .filter(|l| is_code(l) && (l.contains("argmax") || l.contains("Pcg32::new(")))
        .collect();
    assert!(
        raw.is_empty(),
        "a drafter that argmaxes the verifier's logits itself has a second read-back \
         beside the request's draw, and one that builds a draw or seeds a generator \
         of its own has a second stream beside the request's; this one has {raw:?}"
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

/// The stochastic rule commits the accepted prefix and exactly one token the
/// verifier stands behind.
///
/// Driven rather than read, because it is arithmetic and it is the one thing in
/// this drafter that has no shared body to be judged against: `run_rounds` calls
/// it through `verify` and nothing else in the tree does. Four rows, each
/// separating one way it can be wrong from the way it is right.
///
/// - **Every proposal held.** `p == q` makes every ratio one and every coin
///   accept, so the round commits the proposals plus the verifier's own draw
///   past the last of them — and that draw is taken from the **last**
///   distribution, which is what the point mass in the bonus slot reads.
/// - **The first proposal missed.** The verifier gives it no mass at all, so it
///   is rejected whatever the coin reads, and the round commits one token: the
///   correction. A bonus emitted beside it would be a second token the verifier
///   never stood behind, and the rollback keeps the KV of only one.
/// - **The correction is the residual's, not the verifier's.** `normalize((p -
///   q)+)` is a point mass on an id the verifier's own distribution gives almost
///   nothing to, so a correction drawn from `p` instead reads as the other id
///   with probability 0.98 — this row is what parts them.
/// - **The lengths carry the meaning.** A block is the proposals plus one bonus
///   slot and one drawn distribution per proposal; anything else is a swapped or
///   short argument list, which would return a plausible accept count that then
///   drives a KV rollback.
///
/// Mutation: emit the bonus on a partial accept; draw the correction from `p`;
/// draw the bonus from `p[0]`; count a rejection as an acceptance; drop the
/// length refusal.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: every row below is well-formed by construction, so the rule's own length refusal cannot fire on the four that must pass"
)]
fn the_stochastic_rule_commits_the_prefix_and_one_token_the_verifier_stands_behind() {
    // A vocabulary of four. `spread` puts most of the mass on one id and the
    // rest over the others, so no id is impossible and every ratio is finite;
    // `point` is the shape a draw cannot miss.
    let spread = |at: usize| -> Vec<f32> {
        (0..4)
            .map(|i| if i == at { 0.9 } else { 0.033_333_33 })
            .collect()
    };
    let point =
        |at: usize| -> Vec<f32> { (0..4).map(|i| if i == at { 1.0 } else { 0.0 }).collect() };
    // The verifier gives id 0 no mass and the proposal did, so this position is
    // rejected whatever the coin reads. The residual `(p - q)+` is then a point
    // mass on id 2, while the verifier's own distribution puts 0.85 on id 1 —
    // which is what separates a correction drawn from the residual from one
    // drawn from `p`.
    let p_rejects: Vec<f32> = vec![0.0, 0.85, 0.15, 0.0];
    let q_proposed: Vec<f32> = vec![0.05, 0.9, 0.0, 0.05];
    let mut rng = crate::sampler::Pcg32::new(11);

    // Every proposal held: `p == q` at both positions makes every ratio one, so
    // the round commits the proposals and the verifier's own draw past the last
    // of them — taken from the last distribution, which the point mass reads.
    let (accept, commit) = stochastic_prefix(
        &[spread(0), spread(1), point(3)],
        &[spread(0), spread(1)],
        &[0, 1],
        &mut rng,
    )
    .expect("two proposals, two drawn distributions and a bonus slot");
    assert_eq!(
        (accept, commit.as_slice()),
        (2, [0, 1, 3].as_slice()),
        "a round that accepted every proposal commits them and the verifier's own \
         draw past the last one, taken from the last distribution"
    );

    // The first proposal missed: nothing is accepted and the round commits one
    // token, the correction.
    let (accept, commit) = stochastic_prefix(
        &[p_rejects.clone(), spread(1), point(3)],
        &[q_proposed.clone(), spread(1)],
        &[0, 1],
        &mut rng,
    )
    .expect("two proposals, two drawn distributions and a bonus slot");
    assert_eq!(
        (accept, commit.as_slice()),
        (0, [2].as_slice()),
        "a rejected proposal commits the residual's correction and nothing else — a \
         bonus beside it is a second token the verifier never stood behind, and a \
         correction of 1 is one drawn from the verifier's own distribution"
    );

    // One proposal held and the second missed: the prefix is one long, committed
    // as proposed, and the correction closes the round.
    let (accept, commit) = stochastic_prefix(
        &[spread(0), p_rejects, point(3)],
        &[spread(0), q_proposed],
        &[0, 0],
        &mut rng,
    )
    .expect("two proposals, two drawn distributions and a bonus slot");
    assert_eq!(
        (accept, commit.as_slice()),
        (1, [0, 2].as_slice()),
        "the accepted prefix is committed as proposed and the correction closes the \
         round"
    );

    // The lengths are the argument order, and this is what refuses a swapped or
    // short list before an accept count reaches a rollback.
    assert!(
        stochastic_prefix(
            &[spread(0), spread(1)],
            &[spread(0), spread(1)],
            &[0, 1],
            &mut rng
        )
        .is_err(),
        "a block with no bonus slot is a verified block one position short"
    );
    assert!(
        stochastic_prefix(
            &[spread(0), spread(1), point(3)],
            &[spread(0)],
            &[0, 1],
            &mut rng
        )
        .is_err(),
        "a proposal with no distribution behind it is a proposal this rule cannot test"
    );
}
