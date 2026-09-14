//! Three wirings this drafter owns, pinned by text.
//!
//! The round loop is `round_loop.rs`'s and its own gates read it. What moved out
//! of the EAGLE-3 loop body and into this file is wiring — an order and a
//! choice, not an arithmetic — and nothing else in `make ci` sees any of it:
//!
//! 1. **Which read-back the verify pass takes.** The restricted one is an argmax
//!    over a subset of the verifier's row and is sound at temperature 0 only, so
//!    the round's own draw decides it beside the drafter's offer. Drop the `!`
//!    and a sampled request takes a reduction its distribution cannot survive,
//!    with fluent text on the other side; drop the draw and every sampled
//!    request takes it.
//! 2. **The restricted prefix the loop attributes tokens from.** It is the
//!    round's acceptance under that same condition and zero otherwise. Report it
//!    unconditionally and a full-vocabulary request claims a restriction it did
//!    not take; report zero always and the boundary rule in
//!    `docs/SPEC_ANSWER_EQUIVALENCE.md` waives nothing it should.
//! 3. **The drafter-side rollback and the target it reports.** The cache is
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

/// The drafter's own source, read as text.
const ROUND_SRC: &str = include_str!("round.rs");

/// Whether a line is code rather than a whole-line comment.
///
/// The sentences above name these needles, and a reading that counted them
/// would find its own explanation. A trailing comment on a line of code is not
/// stripped: no needle here is one a caller would write at the end of a
/// statement.
fn is_code(line: &str) -> bool {
    !line.trim_start().starts_with("//")
}

/// Every code line carrying `needle`, each with the name of the `fn` it sits
/// in.
fn lines_in_fns<'a>(src: &'a str, needle: &str) -> Vec<(&'a str, String)> {
    let mut current = String::new();
    let mut found = Vec::new();
    for line in src.lines() {
        if let Some(rest) = line.trim_start().strip_prefix("fn ") {
            current = rest
                .split(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or_default()
                .to_owned();
        }
        if is_code(line) && line.contains(needle) {
            found.push((line.trim(), current.clone()));
        }
    }
    found
}

/// The round's draw decides the read-back beside the drafter's offer, and it
/// decides it in `verify`.
///
/// Mutation: drop the `!`; drop the `&& !ctx.draw.sampling()` term; move the
/// read into the entry, where the per-round `Verdict` cannot see it.
#[test]
#[allow(
    clippy::expect_used,
    reason = "the assertion above establishes exactly one element, so the read cannot fail"
)]
fn the_round_draw_decides_which_read_back_the_verify_takes() {
    let reads = lines_in_fns(ROUND_SRC, "hot_path_active()");
    assert_eq!(
        reads.len(),
        1,
        "the read-back is chosen once per round and this drafter chooses it {} \
         time(s): {reads:?}",
        reads.len()
    );
    let (line, owner) = reads.first().expect("one chooser, asserted above");
    assert_eq!(
        *owner, "verify",
        "the choice is the verify pass's, and this one is made in `{owner}` — the \
         loop hands the draw to `verify` and to nothing else"
    );
    assert_eq!(
        *line, "let restricted_read_back = self.drafter.hot_path_active() && !ctx.draw.sampling();",
        "a sampled request cannot take the reduced read-back, and this round \
         chooses it by `{line}` — a dropped negation hands a sampled request an \
         argmax over a subset of the row, and the answer is still fluent"
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
        *line, "restricted: if restricted_read_back { accept } else { 0 },",
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
