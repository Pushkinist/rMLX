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
//!   identical, and nothing at runtime sees it — a report added on a seed-EOS
//!   exit leaves every gate green. What states it is the seven-row
//!   [`DISPOSITIONS`] table below, one row per loop; what reads that table
//!   against today's source is
//!   [`every_loop_reports_the_verifiers_resident_kv_at_the_exit_it_declares`],
//!   positionally, so a call moved to the other exit fails rather than passing
//!   on an unchanged total. The reading is of text and is blind to what the call
//!   does once it is made.
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
//!   runs agreeing on every count can still be scoring different logits. Every
//!   loop emits one event of one shape, so a figure a loop does not carry is
//!   absent from its lines and is invisible to the stream for that loop.
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
use super::dflash::round::AdaptiveRound;
use super::dflash2::round::BlockRound;
use super::eagle3::round::Eagle3Round;
use super::gemma4_assistant::AssistantRound;
use super::mtp::SidecarRound;
use super::round_loop::{ReportSkippedBy, RoundDrafter, VerifierOffsetBasis};
use super::text_scan::{is_code, lines_in_fns};
use super::two_model::TwoModelRound;
use super::{
    accept_prefix, draft_rows_to_drop, guard_restricted_prefix, rollback_target_from_head,
    round_block, two_model_drafts_per_round, SpecLoop, MAX_BLOCK_SIZE,
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

/// The verifier's rollback target keeps the carry and the accepted prefix, and
/// names the position the post-forward read names.
///
/// [`rollback_target_from_head`] is the one producer: the shared loop reads the
/// offset before the verify forward and computes the target from it, whichever
/// read the drafter then reports on its round line. An off-by-one either way
/// leaves a rejected draft in the cache every partial round — the defect the
/// equivalence pairs' broken engines are built from.
///
/// The second reading is the one the round line rests on. Six of the seven
/// drafters report the offset read *after* the forward, and that number names
/// this same position counted back from the tail — `v_offset_before -
/// (proposals - accept)` — because the forward consumed the carry token and
/// every proposal, so `v_offset_before = pre + 1 + proposals`. The tail
/// arithmetic is written out here rather than called: no loop computes a target
/// that way any more, and a helper kept for one test to compare against itself
/// would be a re-derivation rather than a pin.
///
/// Mutation: change `rollback_target_from_head` to `pre_round_offset + accept as
/// i32 + 2`.
#[test]
fn the_rollback_target_retains_the_carry_and_the_accepted_prefix() {
    for pre in [0_i32, 1, 37, 4096] {
        for proposals in 1..=8usize {
            for accept in 0..=proposals {
                let v_k = 1 + proposals as i32;
                let v_offset_before = pre + v_k;
                let from_tail = v_offset_before - (proposals as i32 - accept as i32);
                let from_head = rollback_target_from_head(pre, accept);
                assert_eq!(
                    from_tail, from_head,
                    "the two readings part at pre {pre}, {proposals} proposals, {accept} accepted"
                );
                // The retained positions are the carry and the accepted prefix,
                // and nothing else: the correction the round emits past them is
                // a prediction the verifier has not processed.
                assert_eq!(
                    from_head - pre,
                    accept as i32 + 1,
                    "retained rows at pre {pre}, {proposals} proposals, {accept} accepted"
                );
                // A full acceptance drops nothing.
                if accept == proposals {
                    assert_eq!(from_head, v_offset_before);
                }
            }
        }
    }
    // Absolute rows, one cell each.
    assert_eq!(rollback_target_from_head(4096, 3), 4100);
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

/// The loop hands the round's reduced-vocabulary prefix to its guard, with the
/// request's own declaration beside it.
///
/// The call is what makes the guard reachable, and nothing else in this crate
/// reads it: a deleted call, or one handed the committed count where the bound
/// is the acceptance, leaves every other observable clean. Read as text off the
/// shared loop's own source, which is the file the table below already holds.
///
/// Mutation: delete the call; pass `verdict.commit.len()`; drop the
/// declaration argument.
#[test]
fn the_loop_bounds_every_rounds_reduced_prefix_before_it_emits() {
    let src = LOOP_SOURCES
        .iter()
        .find(|(f, _)| *f == SHARED_LOOP)
        .map(|(_, s)| *s)
        .unwrap_or_default();
    let call: Vec<&str> = src
        .lines()
        .map(str::trim)
        .filter(|l| is_code(l) && l.starts_with("guard_restricted_prefix("))
        .collect();
    assert_eq!(
        call.len(),
        1,
        "the shared loop bounds each round's reduced prefix once and it calls the \
         guard {} time(s): {call:?}",
        call.len()
    );
    let args: Vec<&str> = src
        .lines()
        .skip_while(|l| !l.trim().starts_with("guard_restricted_prefix("))
        .skip(1)
        .take_while(|l| !l.trim().starts_with(")?"))
        .map(str::trim)
        .collect();
    assert_eq!(
        args,
        vec![
            "cfg.loop_kind,",
            "rounds,",
            "restricted_read_back,",
            "verdict.restricted,",
            "verdict.accept,",
        ],
        "the guard reads the request's declaration, the round's prefix and the \
         round's acceptance — the committed count in the last place is a bound a \
         budget-cut round fails for no defect, and a dropped declaration stops the \
         guard seeing a prefix on a request that takes no reduced read-back"
    );
}

/// A round's reduced-vocabulary prefix is bounded by its acceptance, and by what
/// the request declared.
///
/// The correction past the accepted prefix is the verifier's own token over its
/// whole vocabulary, so a prefix that covers it makes the declared boundary in
/// `docs/SPEC_ANSWER_EQUIVALENCE.md` waive the one position that boundary
/// judges — and the answer, the round line and the accept counters all read
/// clean while it does. The bound is the **acceptance** and not the committed
/// count: a round the request's budget cut commits fewer tokens than it
/// accepted and its prefix stays where it was.
///
/// Mutation: bound it on the committed count; drop either refusal.
#[test]
fn the_reduced_vocabulary_prefix_is_bounded_by_the_acceptance() {
    // Every accepted position over the reduced vocabulary, and none of them.
    assert!(guard_restricted_prefix(SpecLoop::Eagle3, 1, true, 4, 4).is_ok());
    assert!(guard_restricted_prefix(SpecLoop::Eagle3, 1, true, 0, 4).is_ok());
    // A round of a budget that cut its commit to one: the acceptance is three
    // and the prefix is three, which a bound on the commit would refuse.
    assert!(guard_restricted_prefix(SpecLoop::Eagle3, 7, true, 3, 3).is_ok());
    // One past the acceptance is the correction.
    let refused = guard_restricted_prefix(SpecLoop::Eagle3, 7, true, 4, 3);
    assert!(
        refused.is_err(),
        "a prefix reaching past the acceptance covers the correction, which is the \
         verifier's own token over its whole vocabulary: {refused:?}"
    );
    // A request that scores every position over the verifier's whole vocabulary
    // has no reduced prefix to report, whatever its rounds accepted. This is the
    // arm a per-round flag made of free constants reaches: the acceptance bound
    // above passes it, and so does every text reading of the branch that
    // produced the flag.
    assert!(guard_restricted_prefix(SpecLoop::MtpSidecar, 1, false, 0, 4).is_ok());
    let undeclared = guard_restricted_prefix(SpecLoop::Eagle3, 2, false, 4, 4);
    assert!(
        undeclared.is_err(),
        "a request that declared no reduced read-back cannot have a round that took \
         one: {undeclared:?}"
    );
}

// ---------------------------------------------------------------------------
// What the seven loops declare at their edges
// ---------------------------------------------------------------------------

/// How a loop refuses a drafter that proposed nothing.
///
/// One arm, because one loop states one refusal. The two-model bodies refused
/// the verifier input the empty chain produced — the same test one statement
/// later, a two-model round's carry being always one token — and that spelling
/// went with the last of them.
#[derive(Debug, Clone, Copy)]
enum ChainRefusedBy {
    /// On the proposals themselves, before the verify forward.
    TheProposalChain,
}

/// The file the migrated loops share, and the order its own edge markers fall
/// in.
///
/// It is not read off a row: the shared loop carries both exits at once and
/// reads [`ReportSkippedBy`] at each of them, so its markers are the seed
/// exit's guard, report and return, the round loop, the empty-chain refusal, and
/// the tail's record, guard and report. What a migrated drafter declares is
/// checked against its constant instead, by the test below.
const SHARED_LOOP: &str = "round_loop.rs";
const SHARED_LOOP_PATTERN: &str = "SPEWGRIP";

/// What each round loop declares at its edges, and the file that holds it.
///
/// Five facts per loop: which exit skips the resident-KV report, how the loop
/// refuses an empty proposal chain, and which read of the verifier's offset its
/// round line reports. This table is the statement; the two tests below are
/// readings of it against today's source. It is data rather than seven assertions because it is what
/// survives the collapse: the resident-KV disposition is a [`RoundDrafter`]
/// declaration for a migrated loop, read here against its constant, and each
/// migration drops its own file from [`LOOP_SOURCES`] and names
/// [`SHARED_LOOP`] in its row instead, while these rows stay as they are. See
/// `docs/SPEC_ROUND_SKELETON.md`.
const DISPOSITIONS: [(
    SpecLoop,
    &str,
    ReportSkippedBy,
    ChainRefusedBy,
    VerifierOffsetBasis,
); 7] = [
    (
        SpecLoop::MtpSidecar,
        SHARED_LOOP,
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
        VerifierOffsetBasis::AfterTheForward,
    ),
    (
        SpecLoop::DFlash,
        SHARED_LOOP,
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
        VerifierOffsetBasis::AfterTheForward,
    ),
    (
        SpecLoop::DFlash2,
        SHARED_LOOP,
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
        VerifierOffsetBasis::AfterTheForward,
    ),
    (
        SpecLoop::Eagle3,
        SHARED_LOOP,
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
        VerifierOffsetBasis::AfterTheForward,
    ),
    (
        SpecLoop::MtpAssistant,
        SHARED_LOOP,
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
        VerifierOffsetBasis::BeforeTheForward,
    ),
    (
        SpecLoop::TwoModelGreedy,
        SHARED_LOOP,
        ReportSkippedBy::TheInRoundExit,
        // Its own body refused the verifier input the empty chain produced, one
        // statement past where the shared loop refuses the chain itself. The two
        // are the same test — a two-model round's carry is one token, so an
        // input under two positions is an empty chain and nothing else — and the
        // shared loop states one of them.
        ChainRefusedBy::TheProposalChain,
        VerifierOffsetBasis::AfterTheForward,
    ),
    (
        SpecLoop::TwoModelStochastic,
        SHARED_LOOP,
        ReportSkippedBy::TheInRoundExit,
        ChainRefusedBy::TheProposalChain,
        VerifierOffsetBasis::AfterTheForward,
    ),
];

/// The files the table names, in the order it names them, each with its source.
///
/// One, as of the last migration: every row names [`SHARED_LOOP`]. It stays a
/// table rather than a constant because a drafter that ever leaves the shared
/// loop names its own file here again, and the readings below are of the table.
const LOOP_SOURCES: [(&str, &str); 1] = [(
    SHARED_LOOP,
    include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/speculative/round_loop.rs"
    )),
)];

/// The seven edge markers, each as the character the reading below renders it
/// as.
///
/// `S` and `I` are the shared loop's two guards on the drafter's declared
/// resident-KV exit, `E` is the early exit, `W` the head of the round loop, `G`
/// either spelling of the empty-chain refusal, `R` the request record a loop
/// closes on, `P` the report of the verifier's resident KV.
///
/// `R` is what makes the reading see a report moved *into* the round loop: with
/// four markers a report placed after the refusal reads the same as one at the
/// tail, because nothing marked where the loop ended.
///
/// **`S` and `I` are the whole guard line, matched exactly, and the other six
/// are substrings.** The arm name alone is not a reader of the declaration: it
/// is there whether the guard says `!matches!(…)` or `matches!(…)`, so dropping
/// either `!` would invert which exit reports while every marker stays where
/// the table says. Carrying the negation in a *substring* needle is not enough
/// either — `|| true` appended, or `true ||` prefixed, leaves the needle inside
/// the line and the sequence unchanged while the guard is now a constant. So
/// these two are compared against the trimmed line, and nothing may be added to
/// either side of them.
///
/// The cost is that a rename, or anything else that makes rustfmt wrap one of
/// these two lines, fails this test rather than quietly passing it. That is the
/// safe direction, and it is why the exactness is confined to these two: the
/// other six markers name a call or a keyword that a loop may legitimately
/// write in more than one shape.
const MARKERS: [(char, &str); 7] = [
    (
        'S',
        "if !matches!(D::KV_REPORT_SKIPPED_BY, ReportSkippedBy::TheSeedExit) {",
    ),
    (
        'I',
        "if !(stopped_in_round && matches!(D::KV_REPORT_SKIPPED_BY, ReportSkippedBy::TheInRoundExit)) {",
    ),
    ('E', "return Ok((emitted"),
    ('W', "while emitted.len() < n_tokens"),
    ('G', "draft_tokens.is_empty()"),
    ('R', "log_request_record("),
    ('P', "report_verifier_kv_bytes("),
];

/// The file's edge markers in source order, as characters.
fn marker_sequence(src: &str) -> String {
    let mut seen: Vec<(usize, char)> = Vec::new();
    for (idx, line) in src.lines().enumerate().filter(|(_, l)| is_code(l)) {
        for (mark, needle) in MARKERS {
            if marks_line(line, mark, needle) {
                seen.push((idx, mark));
            }
        }
    }
    seen.sort_unstable();
    seen.into_iter().map(|(_, mark)| mark).collect()
}

/// The two markers read as a whole line rather than as a substring.
const EXACT_MARKERS: [char; 2] = ['S', 'I'];

/// Whether one line carries one marker, by that marker's own reading.
fn marks_line(line: &str, mark: char, needle: &str) -> bool {
    if EXACT_MARKERS.contains(&mark) {
        line.trim() == needle
    } else {
        line.contains(needle)
    }
}

/// Every loop's exits fall where the table says they do.
///
/// The report of the verifier's resident KV has one writer — a speculative
/// request never goes through `Architecture::generate_greedy` — so a caller that
/// samples the counter around the call reads whatever the previous request left
/// when a loop skips it. Which exit skips it is the drafter's own fact, and it
/// is the one thing a single shared round loop cannot keep without carrying a
/// per-drafter flag for it, so a chunk that collapses the loops decides it here.
///
/// The reading is by position and not by count, because the mutations this holds
/// move a call rather than adding one: a report hoisted above the seed-EOS
/// return inverts the disposition, and one lowered into the round loop skips it
/// on both exits, each leaving every total unchanged.
///
/// **It reads text, in statement order, and is blind past that.** The distance
/// to what it claims is between "the call is written here" and "the call runs
/// and reports the verifier's caches": a report at the declared position inside
/// a branch that never executes reads identical, and so does one handed the
/// drafter's stack. See the mutation table in `docs/SPEC_ROUND_SKELETON.md`.
///
/// Mutation: move any loop's `report_verifier_kv_bytes` call above its early
/// return, or below the head of its round loop, or delete it.
///
/// Mutations on the shared loop's two declared-exit guards, each measured
/// against `SPEWGRIP`: drop the `!` on the seed guard and it reads `PEWGRIP`;
/// drop the `!` on the tail guard, `SPEWGRP`; swap the two arm names between
/// the exits, `PEWGRP`; append `|| true` to either guard, the same as dropping
/// its `!`, because the marker is the whole line.
#[test]
fn every_loop_reports_the_verifiers_resident_kv_at_the_exit_it_declares() {
    // Every row names a file this test reads. Without this, a row pointing at a
    // file [`LOOP_SOURCES`] does not hold is read by neither test — it
    // contributes to no `want` and its source is never scanned — so a migrated
    // row reverted to its old file passes both readings while the loop it names
    // is gone.
    for (loop_kind, file, _, _, _) in DISPOSITIONS {
        assert!(
            LOOP_SOURCES.iter().any(|(f, _)| *f == file),
            "the {loop_kind:?} row names `{file}`, which is not a file this test reads"
        );
    }
    for (_, src) in LOOP_SOURCES {
        assert_eq!(
            marker_sequence(src),
            SHARED_LOOP_PATTERN,
            "{SHARED_LOOP}: the two declared-exit guards (S, I), the early exit (E), \
             the round loop (W), the empty-chain refusal (G), the request record (R) \
             and the resident-KV report (P) do not fall in the order the loop that \
             carries every row's exits is declared to write them in"
        );
    }
    // A migrated loop states its disposition rather than spelling it out, and a
    // drafter can declare the exit the shared loop ignores. The `S` and `I`
    // markers above are the second reader: they hold the loop to both arms of
    // what was declared, so a guard whose sense is inverted fails here.
    for (loop_kind, _, skipped_by, _, basis) in DISPOSITIONS {
        // One arm per migrated loop, named rather than derived — a constant is
        // read through its drafter's own type. Each migration adds its drafter
        // here, the same cliff the refusal reading below carries.
        let declared = match loop_kind {
            SpecLoop::MtpAssistant => (
                <AssistantRound<'_> as RoundDrafter>::KV_REPORT_SKIPPED_BY,
                <AssistantRound<'_> as RoundDrafter>::VERIFIER_OFFSET_BASIS,
            ),
            SpecLoop::MtpSidecar => (
                <SidecarRound<'_> as RoundDrafter>::KV_REPORT_SKIPPED_BY,
                <SidecarRound<'_> as RoundDrafter>::VERIFIER_OFFSET_BASIS,
            ),
            SpecLoop::DFlash2 => (
                <BlockRound<'_> as RoundDrafter>::KV_REPORT_SKIPPED_BY,
                <BlockRound<'_> as RoundDrafter>::VERIFIER_OFFSET_BASIS,
            ),
            SpecLoop::DFlash => (
                <AdaptiveRound<'_> as RoundDrafter>::KV_REPORT_SKIPPED_BY,
                <AdaptiveRound<'_> as RoundDrafter>::VERIFIER_OFFSET_BASIS,
            ),
            SpecLoop::Eagle3 => (
                <Eagle3Round<'_> as RoundDrafter>::KV_REPORT_SKIPPED_BY,
                <Eagle3Round<'_> as RoundDrafter>::VERIFIER_OFFSET_BASIS,
            ),
            // Two paths, one drafter: the acceptance rule is a field of it and
            // nothing either rule does reaches an exit or an offset basis.
            SpecLoop::TwoModelGreedy | SpecLoop::TwoModelStochastic => (
                <TwoModelRound<'_> as RoundDrafter>::KV_REPORT_SKIPPED_BY,
                <TwoModelRound<'_> as RoundDrafter>::VERIFIER_OFFSET_BASIS,
            ),
        };
        assert_eq!(
            declared,
            (skipped_by, basis),
            "the migrated {loop_kind:?} loop declares an exit or an offset basis the \
             table does not"
        );
    }
}

/// Every loop refuses a drafter that proposed nothing, by the measure the table
/// names.
///
/// The acceptance walk answers an empty chain rather than refusing it — pinned
/// by [`the_acceptance_walk_commits_the_rows_the_three_spellings_agreed_on`]
/// above — so the refusal is the only thing standing between a broken drafter
/// and a request that silently emits one token per round at full cost. Nothing
/// at runtime covers its loss: no gate serves a drafter that proposes nothing.
///
/// The two spellings are the same test taken one statement apart, and both
/// families reach it after every drafting forward the round takes — a two-model
/// round's verifier carry is always one token, so an input under two positions
/// is an empty chain and nothing else. Neither family stops anywhere the other
/// would not, and a shared loop with one refusal moves no request. What it
/// replaces is seven `Error::Model` texts with one, and what that one still has
/// to say is that an empty chain is a broken drafter and not the end of the
/// request.
///
/// The shared loop is counted by spelling and not by row, which is the one
/// place this reading differs from a per-file count. Every drafter names
/// [`SHARED_LOOP`] in its row, so seven rows point at one file — and the loop
/// states each refusal once however many drafters route through it. Counting
/// rows there would fail a correct tree.
///
/// **What this reading can still catch, now that the table has one file and one
/// spelling.** A row pointing at the wrong file or the wrong measure is a
/// compile error or is caught by the file check in the test above, so what is
/// left is the source: the refusal deleted from the loop, which reads zero
/// against a wanted one, and the refusal stated twice, which reads two. Those
/// are the mutations this test is for from here, and neither is hypothetical —
/// the first is the "empty-chain refusal lost" row of the mutation table in
/// `docs/SPEC_ROUND_SKELETON.md`, which nothing at runtime covers.
///
/// Mutation: delete `draft_tokens.is_empty()` from the loop; state it twice.
#[test]
fn every_loop_refuses_a_drafter_that_proposed_nothing_by_its_declared_measure() {
    for (file, src) in LOOP_SOURCES {
        for (mark, needle) in MARKERS {
            if mark != 'G' {
                continue;
            }
            let rows = DISPOSITIONS
                .iter()
                .filter(|(_, f, _, refused_by, _)| {
                    *f == file
                        && needle
                            == match refused_by {
                                ChainRefusedBy::TheProposalChain => "draft_tokens.is_empty()",
                            }
                })
                .count();
            // The shared loop states each refusal once however many drafters
            // route through it, which is now all seven.
            let want = rows.min(1);
            let have = src
                .lines()
                .filter(|l| is_code(l) && l.contains(needle))
                .count();
            assert_eq!(
                have, want,
                "{file} states `{needle}` {have} time(s) where the table wants it \
                 stated {want} time(s), over {rows} loop row(s) refusing an empty \
                 chain that way"
            );
        }
    }
}

/// The seed's attribution is written where the seed is emitted, and nowhere
/// else.
///
/// `run_rounds` pushes one `DecidedBy::FullVocab` for the token a request emits
/// before its first round. A pair that emits no seed emits no such token, so the
/// push belongs inside the arm that opens on an emitted seed — between it and
/// the emission. Hoisted above the arm, a request with no seed attributes a
/// token it never emitted and every entry after it names the wrong token.
///
/// Nothing at runtime sees that: the one loop with no seed is the two-model
/// greedy pair, whose entry passes no attribution buffer, so the hoisted push
/// runs on no request that carries one. This reading is what stands in for it,
/// until a drafter that emits no seed also attributes its tokens.
///
/// Mutation: hoist the push above `if let Seed::Emitted(seed)`; move it below
/// the emission.
#[test]
fn the_seeds_attribution_is_written_only_where_a_seed_is_emitted() {
    let src = LOOP_SOURCES
        .iter()
        .find(|(f, _)| *f == SHARED_LOOP)
        .map(|(_, s)| *s)
        .unwrap_or_default();
    let at = |needle: &str| -> Vec<usize> {
        src.lines()
            .enumerate()
            .filter(|(_, l)| is_code(l) && l.contains(needle))
            .map(|(i, _)| i)
            .collect()
    };
    let arm = at("if let Seed::Emitted(seed) = prefilled.seed {");
    let push = at("buf.push(DecidedBy::FullVocab);");
    let emit = at("if emit_seed_token(");
    assert_eq!(
        (arm.len(), push.len(), emit.len()),
        (1, 1, 1),
        "the loop opens one emitted-seed arm, attributes one seed and emits it once, \
         and it states them {} / {} / {} time(s)",
        arm.len(),
        push.len(),
        emit.len()
    );
    assert!(
        arm < push && push < emit,
        "the seed's attribution sits inside the arm that emits it, between the arm \
         and the emission, and this loop writes them at lines {arm:?}, {push:?}, \
         {emit:?} — a push above the arm attributes a token a seedless pair never \
         emitted"
    );
}

/// One generator serves a speculative request, and it is the round's draw.
///
/// `VerifierDraw` holds the request's `Pcg32` and every draw of the request goes
/// through it — the verifier's own tokens, the verifier's distributions, a
/// drafter's sampled proposals and an acceptance rule's coins. A second
/// generator seeded from the same `seed_or_default()` is two correlated streams,
/// and the defect it carries is invisible to every other reading in this tree:
/// each stream reproduces under its seed, parts from a second seed and parts
/// from greedy, so the gate in
/// `crates/rmlx-models/tests/two_model_stochastic.rs` passes on both. Only a
/// diff of one seed's tokens across two commits sees it, and that is a control a
/// reviewer runs, not a gate that runs itself.
///
/// This is the reading that runs itself, and it is over the **whole** module
/// family rather than one drafter: a loop or a drafter that builds
/// `VerifierDraw::new(sampler_cfg)` *and* seeds a `Pcg32` of its own satisfies
/// `make check-spec-sampling`'s RULE 1 and every per-drafter reading beside it.
///
/// **It reads text and is blind past that** — a construction inside a branch
/// that never runs reads identical.
///
/// Mutation: seed a second `Pcg32` in any loop, drafter or entry under
/// `src/speculative/`.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: a source tree this test's own crate cannot read back is a broken checkout, and the panic names the path"
)]
fn the_requests_draw_stream_has_one_generator() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/speculative");
    let mut found: Vec<(String, String, String)> = Vec::new();
    let mut stack = vec![root.clone()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).expect("the speculative source directory") {
            let path = entry.expect("a directory entry").path();
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let name = path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let is_rust = path
                .extension()
                .is_some_and(|ext| ext.eq_ignore_ascii_case("rs"));
            if !is_rust || name.ends_with("_tests.rs") || name == "tests.rs" {
                continue;
            }
            let src = std::fs::read_to_string(&path).expect("a source file this crate ships");
            let rel = path
                .strip_prefix(&root)
                .map(|p| p.to_string_lossy().into_owned())
                .unwrap_or(name);
            for (line, owner) in lines_in_fns(&src, "Pcg32::new(") {
                found.push((rel.clone(), owner, line.to_owned()));
            }
        }
    }
    found.sort();
    assert_eq!(
        found,
        vec![(
            "mod.rs".to_owned(),
            "new".to_owned(),
            "rng: crate::sampler::Pcg32::new(cfg.seed_or_default()),".to_owned()
        )],
        "a speculative request seeds one generator, in `VerifierDraw::new`, and this \
         tree seeds them at {found:?} — a second one off the same seed is a second \
         stream that reproduces just as well and is drawn from just as wrongly"
    );
}
