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
use super::{
    accept_prefix, draft_rows_to_drop, rollback_target_from_head, rollback_target_from_tail,
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

// ---------------------------------------------------------------------------
// What the seven loops declare at their edges
// ---------------------------------------------------------------------------

/// Which of a loop's two exits skips the report of the verifier's resident KV.
#[derive(Debug, Clone, Copy)]
enum ReportSkippedBy {
    /// The seed EOS: the loop emits a token out of its prefill forward, and a
    /// request whose whole output is that token returns before any round.
    TheSeedExit,
    /// The in-round EOS: the loop emits nothing before its first round, so the
    /// only early exit it has is inside one.
    TheInRoundExit,
}

/// How a loop refuses a drafter that proposed nothing.
#[derive(Debug, Clone, Copy)]
enum ChainRefusedBy {
    /// On the proposals themselves, before the verify forward.
    TheProposalChain,
    /// On the verifier input the empty chain produces, one step later.
    TheVerifierInput,
}

/// What each round loop declares at its edges, and the file that holds it.
///
/// This table is the statement; the two tests below are readings of it against
/// today's source. Both facts become per-drafter declarations in the engine when
/// the loops collapse, and this table is then re-keyed onto those declarations
/// with the same seven rows — which is why it is data rather than seven
/// assertions.
const DISPOSITIONS: [(SpecLoop, &str, ReportSkippedBy, ChainRefusedBy); 7] = [
    (
        SpecLoop::MtpSidecar,
        "mtp.rs",
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
    ),
    (
        SpecLoop::DFlash,
        "dflash/mod.rs",
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
    ),
    (
        SpecLoop::DFlash2,
        "dflash2/round.rs",
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
    ),
    (
        SpecLoop::Eagle3,
        "eagle3/mod.rs",
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
    ),
    (
        SpecLoop::MtpAssistant,
        "gemma4_assistant.rs",
        ReportSkippedBy::TheSeedExit,
        ChainRefusedBy::TheProposalChain,
    ),
    (
        SpecLoop::TwoModelGreedy,
        "mod.rs",
        ReportSkippedBy::TheInRoundExit,
        ChainRefusedBy::TheVerifierInput,
    ),
    (
        SpecLoop::TwoModelStochastic,
        "mod.rs",
        ReportSkippedBy::TheInRoundExit,
        ChainRefusedBy::TheVerifierInput,
    ),
];

/// The files the table names, in the order it names them, each with its source.
const LOOP_SOURCES: [(&str, &str); 6] = [
    (
        "mtp.rs",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/speculative/mtp.rs"
        )),
    ),
    (
        "dflash/mod.rs",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/speculative/dflash/mod.rs"
        )),
    ),
    (
        "dflash2/round.rs",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/speculative/dflash2/round.rs"
        )),
    ),
    (
        "eagle3/mod.rs",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/speculative/eagle3/mod.rs"
        )),
    ),
    (
        "gemma4_assistant.rs",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/speculative/gemma4_assistant.rs"
        )),
    ),
    (
        "mod.rs",
        include_str!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/src/speculative/mod.rs"
        )),
    ),
];

/// The four edge markers, each as the character the reading below renders it as.
///
/// `E` is the early exit, `W` the head of the round loop, `G` either spelling of
/// the empty-chain refusal, `P` the report of the verifier's resident KV.
const MARKERS: [(char, &str); 5] = [
    ('E', "return Ok((emitted"),
    ('W', "while emitted.len() < n_tokens"),
    ('G', "draft_tokens.is_empty()"),
    ('G', "v_k < 2"),
    ('P', "report_verifier_kv_bytes("),
];

/// Whether a line is code rather than a whole-line comment.
///
/// A marker written into the sentence that explains it would otherwise read as
/// the statement it describes — the same reading `scripts/check_spec_charge.sh`
/// takes, and for the same reason. A trailing comment on a line of code is not
/// stripped: that needs a quote-aware scan, and no marker here is one a caller
/// would write at the end of a statement.
fn is_code(line: &str) -> bool {
    !line.trim_start().starts_with("//")
}

/// The file's edge markers in source order, as characters.
fn marker_sequence(src: &str) -> String {
    let mut seen: Vec<(usize, char)> = Vec::new();
    for (idx, line) in src.lines().enumerate().filter(|(_, l)| is_code(l)) {
        for (mark, needle) in MARKERS {
            if line.contains(needle) {
                seen.push((idx, mark));
            }
        }
    }
    seen.sort_unstable();
    seen.into_iter().map(|(_, mark)| mark).collect()
}

/// How the markers of one loop fall in source order, given which exit skips the
/// report.
///
/// A loop that skips it on the seed exit returns before the round loop opens and
/// reports after it closes; a loop with no seed has its early exit inside the
/// round loop, past the refusal. Those are two different orderings of the same
/// four markers, which is what makes this reading positional rather than a count.
fn expected_pattern(skipped_by: ReportSkippedBy) -> &'static str {
    match skipped_by {
        ReportSkippedBy::TheSeedExit => "EWGP",
        ReportSkippedBy::TheInRoundExit => "WGEP",
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
/// The reading is by position and not by count, because the mutation this holds
/// moves a call rather than adding one: a report hoisted above the seed-EOS
/// return inverts the disposition and leaves every total unchanged.
///
/// **It reads text, and the distance to what it claims is exactly between "the
/// call is written here" and "the call reports the verifier's caches".** A report
/// computed from the drafter's stack passes this and every other check in the
/// tree — see the mutation table in `docs/SPEC_ROUND_SKELETON.md`.
///
/// Mutation: move any loop's `report_verifier_kv_bytes` call above its early
/// return, or delete it.
#[test]
fn every_loop_reports_the_verifiers_resident_kv_at_the_exit_it_declares() {
    for (file, src) in LOOP_SOURCES {
        let want: String = DISPOSITIONS
            .iter()
            .filter(|(_, f, _, _)| *f == file)
            .map(|&(_, _, skipped_by, _)| expected_pattern(skipped_by))
            .collect();
        assert_eq!(
            marker_sequence(src),
            want,
            "{file}: the early exit (E), the round loop (W), the empty-chain \
             refusal (G) and the resident-KV report (P) do not fall in the order \
             the table declares for the loop(s) it holds"
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
/// The two spellings are not interchangeable. Five loops refuse the proposals;
/// the two-model pair refuses the verifier input the empty chain produces, one
/// step later, having already taken a drafting forward per proposal. A shared
/// loop has one refusal, so both spellings go at the same moment and one of the
/// two families changes where it stops.
///
/// Mutation: drop either spelling from any loop, or move a loop from one
/// spelling to the other without moving its row.
#[test]
fn every_loop_refuses_a_drafter_that_proposed_nothing_by_its_declared_measure() {
    for (file, src) in LOOP_SOURCES {
        for (mark, needle) in MARKERS {
            if mark != 'G' {
                continue;
            }
            let want = DISPOSITIONS
                .iter()
                .filter(|(_, f, _, refused_by)| {
                    *f == file
                        && needle
                            == match refused_by {
                                ChainRefusedBy::TheProposalChain => "draft_tokens.is_empty()",
                                ChainRefusedBy::TheVerifierInput => "v_k < 2",
                            }
                })
                .count();
            let have = src
                .lines()
                .filter(|l| is_code(l) && l.contains(needle))
                .count();
            assert_eq!(
                have, want,
                "{file} states `{needle}` {have} time(s) where the table declares \
                 {want} loop(s) refusing an empty chain that way"
            );
        }
    }
}
