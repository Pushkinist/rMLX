// LOC-exempt: one gate over seven round loops. The oracle it applies — the
// divergence-confidence judgement and the repetition control — is one argument
// whose constants are set against the population of pairs and prompts the file
// runs, and the pair table, the prompt sweep and the per-loop arms are that
// population. Splitting it would put the constants in one file and the readings
// that set them in another.
//! Speculative decoding must not change what the model says.
//!
//! Greedy speculative decoding emits the verifier's own argmax at every
//! position, so at temperature 0 a speculative run and a plain run of the same
//! verifier are two ways of computing one answer. Every change to a drafter, a
//! block policy or an acceptance walk can trade that away for throughput, and
//! nothing about the throughput number says it happened.
//!
//! **It is not bit-identity, and measurement says so.** The verify pass scores a
//! whole block in one forward where plain decode steps one token at a time, and
//! that is a different reduction order. The two arms share a long prefix, flip
//! one token, and then write the same answer — or, on a prompt with many
//! near-equal continuations, two different ones that agree on most of their
//! content.
//!
//! # The oracle
//!
//! **Where a correct pair diverges is decidable, and it is the only thing here
//! that is.** A reduction-order difference is a relative perturbation of order
//! `1e-3` on a logit, so it can only flip a decision the verifier was already
//! nearly indifferent about. [`divergence_confidence`] reads the verifier's own
//! top-two logprob margin at the position the arms first differ and returns
//! where that sits in the same arm's own margin distribution. Both arms saw the
//! same context up to that position, so this judges the **pair**, and it needs
//! no per-prompt calibration: it is a rank, not a number of nats.
//!
//! Over the six prompts in [`PROMPTS`], for each pair, run against the shipped
//! engine and against two deliberately broken ones:
//!
//! | engine | first divergence | percentile of the reference arm's own margins |
//! |---|---|---|
//! | assistant pair, as shipped | 16 to 256 (one prompt bit-identical) | 0.0000 to 0.0820 |
//! | recurrent pair, as shipped | 65 to 256 (three prompts bit-identical) | 0.0000 to 0.0234 |
//! | block pair, as shipped | 9 to 89 (none bit-identical) | 0.0000 to 0.0273 |
//! | assistant pair, SWA ring keeping its rejected block tail | 6 to 9 | 0.4219 to 0.9258 |
//! | recurrent pair, acceptance walk without the final norm | 1 to 24 | 0.0000 to 0.5000 |
//! | block pair, rejected tail never rolled off | 4 to 7 | 0.0117 to 0.9297 |
//! | block pair, one rejected draft kept every partial round | 4 to 10 | 0.1758 to 0.6406 |
//!
//! The block pair's own two broken engines are refused on six of six prompts
//! each. One of their twelve cells reads 0.0117 — inside the confidence ceiling
//! — and is refused by the repetition control instead, which reads 1.0000 on it:
//! the two oracles cover each other, and neither alone gives that recall.
//!
//! **There is deliberately no subsequence floor.** How much of one answer two
//! correct arms share is decided by where their first near-tie lands and by
//! nothing else: on `lcs_ratio` — not the tail — the assistant pair reads 0.9375
//! on the 4k document and 0.4766
//! on a short prose prompt whose arms flip an **exact** tie (top-two margin
//! 0.0000) at token 37, after which both write well-formed, correct, different
//! prose. The two broken engines read 0.2188 to 0.4615 on the same measure —
//! three per cent under the worst correct cell, and the correct minimum is set
//! by where an exact tie happens to land, which nothing bounds from below.
//! [`report`] prints the figure on every run and the gate does not assert it.
//!
//! **The repetition control** is the second oracle, and it exists because the
//! first has nothing to read when both arms are degenerate: two arms in the same
//! loop have no healthy reference arm whose margins mean anything. So every run
//! also checks that neither arm repeats at a short period across more than
//! [`MAX_CYCLE_FRACTION`] of its tokens, over the whole stream and over each
//! tail cut, at every period up to [`MAX_CYCLE_PERIOD`] that leaves
//! [`MIN_CYCLE_SAMPLES`] comparisons.
//!
//! Windowing and the period sweep are both load-bearing; three real
//! degeneracies score under any ceiling without them:
//!
//! | shape | whole stream, period ≤ 8 | whole stream, period ≤ 64 | windowed |
//! |---|---|---|---|
//! | collapse from token 0 | 1.0000 | 1.0000 | 1.0000 |
//! | collapse over the last two fifths | 0.3992 | 0.3992 | 1.0000 |
//! | a twelve-token phrase over four fifths | 0.0000 | 0.7960 | 1.0000 |
//!
//! The control has **no general threshold** and there is none: healthy output
//! spans 0.03 for prose to 0.88 for a markdown table with a yes/no column, and
//! degenerate output 0.37 for a ragged loop to 1.00 for an exact one — two
//! populations overlapping over most of their range. Prose is the regime where
//! they separate, which is why [`PROSE_INSTRUCTION`] exists and why a tokenizer
//! that declares `<think>` gets an empty reasoning block. When the *reference*
//! arm trips the control the input is outside the gate's domain and the gate
//! says so, rather than accusing plain greedy of a repetition loop.
//!
//! # Recall
//!
//! Over the (pair, prompt) cells each broken engine above produces, the gate
//! refuses six of six on the assistant pair and four of six on the recurrent
//! one. The two it does not refuse read 0.0000 and 0.0977 — inside the ceiling —
//! and are why the gate runs **every** prompt rather than one: recall is a
//! property of the set. Both broken engines turn both gates red. The block pair's
//! own two broken engines are refused on six of six prompts each; the adaptive
//! and restricted-vocabulary pairs refuse a rollback off by one on all five
//! prompts they judge, and the two-model pair on all six.
//!
//! # A declared boundary, and the only waiver
//!
//! EAGLE-3's verify pass scores every position it may accept over the drafter's
//! reduced vocabulary, so where the verifier's own answer is a token that
//! vocabulary cannot name the arm emits the drafter's argmax instead. That is
//! upstream's design and no correct loop of this kind avoids it, so a divergence
//! there is reported rather than refused.
//!
//! **Only there.** Widening the same inexactness to the round's correction
//! changes an answer at exactly that kind of token — indistinguishable in the
//! two token streams — and is refused. Which position the loop was at is the
//! only thing that separates them, so the loop reports it per emitted token
//! rather than the gate inferring it from the answer. [`Restriction`] carries
//! the rule; `docs/SPEC_ANSWER_EQUIVALENCE.md` carries the measurements and
//! `scripts/spec_broken_engine.sh` takes them again.
//!
//! # Pairs
//!
//! Six, whose verifiers resolve by slug from `RMLX_O_MODELS_ROOT`:
//!
//! | verifier | drafter | round loop | rollback |
//! |---|---|---|---|
//! | `gemma-4-e2b-it-mxfp8` | `gemma-4-E2B-it-assistant-bf16` | shared-K/V assistant | KV truncation, SWA ring included |
//! | `Qwen3.8-27B-mxfp8` | `Qwen3.8-27B-MTP-mxfp8` | MTP sidecar | KV truncation + recurrent snapshot/replay |
//! | `Qwen3.8-27B-4bit` | `Qwen3.8-27B-DFlash2` | DFlash 2 block drafter | KV truncation + recurrent snapshot/replay |
//! | `Qwen3.6-35B-A3B-8bit` | `Qwen3.6-35B-A3B-DFlash` | DFlash 1, adaptive block | KV truncation + recurrent snapshot/replay |
//! | `Qwen3.6-35B-A3B-8bit` | `specdrift-qwen3.6-35b-a3b-eagle3` | EAGLE-3 | KV truncation + recurrent snapshot/replay |
//! | `Qwen3.8-27B-mxfp8` | `ornith-1.0-9b-mxfp8-mlx` | two full models, greedy | both models' KV + recurrent state |
//!
//! The first runs wherever the snapshots are; the others run on request — see
//! [`DrafterSource`] for the shader-validation reason. `RMLX_DRAFT_TEST_MODEL`
//! names one drafter, so a pair it does not belong to stands down on the kind
//! its snapshot declares rather than loading it as something else.
//!
//! The seventh round loop the engine ships is the two-model **stochastic** one,
//! and it can have no pair here: it runs only above temperature 0, where neither
//! arm is a function of the model alone. `two_model_stochastic.rs` gates it on a
//! different property.
//!
//! The MTP pair is the one whose agreement no subsequence floor could separate
//! from a broken rollback. The divergence oracle settled it: its acceptance walk
//! was scoring an un-normed hidden through the LM head, and with that fixed
//! three of six prompts come back bit-identical.
//!
//! Server-free. `RMLX_KV_TEST_MODEL` / `RMLX_DRAFT_TEST_MODEL` override either
//! half; a verifier of another architecture stands the pair down.
//!
//!     cargo test -p rmlx-models --test spec_greedy_equivalence -- --ignored --nocapture

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::ignore_without_reason
)]

use std::path::Path;

mod common;

use common::round_stream::{
    round_events, CapturedEvent, RoundStreamRecorder, EAGLE3_STEP_SWITCH_TARGET,
    PHASE_SWITCH_TARGET, ROUND_EVENT_FIELDS,
};

use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::speculative::dflash::{dflash_generate, DFlashDrafter};
use rmlx_models::speculative::dflash2::{dflash2_generate, DFlash2Drafter};
use rmlx_models::speculative::eagle3::{eagle3_generate, DecidedBy, Eagle3Drafter};
use rmlx_models::speculative::gemma4_assistant::{mtp_assistant_generate, Gemma4AssistantDrafter};
use rmlx_models::speculative::mtp::{mtp_generate, MtpDrafter};
use rmlx_models::{Declared, DraftKind, SpeculativeDispatcher};

/// Token budget per arm. A ceiling, not a target: both arms stop on the
/// verifier's own stop ids, and every prompt in the sweep answers under it.
///
/// The budget is what makes the horizon long enough to matter — hundreds of
/// rounds and a kilotoken of context, against the 48 tokens the sibling
/// alignment suites compare. Running *past* the answer is the opposite problem:
/// with no stop ids both arms emit end-of-turn forever, and a comparison over
/// that filler measures nothing about the round loop.
const N_TOKENS: usize = 256;

/// How much answer both arms must have produced for the comparison to have
/// power.
///
/// **One number, and both sides of it are measured.** Every arm the gate judged
/// across two pairs and six prompts ran to the full [`N_TOKENS`] budget, so the
/// floor is not what any correct pair is up against. What it decides is where
/// the line falls between two different verdicts: one arm short while the other
/// ran on is the round loop's doing and is refused, and *both* arms short is the
/// prompt's — the recurrent pair answers the 4k summary in 13 and 26 tokens —
/// and is reported rather than failed.
///
/// It is above `TAIL_WINDOWS * MIN_CYCLE_SAMPLES`, which is what the last tail
/// window needs before it can evidence a cycle at all, and under the budget, so
/// a pair that answers in full is never refused for length.
const MIN_ANSWER_TOKENS: usize = 160;

/// The shorter arm over the longer. `lcs_ratio` divides by the shorter stream,
/// so an arm that stopped at a third of the other's length and matched its
/// prefix would otherwise score 1.0.
const MIN_LENGTH_RATIO: f64 = 0.60;

use rmlx_models::speculative::{default_block_for, drafts_per_round};

/// Context both arms run under. Above the 4k prompt plus the budget, and the
/// same on both sides — a different cap on either would make this a measurement
/// of the cap.
const MAX_CTX: i32 = 8192;

/// Where the first divergence's own confidence may sit in the reference arm's
/// margin distribution.
///
/// **Both sides measured**, over six pairs and six prompts each, by running the
/// gate against the shipped engine and against a deliberately broken one per
/// loop — two on the block pair and two on the restricted-vocabulary one:
///
/// | engine | percentile of the reference arm's own margins |
/// |---|---|
/// | assistant pair, as shipped | 0.0000 to 0.0820 |
/// | recurrent pair, as shipped | 0.0000 to 0.0234 |
/// | block pair, as shipped | 0.0000 to 0.0273 |
/// | adaptive pair, as shipped, at the block it is served | 0.0000 to 0.0234 |
/// | adaptive pair, as shipped, over its whole schedule | 0.0000 to 0.0703 |
/// | restricted-vocabulary pair, as shipped | 0.0000 to 0.0703 |
/// | two-model pair, as shipped | 0.0000 to 0.0234 |
/// | assistant pair, SWA ring keeping its rejected block tail | 0.4219 to 0.9258 |
/// | recurrent pair, acceptance walk without the final norm | 0.0000 to 0.5000 |
/// | block pair, rejected tail never rolled off | 0.0117 to 0.9297 |
/// | block pair, one rejected draft kept every partial round | 0.1758 to 0.6406 |
/// | adaptive pair, one rejected draft kept every partial round | 0.0703 to 0.8320 |
/// | restricted-vocabulary pair, one rejected draft kept every round | 0.0000 to 0.8320 |
/// | restricted-vocabulary pair, correction left on the restricted argmax | 0.0000 to 0.8828 |
/// | two-model pair, one rejected draft kept every partial round | 0.0000 to 0.6680 |
///
/// The measurement leaves a band, and the value sits inside it: above the worst
/// correct cell (0.0820, 1.46x) and under the lowest broken cell above it
/// (0.1445, 1.20x). It clears every correct cell, and every broken engine here
/// is refused on at least four of the prompts the gate judged for it — which is
/// what running every prompt rather than one buys, since no broken engine is
/// refused on all of them by this oracle alone.
///
/// **The adaptive pair's two rows are the same engine at two blocks**, and the
/// wider one reads three times the narrower. Both clear the ceiling and both are
/// refused on none of the prompts the gate judges, against at least four for
/// every broken engine here — but 0.0703 is also where the broken adaptive row
/// starts, so the two populations touch on that pair at that block, and the
/// margin the ceiling has over a correct reading there is 1.71x rather than the
/// 1.46x the paragraph above quotes. The narrower row was measured before this
/// pair was split in two and its block is not recorded; the served block
/// reproduces it exactly.
///
/// The exception is the last row, and it is a property of the defect rather than
/// of the ceiling: leaving the correction on the restricted argmax only changes
/// an answer where that vocabulary falls short of the verifier's, which the
/// runs measure at one to five tokens per answer. It is refused on one of the
/// five prompts the gate judges, at 0.8828, at a token the drafter's vocabulary
/// cannot name — and at the round's **correction**, which is the only thing that
/// keeps it refused now the boundary at an accepted position is not. See
/// [`Restriction`].
const MAX_DIVERGENCE_CONFIDENCE: f64 = 0.12;

/// The worst [`weakest_tail`] reading a **correct** pair reached over the prompts
/// the gate judged, on any pair. Not a threshold the gate applies — see below —
/// but the reference `two_arms_in_the_same_ragged_loop_are_refused_until_they_are_no_longer_one_loop`
/// reads to say how far the two populations overlap on this measure.
/// `the_worst_correct_tail_is_the_worst_of_the_tails_measured` holds it to that
/// population, and it moved from 0.2344 to here when four more pairs joined it.
///
/// The paragraph below is about a **different measure** and its figures are not
/// comparable to the one above: [`lcs_ratio`] over the whole arm, where the same
/// runs read higher.
///
/// There is deliberately **no subsequence floor**. How much of one answer two
/// correct arms share is decided by where their first near-tie lands and by
/// nothing else: on `lcs_ratio` the assistant pair reads 0.9375 on the 4k
/// document and 0.4766
/// on a short prose prompt whose arms flip an exact tie at token 37, and the two
/// broken engines measured here read 0.2188 to 0.4615 — three per cent under
/// that, with nothing bounding the correct minimum from below. `report` prints
/// the figure on every run and the gate does not assert it.
const WORST_CORRECT_TAIL_AGREEMENT: f64 = 0.1094;

/// How much of an arm — or of any tail cut of it — may repeat at a short period
/// before it counts as collapsed.
///
/// Set against **the output this gate's own prompts produce**, which is prose:
/// across the prompts the gate judges the real arms read 0.0426 to 0.1351, and 1000
/// synthetic healthy streams at each of six lengths peaked at 0.1351 and tripped
/// this ceiling none (`the_false_positive_rate_on_healthy_output` prints that).
/// 1.48x over the worst of those 6000, and far under every collapse the gate has
/// to catch — the one this gate was built on reads 1.0000.
///
/// The other side is set by the pair regime, and it is **swept rather than
/// sampled**: `two_arms_in_the_same_ragged_loop_are_refused_until_they_are_no_longer_one_loop`
/// walks two arms in the same period-8 loop from 0% to 100% raggedness over four
/// seed pairs and pins where this control stops refusing them — 60%, past which
/// the arms are more noise than loop. Twenty seed pairs over that band leave the
/// range covered at 0.22 and open the first hole at 0.24, so the value here has
/// room rather than sitting on the boundary. An earlier 0.50 — placed from six
/// sampled points — left 34% to 52% passing. Nothing takes over past 60%: what
/// the control admits there agrees better than the worst correct pair does, which
/// is the measurement behind having no subsequence floor.
///
/// It is **not** a general degeneracy threshold and there is none: healthy
/// markdown tables read 0.68 to 0.88 on this measure and ragged loops read 0.37
/// to 0.85, two populations overlapping over most of their range. Structured
/// output therefore trips this control, which is why [`PROSE_INSTRUCTION`]
/// exists and why the value here is only meaningful for arms these prompts
/// produced.
const MAX_CYCLE_FRACTION: f64 = 0.20;

/// Longest cycle the control looks for.
///
/// A degeneracy in real output is usually a repeated phrase or sentence, not a
/// repeated token: a 12-token phrase filling four fifths of a stream scores
/// 0.0000 at any period under 12 and 0.7960 at 12. The collapse this gate was
/// built on repeats at period 8.
///
/// Bounded again by what leaves [`MIN_CYCLE_SAMPLES`] comparisons in whatever
/// window is being read, which is the only bound that turned out to be needed —
/// and which is also this control's declared blind spot. The narrowest window is
/// `len / TAIL_WINDOWS`, so at a 256-token arm the last window can evidence no
/// period above 32; a collapse confined to the last quarter at a longer period
/// than that is read only over the whole stream, where a healthy majority
/// dilutes it. `a_cycle_confined_to_a_window_too_narrow_to_read_it_is_a_declared_blind_spot`
/// is where that is written down.
const MAX_CYCLE_PERIOD: usize = 64;

/// Fewest comparisons a cycle reading may be computed from.
///
/// `strongest_cycle` divides by `len - period`. In the last quarter of a
/// 256-token arm that is a 64-token window, and at period 63 the denominator is
/// **one comparison** — a single coincidental token match read 1.0000 and the
/// gate reported a repetition loop on healthy output. Every false positive
/// observed came from a denominator of one or two, at any ceiling, so this floor
/// is the whole fix; `the_false_positive_rate_on_healthy_output` prints the
/// measured rate per length rather than recording a number nothing can
/// regenerate.
///
/// A bound on the *period relative to the window* was tried alongside it and
/// removed: it changed no false-positive rate and it blinded the sweep to real
/// collapses — a period-40 loop over the last quarter of a 512-token arm read
/// 0.4074 under it and 1.0000 without it, which is the shape the windowed sweep
/// exists for.
const MIN_CYCLE_SAMPLES: usize = 32;

/// The stream is cut at each `1/TAIL_WINDOWS` boundary and the suffixes
/// compared, so a divergence that begins late has a window it dominates.
const TAIL_WINDOWS: usize = 4;

// ── Oracle ───────────────────────────────────────────────────────────────────

/// How many leading tokens two streams share.
fn common_prefix_len(a: &[u32], b: &[u32]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// Longest common subsequence of two token streams, over the shorter of them.
fn lcs_ratio(a: &[u32], b: &[u32]) -> f64 {
    let (n, m) = (a.len(), b.len());
    if n == 0 || m == 0 {
        return 0.0;
    }
    let mut prev = vec![0usize; m + 1];
    let mut cur = vec![0usize; m + 1];
    for i in 1..=n {
        for j in 1..=m {
            cur[j] = if a[i - 1] == b[j - 1] {
                prev[j - 1] + 1
            } else {
                cur[j - 1].max(prev[j])
            };
        }
        std::mem::swap(&mut prev, &mut cur);
        cur.fill(0);
    }
    prev[m] as f64 / n.min(m) as f64
}

/// Where the reference arm's confidence at the first divergence sits in its own
/// distribution: the fraction of its decisions it was **less** sure about.
///
/// `margins[i]` is the verifier's top-two logprob gap at position `i` of the
/// reference arm. The two arms share every position before the first divergence,
/// so `margins[d]` is the gap the speculative arm faced as well.
///
/// `None` when the arms never differ, or when they first differ past the end of
/// the reference arm — the length guard owns that case.
fn divergence_confidence(spec: &[u32], plain: &[u32], margins: &[f32]) -> Option<(usize, f64)> {
    let d = common_prefix_len(spec, plain);
    if d >= spec.len() && d >= plain.len() {
        return None;
    }
    let at = *margins.get(d)?;
    let below = margins.iter().filter(|m| **m < at).count();
    Some((d, below as f64 / margins.len() as f64))
}

/// The strongest short cycle in `tokens`: its period, and the fraction of
/// positions that repeat at that period.
///
/// A stream stuck in a loop matches itself at the loop's period, and two arms in
/// the same loop agree perfectly — so the equivalence oracle says nothing
/// exactly when this is high. It covers every period up to [`MAX_CYCLE_PERIOD`],
/// because a collapse that starts at token 20 and a two-token `A B A B` cycle are
/// both degeneracies a leading-run measure scores at zero.
fn strongest_cycle(tokens: &[u32]) -> (usize, f64) {
    let mut worst = (1usize, 0.0f64);
    for period in 1..=MAX_CYCLE_PERIOD.min(tokens.len().saturating_sub(MIN_CYCLE_SAMPLES)) {
        let samples = tokens.len() - period;
        let matches = tokens[period..]
            .iter()
            .zip(tokens)
            .filter(|(a, b)| a == b)
            .count();
        let fraction = matches as f64 / samples as f64;
        if fraction > worst.1 {
            worst = (period, fraction);
        }
    }
    worst
}

/// The strongest short cycle in `tokens` or in any tail cut of it: where it
/// starts, its period, and the fraction of that window repeating at it.
///
/// Over the whole stream a collapse confined to the last two fifths reads
/// 0.3992 — under the ceiling, because the healthy majority dilutes it. The
/// same cuts `weakest_tail` uses give it a window it fills.
fn strongest_windowed_cycle(tokens: &[u32]) -> (usize, usize, f64) {
    let starts =
        std::iter::once(0).chain((1..TAIL_WINDOWS).map(|n| tokens.len() * n / TAIL_WINDOWS));
    let mut worst = (0usize, 1usize, 0.0f64);
    for start in starts {
        let (period, fraction) = strongest_cycle(&tokens[start..]);
        if fraction > worst.2 {
            worst = (start, period, fraction);
        }
    }
    worst
}

/// The weakest agreement over the tail windows, and where it was found.
///
/// The whole-stream ratio blends a run that diverged once at token 60 and
/// re-converged with a run that diverged at token 700 and never came back: both
/// can land near 0.8. A regression that begins past a cache-size threshold is
/// the second shape, and the repository has shipped that class. Suffix windows
/// separate them — the late one collapses the last window while the early one
/// leaves every window high.
fn weakest_tail(spec: &[u32], plain: &[u32]) -> (usize, f64) {
    let shorter = spec.len().min(plain.len());
    let mut worst = (0usize, 1.0f64);
    for numerator in 1..TAIL_WINDOWS {
        let start = shorter * numerator / TAIL_WINDOWS;
        let ratio = lcs_ratio(&spec[start..], &plain[start..]);
        if ratio < worst.1 {
            worst = (start, ratio);
        }
    }
    worst
}

/// What a pair's loop could and could not have said at each of its positions.
///
/// Only the restricted-vocabulary pair builds one. Its verify pass emits the
/// drafter's own argmax at a position it accepted, and that argmax is the
/// verifier's exactly when the verifier's is a token the drafter can name — so
/// a divergence at an accepted position whose reference token is unnameable is
/// the declared boundary and could be nothing else. The round's correction is
/// taken over the whole vocabulary, so a divergence there is a divergence like
/// any other pair's and is judged like one.
///
/// **That distinction is the whole of the rule, and dropping it forgives the
/// defect the pair exists to catch.** Widening the restriction to the
/// correction as well changes an answer at exactly the same kind of token, so
/// the two are indistinguishable in the token streams; what separates them is
/// which position the loop was at, which only the loop knows.
struct Restriction<'a> {
    /// Target ids the drafter can name.
    vocab: &'a std::collections::HashSet<u32>,
    /// Which vocabulary decided each token of the speculative arm.
    decided_by: &'a [DecidedBy],
}

impl Restriction<'_> {
    /// Whether the declared boundary is what parted the arms at `at`.
    fn explains(&self, at: usize, plain: &[u32]) -> bool {
        self.decided_by.get(at) == Some(&DecidedBy::RestrictedVocab)
            && plain.get(at).is_some_and(|id| !self.vocab.contains(id))
    }

    /// How many of the reference arm's own tokens the drafter cannot name.
    fn unnameable(&self, plain: &[u32]) -> usize {
        plain.iter().filter(|id| !self.vocab.contains(id)).count()
    }
}

/// What one pair of arms says about the round loop.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// The round loop reproduced what the verifier decodes alone.
    Agreed,
    /// The pair says nothing either way, and the reason is a property of the
    /// prompt, the model or the pair's own declared vocabulary rather than of
    /// the round loop. Reported, not failed: a gate that turns red on an input
    /// it cannot read teaches its operator to ignore it.
    Unjudgeable(String),
    /// The round loop did not reproduce plain greedy.
    Refused(String),
}

/// Judge one pair of streams.
///
/// `margins` is the reference arm's per-position top-two logprob gap, or empty
/// when the caller has none — the synthetic fixtures below run that way, and the
/// divergence oracle stands down rather than inventing a distribution.
///
/// `restriction` is `None` for every pair whose loop scores each position over
/// the verifier's whole vocabulary, which is all of them but one.
fn judge(
    spec: &[u32],
    plain: &[u32],
    margins: &[f32],
    restriction: Option<&Restriction<'_>>,
) -> Verdict {
    let shorter = spec.len().min(plain.len());
    let longer = spec.len().max(plain.len());

    // The controls first. A degenerate arm is a more specific verdict than a
    // short one, and a collapse can be what cut the run short.
    let (plain_from, plain_period, plain_cycle) = strongest_windowed_cycle(plain);
    if plain_cycle > MAX_CYCLE_FRACTION {
        return Verdict::Unjudgeable(format!(
            "the reference arm repeats at period {plain_period} across {plain_cycle:.4} of \
             its tokens from {plain_from} on (ceiling {MAX_CYCLE_FRACTION}) — plain greedy \
             is the control here, so this says the prompt did not come back as prose the \
             control can read"
        ));
    }
    let (spec_from, spec_period, spec_cycle) = strongest_windowed_cycle(spec);
    if spec_cycle > MAX_CYCLE_FRACTION {
        return Verdict::Refused(format!(
            "the speculative arm repeats at period {spec_period} across {spec_cycle:.4} of \
             its tokens from {spec_from} on (ceiling {MAX_CYCLE_FRACTION}) while the \
             reference arm reads {plain_cycle:.4} — it has collapsed into a repetition \
             loop the verifier does not produce on its own"
        ));
    }

    if longer < MIN_ANSWER_TOKENS {
        return Verdict::Unjudgeable(format!(
            "both arms answered in {} and {} tokens, under {MIN_ANSWER_TOKENS} — the \
             prompt produced no answer to compare on this model, which is not a \
             statement about the round loop",
            spec.len(),
            plain.len()
        ));
    }
    if plain.len() < MIN_ANSWER_TOKENS {
        // The reference arm is the control, and it answered this prompt in
        // fewer tokens than the gate can read. One shape is still about the
        // round loop: the speculative arm gave the whole reference answer back
        // and then carried on, which says it did not stop where the verifier
        // stopped. An arm that parted from the reference *inside* that answer
        // and then wrote a longer one is a different answer, not a longer
        // version of this one, and there is no answer here to judge it against.
        if spec.starts_with(plain) {
            return Verdict::Refused(format!(
                "the reference arm ended at {} tokens and the speculative arm reproduced \
                 it and ran on to {} — the round loop did not stop where the verifier \
                 stopped",
                plain.len(),
                spec.len()
            ));
        }
        return Verdict::Unjudgeable(format!(
            "the reference arm answered in {} tokens, under {MIN_ANSWER_TOKENS}, and the \
             arms parted inside that answer — plain greedy is the control, so what this \
             says is that the prompt produced no answer to compare on this model, \
             whatever the speculative arm went on to write",
            plain.len()
        ));
    }
    if spec.len() < MIN_ANSWER_TOKENS {
        return Verdict::Refused(format!(
            "one arm stopped early — spec={} plain={}, under {MIN_ANSWER_TOKENS} while \
             the other ran on; a short run is a comparison with no power, not a pass",
            spec.len(),
            plain.len()
        ));
    }
    let length_ratio = shorter as f64 / longer as f64;
    if length_ratio < MIN_LENGTH_RATIO {
        return Verdict::Refused(format!(
            "the arms answered at {} and {} tokens, a ratio of {length_ratio:.4} (floor \
             {MIN_LENGTH_RATIO}) — one stopped well before the other, and there is no \
             divergence to read where the shorter arm has already ended",
            spec.len(),
            plain.len()
        ));
    }

    if let Some((at, confidence)) = divergence_confidence(spec, plain, margins) {
        if confidence > MAX_DIVERGENCE_CONFIDENCE {
            if restriction.is_some_and(|r| r.explains(at, plain)) {
                return Verdict::Unjudgeable(format!(
                    "the arms first differ at token {at}, which the round loop emitted from \
                     the drafter's own argmax and where the verifier's answer is a token \
                     the drafter's vocabulary cannot name — the declared restricted-\
                     vocabulary boundary parted them and no correct loop of this kind could \
                     have said anything else there, so this prompt says nothing about the \
                     round loop"
                ));
            }
            return Verdict::Refused(format!(
                "the arms first differ at token {at}, where the verifier was surer than \
                 {confidence:.4} of its own decisions on this answer (ceiling \
                 {MAX_DIVERGENCE_CONFIDENCE}) — a different reduction order can only flip \
                 a decision that was nearly tied, so the round loop fed the verifier a \
                 different state rather than the same one in a different order"
            ));
        }
    }
    Verdict::Agreed
}

impl Verdict {
    /// The refusal text, or `None` for anything the gate does not fail on.
    fn refusal(&self) -> Option<String> {
        match self {
            Self::Refused(why) => Some(why.clone()),
            Self::Agreed | Self::Unjudgeable(_) => None,
        }
    }
}

// ── Oracle tests (no model, no GPU) ──────────────────────────────────────────

/// A deterministic stand-in for token streams, seeded per case.
///
/// A plain `t % 97` stream was the previous control: 97 is prime and above the
/// period ceiling, so it reads exactly 0.0000 at every period the sweep looks at
/// and could not detect a control that fires on real output. These shapes can.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        // xorshift64*, so the fixtures are reproducible without a dependency.
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    fn below(&mut self, n: u32) -> u32 {
        (self.next() % u64::from(n)) as u32
    }

    /// Word ids drawn from a Zipf-ish distribution, as prose is.
    fn prose(&mut self, len: usize) -> Vec<u32> {
        (0..len)
            .map(|_| {
                let r = self.below(1000);
                r % (1 + r / 8)
            })
            .collect()
    }
}

/// One near-tie flip that both arms write around is the shipped behaviour and
/// must pass; a flip at a decision the verifier was sure about must not.
///
/// The two streams differ by the same single token. Only the reference arm's own
/// margins separate them, which is the whole design: the gate judges the pair
/// against what the verifier itself found difficult.
#[test]
fn a_single_flip_passes_and_a_confident_flip_does_not() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let mut one_flip = plain.clone();
    one_flip[66] = 9999;

    let mut tied = vec![5.0f32; N_TOKENS];
    tied[66] = 0.01;
    assert!(
        judge(&one_flip, &plain, &tied, None).refusal().is_none(),
        "a flip at the arm's own least-confident decision must pass"
    );

    let mut confident = vec![0.01f32; N_TOKENS];
    for m in confident.iter_mut().take(N_TOKENS / 3) {
        *m = 0.001;
    }
    confident[66] = 5.0;
    let failure = judge(&one_flip, &plain, &confident, None)
        .refusal()
        .expect("must refuse");
    assert!(failure.contains("nearly tied"), "{failure}");
}

/// What the divergence oracle reads, and when it stands down.
#[test]
fn the_divergence_oracle_reads_the_first_differing_position_or_nothing() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let mut flipped = plain.clone();
    flipped[66] = 9999;

    let mut tied = vec![5.0f32; N_TOKENS];
    tied[66] = 0.01;
    assert_eq!(
        divergence_confidence(&flipped, &plain, &tied),
        Some((66, 0.0)),
        "the position read is the first the arms differ at"
    );

    // Two identical arms have no divergence to judge, and an empty margin list
    // stands the oracle down rather than reading position 0 of nothing.
    assert_eq!(divergence_confidence(&plain, &plain, &tied), None);
    assert_eq!(divergence_confidence(&flipped, &plain, &[]), None);

    // A divergence past the end of the reference arm has no margin behind it;
    // the length guard owns that pair.
    let extended: Vec<u32> = plain.iter().copied().chain([1_u32]).collect();
    assert_eq!(divergence_confidence(&extended, &plain, &tied), None);
}

/// One pair of arms parting at token 66, at a decision the verifier was sure
/// about, with everything the restricted-vocabulary rule reads left to the
/// caller: the reference token there, and which vocabulary decided the
/// speculative one.
///
/// Every case below is this pair with one of those two moved, so the verdict is
/// the only thing that can differ between them.
fn restricted_case(
    reference_token_nameable: bool,
    decided: DecidedBy,
) -> (
    Vec<u32>,
    Vec<u32>,
    Vec<f32>,
    std::collections::HashSet<u32>,
    Vec<DecidedBy>,
) {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let mut spec = plain.clone();
    spec[66] = 9999;

    // Sure enough at token 66 that the confidence oracle refuses it: every other
    // decision in the arm was a closer call than this one.
    let mut margins = vec![0.01f32; N_TOKENS];
    for m in margins.iter_mut().take(N_TOKENS / 3) {
        *m = 0.001;
    }
    margins[66] = 5.0;

    // The drafter names every id in play except, when the case asks for it, the
    // reference arm's own token at the divergence. Two other reference tokens
    // are unnameable throughout, so a rule that read the answer rather than the
    // divergence would not tell the cases apart.
    let mut vocab: std::collections::HashSet<u32> = plain.iter().copied().collect();
    vocab.insert(9999);
    vocab.remove(&12);
    vocab.remove(&200);
    if !reference_token_nameable {
        vocab.remove(&66);
    }

    let mut decided_by = vec![DecidedBy::FullVocab; N_TOKENS];
    decided_by[66] = decided;

    (spec, plain, margins, vocab, decided_by)
}

/// The declared boundary: the loop emitted the drafter's own argmax at the
/// position the arms parted on, and the verifier's answer there is a token that
/// vocabulary cannot say. No correct loop of this kind could have said anything
/// else, so the prompt is reported rather than refused.
#[test]
fn a_divergence_the_restricted_vocabulary_forced_is_not_a_refusal() {
    let (spec, plain, margins, vocab, decided_by) =
        restricted_case(false, DecidedBy::RestrictedVocab);
    let verdict = judge(
        &spec,
        &plain,
        &margins,
        Some(&Restriction {
            vocab: &vocab,
            decided_by: &decided_by,
        }),
    );
    assert!(verdict.refusal().is_none(), "{verdict:?}");
    let Verdict::Unjudgeable(why) = &verdict else {
        panic!("the boundary is not a statement about the round loop: {verdict:?}");
    };
    assert!(why.contains("cannot name"), "{why}");
    assert!(why.contains("says nothing about the round loop"), "{why}");
}

/// **The mutation this rule has to survive.** Widening the same inexactness to
/// the round's correction changes an answer at exactly the kind of token the
/// boundary does — same reference token, same speculative token, same
/// confidence — and the only thing that separates the two is the position the
/// loop was at. A rule keyed on the token alone forgives it; this one refuses
/// it, for the reason it refuses any other pair.
#[test]
fn the_same_divergence_at_the_correction_is_refused() {
    let (spec, plain, margins, vocab, decided_by) = restricted_case(false, DecidedBy::FullVocab);
    let failure = judge(
        &spec,
        &plain,
        &margins,
        Some(&Restriction {
            vocab: &vocab,
            decided_by: &decided_by,
        }),
    )
    .refusal()
    .expect("the correction reads the whole vocabulary, so the restriction cannot excuse it");
    assert!(failure.contains("nearly tied"), "{failure}");
}

/// A divergence at a token the drafter could have named is judged the old way,
/// wherever the loop was: the restricted argmax and the true one are the same
/// token on the ids the drafter can name, so the restriction explains nothing
/// there.
#[test]
fn a_divergence_the_drafter_could_have_named_is_judged_as_before() {
    for decided in [DecidedBy::RestrictedVocab, DecidedBy::FullVocab] {
        let (spec, plain, margins, vocab, decided_by) = restricted_case(true, decided);
        let failure = judge(
            &spec,
            &plain,
            &margins,
            Some(&Restriction {
                vocab: &vocab,
                decided_by: &decided_by,
            }),
        )
        .refusal()
        .unwrap_or_else(|| panic!("a nameable divergence must still refuse, at {decided:?}"));
        assert!(failure.contains("nearly tied"), "{failure}");

        // And a pair that declares no reduced vocabulary reads the same pair the
        // same way, which is what "judged as before" means.
        let plain_rule = judge(&spec, &plain, &margins, None)
            .refusal()
            .expect("no restriction, no waiver");
        assert_eq!(plain_rule, failure);
    }
}

/// The waiver is scoped to the divergence oracle and reaches neither of the
/// other two verdicts. A speculative arm that collapsed, or one that stopped
/// early, is refused whatever the restriction would have said about where the
/// arms part.
#[test]
fn the_boundary_excuses_no_other_refusal() {
    let (_, plain, margins, vocab, decided_by) = restricted_case(false, DecidedBy::RestrictedVocab);
    let restriction = Restriction {
        vocab: &vocab,
        decided_by: &decided_by,
    };

    let mut collapsed: Vec<u32> = plain[..66].to_vec();
    collapsed.extend(std::iter::repeat_n([7_u32, 8].into_iter(), N_TOKENS).flatten());
    collapsed.truncate(N_TOKENS);
    let failure = judge(&collapsed, &plain, &margins, Some(&restriction))
        .refusal()
        .expect("a collapsed speculative arm is refused whatever parted the arms");
    assert!(failure.contains("repetition loop"), "{failure}");

    let failure = judge(&plain[..16], &plain, &margins, Some(&restriction))
        .refusal()
        .expect("a short speculative arm is refused whatever parted the arms");
    assert!(failure.contains("stopped early"), "{failure}");
}

/// The oracle's own denominator: the confidence is a rank over the reference
/// arm's margins, so it must not move when every margin is scaled.
///
/// A threshold in nats would move. This is the property that lets one ceiling
/// cover two models whose logits are not on the same scale.
#[test]
fn the_confidence_is_a_rank_and_not_a_number_of_nats() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let mut flipped = plain.clone();
    flipped[40] = 7777;

    let base: Vec<f32> = (0..N_TOKENS).map(|i| 0.1 + (i % 17) as f32).collect();
    let scaled: Vec<f32> = base.iter().map(|m| m * 250.0).collect();
    assert_eq!(
        divergence_confidence(&flipped, &plain, &base),
        divergence_confidence(&flipped, &plain, &scaled),
        "scaling every margin must not move the rank"
    );
}

/// Two arms stuck in the same repetition loop agree perfectly and mean nothing.
/// The control is what separates that from a real match — for a loop that
/// starts at the first token, one that starts part-way through, and one whose
/// cycle is longer than a single token. A leading-run measure scores the second
/// and third at zero and lets both through.
#[test]
fn every_shape_of_repetition_loop_is_refused() {
    let healthy: Vec<u32> = (0..N_TOKENS as u32).collect();
    for (shape, stream) in [
        ("from the first token", vec![7u32; N_TOKENS]),
        (
            "beginning at token 20",
            healthy[..20]
                .iter()
                .copied()
                .chain(std::iter::repeat_n(7u32, N_TOKENS - 20))
                .collect(),
        ),
        (
            "a two-token cycle",
            (0..N_TOKENS)
                .map(|i| if i % 2 == 0 { 7 } else { 9 })
                .collect(),
        ),
        (
            "a three-token cycle beginning at token 40",
            healthy[..40]
                .iter()
                .copied()
                .chain((0..N_TOKENS - 40).map(|i| [7u32, 9, 11][i % 3]))
                .collect(),
        ),
        // Over the whole stream this reads 0.3992 at period 1 — under the
        // ceiling, because the healthy three fifths dilute it. Its own tail
        // window is where it is visible.
        (
            "a collapse confined to the last two fifths",
            healthy[..N_TOKENS * 3 / 5]
                .iter()
                .copied()
                .chain(std::iter::repeat_n(7u32, N_TOKENS - N_TOKENS * 3 / 5))
                .collect(),
        ),
        // A repeated sentence is the commonest real degeneracy and reads
        // 0.0000 at every period under its own length.
        (
            "a twelve-token phrase over four fifths",
            healthy[..N_TOKENS / 5]
                .iter()
                .copied()
                .chain((0..N_TOKENS - N_TOKENS / 5).map(|i| 1000 + (i % 12) as u32))
                .collect(),
        ),
    ] {
        let failure = judge(&stream, &healthy, &[], None)
            .refusal()
            .unwrap_or_else(|| panic!("{shape} was not refused"));
        assert!(failure.contains("repeats at period"), "{shape}: {failure}");
    }
}

/// Which arm collapsed decides what the gate is entitled to say.
///
/// A degenerate speculative arm against a healthy reference is a verdict about
/// the round loop. A degenerate *reference* arm is a verdict about the input:
/// plain greedy is the control, and a control the measure cannot read leaves
/// nothing to compare against. Naming it as an engine defect there was a false
/// accusation the gate used to make on any prompt answered with a list.
#[test]
fn a_collapsed_reference_arm_is_reported_as_an_input_the_gate_cannot_judge() {
    let healthy: Vec<u32> = (0..N_TOKENS as u32).collect();
    let looping = vec![7u32; N_TOKENS];

    let spec_side = judge(&looping, &healthy, &[], None);
    let why = spec_side
        .refusal()
        .expect("a degenerate spec arm must be refused");
    assert!(why.contains("speculative arm"), "{why}");
    assert!(
        why.contains("the verifier does not produce on its own"),
        "{why}"
    );

    let plain_side = judge(&healthy, &looping, &[], None);
    assert!(
        matches!(plain_side, Verdict::Unjudgeable(ref why) if why.contains("reference arm")),
        "a collapsed reference arm is an input the gate cannot read, not a \
         refusal of the round loop: {plain_side:?}"
    );
}

/// A prompt neither arm answered is not a failure of the round loop, and a
/// prompt only one arm answered is.
#[test]
fn a_prompt_that_produced_no_answer_is_reported_rather_than_failed() {
    let short: Vec<u32> = Rng(0x5170_0000).prose(MIN_ANSWER_TOKENS - 1);
    assert!(
        matches!(judge(&short, &short, &[], None), Verdict::Unjudgeable(ref why) if why.contains("both arms")),
        "two arms that both stopped short say nothing about the round loop"
    );

    let long: Vec<u32> = Rng(0x5170_0000).prose(N_TOKENS);
    let why = judge(&short, &long, &[], None)
        .refusal()
        .expect("one short arm against one long one must be refused");
    assert!(why.contains("one arm stopped early"), "{why}");
}

/// Structured output trips this control, and that is why the prompts forbid it.
///
/// A markdown table repeats its delimiters every row and a numbered list its
/// `N. **` prefix every item. On this measure they read far above prose and
/// above any ceiling that still refuses a ragged loop — so the control is not a
/// general degeneracy classifier and this test records that rather than
/// asserting it away. [`PROSE_INSTRUCTION`] is what keeps these shapes out of
/// the arms the gate actually judges; if an answer ever comes back structured,
/// the gate refuses it as an input it cannot judge, which is this limit and not
/// a defect in the engine.
#[test]
fn structured_output_trips_the_control_which_is_why_the_prompts_forbid_it() {
    let mut rng = Rng(0x5EED_5EED);
    let mut table = Vec::new();
    while table.len() < N_TOKENS {
        // `| ` name `| ` value `|` newline — two of six tokens vary.
        table.extend_from_slice(&[900, 901, rng.below(50), 902, rng.below(50), 903]);
    }
    let mut boolean_table = Vec::new();
    while boolean_table.len() < N_TOKENS {
        // The same shape with a yes/no column: lower entropy, higher reading.
        boolean_table.extend_from_slice(&[900, 901, rng.below(2), 902, rng.below(2), 903]);
    }
    let mut numbered = Vec::new();
    while numbered.len() < N_TOKENS {
        numbered.extend_from_slice(&[910, 911, 912]);
        numbered.extend((0..3).map(|_| rng.below(80)));
        numbered.extend_from_slice(&[913, 914]);
    }
    for (shape, stream) in [
        ("a markdown table", table[..N_TOKENS].to_vec()),
        (
            "a table with a yes/no column",
            boolean_table[..N_TOKENS].to_vec(),
        ),
        ("a numbered list", numbered[..N_TOKENS].to_vec()),
    ] {
        let (_, _, fraction) = strongest_windowed_cycle(&stream);
        assert!(
            fraction > MAX_CYCLE_FRACTION,
            "{shape} reads {fraction:.4}, at or under the ceiling — if structured \
             output has stopped tripping this control the module docs are wrong \
             about why the prompts are what they are"
        );
    }
}

/// Prose does not, at any length the gate can hand it, and that is the
/// population the ceiling is set against.
#[test]
fn prose_clears_the_control_at_every_length_the_gate_can_hand_it() {
    let mut worst = 0.0f64;
    for len in [MIN_ANSWER_TOKENS, 200, 218, 256] {
        for seed in 0..64u64 {
            let stream = Rng(0x1234_0000 + seed).prose(len);
            let (start, period, fraction) = strongest_windowed_cycle(&stream);
            assert!(
                fraction <= MAX_CYCLE_FRACTION,
                "healthy prose of {len} tokens (seed {seed}) read {fraction:.4} at \
                 period {period} from {start}"
            );
            worst = worst.max(fraction);
        }
    }
    assert!(
        worst * 1.5 < MAX_CYCLE_FRACTION,
        "the margin over healthy prose has fallen to {worst:.4} against a ceiling \
         of {MAX_CYCLE_FRACTION}; the real arms measure 0.05 to 0.10 and the \
         1000-stream sweep peaks at 0.1351"
    );
}

/// The one bound that makes a reading mean something.
///
/// `strongest_cycle` divides by `len - period`. Without a floor on that
/// denominator a reading can be one coincidental comparison out of one, which
/// is what made the control fire on healthy output. A bound on the period
/// relative to the window was tried alongside it and removed: it changed no
/// false-positive rate and blinded the sweep to real collapses.
#[test]
#[allow(
    clippy::float_cmp,
    reason = "the assertion is that no reading was taken at all, which is \
              exactly zero; a band would accept a reading from too few samples"
)]
fn a_cycle_too_short_a_window_to_evidence_is_not_read() {
    // One coincidence at distance 63 in a 64-token window: one comparison.
    let mut coincidence: Vec<u32> = (0..64).collect();
    coincidence[63] = 0;
    assert_eq!(
        strongest_cycle(&coincidence).1,
        0.0,
        "a reading from a single comparison must not be taken"
    );

    // An exact period of 10 in 40 tokens: 30 comparisons, under the floor.
    let short_window: Vec<u32> = (0..40u32).map(|i| i % 10).collect();
    assert_eq!(
        strongest_cycle(&short_window).1,
        0.0,
        "a reading from fewer than the sample floor must not be taken"
    );

    // The floor is a floor on evidence, not a blanket. 140 comparisons all
    // matching at period 60 is overwhelming, and the quarter-window bound used
    // to score it 0.0000 — the case that showed the bound cost real detection.
    let long_period: Vec<u32> = (0..200u32).map(|i| i % 60).collect();
    assert!(
        strongest_cycle(&long_period).1 > MAX_CYCLE_FRACTION,
        "an exact period-60 cycle over 140 comparisons must be read, not bounded away"
    );

    let supported: Vec<u32> = (0..N_TOKENS as u32).map(|i| i % 10).collect();
    assert!(
        strongest_cycle(&supported).1 > MAX_CYCLE_FRACTION,
        "an exact cycle a full window supports must still be caught"
    );
}

/// The false-positive rate of the control on healthy output, per length.
///
/// Not a gate: it prints. Three rounds of review disagreed about this rate by a
/// factor of three to five because the measuring code was never committed and
/// the surviving tests could only observe the "after" state, so nothing in the
/// tree could adjudicate. This is that code, and the numbers the module docs
/// quote are the ones it prints.
///
/// `#[ignore]` because it is a measurement over thousands of streams and says
/// nothing about correctness;
/// `prose_clears_the_control_at_every_length_the_gate_can_hand_it` is the
/// assertion. Run it with `--ignored --nocapture`; it reaches no device.
#[ignore = "measurement, not an assertion: prints the false-positive rate per length"]
#[test]
fn the_false_positive_rate_on_healthy_output() {
    const TRIALS: u64 = 1000;
    println!(
        "healthy-prose false positives, {TRIALS} streams per length, ceiling {MAX_CYCLE_FRACTION}"
    );
    for len in [40usize, 120, 200, 218, 256, 512] {
        let mut trips = 0u32;
        let mut worst = 0.0f64;
        for seed in 0..TRIALS {
            let f = strongest_windowed_cycle(&Rng(0x9E37_0000 + seed).prose(len)).2;
            worst = worst.max(f);
            if f > MAX_CYCLE_FRACTION {
                trips += 1;
            }
        }
        println!(
            "  len={len:4}  trips={trips:4}/{TRIALS}  rate={:.3}%  max reading={worst:.4}",
            f64::from(trips) * 100.0 / TRIALS as f64
        );
    }
}

/// Two arms sharing a real prefix and then locked in the same period-8 loop,
/// **swept** over the whole raggedness range rather than sampled at points.
///
/// This is the regime with no reference to appeal to: both arms are degenerate,
/// so their agreement means nothing. The margins are empty on purpose, so the
/// repetition control is the only oracle in play and the sweep reads it alone.
///
/// It refuses every pair up to [`FIRST_ADMITTED`]% raggedness, and past that the
/// arms are more noise than loop and it admits them. **The tail measure does not
/// take over there**: what the control admits agrees up to
/// [`WORST_ADMITTED_TAIL`], twice the worst reading a correct pair reached. That
/// is the "no subsequence floor" claim as a measurement rather than an
/// assertion, and it is why [`WORST_CORRECT_TAIL_AGREEMENT`] is a recorded
/// figure and not a threshold.
///
/// Both edges are pinned, so the sweep fails when either moves. An earlier
/// revision claimed the range was covered with no gap and sampled six values of
/// the parameter to say so; at 36% the pair passed. A later one closed the gap
/// against a worst-correct figure measured over two pairs — four more pairs took
/// that figure from 0.2344 to 0.1094 and the gap opened again, which is the
/// state recorded here.
#[test]
fn two_arms_in_the_same_ragged_loop_are_refused_until_they_are_no_longer_one_loop() {
    /// Raggedness, in per cent, of the first pair the control admits.
    const FIRST_ADMITTED: u64 = 60;
    /// The best tail agreement any admitted pair reached.
    const WORST_ADMITTED_TAIL: f64 = 0.2188;

    let mut worst_admitted = 0.0f64;
    let mut first_admitted = None;
    for noise in (0..=100).step_by(2) {
        for (sa, sb) in [(0x33u64, 0x44u64), (0x91, 0xA7), (0xB3, 0xC1), (0xD5, 0xE9)] {
            let (a, b) = (ragged_loop_arm(sa, noise), ragged_loop_arm(sb, noise));
            if judge(&a, &b, &[], None) != Verdict::Agreed {
                continue;
            }
            first_admitted = first_admitted.or(Some(noise));
            worst_admitted = worst_admitted.max(weakest_tail(&a, &b).1);
        }
    }
    assert_eq!(
        first_admitted,
        Some(FIRST_ADMITTED),
        "the control admits its first ragged pair at {first_admitted:?}% raggedness, \
         not the recorded {FIRST_ADMITTED}%"
    );
    assert!(
        (worst_admitted - WORST_ADMITTED_TAIL).abs() < 1e-4,
        "the pairs the control admits agree up to {worst_admitted:.4}, not the recorded \
         {WORST_ADMITTED_TAIL}"
    );
    assert!(
        worst_admitted > WORST_CORRECT_TAIL_AGREEMENT,
        "what the control admits agrees no better than the worst correct pair \
         ({WORST_CORRECT_TAIL_AGREEMENT}), so a subsequence floor would separate the \
         two populations after all and this file should have one"
    );
}

/// [`WORST_CORRECT_TAIL_AGREEMENT`] is the worst of the tails actually measured,
/// and this is where that is checked.
///
/// Its only other use is an upper bound, so **raising** it weakens the ragged
/// sweep above toward vacuity with every test still green — the same
/// one-directional asymmetry the divergence ceiling's pin used to have. Holding
/// it to the population it names fails in both directions.
#[test]
fn the_worst_correct_tail_is_the_worst_of_the_tails_measured() {
    /// Every [`weakest_tail`] reading a correct pair reached, over all six pairs
    /// and the prompts each of them judged. These are tails, not [`lcs_ratio`]
    /// over the whole arm — the same runs read 0.4766 to 1.0000 on that measure,
    /// and the two figures the "no subsequence floor" paragraphs quote come from
    /// it.
    const MEASURED: &[f64] = &[
        // Assistant pair, six prompts.
        0.3125, 0.3750, 0.3906, 0.2344, 1.0000, 0.9062,
        // Recurrent pair, the five prompts it judged — the 4k document is
        // answered in 13 and 26 tokens and is reported unjudgeable.
        0.3281, 1.0000, 1.0000, 0.6354, 1.0000,
        // Block pair, six prompts. The worst reading of the whole population is
        // here, and it was outside this list until the pairs below were built.
        0.4323, 0.4844, 0.3021, 0.6562, 0.1094, 0.5938,
        // Adaptive pair, the five it judged — its verifier answers the 4k
        // document in 52 tokens, and both arms reproduce that answer exactly.
        0.2969, 0.2188, 0.4115, 0.3906, 0.2656,
        // Restricted-vocabulary pair, the five it judged — same verifier, and on
        // the 4k document its arms part at the one lowest-margin token in that
        // 52-token answer.
        0.1406, 0.2656, 0.3594, 0.4844, 0.3984,
        // Two-model pair, the five it judged. Its verifier is the recurrent
        // pair's, and on this measure the two loops read alike.
        0.3281, 1.0000, 1.0000, 0.6354, 1.0000,
    ];
    let worst = MEASURED.iter().copied().fold(f64::INFINITY, f64::min);
    assert!(
        (worst - WORST_CORRECT_TAIL_AGREEMENT).abs() < 1e-9,
        "the constant reads {WORST_CORRECT_TAIL_AGREEMENT:.4} and the worst tail a \
         correct pair reached is {worst:.4}"
    );
}

/// Half a healthy prefix, then a period-8 loop `noise` percent of whose tokens
/// are drawn at random instead.
fn ragged_loop_arm(seed: u64, noise: u64) -> Vec<u32> {
    let head = Rng(0xFEED).prose(N_TOKENS / 2);
    let mut rng = Rng(seed);
    let mut out = head;
    for i in 0..N_TOKENS / 2 {
        out.push(if u64::from(rng.below(100)) < noise {
            rng.below(300)
        } else {
            1000 + (i % 8) as u32
        });
    }
    out
}

/// A collapse over the last quarter of a long arm is what the windowed sweep was
/// added for, and a speculative arm doing it must be refused **by the control**,
/// which is the claim, rather than by whichever oracle happens to fire.
#[test]
fn a_speculative_arm_collapsing_over_its_last_quarter_is_refused_by_the_control() {
    let healthy = Rng(0xC0DE).prose(N_TOKENS);
    for period in [8usize, 16, 24] {
        let spec = quarter_collapse(0x55, period);
        let failure = judge(&spec, &healthy, &[], None)
            .refusal()
            .unwrap_or_else(|| {
                panic!(
                    "period {period}: an arm collapsing over its last quarter passed: \
                     cycle {:.4}",
                    strongest_windowed_cycle(&spec).2,
                )
            });
        assert!(
            failure.contains("repeats at period"),
            "period {period}: the control must be what refuses this — {failure}"
        );
    }
}

/// The control's declared blind spot, asserted rather than assumed.
///
/// `strongest_cycle` needs [`MIN_CYCLE_SAMPLES`] comparisons before it reads
/// anything, so the narrowest window — the last `1 / TAIL_WINDOWS` of an arm —
/// can evidence no period above `len / TAIL_WINDOWS - MIN_CYCLE_SAMPLES`, which
/// at this budget is 32. Past that the reading comes from a wider window the
/// collapse only partly fills and it decays: a last-quarter loop reads 0.8750 at
/// period 32, 0.2500 at 40, 0.1875 at 48. The ceiling is crossed between 40 and
/// 48, and that is the blind spot — pinned from both sides so it fails when it
/// moves rather than when someone notices.
#[test]
fn a_cycle_confined_to_a_window_too_narrow_to_read_it_is_a_declared_blind_spot() {
    let readable = N_TOKENS / TAIL_WINDOWS - MIN_CYCLE_SAMPLES;
    assert_eq!(
        readable, 32,
        "the bound the blind spot is stated in terms of"
    );

    let (start, _, inside) = strongest_windowed_cycle(&quarter_collapse(0x55, readable));
    assert!(
        inside > MAX_CYCLE_FRACTION && start == N_TOKENS * 3 / 4,
        "a period the last window can evidence must be read in that window; \
         it read {inside:.4} from {start}"
    );

    let caught = strongest_windowed_cycle(&quarter_collapse(0x55, 40)).2;
    assert!(
        caught > MAX_CYCLE_FRACTION,
        "a period-40 last-quarter collapse reads {caught:.4} from a wider window \
         and must still be caught"
    );
    let missed = strongest_windowed_cycle(&quarter_collapse(0x55, 48)).2;
    assert!(
        missed <= MAX_CYCLE_FRACTION,
        "a period-48 last-quarter collapse reads {missed:.4} and is caught; the \
         blind spot has moved and the docs no longer describe it"
    );

    // And it is the window, not the shape: the same period over four fifths of
    // the arm fills a window wide enough to read it.
    let mut wide = Rng(0xC0DE).prose(N_TOKENS / 5);
    let mut rng = Rng(0x55);
    for i in 0..N_TOKENS - N_TOKENS / 5 {
        wide.push(if rng.below(100) < 5 {
            rng.below(300)
        } else {
            1000 + (i % 48) as u32
        });
    }
    assert!(
        strongest_windowed_cycle(&wide).2 > MAX_CYCLE_FRACTION,
        "the same period over four fifths of the arm must still be caught"
    );
}

/// Three quarters of healthy prose, then a loop at `period` with 5% noise.
fn quarter_collapse(seed: u64, period: usize) -> Vec<u32> {
    let mut out = Rng(0xC0DE).prose(N_TOKENS * 3 / 4);
    let mut rng = Rng(seed);
    for i in 0..N_TOKENS - N_TOKENS * 3 / 4 {
        out.push(if rng.below(100) < 5 {
            rng.below(300)
        } else {
            1000 + (i % period) as u32
        });
    }
    out
}

/// The subsequence ratio is taken over the shorter arm, so an arm that stopped
/// early and matched the other's prefix scores 1.0 and says nothing about the
/// tail it never wrote. The length guard is the only thing between that and a
/// green gate, and this pins both halves: the denominator and the guard.
#[test]
fn the_ratio_is_over_the_shorter_arm_and_the_length_guard_covers_it() {
    // Twice the budget, so the *ratio* is what fires rather than the floor.
    let plain: Vec<u32> = Rng(0x1E17_0000).prose(N_TOKENS * 2);

    let truncated = &plain[..N_TOKENS];
    assert!(
        (lcs_ratio(truncated, &plain) - 1.0).abs() < 1e-9,
        "a true prefix must score 1.0 over the shorter arm; it read {}",
        lcs_ratio(truncated, &plain)
    );
    let failure = judge(truncated, &plain, &[], None)
        .refusal()
        .expect("the length guard must refuse it");
    assert!(failure.contains("stopped well before"), "{failure}");

    // Just inside the guard the same shape scores 1.0 and passes, which is what
    // makes the guard — not the ratio — the thing doing the work above.
    assert!(judge(&plain[..N_TOKENS * 3 / 2], &plain, &[], None)
        .refusal()
        .is_none());
}

/// The length floor is pinned in both directions, and it is under every answer a
/// correct pair produced.
///
/// A floor a real answer cannot reach refuses a *fixed* engine and no reproducer
/// could ever flip to green; a floor nothing can fail is free.
#[test]
fn the_length_floor_admits_every_answer_the_prompts_produce() {
    let at_floor = Rng(0x00F1_7EDD).prose(MIN_ANSWER_TOKENS);
    assert!(
        judge(&at_floor, &at_floor, &[], None).refusal().is_none(),
        "two arms exactly at the floor must pass"
    );
    let under = &at_floor[..MIN_ANSWER_TOKENS - 1];
    assert!(
        judge(under, under, &[], None) != Verdict::Agreed,
        "two arms one token under the floor must not be returned as agreement"
    );

    // Every arm the gate judged ran to the budget, so the floor is under all of
    // them. Driven through `judge` rather than compared as constants.
    let budget = Rng(0x2181_2181).prose(N_TOKENS);
    assert!(
        judge(&budget, &budget, &[], None).refusal().is_none(),
        "a pair that answers in full must clear the floor"
    );
    // Both sides of the floor, at compile time, and each is the tightest form
    // its evidence supports. The band they leave is [160, 165] and the shipped
    // floor sits on its lower limit.
    const {
        // The upper side, and it is not merely the budget. A correct pair can
        // answer and stop well before the budget: the assistant pair run at a
        // 512-token budget over these prompts stops the reference arm on the 4k
        // document at 330, a fraction of 330/512. A floor that only cleared
        // `N_TOKENS` would class a pair that answered and stopped as a prompt
        // that produced nothing, driving `judged` to zero and failing
        // `run_gate`'s own guard for the wrong reason. The measured fraction is
        // asserted rather than a rounder one above it.
        assert!(MIN_ANSWER_TOKENS * 512 <= N_TOKENS * 330);
        // The lower side: the last tail window is `MIN_ANSWER_TOKENS /
        // TAIL_WINDOWS`, and it has to leave `MIN_CYCLE_SAMPLES` comparisons
        // over and above the period-8 cycle the documented collapse repeats at.
        // The looser `MIN_ANSWER_TOKENS > TAIL_WINDOWS * MIN_CYCLE_SAMPLES`
        // admits 129 and rejects nothing this does not.
        assert!(MIN_ANSWER_TOKENS / TAIL_WINDOWS >= MIN_CYCLE_SAMPLES + 8);
    }
}

/// A run that stopped short is refused rather than passed on its short prefix.
#[test]
fn a_truncated_run_is_refused_rather_than_judged() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let failure = judge(&plain[..16], &plain, &[], None)
        .refusal()
        .expect("must refuse");
    assert!(failure.contains("stopped early"), "{failure}");
}

/// Which arm is short decides what the gate is entitled to say, and all three
/// shapes are pinned.
///
/// A *speculative* arm under the floor while the reference ran on is the round
/// loop cutting its own run, and is refused. A *reference* arm under the floor
/// is the prompt — plain greedy is the control, and the same asymmetry the
/// repetition control already applies is what the divergence oracle needs here.
/// The one shape that is still about the loop is a speculative arm that
/// reproduced the whole reference answer and then carried on: that is a loop
/// that did not stop where the verifier stopped, and it is separable from a loop
/// that parted inside the answer and wrote a different, longer one.
///
/// This is the case the restricted-vocabulary pair brought: on its verifier the
/// 4k document is answered in 52 tokens, and two other round loops reproduce
/// that answer exactly while EAGLE-3 flips the single lowest-margin token in it
/// and writes on.
#[test]
fn a_short_reference_arm_is_the_prompt_unless_the_other_arm_only_ran_past_it() {
    let plain: Vec<u32> = Rng(0x5AFE_5AFE).prose(MIN_ANSWER_TOKENS - 1);

    let mut ran_on = plain.clone();
    ran_on.extend(Rng(0x0BAD_0BAD).prose(N_TOKENS));
    let failure = judge(&ran_on, &plain, &[], None)
        .refusal()
        .expect("an arm that reproduced the whole reference answer and kept going must be refused");
    assert!(
        failure.contains("did not stop where the verifier stopped"),
        "{failure}"
    );

    let mut parted = plain[..4].to_vec();
    parted.extend(Rng(0x0F1F_0F1F).prose(N_TOKENS));
    match judge(&parted, &plain, &[], None) {
        Verdict::Unjudgeable(why) => assert!(why.contains("parted inside that answer"), "{why}"),
        other @ (Verdict::Agreed | Verdict::Refused(_)) => {
            panic!("a reference arm under the floor is the prompt, not the loop: {other:?}")
        }
    }

    let long_plain: Vec<u32> = Rng(0x1C1C_1C1C).prose(N_TOKENS);
    let failure = judge(&long_plain[..16], &long_plain, &[], None)
        .refusal()
        .expect("a speculative arm under the floor while the reference ran on must be refused");
    assert!(failure.contains("stopped early"), "{failure}");
}

/// A divergence that begins in the last quarter and never comes back.
///
/// This is the shape of a regression that fires past a cache-size threshold, and
/// the repository has shipped that class. The subsequence ratio blends it with
/// one benign flip — both land near 0.8 over the whole stream — and the
/// divergence oracle does not: it reads the position the arms first differ at,
/// wherever that is, and asks what the verifier thought there.
#[test]
fn a_late_onset_divergence_is_judged_where_it_begins() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let onset = N_TOKENS * 4 / 5;
    let late: Vec<u32> = plain
        .iter()
        .enumerate()
        .map(|(i, t)| if i >= onset { 50_000 + i as u32 } else { *t })
        .collect();

    assert!(
        lcs_ratio(&late, &plain) > 0.75,
        "this pair must agree over most of the answer, or it is not the class \
         the whole-stream ratio blends"
    );

    let mut confident = vec![0.01f32; N_TOKENS];
    confident[onset] = 5.0;
    let failure = judge(&late, &plain, &confident, None)
        .refusal()
        .expect("must refuse");
    assert!(
        failure.contains(&format!("first differ at token {onset}")),
        "{failure}"
    );

    // The same late divergence at a decision the arm was least sure about is
    // the benign case, and passes at exactly the same subsequence ratio.
    let mut tied = vec![5.0f32; N_TOKENS];
    tied[onset] = 0.01;
    assert!(judge(&late, &plain, &tied, None).refusal().is_none());
}

/// The two measured regimes put through `judge` itself, so the test fails when
/// the oracle's composition changes and not only when a constant moves.
///
/// Correct: the worst reading a shipped pair produced, a first divergence at the
/// 8.2nd percentile of the reference arm's own margins. Broken: the lowest
/// reading a broken engine produced above the ceiling, the 14.5th — the cell the
/// recorded recall stands or falls on.
#[test]
fn the_gate_admits_the_worst_correct_regime_and_refuses_the_lowest_broken_one() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).map(|t| t % 97).collect();
    let mut spec = plain.clone();
    spec[64] = 60_000;

    // The reading is the count of margins under the one at the divergence over
    // the arm's length, so the count is what the regime is reconstructed from —
    // a percentage of it lands on whichever reading integer division reaches,
    // which is how a ceiling under the worst correct cell used to pass here.
    let margins_reading = |below: usize| -> Vec<f32> {
        let mut m: Vec<f32> = (0..N_TOKENS)
            .map(|i| if i < below { 0.01 } else { 5.0 })
            .collect();
        m[64] = 0.02;
        m
    };

    let correct = margins_reading(21);
    let (_, worst_correct) = divergence_confidence(&spec, &plain, &correct).expect("a divergence");
    assert!(
        (worst_correct - 0.0820).abs() < 0.0005,
        "the reconstructed correct regime reads {worst_correct:.4}, not the measured 0.0820"
    );
    assert!(
        judge(&spec, &plain, &correct, None).refusal().is_none(),
        "the worst correct regime must pass"
    );

    // 0.1445 is 37/256 — the two-model pair's rollback mutation on the
    // congestion prompt, whose reference arm ran the full budget.
    let broken = margins_reading(37);
    let (_, lowest_broken) = divergence_confidence(&spec, &plain, &broken).expect("a divergence");
    assert!(
        (lowest_broken - 0.1445).abs() < 0.002,
        "the reconstructed broken regime reads {lowest_broken:.4}, not the measured 0.1445"
    );
    let failure = judge(&spec, &plain, &broken, None)
        .refusal()
        .expect("the lowest broken regime must be refused");
    assert!(failure.contains("nearly tied"), "{failure}");
}

/// The field that separates two drafters the arch check cannot.
///
/// Both `-e2b-` and `-e4b-` assistants declare `Gemma4AssistantForCausalLM`, so
/// the golden harness's stand-down passes either against either verifier. This
/// is what the gate reads instead, and it has to read the number rather than
/// merely find the file.
#[test]
fn the_drafter_declares_the_backbone_it_projects_into() {
    let dir = std::env::temp_dir().join(format!("rmlx_spec_equiv_backbone_{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("scratch dir");

    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"gemma4_assistant","architectures":["Gemma4AssistantForCausalLM"],"backbone_hidden_size":1536}"#,
    )
    .expect("write config");
    assert!(is_gemma4_assistant(&dir));
    assert_eq!(declared_backbone_hidden(&dir), Some(1536));

    // The e4b assistant is the same architecture and a different backbone.
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"gemma4_assistant","architectures":["Gemma4AssistantForCausalLM"],"backbone_hidden_size":2560}"#,
    )
    .expect("write config");
    assert!(is_gemma4_assistant(&dir), "still the same architecture");
    assert_eq!(declared_backbone_hidden(&dir), Some(2560));

    // A snapshot that does not say is served by the engine at the verifier's
    // width, so it must not read as a mismatch here.
    std::fs::write(
        dir.join("config.json"),
        r#"{"model_type":"gemma4_assistant","architectures":["Gemma4AssistantForCausalLM"]}"#,
    )
    .expect("write config");
    let silent = declared_backbone_hidden(&dir);
    assert_eq!(silent, None);
    // The gate's own predicate, on the value the gate would have: absent is
    // what the loader defaults to the verifier's width, not a mismatch.
    assert!(
        silent.is_none_or(|w| w == 1536),
        "an absent key must not read as a mismatch"
    );

    std::fs::remove_dir_all(&dir).ok();
}

// ── The pairs ────────────────────────────────────────────────────────────────

/// Which round loop drives a pair's speculative arm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoundLoop {
    /// Gemma4 shared-K/V assistant: the drafter reads the verifier's K/V, and
    /// the rollback is a KV truncation that includes the SWA ring.
    Gemma4Assistant,
    /// Qwen3.5/3.6 MTP sidecar: the verifier carries recurrent state, and the
    /// rollback restores it from a pre-round snapshot and replays the accepted
    /// prefix.
    MtpSidecar,
    /// DFlash 1 block drafter: the same recurrent rollback, and the only loop
    /// here whose block size changes between rounds — it is set from the accept
    /// rate of the recent ones, so a run walks a range of verify widths rather
    /// than repeating one.
    DFlash1,
    /// DFlash 2 block drafter: the same recurrent rollback, and a drafter that
    /// denoises a whole block at once from the verifier's multi-layer hidden
    /// states rather than stepping through it.
    DFlash2,
    /// EAGLE-3: the same recurrent rollback, and an acceptance walk that scores
    /// intermediate positions over the drafter's reduced vocabulary rather than
    /// the verifier's whole one. See [`EAGLE3_PAIR`] for what that costs.
    Eagle3,
    /// Two full models, greedy acceptance: no sidecar head, a second complete
    /// model with its own KV cache rolled back alongside the verifier's.
    TwoModelGreedy,
}

impl RoundLoop {
    /// The drafter kind a pair's snapshot must declare itself.
    ///
    /// One environment variable names the drafter for every pair that is not
    /// resolved by slug, so a run set up for one of them reaches the others.
    /// A snapshot of the wrong kind stands the pair down here rather than
    /// panicking inside a loader that was handed another drafter's tensors.
    fn declares(self) -> Option<DraftKind> {
        match self {
            RoundLoop::Gemma4Assistant => None,
            RoundLoop::MtpSidecar => Some(DraftKind::Mtp),
            RoundLoop::DFlash1 => Some(DraftKind::DFlash),
            RoundLoop::DFlash2 => Some(DraftKind::DFlash2),
            RoundLoop::Eagle3 => Some(DraftKind::Eagle3),
            RoundLoop::TwoModelGreedy => Some(DraftKind::TwoModel),
        }
    }
}

/// One verifier + drafter the gate can run.
struct Pair {
    verifier: common::GoldenModel,
    drafter: DrafterSource,
    round_loop: RoundLoop,
    /// The round block to drive this pair at, or `None` for the block the engine
    /// serves a request that names none.
    ///
    /// `None` is `rmlx_models::speculative::default_block_for` of whatever depth
    /// the drafter declares, which is the serve layer's own rule — so a pair
    /// left at `None` covers the configuration an operator gets, and [`run_gate`]
    /// holds it to that rather than to nothing. It is not the only one the
    /// engine will run: a request names any block up to what one verify forward
    /// can score, and a wider block puts the same round loop through a verify
    /// forward of a different width, an acceptance walk over more positions and
    /// a rollback over a longer rejected tail. Naming it here is what lets a
    /// pair be judged at one of those.
    block: Option<usize>,
}

/// How a pair's drafter is found, and so whether `make gpu-test` selects the
/// pair on a machine that merely holds the snapshots.
enum DrafterSource {
    /// Resolved by slug from `RMLX_O_MODELS_ROOT`, like the verifier — the pair
    /// runs wherever the snapshots are.
    Slug(&'static str),
    /// Run only when an operator asks, and then resolved by the slug named here.
    ///
    /// `RMLX_DRAFT_TEST_MODEL` is what asks. It is one variable and these are
    /// several pairs, so it cannot also be what each of them resolves to: three
    /// MTP pairs across two verifiers would all take whichever sidecar it held,
    /// and a 4-bit sidecar loads against an mxfp8 verifier of the same width
    /// without complaint. So the variable selects, the slug resolves, and the
    /// path the variable holds is the fallback for a machine whose models root
    /// does not carry the slug — checked against the verifier by
    /// [`declared_quant_mode`] either way.
    ///
    /// `None` is a pair with no snapshot to name: the two-model arm's draft is a
    /// full model of the verifier's family, and no sibling of these verifiers is
    /// on this machine's models root, so the variable is its only handle.
    ///
    /// Its verifier drives an MLX quantized matmul whose `load_safe` bound is
    /// the one `scripts/gpu_validation_census.txt` records, so a run under Metal
    /// shader validation reports over a thousand invalid loads from a kernel
    /// this repo does not compile. The census pins one exact count per test, and
    /// a count from a 256-token generation moves with every prompt — so pinning
    /// such a pair would make the census brittle rather than informative. Until
    /// that is settled these pairs run on request and `make gpu-test` reports
    /// them as skipped, with the variable that would run them named.
    Named(Option<&'static str>),
}

/// The pair the floors were measured on: a full-attention-plus-SWA verifier
/// whose rollback is a KV truncation.
const ASSISTANT_PAIR: Pair = Pair {
    verifier: common::GoldenModel {
        slug: "mlx-community__gemma-4-e2b-it-mxfp8",
        archs: &["Gemma4ForConditionalGeneration"],
    },
    drafter: DrafterSource::Slug("mlx-community__gemma-4-E2B-it-assistant-bf16"),
    round_loop: RoundLoop::Gemma4Assistant,
    block: None,
};

/// The recurrent pair. Its agreement is far below the assistant pair's and no
/// subsequence floor separates it from a broken rollback, which is what the
/// divergence-confidence oracle is for.
const MTP_PAIR: Pair = Pair {
    verifier: common::GoldenModel {
        slug: "mlx-community__Qwen3.8-27B-mxfp8",
        archs: &[
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ],
    },
    drafter: DrafterSource::Named(Some("mlx-community__Qwen3.8-27B-MTP-mxfp8")),
    round_loop: RoundLoop::MtpSidecar,
    block: None,
};

/// The recurrent pair on the 4-bit verifier, at the depth its sidecar declares.
///
/// The same round loop as [`MTP_PAIR`] against a verifier of half the weight
/// width. It is the arm the deep-block pair below is compared against: two
/// blocks of the same loop on the same pair, so a difference between them is
/// the block and not the checkpoint.
const MTP_4BIT_PAIR: Pair = Pair {
    verifier: common::GoldenModel {
        slug: "mlx-community__Qwen3.8-27B-4bit",
        archs: &[
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ],
    },
    drafter: DrafterSource::Named(Some("mlx-community__Qwen3.8-27B-MTP-4bit")),
    round_loop: RoundLoop::MtpSidecar,
    block: None,
};

/// The same pair driven past the depth its sidecar declares.
///
/// The sidecar head chains on its own output hidden, so it proposes to any
/// depth asked for and a request may name one; nothing about that changes what
/// the answer must be. What does change is every width downstream of it — the
/// verify forward scores eight positions instead of three, the acceptance walk
/// runs over eight, and a partial round rolls back a longer rejected tail. Each
/// is a width no other pair here drives this loop at, and each is a place an
/// answer could move without the loop reporting anything.
const MTP_4BIT_DEEP_PAIR: Pair = Pair {
    verifier: MTP_4BIT_PAIR.verifier,
    drafter: DrafterSource::Named(Some("mlx-community__Qwen3.8-27B-MTP-4bit")),
    round_loop: RoundLoop::MtpSidecar,
    block: Some(DEEP_BLOCK),
};

/// The block [`MTP_4BIT_DEEP_PAIR`] drives, chosen as the widest the accept
/// curve still returns proposals at.
const DEEP_BLOCK: usize = 8;

/// The block pair at the block a request that names none is served. Its drafter
/// denoises a whole block in one pass and its selector chains the block's
/// independent argmaxes into one sentence, so an error in either reaches the
/// verifier as a rejected proposal rather than as a failure — which the
/// acceptance walk absorbs, and this gate does not. The declared width is
/// [`DFLASH2_DEEP_PAIR`]'s.
///
/// Named for the same reason [`MTP_PAIR`] is, and more so: its verifier is
/// 4-bit, so it drives the same MLX quantized matmul at a group size the
/// census does not pin.
const DFLASH2_PAIR: Pair = Pair {
    verifier: common::GoldenModel {
        slug: "mlx-community__Qwen3.8-27B-4bit",
        archs: &[
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ],
    },
    drafter: DrafterSource::Named(Some("z-lab__Qwen3.8-27B-DFlash2")),
    round_loop: RoundLoop::DFlash2,
    block: None,
};

/// The block pair at the width its checkpoint declares.
///
/// [`DFLASH2_PAIR`] runs the block a request that names none is served, which
/// is the serve default and narrower than the declaration — so the selector
/// chain is driven over four positions where it is defined over eight. The
/// chain is the thing this drafter is, and a chain re-picking three positions
/// against the one before it is not the chain re-picking seven.
const DFLASH2_DEEP_PAIR: Pair = Pair {
    verifier: DFLASH2_PAIR.verifier,
    drafter: DrafterSource::Named(Some("z-lab__Qwen3.8-27B-DFlash2")),
    round_loop: RoundLoop::DFlash2,
    block: Some(8),
};

/// The adaptive pair at the block a request that names none is served. Its
/// drafter carries no dynamic convolution and no selector, and its loop sets
/// each round's block from the accept rate of the recent ones, so its verify
/// width changes between rounds here as at any block. The schedule's full
/// range is [`DFLASH1_DEEP_PAIR`]'s: at the served 5 this one oscillates over
/// {4, 5}.
///
/// Named for the same reason [`MTP_PAIR`] is: an 8-bit verifier drives the same
/// MLX quantized matmul the census records for the affine instantiation.
const DFLASH1_PAIR: Pair = Pair {
    verifier: common::GoldenModel {
        slug: "mlx-community__Qwen3.6-35B-A3B-8bit",
        archs: &[
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ],
    },
    drafter: DrafterSource::Named(Some("z-lab__Qwen3.6-35B-A3B-DFlash")),
    round_loop: RoundLoop::DFlash1,
    block: None,
};

/// The adaptive pair over the range its schedule actually covers.
///
/// [`DFLASH1_PAIR`] runs the served block, and at 5 the schedule oscillates
/// over {4, 5}: `dflash_next_block_size` floors at `min(block, 4)` and grows to
/// the ceiling it was given. The checkpoint declares 16, and the sequence of
/// widths a run then produces — an 8-wide append truncated and followed by a
/// 4- or 6-wide one — is the thing no other pair here reaches. Both are gated
/// because both are real: one is what an operator is served, the other is what
/// the schedule is.
const DFLASH1_DEEP_PAIR: Pair = Pair {
    verifier: DFLASH1_PAIR.verifier,
    drafter: DrafterSource::Named(Some("z-lab__Qwen3.6-35B-A3B-DFlash")),
    round_loop: RoundLoop::DFlash1,
    block: Some(16),
};

/// The restricted-vocabulary pair, and the one place this gate reads a
/// **declared inexactness** rather than a defect.
///
/// EAGLE-3's verify pass scores the whole block against the drafter's reduced
/// vocabulary — 32000 target ids plus the verifier's stop ids — and computes the
/// verifier's full-vocabulary argmax at one position only: the first the draft
/// missed, or the bonus when it missed none. An accepted position is therefore
/// emitted as the draft's token whenever the draft agrees with the *restricted*
/// argmax, and at a position whose true argmax lies outside that set those two
/// are not the same token. The upstream implementation does this too, so it is a
/// design boundary rather than a port defect — but it is still an answer change
/// at temperature 0, which is what this gate reads.
///
/// [`Loaded::draft_vocab`] measures the exposure on every run: how many of the
/// reference arm's own tokens the drafter's vocabulary cannot name — one to five
/// per answer. Each is a position where the boundary could fire; none is a
/// position where it must, because it fires only where the drafter also proposed
/// the restricted argmax. [`Restriction`] is what the verdict does with that.
const EAGLE3_PAIR: Pair = Pair {
    verifier: common::GoldenModel {
        slug: "mlx-community__Qwen3.6-35B-A3B-8bit",
        archs: &[
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ],
    },
    drafter: DrafterSource::Named(Some("Dogacel__specdrift-qwen3.6-35b-a3b-eagle3")),
    round_loop: RoundLoop::Eagle3,
    block: None,
};

/// The two-model pair: no sidecar head at all, a second complete model of the
/// same family whose own KV cache is rolled back beside the verifier's every
/// partial round. Both halves are GDN hybrids, so a round rolls back two kinds
/// of state on each of two models.
///
/// Named, and its drafter is the one kind [`RoundLoop::declares`] cannot pin to
/// a snapshot: `two_model` is an inference from the architecture registry, which
/// every full model satisfies. [`declared_vocab_size`] is the discriminator that
/// stands a mismatched pair down before either model is read.
const TWO_MODEL_PAIR: Pair = Pair {
    verifier: common::GoldenModel {
        slug: "mlx-community__Qwen3.8-27B-mxfp8",
        archs: &[
            "Qwen3_5ForConditionalGeneration",
            "Qwen3_5MoeForConditionalGeneration",
        ],
    },
    drafter: DrafterSource::Named(None),
    round_loop: RoundLoop::TwoModelGreedy,
    block: None,
};

/// Draft-model override, the variable the sibling alignment suites take.
const DRAFT_MODEL_VAR: &str = "RMLX_DRAFT_TEST_MODEL";

/// Resolve the **drafter**, which the golden harness has no variable for: an
/// operator's override if they named one, otherwise the slug under
/// `RMLX_O_MODELS_ROOT`. The verifier goes through `common::model_for`.
///
/// Resolving by slug is what puts this gate inside `make gpu-test` on a machine
/// holding the snapshots: it joins the population `run_gpu_tests.sh` already
/// reports as INCOMPLETE when the models root is unset, instead of being a
/// third variable nobody exports and a green run that asserted nothing.
///
/// A path the operator named that is not a snapshot fails; a models root that
/// simply does not hold the slug skips. That split is the harness's
/// (`tests/common/mod.rs`), not a second copy of its rules.
///
/// Both probes ask for a [`common::Role::Sidecar`]: a drafter is decoded with
/// the verifier's tokenizer and ships none of its own, so requiring one would
/// turn a checkpoint sitting on this machine's disk into a skip — and a skip in
/// this gate reads exactly like the equivalence holding. The harness names its
/// own override variable in the messages it builds; this half of the pair is
/// overridden by a different one, so that name is substituted.
fn resolve(var: &str, slug: &str) -> common::Gate {
    named_path(var).unwrap_or_else(|| by_slug(var, slug))
}

/// What the path in `var` resolves to, or `None` when the variable holds none.
///
/// An operator who named a path meant it, so a typo or a moved snapshot breaks
/// the run rather than skipping it. `override_snapshot` reports only an unset or
/// empty value as `None`, so the inner `unwrap_or_else` covers a case the outer
/// `?` has already excluded.
fn named_path(var: &str) -> Option<common::Gate> {
    let named = std::env::var(var).ok().filter(|v| !v.is_empty())?;
    let probed =
        common::override_snapshot(Some(&named), common::Role::Sidecar).unwrap_or_else(|| {
            common::Snapshot::Misconfigured(format!("{var} is set to an empty value"))
        });
    Some(match probed {
        common::Snapshot::Found { path, .. } => common::Gate::Run { path, note: None },
        common::Snapshot::Absent(why) | common::Snapshot::Misconfigured(why) => {
            common::Gate::Fail(why.replace(common::SINGLE_MODEL_VAR, var))
        }
    })
}

/// What a slug resolves to under `RMLX_O_MODELS_ROOT`, reading no variable.
///
/// `blame` names the variable a message should point at — the one that would
/// have overridden this — because the harness's own messages name its single
/// override and this half of a pair is overridden by a different one.
fn by_slug(blame: &str, slug: &str) -> common::Gate {
    let root = std::env::var(common::MODELS_ROOT_VAR).ok();
    match common::slug_snapshot(root.as_deref(), slug, common::Role::Sidecar) {
        common::Snapshot::Found { path, .. } => common::Gate::Run { path, note: None },
        common::Snapshot::Absent(why) => {
            common::Gate::Skip(why.replace(common::SINGLE_MODEL_VAR, blame))
        }
        common::Snapshot::Misconfigured(why) => common::Gate::Fail(why),
    }
}

/// Whether `draft_path` is the dedicated Gemma4 assistant drafter snapshot.
///
/// Both fields are read because mlx-community snapshots set the family on one
/// or the other depending on the export tool — the same two fields the serve
/// layer routes `--draft-kind mtp` on.
fn is_gemma4_assistant(draft_path: &Path) -> bool {
    draft_config(draft_path).is_some_and(|cfg| {
        let model_type = cfg["model_type"].as_str().unwrap_or_default();
        let arch_name = cfg["architectures"][0].as_str().unwrap_or_default();
        model_type == "gemma4_assistant" || arch_name.contains("Gemma4Assistant")
    })
}

/// The drafter kind a snapshot declares itself, by the serve layer's own rule.
fn declared_kind(draft_path: &Path) -> Option<DraftKind> {
    let cfg = draft_config(draft_path)?;
    Declared::from_snapshot(
        cfg["architectures"][0].as_str().unwrap_or_default(),
        cfg["model_type"].as_str().unwrap_or_default(),
    )
    .kind()
}

fn draft_config(draft_path: &Path) -> Option<serde_json::Value> {
    let raw = std::fs::read_to_string(draft_path.join("config.json")).ok()?;
    serde_json::from_str(&raw).ok()
}

/// The verifier width this drafter declares it projects into, if it declares
/// one.
///
/// Both `-e2b-` and `-e4b-` snapshots declare the same architecture, so the
/// harness's arch stand-down cannot tell them apart — an operator with
/// `RMLX_KV_TEST_MODEL` pointed at e4b for another suite would otherwise run an
/// e4b verifier against the E2B drafter resolved by slug, against floors
/// measured on e2b. This is the field that does tell them apart, and it is read
/// before the drafter is loaded so a mismatched pair skips with a reason rather
/// than panicking inside the loader.
///
/// `None` means the snapshot does not say, which is **not** a mismatch:
/// `Gemma4AssistantDrafter::load` takes the verifier's width when the key is
/// absent, so a drafter the engine serves must not make this gate skip and
/// blame the drafter for it.
fn declared_backbone_hidden(draft_path: &Path) -> Option<usize> {
    draft_config(draft_path)?["backbone_hidden_size"]
        .as_u64()
        .and_then(|v| usize::try_from(v).ok())
}

/// The weight-quantization mode a snapshot declares, if it declares one.
///
/// A sidecar is decoded through the verifier's own LM head and conditioned on
/// its hidden states, so a pair whose halves were quantized differently is a
/// pair in name only — and nothing downstream refuses it: both 27B sidecars
/// carry the same width and the same tensor names, so a 4-bit one loads against
/// an mxfp8 verifier and drafts fluently at a rate no reading here could be
/// attributed to either checkpoint.
///
/// **Mode and not bits.** The shipped Qwen3.6 pair is an 8-bit verifier with a
/// 5-bit sidecar, both affine, and it is the pair the MoE rows were measured on
/// — so a width comparison would stand down a pairing that is real. The mode is
/// what the two 27B pairs differ on and what nothing else separates them by.
///
/// `None` on either side is no opinion: the DFlash 2 and EAGLE-3 drafters
/// declare no `quantization` block at all.
///
/// It speaks for the sidecar loops only. A two-model draft is a whole model that
/// shares the verifier's tokenizer and nothing else, so its weight format is
/// unrelated to the verifier's and a pair of different ones is a pair.
fn declared_quant_mode(path: &Path) -> Option<String> {
    let cfg = draft_config(path)?;
    for at in [&cfg["quantization"], &cfg["text_config"]["quantization"]] {
        if let Some(mode) = at["mode"].as_str() {
            return Some(mode.to_owned());
        }
    }
    None
}

/// The vocabulary a snapshot declares, from its own config or its text tower's.
///
/// The two-model pair needs this because its loop's kind is an inference from
/// the architecture registry rather than a marker: every full model declares
/// itself a `two_model` drafter, so the kind check cannot tell one family's from
/// another's. A draft whose vocabulary is not the verifier's is refused inside
/// `load_speculative`, which would fail the run rather than stand it down.
fn declared_vocab_size(path: &Path) -> Option<usize> {
    let cfg = draft_config(path)?;
    cfg["vocab_size"]
        .as_u64()
        .or_else(|| cfg["text_config"]["vocab_size"].as_u64())
        .and_then(|v| usize::try_from(v).ok())
}

// ── Prompts ──────────────────────────────────────────────────────────────────

/// What every prompt asks for, and why it asks for it.
///
/// This gate owns its prompts, and that is the asymmetry that makes its
/// repetition control possible at all. A general "is this arm degenerate"
/// classifier cannot exist on this measure: healthy output spans 0.03 for prose
/// to 0.88 for a markdown table with a yes/no column, and degenerate output
/// spans 0.37 for a ragged loop to 1.00 for an exact one — two populations
/// overlapping over most of their range, with no threshold between them.
///
/// Prose is the one regime where they separate. Asking for it is not a
/// convenience: it is what lets [`MAX_CYCLE_FRACTION`] sit above every healthy
/// reading measured here and below every collapse the gate has to catch.
const PROSE_INSTRUCTION: &str = "Answer at length, in continuous prose, in at least \
     six full paragraphs. Do not use lists, numbered steps, tables, headings, bullet \
     points or code blocks.";

/// The 4k document the long-context benches use.
const LONG_CONTEXT_DOCUMENT: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../prompts/longctx_4k.json"
));

/// One question, and whether it carries the 4k document with it.
struct Prompt {
    name: &'static str,
    question: &'static str,
    long_context: bool,
}

/// What the sweep runs, and what the gate picks from.
///
/// The last one is the reproducer: a 4k document is what wraps a sliding-window
/// ring, which is where the defect this gate found lived. The others exist so
/// the constants above are set against a population rather than against the one
/// prompt that happened to fail.
const PROMPTS: &[Prompt] = &[
    Prompt {
        name: "hash-map-collisions",
        question: "Explain how a hash map handles two keys that hash to the same bucket, \
                   covering separate chaining and each of the open-addressing probing \
                   strategies.",
        long_context: false,
    },
    Prompt {
        name: "tcp-congestion",
        question: "Explain how TCP congestion control decides how fast to send, covering \
                   slow start, congestion avoidance and what happens when a loss is \
                   detected.",
        long_context: false,
    },
    Prompt {
        name: "virtual-memory",
        question: "Explain how virtual memory works on a modern operating system, covering \
                   page tables, translation lookaside buffers and what happens on a page \
                   fault.",
        long_context: false,
    },
    Prompt {
        name: "database-isolation",
        question: "Explain what a database isolation level is and how the common ones \
                   differ, covering the anomalies each one still permits.",
        long_context: false,
    },
    Prompt {
        name: "photosynthesis",
        question: "Explain how photosynthesis turns light into stored chemical energy, \
                   covering the light-dependent reactions and the carbon-fixing ones.",
        long_context: false,
    },
    Prompt {
        name: "longctx-4k",
        question: "Summarise the document above.",
        long_context: true,
    },
];

impl Prompt {
    /// Chat-formatted prompt ids for whichever turn markers the tokenizer
    /// declares.
    ///
    /// A model served outside its own turn markers answers nothing useful —
    /// Gemma without them and without `<bos>` ends the turn on its first token,
    /// and a gate that only ever runs on a degenerate stream is a gate that
    /// never runs. The markers are read off the tokenizer rather than hard-coded
    /// per pair, so a third pair needs no new branch unless it needs new
    /// markers.
    ///
    /// A tokenizer that declares `<think>` gets an **empty** reasoning block,
    /// which is what its own template emits for `enable_thinking=false`. The
    /// gate asks for prose because prose is the regime its repetition control
    /// can read, and a reasoning block is a plan in numbered steps whatever the
    /// question asked for — measured on this pair, plain greedy's own reasoning
    /// block reads 0.2500 against a ceiling of 0.20.
    fn ids(&self, tk: &tokenizers::Tokenizer) -> Vec<u32> {
        let question = if self.long_context {
            let doc: serde_json::Value = serde_json::from_str(LONG_CONTEXT_DOCUMENT)
                .expect("the long-context prompt is JSON");
            let body = doc["messages"]
                .as_array()
                .expect("messages")
                .iter()
                .filter_map(|m| m["content"].as_str())
                .collect::<Vec<_>>()
                .join("\n\n");
            format!("{body}\n\n{} {PROSE_INSTRUCTION}", self.question)
        } else {
            format!("{} {PROSE_INSTRUCTION}", self.question)
        };

        let mut ids: Vec<u32> = Vec::new();
        if let Some(bos) = tk.token_to_id("<bos>") {
            ids.push(bos);
        }
        // Read off the tokenizer's own added tokens, in an order that keeps a
        // model declaring more than one family on its own markers.
        let turns = [
            ("<|turn>", "<|turn>user\n", "<turn|>\n", "<|turn>model\n"),
            (
                "<start_of_turn>",
                "<start_of_turn>user\n",
                "<end_of_turn>\n",
                "<start_of_turn>model\n",
            ),
            (
                "<|im_start|>",
                "<|im_start|>user\n",
                "<|im_end|>\n",
                "<|im_start|>assistant\n",
            ),
        ];
        let Some(&(_, user, end, assistant)) = turns
            .iter()
            .find(|(marker, ..)| tk.token_to_id(marker).is_some())
        else {
            panic!(
                "this verifier declares none of the turn markers this gate knows, and a \
                 model served outside its own markers answers nothing the gate can judge"
            )
        };
        let mut text = format!("{user}{question}{end}{assistant}");
        if tk.token_to_id("<think>").is_some() {
            text.push_str("<think>\n\n</think>\n\n");
        }
        ids.extend(
            tk.encode(text.as_str(), true)
                .expect("encode")
                .get_ids()
                .iter()
                .copied(),
        );
        ids
    }
}

/// The verifier's own stop ids.
///
/// Without them both arms run past the answer into end-of-turn filler, and a
/// comparison over filler measures nothing about the round loop.
fn eos_ids(model_path: &Path) -> Vec<u32> {
    let Ok(raw) = std::fs::read(model_path.join("config.json")) else {
        return Vec::new();
    };
    let Ok(cfg) = serde_json::from_slice::<serde_json::Value>(&raw) else {
        return Vec::new();
    };
    match cfg.get("eos_token_id") {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(serde_json::Value::Array(a)) => a
            .iter()
            .filter_map(|v| v.as_u64().map(|x| x as u32))
            .collect(),
        _ => Vec::new(),
    }
}

/// Plain greedy decoding on the verifier alone, at temperature 0: the token ids
/// and, per position, the verifier's own top-two logprob gap.
///
/// The gap is what the divergence oracle reads. It costs one log-softmax and a
/// partial top-k per step on the host — the reference arm is not the arm any
/// throughput number comes from.
fn plain_greedy(
    verifier: &arch::Architecture,
    tk: &tokenizers::Tokenizer,
    prompt: &[u32],
    eos: &[u32],
    device: Device,
) -> (Vec<u32>, Vec<f32>) {
    // The speculative arms' configuration, plus the top-two read-back only this
    // arm needs — one temperature-0 setting, and the one field that differs.
    let sampler_cfg = rmlx_models::sampler::SamplerConfig {
        top_logprobs_k: 2,
        ..GREEDY
    };
    let mut rng = rmlx_models::sampler::Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = rmlx_models::sampler::PenaltyConfig::default();
    let mut history: Vec<u32> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    let mut margins: Vec<f32> = Vec::new();
    {
        let mut step = |s: &rmlx_models::ProbeStep| {
            ids.push(s.token_id);
            let mut top: Vec<f32> = s
                .logprobs
                .as_ref()
                .map(|lp| lp.top.iter().map(|t| t.1).collect())
                .unwrap_or_default();
            top.sort_by(|a, b| b.total_cmp(a));
            margins.push(match (top.first(), top.get(1)) {
                (Some(best), Some(second)) => best - second,
                // A one-entry vocabulary is not a thing, but a margin the arm
                // did not report must not read as a tie.
                _ => f32::INFINITY,
            });
            None
        };
        verifier
            .generate_greedy(
                tk,
                prompt,
                N_TOKENS,
                device,
                Some(rmlx_kv_quant::KvQuant::None),
                Some(MAX_CTX),
                1,
                eos,
                &mut step,
                None,
                &sampler_cfg,
                &mut rng,
                &penalty_cfg,
                &mut history,
            )
            .expect("plain greedy generate");
    }
    (ids, margins)
}

/// The greedy sampler configuration both arms run under. Answer equivalence is
/// a temperature-0 claim: at any other temperature the two arms draw from
/// different random streams and the ids are free to differ.
const GREEDY: rmlx_models::sampler::SamplerConfig = rmlx_models::sampler::SamplerConfig {
    temperature: 0.0,
    top_p: 1.0,
    top_k: 0,
    min_p: 0.0,
    seed: Some(0),
    top_logprobs_k: 0,
};

/// A loaded pair, ready to answer prompts.
struct Loaded {
    engine: Engine,
    tokenizer: tokenizers::Tokenizer,
    eos: Vec<u32>,
    /// The pair's [`Pair::block`], carried through to the round-loop call.
    block: Option<usize>,
    /// The drafter snapshot these arms actually ran, so a reading names the
    /// checkpoint it came from rather than the pair it was filed under.
    drafter: std::path::PathBuf,
}

/// The verifier, and whatever drives the speculative arm against it.
///
/// A sidecar head reads the verifier and is held beside it. The two-model
/// dispatcher owns both models, so that pair holds the dispatcher and lends the
/// verifier back for the reference arm.
enum Engine {
    Sidecar {
        verifier: Box<arch::Architecture>,
        drafter: Drafter,
    },
    TwoModel(Box<SpeculativeDispatcher>),
}

impl Engine {
    /// The model the reference arm decodes with, whichever half of the pair
    /// holds it.
    fn verifier(&self) -> &arch::Architecture {
        match self {
            Engine::Sidecar { verifier, .. } => verifier,
            Engine::TwoModel(dispatcher) => &dispatcher.verifier,
        }
    }
}

enum Drafter {
    Assistant(Box<Gemma4AssistantDrafter>),
    Mtp(Box<MtpDrafter>),
    DFlash1(Box<DFlashDrafter>),
    DFlash2(Box<DFlash2Drafter>),
    Eagle3(Box<Eagle3Drafter>),
}

impl Loaded {
    /// The block depth this pair's drafter declares, when it declares one.
    ///
    /// The serve layer resolves a request that names no block from this, so it
    /// is what a `None` pair has to be held to. It is read off the loaded
    /// drafter rather than taken from the pair, which is what keeps the check
    /// from being the harness reading back its own choice.
    fn declared_block(&self) -> Option<usize> {
        match &self.engine {
            Engine::Sidecar { drafter, .. } => match drafter {
                // Neither declares a depth: the assistant's config carries no
                // block key, and a two-model draft is a full model with no
                // drafting depth to declare.
                Drafter::Assistant(_) => None,
                Drafter::Mtp(drafter) => drafter.block_size(),
                Drafter::DFlash1(drafter) => Some(drafter.block_size()),
                Drafter::DFlash2(drafter) => Some(drafter.cfg.block_size),
                Drafter::Eagle3(drafter) => Some(drafter.block_size()),
            },
            Engine::TwoModel(_) => None,
        }
    }

    /// The block this pair's arms run at: what the pair names, or what the serve
    /// layer resolves for a request that names none.
    ///
    /// One producer, so every arm asks for the same width and [`run_gate`] can
    /// hold the answer to it. The two-model loop converts to a draft count at
    /// its own call, which is the only place the two units meet.
    fn round_block(&self) -> usize {
        self.block
            .unwrap_or_else(|| default_block_for(self.declared_block()))
    }

    /// Both arms over one prompt: the block the speculative arm ran at, the
    /// speculative ids, the reference ids, the reference's per-position margins,
    /// and which vocabulary decided each speculative token.
    ///
    /// The last is empty for every loop that scores each position over the
    /// verifier's whole vocabulary, which is all of them but the restricted one.
    ///
    /// The block is returned because a pair that names one has to be checked
    /// against it. Every reading below — the divergence position, its margin,
    /// its confidence — is attributed to a block in the report line, and a pair
    /// whose block quietly fell back to the drafter's declaration would produce
    /// a full set of plausible readings under the wrong label.
    ///
    /// It is **each driver's own answer for the widest block its rounds ran**,
    /// not the number this function was about to pass. Reading back the argument
    /// would check that this function can hold a value, which is not the
    /// question: the resolvers narrow to a checkpoint, and every loop narrows
    /// again per round.
    fn arms(
        &mut self,
        prompt: &Prompt,
        device: Device,
    ) -> (
        usize,
        Vec<u32>,
        Vec<u32>,
        Vec<f32>,
        Vec<DecidedBy>,
        Vec<CapturedEvent>,
    ) {
        let ids = prompt.ids(&self.tokenizer);
        let block = self.round_block();
        let (ran, spec_ids, decided_by, rounds) = self.spec_arm(&ids, block, device);
        let (plain_ids, margins) = plain_greedy(
            self.engine.verifier(),
            &self.tokenizer,
            &ids,
            &self.eos,
            device,
        );
        assert!(
            decided_by.is_empty() || decided_by.len() == spec_ids.len(),
            "the loop reported which vocabulary decided {} tokens and emitted {} — the \
             verdict indexes one by the other",
            decided_by.len(),
            spec_ids.len()
        );
        (ran, spec_ids, plain_ids, margins, decided_by, rounds)
    }

    /// The speculative arm alone: the widest block any of its rounds ran, the
    /// ids it emitted, which vocabulary decided each of them, and the round
    /// stream it reported while doing so.
    ///
    /// The capture is scoped here and nowhere wider. The reference arm runs no
    /// round, and a subscriber left installed past this call would answer the
    /// behaviour switches of every test after it in this binary.
    #[allow(
        clippy::too_many_lines,
        reason = "one arm per drafter, each a twelve-argument driver call; a function per arm would scatter the dispatch without shortening it"
    )]
    fn spec_arm(
        &mut self,
        ids: &[u32],
        block: usize,
        device: Device,
    ) -> (usize, Vec<u32>, Vec<DecidedBy>, Vec<CapturedEvent>) {
        let mut spec_ids: Vec<u32> = Vec::new();
        let mut decided_by: Vec<DecidedBy> = Vec::new();
        // EAGLE-3 is the one loop that consults the second switch, and what the
        // capture is held to is how often. Its round event sits on that same
        // target, so the target is never silent; and the recorder declines
        // TRACE, so no event arrives whatever the guard does. What separates the
        // two worlds is the *question*: the guard asks once per round, and
        // `if step_trace_enabled()` rewritten to `if true` deletes that call and
        // leaves the per-position `trace!` asking once per verified position.
        let asks_step_switch = matches!(
            &self.engine,
            Engine::Sidecar {
                drafter: Drafter::Eagle3(_),
                ..
            }
        );
        let recorder = RoundStreamRecorder::new();
        let ran = tracing::subscriber::with_default(std::sync::Arc::clone(&recorder), || {
            let mut step = |s: &rmlx_models::ProbeStep| {
                spec_ids.push(s.token_id);
                None
            };
            match &mut self.engine {
                Engine::Sidecar { verifier, drafter } => match drafter {
                    Drafter::Assistant(drafter) => {
                        mtp_assistant_generate(
                            verifier,
                            drafter,
                            &self.tokenizer,
                            ids,
                            N_TOKENS,
                            block,
                            Some(rmlx_kv_quant::KvQuant::None),
                            Some(MAX_CTX),
                            &self.eos,
                            &mut step,
                            &GREEDY,
                            device,
                        )
                        .expect("assistant speculative generate")
                        .1
                    }
                    Drafter::Mtp(drafter) => {
                        let block = self
                            .block
                            .unwrap_or_else(|| default_block_for(drafter.block_size()));
                        mtp_generate(
                            verifier,
                            drafter,
                            &self.tokenizer,
                            ids,
                            N_TOKENS,
                            block,
                            Some(rmlx_kv_quant::KvQuant::None),
                            Some(MAX_CTX),
                            &self.eos,
                            &mut step,
                            &GREEDY,
                            device,
                        )
                        .expect("mtp speculative generate")
                        .1
                    }
                    // The loop halves and grows this from the recent accept
                    // rate, so the width varies within a run whatever it starts
                    // at — the schedule is what this pair covers that no other
                    // does, and it runs at whatever block the pair names.
                    Drafter::DFlash1(drafter) => {
                        dflash_generate(
                            verifier,
                            drafter,
                            &self.tokenizer,
                            ids,
                            N_TOKENS,
                            block,
                            Some(rmlx_kv_quant::KvQuant::None),
                            Some(MAX_CTX),
                            &self.eos,
                            &mut step,
                            &GREEDY,
                            device,
                        )
                        .expect("dflash speculative generate")
                        .1
                    }
                    // The whole point of a block drafter is the block, and the
                    // width its selector chain is defined over is the one its
                    // checkpoint declares — so a pair naming none is served the
                    // narrower of that and the default, as anything else is.
                    Drafter::DFlash2(drafter) => {
                        dflash2_generate(
                            verifier,
                            drafter,
                            &self.tokenizer,
                            ids,
                            N_TOKENS,
                            block,
                            Some(rmlx_kv_quant::KvQuant::None),
                            Some(MAX_CTX),
                            &self.eos,
                            &mut step,
                            &GREEDY,
                            device,
                        )
                        .expect("dflash2 speculative generate")
                        .1
                    }
                    Drafter::Eagle3(drafter) => {
                        eagle3_generate(
                            verifier,
                            drafter,
                            &self.tokenizer,
                            ids,
                            N_TOKENS,
                            block,
                            Some(rmlx_kv_quant::KvQuant::None),
                            Some(MAX_CTX),
                            &self.eos,
                            &mut step,
                            &mut decided_by,
                            &GREEDY,
                            device,
                        )
                        .expect("eagle3 speculative generate")
                        .1
                    }
                },
                // This loop takes a draft count where every other takes a
                // block. The two are one apart, so passing a block here runs a
                // round one position wider than the pair names and than the
                // serve layer runs — which reads as a plausible block on both
                // sides of the comparison.
                Engine::TwoModel(dispatcher) => {
                    dispatcher
                        .spec_generate_greedy(
                            &self.tokenizer,
                            ids,
                            N_TOKENS,
                            drafts_per_round(block),
                            Some(rmlx_kv_quant::KvQuant::None),
                            Some(MAX_CTX),
                            0,
                            &self.eos,
                            &mut step,
                            None,
                            &GREEDY,
                        )
                        .expect("two-model speculative generate")
                        .1
                }
            }
        });
        let questions = recorder.questions();
        let enabled_at_trace: Vec<&String> = questions
            .iter()
            .filter(|(_, level, answer)| *level == tracing::Level::TRACE && *answer)
            .map(|(target, ..)| target)
            .collect();
        assert!(
            enabled_at_trace.is_empty(),
            "the capture enabled {enabled_at_trace:?} at TRACE, and this run is then a \
             different, slower one than the engine ships"
        );
        assert!(
            !questions.is_empty(),
            "the capture was never consulted, so it is a recording of nothing rather \
             than a recording of this arm"
        );
        // Which switches this loop asked about is a property of the loop and not
        // of the capture: three consult the phase switch and four pass a literal
        // `false`, so an empty list here is a fact about the loop rather than a
        // capture that stood down.
        let mut switches: Vec<&str> = questions
            .iter()
            .filter(|(target, level, _)| {
                common::round_stream::BEHAVIOUR_SWITCH_TARGETS.contains(&target.as_str())
                    && *level == tracing::Level::TRACE
            })
            .map(|(target, ..)| target.as_str())
            .collect();
        switches.sort_unstable();
        switches.dedup();
        eprintln!("switches declined: {switches:?}");
        let rounds = round_events(&recorder.events());
        let step_questions = questions
            .iter()
            .filter(|(target, level, _)| {
                target == EAGLE3_STEP_SWITCH_TARGET && *level == tracing::Level::TRACE
            })
            .count();
        // Two-sided: the loop that asks it asks once per round, and every other
        // loop asks not at all. A loop that started consulting a switch it does
        // not gate on would be as much of a change as one that stopped.
        let want_step_questions = if asks_step_switch { rounds.len() } else { 0 };
        assert!(
            step_questions == want_step_questions,
            "this pair asks its per-position trace switch {want_step_questions} times \
             over {} rounds and it was asked {step_questions}. A guard replaced by a \
             constant asks it once per verified position, or not at all.",
            rounds.len()
        );
        (ran, spec_ids, decided_by, rounds)
    }

    /// The target-vocabulary ids this pair's drafter can name, or `None` when it
    /// can name every one the verifier can.
    ///
    /// Only EAGLE-3 answers. Its verify pass takes the argmax over the drafter's
    /// reduced vocabulary at every position except the correction, and a
    /// restricted argmax equals the true one **exactly** when the true one is in
    /// the set. So a position whose reference token is in here cannot have been
    /// changed by the restriction, and a position whose reference token is not
    /// can have been.
    fn draft_vocab(&self) -> Option<std::collections::HashSet<u32>> {
        let Engine::Sidecar {
            drafter: Drafter::Eagle3(drafter),
            ..
        } = &self.engine
        else {
            return None;
        };
        let hot: std::collections::HashSet<u32> = drafter.hot_ids_host().iter().copied().collect();
        (!hot.is_empty()).then_some(hot)
    }
}

/// Load a pair, or say why the gate stood down.
fn load(pair: &Pair, test: &str, device: Device) -> Option<Loaded> {
    // The drafter first: a pair the operator has not named stands down before
    // anything loads a verifier.
    let named = std::env::var(DRAFT_MODEL_VAR)
        .ok()
        .filter(|v| !v.is_empty());
    // Both arms below that reach this are guarded on the variable being set, so
    // it always resolves to something; the fallback names the guard it fell
    // through rather than restating the message a `false` arm already gives.
    let named_gate = || {
        named_path(DRAFT_MODEL_VAR).unwrap_or_else(|| {
            common::Gate::Fail(format!(
                "{DRAFT_MODEL_VAR} was set when this pair was selected and is not now"
            ))
        })
    };
    let draft_gate = match (&pair.drafter, named.is_some()) {
        (DrafterSource::Slug(slug), _) => resolve(DRAFT_MODEL_VAR, slug),
        // The variable asked for this pair; the pair says which sidecar it
        // needs. Its own slug first, so a run with several of these selected
        // does not hand all of them whichever one path the variable holds.
        // Only a models root that simply does not carry the slug falls back to
        // the path the variable holds. A root that is misconfigured is the
        // operator's mistake, and swallowing it would reinstate the one
        // cross-pairing the slug is here to prevent — with the drafter that
        // happens to be in the variable, silently.
        (DrafterSource::Named(Some(slug)), true) => match by_slug(DRAFT_MODEL_VAR, slug) {
            found @ common::Gate::Run { .. } => found,
            failed @ common::Gate::Fail(_) => failed,
            common::Gate::Skip(_) => named_gate(),
        },
        (DrafterSource::Named(None), true) => named_gate(),
        (DrafterSource::Named(_), false) => common::Gate::Skip(format!(
            "{DRAFT_MODEL_VAR} is unset and this pair's drafter is not resolved by \
             slug — see the DrafterSource::Named note for why"
        )),
    };
    let draft_path = common::apply(draft_gate, test)?;
    if let Some(want) = pair.round_loop.declares() {
        let declared = declared_kind(&draft_path);
        if declared != Some(want) {
            eprintln!(
                "SKIP {test}: {} declares {declared:?}, and this pair's loop drives a \
                 {want} drafter",
                draft_path.display()
            );
            return None;
        }
    }
    let model_path = common::model_for(&pair.verifier, test)?;
    // A sidecar quantized differently from its verifier is not this pair, and
    // nothing downstream says so — see `declared_quant_mode`. The two-model arm
    // is exempt because its draft is an independent model: it shares the
    // verifier's tokenizer and nothing else, so the two weight formats are
    // unrelated and a pair of different ones is a pair.
    if pair.round_loop != RoundLoop::TwoModelGreedy {
        if let (Some(d), Some(v)) = (
            declared_quant_mode(&draft_path),
            declared_quant_mode(&model_path),
        ) {
            if d != v {
                eprintln!(
                    "SKIP {test}: {} is quantized {d} and {} is quantized {v}, so the \
                     two are not this pair",
                    draft_path.display(),
                    model_path.display(),
                );
                return None;
            }
        }
    }
    // The two-model loop's kind is an inference from the architecture registry,
    // so a full model of any family declares itself this pair's drafter. The
    // vocabulary is what separates them, and reading it here stands a mismatched
    // pair down instead of failing inside the dispatcher's own check.
    if pair.round_loop == RoundLoop::TwoModelGreedy {
        let (draft_vocab, verifier_vocab) = (
            declared_vocab_size(&draft_path),
            declared_vocab_size(&model_path),
        );
        if draft_vocab.is_some() && draft_vocab != verifier_vocab {
            eprintln!(
                "SKIP {test}: {} declares a {:?}-token vocabulary and {} declares {:?}, \
                 so the two are not a two-model pair",
                draft_path.display(),
                draft_vocab.unwrap_or(0),
                model_path.display(),
                verifier_vocab.unwrap_or(0),
            );
            return None;
        }
    }

    let tokenizer =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let eos = eos_ids(&model_path);
    assert!(
        !eos.is_empty(),
        "the verifier config must name its stop ids — without them both arms run past \
         the answer into end-of-turn filler and the comparison is over that"
    );

    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let engine = engine_for(
        pair.round_loop,
        test,
        verifier,
        &draft_path,
        &model_path,
        &eos,
        device,
    )?;

    Some(Loaded {
        engine,
        tokenizer,
        eos,
        block: pair.block,
        drafter: draft_path,
    })
}

/// Build the pair's speculative half around an already-loaded verifier, or say
/// why the gate stood down.
///
/// The drafter is what a pair is, and everything a loader can refuse about it is
/// checked here: whether the verifier is the kind this loop drives, and whether
/// the drafter's own declarations agree with the verifier it was resolved
/// against.
#[allow(clippy::too_many_arguments)]
fn engine_for(
    round_loop: RoundLoop,
    test: &str,
    verifier: arch::Architecture,
    draft_path: &Path,
    model_path: &Path,
    eos: &[u32],
    device: Device,
) -> Option<Engine> {
    let hidden = verifier.hidden_size();
    let vocab = verifier.vocab_size();
    let verifier = Box::new(verifier);
    let engine = match round_loop {
        RoundLoop::Gemma4Assistant => {
            if !is_gemma4_assistant(draft_path) {
                eprintln!(
                    "SKIP {test}: {} is not a Gemma4 assistant drafter",
                    draft_path.display()
                );
                return None;
            }
            let declared = declared_backbone_hidden(draft_path);
            if declared.is_some_and(|width| width != hidden) {
                eprintln!(
                    "SKIP {test}: {} projects into a backbone {} wide and {} is {hidden}",
                    draft_path.display(),
                    declared.unwrap_or(hidden),
                    model_path.display(),
                );
                return None;
            }
            Engine::Sidecar {
                verifier,
                drafter: Drafter::Assistant(Box::new(
                    Gemma4AssistantDrafter::load(draft_path, hidden, device)
                        .expect("load assistant drafter"),
                )),
            }
        }
        RoundLoop::MtpSidecar => {
            if !verifier.needs_lin_caches() {
                eprintln!(
                    "SKIP {test}: {} carries no recurrent state, so it is not the \
                     verifier this loop drives",
                    model_path.display()
                );
                return None;
            }
            Engine::Sidecar {
                verifier,
                drafter: Drafter::Mtp(Box::new(
                    MtpDrafter::load(draft_path, hidden, device).expect("load MTP sidecar"),
                )),
            }
        }
        RoundLoop::DFlash1 => {
            if !verifier.needs_lin_caches() {
                eprintln!(
                    "SKIP {test}: {} carries no recurrent state, so it is not the \
                     verifier this loop drives",
                    model_path.display()
                );
                return None;
            }
            Engine::Sidecar {
                verifier,
                drafter: Drafter::DFlash1(Box::new(
                    DFlashDrafter::load(draft_path, hidden, device).expect("load DFlash 1 drafter"),
                )),
            }
        }
        RoundLoop::DFlash2 => {
            if !verifier.needs_lin_caches() {
                eprintln!(
                    "SKIP {test}: {} carries no recurrent state, so it is not the \
                     verifier this loop drives",
                    model_path.display()
                );
                return None;
            }
            Engine::Sidecar {
                verifier,
                drafter: Drafter::DFlash2(Box::new(
                    DFlash2Drafter::load(draft_path, hidden, device)
                        .expect("load DFlash 2 drafter"),
                )),
            }
        }
        RoundLoop::Eagle3 => {
            if !verifier.needs_lin_caches() {
                eprintln!(
                    "SKIP {test}: {} carries no recurrent state, so it is not the \
                     verifier this loop drives",
                    model_path.display()
                );
                return None;
            }
            // The verifier's stop ids join the drafter's reduced vocabulary, so
            // the restricted argmax can still end a turn. They are the same ids
            // both arms stop on.
            let drafter = Eagle3Drafter::load(draft_path, hidden, vocab, eos, device)
                .expect("load EAGLE-3 drafter");
            Engine::Sidecar {
                verifier,
                drafter: Drafter::Eagle3(Box::new(drafter)),
            }
        }
        RoundLoop::TwoModelGreedy => {
            let draft = arch::load_model(draft_path, device, &arch::LoadOpts::default())
                .expect("load draft model");
            Engine::TwoModel(Box::new(
                SpeculativeDispatcher::new(*verifier, draft, device).expect("two-model dispatcher"),
            ))
        }
    };
    Some(engine)
}

/// Report one pair of arms, whatever the verdict.
#[allow(clippy::too_many_arguments)]
fn report(
    test: &str,
    prompt: &Prompt,
    block: usize,
    drafter: &Path,
    tk: &tokenizers::Tokenizer,
    spec: &[u32],
    plain: &[u32],
    margins: &[f32],
    restriction: Option<&Restriction<'_>>,
    verdict: &Verdict,
) {
    let (tail_start, tail_ratio) = weakest_tail(spec, plain);
    let (spec_from, spec_period, spec_cycle) = strongest_windowed_cycle(spec);
    let (plain_from, plain_period, plain_cycle) = strongest_windowed_cycle(plain);
    let (div, confidence) = divergence_confidence(spec, plain, margins)
        .unwrap_or((common_prefix_len(spec, plain), 0.0));
    // For a pair whose drafter names a reduced vocabulary: how much of the
    // reference answer that vocabulary cannot say, whether the token it could
    // not say is the one the arms parted on, and which vocabulary decided the
    // speculative arm's token there — the two together are what the verdict
    // reads.
    let outside = match restriction {
        Some(r) => format!(
            " unnameable={} divergence_unnameable={} divergence_decided_by={:?}",
            r.unnameable(plain),
            plain.get(div).is_some_and(|id| !r.vocab.contains(id)),
            r.decided_by.get(div),
        ),
        None => String::new(),
    };
    eprintln!(
        "[{test}/{}] block={block} drafter={} lcs={:.4} tail={tail_ratio:.4}@{tail_start} divergence={div} \
         margin={:.4} confidence={confidence:.4}{outside} \
         cycle spec={spec_cycle:.4}/p{spec_period}@{spec_from} \
         plain={plain_cycle:.4}/p{plain_period}@{plain_from} spec={} plain={}\n  \
         verdict = {verdict:?}\n  spec  = {:?}\n  plain = {:?}",
        prompt.name,
        drafter
            .file_name()
            .unwrap_or(drafter.as_os_str())
            .to_string_lossy(),
        lcs_ratio(spec, plain),
        margins.get(div).copied().unwrap_or(f32::NAN),
        spec.len(),
        plain.len(),
        tk.decode(spec, false).unwrap_or_default(),
        tk.decode(plain, false).unwrap_or_default(),
    );
}

/// Write one pair's round stream for one prompt — one JSON object per round —
/// beside the timing-free form two runs are compared on, and say where both
/// went.
///
/// Under `RMLX_HOME`'s `tmp/`. The streams are a run artifact of a suite that
/// names no output directory of its own, and a variable for one would be
/// invisible configuration for a path the run already prints.
///
/// The second file is what
/// `crates/rmlx-models/tests/fixtures/spec_round_baseline/MANIFEST.sha256` pins,
/// so `shasum -a 256` on it is the whole comparison. Its byte length is printed
/// because a cell whose length already differs needs no digest to settle.
fn write_round_stream(test: &str, prompt: &str, rounds: &[CapturedEvent]) {
    let dir = rmlx_core::paths::tmp_dir();
    std::fs::create_dir_all(&dir).expect("create the run's tmp directory");
    let mut full = String::new();
    let mut stable = String::new();
    for event in rounds {
        full.push_str(&event.json_line());
        full.push('\n');
        stable.push_str(&event.stable_json_line());
        stable.push('\n');
    }
    let full_path = dir.join(format!("{test}.{prompt}.jsonl"));
    let stable_path = dir.join(format!("{test}.{prompt}.stable.jsonl"));
    let stable_len = stable.len();
    std::fs::write(&full_path, full).expect("write the round stream");
    std::fs::write(&stable_path, stable).expect("write the timing-free round stream");
    eprintln!(
        "[{test}/{prompt}] rounds={} stream={} stable={} stable_bytes={stable_len}",
        rounds.len(),
        full_path.display(),
        stable_path.display(),
    );
}

/// A round in the shape every loop emits, one from a loop that keeps no phase
/// split, the `error!` a round with a broken phase timer emits beside its own
/// line, and a request-level line that is not a round.
fn emit_two_rounds_an_overrun_and_a_request_line() {
    tracing::debug!(
        target: PHASE_SWITCH_TARGET,
        round = 3,
        accept = 2,
        num_draft = 4,
        n_committed = 3,
        emitted_total = 11,
        projected_rows = 3,
        round_ms = 20.5,
        "speculative round"
    );
    tracing::debug!(
        target: PHASE_SWITCH_TARGET,
        round = 4,
        accept = 2,
        num_draft = 4,
        n_committed = 3,
        emitted_total = 14,
        "speculative round"
    );
    // The overrun report, on the round target, at ERROR, naming the round's own
    // figures — which the recorder admits and the classifier reads by shape.
    tracing::error!(
        target: PHASE_SWITCH_TARGET,
        round = 4,
        accept = 2,
        num_draft = 4,
        refolded = false,
        charged = false,
        round_ms = 1.0,
        draft_ms = 4.0,
        "speculative round phases claim more time than the round has"
    );
    tracing::debug!(target: "rmlx_models::speculative", "a request-level line, not a round");
}

/// The recorder keeps a round as a loop emits one, and charges no phase doing
/// it.
///
/// The events below are emitted at the target every loop's round event uses,
/// and that target is itself a behaviour switch — the phases are charged when
/// it is enabled at TRACE. A capture that declined the target rather than the
/// level would return no round at all.
///
/// Mutation: in `RoundStreamRecorder::enabled`, answer `false` for every
/// metadata whose target is in `BEHAVIOUR_SWITCH_TARGETS`.
#[test]
fn the_round_stream_recorder_keeps_a_round_and_charges_no_phase() {
    let recorder = RoundStreamRecorder::new();
    tracing::subscriber::with_default(std::sync::Arc::clone(&recorder), || {
        // The two questions a round loop asks before it decides what to do.
        assert!(!tracing::enabled!(target: PHASE_SWITCH_TARGET, tracing::Level::TRACE));
        assert!(
            !tracing::enabled!(target: "rmlx_models::speculative::eagle3", tracing::Level::TRACE)
        );
        emit_two_rounds_an_overrun_and_a_request_line();
    });

    let captured = recorder.events();
    let rounds = round_events(&captured);
    assert_eq!(
        rounds.len(),
        2,
        "two rounds were emitted; neither the request-level line nor the overrun \
         report is one: {rounds:?}"
    );
    // The overrun is the negative control that matters. Two lines are emitted on
    // the round's own target and at a level the recorder admits, and this is the
    // one that is nearly a round: it names the round's index, what it accepted
    // and how many proposals it accepted them from — three of the four — so what
    // keeps it out is the fourth, and a classifier narrowed back to three would
    // put a second line in this round's stream. The other, the carry check in
    // `round_stats::log_round`, carries the round and the unforced arrays and
    // neither `accept` nor `num_draft`, so it is already three fields short of a
    // round; it also fires only on a charged round, which no capture is.
    let overrun: Vec<&CapturedEvent> = captured
        .iter()
        .filter(|e| {
            e.message
                .starts_with("speculative round phases claim more time")
        })
        .collect();
    assert_eq!(overrun.len(), 1, "the overrun report was captured");
    assert_eq!(
        overrun[0].field("emitted_total"),
        None,
        "the overrun is a statement about one round's clock and carries no running \
         emitted total, which is what excludes it: {:?}",
        overrun[0]
    );
    for f in ["round", "accept", "num_draft"] {
        assert!(
            overrun[0].field(f).is_some(),
            "the overrun names the round's own figures, so {f} is not what excludes \
             it: {:?}",
            overrun[0]
        );
    }
    // The second negative control, off the round's target: it has to be excluded
    // for the reason `round_events` gives, not by luck. A request-level line
    // that grew all four fields would otherwise quietly join the stream and this
    // fixture would still read 2.
    let control: Vec<&CapturedEvent> = captured
        .iter()
        .filter(|e| e.message == "a request-level line, not a round")
        .collect();
    assert_eq!(control.len(), 1, "the request-level line was captured");
    assert!(
        ROUND_EVENT_FIELDS
            .iter()
            .any(|f| control[0].field(f).is_none()),
        "the line this fixture uses to show a non-round is excluded carries every one \
         of {ROUND_EVENT_FIELDS:?}, so it no longer shows anything: {:?}",
        control[0]
    );
    assert_eq!(rounds[0].field("n_committed"), Some("3"));
    assert_eq!(rounds[0].field("projected_rows"), Some("3"));
    // A loop that carries no conditioning buffer, and no phase split, leaves
    // both off its line rather than reporting a zero for either.
    assert_eq!(rounds[1].field("projected_rows"), None);
    assert_eq!(rounds[1].field("round_ms"), None);

    let line: serde_json::Value =
        serde_json::from_str(&rounds[0].json_line()).expect("a round renders as one JSON object");
    assert_eq!(line["target"], PHASE_SWITCH_TARGET);
    assert_eq!(line["message"], "speculative round");
    assert_eq!(line["accept"], "2");
    assert_eq!(line["num_draft"], "4");
    assert_eq!(line["round_ms"], "20.5");

    // The form the baseline manifest pins: the same object without the fields
    // that move between two runs of the same engine.
    let stable: serde_json::Value = serde_json::from_str(&rounds[0].stable_json_line())
        .expect("a round renders as one JSON object");
    assert!(
        stable.get("round_ms").is_none(),
        "a wall-clock field survived into the form two runs are compared on: {stable}"
    );
    assert_eq!(stable["accept"], "2");
    assert_eq!(stable["num_draft"], "4");

    assert!(
        recorder.declined_both_switches(),
        "both switches were asked at TRACE and both answers must have been no: {:?}",
        recorder.questions()
    );
}

/// Two facts the engine states and its readers restate, held to the engine's
/// copy by reading the source.
///
/// **The round event's target.** `round_stats.rs` owns it and the constant is
/// private, so a capture cannot import the name it must decline to enable at
/// TRACE and restates the literal instead. Nothing held the two together: the
/// only other assertion on the target compares a line this file emitted *with*
/// `PHASE_SWITCH_TARGET` against `PHASE_SWITCH_TARGET`, which is circular, and
/// `round_events` is target-blind. Editing either literal alone left every gate
/// green and every capture reading a target the engine no longer writes on —
/// which is not a failure, it is an empty stream that looks like a run with no
/// rounds.
///
/// **What a round line carries.** `ROUND_EVENT_FIELDS` decides which events
/// reach a capture; `spec_round_stream_compare.py` decides which lines it will
/// read one back from. The silent direction is the engine gaining a field the
/// script does not require: the script then accepts a line the engine would not
/// have written, and a stream missing that field digests as a clean one.
///
/// Read out of the files rather than linked, because one of them is Python. The
/// rendering is the Rust array's, so a fifth field added on this side does not
/// compile into a passing assertion — it changes the string being looked for.
#[test]
fn the_engine_and_its_readers_state_the_same_target_and_round_fields() {
    const ROUND_STATS: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/src/speculative/round_stats.rs"
    ));
    const COMPARE: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../scripts/spec_round_stream_compare.py"
    ));

    let target = format!("const PHASE_TARGET: &str = \"{PHASE_SWITCH_TARGET}\";");
    assert!(
        ROUND_STATS.contains(&target),
        "`round_stats.rs` does not declare `{target}`, so the target this capture \
         declines to enable at TRACE is not the one the engine writes rounds on"
    );

    let rendered = ROUND_EVENT_FIELDS
        .iter()
        .map(|f| format!("\"{f}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let fields = format!("ROUND_EVENT_FIELDS = ({rendered})");
    assert!(
        COMPARE.contains(&fields),
        "`scripts/spec_round_stream_compare.py` does not state `{fields}`, so the \
         filter the engine writes a capture through and the filter that reads one \
         back are two different rules"
    );
}

// ── The gate ─────────────────────────────────────────────────────────────────

/// The assistant pair reproduces plain greedy on every prompt it answers.
///
/// The 4k document is the prompt the SWA-ring defect showed on — 4k is what
/// wraps a sliding-window ring, and past the wrap the ring used to keep the
/// rejected drafts of every round while the full-attention layers dropped
/// theirs. The short prompts never wrap it and were green throughout, which is
/// why the gate does not run on one of them alone.
#[ignore]
#[test]
fn the_assistant_round_loop_reproduces_plain_greedy() {
    run_gate(
        "the_assistant_round_loop_reproduces_plain_greedy",
        &ASSISTANT_PAIR,
    );
}

/// The recurrent pair, whose rollback has no truncation to be exact about — the
/// recurrent state has no sequence axis, so the loop restores a pre-round
/// snapshot and replays the accepted prefix.
///
/// Its subsequence agreement is far below the assistant pair's and no floor
/// separates it from a defect, which is what the divergence oracle is for. It
/// is also the pair that found one: the acceptance walk was scoring an un-normed
/// hidden through the LM head, and with that fixed three of six prompts come
/// back bit-identical.
#[ignore]
#[test]
fn the_recurrent_round_loop_reproduces_plain_greedy() {
    run_gate(
        "the_recurrent_round_loop_reproduces_plain_greedy",
        &MTP_PAIR,
    );
}

/// The recurrent pair on the 4-bit verifier at its sidecar's declared depth,
/// which is the control for the deep-block run below.
#[ignore]
#[test]
fn the_recurrent_round_loop_reproduces_plain_greedy_at_the_declared_block() {
    run_gate(
        "the_recurrent_round_loop_reproduces_plain_greedy_at_the_declared_block",
        &MTP_4BIT_PAIR,
    );
}

/// The same pair and the same loop, driven past the depth its sidecar declares.
///
/// A request may name that block, so the answer it returns is one this engine
/// serves, and it is judged by the same oracle as every other cell here: an
/// arm that parts from plain greedy is refused unless the verifier's own top-two
/// gap at the position they parted says the two were a near-tie.
#[ignore]
#[test]
fn the_recurrent_round_loop_reproduces_plain_greedy_past_the_declared_block() {
    run_gate(
        "the_recurrent_round_loop_reproduces_plain_greedy_past_the_declared_block",
        &MTP_4BIT_DEEP_PAIR,
    );
}

/// The block pair at the block it is served. Its drafter proposes a whole block
/// at once and its selector re-picks every position of that block against the
/// one before it, so a defect in either arrives as a rejected proposal — which
/// the acceptance walk absorbs silently and this gate does not. The rollback is
/// the recurrent one.
#[ignore]
#[test]
fn the_block_round_loop_reproduces_plain_greedy() {
    run_gate(
        "the_block_round_loop_reproduces_plain_greedy",
        &DFLASH2_PAIR,
    );
}

/// The same loop at the width its checkpoint declares, which is the width its
/// selector chain is defined over.
///
/// At the served block the chain re-picks three positions; here it re-picks
/// seven, over a rollback whose rejected tail is correspondingly longer.
#[ignore]
#[test]
fn the_block_round_loop_reproduces_plain_greedy_at_the_declared_block() {
    run_gate(
        "the_block_round_loop_reproduces_plain_greedy_at_the_declared_block",
        &DFLASH2_DEEP_PAIR,
    );
}

/// The adaptive pair at the block it is served, and the only loop here whose
/// verify width is not fixed.
///
/// Its loop halves and grows the block from the accept rate of the recent
/// rounds, so the width varies within a run over the same rollback the
/// recurrent and block pairs use. At the served block that variation is over
/// {4, 5}; the range the schedule is defined over is the next cell's.
#[ignore]
#[test]
fn the_adaptive_round_loop_reproduces_plain_greedy() {
    run_gate(
        "the_adaptive_round_loop_reproduces_plain_greedy",
        &DFLASH1_PAIR,
    );
}

/// The same loop over the range its schedule actually covers.
///
/// A run truncates an 8-wide append and follows it with a 4- or 6-wide one — a
/// sequence of shapes no other pair here produces, and one the served block
/// cannot reach. Every individual width is exercised elsewhere; the schedule is
/// not.
#[ignore]
#[test]
fn the_adaptive_round_loop_reproduces_plain_greedy_over_its_whole_schedule() {
    run_gate(
        "the_adaptive_round_loop_reproduces_plain_greedy_over_its_whole_schedule",
        &DFLASH1_DEEP_PAIR,
    );
}

/// The restricted-vocabulary pair.
///
/// It is judged by the same oracle as every other pair, and it carries one
/// inexactness they do not: an accepted position is emitted as the draft's
/// token, and the verifier's full-vocabulary argmax is computed at the
/// correction position only. Where those two differ the arms differ, by design
/// and in agreement with the upstream implementation, so a divergence there is
/// reported rather than refused — but only there, and [`Restriction`] says why
/// the position and not the token is what decides it.
#[ignore]
#[test]
fn the_restricted_vocab_round_loop_reproduces_plain_greedy() {
    run_gate(
        "the_restricted_vocab_round_loop_reproduces_plain_greedy",
        &EAGLE3_PAIR,
    );
}

/// The two-model pair: two complete models, and the loop with no sidecar head.
///
/// Both halves are GDN hybrids, so a partial round rolls back a KV cache and a
/// recurrent state on the verifier and the draft alike — four restores where
/// the sidecar pairs do two.
#[ignore]
#[test]
fn the_two_model_round_loop_reproduces_plain_greedy() {
    run_gate(
        "the_two_model_round_loop_reproduces_plain_greedy",
        &TWO_MODEL_PAIR,
    );
}

/// Every prompt in [`PROMPTS`], judged, and the whole table printed whatever the
/// verdicts are.
///
/// **Every prompt, not a chosen one.** Recall is a property of the set: on the
/// pairs and prompts measured here, an engine whose acceptance walk skips the
/// final norm reads inside the confidence ceiling on two of six prompts and
/// outside it on four, so a gate pinned to one prompt would be a coin toss and
/// this one is not. The cost is one pair of arms per prompt.
fn run_gate(test: &str, pair: &Pair) {
    let device = Device::Gpu;
    let Some(mut loaded) = load(pair, test, device) else {
        return;
    };
    let draft_vocab = loaded.draft_vocab();
    let mut refusals: Vec<String> = Vec::new();
    let mut judged = 0usize;
    for prompt in PROMPTS {
        let (block, spec, plain, margins, decided_by, rounds) = loaded.arms(prompt, device);
        // Written before it is judged: an empty stream is the case worth having
        // the file for.
        write_round_stream(test, prompt.name, &rounds);
        assert!(
            !rounds.is_empty() || spec.len() <= 1,
            "{test}/{}: the speculative arm emitted {} tokens and reported no round. \
             An arm that stopped on its seed closes no round and is the prompt; this \
             one ran, so the stream the comparison rests on is empty because the loop \
             stopped reporting one.",
            prompt.name,
            spec.len()
        );
        // Every pair is judged at a block or not at all — the one it names, or
        // the one the serve layer resolves for a request that names none. The
        // readings below are all attributed to a block in the report line, and
        // an arm running any other width would fill that line with a plausible
        // set under the wrong label. `want` is built from the pair and from the
        // depth the *drafter* declares, so it is not the harness reading back
        // its own choice.
        let want = pair
            .block
            .unwrap_or_else(|| default_block_for(loaded.declared_block()));
        assert_eq!(
            block, want,
            "{test}/{}: this pair runs block {want} and the loop ran {block}",
            prompt.name
        );
        let restriction = draft_vocab.as_ref().map(|vocab| Restriction {
            vocab,
            decided_by: &decided_by,
        });
        let verdict = judge(&spec, &plain, &margins, restriction.as_ref());
        report(
            test,
            prompt,
            block,
            &loaded.drafter,
            &loaded.tokenizer,
            &spec,
            &plain,
            &margins,
            restriction.as_ref(),
            &verdict,
        );
        match verdict {
            Verdict::Agreed => judged += 1,
            Verdict::Unjudgeable(_) => {}
            Verdict::Refused(why) => {
                judged += 1;
                refusals.push(format!("{}: {why}", prompt.name));
            }
        }
    }
    assert!(
        judged > 0,
        "{test}: no prompt produced a pair this gate could judge, so it asserted \
         nothing — the run is not a pass"
    );
    assert!(
        refusals.is_empty(),
        "{test}: the round loop did not reproduce plain greedy on {} of {judged} \
         judged prompts\n  {}",
        refusals.len(),
        refusals.join("\n  ")
    );
}
