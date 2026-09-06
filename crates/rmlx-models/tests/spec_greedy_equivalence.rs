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
//! Over the twelve (pair, prompt) cells each broken engine above produces, the
//! gate refuses six of six on the assistant pair and four of six on the
//! recurrent one. The two it does not refuse read 0.0000 and 0.0977 — inside the
//! ceiling — and are why the gate runs **every** prompt rather than one: recall
//! is a property of the set. Both broken engines turn both gates red. The block
//! pair's own two broken engines are refused on six of six prompts each.
//!
//! # Pairs
//!
//! Three, whose verifiers resolve by slug from `RMLX_O_MODELS_ROOT`:
//!
//! | verifier | drafter | round loop | rollback |
//! |---|---|---|---|
//! | `gemma-4-e2b-it-mxfp8` | `gemma-4-E2B-it-assistant-bf16` | shared-K/V assistant | KV truncation, SWA ring included |
//! | `Qwen3.8-27B-mxfp8` | `Qwen3.8-27B-MTP-mxfp8` | MTP sidecar | KV truncation + recurrent snapshot/replay |
//! | `Qwen3.8-27B-4bit` | `Qwen3.8-27B-DFlash2` | DFlash 2 block drafter | KV truncation + recurrent snapshot/replay |
//!
//! The first runs wherever the snapshots are; the other two run on request —
//! see [`DrafterSource`] for the shader-validation reason. `RMLX_DRAFT_TEST_MODEL`
//! names one drafter, so the pair it does not belong to stands down on the kind
//! its snapshot declares rather than loading it as something else.
//!
//! The second is the pair whose agreement no subsequence floor could separate
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

use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::speculative::dflash2::{dflash2_generate_greedy, DFlash2Drafter};
use rmlx_models::speculative::gemma4_assistant::{
    mtp_assistant_generate_greedy, Gemma4AssistantDrafter,
};
use rmlx_models::speculative::mtp::{mtp_generate_greedy, MtpDrafter};
use rmlx_models::{Declared, DraftKind};

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

/// Draft block for the assistant pair. Small enough that a rollback runs every
/// few tokens, which is the code path the oracles protect.
const BLOCK_SIZE: usize = 4;

/// Context both arms run under. Above the 4k prompt plus the budget, and the
/// same on both sides — a different cap on either would make this a measurement
/// of the cap.
const MAX_CTX: i32 = 8192;

/// Where the first divergence's own confidence may sit in the reference arm's
/// margin distribution.
///
/// **Both sides measured**, over two pairs and six prompts each, by running the
/// gate against the shipped engine and against two deliberately broken ones:
///
/// | engine | percentile of the reference arm's own margins |
/// |---|---|
/// | assistant pair, as shipped | 0.0000 to 0.0820 |
/// | recurrent pair, as shipped | 0.0000 to 0.0234 |
/// | assistant pair, SWA ring keeping its rejected block tail | 0.4219 to 0.9258 |
/// | recurrent pair, acceptance walk without the final norm | 0.0000 to 0.5000 |
///
/// The measurement leaves a band, and the value sits inside it: above the worst
/// correct cell (0.0820, 1.46x) and under the lowest broken cell above it
/// (0.1538, 1.28x). It clears every correct cell and refuses ten of the twelve
/// broken ones; the two it does not read 0.0000 and 0.0977 and are covered by
/// running every prompt rather than one, since each broken engine is refused on
/// at least four of its six.
const MAX_DIVERGENCE_CONFIDENCE: f64 = 0.12;

/// The worst [`weakest_tail`] reading a **correct** pair reached over the prompts
/// the gate judged, on either pair. Not a threshold the gate applies — see below
/// — but the reference `two_arms_in_the_same_ragged_loop_are_not_returned_as_agreement`
/// uses to decide whether a synthetic pair still looks like agreement at all.
/// `the_worst_correct_tail_is_the_worst_of_the_tails_measured` holds it to that
/// population.
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
const WORST_CORRECT_TAIL_AGREEMENT: f64 = 0.2344;

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
/// sampled**: `the_control_and_the_subsequence_floor_hand_over_inside_the_ragged_range`
/// walks two arms in the same period-8 loop from 0% to 100% raggedness over four
/// seed pairs and asserts that this control and the subsequence floors between
/// them refuse every point. Twenty seed pairs over the band where they hand over
/// leave the range covered at 0.22 and open the first hole at 0.24, so the value
/// here has room rather than sitting on the boundary. An earlier 0.50 — placed
/// from six sampled points — left 34% to 52% passing.
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

/// What one pair of arms says about the round loop.
#[derive(Debug, PartialEq, Eq)]
enum Verdict {
    /// The round loop reproduced what the verifier decodes alone.
    Agreed,
    /// The pair says nothing either way, and the reason is a property of the
    /// prompt or the model rather than of the round loop. Reported, not failed:
    /// a gate that turns red on an input it cannot read teaches its operator to
    /// ignore it.
    Unjudgeable(String),
    /// The round loop did not reproduce plain greedy.
    Refused(String),
}

/// Judge one pair of streams.
///
/// `margins` is the reference arm's per-position top-two logprob gap, or empty
/// when the caller has none — the synthetic fixtures below run that way, and the
/// divergence oracle stands down rather than inventing a distribution.
fn judge(spec: &[u32], plain: &[u32], margins: &[f32]) -> Verdict {
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
    if shorter < MIN_ANSWER_TOKENS {
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
        judge(&one_flip, &plain, &tied).refusal().is_none(),
        "a flip at the arm's own least-confident decision must pass"
    );

    let mut confident = vec![0.01f32; N_TOKENS];
    for m in confident.iter_mut().take(N_TOKENS / 3) {
        *m = 0.001;
    }
    confident[66] = 5.0;
    let failure = judge(&one_flip, &plain, &confident)
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
        let failure = judge(&stream, &healthy, &[])
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

    let spec_side = judge(&looping, &healthy, &[]);
    let why = spec_side
        .refusal()
        .expect("a degenerate spec arm must be refused");
    assert!(why.contains("speculative arm"), "{why}");
    assert!(
        why.contains("the verifier does not produce on its own"),
        "{why}"
    );

    let plain_side = judge(&healthy, &looping, &[]);
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
        matches!(judge(&short, &short, &[]), Verdict::Unjudgeable(ref why) if why.contains("both arms")),
        "two arms that both stopped short say nothing about the round loop"
    );

    let long: Vec<u32> = Rng(0x5170_0000).prose(N_TOKENS);
    let why = judge(&short, &long, &[])
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
/// so their agreement means nothing and the divergence oracle has no healthy
/// reference arm whose margins say anything. The repetition control is what
/// keeps it out, and past the raggedness where the control's reading falls under
/// its ceiling the arms no longer agree either — their tail agreement is at or
/// under [`WORST_CORRECT_TAIL_AGREEMENT`], the worst a correct pair reached.
///
/// The property is asserted over the whole range and over four seed pairs, so
/// the sweep picks the points. An earlier revision claimed the range was covered
/// with no gap and sampled six values of the parameter to say so; at 36% the
/// pair passed.
#[test]
fn two_arms_in_the_same_ragged_loop_are_not_returned_as_agreement() {
    for noise in (0..=100).step_by(2) {
        for (sa, sb) in [(0x33u64, 0x44u64), (0x91, 0xA7), (0xB3, 0xC1), (0xD5, 0xE9)] {
            let (a, b) = (ragged_loop_arm(sa, noise), ragged_loop_arm(sb, noise));
            if judge(&a, &b, &[]) != Verdict::Agreed {
                continue;
            }
            let tail = weakest_tail(&a, &b).1;
            assert!(
                tail <= WORST_CORRECT_TAIL_AGREEMENT,
                "two arms {noise}% ragged in the same period-8 loop (seeds \
                 {sa:#x}/{sb:#x}) were returned as agreement while agreeing better \
                 ({tail:.4}) than the worst correct pair measured \
                 ({WORST_CORRECT_TAIL_AGREEMENT}): cycles {:.4} and {:.4}",
                strongest_windowed_cycle(&a).2,
                strongest_windowed_cycle(&b).2,
            );
        }
    }
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
    /// Every [`weakest_tail`] reading a correct pair reached, over both pairs and
    /// the prompts each of them judged. These are tails, not [`lcs_ratio`] over
    /// the whole arm — the same runs read 0.4766 to 1.0000 on that measure, and
    /// the two figures the "no subsequence floor" paragraphs quote come from it.
    const MEASURED: &[f64] = &[
        // Assistant pair, six prompts.
        0.3125, 0.3750, 0.3906, 0.2344, 1.0000, 0.9062,
        // Recurrent pair, the five prompts it judged — the 4k document is
        // answered in 13 and 26 tokens and is reported unjudgeable.
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
        let failure = judge(&spec, &healthy, &[]).refusal().unwrap_or_else(|| {
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
    let failure = judge(truncated, &plain, &[])
        .refusal()
        .expect("the length guard must refuse it");
    assert!(failure.contains("stopped well before"), "{failure}");

    // Just inside the guard the same shape scores 1.0 and passes, which is what
    // makes the guard — not the ratio — the thing doing the work above.
    assert!(judge(&plain[..N_TOKENS * 3 / 2], &plain, &[])
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
        judge(&at_floor, &at_floor, &[]).refusal().is_none(),
        "two arms exactly at the floor must pass"
    );
    let under = &at_floor[..MIN_ANSWER_TOKENS - 1];
    assert!(
        judge(under, under, &[]) != Verdict::Agreed,
        "two arms one token under the floor must not be returned as agreement"
    );

    // Every arm the gate judged ran to the budget, so the floor is under all of
    // them. Driven through `judge` rather than compared as constants.
    let budget = Rng(0x2181_2181).prose(N_TOKENS);
    assert!(
        judge(&budget, &budget, &[]).refusal().is_none(),
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
    let failure = judge(&plain[..16], &plain, &[])
        .refusal()
        .expect("must refuse");
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
    let failure = judge(&late, &plain, &confident)
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
    assert!(judge(&late, &plain, &tied).refusal().is_none());
}

/// The two measured regimes put through `judge` itself, so the test fails when
/// the oracle's composition changes and not only when a constant moves.
///
/// Correct: the worst reading a shipped pair produced, a first divergence at the
/// 8.2nd percentile of the reference arm's own margins. Broken: the lowest
/// reading a broken engine produced above the ceiling, the 15.4th — the cell the
/// recorded ten-of-twelve recall stands or falls on.
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
        judge(&spec, &plain, &correct).refusal().is_none(),
        "the worst correct regime must pass"
    );

    // 0.1538 is 2/13 — the broken cell's reference arm was 13 tokens long, and a
    // 256-position array gets within one of its own quantum of it.
    let broken = margins_reading(39);
    let (_, lowest_broken) = divergence_confidence(&spec, &plain, &broken).expect("a divergence");
    assert!(
        (lowest_broken - 0.1538).abs() < 0.002,
        "the reconstructed broken regime reads {lowest_broken:.4}, not the measured 0.1538"
    );
    let failure = judge(&spec, &plain, &broken)
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
    /// DFlash 2 block drafter: the same recurrent rollback, and a drafter that
    /// denoises a whole block at once from the verifier's multi-layer hidden
    /// states rather than stepping through it.
    DFlash2,
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
            RoundLoop::DFlash2 => Some(DraftKind::DFlash2),
        }
    }
}

/// One verifier + drafter the gate can run.
struct Pair {
    verifier: common::GoldenModel,
    drafter: DrafterSource,
    round_loop: RoundLoop,
}

/// How a pair's drafter is found, and so whether `make gpu-test` selects the
/// pair on a machine that merely holds the snapshots.
enum DrafterSource {
    /// Resolved by slug from `RMLX_O_MODELS_ROOT`, like the verifier — the pair
    /// runs wherever the snapshots are.
    Slug(&'static str),
    /// Named by an operator or not run at all.
    ///
    /// Its verifier drives an MLX quantized matmul whose `load_safe` bound is
    /// the one `scripts/gpu_validation_census.txt` records, so a run under Metal
    /// shader validation reports over a thousand invalid loads from a kernel
    /// this repo does not compile. The census pins one exact count per test, and
    /// a count from a 256-token generation moves with every prompt — so pinning
    /// such a pair would make the census brittle rather than informative. Until
    /// that is settled these pairs run on request and `make gpu-test` reports
    /// them as skipped, with the variable that would run them named.
    Named,
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
    drafter: DrafterSource::Named,
    round_loop: RoundLoop::MtpSidecar,
};

/// The block pair. Its drafter denoises a whole block in one pass and its
/// selector chains the block's independent argmaxes into one sentence, so an
/// error in either reaches the verifier as a rejected proposal rather than as a
/// failure — which the acceptance walk absorbs, and this gate does not.
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
    drafter: DrafterSource::Named,
    round_loop: RoundLoop::DFlash2,
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
    let Some(named) = std::env::var(var).ok().filter(|v| !v.is_empty()) else {
        let root = std::env::var(common::MODELS_ROOT_VAR).ok();
        return match common::slug_snapshot(root.as_deref(), slug, common::Role::Sidecar) {
            common::Snapshot::Found { path, .. } => common::Gate::Run { path, note: None },
            common::Snapshot::Absent(why) => {
                common::Gate::Skip(why.replace(common::SINGLE_MODEL_VAR, var))
            }
            common::Snapshot::Misconfigured(why) => common::Gate::Fail(why),
        };
    };
    // The operator named this path, so a typo or a moved snapshot breaks the
    // run rather than skipping it. `override_snapshot` reports only an unset or
    // empty value as `None`, and this branch holds a non-empty one.
    let probed =
        common::override_snapshot(Some(&named), common::Role::Sidecar).unwrap_or_else(|| {
            common::Snapshot::Misconfigured(format!("{var} is set to an empty value"))
        });
    match probed {
        common::Snapshot::Found { path, .. } => common::Gate::Run { path, note: None },
        common::Snapshot::Absent(why) | common::Snapshot::Misconfigured(why) => {
            common::Gate::Fail(why.replace(common::SINGLE_MODEL_VAR, var))
        }
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
    let sampler_cfg = rmlx_models::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 2,
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

/// A loaded pair, ready to answer prompts.
struct Loaded {
    verifier: arch::Architecture,
    drafter: Drafter,
    tokenizer: tokenizers::Tokenizer,
    eos: Vec<u32>,
}

enum Drafter {
    Assistant(Box<Gemma4AssistantDrafter>),
    Mtp(Box<MtpDrafter>),
    DFlash2(Box<DFlash2Drafter>),
}

impl Loaded {
    /// Both arms over one prompt: the speculative ids, the reference ids, and
    /// the reference's per-position margins.
    fn arms(&mut self, prompt: &Prompt, device: Device) -> (Vec<u32>, Vec<u32>, Vec<f32>) {
        let ids = prompt.ids(&self.tokenizer);
        let mut spec_ids: Vec<u32> = Vec::new();
        {
            let mut step = |s: &rmlx_models::ProbeStep| {
                spec_ids.push(s.token_id);
                None
            };
            match &mut self.drafter {
                Drafter::Assistant(drafter) => mtp_assistant_generate_greedy(
                    &self.verifier,
                    drafter,
                    &self.tokenizer,
                    &ids,
                    N_TOKENS,
                    BLOCK_SIZE,
                    Some(rmlx_kv_quant::KvQuant::None),
                    Some(MAX_CTX),
                    &self.eos,
                    &mut step,
                    device,
                )
                .expect("assistant speculative generate"),
                Drafter::Mtp(drafter) => {
                    let block = drafter.block_size();
                    mtp_generate_greedy(
                        &self.verifier,
                        drafter,
                        &self.tokenizer,
                        &ids,
                        N_TOKENS,
                        block,
                        Some(rmlx_kv_quant::KvQuant::None),
                        Some(MAX_CTX),
                        &self.eos,
                        &mut step,
                        device,
                    )
                    .expect("mtp speculative generate")
                }
                // The block the drafter was trained at, not the harness's: the
                // whole point of a block drafter is the block, and this is the
                // width its selector chain is defined over.
                Drafter::DFlash2(drafter) => {
                    let block = drafter.cfg.block_size;
                    dflash2_generate_greedy(
                        &self.verifier,
                        drafter,
                        &self.tokenizer,
                        &ids,
                        N_TOKENS,
                        block,
                        Some(rmlx_kv_quant::KvQuant::None),
                        Some(MAX_CTX),
                        &self.eos,
                        &mut step,
                        device,
                    )
                    .expect("dflash2 speculative generate")
                }
            };
        }
        let (plain_ids, margins) =
            plain_greedy(&self.verifier, &self.tokenizer, &ids, &self.eos, device);
        (spec_ids, plain_ids, margins)
    }
}

/// Load a pair, or say why the gate stood down.
fn load(pair: &Pair, test: &str, device: Device) -> Option<Loaded> {
    // The drafter first: a pair the operator has not named stands down before
    // anything loads a verifier.
    let named = std::env::var(DRAFT_MODEL_VAR)
        .ok()
        .filter(|v| !v.is_empty());
    let draft_gate = match (&pair.drafter, named.is_some()) {
        (DrafterSource::Slug(slug), _) => resolve(DRAFT_MODEL_VAR, slug),
        (DrafterSource::Named, true) => resolve(DRAFT_MODEL_VAR, ""),
        (DrafterSource::Named, false) => common::Gate::Skip(format!(
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

    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    let hidden = verifier.hidden_size();
    let drafter = match pair.round_loop {
        RoundLoop::Gemma4Assistant => {
            if !is_gemma4_assistant(&draft_path) {
                eprintln!(
                    "SKIP {test}: {} is not a Gemma4 assistant drafter",
                    draft_path.display()
                );
                return None;
            }
            let declared = declared_backbone_hidden(&draft_path);
            if declared.is_some_and(|width| width != hidden) {
                eprintln!(
                    "SKIP {test}: {} projects into a backbone {} wide and {} is {hidden}",
                    draft_path.display(),
                    declared.unwrap_or(hidden),
                    model_path.display(),
                );
                return None;
            }
            Drafter::Assistant(Box::new(
                Gemma4AssistantDrafter::load(&draft_path, hidden, device)
                    .expect("load assistant drafter"),
            ))
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
            Drafter::Mtp(Box::new(
                MtpDrafter::load(&draft_path, hidden, device).expect("load MTP sidecar"),
            ))
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
            Drafter::DFlash2(Box::new(
                DFlash2Drafter::load(&draft_path, hidden, device).expect("load DFlash 2 drafter"),
            ))
        }
    };

    let tokenizer =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let eos = eos_ids(&model_path);
    assert!(
        !eos.is_empty(),
        "the verifier config must name its stop ids — without them both arms run past \
         the answer into end-of-turn filler and the comparison is over that"
    );
    Some(Loaded {
        verifier,
        drafter,
        tokenizer,
        eos,
    })
}

/// Report one pair of arms, whatever the verdict.
#[allow(clippy::too_many_arguments)]
fn report(
    test: &str,
    prompt: &Prompt,
    tk: &tokenizers::Tokenizer,
    spec: &[u32],
    plain: &[u32],
    margins: &[f32],
    verdict: &Verdict,
) {
    let (tail_start, tail_ratio) = weakest_tail(spec, plain);
    let (spec_from, spec_period, spec_cycle) = strongest_windowed_cycle(spec);
    let (plain_from, plain_period, plain_cycle) = strongest_windowed_cycle(plain);
    let (div, confidence) = divergence_confidence(spec, plain, margins)
        .unwrap_or((common_prefix_len(spec, plain), 0.0));
    eprintln!(
        "[{test}/{}] lcs={:.4} tail={tail_ratio:.4}@{tail_start} divergence={div} \
         margin={:.4} confidence={confidence:.4} \
         cycle spec={spec_cycle:.4}/p{spec_period}@{spec_from} \
         plain={plain_cycle:.4}/p{plain_period}@{plain_from} spec={} plain={}\n  \
         verdict = {verdict:?}\n  spec  = {:?}\n  plain = {:?}",
        prompt.name,
        lcs_ratio(spec, plain),
        margins.get(div).copied().unwrap_or(f32::NAN),
        spec.len(),
        plain.len(),
        tk.decode(spec, false).unwrap_or_default(),
        tk.decode(plain, false).unwrap_or_default(),
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

/// The block pair. Its drafter proposes a whole block at once and its selector
/// re-picks every position of that block against the one before it, so a defect
/// in either arrives as a rejected proposal — which the acceptance walk absorbs
/// silently and this gate does not. The rollback is the recurrent one, driven
/// at a wider block than any other pair here reaches.
#[ignore]
#[test]
fn the_block_round_loop_reproduces_plain_greedy() {
    run_gate(
        "the_block_round_loop_reproduces_plain_greedy",
        &DFLASH2_PAIR,
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
    let mut refusals: Vec<String> = Vec::new();
    let mut judged = 0usize;
    for prompt in PROMPTS {
        let (spec, plain, margins) = loaded.arms(prompt, device);
        let verdict = judge(&spec, &plain, &margins);
        report(
            test,
            prompt,
            &loaded.tokenizer,
            &spec,
            &plain,
            &margins,
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
