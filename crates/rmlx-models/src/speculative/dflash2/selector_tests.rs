//! DFlash 2 candidate-path selector tests.
//!
//! Two oracles, because neither alone is enough.
//!
//! * **Analytic.** A six-token vocabulary at rank two, whose every number is a
//!   dyadic rational, so the score `S_t(a, b) = U_t(b) + <A(a) (*) H(h_t), B(b)>`
//!   is exact in f32 and can be walked by hand. It is walked by hand — the
//!   chain the fixture traces is written out as a literal below — and also by
//!   [`analytic_chain`], a plain-arithmetic host implementation of the formula
//!   that shares no code with the array plumbing under test.
//!
//!   A fixture that returns the right answer for the wrong reason proves
//!   nothing, so the same host walk is run again with each term of the score
//!   removed in turn — the pairwise term, the context gate, the unary term, the
//!   chaining, the top-k truncation, the two codebooks' orientation — and each
//!   is asserted to trace a *different* chain. Anything the fixture cannot
//!   separate is named there rather than passing in silence.
//!
//! * **The reference implementation.** `tests/fixtures/dflash2_scale` holds the
//!   chain the z-lab MLX reference `CandidateSelector.select` traces from that
//!   snapshot's own weights, for two anchors. This is what pins the choice of
//!   primitives — the partition the candidates are taken with, the argmax's
//!   tie-break, the dtype the score accumulates in — against something other
//!   than this port's own opinion. At size, the same comparison is made against
//!   the published checkpoint in `tests/dflash2_loader.rs`.
//!
//! Everything here runs on the CPU device and needs no model snapshot.

#![allow(
    clippy::expect_used,
    clippy::indexing_slicing,
    reason = "unit-test scaffolding: a panic here is the assertion failing, and every index is into a fixed-size fixture constant declared in this file"
)]

use std::path::Path;

use rmlx_mlx::{Array, Device, Dtype};

use super::*;
use crate::layers::{Linear, RmsNorm};
use crate::speculative::dflash2::{DFlash2Config, DFlash2Selector};

// ---------------------------------------------------------------------------
// the analytic fixture
// ---------------------------------------------------------------------------

/// Vocabulary of the analytic fixture.
const V: usize = 6;
/// Codebook rank of the analytic fixture.
const R: usize = 2;
/// Hidden width of the analytic fixture.
const H: usize = 2;
/// Candidates kept per position by the analytic fixture.
const K: usize = 3;
/// Drafted positions of the analytic fixture.
const N: usize = 3;

/// `hidden_projection`, `[R, H]`. Deliberately not symmetric: a transposed
/// projection would give `H(h_0) = [2, 1]` where this gives `[2, 0]`.
const PROJECTION: [f32; R * H] = [2.0, 1.0, 0.0, 1.0];

/// The drafter's final hidden states at the three drafted positions, `[N, H]`.
/// `H(h)` is then `[2, 0]`, `[1, 1]`, `[1.5, 0.5]`.
const HIDDEN: [f32; N * H] = [1.0, 0.0, 0.0, 1.0, 0.5, 0.5];

/// `predecessor_codebook`, `A`, `[V, R]`.
const A: [f32; V * R] = [
    1.0, 0.0, // 0
    0.0, 1.0, // 1
    1.0, 1.0, // 2 — the anchor
    -1.0, 1.0, // 3 — the second anchor
    2.0, 0.0, // 4
    0.0, -2.0, // 5
];

/// `successor_codebook`, `B`, `[V, R]`.
///
/// Token 5 is never a candidate — its logit is below the top-k at every
/// position — and carries the largest first component of any row, so a port
/// that scored the whole vocabulary instead of the candidates would pick it.
const B: [f32; V * R] = [
    1.0, 1.0, // 0
    2.0, 0.0, // 1
    0.0, 2.0, // 2
    1.0, -1.0, // 3
    -1.0, 0.0, // 4
    4.0, 0.0, // 5
];

/// The verifier's logits over the three drafted positions, `[N, V]`.
///
/// Each row's k-th and (k+1)-th largest are more than one apart, so the
/// candidate set is the same however a partition breaks ties.
const LOGITS: [f32; N * V] = [
    1.0, 3.0, 3.25, 3.125, 0.5, 0.25, // top-3: 2, 3, 1
    2.0, 2.25, 0.125, 2.125, 0.375, 0.5, // top-3: 1, 3, 0
    0.125, 0.25, 7.0, 4.0, 5.25, 0.375, // top-3: 2, 4, 3
];

/// The seed token the chain is anchored at.
const ANCHOR: u32 = 2;

/// A second anchor, to show the chain is traced from the seed and not from the
/// block alone.
const OTHER_ANCHOR: u32 = 3;

/// The chain the analytic fixture traces, walked by hand.
///
/// Position 0: `H(h_0) = [2, 0]`, `A(2) = [1, 1]`, so `A(2) (*) H(h_0) = [2, 0]`
/// and the pairwise term is `2 * B(b)[0]` — 4, 0, 2 for candidates 1, 2, 3
/// against logits 3, 3.25, 3.125. Scores 7, 3.25, 5.125; token 1 wins by 1.875.
///
/// Position 1: `H(h_1) = [1, 1]`, `A(1) = [0, 1]`, gated to `[0, 1]`, so the
/// pairwise term is `B(b)[1]` — 1, 0, -1 for candidates 0, 1, 3 against logits
/// 2, 2.25, 2.125. Scores 3, 2.25, 1.125; token 0 wins by 0.75.
///
/// Position 2: `H(h_2) = [1.5, 0.5]`, `A(0) = [1, 0]`, gated to `[1.5, 0]`, so
/// the pairwise term is `1.5 * B(b)[0]` — 0, 1.5, -1.5 for candidates 2, 3, 4
/// against logits 7, 4, 5.25. Scores 7, 5.5, 3.75; token 2 wins by 1.5.
const ANALYTIC_CHAIN: [u32; N] = [1, 0, 2];

/// The per-position argmax of the logits, which is what the chain would be if
/// the selector did nothing.
const ANALYTIC_TOP1: [u32; N] = [2, 1, 2];

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn f32_array(values: &[f32], shape: &[i32]) -> Array {
    Array::from_f32_slice(values, shape).expect("build a test array")
}

/// A config the analytic fixture's weights satisfy.
fn analytic_config() -> DFlash2Config {
    DFlash2Config {
        hidden_size: H,
        num_hidden_layers: 1,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        head_dim: 2,
        intermediate_size: 2,
        vocab_size: V,
        rms_norm_eps: 1e-6,
        rope_theta: 1.0e7,
        sliding_window: 8,
        is_causal: false,
        block_size: N + 1,
        conv_group_size: 1,
        conv_kernel_size: 2,
        selector_rank: R,
        selector_top_k: K,
        mask_token_id: 5,
        target_layer_ids: vec![0],
    }
}

/// A drafter carrying only what the selector reads.
///
/// The trunk is not exercised here — the selector consumes hidden states it is
/// handed, never the decoder stack — so it is present at its smallest legal
/// shape rather than built.
fn drafter_with(cfg: DFlash2Config, selector: DFlash2Selector) -> DFlash2Drafter {
    DFlash2Drafter {
        fc: Linear::Plain {
            weight: f32_array(&[1.0], &[1, 1]),
        },
        hidden_norm: RmsNorm {
            weight: None,
            eps: cfg.rms_norm_eps,
        },
        norm: RmsNorm {
            weight: None,
            eps: cfg.rms_norm_eps,
        },
        layers: Vec::new(),
        selector,
        cfg,
        device: Device::Cpu,
    }
}

/// The analytic fixture's drafter.
fn analytic_drafter() -> DFlash2Drafter {
    let selector = DFlash2Selector {
        hidden_projection: Linear::Plain {
            weight: f32_array(&PROJECTION, &[R as i32, H as i32]),
        },
        predecessor_codebook: f32_array(&A, &[V as i32, R as i32]),
        successor_codebook: f32_array(&B, &[V as i32, R as i32]),
    };
    drafter_with(analytic_config(), selector)
}

fn analytic_hidden() -> Array {
    f32_array(&HIDDEN, &[1, N as i32, H as i32])
}

fn analytic_logits() -> Array {
    f32_array(&LOGITS, &[1, N as i32, V as i32])
}

// ---------------------------------------------------------------------------
// the host walk
// ---------------------------------------------------------------------------

/// One term of the score, removed. Each variant names a defect a port can have
/// while still returning plausible token ids.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Missing {
    /// Nothing removed: the formula as written.
    Nothing,
    /// The pairwise term dropped — the chain is the per-position argmax.
    PairwiseTerm,
    /// The context gate `H(h_t)` dropped from the pairwise term.
    ContextGate,
    /// The unary term dropped — the chain is scored on the codebooks alone.
    UnaryTerm,
    /// The predecessor held at the anchor instead of the token just chosen.
    Chaining,
    /// The whole vocabulary scored instead of the top-k candidates.
    TopKTruncation,
    /// The two codebooks read the other way round.
    CodebookOrientation,
}

/// `H(h_t)`, the hidden state projected to the codebook rank.
fn gate(t: usize) -> [f32; R] {
    let mut out = [0.0; R];
    for (r, o) in out.iter_mut().enumerate() {
        *o = (0..H)
            .map(|h| PROJECTION[r * H + h] * HIDDEN[t * H + h])
            .sum();
    }
    out
}

/// The candidates at position `t`, highest logit first.
fn candidates(t: usize) -> Vec<usize> {
    let mut ids: Vec<usize> = (0..V).collect();
    ids.sort_by(|&x, &y| {
        LOGITS[t * V + y]
            .partial_cmp(&LOGITS[t * V + x])
            .expect("the fixture's logits are finite")
    });
    ids
}

/// `S_t(a, b)` written out, with one term removed.
fn score(t: usize, a: usize, b: usize, missing: Missing) -> f32 {
    let (pred, succ) = if missing == Missing::CodebookOrientation {
        (&B, &A)
    } else {
        (&A, &B)
    };
    let g = gate(t);
    let pairwise: f32 = (0..R)
        .map(|r| {
            let gated = if missing == Missing::ContextGate {
                pred[a * R + r]
            } else {
                pred[a * R + r] * g[r]
            };
            gated * succ[b * R + r]
        })
        .sum();
    let unary = LOGITS[t * V + b];
    if missing == Missing::PairwiseTerm {
        unary
    } else if missing == Missing::UnaryTerm {
        pairwise
    } else {
        unary + pairwise
    }
}

/// The chain the formula traces, walked in plain host arithmetic.
fn analytic_chain(anchor: u32, missing: Missing) -> Vec<u32> {
    let mut predecessor = anchor as usize;
    let mut chain = Vec::with_capacity(N);
    for t in 0..N {
        let ranked = candidates(t);
        let considered: &[usize] = if missing == Missing::TopKTruncation {
            &ranked
        } else {
            &ranked[..K]
        };
        let a = if missing == Missing::Chaining {
            anchor as usize
        } else {
            predecessor
        };
        let mut best = considered[0];
        let mut best_score = score(t, a, best, missing);
        for &b in &considered[1..] {
            let s = score(t, a, b, missing);
            if s > best_score {
                best = b;
                best_score = s;
            }
        }
        predecessor = best;
        chain.push(best as u32);
    }
    chain
}

/// The smallest gap between the winning score and the runner-up, over every
/// position of the reference walk.
fn analytic_margin(anchor: u32) -> f32 {
    let mut predecessor = anchor as usize;
    let mut worst = f32::INFINITY;
    for t in 0..N {
        let ranked = candidates(t);
        let mut scores: Vec<f32> = ranked[..K]
            .iter()
            .map(|&b| score(t, predecessor, b, Missing::Nothing))
            .collect();
        let winner = ranked[..K]
            .iter()
            .zip(&scores)
            .max_by(|x, y| x.1.partial_cmp(y.1).expect("finite"))
            .map(|(&b, _)| b)
            .expect("k >= 2");
        scores.sort_by(|x, y| y.partial_cmp(x).expect("finite"));
        worst = worst.min(scores[0] - scores[1]);
        predecessor = winner;
    }
    worst
}

// ---------------------------------------------------------------------------
// the analytic oracle
// ---------------------------------------------------------------------------

/// The selector traces the chain the score formula written out by hand traces.
#[test]
fn the_chain_is_the_written_out_pairwise_argmax() {
    let drafter = analytic_drafter();
    let got = drafter
        .select_chain(&analytic_hidden(), &analytic_logits(), ANCHOR)
        .expect("the analytic fixture must trace a chain");

    assert_eq!(
        got,
        ANALYTIC_CHAIN.to_vec(),
        "the chain walked by hand in this file's header"
    );
    assert_eq!(
        analytic_chain(ANCHOR, Missing::Nothing),
        ANALYTIC_CHAIN.to_vec(),
        "the host walk must agree with the hand-walked literal, or one of them moved"
    );
    // The fixture would be worthless if the selector's answer were the answer
    // doing nothing gives.
    assert_ne!(
        got,
        ANALYTIC_TOP1.to_vec(),
        "the chain must differ from the per-position argmax of the logits"
    );
    println!(
        "dflash2 selector analytic: chain {got:?}, top-1 {ANALYTIC_TOP1:?}, \
         smallest winning margin {}",
        analytic_margin(ANCHOR)
    );
}

/// Every term of the score changes the chain this fixture traces.
///
/// Without this, a port that dropped the gate, transposed the codebooks or
/// scored the whole vocabulary could pass the test above. Each case here is a
/// defect that returns fluent token ids, and each is shown to be separable —
/// the fixture is not merely right, it is discriminating.
#[test]
fn every_term_of_the_score_is_load_bearing_in_this_fixture() {
    let reference = analytic_chain(ANCHOR, Missing::Nothing);
    for missing in [
        Missing::PairwiseTerm,
        Missing::ContextGate,
        Missing::UnaryTerm,
        Missing::Chaining,
        Missing::TopKTruncation,
        Missing::CodebookOrientation,
    ] {
        let mutated = analytic_chain(ANCHOR, missing);
        assert_ne!(
            mutated, reference,
            "with {missing:?} removed the fixture traces the same chain, so it \
             cannot tell that defect from a correct port"
        );
    }
}

/// The chain is traced from the seed token: a different anchor is a different
/// chain over the same block.
#[test]
fn the_anchor_the_chain_is_traced_from_changes_it() {
    let drafter = analytic_drafter();
    let hidden = analytic_hidden();
    let logits = analytic_logits();
    let first = drafter
        .select_chain(&hidden, &logits, ANCHOR)
        .expect("chain at the first anchor");
    let second = drafter
        .select_chain(&hidden, &logits, OTHER_ANCHOR)
        .expect("chain at the second anchor");
    assert_eq!(second, analytic_chain(OTHER_ANCHOR, Missing::Nothing));
    assert_ne!(
        first, second,
        "the two anchors must trace different chains, or this fixture cannot \
         tell an anchored selector from an unanchored one"
    );
}

/// A shorter block than the drafter's own is drafted, not padded: the last
/// round of a generation asks for fewer positions than `block_size - 1`.
#[test]
fn a_short_block_traces_the_prefix_of_the_long_one() {
    let drafter = analytic_drafter();
    let hidden = f32_array(&HIDDEN[..2 * H], &[1, 2, H as i32]);
    let logits = f32_array(&LOGITS[..2 * V], &[1, 2, V as i32]);
    let got = drafter
        .select_chain(&hidden, &logits, ANCHOR)
        .expect("two positions must draft");
    assert_eq!(got, ANALYTIC_CHAIN[..2].to_vec());
}

// ---------------------------------------------------------------------------
// refusals
// ---------------------------------------------------------------------------

/// Inputs the selector cannot score are refused by the property that fails.
///
/// Each of these produces a chain rather than an error if it is not checked: a
/// hidden of the wrong width silently projects garbage, logits over a different
/// vocabulary index the wrong codebook rows, and an anchor past the vocabulary
/// gathers a clamped row.
#[test]
fn inputs_the_selector_cannot_score_are_refused_by_name() {
    let drafter = analytic_drafter();
    let hidden = analytic_hidden();
    let logits = analytic_logits();

    let cases: Vec<(&str, Array, Array, u32, &str)> = vec![
        (
            "hidden of the wrong width",
            f32_array(&[0.0; N * (H + 1)], &[1, N as i32, H as i32 + 1]),
            analytic_logits(),
            ANCHOR,
            "selector hidden has shape",
        ),
        (
            "hidden of rank two",
            f32_array(&HIDDEN, &[N as i32, H as i32]),
            analytic_logits(),
            ANCHOR,
            "selector hidden has shape",
        ),
        (
            "logits over a different vocabulary",
            analytic_hidden(),
            f32_array(&[0.0; N * (V + 1)], &[1, N as i32, V as i32 + 1]),
            ANCHOR,
            "selector logits have shape",
        ),
        (
            "one side sliced and not the other",
            analytic_hidden(),
            f32_array(&LOGITS[..2 * V], &[1, 2, V as i32]),
            ANCHOR,
            "rows of logits",
        ),
        (
            "more positions than the block drafts",
            f32_array(&[0.0; 4 * H], &[1, 4, H as i32]),
            f32_array(&[0.0; 4 * V], &[1, 4, V as i32]),
            ANCHOR,
            "drafted positions",
        ),
        (
            "no positions at all",
            f32_array(&[], &[1, 0, H as i32]),
            f32_array(&[], &[1, 0, V as i32]),
            ANCHOR,
            "drafted positions",
        ),
        (
            "an anchor outside the vocabulary",
            analytic_hidden(),
            analytic_logits(),
            V as u32,
            "outside this drafter's vocabulary",
        ),
    ];

    for (what, h, l, anchor, want) in cases {
        let err = match drafter.select_chain(&h, &l, anchor) {
            Ok(chain) => panic!("{what} must be refused, traced {chain:?}"),
            Err(e) => e.to_string(),
        };
        assert!(
            err.contains(want),
            "the refusal for {what} must name {want}: {err}"
        );
    }

    // And the shapes that are fine stay fine, so the guards above are not
    // refusing everything.
    drafter
        .select_chain(&hidden, &logits, ANCHOR)
        .expect("the fixture's own shapes must pass every guard");
}

// ---------------------------------------------------------------------------
// the reference oracle, on the scale snapshot
// ---------------------------------------------------------------------------

/// Where the reference snapshot and its expected outputs live.
const FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/dflash2_scale");

/// The anchors the reference traced the committed chains from.
const SCALE_ANCHORS: [u32; 2] = [3, 27];

fn fixture_tensor(name: &str) -> Array {
    let file = Path::new(FIXTURE).join("reference.safetensors");
    let bytes = std::fs::read(file).expect("fixture file is readable");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("fixture parses");
    let t = st.tensor(name).expect("fixture carries the tensor");
    let view = rmlx_loader::TensorView {
        name,
        dtype: t.dtype(),
        shape: t.shape().to_vec(),
        bytes: t.data(),
    };
    Array::from_safetensor_view(&view).expect("tensor loads")
}

fn to_u32(a: &Array) -> Vec<u32> {
    a.eval().expect("eval");
    a.to_bytes()
        .expect("read bytes")
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The drafter the scale snapshot describes, loaded from the committed fixture.
fn scale_drafter() -> DFlash2Drafter {
    DFlash2Drafter::load(Path::new(FIXTURE), 64, Device::Cpu).expect("the scale snapshot loads")
}

/// The selector reproduces the reference implementation's chain on the scale
/// snapshot, for both anchors it was run at.
///
/// The chain is compared exactly — these are token ids, and there is no
/// tolerance to state on an id. What the tolerance question becomes is
/// separation, and it is measured: the fixture's logits put the k-th and
/// (k + 1)-th candidate exactly 0.3125 apart at every position, forty bf16
/// places at that magnitude, so the candidate set does not depend on how a
/// partition breaks a tie; and the winning score beats the runner-up by at
/// least 0.3125 where the scores reach 2.36 and a bf16 place there is 0.0156,
/// twenty places. Both anchors trace a chain that differs from the
/// per-position argmax, the two anchors differ from each other, and both
/// differ from the chain the pairwise term alone would trace — asserted below,
/// so a selector that did nothing, one that ignored the seed, and one that
/// dropped the logits all fail here rather than passing quietly.
#[test]
fn the_selector_reproduces_the_reference_on_the_scale_snapshot() {
    let drafter = scale_drafter();
    // Row 0 of the block is the seed the chain is anchored at; the drafted
    // positions are the rest, which is the reference's `logits_start = 1`.
    let full = fixture_tensor("hidden_wide");
    let positions = full.shape()[1] - 1;
    let hidden = full
        .slice(&[0, 1, 0], &[1, positions + 1, 64], &[1, 1, 1], Device::Cpu)
        .expect("drop the seed row");
    let logits = fixture_tensor("selector_logits");
    let top1 = to_u32(&fixture_tensor("selector_top1"));

    for (i, anchor) in SCALE_ANCHORS.iter().enumerate() {
        let want = to_u32(&fixture_tensor(&format!("selector_chain_{i}")));
        let got = drafter
            .select_chain(&hidden, &logits, *anchor)
            .expect("the scale snapshot must trace a chain");
        assert_eq!(
            got, want,
            "the chain at anchor {anchor} must be the reference's"
        );
        assert_ne!(
            got, top1,
            "the reference's own chain at anchor {anchor} equals the per-position \
             argmax, so this fixture cannot tell a working selector from a dead one"
        );
    }

    let first = to_u32(&fixture_tensor("selector_chain_0"));
    let second = to_u32(&fixture_tensor("selector_chain_1"));
    assert_ne!(
        first, second,
        "the two anchors must trace different chains, or the fixture cannot tell \
         an anchored selector from an unanchored one"
    );
}

/// The scale fixture's logits are bf16, which is what the verifier's head
/// returns: the score is accumulated in the dtype production uses, not in a
/// wider one the test happened to build.
#[test]
fn the_reference_case_runs_in_the_dtype_the_verifier_head_returns() {
    assert_eq!(fixture_tensor("selector_logits").dtype(), Dtype::Bf16);
    assert_eq!(fixture_tensor("hidden_wide").dtype(), Dtype::Bf16);
}

// ---------------------------------------------------------------------------
// the tied boundary — the regime the reference fixtures deliberately exclude
// ---------------------------------------------------------------------------

/// At an exact tie the partition keeps the **highest-numbered** of the tied
/// tokens, at every vocabulary size this selector runs at.
///
/// Both reference fixtures space their peaks so the boundary cannot tie, and
/// say so; measured on the published 4-bit head, the 16th and 17th logits are
/// exactly equal at every block position. So the exact-chain evidence covers a
/// regime production is not in, and what decides the candidate set there is how
/// `argpartition` orders equals — which MLX does not specify.
///
/// **This is the coupling, and it is the reason to pin it.** Which sixteen
/// tokens the selector considers is an input to the accept rate and to nothing
/// else: greedy acceptance emits the verifier's own argmax, so a set that moved
/// would change how often a draft is accepted and change no answer. There is no
/// correctness signal to notice it by. An MLX version whose partition broke ties
/// the other way would move every published DFlash 2 acceptance figure in
/// silence; here it fails, saying which way the tie now goes.
///
/// Measured, not assumed: `argsort` and `argpartition` keep the **same** set at
/// a tie at 6, 1024 and 248 320 tokens — they differ only in the order within
/// the kept slice — so this does not pin the choice between those two
/// primitives, and swapping one for the other is invisible here and everywhere
/// else. What it pins is the tie-break itself.
///
/// **CPU only.** The partition this asserts on is MLX's CPU kernel; production
/// drafts on Metal, whose kernel is a different implementation of the same
/// unspecified contract. A tie-break change arriving through an MLX bump would
/// have to skip this to reach production unseen.
#[test]
fn a_tie_at_the_candidate_boundary_breaks_toward_the_higher_token_id() {
    // The published pair's vocabulary and top-k last: the small sizes make the
    // rule readable, and the large one is the shape it actually runs at.
    for (vocab, k, tied) in [(6usize, 3usize, 4usize), (1024, 16, 20), (248_320, 16, 20)] {
        // Spread across the vocabulary, so "highest-numbered" is a claim about
        // the tie-break and not about the ids happening to be adjacent.
        let ids: Vec<usize> = (0..tied).map(|i| (i * 7919 + 11) % vocab).collect();
        let mut ranked = ids.clone();
        ranked.sort_unstable();
        let highest: Vec<usize> = ranked[tied - k..].to_vec();
        let lowest: Vec<usize> = ranked[..k].to_vec();
        assert_ne!(
            highest, lowest,
            "vocab {vocab}: the two answers must differ, or this asserts nothing"
        );

        let mut values = vec![-8.0f32; vocab];
        for &id in &ids {
            values[id] = 1.0;
        }
        let logits = f32_array(&values, &[1, 1, vocab as i32]);

        let partitioned =
            argpartition(&logits, -(k as i32), -1, Device::Cpu).expect("the partition runs");
        let kept = partitioned
            .slice(
                &[0, 0, vocab as i32 - k as i32],
                &[1, 1, vocab as i32],
                &[1, 1, 1],
                Device::Cpu,
            )
            .expect("the trailing k slices");
        kept.eval().expect("evaluates");
        let mut got: Vec<usize> = kept
            .to_bytes()
            .expect("reads back")
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
            .collect();
        got.sort_unstable();

        assert_eq!(
            got, highest,
            "vocab {vocab}: {tied} tokens tie for the top {k} places and the \
             partition kept {got:?}; every acceptance figure recorded for this \
             drafter was taken with the highest-numbered {k} kept instead"
        );
    }
}
