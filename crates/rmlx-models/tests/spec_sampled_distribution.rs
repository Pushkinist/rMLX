//! A sampled speculative arm must draw from the verifier's own distribution.
//!
//! The claim a sampled arm makes is distributional, and no amount of token
//! matching reaches it. Two arms that draw from different distributions produce
//! different tokens, which is also what two arms drawing from the *same*
//! distribution with different seeds produce, so an id comparison cannot tell
//! them apart. A gate has to read the distribution.
//!
//! # The statistic
//!
//! Every emitted token carries a surprise, `-ln p(token)`, under the
//! distribution the *plain* path would have drawn from at the same prefix. If
//! the arm draws from that distribution then the stream's total surprise has a
//! mean and a variance the distributions themselves fix exactly — the sum of the
//! per-position entropies, and the sum of the per-position variances of
//! `-ln p`. So the gate has a fully specified null hypothesis per position, and
//! the standardised total is a `z` against [`Z_CEILING`] with nothing measured
//! on a healthy engine first.
//!
//! Positions pool across prompts and lengths, because each brings its own mean
//! and variance. That is what lets the gate read a few hundred tokens from each
//! of a handful of answers instead of the thousands of one-token requests a
//! per-position frequency count would need — and it reads every position,
//! including the ones inside a round and the ones after a rollback, which a
//! first-token count never reaches.
//!
//! **The textbook test was tried first and does not have the power.** A
//! Kolmogorov-Smirnov test on the probability-integral transform reads the whole
//! distribution rather than one moment of it, and would be the better statistic
//! against a verifier that hesitates. This one does not: `E[p(argmax)]` is 0.92
//! on the measured pair, so a fully greedy arm moves the transform's deviation
//! by 0.05, and `sqrt(n)*D` reads 0.84 at 256 tokens against that test's own
//! 5e-6 ceiling of 2.30 — accepted. It would need about 1900 tokens per verdict
//! to refuse what the surprise test refuses at 256, because it prices a position
//! by the emitted token's rank while the surprise test prices it by that
//! position's own entropy, and a position the verifier is certain about then
//! contributes almost no variance rather than diluting the sample.
//!
//! **The surprise test is not the only oracle either, and measurement is again
//! why.** A stream drawn without the request's filters reads `z = 4.84` against
//! a ceiling of 5.0 — nine draws in ten still land inside the target's support
//! and are only mildly reweighted — so it is accepted. What sees it is that one
//! draw in ten is a token the request's own filters forbade, which is not a
//! distributional shift but an emission that could not have happened.
//! [`IMPOSSIBLE_FRACTION_CEILING`] reads that, and the two cover each other:
//! neither alone refuses both a greedy arm and an unfiltered one.
//!
//! # The oracle, and what it is not
//!
//! The reference distributions come from a plain forward over
//! `prompt ++ emitted`, not from the arm's own scoring. The arm reaches its
//! logits through a per-drafter verify seam, slices them per position, threads
//! one RNG across rounds and walks an acceptance decision over the result; none
//! of that is on the oracle's path, so a defect in any of it moves the statistic.
//! What the oracle *shares* with the arm is the trunk and the LM head, so this
//! gate cannot see a verifier that computes the wrong logits for everybody —
//! that is `spec_greedy_equivalence`'s job, and it holds the temperature-0 half
//! of the same question.
//!
//! # Shown able to fail
//!
//! The gate carries its own positive control, in the same run and against the
//! same shipped loop: a second arm runs the identical requests at temperature 0,
//! scored against the same oracle, and the run is red unless that one is
//! *refused*. On the measured pair, over five prose questions:
//!
//! | arm | n | z | impossible | verdict |
//! |---|---|---|---|---|
//! | sampled, temperature 0.7 / top-p 0.95 / top-k 20 | 1273 | +0.58 | 0 | accepted |
//! | the same loop at temperature 0 | 1262 | -8.70 | 0 | refused |
//!
//! The control is what makes the prompt set part of the gate rather than a
//! detail of it. A verifier that is nearly certain at every position passes under
//! every acceptance rule, correct or not, because there is no evidence in a
//! deterministic stream — and that is not hypothetical: the first prompt this
//! file ran omitted Gemma's `<bos>`, the verifier answered a three-token loop it
//! predicted with probability one, and both arms were accepted at once. The
//! control caught it. A gate that cannot refuse its own greedy arm now reports
//! that it had no power rather than reporting a pass.
//!
//! The statistic's power against the failures a GPU pair cannot reach is pinned
//! by the CPU tests below, which run everywhere and need no snapshot. Measured,
//! at 300 tokens each:
//!
//! | stream | z | impossible | refused by |
//! |---|---|---|---|
//! | drawn from the target | +0.48 | 0.0000 | — accepted |
//! | the target's argmax at every position | -25.05 | 0.0000 | surprise |
//! | drawn at temperature 1.0 for a request that asked 0.7 | +7.13 | 0.0000 | surprise |
//! | drawn without the request's `top_k` | +4.84 | 0.1000 | support |
//!
//! Server-free. `RMLX_KV_TEST_MODEL` / `RMLX_DRAFT_TEST_MODEL` override either
//! half of the pair; without them the pair resolves by slug from
//! `RMLX_O_MODELS_ROOT`.
//!
//!     cargo test -p rmlx-models --test spec_sampled_distribution -- --ignored --nocapture

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::ignore_without_reason,
    // The transform orders the vocabulary by probability and has to break ties
    // by id, so an exact equality on two f64 is the tie test, not a comparison
    // that should have had a tolerance.
    clippy::float_cmp
)]

use std::path::{Path, PathBuf};

mod common;

use rmlx_kv_quant::{KvCache, KvQuant};
use rmlx_mlx::{Array, Device};
use rmlx_models::arch;
use rmlx_models::sampler::{sampling_distribution, Pcg32, PenaltyConfig, SamplerConfig};
use rmlx_models::speculative::gemma4_assistant::{mtp_assistant_generate, Gemma4AssistantDrafter};

/// The published sampling parameters for this class of model, which is the
/// configuration the gate exists to make runnable through the fast path.
const TEMPERATURE: f32 = 0.7;
const TOP_P: f32 = 0.95;
const TOP_K: u32 = 20;

/// Token budget per answer. A ceiling, not a target: every question in the set
/// is answered under it and both arms stop on the verifier's own stop ids.
const N_TOKENS: usize = 256;

/// Fewest emitted tokens a stream may contribute before it joins the sample.
///
/// One arm answering in a handful of tokens while the other writes a full answer
/// is the loop's doing rather than the question's, and it would put most of the
/// pooled evidence on one side. Refused rather than pooled.
const MIN_TOKENS: usize = 64;

/// Standard deviations of total surprise beyond which the emitted stream is
/// refused.
///
/// The statistic is a sum of independent per-position terms whose mean and
/// variance the target itself fixes, so it is standard normal under the claim
/// whatever the stream length or the model — which is why one number covers
/// every pair and every prompt. 5.0 is a two-sided `p` of 6e-7, low enough that
/// a correct engine does not go red across many runs of many pairs.
///
/// **The other side is measured.** The greedy control over the same five
/// questions reads -8.70, and the four synthetic streams in this file read
/// +0.48, -25.05, +7.13 and +4.84 — so what the number has to separate is a
/// correct arm's own noise from a bias of several standard deviations, and the
/// case it is nearest is the one the support oracle refuses instead. See the
/// module documentation for both tables.
const Z_CEILING: f64 = 5.0;

/// Fraction of emitted tokens that may carry no target mass at all.
///
/// A correct arm reaches this only where its own logits and the oracle's
/// disagree about which token sits at a filter's cut — the verify pass scores a
/// whole block in one forward where the oracle reads one row, and that is a
/// different reduction order, worth a relative `1e-3` on a logit. It can move
/// the twentieth-ranked token across the `top_k` boundary and nothing else, so
/// the rate is bounded by how often the request's cut falls on a near-tie.
///
/// An arm that dropped the request's filters produces them at 10 per cent of
/// positions on the synthetic case below, which is where the two populations
/// sit relative to this number.
const IMPOSSIBLE_FRACTION_CEILING: f64 = 0.02;

/// Context the arm runs under. Above the prompt plus the budget.
const MAX_CTX: i32 = 8192;

/// Draft block. Small enough that rounds turn over often, so the sample covers
/// many first-position, mid-block and post-rollback draws rather than a few.
const BLOCK_SIZE: usize = 4;

/// The questions the sample is pooled over.
///
/// More than one because the evidence a single answer carries is bounded by how
/// much the verifier hesitates while writing it, and that is a property of the
/// question rather than of the engine. Prose, because a numbered plan is a
/// sequence the model is nearly certain about and contributes almost nothing.
///
/// A near-deterministic prompt set passes this gate under every acceptance rule,
/// correct or not, which is what the greedy control is asserted to catch.
const QUESTIONS: &[&str] = &[
    "Explain how virtual memory works on a modern operating system, covering page \
     tables, translation lookaside buffers and what happens on a page fault.",
    "Explain how TCP congestion control decides how fast to send, covering slow \
     start, congestion avoidance and what happens when a loss is detected.",
    "Explain how photosynthesis turns light into stored chemical energy, covering \
     the light-dependent reactions and the carbon-fixing ones.",
    "Explain what a database isolation level is and how the common ones differ, \
     covering the anomalies each one still permits.",
    "Explain how a hash map handles two keys that hash to the same bucket, covering \
     separate chaining and each of the open-addressing probing strategies.",
];

/// Appended to every question. The gate reads how much the verifier hesitates,
/// and a list or a heading is a shape it is certain about.
const PROSE_INSTRUCTION: &str =
    "Answer in prose, in a few short paragraphs, with no lists and no headings.";

/// The verifier of the pair this gate runs by slug.
const VERIFIER: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__gemma-4-e2b-it-mxfp8",
    archs: &["Gemma4ForConditionalGeneration"],
};

/// The drafter slug, and the variable that overrides it.
const DRAFTER_SLUG: &str = "mlx-community__gemma-4-E2B-it-assistant-bf16";
const DRAFT_MODEL_VAR: &str = "RMLX_DRAFT_TEST_MODEL";

// ── the statistic ────────────────────────────────────────────────────────────

/// The target's own mean and variance of `-ln p(t)` for `t` drawn from it: the
/// distribution's entropy, and how much a single draw's surprise varies around
/// it.
///
/// Both are exact functions of the target alone, which is what makes the null
/// hypothesis fully specified per position — no calibration run, no reference
/// arm, no threshold that has to be measured on a healthy engine first.
fn surprise_moments(probs: &[f32]) -> (f64, f64) {
    let mut mean = 0.0f64;
    let mut second = 0.0f64;
    for &p in probs {
        let p = f64::from(p);
        if p > 0.0 {
            let s = -p.ln();
            mean += p * s;
            second += p * s * s;
        }
    }
    (mean, (second - mean * mean).max(0.0))
}

/// The gate's verdict on one emitted stream.
struct Verdict {
    /// Standard deviations between the stream's total surprise under the target
    /// and the total surprise the target itself predicts.
    z: f64,
    tokens: usize,
    /// Emitted tokens the target gives no mass.
    impossible: usize,
    /// Mean of the target's own largest probability, over the stream. It bounds
    /// how much evidence any position can carry, so it is printed whatever the
    /// verdict.
    mean_top: f64,
}

impl Verdict {
    fn impossible_fraction(&self) -> f64 {
        self.impossible as f64 / self.tokens as f64
    }

    /// Whether the gate refuses this stream, and why.
    ///
    /// The two oracles cover each other and neither alone gives the recall. The
    /// surprise test reads a stream drawn from the wrong distribution that stays
    /// inside the right support — a greedy arm, a wrong temperature — and says
    /// nothing about one that leaves it. The support test reads exactly that: a
    /// token the request's own filters forbade is not a distributional shift, it
    /// is an emission that could not have happened, and a dropped `top_k`
    /// produces them at a rate no correct arm reaches while moving the surprise
    /// by well under its ceiling.
    fn refusal(&self) -> Option<String> {
        if self.z.abs() >= Z_CEILING {
            return Some(format!(
                "surprise: z={:.2} at or beyond the ceiling {Z_CEILING} — the stream is \
                 {} than the target says it should be",
                self.z,
                if self.z > 0.0 {
                    "less likely"
                } else {
                    "more likely"
                }
            ));
        }
        if self.impossible_fraction() > IMPOSSIBLE_FRACTION_CEILING {
            return Some(format!(
                "support: {} of {} emitted tokens carry no target mass, a fraction of \
                 {:.4} above the ceiling {IMPOSSIBLE_FRACTION_CEILING}",
                self.impossible,
                self.tokens,
                self.impossible_fraction()
            ));
        }
        None
    }

    fn report(&self, label: &str, test: &str) {
        println!(
            "[{test}] {label}: n={} z={:.2} impossible={} ({:.4}) mean_top={:.4} verdict={}",
            self.tokens,
            self.z,
            self.impossible,
            self.impossible_fraction(),
            self.mean_top,
            self.refusal().unwrap_or_else(|| "accepted".to_owned())
        );
    }
}

/// Score one or more emitted streams against the target distributions at their
/// own prefixes.
///
/// The positions pool: each one contributes its own null mean and variance, so
/// streams from different prompts and different lengths add up into one verdict
/// without any of them needing a threshold of its own.
fn score(streams: &[(Vec<u32>, Vec<Vec<f32>>)]) -> Verdict {
    let mut surprise = 0.0f64;
    let mut null_mean = 0.0f64;
    let mut null_var = 0.0f64;
    let mut impossible = 0usize;
    let mut top_sum = 0.0f64;
    let mut tokens = 0usize;
    for (emitted, targets) in streams {
        assert_eq!(
            emitted.len(),
            targets.len(),
            "one target distribution per emitted token"
        );
        for (i, &token) in emitted.iter().enumerate() {
            let p = &targets[i];
            tokens += 1;
            top_sum += f64::from(p.iter().copied().fold(0.0f32, f32::max));
            let (mean, var) = surprise_moments(p);
            null_mean += mean;
            null_var += var;
            match p.get(token as usize) {
                Some(&m) if m > 0.0 => surprise += -f64::from(m).ln(),
                // An emission the target forbids has infinite surprise, which no
                // z has room for. The support oracle is what reads these; the
                // surprise sum takes the target's own worst finite outcome so a
                // few of them cannot swamp it either way.
                _ => {
                    impossible += 1;
                    let worst = p
                        .iter()
                        .filter(|&&m| m > 0.0)
                        .fold(1.0f32, |acc, &m| acc.min(m));
                    surprise += -f64::from(worst).ln();
                }
            }
        }
    }
    assert!(tokens > 0, "the statistic needs a sample");
    Verdict {
        z: (surprise - null_mean) / null_var.sqrt(),
        tokens,
        impossible,
        mean_top: top_sum / tokens as f64,
    }
}

// ── the CPU power tests ──────────────────────────────────────────────────────
//
// These need no snapshot and no device, so the statistic's power is pinned on
// every machine that runs `cargo test`, not only on one holding the models.

/// A fixed target with real spread, repeated: enough structure that a wrong
/// rule shows, simple enough to reason about.
fn synthetic_target(vocab: usize) -> Vec<f32> {
    let mut p: Vec<f32> = (0..vocab).map(|i| 1.0 / (i as f32 + 1.5)).collect();
    let total: f32 = p.iter().sum();
    for x in &mut p {
        *x /= total;
    }
    p
}

/// Draw an index from `probs` by inverse CDF.
fn draw(probs: &[f32], u: f32) -> u32 {
    let mut cum = 0.0f32;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if cum > u {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Sample size for the CPU cases. Large enough that a correct stream's own
/// noise is nowhere near the ceiling, small enough to be instant.
const SYNTHETIC_N: usize = 300;

#[test]
fn a_stream_drawn_from_the_target_is_accepted() {
    let p = synthetic_target(64);
    let targets: Vec<Vec<f32>> = (0..SYNTHETIC_N).map(|_| p.clone()).collect();
    let mut rng = Pcg32::new(0x5EED);
    let emitted: Vec<u32> = (0..SYNTHETIC_N).map(|_| draw(&p, rng.next_f32())).collect();
    let v = score(&[(emitted, targets)]);
    v.report("drawn from the target", "cpu");
    assert!(
        v.refusal().is_none(),
        "a stream drawn from the target must be accepted, and was refused on {}",
        v.refusal().unwrap_or_default()
    );
}

#[test]
fn a_greedy_stream_is_refused() {
    let p = synthetic_target(64);
    let argmax = p
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).expect("finite"))
        .expect("non-empty")
        .0 as u32;
    let targets: Vec<Vec<f32>> = (0..SYNTHETIC_N).map(|_| p.clone()).collect();
    let emitted: Vec<u32> = vec![argmax; SYNTHETIC_N];
    let v = score(&[(emitted, targets)]);
    v.report("greedy", "cpu");
    let reason = v.refusal().unwrap_or_else(|| {
        panic!(
            "an arm that emits the target's argmax at every position must be refused: \
             z={:.2} mean_top={:.4}",
            v.z, v.mean_top
        )
    });
    assert!(
        reason.starts_with("surprise:"),
        "a greedy stream stays inside the target's support, so the shape oracle is the \
         one that must fire; got {reason}"
    );
}

#[test]
fn a_stream_drawn_at_the_wrong_temperature_is_refused() {
    // The arm asked for 0.7 and the loop divided by 1.0: a flatter distribution
    // over the same logits. Nothing about the emitted ids says so.
    let logits: Vec<f32> = (0..64).map(|i| -(i as f32) * 0.35).collect();
    let asked = softmax(&logits, 1.0 / 0.7);
    let served = softmax(&logits, 1.0);
    let targets: Vec<Vec<f32>> = (0..SYNTHETIC_N).map(|_| asked.clone()).collect();
    let mut rng = Pcg32::new(0x5EED);
    let emitted: Vec<u32> = (0..SYNTHETIC_N)
        .map(|_| draw(&served, rng.next_f32()))
        .collect();
    let v = score(&[(emitted, targets)]);
    v.report("wrong temperature", "cpu");
    let reason = v.refusal().unwrap_or_else(|| {
        panic!(
            "a stream drawn at a temperature the request did not ask for must be \
             refused: z={:.2}",
            v.z
        )
    });
    assert!(
        reason.starts_with("surprise:"),
        "a temperature change moves the shape and not the support, so the shape oracle \
         is the one that must fire; got {reason}"
    );
}

#[test]
fn a_stream_drawn_without_the_requests_filters_is_refused() {
    // The request asked for top-k 20 and the loop drew from the unfiltered
    // distribution. This is the case that needs the second oracle: nine draws in
    // ten still land inside the target's support and are only mildly
    // reweighted, so the shape statistic reads under its own ceiling.
    let logits: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.08).collect();
    let unfiltered = softmax(&logits, 1.0 / 0.7);
    let mut filtered = unfiltered.clone();
    let mut order: Vec<usize> = (0..filtered.len()).collect();
    order.sort_by(|&a, &b| filtered[b].partial_cmp(&filtered[a]).expect("finite"));
    for &i in order.iter().skip(TOP_K as usize) {
        filtered[i] = 0.0;
    }
    let total: f32 = filtered.iter().sum();
    for x in &mut filtered {
        *x /= total;
    }
    let targets: Vec<Vec<f32>> = (0..SYNTHETIC_N).map(|_| filtered.clone()).collect();
    let mut rng = Pcg32::new(0x5EED);
    let emitted: Vec<u32> = (0..SYNTHETIC_N)
        .map(|_| draw(&unfiltered, rng.next_f32()))
        .collect();
    let v = score(&[(emitted, targets)]);
    v.report("filters dropped", "cpu");
    let reason = v.refusal().unwrap_or_else(|| {
        panic!(
            "a stream drawn without the request's filters must be refused: \
             z={:.2} impossible={}",
            v.z, v.impossible
        )
    });
    assert!(
        reason.starts_with("support:"),
        "this case is the reason the support oracle exists — the shape statistic reads \
         under its ceiling here — so a run where the shape oracle fires instead means \
         the fixture stopped exercising the boundary it was built for; got {reason}"
    );
}

fn softmax(logits: &[f32], inv_temp: f32) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut p: Vec<f32> = logits
        .iter()
        .map(|&l| ((l - max) * inv_temp).exp())
        .collect();
    let total: f32 = p.iter().sum();
    for x in &mut p {
        *x /= total;
    }
    p
}

// ── the pair ─────────────────────────────────────────────────────────────────

/// Resolve the drafter: an operator's override, else the slug under the models
/// root. Mirrors the sibling equivalence gate so a machine that runs one runs
/// the other.
fn resolve_drafter() -> common::Gate {
    if let Some(named) = std::env::var(DRAFT_MODEL_VAR)
        .ok()
        .filter(|v| !v.is_empty())
    {
        let path = PathBuf::from(&named);
        return if path.join("config.json").is_file() {
            common::Gate::Run { path, note: None }
        } else {
            common::Gate::Fail(format!("{DRAFT_MODEL_VAR}={named} holds no config.json"))
        };
    }
    let Some(root) = std::env::var(common::MODELS_ROOT_VAR)
        .ok()
        .filter(|r| !r.is_empty())
    else {
        return common::Gate::Skip(format!(
            "no drafter configured — set {} (holding {DRAFTER_SLUG}) or {DRAFT_MODEL_VAR}",
            common::MODELS_ROOT_VAR
        ));
    };
    let path = Path::new(&root).join(DRAFTER_SLUG);
    if path.join("config.json").is_file() {
        common::Gate::Run { path, note: None }
    } else {
        common::Gate::Skip(format!(
            "{}={root} does not hold a runnable {DRAFTER_SLUG}",
            common::MODELS_ROOT_VAR
        ))
    }
}

fn sampler(temperature: f32, seed: u64) -> SamplerConfig {
    SamplerConfig {
        temperature,
        top_p: TOP_P,
        top_k: TOP_K,
        min_p: 0.0,
        seed: Some(seed),
        top_logprobs_k: 0,
    }
}

/// The target distribution at each of `emitted`'s own prefixes, from a plain
/// forward the arm's verify seam is not on.
///
/// The forward runs over `prompt ++ emitted[..len-1]` and reads its last `len`
/// logit rows: row `j` is the position whose continuation is `emitted[j]`.
fn targets_for(
    verifier: &arch::Architecture,
    prompt: &[u32],
    emitted: &[u32],
    cfg: &SamplerConfig,
    device: Device,
) -> Vec<Vec<f32>> {
    let mut ids: Vec<u32> = prompt.to_vec();
    ids.extend_from_slice(&emitted[..emitted.len() - 1]);
    let mut caches: Vec<KvCache> = (0..verifier.num_hidden_layers())
        .map(|i| {
            KvCache::with_quant_max_seq_window(
                KvQuant::None,
                MAX_CTX,
                verifier.layer_sliding_window(i),
            )
            .with_max_seq_ceiling(MAX_CTX)
            .with_layer_idx(i)
            .with_shares_kv(verifier.shares_kv_across_layers())
        })
        .collect();
    let logits = verifier
        .forward_seq_last_k_with_cache(&ids, emitted.len(), &mut caches, None, device)
        .expect("the oracle forward must reach the verifier's logits");
    let vocab = *logits.shape().last().expect("a vocabulary axis");
    let penalties = PenaltyConfig::default();
    (0..emitted.len())
        .map(|i| {
            let i = i as i32;
            let row: Array = logits
                .slice(&[0, i, 0], &[1, i + 1, vocab], &[1, 1, 1], device)
                .and_then(|r| r.reshape(&[1, vocab], device))
                .expect("one logit row per emitted position");
            sampling_distribution(&row, cfg, None, &penalties, &[])
                .expect("the plain path's own distribution builder")
        })
        .collect()
}

/// Run the assistant arm once and return what it emitted.
fn run_arm(
    verifier: &arch::Architecture,
    drafter: &Gemma4AssistantDrafter,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &[u32],
    eos: &[u32],
    cfg: &SamplerConfig,
    device: Device,
) -> Vec<u32> {
    let mut emitted: Vec<u32> = Vec::new();
    {
        let mut step = |s: &rmlx_models::ProbeStep| {
            emitted.push(s.token_id);
            None
        };
        mtp_assistant_generate(
            verifier,
            drafter,
            tokenizer,
            prompt,
            N_TOKENS,
            BLOCK_SIZE,
            Some(KvQuant::None),
            Some(MAX_CTX),
            eos,
            &mut step,
            cfg,
            device,
        )
        .expect("the assistant arm must run");
    }
    emitted
}

#[ignore = "loads two snapshots and drives Metal"]
#[test]
fn a_sampled_sidecar_arm_draws_from_the_verifiers_distribution() {
    const TEST: &str = "a_sampled_sidecar_arm_draws_from_the_verifiers_distribution";
    let device = Device::Gpu;

    let Some(draft_path) = common::apply(resolve_drafter(), TEST) else {
        return;
    };
    let Some(model_path) = common::model_for(&VERIFIER, TEST) else {
        return;
    };

    let verifier = arch::load_model(&model_path, device, &arch::LoadOpts::default())
        .expect("load the verifier");
    let hidden = verifier.hidden_size();
    let drafter =
        Gemma4AssistantDrafter::load(&draft_path, hidden, device).expect("load the drafter");
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let eos = eos_ids(&model_path);
    assert!(
        !eos.is_empty(),
        "the verifier config must name its stop ids, or the arm runs past its answer \
         into filler and the sample is over that"
    );

    // The request the gate is about, at the published sampling parameters, and
    // the same request at temperature 0. Both run the shipped loop; the second
    // is what a silently greedy sidecar used to serve for the first.
    let sampled_cfg = sampler(TEMPERATURE, 0x51D3);
    let greedy_cfg = sampler(0.0, 0x51D3);
    let mut sampled_streams = Vec::new();
    let mut greedy_streams = Vec::new();

    for question in QUESTIONS {
        let prompt = prompt_ids(&tokenizer, question);
        for (cfg, streams, label) in [
            (&sampled_cfg, &mut sampled_streams, "sampled"),
            (&greedy_cfg, &mut greedy_streams, "control"),
        ] {
            let emitted = run_arm(&verifier, &drafter, &tokenizer, &prompt, &eos, cfg, device);
            assert!(
                emitted.len() >= MIN_TOKENS,
                "the {label} arm emitted {} tokens on {question:?} and a stream joins the \
                 sample from {MIN_TOKENS}",
                emitted.len()
            );
            // Both arms are scored against the distributions the *sampled*
            // request asked for: a greedy arm serving that request is exactly
            // what the gate must refuse.
            let targets = targets_for(&verifier, &prompt, &emitted, &sampled_cfg, device);
            streams.push((emitted, targets));
        }
    }

    let sampled_v = score(&sampled_streams);
    sampled_v.report("sampled arm", TEST);
    let greedy_v = score(&greedy_streams);
    greedy_v.report("greedy control", TEST);

    // The positive control. A prompt set on which the control is not refused is
    // one the gate has no power over, and that fails here rather than passing
    // quietly.
    let control = greedy_v.refusal().unwrap_or_else(|| {
        panic!(
            "the gate did not refuse its own greedy control, so it has no power on this \
             prompt set and its verdict on the sampled arm means nothing: control \
             z={:.2} against ceiling {Z_CEILING}, mean_top={:.4} over {} tokens",
            greedy_v.z, greedy_v.mean_top, greedy_v.tokens
        )
    });
    if let Some(why) = sampled_v.refusal() {
        panic!(
            "the sampled arm does not draw from the verifier's distribution — {why}. \
             The greedy control over the same prompts was refused on {control}, so the \
             gate had power here."
        );
    }
}

/// The chat-formatted prompt, with the leading `<bos>` this family needs.
///
/// A Gemma verifier served without `<bos>` and its own turn markers answers a
/// three-token loop that every position predicts with probability one — on which
/// this gate's transform is uniform whatever the acceptance rule, so it passes
/// and asserts nothing. That is not hypothetical: it is what this file measured
/// before the marker went in, and the greedy control is what caught it.
fn prompt_ids(tk: &tokenizers::Tokenizer, question: &str) -> Vec<u32> {
    let mut ids: Vec<u32> = Vec::new();
    if let Some(bos) = tk.token_to_id("<bos>") {
        ids.push(bos);
    }
    let text = format!(
        "<start_of_turn>user\n{question} {PROSE_INSTRUCTION}<end_of_turn>\n\
         <start_of_turn>model\n"
    );
    ids.extend(
        tk.encode(text.as_str(), true)
            .expect("encode the chat-formatted prompt")
            .get_ids()
            .iter()
            .copied(),
    );
    ids
}

/// Stop ids from the verifier's `config.json`, in the shapes it writes them.
fn eos_ids(model_path: &Path) -> Vec<u32> {
    let Ok(text) = std::fs::read_to_string(model_path.join("config.json")) else {
        return Vec::new();
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&text) else {
        return Vec::new();
    };
    let node = cfg
        .get("eos_token_id")
        .or_else(|| cfg.get("text_config").and_then(|t| t.get("eos_token_id")));
    match node {
        Some(serde_json::Value::Number(n)) => n.as_u64().map(|v| v as u32).into_iter().collect(),
        Some(serde_json::Value::Array(items)) => items
            .iter()
            .filter_map(serde_json::Value::as_u64)
            .map(|v| v as u32)
            .collect(),
        _ => Vec::new(),
    }
}
