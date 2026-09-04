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
//! that is a different reduction order. On the Gemma4-e2b assistant pair — a
//! full-attention verifier whose rollback is an exact KV truncation, the most
//! favourable case there is — the two arms share 66 leading tokens and then
//! differ by one word ("Separate Chaining (Most Common Method)" against
//! "(The Most Common Method)"), after which both continue the same explanation.
//! A gate demanding identity would fail on that.
//!
//! **The oracle is therefore length-scaled, not positional**, and it is read
//! twice: over the whole answer, and over each tail window, because those
//! separate two failures the whole-stream ratio blends. One benign flip early
//! and a divergence that begins late and never comes back can both land near
//! 0.8 overall; only the second collapses a window. Measured on that pair at
//! [`N_TOKENS`] tokens:
//!
//! | rollback | whole-stream LCS | weakest tail window | first divergence |
//! |---|---|---|---|
//! | as shipped | 0.8438 | 0.6484 at token 384 | 66 |
//! | one rejected draft key left in the cache | 0.3070 | 0.2093 at token 256 | 5 |
//!
//! [`MIN_LCS_RATIO`] and [`MIN_TAIL_LCS_RATIO`] sit in those gaps. The shared
//! prefix is reported and never asserted on: it collapses the moment a near-tie
//! flips early, which is a property of the prompt and the arithmetic rather
//! than of the round loop.
//!
//! **Length is bounded from both sides.** Two configurations that differ can
//! agree for the first few dozen tokens of an easy answer, so a short run is a
//! gate that cannot fail. A long one is a different trap: greedy decoding
//! compounds, so one benign flip at token 66 makes the two arms write different
//! sections by token 800, and the same pair that reads 0.914 at 256 tokens
//! reads 0.6846 at its natural stop near 800 — below any floor that still
//! refuses a broken rollback. [`N_TOKENS`] is the horizon where the two regimes
//! are separable; it is not a length past which nothing goes wrong.
//!
//! **The control.** A token stream that has collapsed into a repetition loop
//! matches anything, so "the two agree" would pass on garbage. Every run checks
//! that neither arm repeats at any period up to [`MAX_CYCLE_PERIOD`] across more
//! than [`MAX_CYCLE_FRACTION`] of its tokens — a whole-stream measure, because a
//! collapse that begins at token 20 and an `A B A B` cycle are both degeneracies
//! that a leading-run measure scores at zero. Real prose from both arms reads
//! about 0.02 on it.
//!
//! **Known gap: this gate runs at a short context and there is a defect past
//! it.** Driven from the 4k document in `prompts/longctx_4k.json` instead of
//! this prompt, the speculative arm collapses into a period-8 repetition loop
//! (`x86 is:66 is x86 is:66 is ...`, cycle 0.881 across 512 tokens) while plain
//! greedy writes a clean summary and stops at 200 — whole-stream LCS 0.11,
//! first divergence at token 3. That reproduces identically on `main`
//! (8ccc0593), so it is not this branch's, and it is why the horizon here is
//! not the whole story. Moving this gate onto the long prompt is the right
//! thing to do once that defect is fixed.
//!
//! Server-free. Both snapshots resolve by slug from `RMLX_O_MODELS_ROOT`, so
//! `make gpu-test` runs this on a machine holding them and
//! `scripts/run_gpu_tests.sh` reports a machine without them as INCOMPLETE.
//! `RMLX_KV_TEST_MODEL` and `RMLX_DRAFT_TEST_MODEL` override either half:
//!
//! cargo test -p rmlx-models --test spec_greedy_equivalence -- --ignored --nocapture

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::ignore_without_reason
)]

use std::path::{Path, PathBuf};

mod common;

use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::speculative::gemma4_assistant::{
    mtp_assistant_generate_greedy, Gemma4AssistantDrafter,
};

/// Token budget per arm. A ceiling, not a target: both arms stop on the
/// verifier's own stop ids and this answer is shorter than the budget.
///
/// The budget is what makes the horizon long enough to matter — hundreds of
/// rounds and a kilotoken of context, against the 48 tokens the sibling
/// alignment suites compare. Running *past* the answer is the opposite problem:
/// with no stop ids both arms emit end-of-turn forever, and a comparison over
/// that filler measures nothing about the round loop.
const N_TOKENS: usize = 512;

/// The shorter arm must be at least this long, or there is not enough answer to
/// compare. A digest over 32 generated tokens has already given two
/// configurations of this engine the same value where 200 separated them.
const MIN_ANSWER_TOKENS: usize = 256;

/// The shorter arm over the longer. `lcs_ratio` divides by the shorter stream,
/// so an arm that stopped at a third of the other's length and matched its
/// prefix would otherwise score 1.0.
const MIN_LENGTH_RATIO: f64 = 0.60;

/// Draft block. Small enough that a rollback runs every few tokens, which is
/// the code path the oracle protects.
const BLOCK_SIZE: usize = 4;

/// Context both arms run under. Above the 4k prompt plus the budget, and the
/// same on both sides — a different cap on either would make this a measurement
/// of the cap.
const MAX_CTX: i32 = 8192;

/// How much of one answer both arms must have produced.
///
/// Between the two measured regimes in the module docs: 1.3x below the shipped
/// rollback's reading and 2.7x above a rollback that leaves one rejected key in
/// the cache.
const MIN_LCS_RATIO: f64 = 0.70;

/// The most of an arm that may repeat at a short period before the stream counts
/// as degenerate. A loop repeats at its period almost everywhere; the measured
/// readings for real prose from both arms are in the module docs.
const MAX_CYCLE_FRACTION: f64 = 0.50;

/// Longest cycle the control looks for. A loop longer than this is a stream
/// that is still saying different things.
const MAX_CYCLE_PERIOD: usize = 8;

/// The stream is cut at each `1/TAIL_WINDOWS` boundary and the suffixes
/// compared, so a divergence that begins late has a window it dominates.
const TAIL_WINDOWS: usize = 4;

/// How much of one answer both arms must share over every tail window.
///
/// Lower than the whole-stream floor: a suffix starts wherever the cut falls,
/// so it can open mid-divergence where the whole stream does not. Between the
/// two measured regimes in the module docs — 1.6x below the shipped rollback's
/// weakest window and 1.9x above the broken one's.
const MIN_TAIL_LCS_RATIO: f64 = 0.40;

// ── Oracle ───────────────────────────────────────────────────────────────────

/// How many leading tokens two streams share. Reported, never asserted on — see
/// the module docs.
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

/// The strongest short cycle in `tokens`: its period, and the fraction of
/// positions that repeat at that period.
///
/// The control. A stream stuck in a loop matches itself at the loop's period,
/// and two arms in the same loop agree perfectly — so the equivalence oracle
/// says nothing exactly when this is high. It is a whole-stream measure and
/// covers every period up to [`MAX_CYCLE_PERIOD`], because a collapse that
/// starts at token 20 and a two-token `A B A B` cycle are both degeneracies a
/// leading-run measure scores at zero.
fn strongest_cycle(tokens: &[u32]) -> (usize, f64) {
    let mut worst = (1usize, 0.0f64);
    for period in 1..=MAX_CYCLE_PERIOD.min(tokens.len().saturating_sub(1)) {
        let matches = tokens[period..]
            .iter()
            .zip(tokens)
            .filter(|(a, b)| a == b)
            .count();
        let fraction = matches as f64 / (tokens.len() - period) as f64;
        if fraction > worst.1 {
            worst = (period, fraction);
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

/// Judge one pair of streams. Returns the failure text, or `None` when the
/// oracle and both controls held.
fn judge(spec: &[u32], plain: &[u32]) -> Option<String> {
    let shorter = spec.len().min(plain.len());
    let longer = spec.len().max(plain.len());
    if shorter < MIN_ANSWER_TOKENS {
        return Some(format!(
            "an arm stopped early — spec={} plain={}, under {MIN_ANSWER_TOKENS}; a short \
             run is a comparison with no power, not a pass",
            spec.len(),
            plain.len()
        ));
    }
    let length_ratio = shorter as f64 / longer as f64;
    if length_ratio < MIN_LENGTH_RATIO {
        return Some(format!(
            "the arms answered at {} and {} tokens, a ratio of {length_ratio:.4} (floor \
             {MIN_LENGTH_RATIO}) — one stopped well before the other, and the subsequence \
             ratio is taken over the shorter, where a truncated arm reads high on the \
             other's prefix",
            spec.len(),
            plain.len()
        ));
    }

    for (name, arm) in [("plain", plain), ("speculative", spec)] {
        let (period, fraction) = strongest_cycle(arm);
        if fraction > MAX_CYCLE_FRACTION {
            return Some(format!(
                "the {name} arm repeats at period {period} across {fraction:.4} of its \
                 {shorter} tokens (ceiling {MAX_CYCLE_FRACTION}) — it has collapsed into a \
                 repetition loop, so agreement between the arms would say nothing"
            ));
        }
    }

    let ratio = lcs_ratio(spec, plain);
    if ratio < MIN_LCS_RATIO {
        return Some(format!(
            "the arms share {ratio:.4} of one answer (floor {MIN_LCS_RATIO}), sharing \
             {} leading tokens — the round loop did not reproduce what the verifier \
             decodes on its own",
            common_prefix_len(spec, plain)
        ));
    }

    let (start, tail) = weakest_tail(spec, plain);
    if tail < MIN_TAIL_LCS_RATIO {
        return Some(format!(
            "the arms share {ratio:.4} of the whole answer but only {tail:.4} of it from \
             token {start} on (floor {MIN_TAIL_LCS_RATIO}), first differing at token {} — \
             a divergence that begins late and does not come back, which the whole-stream \
             ratio blends with one benign flip",
            common_prefix_len(spec, plain)
        ));
    }
    None
}

// ── Oracle tests (no model, no GPU) ──────────────────────────────────────────

/// One near-tie flip that both arms write around is the shipped behaviour and
/// must pass; a stream that goes its own way must not.
#[test]
fn a_single_flip_passes_and_a_divergent_stream_does_not() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();

    let mut one_flip = plain.clone();
    one_flip[66] = 9999;
    assert!(judge(&one_flip, &plain).is_none(), "one flip must pass");

    let diverged: Vec<u32> = plain
        .iter()
        .enumerate()
        .map(|(i, t)| if i > 8 { 50_000 + i as u32 } else { *t })
        .collect();
    let failure = judge(&diverged, &plain).expect("a divergent stream must fail");
    assert!(failure.contains("did not reproduce"), "{failure}");
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
    ] {
        let failure = judge(&stream, &stream).unwrap_or_else(|| panic!("{shape} was not refused"));
        assert!(failure.contains("repetition loop"), "{shape}: {failure}");
    }
}

/// A speculative arm that degenerated while the plain one did not is caught by
/// the control that reads the speculative arm, not only the plain one.
#[test]
fn a_degenerate_speculative_arm_is_caught() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let looping = vec![7u32; N_TOKENS];
    let failure = judge(&looping, &plain).expect("must refuse");
    assert!(failure.contains("speculative arm"), "{failure}");
}

/// Real prose is not a repetition loop. Without this the control could be made
/// to fire on everything and the suite above would still be green.
#[test]
fn a_healthy_stream_clears_the_repetition_control() {
    let healthy: Vec<u32> = (0..N_TOKENS as u32).map(|t| t % 97).collect();
    let (period, fraction) = strongest_cycle(&healthy);
    assert!(
        fraction <= MAX_CYCLE_FRACTION,
        "a stream with no short cycle scored {fraction} at period {period}"
    );
    assert!(judge(&healthy, &healthy).is_none());
}

/// A run that stopped short is refused rather than passed on its short prefix.
#[test]
fn a_truncated_run_is_refused_rather_than_judged() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let failure = judge(&plain[..16], &plain).expect("must refuse");
    assert!(failure.contains("stopped early"), "{failure}");
}

/// The class the whole-stream ratio blends: a divergence that begins in the
/// last quarter and never comes back scores high overall and collapses one tail
/// window. This is the shape of a regression that fires past a cache-size
/// threshold, and it is why the tail windows exist.
#[test]
fn a_late_onset_divergence_passes_the_whole_stream_ratio_and_fails_the_tail() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let onset = N_TOKENS * 4 / 5;
    let late: Vec<u32> = plain
        .iter()
        .enumerate()
        .map(|(i, t)| if i >= onset { 50_000 + i as u32 } else { *t })
        .collect();

    let whole = lcs_ratio(&late, &plain);
    assert!(
        whole >= MIN_LCS_RATIO,
        "the whole-stream ratio {whole} already refuses this, so the tail windows \
         are not what is being tested"
    );
    let failure = judge(&late, &plain).expect("the tail windows must refuse it");
    assert!(failure.contains("begins late"), "{failure}");
}

/// The two measured regimes, reconstructed as token streams and put through the
/// gate itself — so the test also fails when `lcs_ratio`'s denominator or
/// `judge`'s comparison changes, not only when the constant moves.
#[test]
fn the_gate_admits_the_shipped_regime_and_refuses_the_broken_one() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).map(|t| t % 97).collect();

    // As shipped: one benign flip, then the same answer. Replacing 1 token in
    // 12 leaves a subsequence ratio near the measured 0.91.
    let shipped: Vec<u32> = plain
        .iter()
        .enumerate()
        .map(|(i, t)| if i % 12 == 5 { 60_000 + i as u32 } else { *t })
        .collect();
    assert!(
        lcs_ratio(&shipped, &plain) > MIN_LCS_RATIO,
        "reconstructed shipped regime reads {}",
        lcs_ratio(&shipped, &plain)
    );
    assert!(judge(&shipped, &plain).is_none(), "shipped must pass");

    // The broken rollback: diverged at token 5 and never recovered. Three
    // tokens in four replaced leaves roughly the measured 0.26.
    let broken: Vec<u32> = plain
        .iter()
        .enumerate()
        .map(|(i, t)| {
            if i >= 5 && i % 4 != 0 {
                60_000 + i as u32
            } else {
                *t
            }
        })
        .collect();
    let failure = judge(&broken, &plain).expect("broken must fail");
    assert!(failure.contains("did not reproduce"), "{failure}");
}

// ── The gate ─────────────────────────────────────────────────────────────────

/// Snapshot slug of the verifier this gate is calibrated on.
const VERIFIER_SLUG: &str = "mlx-community__gemma-4-e2b-it-mxfp8";
/// Snapshot slug of its assistant drafter.
const DRAFTER_SLUG: &str = "mlx-community__gemma-4-E2B-it-assistant-bf16";
/// Draft-model override, the variable the sibling alignment suites take.
const DRAFT_MODEL_VAR: &str = "RMLX_DRAFT_TEST_MODEL";

/// Resolve one half of the pair: an operator's override if they named one,
/// otherwise the slug under `RMLX_O_MODELS_ROOT`.
///
/// Resolving by slug is what puts this gate inside `make gpu-test` on a machine
/// holding the snapshots: it joins the population `run_gpu_tests.sh` already
/// reports as INCOMPLETE when the models root is unset, instead of being a
/// third variable nobody exports and a green run that asserted nothing.
///
/// A path the operator named that is not a snapshot fails; a models root that
/// simply does not hold the slug skips. That split is the harness's
/// (`tests/common/mod.rs`), not a second copy of its rules.
fn resolve(var: &str, slug: &str) -> common::Gate {
    let root = std::env::var(common::MODELS_ROOT_VAR).ok();
    let Some(named) = std::env::var(var).ok().filter(|v| !v.is_empty()) else {
        return match common::slug_snapshot(root.as_deref(), slug) {
            common::Snapshot::Found { path, .. } => common::Gate::Run { path, note: None },
            // The harness names its own override variable in that message; this
            // half of the pair is overridden by a different one.
            common::Snapshot::Absent(why) => {
                common::Gate::Skip(why.replace(common::SINGLE_MODEL_VAR, var))
            }
            common::Snapshot::Misconfigured(why) => common::Gate::Fail(why),
        };
    };
    let named_path = PathBuf::from(&named);
    let (parent, leaf) = match (named_path.parent(), named_path.file_name()) {
        (Some(parent), Some(leaf)) => (parent.to_owned(), leaf.to_string_lossy().into_owned()),
        _ => return common::Gate::Fail(format!("{var}={named} is not a directory path")),
    };
    match common::slug_snapshot(parent.to_str(), &leaf) {
        common::Snapshot::Found { path, .. } => common::Gate::Run { path, note: None },
        // The operator named this path, so a typo or a moved snapshot breaks
        // the run rather than skipping it.
        common::Snapshot::Absent(why) | common::Snapshot::Misconfigured(why) => {
            common::Gate::Fail(format!("{var}={named}: {why}"))
        }
    }
}

/// Whether `draft_path` is the dedicated Gemma4 assistant drafter snapshot.
///
/// Both fields are read because mlx-community snapshots set the family on one
/// or the other depending on the export tool — the same two fields the serve
/// layer routes `--draft-kind mtp` on.
fn is_gemma4_assistant(draft_path: &Path) -> bool {
    let Ok(raw) = std::fs::read_to_string(draft_path.join("config.json")) else {
        return false;
    };
    let Ok(cfg) = serde_json::from_str::<serde_json::Value>(&raw) else {
        return false;
    };
    let model_type = cfg["model_type"].as_str().unwrap_or_default();
    let arch_name = cfg["architectures"][0].as_str().unwrap_or_default();
    model_type == "gemma4_assistant" || arch_name.contains("Gemma4Assistant")
}

/// Chat-formatted prompt, with the leading `<bos>`.
///
/// Gemma without its turn markers and without `<bos>` ends the turn on its very
/// first token, and a gate that only ever runs on a degenerate stream is a gate
/// that never runs.
fn build_prompt(tk: &tokenizers::Tokenizer) -> Vec<u32> {
    let text = "<start_of_turn>user\nExplain how a hash map resolves collisions, \
                step by step.<end_of_turn>\n<start_of_turn>model\n";
    let mut ids: Vec<u32> = Vec::new();
    if let Some(bos) = tk.token_to_id("<bos>") {
        ids.push(bos);
    }
    ids.extend(
        tk.encode(text, true)
            .expect("encode")
            .get_ids()
            .iter()
            .copied(),
    );
    ids
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

/// Plain greedy decoding on the verifier alone, at temperature 0.
fn plain_greedy(
    verifier: &arch::Architecture,
    tk: &tokenizers::Tokenizer,
    prompt: &[u32],
    eos: &[u32],
    device: Device,
) -> Vec<u32> {
    let sampler_cfg = rmlx_models::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let mut rng = rmlx_models::sampler::Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = rmlx_models::sampler::PenaltyConfig::default();
    let mut history: Vec<u32> = Vec::new();
    let mut ids: Vec<u32> = Vec::new();
    {
        let mut step = |s: &rmlx_models::ProbeStep| {
            ids.push(s.token_id);
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
    ids
}

/// The gate: the speculative arm reproduces what the verifier decodes alone.
///
/// Both arms run the same verifier, the same prompt, `--kv-quant none` and
/// temperature 0 — a different codec on either side would make this a
/// measurement of the codec instead of the round loop. Both are given the
/// verifier's stop ids and both must produce a real answer of comparable
/// length — see [`MIN_ANSWER_TOKENS`] and [`MIN_LENGTH_RATIO`].
#[ignore]
#[test]
fn speculative_greedy_reproduces_plain_greedy() {
    const TEST: &str = "speculative_greedy_reproduces_plain_greedy";
    let (Some(model_path), Some(draft_path)) = (
        common::apply(resolve(common::SINGLE_MODEL_VAR, VERIFIER_SLUG), TEST),
        common::apply(resolve(DRAFT_MODEL_VAR, DRAFTER_SLUG), TEST),
    ) else {
        return;
    };

    if !is_gemma4_assistant(&draft_path) {
        eprintln!(
            "[spec_equiv] {} is not a Gemma4 assistant drafter - skipping; the floor is \
             calibrated on the exact-rollback regime only",
            draft_path.display()
        );
        return;
    }

    let device = Device::Gpu;
    let verifier =
        arch::load_model(&model_path, device, &arch::LoadOpts::default()).expect("load verifier");
    if verifier.needs_lin_caches() {
        eprintln!(
            "[spec_equiv] {} carries recurrent state - skipping; see the module docs",
            model_path.display()
        );
        return;
    }
    let hidden = verifier.hidden_size();
    let tk =
        tokenizers::Tokenizer::from_file(model_path.join("tokenizer.json")).expect("tokenizer");
    let prompt = build_prompt(&tk);
    let eos = eos_ids(&model_path);
    assert!(
        !eos.is_empty(),
        "the verifier config must name its stop ids — without them both arms run past \
         the answer into end-of-turn filler and the comparison is over that"
    );

    let mut spec_ids: Vec<u32> = Vec::new();
    {
        let mut step = |s: &rmlx_models::ProbeStep| {
            spec_ids.push(s.token_id);
            None
        };
        let drafter = Gemma4AssistantDrafter::load(&draft_path, hidden, device)
            .expect("load assistant drafter");
        mtp_assistant_generate_greedy(
            &verifier,
            &drafter,
            &tk,
            &prompt,
            N_TOKENS,
            BLOCK_SIZE,
            Some(rmlx_kv_quant::KvQuant::None),
            Some(MAX_CTX),
            &eos,
            &mut step,
            device,
        )
        .expect("assistant speculative generate");
    }

    let plain_ids = plain_greedy(&verifier, &tk, &prompt, &eos, device);

    let (tail_start, tail_ratio) = weakest_tail(&spec_ids, &plain_ids);
    let (spec_period, spec_cycle) = strongest_cycle(&spec_ids);
    let (plain_period, plain_cycle) = strongest_cycle(&plain_ids);
    eprintln!(
        "[spec_equiv] lcs={:.4} first_divergence={} tail={tail_ratio:.4}@{tail_start} \
         cycle spec={spec_cycle:.4}/p{spec_period} plain={plain_cycle:.4}/p{plain_period} \
         spec={} plain={}\n  spec  = {:?}\n  plain = {:?}",
        lcs_ratio(&spec_ids, &plain_ids),
        common_prefix_len(&spec_ids, &plain_ids),
        spec_ids.len(),
        plain_ids.len(),
        tk.decode(&spec_ids, false).unwrap_or_default(),
        tk.decode(&plain_ids, false).unwrap_or_default(),
    );

    if let Some(failure) = judge(&spec_ids, &plain_ids) {
        panic!("{failure}");
    }
}
