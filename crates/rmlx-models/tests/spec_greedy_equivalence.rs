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
//! favourable case there is — the two arms share 66 of 256 tokens and then
//! differ by one word ("Separate Chaining (Most Common Method)" against
//! "(The Most Common Method)"), after which both continue the same explanation.
//! A gate demanding identity would fail on that.
//!
//! **The oracle is therefore length-scaled, not positional.** The longest
//! common subsequence of the two token streams, over the shorter of them,
//! measures how much of one answer both arms produced wherever a benign flip
//! put it. Measured on that pair:
//!
//! | rollback | shared prefix | LCS ratio |
//! |---|---|---|
//! | as shipped | 66 / 256 | 0.914 |
//! | one rejected draft key left in the cache | 5 / 256 | 0.258 |
//!
//! [`MIN_LCS_RATIO`] sits in that gap. The shared prefix is not the oracle: it
//! collapses the moment a near-tie flips early, which is a property of the
//! prompt and the arithmetic rather than of the round loop.
//!
//! **Scope: an exact rollback.** A verifier carrying recurrent state has no
//! sequence axis to truncate — the state is restored from a snapshot and
//! replayed — and it diverges far more. Qwen3.8-27B with its MTP sidecar, as
//! shipped, reads 0.520 on this measure (0.336 with the reasoning block
//! suppressed), which is not separable from a broken exact rollback's 0.258.
//! Whether that is the reduction order or the replay is an open question and
//! not one a threshold should paper over, so this gate skips a recurrent
//! verifier and says so. Those loops keep their own shorter-horizon
//! prefix-tracking gates (`qwen3_5_mtp_drafter_alignment` and its siblings).
//!
//! **Length.** Two configurations that differ can agree for the first few dozen
//! tokens of an easy answer, so a short run is a gate that cannot fail. This
//! runs [`N_TOKENS`] tokens for that reason.
//!
//! **The control.** A token stream that has collapsed into a repetition loop
//! matches anything, so "the two agree" would pass on garbage. Every run also
//! checks that neither arm matches its own one-token shift, on the same tokens,
//! in the same run as the assertion it protects.
//!
//! Server-free. Run:
//! RMLX_KV_TEST_MODEL=/path/to/gemma-4-e2b-it-mxfp8 \
//! RMLX_DRAFT_TEST_MODEL=/path/to/gemma-4-E2B-it-assistant-bf16 \
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

use rmlx_mlx::Device;
use rmlx_models::arch;
use rmlx_models::speculative::gemma4_assistant::{
    mtp_assistant_generate_greedy, Gemma4AssistantDrafter,
};

/// Generated tokens per arm.
///
/// Not a round number for its own sake: a digest over 32 generated tokens has
/// already given two configurations of this engine the same value where 200
/// separated them, so a gate at that length cannot fail.
const N_TOKENS: usize = 256;

/// Draft block. Small enough that a rollback runs every few tokens, which is
/// the code path the oracle protects.
const BLOCK_SIZE: usize = 4;

/// How much of one answer both arms must have produced.
///
/// Between the two measured regimes in the module docs: 1.3x below the shipped
/// rollback's reading and 2.7x above a rollback that leaves one rejected key in
/// the cache.
const MIN_LCS_RATIO: f64 = 0.70;

/// The most of an arm that may match its own one-token shift before the stream
/// counts as degenerate. A repetition loop matches its shift almost everywhere;
/// ordinary prose matches it for a token or two.
const MAX_SELF_SHIFT_FRACTION: f64 = 0.25;

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

/// How much of `tokens` matches itself shifted by one position.
///
/// The control: a stream stuck in a repetition loop matches its own shift, so
/// this is high exactly when agreement between the arms would mean nothing.
fn self_shift_prefix_len(tokens: &[u32]) -> usize {
    if tokens.len() < 2 {
        return 0;
    }
    common_prefix_len(tokens, &tokens[1..])
}

/// Judge one pair of streams. Returns the failure text, or `None` when the
/// oracle and both controls held.
fn judge(spec: &[u32], plain: &[u32]) -> Option<String> {
    let shorter = spec.len().min(plain.len());
    if shorter < N_TOKENS {
        return Some(format!(
            "an arm stopped early — spec={} plain={} of {N_TOKENS}; a short run is a \
             comparison with no power, not a pass",
            spec.len(),
            plain.len()
        ));
    }

    let shift_ceiling = (shorter as f64 * MAX_SELF_SHIFT_FRACTION) as usize;
    for (name, arm) in [("plain", plain), ("speculative", spec)] {
        let self_shift = self_shift_prefix_len(arm);
        if self_shift > shift_ceiling {
            return Some(format!(
                "the {name} arm matches its own one-token shift for {self_shift} of \
                 {shorter} tokens (ceiling {shift_ceiling}) — it has collapsed into a \
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
/// The control is what separates that from a real match.
#[test]
fn a_repetition_loop_fails_even_though_the_arms_agree() {
    let looping = vec![7u32; N_TOKENS];
    let failure = judge(&looping, &looping).expect("the control must refuse this");
    assert!(failure.contains("repetition loop"), "{failure}");
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

/// A run that stopped short is refused rather than passed on its short prefix.
#[test]
fn a_truncated_run_is_refused_rather_than_judged() {
    let plain: Vec<u32> = (0..N_TOKENS as u32).collect();
    let failure = judge(&plain[..16], &plain).expect("must refuse");
    assert!(failure.contains("stopped early"), "{failure}");
}

/// The floor admits what the shipped rollback produced and refuses what the
/// broken one did, on the readings recorded in the module docs. Moving the
/// floor out of that gap fails here rather than at the next GPU run.
#[test]
fn the_floor_sits_between_the_two_measured_regimes() {
    // (what produced it, LCS ratio, must pass)
    let measured = [
        ("gemma-4-e2b assistant, as shipped", 0.914_f64, true),
        (
            "gemma-4-e2b assistant, one rejected key kept",
            0.258_f64,
            false,
        ),
    ];
    for (regime, ratio, want_pass) in measured {
        assert_eq!(
            ratio >= MIN_LCS_RATIO,
            want_pass,
            "{regime} read {ratio} against floor {MIN_LCS_RATIO}"
        );
    }
}

// ── The gate ─────────────────────────────────────────────────────────────────

fn env_path(key: &str) -> Option<PathBuf> {
    std::env::var(key)
        .ok()
        .map(PathBuf::from)
        .filter(|p| p.exists())
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

/// Plain greedy decoding on the verifier alone, at temperature 0.
fn plain_greedy(
    verifier: &arch::Architecture,
    tk: &tokenizers::Tokenizer,
    prompt: &[u32],
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
                None,
                1,
                &[],
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
/// measurement of the codec instead of the round loop. Neither arm is given
/// stop ids: both must run the full budget, or the comparison is over whatever
/// prefix the shorter one happened to produce.
#[ignore]
#[test]
fn speculative_greedy_reproduces_plain_greedy() {
    let (Some(model_path), Some(draft_path)) = (
        env_path("RMLX_KV_TEST_MODEL"),
        env_path("RMLX_DRAFT_TEST_MODEL"),
    ) else {
        eprintln!("[spec_equiv] verifier/drafter unset or absent - skipping");
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
            None,
            &[],
            &mut step,
            device,
        )
        .expect("assistant speculative generate");
    }

    let plain_ids = plain_greedy(&verifier, &tk, &prompt, device);

    eprintln!(
        "[spec_equiv] lcs={:.4} prefix={} self_shift={}/{} spec={} plain={}\n  spec  = {:?}\n  plain = {:?}",
        lcs_ratio(&spec_ids, &plain_ids),
        common_prefix_len(&spec_ids, &plain_ids),
        self_shift_prefix_len(&spec_ids),
        self_shift_prefix_len(&plain_ids),
        spec_ids.len(),
        plain_ids.len(),
        tk.decode(&spec_ids, false).unwrap_or_default(),
        tk.decode(&plain_ids, false).unwrap_or_default(),
    );

    if let Some(failure) = judge(&spec_ids, &plain_ids) {
        panic!("{failure}");
    }
}
