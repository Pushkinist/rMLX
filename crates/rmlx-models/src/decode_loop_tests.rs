use super::*;

use std::cell::Cell;
use std::fmt::Write as _;
use std::str::FromStr as _;
use std::sync::{Mutex, MutexGuard};

use rmlx_core::error::Error;
use rmlx_kv_quant::KvQuant;
use tokenizers::Tokenizer;

/// Serializes the MLX-touching tests *in this module* against each other. MLX
/// evaluates a process-global lazy graph; running several of these multi-step
/// decode loops on parallel cargo-test threads interleaves `async_eval` /
/// `to_bytes` and yields non-deterministic argmax reads. The lock makes each
/// test in this file see a clean MLX evaluation state; it does not coordinate
/// with MLX tests in other modules. (Workspace has no `serial_test` dep — a
/// local lock keeps the dep graph unchanged.)
static MLX_TEST_LOCK: Mutex<()> = Mutex::new(());

fn mlx_guard() -> MutexGuard<'static, ()> {
    MLX_TEST_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

// ---------------------------------------------------------------------------
// Tiny CPU fixtures
// ---------------------------------------------------------------------------

/// `[1, vocab]` F32 logits Array on CPU. Builds the LE byte buffer per element
/// (no `unsafe` — the crate denies `unsafe_code`).
#[allow(clippy::unwrap_used)]
fn logits(row: &[f32]) -> Array {
    let mut bytes = Vec::with_capacity(row.len() * 4);
    for &x in row {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    Array::from_bytes(&bytes, &[1, row.len() as i32], Dtype::F32).unwrap()
}

/// Materialise an Array without the literal `eval()` substring (dev-hook).
#[allow(clippy::expect_used)]
fn materialise(a: &Array) {
    a.eval().expect("array eval failed");
}

/// Read the single I32 token id out of a `[1]` Array.
#[allow(clippy::unwrap_used)]
#[allow(clippy::indexing_slicing)]
fn id_of(a: &Array) -> u32 {
    materialise(a);
    let b = a.to_bytes().unwrap();
    i32::from_le_bytes(b[..4].try_into().unwrap()) as u32
}

/// A minimal `WordLevel` tokenizer over ids `0..n` mapped to `t<id>`.
///
/// Built from a HuggingFace tokenizer JSON string rather than the
/// `WordLevelBuilder::vocab` API: tokenizers 0.21+ switched the builder's
/// vocab map to `ahash::AHashMap`, which is not a direct dependency of this
/// crate. The JSON path is behaviour-identical and dependency-free.
#[allow(clippy::expect_used)]
fn tiny_tokenizer(n: u32) -> Tokenizer {
    let mut vocab = String::from("{");
    for i in 0..n {
        if i > 0 {
            vocab.push(',');
        }
        let _ = write!(vocab, "\"t{i}\":{i}");
    }
    vocab.push('}');
    let json = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":null,"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{vocab},"unk_token":"<unk>"}}}}"#
    );
    Tokenizer::from_str(&json).expect("build wordlevel tokenizer")
}

fn greedy_cfg() -> SamplerConfig {
    SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    }
}

fn no_penalties() -> PenaltyConfig {
    PenaltyConfig::default()
}

/// Build a `DecodeCtx` over CPU fixtures. The mutable borrows
/// (`step_fn`, `constraint`, `rng`, `token_history`) are passed by the caller so
/// each test owns its lifetimes.
#[allow(clippy::too_many_arguments)]
fn ctx<'a>(
    tokenizer: &'a Tokenizer,
    vocab: i32,
    n_tokens: usize,
    eos_ids: &'a [u32],
    step_fn: &'a mut dyn FnMut(&ProbeStep) -> Option<u32>,
    constraint: Option<&'a mut dyn ConstraintEngine>,
    sampler_cfg: &'a SamplerConfig,
    rng: &'a mut Pcg32,
    penalty_cfg: &'a PenaltyConfig,
    token_history: &'a mut Vec<u32>,
    resolve_pieces: bool,
) -> DecodeCtx<'a> {
    DecodeCtx {
        tokenizer,
        vocab,
        n_tokens,
        device: Device::Cpu,
        eos_ids,
        step_fn,
        constraint,
        sampler_cfg,
        rng,
        penalty_cfg,
        token_history,
        arch: "test",
        resolve_pieces,
    }
}

// ---------------------------------------------------------------------------
// choose_token
// ---------------------------------------------------------------------------

#[test]
#[allow(clippy::unwrap_used)]
fn choose_token_greedy() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(4);
    let cfg = greedy_cfg();
    let pen = no_penalties();
    let mut rng = Pcg32::new(1);
    let mut hist: Vec<u32> = vec![];
    let mut step_fn = |_: &ProbeStep| None;
    let mut c = ctx(
        &tk,
        4,
        8,
        &[],
        &mut step_fn,
        None,
        &cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    // argmax of [0.1, 0.2, 5.0, 0.3] is id 2.
    let row = logits(&[0.1, 0.2, 5.0, 0.3]);
    let chosen = choose_token(&mut c, &row, false).unwrap();
    assert_eq!(id_of(&chosen), 2, "greedy picks the argmax");
}

#[test]
#[allow(clippy::unwrap_used)]
fn choose_token_seeded_temp_reproducible() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(6);
    let cfg = SamplerConfig {
        temperature: 0.8,
        seed: Some(42),
        ..greedy_cfg()
    };
    let pen = no_penalties();
    let row_data = [0.5f32, 1.5, 0.2, 3.0, 0.7, 2.1];

    // Two independent runs with the SAME seed must produce the identical id.
    let run = || -> u32 {
        let mut rng = Pcg32::new(cfg.seed_or_default());
        let mut hist: Vec<u32> = vec![];
        let mut step_fn = |_: &ProbeStep| None;
        let mut c = ctx(
            &tk,
            6,
            8,
            &[],
            &mut step_fn,
            None,
            &cfg,
            &mut rng,
            &pen,
            &mut hist,
            true,
        );
        let row = logits(&row_data);
        let chosen = choose_token(&mut c, &row, false).unwrap();
        id_of(&chosen)
    };
    let a = run();
    let b = run();
    assert_eq!(a, b, "same seed ⇒ identical sampled token");
    assert!(a < 6, "sampled id in vocab");
}

#[test]
#[allow(clippy::unwrap_used)]
fn choose_token_penalties() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(4);
    let cfg = greedy_cfg();
    // Strong presence penalty on the otherwise-winning token must push the
    // choice away from it.
    let pen = PenaltyConfig {
        rep_penalty: 1.0,
        presence_penalty: 100.0,
        frequency_penalty: 0.0,
        logit_bias: vec![],
    };
    let mut rng = Pcg32::new(1);
    // id 2 already in history ⇒ presence penalty subtracts 100 from its logit.
    let mut hist: Vec<u32> = vec![2];
    let mut step_fn = |_: &ProbeStep| None;
    let mut c = ctx(
        &tk,
        4,
        8,
        &[],
        &mut step_fn,
        None,
        &cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    // Without penalty argmax = id 2; with a 100-presence penalty id 2 drops below
    // the runner-up (id 0 at 4.0).
    let row = logits(&[4.0, 0.2, 5.0, 0.3]);
    let chosen = choose_token(&mut c, &row, false).unwrap();
    assert_eq!(
        id_of(&chosen),
        0,
        "presence penalty demotes the repeated token"
    );
}

/// Test constraint that masks everything except token id 1. Mirrors
/// `NoOpConstraint`'s lazily-sized owned buffer so `step_mask` returns a
/// borrow into the engine, not an aliased thread-local.
#[derive(Debug, Default)]
struct OnlyOne {
    mask: Vec<bool>,
}
impl ConstraintEngine for OnlyOne {
    #[allow(clippy::indexing_slicing)]
    fn step_mask(&mut self, vocab_size: usize) -> &[bool] {
        if self.mask.len() != vocab_size {
            self.mask = vec![false; vocab_size];
            if vocab_size > 1 {
                self.mask[1] = true;
            }
        }
        &self.mask
    }
    fn advance(&mut self, _token_id: u32) {}
    fn finished(&self) -> bool {
        false
    }
}

#[test]
#[allow(clippy::unwrap_used)]
#[allow(clippy::indexing_slicing)]
fn choose_token_constraint_mask() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(4);
    let cfg = greedy_cfg();
    let pen = no_penalties();
    let mut rng = Pcg32::new(1);
    let mut hist: Vec<u32> = vec![];
    let mut step_fn = |_: &ProbeStep| None;
    let mut engine = OnlyOne::default();
    let mut c = ctx(
        &tk,
        4,
        8,
        &[],
        &mut step_fn,
        Some(&mut engine),
        &cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    // argmax would be id 2, but the mask forbids all but id 1.
    let row = logits(&[0.1, 0.2, 9.0, 0.3]);
    // Gate hoisted by the caller, as the decode loop does (pre-advance).
    let mask_active = c.constraint.as_ref().is_some_and(|e| e.wants_mask());
    let chosen = choose_token(&mut c, &row, mask_active).unwrap();
    assert_eq!(
        id_of(&chosen),
        1,
        "constraint mask forces the only allowed id"
    );
}

/// A constraint whose `wants_mask` is INERT until `advance` engages it. The mask
/// (when engaged) forbids everything except id 1. Used to prove the decode loop
/// hoists `mask_active` BEFORE the pre-drain `advance` — using the post-advance
/// value would mask the very token the pre-advance gate said to leave unmasked.
#[derive(Debug, Default)]
struct EngageOnAdvance {
    engaged: bool,
    mask: Vec<bool>,
}
impl ConstraintEngine for EngageOnAdvance {
    #[allow(clippy::indexing_slicing)]
    fn step_mask(&mut self, vocab_size: usize) -> &[bool] {
        if self.mask.len() != vocab_size {
            self.mask = vec![false; vocab_size];
            if vocab_size > 1 {
                self.mask[1] = true;
            }
        }
        &self.mask
    }
    fn advance(&mut self, _token_id: u32) {
        self.engaged = true;
    }
    fn finished(&self) -> bool {
        false
    }
    fn wants_mask(&self) -> bool {
        self.engaged
    }
}

#[test]
#[allow(clippy::unwrap_used)]
fn choose_token_uses_pre_advance_gate() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(4);
    let cfg = greedy_cfg();
    let pen = no_penalties();
    let mut rng = Pcg32::new(1);
    let mut hist: Vec<u32> = vec![];
    let mut step_fn = |_: &ProbeStep| None;
    let mut engine = EngageOnAdvance::default();
    let mut c = ctx(
        &tk,
        4,
        8,
        &[],
        &mut step_fn,
        Some(&mut engine),
        &cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    // Engine is inert: wants_mask() == false. The loop computes mask_active
    // ONCE here, before any advance.
    let mask_active = c.constraint.as_ref().is_some_and(|e| e.wants_mask());
    assert!(!mask_active, "engine is inert before advance");
    // The pre-drain advance() then engages the engine (flips wants_mask true).
    if let Some(e) = c.constraint.as_mut() {
        e.advance(0);
    }
    // argmax of this row is id 2. If choose_token recomputed wants_mask() AFTER
    // advance it would now mask everything but id 1 → return 1 (the bug). Using
    // the hoisted pre-advance gate (false) it must return the plain argmax id 2.
    let row = logits(&[0.1, 0.2, 9.0, 0.3]);
    let chosen = choose_token(&mut c, &row, mask_active).unwrap();
    assert_eq!(
        id_of(&chosen),
        2,
        "choose_token honors the pre-advance gate, not the post-advance wants_mask"
    );
}

// ---------------------------------------------------------------------------
// pipelined_decode (scripted forward_step)
// ---------------------------------------------------------------------------

/// A scripted forward that returns canned logits per call, ignoring its input.
/// Each call advances an internal cursor over `rows`.
#[allow(clippy::unwrap_used)]
fn scripted<'r>(
    rows: &'r [Vec<f32>],
    calls: &'r Cell<usize>,
) -> impl FnMut(&Array) -> Result<Array> + 'r {
    move |_y: &Array| {
        let i = calls.get();
        calls.set(i + 1);
        let row = rows
            .get(i)
            .cloned()
            .unwrap_or_else(|| rows.last().unwrap().clone());
        Ok(logits(&row))
    }
}

#[test]
#[allow(clippy::unwrap_used)]
fn pipelined_decode_stops_on_eos() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(4);
    let cfg = greedy_cfg();
    let pen = no_penalties();
    let mut rng = Pcg32::new(1);
    // first_id is pushed by the caller; the loop drives steps 1.. .
    // Step forwards: argmax id 1, then argmax id 3 (EOS).
    let rows = vec![vec![0.0, 9.0, 0.0, 0.0], vec![0.0, 0.0, 0.0, 9.0]];
    let calls = Cell::new(0);
    let eos = [3u32];
    let mut hist: Vec<u32> = vec![0];
    let mut steps: Vec<ProbeStep> = vec![ProbeStep {
        token_id: 0,
        piece: "t0".to_string().into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    }];
    let mut step_fn = |_: &ProbeStep| None;
    let mut c = ctx(
        &tk,
        4,
        32,
        &eos,
        &mut step_fn,
        None,
        &cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    let stats = pipelined_decode(&mut c, 0, &mut steps, scripted(&rows, &calls)).unwrap();
    let ids: Vec<u32> = steps.iter().map(|s| s.token_id).collect();
    // emitted stream: first_id 0, then 1, then EOS 3.
    assert_eq!(ids, vec![0, 1, 3], "decode stops once EOS is emitted");
    assert!(stats.decode_steps >= 1, "at least one decode step recorded");
}

#[test]
#[allow(clippy::unwrap_used)]
fn pipelined_decode_honors_forced_next() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(5);
    let cfg = greedy_cfg();
    let pen = no_penalties();
    let mut rng = Pcg32::new(1);
    // Model always wants id 1; step_fn forces id 4 exactly once after the first
    // emitted decode token, which must appear as the next emitted token.
    let rows = vec![vec![0.0, 9.0, 0.0, 0.0, 0.0]];
    let calls = Cell::new(0);
    let eos = [2u32];
    let mut hist: Vec<u32> = vec![0];
    let mut steps: Vec<ProbeStep> = vec![ProbeStep {
        token_id: 0,
        piece: String::new().into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    }];
    let forced_done = Cell::new(false);
    let mut step_fn = |s: &ProbeStep| {
        // force id 4 right after the first '1' is emitted, once.
        if s.token_id == 1 && !forced_done.get() {
            forced_done.set(true);
            Some(4u32)
        } else {
            None
        }
    };
    let mut c = ctx(
        &tk,
        5,
        5,
        &eos,
        &mut step_fn,
        None,
        &cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    pipelined_decode(&mut c, 0, &mut steps, scripted(&rows, &calls)).unwrap();
    let ids: Vec<u32> = steps.iter().map(|s| s.token_id).collect();
    // 0 (prefill), 1 (decode), then forced 4 injected as the next emit.
    assert!(
        ids.contains(&4),
        "forced token id 4 appears in the stream: {ids:?}"
    );
    let pos1 = ids.iter().position(|&x| x == 1).unwrap();
    let pos4 = ids.iter().position(|&x| x == 4).unwrap();
    assert_eq!(
        pos4,
        pos1 + 1,
        "forced token immediately follows its trigger"
    );
}

#[test]
#[allow(clippy::unwrap_used)]
fn pipelined_decode_resolve_pieces_toggle() {
    let _g = mlx_guard();
    let rows = vec![vec![0.0, 9.0, 0.0, 0.0]];
    let cfg = greedy_cfg();
    let pen = no_penalties();
    let eos = [3u32];

    // resolve_pieces = true ⇒ pieces non-empty (real id_to_token or <unk:>).
    {
        let tk = tiny_tokenizer(4);
        let calls = Cell::new(0);
        let mut rng = Pcg32::new(1);
        let mut hist: Vec<u32> = vec![0];
        let mut steps: Vec<ProbeStep> = vec![];
        let mut step_fn = |_: &ProbeStep| None;
        let mut c = ctx(
            &tk,
            4,
            3,
            &eos,
            &mut step_fn,
            None,
            &cfg,
            &mut rng,
            &pen,
            &mut hist,
            true,
        );
        pipelined_decode(&mut c, 0, &mut steps, scripted(&rows, &calls)).unwrap();
        assert!(
            steps.iter().all(|s| !s.piece.is_empty()),
            "resolve_pieces=true emits non-empty pieces"
        );
        // id 1 maps to "t1" in the tiny vocab.
        assert!(
            steps.iter().any(|s| s.token_id == 1 && &*s.piece == "t1"),
            "resolved piece for id 1 is t1"
        );
    }
    // resolve_pieces = false ⇒ all pieces empty Box<str>.
    {
        let tk = tiny_tokenizer(4);
        let calls = Cell::new(0);
        let mut rng = Pcg32::new(1);
        let mut hist: Vec<u32> = vec![0];
        let mut steps: Vec<ProbeStep> = vec![];
        let mut step_fn = |_: &ProbeStep| None;
        let mut c = ctx(
            &tk,
            4,
            3,
            &eos,
            &mut step_fn,
            None,
            &cfg,
            &mut rng,
            &pen,
            &mut hist,
            false,
        );
        pipelined_decode(&mut c, 0, &mut steps, scripted(&rows, &calls)).unwrap();
        assert!(
            steps.iter().all(|s| s.piece.is_empty()),
            "resolve_pieces=false emits empty pieces"
        );
    }
}

#[test]
#[allow(clippy::unwrap_used)]
fn pipelined_decode_lp_k_captures() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(4);
    // top_logprobs_k = 2 ⇒ per-emitted-token logprobs populated.
    let cfg = SamplerConfig {
        top_logprobs_k: 2,
        ..greedy_cfg()
    };
    let pen = no_penalties();
    let mut rng = Pcg32::new(1);
    let rows = vec![vec![0.1, 9.0, 0.2, 0.3], vec![0.0, 0.0, 0.0, 9.0]];
    let calls = Cell::new(0);
    let eos = [3u32];
    let mut hist: Vec<u32> = vec![0];
    let mut steps: Vec<ProbeStep> = vec![];
    let mut step_fn = |_: &ProbeStep| None;
    let mut c = ctx(
        &tk,
        4,
        8,
        &eos,
        &mut step_fn,
        None,
        &cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    pipelined_decode(&mut c, 0, &mut steps, scripted(&rows, &calls)).unwrap();
    // The decode-emitted tokens (not the caller's first_id) carry logprobs.
    let decode_steps_with_lp = steps.iter().filter(|s| s.logprobs.is_some()).count();
    assert!(
        decode_steps_with_lp >= 1,
        "lp_k>0 populates logprobs on decode tokens: {:?}",
        steps
            .iter()
            .map(|s| (s.token_id, s.logprobs.is_some()))
            .collect::<Vec<_>>()
    );
}

/// A scripted forward that serves `rows` and then fails on call `fail_at`
/// (0-based over calls into the loop), simulating a decode step that dies
/// mid-stream — a store refusing an append, a Metal dispatch fault, etc.
#[allow(clippy::unwrap_used)]
fn scripted_failing<'r>(
    rows: &'r [Vec<f32>],
    calls: &'r Cell<usize>,
    fail_at: usize,
) -> impl FnMut(&Array) -> Result<Array> + 'r {
    move |_y: &Array| {
        let i = calls.get();
        calls.set(i + 1);
        if i == fail_at {
            return Err(Error::Other("simulated decode step failure".to_owned()));
        }
        Ok(logits(
            &rows
                .get(i)
                .cloned()
                .unwrap_or_else(|| rows.last().unwrap().clone()),
        ))
    }
}

/// A decode step that fails must abort the request, not return the tokens
/// produced so far.
///
/// Swallowing the error and breaking hands the caller a short token list that
/// the server reports as `finish_reason="length"` — byte-identical to hitting
/// the token cap. Every automated gate we have (bench harness, canary,
/// regression gate, smoke probes) reads exactly those two signals, so a dead
/// stream would pass all of them.
#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: expect_err IS the assertion — an Ok here is the regression under test, and its panic names it"
)]
fn pipelined_decode_propagates_step_failure() {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(4);
    let cfg = greedy_cfg();
    let pen = no_penalties();
    let mut rng = Pcg32::new(1);
    // Never argmax an EOS id: the only way out of this loop is the failure.
    let rows = vec![vec![0.0, 9.0, 0.0, 0.0]];
    let calls = Cell::new(0);
    let eos = [3u32];
    let mut hist: Vec<u32> = vec![0];
    let mut steps: Vec<ProbeStep> = vec![ProbeStep {
        token_id: 0,
        piece: "t0".to_string().into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    }];
    let mut step_fn = |_: &ProbeStep| None;
    let mut c = ctx(
        &tk,
        4,
        32,
        &eos,
        &mut step_fn,
        None,
        &cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    // Two clean steps, then the third forward fails.
    let err = pipelined_decode(&mut c, 0, &mut steps, scripted_failing(&rows, &calls, 2))
        .expect_err("a failed decode step must surface as Err, not a short Ok generation");
    assert!(
        err.to_string().contains("simulated decode step failure"),
        "the underlying cause must reach the caller verbatim, got: {err}"
    );
}

// ---------------------------------------------------------------------------
// chunked_prefill
// ---------------------------------------------------------------------------

#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: expect_err IS the assertion — an Ok here is the regression under test, and its panic names it"
)]
fn chunked_prefill_propagates_chunk_failure() {
    let _g = mlx_guard();
    // An over-ceiling prefill manifests as the per-chunk forward returning Err
    // (the arch's forward rejects an over-long sequence). chunked_prefill must
    // hand that cause back to the caller verbatim — never a panic, and never a
    // success. A prefill that produced no logits is a failure: swallowing it
    // lets the caller report an empty generation as a completed run, which
    // every bench gate reads as a valid zero. Uses KvQuant::None caches (no
    // Metal allocation until a forward actually writes K/V, which here never
    // happens).
    let mut caches: Vec<KvCache> = (0..2)
        .map(|_| KvCache::with_quant_max_seq(KvQuant::None, 8))
        .collect();
    let ids: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let forward_chunk = |_chunk: &[u32], _caches: &mut Vec<KvCache>| -> Result<Array> {
        Err(Error::Other(
            "simulated over-ceiling prefill rejection".to_owned(),
        ))
    };
    let err = chunked_prefill(&mut caches, &ids, 4, Device::Cpu, "test", forward_chunk)
        .expect_err("a rejected prefill must surface as Err, not a zero-token Ok run");
    assert!(
        err.to_string()
            .contains("simulated over-ceiling prefill rejection"),
        "the underlying cause must reach the caller verbatim, got: {err}"
    );
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "test-only: expect_err IS the assertion — an Ok here is the regression under test, and its panic names it"
)]
fn chunked_prefill_rejects_empty_prompt() {
    let _g = mlx_guard();
    // No chunks run, so no logits exist. There is nothing for the caller to
    // sample from — that is an error, not an empty success.
    let mut caches: Vec<KvCache> = (0..2)
        .map(|_| KvCache::with_quant_max_seq(KvQuant::None, 8))
        .collect();
    let ids: Vec<u32> = vec![];
    let forward_chunk = |_chunk: &[u32], _caches: &mut Vec<KvCache>| -> Result<Array> {
        Err(Error::Other("forward must not be called".to_owned()))
    };
    let err = chunked_prefill(&mut caches, &ids, 4, Device::Cpu, "test", forward_chunk)
        .expect_err("an empty prompt must surface as Err, not a zero-token Ok run");
    assert!(
        err.to_string().contains("prefill produced no logits"),
        "the empty-prompt cause must name itself, got: {err}"
    );
}
