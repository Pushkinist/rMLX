//! The stop of a request in the loops of this crate. No model, no GPU.
//!
//! Two instruments, and each is blind where the other reads:
//!
//! - The shared decode loop and the shared prefill run under scripted
//!   forwards. These tests read what the loop did after the stop was set, what
//!   it returned, and that its duty after the loop still ran. They cannot read
//!   where the stop was read: a forward that refuses for itself gives the same
//!   counts. The outcome is what tells the two apart here, because a refused
//!   forward is logged as an error and a stop is not.
//! - A source rule finds every loop that does work for a request, from the
//!   work it does, and reads that the first statement of each one is the one
//!   stop helper with `?`. It cannot read that the caller handed over the flag
//!   of the request.
//!
//! The source rule has one test for each group of loops, so a change that
//! gives one group its stop turns one test green. A loop that only one
//! architecture has needs its weights to run, so the source rule is its only
//! check here. `tests/gone_request.rs` of the server crate runs the decode
//! loops of Qwen2, BitNet and Qwen3-VL, and the inline prefills of Qwen2 and
//! Qwen3-VL, on real models, and lists the stops that it does not reach.

use super::*;
use crate::stop_scan::{
    calls_a_flag_fn, calls_off_the_rule, can_leave, error_events, flag_fns_that_disagree,
    flag_uses_off_the_rule, fn_shapes, fns_of, helper_shadowed, loops_of, loops_under,
    not_stopping, opens_with_the_stop_read, prefill_sweeps, sources_under, stop_reads,
    stop_reads_past, stops_by_the_rule, top_level_args, unlisted_and_lost, FnShape, Loop,
    STOP_HELPER, STOP_OUTCOME,
};

use std::sync::atomic::{AtomicBool, Ordering};

/// Hand `stop` to the shared decode loop: one call that passes `stop` to the
/// loop, and wraps neither `forward` nor the step callback.
fn shared_decode(
    c: &mut DecodeCtx<'_>,
    steps: &mut Vec<ProbeStep>,
    stop: &AtomicBool,
    forward: impl FnMut(&Array) -> Result<Array>,
) -> Result<(DecodeStats, PostDecode)> {
    pipelined_decode(c, 0, steps, stop, forward)
}

/// Hand `stop` to the shared prefill, on the terms of [`shared_decode`].
fn shared_prefill(
    caches: &mut Vec<KvCache>,
    ids: &[u32],
    stop: &AtomicBool,
    forward_chunk: impl FnMut(&[u32], &mut Vec<KvCache>) -> Result<Array>,
) -> Result<Array> {
    chunked_prefill(
        caches,
        ids,
        (4, "arch_default"),
        None,
        Device::Cpu,
        "test",
        stop,
        forward_chunk,
    )
}

/// `true` when `e` is the outcome of a stop: an error variant of its own, which
/// the retry envelope does not replay.
fn is_the_stop(e: &Error) -> bool {
    format!("{e:?}").starts_with(STOP_OUTCOME) && !e.is_migratable()
}

/// When a scripted request goes away.
#[derive(Clone, Copy)]
enum Gone {
    Never,
    BeforeTheLoop,
    /// In the step callback of this emitted token, counted from 0 after the
    /// first token of the request.
    AtEmittedToken(usize),
}

/// What one scripted decode did.
struct Decoded {
    /// The ids the loop emitted after the first token of the request.
    ids: Vec<u32>,
    forwards: usize,
    /// Forwards that started with the stop set.
    forwards_after_stop: usize,
    /// Tokens that reached the step callback with the stop set.
    emitted_after_stop: usize,
    /// The error the loop returned, if it returned one.
    error: Option<Error>,
    /// Events the loop logged at `ERROR`.
    logged_errors: usize,
}

/// Decode `n_tokens` with a model that always picks id 1 and never an end
/// token: only the token budget or a stop ends this loop.
fn decode(cfg: &SamplerConfig, n_tokens: usize, gone: Gone) -> Decoded {
    let _g = mlx_guard();
    let tk = tiny_tokenizer(4);
    let pen = no_penalties();
    let mut rng = Pcg32::new(1);
    let eos = [3u32];
    let mut hist: Vec<u32> = vec![0];
    let mut steps: Vec<ProbeStep> = vec![ProbeStep {
        token_id: 0,
        piece: "t0".to_owned().into_boxed_str(),
        max_abs_logit: 0.0,
        nan_count: 0,
        logprobs: None,
    }];
    let stop = AtomicBool::new(matches!(gone, Gone::BeforeTheLoop));
    let emitted = Cell::new(0usize);
    let emitted_after_stop = Cell::new(0usize);
    let forwards = Cell::new(0usize);
    let forwards_after_stop = Cell::new(0usize);

    let mut step_fn = |_: &ProbeStep| {
        if stop.load(Ordering::Relaxed) {
            emitted_after_stop.set(emitted_after_stop.get() + 1);
        }
        if matches!(gone, Gone::AtEmittedToken(at) if at == emitted.get()) {
            stop.store(true, Ordering::Relaxed);
        }
        emitted.set(emitted.get() + 1);
        None
    };
    let forward = |_y: &Array| {
        forwards.set(forwards.get() + 1);
        if stop.load(Ordering::Relaxed) {
            forwards_after_stop.set(forwards_after_stop.get() + 1);
        }
        Ok(logits(&[0.0, 40.0, 0.0, 0.0]))
    };
    let mut c = ctx(
        &tk,
        4,
        n_tokens,
        &eos,
        &mut step_fn,
        None,
        cfg,
        &mut rng,
        &pen,
        &mut hist,
        true,
    );
    let mut error = None;
    let logged_errors = error_events(|| {
        error = shared_decode(&mut c, &mut steps, &stop, forward).err();
    });
    Decoded {
        ids: steps.iter().skip(1).map(|s| s.token_id).collect(),
        forwards: forwards.get(),
        forwards_after_stop: forwards_after_stop.get(),
        emitted_after_stop: emitted_after_stop.get(),
        error,
        logged_errors,
    }
}

/// The two selection paths of the shared loop: the pipelined argmax, and the
/// host selection that a temperature takes.
fn selection_paths() -> [(&'static str, SamplerConfig); 2] {
    [
        ("greedy", greedy_cfg()),
        (
            "sampled",
            SamplerConfig {
                temperature: 0.7,
                ..greedy_cfg()
            },
        ),
    ]
}

const BUDGET: usize = 64;

/// A stop that is never set changes nothing: the loop runs its whole budget
/// and emits what the model picked.
#[test]
fn a_decode_that_is_not_stopped_runs_its_whole_budget() {
    for (path, cfg) in selection_paths() {
        let run = decode(&cfg, BUDGET, Gone::Never);
        assert_eq!(run.ids, vec![1; BUDGET - 1], "{path}");
        assert_eq!(run.forwards, BUDGET - 1, "{path}");
        assert!(run.error.is_none(), "{path}: {:?}", run.error);
    }
}

/// A pass reads the stop before it starts its forward, and the step callback
/// that sets the stop here runs at the end of the pass before. So the loop
/// starts no forward and emits no token after the stop, on both selection
/// paths. The token that the last forward picked is not emitted: its request
/// is gone.
#[test]
fn a_stopped_decode_starts_no_further_forward() {
    for (path, cfg) in selection_paths() {
        let run = decode(&cfg, BUDGET, Gone::AtEmittedToken(2));
        assert_eq!(
            run.forwards_after_stop, 0,
            "{path}: of {} forwards in the request",
            run.forwards
        );
        assert_eq!(run.emitted_after_stop, 0, "{path}");
    }
}

/// A request that is gone before its decode starts runs no forward.
#[test]
fn a_decode_stopped_before_its_first_step_runs_no_forward() {
    for (path, cfg) in selection_paths() {
        let run = decode(&cfg, BUDGET, Gone::BeforeTheLoop);
        assert_eq!(run.forwards, 0, "{path}");
        assert_eq!(run.ids, Vec::<u32>::new(), "{path}");
    }
}

/// A stopped decode returns the stop outcome: not a decode that ran to its
/// end, which the caller would record and report as cut by the budget, and
/// not a failure, which the retry envelope would run again. A request that is
/// gone is a normal event, so the loop logs no error for it.
#[test]
fn a_stopped_decode_returns_the_stop_outcome_and_logs_no_error() {
    for (path, cfg) in selection_paths() {
        let run = decode(&cfg, BUDGET, Gone::AtEmittedToken(2));
        assert!(
            run.error.as_ref().is_some_and(is_the_stop),
            "{path}: {:?}",
            run.error
        );
        assert_eq!(run.logged_errors, 0, "{path}");
    }
}

/// What one scripted prefill did.
struct Prefilled {
    chunks: usize,
    /// Chunks that started with the stop set.
    chunks_after_stop: usize,
    /// The error the prefill returned, if it returned one.
    error: Option<Error>,
    /// Events the prefill logged at `ERROR`.
    logged_errors: usize,
    /// Caches that the prefill left in its prefill state.
    left_in_prefill: usize,
}

/// How a scripted prefill ends early, if it does.
#[derive(Clone, Copy)]
enum Ends {
    Never,
    /// The request goes away while this chunk runs.
    GoneIn(usize),
    /// The forward of this chunk fails.
    FailsIn(usize),
}

/// The message of the error that a scripted forward fails with.
const SCRIPTED_FAILURE: &str = "scripted forward failure";

/// Prefill 16 tokens in 4 chunks, and end as `ends` says.
fn prefill(ends: Ends) -> Prefilled {
    let _g = mlx_guard();
    let mut caches: Vec<KvCache> = (0..3)
        .map(|_| KvCache::with_quant_max_seq(KvQuant::None, 32))
        .collect();
    let ids: Vec<u32> = (0..16).collect();
    let stop = AtomicBool::new(false);
    let chunks = Cell::new(0usize);
    let chunks_after_stop = Cell::new(0usize);
    let forward_chunk = |_chunk: &[u32], caches: &mut Vec<KvCache>| {
        assert!(
            caches.iter().all(KvCache::in_prefill),
            "the prefill did not enter its prefill state on every cache"
        );
        if stop.load(Ordering::Relaxed) {
            chunks_after_stop.set(chunks_after_stop.get() + 1);
        }
        let this_chunk = chunks.get();
        chunks.set(this_chunk + 1);
        match ends {
            Ends::GoneIn(at) if at == this_chunk => stop.store(true, Ordering::Relaxed),
            Ends::FailsIn(at) if at == this_chunk => {
                return Err(Error::Model(SCRIPTED_FAILURE.to_owned()));
            }
            Ends::Never | Ends::GoneIn(_) | Ends::FailsIn(_) => {}
        }
        Ok(logits(&[0.0, 40.0, 0.0, 0.0]))
    };
    let mut error = None;
    let logged_errors = error_events(|| {
        error = shared_prefill(&mut caches, &ids, &stop, forward_chunk).err();
    });
    Prefilled {
        chunks: chunks.get(),
        chunks_after_stop: chunks_after_stop.get(),
        error,
        logged_errors,
        left_in_prefill: caches.iter().filter(|c| c.in_prefill()).count(),
    }
}

/// A stop that is never set changes nothing: every chunk runs and the prefill
/// gives the logits of its last one.
#[test]
fn a_prefill_that_is_not_stopped_runs_every_chunk() {
    let run = prefill(Ends::Never);
    assert_eq!(run.chunks, 4);
    assert!(run.error.is_none(), "{:?}", run.error);
    assert_eq!(run.left_in_prefill, 0);
}

/// A stopped prefill starts no further chunk, and it still takes every cache
/// out of its prefill state. The stop is an early exit as a failed chunk is,
/// and the exit sweep is a duty on every outcome: a stop read with `?` at the
/// head of the chunk loop, with nothing around the loop, skips the sweep and
/// fails here.
#[test]
fn a_stopped_prefill_starts_no_further_chunk_and_leaves_no_cache_in_prefill() {
    let run = prefill(Ends::GoneIn(1));
    assert_eq!(run.chunks, 2, "the stop did not end the prefill");
    assert_eq!(run.chunks_after_stop, 0);
    assert_eq!(run.left_in_prefill, 0, "caches left in their prefill state");
}

/// A stopped prefill gives no logits: a caller that got logits would pick a
/// first token and store the prompt in the cache as prefilled. It returns the
/// stop outcome, logs no error, and leaves no cache in its prefill state.
#[test]
fn a_stopped_prefill_returns_the_stop_outcome_and_logs_no_error() {
    let run = prefill(Ends::GoneIn(1));
    assert!(
        run.error.as_ref().is_some_and(is_the_stop),
        "{:?}",
        run.error
    );
    assert_eq!(run.logged_errors, 0);
    assert_eq!(run.left_in_prefill, 0, "caches left in their prefill state");
}

/// A forward that fails is not a stop. The prefill starts no further chunk,
/// still takes every cache out of its prefill state, and returns the error of
/// the forward as it is, logged once at `ERROR`: the retry envelope reads that
/// error, and a stop in its place would hide the failure.
#[test]
fn a_prefill_whose_forward_fails_starts_no_further_chunk_and_returns_that_error() {
    let run = prefill(Ends::FailsIn(1));
    assert_eq!(run.chunks, 2, "a chunk started after the failed one");
    assert_eq!(run.left_in_prefill, 0, "caches left in their prefill state");
    assert!(
        matches!(&run.error, Some(Error::Model(m)) if m == SCRIPTED_FAILURE),
        "{:?}",
        run.error
    );
    assert_eq!(run.logged_errors, 1);
}

// ── Every loop that does work for a request ─────────────────────────────────
//
// A loop is named by its file, its fn and its header line. A loop that moves
// into a fn of its own, for a duty that must run after it, is named there.

/// The decode loop and the prefill that several architectures share.
const SHARED_LOOPS: [&str; 2] = [
    "decode_loop.rs::chunked_prefill: for (chunk_idx, piece) in pieces.iter().enumerate() {",
    "decode_loop.rs::pipelined_decode: for step_idx in 1..ctx.n_tokens {",
];

/// The decode loops and the prefill loops that one architecture has for
/// itself.
const OWN_BODY_LOOPS: [&str; 11] = [
    "bitnet/generate.rs::decode_loop: for step_idx in 1..n_tokens {",
    "bitnet/generate.rs::generate_greedy: for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {",
    "gemma4/generate/mod.rs::generate_greedy: for (chunk_idx, chunk) in prefill_ids.chunks(prefill_chunk.0).enumerate() {",
    "laguna/generate.rs::decode_loop: for step_idx in 1..n_tokens {",
    "laguna/generate.rs::generate_greedy: for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {",
    "qwen2/generate.rs::decode_loop: for step_idx in 1..n_tokens {",
    "qwen2/generate.rs::generate_greedy: for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {",
    "qwen3_5_moe/generate.rs::generate_greedy: for (chunk_idx, piece) in pieces.iter().enumerate() {",
    "qwen3_vl_moe/generate.rs::decode_from: for _ in 0..n_tokens {",
    "qwen3_vl_moe/generate.rs::generate_image: for g in 0..n_tokens {",
    "qwen3_vl_moe/model.rs::forward_embeds_chunked: while start < seq {",
];

/// The tower loop that runs one evaluation for each block of an image.
const TOWER_LOOPS: [&str; 1] =
    ["qwen3_vl_moe/vision.rs::forward: for (layer_num, blk) in self.blocks.iter().enumerate() {"];

/// The round loop, and the prefills that only the drafters run.
const SPECULATIVE_LOOPS: [&str; 4] = [
    "qwen3_5_moe/model.rs::forward_verify_capture_chunked: for (piece_idx, piece) in pieces.iter().enumerate() {",
    "speculative/eagle3/mod.rs::prefill_from_verifier_hidden: while pos < total {",
    "speculative/mod.rs::prefill_chunked_with: for (chunk_idx, piece) in pieces.iter().enumerate() {",
    "speculative/round_loop.rs::run_rounds: while emitted.len() < n_tokens {",
];

/// The loops that do work and do not read the stop, each with its bound.
const BOUNDED_LOOPS: [(&str, &str); 23] = [
    ("gemma3/vision.rs::build_inputs_embeds: for (img_idx, pv) in images.iter().enumerate() {", "it builds the graph of every image, and one evaluation after the loop runs them all"),
    ("gemma3/vision.rs::forward: for blk in &self.encoder {", "the layers of one tower forward, built as one graph"),
    ("gemma4/vision/mod.rs::build_inputs_embeds: for (img_idx, pv) in images.iter().enumerate() {", "it builds the graph of every image, and one evaluation after the loop runs them all"),
    ("gemma4/vision/unified.rs::build_unified_inputs_embeds: for (img_idx, (pv, &n_soft)) in images.iter().zip(per_image.iter()).enumerate() {", "it builds the graph of every image, and one evaluation after the loop runs them all"),
    ("jina_v4/model.rs::install_task_loras: for layer in 0..self.num_layers() {", "a task adapter is installed over the layers and their projections; no forward"),
    ("jina_v4/model.rs::install_task_loras: for proj in ProjId::ALL {", "a task adapter is installed over the layers and their projections; no forward"),
    ("ppl.rs::compute_ppl_gemma4: while start < tokens.len() {", "the perplexity command, which no request of the server runs"),
    ("ppl.rs::score_window_through_cache: for t in (warmup + 1)..last_scored {", "the perplexity command, which no request of the server runs"),
    ("ppl.rs::sliding_window_ppl: while start < tokens.len() {", "the perplexity command, which no request of the server runs"),
    ("qwen3_5_moe/loader.rs::load_from_path: for i in 0..cfg.num_hidden_layers {", "a model load, which a stop of a request does not cut"),
    ("qwen3_5_moe/loader.rs::load_from_path_paro: for i in 0..cfg.num_hidden_layers {", "a model load, which a stop of a request does not cut"),
    ("qwen3_vl_moe/vision.rs::attend_tiled: while qs < seq {", "the tiles of one attention in one block; the block loop reads the stop"),
    ("speculative/dflash/mod.rs::greedy_block_tokens: for pos in 1..block_size {", "the positions of one draft block"),
    ("speculative/eagle3/mod.rs::draft_block: for _ in 0..n_iters {", "the draft block of one round; the stop is read at the head of the round"),
    ("speculative/eagle3/mod.rs::project_hidden: for (i, norm) in fcs.iter().enumerate() {", "the captured layers of one forward"),
    ("speculative/gemma4_assistant.rs::draft_n: while tokens.len() < block_size - 1 {", "the draft block of one round; the stop is read at the head of the round"),
    ("speculative/mod.rs::draft_decode_n: for _step_idx in 0..n {", "the draft block of one round; the stop is read at the head of the round"),
    ("speculative/mod.rs::draft_decode_n: for arr in emitted_arrays {", "the draft block of one round; the stop is read at the head of the round"),
    ("speculative/mod.rs::draft_decode_n_stochastic: for _step in 0..n {", "the draft block of one round; the stop is read at the head of the round"),
    ("speculative/mtp.rs::draft_n: while tokens.len() < block_size - 1 {", "the draft block of one round; the stop is read at the head of the round"),
    ("speculative/round_common.rs::refold_lin_tapes: for (idx, cache) in lin.iter_mut().enumerate() {", "the rollback of one round, over the recurrent layers and their states"),
    ("speculative/round_common.rs::refold_lin_tapes: for a in &refolded {", "the rollback of one round, over the recurrent layers and their states"),
    ("speculative/round_common.rs::refold_lin_tapes: for a in [&cache.conv_state, &cache.delta_state]", "the rollback of one round, over the recurrent layers and their states"),
];

/// Loops that the scan does not read, each as its file, its fn and its bound.
const UNREAD_LOOPS: [(&str, &str, &str); 2] = [
    (
        "speculative/mod.rs",
        "block_tokens",
        "the rows of one verify block",
    ),
    (
        "speculative/mod.rs",
        "block_distributions",
        "the rows of one verify block",
    ),
];

fn stop_loops() -> Vec<&'static str> {
    [
        &SHARED_LOOPS[..],
        &OWN_BODY_LOOPS[..],
        &TOWER_LOOPS[..],
        &SPECULATIVE_LOOPS[..],
    ]
    .concat()
}

fn crate_src() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

fn crate_loops() -> Vec<(String, Loop)> {
    loops_under(&crate_src())
}

/// Every loop of the crate that runs a forward, an evaluation, a callback or
/// an emission in each pass is on one of the lists, unless its shape bounds
/// it. A new loop is on no list, so this test fails until the loop is given a
/// stop or a bound.
#[test]
fn every_loop_that_does_work_reads_the_stop_or_has_a_stated_bound() {
    let mut listed = stop_loops();
    listed.extend(BOUNDED_LOOPS.iter().map(|(key, _)| *key));
    let (unlisted, lost) = unlisted_and_lost(&crate_loops(), &listed);
    assert_eq!(unlisted, Vec::<String>::new(), "loops on no list");
    assert_eq!(
        lost,
        Vec::<String>::new(),
        "listed loops that the scan lost"
    );
}

/// The scan reads a loop from the work in its body, and it does not know every
/// name of work. These fns hold a loop that it does not read, each with its
/// bound. The test holds only that each fn is still there.
#[test]
fn the_loops_that_the_scan_does_not_read_are_still_there() {
    let sources = sources_under(&crate_src());
    for (file, owner, _bound) in UNREAD_LOOPS {
        let declared = sources
            .iter()
            .any(|(f, code)| f == file && code.contains(&format!("fn {owner}(")));
        assert!(declared, "{file}::{owner}");
    }
}

/// The helper is called in the loops that read the stop and nowhere else in
/// the crate: none in a caller, and none in a loop that a round must run
/// whole.
#[test]
fn the_stop_helper_is_called_in_no_other_place() {
    let past = stop_reads_past(&sources_under(&crate_src()), &crate_loops(), &stop_loops());
    assert_eq!(past, 0);
}

/// The shared decode loop and the shared prefill read the stop by the rule:
/// the helper is the first statement of a pass, with `?`.
#[test]
fn the_shared_loops_read_the_stop() {
    assert_eq!(
        not_stopping(&crate_loops(), &SHARED_LOOPS),
        Vec::<String>::new()
    );
}

/// The decode loops and the prefill loops that one architecture has for itself
/// read the stop by the rule.
#[test]
fn the_own_body_loops_and_the_inline_prefills_read_the_stop() {
    assert_eq!(
        not_stopping(&crate_loops(), &OWN_BODY_LOOPS),
        Vec::<String>::new()
    );
}

/// The tower loop that evaluates in each pass reads the stop by the rule.
#[test]
fn the_tower_loop_reads_the_stop() {
    assert_eq!(
        not_stopping(&crate_loops(), &TOWER_LOOPS),
        Vec::<String>::new()
    );
}

/// The round loop and the prefills of the drafters read the stop by the rule.
/// For the round loop that is the head of a round, so a round commits whole
/// or does not start.
#[test]
fn the_round_loop_and_the_drafter_prefills_read_the_stop() {
    assert_eq!(
        not_stopping(&crate_loops(), &SPECULATIVE_LOOPS),
        Vec::<String>::new()
    );
}

// ── The flag of a request ───────────────────────────────────────────────────
//
// The source rule above reads that a loop calls the helper. It cannot read
// whose flag the loop holds: a loop handed a flag of its own stops for nobody.
// The rule here reads the use of the flag in every file that a stop touches.

/// The files that call a fn of a request with a flag that is never set, each
/// with its reason. They have no request, so they have no stop.
const FILES_WITHOUT_A_REQUEST: [(&str, &str); 2] = [
    ("ppl.rs", "the perplexity command has no request"),
    ("arch/loader.rs", "the smoke probe of a load has no request"),
];

fn is_without_a_request(file: &str) -> bool {
    FILES_WITHOUT_A_REQUEST.iter().any(|(f, _)| *f == file)
}

/// The files that the flag rule reads: each file that holds a loop of the stop
/// lists, each file that takes the flag, and each file that calls a fn that
/// takes it.
fn files_that_touch_the_flag(
    sources: &[(String, String)],
    shapes: &[FnShape],
) -> Vec<(String, String)> {
    let loop_files: Vec<&str> = stop_loops()
        .into_iter()
        .filter_map(|key| key.split_once("::").map(|(file, _)| file))
        .collect();
    sources
        .iter()
        .filter(|(file, code)| {
            loop_files.contains(&file.as_str())
                || fn_shapes(code).iter().any(|s| s.flag.is_some())
                || calls_a_flag_fn(code, shapes)
        })
        .cloned()
        .collect()
}

/// In every file that a stop touches, a flag parameter (a fn parameter of type
/// `&AtomicBool`, whatever its name) and the name `cancel` are used as the
/// argument of the stop helper and as the bare argument of a call, and in no
/// other way, and never in a pattern; a closure parameter `cancel` is refused
/// in the models crate, which has no job runner. Each call of a fn that takes a
/// flag hands on a bare flag of its caller at the place of the flag. So a fresh
/// flag, a `static`, a binding of the flag to another name, a pattern and a
/// read of it are all red. A file with no request is read for its uses and not
/// for its calls. Two fns that take the flag at two places under one name and
/// one count of parameters are refused, because a call cannot be matched to
/// either.
#[test]
fn the_flag_of_a_request_is_only_handed_on_and_read_by_the_helper() {
    let sources = sources_under(&crate_src());
    let shapes: Vec<FnShape> = sources
        .iter()
        .flat_map(|(_, code)| fn_shapes(code))
        .collect();
    assert_eq!(flag_fns_that_disagree(&shapes), Vec::<String>::new());
    let scope = files_that_touch_the_flag(&sources, &shapes);
    assert!(scope.len() > 10, "the scan reads {} files", scope.len());
    let mut wrong = Vec::new();
    for (file, code) in &scope {
        wrong.extend(
            flag_uses_off_the_rule(code)
                .into_iter()
                .map(|line| format!("{file}: {line}")),
        );
        if !is_without_a_request(file) {
            wrong.extend(
                calls_off_the_rule(code, &shapes)
                    .into_iter()
                    .map(|line| format!("{file}: {line}")),
            );
        }
    }
    assert_eq!(wrong, Vec::<String>::new());
    for (excepted, _reason) in FILES_WITHOUT_A_REQUEST {
        let applies = scope
            .iter()
            .any(|(file, code)| file == excepted && !calls_off_the_rule(code, &shapes).is_empty());
        assert!(
            applies,
            "{excepted} no longer calls a fn of a request with a flag of its own"
        );
    }
}

/// The source of every crate of the workspace, each file with its path from
/// `crates`.
#[allow(
    clippy::expect_used,
    reason = "a source tree that cannot be read is a test failure"
)]
fn workspace_sources() -> Vec<(String, String)> {
    let crates = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut dirs: Vec<std::path::PathBuf> = std::fs::read_dir(&crates)
        .expect("the crates directory")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.join("src").is_dir())
        .collect();
    dirs.sort();
    dirs.iter()
        .flat_map(|dir| {
            let name = dir
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            sources_under(&dir.join("src"))
                .into_iter()
                .map(move |(file, code)| (format!("{name}/src/{file}"), code))
        })
        .collect()
}

/// The stop helper is declared in one place in the workspace, and its name is
/// never the target of a rename on import and never bound by `let`, `const` or
/// `static`. A fn, a rename or a closure of the same name would give a loop a
/// read that stops for nobody, and every rule above would still pass.
#[test]
fn the_stop_helper_is_declared_in_one_place_and_never_shadowed() {
    let sources = workspace_sources();
    let declared: Vec<&str> = sources
        .iter()
        .filter(|(_, code)| fns_of(code).iter().any(|(name, _)| name == STOP_HELPER))
        .map(|(file, _)| file.as_str())
        .collect();
    assert_eq!(declared, ["rmlx-core/src/cancel.rs"]);
    let shadowed: Vec<String> = sources
        .iter()
        .flat_map(|(file, code)| {
            helper_shadowed(code)
                .into_iter()
                .map(move |line| format!("{file}: {line}"))
        })
        .collect();
    assert_eq!(shadowed, Vec::<String>::new());
}

/// A fn that enters the prefill state of its caches sweeps them before any `?`
/// and any `return` outside a closure. The result of the chunks is read after
/// the sweep: a stop and a failed chunk both end the prefill early, and the
/// sweep is owed on both. The sweep is a loop over the same caches as the entry
/// loop, in the block of the entry or in plain blocks inside it, not inside an
/// `if`, a `match` or another loop and with no adapter such as `take` on the
/// caches, and each entry has its own. The fns are read from the source: each
/// fn that calls `enter_prefill`. The shared `chunked_prefill` is one of them,
/// and its chunk loop is a closure, so the `?` of that loop ends the closure
/// and not the fn. An entry in a helper whose caller sweeps is not a shape of
/// the tree, and it is refused as an entry with no sweep.
#[test]
fn every_prefill_is_swept_before_its_result_is_read() {
    let mut entries = 0;
    let mut wrong = Vec::new();
    for (file, code) in sources_under(&crate_src()) {
        for (name, text) in fns_of(&code) {
            for sweep in prefill_sweeps(&text) {
                entries += 1;
                match sweep {
                    Err(reason) => wrong.push(format!("{file}::{name}: {reason}")),
                    Ok(stretch) if can_leave(&stretch) => {
                        wrong.push(format!(
                            "{file}::{name}: a `?` or a `return` before the sweep"
                        ));
                    }
                    Ok(_) => {}
                }
            }
        }
    }
    assert!(entries > 5, "the scan found {entries} prefill entries");
    assert_eq!(wrong, Vec::<String>::new());
}

/// The reader of the flag rule: the parameters of a fn, the uses of the flag,
/// and the calls that hand it on.
#[test]
fn the_flag_scan_reads_uses_and_calls() {
    assert_eq!(
        top_level_args("&mut c, f(a, b), [1, 2], |x| { y(x, z) }, cancel,"),
        [
            "&mut c",
            " f(a, b)",
            " [1, 2]",
            " |x| { y(x, z) }",
            " cancel"
        ]
    );

    let shape = |code: &str| fn_shapes(code);
    assert_eq!(
        shape("fn run<T>(&self, a: Map<K, V>, f: impl FnMut(u8) -> u8, stop: &AtomicBool) {}"),
        [FnShape {
            name: "run".to_owned(),
            params: 3,
            receiver: true,
            flag: Some((2, "stop".to_owned())),
        }]
    );
    for no_flag in [
        "fn run(a: u8, cancel: AtomicBool) {}",
        "fn run(a: u8, other: &mut AtomicBool) {}",
        "fn run(a: u8, other: &Arc<AtomicBool>) {}",
    ] {
        assert_eq!(
            shape(no_flag).first().map(|s| s.flag.clone()),
            Some(None),
            "{no_flag}"
        );
    }

    let inner = shape("fn inner(a: u8, cancel: &AtomicBool, b: u8) {}");
    let uses = |code: &str| flag_uses_off_the_rule(code).len();
    let calls = |code: &str| calls_off_the_rule(code, &inner).len();
    let fine = "fn run(a: u8, cancel: &AtomicBool) -> R {\n    \
                ensure_not_cancelled(cancel)?;\n    inner(a, cancel, b)\n}\n\
                use rmlx_core::cancel::ensure_not_cancelled;\n\
                fn job() {\n    run_admitted(guard, move |cancel| {\n        \
                inner(1, cancel, 2,)\n    })\n}\n\
                fn other(stop: &AtomicBool) {\n    inner(1, stop, 2);\n    \
                ensure_not_cancelled(stop)?;\n}";
    assert_eq!((uses(fine), calls(fine)), (0, 0));
    for (code, red_uses, red_calls) in [
        ("fn f(cancel: &AtomicBool) {\n    let c = cancel;\n}", 1, 0),
        (
            "fn f(cancel: &AtomicBool) {\n    if cancel.load(o) {}\n}",
            1,
            0,
        ),
        (
            "fn f(cancel: &AtomicBool) {\n    let x = || cancel.load(o);\n}",
            1,
            0,
        ),
        ("fn f(cancel: &AtomicBool) {\n    g(&cancel);\n}", 1, 0),
        (
            "fn f(cancel: &AtomicBool) {\n    g(cancel.clone());\n}",
            1,
            0,
        ),
        ("fn f(cancel: &AtomicBool) {\n    debug!(?cancel);\n}", 1, 0),
        (
            "fn f() {\n    let cancel = AtomicBool::new(false);\n}",
            1,
            0,
        ),
        ("fn f() {\n    let cancel: &AtomicBool = &S;\n}", 1, 0),
        ("struct S<'a> {\n    cancel: &'a AtomicBool,\n}", 1, 0),
        ("fn f() {\n    S { cancel }\n}", 1, 0),
        // A name of another flag parameter is read in its own fn.
        (
            "fn stopped(f: &AtomicBool) -> bool {\n    f.load(o)\n}",
            1,
            0,
        ),
        ("fn stopped(f: &AtomicBool) {\n    let g = f;\n}", 1, 0),
        // A closure parameter outside the runner, and a pattern.
        ("fn f() {\n    (|cancel| g(cancel))(&NEVER);\n}", 1, 0),
        ("fn f() {\n    let Flag(cancel) = Flag(&NEVER);\n}", 1, 0),
        ("fn f() {\n    if let Some(cancel) = x {}\n}", 1, 0),
        ("fn f() {\n    while let Some(cancel) = x {}\n}", 1, 0),
        (
            "fn f() {\n    match x {\n        Flag(cancel) => {}\n    }\n}",
            1,
            0,
        ),
        (
            "fn f() {\n    match x {\n        Some(cancel) if ok => {}\n    }\n}",
            1,
            0,
        ),
        ("fn f() {\n    for Flag(cancel) in v {}\n}", 1, 0),
        (
            "fn f(cancel: &AtomicBool) {\n    inner(a, &AtomicBool::new(false), b);\n}",
            0,
            1,
        ),
        (
            "fn f(cancel: &AtomicBool) {\n    inner(a, &STOP, b);\n}",
            0,
            1,
        ),
        (
            "fn f(cancel: &AtomicBool) {\n    inner(a, cancel, cancel);\n}",
            0,
            0,
        ),
        (
            "fn f(cancel: &AtomicBool) {\n    inner(cancel, a, b);\n}",
            0,
            1,
        ),
        (
            "fn f(cancel: &AtomicBool) {\n    ensure_not_cancelled(&STOP)?;\n}",
            0,
            1,
        ),
        ("fn f(x: u8) {\n    inner(a, x, b);\n}", 0, 1),
    ] {
        assert_eq!((uses(code), calls(code)), (red_uses, red_calls), "{code}");
    }
    // A call whose count of arguments matches no fn of its name is a failure
    // that names the line, the name and the counts, not a pass.
    let unmatched = calls_off_the_rule("fn f() {\n    inner(a, &STOP);\n}", &inner);
    assert_eq!(unmatched.len(), 1, "{unmatched:?}");
    assert!(
        unmatched.first().is_some_and(
            |m| m.starts_with("line 2: `inner` called with 2 arguments") && m.contains("[3]")
        ),
        "{unmatched:?}"
    );
    // A call of a name that a fn with no flag shares is not read.
    let shared = [inner.clone(), shape("fn inner(a: u8, b: u8, c: u8) {}")].concat();
    let shared_calls = |code: &str| calls_off_the_rule(code, &shared).len();
    assert_eq!(shared_calls("fn f() {\n    inner(a, &STOP, b);\n}"), 0);
}

/// A closure parameter list `|a, b|` is one piece of its argument, so a call
/// with such a closure has the arguments it reads, and the flag place of the
/// fn is found. A call that no fn of its name can take is a failure.
#[test]
fn a_closure_parameter_list_is_one_argument_and_an_unmatched_call_is_a_failure() {
    assert_eq!(
        top_level_args("a, |x, y| x + y, b"),
        ["a", " |x, y| x + y", " b"]
    );
    assert_eq!(
        top_level_args("move |row, temperature| picker.pick(row, temperature), cancel"),
        [
            "move |row, temperature| picker.pick(row, temperature)",
            " cancel"
        ]
    );
    assert_eq!(
        top_level_args("a, &mut |x, y| x + y, b"),
        ["a", " &mut |x, y| x + y", " b"]
    );
    assert_eq!(top_level_args("a, || f(1), b"), ["a", " || f(1)", " b"]);
    assert_eq!(top_level_args("a | b, c || d"), ["a | b", " c || d"]);
    let shapes = fn_shapes(
        "fn transcribe(a: u8, b: u8, pick: impl FnMut(&[f32], f32) -> u8, cancel: &AtomicBool, c: u8) {}",
    );
    let calls = |code: &str| calls_off_the_rule(code, &shapes);
    let call = |flag: &str| {
        format!(
            "fn f(cancel: &AtomicBool) {{\n    transcribe(a, b, &mut |row, temperature| picker.pick(row, temperature), {flag}, c);\n}}"
        )
    };
    assert_eq!(calls(&call("cancel")), Vec::<String>::new());
    assert_eq!(calls(&call("&STOP")).len(), 1);
    // A closure that the fn binds is not a fn of the same name: its calls are
    // not matched to fn shapes, and its body is still read as code of the fn.
    let tower = fn_shapes("fn forward(&self, a: u8, b: u8, c: u8, cancel: &AtomicBool) {}");
    let closure_calls = |body: &str, bind: &str| {
        calls_off_the_rule(
            &format!(
                "fn f(cancel: &AtomicBool) {{\n    let {bind}forward = {body};\n    forward()?;\n    forward()?;\n}}"
            ),
            &tower,
        )
    };
    for bind in ["", "mut "] {
        for body in [
            "|| { tower.forward(a, b, c, cancel) }",
            "move || { tower.forward(a, b, c, cancel) }",
        ] {
            assert_eq!(
                closure_calls(body, bind),
                Vec::<String>::new(),
                "{bind}{body}"
            );
        }
    }
    assert_eq!(closure_calls("|| { tower.forward(a, b, c) }", "").len(), 1);
    assert_eq!(
        closure_calls("|| { tower.forward(a, b, c, &STOP) }", "").len(),
        1
    );
    // Another fn's closure binding does not excuse a call in this fn.
    let other_fn = "fn g() {\n    let forward = || 1;\n}\nfn h() {\n    forward()?;\n}";
    assert_eq!(calls_off_the_rule(other_fn, &tower).len(), 1);
    // Too few arguments for every fn of the name: reported, with its counts.
    let short = calls("fn f(cancel: &AtomicBool) {\n    transcribe(a, cancel);\n}");
    assert_eq!(short.len(), 1, "{short:?}");
    assert!(
        short.first().is_some_and(|m| m
            .starts_with("line 2: `transcribe` called with 2 arguments")
            && m.contains("[5]")),
        "{short:?}"
    );
}

/// The reader of the flag rule for the calls of a method by its path, for two
/// fns that take the flag at two places, and for the shadows of the helper.
#[test]
fn the_flag_scan_reads_paths_pairs_and_shadows() {
    let shape = |code: &str| fn_shapes(code);
    // A method called by its path has the receiver as one argument more.
    let method = shape("impl S {\n    fn go(&self, a: u8, cancel: &AtomicBool) {}\n}");
    let method_calls = |code: &str| calls_off_the_rule(code, &method).len();
    assert_eq!(
        method_calls("fn f(cancel: &AtomicBool) {\n    me.go(1, cancel);\n}"),
        0
    );
    assert_eq!(
        method_calls("fn f(cancel: &AtomicBool) {\n    me.go(1, &STOP);\n}"),
        1
    );
    assert_eq!(
        method_calls("fn f(cancel: &AtomicBool) {\n    S::go(me, 1, cancel);\n}"),
        0
    );
    assert_eq!(
        method_calls("fn f(cancel: &AtomicBool) {\n    Self::go(me, 1, &STOP);\n}"),
        1
    );

    // Two fns of one name and count with the flag at two places.
    let apart = shape("fn go(a: u8, cancel: &AtomicBool) {}\nfn go(cancel: &AtomicBool, a: u8) {}");
    assert_eq!(flag_fns_that_disagree(&apart).len(), 1);
    let together =
        shape("fn go(a: u8, cancel: &AtomicBool) {}\nfn go(b: u8, stop: &AtomicBool) {}");
    assert_eq!(flag_fns_that_disagree(&together), Vec::<String>::new());

    // The helper under another name, a local closure of its name, a local fn.
    let shadows = |code: &str| helper_shadowed(code).len();
    assert_eq!(shadows("use rmlx_core::cancel::ensure_not_cancelled;"), 0);
    assert_eq!(shadows("use x::y as ensure_not_cancelled;"), 1);
    assert_eq!(
        shadows("let ensure_not_cancelled = |_: &AtomicBool| Ok(());"),
        1
    );
    assert_eq!(shadows("let mut ensure_not_cancelled = f;"), 1);
    assert_eq!(shadows("ensure_not_cancelled(cancel)?;"), 0);
}

/// The reader of the sweep rule: Ok(can leave before the sweep) or the reason
/// for the refusal.
#[test]
fn the_sweep_scan_reads_what_it_should() {
    let enter = "    for c in &mut caches {\n        c.enter_prefill();\n    }\n";
    let sweep = "    for c in &mut caches {\n        c.exit_prefill(d);\n    }\n";
    let read = |code: &str| -> Vec<std::result::Result<bool, &str>> {
        prefill_sweeps(code)
            .into_iter()
            .map(|s| s.map(|s| can_leave(&s)))
            .collect()
    };
    let swept = format!(
        "fn f() {{\n{enter}    let run = |c: &mut [K]| -> R {{\n        g()?;\n        \
         return Err(e);\n    }};\n    let r = run(c);\n{sweep}    r?\n}}"
    );
    assert_eq!(read(&swept), [Ok(false)]);
    let in_a_let_block =
        format!("fn f() {{\n{enter}    let x = {{\n    {sweep}    }};\n    r?\n}}");
    assert_eq!(read(&in_a_let_block), [Ok(false)]);
    let skipped = format!("fn f() {{\n{enter}    let r = run(c)?;\n{sweep}}}");
    assert_eq!(read(&skipped), [Ok(true)]);
    let returned =
        format!("fn f() {{\n{enter}    if bad {{\n        return Err(e);\n    }}\n{sweep}}}");
    assert_eq!(read(&returned), [Ok(true)]);
    let after_the_sweep = format!("fn f() {{\n{enter}{sweep}    run(c)?;\n}}");
    assert_eq!(read(&after_the_sweep), [Ok(false)]);
    let a_char = format!("fn f() {{\n{enter}    let q = '?';\n    let r = '\\'';\n{sweep}}}");
    assert_eq!(read(&a_char), [Ok(false)]);
    let no_sweep = format!("fn f() {{\n{enter}    run(c);\n}}");
    assert_eq!(read(&no_sweep), [Err("no sweep after the entry")]);
    let no_loop = "fn f() {\n    c.enter_prefill();\n    c.exit_prefill(d);\n}";
    assert_eq!(read(no_loop), [Err("the entry is not in a loop")]);
    let conditional = format!("fn f() {{\n{enter}    if r.is_ok() {{\n    {sweep}    }}\n}}");
    assert_eq!(
        read(&conditional),
        [Err("the sweep is inside a branch or a loop")]
    );
    let one_cache = "fn f() {\n    for c in &mut caches {\n        c.enter_prefill();\n    }\n    \
                     for c in caches.iter_mut().take(1) {\n        c.exit_prefill(d);\n    }\n}";
    assert_eq!(
        read(one_cache),
        [Err(
            "the sweep loop does not iterate the caches of the entry loop"
        )]
    );
    let sweep_in_an_if =
        format!("fn f() {{\n{enter}    for c in &mut caches {{\n        if ok {{\n            c.exit_prefill(d);\n        }}\n    }}\n}}");
    assert_eq!(
        read(&sweep_in_an_if),
        [Err("the sweep is not directly in a loop")]
    );
    let two_entries = format!("fn f() {{\n{enter}{enter}{sweep}}}");
    assert_eq!(
        read(&two_entries),
        [Err("another entry comes before the sweep"), Ok(false)]
    );
}

/// The scan finds a loop from the work in its body, whatever opens the loop,
/// and the stop rule reads the first statement of the body and nothing less.
#[test]
fn the_loop_scan_reads_work_and_the_stop_rule_reads_the_first_statement() {
    let found =
        |code: &str| -> Vec<String> { loops_of(code).into_iter().map(|l| l.owner).collect() };
    let emits = "fn decode() {\n    for i in 0..n {\n        step_fn(&step);\n    }\n}";
    let labelled = "fn fill() {\n    'c: loop {\n        m.forward_seq(ids)?;\n    }\n}";
    let labelled_while =
        "fn fill() {\n    'c: while more {\n        m.forward_seq(ids)?;\n    }\n}";
    let bound_to_a_name =
        "fn fill() {\n    let stop = loop {\n        row = m.forward(c)?;\n    };\n}";
    let bound_and_labelled =
        "fn fill() {\n    let stop = 'c: loop {\n        row = m.forward(c)?;\n    };\n}";
    let while_let =
        "fn fill() {\n    while let Some(c) = it.next() {\n        m.forward(c)?;\n    }\n}";
    let adaptor = "fn fill() {\n    chunks.try_for_each(|c| {\n        m.forward(c)?;\n    })\n}";
    let unknown_adaptor =
        "fn fill() {\n    chunks.scan_all(0, |s, c| {\n        m.forward(c)?;\n    })\n}";
    let callback = "fn frames(\n    step: &mut impl FnMut(usize) -> u8,\n) {\n    for i in 0..n {\n        step(i);\n    }\n}";
    let evaluates = "fn tower() {\n    for blk in &self.blocks {\n        h.eval()?;\n    }\n}";
    let once = "fn load() {\n    x.map_err(|e| {\n        m.forward(c)?;\n    })\n}";
    let before_a_loop = "fn first() {\n    step_fn(&step);\n    for c in caches {\n    }\n}";
    for code in [
        emits,
        labelled,
        labelled_while,
        bound_to_a_name,
        bound_and_labelled,
        while_let,
        adaptor,
        unknown_adaptor,
        callback,
        evaluates,
    ] {
        assert_eq!(found(code).len(), 1, "{code}");
    }
    assert_eq!(found(once), Vec::<String>::new());
    assert_eq!(found(before_a_loop), Vec::<String>::new());

    // A new loop is on no list, and a loop that the scan loses is reported. A
    // loop that replaces a listed one in the same fn is a new loop.
    let new: Vec<(String, Loop)> = loops_of(adaptor)
        .into_iter()
        .map(|l| ("new.rs".to_owned(), l))
        .collect();
    let (unlisted, lost) = unlisted_and_lost(&new, &["new.rs::fill: for c in chunks {"]);
    assert_eq!(unlisted, ["new.rs::fill: chunks.try_for_each(|c| {"]);
    assert_eq!(lost, ["new.rs::fill: for c in chunks {"]);

    // The shape bounds a loop over layers that only runs their forward, and a
    // loop over caches that only evaluates their state. It bounds no loop
    // that does other work, whatever its header names.
    let shape = |code: &str| {
        loops_of(code)
            .iter()
            .all(crate::stop_scan::bounded_by_shape)
    };
    let layers = "fn forward() {\n    for layer in &self.layers {\n        h = layer.forward(&h)?;\n    }\n}";
    let caches = "fn flush() {\n    for c in &caches {\n        c.eval_prefill_state()?;\n    }\n}";
    let prefill_over_blocks = "fn fill() {\n    for chunk in prompt_blocks.chunks(n) {\n        forward_chunk(chunk)?;\n        c.eval_prefill_state()?;\n    }\n}";
    let blocks_that_emit = "fn decode() {\n    for b in blocks {\n        m.forward(b)?;\n        step_fn(&s);\n    }\n}";
    let caches_that_forward = "fn fill() {\n    for c in &caches {\n        m.forward(c)?;\n        c.eval_prefill_state()?;\n    }\n}";
    assert!(shape(layers));
    assert!(shape(caches));
    for code in [
        evaluates,
        emits,
        prefill_over_blocks,
        blocks_that_emit,
        caches_that_forward,
    ] {
        assert!(!shape(code), "{code}");
    }

    let in_a_loop = |body: &str| format!("for i in 0..n {{\n    {body}\n    step_fn(&s);\n}}");
    let stops = |body: &str| stops_by_the_rule(&in_a_loop(body));
    assert!(stops("ensure_not_cancelled(cancel)?;"));
    assert!(stops("ensure_not_cancelled(\n        ctx.cancel,\n    )?;"));
    assert!(stops(
        "\n    rmlx_core::cancel::ensure_not_cancelled(cancel)?;"
    ));
    assert!(stops(
        "ensure_not_cancelled(cancel)?;\n    let n = other.load(o);"
    ));
    assert!(!stops("let _ = cancel;"));
    assert!(!stops("debug!(?cancel);"));
    assert!(!stops("if cancel.load(o) {\n        break;\n    }"));
    assert!(!stops("let _ = ensure_not_cancelled(cancel);"));
    assert!(!stops(
        "if ensure_not_cancelled(cancel).is_err() {\n        break;\n    }"
    ));
    assert!(!stops("my_ensure_not_cancelled(cancel)?;"));
    assert!(!stops("m.forward(x)?;\n    ensure_not_cancelled(cancel)?;"));
    assert!(!stops(
        "if late {\n        ensure_not_cancelled(cancel)?;\n    }"
    ));
    assert!(!stops(
        "if skip {\n        continue;\n    }\n    ensure_not_cancelled(cancel)?;"
    ));
    assert!(!stops(
        "ensure_not_cancelled(cancel)?;\n    ensure_not_cancelled(cancel)?;"
    ));
    assert!(!stops(
        "ensure_not_cancelled(cancel)?;\n    if cancel.load(o) {\n    }"
    ));
    assert_eq!(
        stop_reads(&in_a_loop("let _ = ensure_not_cancelled(cancel);")),
        (0, 1)
    );
    assert!(opens_with_the_stop_read(
        "'c: for chunk in ids\n    .chunks(4)\n{\n    ensure_not_cancelled(cancel)?;\n}"
    ));
}
