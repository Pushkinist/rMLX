//! A prefill issues no attention call in the configuration where the pinned
//! MLX returns non-finite rows under Metal device-memory shader validation.
//!
//! # The configuration
//!
//! MLX 0.32.3 routes `fast::scaled_dot_product_attention` at head dim 256 to
//! its head-dim-split kernel when the call has at least 1024 query rows and a
//! mask (`ScaledDotProductAttention::use_fallback`). That kernel is compiled
//! per call for `align_Q = (query rows % 64 == 0)` and
//! `align_K = (key rows % 32 == 0)` (`sdpa_full_self_attention_nax`). With an
//! array mask and both of them false it returns `+inf` rows under device-memory
//! validation. [`MEASURED_CELLS`] holds the cells that were measured in pure
//! MLX, and `common::attention_calls::is_faulty` is the statement they support.
//!
//! # The observable
//!
//! `rmlx_mlx::ATTENTION_CALL_TARGET`: one TRACE event for each attention call,
//! emitted where the call leaves `rmlx-mlx` for mlx-c. It reports the call MLX
//! receives. A rule above that point (a chunk split) and a rule at that point
//! (a split of one call) are both visible in it.
//!
//! # What cannot move
//!
//! - Decode: a decode step has one query row and is never in the
//!   configuration, so no decode call changes.
//! - The served tokens, except at a bf16 near-tie: another split of the prompt
//!   is the same arithmetic in another order. The tail-logit cells hold that
//!   against a reference split: the argmax at each position, then a limit on
//!   each logit that the rule-free splits set in the same run.
//!
//! # Mutations and the assertion that catches each
//!
//! | Mutation | Caught by |
//! |---|---|
//! | no rule at all (this tree) | the fresh, resumed and speculative cells of the Qwen3.5-family tests; the override cells of the gemma-4 test |
//! | the rule is applied in the fresh prefill only | the `resumed prefix` cells: gemma-4 here, and the Qwen3.5 hydrated tail in `src/qwen3_5_moe/tests.rs`, which needs a crate-private harness |
//! | the rule is applied in `generate_greedy` only | the `speculative prefill` cell |
//! | the rule is applied for one architecture only | the test of the other family (Qwen3.5 `kv_h > 1` dense, gemma-4 `kv_h == 1` shared KV) |
//! | the rule reads the default chunk and not the resolved one | the runtime-override cells |
//! | the rule aligns the query rows only when the prompt is short of one chunk | the cells past the second chunk boundary |
//! | the rule drops, repeats or shifts rows of the call it changes | the tail-logit cells: argmax at each position, the per-position limit, the first generated token |
//! | the event is removed, or reports another field under a name | `an_attention_call_reports_what_mlx_receives` and the positive control at the start of each GPU test |
//! | the oracle states another configuration (a block size, the query-row floor, the head dim, the mask kind, the device) | `the_oracle_agrees_with_every_measured_cell` |
//!
//! # What these tests cannot see
//!
//! - A vision or audio prefix, the server and the CLI: no cell drives them.
//!   They reach MLX through the same call, so the event covers them, but no
//!   test here issues their prefill.
//! - A machine without the NAX kernels: the oracle does not ask for them, so
//!   it is stricter than MLX's route there.
//! - Whether the kernel is correct: the oracle is a list of measured cells,
//!   not a proof.
//!
//! The GPU tests are `#[ignore]`d: they load a model and need the Metal
//! context. Snapshots resolve from `RMLX_O_MODELS_ROOT` by slug (see
//! `tests/common/mod.rs`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss
)]

mod common;

use common::attention_calls::{
    is_faulty, recorded, AttentionCall, KEY_BLOCK, QUERY_BLOCK, SPLIT_KERNEL_HEAD_DIM,
    SPLIT_KERNEL_MIN_QUERY_ROWS,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_mlx::{Array, Device, Dtype};
use rmlx_models::arch::{self, Architecture};
use rmlx_models::prefill_chunk::{module_key_for_class, resolve, set_prefill_chunk};
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

// ---------------------------------------------------------------------------
// The oracle against the measurement
// ---------------------------------------------------------------------------

/// `(query rows, key rows, head dim, query heads, kv heads, mask mode, bad
/// repeats, repeats)` — pure-MLX probe cells on mlx 0.32.3 under device-memory
/// validation, bf16, on the GPU, with the Metal library of the source build
/// that `docs/MLX_PAIR.md` describes. A cell is "measured bad" when at least
/// one repeat returned a non-finite cell. With validation off, every cell that
/// was run returned one digest and no non-finite cell.
///
/// A clean cell is a bound and not a proof: 0 of 24 bounds the rate of one cell
/// near 12 %. The cells with one row count aligned pool to 0 of 224.
const MEASURED_CELLS: [(i64, i64, i64, i64, i64, &str, u32, u32); 21] = [
    (2012, 2012, 256, 8, 2, "array", 11, 24),
    (1806, 3854, 256, 16, 2, "array", 9, 16),
    (1025, 3073, 256, 16, 2, "array", 8, 16),
    (1296, 1296, 256, 8, 1, "array", 6, 24),
    (1100, 3148, 256, 8, 1, "array", 2, 16),
    // Query rows a multiple of 32 and not of 64.
    (1056, 3089, 256, 16, 2, "array", 7, 16),
    // Key rows aligned, query rows not.
    (2012, 2048, 256, 8, 2, "array", 0, 24),
    (2016, 2016, 256, 8, 2, "array", 0, 32),
    (1806, 3840, 256, 16, 2, "array", 0, 24),
    (1025, 3072, 256, 16, 2, "array", 0, 24),
    (1296, 1312, 256, 8, 1, "array", 0, 24),
    // Query rows aligned, key rows not.
    (2048, 2012, 256, 8, 2, "array", 0, 32),
    (1984, 2012, 256, 8, 2, "array", 0, 32),
    (1856, 3854, 256, 16, 2, "array", 0, 32),
    // Both aligned.
    (2048, 2048, 256, 8, 2, "array", 0, 24),
    (1792, 3840, 256, 16, 2, "array", 0, 24),
    // No array mask.
    (2012, 2012, 256, 8, 2, "causal", 0, 32),
    (2012, 2048, 256, 8, 2, "causal", 0, 32),
    // Under the route's query-row floor.
    (1023, 3071, 256, 16, 2, "array", 0, 32),
    (1001, 3049, 256, 16, 2, "array", 0, 32),
    // Another head dim: not the split kernel.
    (2012, 2012, 128, 32, 8, "array", 0, 16),
];

#[test]
fn the_oracle_agrees_with_every_measured_cell() {
    let mut bad_cells = 0;
    for (q_rows, k_rows, head_dim, q_heads, kv_heads, mask, bad, repeats) in MEASURED_CELLS {
        let call = AttentionCall {
            q_heads,
            q_rows,
            head_dim,
            kv_heads,
            k_rows,
            v_head_dim: head_dim,
            dtype: "Bf16".to_owned(),
            mask: mask.to_owned(),
            device: "Gpu".to_owned(),
        };
        assert_eq!(
            is_faulty(&call),
            bad > 0,
            "the oracle disagrees with the measurement ({bad} bad of {repeats}) at {call:?}"
        );
        bad_cells += u32::from(bad > 0);
    }
    assert_eq!(bad_cells, 6, "the table must hold both verdicts");

    // The same shape on the CPU never reaches a Metal kernel.
    let on_cpu = AttentionCall {
        q_heads: 8,
        q_rows: 2012,
        head_dim: 256,
        kv_heads: 2,
        k_rows: 2012,
        v_head_dim: 256,
        dtype: "Bf16".to_owned(),
        mask: "array".to_owned(),
        device: "Cpu".to_owned(),
    };
    assert!(!is_faulty(&on_cpu));
}

#[test]
fn an_attention_call_reports_what_mlx_receives() {
    let device = Device::Cpu;
    let ((), calls) = recorded(|| {
        let q = rmlx_mlx::zeros(&[1, 2, 5, 8], Dtype::F32, device).expect("q");
        let k = rmlx_mlx::zeros(&[1, 1, 7, 8], Dtype::F32, device).expect("k");
        let v = rmlx_mlx::zeros(&[1, 1, 7, 8], Dtype::F32, device).expect("v");
        let mask = rmlx_mlx::zeros(&[1, 1, 5, 7], Dtype::F32, device).expect("mask");
        rmlx_mlx::scaled_dot_product_attention(&q, &k, &v, 1.0, "array", Some(&mask), device)
            .expect("array-mask call");
        rmlx_mlx::scaled_dot_product_attention(&q, &k, &v, 1.0, "", None, device)
            .expect("unmasked call");
    });
    let expected = |mask: &str| AttentionCall {
        q_heads: 2,
        q_rows: 5,
        head_dim: 8,
        kv_heads: 1,
        k_rows: 7,
        v_head_dim: 8,
        dtype: "F32".to_owned(),
        mask: mask.to_owned(),
        device: "Cpu".to_owned(),
    };
    assert_eq!(calls, vec![expected("array"), expected("")]);
}

// ---------------------------------------------------------------------------
// GPU harness
// ---------------------------------------------------------------------------

const DEVICE: Device = Device::Gpu;

/// Context ceiling of every generation and cache here: above the longest
/// prompt a cell builds.
const MAX_CTX: i32 = 8192;

/// A chunk the adaptive prefill controller reaches from a default of 256
/// (256, 384, 576, 864, 1296) and installs for every architecture of the
/// process. It is at least 1024 rows and not a multiple of 64.
const ADAPTIVE_CHUNK: usize = 1296;

/// A chunk under which no call is in the configuration: a whole chunk is 1024
/// rows, a multiple of 64, and a last chunk is shorter than the route's floor.
/// It is the reference split of the tail-logit cells.
const REFERENCE_CHUNK: usize = 1024;

/// Installs a process-wide prefill chunk, and clears it when dropped.
struct ChunkOverride;

impl ChunkOverride {
    fn install(chunk: usize) -> Self {
        set_prefill_chunk(chunk);
        Self
    }
}

impl Drop for ChunkOverride {
    fn drop(&mut self) {
        set_prefill_chunk(0);
    }
}

struct Loaded {
    model: Architecture,
    tokenizer: tokenizers::Tokenizer,
    /// Token ids of one passage, long enough for every cell.
    passage: Vec<u32>,
    /// Token ids of another passage: the replaced token of the logit control.
    other_passage: Vec<u32>,
}

fn load(model: &common::GoldenModel, test: &str) -> Option<Loaded> {
    let path = common::model_for(model, test)?;
    let loaded =
        arch::load_model(&path, DEVICE, &arch::LoadOpts::default()).expect("arch::load_model");
    let tokenizer =
        tokenizers::Tokenizer::from_file(path.join("tokenizer.json")).expect("tokenizer.json");
    let encode = |sentence: &str| -> Vec<u32> {
        tokenizer
            .encode(sentence.repeat(600), false)
            .expect("tokenize")
            .get_ids()
            .to_vec()
    };
    let passage = encode(
        "A cartographer walked the ridge at dawn, tracing every river and switchback onto \
         oiled linen. ",
    );
    let other_passage = encode(
        "The harbour master counted nine ships before the fog came in from the north and \
         closed the channel. ",
    );
    assert!(
        passage.len() >= 6000 && other_passage.len() >= 2000,
        "the passages are too short for the cells: {} and {} tokens",
        passage.len(),
        other_passage.len()
    );
    Some(Loaded {
        model: loaded,
        tokenizer,
        passage,
        other_passage,
    })
}

/// The chunk production resolves for this model now, the runtime override
/// included.
fn resolved_chunk(model: &Architecture) -> usize {
    resolve(module_key_for_class(model.arch_class())).0
}

/// One greedy generation of one token through the entry the CLI and the server
/// call: its step, and the attention calls it issued.
fn generate_one(
    loaded: &Loaded,
    prompt: &[u32],
    cache_slots: usize,
    top_logprobs_k: u32,
) -> (Vec<rmlx_models::ProbeStep>, Vec<AttentionCall>) {
    let sampler_cfg = SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k,
    };
    let mut rng = Pcg32::new(sampler_cfg.seed_or_default());
    let penalty_cfg = PenaltyConfig::default();
    let mut token_history: Vec<u32> = Vec::new();
    let (steps, calls) = recorded(|| {
        loaded
            .model
            .generate_greedy(
                &loaded.tokenizer,
                prompt,
                1,
                DEVICE,
                None,
                Some(MAX_CTX),
                cache_slots,
                &[],
                &mut |_| None,
                None,
                &sampler_cfg,
                &mut rng,
                &penalty_cfg,
                &mut token_history,
            )
            .expect("generate_greedy")
    });
    assert_eq!(steps.len(), 1, "the generation must emit its one token");
    (steps, calls)
}

fn generate_calls(loaded: &Loaded, prompt: &[u32], cache_slots: usize) -> Vec<AttentionCall> {
    generate_one(loaded, prompt, cache_slots, 0).1
}

fn fresh_caches(model: &Architecture) -> (Vec<KvCache>, Option<Vec<LinearAttnCache>>) {
    let layers = model.num_hidden_layers();
    let kv = (0..layers)
        .map(|i| {
            KvCache::with_quant_max_seq_window(
                KvQuant::None,
                MAX_CTX,
                model.layer_sliding_window(i),
            )
            .with_shares_kv(model.shares_kv_across_layers())
        })
        .collect();
    let lin = model
        .needs_lin_caches()
        .then(|| (0..layers).map(|_| LinearAttnCache::new()).collect());
    (kv, lin)
}

/// The calls of one fresh prefill through the speculative paths' entry.
fn speculative_prefill_calls(model: &Architecture, prompt: &[u32]) -> Vec<AttentionCall> {
    let ((), calls) = recorded(|| {
        let (mut kv, mut lin) = fresh_caches(model);
        rmlx_models::speculative::prefill_chunked(
            model,
            prompt,
            &mut kv,
            lin.as_deref_mut(),
            DEVICE,
        )
        .expect("speculative prefill");
    });
    calls
}

/// The distinct `(query rows, key rows)` of a run, in the order they first
/// occur.
fn distinct_shapes(calls: &[AttentionCall]) -> Vec<(i64, i64)> {
    let mut shapes = Vec::new();
    for call in calls {
        let shape = (call.q_rows, call.k_rows);
        if !shapes.contains(&shape) {
            shapes.push(shape);
        }
    }
    shapes
}

/// Positive control, in the same process as the cells: a call in the
/// configuration, built and never evaluated, reaches the recorder and the
/// oracle flags it.
fn assert_the_recorder_and_the_oracle_see_a_faulty_call() {
    let ((), calls) = recorded(|| {
        let q = rmlx_mlx::zeros(&[1, 8, 1025, 256], Dtype::Bf16, DEVICE).expect("q");
        let kv = rmlx_mlx::zeros(&[1, 1, 3073, 256], Dtype::Bf16, DEVICE).expect("kv");
        let mask = rmlx_mlx::zeros(&[1, 1, 1025, 3073], Dtype::Bf16, DEVICE).expect("mask");
        let _lazy: Array =
            rmlx_mlx::scaled_dot_product_attention(&q, &kv, &kv, 1.0, "array", Some(&mask), DEVICE)
                .expect("lazy call");
    });
    assert_eq!(calls.len(), 1, "positive control: one call, one event");
    assert!(
        is_faulty(&calls[0]),
        "positive control: the oracle must flag {:?}",
        calls[0]
    );
}

/// Collects every cell that reached the configuration, so one run reports all
/// of them.
#[derive(Default)]
struct Report {
    cells: usize,
    offenders: Vec<String>,
}

impl Report {
    /// `all_keys` is the prompt length. The cell must show a call over the whole
    /// prompt: without that, a recorder that saw another prefill, or none, would
    /// pass.
    fn check(&mut self, cell: &str, calls: &[AttentionCall], all_keys: usize) {
        self.cells += 1;
        assert!(
            calls.iter().any(|c| c.k_rows == all_keys as i64),
            "{cell}: no attention call covered the whole prompt of {all_keys} tokens"
        );
        let faulty: Vec<&AttentionCall> = calls.iter().filter(|c| is_faulty(c)).collect();
        if let Some(first) = faulty.first() {
            self.offenders.push(format!(
                "{cell}: {} of {} calls, first q_rows={} k_rows={} head_dim={} \
                 q_heads={} kv_heads={} mask={}",
                faulty.len(),
                calls.len(),
                first.q_rows,
                first.k_rows,
                first.head_dim,
                first.q_heads,
                first.kv_heads,
                first.mask
            ));
        }
    }

    fn finish(self, test: &str) {
        println!(
            "[{test}] {} cells, {} reached the configuration",
            self.cells,
            self.offenders.len()
        );
        assert!(
            self.offenders.is_empty(),
            "a prefill issued an attention call in the configuration where the pinned MLX \
             returns non-finite rows under device-memory validation (head dim \
             {SPLIT_KERNEL_HEAD_DIM}, array mask, at least {SPLIT_KERNEL_MIN_QUERY_ROWS} query \
             rows, query rows not a multiple of {QUERY_BLOCK}, key rows not a multiple of \
             {KEY_BLOCK}):\n  {}",
            self.offenders.join("\n  ")
        );
    }
}

/// Prompt lengths on both sides of the first two chunk boundaries, and tails
/// on both sides of the route's query-row floor.
fn lengths_across_the_chunk_boundaries(chunk: usize) -> Vec<usize> {
    let floor = SPLIT_KERNEL_MIN_QUERY_ROWS as usize;
    let mut lengths = vec![
        chunk - 1,
        chunk,
        chunk + 1,
        chunk + floor - 1,
        chunk + floor,
        chunk + floor + 1,
        chunk + 1806,
        2 * chunk - 1,
        2 * chunk + floor + 1,
    ];
    lengths.sort_unstable();
    lengths.dedup();
    lengths
}

/// Fresh prefills at the chunk production resolves now.
fn check_fresh_prefills(loaded: &Loaded, label: &str, report: &mut Report) {
    let chunk = resolved_chunk(&loaded.model);
    for len in lengths_across_the_chunk_boundaries(chunk) {
        loaded.model.clear_prompt_cache();
        let calls = generate_calls(loaded, &loaded.passage[..len], 0);
        report.check(
            &format!("{label} fresh prefill, chunk {chunk}, {len} tokens"),
            &calls,
            len,
        );
    }
}

/// A prompt that extends a stored prompt by `tail` tokens, as the next turn of
/// a conversation does. An extension is the one resume every architecture
/// takes: gemma-4 does not truncate a stored prompt once a sliding-window ring
/// has wrapped.
///
/// A miss prefills the whole prompt and agrees with a fresh prefill for free,
/// so the cell asserts the branch it reached: the cache reports a hit, and no
/// prefill call of the second request starts at key row zero.
fn check_resumed_prefix(loaded: &Loaded, label: &str, tail: usize, report: &mut Report) {
    const SHARED: usize = 2048;
    const SLOTS: usize = 4;
    let chunk = resolved_chunk(&loaded.model);
    let stats = || {
        loaded
            .model
            .cache_stats()
            .map_or((0, 0), |s| (s.hits, s.misses))
    };

    loaded.model.clear_prompt_cache();
    let _ = generate_calls(loaded, &loaded.passage[..SHARED], SLOTS);

    let second = &loaded.passage[..SHARED + tail];
    let (hits_before, _) = stats();
    let calls = generate_calls(loaded, second, SLOTS);
    let (hits_after, _) = stats();
    assert_eq!(
        hits_after - hits_before,
        1,
        "{label}: the second request must find the stored prefix"
    );
    let resumed_at = calls.iter().map(|c| c.k_rows - c.q_rows).min().unwrap_or(0);
    assert!(
        resumed_at > 0,
        "{label}: a call of the second request starts at key row zero, so it did not \
         resume the stored prefix; (query rows, key rows) of its calls: {:?}",
        distinct_shapes(&calls)
    );
    report.check(
        &format!("{label} resumed prefix, chunk {chunk}, tail of {tail} tokens"),
        &calls,
        second.len(),
    );
    loaded.model.clear_prompt_cache();
}

/// The speculative paths' prefill, one chunk and a tail past the route's floor.
fn check_speculative_prefill(loaded: &Loaded, label: &str, report: &mut Report) {
    let chunk = resolved_chunk(&loaded.model);
    let len = chunk + SPLIT_KERNEL_MIN_QUERY_ROWS as usize + 1;
    let calls = speculative_prefill_calls(&loaded.model, &loaded.passage[..len]);
    report.check(
        &format!("{label} speculative prefill, chunk {chunk}, {len} tokens"),
        &calls,
        len,
    );
}

/// The prefill paths one architecture has beside the fresh prefill.
#[derive(Clone, Copy)]
struct Paths {
    /// `speculative::prefill_chunked` serves this architecture as a verifier.
    speculative: bool,
    /// The prompt cache resumes a stored prompt that the request extends. The
    /// other architectures reuse a RAM entry on an exact match only.
    resumes_an_extended_prompt: bool,
}

/// Every cell of one model: at the default chunk, then at a chunk the adaptive
/// controller installs.
fn check_every_prefill_path(loaded: &Loaded, test: &str, paths: Paths) {
    assert_the_recorder_and_the_oracle_see_a_faulty_call();
    let mut report = Report::default();

    for (label, chunk, tail) in [
        ("default:", None, 1025),
        ("override:", Some(ADAPTIVE_CHUNK), 1100),
    ] {
        let _chunk = chunk.map(ChunkOverride::install);
        check_fresh_prefills(loaded, label, &mut report);
        if paths.resumes_an_extended_prompt {
            check_resumed_prefix(loaded, label, tail, &mut report);
        }
        if paths.speculative {
            check_speculative_prefill(loaded, label, &mut report);
        }
    }
    report.finish(test);
}

// ---------------------------------------------------------------------------
// The seam cells
// ---------------------------------------------------------------------------

const QWEN3_6_MOE: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__Qwen3.6-35B-A3B-8bit",
    archs: &["Qwen3_5MoeForConditionalGeneration"],
};

const QWEN3_5_DENSE: common::GoldenModel = common::GoldenModel {
    slug: "prism-ml__Ternary-Bonsai-27B-mlx-2bit",
    archs: &["Qwen3_5ForConditionalGeneration"],
};

const GEMMA4_E2B: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__gemma-4-e2b-it-mxfp8",
    archs: &["Gemma4ForConditionalGeneration"],
};

const BONSAI_8B: common::GoldenModel = common::GoldenModel {
    slug: "prism-ml__Ternary-Bonsai-8B-mlx-2bit",
    archs: &["Qwen3ForCausalLM"],
};

/// Head dim 256, 16 query heads over 2 KV heads, no KV sharing, chunk 2048.
#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn qwen3_6_moe_prefill_stays_off_the_faulty_attention_configuration() {
    const TEST: &str = "qwen3_6_moe_prefill_stays_off_the_faulty_attention_configuration";
    let Some(loaded) = load(&QWEN3_6_MOE, TEST) else {
        return;
    };
    assert_eq!(loaded.model.head_dim(), 256);
    assert!(loaded.model.num_key_value_heads() > 1);
    check_every_prefill_path(
        &loaded,
        TEST,
        Paths {
            speculative: true,
            resumes_an_extended_prompt: false,
        },
    );
}

/// Head dim 256, 24 query heads over 4 KV heads, dense, no KV sharing.
#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn qwen3_5_dense_prefill_stays_off_the_faulty_attention_configuration() {
    const TEST: &str = "qwen3_5_dense_prefill_stays_off_the_faulty_attention_configuration";
    let Some(loaded) = load(&QWEN3_5_DENSE, TEST) else {
        return;
    };
    assert_eq!(loaded.model.head_dim(), 256);
    assert!(loaded.model.num_key_value_heads() > 1);
    check_every_prefill_path(
        &loaded,
        TEST,
        Paths {
            speculative: true,
            resumes_an_extended_prompt: false,
        },
    );
}

/// Head dim 256 on the sliding-window layers, one KV head, KV shared across
/// layers. The sliding-window mask is an array at every offset.
#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn gemma4_e2b_prefill_stays_off_the_faulty_attention_configuration() {
    const TEST: &str = "gemma4_e2b_prefill_stays_off_the_faulty_attention_configuration";
    let Some(loaded) = load(&GEMMA4_E2B, TEST) else {
        return;
    };
    assert_eq!(loaded.model.num_key_value_heads(), 1);
    assert!(loaded.model.shares_kv_across_layers());
    check_every_prefill_path(
        &loaded,
        TEST,
        Paths {
            speculative: true,
            resumes_an_extended_prompt: true,
        },
    );
}

/// Head dim 128: no call of this model is on the split kernel, so this test is
/// a guard that the rule is stated by shape and not a cell that can show the
/// defect.
#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn bonsai_8b_prefill_stays_off_the_faulty_attention_configuration() {
    const TEST: &str = "bonsai_8b_prefill_stays_off_the_faulty_attention_configuration";
    let Some(loaded) = load(&BONSAI_8B, TEST) else {
        return;
    };
    assert_eq!(loaded.model.head_dim(), 128);
    assert!(loaded.model.num_key_value_heads() > 1);
    check_every_prefill_path(
        &loaded,
        TEST,
        Paths {
            speculative: false,
            resumes_an_extended_prompt: false,
        },
    );
}

// ---------------------------------------------------------------------------
// The tail-logit cells
// ---------------------------------------------------------------------------

/// Positions judged after the prefill. Eight query rows keep the judging
/// forward itself off the full-attention kernels.
const TAIL: usize = 8;

/// Two more chunks under which no call is in the configuration: every chunk is
/// shorter than the route's floor. Each against [`REFERENCE_CHUNK`] measures,
/// in the same run, how far a change of split alone moves a logit.
const NOISE_CHUNKS: [usize; 2] = [512, 768];

/// The production split may move a logit, at one tail position, no further
/// than `NOISE_FACTOR` times what a [`NOISE_CHUNKS`] split moves it there, plus
/// `NOISE_FLOOR`.
///
/// One absolute bound does not hold across the models. Measured with no rule
/// in place and no instrument, largest difference at one position over the
/// whole vocabulary, each against chunk 1024:
///
/// | Model, prefill | production split | chunk 512 | chunk 768 | one token replaced |
/// |---|---|---|---|---|
/// | Qwen3.6-35B-A3B-8bit, 3073 | 1.07 to 4.46 | 1.16 to 5.51 | 1.19 to 4.20 | 6.14 to 11.53 |
/// | Ternary-Bonsai-27B 2-bit, 3073 | 0.08 to 0.61 | 0.09 to 0.59 | 0.09 to 0.59 | 2.56 to 6.19 |
/// | gemma-4-e2b mxfp8, 2321 | 0.54 to 1.60 | 0.42 to 1.47 | 0.30 to 1.57 | 2.07 to 6.99 |
///
/// The position where a split moves a logit most is the same position for
/// every split, so the limit is taken per position. The largest ratio of the
/// production split to the larger noise split at one position is 3.0, at 0.375
/// against 0.125, which is why the limit has a floor. A prompt with one token
/// replaced 40 rows before the end of the prefill exceeds the limit at every
/// position on each model; at one position of Qwen3.6 by 0.02.
const NOISE_FACTOR: f32 = 2.0;
const NOISE_FLOOR: f32 = 0.5;

fn logit_rows(logits: &Array, rows: usize) -> Vec<Vec<f32>> {
    let f32_logits = logits.astype(Dtype::F32, DEVICE).expect("astype f32");
    Array::eval(&f32_logits).expect("materialise logits");
    let flat: Vec<f32> = f32_logits
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let vocab = flat.len() / rows;
    flat.chunks_exact(vocab).map(<[f32]>::to_vec).collect()
}

/// Prefill `ids[..prefill_len]` at the chunk production resolves now, then
/// return the logits of the next [`TAIL`] positions.
fn tail_rows_after_prefill(model: &Architecture, ids: &[u32], prefill_len: usize) -> Vec<Vec<f32>> {
    let (mut kv, mut lin) = fresh_caches(model);
    rmlx_models::speculative::prefill_chunked(
        model,
        &ids[..prefill_len],
        &mut kv,
        lin.as_deref_mut(),
        DEVICE,
    )
    .expect("prefill");
    let logits = model
        .forward_seq_last_k_with_cache(
            &ids[prefill_len..prefill_len + TAIL],
            TAIL,
            &mut kv,
            lin.as_deref_mut(),
            DEVICE,
        )
        .expect("tail forward");
    let rows = logit_rows(&logits, TAIL);
    assert!(
        rows.iter().flatten().all(|x| x.is_finite()),
        "the tail logits hold a non-finite cell"
    );
    rows
}

/// `(argmax, top-1 minus top-2)` of one row.
fn argmax_and_margin(row: &[f32]) -> (usize, f32) {
    let mut best = (0usize, f32::NEG_INFINITY);
    let mut second = f32::NEG_INFINITY;
    for (i, &x) in row.iter().enumerate() {
        if x > best.1 {
            second = best.1;
            best = (i, x);
        } else if x > second {
            second = x;
        }
    }
    (best.0, best.1 - second)
}

/// The largest difference at each tail position.
fn moved_per_position(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<f32> {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            x.iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0.0f32, f32::max)
        })
        .collect()
}

/// The split production uses for a prompt whose last chunk is in the
/// configuration without a rule, against [`REFERENCE_CHUNK`].
///
/// `production_chunk` is `None` for the architecture's default and `Some` for
/// a chunk the adaptive controller installs.
fn check_tail_logits_against_the_reference_split(
    loaded: &Loaded,
    test: &str,
    production_chunk: Option<usize>,
) {
    let model = &loaded.model;
    let rows_at = |chunk: Option<usize>, ids: &[u32], prefill_len: usize| {
        let _chunk = chunk.map(ChunkOverride::install);
        (
            resolved_chunk(model),
            tail_rows_after_prefill(model, ids, prefill_len),
        )
    };

    let production = production_chunk.unwrap_or_else(|| resolved_chunk(model));
    assert!(
        production != REFERENCE_CHUNK && !NOISE_CHUNKS.contains(&production),
        "the production split must differ from the reference and noise splits"
    );
    let prefill_len = production + SPLIT_KERNEL_MIN_QUERY_ROWS as usize + 1;
    let ids = &loaded.passage[..prefill_len + TAIL];

    let (used, production_rows) = rows_at(production_chunk, ids, prefill_len);
    assert_eq!(used, production);
    let (used, reference_rows) = rows_at(Some(REFERENCE_CHUNK), ids, prefill_len);
    assert_eq!(used, REFERENCE_CHUNK);

    let mut limit = vec![0.0f32; TAIL];
    for chunk in NOISE_CHUNKS {
        let (_, noise_rows) = rows_at(Some(chunk), ids, prefill_len);
        for (l, n) in limit
            .iter_mut()
            .zip(moved_per_position(&noise_rows, &reference_rows))
        {
            *l = l.max(n);
        }
    }
    for l in &mut limit {
        *l = NOISE_FACTOR * *l + NOISE_FLOOR;
    }

    // Control: the reference split of a prompt that differs in one token of
    // the last chunk. The limit must separate it from the production split.
    let mut one_token_off = ids.to_vec();
    let at = prefill_len - 40;
    let other = &loaded.other_passage;
    one_token_off[at] = if other[0] == ids[at] {
        other[1]
    } else {
        other[0]
    };
    let (_, control_rows) = rows_at(Some(REFERENCE_CHUNK), &one_token_off, prefill_len);

    let moved = moved_per_position(&production_rows, &reference_rows);
    let control = moved_per_position(&control_rows, &reference_rows);
    println!(
        "[{test}] production chunk {production}, reference chunk {REFERENCE_CHUNK}, prefill \
         {prefill_len} tokens\n  production split: {moved:?}\n  limit:            {limit:?}\n  \
         one token off:    {control:?}"
    );

    for (pos, (p, r)) in production_rows.iter().zip(&reference_rows).enumerate() {
        let (p_arg, p_margin) = argmax_and_margin(p);
        let (r_arg, r_margin) = argmax_and_margin(r);
        assert_eq!(
            p_arg, r_arg,
            "{test}: the argmax at tail position {pos} differs between the production split \
             and the reference split (top-2 margin: production {p_margin}, reference \
             {r_margin}; largest logit difference there {})",
            moved[pos]
        );
        assert!(
            moved[pos] <= limit[pos],
            "{test}: the production split moves a logit at tail position {pos} by {}, beyond \
             the limit {} that the rule-free splits set",
            moved[pos],
            limit[pos]
        );
    }
    let separated = control.iter().zip(&limit).filter(|(c, l)| c > l).count();
    assert!(
        separated * 2 > TAIL,
        "{test}: the limit does not separate a prompt with one token replaced: it exceeds \
         the limit at {separated} of {TAIL} positions"
    );

    // The prefill's own last row, through the entry the CLI and the server
    // call: the first token of a fresh generation.
    let first_token = |chunk: Option<usize>| {
        let _chunk = chunk.map(ChunkOverride::install);
        model.clear_prompt_cache();
        let (steps, _) = generate_one(loaded, &ids[..prefill_len], 0, 2);
        let top = &steps[0].logprobs.as_ref().expect("top logprobs").top;
        (steps[0].token_id, top[0].1 - top[1].1)
    };
    let (production_token, production_margin) = first_token(production_chunk);
    let (reference_token, reference_margin) = first_token(Some(REFERENCE_CHUNK));
    println!(
        "  first token: production {production_token} (top-2 margin {production_margin}), \
         reference {reference_token} (top-2 margin {reference_margin})"
    );
    assert_eq!(
        production_token, reference_token,
        "{test}: the first generated token differs between the production split and the \
         reference split (top-2 logprob margin: production {production_margin}, reference \
         {reference_margin})"
    );
}

#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn qwen3_6_moe_tail_logits_hold_across_the_prefill_split() {
    const TEST: &str = "qwen3_6_moe_tail_logits_hold_across_the_prefill_split";
    let Some(loaded) = load(&QWEN3_6_MOE, TEST) else {
        return;
    };
    check_tail_logits_against_the_reference_split(&loaded, TEST, None);
}

#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn qwen3_5_dense_tail_logits_hold_across_the_prefill_split() {
    const TEST: &str = "qwen3_5_dense_tail_logits_hold_across_the_prefill_split";
    let Some(loaded) = load(&QWEN3_5_DENSE, TEST) else {
        return;
    };
    check_tail_logits_against_the_reference_split(&loaded, TEST, None);
}

#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn gemma4_e2b_tail_logits_hold_across_the_prefill_split() {
    const TEST: &str = "gemma4_e2b_tail_logits_hold_across_the_prefill_split";
    let Some(loaded) = load(&GEMMA4_E2B, TEST) else {
        return;
    };
    check_tail_logits_against_the_reference_split(&loaded, TEST, Some(ADAPTIVE_CHUNK));
}
