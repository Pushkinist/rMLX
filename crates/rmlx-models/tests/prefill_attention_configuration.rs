//! No attention call reaches mlx-c in the configuration where the pinned MLX
//! returns non-finite rows under Metal device-memory shader validation.
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
//! # The observable, and why no cell for each path is needed
//!
//! `rmlx_mlx::ATTENTION_CALL_TARGET`: one TRACE event for each attention node
//! handed to mlx-c, emitted beside the one call of
//! `mlx_fast_scaled_dot_product_attention` in the workspace.
//! [`one_function_hands_attention_to_mlx_c`] reads that from the source: the
//! binding module is private to `rmlx-mlx` and one line of it names the
//! symbol. Every caller on every architecture, codec route, cache resume,
//! image or audio prefix, perplexity run and drafter therefore passes that
//! line, and none of them has to be listed.
//!
//! [`no_call_in_the_configuration_reaches_mlx_c`] asks the public wrapper
//! itself for each measured-bad shape and reads what reached mlx-c. It names no
//! model, module or entry point, so a rule above the wrapper (a chunk split in
//! some prefill loops), or a rule keyed on a name, leaves it red.
//! [`a_call_outside_the_configuration_reaches_mlx_c_unchanged`] holds the other
//! side: the rule is stated by shape and touches nothing else, decode included.
//! [`a_call_in_the_configuration_returns_the_attention_of_its_rows`] holds the
//! output.
//!
//! The model cells below those three are evidence on real prefills, not the
//! proof of completeness.
//!
//! # What cannot move
//!
//! - A call outside the configuration: it reaches mlx-c as one node with the
//!   shape the caller gave. A decode step has one query row.
//! - The output of a call in the configuration: it is bit-equal to the
//!   concatenation of the nodes that reached mlx-c, issued by hand, and within
//!   one bf16 unit of an f32 reference.
//! - The served tokens, except at a bf16 near-tie. The model cells hold the
//!   argmax at eight positions after the prefill, and the first generated
//!   token, against a reference split. The size of a logit move across two
//!   kernels has no oracle at the model level: two rule-free splits of one
//!   prompt move a logit of Qwen3.6-35B-A3B by 5.5, and one replaced token
//!   moves it by 6.1. No cell asserts on it.
//!
//! # Mutations and the assertion that catches each
//!
//! | Mutation | Caught by |
//! |---|---|
//! | no rule (this tree) | `no_call_in_the_configuration_reaches_mlx_c`; the model cells |
//! | a rule in some prefill loops and not at the call (an unchunked image prefill, a hydrated tail, `ppl`, a codec route is left out) | `no_call_in_the_configuration_reaches_mlx_c` |
//! | a rule keyed on an architecture or module name | the same: the direct call carries no name |
//! | a rule that also changes a call outside the configuration (head dim 128, 192 or 512, no array mask, under the floor, query rows aligned) | `a_call_outside_the_configuration_reaches_mlx_c_unchanged`; the Bonsai-8B cell, which holds the plain chunk split |
//! | a rule that drops, repeats, reorders or shifts query rows, or slices the mask wrongly | `a_call_in_the_configuration_returns_the_attention_of_its_rows` |
//! | a second call of the mlx-c symbol, or the event moved away from the call | `one_function_hands_attention_to_mlx_c` |
//! | the event is removed, or reports another field under a name | `an_attention_call_reports_what_mlx_receives` |
//! | the oracle states another configuration (a block size, the query-row floor, the head dim, the value head dim, the mask kind, the device) | `the_oracle_agrees_with_every_measured_cell` |
//!
//! # What these tests cannot see
//!
//! - An attention node that MLX builds inside another op: the event is at the
//!   mlx-c call, not inside MLX.
//! - A machine without the NAX kernels: the oracle does not ask for them, so
//!   it is stricter than MLX's route there.
//! - Whether the kernel is correct: the oracle is a list of measured cells. A
//!   clean cell bounds a rate; it does not prove the cell clean.
//!
//! The GPU tests are `#[ignore]`d: they drive the Metal context. The model
//! cells resolve their snapshots from `RMLX_O_MODELS_ROOT` by slug (see
//! `tests/common/mod.rs`).

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::print_stdout
)]

mod common;

use common::attention_calls::{
    assert_the_recorder_is_live, is_faulty, recorded, AttentionCall, KEY_BLOCK, QUERY_BLOCK,
    SPLIT_KERNEL_HEAD_DIM, SPLIT_KERNEL_MIN_QUERY_ROWS,
};
use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_mlx::{Array, Device, Dtype};
use rmlx_models::arch::{self, Architecture};
use rmlx_models::prefill_chunk::{module_key_for_class, resolve, set_prefill_chunk};
use rmlx_models::{Pcg32, PenaltyConfig, SamplerConfig};

// ---------------------------------------------------------------------------
// The oracle against the measurement
// ---------------------------------------------------------------------------

/// One pure-MLX probe cell on mlx 0.32.3 under device-memory validation, bf16,
/// on the GPU, with the Metal library of the source build that
/// `docs/MLX_PAIR.md` describes.
#[derive(Clone, Copy)]
struct Cell {
    q_rows: i64,
    k_rows: i64,
    head_dim: i64,
    v_head_dim: i64,
    q_heads: i64,
    kv_heads: i64,
    mask: &'static str,
    /// Repeats that returned a non-finite cell, of `repeats`.
    bad: u32,
    repeats: u32,
}

const fn cell(
    (q_rows, k_rows): (i64, i64),
    (head_dim, v_head_dim): (i64, i64),
    (q_heads, kv_heads): (i64, i64),
    mask: &'static str,
    (bad, repeats): (u32, u32),
) -> Cell {
    Cell {
        q_rows,
        k_rows,
        head_dim,
        v_head_dim,
        q_heads,
        kv_heads,
        mask,
        bad,
        repeats,
    }
}

/// A cell is "measured bad" when at least one repeat returned a non-finite
/// cell. With validation off, every cell that was run returned one digest and
/// no non-finite cell.
///
/// A clean cell is a bound and not a proof: 0 of 24 bounds the rate of one cell
/// near 12 %. The cells with one row count aligned pool to 0 of 224.
const MEASURED_CELLS: [Cell; 29] = [
    cell((2012, 2012), (256, 256), (8, 2), "array", (11, 24)),
    cell((1806, 3854), (256, 256), (16, 2), "array", (9, 16)),
    cell((1025, 3073), (256, 256), (16, 2), "array", (8, 16)),
    cell((1296, 1296), (256, 256), (8, 1), "array", (6, 24)),
    cell((1100, 3148), (256, 256), (8, 1), "array", (2, 16)),
    // Query rows a multiple of 32 and not of 64.
    cell((1056, 3089), (256, 256), (16, 2), "array", (7, 16)),
    // Key rows aligned, query rows not.
    cell((2012, 2048), (256, 256), (8, 2), "array", (0, 24)),
    cell((2016, 2016), (256, 256), (8, 2), "array", (0, 32)),
    cell((1806, 3840), (256, 256), (16, 2), "array", (0, 24)),
    cell((1025, 3072), (256, 256), (16, 2), "array", (0, 24)),
    cell((1296, 1312), (256, 256), (8, 1), "array", (0, 24)),
    // Query rows aligned, key rows not.
    cell((2048, 2012), (256, 256), (8, 2), "array", (0, 32)),
    cell((1984, 2012), (256, 256), (8, 2), "array", (0, 32)),
    cell((1856, 3854), (256, 256), (16, 2), "array", (0, 32)),
    // Both aligned.
    cell((2048, 2048), (256, 256), (8, 2), "array", (0, 24)),
    cell((1792, 3840), (256, 256), (16, 2), "array", (0, 24)),
    // No array mask: the causal mode, and no mask.
    cell((2012, 2012), (256, 256), (8, 2), "causal", (0, 32)),
    cell((2012, 2048), (256, 256), (8, 2), "causal", (0, 32)),
    cell((2047, 2047), (256, 256), (16, 2), "causal", (0, 48)),
    cell((1806, 3854), (256, 256), (16, 2), "causal", (0, 48)),
    cell((2012, 2012), (256, 256), (8, 2), "", (0, 24)),
    // Under the route's query-row floor.
    cell((1023, 3071), (256, 256), (16, 2), "array", (0, 32)),
    cell((1001, 3049), (256, 256), (16, 2), "array", (0, 32)),
    // Another head dim.
    cell((2012, 2012), (128, 128), (32, 8), "array", (0, 16)),
    cell((2012, 2012), (192, 192), (8, 2), "array", (0, 24)),
    cell((2012, 2012), (512, 512), (8, 2), "array", (0, 24)),
    cell((1025, 3073), (512, 512), (16, 2), "array", (0, 24)),
    // Another value head dim.
    cell((2012, 2012), (256, 128), (8, 2), "array", (0, 24)),
    cell((2012, 2012), (512, 256), (8, 2), "array", (0, 24)),
];

impl Cell {
    fn as_gpu_call(self) -> AttentionCall {
        AttentionCall {
            q_heads: self.q_heads,
            q_rows: self.q_rows,
            head_dim: self.head_dim,
            kv_heads: self.kv_heads,
            k_rows: self.k_rows,
            v_head_dim: self.v_head_dim,
            dtype: "Bf16".to_owned(),
            mask: self.mask.to_owned(),
            device: "Gpu".to_owned(),
        }
    }

    /// Ask the public wrapper for this call on `device`. The node is built and
    /// never evaluated.
    fn issue(self, device: Device) -> Array {
        let dims = |heads: i64, rows: i64, dim: i64| [1, heads as i32, rows as i32, dim as i32];
        let q = rmlx_mlx::zeros(
            &dims(self.q_heads, self.q_rows, self.head_dim),
            Dtype::Bf16,
            device,
        )
        .expect("q");
        let k = rmlx_mlx::zeros(
            &dims(self.kv_heads, self.k_rows, self.head_dim),
            Dtype::Bf16,
            device,
        )
        .expect("k");
        let v = rmlx_mlx::zeros(
            &dims(self.kv_heads, self.k_rows, self.v_head_dim),
            Dtype::Bf16,
            device,
        )
        .expect("v");
        let mask = (self.mask == "array").then(|| {
            rmlx_mlx::zeros(
                &[1, 1, self.q_rows as i32, self.k_rows as i32],
                Dtype::Bf16,
                device,
            )
            .expect("mask")
        });
        rmlx_mlx::scaled_dot_product_attention(&q, &k, &v, 1.0, self.mask, mask.as_ref(), device)
            .expect("attention call")
    }
}

#[test]
fn the_oracle_agrees_with_every_measured_cell() {
    let mut bad_cells = 0;
    for measured in MEASURED_CELLS {
        let call = measured.as_gpu_call();
        assert_eq!(
            is_faulty(&call),
            measured.bad > 0,
            "the oracle disagrees with the measurement ({} bad of {}) at {call:?}",
            measured.bad,
            measured.repeats
        );
        bad_cells += u32::from(measured.bad > 0);
    }
    assert_eq!(bad_cells, 6, "the table must hold both verdicts");

    // The same shape on the CPU never reaches a Metal kernel.
    let on_cpu = AttentionCall {
        device: "Cpu".to_owned(),
        ..MEASURED_CELLS[0].as_gpu_call()
    };
    assert!(!is_faulty(&on_cpu));
}

/// The value head dim differs from the query head dim, so a field read from
/// the wrong array shows. Key rows and value rows are one number in every
/// valid call, so this test cannot tell which of the two arrays `k_rows` is
/// read from.
#[test]
fn an_attention_call_reports_what_mlx_receives() {
    let device = Device::Cpu;
    let ((), calls) = recorded(|| {
        let q = rmlx_mlx::zeros(&[1, 2, 5, 8], Dtype::F32, device).expect("q");
        let k = rmlx_mlx::zeros(&[1, 1, 7, 8], Dtype::F32, device).expect("k");
        let v = rmlx_mlx::zeros(&[1, 1, 7, 4], Dtype::F32, device).expect("v");
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
        v_head_dim: 4,
        dtype: "F32".to_owned(),
        mask: mask.to_owned(),
        device: "Cpu".to_owned(),
    };
    assert_eq!(calls, vec![expected("array"), expected("")]);
}

/// The caller enumeration, read from the source and not from a list.
///
/// The mlx-c bindings are the private module `sys` of `rmlx-mlx`, so no other
/// crate can name the symbol. Inside `rmlx-mlx`, one line of non-test code
/// calls it, and the event is emitted in the same function before that line.
///
/// The scan reads names. It cannot see a name that a macro builds from parts.
#[test]
fn one_function_hands_attention_to_mlx_c() {
    // The path with no parenthesis, so a function pointer counts as a call.
    // An import of the name is a second way to reach it.
    const NAME: &str = "mlx_fast_scaled_dot_product_attention";
    const PATH: &str = "sys::mlx_fast_scaled_dot_product_attention";
    let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../rmlx-mlx/src");
    let lib = std::fs::read_to_string(src.join("lib.rs")).expect("rmlx-mlx lib.rs");
    assert!(
        lib.lines().any(|l| l.trim() == "mod sys;"),
        "the mlx-c bindings must stay a private module of rmlx-mlx"
    );

    let mut files = Vec::new();
    let mut dirs = vec![src];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).expect("read_dir") {
            let path = entry.expect("entry").path();
            if path.is_dir() {
                dirs.push(path);
            } else if path.extension().is_some_and(|e| e == "rs") {
                files.push(path);
            }
        }
    }
    assert!(files.len() > 10, "the scan read {} files", files.len());

    let mut call_sites = Vec::new();
    for file in &files {
        let name = file
            .file_name()
            .expect("file name")
            .to_string_lossy()
            .into_owned();
        if name == "sys.rs" || name == "tests.rs" || name.ends_with("_tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(file).expect("read source");
        for (n, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            let imports = code.trim_start().starts_with("use ")
                && (code.contains(NAME) || code.contains("sys::*"));
            if code.contains(PATH) || imports {
                call_sites.push((name.clone(), n, text.clone()));
            }
        }
    }
    assert_eq!(
        call_sites.len(),
        1,
        "exactly one line may hand an attention node to mlx-c; found {:?}",
        call_sites
            .iter()
            .map(|(name, n, _)| format!("{name}:{}", n + 1))
            .collect::<Vec<_>>()
    );
    let (name, line, text) = &call_sites[0];
    assert_eq!(name, "fast_ops.rs");
    let before: Vec<&str> = text.lines().take(*line).collect();
    let fn_start = before
        .iter()
        .rposition(|l| l.starts_with("fn ") || l.starts_with("pub fn "))
        .expect("the function that holds the call");
    assert!(
        before[fn_start..]
            .iter()
            .any(|l| l.trim_start().starts_with("trace_attention_call(")),
        "the attention-call event must be emitted in the function that calls mlx-c, before \
         the call"
    );
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
/// It is the reference split of the tail-argmax cells.
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

// ---------------------------------------------------------------------------
// The seam cells: the wrapper itself, no model
// ---------------------------------------------------------------------------

/// Whether a rule may change this call. The oracle's configuration is inside
/// this set. The set is wider by the calls with unaligned query rows and
/// aligned key rows: those were measured clean, a clean cell is a bound, and a
/// rule that also keeps them off the kernel is allowed.
fn a_rule_may_change(call: &AttentionCall) -> bool {
    call.device == "Gpu"
        && call.mask == "array"
        && call.head_dim == SPLIT_KERNEL_HEAD_DIM
        && call.v_head_dim == SPLIT_KERNEL_HEAD_DIM
        && call.q_rows >= SPLIT_KERNEL_MIN_QUERY_ROWS
        && call.q_rows % QUERY_BLOCK != 0
}

/// The nodes must be the caller's query rows in order, each over all keys.
fn assert_the_nodes_partition_the_query_rows(asked: &AttentionCall, nodes: &[AttentionCall]) {
    assert!(!nodes.is_empty(), "no node reached mlx-c for {asked:?}");
    assert_eq!(
        nodes.iter().map(|n| n.q_rows).sum::<i64>(),
        asked.q_rows,
        "the nodes do not hold the query rows of {asked:?}: {nodes:?}"
    );
    for node in nodes {
        let whole = AttentionCall {
            q_rows: asked.q_rows,
            ..node.clone()
        };
        assert_eq!(
            &whole, asked,
            "a node differs from the call in more than its query rows"
        );
    }
}

#[ignore = "builds attention nodes on the Metal GPU stream"]
#[test]
fn no_call_in_the_configuration_reaches_mlx_c() {
    let mut reached = Vec::new();
    let mut asked_cells = 0;
    for measured in MEASURED_CELLS.iter().filter(|c| c.bad > 0) {
        asked_cells += 1;
        let asked = measured.as_gpu_call();
        let (_node, nodes) = recorded(|| measured.issue(DEVICE));
        assert_the_nodes_partition_the_query_rows(&asked, &nodes);
        reached.extend(nodes.into_iter().filter(is_faulty));
    }
    assert_eq!(asked_cells, 6);
    assert!(
        reached.is_empty(),
        "the wrapper handed mlx-c {} attention nodes in the configuration where the pinned \
         MLX returns non-finite rows under device-memory validation (head dim \
         {SPLIT_KERNEL_HEAD_DIM}, array mask, at least {SPLIT_KERNEL_MIN_QUERY_ROWS} query rows, \
         query rows not a multiple of {QUERY_BLOCK}, key rows not a multiple of {KEY_BLOCK}): \
         {reached:#?}",
        reached.len()
    );
}

#[ignore = "builds attention nodes on the Metal GPU stream"]
#[test]
fn a_call_outside_the_configuration_reaches_mlx_c_unchanged() {
    // The measured cells a rule may not change, a decode step, and a short
    // verify block over a long cache.
    let mut cells: Vec<Cell> = MEASURED_CELLS
        .iter()
        .copied()
        .filter(|c| !a_rule_may_change(&c.as_gpu_call()))
        .collect();
    cells.push(cell((1, 4097), (256, 256), (16, 2), "", (0, 0)));
    cells.push(cell((8, 3081), (256, 256), (16, 2), "array", (0, 0)));
    assert!(cells.len() >= 18, "the grid lost cells: {}", cells.len());
    for outside in cells {
        let asked = outside.as_gpu_call();
        assert!(!is_faulty(&asked));
        let (_node, nodes) = recorded(|| outside.issue(DEVICE));
        assert_eq!(
            nodes,
            vec![asked],
            "a call outside the configuration must reach mlx-c as the one node the caller built"
        );
    }
}

/// Uniform values in `[-1, 1)`, as bf16 on the GPU.
fn uniform_bf16(rng: &mut Pcg32, shape: &[i32]) -> Array {
    let len: i32 = shape.iter().product();
    let data: Vec<f32> = (0..len).map(|_| 2.0 * rng.next_f32() - 1.0).collect();
    Array::from_f32_slice(&data, shape)
        .expect("host array")
        .astype(Dtype::Bf16, DEVICE)
        .expect("astype bf16")
}

fn to_f32_vec(a: &Array) -> Vec<f32> {
    let f32_array = a.astype(Dtype::F32, DEVICE).expect("astype f32");
    Array::eval(&f32_array).expect("materialise");
    f32_array
        .to_bytes()
        .expect("to_bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// One call in the configuration, with data, evaluated with no instrument.
///
/// Two oracles, and neither is a tolerance fitted to a result:
///
/// - The nodes that reached mlx-c, issued again by hand over the same query
///   rows and mask rows and concatenated, give the wrapper's output bit for
///   bit. Query rows are independent in attention, so this holds for every
///   partition of the rows and fails for a dropped, repeated, reordered or
///   shifted row and for a mask row given to another query row.
/// - Each output is a convex combination of value rows, so the exact result
///   is no larger than the largest value. An f32 reference, built from matmul
///   and softmax with no attention kernel, must agree within one bf16 unit at
///   that magnitude. This is what holds a node on another kernel to the same
///   arithmetic.
#[ignore = "evaluates attention on the Metal GPU"]
#[test]
fn a_call_in_the_configuration_returns_the_attention_of_its_rows() {
    const Q_ROWS: i32 = 1025;
    const K_ROWS: i32 = 3073;
    const DIM: i32 = 256;
    const Q_HEADS: i32 = 8;
    const KV_HEADS: i32 = 2;
    // Sharp attention weights, so each output row is close to a few value rows
    // and a wrong row shows at the size of a value.
    let scale = 0.5;
    let full = [1, 1, 1, 1];

    let mut rng = Pcg32::new(7);
    let q = uniform_bf16(&mut rng, &[1, Q_HEADS, Q_ROWS, DIM]);
    let k = uniform_bf16(&mut rng, &[1, KV_HEADS, K_ROWS, DIM]);
    let v = uniform_bf16(&mut rng, &[1, KV_HEADS, K_ROWS, DIM]);
    // A chunk's causal mask: query row `i` is prompt row `K_ROWS - Q_ROWS + i`.
    let offset = (K_ROWS - Q_ROWS) as usize;
    let mut mask_host = vec![0.0f32; (Q_ROWS * K_ROWS) as usize];
    for (i, row) in mask_host.chunks_exact_mut(K_ROWS as usize).enumerate() {
        for cell in &mut row[offset + i + 1..] {
            *cell = -1e30;
        }
    }
    let mask_f32 = Array::from_f32_slice(&mask_host, &[1, 1, Q_ROWS, K_ROWS]).expect("mask");
    let mask = mask_f32.astype(Dtype::Bf16, DEVICE).expect("mask bf16");

    let (output, nodes) = recorded(|| {
        rmlx_mlx::scaled_dot_product_attention(&q, &k, &v, scale, "array", Some(&mask), DEVICE)
            .expect("the call in the configuration")
    });
    let asked = AttentionCall {
        q_heads: i64::from(Q_HEADS),
        q_rows: i64::from(Q_ROWS),
        head_dim: i64::from(DIM),
        kv_heads: i64::from(KV_HEADS),
        k_rows: i64::from(K_ROWS),
        v_head_dim: i64::from(DIM),
        dtype: "Bf16".to_owned(),
        mask: "array".to_owned(),
        device: "Gpu".to_owned(),
    };
    assert!(is_faulty(&asked), "the cell must ask for the configuration");
    assert_the_nodes_partition_the_query_rows(&asked, &nodes);
    let output = to_f32_vec(&output);
    assert!(output.iter().all(|x| x.is_finite()));

    // Oracle 1: the same nodes by hand.
    let mut by_hand = Vec::new();
    let mut row = 0i32;
    for node in &nodes {
        let rows = node.q_rows as i32;
        let q_part = q
            .slice(
                &[0, 0, row, 0],
                &[1, Q_HEADS, row + rows, DIM],
                &full,
                DEVICE,
            )
            .expect("q rows");
        let mask_part = mask
            .slice(&[0, 0, row, 0], &[1, 1, row + rows, K_ROWS], &full, DEVICE)
            .expect("mask rows");
        let part = rmlx_mlx::scaled_dot_product_attention(
            &q_part,
            &k,
            &v,
            scale,
            "array",
            Some(&mask_part),
            DEVICE,
        )
        .expect("node by hand");
        by_hand.extend(to_f32_vec(&part));
        row += rows;
    }
    // Rows are axis 2, so a concatenation of row ranges is not a concatenation
    // of the flat buffers: compare head by head.
    let row_len = DIM as usize;
    let head_len = Q_ROWS as usize * row_len;
    let mut start = 0usize;
    let mut first_row = 0usize;
    for node in &nodes {
        let rows = node.q_rows as usize;
        for head in 0..Q_HEADS as usize {
            let got = &output[head * head_len + first_row * row_len..][..rows * row_len];
            let hand = &by_hand[start + head * rows * row_len..][..rows * row_len];
            assert!(
                got.iter()
                    .zip(hand)
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "head {head}, query rows {first_row}..{}: the wrapper's output is not the \
                 output of the node that reached mlx-c for those rows",
                first_row + rows
            );
        }
        start += Q_HEADS as usize * rows * row_len;
        first_row += rows;
    }

    // Oracle 2: an f32 reference with no attention kernel.
    let group = Q_HEADS / KV_HEADS;
    let mut reference = Vec::new();
    for kv_head in 0..KV_HEADS {
        let f32_of = |a: &Array, from: i32, to: i32, rows: i32| {
            a.slice(&[0, from, 0, 0], &[1, to, rows, DIM], &full, DEVICE)
                .expect("head slice")
                .astype(Dtype::F32, DEVICE)
                .expect("astype f32")
        };
        let q_f32 = f32_of(&q, kv_head * group, (kv_head + 1) * group, Q_ROWS);
        let k_f32 = f32_of(&k, kv_head, kv_head + 1, K_ROWS);
        let v_f32 = f32_of(&v, kv_head, kv_head + 1, K_ROWS);
        let k_t = k_f32.transpose(&[0, 1, 3, 2], DEVICE).expect("transpose");
        let scores = rmlx_mlx::matmul(&q_f32, &k_t, DEVICE).expect("q k^T");
        let scores =
            rmlx_mlx::multiply(&scores, &rmlx_mlx::scalar_f32(scale), DEVICE).expect("scale");
        let scores = rmlx_mlx::add(&scores, &mask_f32, DEVICE).expect("mask");
        let weights = rmlx_mlx::softmax_precise(&scores, -1, DEVICE).expect("softmax");
        reference.extend(to_f32_vec(
            &rmlx_mlx::matmul(&weights, &v_f32, DEVICE).expect("p v"),
        ));
    }
    assert_eq!(reference.len(), output.len());
    let largest_value = to_f32_vec(&v).iter().fold(0.0f32, |m, x| m.max(x.abs()));
    // bf16 keeps 8 significant bits: one unit at magnitude `x` is 2^(exp(x) - 7).
    let one_unit = 2.0f32.powi(largest_value.log2().floor() as i32 - 7);
    let worst = output
        .iter()
        .zip(&reference)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!(
        "[a_call_in_the_configuration_returns_the_attention_of_its_rows] nodes {:?}, largest \
         |output - f32 reference| {worst}, one bf16 unit at the largest value {one_unit}",
        nodes
            .iter()
            .map(|n| (n.q_rows, n.k_rows))
            .collect::<Vec<_>>()
    );
    assert!(
        worst <= one_unit,
        "the output differs from the f32 reference by {worst}, more than one bf16 unit \
         ({one_unit}) at the largest value {largest_value}"
    );
}

// ---------------------------------------------------------------------------
// The model cells: evidence on real prefills
// ---------------------------------------------------------------------------

struct Loaded {
    model: Architecture,
    tokenizer: tokenizers::Tokenizer,
    /// Token ids of one passage, long enough for every cell.
    passage: Vec<u32>,
}

fn load(model: &common::GoldenModel, test: &str) -> Option<Loaded> {
    let path = common::model_for(model, test)?;
    let loaded =
        arch::load_model(&path, DEVICE, &arch::LoadOpts::default()).expect("arch::load_model");
    let tokenizer =
        tokenizers::Tokenizer::from_file(path.join("tokenizer.json")).expect("tokenizer.json");
    let passage: Vec<u32> = tokenizer
        .encode(
            "A cartographer walked the ridge at dawn, tracing every river and switchback \
             onto oiled linen. "
                .repeat(600),
            false,
        )
        .expect("tokenize")
        .get_ids()
        .to_vec();
    assert!(
        passage.len() >= 6000,
        "the passage is too short for the cells: {} tokens",
        passage.len()
    );
    Some(Loaded {
        model: loaded,
        tokenizer,
        passage,
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
    kv_quant: Option<KvQuant>,
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
                kv_quant,
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

fn generate_calls(
    loaded: &Loaded,
    prompt: &[u32],
    cache_slots: usize,
    kv_quant: Option<KvQuant>,
) -> Vec<AttentionCall> {
    generate_one(loaded, prompt, cache_slots, kv_quant, 0).1
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
///
/// `plain_split`: the model has no call a rule may change, so the nodes of its
/// prefill must be the plain split of the prompt into chunks, each chunk one
/// node over the keys so far.
fn check_fresh_prefills(loaded: &Loaded, label: &str, plain_split: bool, report: &mut Report) {
    let chunk = resolved_chunk(&loaded.model);
    for len in lengths_across_the_chunk_boundaries(chunk) {
        loaded.model.clear_prompt_cache();
        let calls = generate_calls(loaded, &loaded.passage[..len], 0, None);
        if plain_split {
            let prefill: Vec<AttentionCall> = calls
                .iter()
                .filter(|c| c.k_rows <= len as i64)
                .cloned()
                .collect();
            let expected: Vec<(i64, i64)> = (0..len)
                .step_by(chunk)
                .map(|at| (chunk.min(len - at) as i64, (at + chunk).min(len) as i64))
                .collect();
            assert_eq!(
                distinct_shapes(&prefill),
                expected,
                "{label} {len} tokens at chunk {chunk}: the prefill is not the plain chunk split"
            );
        }
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
fn check_resumed_prefix(
    loaded: &Loaded,
    label: &str,
    tail: usize,
    kv_quant: Option<KvQuant>,
    report: &mut Report,
) {
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
    let _ = generate_calls(loaded, &loaded.passage[..SHARED], SLOTS, kv_quant);

    let second = &loaded.passage[..SHARED + tail];
    let (hits_before, _) = stats();
    let calls = generate_calls(loaded, second, SLOTS, kv_quant);
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
        &format!(
            "{label} resumed prefix, chunk {chunk}, tail of {tail} tokens, codec {kv_quant:?}"
        ),
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
    /// No call of this model is one a rule may change.
    plain_split: bool,
}

/// Every cell of one model: at the default chunk, then at a chunk the adaptive
/// controller installs.
fn check_every_prefill_path(loaded: &Loaded, test: &str, paths: Paths) {
    assert_the_recorder_is_live(DEVICE);
    let mut report = Report::default();

    for (label, chunk, tail) in [
        ("default:", None, 1025),
        ("override:", Some(ADAPTIVE_CHUNK), 1100),
    ] {
        let _chunk = chunk.map(ChunkOverride::install);
        check_fresh_prefills(loaded, label, paths.plain_split, &mut report);
        if paths.resumes_an_extended_prompt {
            // The default codec, and one quantized store: a resumed tail
            // appends through the codec's own route to the attention call.
            for kv_quant in [None, Some(KvQuant::K8V8)] {
                check_resumed_prefix(loaded, label, tail, kv_quant, &mut report);
            }
        }
        if paths.speculative {
            check_speculative_prefill(loaded, label, &mut report);
        }
    }
    report.finish(test);
}

// ---------------------------------------------------------------------------
// The model cells, one test for each model
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
            plain_split: false,
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
            plain_split: false,
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
            plain_split: false,
        },
    );
}

/// Head dim 128: no call of this model is on the split kernel. The test holds
/// the other side of the rule: every prefill here is the plain chunk split.
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
            plain_split: true,
        },
    );
}

// ---------------------------------------------------------------------------
// The tail-argmax cells
// ---------------------------------------------------------------------------

/// Positions judged after the prefill. Eight query rows keep the judging
/// forward itself off the full-attention kernels.
const TAIL: usize = 8;

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

/// The largest difference at each tail position. Printed, never asserted on:
/// it has no oracle at this level (see the module doc).
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
/// configuration without a rule, against [`REFERENCE_CHUNK`]: the argmax at
/// each of the next [`TAIL`] positions, and the first generated token.
///
/// A token that differs is a stop, not a re-pin: the message carries both
/// top-2 margins, which is what tells a near-tie from a defect.
///
/// `production_chunk` is `None` for the architecture's default and `Some` for
/// a chunk the adaptive controller installs.
fn check_tail_argmax_against_the_reference_split(
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
    assert_ne!(
        production, REFERENCE_CHUNK,
        "the production split must differ from the reference split"
    );
    let prefill_len = production + SPLIT_KERNEL_MIN_QUERY_ROWS as usize + 1;
    let ids = &loaded.passage[..prefill_len + TAIL];

    let (used, production_rows) = rows_at(production_chunk, ids, prefill_len);
    assert_eq!(used, production);
    let (used, reference_rows) = rows_at(Some(REFERENCE_CHUNK), ids, prefill_len);
    assert_eq!(used, REFERENCE_CHUNK);

    let moved = moved_per_position(&production_rows, &reference_rows);
    println!(
        "[{test}] production chunk {production}, reference chunk {REFERENCE_CHUNK}, prefill \
         {prefill_len} tokens; largest logit difference at each position: {moved:?}"
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
    }

    // The prefill's own last row, through the entry the CLI and the server
    // call: the first token of a fresh generation.
    let first_token = |chunk: Option<usize>| {
        let _chunk = chunk.map(ChunkOverride::install);
        model.clear_prompt_cache();
        let (steps, _) = generate_one(loaded, &ids[..prefill_len], 0, None, 2);
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
fn qwen3_6_moe_tail_argmax_holds_across_the_prefill_split() {
    const TEST: &str = "qwen3_6_moe_tail_argmax_holds_across_the_prefill_split";
    let Some(loaded) = load(&QWEN3_6_MOE, TEST) else {
        return;
    };
    check_tail_argmax_against_the_reference_split(&loaded, TEST, None);
}

#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn qwen3_5_dense_tail_argmax_holds_across_the_prefill_split() {
    const TEST: &str = "qwen3_5_dense_tail_argmax_holds_across_the_prefill_split";
    let Some(loaded) = load(&QWEN3_5_DENSE, TEST) else {
        return;
    };
    check_tail_argmax_against_the_reference_split(&loaded, TEST, None);
}

#[ignore = "loads a model and drives the Metal GPU"]
#[test]
fn gemma4_e2b_tail_argmax_holds_across_the_prefill_split() {
    const TEST: &str = "gemma4_e2b_tail_argmax_holds_across_the_prefill_split";
    let Some(loaded) = load(&GEMMA4_E2B, TEST) else {
        return;
    };
    check_tail_argmax_against_the_reference_split(&loaded, TEST, Some(ADAPTIVE_CHUNK));
}
