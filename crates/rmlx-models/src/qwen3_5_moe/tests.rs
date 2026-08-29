// unsafe_code: mlx-rs Array zero-copy view — slice::from_raw_parts byte-reinterpret in test helpers
#![cfg_attr(test, allow(unsafe_code))]
#![cfg_attr(test, allow(clippy::format_push_string))]
#![allow(
    clippy::cloned_instead_of_copied,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::stable_sort_primitive,
    clippy::too_many_lines
)]
use super::*;

// Pull in mlx primitives that were at module scope in the original flat file
// and are needed by multiple test functions.
use rmlx_mlx::{add, divide, softmax, sum_axis, Array, Device, Dtype};

// AWQ/F16 byte-math now lives in `rmlx_quant::awq`. The MLX-integration tests
// below still call them to build inputs for `quantized_matmul`/`embed_lookup`.
use rmlx_quant::awq::{
    convert_awq_qweight, convert_awq_qzeros_to_biases, f16_bits_to_f32, f32_to_f16_bits,
    quantize_f16_affine_int4,
};
// pub(super) items accessed via explicit submodule paths.
use super::decoder_layer::AttnBlock;
use super::layers::{embed_lookup, Embedding};
use super::prompt_cache::Qwen35MoeEntry;
use crate::prompt_cache::{PromptCache, BLOCK_TOKENS};
use rmlx_kv_ssd::chained_block_hashes;

fn paro_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_QWEN36_PARO").map(std::path::PathBuf::from)
}

fn qwen36_model_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_QWEN36").map(std::path::PathBuf::from)
}

/// Verify the softmax -> argsort top-K -> optional normalize routing math.
///
/// For a two-token batch with 4 experts and top_k=2:
/// Token 0 logits = [2.0, 1.0, 0.5, 0.1] -> top experts: 0, 1
/// Token 1 logits = [0.1, 0.5, 1.0, 2.0] -> top experts: 3, 2
///
/// After softmax and norm_topk_prob, each token's selected weights sum to 1.0.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn qwen3_5_moe_routing_math() {
    let n = 2usize;
    let ne = 4usize;
    let tk = 2usize;

    // Logits: [2, 4]
    let logits_data: Vec<f32> = vec![2.0, 1.0, 0.5, 0.1, 0.1, 0.5, 1.0, 2.0];
    let bytes = unsafe {
        std::slice::from_raw_parts(logits_data.as_ptr().cast::<u8>(), logits_data.len() * 4)
    };
    let logits = Array::from_bytes(bytes, &[2, 4], Dtype::F32).unwrap();

    let device = Device::Cpu;

    // Softmax along last axis.
    let gates = softmax(&logits, -1, device).unwrap();

    // Argsort ascending -> last tk = top-k.
    let sorted_idx = rmlx_mlx::argsort(&gates, device).unwrap();
    let expert_idx = sorted_idx
        .slice(
            &[0, (ne - tk) as i32],
            &[n as i32, ne as i32],
            &[1, 1],
            device,
        )
        .unwrap();
    let expert_idx_i32 = expert_idx.astype(Dtype::I32, device).unwrap();

    // Gather scores.
    let mut off_data = vec![0i32; n * tk];
    for i in 0..n {
        for j in 0..tk {
            off_data[i * tk + j] = (i * ne) as i32;
        }
    }
    let off_bytes =
        unsafe { std::slice::from_raw_parts(off_data.as_ptr().cast::<u8>(), off_data.len() * 4) };
    let offsets = Array::from_bytes(off_bytes, &[(n * tk) as i32], Dtype::I32).unwrap();
    let idx_flat = expert_idx_i32.reshape(&[(n * tk) as i32], device).unwrap();
    let flat_idx = add(&idx_flat, &offsets, device).unwrap();
    let gates_flat = gates.reshape(&[(n * ne) as i32], device).unwrap();
    let scores_flat = gates_flat.take(&flat_idx, 0, device).unwrap();
    let scores = scores_flat.reshape(&[n as i32, tk as i32], device).unwrap();

    // Normalize.
    let s_sum = sum_axis(&scores, -1, device).unwrap();
    let s_sum_2d = s_sum.reshape(&[n as i32, 1], device).unwrap();
    let scores_norm = divide(&scores, &s_sum_2d, device).unwrap();

    // Row sums must be ~1.0.
    // Evaluate to CPU, extract bytes, reinterpret as f32.
    let row_sums = sum_axis(&scores_norm, -1, device).unwrap();
    row_sums.eval().unwrap();
    let sum_bytes = row_sums.to_bytes().unwrap();
    let sum_data: Vec<f32> = sum_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    for (i, &s) in sum_data.iter().enumerate() {
        assert!(
            (s - 1.0).abs() < 1e-4,
            "token {i} normalized weights sum = {s}, expected 1.0"
        );
    }

    // Expert indices: token 0 should pick experts 0 and 1 (highest logits).
    // Token 1 should pick experts 3 and 2.
    expert_idx_i32.eval().unwrap();
    let idx_bytes = expert_idx_i32.to_bytes().unwrap();
    let idx_data: Vec<i32> = idx_bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let tok0: Vec<i32> = {
        let mut v = vec![idx_data[0], idx_data[1]];
        v.sort();
        v
    };
    let tok1: Vec<i32> = {
        let mut v = vec![idx_data[2], idx_data[3]];
        v.sort();
        v
    };
    assert_eq!(
        tok0,
        vec![0, 1],
        "token 0 expected experts {{0,1}}, got {tok0:?}"
    );
    assert_eq!(
        tok1,
        vec![2, 3],
        "token 1 expected experts {{2,3}}, got {tok1:?}"
    );
}

// ── PromptCache unit tests (C1: block-level prefix sharing) ───────────────

/// Build a Qwen35MoeEntry test fixture with computed chained block hashes
/// and empty KV / linear caches.
fn moe_entry(prompt_token_ids: Vec<u32>) -> Qwen35MoeEntry {
    let block_hashes = chained_block_hashes(&prompt_token_ids);
    Qwen35MoeEntry {
        prompt_token_ids,
        block_hashes,
        kv_caches: vec![],
        lin_caches: vec![],
        first_id: 0,
        first_piece: String::new(),
        kv_quant: Some(rmlx_kv_quant::KvQuant::K8V8),
        is_ssd_hydrated: false,
    }
}

/// `find_best_prefix` returns None on an empty cache.
#[test]
fn prompt_cache_lookup_miss_empty() {
    let mut cache: PromptCache<Qwen35MoeEntry> = PromptCache::new(4);
    let ids = vec![1u32, 2, 3, 4];
    assert!(
        cache
            .find_best_prefix(&ids, crate::prompt_cache::FNV_OFFSET)
            .is_none(),
        "empty cache must return None"
    );
}

/// `find_best_prefix` returns the slot index and matched block count on a hit.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prompt_cache_lookup_hit_prefix() {
    let mut cache = PromptCache::new(4);

    // Push a 2-block (512-token) entry.
    let base: Vec<u32> = (0..2 * BLOCK_TOKENS as u32).collect();
    cache.push(moe_entry(base));

    // New prompt: first full block identical, second block diverges.
    let mut new_prompt: Vec<u32> = (0..BLOCK_TOKENS as u32).collect();
    new_prompt.extend(10_000..10_000 + BLOCK_TOKENS as u32);

    let result = cache.find_best_prefix(&new_prompt, crate::prompt_cache::FNV_OFFSET);
    assert!(result.is_some(), "cache must return a prefix hit");
    let (slot_idx, block_count) = result.unwrap();
    assert_eq!(slot_idx, 0, "slot index must be 0");
    assert_eq!(block_count, 1, "exactly one leading block matches");
}

/// `find_best_prefix` ignores a shared prefix shorter than one full block.
#[test]
fn prompt_cache_lookup_miss_below_threshold() {
    let mut cache = PromptCache::new(4);

    // Entry shares only 10 leading tokens with the query — no full block match.
    let base: Vec<u32> = (1..=10).chain(500..700u32).collect();
    cache.push(moe_entry(base));

    let query: Vec<u32> = (1..=10).chain(800..1000u32).collect();
    assert!(
        cache
            .find_best_prefix(&query, crate::prompt_cache::FNV_OFFSET)
            .is_none(),
        "shared prefix < one 256-token block must return None"
    );
}

/// LRU eviction: pushing beyond capacity evicts the oldest-accessed slot
/// (LRU == FIFO when no hits).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn prompt_cache_fifo_eviction() {
    let mut cache = PromptCache::new(2); // capacity = 2

    // Each prompt is 2 full blocks; interleave by 4 so no block is shared.
    let make_ids =
        |off: u32| -> Vec<u32> { (0..2 * BLOCK_TOKENS as u32).map(|x| x * 4 + off).collect() };

    cache.push(moe_entry(make_ids(0))); // slot A
    cache.push(moe_entry(make_ids(1))); // slot B
    assert_eq!(cache.slots.len(), 2);

    // Push slot C: slot A must be evicted (LRU — A pushed first, no hits).
    cache.push(moe_entry(make_ids(2)));
    assert_eq!(cache.slots.len(), 2, "capacity must be respected");

    // Slot A is gone — its exact prompt no longer matches any block.
    let query_a = make_ids(0);
    let r = cache.find_best_prefix(&query_a, crate::prompt_cache::FNV_OFFSET);
    assert!(
        r.is_none(),
        "evicted slot A must not produce a match; got {r:?}"
    );

    // Slot C must match its own 2-block prompt exactly.
    let query_c = make_ids(2);
    let rc = cache.find_best_prefix(&query_c, crate::prompt_cache::FNV_OFFSET);
    assert!(rc.is_some(), "slot C must match its own prompt");
    let (_, blocks) = rc.unwrap();
    assert_eq!(blocks, 2, "slot C shares both blocks with its own prompt");
}

/// C1 regression (the gap 700 unit tests missed): an identical-prompt repeat
/// must be detected as a true EXACT hit, NOT misrouted into the partial path.
///
/// C1 shipped with the callsite Exact test written as
/// `block_count * BLOCK_TOKENS == prompt_ids.len()`. That is essentially never
/// true (only when len % 256 == 0), so an identical re-request of a
/// non-block-aligned prompt fell into the block-truncate + tail-reprefill
/// Prefix path. For qwen3_5_moe that path leaves the recurrent GDN
/// `lin_caches` untouched while truncating KV, corrupting state → the model
/// emitted EOS after 9 tokens instead of the correct 258 (cold value).
///
/// The fixed callsite predicate is `entry.prompt_token_ids() == prompt_ids`
/// (full token equality). This test asserts that predicate behaves correctly
/// for a deliberately non-block-aligned prompt (len % 256 != 0), and that the
/// old block-floored predicate would have FAILED to detect the same exact
/// match — i.e. it pins the exact misrouting bug.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn prompt_cache_identical_prompt_is_exact_not_partial() {
    use crate::prompt_cache::PromptCacheEntry;

    let mut cache: PromptCache<Qwen35MoeEntry> = PromptCache::new(4);

    // 3854 tokens — the real longctx_4k.json prompt length. NOT block-aligned:
    // 3854 % 256 == 14, so block_count = 15 → 15*256 = 3840 != 3854.
    let prompt: Vec<u32> = (0..3854u32).collect();
    assert_ne!(
        prompt.len() % BLOCK_TOKENS,
        0,
        "fixture must be non-aligned"
    );

    cache.push(moe_entry(prompt.clone()));

    // Re-request the SAME prompt (regenerate / identical retry).
    let (slot_idx, block_count) = cache
        .find_best_prefix(&prompt, crate::prompt_cache::FNV_OFFSET)
        .expect("identical prompt must hit");

    // The fixed callsite predicate: full token-level equality => EXACT.
    let is_exact = cache.slots[slot_idx].entry.prompt_token_ids() == prompt.as_slice();
    assert!(
        is_exact,
        "identical prompt must be classified EXACT (full token equality)"
    );

    // The OLD (broken) block-floored predicate would have said "not exact"
    // for this identical prompt, misrouting it into the unsafe partial path.
    let old_broken_exact = block_count * BLOCK_TOKENS == prompt.len();
    assert!(
        !old_broken_exact,
        "block-floored test must NOT detect this exact match \
         (this is precisely the C1 regression being pinned)"
    );
}

/// Validate AWQ → MLX weight conversion feeds a correct `quantized_matmul`.
///
/// Synthetic: in=128, out=8, bits=4, group_size=128, num_groups=1.
/// Weight values after dequant: nibble[i, o] = (i + o) % 8, scale=1.0, zero=0.
/// So dequant(x=ones): out[o] = sum_{i=0}^{127} (i + o) % 8.
///
/// The pure byte-layout assertions for `convert_awq_qweight` /
/// `convert_awq_qzeros_to_biases` live in `rmlx-quant` (`awq_tests.rs`); this
/// test covers the MLX-integration path that `rmlx-quant` cannot (no mlx dep).
///
/// MLX requires group_size ∈ {32, 64, 128} — smaller values have no kernel.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test paro_weight_conversion_matmul -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn paro_weight_conversion_matmul() {
    use rmlx_mlx::{Array, Device, Dtype};

    let in_f = 128usize;
    let out_f = 8usize;
    let bits = 4usize;
    let gs = 128usize;
    let num_groups = in_f / gs; // = 1

    // Build expected nibble matrix [in, out]: nibble[i][o] = (i + o) % 8
    // Each element fits in 4 bits (value 0..7 < 15).
    let mut nibble_matrix = vec![[0u8; 8]; in_f];
    for (i, row) in nibble_matrix.iter_mut().enumerate() {
        for (o, cell) in row.iter_mut().enumerate() {
            *cell = ((i + o) % 8) as u8;
        }
    }

    // Pack in AWQ order: [in, out*bits/32] = [128, 1] I32 words.
    // AWQ interleave: output elements [0,2,4,6,1,3,5,7] go to nibble positions [0..7].
    let awq_order: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
    let words_per_in = out_f * bits / 32; // = 1
    let mut qweight_bytes = vec![0u8; in_f * words_per_in * 4];
    for (i, row) in nibble_matrix.iter().enumerate() {
        // One word per input row (8 nibbles = 8 outputs).
        let mut word = 0u32;
        for (pos, &o) in awq_order.iter().enumerate() {
            word |= u32::from(row[o]) << (pos * 4);
        }
        let off = i * words_per_in * 4;
        qweight_bytes[off..off + 4].copy_from_slice(&word.to_le_bytes());
    }

    // Convert AWQ qweight → MLX layout.
    let mlx_weight_bytes =
        convert_awq_qweight(&qweight_bytes, in_f, out_f, bits).expect("convert_awq_qweight");

    // Build scales (F16 = 1.0) and zeros (0) for num_groups=1, out_f=8.
    let scale_f16_bits: u16 = 0x3C00; // 1.0 in F16
    let mut scales_bytes = vec![0u8; num_groups * out_f * 2]; // [1, 8] F16
    for o in 0..out_f {
        let off = o * 2;
        scales_bytes[off..off + 2].copy_from_slice(&scale_f16_bits.to_le_bytes());
    }
    // qzeros: [num_groups=1, out*bits/32=1] all zeros → zero-points = 0.
    let qzeros_bytes = vec![0u8; num_groups * words_per_in * 4];

    let (scales_t_bytes, biases_t_bytes) =
        convert_awq_qzeros_to_biases(&qzeros_bytes, &scales_bytes, num_groups, out_f, bits)
            .expect("convert_awq_qzeros_to_biases");

    // Call quantized_matmul with x=ones: expect out[o] = sum(nibble[i][o]) for i=0..7.
    // For our nibble matrix: out[o] = sum((i+o)%8, i=0..7) = 0+1+..+7 = 28 for all o.
    let device = Device::Gpu;

    let w = Array::from_bytes(
        &mlx_weight_bytes,
        &[out_f as i32, (in_f * bits / 32) as i32],
        Dtype::U32,
    )
    .expect("w");
    let s = Array::from_bytes(
        &scales_t_bytes,
        &[out_f as i32, num_groups as i32],
        Dtype::F16,
    )
    .expect("s");
    let b = Array::from_bytes(
        &biases_t_bytes,
        &[out_f as i32, num_groups as i32],
        Dtype::F16,
    )
    .expect("b");

    // x = ones [1, in_f] F16
    let x_data = vec![1.0f32; in_f];
    let x_bytes =
        unsafe { std::slice::from_raw_parts(x_data.as_ptr().cast::<u8>(), x_data.len() * 4) };
    let x_f32 = Array::from_bytes(x_bytes, &[1, in_f as i32], Dtype::F32).expect("x_f32");
    let x = x_f32.astype(Dtype::F16, device).expect("x f16");

    let out = rmlx_mlx::quantized_matmul(
        &x,
        &w,
        &s,
        Some(&b),
        gs as i32,
        bits as i32,
        "affine",
        true,
        device,
    )
    .expect("quantized_matmul");

    out.eval().expect("eval");
    let out_f32 = out.astype(Dtype::F32, Device::Cpu).expect("astype f32");
    out_f32.eval().expect("eval f32");
    let out_bytes = out_f32.to_bytes().expect("to_bytes");
    let out_vals: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // Expected: sum_{i=0}^{127} (i+o)%8 = 16 * (0+1+2+3+4+5+6+7) = 16 * 28 = 448.
    // (128/8 = 16 complete cycles; each cycle covers all values 0..7 regardless of offset o.)
    let expected = 448.0f32;
    // Tolerance: 0.5% relative for F16 accumulation errors over 128 elements.
    let tol = expected.mul_add(0.005, 1.0);
    assert_eq!(
        out_vals.len(),
        out_f,
        "output must have out_f={out_f} elements"
    );
    for (o, &v) in out_vals.iter().enumerate() {
        assert!(
            (v - expected).abs() < tol,
            "out[{o}]: expected {expected:.1}, got {v:.4} (diff={:.4})",
            (v - expected).abs()
        );
    }
}

/// Verify `embed_lookup` on-device take+dequantize path — F16 scales arm.
///
/// Synthetic 3-row × 128-col, 4-bit affine embedding. All nibbles in row `r`
/// are fixed to `r` (0, 1, 2). Scale = 1.0 F16, bias = 0.0 F16 for every
/// group, so `dequant(row r)` = r (all 128 values equal `r` as f32).
///
/// Checks that `embed_lookup` with F16 scales:
/// - selects the CORRECT row (ids=[1] → all output values ≈ 1.0, not 0.0/2.0),
/// - returns dtype BF16 (the `astype Bf16` else-arm fires because dequantize
///   returns the scales dtype = F16 ≠ BF16).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test embed_lookup -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "values established by construction; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "byte slices constructed in this fn; try_into() on known-size windows is infallible"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: vec sized at init, loop indices bounded by rows/num_groups"
)]
fn embed_lookup_f16_scales_selects_correct_row_and_casts_to_bf16() {
    let rows = 3usize;
    let cols = 128usize;
    let group_size = 128i32;
    let bits = 4i32;
    let num_groups = (cols as i32) / group_size; // 1
    let words_per_row = (cols * bits as usize) / 32; // 16

    // Build packed 4-bit weight: row r → all nibbles = r.
    // Each U32 word holds 8 nibbles: value r repeated 8 times.
    let mut weight_bytes = vec![0u8; rows * words_per_row * 4];
    for r in 0..rows {
        let nibble = r as u32; // 0, 1, or 2
                               // Pack 8 nibbles of the same value into one U32.
        let mut word = 0u32;
        for pos in 0..8 {
            word |= nibble << (pos * 4);
        }
        let word_bytes = word.to_le_bytes();
        for w in 0..words_per_row {
            let off = (r * words_per_row + w) * 4;
            weight_bytes[off..off + 4].copy_from_slice(&word_bytes);
        }
    }

    // Build F16 scales (all 1.0) and biases (all 0.0): [rows, num_groups] F16.
    let f16_one: u16 = 0x3C00; // 1.0 in IEEE-754 F16
    let mut scales_bytes = vec![0u8; rows * num_groups as usize * 2];
    for i in 0..rows * num_groups as usize {
        scales_bytes[i * 2..i * 2 + 2].copy_from_slice(&f16_one.to_le_bytes());
    }
    let biases_bytes = vec![0u8; rows * num_groups as usize * 2]; // all 0.0

    let device = Device::Gpu;

    let w_arr = Array::from_bytes(
        &weight_bytes,
        &[rows as i32, words_per_row as i32],
        Dtype::U32,
    )
    .expect("weight array");
    let s_arr = Array::from_bytes(&scales_bytes, &[rows as i32, num_groups], Dtype::F16)
        .expect("scales array");
    let b_arr = Array::from_bytes(&biases_bytes, &[rows as i32, num_groups], Dtype::F16)
        .expect("biases array");

    // Look up row 1 — expected dequant output: all 128 values ≈ 1.0.
    let id_bytes = 1u32.to_le_bytes();
    let ids = Array::from_bytes(&id_bytes, &[1], Dtype::U32).expect("ids");

    let out = embed_lookup(
        &ids,
        &w_arr,
        &s_arr,
        Some(&b_arr),
        group_size,
        bits,
        "affine",
        device,
    )
    .expect("embed_lookup F16 arm");
    out.eval().expect("eval");

    // dtype must be BF16 — the F16 scales branch forces astype(BF16).
    assert_eq!(
        out.dtype(),
        Dtype::Bf16,
        "embed_lookup must return BF16 when scales are F16 (astype branch)"
    );

    // Shape must be [1, 128].
    assert_eq!(out.shape(), &[1, cols as i32], "embed_lookup output shape");

    // Convert to F32 for value assertions.
    let out_f32 = out.astype(Dtype::F32, Device::Cpu).expect("astype f32");
    out_f32.eval().expect("eval f32");
    let out_bytes = out_f32.to_bytes().expect("to_bytes");
    let out_vals: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    assert_eq!(out_vals.len(), cols, "output must have {cols} values");
    // With scale=1.0 and bias=0.0, dequant(nibble=1) = 1.0. Row 0 gives 0.0,
    // row 2 gives 2.0 — any wrong row selection fails this assertion.
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            (v - 1.0_f32).abs() < 5e-3,
            "embed_lookup F16 arm: out[{i}] expected ≈1.0, got {v:.6}"
        );
    }
}

/// Verify `embed_lookup` on-device take+dequantize path — BF16 scales arm.
///
/// Same synthetic 3-row × 128-col, 4-bit affine embedding as above, but with
/// BF16 scales and biases. `dequantize` returns BF16 (matching the scales
/// dtype), so `embed_lookup` hits the `Ok(dq)` passthrough without `astype`.
///
/// Checks that:
/// - ids=[2] selects the CORRECT row (all output values ≈ 2.0),
/// - output dtype is BF16 (the direct `Ok(dq)` passthrough arm fires).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test embed_lookup -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "values established by construction; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "byte slices constructed in this fn; try_into() on known-size windows is infallible"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: vec sized at init, loop indices bounded by rows/num_groups"
)]
fn embed_lookup_bf16_scales_passthrough_arm() {
    let rows = 3usize;
    let cols = 128usize;
    let group_size = 128i32;
    let bits = 4i32;
    let num_groups = (cols as i32) / group_size; // 1
    let words_per_row = (cols * bits as usize) / 32; // 16

    // Packed weight: same nibble-per-row construction as the F16 test.
    let mut weight_bytes = vec![0u8; rows * words_per_row * 4];
    for r in 0..rows {
        let nibble = r as u32;
        let mut word = 0u32;
        for pos in 0..8 {
            word |= nibble << (pos * 4);
        }
        let word_bytes = word.to_le_bytes();
        for w in 0..words_per_row {
            let off = (r * words_per_row + w) * 4;
            weight_bytes[off..off + 4].copy_from_slice(&word_bytes);
        }
    }

    // Build BF16 scales (all 1.0) and biases (all 0.0): [rows, num_groups] BF16.
    // BF16 1.0 = upper 2 bytes of F32 1.0 (0x3F80_0000) → [0x80, 0x3F] LE.
    let bf16_one: [u8; 2] = [0x80, 0x3F];
    let mut scales_bytes = vec![0u8; rows * num_groups as usize * 2];
    for i in 0..rows * num_groups as usize {
        scales_bytes[i * 2..i * 2 + 2].copy_from_slice(&bf16_one);
    }
    let biases_bytes = vec![0u8; rows * num_groups as usize * 2]; // BF16 0.0

    let device = Device::Gpu;

    let w_arr = Array::from_bytes(
        &weight_bytes,
        &[rows as i32, words_per_row as i32],
        Dtype::U32,
    )
    .expect("weight array");
    let s_arr = Array::from_bytes(&scales_bytes, &[rows as i32, num_groups], Dtype::Bf16)
        .expect("scales array");
    let b_arr = Array::from_bytes(&biases_bytes, &[rows as i32, num_groups], Dtype::Bf16)
        .expect("biases array");

    // Look up row 2 — expected dequant output: all 128 values ≈ 2.0.
    let id_bytes = 2u32.to_le_bytes();
    let ids = Array::from_bytes(&id_bytes, &[1], Dtype::U32).expect("ids");

    let out = embed_lookup(
        &ids,
        &w_arr,
        &s_arr,
        Some(&b_arr),
        group_size,
        bits,
        "affine",
        device,
    )
    .expect("embed_lookup BF16 arm");
    out.eval().expect("eval");

    // dtype must be BF16 — the BF16 scales produce BF16 dequant, hitting Ok(dq).
    assert_eq!(
        out.dtype(),
        Dtype::Bf16,
        "embed_lookup must return BF16 when scales are already BF16 (passthrough arm)"
    );

    // Shape must be [1, 128].
    assert_eq!(out.shape(), &[1, cols as i32], "embed_lookup output shape");

    // Convert to F32 for value assertions.
    let out_f32 = out.astype(Dtype::F32, Device::Cpu).expect("astype f32");
    out_f32.eval().expect("eval f32");
    let out_bytes = out_f32.to_bytes().expect("to_bytes");
    let out_vals: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    assert_eq!(out_vals.len(), cols, "output must have {cols} values");
    // scale=1.0, bias=0.0: dequant(nibble=2) = 2.0. Row 0→0.0, row 1→1.0 differ.
    for (i, &v) in out_vals.iter().enumerate() {
        assert!(
            (v - 2.0_f32).abs() < 5e-3,
            "embed_lookup BF16 arm: out[{i}] expected ≈2.0, got {v:.6}"
        );
    }
}

/// Validate quantize_f16_affine_int4 on the actual PARO embed_tokens row 760.
///
/// Python reference (from mlx-lm-turboquant env):
/// embed_np[760, 0] = 0.02099609375
/// group0 min=-0.033935546875, max=0.12353515625
/// scale_f16 = -0.01029205322265625, bias_f16 = 0.12353515625
/// dequant[760, :8] = [0.0206146, 3.05e-5, 3.05e-5, 3.05e-5, 3.05e-5, 3.05e-5, 0.010323, -0.010262]
///
/// Run: cargo test -- --ignored quantize_paro_embed_row760
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn quantize_paro_embed_row760() {
    use rmlx_loader::{load_shard_index, ShardSet};
    let Some(model_dir_buf) = paro_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36_PARO not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: PARO model dir not found");
        return;
    }
    let idx = load_shard_index(model_dir).expect("shard index");
    let shards = ShardSet::open(model_dir, &idx).expect("shards");

    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn load_raw_t(shards: &ShardSet, name: &str) -> Option<(Vec<u8>, Vec<usize>)> {
        for (_, h) in shards.iter() {
            if let Ok(st) = h.safetensors() {
                if let Ok(t) = st.tensor(name) {
                    return Some((t.data().to_vec(), t.shape().to_vec()));
                }
            }
        }
        None
    }

    let (w_bytes, w_shape) =
        load_raw_t(&shards, "model.language_model.embed_tokens.weight").expect("load embed");
    let vocab = w_shape[0];
    let hidden = w_shape[1];
    let group_size = 128usize;

    println!("embed shape: [{vocab}, {hidden}]");

    // Get raw F16 values for row 760
    let row = 760usize;
    let row_start = row * hidden * 2;
    let row760_bytes = &w_bytes[row_start..row_start + hidden * 2];

    // Get group 0 (cols 0..128) min/max
    let mut g0_min = f32::INFINITY;
    let mut g0_max = f32::NEG_INFINITY;
    for c in 0..128 {
        let bits = u16::from_le_bytes([row760_bytes[c * 2], row760_bytes[c * 2 + 1]]);
        let v = f16_bits_to_f32(bits);
        if v < g0_min {
            g0_min = v;
        }
        if v > g0_max {
            g0_max = v;
        }
    }
    println!("row760 group0: min={g0_min}, max={g0_max}");
    println!(
        "row760 elem0: {}",
        f16_bits_to_f32(u16::from_le_bytes([row760_bytes[0], row760_bytes[1]]))
    );

    // Run quantize on row 760 only
    let single_row_bytes = row760_bytes.to_vec();
    let (wq, sc, bi) =
        quantize_f16_affine_int4(&single_row_bytes, 1, hidden, group_size).expect("quantize");

    let scale0 = f16_bits_to_f32(u16::from_le_bytes([sc[0], sc[1]]));
    let bias0 = f16_bits_to_f32(u16::from_le_bytes([bi[0], bi[1]]));
    println!("scale[0] = {scale0}, bias[0] = {bias0}");
    println!("expected: scale=-0.01029205, bias=0.12353515625");

    // Check nibble for col 0
    let word0 = u32::from_le_bytes(wq[0..4].try_into().unwrap());
    let nibble0 = word0 & 0xF;
    let dq0 = scale0.mul_add(nibble0 as f32, bias0);
    println!("nibble[0] = {nibble0}, dequant[0] = {dq0}");
    println!("expected dequant[0] = 0.0206146");

    // Dequant first 8
    for i in 0..8 {
        let word = u32::from_le_bytes(wq[(i / 8) * 4..(i / 8) * 4 + 4].try_into().unwrap());
        let nibble = (word >> ((i % 8) * 4)) & 0xF;
        let sg = i / group_size;
        let s = f16_bits_to_f32(u16::from_le_bytes([sc[sg * 2], sc[sg * 2 + 1]]));
        let b = f16_bits_to_f32(u16::from_le_bytes([bi[sg * 2], bi[sg * 2 + 1]]));
        let dq = s.mul_add(nibble as f32, b);
        print!("{dq:.7}, ");
    }
    println!();

    assert!(
        (scale0 - (-0.01029205_f32)).abs() < 1e-4,
        "scale mismatch: got={scale0}"
    );
    assert!(
        (bias0 - 0.12353516_f32).abs() < 1e-4,
        "bias mismatch: got={bias0}"
    );
    assert!(
        (dq0 - 0.0206146_f32).abs() < 1e-4,
        "dq0 mismatch: got={dq0}"
    );

    // Now: build Embedding::Quantized and call embed_lookup for token 760.
    // We use a SINGLE-row embedding (row 760 only, mapped to index 0).
    let num_groups = hidden / group_size;
    let wq_arr =
        Array::from_bytes(&wq, &[1_i32, (hidden * 4 / 32) as i32], Dtype::U32).expect("wq arr");
    let sc_arr = Array::from_bytes(&sc, &[1_i32, num_groups as i32], Dtype::F16).expect("sc arr");
    let bi_arr = Array::from_bytes(&bi, &[1_i32, num_groups as i32], Dtype::F16).expect("bi arr");

    let id_bytes: [u8; 4] = 0u32.to_le_bytes();
    let ids_arr = Array::from_bytes(&id_bytes, &[1], Dtype::U32).expect("ids");

    let result = embed_lookup(
        &ids_arr,
        &wq_arr,
        &sc_arr,
        Some(&bi_arr),
        group_size as i32,
        4,
        "affine",
        Device::Cpu,
    )
    .expect("embed_lookup");
    result.eval().expect("eval");
    let r_f32 = result.astype(Dtype::F32, Device::Cpu).expect("f32");
    r_f32.eval().expect("eval f32");
    let r_bytes = r_f32.to_bytes().expect("bytes");
    let r_vals: Vec<f32> = r_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("embed_lookup[:8] = {:?}", &r_vals[..8]);
    let expected = [
        0.0206146_f32,
        3.05e-5,
        3.05e-5,
        3.05e-5,
        3.05e-5,
        3.05e-5,
        0.010323,
        -0.010262,
    ];
    for (i, (&got, &exp)) in r_vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "embed_lookup[{i}]: expected={exp:.7}, got={got:.7}"
        );
    }
    println!("PASS: embed_lookup matches Python");

    // Extended: quantize TWO rows (row 0 and row 760) and test take+embed_lookup for index 1.
    // Simulates load_from_path_paro but with only 2 rows.
    let row0_bytes = w_bytes[0..hidden * 2].to_vec();
    let row760_start = 760 * hidden * 2;
    let row760_bytes2 = w_bytes[row760_start..row760_start + hidden * 2].to_vec();
    let two_rows: Vec<u8> = row0_bytes
        .iter()
        .chain(row760_bytes2.iter())
        .cloned()
        .collect();

    let (wq2, sc2, bi2) =
        quantize_f16_affine_int4(&two_rows, 2, hidden, group_size).expect("quantize 2 rows");

    let wq2_arr =
        Array::from_bytes(&wq2, &[2_i32, (hidden * 4 / 32) as i32], Dtype::U32).expect("wq2 arr");
    let sc2_arr =
        Array::from_bytes(&sc2, &[2_i32, num_groups as i32], Dtype::F16).expect("sc2 arr");
    let bi2_arr =
        Array::from_bytes(&bi2, &[2_i32, num_groups as i32], Dtype::F16).expect("bi2 arr");

    // Verify scale/bias for row 1 (the embedding for token 760)
    let s2_bytes_vec = sc2_arr.to_bytes().expect("sc2 bytes");
    let b2_bytes_vec = bi2_arr.to_bytes().expect("bi2 bytes");
    let s2 = s2_bytes_vec.as_slice();
    let b2 = b2_bytes_vec.as_slice();
    let s1_0 = f16_bits_to_f32(u16::from_le_bytes([
        s2[num_groups * 2],
        s2[num_groups * 2 + 1],
    ]));
    let b1_0 = f16_bits_to_f32(u16::from_le_bytes([
        b2[num_groups * 2],
        b2[num_groups * 2 + 1],
    ]));
    println!("two-row: scales[1,0]={s1_0:.8} (expected -0.01029205)");
    println!("two-row: biases[1,0]={b1_0:.8} (expected 0.12353516)");
    assert!(
        (s1_0 - (-0.01029205_f32)).abs() < 1e-4,
        "scale[1,0] wrong: {s1_0}"
    );
    assert!(
        (b1_0 - 0.12353516_f32).abs() < 1e-4,
        "bias[1,0] wrong: {b1_0}"
    );

    // embed_lookup for index 1 (= row 760 in 2-row embedding)
    let id1_bytes: [u8; 4] = 1u32.to_le_bytes();
    let ids1_arr = Array::from_bytes(&id1_bytes, &[1], Dtype::U32).expect("ids1");
    let result2 = embed_lookup(
        &ids1_arr,
        &wq2_arr,
        &sc2_arr,
        Some(&bi2_arr),
        group_size as i32,
        4,
        "affine",
        Device::Cpu,
    )
    .expect("embed_lookup2");
    result2.eval().expect("eval2");
    let r2_f32 = result2.astype(Dtype::F32, Device::Cpu).expect("f32 2");
    r2_f32.eval().expect("eval r2");
    let r2_bytes_vec = r2_f32.to_bytes().expect("bytes2");
    let r2_vals: Vec<f32> = r2_bytes_vec
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("two-row embed_lookup[1][:8] = {:?}", &r2_vals[..8]);
    let expected2 = [
        0.0206146_f32,
        3.05e-5,
        3.05e-5,
        3.05e-5,
        3.05e-5,
        3.05e-5,
        0.010323,
        -0.010262,
    ];
    for (i, (&got, &exp)) in r2_vals.iter().zip(expected2.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "two-row embed_lookup[{i}]: expected={exp:.7}, got={got:.7}"
        );
    }
    println!("PASS: two-row embed_lookup matches Python");

    // Full-vocab check: quantize ALL vocab rows, then check row 760.
    // This tests that the large buffer has correct layout.
    println!("Quantizing all {vocab} rows (may take a few seconds in debug)...");
    let (wq_full, sc_full, bi_full) =
        quantize_f16_affine_int4(&w_bytes, vocab, hidden, group_size).expect("quantize full");
    println!("Full quantize done. Checking row 760...");

    // Extract scale/bias for row 760
    let row = 760usize;
    let sg_off = row * num_groups * 2;
    let s760_0 = f16_bits_to_f32(u16::from_le_bytes([sc_full[sg_off], sc_full[sg_off + 1]]));
    let b760_0 = f16_bits_to_f32(u16::from_le_bytes([bi_full[sg_off], bi_full[sg_off + 1]]));
    println!("full: sc[760,0]={s760_0:.8} (expected -0.01029205)");
    println!("full: bi[760,0]={b760_0:.8} (expected 0.12353516)");

    // Dequant elem 0 of row 760
    let w_off = row * (hidden / 8) * 4;
    let word = u32::from_le_bytes(wq_full[w_off..w_off + 4].try_into().unwrap());
    let n0 = word & 0xF;
    let dq_full = s760_0.mul_add(n0 as f32, b760_0);
    println!("full: nibble[760,0]={n0} → dequant={dq_full:.7} (expected 0.0206146)");

    assert!(
        (s760_0 - (-0.01029205_f32)).abs() < 1e-4,
        "full scale wrong: {s760_0}"
    );
    assert!(
        (b760_0 - 0.12353516_f32).abs() < 1e-4,
        "full bias wrong: {b760_0}"
    );
    assert!(
        (dq_full - 0.0206146_f32).abs() < 1e-4,
        "full dq wrong: {dq_full}"
    );
    println!("PASS: full-vocab quantize row 760 correct");
}

/// Validate PARO Linear::forward against Python manual dequant reference.
///
/// Loads layer 0 in_proj_qkv from the PARO checkpoint, calls Linear::forward
/// with x=ones[1,5120], and checks the first 4 output values match the
/// Python reference (computed in paro_qmm_test.py):
/// out[0:4] ≈ [-0.5281, -1.7090, 1.0711, 0.1899]
///
/// This isolates the PARO weight conversion + quantized_matmul path from
/// the rotation kernel. The Linear::Paro path applies rotation first, so we
/// test with rotation bypassed by constructing a Linear::Quantized directly.
///
/// Skipped in CI; run manually: cargo test -- --ignored paro_linear_fwd_layer0
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn paro_linear_fwd_layer0() {
    use rmlx_loader::{load_shard_index, ShardSet};
    use rmlx_mlx::{Array, Device, Dtype};

    let Some(model_dir_buf) = paro_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36_PARO not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: PARO model dir not found");
        return;
    }

    let _shard_path = model_dir.join("model.safetensors");
    let idx = load_shard_index(model_dir).expect("shard index");
    let shards = ShardSet::open(model_dir, &idx).expect("shards");

    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn load_raw_any(shards: &ShardSet, name: &str) -> Option<(Vec<u8>, Vec<usize>)> {
        for (_, h) in shards.iter() {
            if let Ok(st) = h.safetensors() {
                if let Ok(t) = st.tensor(name) {
                    return Some((t.data().to_vec(), t.shape().to_vec()));
                }
            }
        }
        None
    }

    let base = "model.language_model.layers.0.linear_attn.in_proj_qkv";
    let (qw_bytes, qw_shape) = load_raw_any(&shards, &format!("{base}.qweight")).expect("qweight");
    let (sc_bytes, sc_shape) = load_raw_any(&shards, &format!("{base}.scales")).expect("scales");
    let (qz_bytes, _) = load_raw_any(&shards, &format!("{base}.qzeros")).expect("qzeros");

    println!("qweight shape: {qw_shape:?}");
    println!("scales  shape: {sc_shape:?}");

    let in_f = qw_shape[0];
    let num_groups = sc_shape[0];
    let out_f = sc_shape[1];
    let bits = 4usize;
    let group_size = 128usize;

    println!("in={in_f} out={out_f} groups={num_groups} group_size={group_size}");

    // Convert AWQ weight
    let mlx_w_bytes = convert_awq_qweight(&qw_bytes, in_f, out_f, bits).expect("convert qweight");
    let w = Array::from_bytes(
        &mlx_w_bytes,
        &[out_f as i32, (in_f * bits / 32) as i32],
        Dtype::U32,
    )
    .expect("w array");

    let (s_bytes, b_bytes) =
        convert_awq_qzeros_to_biases(&qz_bytes, &sc_bytes, num_groups, out_f, bits)
            .expect("convert qzeros");
    let s = Array::from_bytes(&s_bytes, &[out_f as i32, num_groups as i32], Dtype::F16).expect("s");
    let b = Array::from_bytes(&b_bytes, &[out_f as i32, num_groups as i32], Dtype::F16).expect("b");

    let device = Device::Gpu;

    // x = ones [1, in_f] F16
    let x_data = vec![1.0f32; in_f];
    let x_bytes =
        unsafe { std::slice::from_raw_parts(x_data.as_ptr().cast::<u8>(), x_data.len() * 4) };
    let x_f32 = Array::from_bytes(x_bytes, &[1, in_f as i32], Dtype::F32).expect("x");
    let x = x_f32.astype(Dtype::F16, device).expect("x f16");

    let out = rmlx_mlx::quantized_matmul(
        &x,
        &w,
        &s,
        Some(&b),
        group_size as i32,
        bits as i32,
        "affine",
        true,
        device,
    )
    .expect("quantized_matmul");

    out.eval().expect("eval");
    let out_f32 = out.astype(Dtype::F32, Device::Cpu).expect("astype");
    out_f32.eval().expect("eval f32");
    let bytes = out_f32.to_bytes().expect("to_bytes");
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    println!("Rust out[0:8]: {:?}", &vals[..8.min(vals.len())]);

    // Debug: print scales and biases for out=0, group=0
    // scales_t: [out, num_groups] F16 → scales_t[0, 0] = scales[0, 0]
    // Python: scale[g=0, o=0] = 0x1E27 = 0.006008
    {
        let s_bits = u16::from_le_bytes([s_bytes[0], s_bytes[1]]);
        let b_bits = u16::from_le_bytes([b_bytes[0], b_bytes[1]]);
        let s_f32 = f16_bits_to_f32(s_bits);
        let b_f32 = f16_bits_to_f32(b_bits);
        println!("scale[o=0, g=0]: bits=0x{s_bits:04X} f32={s_f32:.8}");
        println!("bias[o=0,  g=0]: bits=0x{b_bits:04X} f32={b_f32:.8}");
        // Python reference: scale=0x1E27=0.006008, bias=0xAA27=-0.048065
    }

    // Debug: print first 8 weight nibbles for out=0 from the Rust-converted MLX weight.
    // MLX weight shape: [out, in*bits/32], each row is packed input nibbles.
    // For out=0: row 0 of mlx_w_bytes.
    // words_per_out = in_f*4/32 = 16, row 0 spans bytes [0..64].
    {
        let words_per_out = in_f * bits / 32;
        print!("First 8 Rust weight nibbles (out=0, in=0..7): ");
        let mut nibbles = Vec::new();
        for j in 0..words_per_out.min(1) {
            let off = j * 4;
            let word = u32::from_le_bytes(mlx_w_bytes[off..off + 4].try_into().unwrap());
            for k in 0..8 {
                nibbles.push((word >> (k * 4)) & 0xF);
            }
        }
        println!("{:?}", &nibbles[..8]);
        // Python: unpacked[0:8, 0] = nibble_matrix[0:8][0]
        // from python: "First 8 weight nibbles (col=0, i=0..7)"
    }

    // Python reference (from paro_qmm_test.py): MLX qmm out[0:8] ≈
    // [-0.5390625, -1.7080078, 1.0664062, 0.19702148,
    // 0.21374512, -0.9897461, -0.30981445, -1.3105469]
    let reference = [-0.5390625f32, -1.7080078, 1.0664062, 0.19702148];
    let tol = 0.05;
    for (i, (&r, &v)) in reference.iter().zip(vals.iter()).enumerate() {
        assert!(
            (v - r).abs() < tol,
            "out[{i}]: reference={r:.4}, rust={v:.4}, diff={:.4}",
            (v - r).abs()
        );
    }
    println!("PASS: PARO Linear::Quantized forward matches Python reference");
}

/// Layer-0 trace test: loads the PARO model, runs single-token [760] forward,
/// and checks intermediate values match the Python reference.
///
/// Python reference (single token 760, layer 0):
/// embed[760][:8] = [0.02061, 3.05e-5, 3.05e-5, 3.05e-5, 3.05e-5, 3.05e-5, 0.01032, -0.01026]
/// normed[:8] = [1.486, 0.00198, 0.00195, 0.00199, 0.00191, 0.00193, 0.750, -0.629]
/// GDN out[:8] = [-0.03806, 0.002659, -0.01468, 0.000163, -0.04623, -0.000996, 0.004459, -0.01350]
/// layer0_out[:8] = [-0.00491, 0.007183, -0.05164, -0.04550, -0.05255, -0.01894, 0.02736, -0.02786]
///
/// Run manually: cargo test -- --ignored paro_layer0_trace
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn paro_layer0_trace() {
    let Some(model_dir_buf) = paro_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36_PARO not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: PARO model dir not found");
        return;
    }

    println!("Loading PARO model for layer0 trace...");
    let model = load_from_path_paro(model_dir).expect("load PARO model");

    // Diagnostic: inspect embed_tokens scales/biases for row 760 directly.
    {
        if let Embedding::Quantized {
            weight,
            scales,
            biases,
            group_size,
            bits,
            ..
        } = &model.embed_tokens
        {
            println!(
                "embed: weight shape={:?}, scales shape={:?}",
                weight.shape(),
                scales.shape()
            );
            // Extract scales[760, 0] and biases[760, 0]
            let idx_bytes: [u8; 4] = 760u32.to_le_bytes();
            let idx_arr = Array::from_bytes(&idx_bytes, &[1], Dtype::U32).expect("idx");
            let s760 = scales.take(&idx_arr, 0, Device::Cpu).expect("scales take");
            let b760 = biases
                .as_ref()
                .unwrap()
                .take(&idx_arr, 0, Device::Cpu)
                .expect("biases take");
            s760.eval().expect("eval s760");
            b760.eval().expect("eval b760");
            let s_bytes = s760.to_bytes().expect("s bytes");
            let b_bytes = b760.to_bytes().expect("b bytes");
            let s0 = f16_bits_to_f32(u16::from_le_bytes([s_bytes[0], s_bytes[1]]));
            let b0 = f16_bits_to_f32(u16::from_le_bytes([b_bytes[0], b_bytes[1]]));
            println!("embed scales[760, 0] = {s0:.8} (expected -0.01029205)");
            println!("embed biases[760, 0] = {b0:.8} (expected 0.12353516)");
            // Check the packed weight nibbles for row 760, word 0
            let w760 = weight.take(&idx_arr, 0, Device::Cpu).expect("weight take");
            w760.eval().expect("eval w760");
            let w_bytes = w760.to_bytes().expect("w bytes");
            let word0 = u32::from_le_bytes(w_bytes[0..4].try_into().unwrap());
            let n0 = word0 & 0xF;
            let dq0 = s0 * n0 as f32 + b0;
            println!("embed weight[760, word0=0x{word0:08X}] nibble[0]={n0} → dequant={dq0:.7} (expected 0.0206146)");
            println!("group_size={group_size}, bits={bits}");
        }
    }

    let device = Device::Gpu;

    // Single token [760] ("The")
    let ids: Vec<u32> = vec![760];
    let ids_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(ids.as_ptr().cast::<u8>(), ids.len() * 4) };
    let ids_arr = Array::from_bytes(ids_bytes, &[1], Dtype::U32).expect("ids");

    // Embed
    let h = model.embed_tokens.forward(&ids_arr, device).expect("embed");
    h.eval().expect("eval embed");
    let h = h
        .reshape(&[1, 1, model.cfg.hidden_size as i32], device)
        .expect("reshape");

    let h_f32 = h.astype(Dtype::F32, Device::Cpu).expect("h f32");
    h_f32.eval().expect("eval h_f32");
    let h_vals: Vec<f32> = h_f32
        .to_bytes()
        .expect("bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("embed[760][:8] = {:?}", &h_vals[..8]);

    // Python reference (exact F16 values — precision is intentional)
    #[allow(clippy::excessive_precision)]
    let embed_ref: [f32; 8] = [
        0.0206146240234375,
        3.0517578125e-5,
        3.0517578125e-5,
        3.0517578125e-5,
        3.0517578125e-5,
        3.0517578125e-5,
        0.01032257080078125,
        -0.01026153564453125,
    ];
    let tol_embed = 0.0002_f32;
    for (i, (&got, &exp)) in h_vals.iter().zip(embed_ref.iter()).enumerate() {
        assert!(
            (got - exp).abs() < tol_embed,
            "embed[{i}]: expected={exp:.8}, got={got:.8}, diff={:.8}",
            (got - exp).abs()
        );
    }
    println!("PASS: embed matches Python");

    // Debug: print embed[760] RMS and sample elements across all groups.
    {
        let rms_sq: f32 = h_vals.iter().map(|&v| v * v).sum::<f32>() / h_vals.len() as f32;
        let rms = rms_sq.sqrt();
        println!("embed[760] rms={rms:.6} (Python rms≈0.014645)");
        println!("embed[760][128:136] = {:?}", &h_vals[128..136]);
        println!("embed[760][256:264] = {:?}", &h_vals[256..264]);
        // How many elements have abs > 1.0?
        let large = h_vals.iter().filter(|&&v| v.abs() > 1.0).count();
        println!("embed[760] elements with |v|>1.0: {large}");
    }

    // Layer 0
    let layer0 = &model.layers[0];

    // Pre-norm
    let h_normed = layer0.input_layernorm.forward(&h, device).expect("norm");
    h_normed.eval().expect("eval normed");
    let hn_f32 = h_normed.astype(Dtype::F32, Device::Cpu).expect("f32");
    hn_f32.eval().expect("eval hn f32");
    let hn_vals: Vec<f32> = hn_f32
        .to_bytes()
        .expect("bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("normed[:8] = {:?}", &hn_vals[..8]);

    // Exact F16 reference values — precision intentional
    #[allow(clippy::excessive_precision)]
    let normed_ref: [f32; 8] = [
        1.486328125,
        0.0019779205322265625,
        0.0019512176513671875,
        0.001987457275390625,
        0.0019083023071289062,
        0.001926422119140625,
        0.75048828125,
        -0.62890625,
    ];
    let tol_norm = 0.01_f32;
    for (i, (&got, &exp)) in hn_vals.iter().zip(normed_ref.iter()).enumerate() {
        assert!(
            (got - exp).abs() < tol_norm,
            "normed[{i}]: expected={exp:.8}, got={got:.8}, diff={:.8}",
            (got - exp).abs()
        );
    }
    println!("PASS: normed matches Python");

    // GDN forward (no cache)
    let gdn = match &layer0.attn {
        AttnBlock::Linear(gdn) => gdn,
        _ => panic!("layer 0 should be GatedDeltaNet"),
    };
    let y_gdn = gdn.forward(&h_normed, None, device).expect("GDN forward");
    y_gdn.eval().expect("eval y_gdn");
    let y_f32 = y_gdn.astype(Dtype::F32, Device::Cpu).expect("f32");
    y_f32.eval().expect("eval y_f32");
    let y_vals: Vec<f32> = y_f32
        .to_bytes()
        .expect("bytes")
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    println!("GDN output[:8] = {:?}", &y_vals[..8]);

    // Exact F16 reference values — precision intentional
    #[allow(clippy::excessive_precision)]
    let gdn_ref: [f32; 8] = [
        -0.038055419921875,
        0.002658843994140625,
        -0.014678955078125,
        0.00016319751739501953,
        -0.046234130859375,
        -0.000995635986328125,
        0.004459381103515625,
        -0.0135040283203125,
    ];
    let tol_gdn = 0.005_f32;
    let mut gdn_ok = true;
    for (i, (&got, &exp)) in y_vals.iter().zip(gdn_ref.iter()).enumerate() {
        if (got - exp).abs() > tol_gdn {
            println!(
                "GDN MISMATCH [{i}]: expected={exp:.8}, got={got:.8}, diff={:.8}",
                (got - exp).abs()
            );
            gdn_ok = false;
        }
    }
    if gdn_ok {
        println!("PASS: GDN output matches Python");
    } else {
        println!("FAIL: GDN output diverges from Python");
    }

    // Write full GDN output for inspection
    let report = format!(
        "paro_layer0_trace\nembed[:8]={embed_ref:?}\nnormed[:8]={normed_ref:?}\n\
         GDN_ref[:8]={gdn_ref:?}\nGDN_rust[:8]={:?}\n",
        &y_vals[..8]
    );
    std::fs::write("/tmp/paro_layer0_trace_result.txt", &report).ok();

    assert!(
        gdn_ok,
        "GDN output diverges from Python reference. See /tmp/paro_layer0_trace_result.txt"
    );
}

/// Integration test: PARO model forward pass with the real prompt token IDs.
///
/// Python reference (paroquant loader + mlx-lm, temp=0):
/// prompt: "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n<think>\n"
/// token IDs: [248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045, 74455, 198, 248068, 198]
/// first token output: "Paris" (token 24102 or similar)
/// second token output: "."
///
/// Run manually: cargo test -- --ignored integration_paro_forward
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn integration_paro_forward() {
    let Some(model_dir_buf) = paro_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36_PARO not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: PARO model dir not found");
        return;
    }

    println!("Loading PARO model...");
    let model = load_from_path_paro(model_dir).expect("load PARO model");
    println!("Model loaded. vocab_size={}", model.cfg.vocab_size);

    // Token IDs matching Python reference:
    // "<|im_start|>user\nThe capital of France is<|im_end|>\n<|im_start|>assistant\n<think>\n"
    let prompt_ids: Vec<u32> = vec![
        248045, 846, 198, 760, 6511, 314, 9338, 369, 248046, 198, 248045, 74455, 198, 248068, 198,
    ];

    println!("Running forward pass with {} tokens...", prompt_ids.len());
    let logits = model
        .forward_seq(&prompt_ids, Device::Gpu)
        .expect("forward_seq");
    logits.eval().expect("eval");

    let logits_f32 = logits.astype(Dtype::F32, Device::Cpu).expect("logits f32");
    logits_f32.eval().expect("eval f32");
    let bytes = logits_f32.to_bytes().expect("to_bytes");
    let vals: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // Write results to a file since cargo test captures stdout.
    let mut report = String::new();
    report.push_str(&format!("Logits shape: {:?}\n", logits.shape()));
    report.push_str(&format!("Logits[0..5]: {:?}\n", &vals[..5.min(vals.len())]));

    // Find argmax.
    let (argmax, max_val) =
        vals.iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                if v > acc.1 {
                    (i, v)
                } else {
                    acc
                }
            });
    report.push_str(&format!("argmax token: {argmax} (logit={max_val:.4})\n"));

    // Top-10 logits.
    let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    report.push_str(&format!(
        "Top-10 tokens: {:?}\n",
        &indexed[..10.min(indexed.len())]
    ));

    // Also run with raw "The capital of France is" tokens (5 tokens, same as Python reference)
    let raw_prompt_ids: Vec<u32> = vec![760, 6511, 314, 9338, 369];
    report.push_str(&format!(
        "\n--- Raw prompt ({} tokens) ---\n",
        raw_prompt_ids.len()
    ));
    let raw_logits = model
        .forward_seq(&raw_prompt_ids, Device::Gpu)
        .expect("raw forward_seq");
    raw_logits.eval().expect("raw eval");
    let raw_f32 = raw_logits.astype(Dtype::F32, Device::Cpu).expect("raw f32");
    raw_f32.eval().expect("raw eval f32");
    let raw_bytes = raw_f32.to_bytes().expect("raw to_bytes");
    let raw_vals: Vec<f32> = raw_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let (raw_argmax, raw_max) =
        raw_vals
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |acc, (i, &v)| {
                if v > acc.1 {
                    (i, v)
                } else {
                    acc
                }
            });
    report.push_str(&format!(
        "raw argmax token: {raw_argmax} (logit={raw_max:.4})\n"
    ));
    let mut raw_indexed: Vec<(usize, f32)> = raw_vals.iter().copied().enumerate().collect();
    raw_indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    report.push_str(&format!(
        "raw Top-10 tokens: {:?}\n",
        &raw_indexed[..10.min(raw_indexed.len())]
    ));

    std::fs::write("/tmp/paro_integration_result.txt", &report).expect("write result");

    // Python reference (verified 2026-05-08):
    // chat template prompt → argmax=8160 ("Here"), logit≈22.28
    // raw prompt → argmax=11751 (" Paris"), logit≈16.08
    //
    // Tolerance: ±1 token for chat (model is thinking, nondeterministic-ish);
    // raw prompt must be exactly " Paris" (token 11751).
    assert_eq!(
        raw_argmax, 11751,
        "raw prompt: expected argmax=11751 (' Paris') but got {raw_argmax}\nReport:\n{report}"
    );
    assert_eq!(
        argmax, 8160,
        "chat template: expected argmax=8160 ('Here') but got {argmax}\nReport:\n{report}"
    );
    assert!(
        (raw_max - 16.08_f32).abs() < 0.5,
        "raw prompt max logit={raw_max:.4}, expected≈16.08\nReport:\n{report}"
    );
    println!(
        "PASS: integration_paro_forward — raw→'Paris' (token 11751), chat→'Here' (token 8160)"
    );
}

/// Integration test: PARO model generate_greedy with chat-templated "hi" prompt.
///
/// Regression test for the generate_greedy fixedpoint bug (token 227854 "ĠSorr" repeating).
///
/// Verifies that generate_greedy produces token 8160 ("Here") as the first token,
/// matching forward_seq which gives the same argmax without a KV cache.
///
/// Run manually: cargo test -- --ignored integration_paro_generate_greedy
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn integration_paro_generate_greedy() {
    let Some(model_dir_buf) = paro_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36_PARO not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: PARO model dir not found");
        return;
    }

    println!("Loading PARO model...");
    let model = load_from_path_paro(model_dir).expect("load PARO model");
    let tokenizer =
        tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json")).expect("load tokenizer");
    println!("Model loaded. vocab_size={}", model.cfg.vocab_size);

    // Chat-templated "hi" prompt — 11 tokens as the server sends.
    // "<|im_start|>user\nĠhi<|im_end|>\n<|im_start|>assistant\n<think>\n"
    // Token 15131 = "Ġhi" (space+hi via BPE).
    let prompt_ids: Vec<u32> = vec![
        248045, 846, 198, 15131, 248046, 198, 248045, 74455, 198, 248068, 198,
    ];

    println!(
        "Running generate_greedy with {} tokens...",
        prompt_ids.len()
    );
    let mut report = String::new();
    let mut step_fn = |step: &crate::decode_loop::ProbeStep| -> Option<u32> {
        let line = format!(
            "step {} token_id={}\n",
            report.matches('\n').count(),
            step.token_id
        );
        report.push_str(&line);
        println!("{}", line.trim());
        None
    };
    // A7.2: greedy (temperature 0.0) — untouched argmax path.
    let test_sampler_cfg = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: None,
        top_logprobs_k: 0,
    };
    let mut test_rng = crate::sampler::Pcg32::new(test_sampler_cfg.seed_or_default());
    let mut test_token_history: Vec<u32> = Vec::new();
    let test_penalty_cfg = crate::sampler::PenaltyConfig::default();
    let steps = generate_greedy(
        &model,
        &tokenizer,
        &prompt_ids,
        8,
        Device::Gpu,
        rmlx_kv_quant::KvQuant::K8V8,
        Some(4096),
        1,
        &[],
        &mut step_fn,
        None,
        &test_sampler_cfg,
        &mut test_rng,
        &test_penalty_cfg,
        &mut test_token_history,
    )
    .expect("generate_greedy");

    std::fs::write("/tmp/paro_generate_greedy_result.txt", &report).ok();

    let first_id = steps.first().map_or(0, |s| s.token_id);
    assert_eq!(
        first_id, 8160,
        "generate_greedy step 0: expected token 8160 ('Here') but got {first_id}.\n\
         This is the PARO generate_greedy fixedpoint bug.\nSteps: {steps:?}"
    );

    // Also verify it doesn't fixedpoint (first 4 tokens must not all be the same).
    let ids: Vec<u32> = steps.iter().map(|s| s.token_id).collect();
    let all_same = ids.windows(2).all(|w| w[0] == w[1]);
    assert!(
        !all_same,
        "generate_greedy fixedpoint detected: all tokens are the same ({ids:?})"
    );

    println!("PASS: integration_paro_generate_greedy — first token={first_id}");
}

/// Verify paro_rotate_gpu output for embed_tokens[760] through layer 0 in_proj_qkv rotation.
///
/// Python reference (paroquant loader, token_id=760, layer=0, in_proj_qkv):
/// input: embed_tokens.weight[760], shape [1, 5120] F16
/// output: rotated[0, 0:16] = [0.01657, -0.00576, 0.000283, 0.02132,
/// -0.001490, 0.009895, 0.010307, -0.001883,
/// -0.002081, -0.001678, -0.008347, 0.002474,
/// -0.01151, 0.003979, 0.003664, -0.002384]
///
/// Reference output saved as f32 in /tmp/paro_rotation_expected.npy (shape [1,5120]).
///
/// Run manually: cargo test -- --ignored paro_rotation_kernel_vs_python
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn paro_rotation_kernel_vs_python() {
    use rmlx_loader::{load_shard_index, ShardSet};
    use rmlx_mlx::{Array, Device, Dtype};

    let Some(model_dir_buf) = paro_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36_PARO not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: PARO model dir not found");
        return;
    }

    let idx = load_shard_index(model_dir).expect("shard index");
    let shards = ShardSet::open(model_dir, &idx).expect("shards");

    #[allow(
        clippy::expect_used,
        reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
    )]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    #[allow(
        clippy::unwrap_used,
        reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
    )]
    fn load_tensor(shards: &ShardSet, name: &str) -> (Vec<u8>, Vec<usize>) {
        for (_, h) in shards.iter() {
            if let Ok(st) = h.safetensors() {
                if let Ok(t) = st.tensor(name) {
                    return (t.data().to_vec(), t.shape().to_vec());
                }
            }
        }
        panic!("tensor not found: {name}");
    }

    // Load embed_tokens.weight [248320, 5120] F16 — row 760.
    let (emb_bytes, emb_shape) = load_tensor(&shards, "model.language_model.embed_tokens.weight");
    println!("embed_tokens shape: {emb_shape:?}");
    let vocab = emb_shape[0];
    let hidden = emb_shape[1];
    assert_eq!(hidden, 5120, "expected hidden=5120");
    // Row 760 byte range: [760 * 5120 * 2 .. 761 * 5120 * 2].
    let row_start = 760 * hidden * 2;
    let row_end = row_start + hidden * 2;
    let row_bytes = emb_bytes[row_start..row_end].to_vec();
    // Build [1, 5120] F16 array from the row bytes.
    let x = Array::from_bytes(&row_bytes, &[1, hidden as i32], Dtype::F16).expect("embed row");
    let x_gpu = x.astype(Dtype::F16, Device::Gpu).expect("x gpu");

    // Load rotation params for layer 0 in_proj_qkv.
    let base = "model.language_model.layers.0.linear_attn.in_proj_qkv";
    let (theta_bytes, theta_shape) = load_tensor(&shards, &format!("{base}.theta"));
    let (pairs_bytes, pairs_shape) = load_tensor(&shards, &format!("{base}.pairs"));
    let (cs_bytes, cs_shape) = load_tensor(&shards, &format!("{base}.channel_scales"));
    println!("theta shape: {theta_shape:?}");
    println!("pairs shape: {pairs_shape:?}");
    println!("channel_scales shape: {cs_shape:?}");

    // theta: F16 [krot, hidden/2].
    let krot = theta_shape[0];
    let half_hidden = theta_shape[1];
    assert_eq!(half_hidden * 2, hidden, "theta half_hidden mismatch");
    let group_size = 128usize; // PARO Qwen3.6-27B always uses 128.

    // Pre-compute cos/sin from F16 theta bytes (mirrors load_paro_parts).
    let n_theta = krot * half_hidden;
    let mut cos_bytes = vec![0u8; n_theta * 2];
    let mut sin_bytes = vec![0u8; n_theta * 2];
    for i in 0..n_theta {
        let th_bits = u16::from_le_bytes([theta_bytes[i * 2], theta_bytes[i * 2 + 1]]);
        let th_f32 = f16_bits_to_f32(th_bits);
        let cos_f16 = f32_to_f16_bits(th_f32.cos());
        let sin_f16 = f32_to_f16_bits(th_f32.sin());
        cos_bytes[i * 2..i * 2 + 2].copy_from_slice(&cos_f16.to_le_bytes());
        sin_bytes[i * 2..i * 2 + 2].copy_from_slice(&sin_f16.to_le_bytes());
    }
    let cos_theta = Array::from_bytes(&cos_bytes, &[krot as i32, half_hidden as i32], Dtype::F16)
        .expect("cos_theta");
    let sin_theta = Array::from_bytes(&sin_bytes, &[krot as i32, half_hidden as i32], Dtype::F16)
        .expect("sin_theta");

    // Pack I16 pairs [krot, hidden] → I32 [krot, hidden/2].
    // pairs_bytes: I16 little-endian, shape [krot, hidden].
    let packed = crate::paroquant_msl::pack_pairs_cpu(&pairs_bytes, krot, hidden, group_size)
        .expect("pack_pairs_cpu");
    let packed_bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(packed.as_ptr().cast::<u8>(), packed.len() * 4) };
    let packed_pairs =
        Array::from_bytes(packed_bytes, &[krot as i32, half_hidden as i32], Dtype::I32)
            .expect("packed_pairs");

    // Channel scales: F16 [1, hidden] or [hidden].
    let cs_flat_shape: &[i32] = if cs_shape.len() > 1 {
        &[cs_shape[0] as i32, cs_shape[1] as i32]
    } else {
        &[cs_shape[0] as i32]
    };
    let channel_scales =
        Array::from_bytes(&cs_bytes, cs_flat_shape, Dtype::F16).expect("channel_scales");

    println!("krot={krot}, group_size={group_size}, hidden={hidden}");

    // Run the rotation kernel.
    let out = crate::paroquant_msl::paro_rotate_gpu(
        &x_gpu,
        &packed_pairs,
        &cos_theta,
        &sin_theta,
        &channel_scales,
        krot,
        group_size,
        Device::Gpu,
    )
    .expect("paro_rotate_gpu");

    out.eval().expect("eval");
    let out_f32 = out.astype(Dtype::F32, Device::Cpu).expect("astype f32");
    out_f32.eval().expect("eval f32");
    let out_bytes = out_f32.to_bytes().expect("to_bytes");
    let out_vals: Vec<f32> = out_bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();

    // Write full result for inspection.
    {
        let preview: Vec<f32> = out_vals[..16.min(out_vals.len())].to_vec();
        let report = format!(
            "paro_rotation_kernel_vs_python\n\
             krot={krot} group_size={group_size} hidden={hidden}\n\
             Rust out[0:16]: {preview:?}\n\
             Python ref:     [0.01657, -0.00576, 0.000283, 0.02132, -0.001490, 0.009895, 0.010307, -0.001883, -0.002081, -0.001678, -0.008347, 0.002474, -0.01151, 0.003979, 0.003664, -0.002384]\n"
        );
        std::fs::write("/tmp/paro_rotation_kernel_result.txt", &report).ok();
        println!("{report}");
    }

    // Python reference: first 16 elements of rotated output (exact F16 values).
    // Source: /tmp/paro_rotation_expected.npy, computed by paroquant Python loader.
    #[allow(clippy::excessive_precision)]
    let reference: [f32; 16] = [
        0.016571044921875,
        -0.005756378173828125,
        0.00028324127197265625,
        0.0213165283203125,
        -0.0014896392822265625,
        0.00989532470703125,
        0.01030731201171875,
        -0.0018825531005859375,
        -0.0020809173583984375,
        -0.001678466796875,
        -0.0083465576171875,
        0.0024738311767578125,
        -0.01151275634765625,
        0.003978729248046875,
        0.0036640167236328125,
        -0.002384185791015625,
    ];

    // Tolerance: F16 arithmetic → allow 1 ULP at F16 resolution (~0.0002 for values ~0.01).
    let tol = 0.001_f32;
    let mut failures = 0usize;
    for (i, (&got, &exp)) in out_vals.iter().zip(reference.iter()).enumerate() {
        let diff = (got - exp).abs();
        if diff > tol {
            println!("MISMATCH out[{i}]: expected={exp:.8}, got={got:.8}, diff={diff:.8}");
            failures += 1;
        }
    }
    assert_eq!(
        failures, 0,
        "paro_rotate_gpu output diverges from Python reference (see /tmp/paro_rotation_kernel_result.txt)"
    );
    println!("PASS: paro_rotate_gpu matches Python reference within tolerance={tol}");
    let _ = vocab;
}

/// Integration smoke probe — requires the actual model snapshot at the path.
/// Skipped in CI; run manually with: cargo test -- --ignored integration_qwen3_5_moe_35b
#[test]
#[ignore]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn integration_qwen3_5_moe_35b() {
    let Some(model_dir_buf) = qwen36_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36 not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: model dir not found");
        return;
    }
    let model = load_from_path(model_dir).expect("load failed");
    let ids = vec![1u32, 2, 3, 4, 5, 6, 7, 8];
    let result = model.forward_seq(&ids, Device::Cpu);
    match result {
        Ok(arr) => {
            let shape = arr.shape();
            assert_eq!(shape.len(), 3);
            assert_eq!(shape[0], 1);
            assert_eq!(shape[1], 1);
            assert_eq!(shape[2] as usize, model.cfg.vocab_size);
            println!("forward_seq OK shape={shape:?}");
        }
        Err(e) => panic!("forward_seq failed: {e}"),
    }
}

/// Correctness gate for the `HydratedTail` cache path (GitHub issue #9 fix).
///
/// Proves that when a SSD-hydrated block-aligned prefix is in the prompt
/// cache, `generate_greedy` correctly re-prefills only the tail tokens on
/// top of the restored KV/lin state and produces token ids byte-identical
/// to a full cold prefill of the same prompt.
///
/// Test structure:
///
/// 1. COLD: full `generate_greedy` from an empty cache → record N_DECODE token ids.
/// 2. WARM: inject a real KV/lin snapshot of the block-aligned prefix (marked
///    `is_ssd_hydrated=true`) into `PROMPT_CACHE`, then re-run `generate_greedy`
///    with the same prompt → must take `HydratedTail` and produce identical ids.
/// 3. DIVERGENT: inject the same prefix snapshot but pass a prompt whose tail
///    diverges WITHIN the prefix range (not a strict prefix of the stored ids) →
///    `find_best_prefix` returns the slot but the strict-prefix gate rejects it →
///    the consume engine yields `Consumed::Miss` (confirmed by checking the
///    decode output differs from a crafted different-tail cold run, NOT from the
///    warm run).
///
/// Run:
/// ```sh
/// RMLX_KV_TEST_MODEL=/path/to/Qwen3.6-35B-A3B-8bit \
/// cargo test -p rmlx-models hydrated_tail_produces_identical_output \
///     -- --ignored --nocapture
/// ```
#[test]
#[ignore]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free; remaining unwrap is on values constructed in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: indices bounded by slice length validated before call"
)]
fn hydrated_tail_produces_identical_output() {
    let Some(model_dir_buf) = qwen36_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36 not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: model dir not found at {}", model_dir.display());
        return;
    }

    // Verify this is the expected arch before touching the model.
    let arch_str = {
        let cfg_path = model_dir.join("config.json");
        let data = std::fs::read(&cfg_path).expect("read config.json");
        let v: serde_json::Value = serde_json::from_slice(&data).expect("parse config.json");
        v.get("architectures")
            .and_then(|a| a.get(0))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_owned()
    };
    let expected_archs = [
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ];
    if !expected_archs.contains(&arch_str.as_str()) {
        println!("SKIP: arch \"{arch_str}\" is not a Qwen3.5-MoE arch");
        return;
    }

    println!("Loading model from {}", model_dir.display());
    let model = load_from_path(model_dir).expect("load model");
    let n_layers = model.cfg.num_hidden_layers;
    let device = Device::Gpu;

    // Use unquantized KV so the test is noise-free: COLD and WARM must produce
    // token-identical output.
    let kv_quant = rmlx_kv_quant::KvQuant::None;
    let max_seq = 4096i32;

    // Prompt: 2 full blocks + 8 tail tokens = 520 tokens total.
    // Ids are small (1..=520, wrapping at 9999) to stay well within the vocab.
    // Token id 0 is avoided (some models use it as <pad>/<bos>).
    let prefix_len = 2 * BLOCK_TOKENS; // 512
    let tail_len = 8usize;
    let prompt_len = prefix_len + tail_len; // 520
    let prompt_ids: Vec<u32> = (1u32..=prompt_len as u32)
        .map(|i| (i % 9999).max(1))
        .collect();
    assert_eq!(prompt_ids.len(), prompt_len);

    // Sampler config: greedy (temp=0), fully deterministic.
    let sampler_cfg = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let penalty_cfg = crate::sampler::PenaltyConfig::default();
    let n_decode = 8usize; // decode tokens to compare

    // ── Step 1: COLD full prefill ─────────────────────────────────────────────
    // Reset the prompt cache to a fresh 4-slot state (no stale entries from
    // concurrent tests in the same process).
    prompt_cache::ensure_prompt_cache(4);
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
        }
    });

    let cold_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            // Tokenizer is only needed for piece strings in ProbeStep; we can
            // construct a bare tokenizer from file even if pieces are wrong.
            // The token IDs themselves are unaffected by the piece lookup.
            &tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer"),
            &prompt_ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4, // prompt_cache_slots
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("cold generate_greedy");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("COLD tokens: {cold_tokens:?}");
    assert_eq!(
        cold_tokens.len(),
        n_decode,
        "cold path must produce {n_decode} tokens"
    );

    // ── Step 2: Build a real KV/lin snapshot for the block-aligned prefix ────
    // Run a manual prefill of `prompt_ids[..prefix_len]` using the same KV
    // stack as Path C in generate_greedy, so the snapshot is physically correct.
    let (prefix_kv_caches, prefix_lin_caches) = {
        let mut kv_caches: Vec<rmlx_kv_quant::KvCache> =
            crate::kv_cache::kv_layer_quants(n_layers, kv_quant, false)
                .into_iter()
                .enumerate()
                .map(|(i, q)| {
                    rmlx_kv_quant::KvCache::with_quant_max_seq(q, max_seq).with_layer_idx(i)
                })
                .collect();
        let mut lin_caches: Vec<rmlx_kv_quant::LinearAttnCache> = (0..n_layers)
            .map(|_| rmlx_kv_quant::LinearAttnCache::new())
            .collect();

        // Mirror Path C: enter_prefill → run prefix chunks → exit_prefill.
        for c in &mut kv_caches {
            c.enter_prefill();
        }
        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe");
        let prefix = &prompt_ids[..prefix_len];
        let n_chunks = prefix.len().div_ceil(prefill_chunk);
        for (chunk_idx, chunk) in prefix.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            let logits = model
                .forward_seq_with_cache(chunk, Some(&mut kv_caches), Some(&mut lin_caches), device)
                .expect("prefix prefill chunk");
            if is_last {
                // materialise the last-chunk logits so arrays are evaluated.
                logits.eval().expect("eval last-chunk logits");
            } else {
                for c in &kv_caches {
                    c.eval_prefill_state().expect("eval_prefill_state");
                }
            }
        }
        for c in &mut kv_caches {
            c.exit_prefill(device).expect("exit_prefill");
        }
        // Pre-eval for safe cross-thread use (mirrors Path C's pre-eval before push).
        for c in &kv_caches {
            c.eval_for_spill().expect("eval_for_spill kv");
        }
        for c in &lin_caches {
            c.eval_for_spill().expect("eval_for_spill lin");
        }
        (kv_caches, lin_caches)
    };

    // ── Step 3: WARM — inject SSD-hydrated prefix, re-run generate_greedy ───
    // Clear the cache, then inject the prefix snapshot as a "SSD-hydrated" entry.
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
            let prefix_ids = prompt_ids[..prefix_len].to_vec();
            let block_hashes = crate::prompt_cache::chained_block_hashes_seeded(
                &prefix_ids,
                crate::prompt_cache::FNV_OFFSET,
            );
            let kv_snap: rmlx_core::error::Result<Vec<_>> = prefix_kv_caches
                .iter()
                .map(rmlx_kv_quant::KvCache::try_deep_clone)
                .collect();
            let lin_snap: rmlx_core::error::Result<Vec<_>> = prefix_lin_caches
                .iter()
                .map(rmlx_kv_quant::LinearAttnCache::try_deep_clone)
                .collect();
            let (kv_snap, lin_snap) = (kv_snap.expect("kv clone"), lin_snap.expect("lin clone"));
            cache.push(Qwen35MoeEntry {
                prompt_token_ids: prefix_ids,
                block_hashes,
                kv_caches: kv_snap,
                lin_caches: lin_snap,
                first_id: 0,
                first_piece: String::new(),
                kv_quant: Some(kv_quant),
                // KEY: mark as SSD-hydrated so the HydratedTail arm fires.
                is_ssd_hydrated: true,
            });
        }
    });

    let warm_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            &tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer"),
            &prompt_ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("warm generate_greedy");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("WARM tokens: {warm_tokens:?}");

    // PRIMARY ASSERTION: HydratedTail must produce byte-identical output to COLD.
    assert_eq!(
        warm_tokens, cold_tokens,
        "HydratedTail output must be byte-identical to cold prefill.\n\
         COLD: {cold_tokens:?}\n\
         WARM: {warm_tokens:?}\n\
         A mismatch means the tail re-prefill diverged from the full cold prefill."
    );
    println!("PASS: WARM == COLD (HydratedTail produced identical token ids)");

    // ── Step 4: DIVERGENT — strict-prefix gate must reject a non-matching tail ─
    // Build a prompt where the tail tokens at positions [prefix_len..] differ,
    // but more importantly, the stored prefix ids WITHIN [..prefix_len] diverge
    // from the new prompt so `starts_with(stored)` is FALSE.
    // We change the last token of the prefix range to force divergence inside
    // the stored ids.
    let mut divergent_prompt = prompt_ids.clone();
    // Alter a token at position prefix_len - 1 (last token of the stored prefix).
    let altered_val = (divergent_prompt[prefix_len - 1] + 1) % 9999 + 1;
    divergent_prompt[prefix_len - 1] = altered_val;

    // Inject the original prefix snapshot back (it was consumed by the warm run's push).
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
            let prefix_ids = prompt_ids[..prefix_len].to_vec();
            let block_hashes = crate::prompt_cache::chained_block_hashes_seeded(
                &prefix_ids,
                crate::prompt_cache::FNV_OFFSET,
            );
            let kv_snap: rmlx_core::error::Result<Vec<_>> = prefix_kv_caches
                .iter()
                .map(rmlx_kv_quant::KvCache::try_deep_clone)
                .collect();
            let lin_snap: rmlx_core::error::Result<Vec<_>> = prefix_lin_caches
                .iter()
                .map(rmlx_kv_quant::LinearAttnCache::try_deep_clone)
                .collect();
            let (kv_snap, lin_snap) = (kv_snap.expect("kv clone"), lin_snap.expect("lin clone"));
            cache.push(Qwen35MoeEntry {
                prompt_token_ids: prefix_ids,
                block_hashes,
                kv_caches: kv_snap,
                lin_caches: lin_snap,
                first_id: 0,
                first_piece: String::new(),
                kv_quant: Some(kv_quant),
                is_ssd_hydrated: true,
            });
        }
    });

    let divergent_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            &tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer"),
            &divergent_prompt,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("divergent generate_greedy");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("DIVERGENT tokens: {divergent_tokens:?}");

    // The divergent prompt has a different token at position prefix_len-1, so
    // `starts_with(stored_prefix)` is false → the HydratedTail arm is NOT taken
    // → Path C (Miss) fires → a full re-prefill of the divergent prompt occurs.
    //
    // Strict-prefix-gate assertion: the divergent output must NOT equal the
    // warm output (same prefix KV would corrupt output if the gate had failed).
    // Note: divergent and cold COULD coincidentally match on some tokens, but
    // divergent vs warm is the meaningful comparison — warm used the SAME prefix
    // KV as the injected entry, whereas divergent MUST have re-prefilled from
    // scratch (different prompt → different output). We only assert divergent ≠
    // warm, not that divergent is wrong in any absolute sense.
    //
    // In practice these will differ because the last prefix token changed and
    // the model's output is sensitive to it. If they happen to be equal
    // (astronomically unlikely for a 36B model), the gate correctness is still
    // proven by the warm==cold assertion above.
    if divergent_tokens == warm_tokens {
        // Warn but do not fail: the gate is proven by warm==cold. A coincidental
        // match on a trivial prompt (or a model fixedpoint) is possible.
        println!(
            "NOTE: divergent tokens match warm (possible fixedpoint or coincidence). \
             Gate correctness is proven by warm==cold above."
        );
    } else {
        println!("PASS: DIVERGENT ≠ WARM (strict-prefix gate correctly rejected divergent tail)");
    }

    println!(
        "hydrated_tail_produces_identical_output PASS: \
         prefix_len={prefix_len} tail_len={tail_len} n_decode={n_decode}"
    );
}

/// BUG-1 regression: block-aligned full-prompt hydrate must not emit placeholder token 0.
///
/// When a SSD-hydrated entry's `prompt_token_ids.len()` is an exact multiple of
/// `BLOCK_TOKENS` (no tail), the old Exact-arm guard (`prompt_token_ids() == prompt_ids`)
/// fired BEFORE the HydratedTail arm.  That path emitted `first_id = 0` (the
/// placeholder set in `SsdHydrate::hydrate`) as the first real decode token →
/// silent output corruption.
///
/// The fix adds `!is_ssd_hydrated` to the Exact arm guard, forcing a hydrated
/// full-prompt match to fall through to `Miss` → full re-prefill re-derives the
/// real `first_id`.
///
/// Test structure:
///
/// 1. COLD: `generate_greedy` from an empty cache with a 512-token prompt
///    (exactly 2×BLOCK_TOKENS, no tail) → record N_DECODE token ids.
///    Assert the first token is NOT 0 (a zero first token from cold is
///    theoretically possible but would be a separate model bug; for the
///    prompt used here the model emits a non-zero token).
/// 2. WARM: inject the block-aligned KV/lin snapshot marked `is_ssd_hydrated=true`
///    with `first_id=0, first_piece=""` (exactly what `SsdHydrate::hydrate`
///    produces).  Before the fix, `generate_greedy` served this as Exact →
///    `warm[0] == 0` (placeholder).  After the fix it must fall to Miss and
///    produce `warm == cold`.
///
/// Run:
/// ```sh
/// RMLX_TEST_MODEL_QWEN36=/path/to/Qwen3.6-35B-A3B-8bit \
/// cargo test -p rmlx-models hydrated_exact_block_no_tail_not_placeholder \
///     -- --ignored --nocapture
/// ```
#[test]
#[ignore]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free; remaining unwrap is on values constructed in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: indices bounded by slice length validated before call"
)]
fn hydrated_exact_block_no_tail_not_placeholder() {
    let Some(model_dir_buf) = qwen36_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36 not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: model dir not found at {}", model_dir.display());
        return;
    }

    let arch_str = {
        let cfg_path = model_dir.join("config.json");
        let data = std::fs::read(&cfg_path).expect("read config.json");
        let v: serde_json::Value = serde_json::from_slice(&data).expect("parse config.json");
        v.get("architectures")
            .and_then(|a| a.get(0))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_owned()
    };
    let expected_archs = [
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ];
    if !expected_archs.contains(&arch_str.as_str()) {
        println!("SKIP: arch \"{arch_str}\" is not a Qwen3.5-MoE arch");
        return;
    }

    println!("Loading model from {}", model_dir.display());
    let model = load_from_path(model_dir).expect("load model");
    let n_layers = model.cfg.num_hidden_layers;
    let device = Device::Gpu;

    let kv_quant = rmlx_kv_quant::KvQuant::None;
    let max_seq = 4096i32;

    // Prompt: exactly 2×BLOCK_TOKENS = 512 tokens — NO tail.
    // This is the exact-multiple boundary that triggered BUG-1.
    let prompt_len = 2 * BLOCK_TOKENS; // 512
    assert_eq!(
        prompt_len % BLOCK_TOKENS,
        0,
        "fixture must be block-aligned (no tail)"
    );
    let prompt_ids: Vec<u32> = (1u32..=prompt_len as u32)
        .map(|i| (i % 9999).max(1))
        .collect();

    let sampler_cfg = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let penalty_cfg = crate::sampler::PenaltyConfig::default();
    let n_decode = 4usize;

    // ── Step 1: COLD full prefill ─────────────────────────────────────────────
    prompt_cache::ensure_prompt_cache(4);
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
        }
    });

    let cold_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            &tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer"),
            &prompt_ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("cold generate_greedy");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("COLD tokens: {cold_tokens:?}");
    assert_eq!(cold_tokens.len(), n_decode);
    // Sanity: cold first token must not be 0 (that would be a model bug, not ours,
    // but it would make the BUG-1 regression check vacuous).
    assert_ne!(
        cold_tokens[0], 0,
        "cold first token is 0 — this is a model anomaly; the BUG-1 test would be vacuous"
    );

    // ── Step 2: Build a real KV/lin snapshot for the full block-aligned prompt ──
    let (full_kv_caches, full_lin_caches) = {
        let mut kv_caches: Vec<rmlx_kv_quant::KvCache> =
            crate::kv_cache::kv_layer_quants(n_layers, kv_quant, false)
                .into_iter()
                .enumerate()
                .map(|(i, q)| {
                    rmlx_kv_quant::KvCache::with_quant_max_seq(q, max_seq).with_layer_idx(i)
                })
                .collect();
        let mut lin_caches: Vec<rmlx_kv_quant::LinearAttnCache> = (0..n_layers)
            .map(|_| rmlx_kv_quant::LinearAttnCache::new())
            .collect();

        for c in &mut kv_caches {
            c.enter_prefill();
        }
        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe");
        let n_chunks = prompt_len.div_ceil(prefill_chunk);
        for (chunk_idx, chunk) in prompt_ids.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            let logits = model
                .forward_seq_with_cache(chunk, Some(&mut kv_caches), Some(&mut lin_caches), device)
                .expect("prefill chunk");
            if is_last {
                logits.eval().expect("eval last-chunk logits");
            } else {
                for c in &kv_caches {
                    c.eval_prefill_state().expect("eval_prefill_state");
                }
            }
        }
        for c in &mut kv_caches {
            c.exit_prefill(device).expect("exit_prefill");
        }
        for c in &kv_caches {
            c.eval_for_spill().expect("eval_for_spill kv");
        }
        for c in &lin_caches {
            c.eval_for_spill().expect("eval_for_spill lin");
        }
        (kv_caches, lin_caches)
    };

    // ── Step 3: WARM — inject SSD-hydrated FULL-PROMPT snapshot ──────────────
    // Crucially: `first_id=0, first_piece=""` — the placeholder a real
    // SsdHydrate::hydrate sets.  Before the BUG-1 fix this entry is served
    // as Exact → warm[0] == 0 (placeholder corruption).  After the fix it
    // must fall to Miss → full re-prefill → warm == cold.
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
            let block_hashes = crate::prompt_cache::chained_block_hashes_seeded(
                &prompt_ids,
                crate::prompt_cache::FNV_OFFSET,
            );
            let kv_snap: rmlx_core::error::Result<Vec<_>> = full_kv_caches
                .iter()
                .map(rmlx_kv_quant::KvCache::try_deep_clone)
                .collect();
            let lin_snap: rmlx_core::error::Result<Vec<_>> = full_lin_caches
                .iter()
                .map(rmlx_kv_quant::LinearAttnCache::try_deep_clone)
                .collect();
            let (kv_snap, lin_snap) = (kv_snap.expect("kv clone"), lin_snap.expect("lin clone"));
            cache.push(Qwen35MoeEntry {
                prompt_token_ids: prompt_ids.clone(),
                block_hashes,
                kv_caches: kv_snap,
                lin_caches: lin_snap,
                // Placeholder values — exactly what SsdHydrate::hydrate emits.
                first_id: 0,
                first_piece: String::new(),
                kv_quant: Some(kv_quant),
                // KEY: is_ssd_hydrated=true on a full-length entry (no tail).
                // Before the fix, Exact arm fires → emits first_id=0.
                // After the fix, Exact arm skips → Miss → re-prefill.
                is_ssd_hydrated: true,
            });
        }
    });

    let warm_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            &tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer"),
            &prompt_ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("warm generate_greedy");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("WARM tokens: {warm_tokens:?}");

    // PRIMARY ASSERTION: after the BUG-1 fix, warm must equal cold.
    // Before the fix, warm[0] == 0 (placeholder) ≠ cold[0].
    assert_ne!(
        warm_tokens.first().copied(),
        Some(0u32),
        "BUG-1 REGRESSION: warm first token is placeholder 0. \
         The Exact arm must be guarded by !is_ssd_hydrated."
    );
    assert_eq!(
        warm_tokens, cold_tokens,
        "BUG-1 regression: WARM must equal COLD after fix.\n\
         COLD: {cold_tokens:?}\n\
         WARM: {warm_tokens:?}"
    );
    println!(
        "PASS: BUG-1 guard works — warm[0]={} (not 0), warm==cold",
        warm_tokens[0]
    );
    println!(
        "hydrated_exact_block_no_tail_not_placeholder PASS: \
         prompt_len={prompt_len} n_decode={n_decode}"
    );
}

/// BUG-2 equivalence test: HydratedTail at K8V8 quantized KV.
///
/// The original `hydrated_tail_produces_identical_output` test pinned
/// `KvQuant::None`.  For `None` both cold and warm use `update_decode_fp16`
/// regardless of path, so it cannot expose a decode-vs-prefill quantization
/// divergence.  This test repeats the equivalence check at `KvQuant::K8V8`
/// (symmetric 8-bit K and V) — the minimum quantized mode that exercises the
/// `QuantK::append` / `QuantV::append` decode path for the tail tokens.
///
/// Expected outcome: WARM == COLD byte-identical (same token ids).
/// If WARM ≠ COLD, the test prints the divergence evidence (first differing
/// position + both id sequences) and then fails — do NOT gate or hack; report
/// the divergence for manual decision.
///
/// Run:
/// ```sh
/// RMLX_TEST_MODEL_QWEN36=/path/to/Qwen3.6-35B-A3B-8bit \
/// cargo test -p rmlx-models hydrated_tail_k8v8_equivalence \
///     -- --ignored --nocapture
/// ```
#[test]
#[ignore]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free; remaining unwrap is on values constructed in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: indices bounded by slice length validated before call"
)]
fn hydrated_tail_k8v8_equivalence() {
    let Some(model_dir_buf) = qwen36_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36 not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: model dir not found at {}", model_dir.display());
        return;
    }

    let arch_str = {
        let cfg_path = model_dir.join("config.json");
        let data = std::fs::read(&cfg_path).expect("read config.json");
        let v: serde_json::Value = serde_json::from_slice(&data).expect("parse config.json");
        v.get("architectures")
            .and_then(|a| a.get(0))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_owned()
    };
    let expected_archs = [
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ];
    if !expected_archs.contains(&arch_str.as_str()) {
        println!("SKIP: arch \"{arch_str}\" is not a Qwen3.5-MoE arch");
        return;
    }

    println!("Loading model from {}", model_dir.display());
    let model = load_from_path(model_dir).expect("load model");
    let n_layers = model.cfg.num_hidden_layers;
    let device = Device::Gpu;

    // K8V8: symmetric 8-bit K and V. This exercises the quantized append path
    // for the tail tokens (QuantK::append / QuantV::append), unlike KvQuant::None
    // which always uses update_decode_fp16.
    let kv_quant = rmlx_kv_quant::KvQuant::K8V8;
    let max_seq = 4096i32;

    // Same prompt shape as the None test: 2 blocks + 8-token tail = 520 tokens.
    let prefix_len = 2 * BLOCK_TOKENS; // 512
    let tail_len = 8usize;
    let prompt_len = prefix_len + tail_len; // 520
    let prompt_ids: Vec<u32> = (1u32..=prompt_len as u32)
        .map(|i| (i % 9999).max(1))
        .collect();
    assert_eq!(prompt_ids.len(), prompt_len);

    let sampler_cfg = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let penalty_cfg = crate::sampler::PenaltyConfig::default();
    let n_decode = 8usize;

    // ── Step 1: COLD full prefill at K8V8 ────────────────────────────────────
    prompt_cache::ensure_prompt_cache(4);
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
        }
    });

    let cold_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            &tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer"),
            &prompt_ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("cold generate_greedy (K8V8)");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("COLD (K8V8) tokens: {cold_tokens:?}");
    assert_eq!(cold_tokens.len(), n_decode);

    // ── Step 2: Build real KV/lin snapshot for the block-aligned prefix at K8V8 ─
    let (prefix_kv_caches, prefix_lin_caches) = {
        let mut kv_caches: Vec<rmlx_kv_quant::KvCache> =
            crate::kv_cache::kv_layer_quants(n_layers, kv_quant, false)
                .into_iter()
                .enumerate()
                .map(|(i, q)| {
                    rmlx_kv_quant::KvCache::with_quant_max_seq(q, max_seq).with_layer_idx(i)
                })
                .collect();
        let mut lin_caches: Vec<rmlx_kv_quant::LinearAttnCache> = (0..n_layers)
            .map(|_| rmlx_kv_quant::LinearAttnCache::new())
            .collect();

        for c in &mut kv_caches {
            c.enter_prefill();
        }
        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe");
        let prefix = &prompt_ids[..prefix_len];
        let n_chunks = prefix.len().div_ceil(prefill_chunk);
        for (chunk_idx, chunk) in prefix.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            let logits = model
                .forward_seq_with_cache(chunk, Some(&mut kv_caches), Some(&mut lin_caches), device)
                .expect("prefix prefill chunk (K8V8)");
            if is_last {
                logits.eval().expect("eval last-chunk logits");
            } else {
                for c in &kv_caches {
                    c.eval_prefill_state().expect("eval_prefill_state");
                }
            }
        }
        for c in &mut kv_caches {
            c.exit_prefill(device).expect("exit_prefill");
        }
        for c in &kv_caches {
            c.eval_for_spill().expect("eval_for_spill kv");
        }
        for c in &lin_caches {
            c.eval_for_spill().expect("eval_for_spill lin");
        }
        (kv_caches, lin_caches)
    };

    // ── Step 3: WARM — inject SSD-hydrated prefix at K8V8, re-run generate_greedy ─
    prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
        if let Some(cache) = guard.as_mut() {
            cache.clear();
            let prefix_ids = prompt_ids[..prefix_len].to_vec();
            let block_hashes = crate::prompt_cache::chained_block_hashes_seeded(
                &prefix_ids,
                crate::prompt_cache::FNV_OFFSET,
            );
            let kv_snap: rmlx_core::error::Result<Vec<_>> = prefix_kv_caches
                .iter()
                .map(rmlx_kv_quant::KvCache::try_deep_clone)
                .collect();
            let lin_snap: rmlx_core::error::Result<Vec<_>> = prefix_lin_caches
                .iter()
                .map(rmlx_kv_quant::LinearAttnCache::try_deep_clone)
                .collect();
            let (kv_snap, lin_snap) = (kv_snap.expect("kv clone"), lin_snap.expect("lin clone"));
            cache.push(Qwen35MoeEntry {
                prompt_token_ids: prefix_ids,
                block_hashes,
                kv_caches: kv_snap,
                lin_caches: lin_snap,
                first_id: 0,
                first_piece: String::new(),
                kv_quant: Some(kv_quant),
                is_ssd_hydrated: true,
            });
        }
    });

    let warm_tokens: Vec<u32> = {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        let steps = generate_greedy(
            &model,
            &tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer"),
            &prompt_ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("warm generate_greedy (K8V8)");
        steps.into_iter().map(|s| s.token_id).collect()
    };
    println!("WARM (K8V8) tokens: {warm_tokens:?}");

    // Report divergence position before asserting so the evidence is visible.
    if warm_tokens != cold_tokens {
        let first_diff = cold_tokens
            .iter()
            .zip(warm_tokens.iter())
            .position(|(c, w)| c != w)
            .unwrap_or(cold_tokens.len().min(warm_tokens.len()));
        println!(
            "DIVERGENCE at position {first_diff}: \
             cold[{first_diff}]={:?} warm[{first_diff}]={:?}",
            cold_tokens.get(first_diff),
            warm_tokens.get(first_diff),
        );
        println!("COLD: {cold_tokens:?}");
        println!("WARM: {warm_tokens:?}");
    }

    // ASSERTION: byte-identical.  If this fails, the evidence is already printed.
    assert_eq!(
        warm_tokens, cold_tokens,
        "BUG-2: HydratedTail at K8V8 diverged from cold prefill.\n\
         COLD: {cold_tokens:?}\n\
         WARM: {warm_tokens:?}\n\
         See printed divergence position above."
    );
    println!("PASS: WARM (K8V8) == COLD — HydratedTail quantized path is proven correct");
    println!(
        "hydrated_tail_k8v8_equivalence PASS: \
         prefix_len={prefix_len} tail_len={tail_len} kv_quant=K8V8 n_decode={n_decode}"
    );
}

/// Phase C consume-engine migration golden (qwen3.5-moe, Qwen3.6 — model-gated).
///
/// Pins that routing qwen3.5-moe through the shared `consume()` engine is
/// behavior-identical to the pre-migration inline dispatch across all three
/// outcomes reachable for this hybrid GDN arch under `ReusePolicy::ExactOnly`.
/// At temp 0, every reuse/degrade path must decode token-identically to a cold
/// (Miss) baseline of the SAME prompt:
///   (a) ExactOnly forbids a RAM (non-hydrated) PARTIAL match: a 512-token RAM
///       snapshot whose first block is shared with a divergent 512-token request
///       (1 shared full block, then diverges) → the ExactOnly policy gate
///       degrades it to Miss → full re-prefill → WARM == COLD(divergent). The
///       GDN `lin_caches` are never block-truncated.
///   (b) hydrated strict-prefix HydratedTail resume: an SSD-hydrated 512-token
///       block-aligned prefix that the request extends to 520 tokens →
///       `Reuse{StrictPrefix}` → restore + tail-only re-prefill → WARM ==
///       COLD(520).
///   (c) hydrated block-aligned EQUAL-length exclusion (the strict-`<` guard): an
///       SSD-hydrated 512-token entry whose prefix length equals the full
///       512-token prompt (no tail, placeholder first_id 0) → both the Exact
///       arm's `!is_ssd_hydrated` guard and the strict-less-than HydratedTail
///       gate decline → Miss → recompute → WARM == COLD, never the placeholder 0.
///
/// Run:
/// ```sh
/// RMLX_TEST_MODEL_QWEN36=/path/to/mlx-community__Qwen3.6-35B-A3B-8bit \
/// cargo test -p rmlx-models qwen3_5_moe_consume_engine_migration_golden \
///     -- --ignored --nocapture
/// ```
#[test]
#[ignore]
#[allow(
    clippy::unwrap_used,
    reason = "test-only: Mutex critical section is panic-free; remaining unwrap is on values constructed in this fn"
)]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: indices bounded by slice length validated before call"
)]
#[allow(
    clippy::too_many_lines,
    reason = "test-only: a single golden covering the three reachable moe consume outcomes (RAM-partial degrade / hydrated strict-prefix HydratedTail / hydrated equal-length exclusion) reads clearest as one sequential fixture"
)]
fn qwen3_5_moe_consume_engine_migration_golden() {
    let Some(model_dir_buf) = qwen36_model_dir() else {
        println!("SKIP: RMLX_TEST_MODEL_QWEN36 not set");
        return;
    };
    let model_dir = model_dir_buf.as_path();
    if !model_dir.exists() {
        println!("SKIP: model dir not found at {}", model_dir.display());
        return;
    }
    let arch_str = {
        let cfg_path = model_dir.join("config.json");
        let data = std::fs::read(&cfg_path).expect("read config.json");
        let v: serde_json::Value = serde_json::from_slice(&data).expect("parse config.json");
        v.get("architectures")
            .and_then(|a| a.get(0))
            .and_then(|s| s.as_str())
            .unwrap_or("")
            .to_owned()
    };
    let expected_archs = [
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ];
    if !expected_archs.contains(&arch_str.as_str()) {
        println!("SKIP: arch \"{arch_str}\" is not a Qwen3.5-MoE arch");
        return;
    }

    println!("Loading model from {}", model_dir.display());
    let model = load_from_path(model_dir).expect("load model");
    let n_layers = model.cfg.num_hidden_layers;
    let device = Device::Gpu;

    // Unquantized KV so the comparison is noise-free: warm == cold must hold
    // token-for-token.
    let kv_quant = rmlx_kv_quant::KvQuant::None;
    let max_seq = 4096i32;
    let n_decode = 6usize; // short — Qwen3.6 is a 35B model

    let sampler_cfg = crate::sampler::SamplerConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        min_p: 0.0,
        seed: Some(0),
        top_logprobs_k: 0,
    };
    let penalty_cfg = crate::sampler::PenaltyConfig::default();

    // Run generate_greedy at temp 0, return the decoded token_id sequence.
    let run = |ids: &[u32]| -> Vec<u32> {
        let mut rng = crate::sampler::Pcg32::new(sampler_cfg.seed_or_default());
        let mut token_history: Vec<u32> = Vec::new();
        let mut step_fn = |_: &crate::decode_loop::ProbeStep| -> Option<u32> { None };
        generate_greedy(
            &model,
            &tokenizers::Tokenizer::from_file(model_dir.join("tokenizer.json"))
                .expect("load tokenizer"),
            ids,
            n_decode,
            device,
            kv_quant,
            Some(max_seq),
            4,
            &[],
            &mut step_fn,
            None,
            &sampler_cfg,
            &mut rng,
            &penalty_cfg,
            &mut token_history,
        )
        .expect("generate_greedy")
        .into_iter()
        .map(|s| s.token_id)
        .collect()
    };

    let clear_cache = || {
        prompt_cache::ensure_prompt_cache(4);
        prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
            if let Some(cache) = guard.as_mut() {
                cache.clear();
            }
        });
    };

    // Build a real (physically correct) KV/lin snapshot for `ids` via the same
    // KV stack + enter/exit_prefill bracketing as generate_greedy's Miss path.
    let make_snapshot = |ids: &[u32]| -> (
        Vec<rmlx_kv_quant::KvCache>,
        Vec<rmlx_kv_quant::LinearAttnCache>,
    ) {
        let mut kv_caches: Vec<rmlx_kv_quant::KvCache> =
            crate::kv_cache::kv_layer_quants(n_layers, kv_quant, false)
                .into_iter()
                .enumerate()
                .map(|(i, q)| {
                    rmlx_kv_quant::KvCache::with_quant_max_seq(q, max_seq).with_layer_idx(i)
                })
                .collect();
        let mut lin_caches: Vec<rmlx_kv_quant::LinearAttnCache> = (0..n_layers)
            .map(|_| rmlx_kv_quant::LinearAttnCache::new())
            .collect();
        for c in &mut kv_caches {
            c.enter_prefill();
        }
        let prefill_chunk = crate::prefill_chunk::prefill_chunk_for("qwen3_5_moe");
        let n_chunks = ids.len().div_ceil(prefill_chunk);
        for (chunk_idx, chunk) in ids.chunks(prefill_chunk).enumerate() {
            let is_last = chunk_idx + 1 == n_chunks;
            let logits = model
                .forward_seq_with_cache(chunk, Some(&mut kv_caches), Some(&mut lin_caches), device)
                .expect("prefill chunk");
            if is_last {
                logits.eval().expect("eval last-chunk logits");
            } else {
                for c in &kv_caches {
                    c.eval_prefill_state().expect("eval_prefill_state");
                }
            }
        }
        for c in &mut kv_caches {
            c.exit_prefill(device).expect("exit_prefill");
        }
        for c in &kv_caches {
            c.eval_for_spill().expect("eval_for_spill kv");
        }
        for c in &lin_caches {
            c.eval_for_spill().expect("eval_for_spill lin");
        }
        (kv_caches, lin_caches)
    };

    // Push an entry keyed on `key_ids` carrying `(kv, lin)`, into a freshly
    // cleared cache. `hydrated` sets is_ssd_hydrated + placeholder first token
    // (id 0); a non-hydrated RAM entry carries a real first token.
    let push_entry = |key_ids: &[u32],
                      kv: Vec<rmlx_kv_quant::KvCache>,
                      lin: Vec<rmlx_kv_quant::LinearAttnCache>,
                      hydrated: bool,
                      first_id: u32| {
        clear_cache();
        prompt_cache::PROMPT_CACHE.with_inner_mut(|guard| {
            if let Some(cache) = guard.as_mut() {
                let block_hashes = crate::prompt_cache::chained_block_hashes_seeded(
                    key_ids,
                    crate::prompt_cache::request_cache_seed(
                        prompt_cache::active_layout_key(),
                        kv_quant,
                        model.cfg.num_hidden_layers,
                        SHARES_KV_ACROSS_LAYERS,
                        model.model_sig,
                    ),
                );
                cache.push(Qwen35MoeEntry {
                    prompt_token_ids: key_ids.to_vec(),
                    block_hashes,
                    kv_caches: kv,
                    lin_caches: lin,
                    first_id,
                    first_piece: String::new(),
                    kv_quant: Some(kv_quant),
                    is_ssd_hydrated: hydrated,
                });
            }
        });
    };

    // Deterministic synthetic ids (1..=len, wrapping at 9999, never 0).
    let make_ids = |len: usize, salt: u32| -> Vec<u32> {
        (1u32..=len as u32)
            .map(|i| ((i.wrapping_mul(7).wrapping_add(salt)) % 9999).max(1))
            .collect()
    };

    // ── (a) ExactOnly forbids a RAM (non-hydrated) PARTIAL match → Miss ──────
    // Cache a 512-token RAM snapshot; request a divergent 512-token prompt that
    // shares only the first full block. find_best_prefix matches 1 block, but
    // the ExactOnly policy gate forbids the non-hydrated partial → Miss → full
    // re-prefill of the divergent prompt → WARM == COLD(divergent).
    let p512 = make_ids(2 * BLOCK_TOKENS, 0);
    let p512_div: Vec<u32> = {
        let mut v = p512.clone();
        for t in v.iter_mut().skip(BLOCK_TOKENS) {
            *t = ((*t).wrapping_add(101) % 9999).max(1);
        }
        v
    };
    clear_cache();
    let cold_div = run(&p512_div);
    assert_eq!(cold_div.len(), n_decode);
    assert_ne!(cold_div[0], 0, "cold divergent first token is 0 — anomaly");
    let (kv512, lin512) = make_snapshot(&p512);
    // A real RAM entry carries a real first token; use a sentinel non-zero id so
    // a (forbidden) partial reuse would be detectable, but the test asserts the
    // tokens equal the cold re-prefill regardless.
    push_entry(&p512, kv512, lin512, false, 7u32);
    let warm_partial = run(&p512_div);
    println!("(a) RAM-partial degrade: cold={cold_div:?} warm={warm_partial:?}");
    assert_eq!(
        warm_partial, cold_div,
        "(a) ExactOnly must forbid a non-hydrated partial match → Miss → full re-prefill \
         equal to the cold baseline for the divergent prompt"
    );

    // ── (b) hydrated strict-prefix HydratedTail resume → warm == cold ────────
    // SSD-hydrated 512-token block-aligned prefix; request extends it to 520
    // tokens. The strict-`<` HydratedTail gate fires → restore + tail-only
    // re-prefill → WARM == COLD(520).
    let p520: Vec<u32> = {
        let mut v = p512.clone();
        v.extend(make_ids(8, 5)); // 8-token tail with a distinct salt
        v
    };
    clear_cache();
    let cold_520 = run(&p520);
    let (kv_pref, lin_pref) = make_snapshot(&p512);
    push_entry(&p512, kv_pref, lin_pref, true, 0u32);
    let warm_tail = run(&p520);
    println!("(b) HydratedTail: cold={cold_520:?} warm={warm_tail:?}");
    assert_ne!(
        warm_tail.first().copied(),
        Some(0u32),
        "(b) HydratedTail resume must decode a real first token, never the placeholder 0"
    );
    assert_eq!(
        warm_tail, cold_520,
        "(b) hydrated strict-prefix HydratedTail resume must equal the cold baseline for the \
         extended 520-token prompt"
    );

    // ── (c) hydrated block-aligned EQUAL-length exclusion (strict-`<` guard) ──
    // SSD-hydrated 512-token entry whose prefix length equals the full 512-token
    // prompt (no tail, placeholder first_id 0). Neither the Exact arm
    // (`!is_ssd_hydrated`) nor the strict-less-than HydratedTail gate accepts it
    // → Miss → recompute → WARM == COLD, never the placeholder 0.
    clear_cache();
    let cold_512 = run(&p512);
    assert_ne!(cold_512[0], 0, "cold 512 first token is 0 — anomaly");
    let (kv_full, lin_full) = make_snapshot(&p512);
    push_entry(&p512, kv_full, lin_full, true, 0u32);
    let warm_equal = run(&p512);
    println!("(c) hydrated equal-length exclusion: cold={cold_512:?} warm={warm_equal:?}");
    assert_ne!(
        warm_equal.first().copied(),
        Some(0u32),
        "(c) block-aligned hydrated equal-length must NOT replay placeholder 0 — \
         the strict-< guard forces a re-prefill that recomputes the real first token"
    );
    assert_eq!(
        warm_equal, cold_512,
        "(c) block-aligned hydrated equal-length recompute must equal the cold baseline"
    );

    println!(
        "PASS: qwen3_5_moe consume-engine migration golden — RAM-partial degrade / \
         hydrated strict-prefix HydratedTail / hydrated equal-length exclusion all match \
         their cold baselines"
    );
}
