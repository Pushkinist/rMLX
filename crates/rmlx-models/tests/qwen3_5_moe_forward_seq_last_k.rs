//! Cached last-K forward equivalence for the Qwen3.5-MoE hybrid stack.
//!
//! Asserts `Qwen3_5MoeText::forward_seq_last_k_with_cache` (the speculative
//! verifier path) produces the same last-K logits as a single cold forward
//! over the identical token sequence — i.e. splitting the sequence into a
//! cached prefill (`ids[..seq-k]`, advancing BOTH the KV caches and the GDN
//! `lin_caches`) followed by a K-position pass equals the reference one-shot
//! forward, within fp tolerance.
//!
//! This is the core correctness gate for the recurrent-state plumbing: the
//! GatedDeltaNet `lin_caches` must be advanced through the prefill split so
//! the K-position pass continues the recurrence at the right point.
//!
//! Model: `mlx-community__Qwen3.6-35B-A3B-8bit`.
//!
//! cargo test -p rmlx-models --test qwen3_5_moe_forward_seq_last_k -- --ignored
//!
//! `#[ignore]`d for the large model load. The snapshot resolves from
//! `RMLX_O_MODELS_ROOT` by slug (see `tests/common/mod.rs`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::items_after_statements,
    clippy::trivially_copy_pass_by_ref,
    clippy::too_many_lines,
    clippy::wildcard_enum_match_arm,
    clippy::needless_pass_by_value
)]

mod common;

use rmlx_kv_quant::{KvCache, KvQuant, LinearAttnCache};
use rmlx_mlx::Device;
use rmlx_models::qwen3_5_moe;

/// The snapshot this test covers, and the architectures it was written against.
const MODEL: common::GoldenModel = common::GoldenModel {
    slug: "mlx-community__Qwen3.6-35B-A3B-8bit",
    archs: &[
        "Qwen3_5MoeForCausalLM",
        "Qwen3_5MoeForConditionalGeneration",
    ],
};

/// Read all logit rows of a `[1, n, vocab]` Array into a flat `Vec<f32>`.
/// Logits come back in the model dtype (bf16/fp16), so cast to F32 first.
fn logits_to_vec(logits: &rmlx_mlx::Array, device: Device) -> Vec<f32> {
    let f32_logits = logits
        .astype(rmlx_mlx::Dtype::F32, device)
        .expect("astype f32");
    rmlx_mlx::Array::eval(&f32_logits).expect("materialise logits");
    let bytes = f32_logits.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Index of the max element of `row`.
fn argmax_row(row: &[f32]) -> usize {
    let mut best = (0usize, f32::NEG_INFINITY);
    for (i, &x) in row.iter().enumerate() {
        if x > best.1 {
            best = (i, x);
        }
    }
    best.0
}

#[ignore]
#[test]
fn qwen3_5_moe_forward_seq_last_k_equals_reference() {
    let Some(model_path) =
        common::model_for(&MODEL, "qwen3_5_moe_forward_seq_last_k_equals_reference")
    else {
        return;
    };

    let device = Device::Gpu;
    let model = qwen3_5_moe::load_from_path(&model_path).expect("load MoE model");
    let n_layers = model.cfg.num_hidden_layers;

    // Deterministic coherent-ish token sequence well inside vocab. The exact
    // ids don't matter for the equivalence assertion — only that the two
    // paths see the same ids and the GDN recurrence is non-trivial.
    let ids: Vec<u32> = vec![
        2, 9707, 11, 1879, 13, 576, 6722, 315, 9625, 374, 12095, 13, 576, 6722, 315, 6323, 374,
        26867, 13,
    ];
    let seq = ids.len();
    let k = 5usize;
    assert!(k < seq, "need k < seq for a non-empty prefill split");

    // Build fresh per-layer caches. Unquantized KV keeps the equivalence
    // exact — quant noise would force a much looser tolerance.
    let make_kv = || -> Vec<KvCache> {
        (0..n_layers)
            .map(|_| KvCache::with_quant(KvQuant::None))
            .collect()
    };
    let make_lin =
        || -> Vec<LinearAttnCache> { (0..n_layers).map(|_| LinearAttnCache::new()).collect() };

    // --- Reference: single cold forward over ALL ids, slice last-K. -------
    // Fresh empty caches => equivalent to a no-cache full forward.
    let ref_logits = {
        let mut kv = make_kv();
        let mut lin = make_lin();
        model
            .forward_seq_last_k_with_cache(&ids, k, &mut kv, Some(&mut lin), device)
            .expect("reference forward_seq_last_k_with_cache")
    };
    let ref_vec = logits_to_vec(&ref_logits, device);
    let ref_shape = ref_logits.shape();
    let vocab = ref_shape[2] as usize;
    assert_eq!(ref_shape, &[1, k as i32, vocab as i32]);

    // --- Cached: prefill ids[..seq-k] into the caches (advancing BOTH KV +
    // GDN lin state), then a K-position pass over the tail. -----------------
    let cached_logits = {
        let mut kv = make_kv();
        let mut lin = make_lin();
        let split = seq - k;
        // Prefill the prefix (last_k=1 — only the cache advance matters).
        let _ = model
            .forward_seq_last_k_with_cache(&ids[..split], 1, &mut kv, Some(&mut lin), device)
            .expect("cached prefill");
        // K-position pass over the remaining tail tokens.
        model
            .forward_seq_last_k_with_cache(&ids[split..], k, &mut kv, Some(&mut lin), device)
            .expect("cached last-K pass")
    };
    let cached_vec = logits_to_vec(&cached_logits, device);
    assert_eq!(cached_logits.shape(), &[1, k as i32, vocab as i32]);
    assert_eq!(cached_vec.len(), ref_vec.len());

    // --- Compare per-element within fp tolerance. -------------------------
    // Same arithmetic, different chunking — only fp reassociation differs.
    let mut max_abs_diff = 0.0f32;
    let mut nan_count = 0usize;
    for (a, b) in cached_vec.iter().zip(ref_vec.iter()) {
        if a.is_nan() || b.is_nan() {
            nan_count += 1;
            continue;
        }
        let d = (a - b).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert_eq!(nan_count, 0, "cached/reference logits contain NaN");

    // Primary invariant: the greedy (argmax) token at each of the K positions
    // must match. This is the decode-relevant correctness check and the same
    // bar the gemma4 KV-equivalence tests use (token-id equality, not raw
    // logit closeness). A genuine GDN-split / recurrent-state desync flips
    // argmaxes immediately; fp reassociation noise does not.
    for pos in 0..k {
        let off = pos * vocab;
        let c = argmax_row(&cached_vec[off..off + vocab]);
        let r = argmax_row(&ref_vec[off..off + vocab]);
        assert_eq!(
            c, r,
            "argmax mismatch at K-position {pos} (max_abs_diff={max_abs_diff})"
        );
    }

    // Secondary sanity bound on raw logits. Both paths run identical
    // arithmetic in a different chunking; the only difference is bf16
    // floating-point reassociation over a 248K-wide vocab, which is bounded
    // by a few bf16 ulps on the larger logits. A real desync would produce
    // diffs in the tens/hundreds (and would already have flipped an argmax
    // above), so a small bound here is a meaningful guard.
    assert!(
        max_abs_diff < 2.0,
        "cached last-K logits diverge from reference beyond bf16 noise: \
         max_abs_diff={max_abs_diff}"
    );

    eprintln!(
        "qwen3_5_moe forward_seq_last_k cached==reference OK: \
         seq={seq} k={k} vocab={vocab} max_abs_diff={max_abs_diff}"
    );
}
