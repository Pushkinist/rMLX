//! Unit tests for [`QuantIsoK4`].

use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::storage::quant_iso_k4::{QuantIsoK4, ISO_K4_BITS, ISO_K4_GROUP_SIZE};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

#[test]
fn quant_iso_k4_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let max_seq = 64_i32;
    let q = QuantIsoK4::new(init_shape.clone(), max_seq);
    assert_eq!(q.shape, init_shape);
    assert_eq!(q.max_seq, max_seq);
    assert_eq!(q.bits, ISO_K4_BITS, "bits should be ISO_K4_BITS (4)");
    assert!(q.blocks.is_empty());
    assert_eq!(q.byte_size(), 0);
}

#[test]
fn quant_iso_k4_roundtrip_dequant() {
    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 8;
    let n_rows = b * kv_h * n_seq;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);

    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qk = QuantIsoK4::new(
        vec![b as i32, kv_h as i32, 0_i32, head_dim as i32],
        n_seq as i32,
    );
    qk.append(&data, &new_shape).expect("append should succeed");

    assert_eq!(qk.blocks.len(), 1);
    assert_eq!(qk.shape[2], n_seq as i32);
    assert!(qk.byte_size() > 0);

    let decoded = qk.dequant().expect("dequant should succeed");

    let (ref_codes, ref_scales, ref_quats, ref_norms) =
        iso_encode_fast(&data, head_dim, ISO_K4_GROUP_SIZE, ISO_K4_BITS).expect("encode reference");
    let reference = iso_decode_fast(
        &ref_codes,
        &ref_scales,
        &ref_quats,
        &ref_norms,
        head_dim,
        ISO_K4_GROUP_SIZE,
        ISO_K4_BITS,
    )
    .expect("decode reference");

    assert_eq!(decoded.len(), reference.len());
    let max_abs_err = decoded
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_err < 1e-3,
        "QuantIsoK4::dequant vs iso_decode_fast max_abs_err = {max_abs_err:.6} (>= 1e-3)"
    );
}

#[test]
fn quant_iso_k4_reset_clears_seq() {
    let head_dim = 8;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantIsoK4::new(vec![1, 1, 0, head_dim as i32], 16);
    qk.append(&data, &new_shape).unwrap();
    assert_eq!(qk.shape[2], n_seq as i32);

    qk.reset();
    assert_eq!(qk.shape[2], 0);
    assert!(qk.blocks.is_empty());
}

/// Empirical cosine floor for iso_k4 at head_dim=128. 4-bit codebook
/// improves fidelity over iso3, so floor is set above iso_k3.
#[test]
fn quant_iso_k4_cosine_empirical_floor_head_dim_128() {
    let head_dim = 128;
    let n_rows = 16;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    let mut qk = QuantIsoK4::new(vec![1, 1, 0, head_dim as i32], n_rows as i32);
    qk.append(&data, &new_shape).unwrap();
    let decoded = qk.dequant().unwrap();

    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    // 4-bit Lloyd-Max + SO(4) rotation: cosine ≥ 0.99 on Gaussian-like data.
    assert!(
        stats.min >= 0.99,
        "iso_k4 cosine min={:.6} below empirical floor 0.99",
        stats.min
    );
}

/// Multi-append with `kv_h > 1` must match a single-shot append of the
/// concatenated head-major buffer (head↔seq layout invariant). Per-(head,
/// token, dim) distinct values surface any head transposition as a large error.
#[test]
fn quant_iso_k4_multi_append_matches_single_shot_gqa() {
    let kv_h = 3_usize;
    let head_dim = 8_usize;
    let chunk_a = 2_usize;
    let chunk_b = 3_usize;
    let s_total = chunk_a + chunk_b;
    let val = |h: usize, s: usize, d: usize| {
        (h as f32) * 100.0 + (s as f32) * 10.0 + (d as f32) * 0.5 + 1.0
    };
    let build = |s_lo: usize, s_hi: usize| -> Vec<f32> {
        let s = s_hi - s_lo;
        let mut out = vec![0.0_f32; kv_h * s * head_dim];
        for h in 0..kv_h {
            for si in 0..s {
                for d in 0..head_dim {
                    out[(h * s + si) * head_dim + d] = val(h, s_lo + si, d);
                }
            }
        }
        out
    };
    let mut qref = QuantIsoK4::new(vec![1, kv_h as i32, 0, head_dim as i32], 64);
    qref.append(
        &build(0, s_total),
        &[1, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .expect("single-shot append");
    let reference = qref.dequant().expect("single-shot dequant");

    let mut qv = QuantIsoK4::new(vec![1, kv_h as i32, 0, head_dim as i32], 64);
    qv.append(
        &build(0, chunk_a),
        &[1, kv_h as i32, chunk_a as i32, head_dim as i32],
    )
    .expect("append A");
    qv.append(
        &build(chunk_a, s_total),
        &[1, kv_h as i32, chunk_b as i32, head_dim as i32],
    )
    .expect("append B");
    let multi = qv.dequant().expect("multi dequant");

    assert_eq!(multi.len(), reference.len());
    let max_abs = multi
        .iter()
        .zip(reference.iter())
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_abs < 1.0,
        "iso_k4 multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — head↔seq scramble"
    );
    let _ = (ISO_K4_BITS, ISO_K4_GROUP_SIZE);
}

/// Falsifies #284: at `kv_h > 1`, `truncate_to(n)` must keep exactly the
/// leading blocks covering sequence `[0, n)`, not `floor(n / kv_h)` of them.
/// CPU-only (no GPU ring touched). See `quant_iso_k_tests.rs` for the full
/// rationale + mutation-check note.
#[test]
fn quant_iso_k4_truncate_to_kv_h_gt_1_keeps_exact_prefix() {
    let head_dim = 8_usize;
    let total_tokens = 4_usize;
    let keep_tokens = 2_usize;
    let val = |h: usize, tok: usize, d: usize| {
        (h as f32) * 100.0 + (tok as f32) * 10.0 + (d as f32) * 0.5 + 1.0
    };

    for kv_h in [1_usize, 4_usize] {
        let token_data = |tok: usize| -> Vec<f32> {
            let mut out = vec![0.0_f32; kv_h * head_dim];
            for h in 0..kv_h {
                for d in 0..head_dim {
                    out[h * head_dim + d] = val(h, tok, d);
                }
            }
            out
        };
        let new_shape = [1_i32, kv_h as i32, 1, head_dim as i32];

        let mut store = QuantIsoK4::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 64);
        for tok in 0..total_tokens {
            store.append(&token_data(tok), &new_shape).unwrap();
        }
        assert_eq!(
            store.blocks.len(),
            total_tokens,
            "one block per token append (kv_h={kv_h})"
        );

        store.truncate_to(keep_tokens as i32);

        assert_eq!(
            store.shape[2], keep_tokens as i32,
            "shape[2] must equal keep_tokens (kv_h={kv_h})"
        );
        assert_eq!(
            store.blocks.len(),
            keep_tokens,
            "truncate_to must keep exactly keep_tokens blocks, not floor(keep_tokens / kv_h) (kv_h={kv_h})"
        );
        let kept_rows: usize = store.blocks.iter().map(|blk| blk.n_tokens).sum();
        assert_eq!(
            kept_rows,
            keep_tokens * kv_h,
            "kept rows must equal keep_tokens * b * kv_h (kv_h={kv_h})"
        );

        let decoded = store
            .dequant()
            .expect("dequant must succeed after truncate at kv_h>1 (#284)");

        let mut reference = QuantIsoK4::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 64);
        for tok in 0..keep_tokens {
            reference.append(&token_data(tok), &new_shape).unwrap();
        }
        let ref_decoded = reference.dequant().unwrap();

        assert_eq!(
            decoded, ref_decoded,
            "truncated store must exactly match a store built from only the \
             first keep_tokens (kv_h={kv_h})"
        );
    }
}
