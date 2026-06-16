//! Unit tests for [`QuantIsoV4`].

use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::storage::quant_iso_v4::{QuantIsoV4, ISO4_BITS, ISO4_GROUP_SIZE};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

/// Newly-constructed `QuantIsoV4` carries the requested init shape and bit
/// width; no blocks yet.
#[test]
fn quant_iso_v4_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let max_seq = 64_i32;
    let q = QuantIsoV4::new(init_shape.clone(), max_seq);
    assert_eq!(q.shape, init_shape, "shape preserved after new()");
    assert_eq!(q.max_seq, max_seq, "max_seq preserved");
    assert_eq!(q.bits, ISO4_BITS, "bits should be ISO4_BITS (4)");
    assert!(q.blocks.is_empty(), "no blocks after new()");
    assert_eq!(q.byte_size(), 0, "byte_size 0 with no blocks");
}

/// Roundtrip: encode → store in QuantIsoV4 → `dequant` → compare against the
/// raw `iso_decode_fast` reference (bits=4). Equal element-by-element.
#[test]
fn quant_iso_v4_roundtrip_dequant() {
    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 8;
    let n_rows = b * kv_h * n_seq;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);

    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qv = QuantIsoV4::new(
        vec![b as i32, kv_h as i32, 0_i32, head_dim as i32],
        n_seq as i32,
    );
    qv.append(&data, &new_shape).expect("append should succeed");

    assert_eq!(qv.blocks.len(), 1, "one append → one block");
    assert_eq!(qv.shape[2], n_seq as i32, "shape[2] advanced by n_seq");
    assert!(
        qv.byte_size() > 0,
        "byte_size should be non-zero after append"
    );

    let decoded = qv.dequant().expect("dequant should succeed");

    let (ref_codes, ref_scales, ref_quats, ref_norms) =
        iso_encode_fast(&data, head_dim, ISO4_GROUP_SIZE, ISO4_BITS).expect("encode reference");
    let reference = iso_decode_fast(
        &ref_codes,
        &ref_scales,
        &ref_quats,
        &ref_norms,
        head_dim,
        ISO4_GROUP_SIZE,
        ISO4_BITS,
    )
    .expect("decode reference");

    assert_eq!(
        decoded.len(),
        reference.len(),
        "QuantIsoV4::dequant length should match iso_decode_fast"
    );

    let mut max_abs_err = 0.0_f32;
    for (a, b) in decoded.iter().zip(reference.iter()) {
        let d = (a - b).abs();
        if d > max_abs_err {
            max_abs_err = d;
        }
    }
    assert!(
        max_abs_err < 1e-3,
        "QuantIsoV4::dequant vs iso_decode_fast max_abs_err = {max_abs_err:.6} (>= 1e-3)"
    );
}

/// After `append` then `reset`, the storage reports seq = 0.
#[test]
fn quant_iso_v4_reset_clears_seq() {
    let head_dim = 8;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qv = QuantIsoV4::new(vec![1, 1, 0, head_dim as i32], 16);
    qv.append(&data, &new_shape).unwrap();
    assert_eq!(qv.shape[2], n_seq as i32);

    qv.reset();
    assert_eq!(qv.shape[2], 0, "shape[2] reset to 0");
    assert!(qv.blocks.is_empty(), "blocks cleared on reset");
}

/// After `append` of N tokens then `truncate_to(N/2)`, the dequant prefix
/// retains the first N/2 tokens (when appended one token per call).
#[test]
fn quant_iso_v4_truncate_to_keeps_first_n() {
    let head_dim = 8;
    let n_seq_each = 1;
    let total_tokens = 4;
    let data_full = lcg_data(total_tokens * head_dim, TEST_SEED);

    let mut qv = QuantIsoV4::new(vec![1, 1, 0, head_dim as i32], 16);
    for tok in 0..total_tokens {
        let row = &data_full[tok * head_dim..(tok + 1) * head_dim];
        let new_shape = [1_i32, 1, n_seq_each, head_dim as i32];
        qv.append(row, &new_shape).unwrap();
    }
    assert_eq!(qv.shape[2], total_tokens as i32);
    assert_eq!(qv.blocks.len(), total_tokens);

    let keep = (total_tokens / 2) as i32;
    qv.truncate_to(keep);
    assert_eq!(
        qv.shape[2], keep,
        "shape[2] should be `keep` after truncate"
    );
    assert_eq!(
        qv.blocks.len(),
        keep as usize,
        "block count should equal `keep` when one token per block"
    );

    let decoded = qv.dequant().unwrap();
    let prefix = &data_full[..(keep as usize) * head_dim];

    // Cosine match within tight tolerance — iso4 (4-bit) yields higher fidelity
    // than iso3, so the 0.99 floor inherited from iso3 is comfortably met.
    let stats = cosine_similarity_per_row(prefix, &decoded, head_dim);
    assert!(
        stats.min >= 0.99_f32,
        "truncated-prefix cosine min={:.6} below 0.99",
        stats.min
    );
}

/// Multi-append with `kv_h > 1` must match a single-shot append of the
/// concatenated head-major buffer (head↔seq layout invariant). Per-(head,
/// token, dim) distinct values surface any head transposition as a large error.
#[test]
fn quant_iso_v4_multi_append_matches_single_shot_gqa() {
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
    let mut qref = QuantIsoV4::new(vec![1, kv_h as i32, 0, head_dim as i32], 64);
    qref.append(
        &build(0, s_total),
        &[1, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .expect("single-shot append");
    let reference = qref.dequant().expect("single-shot dequant");

    let mut qv = QuantIsoV4::new(vec![1, kv_h as i32, 0, head_dim as i32], 64);
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
        "iso_v4 multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — head↔seq scramble"
    );
    let _ = (ISO4_BITS, ISO4_GROUP_SIZE);
}
