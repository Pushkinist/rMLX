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
