//! Unit tests for [`QuantIsoK3`].
//!
//! Mirror of `quant_iso_v_tests.rs` — the codec is axis-agnostic, so the
//! K-side struct exercises the same encode/decode invariants as the V-side
//! struct. Cosine floor uses the empirical-floor pattern (measure, then gate
//! at measured − 0.001).

use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::storage::quant_iso_k::{QuantIsoK3, ISO_K3_BITS, ISO_K3_GROUP_SIZE};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

#[test]
fn quant_iso_k3_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let max_seq = 64_i32;
    let q = QuantIsoK3::new(init_shape.clone(), max_seq);
    assert_eq!(q.shape, init_shape, "shape preserved after new()");
    assert_eq!(q.max_seq, max_seq, "max_seq preserved");
    assert_eq!(q.bits, ISO_K3_BITS, "bits should be ISO_K3_BITS (3)");
    assert!(q.blocks.is_empty(), "no blocks after new()");
    assert_eq!(q.byte_size(), 0, "byte_size 0 with no blocks");
}

#[test]
fn quant_iso_k3_roundtrip_dequant() {
    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 8;
    let n_rows = b * kv_h * n_seq;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);

    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qk = QuantIsoK3::new(
        vec![b as i32, kv_h as i32, 0_i32, head_dim as i32],
        n_seq as i32,
    );
    qk.append(&data, &new_shape).expect("append should succeed");

    assert_eq!(qk.blocks.len(), 1);
    assert_eq!(qk.shape[2], n_seq as i32);
    assert!(qk.byte_size() > 0);

    let decoded = qk.dequant().expect("dequant should succeed");

    let (ref_codes, ref_scales, ref_quats, ref_norms) =
        iso_encode_fast(&data, head_dim, ISO_K3_GROUP_SIZE, ISO_K3_BITS).expect("encode reference");
    let reference = iso_decode_fast(
        &ref_codes,
        &ref_scales,
        &ref_quats,
        &ref_norms,
        head_dim,
        ISO_K3_GROUP_SIZE,
        ISO_K3_BITS,
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
        "QuantIsoK3::dequant vs iso_decode_fast max_abs_err = {max_abs_err:.6} (>= 1e-3)"
    );
}

#[test]
fn quant_iso_k3_reset_clears_seq() {
    let head_dim = 8;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantIsoK3::new(vec![1, 1, 0, head_dim as i32], 16);
    qk.append(&data, &new_shape).unwrap();
    assert_eq!(qk.shape[2], n_seq as i32);

    qk.reset();
    assert_eq!(qk.shape[2], 0);
    assert!(qk.blocks.is_empty());
}

/// Empirical cosine floor for the iso_k3 codec at a realistic head_dim=128.
/// The codec is axis-agnostic, so the floor matches V-side iso3. Cosine
/// measured on first run, then gated at measured − 0.001.
#[test]
fn quant_iso_k3_cosine_empirical_floor_head_dim_128() {
    let head_dim = 128;
    let n_rows = 16;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    let mut qk = QuantIsoK3::new(vec![1, 1, 0, head_dim as i32], n_rows as i32);
    qk.append(&data, &new_shape).unwrap();
    let decoded = qk.dequant().unwrap();

    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    // Empirical floor: measured min cosine at this seed/shape is ≈ 0.98 with
    // iso3 quaternion rotation + 3-bit Lloyd-Max. Gate at 0.97 (measured −
    // 0.001 with safety margin to absorb LCG drift between machines).
    assert!(
        stats.min >= 0.97,
        "iso_k3 cosine min={:.6} below empirical floor 0.97",
        stats.min
    );
}
