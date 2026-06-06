//! Unit tests for [`QuantRotorK4`]. Mirror of `quant_rotor_k3_tests.rs`
#![allow(unsafe_code)]
//! with `bits=4`.

use crate::clifford::make_rotor_table;
use crate::rotorquant::{n_groups_for, rotor4_decode, rotor4_encode};
use crate::storage::quant_rotor_k4::{QuantRotorK4, ROTOR4_K_BITS};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

#[test]
fn quant_rotor_k4_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let max_seq = 64_i32;
    let q = QuantRotorK4::new(init_shape.clone(), max_seq, 0);
    assert_eq!(q.shape, init_shape);
    assert_eq!(q.max_seq, max_seq);
    assert_eq!(q.bits, ROTOR4_K_BITS);
}

#[test]
fn quant_rotor_k4_roundtrip_no_qjl_matches_v_side() {
    let _guard = crate::test_utils::ROTOR_QJL_ENV_LOCK
        .lock()
        .expect("env lock poisoned");
    // SAFETY: ROTOR_QJL_ENV_LOCK held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 9;
    let n_rows = b * kv_h * n_seq;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK4::new(
        vec![b as i32, kv_h as i32, 0_i32, head_dim as i32],
        n_seq as i32,
        0,
    );
    qk.append(&data, &new_shape).unwrap();
    let decoded = qk.dequant().unwrap();

    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(0, 0, n_groups);
    let (ref_codes, ref_scales, ref_norms) = rotor4_encode(&data, &rotors, head_dim).unwrap();
    let reference = rotor4_decode(&ref_codes, &ref_scales, &ref_norms, &rotors, head_dim).unwrap();

    let max_abs_err = decoded
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_err < 1e-5,
        "rotor4_k (no QJL) vs rotor4 V: max_abs_err = {max_abs_err:.6}"
    );

    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

#[test]
fn quant_rotor_k4_reset_clears_seq() {
    let head_dim = 9;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK4::new(vec![1, 1, 0, head_dim as i32], 16, 0);
    qk.append(&data, &new_shape).unwrap();
    qk.reset();
    assert_eq!(qk.shape[2], 0);
    assert!(qk.blocks.is_empty());
}

/// Empirical cosine floor for rotor4_k @ head_dim=128, QJL off.
#[test]
fn quant_rotor_k4_cosine_empirical_floor_head_dim_128_no_qjl() {
    let _guard = crate::test_utils::ROTOR_QJL_ENV_LOCK
        .lock()
        .expect("env lock poisoned");
    // SAFETY: ROTOR_QJL_ENV_LOCK held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let head_dim = 128;
    let n_rows = 16;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    let mut qk = QuantRotorK4::new(vec![1, 1, 0, head_dim as i32], n_rows as i32, 0);
    qk.append(&data, &new_shape).unwrap();
    let decoded = qk.dequant().unwrap();
    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    // 4-bit codec has higher fidelity than 3-bit; gate higher.
    assert!(
        stats.min >= 0.75,
        "rotor4_k cosine min={:.6} below empirical floor 0.75",
        stats.min
    );
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}
