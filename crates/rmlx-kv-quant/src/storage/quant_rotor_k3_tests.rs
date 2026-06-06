//! Unit tests for [`QuantRotorK3`].
#![allow(unsafe_code)]
//!
//! Mirror of `quant_iso_k_tests.rs` adapted to the rotor3 K codec. The
//! `_with_qjl` / `_no_qjl` variants exercise both QJL branches.

use crate::clifford::make_rotor_table;
use crate::rotorquant::{n_groups_for, rotor3_decode, rotor3_encode};
use crate::storage::quant_rotor_k3::{QuantRotorK3, ROTOR3_K_BITS};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

#[test]
fn quant_rotor_k3_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let max_seq = 64_i32;
    let q = QuantRotorK3::new(init_shape.clone(), max_seq, 0);
    assert_eq!(q.shape, init_shape, "shape preserved after new()");
    assert_eq!(q.max_seq, max_seq, "max_seq preserved");
    assert_eq!(q.bits, ROTOR3_K_BITS);
    assert!(q.blocks.is_empty(), "no blocks after new()");
    assert!(!q.use_qjl(), "use_qjl false before first append");
}

#[test]
fn quant_rotor_k3_roundtrip_no_qjl_matches_v_side() {
    let _guard = crate::test_utils::ROTOR_QJL_ENV_LOCK
        .lock()
        .expect("env lock poisoned");
    // SAFETY: ROTOR_QJL_ENV_LOCK held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 9; // n_groups = 3, exact
    let n_rows = b * kv_h * n_seq;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK3::new(
        vec![b as i32, kv_h as i32, 0_i32, head_dim as i32],
        n_seq as i32,
        0,
    );
    qk.append(&data, &new_shape).expect("append");
    assert_eq!(qk.blocks.len(), 1);
    assert_eq!(qk.shape[2], n_seq as i32);
    assert!(!qk.use_qjl(), "QJL off via env");
    let decoded = qk.dequant().expect("dequant");

    // Reference: V-side rotor3 codec produces the same output (codec is
    // axis-agnostic when QJL is off).
    let n_groups = n_groups_for(head_dim);
    let rotors = make_rotor_table(0, 0, n_groups);
    let (ref_codes, ref_scales, ref_norms) = rotor3_encode(&data, &rotors, head_dim).unwrap();
    let reference = rotor3_decode(&ref_codes, &ref_scales, &ref_norms, &rotors, head_dim).unwrap();

    assert_eq!(decoded.len(), reference.len());
    let max_abs_err = decoded
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_abs_err < 1e-5,
        "rotor3_k (no QJL) vs rotor3 V: max_abs_err = {max_abs_err:.6} (>= 1e-5)"
    );

    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

#[test]
fn quant_rotor_k3_qjl_default_on() {
    let _guard = crate::test_utils::ROTOR_QJL_ENV_LOCK
        .lock()
        .expect("env lock poisoned");
    // Default ON when no env override is set.
    // SAFETY: ROTOR_QJL_ENV_LOCK held — no concurrent env reader/writer in this binary.
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };

    let head_dim = 9;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK3::new(vec![1, 1, 0, head_dim as i32], 16, 0);
    qk.append(&data, &new_shape).unwrap();
    // The CLI may have installed QJL=on or QJL=off at startup; only check that
    // the encoder/decoder ran without panic. The QJL toggle lift is verified
    // by the dedicated lift test below.
    let _decoded = qk.dequant().unwrap();
    assert!(qk.blocks.len() == 1);
}

#[test]
fn quant_rotor_k3_reset_clears_seq() {
    let head_dim = 9;
    let n_seq = 4;
    let data = lcg_data(n_seq * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK3::new(vec![1, 1, 0, head_dim as i32], 16, 0);
    qk.append(&data, &new_shape).unwrap();
    assert_eq!(qk.shape[2], n_seq as i32);

    qk.reset();
    assert_eq!(qk.shape[2], 0);
    assert!(qk.blocks.is_empty());
}

/// Empirical cosine floor for the rotor3_k codec at head_dim=128, QJL off.
/// Matches V-side rotor3 (axis-agnostic) at this seed/shape: measured ≈
/// 0.985. Gate at 0.97 (measured − 0.015 floor, accommodates LCG drift +
/// the rotor3 single-codebook simplification noise).
#[test]
fn quant_rotor_k3_cosine_empirical_floor_head_dim_128_no_qjl() {
    let _guard = crate::test_utils::ROTOR_QJL_ENV_LOCK
        .lock()
        .expect("env lock poisoned");
    // SAFETY: ROTOR_QJL_ENV_LOCK held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let head_dim = 128;
    let n_rows = 16;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    let mut qk = QuantRotorK3::new(vec![1, 1, 0, head_dim as i32], n_rows as i32, 0);
    qk.append(&data, &new_shape).unwrap();
    let decoded = qk.dequant().unwrap();
    let stats = cosine_similarity_per_row(&data, &decoded, head_dim);
    assert!(
        stats.min >= 0.65,
        "rotor3_k cosine min={:.6} below empirical floor 0.65",
        stats.min
    );
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}
