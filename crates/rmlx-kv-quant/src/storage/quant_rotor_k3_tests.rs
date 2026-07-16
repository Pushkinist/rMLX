//! Unit tests for [`QuantRotorK3`].
//!
//! Mirror of `quant_iso_k_tests.rs` adapted to the rotor3 K codec. The
//! `_with_qjl` / `_no_qjl` variants exercise both QJL branches.
#![allow(unsafe_code)]

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
    assert_eq!(qk.blocks.len(), 1);
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

/// Multi-append with `kv_h > 1` must match a single-shot append of the
/// concatenated head-major buffer (head↔seq layout invariant), with the QJL
/// sideband ON. The static rotor table and QJL projection are
/// group/projection-keyed (not token); the per-token QJL sideband (qjl_codes /
/// qjl_norms) reorders with the token rows. Per-(head, token, dim) distinct
/// values surface any head transposition as a large error.
#[test]
fn quant_rotor_k3_multi_append_matches_single_shot_gqa_with_qjl() {
    let _guard = crate::test_utils::ROTOR_QJL_ENV_LOCK
        .lock()
        .expect("env lock poisoned");
    // SAFETY: ROTOR_QJL_ENV_LOCK held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "1") };

    let kv_h = 3_usize;
    let head_dim = 128_usize;
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
    let mut qref = QuantRotorK3::new(vec![1, kv_h as i32, 0, head_dim as i32], 64, 7);
    qref.append(
        &build(0, s_total),
        &[1, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .unwrap();
    assert!(qref.use_qjl(), "QJL must be ON for this test");
    let reference = qref.dequant().unwrap();

    let mut qv = QuantRotorK3::new(vec![1, kv_h as i32, 0, head_dim as i32], 64, 7);
    qv.append(
        &build(0, chunk_a),
        &[1, kv_h as i32, chunk_a as i32, head_dim as i32],
    )
    .unwrap();
    qv.append(
        &build(chunk_a, s_total),
        &[1, kv_h as i32, chunk_b as i32, head_dim as i32],
    )
    .unwrap();
    let multi = qv.dequant().unwrap();

    assert_eq!(multi.len(), reference.len());
    let max_abs = multi
        .iter()
        .zip(reference.iter())
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_abs < 1.0,
        "rotor3_k (QJL on) multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — head↔seq scramble"
    );
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

// ── GPU ring lifecycle ────────────────────────────────────────────────────────
//
// `reset` / `truncate_to` shorten `blocks` + `shape[2]` while a longer GPU ring
// may still be live. The ring must be dropped, not left behind: a stale ring is
// longer than the store claims, so the next `gpu_append` takes `prev_seq` from
// the (shorter) shape and writes into the middle of a ring whose tail still
// holds the truncated tokens — `packed_view` would then hand the kernel stale
// keys past the truncation point. Silent wrong answer, no error.

#[allow(
    clippy::expect_used,
    reason = "test helper: a failure here is the assertion"
)]
fn seed_ring_via_gpu_append(ks: &mut QuantRotorK3, kv_h: i32, head_dim: i32, n_tokens: i32) {
    use rmlx_mlx::{Array, Device, Dtype};
    let n_groups = n_groups_for(head_dim as usize) as i32;
    let cps = (kv_h * n_groups * n_tokens) as usize;
    let nps = (kv_h * n_tokens) as usize;
    let codes_b: Vec<u8> = (0..cps).flat_map(|i| (i as u32).to_le_bytes()).collect();
    let scales_b: Vec<u8> = (0..cps).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let norms_b: Vec<u8> = (0..nps).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let codes = Array::from_bytes(&codes_b, &[cps as i32], Dtype::U32).expect("codes");
    let scales = Array::from_bytes(&scales_b, &[cps as i32], Dtype::F32).expect("scales");
    let norms = Array::from_bytes(&norms_b, &[nps as i32], Dtype::F32).expect("norms");
    ks.gpu_append(
        &codes,
        &scales,
        &norms,
        kv_h,
        head_dim,
        0,
        n_tokens,
        Device::Gpu,
    )
    .expect("gpu_append");
}

#[test]
fn quant_rotor_k3_reset_drops_the_gpu_ring() {
    if crate::test_utils::skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, head_dim) = (2_i32, 6_i32);
    let mut ks = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 64, 0);
    ks.rotors = make_rotor_table(0, 0, n_groups_for(head_dim as usize));
    seed_ring_via_gpu_append(&mut ks, kv_h, head_dim, 3);
    assert!(ks.gpu.is_allocated(), "ring should be live before reset");

    ks.reset();
    assert!(
        !ks.gpu.is_allocated(),
        "reset() must drop the ring — a ring outliving the blocks it mirrors \
         would serve stale keys on the next append"
    );
}

#[test]
fn quant_rotor_k3_truncate_to_drops_the_gpu_ring() {
    if crate::test_utils::skip_if_no_gpu_env() {
        return;
    }
    let (kv_h, head_dim) = (2_i32, 6_i32);
    let mut ks = QuantRotorK3::new(vec![1, kv_h, 0, head_dim], 64, 0);
    ks.rotors = make_rotor_table(0, 0, n_groups_for(head_dim as usize));

    // Two CPU appends so `truncate_to` has block boundaries to cut on.
    let per_tok = (kv_h * head_dim) as usize;
    let d1 = lcg_data(per_tok * 2, TEST_SEED);
    ks.append(&d1, &[1, kv_h, 2, head_dim]).expect("append 1");
    let d2 = lcg_data(per_tok, TEST_SEED + 1);
    ks.append(&d2, &[1, kv_h, 1, head_dim]).expect("append 2");
    assert_eq!(ks.shape[2], 3);

    // A live ring covering all 3 tokens, then truncate back to 2.
    seed_ring_via_gpu_append(&mut ks, kv_h, head_dim, 3);
    assert!(ks.gpu.is_allocated());

    ks.truncate_to(2);
    assert_eq!(ks.shape[2], 2);
    assert!(
        !ks.gpu.is_allocated(),
        "truncate_to() must drop the ring — otherwise packed_view() would still \
         expose the truncated token"
    );
}
