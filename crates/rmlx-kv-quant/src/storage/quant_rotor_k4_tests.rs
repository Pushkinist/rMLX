//! Unit tests for [`QuantRotorK4`]. Mirror of `quant_rotor_k3_tests.rs`
//! with `bits=4`.
#![allow(unsafe_code)]

use crate::clifford::make_rotor_table;
use crate::rotorquant::{n_groups_for, rotor4_decode, rotor4_encode};
use crate::storage::quant_rotor_k4::{QuantRotorK4, ROTOR4_K_BITS};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

#[test]
fn quant_rotor_k4_new_shapes_correct() {
    let init_shape = vec![1_i32, 4, 0, 128];
    let q = QuantRotorK4::new(init_shape.clone(), 0);
    assert_eq!(q.shape, init_shape);
    assert_eq!(q.bits, ROTOR4_K_BITS);
}

#[test]
fn quant_rotor_k4_roundtrip_no_qjl_matches_v_side() {
    let _guard = crate::test_utils::env_lock();
    // SAFETY: env lock held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let b = 1;
    let kv_h = 2;
    let n_seq = 4;
    let head_dim = 9;
    let n_rows = b * kv_h * n_seq;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [b as i32, kv_h as i32, n_seq as i32, head_dim as i32];

    let mut qk = QuantRotorK4::new(vec![b as i32, kv_h as i32, 0_i32, head_dim as i32], 0);
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

    let mut qk = QuantRotorK4::new(vec![1, 1, 0, head_dim as i32], 0);
    qk.append(&data, &new_shape).unwrap();
    qk.reset();
    assert_eq!(qk.shape[2], 0);
    assert!(qk.blocks.is_empty());
}

/// Empirical cosine floor for rotor4_k @ head_dim=128, QJL off.
#[test]
fn quant_rotor_k4_cosine_empirical_floor_head_dim_128_no_qjl() {
    let _guard = crate::test_utils::env_lock();
    // SAFETY: env lock held — no concurrent env reader/writer in this binary.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let head_dim = 128;
    let n_rows = 16;
    let data = lcg_data(n_rows * head_dim, TEST_SEED);
    let new_shape = [1_i32, 1, n_rows as i32, head_dim as i32];

    let mut qk = QuantRotorK4::new(vec![1, 1, 0, head_dim as i32], 0);
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

/// Multi-append with `kv_h > 1` must match a single-shot append of the
/// concatenated head-major buffer (head↔seq layout invariant), QJL ON. Static
/// rotor/QJL projection are group/projection-keyed; per-token QJL sideband
/// reorders with the token rows. Distinct per-(head, token, dim) values surface
/// any head transposition as a large error.
#[test]
fn quant_rotor_k4_multi_append_matches_single_shot_gqa_with_qjl() {
    let _guard = crate::test_utils::env_lock();
    // SAFETY: env lock held — no concurrent env reader/writer in this binary.
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
    let mut qref = QuantRotorK4::new(vec![1, kv_h as i32, 0, head_dim as i32], 7);
    qref.append(
        &build(0, s_total),
        &[1, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .unwrap();
    assert!(qref.use_qjl(), "QJL must be ON for this test");
    let reference = qref.dequant().unwrap();

    let mut qv = QuantRotorK4::new(vec![1, kv_h as i32, 0, head_dim as i32], 7);
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
        "rotor4_k (QJL on) multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — head↔seq scramble"
    );
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

/// Falsifies #284: at `kv_h > 1`, `truncate_to(n)` must keep exactly the
/// leading blocks covering sequence `[0, n)`, not `floor(n / kv_h)` of them.
/// CPU-only (no GPU ring touched). See `quant_rotor_k3_tests.rs` for the
/// full rationale + mutation-check note.
#[test]
fn quant_rotor_k4_truncate_to_kv_h_gt_1_keeps_exact_prefix() {
    let _guard = crate::test_utils::env_lock();
    // SAFETY: env lock held — no concurrent env reader/writer.
    unsafe { std::env::set_var("RMLX_ROTOR_QJL", "0") };

    let head_dim = 9_usize;
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

        let mut store = QuantRotorK4::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 5);
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

        let mut reference = QuantRotorK4::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 5);
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

    // SAFETY: env lock still held.
    unsafe { std::env::remove_var("RMLX_ROTOR_QJL") };
}

// ── Batch-axis block-boundary parity ──────────────────────────────────

/// Two appends must decode exactly like one append of the same tokens, at
/// `B > 1` as well as `B == 1`.
///
/// Each block covers `[B, S_block, kv_h, D]`, so the concatenation of two
/// blocks is not one `[B, S_total, kv_h, D]` run — reading it as one maps the
/// second block's batch-0 rows onto batch-1 sequence slots. The single-append
/// store holds exactly one block and therefore concatenates nothing, which is
/// what makes it the oracle here.
///
/// Mutation check: put `seq_layout::transpose_seq_heads` over the whole
/// concatenation back in `QuantRotorK4::dequant` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
#[test]
fn quant_rotor_k4_two_block_decode_matches_one_block_at_b_gt_1() {
    // The QJL sideband is read from the process env at each `append`, so a
    // concurrent env-mutating test could otherwise encode the two stores under
    // different settings. Hold the lock and pin both settings explicitly — the
    // per-token sideband has to reorder with its token rows either way.
    let _guard = crate::test_utils::env_lock();
    for (b, kv_h, qjl) in [
        (1_usize, 1_usize, "0"),
        (1, 2, "0"),
        (2, 1, "0"),
        (2, 2, "0"),
        (1, 2, "1"),
        (2, 2, "1"),
    ] {
        // SAFETY: env lock held — no concurrent env reader/writer in this binary.
        unsafe { std::env::set_var("RMLX_ROTOR_QJL", qjl) };
        let head_dim = 128_usize;
        let (n0, n1) = (2_usize, 3_usize);
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];

        let mut one = QuantRotorK4::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], 5);
        one.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0 + n1, head_dim),
            &shape(n0 + n1),
        )
        .expect("single append");
        let oracle = one.dequant().expect("one-block dequant");

        let mut two = QuantRotorK4::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], 5);
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0, head_dim),
            &shape(n0),
        )
        .expect("append chunk 0");
        two.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, n0, n1, head_dim),
            &shape(n1),
        )
        .expect("append chunk 1");
        let got = two.dequant().expect("two-block dequant");

        assert_eq!(
            got, oracle,
            "two-block decode must equal the one-block oracle at b={b} kv_h={kv_h}, qjl={qjl}"
        );
    }
}
