//! Unit tests for [`QuantIsoK3`].
//!
//! Mirror of `quant_iso_v_tests.rs` — the codec is axis-agnostic, so the
//! K-side struct exercises the same encode/decode invariants as the V-side
//! struct. Cosine floor uses the empirical-floor pattern (measure, then gate
//! at measured − 0.001).

use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::storage::quant_iso_k::{QuantIsoK3, ISO_K3_BITS, ISO_K3_GROUP_SIZE};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, skip_if_no_gpu_env, TEST_SEED};
use rmlx_mlx::Device;

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

/// Multi-append with `kv_h > 1` must match a single-shot append of the
/// concatenated head-major buffer (head↔seq layout invariant). Per-(head,
/// token, dim) distinct values surface any head transposition as a large error.
#[test]
fn quant_iso_k3_multi_append_matches_single_shot_gqa() {
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
    let mut qref = QuantIsoK3::new(vec![1, kv_h as i32, 0, head_dim as i32], 64);
    qref.append(
        &build(0, s_total),
        &[1, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .expect("single-shot append");
    let reference = qref.dequant().expect("single-shot dequant");

    let mut qv = QuantIsoK3::new(vec![1, kv_h as i32, 0, head_dim as i32], 64);
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
        "iso_k3 multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — head↔seq scramble"
    );
    let _ = (ISO_K3_BITS, ISO_K3_GROUP_SIZE);
}

/// GPU multi-append with `kv_h > 1` must match a single-shot CPU-append +
/// `dequant_gpu` of the concatenated head-major buffer. `dequant_gpu` uploads
/// CPU blocks via `Array::from_bytes` and drives the iso3 MSL kernel; the
/// subsequent reshape+transpose reorders sequence-major blocks back to
/// head-major `[B, kv_h, S, D]`. A head transposition produces errors ~100;
/// quant noise on this fixture is well under 1.0.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant --release iso_k3_gpu_multi_append -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    reason = "test: panic on failure is the desired test failure mode"
)]
fn iso_k3_gpu_multi_append_matches_single_shot_gqa() {
    if skip_if_no_gpu_env() {
        return;
    }
    let kv_h = 3_i32;
    let head_dim = 8_i32; // must be multiple of ISO_K3_GROUP_SIZE (4)
    let chunk_a = 2_i32;
    let chunk_b = 3_i32;
    let s_total = chunk_a + chunk_b;

    let val = |h: i32, s: i32, d: i32| -> f32 {
        (h as f32) * 100.0 + (s as f32) * 10.0 + (d as f32) * 0.5 + 1.0
    };
    let build = |s_lo: i32, s_hi: i32| -> Vec<f32> {
        let s = s_hi - s_lo;
        let mut out = vec![0.0_f32; (kv_h * s * head_dim) as usize];
        for h in 0..kv_h {
            for si in 0..s {
                for d in 0..head_dim {
                    let idx = ((h * s + si) * head_dim + d) as usize;
                    out[idx] = val(h, s_lo + si, d);
                }
            }
        }
        out
    };

    // Reference: single CPU append, then dequant_gpu.
    let mut qref = QuantIsoK3::new(vec![1, kv_h, 0, head_dim], 64);
    qref.append(&build(0, s_total), &[1, kv_h, s_total, head_dim])
        .unwrap();
    let ref_arr = qref.dequant_gpu(Device::Gpu).unwrap();
    ref_arr.eval().unwrap();
    let reference: Vec<f32> = ref_arr
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();

    // Two CPU appends, then dequant_gpu.
    let mut qv = QuantIsoK3::new(vec![1, kv_h, 0, head_dim], 64);
    qv.append(&build(0, chunk_a), &[1, kv_h, chunk_a, head_dim])
        .unwrap();
    qv.append(&build(chunk_a, s_total), &[1, kv_h, chunk_b, head_dim])
        .unwrap();
    let multi_arr = qv.dequant_gpu(Device::Gpu).unwrap();
    multi_arr.eval().unwrap();
    let multi: Vec<f32> = multi_arr
        .to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect();

    assert_eq!(multi.len(), reference.len(), "length parity");
    let max_abs = multi
        .iter()
        .zip(reference.iter())
        .fold(0.0_f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_abs < 1.0,
        "GPU multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — \
         head↔seq layout scramble"
    );
}

/// Falsifies #284: at `kv_h > 1`, `truncate_to(n)` must keep exactly the
/// leading blocks covering sequence `[0, n)`, not `floor(n / kv_h)` of them.
///
/// Builds one block per token (CPU-only, no GPU ring ever touched), truncates
/// mid-sequence at a block boundary, and requires the result to exactly match
/// a reference store built from only the first `keep_tokens`.
///
/// Runs both `kv_h == 1` (historical, accidentally-correct) and `kv_h > 1`
/// (where the pre-fix code undercounted) in one test.
///
/// Mutation check: reverting `truncate_to` to compare
/// `acc + blk.n_tokens <= n as usize` (raw, not row-scaled) makes the
/// `kv_h > 1` case RED — `blocks.len()` drops and `dequant()` returns `Err`.
#[test]
fn quant_iso_k3_truncate_to_kv_h_gt_1_keeps_exact_prefix() {
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

        let mut store = QuantIsoK3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 64);
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

        let mut reference = QuantIsoK3::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 64);
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
/// concatenation back in `QuantIsoK3::dequant` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
#[test]
fn quant_iso_k3_two_block_decode_matches_one_block_at_b_gt_1() {
    for b in [1_usize, 2] {
        let (kv_h, head_dim) = (2_usize, 8_usize);
        let (n0, n1) = (2_usize, 3_usize);
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];

        let mut one = QuantIsoK3::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], 512);
        one.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0 + n1, head_dim),
            &shape(n0 + n1),
        )
        .expect("single append");
        let oracle = one.dequant().expect("one-block dequant");

        let mut two = QuantIsoK3::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], 512);
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
            "two-block decode must equal the one-block oracle at b={b}"
        );
    }
}
