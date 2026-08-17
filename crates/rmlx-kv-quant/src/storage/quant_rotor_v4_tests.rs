//! Tests for [`QuantRotorV4`].
#![allow(
    clippy::identity_op,
    reason = "explicit `B * kv_h * seq * head_dim` element counts kept as-is for readability — `1 * 2 * n_tokens * head_dim` mirrors the canonical [B, kv_h, S, D] shape literally"
)]

use crate::storage::quant_rotor_v4::{QuantRotorV4, ROTOR4_V_BITS};

/// Newly-constructed `QuantRotorV4` carries the requested init shape and bit
/// tag; no rotors yet (lazily generated on first append).
#[test]
fn quant_rotor_v4_new_carries_shape_and_bits() {
    let shape = vec![1_i32, 2, 0, 96];
    let qv = QuantRotorV4::new(shape.clone(), 4096, 7);
    assert_eq!(qv.shape, shape, "shape preserved");
    assert_eq!(qv.max_seq, 4096, "max_seq preserved");
    assert_eq!(qv.bits, ROTOR4_V_BITS, "bits tag");
    assert_eq!(qv.layer_idx, 7, "layer_idx preserved");
    assert!(qv.rotors.is_empty(), "rotors empty before first append");
    assert!(qv.blocks.is_empty(), "no blocks before append");
}

/// First `append` lazily generates the rotor table (`n_groups * 4` f32).
#[test]
fn quant_rotor_v4_append_generates_rotor_table() {
    let mut qv = QuantRotorV4::new(vec![1_i32, 2, 0, 96], 4096, 0);
    let n_tokens = 4;
    let head_dim = 96;
    let n: usize = 1 * 2 * n_tokens * head_dim;
    let data = vec![0.5_f32; n];
    let new_shape = [1_i32, 2, n_tokens as i32, head_dim as i32];
    qv.append(&data, &new_shape).unwrap();

    let expected_rotors_len = (head_dim / 3) * 4; // head_dim divisible by 3
    assert_eq!(
        qv.rotors.len(),
        expected_rotors_len,
        "rotor table generated with n_groups * 4 entries"
    );
    assert_eq!(qv.shape, new_shape, "shape advanced by first append");
    assert_eq!(qv.blocks.len(), 1, "one block per append call");
}

/// `truncate_to` drops trailing blocks but keeps the rotor table.
#[test]
fn quant_rotor_v4_truncate_keeps_rotors() {
    let mut qv = QuantRotorV4::new(vec![1_i32, 1, 0, 96], 4096, 3);
    let head_dim = 96;
    let new_shape = [1_i32, 1, 4, head_dim as i32];
    let data = vec![0.3_f32; 1 * 1 * 4 * head_dim];
    qv.append(&data, &new_shape).unwrap();
    let rotors_before = qv.rotors.clone();

    qv.truncate_to(0);
    assert!(qv.blocks.is_empty(), "blocks dropped on truncate");
    assert_eq!(qv.rotors, rotors_before, "rotor table preserved");
    assert_eq!(qv.shape[2], 0, "seq dim truncated");
}

/// `reset` clears blocks but keeps the rotor table.
#[test]
fn quant_rotor_v4_reset_keeps_rotors() {
    let mut qv = QuantRotorV4::new(vec![1_i32, 1, 0, 96], 4096, 2);
    let head_dim = 96;
    let new_shape = [1_i32, 1, 4, head_dim as i32];
    let data = vec![0.3_f32; 1 * 1 * 4 * head_dim];
    qv.append(&data, &new_shape).unwrap();
    let rotors_before = qv.rotors.clone();

    qv.reset();
    assert!(qv.blocks.is_empty(), "blocks dropped on reset");
    assert_eq!(qv.rotors, rotors_before, "rotor table preserved");
    assert_eq!(qv.shape[2], 0, "seq dim reset to 0");
}

/// `byte_size` counts rotors exactly once + accumulated blocks.
#[test]
fn quant_rotor_v4_byte_size_counts_rotors_once() {
    let mut qv = QuantRotorV4::new(vec![1_i32, 1, 0, 96], 4096, 0);
    let head_dim = 96;
    let new_shape = [1_i32, 1, 4, head_dim as i32];
    let data = vec![0.5_f32; 1 * 1 * 4 * head_dim];
    qv.append(&data, &new_shape).unwrap();

    let rotors_bytes = (qv.rotors.len() * size_of::<f32>()) as u64;
    let blocks_bytes = qv
        .blocks
        .iter()
        .map(|b| (b.codes.len() * 4 + b.scales.len() * 4 + b.norms.len() * 4) as u64)
        .sum::<u64>();
    assert_eq!(qv.byte_size(), rotors_bytes + blocks_bytes);

    // Appending a second block must not double-count rotors.
    qv.append(&data, &new_shape).unwrap();
    let new_blocks_bytes = qv
        .blocks
        .iter()
        .map(|b| (b.codes.len() * 4 + b.scales.len() * 4 + b.norms.len() * 4) as u64)
        .sum::<u64>();
    assert_eq!(qv.byte_size(), rotors_bytes + new_blocks_bytes);
}

/// `try_deep_clone` clones rotors + blocks + meta.
#[test]
fn quant_rotor_v4_deep_clone() {
    let mut qv = QuantRotorV4::new(vec![1_i32, 1, 0, 96], 4096, 1);
    let head_dim = 96;
    let new_shape = [1_i32, 1, 2, head_dim as i32];
    let data = vec![0.1_f32; 1 * 1 * 2 * head_dim];
    qv.append(&data, &new_shape).unwrap();

    let cloned = qv.try_deep_clone().unwrap();
    assert_eq!(cloned.rotors, qv.rotors);
    assert_eq!(cloned.shape, qv.shape);
    assert_eq!(cloned.blocks.len(), qv.blocks.len());
    assert_eq!(cloned.blocks[0].codes, qv.blocks[0].codes);
    assert_eq!(cloned.layer_idx, qv.layer_idx);
    assert_eq!(cloned.bits, ROTOR4_V_BITS, "bits tag preserved in clone");
}

/// Multi-append with `kv_h > 1` must match a single-shot append of the
/// concatenated head-major buffer (head↔seq layout invariant). The static
/// rotor table is group-position-keyed (not token); per-(head, token, dim)
/// distinct values surface any head transposition as a large error.
#[test]
fn quant_rotor_v4_multi_append_matches_single_shot_gqa() {
    let kv_h = 3_usize;
    let head_dim = 96_usize;
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
    let mut qref = QuantRotorV4::new(vec![1, kv_h as i32, 0, head_dim as i32], 64, 7);
    qref.append(
        &build(0, s_total),
        &[1, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .unwrap();
    let reference = qref.dequant().unwrap();

    let mut qv = QuantRotorV4::new(vec![1, kv_h as i32, 0, head_dim as i32], 64, 7);
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
        "rotor4 multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — head↔seq scramble"
    );
}

/// Falsifies #284: at `kv_h > 1`, `truncate_to(n)` must keep exactly the
/// leading blocks covering sequence `[0, n)`, not `floor(n / kv_h)` of them.
/// CPU-only (no GPU ring touched). See `quant_rotor_v3_tests.rs` for the
/// full rationale + mutation-check note.
#[test]
fn quant_rotor_v4_truncate_to_kv_h_gt_1_keeps_exact_prefix() {
    let head_dim = 96_usize;
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

        let mut store = QuantRotorV4::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 64, 5);
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

        let mut reference = QuantRotorV4::new(vec![1_i32, kv_h as i32, 0, head_dim as i32], 64, 5);
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
/// concatenation back in `QuantRotorV4::dequant` and this goes red at
/// `b = 2` while staying green at `b = 1` — which is how the defect stayed
/// invisible.
#[test]
fn quant_rotor_v4_two_block_decode_matches_one_block_at_b_gt_1() {
    for b in [1_usize, 2] {
        let (kv_h, head_dim) = (2_usize, 96_usize);
        let (n0, n1) = (2_usize, 3_usize);
        let shape = |n: usize| [b as i32, kv_h as i32, n as i32, head_dim as i32];

        let mut one = QuantRotorV4::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], 512, 5);
        one.append(
            &crate::test_utils::batch_head_chunk(b, kv_h, 0, n0 + n1, head_dim),
            &shape(n0 + n1),
        )
        .expect("single append");
        let oracle = one.dequant().expect("one-block dequant");

        let mut two = QuantRotorV4::new(vec![b as i32, kv_h as i32, 0, head_dim as i32], 512, 5);
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
