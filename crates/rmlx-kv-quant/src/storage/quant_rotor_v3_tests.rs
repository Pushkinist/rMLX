//! Tests for [`QuantRotorV3`].
#![allow(
    clippy::identity_op,
    reason = "explicit `B * kv_h * seq * head_dim` element counts kept as-is for readability — `1 * 2 * n_tokens * head_dim` mirrors the canonical [B, kv_h, S, D] shape literally"
)]

use crate::storage::quant_rotor_v3::{QuantRotorV3, ROTOR3_V_BITS};

/// Newly-constructed `QuantRotorV3` carries the requested init shape and bit
/// tag; no rotors yet (lazily generated on first append).
#[test]
fn quant_rotor_v3_new_carries_shape_and_bits() {
    let shape = vec![1_i32, 2, 0, 96];
    let qv = QuantRotorV3::new(shape.clone(), 4096, 7);
    assert_eq!(qv.shape, shape, "shape preserved");
    assert_eq!(qv.max_seq, 4096, "max_seq preserved");
    assert_eq!(qv.bits, ROTOR3_V_BITS, "bits tag");
    assert_eq!(qv.layer_idx, 7, "layer_idx preserved");
    assert!(qv.rotors.is_empty(), "rotors empty before first append");
    assert!(qv.blocks.is_empty(), "no blocks before append");
}

/// First `append` lazily generates the rotor table (`n_groups * 4` f32).
#[test]
fn quant_rotor_v3_append_generates_rotor_table() {
    let mut qv = QuantRotorV3::new(vec![1_i32, 2, 0, 96], 4096, 0);
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
fn quant_rotor_v3_truncate_keeps_rotors() {
    let mut qv = QuantRotorV3::new(vec![1_i32, 1, 0, 96], 4096, 3);
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
fn quant_rotor_v3_reset_keeps_rotors() {
    let mut qv = QuantRotorV3::new(vec![1_i32, 1, 0, 96], 4096, 2);
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
fn quant_rotor_v3_byte_size_counts_rotors_once() {
    let mut qv = QuantRotorV3::new(vec![1_i32, 1, 0, 96], 4096, 0);
    let head_dim = 96;
    let new_shape = [1_i32, 1, 4, head_dim as i32];
    let data = vec![0.5_f32; 1 * 1 * 4 * head_dim];
    qv.append(&data, &new_shape).unwrap();

    let rotors_bytes = qv.rotors.len() * size_of::<f32>();
    let blocks_bytes = qv
        .blocks
        .iter()
        .map(|b| b.codes.len() * 4 + b.scales.len() * 4 + b.norms.len() * 4)
        .sum::<usize>();
    assert_eq!(qv.byte_size(), rotors_bytes + blocks_bytes);

    // Appending a second block must not double-count rotors.
    qv.append(&data, &new_shape).unwrap();
    let new_blocks_bytes = qv
        .blocks
        .iter()
        .map(|b| b.codes.len() * 4 + b.scales.len() * 4 + b.norms.len() * 4)
        .sum::<usize>();
    assert_eq!(qv.byte_size(), rotors_bytes + new_blocks_bytes);
}

/// `try_deep_clone` clones rotors + blocks + meta.
#[test]
fn quant_rotor_v3_deep_clone() {
    let mut qv = QuantRotorV3::new(vec![1_i32, 1, 0, 96], 4096, 1);
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
}

/// Multi-append with `kv_h > 1` must match a single-shot append of the
/// concatenated head-major buffer (head↔seq layout invariant). The static
/// rotor table is group-position-keyed (not token), so the seq-major reorder
/// leaves it correctly associated. Per-(head, token, dim) distinct values
/// surface any head transposition as a large error.
#[test]
fn quant_rotor_v3_multi_append_matches_single_shot_gqa() {
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
    // Same layer_idx so both caches share the same static rotor table.
    let mut qref = QuantRotorV3::new(vec![1, kv_h as i32, 0, head_dim as i32], 64, 7);
    qref.append(
        &build(0, s_total),
        &[1, kv_h as i32, s_total as i32, head_dim as i32],
    )
    .unwrap();
    let reference = qref.dequant().unwrap();

    let mut qv = QuantRotorV3::new(vec![1, kv_h as i32, 0, head_dim as i32], 64, 7);
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
        "rotor3 multi-append vs single-shot max_abs_err = {max_abs:.6} (>= 1.0) — head↔seq scramble"
    );
    let _ = ROTOR3_V_BITS;
}
