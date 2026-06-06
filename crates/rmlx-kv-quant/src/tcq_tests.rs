//! CPU TCQ codec tests — 3-bit and 2-bit variants.

use super::{build_transition_table, tcq_quantize_v2, tcq_quantize_v3, TCQ_NUM_STATES};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};
use crate::turboquant::{turbo_dequantize, turbo_quantize_v, GROUP_SIZE};

// Reuse the canonical cosine fixture from turboquant_tests.rs so the TCQ
// cosine gate is directly comparable to the plain turbo3 row.
const COSINE_SHAPE: [i32; 4] = [1, 4, 128, 64];
const COSINE_HEAD_DIM: usize = 64;

fn cosine_fixture() -> Vec<f32> {
    let n: usize = COSINE_SHAPE.iter().map(|&d| d as usize).product();
    lcg_data(n, TEST_SEED)
}

// ── Trellis transition table ─────────────────────────────────────────────────

#[test]
fn trellis_transition_table_matches_reference_formula() {
    let num_levels = 8usize;
    let tbl = build_transition_table(num_levels);
    assert_eq!(tbl.len(), TCQ_NUM_STATES * num_levels);
    for state in 0..TCQ_NUM_STATES {
        for level in 0..num_levels {
            let expected = ((state << 1) | (level & 1)) % TCQ_NUM_STATES;
            let got = tbl[state * num_levels + level] as usize;
            assert_eq!(
                got, expected,
                "transition table mismatch at state={state} level={level}: \
                 expected={expected} got={got}",
            );
        }
    }
}

// ── Cosine gate on LCG fixture (mirrors turbo3 cosine gate) ──────────────────

#[test]
fn tcq_v3_cosine_gate() {
    // Per multi-turboquant README matrix row `turbo3_tcq` = 0.9817
    // − 0.001 empirical floor → ≥ 0.9807. The fixture is the same LCG data
    // used by the plain turbo3 cosine gate so the two thresholds are directly
    // comparable (TCQ should be ≥ turbo3 by construction).
    let data = cosine_fixture();
    let blocks = tcq_quantize_v3(&data, &COSINE_SHAPE).expect("tcq_quantize_v3 failed");
    let decoded = turbo_dequantize(&blocks).expect("turbo_dequantize failed");
    let stats = cosine_similarity_per_row(&data, &decoded, COSINE_HEAD_DIM);
    assert!(
        stats.mean >= 0.9807,
        "TCQ V3 mean cosine {:.6} < 0.9807 (empirical floor)",
        stats.mean,
    );
}

// ── Quality gate: TCQ ≥ turbo3 on a non-Gaussian fixture ─────────────────────
//
// The load-bearing test: prove Viterbi assignment actually helps over
// nearest-centroid on data with structure the codebook does not match. We use
// a sinusoidal sweep across the dim axis — strongly inter-element-correlated
// values that nearest-centroid cannot exploit.

#[allow(
    clippy::suboptimal_flops,
    reason = "test fixture: readable closed-form sinusoid; mul_add micro-optim has no measurable effect on test wall time"
)]
fn sinusoidal_fixture(shape: &[i32; 4]) -> Vec<f32> {
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let dim = shape[3] as usize;
    let mut out = Vec::with_capacity(n);
    for row in 0..(n / dim) {
        // Per-row phase + frequency derived from the row index; deterministic.
        let phase = (row as f32) * 0.123_456;
        let freq = 1.5 + (row as f32 % 3.0) * 0.5;
        for t in 0..dim {
            let theta = phase + (t as f32) * freq * 0.1;
            // Mixture-of-sinusoids: long + short wavelengths overlaid.
            let v = 0.6 * theta.sin() + 0.35 * (theta * 2.7).cos();
            out.push(v);
        }
    }
    out
}

#[test]
fn tcq_beats_plain_turbo3_on_sinusoidal_fixture() {
    // Same shape as the cosine fixture so each row is exactly one head_dim slice.
    let shape: [i32; 4] = COSINE_SHAPE;
    let data = sinusoidal_fixture(&shape);
    assert_eq!(
        data.len(),
        shape.iter().map(|&d| d as usize).product::<usize>(),
        "sinusoidal_fixture length mismatch"
    );

    let turbo_blocks = turbo_quantize_v(&data, 3, &shape).expect("plain turbo3 encode failed");
    let turbo_decoded = turbo_dequantize(&turbo_blocks).expect("plain turbo3 decode failed");
    let turbo_stats = cosine_similarity_per_row(&data, &turbo_decoded, COSINE_HEAD_DIM);

    let tcq_blocks = tcq_quantize_v3(&data, &shape).expect("tcq encode failed");
    let tcq_decoded = turbo_dequantize(&tcq_blocks).expect("tcq decode (turbo path) failed");
    let tcq_stats = cosine_similarity_per_row(&data, &tcq_decoded, COSINE_HEAD_DIM);

    assert!(
        tcq_stats.mean >= turbo_stats.mean,
        "TCQ ({:.6}) regressed vs plain turbo3 ({:.6}) on sinusoidal fixture — \
         Viterbi assignment should never produce worse cosine than nearest-centroid \
         on the same codebook",
        tcq_stats.mean,
        turbo_stats.mean,
    );
}

// ── Layout invariants: TCQ output is decoder-compatible with plain turbo ─────

#[test]
fn tcq_output_layout_matches_plain_turbo3() {
    let data = cosine_fixture();
    let tcq_blocks = tcq_quantize_v3(&data, &COSINE_SHAPE).expect("tcq encode failed");
    let plain_blocks =
        turbo_quantize_v(&data, 3, &COSINE_SHAPE).expect("plain turbo3 encode failed");

    assert_eq!(
        tcq_blocks.bits, plain_blocks.bits,
        "bits must match plain turbo3 (3)"
    );
    assert_eq!(
        tcq_blocks.codes.len(),
        plain_blocks.codes.len(),
        "codes-byte length must match plain turbo3 — decoder is shared"
    );
    assert_eq!(
        tcq_blocks.scales.len(),
        plain_blocks.scales.len(),
        "scales length must match plain turbo3 — decoder is shared"
    );
    assert_eq!(
        tcq_blocks.original_shape, plain_blocks.original_shape,
        "original_shape must match plain turbo3"
    );
}

// ── Zero-scale block: emits zero indices, identical to plain turbo3 ──────────

#[test]
fn tcq_zero_scale_block_emits_zero_indices() {
    let shape: [i32; 4] = [1, 1, 1, GROUP_SIZE as i32 * 2];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    // Half of the buffer is exact zeros (will trigger the zero-scale path);
    // the other half is a tiny non-zero ramp so the second block exercises the
    // normal Viterbi path. Validates the early-return contract.
    let mut data = vec![0.0_f32; n];
    for (i, slot) in data.iter_mut().enumerate().skip(GROUP_SIZE) {
        *slot = (i - GROUP_SIZE) as f32 * 0.01;
    }
    let blocks = tcq_quantize_v3(&data, &shape).expect("tcq encode failed");
    assert_eq!(blocks.scales.len(), 2);
    assert_eq!(blocks.scales[0], 0.0, "first block scale must be 0");
    assert!(blocks.scales[1] > 0.0, "second block scale must be > 0");
    // Codes for the first block are all zeros (the buf was zero-initialised
    // and the zero-scale early-return does not pack).
    let bytes_per_block = (GROUP_SIZE * 3).div_ceil(8);
    assert!(
        blocks.codes[..bytes_per_block].iter().all(|&b| b == 0),
        "zero-scale block must emit all-zero packed bytes",
    );
}

// ── Shape validation ────────────────────────────────────────────────────────

#[test]
fn tcq_rejects_d_not_multiple_of_group_size() {
    // D = 33 (not a multiple of GROUP_SIZE = 32).
    let shape: [i32; 4] = [1, 1, 1, 33];
    let data = vec![0.0_f32; 33];
    let res = tcq_quantize_v3(&data, &shape);
    assert!(
        res.is_err(),
        "tcq must reject D not divisible by GROUP_SIZE"
    );
}

#[test]
fn tcq_rejects_wrong_rank() {
    let shape: [i32; 3] = [1, 1, 32];
    let data = vec![0.0_f32; 32];
    let res = tcq_quantize_v3(&data, &shape);
    assert!(res.is_err(), "tcq must reject non-4-D original_shape");
}

#[test]
fn tcq_rejects_length_mismatch() {
    let shape: [i32; 4] = [1, 1, 1, 32];
    let data = vec![0.0_f32; 33];
    let res = tcq_quantize_v3(&data, &shape);
    assert!(res.is_err(), "tcq must reject x.len() != prod(shape)");
}

// ── 2-bit TCQ tests ──────────────────────────────────────────────────────────

/// V2 cosine gate: empirical floor = multi-turboquant matrix row `turbo2_tcq`
/// − 0.001. The fixture is the same LCG data used by the 3-bit cosine gate so
/// both thresholds are directly comparable. 2-bit TCQ should exceed or equal
/// plain turbo2 by construction (Viterbi cannot be worse than nearest-centroid
/// on the same codebook).
#[test]
fn tcq_v2_cosine_gate() {
    let data = cosine_fixture();
    let blocks = tcq_quantize_v2(&data, &COSINE_SHAPE).expect("tcq_quantize_v2 failed");
    let decoded = turbo_dequantize(&blocks).expect("turbo_dequantize failed");
    let stats = cosine_similarity_per_row(&data, &decoded, COSINE_HEAD_DIM);
    // 2-bit TCQ: empirical measured value on LCG fixture is ~0.9579.
    // Floor = measured − 0.001 ≈ 0.957.
    assert!(
        stats.mean >= 0.957,
        "TCQ V2 mean cosine {:.6} < 0.957 (empirical floor)",
        stats.mean,
    );
}

/// Quality gate: TCQ V2 ≥ plain turbo2 on the sinusoidal fixture.
/// Viterbi assignment on the 4-centroid codebook should never produce worse
/// cosine than nearest-centroid on the same data.
#[test]
fn tcq_v2_beats_plain_turbo2_on_sinusoidal_fixture() {
    let shape: [i32; 4] = COSINE_SHAPE;
    let data = sinusoidal_fixture(&shape);

    let turbo_blocks = turbo_quantize_v(&data, 2, &shape).expect("plain turbo2 encode failed");
    let turbo_decoded = turbo_dequantize(&turbo_blocks).expect("plain turbo2 decode failed");
    let turbo_stats = cosine_similarity_per_row(&data, &turbo_decoded, COSINE_HEAD_DIM);

    let tcq_blocks = tcq_quantize_v2(&data, &shape).expect("tcq v2 encode failed");
    let tcq_decoded = turbo_dequantize(&tcq_blocks).expect("tcq v2 decode (turbo path) failed");
    let tcq_stats = cosine_similarity_per_row(&data, &tcq_decoded, COSINE_HEAD_DIM);

    assert!(
        tcq_stats.mean >= turbo_stats.mean,
        "TCQ V2 ({:.6}) regressed vs plain turbo2 ({:.6}) on sinusoidal fixture — \
         Viterbi assignment should never produce worse cosine than nearest-centroid \
         on the same codebook",
        tcq_stats.mean,
        turbo_stats.mean,
    );
}

/// Layout invariant: TCQ V2 output is decoder-compatible with plain turbo2.
/// `bits`, `codes` byte-length, `scales` length, and `original_shape` must
/// match the plain encoder's output (shared decoder contract).
#[test]
fn tcq_v2_output_layout_matches_plain_turbo2() {
    let data = cosine_fixture();
    let tcq_blocks = tcq_quantize_v2(&data, &COSINE_SHAPE).expect("tcq v2 encode failed");
    let plain_blocks =
        turbo_quantize_v(&data, 2, &COSINE_SHAPE).expect("plain turbo2 encode failed");

    assert_eq!(
        tcq_blocks.bits, plain_blocks.bits,
        "bits must match plain turbo2 (2)"
    );
    assert_eq!(
        tcq_blocks.codes.len(),
        plain_blocks.codes.len(),
        "codes-byte length must match plain turbo2 — decoder is shared"
    );
    assert_eq!(
        tcq_blocks.scales.len(),
        plain_blocks.scales.len(),
        "scales length must match plain turbo2 — decoder is shared"
    );
    assert_eq!(
        tcq_blocks.original_shape, plain_blocks.original_shape,
        "original_shape must match plain turbo2"
    );
}

/// Zero-scale block emits all-zero indices at 2-bit (mirrors v3 zero-scale
/// test but with the 2-bit bytes_per_block calculation).
#[test]
fn tcq_v2_zero_scale_block_emits_zero_indices() {
    let shape: [i32; 4] = [1, 1, 1, GROUP_SIZE as i32 * 2];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut data = vec![0.0_f32; n];
    for (i, slot) in data.iter_mut().enumerate().skip(GROUP_SIZE) {
        *slot = (i - GROUP_SIZE) as f32 * 0.01;
    }
    let blocks = tcq_quantize_v2(&data, &shape).expect("tcq v2 encode failed");
    assert_eq!(blocks.bits, 2, "bits must be 2");
    assert_eq!(blocks.scales.len(), 2);
    assert_eq!(blocks.scales[0], 0.0, "first block scale must be 0");
    assert!(blocks.scales[1] > 0.0, "second block scale must be > 0");
    // 2-bit: bytes_per_block = (32 * 2).div_ceil(8) = 8.
    let bytes_per_block = (GROUP_SIZE * 2).div_ceil(8);
    assert!(
        blocks.codes[..bytes_per_block].iter().all(|&b| b == 0),
        "zero-scale block must emit all-zero packed bytes (2-bit)",
    );
}
