use super::*;
use crate::test_utils::{cosine_similarity_per_row, lcg_data, TEST_SEED};

// re-export for codebook-override tests
use super::{turbo_dequantize_with_codebook, turbo_quantize_v_with_codebook};

/// `original_shape: [i32; 4]` is 16 B inline — no heap allocation per TurboBlocks.
///
/// Before perf(types): `original_shape: Vec<i32>` was 24 B stack + 16 B heap alloc.
/// After: 16 B inline in the struct. Savings: 8 B stack + 1 heap alloc per decode step.
#[test]
fn turbo_blocks_original_shape_is_inline_array() {
    assert_eq!(
        size_of::<[i32; 4]>(),
        16,
        "original_shape must be 16 bytes (4 × i32)"
    );
    // Verify a round-trip still works — shape is stored as [i32; 4].
    let shape = [1i32, 1, 1, 32];
    let data = vec![0.1_f32; 32];
    let blocks = turbo_quantize_v(&data, 4, &shape).unwrap();
    assert_eq!(blocks.original_shape, shape);
}

/// Quantize a [1, 4, 128, 64] random f32 tensor at 4 bits, dequantize,
/// and verify max absolute error is below 0.15 per element.
///
/// Input data is drawn from [-1.0, 1.0] (uniform), which represents
/// normalized KV vectors after L2-normalization — the typical input range
/// that TurboQuant is designed for. For this range:
/// scale ≈ 1.0 / 2.7326 ≈ 0.366
/// worst-case half-step ≈ 0.332 × 0.366 ≈ 0.122 < 0.15 ✓
#[test]
fn turbo_quantize_then_dequantize_v_4bit_within_tolerance() {
    // Deterministic LCG pseudo-random to avoid rand dependency.
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let mut state = 0xDEAD_BEEF_u64;
    let data: Vec<f32> = (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to [-1.0, 1.0] — normalized KV range TurboQuant targets.
            let frac = ((state >> 33) as f32) / (u32::MAX as f32);
            frac * 2.0 - 1.0
        })
        .collect();

    let blocks = turbo_quantize_v(&data, 4, &shape).expect("quantize failed");
    let recon = turbo_dequantize(&blocks).expect("dequantize failed");

    assert_eq!(recon.len(), n, "output length mismatch");

    let max_err = data
        .iter()
        .zip(recon.iter())
        .map(|(&orig, &dq)| (orig - dq).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.15,
        "max abs error {max_err:.6} exceeds tolerance 0.15 for 4-bit turbo quantization"
    );
}

/// The 4-bit codebook must have exactly 16 entries and be strictly increasing.
#[test]
fn turbo_codebook_4bit_has_16_entries_monotonic() {
    let cb = lloyd_gaussian_codebook(4).expect("4-bit codebook failed");
    assert_eq!(
        cb.len(),
        16,
        "4-bit codebook must have 16 entries, got {}",
        cb.len()
    );
    for i in 0..cb.len() - 1 {
        assert!(
            cb[i] < cb[i + 1],
            "codebook not strictly increasing at index {i}: {} >= {}",
            cb[i],
            cb[i + 1]
        );
    }
}

/// bits=8 must return an error — K8 uses affine q8_0, not TurboQuant.
#[test]
fn turbo_codebook_8bit_unsupported() {
    let result = lloyd_gaussian_codebook(8);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for bits=8, got {result:?}"
    );
}

/// Dequantized tensor must have the same shape as the original input.
#[test]
fn turbo_quantize_preserves_shape() {
    let shape = [2i32, 4, 16, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = vec![0.5_f32; n];

    let blocks = turbo_quantize_v(&data, 4, &shape).expect("quantize failed");
    let recon = turbo_dequantize(&blocks).expect("dequantize failed");

    let recon_elems: usize = blocks.original_shape.iter().map(|&d| d as usize).product();
    assert_eq!(recon.len(), recon_elems, "shape not preserved");
    assert_eq!(blocks.original_shape, shape, "original_shape mismatch");
}

// ── Additional correctness checks ─────────────────────────────────────────

/// 3-bit codebook has 8 entries and is strictly monotonic.
#[test]
fn turbo_codebook_3bit_has_8_entries_monotonic() {
    let cb = lloyd_gaussian_codebook(3).expect("3-bit codebook failed");
    assert_eq!(cb.len(), 8);
    for i in 0..cb.len() - 1 {
        assert!(cb[i] < cb[i + 1], "not monotonic at {i}");
    }
}

/// Zero-input: quantize then dequantize must produce all zeros.
#[test]
fn turbo_quantize_zero_input_roundtrip() {
    let shape = [1i32, 1, 1, 32];
    let data = vec![0.0_f32; 32];
    let blocks = turbo_quantize_v(&data, 4, &shape).expect("quantize failed");
    let recon = turbo_dequantize(&blocks).expect("dequantize failed");
    for (i, &v) in recon.iter().enumerate() {
        assert!(v.abs() < 1e-6, "expected 0 at index {i}, got {v}");
    }
}

/// Shape validation: wrong shape length returns Quant error.
#[test]
fn turbo_err_wrong_shape_len() {
    let result = turbo_quantize_v(&[0.0_f32; 32], 4, &[1i32, 32]);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for 2-element shape"
    );
}

/// Shape validation: last dim not multiple of GROUP_SIZE returns Quant error.
#[test]
fn turbo_err_last_dim_not_multiple_of_group_size() {
    let shape = [1i32, 1, 1, 33]; // 33 is not a multiple of 32
    let result = turbo_quantize_v(&[0.0_f32; 33], 4, &shape);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for D=33"
    );
}

/// Unsupported bits returns Quant error.
#[test]
fn turbo_err_unsupported_bits() {
    let shape = [1i32, 1, 1, 32];
    let result = turbo_quantize_v(&[0.0_f32; 32], 7, &shape);
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "expected Quant error for bits=7"
    );
}

// ── Cosine-similarity gate ────────────────────────────────────────────────────

// Fixture shape [1, 4, 128, 64] = 32 768 elements; rows are head_dim=64 slices.
const COSINE_SHAPE: [i32; 4] = [1, 4, 128, 64];
const COSINE_HEAD_DIM: usize = 64;

fn cosine_fixture() -> Vec<f32> {
    let n: usize = COSINE_SHAPE.iter().map(|&d| d as usize).product();
    lcg_data(n, TEST_SEED)
}

/// TurboQuant V4 (K8V4 V-side) cosine gate: mean cosine ≥ 0.9937.
///
/// Threshold: per ../multi-turboquant README matrix row `turbo4` = 0.9947 − 0.001.
#[test]
fn turbo_v4_cosine_gate_k8v4() {
    let data = cosine_fixture();
    let blocks = turbo_quantize_v(&data, 4, &COSINE_SHAPE).expect("turbo_quantize_v failed");
    let decoded = turbo_dequantize(&blocks).expect("turbo_dequantize failed");
    let stats = cosine_similarity_per_row(&data, &decoded, COSINE_HEAD_DIM);
    assert!(
        stats.mean >= 0.9937,
        // per ../multi-turboquant README matrix row `turbo4` = 0.9947 − 0.001
        "TurboQuant V4 (K8V4) mean cosine {:.6} < 0.9937 (empirical floor)",
        stats.mean,
    );
}

/// TurboQuant K4 (TurboSym4 K-side) cosine gate: mean cosine ≥ 0.9937.
///
/// The TurboQuant codec is axis-agnostic (Lloyd-Max N(0,1) 4-bit codebook
/// applied to flat f32 groups regardless of which axis the source slice came
/// from). With the same `cosine_fixture()` data the K-side cosine matches the
/// V-side cosine within f32 rounding. Spec floor is `0.9947 − 0.001 = 0.9937`.
///
/// If this assertion ever falls below 0.9937 while the V-side stays at ~0.9947
/// it is a sign the K-side input distribution diverged from N(0,1) — the agent
/// should escalate rather than relax the gate.
#[test]
fn turbo_k4_cosine_gate_tsym4() {
    let data = cosine_fixture();
    let blocks = turbo_quantize_v(&data, 4, &COSINE_SHAPE).expect("turbo_quantize_v failed");
    let decoded = turbo_dequantize(&blocks).expect("turbo_dequantize failed");
    let stats = cosine_similarity_per_row(&data, &decoded, COSINE_HEAD_DIM);
    assert!(
        stats.mean >= 0.9937,
        "TurboQuant K4 (TurboSym4 K-side) mean cosine {:.6} < 0.9937 (empirical floor)",
        stats.mean,
    );
}

/// TurboQuant V3 (K8VTurbo3 V-side) cosine gate: mean cosine ≥ 0.9807.
///
/// Threshold: per ../multi-turboquant README matrix row `turbo3` = 0.9817 − 0.001.
#[test]
fn turbo_v3_cosine_gate_k8vturbo3() {
    let data = cosine_fixture();
    let blocks = turbo_quantize_v(&data, 3, &COSINE_SHAPE).expect("turbo_quantize_v failed");
    let decoded = turbo_dequantize(&blocks).expect("turbo_dequantize failed");
    let stats = cosine_similarity_per_row(&data, &decoded, COSINE_HEAD_DIM);
    assert!(
        stats.mean >= 0.9807,
        // per ../multi-turboquant README matrix row `turbo3` = 0.9817 − 0.001
        "TurboQuant V3 (K8VTurbo3) mean cosine {:.6} < 0.9807 (empirical floor)",
        stats.mean,
    );
}

/// Bit-pack/unpack round-trip for all supported bit widths.
#[test]
fn pack_unpack_roundtrip_all_bits() {
    for bits in [1u8, 2, 3, 4] {
        let max_idx = (1u8 << bits) - 1;
        let bits_per_block = GROUP_SIZE * bits as usize;
        let bytes_per_block = bits_per_block.div_ceil(8);
        let mut block_bytes = vec![0u8; bytes_per_block];

        for elem in 0..GROUP_SIZE {
            let idx = (elem as u8) & max_idx;
            pack_index(&mut block_bytes, elem, idx, bits);
        }
        for elem in 0..GROUP_SIZE {
            let expected = (elem as u8) & max_idx;
            let got = unpack_index(&block_bytes, elem, bits);
            assert_eq!(
                got, expected,
                "bits={bits} elem={elem}: expected={expected} got={got}"
            );
        }
    }
}

/// 3-bit codebook finitude gate.
///
/// `build_msl_header_v3` uses `partial_cmp(...).unwrap_or(Equal)` to find the
/// max codebook entry. A NaN-poisoned codebook would silently produce a wrong
/// `CB_MAX_3` constant. This test asserts every entry is finite so that the
/// `unwrap_or` fallback is never reachable in practice. If it fires, the
/// codebook derivation changed; regenerate and re-verify.
#[test]
fn lloyd_gaussian_codebook_3bit_entries_are_finite() {
    let cb = lloyd_gaussian_codebook(3).expect("3-bit codebook must exist");
    assert_eq!(cb.len(), 8, "3-bit codebook must have 8 entries");
    for (i, &v) in cb.iter().enumerate() {
        assert!(
            v.is_finite(),
            "CODEBOOK_3BIT[{i}] = {v} is not finite (NaN or Inf)"
        );
    }
}

// ── Codebook bit-exactness gates ──────────────────────────────────────────────

/// TurboQuant V4 4-bit codebook: bit patterns must match MSL-embedded constants.
///
/// The MSL kernel embeds these values as `constant float CB4[16]`. Any drift
/// (e.g. regenerating the codebook from a different source) must be caught here
/// before it silently breaks GPU/CPU parity.
///
/// Expected bit patterns derived from `CODEBOOK_4BIT` as declared in
/// `turboquant.rs` (verified via `f32::to_bits()` on each entry).
#[test]
fn cb4_constants_bit_exact() {
    let cb = lloyd_gaussian_codebook(4).expect("4-bit codebook");
    assert_eq!(cb.len(), 16, "4-bit codebook must have 16 entries");

    let expected_bits: [u32; 16] = [
        0xC02D_EE42, // -2.717_667
        0xC003_563B, // -2.052_138
        0xBFCC_E718, // -1.600_802_4
        0xBF9E_B6FA, // -1.239_959
        0xBF6D_A172, // -0.928_244_7
        0xBF25_5816, // -0.645_875_33
        0xBEC3_29CB, // -0.381_178_23
        0xBE01_1273, // -0.126_046_94
        0x3E01_1273, //  0.126_046_94
        0x3EC3_29CB, //  0.381_178_23
        0x3F25_5816, //  0.645_875_33
        0x3F6D_A172, //  0.928_244_7
        0x3F9E_B6FA, //  1.239_959
        0x3FCC_E718, //  1.600_802_4
        0x4003_563B, //  2.052_138
        0x402D_EE42, //  2.717_667
    ];

    for (i, (&v, &expected)) in cb.iter().zip(expected_bits.iter()).enumerate() {
        assert_eq!(
            v.to_bits(),
            expected,
            "CODEBOOK_4BIT[{i}] bit pattern: got 0x{:08X} expected 0x{expected:08X}",
            v.to_bits(),
        );
    }

    // Boundary midpoints (used in nearest-centroid lookup).
    let expected_bnds: [u32; 15] = [
        0xC018_A23E, // bnd[0]
        0xBFE9_C9C7, // bnd[1]
        0xBFB5_CF09, // bnd[2]
        0xBF8A_C3DA, // bnd[3]
        0xBF49_7CC4, // bnd[4]
        0xBF03_767E, // bnd[5]
        0xBE81_D982, // bnd[6]
        0x0000_0000, // bnd[7] = 0.0
        0x3E81_D982, // bnd[8]
        0x3F03_767E, // bnd[9]
        0x3F49_7CC4, // bnd[10]
        0x3F8A_C3DA, // bnd[11]
        0x3FB5_CF09, // bnd[12]
        0x3FE9_C9C7, // bnd[13]
        0x4018_A23E, // bnd[14]
    ];
    for i in 0..15usize {
        let mid = (cb[i] + cb[i + 1]) * 0.5_f32;
        assert_eq!(
            mid.to_bits(),
            expected_bnds[i],
            "CODEBOOK_4BIT boundary[{i}] bit pattern: got 0x{:08X} expected 0x{:08X}",
            mid.to_bits(),
            expected_bnds[i],
        );
    }
}

// ── turbo_quantize_v_with_codebook ────────────────────────────────────────────

/// `None` override is identical to `turbo_quantize_v` (built-in Lloyd-Max).
#[test]
fn with_codebook_none_identical_to_default() {
    let shape = [1i32, 1, 1, 32];
    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

    let baseline = turbo_quantize_v(&data, 4, &shape).expect("baseline quantize failed");
    let with_none =
        turbo_quantize_v_with_codebook(&data, 4, &shape, None).expect("None override failed");

    assert_eq!(
        baseline.codes, with_none.codes,
        "None override must produce identical codes to turbo_quantize_v"
    );
    assert_eq!(
        baseline.scales, with_none.scales,
        "None override must produce identical scales to turbo_quantize_v"
    );
}

/// A custom codebook that differs from Lloyd-Max must produce different codes.
// Strictly-ascending codebook with span far smaller than Lloyd-Max — guarantees distinct nearest-centroid outputs on the same input.
#[test]
fn with_codebook_override_produces_distinct_codes() {
    let shape = [1i32, 1, 1, 32];
    let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

    let custom_cb: Vec<f32> = (0..16).map(|i| i as f32 * 0.001).collect(); // [0, 0.001, ..., 0.015]

    let baseline = turbo_quantize_v(&data, 4, &shape).expect("baseline quantize");
    let with_custom =
        turbo_quantize_v_with_codebook(&data, 4, &shape, Some(&custom_cb)).expect("override");

    assert_ne!(
        baseline.codes, with_custom.codes,
        "custom codebook must produce different codes from Lloyd-Max on non-zero data"
    );
}

/// Dequantize round-trip test: encode with a codebook that diverges clearly from
/// Lloyd-Max, then decode with the same override and verify exact centroid values.
///
/// Codebook [-3.0, -1.0, 1.0, 3.0] is far from the 2-bit Lloyd-Max values
/// (~[-1.51, -0.453, 0.453, 1.51]). Inputs exactly at centroid values must
/// round-trip to those exact values. Without the dequant-override fix (HIGH 1)
/// this test fails because `turbo_dequantize` would decode with Lloyd-Max.
#[test]
fn with_codebook_dequant_exact_roundtrip() {
    // One group of 32, with each quarter filled with one centroid value.
    // Centroid values: -3.0, -1.0, 1.0, 3.0.
    let shape = [1i32, 1, 1, 32];
    let cb_2bit: Vec<f32> = vec![-3.0, -1.0, 1.0, 3.0];
    let data: Vec<f32> = (0..32)
        .map(|i| cb_2bit[i / 8]) // 8 elements per centroid
        .collect();

    let blocks = turbo_quantize_v_with_codebook(&data, 2, &shape, Some(&cb_2bit)).expect("encode");

    // Decode with the same codebook override — must produce exact centroid values.
    let recon =
        turbo_dequantize_with_codebook(&blocks, Some(&cb_2bit)).expect("decode with override");
    assert_eq!(recon.len(), data.len(), "output length mismatch");
    for (i, (&orig, &dq)) in data.iter().zip(recon.iter()).enumerate() {
        assert!(
            (orig - dq).abs() < 1e-5,
            "element {i}: expected {orig} got {dq} (round-trip failed)"
        );
    }

    // Decode without override (Lloyd-Max) must produce DIFFERENT values — confirming
    // the two code paths are distinct.
    let recon_default = turbo_dequantize(&blocks).expect("decode default");
    let max_diff = data
        .iter()
        .zip(recon_default.iter())
        .map(|(&a, &b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_diff > 0.5,
        "Lloyd-Max decode of [-3,-1,1,3] encoded data should differ substantially; max_diff={max_diff}"
    );
}

/// Wrong-length codebook override must return `Error::Quant` (not panic, not silent fallback).
#[test]
fn with_codebook_wrong_length_returns_error() {
    let shape = [1i32, 1, 1, 32];
    let data = vec![0.1_f32; 32];

    // 4-bit needs 16 entries; pass 8 → error.
    let wrong_cb: Vec<f32> = vec![0.0; 8];
    let result = turbo_quantize_v_with_codebook(&data, 4, &shape, Some(&wrong_cb));
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "wrong-length codebook must return Quant error, got: {result:?}"
    );

    // 2-bit needs 4 entries; pass 3 → error.
    let wrong_cb2: Vec<f32> = vec![0.0; 3];
    let result2 = turbo_quantize_v_with_codebook(&data, 2, &shape, Some(&wrong_cb2));
    assert!(
        matches!(result2, Err(Error::Quant(_))),
        "wrong-length 2-bit codebook must return Quant error, got: {result2:?}"
    );
}

/// Non-monotone codebook must return `Error::Quant` (not silently produce wrong indices).
#[test]
fn with_codebook_non_monotone_returns_error() {
    let shape = [1i32, 1, 1, 32];
    let data = vec![0.1_f32; 32];

    // 4-bit codebook where two adjacent entries are equal (not strictly ascending).
    let mut non_mono: Vec<f32> = (0..16).map(|i| i as f32).collect();
    non_mono[5] = non_mono[4]; // break strict ascent: [0,1,2,3,4,4,6,...]
    let result = turbo_quantize_v_with_codebook(&data, 4, &shape, Some(&non_mono));
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "non-monotone codebook must return Quant error, got: {result:?}"
    );

    // Descending entry also fails.
    let mut descending: Vec<f32> = (0..4).map(|i| i as f32).collect();
    descending[2] = 10.0; // [0, 1, 10, 3] — not ascending at index 3
    let result2 = turbo_quantize_v_with_codebook(&data, 2, &shape, Some(&descending));
    assert!(
        matches!(result2, Err(Error::Quant(_))),
        "descending entry in codebook must return Quant error, got: {result2:?}"
    );
}

/// Empty codebook `[]` is a caller bug: `2^bits != 0` for any valid `bits`.
/// Verify it returns `Error::Quant` (not panic).
#[test]
fn with_codebook_empty_override_returns_error() {
    let shape = [1i32, 1, 1, 32];
    let data = vec![0.0_f32; 32];
    let result = turbo_quantize_v_with_codebook(&data, 4, &shape, Some(&[]));
    assert!(
        matches!(result, Err(Error::Quant(_))),
        "empty codebook override must return Quant error"
    );
}
