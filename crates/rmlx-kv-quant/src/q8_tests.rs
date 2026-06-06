use super::{q8_dequantize, q8_quantize, Q8_GROUP_SIZE};
use crate::test_utils::{cosine_similarity_per_row, lcg_data, CosineStats, TEST_SEED};

// Fixture: [1, 4, 128, 64] = 32 768 elements, rows are head_dim=64 slices.
const SHAPE: [i32; 4] = [1, 4, 128, 64];
const HEAD_DIM: usize = 64;

fn fixture() -> Vec<f32> {
    let n: usize = SHAPE.iter().map(|&d| d as usize).product();
    lcg_data(n, TEST_SEED)
}

// ── Cosine-similarity gate ─────────────────────────────────────────────────

/// q8_0 (K8V8) cosine gate: mean cosine ≥ 0.9990.
///
/// Threshold: empirical floor measured 2026-05-30 (mean ~0.9998), set to
/// `measured − 0.001 = 0.9988` then raised to 0.9990 to match the ticket
/// spec conservative floor. q8_0 at group_size=128 on uniform [-1,1] data
/// is near-lossless; the floor is intentionally conservative.
#[test]
fn q8_cosine_gate_k8v8() {
    let data = fixture();
    let (codes, scales) = q8_quantize(&data);
    let decoded = q8_dequantize(&codes, &scales);

    let stats: CosineStats = cosine_similarity_per_row(&data, &decoded, HEAD_DIM);

    assert!(
        stats.mean >= 0.9990,
        // empirical floor measured 2026-05-30
        "q8_0 (K8V8) mean cosine {:.6} < 0.9990 (empirical floor)",
        stats.mean,
    );
    assert!(
        stats.min >= 0.9970,
        "q8_0 (K8V8) min cosine {:.6} < 0.9970",
        stats.min,
    );
}

// ── Round-trip correctness ─────────────────────────────────────────────────

/// Quantize then dequantize on uniform data: max abs error within expected bound.
///
/// q8_0 step ≈ 1/127 of the group max. For uniform [-1,1] data:
/// scale ≈ 1/127 ≈ 0.00787 → half-step ≈ 0.00394. Round up to 0.01 for margin.
#[test]
fn q8_roundtrip_within_tolerance() {
    let data = fixture();
    let (codes, scales) = q8_quantize(&data);
    let decoded = q8_dequantize(&codes, &scales);

    assert_eq!(decoded.len(), data.len(), "output length mismatch");

    let max_err = data
        .iter()
        .zip(decoded.iter())
        .map(|(&orig, &dq)| (orig - dq).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.01,
        "q8_0 roundtrip max abs error {max_err:.6} exceeds 0.01"
    );
}

/// Zero input roundtrip: all-zero output.
#[test]
fn q8_zero_input_roundtrip() {
    let n = Q8_GROUP_SIZE * 4;
    let data = vec![0.0_f32; n];
    let (codes, scales) = q8_quantize(&data);
    let decoded = q8_dequantize(&codes, &scales);
    for (i, &v) in decoded.iter().enumerate() {
        assert!(v.abs() < 1e-6, "expected 0 at index {i}, got {v}");
    }
}

/// Constant input: all codes should be identical (max, clamped to ±127).
#[test]
fn q8_constant_input_all_same_code() {
    let n = Q8_GROUP_SIZE;
    let data = vec![0.5_f32; n];
    let (codes, scales) = q8_quantize(&data);
    assert_eq!(scales.len(), 1);
    assert!((scales[0] - 0.5_f32 / 127.0).abs() < 1e-6, "scale mismatch");
    // All codes should be 127 (0.5 / scale = 127.0).
    for &c in &codes {
        assert_eq!(c as i8, 127, "expected all codes == 127");
    }
}

/// Length not a multiple of Q8_GROUP_SIZE must panic.
#[test]
#[should_panic(expected = "not a multiple of Q8_GROUP_SIZE")]
fn q8_bad_length_panics() {
    q8_quantize(&[0.0_f32; 100]);
}
