use super::super::state::MixedKvState;
use super::*;
use crate::test_utils::{cosine_similarity_per_row, gaussian_data, lcg_data, TEST_SEED};
use rmlx_core::DispatchPolicy;
use rmlx_mlx::Dtype;
use rmlx_mlx::{dequantize, matmul, scaled_dot_product_attention};

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn from_vec(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn to_vec(a: &Array) -> Vec<f32> {
    a.eval().unwrap();
    a.to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// DoD: rotated-and-quantized K with pre-rotated Q reproduces the
/// unrotated Q*Kt scores **within 8-bit affine quant tolerance**.
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rot_k_quantized_score_identity_within_tolerance() {
    let device = Device::Cpu;
    let b = 1i32;
    let kv_h = 1i32;
    let lk = 8i32;
    let d = 128i32;
    let lq = 4i32;

    let mut s = 0x1234_5678u64;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((s >> 33) as f32) / (u32::MAX as f32)).mul_add(2.0, -1.0)
    };
    let k_data: Vec<f32> = (0..(b * kv_h * lk * d)).map(|_| next()).collect();
    let q_data: Vec<f32> = (0..(b * kv_h * lq * d)).map(|_| next()).collect();
    let k = from_vec(&k_data, &[b, kv_h, lk, d]);
    let q = from_vec(&q_data, &[b, kv_h, lq, d]);

    let k_t = k.transpose(&[0, 1, 3, 2], device).unwrap();
    let scores_ref = matmul(&q, &k_t, device).unwrap();

    let mut state = MixedKvState::new_rotated(4, 64);
    let k_rot = state.maybe_rotate_k(&k, device).unwrap();
    let (codes, scales, biases) =
        rmlx_mlx::quantize(&k_rot, state.k_group_size, state.k_bits, device).unwrap();
    let k_rot_dq = dequantize(
        &codes,
        &scales,
        Some(&biases),
        state.k_group_size,
        state.k_bits,
        "affine",
        device,
    )
    .unwrap();

    let r = state.k_rotation.as_ref().unwrap();
    let q_rot = super::super::super::rot_k::rotate_last_axis(&q, r, device).unwrap();
    let k_rot_dq_t = k_rot_dq.transpose(&[0, 1, 3, 2], device).unwrap();
    let scores_rot = matmul(&q_rot, &k_rot_dq_t, device).unwrap();

    let a = to_vec(&scores_ref);
    let bb = to_vec(&scores_rot);
    let max_abs_ref = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let max_err = a
        .iter()
        .zip(&bb)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 0.02 * max_abs_ref.max(1.0),
        "RotK quantized score error {max_err} too large vs |scores|={max_abs_ref} \
         (pre-rotate-Q identity must survive 8-bit quant)"
    );
}

#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn plain_mixed_does_not_rotate() {
    let device = Device::Cpu;
    let mut state = MixedKvState::new(8, 4, 64, 64);
    assert!(!state.rotate_k);
    let k = from_vec(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 1, 4]);
    let out = state.maybe_rotate_k(&k, device).unwrap();
    assert_eq!(to_vec(&out), vec![1.0, 2.0, 3.0, 4.0]);
    assert!(state.k_rotation.is_none());
}

#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn rot_k_tq4v_k_dequant_same_tolerance_as_rot_k() {
    use rmlx_mlx::dequantize;

    let device = Device::Cpu;
    let b = 1i32;
    let kv_h = 1i32;
    let lk = 8i32;
    let d = 128i32;
    let lq = 4i32;

    let mut s = 0xABCD_1234u64;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (((s >> 33) as f32) / (u32::MAX as f32)).mul_add(2.0, -1.0)
    };
    let k_data: Vec<f32> = (0..(b * kv_h * lk * d)).map(|_| next()).collect();
    let q_data: Vec<f32> = (0..(b * kv_h * lq * d)).map(|_| next()).collect();
    let k = from_vec(&k_data, &[b, kv_h, lk, d]);
    let q = from_vec(&q_data, &[b, kv_h, lq, d]);

    let mut k_state = MixedKvState::new_rotated(4, 64);
    let (k_codes, k_scales, k_biases) = k_state
        .bulk_init_k_from_fp16(&k, device, DispatchPolicy::default())
        .unwrap();

    let k_dq = dequantize(
        &k_codes,
        &k_scales,
        Some(&k_biases),
        k_state.k_group_size,
        k_state.k_bits,
        "affine",
        device,
    )
    .unwrap();

    let r = k_state.k_rotation.as_ref().expect("rotation built");
    let q_rot = super::super::super::rot_k::rotate_last_axis(&q, r, device).unwrap();

    let k_dq_t = k_dq.transpose(&[0, 1, 3, 2], device).unwrap();
    let scores_rot = matmul(&q_rot, &k_dq_t, device).unwrap();

    let k_t = k.transpose(&[0, 1, 3, 2], device).unwrap();
    let scores_ref = matmul(&q, &k_t, device).unwrap();

    let a = to_vec(&scores_ref);
    let b_vals = to_vec(&scores_rot);
    let max_abs_ref = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let max_err = a
        .iter()
        .zip(&b_vals)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);

    assert!(
        max_err < 0.02 * max_abs_ref.max(1.0),
        "RotKTq4V K-dequant score error {max_err} too large vs \
         |scores|={max_abs_ref} (must match RotK tolerance)"
    );
}

// ── Decode-path oracle gate, straddling the context length ───────────────────
//
// `mixed_quantized_sdpa` at L=1 is the single-token decode path for every
// `mixed_*` / `rot_k_*` cell. The oracle below dequantizes the *same* affine
// 3-tuples the codec holds (`rmlx_mlx::dequantize`) and runs stock
// `scaled_dot_product_attention` on the resulting float K/V. It shares no
// arithmetic with the codec path, so it stays a valid reference whatever that
// path does internally — a reference that re-derives the codec's own unpack
// formula instead agrees with whatever convention the codec happens to use, and
// that is precisely how an affine-vs-symmetric V-dequant offset once passed a
// green test suite here.
//
// The two context lengths are the substance of the test, not thoroughness: a
// decode path that switches implementation on `t_seq` must return the same
// answer either side of the switch, so the gate has to sample both.

/// Relative L2 error of `mixed_quantized_sdpa` (L=1, k8g64/v4g64) against the
/// dequantize + stock-SDPA oracle, for one `(kv_heads, repeats, head_dim)`
/// shape at one context length.
///
/// Relative L2, not cosine: cosine compares direction only, so it scores a
/// uniformly shrunk output as a perfect match. An attention path can lose
/// probability mass without turning, and that is a defect this gate has to see.
#[allow(
    clippy::unwrap_used,
    reason = "every unwrap here is on a fixture this fn just built at a known shape, or on an MLX op over it; a failure is a broken test fixture, which should abort the test"
)]
fn mixed_decode_oracle_rel_l2(kv_heads: i32, repeats: i32, head_dim: i32, t_seq: i32) -> f64 {
    let device = Device::Gpu;
    let b = 1i32;
    let n_q_heads = kv_heads * repeats;
    let (k_group, k_bits, v_group, v_bits) = (64i32, 8i32, 64i32, 4i32);

    let q_n = (b * n_q_heads * head_dim) as usize;
    let kv_n = (b * kv_heads * t_seq * head_dim) as usize;
    let q = from_vec(&gaussian_data(q_n, TEST_SEED), &[b, n_q_heads, 1, head_dim]);
    let k = from_vec(
        &gaussian_data(kv_n, TEST_SEED ^ 0xA5A5),
        &[b, kv_heads, t_seq, head_dim],
    );
    let v = from_vec(
        &gaussian_data(kv_n, TEST_SEED ^ 0x5A5A),
        &[b, kv_heads, t_seq, head_dim],
    );

    let (kc, ks, kb) = rmlx_mlx::quantize(&k, k_group, k_bits, device).unwrap();
    let (vc, vs, vb) = rmlx_mlx::quantize(&v, v_group, v_bits, device).unwrap();
    let scale = ORACLE_PEAK_FACTOR / (head_dim as f32).sqrt();

    let out = mixed_quantized_sdpa(
        &q,
        &MixedTuple {
            codes: kc.try_clone().unwrap(),
            scales: ks.try_clone().unwrap(),
            biases: kb.try_clone().unwrap(),
        },
        &MixedTuple {
            codes: vc.try_clone().unwrap(),
            scales: vs.try_clone().unwrap(),
            biases: vb.try_clone().unwrap(),
        },
        scale,
        None,
        k_group,
        k_bits,
        v_group,
        v_bits,
        None,
        device,
        DispatchPolicy::default(),
    )
    .unwrap();

    let k_dq = dequantize(&kc, &ks, Some(&kb), k_group, k_bits, "affine", device).unwrap();
    let v_dq = dequantize(&vc, &vs, Some(&vb), v_group, v_bits, "affine", device).unwrap();
    let oracle = scaled_dot_product_attention(&q, &k_dq, &v_dq, scale, "", None, device).unwrap();

    let (want, got) = (to_vec(&oracle), to_vec(&out));
    let err: f64 = want
        .iter()
        .zip(&got)
        .map(|(a, b)| f64::from(a - b).powi(2))
        .sum();
    let norm: f64 = want.iter().map(|a| f64::from(*a).powi(2)).sum();
    err.sqrt() / norm.sqrt()
}

/// Query scale multiplier, in units of `1/sqrt(head_dim)`, that sharpens the
/// softmax away from the near-uniform distribution a unit-scaled Gaussian
/// fixture produces.
///
/// Chosen by measuring sensitivity, not by intuition. The defect class this
/// fixture has to expose is *lost probability mass* — a path that silently drops
/// small weights — and the mass available to lose is not monotone in peaking.
/// Measured worst-case relative L2 for a mass-dropping path against the correct
/// one, over these shapes: 1.7e-4 vs 2.4e-6 at factor 3, 2.3e-5 vs 3.1e-6 at 6,
/// and 3.3e-6 vs 2.7e-6 by factor 16 — where the two are indistinguishable. A
/// near-one-hot softmax has almost no tail left to drop, so peaking the fixture
/// harder makes the gate blinder, not sharper. Factor 3 gives the widest
/// separation (73x) and is what the ceiling below is sized against.
const ORACLE_PEAK_FACTOR: f32 = 3.0;

/// Ceiling on relative L2 error against the oracle, placed in the gap between
/// the two measured populations rather than near either edge.
///
/// A correct path floors at the accumulation-order difference between
/// `quantized_matmul` and stock SDPA — 2.4e-6 worst case here, and deterministic
/// (this is pinned-fixture GPU arithmetic, not a timing measurement; repeated
/// runs agree to ~1e-8 relative). The two defects this gate exists for land at
/// 1.7e-4 (probability mass dropped before the V matmul) and ~2.9e0 (V
/// dequantized with the wrong convention). 2e-5 is the geometric midpoint of the
/// floor and the nearer defect: 8.4x of headroom above a correct path, 8.7x of
/// margin below the closest thing it must reject.
const ORACLE_REL_L2_CEILING: f64 = 2e-5;

/// The decode path must agree with the dequantize + stock-SDPA oracle at every
/// context length and every head layout it serves.
///
/// 4 096 and 8 448 sit either side of 8 192, the context length at which the V
/// side once switched to a separate kernel. The three head layouts cover both
/// arms of the GQA branch: `repeats > 1` takes the `expand_dims` broadcast and
/// the trailing reshape, `repeats == 1` takes neither and returns the
/// `quantized_matmul` output unreshaped — the MHA-shaped Mixed cache, and now
/// the only path serving it.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant --lib -- --ignored mixed_decode --test-threads=1"]
fn mixed_decode_matches_dequant_oracle_across_context() {
    // Every cell is measured before anything is asserted: the point is to see
    // both sides of the context sweep, and a first-failure abort would hide
    // whichever side happened to run second.
    use std::fmt::Write as _;

    let mut report = String::new();
    let mut worst = 0.0f64;
    for (kv_heads, repeats, head_dim) in [
        (2i32, 4i32, 128i32), // GQA
        (1i32, 4i32, 128i32), // single KV head, shared-KV shape
        (2i32, 1i32, 128i32), // MHA — the n_repeats == 1 arm
    ] {
        for t_seq in [4096i32, 8448i32] {
            let rel_l2 = mixed_decode_oracle_rel_l2(kv_heads, repeats, head_dim, t_seq);
            worst = worst.max(rel_l2);
            let _ = write!(
                report,
                "\n  kv_heads={kv_heads} repeats={repeats} head_dim={head_dim} \
                 t_seq={t_seq}: relative L2 {rel_l2:e}"
            );
        }
    }
    assert!(
        worst <= ORACLE_REL_L2_CEILING,
        "mixed_quantized_sdpa disagrees with the dequantize + stock-SDPA oracle \
         (ceiling {ORACLE_REL_L2_CEILING:e}):{report}"
    );
}

// ── Mixed cosine-similarity gate ─────────────────────────────────────────────
//
// Each test quantizes a [1, 4, 64, 128] V-tensor with `mx.quantize` (affine
// mode) at the specified (v_bits, v_group_size) and dequantizes it back, then
// measures per-row (head_dim=128) cosine similarity against the original.
//
// Shape [1, 4, 64, 128]: last dim=128 is divisible by all three group_sizes
// (128, 64, 32), so all three Mixed cells can share one fixture.
//
// Fixture: LCG-seeded [-1, 1] uniform, TEST_SEED — same as other codec tests.

// [1, 4, 64, 128] = 32 768 elements; last dim=128 divisible by group_sizes 128/64/32.
const MIXED_SHAPE: [i32; 4] = [1, 4, 64, 128];
const MIXED_HEAD_DIM: usize = 128;

fn mixed_cosine_fixture() -> Vec<f32> {
    let n: usize = MIXED_SHAPE.iter().map(|&d| d as usize).product();
    lcg_data(n, TEST_SEED)
}

/// Helper: quantize + dequantize with `mx.quantize` affine.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn mlx_affine_roundtrip(data: &[f32], shape: &[i32], bits: i32, group_size: i32) -> Vec<f32> {
    let device = Device::Cpu;
    let arr = from_vec(data, shape);
    let (codes, scales, biases) = rmlx_mlx::quantize(&arr, group_size, bits, device).unwrap();
    let dq = dequantize(
        &codes,
        &scales,
        Some(&biases),
        group_size,
        bits,
        "affine",
        device,
    )
    .unwrap();
    to_vec(&dq)
}

/// Mixed {k_bits:8, v_bits:4, k_group:128, v_group:64} V-side cosine gate: mean ≥ 0.9937.
///
/// Threshold: per ../multi-turboquant README matrix row `turbo4` = 0.9947 − 0.001.
#[test]
fn mixed_k8v4_g128_64_cosine_gate() {
    let data = mixed_cosine_fixture();
    let decoded = mlx_affine_roundtrip(&data, &MIXED_SHAPE, 4, 64);
    let stats = cosine_similarity_per_row(&data, &decoded, MIXED_HEAD_DIM);
    assert!(
        stats.mean >= 0.9937,
        // per ../multi-turboquant README matrix row `turbo4` = 0.9947 − 0.001
        "Mixed{{k_bits:8, v_bits:4, k_group:128, v_group:64}} mean cosine {:.6} < 0.9937 \
         (empirical floor)",
        stats.mean,
    );
}

/// Mixed {k_bits:8, v_bits:8, k_group:128, v_group:128} V-side cosine gate: mean ≥ 0.9990.
///
/// Threshold: empirical floor measured 2026-05-30, set to `measured − 0.001`.
#[test]
fn mixed_k8v8_g128_128_cosine_gate() {
    let data = mixed_cosine_fixture();
    let decoded = mlx_affine_roundtrip(&data, &MIXED_SHAPE, 8, 128);
    let stats = cosine_similarity_per_row(&data, &decoded, MIXED_HEAD_DIM);
    assert!(
        stats.mean >= 0.9990,
        // empirical floor measured 2026-05-30
        "Mixed{{k_bits:8, v_bits:8, k_group:128, v_group:128}} mean cosine {:.6} < 0.9990 \
         (empirical floor)",
        stats.mean,
    );
}

/// Mixed {k_bits:8, v_bits:2, k_group:128, v_group:32} V-side cosine gate: mean ≥ 0.9000.
///
/// Threshold: empirical floor measured 2026-05-30, set to `measured − 0.001`.
#[test]
fn mixed_k8v2_g128_32_cosine_gate() {
    let data = mixed_cosine_fixture();
    let decoded = mlx_affine_roundtrip(&data, &MIXED_SHAPE, 2, 32);
    let stats = cosine_similarity_per_row(&data, &decoded, MIXED_HEAD_DIM);
    assert!(
        stats.mean >= 0.9000,
        // empirical floor measured 2026-05-30
        "Mixed{{k_bits:8, v_bits:2, k_group:128, v_group:32}} mean cosine {:.6} < 0.9000 \
         (empirical floor)",
        stats.mean,
    );
}
