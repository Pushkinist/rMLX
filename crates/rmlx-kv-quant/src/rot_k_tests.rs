use super::*;
use crate::test_utils::{cosine_similarity_per_row, fwht_normalize, lcg_data, TEST_SEED};

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

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn from_vec(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn hadamard_is_orthogonal() {
    let d = 8;
    let r = hadamard_rotation(d, Dtype::F32, Device::Cpu).unwrap();
    // R is symmetric so R R == I.
    let rr = matmul(&r, &r, Device::Cpu).unwrap();
    let v = to_vec(&rr);
    for i in 0..d {
        for j in 0..d {
            let want = if i == j { 1.0 } else { 0.0 };
            assert!(
                (v[i * d + j] - want).abs() < 1e-5,
                "R R[{i},{j}] = {} expected {want}",
                v[i * d + j]
            );
        }
    }
}

#[test]
fn non_power_of_two_rejected() {
    assert!(hadamard_rotation(96, Dtype::F32, Device::Cpu).is_err());
    assert!(hadamard_rotation(0, Dtype::F32, Device::Cpu).is_err());
}

/// The crux numerical proof: rotated-K with pre-rotated-Q reproduces the
/// unrotated Q·Kᵀ scores exactly (no quantization — pure rotation
/// cancellation `(Q Rᵀ)(K Rᵀ)ᵀ = Q Kᵀ`).
#[test]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn pre_rotate_q_cancellation_identity() {
    let d = 16i32;
    let lq = 3i32;
    let lk = 5i32;
    let device = Device::Cpu;

    let q_data: Vec<f32> = (0..(lq * d)).map(|i| ((i * 7 % 13) as f32) - 6.0).collect();
    let k_data: Vec<f32> = (0..(lk * d))
        .map(|i| ((i * 11 % 17) as f32) - 8.0)
        .collect();
    let q = from_vec(&q_data, &[lq, d]);
    let k = from_vec(&k_data, &[lk, d]);

    // Unrotated scores: Q · Kᵀ → [Lq, Lk].
    let k_t = k.transpose(&[1, 0], device).unwrap();
    let scores_ref = matmul(&q, &k_t, device).unwrap();

    // Rotated path: K_rot = K @ R, Q_rot = Q @ R, scores = Q_rot · K_rotᵀ.
    let r = hadamard_rotation(d as usize, Dtype::F32, device).unwrap();
    let k_rot = rotate_last_axis(&k, &r, device).unwrap();
    let q_rot = rotate_last_axis(&q, &r, device).unwrap();
    let k_rot_t = k_rot.transpose(&[1, 0], device).unwrap();
    let scores_rot = matmul(&q_rot, &k_rot_t, device).unwrap();

    let a = to_vec(&scores_ref);
    let b = to_vec(&scores_rot);
    let max_err = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_err < 1e-3,
        "pre-rotate-Q cancellation failed: max score error {max_err}"
    );
}

// ── Cosine-similarity gate ────────────────────────────────────────────────────

// Fixture shape [1, 4, 128, 64] = 32 768 elements; rows are head_dim=64 slices.
// head_dim=64 must be a power-of-two for the Hadamard transform.
const COSINE_N: usize = 4 * 128 * 64;
const COSINE_HEAD_DIM: usize = 64;

fn cosine_fixture() -> Vec<f32> {
    lcg_data(COSINE_N, TEST_SEED)
}

/// rot_k Hadamard 8-bit cosine gate (CPU path): mean cosine ≥ 0.9970.
///
/// Protocol: apply normalized FWHT to each head_dim=64 row (the rot_k
/// "K_rot = K @ R" step), quantize with symmetric 8-bit affine at group_size=64
/// (matching the rot_k default `k_group_size=64`), dequantize, apply FWHT again
/// (self-inverse: R R = I for the normalized Hadamard), then measure cosine vs
/// original K.
///
/// Threshold: empirical floor measured 2026-05-30, set to `measured − 0.001`.
/// Uses the CPU reference FWHT path; the MLX path (hadamard_rotation +
/// rotate_last_axis) is tested separately in pre_rotate_q_cancellation_identity.
#[test]
fn rot_k_hadamard_8bit_cosine_gate() {
    const GROUP: usize = 64; // rot_k default k_group_size=64

    let data = cosine_fixture();

    // Rotate: K_rot = data @ R (FWHT per row, normalized).
    let mut rotated = data.clone();
    fwht_normalize(&mut rotated, COSINE_HEAD_DIM);

    // Quantize each row as one symmetric 8-bit affine group of 64 elements.
    // (q8_quantize uses group_size=128; we inline the same logic for group_size=64.)
    let n_rows = rotated.len() / GROUP;
    let mut decoded_rotated = vec![0.0f32; rotated.len()];
    for row in 0..n_rows {
        let start = row * GROUP;
        let chunk = &rotated[start..start + GROUP];
        let abs_max = chunk.iter().copied().fold(0.0f32, |a, v| a.max(v.abs()));
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 0.0 };
        let inv_scale = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let out = &mut decoded_rotated[start..start + GROUP];
        for (&v, slot) in chunk.iter().zip(out.iter_mut()) {
            let code = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
            *slot = scale * f32::from(code);
        }
    }

    // Inverse rotate: apply FWHT again (R is self-inverse for normalized Hadamard).
    fwht_normalize(&mut decoded_rotated, COSINE_HEAD_DIM);

    let stats = cosine_similarity_per_row(&data, &decoded_rotated, COSINE_HEAD_DIM);

    assert!(
        stats.mean >= 0.9970,
        // empirical floor measured 2026-05-30
        "rot_k Hadamard 8-bit mean cosine {:.6} < 0.9970 (rot_k gate)",
        stats.mean,
    );
    // min assertion: docs/TESTING.md originally claimed min≥0.9950 but only checked mean.
    // Remeasured on corrected LCG fixture (>> 32 fix): min=0.9999617
    // → floor 0.9999617 − 0.001 = 0.9990 (empirical, 2026-05-30).
    assert!(
        stats.min >= 0.9990,
        "rot_k Hadamard 8-bit min cosine {:.6} < 0.9990 (empirical floor)",
        stats.min,
    );
}
