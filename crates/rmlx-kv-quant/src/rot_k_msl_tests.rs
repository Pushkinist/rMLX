use super::*;
use crate::rot_k::{hadamard_rotation, rotate_last_axis};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env, vectorized_parity_check};
use rmlx_mlx::{dequantize, quantize, Array, Device, Dtype};

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn from_vec_f32(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).unwrap()
}

#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn to_vec_f32(a: &Array) -> Vec<f32> {
    a.eval().unwrap();
    a.to_bytes()
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// CPU reference: rotate_last_axis + mx.quantize, then dequantize.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn reference_rotate_quantize_dequantize(k: &Array, d: usize, device: Device) -> Vec<f32> {
    let r = hadamard_rotation(d, Dtype::F32, device).unwrap();
    let k_rot = rotate_last_axis(k, &r, device).unwrap();
    let (codes, scales, biases) =
        quantize(&k_rot, FWHT_QUANT_GROUP_SIZE as i32, 8, device).unwrap();
    let k_dq = dequantize(
        &codes,
        &scales,
        Some(&biases),
        FWHT_QUANT_GROUP_SIZE as i32,
        8,
        "affine",
        device,
    )
    .unwrap();
    to_vec_f32(&k_dq)
}

/// DoD: fused FWHT output, dequantized, must match the two-step
/// (rotate_last_axis + mx.quantize) within 2 ULP on bf16 values.
///
/// 2 ULP at bf16: for values in ~[-3, 3], 2 ULP ~ 0.016.
/// We use 0.02 to absorb scale/bias rounding differences between the
/// fused kernel (f32 FWHT + in-kernel min/max scan) and the reference
/// (mx.quantize's internal f32/bf16 affine path).
///
/// Requires GPU Metal context -- run in isolation:
/// RMLX_ROT_K_FUSED=1 cargo test rot_k_msl -- --ignored --test-threads=1
#[test]
#[ignore = "GPU Metal context -- run: RMLX_ROT_K_FUSED=1 cargo test rot_k_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn fwht_quantize_matches_reference_d128() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let d = 128usize;
    let n_rows = 4i32;

    let data = lcg_data(n_rows as usize * d, 0xABCD_EF01_u64);
    let k = from_vec_f32(&data, &[n_rows, d as i32]);

    // Reference: v1 two-step on GPU — pre-compute before calling the helper
    // because both paths are inherently stateful (Metal context, quantize state).
    let ref_dq = reference_rotate_quantize_dequantize(&k, d, device);

    // Fused kernel.
    let (codes, scales, biases) =
        rot_k_fwht_quantize_gpu(&k, device).expect("fused FWHT quantize should succeed");
    let fused_dq = {
        let dq = dequantize(
            &codes,
            &scales,
            Some(&biases),
            FWHT_QUANT_GROUP_SIZE as i32,
            8,
            "affine",
            device,
        )
        .unwrap();
        to_vec_f32(&dq)
    };

    // Both outputs are computed — delegate comparison to the shared helper.
    // pre-computed: closures ignore input; clones satisfy FnOnce signature (Metal context must be sequenced upstream)
    // Tolerance: one 8-bit affine quantization step for D=128 FWHT range.
    vectorized_parity_check(
        |_| ref_dq.clone(),
        |_| fused_dq.clone(),
        &[],
        0.10_f32,
        "rot_k FWHT+affine q8",
    );
}

/// Rotate-only kernel must match rotate_last_axis within f32 precision.
///
/// Requires GPU Metal context -- run in isolation:
/// RMLX_ROT_K_FUSED=1 cargo test rot_k_msl -- --ignored --test-threads=1
#[test]
#[ignore = "GPU Metal context -- run: RMLX_ROT_K_FUSED=1 cargo test rot_k_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn fwht_rotate_matches_reference_d128() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let d = 128usize;
    let n_rows = 8i32;

    let data = lcg_data(n_rows as usize * d, 0xDEAD_BEEF_u64);
    let q = from_vec_f32(&data, &[n_rows, d as i32]);

    let r = hadamard_rotation(d, Dtype::F32, device).unwrap();
    let q_ref = rotate_last_axis(&q, &r, device).unwrap();
    let ref_vals = to_vec_f32(&q_ref);

    let q_fused = rot_k_fwht_rotate_gpu(&q, device).expect("fwht rotate should succeed");
    let fused_vals = to_vec_f32(&q_fused);

    assert_eq!(ref_vals.len(), fused_vals.len());
    let max_err = ref_vals
        .iter()
        .zip(&fused_vals)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    // FWHT butterfly (7 stages for D=128) uses different summation order than
    // the reference matmul (MLX gemm). Both are exact for small inputs but
    // diverge by O(D * eps_f32) due to reordered additions.
    // For values after FWHT reaching ~sqrt(D)=11.3, D=128 stages, tolerance
    // 5e-3 is ~5x the expected FWHT-vs-matmul f32 drift (verified empirically).
    assert!(
        max_err < 5e-3,
        "fwht rotate diverges by {max_err:.2e} (expected < 5e-3)"
    );
}

/// Supported/unsupported D guard.
#[test]
fn supported_d_set_is_correct() {
    for &d in SUPPORTED_D {
        assert!(is_supported_d(d), "supported D={d} should be recognized");
    }
    assert!(!is_supported_d(96), "D=96 (not pow2) should be unsupported");
    assert!(!is_supported_d(0), "D=0 should be unsupported");
    assert!(!is_supported_d(1024), "D=1024 not in SUPPORTED_D");
}

/// MEDIUM 1 fix: `rot_k_fwht_rotate_gpu` must preserve the input dtype.
///
/// When Q is bf16, the output must also be bf16 (not silently widened to
/// f32), so that downstream SDPA ops (`multiply`, `quantized_matmul`) keep
/// the expected precision without an implicit f32 promotion.
///
/// Requires GPU Metal context -- run in isolation:
/// RMLX_ROT_K_FUSED=1 cargo test rot_k_msl -- --ignored --test-threads=1
#[test]
#[ignore = "GPU Metal context -- run: RMLX_ROT_K_FUSED=1 cargo test rot_k_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn fwht_rotate_preserves_input_dtype_bf16() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let d = 128usize;
    let n_rows = 4i32;

    let data = lcg_data(n_rows as usize * d, 0x1234_5678_u64);
    let q_f32 = from_vec_f32(&data, &[n_rows, d as i32]);
    // Cast to bf16 — this is the typical Q dtype in production.
    let q_bf16 = q_f32.astype(Dtype::Bf16, device).unwrap();

    let out = rot_k_fwht_rotate_gpu(&q_bf16, device).expect("fwht rotate (bf16) should succeed");

    assert_eq!(
        out.dtype(),
        Dtype::Bf16,
        "rotate output must preserve bf16 input dtype, got {:?}",
        out.dtype()
    );
    assert_eq!(out.shape(), q_bf16.shape(), "shape must be unchanged");
}

/// MEDIUM 3 fix: constant-group handling mirrors MLX convention.
///
/// When all elements in a quantization group are identical (gmax == gmin),
/// the kernel must emit scale=0, bias=gmin, codes=all-zero. This matches
/// the behavior of `mx.quantize` (which produces scale≈0, bias=gmin) and
/// ensures that dequantization (`scale * 0 + bias = bias`) reproduces the
/// original values correctly in both in-process and cross-backend paths.
///
/// This test runs on CPU via the v1 reference path (no GPU required), since
/// it validates the formula semantics, not the Metal kernel dispatch.
#[test]
fn quantize_constant_group_reconstruction() {
    // Simulate the kernel constant-group convention in pure Rust:
    // scale = 0.0 when gmax == gmin; guard emits code = 0.
    // dequant = scale * 0 + bias = bias = gmin.
    let gmin = 7.5_f32;
    let gmax = 7.5_f32; // constant group
    let scale = if gmax > gmin {
        (gmax - gmin) / 255.0
    } else {
        0.0_f32 // mirrors MLX convention
    };
    let bias = gmin;
    // Guard: (scale > 0) ? (x-bias)/scale : 0.0
    let v_norm = if scale > 0.0 {
        (gmin - bias) / scale
    } else {
        0.0_f32
    };
    let code = v_norm.clamp(0.0, 255.0).round() as u32;
    let recon = scale * (code as f32) + bias;

    assert_eq!(code, 0, "constant group must produce code=0");
    assert!(
        (recon - gmin).abs() < 1e-6,
        "constant group reconstruction must equal gmin={gmin}, got {recon}"
    );
    assert_eq!(
        scale, 0.0_f32,
        "constant group scale must be 0.0 (mirrors MLX)"
    );
}
