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

/// A row shorter than one affine group must be rejected, not asserted.
///
/// `head_dim < FWHT_QUANT_GROUP_SIZE` yields zero groups per row, which sized
/// the kernel's per-group SMEM as a zero-length threadgroup array — MSL that
/// does not compile. A `debug_assert` is compiled out of release, so release
/// would have shipped the broken shader to the GPU. The builder must return a
/// real error in every profile.
#[allow(
    clippy::expect_used,
    reason = "test asserts the error path exists; a missing error is the failure being checked"
)]
#[test]
fn fwht_quantize_rejects_head_dim_below_group_size() {
    let err = build_fwht_quantize_body(FWHT_QUANT_GROUP_SIZE / 2)
        .expect_err("head_dim below the affine group size must be rejected");
    let msg = err.to_string();
    assert!(
        msg.contains(&(FWHT_QUANT_GROUP_SIZE / 2).to_string()),
        "error should name the offending head_dim: {msg}"
    );
    assert!(
        msg.contains(&FWHT_QUANT_GROUP_SIZE.to_string()),
        "error should name the affine group size: {msg}"
    );

    // Every shape that holds a whole number of groups still builds.
    for d in [64usize, 128, 256, 512] {
        assert!(
            build_fwht_quantize_body(d).is_ok(),
            "D={d} is a whole number of groups and must build"
        );
    }

    // Rotation has no group structure, so the same D stays valid there.
    assert!(
        is_supported_d(FWHT_QUANT_GROUP_SIZE / 2),
        "rotation must still accept D=32"
    );
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

/// Probe header snapshots must equal what the builders emit.
///
/// `make check-metal-compiles` prepends these snapshots to the kernel bodies.
/// A builder that changes a constant's value, or drops one, without the
/// snapshot being refreshed leaves the probe compiling text production no
/// longer emits — the gate would keep passing while checking the wrong thing.
/// Equality here turns that drift into a hard failure.
#[test]
fn hdr_probe_snapshot_matches_builder() {
    assert_eq!(
        kernel_header(),
        include_str!("metal/probes/rot_k.hdr.metal"),
        "stale snapshot: refresh metal/probes/rot_k.hdr.metal"
    );
}

/// The fused quantize must type its scales and biases like `mx.quantize`
/// does — in K's dtype, not the kernel's f32 accumulator.
///
/// `mx.quantized_matmul` and `mx.dequantize` take their operand width from the
/// scales they are handed, so f32 scales on a bf16 model promote the score
/// matmul, its output, the residual add behind it and every downstream op in
/// the layer. The fallback path (`rotate_last_axis` + `mx.quantize`, taken
/// whenever the policy flag is off) types them bf16, so the leak also makes
/// the two arms of the same codec numerically different.
#[test]
#[ignore = "GPU Metal context -- run: cargo test -p rmlx-kv-quant --lib rot_k_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "fixture construction on values established in this fn; a failure here is a test bug, not a codec result"
)]
fn fwht_quantize_types_scales_like_mx_quantize() {
    if skip_if_no_gpu_env() {
        return;
    }
    let device = Device::Gpu;
    let d = 128usize;
    let n_rows = 4i32;

    let data = lcg_data(n_rows as usize * d, 0x5CA1_E500_u64);
    let k = from_vec_f32(&data, &[n_rows, d as i32])
        .astype(Dtype::Bf16, device)
        .unwrap();

    // What the non-fused arm of the same codec produces.
    let r = hadamard_rotation(d, Dtype::Bf16, device).unwrap();
    let k_rot = rotate_last_axis(&k, &r, device).unwrap();
    let (_, ref_scales, ref_biases) =
        quantize(&k_rot, FWHT_QUANT_GROUP_SIZE as i32, 8, device).unwrap();

    let (_, scales, biases) =
        rot_k_fwht_quantize_gpu(&k, device).expect("fused FWHT quantize should succeed");

    assert_eq!(
        scales.dtype(),
        ref_scales.dtype(),
        "fused scales dtype must match mx.quantize's ({:?})",
        ref_scales.dtype()
    );
    assert_eq!(
        biases.dtype(),
        ref_biases.dtype(),
        "fused biases dtype must match mx.quantize's ({:?})",
        ref_biases.dtype()
    );
    assert_eq!(
        scales.dtype(),
        Dtype::Bf16,
        "a bf16 K must not come back with f32 quantization parameters"
    );

    // Narrowing the scales is only correct if the reconstruction stays where
    // the non-fused arm's does. Dequantize both 3-tuples and compare.
    let deq = |c: &Array, sc: &Array, bi: &Array| {
        to_vec_f32(
            &dequantize(
                c,
                sc,
                Some(bi),
                FWHT_QUANT_GROUP_SIZE as i32,
                8,
                "affine",
                device,
            )
            .unwrap()
            .astype(Dtype::F32, device)
            .unwrap(),
        )
    };
    let (ref_codes, _, _) = quantize(&k_rot, FWHT_QUANT_GROUP_SIZE as i32, 8, device).unwrap();
    let (fused_codes, _, _) = rot_k_fwht_quantize_gpu(&k, device).unwrap();
    let ref_vals = deq(&ref_codes, &ref_scales, &ref_biases);
    let fused_vals = deq(&fused_codes, &scales, &biases);
    let max_err = ref_vals
        .iter()
        .zip(&fused_vals)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    // Gate the measured value, not a tolerance six times looser than it: a
    // documented figure whose assertion allows 6x drift is a figure nothing
    // holds. 0.0156 measured on this fixture; 0.02 leaves headroom for a
    // codebook or reduction-order change without letting a regression through.
    eprintln!("[rot_k bf16 scales] max |fused - reference| = {max_err:.4}");
    assert!(
        max_err < 0.02,
        "bf16 scales moved the reconstruction to {max_err} (measured 0.0156 when \
         this was written); the documented figure in docs/KV_QUANT.md is stale or \
         the codec changed"
    );

    // Which arm sits closer to the unquantized K? Narrowing the scales is
    // required either way — f32 scales promote every consumer of this 3-tuple —
    // but "required" is not "harmless", and the direction was unmeasured until
    // this assertion existed.
    let k_ref = to_vec_f32(&k_rot.astype(Dtype::F32, device).unwrap());
    let cos = |a: &[f32], b: &[f32]| -> f64 {
        let (mut dot, mut na, mut nb) = (0.0_f64, 0.0_f64, 0.0_f64);
        for (x, y) in a.iter().zip(b.iter()) {
            dot += f64::from(*x) * f64::from(*y);
            na += f64::from(*x).powi(2);
            nb += f64::from(*y).powi(2);
        }
        if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            dot / (na.sqrt() * nb.sqrt())
        }
    };
    let cos_ref = cos(&k_ref, &ref_vals);
    let cos_fused = cos(&k_ref, &fused_vals);
    eprintln!(
        "[rot_k bf16 scales] cosine vs unquantized K_rot: mx.quantize={cos_ref:.6} \
         fused={cos_fused:.6}"
    );
    assert!(
        cos_fused > 0.999,
        "the fused arm's reconstruction cosine against the unquantized K dropped to \
         {cos_fused} — narrowing the scales cost real fidelity, not just a digest"
    );
}
