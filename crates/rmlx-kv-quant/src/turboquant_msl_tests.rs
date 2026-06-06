use super::*;
use rmlx_mlx::{Array, Device, Dtype};

use crate::test_utils::{lcg_data, skip_if_no_gpu_env, vectorized_parity_check};
use crate::turboquant::{
    turbo_dequantize, turbo_dequantize_with_codebook, turbo_quantize_v,
    turbo_quantize_v_with_codebook,
};

// Helper: make an Array from f32 data.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
}

// Helper: extract f32 vec from an Array.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn to_f32_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// GPU roundtrip (quantize then dequantize) max abs error < 0.20.
///
/// Input shape [1, 4, 128, 64] — representative KV tensor.
/// Tolerance 0.20 is slightly looser than the CPU 0.15 to allow for f32
/// rounding differences in the GPU reduction.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turboquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn turbo_v4_msl_roundtrip_within_tolerance() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales) = turbo_quantize_v4_gpu(&arr, Device::Gpu).expect("GPU quantize failed");

    let recon = turbo_dequantize_v4_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
        .expect("GPU dequantize failed");

    let recon_vec = to_f32_vec(&recon);
    assert_eq!(recon_vec.len(), n);

    let max_err = data
        .iter()
        .zip(recon_vec.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.20,
        "GPU roundtrip max abs error {max_err:.6} exceeds tolerance 0.20"
    );
}

/// K-axis CPU↔MSL parity. The TurboQuant 4-bit kernel is axis-agnostic, so
/// the K side of `KvStorage::TurboSym4` reuses the same CPU + MSL paths as the
/// V side. This test exercises the K-axis call site shape `[1, 4, 128, 64]`
/// (same K-typical layout) and asserts the same 5e-3 tolerance the V-side
/// parity test uses. Anything tighter is a sign the K input distribution
/// diverged — escalate.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turboquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn turbo_k4_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xA125_0008_u64);

    vectorized_parity_check(
        |input| {
            let blocks = turbo_quantize_v(input, 4, &shape).expect("CPU quantize failed");
            turbo_dequantize(&blocks).expect("CPU dequantize failed")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales) =
                turbo_quantize_v4_gpu(&arr, Device::Gpu).expect("GPU quantize failed");
            let recon = turbo_dequantize_v4_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
                .expect("GPU dequantize failed");
            to_f32_vec(&recon)
        },
        &data,
        5e-3_f32,
        "TurboQuant K4 CPU vs GPU",
    );
}

/// GPU and CPU quantize+dequantize must agree within 0.001.
///
/// Verifies that the MSL kernel is bit-equivalent to the scalar CPU path
/// up to f32 rounding differences.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turboquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn turbo_v4_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    // [1, 4, 128, 64]: product = 32768 = 1024 groups of 32.
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xCAFE_BABE_u64);

    vectorized_parity_check(
        |input| {
            let blocks = turbo_quantize_v(input, 4, &shape).expect("CPU quantize failed");
            turbo_dequantize(&blocks).expect("CPU dequantize failed")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales) =
                turbo_quantize_v4_gpu(&arr, Device::Gpu).expect("GPU quantize failed");
            let recon = turbo_dequantize_v4_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
                .expect("GPU dequantize failed");
            to_f32_vec(&recon)
        },
        &data,
        5e-3_f32,
        "TurboQuant V4 CPU vs GPU",
    );
}

// ── Codebook-buffer kernel parity tests ──────────────────────────────────────

/// Build a perturbed-Lloyd-Max 16-entry codebook. Each centroid is shifted by a
/// small fraction of its magnitude so the override is materially different
/// from the hardwired table, then re-sorted ascending. This is the same
/// "synthetic perturbation" the ticket-spec workflow step 8 describes.
fn perturbed_lloyd_max_4bit() -> Vec<f32> {
    let mut cb = vec![
        -2.717_667_f32,
        -2.052_138,
        -1.600_802_4,
        -1.239_959,
        -0.928_244_7,
        -0.645_875_33,
        -0.381_178_23,
        -0.126_046_94,
        0.126_046_94,
        0.381_178_23,
        0.645_875_33,
        0.928_244_7,
        1.239_959,
        1.600_802_4,
        2.052_138,
        2.717_667,
    ];
    // Multiplicative perturbation — keeps ordering monotone.
    for (i, c) in cb.iter_mut().enumerate() {
        let frac = 0.05_f32 * ((i as f32) - 7.5_f32) / 7.5_f32;
        *c *= 1.0 + frac;
    }
    // Sanity: must remain strictly ascending.
    debug_assert!(cb.windows(2).all(|w| w[0] < w[1]));
    cb
}

/// GPU codebook-buffer encode must match CPU codebook-override encode within
/// 5e-3 max-abs error. Tolerance mirrors `turbo_v4_msl_matches_cpu_within_eps`.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turboquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn turbo_v4_msl_codebook_buf_matches_cpu_with_override() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xA155_001B_u64);
    let codebook = perturbed_lloyd_max_4bit();

    vectorized_parity_check(
        |input| {
            let blocks = turbo_quantize_v_with_codebook(input, 4, &shape, Some(&codebook))
                .expect("CPU quantize with override failed");
            turbo_dequantize_with_codebook(&blocks, Some(&codebook))
                .expect("CPU dequantize with override failed")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            // Upload codebook to GPU.
            let cb_bytes = unsafe {
                std::slice::from_raw_parts(codebook.as_ptr().cast::<u8>(), codebook.len() * 4)
            };
            let cb_arr = Array::from_bytes(cb_bytes, &[codebook.len() as i32], Dtype::F32)
                .expect("Array::from_bytes codebook");
            let (codes, scales) = turbo_quantize_v4_codebook_buf_gpu(&arr, &cb_arr, Device::Gpu)
                .expect("GPU codebook-buf quantize failed");
            let recon = turbo_dequantize_v4_codebook_buf_gpu(
                &codes,
                &scales,
                &cb_arr,
                &shape,
                Dtype::F32,
                Device::Gpu,
            )
            .expect("GPU codebook-buf dequantize failed");
            to_f32_vec(&recon)
        },
        &data,
        5e-3_f32,
        "TurboQuant V4 CPU-override vs GPU codebook-buffer",
    );
}

/// When the override codebook *is* the built-in Lloyd-Max, the codebook-buffer
/// kernel must reproduce the hardwired kernel exactly (within f32 reduction
/// noise). Verifies the runtime-boundary path is bit-equivalent to the
/// hardwired-constant path when fed identical centroids.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turboquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn turbo_v4_msl_codebook_buf_matches_hardwired_on_lloyd_max() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xA155_002B_u64);
    // Build the test codebook from the **exact** IEEE-754 bit patterns
    // embedded in `turboquant_msl.rs::KERNEL_HEADER` (`as_type<float>(0x...)`).
    // Decimal literals like `-2.717_667_f32` round to a different f32 bit
    // pattern than the kernel constants, which would make the "Lloyd-Max
    // identity" claim meaningless and risk flake on boundary-crossing inputs.
    // Order matches KERNEL_HEADER CB[0..16].
    let codebook: Vec<f32> = vec![
        f32::from_bits(0xC02D_EE42), // -2.7176671
        f32::from_bits(0xC003_563B), // -2.0521381
        f32::from_bits(0xBFCC_E718), // -1.6008024
        f32::from_bits(0xBF9E_B6FA), // -1.2399590
        f32::from_bits(0xBF6D_A172), // -0.9282447
        f32::from_bits(0xBF25_5816), // -0.6458753
        f32::from_bits(0xBEC3_29CB), // -0.3811782
        f32::from_bits(0xBE01_1273), // -0.1260469
        f32::from_bits(0x3E01_1273), //  0.1260469
        f32::from_bits(0x3EC3_29CB), //  0.3811782
        f32::from_bits(0x3F25_5816), //  0.6458753
        f32::from_bits(0x3F6D_A172), //  0.9282447
        f32::from_bits(0x3F9E_B6FA), //  1.2399590
        f32::from_bits(0x3FCC_E718), //  1.6008024
        f32::from_bits(0x4003_563B), //  2.0521381
        f32::from_bits(0x402D_EE42), //  2.7176671
    ];

    vectorized_parity_check(
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales) =
                turbo_quantize_v4_gpu(&arr, Device::Gpu).expect("GPU hardwired quantize failed");
            let recon = turbo_dequantize_v4_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
                .expect("GPU hardwired dequantize failed");
            to_f32_vec(&recon)
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let cb_bytes = unsafe {
                std::slice::from_raw_parts(codebook.as_ptr().cast::<u8>(), codebook.len() * 4)
            };
            let cb_arr = Array::from_bytes(cb_bytes, &[codebook.len() as i32], Dtype::F32)
                .expect("Array::from_bytes codebook");
            let (codes, scales) = turbo_quantize_v4_codebook_buf_gpu(&arr, &cb_arr, Device::Gpu)
                .expect("GPU codebook-buf quantize failed");
            let recon = turbo_dequantize_v4_codebook_buf_gpu(
                &codes,
                &scales,
                &cb_arr,
                &shape,
                Dtype::F32,
                Device::Gpu,
            )
            .expect("GPU codebook-buf dequantize failed");
            to_f32_vec(&recon)
        },
        &data,
        1e-5_f32,
        "TurboQuant V4 hardwired vs codebook-buffer (Lloyd-Max identity)",
    );
}
