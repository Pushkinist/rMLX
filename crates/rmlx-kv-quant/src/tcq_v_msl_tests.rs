//! MSL Viterbi (TCQ) 3-bit kernel parity tests.
#![allow(unsafe_code)]

use super::tcq_quantize_v3_gpu;
use crate::k8vturbo3_append_msl::turbo_dequantize_v3_gpu;
use crate::tcq::tcq_quantize_v3;
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use crate::turboquant::turbo_dequantize;
use rmlx_mlx::{Array, Device, Dtype};

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn array_to_u32_vec(a: &Array) -> Vec<u32> {
    a.eval().expect("array materialise");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn array_to_f32_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("array materialise");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Reinterpret a `TurboBlocks` `codes` byte vector as a sequence of LE u32
/// words. Plain turbo3 packs `3 * GROUP_SIZE = 96` bits per block = exactly
/// `3 u32`; CPU + MSL must emit the same byte stream.
fn cpu_codes_as_u32_words(bytes: &[u8]) -> Vec<u32> {
    assert!(
        bytes.len().is_multiple_of(4),
        "codes length must be u32-aligned"
    );
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// CPU Viterbi == MSL Viterbi (bit-identical code emission) for the canonical
/// V-side fixture shape. **The load-bearing parity test** — proves the GPU
/// kernel implements the same trellis as the CPU codec.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test tcq_v_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn tcq_cpu_msl_codes_bit_identical() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1_i32, 4, 32, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0x5A5A_BEEF_u64);

    let cpu_blocks = tcq_quantize_v3(&data, &shape).expect("CPU TCQ encode failed");
    let cpu_codes_u32 = cpu_codes_as_u32_words(&cpu_blocks.codes);

    let arr = make_f32_array(&data, &shape);
    let (gpu_codes, gpu_scales) =
        tcq_quantize_v3_gpu(&arr, Device::Gpu).expect("GPU TCQ encode failed");
    let gpu_codes_u32 = array_to_u32_vec(&gpu_codes);
    let gpu_scales_f32 = array_to_f32_vec(&gpu_scales);

    assert_eq!(
        cpu_codes_u32.len(),
        gpu_codes_u32.len(),
        "CPU codes-u32 length {} != GPU {}",
        cpu_codes_u32.len(),
        gpu_codes_u32.len()
    );
    assert_eq!(
        cpu_blocks.scales.len(),
        gpu_scales_f32.len(),
        "CPU scales length {} != GPU {}",
        cpu_blocks.scales.len(),
        gpu_scales_f32.len()
    );

    for (i, (&cpu_w, &gpu_w)) in cpu_codes_u32.iter().zip(gpu_codes_u32.iter()).enumerate() {
        assert_eq!(
            cpu_w, gpu_w,
            "TCQ CPU/MSL codes diverge at u32 word {i}: CPU=0x{cpu_w:08X} GPU=0x{gpu_w:08X}",
        );
    }
    for (i, (&cs, &gs)) in cpu_blocks
        .scales
        .iter()
        .zip(gpu_scales_f32.iter())
        .enumerate()
    {
        // Scales are computed identically (max(|x|) / max_centroid) — must be
        // bit-exact f32 (no Viterbi reachability dependency).
        assert_eq!(
            cs.to_bits(),
            gs.to_bits(),
            "TCQ CPU/MSL scales diverge at group {i}: CPU={cs} GPU={gs}"
        );
    }
}

/// GPU TCQ encode + GPU turbo3 dequant round-trip stays within the 3-bit
/// Lloyd-Max half-step error bound on N(0,1) data. Smoke for the kernel
/// dispatch + decode reuse path.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test tcq_v_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn tcq_msl_roundtrip_within_tolerance() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1_i32, 2, 16, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xC0DE_FACE_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales) = tcq_quantize_v3_gpu(&arr, Device::Gpu).expect("GPU TCQ encode failed");
    // Decode reuses plain turbo3 kernel — TCQ is encode-side only.
    let recon = turbo_dequantize_v3_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
        .expect("GPU turbo3 decode failed");
    let recon_vec = array_to_f32_vec(&recon);
    assert_eq!(recon_vec.len(), n);

    let max_err = data
        .iter()
        .zip(recon_vec.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);
    // Same threshold as the plain turbo3 GPU round-trip — TCQ assignment
    // never produces worse per-element error than nearest-centroid.
    assert!(
        max_err < 0.55,
        "GPU TCQ round-trip max abs error {max_err:.6} exceeds tolerance 0.55"
    );
}

/// CPU TCQ encode + CPU turbo3 decode round-trip — non-GPU sanity check that
/// runs in the default `cargo test` invocation (no Metal context required).
/// Validates the decoder-reuse contract.
#[test]
fn tcq_cpu_roundtrip_within_tolerance() {
    let shape = [1_i32, 2, 16, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xC0DE_FACE_u64);
    let blocks = tcq_quantize_v3(&data, &shape).expect("CPU TCQ encode failed");
    let recon = turbo_dequantize(&blocks).expect("CPU turbo3 decode failed");
    let max_err = data
        .iter()
        .zip(recon.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 0.55,
        "CPU TCQ round-trip max abs error {max_err:.6} exceeds tolerance 0.55"
    );
}
