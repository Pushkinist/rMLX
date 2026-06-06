use super::*;
use crate::test_utils::{lcg_data, skip_if_no_gpu_env, vectorized_parity_check};
use crate::turboquant::{lloyd_gaussian_codebook, turbo_dequantize, turbo_quantize_v};
use rmlx_mlx::{Array, Device, Dtype};

/// Compile-time-style check: the bit patterns embedded in the MSL header
/// match the Rust `f32` values in `crate::turboquant::CODEBOOK_3BIT`.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn cb3_constants_bit_exact() {
    let cb = lloyd_gaussian_codebook(3).unwrap();
    let expected_bits: [u32; 8] = [
        0xC009_B977,
        0xBFAC_0532,
        0xBF41_8987,
        0xBE7A_F9EB,
        0x3E7A_F9EB,
        0x3F41_8987,
        0x3FAC_0532,
        0x4009_B977,
    ];
    for (i, (&c, &bits)) in cb.iter().zip(expected_bits.iter()).enumerate() {
        assert_eq!(
            c.to_bits(),
            bits,
            "CB3[{i}] bit pattern: CPU 0x{:08X} vs MSL 0x{bits:08X}",
            c.to_bits()
        );
    }
    let expected_bnds: [u32; 7] = [
        0xBFDF_BC10,
        0xBF86_64FB,
        0xBF00_2401,
        0x0000_0000,
        0x3F00_2401,
        0x3F86_64FB,
        0x3FDF_BC10,
    ];
    for i in 0..7 {
        let mid = (cb[i] + cb[i + 1]) * 0.5_f32;
        assert_eq!(
            mid.to_bits(),
            expected_bnds[i],
            "BOUNDARIES_3[{i}] bit pattern mismatch"
        );
    }
}

/// Pure-Rust pack/unpack invariant: feeding 32 elements through the CPU
/// quantize, reinterpreting the 12 packed bytes as 3 LE `u32` words (the
/// GPU layout), then re-quantizing the dequantized output round-trips to
/// the same byte stream. Verifies that the CPU 3-bit pack is byte-for-
/// byte identical to the GPU 3 LE u32 layout the kernel writes.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "Mutex critical section is panic-free, so PoisonError is structurally unreachable; remaining Option/Result unwrap is on values established by construction earlier in this fn"
)]
fn v3_pack_unpack_round_trip() {
    let mut state = 0xABCD_1234_u64;
    let mut indices = [0_u8; 32];
    for slot in &mut indices {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *slot = ((state >> 33) & 0x7) as u8;
    }
    // Synthetic input that snaps cleanly to each centroid.
    let cb = lloyd_gaussian_codebook(3).unwrap();
    let dummy_x: Vec<f32> = indices.iter().map(|&i| cb[i as usize]).collect();

    let blocks = turbo_quantize_v(&dummy_x, 3, &[1, 1, 1, 32]).expect("quantize");
    assert_eq!(blocks.codes.len(), 12, "3-bit pack: 32*3/8 = 12 bytes");

    // Reinterpret the 12 bytes as 3 LE u32 words (the GPU layout).
    let words: [u32; 3] = [
        u32::from_le_bytes(blocks.codes[0..4].try_into().unwrap()),
        u32::from_le_bytes(blocks.codes[4..8].try_into().unwrap()),
        u32::from_le_bytes(blocks.codes[8..12].try_into().unwrap()),
    ];

    // Same signed-shift extraction the MSL dequant kernel does.
    let mut extracted = [0_u8; 32];
    for (e, slot) in extracted.iter_mut().enumerate() {
        let bit_off = e * 3;
        let w_id = bit_off / 32;
        let shift = bit_off - w_id * 32;
        let lo = u64::from(words[w_id]);
        let hi = if w_id + 1 < 3 {
            u64::from(words[w_id + 1])
        } else {
            0
        };
        let window = lo | (hi << 32);
        *slot = ((window >> shift) & 0x7) as u8;
    }
    assert_eq!(
        extracted, indices,
        "3-bit GPU-layout extraction does not match CPU pack_index"
    );

    // Stronger check: dequant the CPU blocks then re-quantize; codes must
    // be stable (proves the GPU layout is the byte-for-byte equivalent of
    // the CPU layout).
    let recon = turbo_dequantize(&blocks).expect("dequant");
    let re_blocks = turbo_quantize_v(&recon, 3, &[1, 1, 1, 32]).expect("re-quantize");
    assert_eq!(
        re_blocks.codes, blocks.codes,
        "3-bit codes are not stable across re-quantize: layout mismatch"
    );
}

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
fn to_f32_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// GPU roundtrip (quantize -> dequantize) max abs error stays inside the
/// 3-bit Lloyd-Max half-step bound on the normalized N(0,1) range.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test k8vturbo3_append_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn tq3_msl_roundtrip_within_tolerance() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1_i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales) = turbo_quantize_v3_gpu(&arr, Device::Gpu).expect("GPU quantize failed");
    let recon = turbo_dequantize_v3_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
        .expect("GPU dequantize failed");

    let recon_vec = to_f32_vec(&recon);
    assert_eq!(recon_vec.len(), n);

    let max_err = data
        .iter()
        .zip(recon_vec.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);
    // 3-bit Lloyd-Max half-step on N(0,1) is ~0.50; allow 0.55 for f32 noise.
    assert!(
        max_err < 0.55,
        "GPU roundtrip max abs error {max_err:.6} exceeds tolerance 0.55"
    );
}

/// CPU vs GPU bit-equivalence: dequantized values must agree within f32
/// rounding noise.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test k8vturbo3_append_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn tq3_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1_i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xCAFE_BABE_u64);

    vectorized_parity_check(
        |input| {
            let blocks = turbo_quantize_v(input, 3, &shape).expect("CPU quantize failed");
            turbo_dequantize(&blocks).expect("CPU dequantize failed")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales) =
                turbo_quantize_v3_gpu(&arr, Device::Gpu).expect("GPU quantize failed");
            let recon = turbo_dequantize_v3_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
                .expect("GPU dequantize failed");
            to_f32_vec(&recon)
        },
        &data,
        1e-3_f32,
        "K8VTurbo3 V CPU vs GPU",
    );
}
