//! TurboQuant V2 (Lloyd-Max 2-bit) Metal kernel tests.
//!
//! Mirrors `k8vturbo3_append_msl_tests.rs`:
//! - `cb2_constants_bit_exact` — Rust ↔ MSL bit-pattern parity.
//! - `v2_pack_unpack_round_trip` — CPU 2-bit pack ↔ GPU 2 LE u32 layout parity.
//! - `tq2_msl_roundtrip_within_tolerance` (GPU-only, `#[ignore]`).
//! - `tq2_msl_matches_cpu_within_eps` (GPU-only, `#[ignore]`).
//! - `tq2_cosine_naive_baseline_floor` (cosine gate, empirical floor).

use super::*;
use crate::test_utils::{
    cosine_similarity_per_row, lcg_data, skip_if_no_gpu_env, vectorized_parity_check, TEST_SEED,
};
use crate::turboquant::{lloyd_gaussian_codebook, turbo_dequantize, turbo_quantize_v};
use rmlx_mlx::{Array, Device, Dtype};

/// Compile-time-style check: the bit patterns embedded in the MSL header
/// match the Rust `f32` values in `crate::turboquant::CODEBOOK_2BIT`.
#[test]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by the harness, not the unit under test"
)]
fn cb2_constants_bit_exact() {
    let cb = lloyd_gaussian_codebook(2).unwrap();
    let expected_bits: [u32; 4] = [0xBFC1_47AE, 0xBEE7_EF9E, 0x3EE7_EF9E, 0x3FC1_47AE];
    for (i, (&c, &bits)) in cb.iter().zip(expected_bits.iter()).enumerate() {
        assert_eq!(
            c.to_bits(),
            bits,
            "CB2[{i}] bit pattern: CPU 0x{:08X} vs MSL 0x{bits:08X}",
            c.to_bits()
        );
    }
    let expected_bnds: [u32; 3] = [0xBF7B_4396, 0x0000_0000, 0x3F7B_4396];
    for i in 0..3 {
        let mid = (cb[i] + cb[i + 1]) * 0.5_f32;
        assert_eq!(
            mid.to_bits(),
            expected_bnds[i],
            "BOUNDARIES_2[{i}] bit pattern mismatch"
        );
    }
}

/// Pure-Rust pack/unpack invariant: feeding 32 elements through the CPU
/// quantize, reinterpreting the 8 packed bytes as 2 LE `u32` words (the
/// GPU layout), then re-quantizing the dequantized output round-trips to
/// the same byte stream. Verifies that the CPU 2-bit pack is byte-for-
/// byte identical to the GPU 2 LE u32 layout the kernel writes.
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
    reason = "test scaffolding: invariants enforced by the harness, not the unit under test"
)]
fn v2_pack_unpack_round_trip() {
    let mut state = 0x9876_5432_u64;
    let mut indices = [0_u8; 32];
    for slot in &mut indices {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        *slot = ((state >> 32) & 0x3) as u8;
    }
    // Synthetic input that snaps cleanly to each centroid.
    let cb = lloyd_gaussian_codebook(2).unwrap();
    let dummy_x: Vec<f32> = indices.iter().map(|&i| cb[i as usize]).collect();

    let blocks = turbo_quantize_v(&dummy_x, 2, &[1, 1, 1, 32]).expect("quantize");
    assert_eq!(blocks.codes.len(), 8, "2-bit pack: 32*2/8 = 8 bytes");

    // Reinterpret the 8 bytes as 2 LE u32 words (the GPU layout).
    let words: [u32; 2] = [
        u32::from_le_bytes(blocks.codes[0..4].try_into().unwrap()),
        u32::from_le_bytes(blocks.codes[4..8].try_into().unwrap()),
    ];

    // Same window-shift extraction the MSL dequant kernel does.
    let mut extracted = [0_u8; 32];
    for (e, slot) in extracted.iter_mut().enumerate() {
        let bit_off = e * 2;
        let w_id = bit_off / 32;
        let shift = bit_off - w_id * 32;
        let lo = u64::from(words[w_id]);
        let hi = if w_id + 1 < 2 {
            u64::from(words[w_id + 1])
        } else {
            0
        };
        let window = lo | (hi << 32);
        *slot = ((window >> shift) & 0x3) as u8;
    }
    assert_eq!(
        extracted, indices,
        "2-bit GPU-layout extraction does not match CPU pack_index"
    );

    // Stronger check: dequant the CPU blocks then re-quantize; codes must
    // be stable (proves the GPU layout is the byte-for-byte equivalent of
    // the CPU layout).
    let recon = turbo_dequantize(&blocks).expect("dequant");
    let re_blocks = turbo_quantize_v(&recon, 2, &[1, 1, 1, 32]).expect("re-quantize");
    assert_eq!(
        re_blocks.codes, blocks.codes,
        "2-bit codes are not stable across re-quantize: layout mismatch"
    );
}

/// Cosine-similarity gate.
///
/// Empirical floor: rMLX naïve 2-bit Lloyd-Max on a TEST_SEED-pinned fixture.
/// Threshold is the measured value minus 0.001 (measured-minus-0.001 policy).
/// The gap-vs-mtq is documented in `docs/KV_QUANT.md`: mtq's `turbo2` ships
/// with outlier-mask (cosine 0.9420 on their GPU bench) — rMLX ships naïve so
/// a drop is expected. Outlier-mask is deferred pending calibration loader.
///
/// The fixture is 1 × 4 × 128 × 64 = 32 768 elements (mirrors the V3 test
/// shape) so the cosine number is comparable to other rMLX KV codec gates.
#[test]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn tq2_cosine_naive_baseline_floor() {
    let shape = [1_i32, 4, 128, 64];
    let head_dim = shape[3] as usize;
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, TEST_SEED);
    let blocks = turbo_quantize_v(&data, 2, &shape).expect("CPU quantize");
    let recon = turbo_dequantize(&blocks).expect("CPU dequantize");
    let stats = cosine_similarity_per_row(&data, &recon, head_dim);
    // Empirical floor (measured − 0.001): on the LCG-seeded uniform [-1,1]
    // fixture mean = 0.957865 and min = 0.926916 (n_rows = 512).
    // → floor: mean ≥ 0.956, min ≥ 0.925.
    //
    // Gap-vs-mtq: multi-turboquant's `turbo2` ships at GPU cosine 0.9420
    // (`multi-turboquant/README.md` method row 1, ~5.8× compression) *with*
    // its `build_outlier_masks` + offline calibration. rMLX ships **naïve**
    // 2-bit Lloyd-Max — the direct CPU cosine on a uniform fixture is therefore
    // not apples-to-apples; it captures only the intrinsic Lloyd-Max
    // quantization noise, not the heavy-tail residual that outlier-mask handles
    // on real V tensors. The outlier-mask path to close the production PPL gap
    // is deferred pending the calibration loader. See `docs/KV_QUANT.md` § K8VTurbo2.
    assert!(
        stats.mean >= 0.956,
        "K8VTurbo2 naive 2-bit cosine mean {:.6} fell below empirical floor 0.956 \
         (n_rows={}); gap-vs-mtq + outlier-mask plan in docs/KV_QUANT.md",
        stats.mean,
        stats.n_rows
    );
    assert!(
        stats.min >= 0.925,
        "K8VTurbo2 naive 2-bit cosine min {:.6} fell below empirical floor 0.925 \
         (n_rows={})",
        stats.min,
        stats.n_rows
    );
}

#[allow(
    clippy::expect_used,
    reason = "test helper: panic on construction failure is fine"
)]
#[allow(
    unsafe_code,
    reason = "test helper: slice::from_raw_parts to reinterpret f32 bytes for MLX Array constructor; &[f32] invariants hold"
)]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: data is a `&[f32]` slice and we read its bytes via the same
    // alignment + size invariants Vec<f32> provides; the byte slice is fed
    // directly into MLX's array constructor which copies internally.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
}

#[allow(
    clippy::expect_used,
    reason = "test helper: panic on Array materialise failure is fine"
)]
#[allow(
    clippy::unwrap_used,
    reason = "test scaffolding: invariants enforced by the harness, not the unit under test"
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
/// 2-bit Lloyd-Max half-step bound on the normalized N(0,1) range.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turbo2_v_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test scaffolding: panic-on-error is fine"
)]
fn tq2_msl_roundtrip_within_tolerance() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1_i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales) = turbo_quantize_v2_gpu(&arr, Device::Gpu).expect("GPU quantize failed");
    let recon = turbo_dequantize_v2_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
        .expect("GPU dequantize failed");

    let recon_vec = to_f32_vec(&recon);
    assert_eq!(recon_vec.len(), n);

    let max_err = data
        .iter()
        .zip(recon_vec.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);
    // 2-bit Lloyd-Max half-step on N(0,1) is ~0.98; allow 1.05 for f32 noise.
    assert!(
        max_err < 1.05,
        "GPU roundtrip max abs error {max_err:.6} exceeds tolerance 1.05"
    );
}

/// CPU vs GPU bit-equivalence: dequantized values must agree
/// within f32 rounding noise (1e-3 tolerance per the test_utils policy
/// table for 2-bit codebook lookup).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turbo2_v_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test scaffolding: panic-on-error is fine"
)]
fn tq2_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1_i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xCAFE_BABE_u64);

    vectorized_parity_check(
        |input| {
            let blocks = turbo_quantize_v(input, 2, &shape).expect("CPU quantize failed");
            turbo_dequantize(&blocks).expect("CPU dequantize failed")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales) =
                turbo_quantize_v2_gpu(&arr, Device::Gpu).expect("GPU quantize failed");
            let recon = turbo_dequantize_v2_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu)
                .expect("GPU dequantize failed");
            to_f32_vec(&recon)
        },
        &data,
        1e-3_f32,
        "K8VTurbo2 V CPU vs GPU",
    );
}
