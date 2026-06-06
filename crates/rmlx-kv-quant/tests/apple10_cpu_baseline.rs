// CPU baseline cosine at head_dim = 256.
//
// The on-GPU smoke + stress tests (tests/apple10_head_dim_256.rs) land
// cosine ≈ 0.997 vs the bf16 SDPA reference at head_dim = 256. The K8V4
// fused-QK parity baseline records cosine ≈ 0.999998 at head_dim = 128.
// That is ~3 orders of magnitude noisier — the question is whether the residual
// is the **codec floor** (q8 on K + turbo-4 on V, scalar quantization noise that
// any encode→dequantize round-trip would inherit at this head_dim) or a
// **kernel issue** in the TurboFlash MSL dispatch at head_dim = 256.
//
// This file isolates the codec contribution from the kernel contribution by
// running the round-trip on CPU with NO MSL kernel involvement:
//
//   1. Encode K via `q8_quantize` (group_size 128, the K8 codec the GPU path
//      uses for K). Encode V via `turbo_quantize_v` at bits = 4 (group_size 32,
//      the V4 codec the GPU path uses).
//   2. Decode each back to f32. Compute per-row cosine vs the f32 source for
//      both K and V, at head_dim = 128 (fused-QK anchor) and head_dim = 256
//      (Apple10 hazard cell). Same LCG seeds as the GPU smoke test.
//   3. Print a labelled summary so the revalidation report can cite the real
//      numbers.
//
// Decision tree:
//
//   * If head_dim = 256 codec round-trip cosine is **at or below** the GPU
//     0.997 number → the residual is the codec floor. The GPU kernel is doing
//     its job; the bf16-SDPA-vs-K8V4-SDPA gap is dominated by quantization
//     noise that gets amplified through softmax + V projection at 256 dims.
//     No kernel bug — tighten the GPU assertion to parity vs CPU baseline,
//     document the codec floor.
//   * If head_dim = 256 codec round-trip cosine is **≥ 0.999** (parity with
//     head_dim = 128) → the residual is a kernel-side numerics issue at
//     head_dim = 256. Revert the Auto-flip on Apple10 to OFF until fixed.
//
// This test runs on CPU only — no Metal context required, no `#[ignore]`.
#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::print_stderr,
    clippy::unusual_byte_groupings,
    clippy::indexing_slicing,
    clippy::cast_precision_loss,
    missing_docs
)]
//! CPU codec baseline at head_dim ∈ {128, 256}.

use rmlx_kv_quant::q8::{q8_dequantize, q8_quantize};
use rmlx_kv_quant::turboquant::{turbo_dequantize, turbo_quantize_v};

// ── helpers ───────────────────────────────────────────────────────────────────

/// LCG pseudo-random data — same generator as the GPU smoke test so seeds round-trip.
fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let frac = (state >> 32) as u32 as f32 / u32::MAX as f32;
            frac * 2.0 - 1.0
        })
        .collect()
}

/// Per-row cosine — same helper as the GPU smoke test so the numbers are
/// directly comparable across reports.
fn cosine_per_row(a: &[f32], b: &[f32], row_len: usize) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "cosine: len mismatch");
    assert!(row_len > 0);
    let n_rows = a.len() / row_len;
    let mut mn = f32::INFINITY;
    let mut sum = 0.0_f64;
    for r in 0..n_rows {
        let ra = &a[r * row_len..(r + 1) * row_len];
        let rb = &b[r * row_len..(r + 1) * row_len];
        let dot: f64 = ra
            .iter()
            .zip(rb.iter())
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        let na: f64 = ra.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let nb: f64 = rb.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
        let c = if na == 0.0 || nb == 0.0 {
            1.0
        } else {
            (dot / (na * nb)) as f32
        };
        if c < mn {
            mn = c;
        }
        sum += f64::from(c);
    }
    let mean = (sum / n_rows as f64) as f32;
    (mean, mn)
}

// Same cache shape constants as the GPU smoke test.
const B: i32 = 1;
const KV_H: i32 = 2;
const SMOKE_PREFILL: i32 = 64;

/// Single-codec round-trip cosine for K (q8) and V (turbo-4) at a given head_dim.
/// Returns `(k_mean, k_min, v_mean, v_min)`.
fn codec_roundtrip_cosines(head_dim: i32) -> (f32, f32, f32, f32) {
    let shape = [B, KV_H, SMOKE_PREFILL, head_dim];
    let n: usize = shape.iter().map(|&d| d as usize).product();

    // Same seeds as the GPU smoke test prefill so the codec sees the same data.
    let k_src = lcg_data(n, 0xC1A2D3E4_F5A6_0001);
    let v_src = lcg_data(n, 0xC1A2D3E4_F5A6_0002);

    // K — q8 (group_size = 128). The GPU K8 path uses affine q8_0 with this
    // exact group size; the scalar q8 codec here is the matching reference.
    let (k_codes, k_scales) = q8_quantize(&k_src);
    let k_decoded = q8_dequantize(&k_codes, &k_scales);
    let (k_mean, k_min) = cosine_per_row(&k_src, &k_decoded, head_dim as usize);

    // V — turbo-4 (group_size = 32). Matches the V4 codec the GPU TurboFlash
    // path uses for the V tensor.
    let v_blocks = turbo_quantize_v(&v_src, 4, &shape).expect("turbo_quantize_v");
    let v_decoded = turbo_dequantize(&v_blocks).expect("turbo_dequantize");
    let (v_mean, v_min) = cosine_per_row(&v_src, &v_decoded, head_dim as usize);

    (k_mean, k_min, v_mean, v_min)
}

/// CPU baseline at head_dim = 256 — the Apple10 hazard cell. This is the
/// load-bearing test for the decision tree.
#[test]
fn cpu_baseline_cosine_at_head_dim_256() {
    let (k_mean, k_min, v_mean, v_min) = codec_roundtrip_cosines(256);
    eprintln!(
        "CPU baseline head_dim=256: K q8 round-trip cosine \
         mean={k_mean:.6} min={k_min:.6}; V turbo-4 round-trip cosine \
         mean={v_mean:.6} min={v_min:.6}"
    );

    // Soft assertion — every codec round-trip must at minimum be coherent
    // (cosine > 0.9). This is just a sanity guard; the verdict comes from
    // comparing these numbers to the head_dim=128 baseline below and to the
    // GPU SDPA 0.997.
    assert!(
        k_min > 0.9,
        "K q8 round-trip cosine min {k_min} <= 0.9 at head_dim=256 — codec smashed"
    );
    assert!(
        v_min > 0.9,
        "V turbo-4 round-trip cosine min {v_min} <= 0.9 at head_dim=256 — codec smashed"
    );
}

/// CPU baseline at head_dim = 128 — fused-QK anchor cell. Provides the direct
/// comparison point for the hazard decision.
#[test]
fn cpu_baseline_cosine_at_head_dim_128() {
    let (k_mean, k_min, v_mean, v_min) = codec_roundtrip_cosines(128);
    eprintln!(
        "CPU baseline head_dim=128: K q8 round-trip cosine \
         mean={k_mean:.6} min={k_min:.6}; V turbo-4 round-trip cosine \
         mean={v_mean:.6} min={v_min:.6}"
    );
    assert!(
        k_min > 0.9,
        "K q8 round-trip cosine min {k_min} <= 0.9 at head_dim=128 — codec smashed"
    );
    assert!(
        v_min > 0.9,
        "V turbo-4 round-trip cosine min {v_min} <= 0.9 at head_dim=128 — codec smashed"
    );
}

/// Side-by-side delta — head_dim=256 vs head_dim=128 codec floor. If the
/// deltas are small (<1e-3), head_dim=256 has no codec-side penalty beyond
/// head_dim=128, and any GPU-side residual must come from kernel numerics.
/// If they're large, the codec already produces the residual at head_dim=256
/// and the GPU is faithful.
#[test]
fn cpu_baseline_delta_head_dim_128_vs_256() {
    let (k128_mean, k128_min, v128_mean, v128_min) = codec_roundtrip_cosines(128);
    let (k256_mean, k256_min, v256_mean, v256_min) = codec_roundtrip_cosines(256);

    let k_mean_delta = (k128_mean - k256_mean).abs();
    let k_min_delta = (k128_min - k256_min).abs();
    let v_mean_delta = (v128_mean - v256_mean).abs();
    let v_min_delta = (v128_min - v256_min).abs();

    eprintln!(
        "CPU baseline delta hd128 vs hd256: \
         K q8 mean delta={k_mean_delta:.6} min delta={k_min_delta:.6}; \
         V turbo-4 mean delta={v_mean_delta:.6} min delta={v_min_delta:.6}"
    );
    eprintln!(
        "CPU baseline hd128: k_mean={k128_mean:.6} k_min={k128_min:.6} \
         v_mean={v128_mean:.6} v_min={v128_min:.6}"
    );
    eprintln!(
        "CPU baseline hd256: k_mean={k256_mean:.6} k_min={k256_min:.6} \
         v_mean={v256_mean:.6} v_min={v256_min:.6}"
    );
}
