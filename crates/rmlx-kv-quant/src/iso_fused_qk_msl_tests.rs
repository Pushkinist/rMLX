//! GPU parity tests for the fused IsoQuant K-side QK kernel (BITS in {3, 4}).

use super::*;
use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_mlx::{Array, Device, Dtype};

// Test helpers

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("make_f32_array")
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn make_u32_array(data: &[u32], shape: &[i32]) -> Array {
    let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::U32).expect("make_u32_array")
}

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
#[allow(
    clippy::unwrap_used,
    reason = "test helper: chunks_exact(4) guarantees length"
)]
fn array_to_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("array eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Encode K with iso codec on CPU; return GPU-resident
/// `(codes_arr, scales_arr, norms_arr, dequant_vec)`.
#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn iso_encode_for_test(
    k_flat: &[f32],
    head_dim: usize,
    bits: u8,
) -> (Array, Array, Array, Vec<f32>) {
    let (codes, scales, quaternions, norms) =
        iso_encode_fast(k_flat, head_dim, 4, bits).expect("iso_encode_fast");

    let codes_arr = make_u32_array(&codes, &[codes.len() as i32]);
    let scales_arr = make_f32_array(&scales, &[scales.len() as i32]);
    let norms_arr = make_f32_array(&norms, &[norms.len() as i32]);

    let dequant = iso_decode_fast(&codes, &scales, &quaternions, &norms, head_dim, 4, bits)
        .expect("iso_decode_fast");

    (codes_arr, scales_arr, norms_arr, dequant)
}

fn ref_qk_scores(
    q: &[f32],
    k: &[f32],
    b: usize,
    n_q_heads: usize,
    kv_h: usize,
    kv_seq: usize,
    head_dim: usize,
    scale: f32,
) -> Vec<f32> {
    let heads_per_kv = n_q_heads / kv_h;
    let mut out = vec![0.0_f32; b * n_q_heads * kv_seq];
    for bi in 0..b {
        for hq in 0..n_q_heads {
            let kv_h_idx = hq / heads_per_kv;
            let q_base = (bi * n_q_heads + hq) * head_dim;
            for s in 0..kv_seq {
                let k_base = ((bi * kv_h + kv_h_idx) * kv_seq + s) * head_dim;
                let mut acc = 0.0_f32;
                for d in 0..head_dim {
                    acc += q[q_base + d] * k[k_base + d];
                }
                out[(bi * n_q_heads + hq) * kv_seq + s] = acc * scale;
            }
        }
    }
    out
}

// Dispatch-counter sanity (no GPU)

#[test]
fn iso3_fused_qk_dispatch_count_starts_at_zero() {
    let _ = iso3_fused_qk_dispatch_count();
}

#[test]
fn iso_kernel_source_smoke() {
    // Smoke check that both BITS=3 and BITS=4 header + source builders run
    // without panicking and embed the correct codebook table size.
    let h3 = build_iso_fused_qk_header(3);
    let h4 = build_iso_fused_qk_header(4);
    assert!(h3.contains("ISO_CB[8]"));
    assert!(h4.contains("ISO_CB[16]"));
    assert!(!build_iso_fused_qk_source(3).is_empty());
    assert!(!build_iso_fused_qk_source(4).is_empty());
}

#[test]
fn iso4_fused_qk_dispatch_count_starts_at_zero() {
    let _ = iso4_fused_qk_dispatch_count();
}

#[test]
fn iso_fused_qk_rejects_unsupported_head_dim_bits3() {
    let dummy_q = make_f32_array(&[0.0_f32], &[1]);
    let dummy_codes = make_u32_array(&[0_u32], &[1]);
    let dummy_scales = make_f32_array(&[0.0_f32], &[1]);
    let dummy_norms = make_f32_array(&[0.0_f32], &[1]);

    let err = iso_fused_qk_sdpa::<3>(
        &dummy_q,
        &dummy_codes,
        &dummy_scales,
        &dummy_norms,
        None,
        1,
        1,
        1,
        64,
        1,
        1.0,
        Device::Cpu,
    )
    .expect_err("iso_fused_qk_sdpa<3> should reject head_dim=64");
    let msg = err.to_string();
    assert!(
        msg.contains("head_dim=64") || msg.contains("not supported"),
        "expected head_dim gate error, got: {msg}"
    );
}

#[test]
fn iso_fused_qk_rejects_unsupported_head_dim_bits4() {
    let dummy_q = make_f32_array(&[0.0_f32], &[1]);
    let dummy_codes = make_u32_array(&[0_u32], &[1]);
    let dummy_scales = make_f32_array(&[0.0_f32], &[1]);
    let dummy_norms = make_f32_array(&[0.0_f32], &[1]);

    let err = iso_fused_qk_sdpa::<4>(
        &dummy_q,
        &dummy_codes,
        &dummy_scales,
        &dummy_norms,
        None,
        1,
        1,
        1,
        64,
        1,
        1.0,
        Device::Cpu,
    )
    .expect_err("iso_fused_qk_sdpa<4> should reject head_dim=64");
    let msg = err.to_string();
    assert!(
        msg.contains("head_dim=64") || msg.contains("not supported"),
        "expected head_dim gate error, got: {msg}"
    );
}

// GPU parity tests, BITS = 3

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits3_single_tile_head_dim_128() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads = kv_h * heads_per_kv;
    let kv_seq: i32 = 64;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xA101_B201_u64);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xA101_B202_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, norms, k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 3);

    let fused = iso_fused_qk_sdpa::<3>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<3>");

    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
        scale,
    );
    let fused_vec = array_to_f32(&fused);
    assert_eq!(fused_vec.len(), ref_scores.len());
    let max_err = fused_vec
        .iter()
        .zip(ref_scores.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4_f32,
        "iso_fused_qk bits=3 single-tile head_dim=128: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits3_multi_tile_head_dim_128_kv_seq_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads = kv_h * heads_per_kv;
    let kv_seq: i32 = 256;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xA103_B201_u64);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xA103_B202_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, norms, k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 3);

    let fused = iso_fused_qk_sdpa::<3>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<3>");

    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
        scale,
    );
    let fused_vec = array_to_f32(&fused);
    let max_err = fused_vec
        .iter()
        .zip(ref_scores.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4_f32,
        "iso_fused_qk bits=3 multi-tile (kv_seq=256) head_dim=128: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits3_head_dim_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 2;
    let n_q_heads = kv_h * heads_per_kv;
    let kv_seq: i32 = 32;
    let head_dim: i32 = 256;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xA105_B201_u64);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xA105_B202_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, norms, k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 3);

    let fused = iso_fused_qk_sdpa::<3>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<3>");

    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
        scale,
    );
    let fused_vec = array_to_f32(&fused);
    let max_err = fused_vec
        .iter()
        .zip(ref_scores.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4_f32,
        "iso_fused_qk bits=3 head_dim=256: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits3_with_additive_mask() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads = kv_h * heads_per_kv;
    let kv_seq: i32 = 64;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xDEAD_3301_u64);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xDEAD_3302_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let mask_shape = [b, n_q_heads, 1, kv_seq];
    let mask_n: usize = mask_shape.iter().map(|&d| d as usize).product();
    let mut mask_data = vec![0.0_f32; mask_n];
    let half = (kv_seq / 2) as usize;
    for bh in 0..(b as usize * n_q_heads as usize) {
        for s in 0..(kv_seq as usize) {
            if s >= half {
                mask_data[bh * (kv_seq as usize) + s] = -1.0e4_f32;
            }
        }
    }
    let mask_arr = make_f32_array(&mask_data, &mask_shape);

    let (codes, scales, norms, k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 3);

    let fused = iso_fused_qk_sdpa::<3>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        Some(&mask_arr),
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<3> with mask");

    let mut ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
        scale,
    );
    for (idx, m) in mask_data.iter().enumerate() {
        ref_scores[idx] += m;
    }

    let fused_vec = array_to_f32(&fused);
    let max_err = fused_vec
        .iter()
        .zip(ref_scores.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-3_f32,
        "iso_fused_qk bits=3 with mask: max abs error {max_err:.6} > 1e-3"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits3_dispatch_count_increments() {
    if skip_if_no_gpu_env() {
        return;
    }
    let before = iso3_fused_qk_dispatch_count();

    let b: i32 = 1;
    let kv_h: i32 = 1;
    let heads_per_kv: i32 = 1;
    let kv_seq: i32 = 32;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_n: usize = (b * kv_h * kv_seq * head_dim) as usize;
    let k_data = lcg_data(k_n, 0xA107_0001_u64);

    let q_n: usize = (b * kv_h * heads_per_kv * head_dim) as usize;
    let q_data = lcg_data(q_n, 0xA107_0002_u64);
    let q_arr = make_f32_array(&q_data, &[b, kv_h * heads_per_kv, 1, head_dim]);

    let (codes, scales, norms, _k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 3);

    let _ = iso_fused_qk_sdpa::<3>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<3>");

    let after = iso3_fused_qk_dispatch_count();
    assert!(
        after > before,
        "iso3 dispatch counter must increment: before={before} after={after}"
    );
}

// GPU parity tests, BITS = 4

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits4_single_tile_head_dim_128() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads = kv_h * heads_per_kv;
    let kv_seq: i32 = 64;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xB201_C101_u64);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xB201_C102_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, norms, k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 4);

    let fused = iso_fused_qk_sdpa::<4>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<4>");

    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
        scale,
    );
    let fused_vec = array_to_f32(&fused);
    assert_eq!(fused_vec.len(), ref_scores.len());
    let max_err = fused_vec
        .iter()
        .zip(ref_scores.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4_f32,
        "iso_fused_qk bits=4 single-tile head_dim=128: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits4_multi_tile_head_dim_128_kv_seq_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads = kv_h * heads_per_kv;
    let kv_seq: i32 = 256;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xB203_C101_u64);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xB203_C102_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, norms, k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 4);

    let fused = iso_fused_qk_sdpa::<4>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<4>");

    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
        scale,
    );
    let fused_vec = array_to_f32(&fused);
    let max_err = fused_vec
        .iter()
        .zip(ref_scores.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4_f32,
        "iso_fused_qk bits=4 multi-tile (kv_seq=256) head_dim=128: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits4_head_dim_256() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 2;
    let n_q_heads = kv_h * heads_per_kv;
    let kv_seq: i32 = 32;
    let head_dim: i32 = 256;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xB205_C101_u64);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xB205_C102_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, norms, k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 4);

    let fused = iso_fused_qk_sdpa::<4>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<4>");

    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
        scale,
    );
    let fused_vec = array_to_f32(&fused);
    let max_err = fused_vec
        .iter()
        .zip(ref_scores.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-4_f32,
        "iso_fused_qk bits=4 head_dim=256: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits4_with_additive_mask() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 2;
    let heads_per_kv: i32 = 4;
    let n_q_heads = kv_h * heads_per_kv;
    let kv_seq: i32 = 64;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xDEAD_4401_u64);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xDEAD_4402_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let mask_shape = [b, n_q_heads, 1, kv_seq];
    let mask_n: usize = mask_shape.iter().map(|&d| d as usize).product();
    let mut mask_data = vec![0.0_f32; mask_n];
    let half = (kv_seq / 2) as usize;
    for bh in 0..(b as usize * n_q_heads as usize) {
        for s in 0..(kv_seq as usize) {
            if s >= half {
                mask_data[bh * (kv_seq as usize) + s] = -1.0e4_f32;
            }
        }
    }
    let mask_arr = make_f32_array(&mask_data, &mask_shape);

    let (codes, scales, norms, k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 4);

    let fused = iso_fused_qk_sdpa::<4>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        Some(&mask_arr),
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<4> with mask");

    let mut ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant,
        b as usize,
        n_q_heads as usize,
        kv_h as usize,
        kv_seq as usize,
        head_dim as usize,
        scale,
    );
    for (idx, m) in mask_data.iter().enumerate() {
        ref_scores[idx] += m;
    }

    let fused_vec = array_to_f32(&fused);
    let max_err = fused_vec
        .iter()
        .zip(ref_scores.iter())
        .map(|(&f, &r)| (f - r).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        max_err < 1e-3_f32,
        "iso_fused_qk bits=4 with mask: max abs error {max_err:.6} > 1e-3"
    );
}

#[test]
#[ignore = "GPU Metal context: cargo test iso_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn iso_fused_qk_bits4_dispatch_count_increments() {
    if skip_if_no_gpu_env() {
        return;
    }
    let before = iso4_fused_qk_dispatch_count();

    let b: i32 = 1;
    let kv_h: i32 = 1;
    let heads_per_kv: i32 = 1;
    let kv_seq: i32 = 32;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_n: usize = (b * kv_h * kv_seq * head_dim) as usize;
    let k_data = lcg_data(k_n, 0xB207_0001_u64);

    let q_n: usize = (b * kv_h * heads_per_kv * head_dim) as usize;
    let q_data = lcg_data(q_n, 0xB207_0002_u64);
    let q_arr = make_f32_array(&q_data, &[b, kv_h * heads_per_kv, 1, head_dim]);

    let (codes, scales, norms, _k_dequant) = iso_encode_for_test(&k_data, head_dim as usize, 4);

    let _ = iso_fused_qk_sdpa::<4>(
        &q_arr,
        &codes,
        &scales,
        &norms,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("iso_fused_qk_sdpa<4>");

    let after = iso4_fused_qk_dispatch_count();
    assert!(
        after > before,
        "iso4 dispatch counter must increment: before={before} after={after}"
    );
}
