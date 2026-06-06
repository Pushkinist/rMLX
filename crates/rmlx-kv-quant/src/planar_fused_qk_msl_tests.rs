//! Parity tests for the fused PlanarQuant QK MSL kernel.

use super::*;
use crate::planarquant_msl::{
    planar_dequantize_v3_gpu, planar_dequantize_v4_gpu, planar_quantize_v3_gpu,
    planar_quantize_v4_gpu,
};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use rmlx_mlx::{Array, Device, Dtype};

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    // Safe `to_le_bytes` collect avoids `unsafe` slice transmute and any
    // aliasing / endianness assumptions on the host.
    let bytes: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Array::from_bytes(&bytes, shape, Dtype::F32).expect("make_f32_array")
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

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_fused_qk_v4_matches_reference() {
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
    let k_data = lcg_data(k_n, 0xCAFE_1234_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xBEEF_5678_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, rot32) = planar_quantize_v4_gpu(&k_arr, Device::Gpu).expect("v4 quantize");
    let fused = planar_fused_qk(
        &q_arr,
        &codes,
        &scales,
        &rot32,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        4,
        scale,
        Device::Gpu,
    )
    .expect("fused QK kernel");

    let k_dequant = planar_dequantize_v4_gpu(&codes, &scales, &rot32, &k_shape, Device::Gpu)
        .expect("v4 dequant");
    let k_dequant_vec = array_to_f32(&k_dequant);
    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant_vec,
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
        "planar_fused_qk v4: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_fused_qk_v3_matches_reference() {
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
    let k_data = lcg_data(k_n, 0xC0DE_4321_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xFACE_8765_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, rot32) = planar_quantize_v3_gpu(&k_arr, Device::Gpu).expect("v3 quantize");
    let fused = planar_fused_qk(
        &q_arr,
        &codes,
        &scales,
        &rot32,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        3,
        scale,
        Device::Gpu,
    )
    .expect("fused QK kernel");

    let k_dequant = planar_dequantize_v3_gpu(&codes, &scales, &rot32, &k_shape, Device::Gpu)
        .expect("v3 dequant");
    let k_dequant_vec = array_to_f32(&k_dequant);
    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant_vec,
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
        "planar_fused_qk v3: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_fused_qk_mha_heads_per_kv_one() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: i32 = 1;
    let kv_h: i32 = 4;
    let heads_per_kv: i32 = 1;
    let n_q_heads = kv_h;
    let kv_seq: i32 = 32;
    let head_dim: i32 = 64;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0xAAAA_BBBB_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xCCCC_DDDD_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, rot32) = planar_quantize_v4_gpu(&k_arr, Device::Gpu).expect("v4 quantize");
    let fused = planar_fused_qk(
        &q_arr,
        &codes,
        &scales,
        &rot32,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        4,
        scale,
        Device::Gpu,
    )
    .expect("fused QK kernel");

    let k_dequant = planar_dequantize_v4_gpu(&codes, &scales, &rot32, &k_shape, Device::Gpu)
        .expect("v4 dequant");
    let k_dequant_vec = array_to_f32(&k_dequant);
    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant_vec,
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
        "planar_fused_qk MHA: max abs error {max_err:.6} > 1e-4"
    );
}

// head_dim=256 parity test (Gemma4-26B class) + CPU-only pow-2 guard test.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planar_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn planar_fused_qk_v4_head_dim_256() {
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
    let k_data = lcg_data(k_n, 0x2560_0001_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0x2560_0002_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales, rot32) = planar_quantize_v4_gpu(&k_arr, Device::Gpu).expect("v4 quantize");
    let fused = planar_fused_qk(
        &q_arr,
        &codes,
        &scales,
        &rot32,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        4,
        scale,
        Device::Gpu,
    )
    .expect("fused QK kernel");

    let k_dequant = planar_dequantize_v4_gpu(&codes, &scales, &rot32, &k_shape, Device::Gpu)
        .expect("v4 dequant");
    let k_dequant_vec = array_to_f32(&k_dequant);
    let ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant_vec,
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
        "planar_fused_qk head_dim=256: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
fn planar_fused_qk_rejects_non_pow2_head_dim() {
    // head_dim=80 is divisible by GROUP_SIZE=32 (well, 80%32=16 → fails that
    // check first).  Use head_dim=96 (96 % 32 == 0, but 96 is not a power of
    // two) to exercise the pow-2 guard specifically.  No GPU call is made —
    // the dispatcher returns before any kernel dispatch.
    //
    // Inputs are dummy 1-element f32 arrays — shape validation comes first,
    // but since head_dim=96 passes the GROUP_SIZE check, the next guard
    // (pow-2) is what should fire.
    let dummy_q = make_f32_array(&[0.0_f32], &[1]);
    let dummy_codes = Array::from_bytes(&[0u8, 0, 0, 0], &[1], Dtype::U32).expect("codes");
    let dummy_scales = make_f32_array(&[0.0_f32], &[1]);
    let dummy_rot = Array::from_bytes(&[0u8, 0, 0, 0], &[1], Dtype::U32).expect("rot");

    let err = planar_fused_qk(
        &dummy_q,
        &dummy_codes,
        &dummy_scales,
        &dummy_rot,
        1,
        1,
        1,
        96,
        1,
        4,
        1.0,
        Device::Cpu,
    )
    .expect_err("planar_fused_qk should reject non-pow-2 head_dim");
    let msg = err.to_string();
    assert!(
        msg.contains("power of two"),
        "expected pow-2 guard error, got: {msg}"
    );
}
