//! Parity tests for the fused TurboSym4 K-side QK kernel.

use super::*;
use crate::test_utils::{lcg_data, skip_if_no_gpu_env};
use crate::turboquant_msl::{turbo_dequantize_v4_gpu, turbo_quantize_v4_gpu};
use rmlx_mlx::{Array, Device, Dtype};

// ── Test helpers ─────────────────────────────────────────────────────────────

#[allow(clippy::expect_used, reason = "test helper: invariants documented")]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
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

// ── Dispatch-counter sanity (always runs, no GPU) ────────────────────────────

#[test]
fn turbo_k4_fused_qk_dispatch_count_starts_at_zero() {
    // Counter is process-global; verify it is readable without panicking.
    let _ = turbo_k4_fused_qk_dispatch_count();
}

#[test]
fn turbo_k4_fused_qk_rejects_unsupported_head_dim() {
    // head_dim=64 is not supported (kernel uses fixed shared-mem of 256
    // floats for 128/256 only).  Dispatcher rejects before any GPU work.
    let dummy_q = make_f32_array(&[0.0_f32], &[1]);
    let dummy_codes = Array::from_bytes(&[0u8, 0, 0, 0], &[1], Dtype::U32).expect("codes");
    let dummy_scales = make_f32_array(&[0.0_f32], &[1]);

    let err = turbo_k4_fused_qk_sdpa(
        &dummy_q,
        &dummy_codes,
        &dummy_scales,
        None,
        None,
        None,
        1,
        1,
        1,
        64, // head_dim
        1,
        1.0,
        Device::Cpu,
    )
    .expect_err("turbo_k4_fused_qk_sdpa should reject head_dim=64");
    let msg = err.to_string();
    assert!(
        msg.contains("head_dim=64") || msg.contains("not supported"),
        "expected head_dim gate error, got: {msg}"
    );
}

// ── GPU parity tests ─────────────────────────────────────────────────────────

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turbo_k4_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn turbo_k4_fused_qk_single_tile_head_dim_128() {
    // Single-tile config:
    //   B=1, n_q_heads=8, kv_h=2, head_dim=128, kv_seq=64.
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
    let k_data = lcg_data(k_n, 0x7C40_5ED1_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0x7C40_5ED2_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    // Quant + dequant K via turbo4 GPU codec; the dequantized K is the
    // reference the fused kernel must match.
    let (codes, scales) = turbo_quantize_v4_gpu(&k_arr, Device::Gpu).expect("turbo4 quantize");
    let k_dequant = turbo_dequantize_v4_gpu(&codes, &scales, &k_shape, Dtype::F32, Device::Gpu)
        .expect("turbo4 dequant");
    let k_dequant_vec = array_to_f32(&k_dequant);

    let fused = turbo_k4_fused_qk_sdpa(
        &q_arr,
        &codes,
        &scales,
        None,
        None,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("turbo_k4_fused_qk_sdpa");

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
        "turbo_k4_fused_qk single-tile head_dim=128: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turbo_k4_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn turbo_k4_fused_qk_multi_tile_head_dim_128_kv_seq_256() {
    // Multi-tile config:
    //   B=1, n_q_heads=8, kv_h=2, head_dim=128, kv_seq=256 (4 tiles of 64).
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
    let k_data = lcg_data(k_n, 0x7C40_4321_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0x7C40_8765_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales) = turbo_quantize_v4_gpu(&k_arr, Device::Gpu).expect("turbo4 quantize");
    let k_dequant = turbo_dequantize_v4_gpu(&codes, &scales, &k_shape, Dtype::F32, Device::Gpu)
        .expect("turbo4 dequant");
    let k_dequant_vec = array_to_f32(&k_dequant);

    let fused = turbo_k4_fused_qk_sdpa(
        &q_arr,
        &codes,
        &scales,
        None,
        None,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("turbo_k4_fused_qk_sdpa");

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
        "turbo_k4_fused_qk multi-tile (kv_seq=256) head_dim=128: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turbo_k4_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn turbo_k4_fused_qk_head_dim_256() {
    // head_dim=256 path (Gemma4-26B class): two turbo4 groups per head_dim.
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
    let k_data = lcg_data(k_n, 0x2C60_0001_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0x2C60_0002_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales) = turbo_quantize_v4_gpu(&k_arr, Device::Gpu).expect("turbo4 quantize");
    let k_dequant = turbo_dequantize_v4_gpu(&codes, &scales, &k_shape, Dtype::F32, Device::Gpu)
        .expect("turbo4 dequant");
    let k_dequant_vec = array_to_f32(&k_dequant);

    let fused = turbo_k4_fused_qk_sdpa(
        &q_arr,
        &codes,
        &scales,
        None,
        None,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("turbo_k4_fused_qk_sdpa");

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
        "turbo_k4_fused_qk head_dim=256: max abs error {max_err:.6} > 1e-4"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turbo_k4_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn turbo_k4_fused_qk_with_additive_mask() {
    // Smoke: in-kernel additive mask is summed into the pre-softmax score.
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
    let k_data = lcg_data(k_n, 0xDEAD_C401_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, n_q_heads, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0xDEAD_C402_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    // Causal-style mask: 0 on positions s_kv < kv_seq/2, -1e4 on positions above.
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

    let (codes, scales) = turbo_quantize_v4_gpu(&k_arr, Device::Gpu).expect("turbo4 quantize");
    let k_dequant = turbo_dequantize_v4_gpu(&codes, &scales, &k_shape, Dtype::F32, Device::Gpu)
        .expect("turbo4 dequant");
    let k_dequant_vec = array_to_f32(&k_dequant);

    let fused = turbo_k4_fused_qk_sdpa(
        &q_arr,
        &codes,
        &scales,
        None,
        None,
        Some(&mask_arr),
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("turbo_k4_fused_qk_sdpa with mask");

    let mut ref_scores = ref_qk_scores(
        &q_data,
        &k_dequant_vec,
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
    // Mask values are large (-1e4); use 1e-3 to leave headroom for f32
    // epsilon × |mask| at the score-add site.
    assert!(
        max_err < 1e-3_f32,
        "turbo_k4_fused_qk with mask: max abs error {max_err:.6} > 1e-3"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test turbo_k4_fused_qk -- --ignored --test-threads=1"]
#[allow(clippy::expect_used, reason = "test invariants documented")]
fn turbo_k4_fused_qk_dispatch_count_increments() {
    if skip_if_no_gpu_env() {
        return;
    }
    let before = turbo_k4_fused_qk_dispatch_count();

    let b: i32 = 1;
    let kv_h: i32 = 1;
    let heads_per_kv: i32 = 1;
    let kv_seq: i32 = 32;
    let head_dim: i32 = 128;
    let scale: f32 = 1.0 / (head_dim as f32).sqrt();

    let k_shape = [b, kv_h, kv_seq, head_dim];
    let k_n: usize = k_shape.iter().map(|&d| d as usize).product();
    let k_data = lcg_data(k_n, 0x7C40_0001_u64);
    let k_arr = make_f32_array(&k_data, &k_shape);

    let q_shape = [b, kv_h * heads_per_kv, 1, head_dim];
    let q_n: usize = q_shape.iter().map(|&d| d as usize).product();
    let q_data = lcg_data(q_n, 0x7C40_0002_u64);
    let q_arr = make_f32_array(&q_data, &q_shape);

    let (codes, scales) = turbo_quantize_v4_gpu(&k_arr, Device::Gpu).expect("turbo4 quantize");
    let _ = turbo_k4_fused_qk_sdpa(
        &q_arr,
        &codes,
        &scales,
        None,
        None,
        None,
        b,
        kv_h,
        kv_seq,
        head_dim,
        heads_per_kv,
        scale,
        Device::Gpu,
    )
    .expect("turbo_k4_fused_qk_sdpa");

    let after = turbo_k4_fused_qk_dispatch_count();
    assert!(
        after > before,
        "dispatch counter must increment: before={before} after={after}"
    );
}
