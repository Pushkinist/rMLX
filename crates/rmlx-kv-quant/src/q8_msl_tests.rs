use super::*;
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
fn to_f32_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn lcg_data(n: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let frac = ((state >> 33) as f32) / (u32::MAX as f32);
            frac * 2.0 - 1.0
        })
        .collect()
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test q8_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn q8_msl_roundtrip_within_tolerance() {
    let shape = [1i32, 4, 128, 128];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales) = q8_quantize_gpu(&arr, Device::Gpu).expect("GPU quantize");
    let recon =
        q8_dequantize_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu).expect("GPU dequant");

    let recon_vec = to_f32_vec(&recon);
    assert_eq!(recon_vec.len(), n);

    let max_err = data
        .iter()
        .zip(recon_vec.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.01,
        "GPU q8 roundtrip max abs error {max_err:.6} exceeds 0.01"
    );
}

#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test q8_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
)]
fn q8_msl_matches_cpu_within_eps() {
    let shape = [1i32, 4, 128, 128];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xCAFE_BABE_u64);

    let mut cpu_recon = vec![0.0_f32; n];
    for g in 0..(n / 128) {
        let start = g * 128;
        let group = &data[start..start + 128];
        let abs_max = group
            .iter()
            .cloned()
            .fold(0.0_f32, |acc, v| acc.max(v.abs()));
        let scale = if abs_max > 0.0 { abs_max / 127.0 } else { 0.0 };
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for (i, &v) in group.iter().enumerate() {
            let c = (v * inv).round().clamp(-128.0, 127.0) as i32;
            cpu_recon[start + i] = scale * (c as f32);
        }
    }

    let arr = make_f32_array(&data, &shape);
    let (codes, scales) = q8_quantize_gpu(&arr, Device::Gpu).expect("GPU quantize");
    let recon =
        q8_dequantize_gpu(&codes, &scales, &shape, Dtype::F32, Device::Gpu).expect("GPU dequant");
    let gpu_recon = to_f32_vec(&recon);

    let max_diff = cpu_recon
        .iter()
        .zip(gpu_recon.iter())
        .map(|(&c, &g)| (c - g).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_diff < 1.0e-4,
        "CPU vs GPU max abs diff {max_diff:.6} exceeds 1e-4"
    );
}
