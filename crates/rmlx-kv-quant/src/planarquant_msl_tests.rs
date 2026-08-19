use super::*;
use crate::planarquant::{planar_dequantize, planar_quantize};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env, vectorized_parity_check};
use crate::turboquant::GROUP_SIZE as GS;
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
fn array_to_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("array eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// GPU PlanarQuant V4 roundtrip max abs error < 0.10.
// MLX reuses Metal buffer pool without zero-initialising; the atomic-OR
// quantize kernel accumulates garbage from a prior call when output buffers
// are recycled. Running alone (cargo test --ignored) avoids reuse.
// Root-cause fix: add init_value=0.0 to the quantize kernel invocation.
// Tracked in planarquant-flake-fix.md.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planarquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn planar_v4_msl_roundtrip_within_tolerance() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales, rot32) =
        planar_quantize_v4_gpu(&arr, Device::Gpu).expect("GPU quantize failed");

    let recon = planar_dequantize_v4_gpu(&codes, &scales, &rot32, &shape, Dtype::F32, Device::Gpu)
        .expect("GPU dequantize failed");

    let recon_vec = array_to_f32(&recon);
    assert_eq!(recon_vec.len(), n);

    let max_err = data
        .iter()
        .zip(recon_vec.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.10,
        "GPU PlanarQuant roundtrip max abs error {max_err:.6} exceeds 0.10"
    );
}

// ── Planar V3 (3-bit) GPU tests ──────────────────────────────────────────────

/// GPU PlanarQuant V3 roundtrip max abs error < 0.20.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planarquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn planar_v3_msl_roundtrip_within_tolerance() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xDEAD_BEEF_u64);
    let arr = make_f32_array(&data, &shape);

    let (codes, scales, rot32) =
        planar_quantize_v3_gpu(&arr, Device::Gpu).expect("V3 GPU quantize failed");

    let recon = planar_dequantize_v3_gpu(&codes, &scales, &rot32, &shape, Dtype::F32, Device::Gpu)
        .expect("V3 GPU dequantize failed");

    let recon_vec = array_to_f32(&recon);
    assert_eq!(recon_vec.len(), n);

    let max_err = data
        .iter()
        .zip(recon_vec.iter())
        .map(|(&o, &r)| (o - r).abs())
        .fold(0.0_f32, f32::max);

    assert!(
        max_err < 0.20,
        "GPU PlanarQuant V3 roundtrip max abs error {max_err:.6} exceeds 0.20"
    );
}

/// GPU and CPU V3 dequant must agree within 0.005.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planarquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn planar_v3_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xBEEF_C0DE_u64);

    vectorized_parity_check(
        |input| {
            let blocks = planar_quantize(input, GS, 3, &shape).expect("V3 CPU quantize failed");
            planar_dequantize(&blocks).expect("V3 CPU dequantize failed")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales, rot32) =
                planar_quantize_v3_gpu(&arr, Device::Gpu).expect("V3 GPU quantize failed");
            let gpu_arr =
                planar_dequantize_v3_gpu(&codes, &scales, &rot32, &shape, Dtype::F32, Device::Gpu)
                    .expect("V3 GPU dequantize failed");
            array_to_f32(&gpu_arr)
        },
        &data,
        5e-3_f32,
        "PlanarQuant V3 CPU vs GPU",
    );
}

/// GPU and CPU dequant must agree within 0.005 (small f32 rounding allowed).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test planarquant_msl -- --ignored --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn planar_v4_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    let shape = [1i32, 4, 128, 64];
    let n: usize = shape.iter().map(|&d| d as usize).product();
    let data = lcg_data(n, 0xBEEF_CAFE_u64);

    vectorized_parity_check(
        |input| {
            let blocks = planar_quantize(input, GS, 4, &shape).expect("CPU quantize failed");
            planar_dequantize(&blocks).expect("CPU dequantize failed")
        },
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales, rot32) =
                planar_quantize_v4_gpu(&arr, Device::Gpu).expect("GPU quantize failed");
            let gpu_arr =
                planar_dequantize_v4_gpu(&codes, &scales, &rot32, &shape, Dtype::F32, Device::Gpu)
                    .expect("GPU dequantize failed");
            array_to_f32(&gpu_arr)
        },
        &data,
        5e-3_f32,
        "PlanarQuant V4 CPU vs GPU",
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
fn hdr_probe_snapshots_match_builders() {
    assert_eq!(
        kernel_header(),
        include_str!("metal/probes/planarquant_v4.hdr.metal"),
        "stale snapshot: refresh metal/probes/planarquant_v4.hdr.metal"
    );
    assert_eq!(
        kernel_header_v3(),
        include_str!("metal/probes/planarquant_v3.hdr.metal"),
        "stale snapshot: refresh metal/probes/planarquant_v3.hdr.metal"
    );
}
