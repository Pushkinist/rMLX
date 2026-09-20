//! iso4 MSL kernel parity tests.
//!
//! All GPU tests are `#[ignore]`-gated. They run only when explicitly invoked:
//!
//! ```text
//! cargo test -p rmlx-kv-quant --lib -- --ignored isoquant_msl_v4 --test-threads=1
//! ```
//!
//! `RMLX_SKIP_GPU=1` causes GPU tests to exit silently even with `--include-ignored`.

use super::*;
use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::storage::{IsoBlocks, QuantIsoK4, QuantIsoV4};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env, vectorized_parity_check};

/// Build a test `Array` from a f32 slice and shape.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    unsafe_code,
    reason = "Metal FFI test helper: reinterpret f32 slice as bytes for Array::from_bytes; \
              slice lifetime is tied to `data`, no aliasing — safe by construction"
)]
fn make_f32_array(data: &[f32], shape: &[i32]) -> Array {
    // SAFETY: `f32` has no padding, alignment is 4; reinterpreting as `u8` is
    // well-defined. The resulting byte slice borrows from `data` for the duration
    // of this call. `Array::from_bytes` copies the bytes immediately.
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, shape, Dtype::F32).expect("make_f32_array")
}

/// Materialise an `Array` to `Vec<f32>` via the MLX graph executor.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[allow(
    clippy::unwrap_used,
    reason = "chunks_exact(4) + try_into are infallible given the f32 element size invariant"
)]
fn array_to_f32_vec(a: &Array) -> Vec<f32> {
    a.eval().expect("array materialise");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// iso4 GPU matches CPU within 5e-3.
///
/// CPU path: `iso_encode_fast(..., bits=4)` → `iso_decode_fast(..., bits=4)`.
/// MSL path: `iso_quantize_v4_gpu` → `iso_dequantize_v4_gpu`.
/// Tolerance 5e-3 per codebook-codec tolerance policy (same as iso3).
//
// `#[ignore]`-gated per project policy: any GPU Metal test acquires the
// single-MLX claim and would race the cargo-test process pool; the canonical
// run-mode is in isolation via the docstring command above.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored isoquant_msl_v4 --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn iso_v4_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    // n_tokens=32, head_dim=128, group_size=4: 32 groups/token, 4 elems/group.
    let head_dim: usize = 128;
    let n_tokens: usize = 32;
    let group_size: usize = 4;
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x1504_BEEF_u64);

    let shape = [n_tokens as i32, head_dim as i32];

    vectorized_parity_check(
        // CPU path.
        |input| {
            let (codes, scales, quats, norms) =
                iso_encode_fast(input, head_dim, group_size, 4).expect("iso_encode_fast bits=4");
            iso_decode_fast(&codes, &scales, &quats, &norms, head_dim, group_size, 4)
                .expect("iso_decode_fast bits=4")
        },
        // MSL path.
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales, quats, norms) =
                iso_quantize_v4_gpu(&arr, head_dim, Device::Gpu).expect("iso_quantize_v4_gpu");
            let out = iso_dequantize_v4_gpu(
                &codes,
                &scales,
                &quats,
                &norms,
                head_dim,
                Dtype::F32,
                Device::Gpu,
            )
            .expect("iso_dequantize_v4_gpu");
            array_to_f32_vec(&out)
        },
        &data,
        5e-3_f32,
        "iso4 CPU vs MSL",
    );
}

/// `QuantIsoV4::dequant_gpu` matches `QuantIsoV4::dequant` (CPU) within codec
/// tolerance, and within the strict numeric bound the 3-bit sibling holds.
///
/// The 4-bit width had no `dequant_gpu` until the storage collapse gave it
/// one, so `iso_v4_msl_matches_cpu_within_eps` above could only compare the
/// kernels. This compares the store entries a decode step actually calls, at
/// the same two bounds as `iso_v3_dequant_gpu_matches_dequant_cpu`.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored isoquant_msl_v4 --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Result::Err from encode/dequant paths is the desired test failure mode"
)]
fn iso_v4_dequant_gpu_matches_dequant_cpu() {
    if skip_if_no_gpu_env() {
        return;
    }
    // A [B, kv_h, S, D] layout matching the iso V decode contract.
    let b: usize = 1;
    let kv_h: usize = 4;
    let s_tokens: usize = 16;
    let head_dim: usize = 128;
    let n_tokens = b * kv_h * s_tokens;
    let data = lcg_data(n_tokens * head_dim, 0x179C_BEEF_u64);

    let storage_shape: Vec<i32> = vec![b as i32, kv_h as i32, s_tokens as i32, head_dim as i32];

    // Build the storage by stuffing one CPU-encoded block.
    let (codes, scales, quaternions, norms) =
        iso_encode_fast(&data, head_dim, 4, 4).expect("iso_encode_fast bits=4");
    let mut vs = QuantIsoV4::new(storage_shape);
    vs.blocks.push(IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens,
    });

    let cpu = vs.dequant().expect("dequant cpu");
    let arr = vs.dequant_gpu(Device::Gpu).expect("dequant_gpu");
    let gpu = array_to_f32_vec(&arr);

    assert_dequant_parity(&cpu, &gpu, "V4");
}

/// `QuantIsoK4::dequant_gpu` matches `QuantIsoK4::dequant` (CPU) within codec
/// tolerance and the strict bound.
///
/// K-side mirror of the V4 test; same axis-agnostic kernel, same tolerances.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored isoquant_msl_v4 --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Result::Err from encode/dequant paths is the desired test failure mode"
)]
fn iso_k4_dequant_gpu_matches_dequant_cpu() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: usize = 1;
    let kv_h: usize = 4;
    let s_tokens: usize = 16;
    let head_dim: usize = 128;
    let n_tokens = b * kv_h * s_tokens;
    let data = lcg_data(n_tokens * head_dim, 0x179D_BEEF_u64);

    let storage_shape: Vec<i32> = vec![b as i32, kv_h as i32, s_tokens as i32, head_dim as i32];

    let (codes, scales, quaternions, norms) =
        iso_encode_fast(&data, head_dim, 4, 4).expect("iso_encode_fast bits=4");
    let mut ks = QuantIsoK4::new(storage_shape, s_tokens as i32);
    ks.blocks.push(IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens,
    });

    let cpu = ks.dequant().expect("dequant cpu");
    let arr = ks.dequant_gpu(Device::Gpu).expect("dequant_gpu");
    let gpu = array_to_f32_vec(&arr);

    assert_dequant_parity(&cpu, &gpu, "K4");
}

/// The two bounds a `dequant_gpu` parity test holds, shared by the V and K
/// cases above.
///
/// `5e-3` is the codebook tolerance. The strict `1e-6` is the real gate: the
/// CPU and GPU paths reduce through different fp32 summation orders inside the
/// MSL kernel, so a few ULPs are expected and anything larger is codec drift,
/// which would otherwise surface only as a perplexity regression.
fn assert_dequant_parity(cpu: &[f32], gpu: &[f32], axis: &str) {
    assert_eq!(cpu.len(), gpu.len(), "{axis} dequant length mismatch");
    let mut max_abs = 0.0_f32;
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        assert!(
            diff <= 5e-3_f32,
            "{axis} dequant mismatch @ idx {i}: cpu={c}, gpu={g}, diff={diff}"
        );
    }
    eprintln!("{axis} dequant_gpu max|cpu-gpu| = {max_abs:.8}");
    assert!(
        max_abs <= 1e-6_f32,
        "{axis} dequant_gpu strict bound broken: max|cpu-gpu| = {max_abs}; expected ≤ 1e-6. \
         If the new observation is intentional, update docs/KV_QUANT.md to the new bound."
    );
}

/// Probe header snapshot must equal what the builder emits.
///
/// `make check-metal-compiles` prepends this snapshot to the iso4 kernel
/// bodies. A builder that changes a constant's value, or drops one, without the
/// snapshot being refreshed leaves the probe compiling text production no
/// longer emits — the gate would keep passing while checking the wrong thing.
/// Equality here turns that drift into a hard failure.
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
#[test]
fn hdr_probe_snapshot_matches_builder() {
    assert_eq!(
        kernel_header_iso4().expect("iso4 header builder"),
        include_str!("metal/probes/isoquant_iso4.hdr.metal"),
        "stale snapshot: refresh metal/probes/isoquant_iso4.hdr.metal"
    );
}
