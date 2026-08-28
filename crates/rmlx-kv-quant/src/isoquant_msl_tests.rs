//! iso3 MSL kernel tests.
//!
//! All GPU tests are `#[ignore]`-gated. They run only when explicitly invoked:
//!
//! ```text
//! cargo test -p rmlx-kv-quant --lib -- --ignored isoquant_msl --test-threads=1
//! ```
//!
//! `RMLX_SKIP_GPU=1` causes GPU tests to exit silently even with `--include-ignored`.

use super::*;
use crate::isoquant::{iso_decode_fast, iso_encode_fast};
use crate::storage::{IsoBlocks, QuantIsoK3, QuantIsoV3};
use crate::test_utils::{lcg_data, skip_if_no_gpu_env, vectorized_parity_check};

/// Build a test `Array` from a f32 slice and shape.
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Result::Err from Array::from_bytes is the desired test failure mode"
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
    reason = "test fixture: panic on Result::Err from Array::eval / to_bytes is the desired test failure mode"
)]
#[allow(
    clippy::unwrap_used,
    reason = "chunks_exact(4) + try_into are infallible given the f32 element size invariant"
)]
fn array_to_f32_vec(a: &Array) -> Vec<f32> {
    // Materialise the lazy MLX graph before reading bytes.
    a.eval().expect("array materialise");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// iso3 GPU matches CPU within 5e-3.
///
/// CPU path: `iso_encode_fast` → `iso_decode_fast`.
/// MSL path: `iso_quantize_v3_gpu` → `iso_dequantize_v3_gpu`.
/// Tolerance 5e-3 per codebook-codec tolerance policy.
// Parity test stays `#[ignore]`-gated per project policy: any GPU
// Metal test acquires the single-MLX claim and would race the cargo-test
// process pool; the canonical run-mode is in isolation via the docstring
// command above. The CPU↔GPU bit-identity check itself is unchanged.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored isoquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "structural invariant: value present by construction in calling context; .expect() message documents the invariant"
)]
fn iso_v3_msl_matches_cpu_within_eps() {
    if skip_if_no_gpu_env() {
        return;
    }
    // n_tokens=32, head_dim=128, group_size=4: 32 groups/token, 4 elems/group.
    let head_dim: usize = 128;
    let n_tokens: usize = 32;
    let group_size: usize = 4;
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x1503_BEEF_u64);

    let shape = [n_tokens as i32, head_dim as i32];

    vectorized_parity_check(
        // CPU path.
        |input| {
            let (codes, scales, quats, norms) =
                iso_encode_fast(input, head_dim, group_size, 3).expect("iso_encode_fast");
            iso_decode_fast(&codes, &scales, &quats, &norms, head_dim, group_size, 3)
                .expect("iso_decode_fast")
        },
        // MSL path.
        |input| {
            let arr = make_f32_array(input, &shape);
            let (codes, scales, quats, norms) =
                iso_quantize_v3_gpu(&arr, head_dim, Device::Gpu).expect("iso_quantize_v3_gpu");
            let out = iso_dequantize_v3_gpu(
                &codes,
                &scales,
                &quats,
                &norms,
                head_dim,
                Dtype::F32,
                Device::Gpu,
            )
            .expect("iso_dequantize_v3_gpu");
            array_to_f32_vec(&out)
        },
        &data,
        5e-3_f32,
        "iso3 CPU vs MSL",
    );
}

/// `QuantIsoV3::dequant_gpu` matches `QuantIsoV3::dequant` (CPU) within
/// codec tolerance.
///
/// Encodes a known V chunk via the CPU `iso_encode_fast`, stuffs the blocks
/// into `QuantIsoV3`, then compares the GPU dequant path
/// (`dequant_gpu` → `Array::from_bytes` → MSL dispatch → reshape) against the
/// CPU dequant path (`dequant` → `Vec<f32>`). Tolerance 5e-3 per codebook-codec
/// tolerance policy.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored isoquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Result::Err from encode/dequant paths is the desired test failure mode"
)]
fn iso_v3_dequant_gpu_matches_dequant_cpu() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Use a [B, kv_h, S, D] layout matching update_iso3 contract.
    let b: usize = 1;
    let kv_h: usize = 4;
    let s_tokens: usize = 16;
    let head_dim: usize = 128;
    let n_tokens = b * kv_h * s_tokens;
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x179A_BEEF_u64);

    let storage_shape: Vec<i32> = vec![b as i32, kv_h as i32, s_tokens as i32, head_dim as i32];

    // Build storage by stuffing one CPU-encoded block.
    let (codes, scales, quaternions, norms) =
        iso_encode_fast(&data, head_dim, 4, 3).expect("iso_encode_fast bits=3");
    let mut vs = QuantIsoV3::new(storage_shape);
    vs.blocks.push(IsoBlocks {
        codes,
        scales,
        quaternions,
        norms,
        n_tokens,
    });

    // CPU path: dequant -> Vec<f32>.
    let cpu = vs.dequant().expect("dequant cpu");

    // GPU path: dequant_gpu -> Array -> Vec<f32>.
    let arr = vs.dequant_gpu(Device::Gpu).expect("dequant_gpu");
    let gpu = array_to_f32_vec(&arr);

    assert_eq!(cpu.len(), gpu.len(), "dequant length mismatch");
    let mut max_abs = 0.0f32;
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        assert!(
            diff <= 5e-3_f32,
            "V3 dequant mismatch @ idx {i}: cpu={c}, gpu={g}, diff={diff}"
        );
    }
    eprintln!("V3 dequant_gpu max|cpu-gpu| = {max_abs:.8}");
    // Strict numeric bound. The CPU and GPU paths reduce through different
    // fp32 summation orders inside the MSL kernel; observed `max|cpu-gpu| ≈ 2.4e-7`
    // on the LCG fixture (a few ULPs at f32 codebook magnitudes). This gates
    // well below the 5e-3 codebook tolerance and catches future codec drift
    // before it surfaces as PPL regression.
    assert!(
        max_abs <= 1e-6_f32,
        "V3 dequant_gpu strict bound broken: max|cpu-gpu| = {max_abs}; expected ≤ 1e-6 \
         (observed ≈ 2.4e-7 historically). If the new observation is intentional, update \
         docs/PERF_BASELINE.md + docs/KV_QUANT.md to the new bound."
    );
}

/// `QuantIsoK3::dequant_gpu` matches `QuantIsoK3::dequant` (CPU) within
/// codec tolerance.
///
/// K-side mirror of the V3 test; same axis-agnostic kernel, same tolerance.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored isoquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Result::Err from encode/dequant paths is the desired test failure mode"
)]
fn iso_k3_dequant_gpu_matches_dequant_cpu() {
    if skip_if_no_gpu_env() {
        return;
    }
    let b: usize = 1;
    let kv_h: usize = 4;
    let s_tokens: usize = 16;
    let head_dim: usize = 128;
    let n_tokens = b * kv_h * s_tokens;
    let n = n_tokens * head_dim;
    let data = lcg_data(n, 0x179B_BEEF_u64);

    let storage_shape: Vec<i32> = vec![b as i32, kv_h as i32, s_tokens as i32, head_dim as i32];

    let (codes, scales, quaternions, norms) =
        iso_encode_fast(&data, head_dim, 4, 3).expect("iso_encode_fast bits=3");
    let mut ks = QuantIsoK3::new(storage_shape, s_tokens as i32);
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

    assert_eq!(cpu.len(), gpu.len(), "dequant length mismatch");
    let mut max_abs = 0.0f32;
    for (i, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        let diff = (c - g).abs();
        if diff > max_abs {
            max_abs = diff;
        }
        assert!(
            diff <= 5e-3_f32,
            "K3 dequant mismatch @ idx {i}: cpu={c}, gpu={g}, diff={diff}"
        );
    }
    eprintln!("K3 dequant_gpu max|cpu-gpu| = {max_abs:.8}");
    // Strict numeric bound — see V-side mirror comment for rationale.
    assert!(
        max_abs <= 1e-6_f32,
        "K3 dequant_gpu strict bound broken: max|cpu-gpu| = {max_abs}; expected ≤ 1e-6 \
         (observed ≈ 2.4e-7 historically). If the new observation is intentional, update \
         docs/PERF_BASELINE.md + docs/KV_QUANT.md to the new bound."
    );
}

/// `dequant_gpu` on an empty cache returns a zero-length Array of the declared
/// rank-4 shape (no blocks, `shape[2] == 0`).
///
/// Mirrors `dequant()` empty behaviour and ensures the shape guard does not
/// spuriously reject the canonical empty-cache case.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored isoquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Result::Err from dequant_gpu is the desired test failure mode"
)]
fn iso_v3_dequant_gpu_empty_cache_returns_zero_array() {
    if skip_if_no_gpu_env() {
        return;
    }
    let storage_shape: Vec<i32> = vec![1, 4, 0, 128];
    let vs = QuantIsoV3::new(storage_shape);
    let arr = vs.dequant_gpu(Device::Gpu).expect("dequant_gpu empty");
    assert_eq!(arr.shape(), vec![1, 4, 0, 128]);
}

/// `dequant_gpu` with declared shape `[B, kv_h, N>0, D]` but zero accumulated
/// blocks refuses, rather than zero-padding the missing tail or panicking.
///
/// **Two guards can refuse this, and which one speaks is not the contract.**
/// `synced_iso_v_blocks` runs first and rejects a blocks-vs-shape shortfall
/// with no ring to cover it; the `actual_total != declared_total` accounting in
/// the kernel-input builder is behind it and only sees inputs that already
/// agree. This asserted the second guard's wording and went red when the first
/// one was added, while the behaviour it exists to protect never moved. So it
/// asserts the behaviour: an `Err` that names this store and says it is
/// refusing, and no silently-padded array.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test -p rmlx-kv-quant -- --ignored isoquant_msl --test-threads=1"]
#[allow(
    clippy::expect_used,
    reason = "test fixture: panic on Result::Ok from dequant_gpu (we expect Err) is the desired test failure mode"
)]
fn iso_v3_dequant_gpu_shape_divergence_errors() {
    if skip_if_no_gpu_env() {
        return;
    }
    // Declare a non-empty shape but leave `blocks` empty: the shape claims
    // 1 * 4 * 16 = 64 tokens and nothing holds them, on either the CPU or the
    // (unallocated) ring.
    let storage_shape: Vec<i32> = vec![1, 4, 16, 128];
    let vs = QuantIsoV3::new(storage_shape);
    let err = vs
        .dequant_gpu(Device::Gpu)
        .expect_err("a declared tail nothing holds must error, not zero-pad");
    let msg = err.to_string();
    assert!(
        msg.contains("iso V store") && msg.contains("refusing"),
        "the refusal must name this store and say it is refusing, so an operator can tell \
         it from an unrelated failure; got: {msg}"
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
fn hdr_probe_snapshot_matches_builder() {
    assert_eq!(
        kernel_header_iso3(),
        include_str!("metal/probes/isoquant_iso3.hdr.metal"),
        "stale snapshot: refresh metal/probes/isoquant_iso3.hdr.metal"
    );
}
