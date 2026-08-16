use super::*;
use crate::{Array, Device, Dtype};

/// Helper: make a [N] f32 array from a slice.
fn make_f32_array(data: &[f32]) -> Array {
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    Array::from_bytes(bytes, &[data.len() as i32], Dtype::F32).expect("make_f32_array")
}

/// Helper: extract f32 values from an evaluated Array.
fn array_to_f32(a: &Array) -> Vec<f32> {
    a.eval().expect("eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Trivial GPU "add_one" kernel: input [8] f32 -> output [8] f32.
///
/// Each thread adds 1.0 to its element. Grid=(8,1,1), threadgroup=(8,1,1).
/// Input is zeros; expected output is all 1.0.
///
/// Runs unconditionally — the CI/dev host has a GPU (Apple Silicon M-series).
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test metal_kernel -- --ignored --test-threads=1"]
fn metal_kernel_add_one_gpu() {
    let kernel = MetalKernel::new(
        "rmlx_add_one",
        // header: empty
        "",
        // source: kernel body.
        // `thread_position_in_grid` is a Metal built-in (uint3).
        // `inp` and `out` are `device float*` buffers provided by MLX.
        "uint gid = thread_position_in_grid.x; \
         out[gid] = inp[gid] + 1.0f;",
        &["inp"],
        &["out"],
    )
    .expect("kernel registration failed");

    let input = make_f32_array(&[0.0_f32; 8]);
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&input).expect("add_input");
    invoke
        .add_output_shape(&[8_i32], Dtype::F32)
        .expect("add_output_shape");
    invoke.set_grid(8, 1, 1).expect("set_grid");
    invoke.set_thread_group(8, 1, 1).expect("set_thread_group");

    let outputs = kernel
        .apply(invoke, Device::Gpu)
        .expect("kernel apply failed");
    assert_eq!(outputs.len(), 1, "expected 1 output");

    let result = array_to_f32(&outputs[0]);
    assert_eq!(result.len(), 8);
    for (i, &v) in result.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-6, "element {i}: expected 1.0, got {v}");
    }
}

/// A NUL byte in an output name must return Err (not panic) — a regression
/// guard for the leak-free error paths in `new` and `strings_to_vec_string`
/// (this asserts the `Err` contract those frees depend on; it cannot itself
/// observe a leak).
#[test]
fn new_rejects_nul_in_output_name() {
    let r = MetalKernel::new("k", "", "kernel void k() {}", &["inp"], &["out\0bad"]);
    assert!(r.is_err());
}

/// Helper: extract i32 values from an evaluated Array.
fn array_to_i32(a: &Array) -> Vec<i32> {
    a.eval().expect("eval");
    let bytes = a.to_bytes().expect("to_bytes");
    bytes
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

/// Which MSL language version does MLX's runtime JIT compile a custom kernel
/// body at, and does the cooperative-tensor header survive MLX's source
/// wrapping?
///
/// This is an observation, not an assertion about a desired value: the answer
/// decides whether an rMLX kernel body may ever use `mpp::tensor_ops`, and
/// nothing on our side can force it — mlx-c exposes no compile-options surface,
/// so the language version is whatever MLX passes to the runtime compiler.
///
/// Reading the four outputs:
///
/// - `out[0]` — `__METAL_VERSION__` as MLX's JIT saw it. `400` is Metal 4.0.
/// - `out[1]` — `1` if `__HAVE_TENSOR__` reached the body. That macro is only
///   defined from `-std=metal4.0`, so it is a second, independent read of the
///   same fact.
/// - `out[2]` — the `.m` field of a `constexpr matmul2d_descriptor`, so `8`
///   means the `MetalPerformancePrimitives` include actually resolved and a
///   descriptor instantiated *inside* the JIT'd body. This is the only one of
///   the three that proves the include path survives MLX's source wrapping;
///   `-1` means the guard was inactive.
/// - `out[3]` — `0`, a liveness marker: it distinguishes "the kernel ran and
///   wrote every slot" from "the buffer was never written".
///
/// The assert is on `apply`, not `new`: MLX compiles lazily on first dispatch,
/// so a compile failure surfaces there.
///
/// Recorded result: see docs/FFI.md, "MLX JIT language version".
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test metal_kernel -- --ignored --test-threads=1"]
fn rmlx_nax_probe_gpu() {
    let kernel = MetalKernel::new(
        "rmlx_nax_probe",
        include_str!("metal/nax_probe_header.metal"),
        include_str!("metal/nax_probe.metal"),
        &[],
        &["out"],
    )
    .expect("kernel registration failed");

    let mut invoke = MetalKernelInvoke::new();
    invoke
        .add_output_shape(&[4_i32], Dtype::I32)
        .expect("add_output_shape");
    invoke.set_grid(1, 1, 1).expect("set_grid");
    invoke.set_thread_group(1, 1, 1).expect("set_thread_group");

    let outputs = kernel
        .apply(invoke, Device::Gpu)
        .expect("MLX JIT refused the probe body");
    assert_eq!(outputs.len(), 1, "expected 1 output");

    let v = array_to_i32(&outputs[0]);
    assert_eq!(v.len(), 4, "expected 4 int32 slots, got {v:?}");
    println!(
        "rmlx_nax_probe: __METAL_VERSION__={} __HAVE_TENSOR__={} matmul2d_descriptor.m={} live={}",
        v[0], v[1], v[2], v[3]
    );

    // Liveness: without this a buffer MLX never wrote reads as all-zero and
    // would be indistinguishable from "compiled below 4.0".
    assert_eq!(v[3], 0, "probe did not run: {v:?}");
    assert!(v[0] >= 300, "implausible __METAL_VERSION__: {v:?}");

    // The two independent reads of the same fact must agree. They cannot
    // disagree unless the toolchain's own `__HAVE_TENSOR__` / version coupling
    // changed, which is exactly the thing worth failing on.
    let have_tensor = v[0] >= 400;
    assert_eq!(
        v[1],
        i32::from(have_tensor),
        "__METAL_VERSION__ and __HAVE_TENSOR__ disagree: {v:?}"
    );
    assert_eq!(
        v[2],
        if have_tensor { 8 } else { -1 },
        "matmul2d_descriptor did not instantiate as expected: {v:?}"
    );
}

/// Kernel registration with bad source should not panic.
///
/// MLX may defer compilation to first dispatch, so the error may surface
/// in `apply` rather than `new`. Both outcomes are acceptable; what must
/// NOT happen is a panic or UB.
#[test]
#[ignore = "GPU Metal context — run in isolation: cargo test metal_kernel -- --ignored --test-threads=1"]
fn metal_kernel_bad_source_does_not_panic() {
    let result = MetalKernel::new(
        "rmlx_bad_kernel_test",
        "",
        // Invalid MSL: undeclared identifier.
        "this_is_not_valid_msl_XXXXXXXXXX;",
        &["inp"],
        &["out"],
    );
    if let Ok(kernel) = result {
        // Deferred compile: try to apply and expect a clean error.
        let input = make_f32_array(&[1.0_f32]);
        let mut invoke = MetalKernelInvoke::new();
        invoke.add_input(&input).expect("add_input");
        let _ = invoke.add_output_shape(&[1_i32], Dtype::F32);
        let _ = invoke.set_grid(1, 1, 1);
        let _ = invoke.set_thread_group(1, 1, 1);
        // Either Ok or Err — must not panic.
        let _ = kernel.apply(invoke, Device::Gpu);
    } else {
        // Eager compile rejected bad source — correct.
    }
}
