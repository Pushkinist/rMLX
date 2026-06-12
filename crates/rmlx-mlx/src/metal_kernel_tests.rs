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

/// Kernel registration with bad source should not panic.
///
/// MLX may defer compilation to first dispatch, so the error may surface
/// in `apply` rather than `new`. Both outcomes are acceptable; what must
/// NOT happen is a panic or UB.
#[test]
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
