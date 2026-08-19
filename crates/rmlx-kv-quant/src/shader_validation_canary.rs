//! Positive control for the shader-validation gate.
//!
//! `scripts/run_gpu_tests.sh` scans test output for Metal's invalid-access
//! report. That scan is a hand-written pattern over an undocumented,
//! version-specific message, and its *absence* is what the gate reports as
//! success — so a wording change at a toolchain bump would silently convert it
//! into a gate that runs, prints its banner, and can never fire.
//!
//! This module dispatches a kernel that stores out of bounds on purpose, so the
//! gate can confirm the detector still matches before trusting a clean scan.
//!
//! Built only under the `shader-validation-canary` feature, and the test
//! refuses to dispatch unless shader validation is actually enabled: with
//! validation on the invalid write is instrumented and (under the gate's
//! `FAIL_MODE=zerofill`) discarded, whereas dispatching it uninstrumented would
//! be a real out-of-bounds write into an MLX-owned buffer.

use rmlx_core::error::Result;
use rmlx_mlx::metal_kernel::{MetalKernel, MetalKernelInvoke};
use rmlx_mlx::{Array, Device, Dtype};
use std::sync::OnceLock;

const CANARY_SOURCE: &str = include_str!("metal/shader_validation_canary.metal");

static CANARY_KERNEL: OnceLock<Result<MetalKernel>> = OnceLock::new();

fn canary_kernel() -> Result<&'static MetalKernel> {
    CANARY_KERNEL
        .get_or_init(|| {
            MetalKernel::new(
                "rmlx_shader_validation_canary",
                "",
                CANARY_SOURCE,
                &["inp"],
                &["out"],
            )
        })
        .as_ref()
        .map_err(|e| {
            rmlx_core::error::Error::Mlx(format!("shader_validation_canary kernel init: {e}"))
        })
}

/// Whether Metal's shader validation is enabled for this process.
///
/// Read from the environment because that is the only place it can be set:
/// Metal inserts the validation layer at device creation and there is no
/// in-process way to add it later.
#[must_use]
pub fn shader_validation_enabled() -> bool {
    std::env::var("MTL_SHADER_VALIDATION").is_ok_and(|v| v != "0" && !v.is_empty())
}

/// Dispatch the deliberately out-of-bounds kernel.
///
/// # Errors
/// Any FFI failure building or running the kernel. Note that the invalid store
/// itself does **not** produce an error: the command buffer completes, its
/// `error` is nil and the process exits 0. That is the whole point — the signal
/// is a diagnostic in the output, not a status code.
// f32-out-ok: deliberately-invalid canary dispatch; the output buffer exists
// only to give the out-of-bounds store somewhere to aim and is never returned.
pub fn dispatch_out_of_bounds_store(device: Device) -> Result<()> {
    let n = 256_i32;
    let input = Array::from_f32_slice(&vec![1.0_f32; n as usize], &[n])?;

    let kernel = canary_kernel()?;
    let mut invoke = MetalKernelInvoke::new();
    invoke.add_input(&input)?;
    invoke.add_output_shape(&[n], Dtype::F32)?;
    invoke.set_grid(n, 1, 1)?;
    invoke.set_thread_group(64, 1, 1)?;

    let mut outputs = kernel.apply(invoke, device)?;
    if let Some(out) = outputs.first_mut() {
        // eval-ok: the dispatch IS the product here, not the value. Nothing
        // ever reads this output — the canary exists to make the GPU execute an
        // invalid store so the validation layer reports it — so MLX has no
        // reason to materialise the node and, left lazy, the kernel never runs.
        // Measured with validation on: with this eval the run emits one
        // "Invalid device store … custom_kernel_rmlx_shader_validation_canary";
        // without it, zero, with the validation banner present either way. A
        // canary that does not dispatch would let the gate certify its detector
        // against silence.
        out.eval()?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "shader_validation_canary_tests.rs"]
mod shader_validation_canary_tests;
