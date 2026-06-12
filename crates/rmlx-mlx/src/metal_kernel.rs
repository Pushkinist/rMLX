//! RAII wrapper around `mlx_fast_metal_kernel` — custom Metal/MSL dispatch.
//!
//! # What this is
//!
//! `MetalKernel` wraps the `mlx-c 0.6` `mlx_fast_metal_kernel` API, which lets
//! Rust code compile and dispatch MSL (Metal Shading Language) kernels on
//! Apple Silicon's GPU without leaving the MLX compute graph.
//!
//! The wrapper is **not** thread-safe (MLX Metal context is single-process, see
//! CLAUDE.md "Single MLX process per Mac" rule). Callers must hold the rMLX
//! process-level GPU claim before invoking.
//!
//! # MSL source conventions
//!
//! - `source` is the **body** of the kernel function. MLX wraps it with the
//!   appropriate Metal function signature and buffer declarations automatically.
//! - Input buffers arrive as `device const T* <name>` (auto-typed by MLX from
//!   the array dtype).
//! - Output buffers arrive as `device T* <name>`.
//! - Thread position built-ins: `thread_position_in_grid` (uint3),
//!   `threadgroup_position_in_grid` (uint3), `thread_position_in_threadgroup` (uint3).
//! - `header` is MSL code inserted *before* the kernel body (use it for
//!   `constant` arrays, helper functions).

// unsafe_code: mlx-rs FFI bridge — calls mlx_fast_metal_kernel C API via unsafe blocks
#![allow(unsafe_code)]

use std::ffi::CString;

use rmlx_core::error::{Error, Result};

use crate::{install_error_handler, sys, Array, Device, Dtype};

// ---------------------------------------------------------------------------
// MetalKernel — compiled MSL kernel handle
// ---------------------------------------------------------------------------

/// Compiled Metal/MSL kernel. RAII: frees the underlying `mlx_fast_metal_kernel`
/// on drop.
///
/// Register once per process; apply many times.
#[allow(missing_debug_implementations)]
pub struct MetalKernel {
    handle: sys::mlx_fast_metal_kernel,
}

// SAFETY: mlx_fast_metal_kernel is a void* handle. The kernel object itself
// is immutable after `new` (no mutation through &self). MLX's Metal device
// context is process-global; we document the single-process requirement.
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        // NOTE: tracing::trace! removed here — same class as the Array::drop
        // trace (rmlx-mlx/src/lib.rs:281-293). MetalKernel objects are dropped
        // when a model is unloaded (one per custom kernel, not per-token), so
        // cost here is negligible in practice. Removed for consistency: any
        // per-handle trace in rmlx-mlx fires under the default rmlx=trace
        // filter and pays JSON formatting on the emitter thread.
        // SAFETY: handle is valid (created in `new`, not aliased).
        unsafe { sys::mlx_fast_metal_kernel_free(self.handle) };
    }
}

impl MetalKernel {
    /// Register a Metal kernel from MSL source.
    ///
    /// # Arguments
    ///
    /// - `name` — kernel function identifier (also the Metal function name).
    /// - `header` — MSL code inserted before the kernel body (use for
    ///   `constant` declarations, helper functions).
    /// - `source` — body statements of the kernel function (no function
    ///   signature needed; MLX generates it from `input_names`/`output_names`).
    /// - `input_names` — names of input buffer parameters (order-sensitive).
    /// - `output_names` — names of output buffer parameters (order-sensitive).
    ///
    /// Returns `Err(Error::Mlx)` if the kernel cannot be compiled (e.g. bad MSL
    /// syntax, name conflict). On success, the kernel may be compiled lazily by
    /// MLX on first dispatch.
    pub fn new(
        name: &str,
        header: &str,
        source: &str,
        input_names: &[&str],
        output_names: &[&str],
    ) -> Result<Self> {
        install_error_handler();

        let name_c = CString::new(name)
            .map_err(|_| Error::Mlx(format!("MetalKernel::new: name '{name}' contains NUL")))?;
        let header_c = CString::new(header)
            .map_err(|_| Error::Mlx("MetalKernel::new: header contains NUL".to_owned()))?;
        let source_c = CString::new(source)
            .map_err(|_| Error::Mlx("MetalKernel::new: source contains NUL".to_owned()))?;

        // Build mlx_vector_string for input_names and output_names.
        let in_vec = strings_to_vec_string(input_names)?;
        let out_vec = strings_to_vec_string(output_names).inspect_err(|_e| {
            // SAFETY: in_vec was created just above, is not aliased, and is not
            // used after this free — release it so the output-name failure path
            // does not leak the input handle.
            unsafe { sys::mlx_vector_string_free(in_vec) };
        })?;

        // SAFETY: all CStrings are valid NUL-terminated strings; vectors were
        // created above and are valid for the duration of this call.
        let handle = unsafe {
            sys::mlx_fast_metal_kernel_new(
                name_c.as_ptr(),
                in_vec,
                out_vec,
                source_c.as_ptr(),
                header_c.as_ptr(),
                true,  // ensure_row_contiguous: safer default for custom kernels
                false, // atomic_outputs: not needed for our kernels
            )
        };

        // SAFETY: free after use (mlx_fast_metal_kernel_new copies strings).
        unsafe {
            sys::mlx_vector_string_free(in_vec);
            sys::mlx_vector_string_free(out_vec);
        };

        if handle.ctx.is_null() {
            let msg = crate::LAST_ERROR.with(std::cell::Cell::take);
            return Err(Error::Mlx(format!(
                "MetalKernel::new '{}': kernel registration returned null handle: {}",
                name,
                msg.unwrap_or_else(|| "<no mlx error>".to_owned())
            )));
        }

        tracing::debug!(name, "MetalKernel registered");
        Ok(MetalKernel { handle })
    }

    /// Dispatch this kernel with the given invocation config.
    ///
    /// Returns the output `Array`s in the order declared by `output_names`.
    ///
    /// # Errors
    ///
    /// Returns `Err(Error::Mlx)` if the kernel dispatch fails (e.g. mismatched
    /// argument count, OOB grid, GPU OOM).
    #[allow(
        clippy::needless_pass_by_value,
        reason = "MetalKernelInvoke is consumed: invoke.config is passed to mlx_fast_metal_kernel_apply (FFI ownership) and invoke.inputs are moved for array ref collection"
    )]
    pub fn apply(&self, invoke: MetalKernelInvoke, device: Device) -> Result<Vec<Array>> {
        install_error_handler();

        // Build input vector_array.
        let in_inners: Vec<sys::mlx_array> = invoke.inputs.iter().map(|a| a.inner).collect();
        let in_vec = unsafe { sys::mlx_vector_array_new_data(in_inners.as_ptr(), in_inners.len()) };

        // Prepare output vector_array (starts empty; mlx_fast_metal_kernel_apply
        // fills it with the declared outputs).
        let mut out_vec = unsafe { sys::mlx_vector_array_new() };

        let status = unsafe {
            crate::with_stream(device, |stream| {
                sys::mlx_fast_metal_kernel_apply(
                    &raw mut out_vec,
                    self.handle,
                    in_vec,
                    invoke.config,
                    stream,
                )
            })
        };

        // Free input vector (we own the handle; arrays are ref-counted by MLX).
        unsafe { sys::mlx_vector_array_free(in_vec) };

        // Check error *before* extracting outputs to avoid leaking on error.
        // SAFETY: called on the same thread immediately after the C call.
        unsafe { crate::check_status(status, "MetalKernel::apply") }?;

        // Extract output arrays.
        let n_out = unsafe { sys::mlx_vector_array_size(out_vec) };
        let mut outputs = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let mut arr = unsafe { sys::mlx_array_new() };
            let rc = unsafe { sys::mlx_vector_array_get(&raw mut arr, out_vec, i) };
            if rc != 0 {
                unsafe { sys::mlx_vector_array_free(out_vec) };
                return Err(Error::Mlx(format!(
                    "MetalKernel::apply: failed to extract output array at index {i}"
                )));
            }
            outputs.push(Array { inner: arr });
        }

        unsafe { sys::mlx_vector_array_free(out_vec) };

        Ok(outputs)
    }
}

// ---------------------------------------------------------------------------
// MetalKernelInvoke — builder for one kernel dispatch
// ---------------------------------------------------------------------------

/// Builder for a single Metal kernel dispatch.
///
/// Call `add_input`, `add_output_shape`, `set_grid`, `set_thread_group` to
/// configure the dispatch, then pass to [`MetalKernel::apply`].
///
/// The builder takes ownership of the `mlx_fast_metal_kernel_config` handle.
#[allow(missing_debug_implementations)]
pub struct MetalKernelInvoke {
    /// mlx-c config handle. Created in `new`; consumed by `apply`.
    config: sys::mlx_fast_metal_kernel_config,
    /// Input arrays in declaration order.
    inputs: Vec<Array>,
}

impl Drop for MetalKernelInvoke {
    fn drop(&mut self) {
        // SAFETY: config was created in `new` and is not aliased.
        unsafe { sys::mlx_fast_metal_kernel_config_free(self.config) };
    }
}

impl MetalKernelInvoke {
    /// Create a new, empty invocation builder.
    pub fn new() -> Self {
        install_error_handler();
        // SAFETY: no preconditions; always returns a valid config.
        let config = unsafe { sys::mlx_fast_metal_kernel_config_new() };
        MetalKernelInvoke {
            config,
            inputs: Vec::new(),
        }
    }

    /// Append an input array.
    ///
    /// Inputs must be added in the same order as `input_names` passed to
    /// [`MetalKernel::new`].
    pub fn add_input(&mut self, arr: &Array) -> Result<()> {
        let cloned = arr.try_clone()?;
        self.inputs.push(cloned);
        Ok(())
    }

    /// Declare one output shape+dtype.
    ///
    /// Outputs must be declared in the same order as `output_names` passed to
    /// [`MetalKernel::new`].
    pub fn add_output_shape(&mut self, shape: &[i32], dtype: Dtype) -> Result<()> {
        let rc = unsafe {
            sys::mlx_fast_metal_kernel_config_add_output_arg(
                self.config,
                shape.as_ptr(),
                shape.len(),
                dtype.to_sys(),
            )
        };
        if rc != 0 {
            return Err(Error::Mlx(
                "MetalKernelInvoke::add_output_shape: mlx returned non-zero".to_owned(),
            ));
        }
        Ok(())
    }

    /// Zero-initialise all output buffers before the kernel runs.
    ///
    /// Required for kernels that use `atomic_fetch_or_explicit` (or any other
    /// read-modify-write atomic) to accumulate bit-patterns into the outputs.
    /// Without this, MLX may reuse a Metal buffer from its pool whose previous
    /// contents are non-zero, causing those bits to corrupt the result.
    ///
    /// Wraps `mlx_fast_metal_kernel_config_set_init_value(config, value)`.
    pub fn set_init_value(&mut self, value: f32) -> Result<()> {
        let rc = unsafe { sys::mlx_fast_metal_kernel_config_set_init_value(self.config, value) };
        if rc != 0 {
            return Err(Error::Mlx(
                "MetalKernelInvoke::set_init_value: mlx returned non-zero".to_owned(),
            ));
        }
        Ok(())
    }

    /// Set the Metal dispatch grid (total threads, 3-D).
    pub fn set_grid(&mut self, g1: i32, g2: i32, g3: i32) -> Result<()> {
        let rc = unsafe { sys::mlx_fast_metal_kernel_config_set_grid(self.config, g1, g2, g3) };
        if rc != 0 {
            return Err(Error::Mlx(
                "MetalKernelInvoke::set_grid: mlx returned non-zero".to_owned(),
            ));
        }
        Ok(())
    }

    /// Set the Metal threadgroup size (threads per group, 3-D).
    pub fn set_thread_group(&mut self, t1: i32, t2: i32, t3: i32) -> Result<()> {
        let rc =
            unsafe { sys::mlx_fast_metal_kernel_config_set_thread_group(self.config, t1, t2, t3) };
        if rc != 0 {
            return Err(Error::Mlx(
                "MetalKernelInvoke::set_thread_group: mlx returned non-zero".to_owned(),
            ));
        }
        Ok(())
    }

    /// Add a template argument of `int` type.
    pub fn set_template_int(&mut self, name: &str, value: i32) -> Result<()> {
        let name_c = CString::new(name).map_err(|_| {
            Error::Mlx(format!(
                "MetalKernelInvoke::set_template_int: name '{name}' contains NUL"
            ))
        })?;
        let rc = unsafe {
            sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                self.config,
                name_c.as_ptr(),
                value,
            )
        };
        if rc != 0 {
            return Err(Error::Mlx(
                "MetalKernelInvoke::set_template_int: mlx returned non-zero".to_owned(),
            ));
        }
        Ok(())
    }

    /// Add a template argument of dtype type.
    pub fn set_template_dtype(&mut self, name: &str, dtype: Dtype) -> Result<()> {
        let name_c = CString::new(name).map_err(|_| {
            Error::Mlx(format!(
                "MetalKernelInvoke::set_template_dtype: name '{name}' contains NUL"
            ))
        })?;
        let rc = unsafe {
            sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                self.config,
                name_c.as_ptr(),
                dtype.to_sys(),
            )
        };
        if rc != 0 {
            return Err(Error::Mlx(
                "MetalKernelInvoke::set_template_dtype: mlx returned non-zero".to_owned(),
            ));
        }
        Ok(())
    }
}

impl Default for MetalKernelInvoke {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Build a `mlx_vector_string` from a `&[&str]`.
///
/// The caller must free the returned handle with `mlx_vector_string_free`.
fn strings_to_vec_string(names: &[&str]) -> Result<sys::mlx_vector_string> {
    // SAFETY: mlx_vector_string_new returns a valid (empty) handle.
    let vec = unsafe { sys::mlx_vector_string_new() };
    for &s in names {
        let cs = CString::new(s).map_err(|_| {
            // SAFETY: vec was created above and is not aliased; free it before
            // returning so the NUL-byte path does not leak the handle (mirrors
            // the append_value-failure free below).
            unsafe { sys::mlx_vector_string_free(vec) };
            Error::Mlx(format!(
                "strings_to_vec_string: string '{s}' contains a NUL byte"
            ))
        })?;
        let rc = unsafe { sys::mlx_vector_string_append_value(vec, cs.as_ptr()) };
        if rc != 0 {
            unsafe { sys::mlx_vector_string_free(vec) };
            return Err(Error::Mlx(format!(
                "strings_to_vec_string: append_value failed for '{s}'"
            )));
        }
    }
    Ok(vec)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "metal_kernel_tests.rs"]
mod tests;
