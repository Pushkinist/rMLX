//! Safe wrapper around the `mlx_compile` / `mlx_closure` C API.
//!
//! # What mlx_compile does
//!
//! `mlx_compile` takes an `mlx_closure` (a function `Vec<Array> -> Vec<Array>`),
//! traces it once to build the MLX lazy graph, compiles the resulting graph to a
//! Metal program, and returns a new `mlx_closure` that replays the compiled program
//! on subsequent calls — skipping the Rust tracing loop entirely.
//!
//! This is the Rust equivalent of Python's `@mx.compile` decorator.
//!
//! # Usage pattern
//!
//! ```ignore
//! use rmlx_mlx::compile::{Closure, compile_shapeless};
//!
//! // Build a compiled closure once (e.g. in a OnceLock):
//! let raw = Closure::from_fn(|inputs| {
//! // your ops here — Array slices in, Vec<Array> out
//! Ok(vec![/* ... */])
//! });
//! let compiled = compile_shapeless(raw)?;
//!
//! // Re-apply on every call — graph traces only on the first invocation:
//! let outputs = compiled.apply(&[&q, &k, &v])?;
//! ```
//!
//! # Shape caching
//!
//! When `shapeless=false` (plain `compile`), MLX re-traces if the input shapes
//! change. When `shapeless=true` (`compile_shapeless`), MLX re-uses the compiled
//! program regardless of shape changes — Metal dispatch grid is adjusted at
//! runtime. Use `compile_shapeless` for the GDN prefill ops where T varies.
//!
//! # Thread safety
//!
//! `Closure` is `Send + Sync`. The underlying `mlx_closure` is ref-counted by
//! mlx-c. Do not share the same `Closure` across concurrent MLX dispatches (MLX
//! is single-threaded inside a process on Apple Silicon).

// unsafe_code: mlx-rs FFI bridge — calls mlx_closure / mlx_compile C API via unsafe blocks
#![allow(unsafe_code)]

use rmlx_core::error::{Error, Result};

use crate::{check_status, install_error_handler, sys, Array};

// ---------------------------------------------------------------------------
// Closure — RAII wrapper around mlx_closure
// ---------------------------------------------------------------------------

/// Owned RAII handle around an `mlx_closure`.
///
/// Freed on `Drop` via `mlx_closure_free`.
pub struct Closure {
    pub(crate) inner: sys::mlx_closure,
}

impl std::fmt::Debug for Closure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Closure(ctx={:p})", self.inner.ctx)
    }
}

// SAFETY: mlx_closure is a ref-counted pointer (analogous to Arc<T>).
// mlx-c docs guarantee thread-safe ref-counting.
unsafe impl Send for Closure {}
unsafe impl Sync for Closure {}

impl Drop for Closure {
    fn drop(&mut self) {
        // SAFETY: inner is a valid handle created by mlx-c. Free decrements
        // ref-count; the C++ closure is destroyed when count reaches zero.
        unsafe {
            sys::mlx_closure_free(self.inner);
        }
    }
}

// Payload type for the generic Rust-closure bridge.
// Box is heap-allocated; the raw pointer is stored in the mlx_closure payload.
type BoxFn = Box<dyn Fn(Vec<Array>) -> Result<Vec<Array>> + Send + Sync + 'static>;

// The C callback invoked by mlx-c when calling a closure created via `from_fn`.
//
// SAFETY:
// - `payload` is a `*mut BoxFn` cast to `*mut c_void`, valid for the lifetime
// of the mlx_closure (the dtor frees the Box when the closure is dropped).
// - `output` is a freshly created mlx_vector_array we must populate.
// - `input` is a borrowed mlx_vector_array; we must NOT free it.
// - Panics must NOT propagate across the FFI boundary. We convert to an error
// code (non-zero return) via catch_unwind.
unsafe extern "C" fn rust_closure_callback(
    output: *mut sys::mlx_vector_array,
    input: sys::mlx_vector_array,
    payload: *mut std::os::raw::c_void,
) -> std::os::raw::c_int {
    // SAFETY: payload was constructed in from_fn as Box<BoxFn> leaked to raw ptr.
    let f = unsafe { &*(payload as *const BoxFn) };

    // Unpack input vector → Vec<Array>
    // SAFETY: mlx_vector_array_size and mlx_vector_array_get are safe to call.
    let n = unsafe { sys::mlx_vector_array_size(input) };
    let mut inputs: Vec<Array> = Vec::with_capacity(n);
    for i in 0..n {
        let mut arr = unsafe { sys::mlx_array_new() };
        let st = unsafe { sys::mlx_vector_array_get(&raw mut arr, input, i) };
        if st != 0 {
            unsafe { sys::mlx_array_free(arr) };
            return 1;
        }
        // SAFETY: mlx_vector_array_get writes a valid array handle into arr.
        // mlx-c ref-counts it; wrapping in Array takes ownership.
        inputs.push(Array { inner: arr });
    }

    // Call the Rust closure. Catch panics so we never unwind across FFI.
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(inputs)));
    let arrays = match result {
        Ok(Ok(arrs)) => arrs,
        Ok(Err(e)) => {
            tracing::error!(target: "rmlx::compile", error = %e, "callback: Rust fn returned Err");
            return 1;
        }
        Err(panic_val) => {
            let msg = panic_val
                .downcast_ref::<&str>()
                .copied()
                .unwrap_or("<non-string panic>");
            tracing::error!(target: "rmlx::compile", panic_msg = msg, "callback: Rust fn panicked");
            return 1;
        }
    };

    // Pack outputs → mlx_vector_array at *output.
    //
    // The C++ closure lambda creates `*output` as `mlx_vector_array({nullptr})` —
    // ctx is null. We CANNOT append to it (mlx_vector_array_get_ throws).
    //
    // The correct pattern (from mlx-rs trampoline): create a NEW initialized
    // vector via mlx_vector_array_new() (which sets ctx = new std::vector<>),
    // append our arrays into it, then OVERWRITE *output with the new struct.
    // The struct is plain-old-data: copying it copies the ctx pointer.
    // The C++ lambda then reads ctx from *output and frees it — no leak.
    let new_vec = unsafe { sys::mlx_vector_array_new() };
    let raw_handles: Vec<sys::mlx_array> = arrays.iter().map(|a| a.inner).collect();
    let st = unsafe {
        sys::mlx_vector_array_append_data(new_vec, raw_handles.as_ptr(), raw_handles.len())
    };
    if st != 0 {
        unsafe { sys::mlx_vector_array_free(new_vec) };
        tracing::error!(
            target: "rmlx::compile",
            status = st,
            "callback: mlx_vector_array_append_data returned non-zero"
        );
        return 1;
    }
    // Overwrite *output with the new initialized struct (copies the ctx pointer).
    // SAFETY: output is a valid pointer from the C++ lambda (&res).
    // We do NOT free new_vec — the C++ lambda will free *output (= new_vec) for us.
    // The `arrays` Vec still holds its own refs (dropped after this line),
    // and `mlx_vector_array_append_data` bumped the ref-count inside new_vec.
    unsafe { *output = new_vec };
    // Drop arrays here explicitly to control timing.
    drop(arrays);
    0
}

// Destructor for the payload: frees the Box<BoxFn>.
//
// SAFETY: closure user data is owned by mlx-c for the lifetime of the mlx_closure
// ref-count. When the ref-count reaches zero, mlx-c calls this dtor with the
// `payload` pointer we registered in `from_fn`. `Box::from_raw` here returns
// ownership back to Rust and drops the closure. The `catch_unwind` wrapper
// prevents any panic inside a user-supplied closure destructor from unwinding
// through the C frame, which would be UB per RFC 2945. On caught panic we abort
// instead — matching the defensive pattern in `rust_closure_callback`.
unsafe extern "C" fn rust_closure_dtor(payload: *mut std::os::raw::c_void) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        // SAFETY: payload was created by Box::into_raw(Box::new(f)) in from_fn.
        let _ = unsafe { Box::from_raw(payload.cast::<BoxFn>()) };
    }));
    if result.is_err() {
        tracing::error!(
            target: "rmlx::ffi",
            "closure destructor panicked — aborting to avoid UB across FFI"
        );
        std::process::abort();
    }
}

impl Closure {
    /// Create a closure from an arbitrary Rust function.
    ///
    /// `f` receives the packed input arrays and must return the packed output
    /// arrays. It must be `Send + Sync + 'static` (no borrowed captures).
    ///
    /// The closure is suitable for passing to [`compile`] or [`compile_shapeless`].
    pub fn from_fn<F>(f: F) -> Self
    where
        F: Fn(Vec<Array>) -> Result<Vec<Array>> + Send + Sync + 'static,
    {
        install_error_handler();
        let boxed: BoxFn = Box::new(f);
        let payload = Box::into_raw(Box::new(boxed)).cast::<std::os::raw::c_void>();
        // SAFETY: payload is a valid pointer; rust_closure_callback and
        // rust_closure_dtor are 'static extern "C" fns.
        let inner = unsafe {
            sys::mlx_closure_new_func_payload(
                Some(rust_closure_callback),
                payload,
                Some(rust_closure_dtor),
            )
        };
        Self { inner }
    }

    /// Apply this closure to a slice of input arrays.
    ///
    /// Returns the output arrays in a `Vec<Array>`.
    pub fn apply(&self, inputs: &[&Array]) -> Result<Vec<Array>> {
        install_error_handler();

        // Pack inputs.
        let vec_in = unsafe { sys::mlx_vector_array_new() };
        for arr in inputs {
            let st = unsafe { sys::mlx_vector_array_append_value(vec_in, arr.inner) };
            if st != 0 {
                unsafe { sys::mlx_vector_array_free(vec_in) };
                return Err(Error::Mlx(
                    "Closure::apply: mlx_vector_array_append_value failed".to_owned(),
                ));
            }
        }

        let mut vec_out = unsafe { sys::mlx_vector_array_new() };
        let status = unsafe { sys::mlx_closure_apply(&raw mut vec_out, self.inner, vec_in) };
        unsafe { sys::mlx_vector_array_free(vec_in) };
        unsafe { check_status(status, "Closure::apply") }?;

        // Unpack outputs.
        let n = unsafe { sys::mlx_vector_array_size(vec_out) };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut arr = unsafe { sys::mlx_array_new() };
            let st = unsafe { sys::mlx_vector_array_get(&raw mut arr, vec_out, i) };
            if st != 0 {
                unsafe { sys::mlx_vector_array_free(vec_out) };
                return Err(Error::Mlx(format!(
                    "Closure::apply: mlx_vector_array_get[{i}] failed"
                )));
            }
            out.push(Array { inner: arr });
        }
        unsafe { sys::mlx_vector_array_free(vec_out) };
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// compile / compile_shapeless
// ---------------------------------------------------------------------------

/// Compile a closure with shape-aware caching (`shapeless=false`).
///
/// On the first invocation with a given input shape, MLX traces the closure,
/// compiles the resulting graph to a Metal program, and caches the result.
/// Subsequent calls with the same shapes replay the compiled program directly.
///
/// Returns a compiled `Closure`. The returned closure may be applied with
/// [`Closure::apply`].
#[allow(
    clippy::needless_pass_by_value,
    reason = "Closure is consumed via fun.inner passed to mlx_compile; FFI takes ownership of the handle"
)]
pub fn compile(fun: Closure) -> Result<Closure> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_closure_new() };
    let status = unsafe { sys::mlx_compile(&raw mut res, fun.inner, false) };
    unsafe { check_status(status, "compile") }?;
    // SAFETY: mlx_compile filled `res` with a valid compiled closure.
    Ok(Closure { inner: res })
}

/// Compile a closure with shapeless caching (`shapeless=true`).
///
/// MLX caches a single compiled Metal program regardless of input shapes.
/// The dispatch grid is adjusted at runtime for each invocation. This is
/// the equivalent of `@partial(mx.compile, shapeless=True)` in Python.
///
/// Prefer this for functions called with variable sequence lengths (e.g. the
/// GDN prefill ops with varying T per chunk).
#[allow(
    clippy::needless_pass_by_value,
    reason = "Closure is consumed via fun.inner passed to mlx_compile; FFI takes ownership of the handle"
)]
pub fn compile_shapeless(fun: Closure) -> Result<Closure> {
    install_error_handler();
    let mut res = unsafe { sys::mlx_closure_new() };
    let status = unsafe { sys::mlx_compile(&raw mut res, fun.inner, true) };
    unsafe { check_status(status, "compile_shapeless") }?;
    // SAFETY: mlx_compile filled `res` with a valid compiled closure.
    Ok(Closure { inner: res })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "compile_tests.rs"]
mod tests;
