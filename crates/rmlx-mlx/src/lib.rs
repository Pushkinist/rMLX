// LOC-exempt: crate root of the mlx-c wrapper. `Array` — the owned handle every
// other module in the workspace passes around — is inseparable from the
// crate-private FFI plumbing it calls on every op: the thread-local error
// capture behind `check_status`, the process-global default-stream helper, the
// quant-mode CString cache, and the null-handle sentinel for optional
// arguments. Splitting `Array` out would export that plumbing across a module
// boundary purely to move lines, with no reader benefit.
//! Safe Rust wrapper around the brew-prebuilt `mlx-c` library.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use rmlx_mlx::{Array, Device, Dtype, add};
//! ```

// unsafe_code: mlx-rs FFI bridge — entire crate is the safe Rust wrapper over mlx-c
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
#![warn(missing_docs)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing,
        clippy::print_stdout,
        clippy::print_stderr,
        clippy::unreachable,
        clippy::todo,
        clippy::unimplemented,
        clippy::float_cmp,
        // disallowed_methods is a separate lint from unwrap_used;
        // test code (bucket-B) is already exempted for unwrap_used, extend here.
        clippy::disallowed_methods,
    )
)]

pub mod compile;
/// Bounded-window Metal GPU trace capture. Debug-only: compiled out entirely
/// unless the `metal-capture` feature is enabled, so a release build carries no
/// state, no decode-path branch, and no route to `MTLCaptureManager`.
#[cfg(feature = "metal-capture")]
pub mod metal_capture;
pub mod metal_kernel;
mod nax;
mod sys;
/// Parser for `xcrun xctrace export` XML — the headless route to GPU wall-clock
/// timing and the CPU→GPU gap. Debug-only, behind the same feature as
/// `metal_capture`: it is profiling tooling, not part of the served binary.
#[cfg(feature = "metal-capture")]
pub mod xctrace;

use std::cell::Cell;
use std::ptr;

use rmlx_core::error::{Error, Result};

// ---------------------------------------------------------------------------
// Thread-local error capture
// ---------------------------------------------------------------------------
//
// mlx-c delivers errors via a registered callback. We store the message in a
// thread-local so check_status can retrieve it after a failing C call.
//
// `Cell<Option<String>>` instead of `RefCell<Option<String>>`: the access
// pattern is always `take()` or `set(Some(s))`, both of which compile to
// a plain swap/store with no runtime borrow-check branch. `RefCell` would
// add a borrow-count field (8 bytes) and an isize compare on every access.
// ch-15 (wrapper-types): prefer `Cell` over `RefCell` when `T` is only ever
// `take()`/`set()`/`replace()` — no shared borrows needed.

thread_local! {
    pub(crate) static LAST_ERROR: Cell<Option<String>> = const { Cell::new(None) };
}

/// Install the thread-local error handler. Called once per process, lazily.
///
/// # Safety
/// mlx_set_error_handler is thread-safe. The callback receives a valid
/// NUL-terminated `*const c_char` for the duration of the call.
pub(crate) fn install_error_handler() {
    use std::sync::Once;
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        unsafe extern "C" fn handler(msg: *const std::ffi::c_char, _data: *mut std::ffi::c_void) {
            // SAFETY: msg is a valid NUL-terminated string for the duration of
            // this callback, as specified by the mlx-c error handler contract.
            let s = unsafe {
                if msg.is_null() {
                    "<null error>".to_owned()
                } else {
                    std::ffi::CStr::from_ptr(msg).to_string_lossy().into_owned()
                }
            };
            LAST_ERROR.with(|cell| {
                cell.set(Some(s));
            });
        }
        // SAFETY: registering a C-compatible callback with no captured state.
        unsafe {
            sys::mlx_set_error_handler(Some(handler), ptr::null_mut(), None);
        }
        warn_on_mlx_version_skew();
        nax::warn_if_nax_kernels_missing();
    });
}

/// The MLX version `rmlx-mlx` was compiled against (from `mlx/version.h`,
/// captured by `build.rs`). `"unknown"` when the header was unreadable.
const MLX_BUILD_VERSION: &str = env!("RMLX_MLX_BUILD_VERSION");

/// Whether the MLX this build links against ships the `steel_gemm_fused_nax*`
/// GEMM kernels — `"present"` / `"absent"` / `"unknown"`. Stamped by
/// `build.rs` from the same metallib scan that drives the missing-kernel
/// build warning (`crates/rmlx-mlx/build.rs`, `check_mlx_pin`).
///
/// `pub` (unlike `MLX_BUILD_VERSION`): this is the one fact about the MLX
/// build that a downstream binary genuinely cannot derive itself, and the
/// only channel it has out of this crate. `rmlx-metrics::identity` cannot
/// read `RMLX_MLX_NAX` directly — `cargo:rustc-env` from this crate's build
/// script only reaches this crate's own compiler invocation, and
/// `rmlx-metrics` deliberately does not depend on `rmlx-mlx` (its build
/// script hard-requires a working Homebrew MLX/mlx-c install, which
/// `rmlx-metrics` must not need). So `rmlx-cli::main()` — the one binary that
/// links both — reads this constant and calls
/// `rmlx_metrics::identity::set_mlx_nax` with it once at startup, before any
/// metrics recording. See `docs/METRICS_DB.md`.
///
/// This describes the machine that **built** the binary — the right answer for
/// a source install, the wrong one for a prebuilt bottle or release tarball,
/// which link MLX through a package-manager symlink and load whatever the
/// installing user has. The operator-facing answer therefore comes from the
/// crate-private `nax` module's runtime probe of the library actually loaded,
/// not from here.
pub const NAX_CAPABILITY: &str = env!("RMLX_MLX_NAX");

/// Read the MLX version of the dylib actually loaded into this process.
///
/// Returns `None` if mlx-c reports an error or hands back a null string.
fn runtime_mlx_version() -> Option<String> {
    // SAFETY: mlx_version writes an owned mlx_string on success (status 0).
    // mlx_string_data borrows its NUL-terminated buffer, which stays valid
    // until mlx_string_free; the value is copied out before freeing.
    unsafe {
        let mut s = sys::mlx_string_new();
        if sys::mlx_version(&raw mut s) != 0 {
            let _ = sys::mlx_string_free(s);
            return None;
        }
        let data = sys::mlx_string_data(s);
        let out = if data.is_null() {
            None
        } else {
            Some(
                std::ffi::CStr::from_ptr(data)
                    .to_string_lossy()
                    .into_owned(),
            )
        };
        let _ = sys::mlx_string_free(s);
        out
    }
}

/// Warn when the loaded MLX differs from the one this binary was built against.
///
/// The linked dylib's install name is a Homebrew `opt` symlink, so upgrading
/// MLX silently redirects this binary to a different library and metallib with
/// no rebuild. That skew is an ABI hazard and has been observed to cost a large
/// factor of GPU matmul throughput, which shows up only as slow prefill — an
/// easy defect to misattribute to model code. Surface it instead of letting it
/// pass silently.
///
/// Warn (not fail): a skew is usually benign-but-slow, and a hard error here
/// would brick every run on a host whose package manager moved ahead.
fn warn_on_mlx_version_skew() {
    if MLX_BUILD_VERSION == "unknown" {
        return;
    }
    let Some(runtime) = runtime_mlx_version() else {
        return;
    };
    // The runtime string carries a ".devYYYYMMDD+hash" suffix on dev builds;
    // compare only the leading "major.minor.patch".
    let runtime_core = runtime.split(['-', '+']).next().unwrap_or(&runtime);
    let runtime_core = runtime_core
        .split('.')
        .take(3)
        .collect::<Vec<_>>()
        .join(".");
    if runtime_core != MLX_BUILD_VERSION {
        tracing::warn!(
            build_version = MLX_BUILD_VERSION,
            runtime_version = %runtime,
            "MLX build/runtime version skew: this binary was compiled against \
             MLX {MLX_BUILD_VERSION} but loaded MLX {runtime}. The linked dylib \
             resolves through a package-manager symlink, so an upgrade swaps it \
             without a rebuild. Expect ABI risk and possible large GPU-throughput \
             differences (prefill). Rebuild against the loaded version, or pin \
             the package back to {MLX_BUILD_VERSION}."
        );
    }
}

/// Check the return code of a mlx-c function call.
///
/// Returns `Ok(())` if `status == 0`, otherwise extracts the error message
/// captured by the thread-local handler and wraps it in `Error::Mlx`.
///
/// # Safety
/// Must be called immediately after the mlx-c call whose status is being
/// checked, on the same thread, before any other mlx-c call that could
/// overwrite the thread-local error slot.
pub(crate) unsafe fn check_status(status: i32, context: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let msg = LAST_ERROR.with(Cell::take);
    let msg = msg.unwrap_or_else(|| format!("mlx-c returned non-zero status {status}"));
    Err(Error::Mlx(format!("{context}: {msg}")))
}

// ---------------------------------------------------------------------------
// Stream helper: borrow the process-global default stream, run a closure.
// ---------------------------------------------------------------------------
//
// The previous implementation called `mlx_stream_new_device`
// on every op invocation. Each call to `mlx_stream_new_device` spawns a new
// OS worker thread inside MLX. A single Gemma4 forward pass invokes ~42
// layers × several ops per layer — hundreds of `with_stream` calls per step.
// After 3–6 decode steps the per-process thread limit (~2 048 on macOS) is
// exhausted and `pthread_create` returns EAGAIN, manifesting as:
//
// mlx: Array::eval: thread constructor failed: Resource temporarily unavailable
//
// Fix: use `mlx_default_cpu_stream_new` / `mlx_default_gpu_stream_new` which
// return a *reference-counted handle to the already-running default stream*
// (no new thread). Freeing the handle with `mlx_stream_free` decrements the
// ref-count — it does NOT tear down the stream or its thread.

/// Borrow the process-global default stream for `device`, call `f(stream)`,
/// release the handle, and return the result.
///
/// All ops that need a stream use this helper to avoid repeating the boilerplate.
///
/// # Safety
/// `f` must not store the stream handle past the duration of the call.
pub(crate) unsafe fn with_stream<T>(device: Device, f: impl FnOnce(sys::mlx_stream) -> T) -> T {
    // Obtain a reference-counted handle to the existing default stream.
    // This does NOT spawn a new thread — the stream already owns its thread.
    let stream = match device {
        Device::Cpu => unsafe { sys::mlx_default_cpu_stream_new() },
        Device::Gpu => unsafe { sys::mlx_default_gpu_stream_new() },
    };
    let result = f(stream);
    // Release our ref-count handle. The stream itself stays alive (process lifetime).
    // SAFETY: stream is a valid handle obtained just above.
    unsafe { sys::mlx_stream_free(stream) };
    result
}

/// Ensure a GPU stream with a Metal command encoder is registered as the
/// default stream for the **calling thread**.
///
/// MLX's `eval.cpp::eval_impl` calls `metal::get_command_encoder(stream)`
/// on the **calling thread**. That function looks up a thread-local map of
/// `{stream_index → CommandEncoder}`.  The entry for a given stream index
/// is created only when `mlx::core::new_stream` (which calls
/// `metal::new_stream`) is called on that thread.  Tokio blocking-pool
/// threads never call `new_stream`, so any `Array::eval()` on them fails
/// with "There is no Stream(gpu, N) in current thread."
///
/// This function:
///   1. Creates a new GPU stream via `mlx_stream_new_device` — which calls
///      `mlx::core::new_stream` → `metal::new_stream` on the calling thread,
///      registering a fresh `CommandEncoder` in the thread-local map.
///   2. Sets the new stream as the calling thread's default so that all
///      subsequent `with_stream(Device::Gpu, …)` calls return it.
///   3. Stores the handle in a thread-local so it lives for the thread's
///      lifetime (do NOT free — that would drop the CommandEncoder).
///
/// Idempotent — subsequent calls from the same thread are no-ops.
///
/// No-op if the GPU device is unavailable.
pub fn ensure_gpu_default_stream() {
    // Thread-local storage: init flag + stream handle.
    // The handle is held for the thread's lifetime so the CommandEncoder
    // entry in the thread-local encoders map stays alive.
    thread_local! {
        static GPU_STREAM_INIT: Cell<bool> = const { Cell::new(false) };
        // The mlx_stream handle keeps the stream and its CommandEncoder alive.
        // Intentionally leaked on thread exit — each leaked handle is also an
        // MLX-internal OS thread. This is only safe because callers bound the
        // set of distinct threads that ever run this: `rmlx serve` builds its
        // tokio runtime with a capped `max_blocking_threads` and a long
        // `thread_keep_alive` (see `crates/rmlx-cli/src/commands/serve.rs`)
        // so the blocking pool's worker threads are reused, not
        // idle-reaped-and-replaced — the cumulative leak is bounded by that
        // cap, not merely "these threads are long-lived."
        static GPU_STREAM_HANDLE: Cell<sys::mlx_stream> =
            const { Cell::new(sys::mlx_stream { ctx: ptr::null_mut() }) };
    }

    if GPU_STREAM_INIT.with(Cell::get) {
        return;
    }

    // SAFETY:
    // - mlx_device_new_type(MLX_GPU, 0): creates a handle to the GPU device.
    //   Always succeeds on Apple Silicon; ctx is non-null on success.
    // - mlx_stream_new_device(gpu_dev): calls mlx::core::new_stream which
    //   calls metal::new_stream on the calling thread, registering a new
    //   CommandEncoder in the calling thread's thread-local encoder map.
    //   Returns a ref-counted handle to the new stream.
    // - mlx_set_default_stream(stream): stores Stream(gpu, N) as the
    //   calling thread's default.  All subsequent mlx_default_gpu_stream_new()
    //   calls on this thread will return Stream(gpu, N).
    // - mlx_device_free: releases the temporary device handle; the stream
    //   retains a reference to the underlying device.
    // - We store the stream handle in GPU_STREAM_HANDLE and do NOT call
    //   mlx_stream_free — the CommandEncoder entry stays alive as long as
    //   the handle is alive.
    unsafe {
        let gpu_dev = sys::mlx_device_new_type(sys::mlx_device_type_::MLX_GPU, 0);
        if gpu_dev.ctx.is_null() {
            return;
        }
        let stream = sys::mlx_stream_new_device(gpu_dev);
        let _ = sys::mlx_device_free(gpu_dev);

        if stream.ctx.is_null() {
            return;
        }

        let _ = sys::mlx_set_default_stream(stream);

        // Store the handle; do NOT call mlx_stream_free.
        GPU_STREAM_HANDLE.with(|cell| cell.set(stream));
        GPU_STREAM_INIT.with(|cell| cell.set(true));
    }
}

/// Ensure a CPU stream is registered as the default stream for the **calling
/// thread**, so any `Array::eval()` that schedules work on the CPU stream from
/// this thread finds a registered command encoder.
///
/// This is the CPU analog of [`ensure_gpu_default_stream`]. Since MLX 0.31/0.32
/// the default CPU/GPU streams are **thread-local** (`mlx/stream.cpp`:
/// `static thread_local ... default_streams`) and the CPU backend resolves a
/// stream's `CommandEncoder` through a **thread-local** map first
/// (`mlx/backend/cpu/encoder.cpp::get_command_encoder`), falling back to a
/// process-global map and otherwise throwing
/// "There is no Stream(cpu, N) in current thread." A tokio blocking-pool worker
/// that never called `mlx::core::new_stream` for the CPU device therefore
/// faults the first time an op is scheduled on the CPU stream — e.g. the K8V8
/// `exit_prefill` quantization, whose reduction MLX evaluates lazily on the CPU
/// stream even when the model forward runs on the GPU device.
///
/// The mechanism mirrors the GPU guard exactly:
///   1. Create a fresh CPU stream via `mlx_stream_new_device` — which calls
///      `mlx::core::new_stream` → `cpu::new_stream`, registering a
///      `CommandEncoder` for this stream index in the calling thread's
///      thread-local encoder map.
///   2. Set it as the calling thread's default CPU stream so every subsequent
///      `default_stream(cpu)` / `with_stream(Device::Cpu, …)` on this thread
///      resolves to a stream whose encoder is registered here.
///   3. Store the handle in a thread-local for the thread's lifetime (do NOT
///      free — that would drop the encoder entry).
///
/// Idempotent — subsequent calls from the same thread are no-ops.
pub fn ensure_cpu_default_stream() {
    // Thread-local storage: init flag + stream handle. The handle is held for
    // the thread's lifetime so the CommandEncoder entry in the thread-local
    // encoder map stays alive — intentionally leaked on thread exit, same
    // tradeoff and same bound as `ensure_gpu_default_stream` above: this is
    // only safe because the tokio blocking pool this guards is capped
    // (`max_blocking_threads`) and kept warm (`thread_keep_alive`) by
    // `rmlx serve`, so the cumulative leak is bounded by that cap rather than
    // relying on the workers simply being long-lived.
    thread_local! {
        static CPU_STREAM_INIT: Cell<bool> = const { Cell::new(false) };
        static CPU_STREAM_HANDLE: Cell<sys::mlx_stream> =
            const { Cell::new(sys::mlx_stream { ctx: ptr::null_mut() }) };
    }

    if CPU_STREAM_INIT.with(Cell::get) {
        return;
    }

    // SAFETY: mirror of ensure_gpu_default_stream for the CPU device.
    // - mlx_device_new_type(MLX_CPU, 0): handle to the CPU device (always
    //   available on Apple Silicon; ctx is non-null on success).
    // - mlx_stream_new_device(cpu_dev): mlx::core::new_stream(cpu) →
    //   cpu::new_stream(s), which registers a CommandEncoder for the new stream
    //   index in the calling thread's thread-local encoder map. Returns a
    //   ref-counted handle to the new stream.
    // - mlx_set_default_stream(stream): store Stream(cpu, N) as the calling
    //   thread's default; all subsequent mlx_default_cpu_stream_new() calls on
    //   this thread return it.
    // - mlx_device_free: releases the temporary device handle; the stream keeps
    //   its own reference to the underlying device.
    // - We store the stream handle in CPU_STREAM_HANDLE and do NOT call
    //   mlx_stream_free — the CommandEncoder entry stays alive while the handle
    //   is alive.
    unsafe {
        let cpu_dev = sys::mlx_device_new_type(sys::mlx_device_type_::MLX_CPU, 0);
        if cpu_dev.ctx.is_null() {
            return;
        }
        let stream = sys::mlx_stream_new_device(cpu_dev);
        let _ = sys::mlx_device_free(cpu_dev);

        if stream.ctx.is_null() {
            return;
        }

        let _ = sys::mlx_set_default_stream(stream);

        // Store the handle; do NOT call mlx_stream_free.
        CPU_STREAM_HANDLE.with(|cell| cell.set(stream));
        CPU_STREAM_INIT.with(|cell| cell.set(true));
    }
}

// ---------------------------------------------------------------------------
// CString cache for quantization mode strings
// ---------------------------------------------------------------------------
//
// `quantized_matmul`, `dequantize`, `gather_qmm`, and
// `scaled_dot_product_attention` each accept a mode `&str` and convert it to
// a `CString` per call — which is per-layer per-token on the hot path.
//
// The set of legal mode strings is small and fixed. Caching them in
// `OnceLock<CString>` eliminates those heap allocations. Unknown strings
// fall through to a dynamic `CString::new` for forward-compatibility.

static CSTR_AFFINE: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
static CSTR_MXFP8: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
static CSTR_MXFP4: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
static CSTR_NVFP4: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
static CSTR_ARRAY: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
static CSTR_CAUSAL: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
static CSTR_CONSTANT: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();
static CSTR_EMPTY: std::sync::OnceLock<std::ffi::CString> = std::sync::OnceLock::new();

/// Return a `&'static CStr` for a known mode string, or a heap-allocated `CString`.
///
/// Uses `Cow<'static, CStr>` as the return type:
/// - `Cow::Borrowed(&'static CStr)` — zero allocation for known modes.
/// - `Cow::Owned(CString)` — dynamic allocation for unknown modes.
///
/// Call `.as_ptr()` on the result to get the `*const c_char` expected by mlx-c.
#[allow(
    clippy::expect_used,
    reason = "CString::new on a hardcoded ASCII literal — interior NUL is impossible; these are OnceLock initialisers"
)]
pub(crate) fn mode_to_cstr(
    mode: &str,
    ctx: &str,
) -> Result<std::borrow::Cow<'static, std::ffi::CStr>> {
    use std::borrow::Cow;
    use std::ffi::CString;
    match mode {
        "affine" => Ok(Cow::Borrowed(
            CSTR_AFFINE
                .get_or_init(|| CString::new("affine").expect("affine cstr"))
                .as_c_str(),
        )),
        "mxfp8" => Ok(Cow::Borrowed(
            CSTR_MXFP8
                .get_or_init(|| CString::new("mxfp8").expect("mxfp8 cstr"))
                .as_c_str(),
        )),
        "mxfp4" => Ok(Cow::Borrowed(
            CSTR_MXFP4
                .get_or_init(|| CString::new("mxfp4").expect("mxfp4 cstr"))
                .as_c_str(),
        )),
        "nvfp4" => Ok(Cow::Borrowed(
            CSTR_NVFP4
                .get_or_init(|| CString::new("nvfp4").expect("nvfp4 cstr"))
                .as_c_str(),
        )),
        "array" => Ok(Cow::Borrowed(
            CSTR_ARRAY
                .get_or_init(|| CString::new("array").expect("array cstr"))
                .as_c_str(),
        )),
        "causal" => Ok(Cow::Borrowed(
            CSTR_CAUSAL
                .get_or_init(|| CString::new("causal").expect("causal cstr"))
                .as_c_str(),
        )),
        "constant" => Ok(Cow::Borrowed(
            CSTR_CONSTANT
                .get_or_init(|| CString::new("constant").expect("constant cstr"))
                .as_c_str(),
        )),
        "" => Ok(Cow::Borrowed(
            CSTR_EMPTY
                .get_or_init(|| CString::new("").expect("empty cstr"))
                .as_c_str(),
        )),
        other => CString::new(other)
            .map(Cow::Owned)
            .map_err(|e| Error::Mlx(format!("{ctx}: invalid mode string: {e}"))),
    }
}

// ---------------------------------------------------------------------------
// Null-handle sentinel for optional FFI arguments (ch-18 F1)
// ---------------------------------------------------------------------------
//
// Several mlx-c wrappers accept optional Array arguments (biases, freqs,
// sinks, lhs_indices, weight). When the argument is `None`, the idiom is to
// pass a freshly allocated empty handle (`mlx_array_new()`) and free it after
// the call — one heap alloc + one atomic ref-count free per invocation.
//
// These calls happen per layer per decode step:
// - rope / rope_dynamic: freqs_null — once per attention layer.
// - scaled_dot_product_attention: sinks_null, and mask_inner when None.
// - rms_norm: w_arr when weight.is_none() — Gemma4 v_norm.
// - quantized_matmul / dequantize / gather_qmm: biases when None.
// - dequantize / quantize: global_scale — always None.
//
// Fix: keep one process-global empty handle in `EMPTY_ARRAY`. The raw
// `mlx_array` inner value (ctx=null) is passed to every optional-None site.
// The sentinel is never freed — it lives for the process lifetime, and its
// ctx is null so the (non-)free is a no-op anyway.
//
// Safety: mlx-c arrays are reference-counted shared_ptr internally.
// Passing the same ctx=null handle to multiple concurrent FFI calls is safe:
// the null ctx signals "absent" to the C++ side; no ref-count manipulation
// occurs for null-ctx arrays per the mlx-c contract.

pub(crate) static EMPTY_ARRAY_SENTINEL: std::sync::OnceLock<Array> = std::sync::OnceLock::new();

/// Return the inner `mlx_array` handle of the process-global null sentinel.
///
/// Use this instead of `mlx_array_new()` for every optional-absent argument.
/// The returned handle must NOT be freed — omit the matching `mlx_array_free`.
///
/// # Safety
/// The null-ctx handle returned here is only valid as a "sentinel absent"
/// argument to mlx-c functions that document "may be null". Never store or
/// evaluate the returned handle as a real Array.
#[inline]
pub(crate) fn null_sentinel() -> sys::mlx_array {
    EMPTY_ARRAY_SENTINEL
        .get_or_init(|| {
            // SAFETY: mlx_array_new returns a default-constructed handle with
            // ctx=null. This is the empty/absent sentinel value mlx-c uses.
            let inner = unsafe { sys::mlx_array_new() };
            Array { inner }
        })
        .inner
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The device to run ops on.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed device enum — two MLX device targets (Cpu/Gpu); adding a device requires updating all Device match arms and the mlx-c FFI binding"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    /// Run ops on the host CPU stream.
    Cpu,
    /// Run ops on the Metal GPU stream.
    Gpu,
}

/// Element dtype subset. Extend in S1.4b as needed.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dtype enum — six MLX element types (Bf16/F16/F32/U8/U32/I32); adding a dtype requires updating to_sys(), from_sys(), and all Dtype match arms across the codebase"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// Brain float 16 (bfloat16).
    Bf16,
    /// IEEE float 16 (half precision).
    F16,
    /// IEEE float 32 (single precision).
    F32,
    /// Unsigned 8-bit integer.
    U8,
    /// Unsigned 32-bit integer.
    U32,
    /// Signed 32-bit integer.
    I32,
}

impl Dtype {
    pub(crate) fn to_sys(self) -> sys::mlx_dtype_ {
        match self {
            Dtype::Bf16 => sys::mlx_dtype_::MLX_BFLOAT16,
            Dtype::F16 => sys::mlx_dtype_::MLX_FLOAT16,
            Dtype::F32 => sys::mlx_dtype_::MLX_FLOAT32,
            Dtype::U8 => sys::mlx_dtype_::MLX_UINT8,
            Dtype::U32 => sys::mlx_dtype_::MLX_UINT32,
            Dtype::I32 => sys::mlx_dtype_::MLX_INT32,
        }
    }

    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "mlx_dtype_ is a C FFI enum; returning None for unrecognised variants is the correct and intentional fall-through"
    )]
    fn from_sys(d: sys::mlx_dtype_) -> Option<Self> {
        match d {
            sys::mlx_dtype_::MLX_BFLOAT16 => Some(Dtype::Bf16),
            sys::mlx_dtype_::MLX_FLOAT16 => Some(Dtype::F16),
            sys::mlx_dtype_::MLX_FLOAT32 => Some(Dtype::F32),
            sys::mlx_dtype_::MLX_UINT8 => Some(Dtype::U8),
            sys::mlx_dtype_::MLX_UINT32 => Some(Dtype::U32),
            sys::mlx_dtype_::MLX_INT32 => Some(Dtype::I32),
            _ => None,
        }
    }

    /// Element size in bytes.
    pub fn itemsize(self) -> usize {
        match self {
            Dtype::Bf16 | Dtype::F16 => 2,
            Dtype::F32 | Dtype::U32 | Dtype::I32 => 4,
            Dtype::U8 => 1,
        }
    }
}

/// Map a `safetensors::Dtype` to our `Dtype`.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "safetensors::Dtype may gain new variants; capturing all unsupported ones and returning an error is the correct and intentional pattern"
)]
pub fn dtype_from_safetensors(st: safetensors::Dtype) -> Result<Dtype> {
    match st {
        safetensors::Dtype::BF16 => Ok(Dtype::Bf16),
        safetensors::Dtype::F16 => Ok(Dtype::F16),
        safetensors::Dtype::F32 => Ok(Dtype::F32),
        safetensors::Dtype::U8 => Ok(Dtype::U8),
        safetensors::Dtype::U32 => Ok(Dtype::U32),
        safetensors::Dtype::I32 => Ok(Dtype::I32),
        other => Err(Error::Mlx(format!(
            "unsupported safetensors dtype {other:?}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Array
// ---------------------------------------------------------------------------

/// Heap-allocated MLX array. Dropping frees the underlying mlx-c handle.
pub struct Array {
    inner: sys::mlx_array,
}

// SAFETY: mlx_array wraps a std::shared_ptr<mlx::core::array> under the hood.
// The mlx-c docs state that arrays are reference-counted and thread-safe to
// pass (same semantics as shared_ptr).
unsafe impl Send for Array {}
unsafe impl Sync for Array {}

impl Drop for Array {
    fn drop(&mut self) {
        // NOTE: previously this fired a `tracing::trace!` per drop, which
        // turned out to be the dominant cost in `RUST_LOG=debug,rmlx=trace`
        // mode (every MLX op allocates and drops several arrays — millions
        // of trace events per decode session). Bench p50 went from ~17 TPS
        // (with the trace) to ~35 TPS (without). The drop event is a
        // memory-leak debugging aid, not load-bearing — skip it on the hot
        // path. If you need to debug a leak, re-enable temporarily.
        // SAFETY: self.inner is a valid handle created by mlx-c. Freeing it
        // once on drop is correct; we never alias the handle.
        unsafe {
            sys::mlx_array_free(self.inner);
        }
    }
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Print shape + dtype only; contents may be unevaluated.
        write!(
            f,
            "Array(shape={:?}, dtype={:?})",
            self.shape(),
            self.dtype()
        )
    }
}

impl Array {
    /// Create an array from row-major host bytes.
    ///
    /// `data` must be exactly `product(shape) * dtype.itemsize()` bytes.
    pub fn from_bytes(data: &[u8], shape: &[i32], dtype: Dtype) -> Result<Self> {
        install_error_handler();

        let n_elems: usize = shape.iter().map(|&d| d as usize).product();
        let expected = n_elems * dtype.itemsize();
        if data.len() != expected {
            return Err(Error::Mlx(format!(
                "Array::from_bytes: data length {} != expected {} \
                 (shape={shape:?}, dtype={dtype:?})",
                data.len(),
                expected,
            )));
        }

        // SAFETY: mlx_array_new_data copies the buffer immediately. The
        // returned handle owns its data; `data` need not outlive this call.
        let inner = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as i32,
                dtype.to_sys(),
            )
        };
        // NOTE: tracing::trace! removed here — same class as the Array::drop
        // trace removed earlier (lib.rs:281-293). from_bytes is called from
        // every safetensor load (thousands of calls at startup) and once per
        // decode step to wrap the next token id. Under RUST_LOG=debug,rmlx=trace
        // the trace fired millions of times per session and paid JSON formatting
        // + non-blocking-channel overhead on the hot path. The pointer/shape/
        // dtype snapshot is a memory-leak aid, not a correctness invariant;
        // re-enable temporarily with RUST_LOG=trace if debugging a leak.
        Ok(Array { inner })
    }

    /// Create an array from a `&[f32]` slice.
    ///
    /// Convenience wrapper over [`Array::from_bytes`] that handles the
    /// `&[f32]` → `&[u8]` byte-reinterpret in the one allowed place (here,
    /// inside the `rmlx-mlx` FFI module). Callers in other crates MUST use
    /// this instead of writing their own `unsafe` reinterpret.
    pub fn from_f32_slice(data: &[f32], shape: &[i32]) -> Result<Self> {
        // SAFETY: f32 is Pod (no padding, no uninitialised bytes); the byte
        // slice is used read-only inside from_bytes which copies to MLX
        // immediately. The original `data` slice remains valid for the call.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        Self::from_bytes(bytes, shape, Dtype::F32)
    }

    /// Create an array from a `&[i32]` slice.
    ///
    /// Convenience wrapper over [`Array::from_bytes`] for `i32` data.
    /// See [`Array::from_f32_slice`] for the safety rationale.
    pub fn from_i32_slice(data: &[i32], shape: &[i32]) -> Result<Self> {
        // SAFETY: i32 is Pod; bytes are used read-only inside from_bytes.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
        Self::from_bytes(bytes, shape, Dtype::I32)
    }

    /// Create an Array from a `TensorView` obtained via the loader.
    ///
    /// The view carries safetensors dtype + shape + bytes. The bytes are
    /// copied into MLX immediately (MLX owns them after this call).
    /// For bf16 the native MLX `BF16` dtype is used — no host-side conversion.
    pub fn from_safetensor_view(view: &rmlx_loader::TensorView<'_>) -> Result<Self> {
        install_error_handler();

        let dtype = dtype_from_safetensors(view.dtype)?;
        let shape: Vec<i32> = view.shape.iter().map(|&d| d as i32).collect();
        Self::from_bytes(view.bytes, &shape, dtype)
    }

    /// Number of dimensions.
    pub fn ndim(&self) -> usize {
        // SAFETY: inner is a valid mlx_array handle.
        unsafe { sys::mlx_array_ndim(self.inner) }
    }

    /// Size of a single dimension.
    pub fn dim(&self, axis: usize) -> Result<i32> {
        if axis >= self.ndim() {
            return Err(Error::Mlx(format!(
                "Array::dim: axis {axis} out of bounds (ndim={})",
                self.ndim()
            )));
        }
        // SAFETY: axis < ndim, inner valid.
        Ok(unsafe { sys::mlx_array_dim(self.inner, axis as i32) })
    }

    /// All dimension sizes.
    pub fn shape(&self) -> Vec<i32> {
        let ndim = self.ndim();
        if ndim == 0 {
            return Vec::new();
        }
        // SAFETY: inner is valid; the returned pointer is valid for the
        // lifetime of the array object (mlx-c contract).
        let ptr = unsafe { sys::mlx_array_shape(self.inner) };
        if ptr.is_null() {
            return Vec::new();
        }
        // SAFETY: ptr points to ndim contiguous i32 values.
        unsafe { std::slice::from_raw_parts(ptr, ndim) }.to_vec()
    }

    /// Element dtype.
    pub fn dtype(&self) -> Dtype {
        // SAFETY: inner is valid.
        let raw = unsafe { sys::mlx_array_dtype(self.inner) };
        Dtype::from_sys(raw).unwrap_or(Dtype::U8)
    }

    /// Force evaluation (MLX is lazy — ops are deferred until materialized).
    pub fn eval(&self) -> Result<()> {
        install_error_handler();
        // SAFETY: inner is a valid mlx_array.
        let status = unsafe { sys::mlx_array_eval(self.inner) };
        // SAFETY: called immediately after the C function on the same thread.
        unsafe { check_status(status, "Array::eval") }
    }

    /// Asynchronously schedule this array's compute graph on the GPU stream
    /// without blocking the calling thread. The actual evaluation happens
    /// in the background; subsequent `to_bytes`/`eval` will wait if needed.
    ///
    /// Used to pipeline the next decode step's forward pass on the GPU
    /// while the current step's argmax is still being read out (mirrors
    /// mlx-lm's `mx.async_eval` pattern in `generate.py`).
    pub fn async_eval(&self) -> Result<()> {
        install_error_handler();
        // mlx_async_eval takes a vector_array; build a single-element vec.
        let vec = unsafe { sys::mlx_vector_array_new_value(self.inner) };
        let status = unsafe { sys::mlx_async_eval(vec) };
        unsafe { sys::mlx_vector_array_free(vec) };
        unsafe { check_status(status, "Array::async_eval") }
    }

    /// Copy array contents into a fresh `Vec<u8>`.
    ///
    /// Forces evaluation first so the read is always of materialized data.
    /// MLX is lazy and `async_eval` only *schedules* compute; the data pointer
    /// returned by `mlx_array_data_uint8` is not guaranteed to hold the result
    /// until the graph has actually run. Calling `eval()` here blocks until it
    /// has — idempotent and near-free on an already-evaluated array. Without
    /// it, reading the pointer can race the asynchronous evaluation and return
    /// stale/recycled buffer bytes (another array's data) rather than this
    /// array's value.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        install_error_handler();
        self.eval()?;
        let nbytes = unsafe { sys::mlx_array_nbytes(self.inner) };
        if nbytes == 0 {
            return Ok(Vec::new());
        }

        // SAFETY: mlx_array_data_uint8 returns a raw pointer valid while the
        // array is alive and evaluated. We copy out immediately.
        let ptr = unsafe { sys::mlx_array_data_uint8(self.inner) };
        if ptr.is_null() {
            return Err(Error::Mlx(
                "Array::to_bytes: data pointer is null — was eval() called?".into(),
            ));
        }
        // SAFETY: ptr is non-null and points to `nbytes` contiguous bytes
        // owned by the mlx array. Copied into Vec before returning.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec();
        Ok(bytes)
    }

    /// Create a logical copy of this array using `mlx_array_set`.
    ///
    /// mlx-c is reference-counted internally, so this is cheap — no data
    /// duplication on the hot path.
    pub fn try_clone(&self) -> Result<Self> {
        install_error_handler();
        // SAFETY: mlx_array_new returns a valid empty handle.
        let mut new_arr = unsafe { sys::mlx_array_new() };
        // SAFETY: mlx_array_set increments the ref-count of src and assigns
        // it to *arr. Both handles are valid mlx_array structs.
        let status = unsafe { sys::mlx_array_set(&raw mut new_arr, self.inner) };
        // SAFETY: called immediately after the C function on the same thread.
        unsafe { check_status(status, "Array::try_clone") }?;
        // NOTE: tracing::trace! removed here — same class as the Array::drop
        // trace removed earlier (lib.rs:281-293) and the from_bytes trace.
        // try_clone is called once per decode step (to alias the next-token
        // array into the cache); under RUST_LOG=debug,rmlx=trace the trace
        // fired once per token per step. The src/dst pointer snapshot is a
        // ref-count debugging aid; re-enable temporarily with RUST_LOG=trace.
        Ok(Array { inner: new_arr })
    }

    /// Cast this array to `dtype`.
    pub fn astype(&self, dtype: Dtype, device: Device) -> Result<Array> {
        install_error_handler();
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe {
            with_stream(device, |s| {
                sys::mlx_astype(&raw mut res, self.inner, dtype.to_sys(), s)
            })
        };
        unsafe { check_status(status, "astype") }?;
        Ok(Array { inner: res })
    }

    /// Reshape. Shape values of -1 are inferred (MLX convention).
    pub fn reshape(&self, shape: &[i32], device: Device) -> Result<Array> {
        install_error_handler();
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe {
            with_stream(device, |s| {
                sys::mlx_reshape(&raw mut res, self.inner, shape.as_ptr(), shape.len(), s)
            })
        };
        unsafe { check_status(status, "reshape") }?;
        Ok(Array { inner: res })
    }

    /// Permute dimensions. `axes` must be a permutation of `0..ndim`.
    pub fn transpose(&self, axes: &[i32], device: Device) -> Result<Array> {
        install_error_handler();
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe {
            with_stream(device, |s| {
                sys::mlx_transpose_axes(&raw mut res, self.inner, axes.as_ptr(), axes.len(), s)
            })
        };
        unsafe { check_status(status, "transpose") }?;
        Ok(Array { inner: res })
    }

    /// Materialize a row-major contiguous copy of this array.
    ///
    /// MLX ops like `transpose` produce a logical view with permuted strides —
    /// the data is *not* re-laid-out until something forces it. Built-in MLX
    /// ops honor those strides, so a transpose is usually free. Custom MSL
    /// kernels (`MetalKernel`) do **not**: they read the input buffer by raw
    /// linear index, so they see the un-permuted physical order and silently
    /// produce scrambled results. Call this before feeding a transposed (or
    /// otherwise non-contiguous) array to such a kernel so the physical layout
    /// matches the logical shape. `mlx_reshape` to the same element count does
    /// not guarantee this — a flattened strided view can still report as
    /// "contiguous" while its bytes follow the original strides.
    pub fn contiguous(&self, device: Device) -> Result<Array> {
        install_error_handler();
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe {
            with_stream(device, |s| {
                // allow_col_major = false → force row-major layout.
                sys::mlx_contiguous(&raw mut res, self.inner, false, s)
            })
        };
        unsafe { check_status(status, "contiguous") }?;
        Ok(Array { inner: res })
    }

    /// Slice `a[start:stop:strides]` along all axes simultaneously.
    ///
    /// `start`, `stop`, `strides` must all have length == `self.ndim()`.
    pub fn slice(
        &self,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
        device: Device,
    ) -> Result<Array> {
        install_error_handler();
        let ndim = self.ndim();
        if start.len() != ndim || stop.len() != ndim || strides.len() != ndim {
            return Err(Error::Mlx(format!(
                "slice: start/stop/strides length must equal ndim={ndim}"
            )));
        }
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe {
            with_stream(device, |s| {
                sys::mlx_slice(
                    &raw mut res,
                    self.inner,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    s,
                )
            })
        };
        unsafe { check_status(status, "slice") }?;
        Ok(Array { inner: res })
    }

    /// Write `update` into a slice of `self` and return the resulting array.
    ///
    /// Equivalent to `mlx_slice_update`: `res = src; res[start:stop:strides] = update`.
    /// `start`, `stop`, `strides` must all have length == `self.ndim()`.
    ///
    /// MLX may reuse the underlying buffer when the graph is compiled (lazy eval),
    /// making this cheaper than concat for fixed-size pre-allocated KV buffers.
    pub fn slice_update(
        &self,
        update: &Array,
        start: &[i32],
        stop: &[i32],
        strides: &[i32],
        device: Device,
    ) -> Result<Array> {
        install_error_handler();
        let ndim = self.ndim();
        if start.len() != ndim || stop.len() != ndim || strides.len() != ndim {
            return Err(Error::Mlx(format!(
                "slice_update: start/stop/strides length must equal ndim={ndim}"
            )));
        }
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe {
            with_stream(device, |s| {
                sys::mlx_slice_update(
                    &raw mut res,
                    self.inner,
                    update.inner,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    s,
                )
            })
        };
        unsafe { check_status(status, "slice_update") }?;
        Ok(Array { inner: res })
    }

    /// Gather elements at `indices` along `axis`. Equivalent to `np.take`.
    pub fn take(&self, indices: &Array, axis: i32, device: Device) -> Result<Array> {
        install_error_handler();
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe {
            with_stream(device, |s| {
                sys::mlx_take_axis(&raw mut res, self.inner, indices.inner, axis, s)
            })
        };
        unsafe { check_status(status, "take") }?;
        Ok(Array { inner: res })
    }
}

mod fast_ops;
mod ops;

pub use fast_ops::*;
pub use ops::*;

// ---------------------------------------------------------------------------
// Memory query helpers
// ---------------------------------------------------------------------------

/// High-water Metal allocator peak, in bytes.
///
/// Wraps `mlx_get_peak_memory` from mlx-c 0.6.0. Returns `None` if the C
/// call reports an error (e.g. on non-Metal / CPU-only builds).
///
/// The value is the maximum of the allocator's *live* byte count observed
/// since the process started, or since the last [`mlx_reset_peak_memory`].
/// A bare reading is therefore a process-lifetime dashboard number; to scope
/// it to one region use [`PeakBracket`].
///
/// Used by the `metal_peak_alloc_mb` metric emit in the server engine.
pub fn mlx_peak_memory_bytes() -> Option<u64> {
    install_error_handler();
    let mut res: usize = 0;
    // SAFETY: writing to a stack `usize` we own; `mlx_get_peak_memory` is
    // thread-safe per mlx-c contract.
    let status = unsafe { sys::mlx_get_peak_memory(&raw mut res) };
    if status != 0 {
        return None;
    }
    Some(res as u64)
}

/// Currently live (allocated and not yet freed) Metal allocator bytes.
///
/// Wraps `mlx_get_active_memory`. Returns `None` if the C call reports an
/// error. Pooled-but-free buffers are *not* counted here — they sit in the
/// allocator's cache, which `mlx_get_cache_memory` reports separately.
pub fn mlx_active_memory_bytes() -> Option<u64> {
    install_error_handler();
    let mut res: usize = 0;
    // SAFETY: writing to a stack `usize` we own; `mlx_get_active_memory` is
    // thread-safe per mlx-c contract.
    let status = unsafe { sys::mlx_get_active_memory(&raw mut res) };
    if status != 0 {
        return None;
    }
    Some(res as u64)
}

/// Zero the Metal allocator's peak-memory high-water mark.
///
/// Wraps `mlx_reset_peak_memory`. Returns `false` if the C call reports an
/// error. After a successful reset [`mlx_peak_memory_bytes`] reads `0` until
/// the next allocation, then tracks the live byte count from there — so the
/// reading becomes "the most bytes that were live at once *since the reset*",
/// which is what makes a scoped measurement possible.
///
/// The high-water mark is process-global. Two brackets running concurrently
/// on different threads reset each other; scope one at a time.
pub fn mlx_reset_peak_memory() -> bool {
    install_error_handler();
    // SAFETY: no arguments, no out-params; `mlx_reset_peak_memory` is
    // thread-safe per mlx-c contract.
    let status = unsafe { sys::mlx_reset_peak_memory() };
    status == 0
}

/// A scoped Metal-allocator measurement: reset the peak, run a region, read it.
///
/// MLX pools its device buffers, so an absolute byte count says as much about
/// what ran before as about the region under test. What is comparable is the
/// *delta* across a bracket, which is what this type reports.
///
/// ```ignore
/// let bracket = PeakBracket::open();
/// // ... the region under test, materialised (lazy arrays must be eval'd here) ...
/// let reading = bracket.close();
/// assert!(reading.observed_allocation());       // or the bound below is vacuous
/// assert!(reading.headroom_bytes() <= budget);  // budget scaled to the workload
/// ```
///
/// The high-water mark is process-global — see [`mlx_reset_peak_memory`].
#[derive(Debug, Clone, Copy)]
pub struct PeakBracket {
    live_at_open: u64,
    reset_ok: bool,
}

/// What a [`PeakBracket`] observed. All three raw fields are in bytes.
///
/// Construct one only via [`PeakBracket::close`]; the derived accessors are
/// the intended reading surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub struct PeakReading {
    /// Most bytes live at any one moment inside the bracket. `0` when the
    /// region allocated nothing at all (the reset leaves the mark at zero and
    /// only an allocation raises it).
    pub peak_bytes: u64,
    /// Live bytes when the bracket opened.
    pub live_at_open_bytes: u64,
    /// Live bytes when the bracket closed.
    pub live_at_close_bytes: u64,
    /// Whether the peak mark was successfully zeroed when the bracket opened.
    ///
    /// When this is `false` the peak reading is still process-lifetime, so no
    /// delta computed from it is scoped to anything. Every derived accessor
    /// reports "measured nothing" rather than a plausible number.
    pub reset_ok: bool,
}

impl PeakBracket {
    /// Record the live byte count and zero the peak mark.
    ///
    /// Neither C call is required to succeed — on a build with no Metal
    /// allocator both fail. What matters is that the two failures are tracked
    /// separately: if the *reset* fails while the live reading succeeds, the
    /// peak stays process-lifetime and `peak - live_at_open` would be a large,
    /// stable, entirely meaningless number. [`PeakReading::measurable`] is what
    /// distinguishes that from a real measurement.
    #[must_use]
    pub fn open() -> Self {
        let live_at_open = mlx_active_memory_bytes().unwrap_or(0);
        let reset_ok = mlx_reset_peak_memory();
        Self {
            live_at_open,
            reset_ok,
        }
    }

    /// Read the peak and live counts back.
    #[must_use]
    pub fn close(self) -> PeakReading {
        PeakReading {
            peak_bytes: mlx_peak_memory_bytes().unwrap_or(0),
            live_at_open_bytes: self.live_at_open,
            live_at_close_bytes: mlx_active_memory_bytes().unwrap_or(0),
            reset_ok: self.reset_ok,
        }
    }
}

impl PeakReading {
    /// Whether this reading is scoped to the bracket at all.
    ///
    /// `false` when the peak mark could not be zeroed at `open()`. The peak is
    /// then still process-lifetime, so every delta derived from it describes
    /// the process rather than the region, and all of them report zero instead.
    #[must_use]
    pub const fn measurable(&self) -> bool {
        self.reset_ok
    }

    /// Bytes the region needed *on top of* what was already live when it
    /// opened. This is the number an allocation gate should assert on: it is
    /// independent of the resident weights and of whatever ran before.
    ///
    /// Saturating: a region that allocated nothing reports `0`, and so does a
    /// reading that is not [`measurable`](Self::measurable).
    #[must_use]
    pub const fn headroom_bytes(&self) -> u64 {
        if !self.reset_ok {
            return 0;
        }
        self.peak_bytes.saturating_sub(self.live_at_open_bytes)
    }

    /// Bytes allocated inside the region and released again before it closed.
    ///
    /// Non-zero means the region peaked above what it was still holding at the
    /// end — it allocated something and threw it away.
    ///
    /// **Zero does not mean no scratch buffer existed.** A transient smaller
    /// than the region's own steady-state peak hides underneath it: the peak
    /// is reached by the surviving buffers regardless, so this reads zero.
    /// What it does catch is a transient *larger* than the steady state, which
    /// is the allocation regression worth a gate — and one no numerics test
    /// can see, because the output bits are unchanged.
    #[must_use]
    pub const fn transient_bytes(&self) -> u64 {
        if !self.reset_ok {
            return 0;
        }
        self.peak_bytes.saturating_sub(self.live_at_close_bytes)
    }

    /// `true` when this region's live bytes rose above where they started.
    ///
    /// A gate that asserts an upper bound passes vacuously against a region
    /// that never allocated (or against a build with no Metal allocator at
    /// all); assert this first so the gate cannot pass by measuring nothing.
    ///
    /// This is `headroom_bytes() > 0`, **not** `peak_bytes > 0`. MLX updates
    /// the mark as `peak = max(peak, active)` on every allocation, and `active`
    /// is the whole live count — so after a reset a single one-byte allocation
    /// anywhere in the process lifts `peak_bytes` to gigabytes wherever weights
    /// are resident. `peak_bytes > 0` therefore means "something, somewhere,
    /// allocated since the reset", which in any real process is always true.
    ///
    /// The residual limit: a region that frees before it allocates, and never
    /// climbs back above where it opened, reads `false`. That is accurate —
    /// nothing about this region's peak was observable — but it is not the same
    /// as "no allocation call was made".
    #[must_use]
    pub const fn observed_allocation(&self) -> bool {
        self.headroom_bytes() > 0
    }
}

// ---------------------------------------------------------------------------
// Metal-specific helpers
// ---------------------------------------------------------------------------
//
// Byte-to-byte port of the mlx-lm server.py startup sequence:
//
// if mx.metal.is_available():
// wired_limit = mx.device_info()["max_recommended_working_set_size"]
// mx.set_wired_limit(wired_limit)
//
// Wiring `max_recommended_working_set_size` as the wired limit asks the
// kernel to keep the model's resident pages locked, eliminating page-fault
// stalls during decode. mlx-lm calls this once at server startup; rMLX does
// the same from `crates/rmlx-cli/src/commands/serve.rs`.

/// Metal-specific helpers: availability check, device info, and wired-memory limit.
pub mod metal {
    use super::{check_status, install_error_handler, sys, Error, Result};
    use std::ffi::CString;

    /// Returns true if a Metal-capable GPU backend is available.
    ///
    /// Mirrors `mlx.core.metal.is_available()`.
    pub fn is_available() -> Result<bool> {
        install_error_handler();
        let mut avail = false;
        // SAFETY: writing to a stack `bool` we own.
        let status = unsafe { sys::mlx_metal_is_available(&raw mut avail) };
        unsafe { check_status(status, "metal::is_available") }?;
        Ok(avail)
    }

    /// Returns the size_t value for `key` from MLX's device info dict for
    /// the default device.
    ///
    /// Mirrors `mlx.core.device_info()[key]` for size_t-typed keys
    /// (e.g. `max_recommended_working_set_size`, `memory_size`).
    #[allow(
        clippy::unwrap_used,
        reason = "check_status returns Err when status != 0; .unwrap_err() is infallible because the guard `status != 0` ensures the Result is Err"
    )]
    pub fn device_info_size(key: &str) -> Result<usize> {
        install_error_handler();
        let key_c = CString::new(key).map_err(|e| Error::Mlx(format!("device_info key: {e}")))?;

        // Look up the default device.
        let mut dev = unsafe { sys::mlx_device_new() };
        // SAFETY: `dev` is a valid empty device handle just created.
        let status = unsafe { sys::mlx_get_default_device(&raw mut dev) };
        if status != 0 {
            // Free dev before returning.
            unsafe { sys::mlx_device_free(dev) };
            return Err(unsafe {
                check_status(status, "metal::device_info_size: get_default_device")
            }
            .unwrap_err());
        }

        // Fetch the device info struct.
        let mut info = unsafe { sys::mlx_device_info_new() };
        // SAFETY: `info` and `dev` are both valid handles.
        let status = unsafe { sys::mlx_device_info_get(&raw mut info, dev) };
        if status != 0 {
            unsafe { sys::mlx_device_info_free(info) };
            unsafe { sys::mlx_device_free(dev) };
            return Err(unsafe {
                check_status(status, "metal::device_info_size: device_info_get")
            }
            .unwrap_err());
        }

        // Read the size_t-typed value for `key`.
        let mut value: usize = 0;
        // SAFETY: `info` is valid; `key_c.as_ptr()` lives until the end of this fn.
        let status = unsafe { sys::mlx_device_info_get_size(&raw mut value, info, key_c.as_ptr()) };
        // Status: 0 = ok, 1 = error, 2 = key missing or wrong type.
        let result = if status == 0 {
            Ok(value)
        } else if status == 2 {
            Err(Error::Mlx(format!(
                "metal::device_info_size: key '{key}' not found or not a size_t"
            )))
        } else {
            Err(unsafe { check_status(status, "metal::device_info_size: get_size") }.unwrap_err())
        };

        // SAFETY: both handles are valid; freeing each exactly once.
        unsafe { sys::mlx_device_info_free(info) };
        unsafe { sys::mlx_device_free(dev) };
        result
    }

    /// Set the GPU wired-memory limit. Returns the previous limit.
    ///
    /// Mirrors `mlx.core.set_wired_limit(limit) -> int`.
    pub fn set_wired_limit(limit: usize) -> Result<usize> {
        install_error_handler();
        let mut old: usize = 0;
        // SAFETY: writing to a stack `usize` we own.
        let status = unsafe { sys::mlx_set_wired_limit(&raw mut old, limit) };
        unsafe { check_status(status, "metal::set_wired_limit") }?;
        Ok(old)
    }

    /// One-shot startup helper: byte-to-byte port of mlx-lm's
    /// `server.py` startup sequence.
    ///
    /// ```python
    /// if mx.metal.is_available():
    /// wired_limit = mx.device_info()["max_recommended_working_set_size"]
    /// mx.set_wired_limit(wired_limit)
    /// ```
    ///
    /// On non-Metal backends (e.g. CPU-only build), returns `Ok(None)` and
    /// does nothing.
    pub fn set_wired_limit_to_recommended() -> Result<Option<(usize, usize)>> {
        if !is_available()? {
            return Ok(None);
        }
        let recommended = device_info_size("max_recommended_working_set_size")?;
        let old = set_wired_limit(recommended)?;
        Ok(Some((recommended, old)))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod lib_tests;
