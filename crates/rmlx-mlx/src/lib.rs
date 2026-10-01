// LOC-exempt: crate root of the mlx-c wrapper. `Array` — the owned handle every
// other module in the workspace passes around — is inseparable from the
// crate-private FFI plumbing it calls on every op: the thread-local error
// capture behind `check_status`, the stream helper, the
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

/// The mlx-c C API this crate compiled against, and the one the process loaded.
mod c_api;
pub mod compile;
/// Bounded-window Metal GPU trace capture. Debug-only: compiled out entirely
/// unless the `metal-capture` feature is enabled, so a release build carries no
/// state, no decode-path branch, and no route to `MTLCaptureManager`.
#[cfg(feature = "metal-capture")]
pub mod metal_capture;
pub mod metal_kernel;
/// The one thread that owns the streams every op is built on, and evaluates.
mod mlx_thread;
mod nax;
/// Whether the MLX this process loaded is the pair rMLX is validated against.
mod pin;
mod sys;
/// Parser for `xcrun xctrace export` XML — the headless route to GPU wall-clock
/// timing and the CPU→GPU gap. Debug-only, behind the same feature as
/// `metal_capture`: it is profiling tooling, not part of the served binary.
#[cfg(feature = "metal-capture")]
pub mod xctrace;

use std::cell::Cell;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

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

// ---------------------------------------------------------------------------
// Evaluation serialisation
// ---------------------------------------------------------------------------
//
// This crate funnels every MLX evaluation through one process-wide lock, on
// one thread: `with_eval_lock` runs the call on the MLX thread
// (`mlx_thread.rs`), and takes the lock there.
//
// MLX finds the command encoder of a stream only in the thread that created
// the stream (`mlx/backend/cpu/encoder.cpp` from MLX 0.32, the Metal device on
// every MLX), so every op is built on a stream of the MLX thread and every
// evaluation runs there. The lock is kept as a second guard: it makes serial
// evaluation a property of this crate even if evaluation leaves that thread,
// and the gates below keep every evaluating FFI call under it.
//
// Cost is one uncontended mutex acquire + release per evaluation, plus the
// hand-off to the MLX thread when the caller is another thread.
//
// **Which C entry points need it.** Not just the eval-named ones: every mlx-c
// function that reaches `mlx::core::eval_impl` evaluates. The set is
// **25** — 24 found by reverse reachability over the linked dylibs, plus
// `mlx_closure_apply`, which that automated pass structurally cannot see
// because the call goes through a `std::function` vtable. Re-running the
// procedure at a pin bump yields 24 and no closure entry; that is the pass's
// blind spot, not a stale guard. `scripts/check_eval_lock.sh` records both
// passes.
//
//   - `mlx_array_eval`, `mlx_async_eval`, `mlx_eval`
//   - 14 × `mlx_array_item_*` — `array::item<T>()` calls `eval()` first
//     (`mlx/array.h`)
//   - `mlx_array_tostring` — `operator<<(ostream&, array)` evaluates
//   - `mlx_save`, `mlx_save_writer`, `mlx_save_safetensors`,
//     `mlx_save_safetensors_writer`, `mlx_save_gguf`, `mlx_load_gguf` — the
//     serialisation paths materialise before writing
//   - `mlx_closure_apply` — indirect: building the fused `Compiled` primitive
//     bakes scalar constants into the kernel library name via
//     `print_constant` → `array::item<T>()` → `array::eval()`
//     (`mlx/backend/common/compiled.cpp`)
//
// The data accessors `mlx_array_data_*` do *not* evaluate and need no lock.
// Three of the 25 are called here and all three are guarded; the other 22 are
// one call away, and adding one unguarded makes evaluation concurrent again —
// `mlx_save_safetensors` is the write side of `rmlx convert`, and
// `mlx_array_tostring` is what an `impl Debug for Array` would reach for.
// `make check-eval-lock` fails the build on that, because a doc sentence is
// not a gate.
//
// **Re-entrancy.** The mutex is not reentrant, so nothing running under it may
// *take the lock again* — a broader ban than "must not evaluate", because
// `Closure::apply` takes it too, so applying one compiled closure from inside
// another's body deadlocks without calling `eval` anywhere. That is only a live
// question for `mlx_closure_apply`, which invokes a Rust closure body on the
// MLX thread while the lock is held. No body does it today and the gate's
// RULE 3 enforces it; see `Closure::apply`.
static EVAL_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Run `f` on the MLX thread, holding the process-wide evaluation lock.
///
/// Takes a closure rather than returning the guard on purpose: a
/// `MutexGuard`-returning helper is only correct if every caller binds it to a
/// named local, and `let _ = acquire();` — which drops the guard *before* the
/// FFI call and leaves that call unserialised — compiles silently, because `#[must_use]`
/// does not fire on `let _ =`. This signature makes "held across the FFI call
/// and nothing else" a property of the API instead of a rule callers have to
/// remember.
///
/// Poisoning is unreachable here, though not because the guarded region is
/// Rust-free — it is not. Two different bits of Rust run under this lock, and
/// each is contained by a different mechanism:
///
/// - The error handler installed by [`install_error_handler`], which MLX calls
///   from *inside* `mlx_array_eval` / `mlx_async_eval`, allocating a `String`
///   and writing a thread-local. It is an `unsafe extern "C" fn`, and since
///   Rust 1.81 a panic escaping one aborts the process rather than unwinding.
/// - A compiled closure body, run from inside `mlx_closure_apply`. The abort
///   rule does *not* apply to it, because `rust_closure_callback` wraps the
///   call in `catch_unwind` and converts a panic into a non-zero return — so
///   no unwind ever crosses the `MutexGuard` from that direction either.
///
/// Recovering from `PoisonError` rather than propagating it keeps that
/// unreachable state from turning a later evaluation into a spurious failure.
///
/// The mlx-c error message that `f` leaves on the MLX thread moves to the
/// calling thread, so `check_status` after this call reads it.
///
/// # Errors
/// An error when the MLX thread cannot start; `f` did not run.
pub(crate) fn with_eval_lock<T: Send>(f: impl FnOnce() -> T + Send) -> Result<T> {
    mlx_thread::run(|| under_eval_lock(f))
}

/// Run `f` on the calling thread, holding the process-wide evaluation lock.
fn under_eval_lock<T>(f: impl FnOnce() -> T) -> T {
    let _guard = EVAL_LOCK
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    f()
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
        c_api::report_at_init();
        warn_on_mlx_version_skew();
        nax::warn_if_nax_kernels_missing();
    });
}

/// The MLX version `rmlx-mlx` was compiled against (from `mlx/version.h`,
/// captured by `build.rs`). `"unknown"` when the header was unreadable.
const MLX_BUILD_VERSION: &str = env!("RMLX_MLX_BUILD_VERSION");

/// Whether the MLX **this process loaded** ships the `steel_gemm_fused_nax*`
/// GEMM kernels — `"present"` / `"absent"` / `"unknown"`.
///
/// Read off the `mlx.metallib` beside the `libmlx.dylib` dyld resolved, once
/// per process. A binary links MLX through a package-manager symlink, so it
/// runs against whatever the installing user has — a different bottle of the
/// same version can differ in exactly this. Anything derived at build time
/// describes the build machine, which is the wrong machine.
///
/// `pub` (unlike `MLX_BUILD_VERSION`): this is the one fact about the loaded
/// MLX that a downstream binary cannot derive itself, and the only channel it
/// has out of this crate. `rmlx-metrics` deliberately does not depend on
/// `rmlx-mlx` (its build script hard-requires a working Homebrew MLX/mlx-c
/// install, which `rmlx-metrics` must not need), so `rmlx-cli::main()` — the
/// one binary that links both — calls this and forwards it to
/// `rmlx_metrics::identity::set_mlx_nax` once at startup, before any metrics
/// recording. See `docs/METRICS_SCHEMA.md` §3.6.
#[must_use]
pub fn nax_capability() -> &'static str {
    nax::loaded_nax_capability()
}

pub use pin::{pin_check, PinCheck, PinEnforcement, PinRefusal};

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

/// The version of the MLX this process loaded, as `(major, minor, patch)`.
///
/// `None` when mlx-c cannot report it or it does not parse. A test that pins
/// numerics which changed between MLX releases keys its expectation on this.
#[must_use]
pub fn loaded_mlx_version() -> Option<(u32, u32, u32)> {
    install_error_handler();
    parse_mlx_version(&runtime_mlx_version()?)
}

/// `"0.32.3"` or `"0.32.3.dev20260901+abc"` -> `(0, 32, 3)`.
fn parse_mlx_version(s: &str) -> Option<(u32, u32, u32)> {
    let core = s.split(['-', '+']).next()?;
    let mut parts = core.split('.').map(|p| p.parse::<u32>().ok());
    Some((parts.next()??, parts.next()??, parts.next()??))
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
    let Some(built) = parse_mlx_version(MLX_BUILD_VERSION) else {
        return;
    };
    let Some(runtime) = runtime_mlx_version() else {
        return;
    };
    if parse_mlx_version(&runtime) != Some(built) {
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
// The CPU-device latch: once set, nothing in this crate asks for a GPU stream
// or calls a Metal API.
// ---------------------------------------------------------------------------

/// Set once, before any model work starts, by the process that decided on the
/// CPU device; never cleared. Worker threads start after it is set, so a
/// `Relaxed` load observes it.
static GPU_FORBIDDEN: AtomicBool = AtomicBool::new(false);

/// Refuse every later GPU stream and Metal API call in this process with
/// [`Error::GpuForbidden`], and make the CPU the MLX default device, so that
/// an op MLX builds inside another op goes to a CPU stream too.
///
/// Call it once, from the place that makes the process's device decision,
/// before any model work. There is no way back.
///
/// # Errors
/// An error when MLX does not accept the CPU as its default device.
pub fn forbid_gpu() -> Result<()> {
    install_error_handler();
    GPU_FORBIDDEN.store(true, Ordering::Relaxed);
    // SAFETY: a CPU device value that this function owns and frees below.
    let cpu = unsafe { sys::mlx_device_new_type(sys::mlx_device_type::MLX_CPU, 0) };
    // SAFETY: `cpu` is valid. MLX copies it into its process-wide default.
    let status = unsafe { sys::mlx_set_default_device(cpu) };
    // SAFETY: called immediately after the C function on this thread.
    let set = unsafe { check_status(status, "mlx_set_default_device") };
    // SAFETY: `cpu` is not used after this call.
    unsafe { sys::mlx_device_free(cpu) };
    set?;
    tracing::info!("GPU refused for this process: the device decision is cpu");
    Ok(())
}

/// True once [`forbid_gpu`] has run in this process.
#[must_use]
pub fn gpu_forbidden() -> bool {
    GPU_FORBIDDEN.load(Ordering::Relaxed)
}

/// `Err(Error::GpuForbidden)` naming `op` once [`forbid_gpu`] has run.
fn check_gpu_allowed(op: &'static str) -> Result<()> {
    if gpu_forbidden() {
        return Err(Error::GpuForbidden { op });
    }
    Ok(())
}

/// Call `f` with the stream for `device` that every op is built on: a stream
/// of the MLX thread (`mlx_thread.rs`). An op built on it evaluates from any
/// thread, because every evaluation runs on the MLX thread. Never create a
/// stream per op: MLX keeps a worker thread for each CPU stream for the life
/// of the process, so `pthread_create` fails after a few thousand ops.
///
/// # Errors
/// [`Error::GpuForbidden`] for the GPU device after [`forbid_gpu`]; `f` is not
/// called and no stream is requested. An error when the MLX thread cannot
/// start.
pub(crate) fn with_stream<T>(device: Device, f: impl FnOnce(sys::mlx_stream) -> T) -> Result<T> {
    if device == Device::Gpu {
        check_gpu_allowed("GPU stream")?;
    }
    Ok(f(mlx_thread::stream(device)?))
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
// Null-handle sentinel for optional FFI arguments
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

/// Element dtype subset.
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
    #[must_use]
    pub const fn itemsize(self) -> usize {
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
    ///
    /// Serialised process-wide against every other evaluation — see
    /// `EVAL_LOCK`.
    pub fn eval(&self) -> Result<()> {
        install_error_handler();
        // SAFETY: inner is a valid mlx_array, and `self` keeps it alive until
        // the evaluation returns.
        let status = with_eval_lock(|| unsafe { sys::mlx_array_eval(self.inner) })?;
        // SAFETY: `with_eval_lock` moved the error message of the call to this
        // thread's error slot, and no mlx-c call ran since.
        unsafe { check_status(status, "Array::eval") }
    }

    /// Whether this array's data is materialised, so reading it would not
    /// first run a graph.
    ///
    /// False while the array is still an unevaluated node — the work it stands
    /// for has been described and not yet paid for, and whoever blocks on it
    /// next is who pays. That is the question a wall-clock span around a call
    /// site cannot answer about itself, and the reason this is exposed at all.
    ///
    /// It does not block and does not schedule: an array left mid-flight by
    /// [`Self::async_eval`] reads false until its event fires.
    pub fn is_available(&self) -> Result<bool> {
        install_error_handler();
        let mut available = false;
        // SAFETY: inner is a valid mlx_array; the out-pointer is a stack
        // `bool` we own. This reads the array's status and evaluates nothing,
        // so it is outside the evaluation lock's remit.
        let status = unsafe { sys::_mlx_array_is_available(&raw mut available, self.inner) };
        // SAFETY: called immediately after the C function on the same thread,
        // whose thread-local error slot it reads.
        unsafe { check_status(status, "Array::is_available") }?;
        Ok(available)
    }

    /// Evaluate this array, then return the address of its first element.
    ///
    /// For identity checks only: an update that MLX did in place gives its
    /// result the address of its input, and a copy gives a new address. Never
    /// read through it.
    ///
    /// # Errors
    /// An error when the evaluation fails.
    pub fn data_address(&self) -> Result<usize> {
        self.eval()?;
        // SAFETY: `self.inner` is a valid mlx_array, and the evaluation above
        // gave it a buffer, which MLX reads to make the pointer. The pointer
        // is not dereferenced here.
        let ptr = unsafe { sys::mlx_array_data_uint8(self.inner) };
        Ok(ptr.addr())
    }

    /// Schedule this array's compute graph without blocking the calling
    /// thread. The work runs in the background; a later `to_bytes` or `eval`
    /// waits for it.
    ///
    /// The decode loop uses it to queue the next forward pass while the
    /// current argmax is read back (as `mx.async_eval` in mlx-lm's
    /// `generate.py`).
    ///
    /// Serialised process-wide against every other evaluation — see
    /// `EVAL_LOCK`. Only the graph walk and dispatch run under the lock; the
    /// scheduled work completes after it is released, so the pipelining this
    /// exists for is kept.
    ///
    /// # Errors
    /// An error when MLX cannot schedule the graph.
    pub fn async_eval(&self) -> Result<()> {
        install_error_handler();
        let status = with_eval_lock(|| {
            // SAFETY: `self.inner` is a valid mlx_array that `self` keeps alive
            // until this returns. The vector is freed before the job ends.
            unsafe {
                let vec = sys::mlx_vector_array_new_value(self.inner);
                let status = sys::mlx_async_eval(vec);
                sys::mlx_vector_array_free(vec);
                status
            }
        })?;
        // SAFETY: `with_eval_lock` moved the error message of the failed call to
        // this thread's error slot, and no mlx-c call ran since.
        unsafe { check_status(status, "Array::async_eval") }
    }

    /// Copy the array's logical elements, row-major, into a fresh `Vec<u8>`.
    ///
    /// Forces evaluation first so the read is always of materialized data.
    /// MLX is lazy and `async_eval` only *schedules* compute; the data pointer
    /// returned by `mlx_array_data_uint8` is not guaranteed to hold the result
    /// until the graph has actually run. Calling `eval()` here blocks until it
    /// has — idempotent and near-free on an already-evaluated array. Without
    /// it, reading the pointer can race the asynchronous evaluation and return
    /// stale/recycled buffer bytes (another array's data) rather than this
    /// array's value.
    ///
    /// Evaluation is not enough on its own: a transpose, a strided slice and a
    /// broadcast all evaluate to the *parent's* allocation with adjusted
    /// strides, and a linear read of one returns the parent's leading elements
    /// under the view's shape — right length, right dtype, wrong values. Such
    /// an input is relaid out row-major before the read, so the returned bytes
    /// always mean what the array's shape says. The relayout costs a copy, and
    /// only on inputs whose linear read would otherwise be wrong; a dense array
    /// — every reduction, elementwise, matmul and kernel output — pays one
    /// layout-flag read on top of what it paid before.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        install_error_handler();
        self.eval()?;
        let nbytes = unsafe { sys::mlx_array_nbytes(self.inner) };
        if nbytes == 0 {
            return Ok(Vec::new());
        }
        if self.is_row_contiguous()? {
            return self.copy_row_major_bytes(nbytes);
        }
        // The bytes are bound for host memory and the caller is already
        // blocking on them, so relaying out on the CPU stream keeps the read
        // path free of a GPU dispatch to wait on. Reading an evaluated array's
        // buffer from the host is what this method does either way.
        tracing::debug!(
            shape = ?self.shape(),
            nbytes,
            "to_bytes: relaying out a non-row-contiguous array"
        );
        let dense = self.contiguous(Device::Cpu)?;
        dense.eval()?;
        dense.copy_row_major_bytes(nbytes)
    }

    /// Copy `nbytes` straight off the data pointer.
    ///
    /// Callers must have evaluated the array and established that it is
    /// row-contiguous, and must pass this array's own `mlx_array_nbytes`;
    /// otherwise the read follows the parent allocation's layout instead of
    /// this array's, and for a broadcast it runs past the end of that
    /// allocation.
    fn copy_row_major_bytes(&self, nbytes: usize) -> Result<Vec<u8>> {
        // SAFETY: mlx_array_data_uint8 returns a raw pointer valid while the
        // array is alive and evaluated. We copy out immediately.
        let ptr = unsafe { sys::mlx_array_data_uint8(self.inner) };
        if ptr.is_null() {
            return Err(Error::Mlx(
                "Array::to_bytes: data pointer is null after eval".into(),
            ));
        }
        // SAFETY: ptr is non-null and points to `nbytes` contiguous bytes
        // owned by the mlx array. Copied into Vec before returning.
        let bytes = unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec();
        Ok(bytes)
    }

    /// Whether the elements are laid out row-major and dense from the data
    /// pointer — i.e. whether reading `nbytes` linearly off that pointer yields
    /// the logical elements in logical order.
    ///
    /// False for a transpose, a strided slice, and a broadcast: MLX gives each
    /// of those the parent's allocation with adjusted strides rather than a
    /// copy, and `eval` materialises the graph without relaying anything out.
    /// A broadcast is the sharpest case — its logical size exceeds the elements
    /// the parent owns, so a linear read of `nbytes` runs off the buffer.
    ///
    /// **Evaluate first.** The flag describes a materialised buffer, and MLX
    /// only sets it when one is attached; on an unevaluated array it reports
    /// the layout of nothing and answers `true` for a view that is not dense.
    ///
    /// `mlx/c/array.h` declares `_mlx_array_is_row_contiguous` an internal
    /// function with no stability promise, so every host readback now rests on
    /// an mlx-c internal. The behaviour pin is the layout-flag test in
    /// `lib_tests.rs`; re-run it whenever the pinned mlx / mlx-c pair moves.
    fn is_row_contiguous(&self) -> Result<bool> {
        install_error_handler();
        let mut res = false;
        // SAFETY: inner is a valid mlx_array. The call reads the array's layout
        // flags and never touches the data buffer, so it cannot fault on an
        // unevaluated array — it just answers about a buffer that is not there.
        let status = unsafe { sys::_mlx_array_is_row_contiguous(&raw mut res, self.inner) };
        // SAFETY: called immediately after the C function on the same thread.
        unsafe { check_status(status, "Array::is_row_contiguous") }?;
        Ok(res)
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
        }?;
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
        }?;
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
        }?;
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
    /// matches the logical shape. The case that needs it is a transpose or a
    /// strided slice fed straight to the kernel: a `reshape` that changes the
    /// shape already relayouts, because MLX copies a non-row-contiguous
    /// reshape input dense instead of sharing its buffer. A reshape to the
    /// shape the array already has returns the array itself and relayouts
    /// nothing.
    pub fn contiguous(&self, device: Device) -> Result<Array> {
        install_error_handler();
        let mut res = unsafe { sys::mlx_array_new() };
        let status = unsafe {
            with_stream(device, |s| {
                // allow_col_major = false → force row-major layout.
                sys::mlx_contiguous(&raw mut res, self.inner, false, s)
            })
        }?;
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
        }?;
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
        }?;
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
        }?;
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
/// Wraps mlx-c `mlx_get_peak_memory`. Returns `None` if the C
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
///
/// Every function here returns [`Error::GpuForbidden`] after
/// [`super::forbid_gpu`], before any FFI call.
pub mod metal {
    use super::{check_gpu_allowed, check_status, install_error_handler, sys, Error, Result};
    use std::ffi::CString;

    /// Returns true if a Metal-capable GPU backend is available.
    ///
    /// Mirrors `mlx.core.metal.is_available()`.
    pub fn is_available() -> Result<bool> {
        check_gpu_allowed("metal::is_available")?;
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
        check_gpu_allowed("metal::device_info_size")?;
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
        check_gpu_allowed("metal::set_wired_limit")?;
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
        check_gpu_allowed("metal::set_wired_limit_to_recommended")?;
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

#[cfg(test)]
mod gpu_latch_tests;

#[cfg(test)]
#[path = "../tests/common/within_limit.rs"]
mod within_limit;
