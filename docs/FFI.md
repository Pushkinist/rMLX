# FFI Bridge Reference

rMLX ↔ mlx-c FFI bridge: how Rust talks to MLX without any Python at
runtime.

---

## Overview

`mlx-c` is Apple's stable C ABI layer over the MLX C++ library. It exposes
a pure-C interface (`mlx_array`, `mlx_closure`, `mlx_fast_metal_kernel`, …)
that avoids the fragile ABI of C++, making it safe to link from Rust via
`bindgen`-generated bindings.

The `rmlx-mlx` crate is the only crate in the workspace that touches mlx-c
directly. All other crates (`rmlx-models`, `rmlx-quant`, `rmlx-loader`,
`rmlx-server`, `rmlx-cli`) call through the public API of `rmlx-mlx` and
never see an `unsafe` block related to mlx-c. `rmlx-models` ships MSL byte
blobs that are registered and dispatched through `rmlx-mlx`'s
`metal_kernel` module.

The crate does **not** depend on `mlx-rs` (the community Rust binding). It
drives mlx-c directly to control every detail of the FFI contract.

---

## mlx-c Contract

### Versioned ABI

mlx-c pins a specific MLX version. The build links against:

- `libmlxc.dylib` — mlx-c 0.6.0, default prefix
  `/opt/homebrew/Cellar/mlx-c/0.6.0_2/lib`.
- `libmlx.dylib` — MLX 0.31.2, default prefix
  `/opt/homebrew/Cellar/mlx/0.31.2/lib`.

Both prefixes are overridable via `MLX_C_PREFIX` / `MLX_PREFIX` env vars.
`build.rs` asserts the dylibs exist and aborts the build with an actionable
message if they are absent.

### Build pipeline (`build.rs`)

1. Target guard: `aarch64-apple-darwin` only. The assertion fires at compile
   time for any other target.
2. `bindgen` runs against `wrapper.h`, producing `$OUT_DIR/bindings.rs`. Only
   `mlx_*` symbols are allowlisted; `mlx_dtype_` and `mlx_device_type_` are
   emitted as Rust enums.
3. `build.rs` post-processes `bindings.rs` to strip any `#![...]` inner
   attributes bindgen 0.71 emits at the file head — these are illegal inside
   the `include!()` call site in `sys.rs`.
4. Rebuild is triggered on `MLX_C_PREFIX`, `MLX_PREFIX`, or `wrapper.h`
   changes.
5. rpath entries are emitted so the binary finds both dylibs at runtime
   without `DYLD_LIBRARY_PATH`.

### `sys.rs` — raw bindings

`sys.rs` wraps the generated `bindings.rs` inside `pub(crate) mod ffi { … }`
with blanket lint suppression, then re-exports everything via
`pub(crate) use ffi::*`. The suppressed lints cover naming convention
violations and dead-code warnings that are inherent in generated C bindings.
No public symbol from `sys.rs` escapes the crate.

### Error delivery

mlx-c delivers runtime errors via a registered C callback rather than return
values alone. `lib.rs` installs a thread-local handler exactly once per
process via `std::sync::Once`:

```rust
// SAFETY: mlx_set_error_handler is thread-safe. The callback receives
// a valid NUL-terminated *const c_char for the duration of the call.
unsafe extern "C" fn handler(msg: *const c_char, _data: *mut c_void) { … }
```

The handler writes the message into `LAST_ERROR: thread_local! { Cell<Option<String>> }`.
`Cell` is used instead of `RefCell` because the only access pattern is
`take()` and `set()`, which need no shared borrows and avoid the runtime
borrow-count overhead of `RefCell`.

After every mlx-c call, `check_status(status, context)` retrieves the stored
message and returns `Err(Error::Mlx(...))` on any non-zero status code.
`check_status` must be called immediately on the same thread, before any
other mlx-c call could overwrite the slot.

### Default stream

Every op that requires a stream calls `with_stream(device, |s| …)`, which
borrows the process-global default stream and releases the handle after the
closure returns.

An earlier implementation called `mlx_stream_new_device` per op, which
spawns a new OS thread inside MLX on each call. A single Gemma4 forward
pass dispatches hundreds of ops per decode step; after 3–6 steps the macOS
thread limit (~2 048) was exhausted and `pthread_create` returned `EAGAIN`.
The fix uses `mlx_default_gpu_stream_new` / `mlx_default_cpu_stream_new`,
which return a reference-counted handle to the already-running default
stream. Freeing the handle decrements the ref-count; the stream and its
thread are never torn down.

### Null sentinel for optional arguments

Several mlx-c functions accept optional `mlx_array` arguments. When the
argument is absent the correct idiom is a default-constructed handle with
`ctx = null`. The naive approach allocates a fresh handle per call via
`mlx_array_new()` and frees it after — one heap allocation and one atomic
ref-count free per op per layer per decode step.

`rmlx-mlx` keeps one process-global null sentinel in
`EMPTY_ARRAY_SENTINEL: OnceLock<Array>`. The raw inner handle is retrieved
via `null_sentinel()` and passed directly to the C function. The sentinel is
never freed and must never be passed to any function that will free it or
materialize it as a real array. This pattern is used for:

- `weight` in `rms_norm` when `None` (Gemma4 `v_norm` layers).
- `freqs` in `rope` / `rope_dynamic` when base-only computation is wanted.
- `biases` in `quantized_matmul`, `dequantize`, `gather_qmm` when `None`
  (mxfp8/mxfp4 modes do not use bias).
- `lhs_indices` in `gather_qmm` when `None`.
- `mask` and `sinks` in `scaled_dot_product_attention` when `None`.
- `global_scale` in `dequantize` and `quantize` (unused for affine/mxfp8).

### Mode string cache

mlx-c functions that accept a mode string (`"affine"`, `"mxfp8"`,
`"causal"`, etc.) receive a `*const c_char`. Converting a `&str` to
`CString` per call allocates on the heap. The set of legal mode strings is
fixed, so each is cached in a `OnceLock<CString>` and returned as a
`Cow<'static, CStr>`. Unknown strings fall through to a dynamic
`CString::new` for forward-compatibility. Callers use `.as_ptr()` on the
result.

---

## Array Lifetime and Ownership

### Type

`Array` is a newtype around `sys::mlx_array`, which wraps a C++
`std::shared_ptr<mlx::core::array>` under the hood. mlx-c reference-counts
arrays; every call to `mlx_array_new`, `mlx_array_new_data`,
`mlx_array_new_float`, or `mlx_vector_array_get` increments the count.

```rust
pub struct Array {
    inner: sys::mlx_array,
}
```

`Array` implements `Send + Sync`:

```rust
// SAFETY: mlx_array wraps a std::shared_ptr<mlx::core::array>. The
// mlx-c docs state that arrays are reference-counted and thread-safe
// to pass (same semantics as shared_ptr).
unsafe impl Send for Array {}
unsafe impl Sync for Array {}
```

### Drop

`Drop` calls `mlx_array_free`, which decrements the ref-count. The handle
is never aliased inside Rust — each `Array` value owns exactly one handle.
A `trace!` event that was previously fired on drop was removed: it fired
millions of times per decode session and was the dominant overhead in
`RUST_LOG=trace` mode, reducing decode throughput from ~35 TPS to ~17 TPS.

### Construction

| Method | mlx-c call | Ownership |
|--------|-----------|-----------|
| `Array::from_bytes(data, shape, dtype)` | `mlx_array_new_data` | MLX copies the buffer; `data` need not outlive the call. |
| `Array::from_safetensor_view(view)` | `mlx_array_new_data` | Same: copy-on-construct from the mmap view. |
| `scalar_f32(v)` | `mlx_array_new_float` | MLX owns the scalar. |
| `Array::try_clone(&self)` | `mlx_array_set` | Increments ref-count; cheap — no data duplication. |

`Array::from_bytes` validates that `data.len() == product(shape) * dtype.itemsize()`
before calling into C:

```rust
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
```

### Evaluation (lazy graph)

MLX is lazy: ops build a compute graph but do not execute immediately. Two
methods trigger execution:

- `Array::eval()` — synchronous; wraps `mlx_array_eval` and blocks until
  the result is materialized on the GPU.
- `Array::async_eval()` — schedules the array's compute graph on the GPU
  stream without blocking the caller. Implemented by wrapping the array in
  an `mlx_vector_array` and calling `mlx_async_eval`. A subsequent
  `to_bytes()` or `eval()` call will wait if the GPU has not finished. Used
  to pipeline the next decode step's forward pass on the GPU while the
  current step's argmax is still being read back to the CPU, mirroring
  mlx-lm's `mx.async_eval` pattern in `generate.py`.

### Data readback

`Array::to_bytes()` forces evaluation (`eval()`) before reading, then calls
`mlx_array_data_uint8` to obtain a raw pointer to the materialized buffer and
copies into a `Vec<u8>`. Because it materializes internally, callers do not
need a preceding `eval()` / `async_eval()` for correctness — an upstream
`async_eval()` stays a pipelining hint, not a correctness prerequisite (the
data pointer is not guaranteed valid until evaluation has actually run). The
pointer is valid only for the lifetime of the `Array` object; the method
returns `Err` if the pointer is null.

```rust
// SAFETY: ptr is non-null and points to `nbytes` contiguous bytes owned
// by the mlx array. Copied into Vec before returning.
let bytes = unsafe { std::slice::from_raw_parts(ptr, nbytes) }.to_vec();
```

---

## Core Ops

Every op in `ops.rs` and `fast_ops.rs` follows the same pattern:

1. Call `install_error_handler()` (idempotent; `Once`-guarded).
2. Allocate an output handle with `mlx_array_new()`.
3. Call the C function inside `with_stream(device, |s| …)`, writing into the
   output handle via `&raw mut res`.
4. Call `check_status(status, "op_name")` to convert non-zero returns to
   `Error::Mlx`.
5. Return `Ok(Array { inner: res })`.

### `matmul`

Standard dense matrix multiplication. Wraps `mlx_matmul`. Batch-compatible.

### `quantized_matmul`

Integer-affine or OCP mxfp8 quantized matmul. Wraps `mlx_quantized_matmul`.

- `w` is bit-packed (U32 for affine/mxfp8).
- `scales` carries per-group scale factors.
- `biases` is `None` for mxfp8/mxfp4; present for integer affine.
- `mode` must match the quantization codec: `"affine"` or `"mxfp8"`.
  MLX 0.31+ rejects the legacy `"default"` string.
- `transpose_w = true` is the common case for linear layers (`w` is
  `[out_features, packed_in_features]`; matmul computes `x @ w.T`).
- `group_size` and `bits` are passed as `mlx_optional_int` structs.

### `dequantize`

Reconstructs a floating-point tensor from a quantized triple `(codes, scales,
biases)`. Wraps `mlx_dequantize`. Used for the on-device embedding lookup
path — avoids the `device → host → device` round-trip that an `eye(seq) @ w`
workaround required.

### `quantize`

Quantizes a floating-point tensor using MLX's affine integer-affine codec.
Returns the canonical triple `(codes_u32, scales, biases)` that MLX uses for
affine-quantized tensors. Wraps `mlx_quantize`, which returns its result as
an `mlx_vector_array`; the wrapper extracts the three handles and wraps them
in `Array`.

### `gather_qmm`

Batched MoE expert dispatch: computes `x[lhs_indices] @ W[rhs_indices].T`
with quantized weights. Wraps `mlx_gather_qmm`. Both `lhs_indices` and
`biases` are optional; the null sentinel is used when absent.

### `Array::slice` / `Array::slice_update`

`slice` extracts a sub-tensor via per-axis `start:stop:stride` ranges.
`slice_update` returns a copy of `self` with one slice replaced by `update`.
Both wrap the corresponding `mlx_slice` / `mlx_slice_update` C functions.
`slice_update` is the preferred write path for pre-allocated KV cache buffers
inside compiled graphs — MLX may reuse the underlying buffer when the
original is no longer live, making it cheaper than `concat`.

### `Array::take` / `take_along_axis`

`take` wraps `mlx_take_axis` — equivalent to `np.take`. `take_along_axis`
wraps `mlx_take_along_axis` — equivalent to `np.take_along_axis`.

### `scatter_add`

Wraps `mlx_scatter_add_axis`. Performs `out[indices] += values` along axis 0.
Used for sparse accumulation in MoE expert output aggregation.

---

## Fast Ops (Fused Metal Kernels)

`fast_ops.rs` wraps the `mlx_fast_*` family, which bypass the elementwise
graph and dispatch directly to optimised Metal kernels.

### `rms_norm`

Fused RMSNorm: `x / sqrt(mean(x^2) + eps) * weight`. Wraps
`mlx_fast_rms_norm`. `weight` is `None` for `RMSNormNoScale` layers (Gemma4
`v_norm`); the null sentinel is passed when weight is absent.

### `rope` / `rope_dynamic` / `rope_with_freqs` / `rope_with_freqs_dynamic`

Four variants of Rotary Position Embedding, all wrapping `mlx_fast_rope` or
`mlx_fast_rope_dynamic`:

| Variant | Offset type | Frequency source |
|---------|-------------|-----------------|
| `rope` | `i32` (captured) | Base theta (computed by kernel) |
| `rope_dynamic` | `Array` (0-D i32 scalar) | Base theta |
| `rope_with_freqs` | `i32` | Explicit `[dims/2]` freq table |
| `rope_with_freqs_dynamic` | `Array` (0-D i32 scalar) | Explicit freq table |

`rope_dynamic` is required inside `mx.compile` closures: a captured `i32`
offset forces a retrace on every step (new unique literal in the graph),
whereas an `Array`-valued offset flows through the compiled graph as a runtime
operand, enabling a single compiled program across all decode steps.

`rope_with_freqs` is used for Gemma4 full-attention layers that apply
ProportionalRoPE, where the frequency exponent is divided by the global head
dimension (512) rather than the rotated dimension (128). The `base` parameter
is ignored when `freqs` is provided; `has_value = false` is set on the
`mlx_optional_float` struct to signal this explicitly.

### `scaled_dot_product_attention`

Fused FlashAttention-style SDPA. Wraps
`mlx_fast_scaled_dot_product_attention`. Arguments:

- `q`, `k`, `v`: `[batch, n_heads, seq_len, head_dim]`.
- `scale`: `1/sqrt(head_dim)` or `1.0` (Gemma4 uses `1.0`).
- `mask_mode`: `"causal"` (kernel handles masking internally, fastest),
  `"additive"` (caller supplies additive mask), or `""` (no mask).
- `mask_arr`: ignored when `mask_mode = "causal"`; null sentinel when `None`.
- `sinks`: always null sentinel (not used at this stage).

---

## Compiled Kernels (`compile` module)

`compile.rs` wraps the `mlx_compile` / `mlx_closure` C API. This is the
Rust equivalent of Python's `@mx.compile` decorator.

### `Closure`

`Closure` is an owned RAII handle around `mlx_closure`. It is freed on drop
via `mlx_closure_free`. The underlying handle is ref-counted by mlx-c.

```rust
// SAFETY: mlx_closure is a ref-counted pointer (analogous to Arc<T>).
// mlx-c docs guarantee thread-safe ref-counting.
unsafe impl Send for Closure {}
unsafe impl Sync for Closure {}
```

`Closure::from_fn` accepts any
`Fn(Vec<Array>) -> Result<Vec<Array>> + Send + Sync + 'static` and bridges
it to the C callback ABI via a heap-allocated `Box<BoxFn>` whose raw pointer
is stored as the closure payload:

```rust
// SAFETY: payload is a *mut BoxFn cast to *mut c_void, valid for the
// lifetime of the mlx_closure (the dtor frees the Box when the closure
// is dropped). input is a borrowed mlx_vector_array; do NOT free it.
// output is a freshly created mlx_vector_array we must populate.
// Panics must NOT propagate across the FFI boundary.
unsafe extern "C" fn rust_closure_callback(
    output: *mut sys::mlx_vector_array,
    input: sys::mlx_vector_array,
    payload: *mut c_void,
) -> c_int { … }
```

The callback catches panics via `std::panic::catch_unwind` so they never
propagate across the FFI boundary; a panic is converted to a non-zero return
code and logged via `tracing::error!`.

Output packing requires creating a new `mlx_vector_array` handle and
overwriting `*output` with it. The C++ lambda that calls the Rust callback
creates `*output` with a null ctx; appending to a null-ctx vector is
undefined. The correct pattern: allocate a new initialized vector, append,
then overwrite `*output` (a plain struct copy of the ctx pointer). The C++
lambda reads `ctx` from `*output` and frees it — no leak occurs.

### `compile` / `compile_shapeless`

Both functions consume a `Closure`, pass its inner handle to `mlx_compile`,
and return a new compiled `Closure`. The compiled closure replays the cached
Metal program on every subsequent call without re-tracing the Rust op loop.

- `compile` (`shapeless = false`): re-traces when input shapes change.
- `compile_shapeless` (`shapeless = true`): reuses one compiled program
  regardless of shape; the Metal dispatch grid is adjusted at runtime. Use
  this for ops called with variable sequence lengths (e.g. chunked prefill).

---

## MSL Kernels (`metal_kernel` module)

`metal_kernel.rs` wraps the `mlx_fast_metal_kernel` API from mlx-c 0.6,
which compiles and dispatches arbitrary MSL (Metal Shading Language) kernels
within the MLX compute graph.

### `MetalKernel`

Compiled Metal kernel handle. RAII: freed on drop via
`mlx_fast_metal_kernel_free`.

```rust
// SAFETY: mlx_fast_metal_kernel is a void* handle. The kernel object is
// immutable after `new` (no mutation through &self). MLX's Metal device
// context is process-global; callers must hold the rMLX process-level
// GPU claim before invoking.
unsafe impl Send for MetalKernel {}
unsafe impl Sync for MetalKernel {}
```

`MetalKernel::new` converts all string arguments to `CString`, builds
`mlx_vector_string` handles for `input_names` and `output_names`, then calls
`mlx_fast_metal_kernel_new`. String vectors are freed after the call;
mlx-c copies them internally.

`ensure_row_contiguous = true` is always set — the safer default for custom
kernels. `atomic_outputs = false` unless the kernel requires read-modify-write
atomics (e.g. `atomic_fetch_or_explicit`).

`MetalKernel::apply` takes a `MetalKernelInvoke` by value (consumed), builds
input/output `mlx_vector_array` handles, dispatches via
`mlx_fast_metal_kernel_apply`, extracts output `Array` handles, and returns
them.

**Lazy compile.** `MetalKernel::new` only *registers* the kernel
with MLX; the MSL → Metal pipeline compiles on the **first `apply()`
dispatch**, not at `new`. For KV codecs this means the shader cold-compile lands
inside the first user request unless it is warmed earlier. The KV layer warms
its shader-heavy codecs at model-load time via
`rmlx_kv_quant::precompile::precompile_kv_codec_msl` (one representative dispatch
per codec kernel during the eager-preload window); see `docs/KV_QUANT.md`
§ "Metal-vs-CPU hot path + load-time MSL precompile". The `gdn_warmup` in
`rmlx-models::arch::loader` is the analogous warm for the GatedDeltaNet compiled
graph.

### MSL source conventions

The `source` parameter is the **body** of the kernel function. MLX wraps it
with the Metal function signature and buffer declarations automatically:

- Input buffers: `device const T* <name>` (auto-typed from array dtype).
- Output buffers: `device T* <name>`.
- Built-ins: `thread_position_in_grid` (uint3),
  `threadgroup_position_in_grid` (uint3),
  `thread_position_in_threadgroup` (uint3).
- `header` is MSL inserted before the kernel body; use it for `constant`
  array declarations and helper functions.

### `MetalKernelInvoke`

Builder for a single dispatch. `add_input` clones the input `Array` (via
`mlx_array_set` ref-count increment) into the builder's `inputs` vec.
`add_output_shape` declares an output buffer via
`mlx_fast_metal_kernel_config_add_output_arg`. `set_grid` and
`set_thread_group` configure the 3-D Metal dispatch geometry. Template
arguments (`set_template_int`, `set_template_dtype`) specialize the MSL
template at JIT compile time.

`set_init_value` zeroes output buffers before the kernel runs. Required for
kernels that accumulate into outputs via `atomic_fetch_or_explicit` — without
it, MLX may reuse a Metal buffer from its pool whose previous contents are
non-zero, corrupting the result.

### MSL kernels in `rmlx-models`

`rmlx-models` ships MSL source strings for each KV-quant codec as
`OnceLock<MetalKernel>` singletons registered on first use:

| Module | Codec | Group size |
|--------|-------|-----------|
| `q8_msl` | Symmetric 8-bit affine (q8_0) | 128 |
| `turboquant_msl` | TurboQuant V4 (Lloyd-Max 4-bit) | 32 |
| `planarquant_msl` | PlanarQuant | codec-specific |
| `paroquant_msl` | ParoQuant | codec-specific |
| `turbo_flash_msl` | TurboFlash (tq4 V, fused SDPA) | 32 |
| `gated_delta_msl` | GatedDelta | codec-specific |
| `sparse_v_msl` | Sparse-V | codec-specific |

Each module declares its kernel once, encodes codebook constants as
`constant float` MSL declarations in `header`, and writes MSL body source
that matches the CPU reference path in the corresponding `*quant.rs` file.

---

## Unsafe Policy

Every file in `rmlx-mlx` carries:

```rust
// unsafe_code: mlx-rs FFI bridge — <per-file justification>
#![allow(unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]
```

`#![deny(unsafe_op_in_unsafe_fn)]` is preserved throughout: every `unsafe`
operation inside an `unsafe fn` must be individually justified with its own
`// SAFETY:` comment. This prevents the common anti-pattern of marking an
entire function `unsafe` and then writing unjustified unsafe code in its
body.

### SAFETY contracts by pattern

**`Array::from_bytes` / `mlx_array_new_data`**

```
// SAFETY: mlx_array_new_data copies the buffer immediately. The
// returned handle owns its data; `data` need not outlive this call.
```

**`mlx_array_shape` / `mlx_array_data_uint8` (borrowed pointer)**

```
// SAFETY: ptr is valid for the lifetime of the Array object (mlx-c
// contract). Copied into Vec before returning; no pointer escapes.
```

**`null_sentinel` / `EMPTY_ARRAY_SENTINEL` (process-global null handle)**

```
// SAFETY: the null-ctx handle returned here is only valid as a
// "sentinel absent" argument to mlx-c functions that document
// "may be null". Never store or materialize the returned handle
// as a real Array.
```

**`with_stream` (stream handle borrow)**

```
// SAFETY: f must not store the stream handle past the duration of
// the call. mlx_default_gpu_stream_new returns a ref-counted handle;
// freeing it after the call decrements the count without tearing down
// the stream or its thread.
```

**`check_status` (thread-local error retrieval)**

```
// SAFETY: must be called immediately after the mlx-c call whose status
// is being checked, on the same thread, before any other mlx-c call
// that could overwrite the thread-local error slot.
```

**`rust_closure_callback` (FFI → Rust trampoline)**

```
// SAFETY: payload is a *mut BoxFn cast to *mut c_void, valid for the
// lifetime of the mlx_closure (the dtor frees the Box when the closure
// is dropped). input is a borrowed mlx_vector_array; we must NOT free
// it. output is a freshly created mlx_vector_array we must populate.
// Panics must NOT propagate across the FFI boundary.
```

**`MetalKernel` / `Closure` (`Send + Sync`)**

```
// SAFETY: the handle is immutable after construction and is ref-counted
// by mlx-c. MLX's Metal device context is process-global; rMLX enforces
// single-process ownership via the /tmp/rmlx.<port>.claim file.
```

**`mlx_array_data_uint8` readback**

```
// SAFETY: ptr is non-null and points to `nbytes` contiguous bytes owned
// by the mlx array. Copied into Vec before returning.
```

---

## See also

- `docs/KV_CACHE.md` — KV cache design; describes how `slice_update`,
  `take`, and the MSL quant kernels are composed to implement the KV quant
  families.
- `docs/WEIGHT_QUANTS.md` — weight quantization formats and how
  `quantized_matmul` / `dequantize` / `gather_qmm` interact with them.
- `docs/KV_QUANT.md` — KV quantization format details; how the MSL kernels
  in `rmlx-models/*_msl.rs` map to each codec.
