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
never see an `unsafe` block related to mlx-c. `rmlx-kv-quant` (KV-cache
codecs) and `rmlx-models` (per-arch kernels) ship MSL source that is
registered and dispatched through `rmlx-mlx`'s `metal_kernel` module.

The crate does **not** depend on `mlx-rs` (the community Rust binding). It
drives mlx-c directly to control every detail of the FFI contract.

---

## mlx-c Contract

### Versioned ABI

mlx-c pins a specific MLX version. The build links `libmlxc.dylib` and
`libmlx.dylib`, resolving each prefix in this order:

1. `MLX_C_PREFIX` / `MLX_PREFIX` — explicit override.
2. `brew --prefix mlx-c` / `brew --prefix mlx`.
3. `/opt/homebrew/opt/mlx-c` / `/opt/homebrew/opt/mlx` — the conventional
   Homebrew `opt` symlink.

`build.rs` asserts the dylibs exist and aborts the build with an actionable
message if they are absent.

Resolving to the `opt` path rather than a Cellar path is deliberate. Both
dylibs' install names **are** `opt` paths:

```
$ otool -D /opt/homebrew/opt/mlx/lib/libmlx.dylib
/opt/homebrew/opt/mlx/lib/libmlx.dylib
```

so that symlink decides what gets loaded at run time regardless of what the
build pointed at. An upgrade repoints it and silently retargets an
already-built binary — no rebuild, no relink, no diagnostic. Building against
the same path the loader uses is what keeps compile-time and run-time on the
same file; a hard-coded Cellar path drifts from it on the next upgrade.

### Pinned MLX / mlx-c pair

rMLX declares the MLX stack it is validated against in **one** place:

```
crates/rmlx-mlx/mlx-pin.txt
```

```
mlx    0.31.2
mlx-c  0.6.0_2
```

Nothing else in the tree declares these versions — `build.rs` reads that file,
and bumping a line there is the whole change.

**They bump together.** mlx-c is compiled against a specific mlx, and both
resolve the moving `opt` symlink at run time, so a mismatched pair aborts at
load:

```
dyld: Symbol not found: __ZN3mlx4core4fast12metal_kernelE...
  Referenced from: .../mlx-c/0.6.0_3/lib/libmlxc.dylib
  Expected in:     .../mlx/0.31.2/lib/libmlx.dylib
```

(mlx-c 0.6.0_3 is built against mlx 0.32.0. Same upstream 0.6.0 as `_2` — the
Homebrew revision suffix is the only thing that distinguishes them, which is
why the pin carries it.)

#### Why the pin exists

Homebrew's `mlx` 0.32.0 bottle ships **zero** `steel_gemm_fused_nax_*` kernels
— the M5 Neural-Accelerator GEMM path. The pinned 0.31.2 bottle ships 145 of
them:

```sh
strings "$(brew --prefix mlx)/lib/mlx.metallib" | grep -c steel_gemm_fused_nax
# 0.31.2 -> 288      0.32.0 -> 0
```

(`grep -c` counts matching *lines*, not kernels: 145 distinct kernel functions
appear on 288 lines. Only the zero is load-bearing — the build's probe is
boolean and never counts.)

Measured cost: ~3.8× lower GPU matmul throughput, 2.2–3.7× slower prefill on
Neural-Accelerator-class hardware. Decode is bandwidth-bound and barely moves
(gemma-4-e2b @4k, median of 3, same binary: prefill 6,916 → 15,352 t/s, decode
124.5 → 119.9), so the symptom looks like a model-code defect rather than a
toolchain one — it cost one investigation days of misattribution before the
metallib was inspected.

This is a **Homebrew bottle regression, not an MLX regression**: the upstream
0.32.0 PyPI wheel ships the kernels and is fine. Tracked in
[#216](https://github.com/Pushkinist/rMLX/issues/216).

#### What the build checks

`build.rs` warns — never fails — on two independent things:

| Check | Fires when | Meaning |
|---|---|---|
| **Capability** | the resolved `mlx.metallib` contains no `steel_gemm_fused_nax` | the real defect; names both versions and the fix command |
| **Pin drift** | resolved mlx/mlx-c ≠ the pinned pair, but the kernels are present | informational; the pair is unvalidated, and bumping the pin may be due |

Warning rather than failing is deliberate: the pin records what *was validated
here*, not a claim that everything else is broken. A correct non-bottle build
of another version must still compile.

#### Run identity: `RMLX_MLX_NAX`

The same metallib scan also stamps `cargo:rustc-env=RMLX_MLX_NAX=<present|absent|unknown>`
(exposed as `rmlx_mlx::NAX_CAPABILITY`) — not a second detection path, the same
`fast_gemm` result the two warnings above already computed. `rmlx-cli::main()`
forwards it into `rmlx-metrics`'s run identity
(`rmlx_metrics::identity::set_mlx_nax`), so every `events` row records whether
that run built against a nax-capable MLX. See `docs/METRICS_DB.md` §3.6 for
the column and why the propagation goes through a runtime setter rather than
a second `env!()` read (`cargo:rustc-env` only reaches the compiler
invocation of the crate whose build script set it).

The capability probe is the ground truth; the version pin only proxies for it.
The probe is what keeps this from nagging forever once a fixed bottle ships —
it simply passes. Neither check can be verified from a version number alone,
which is why both exist.

Version identity comes from different places per formula, of necessity: mlx
ships `include/mlx/version.h` (authoritative, and works on non-Homebrew
layouts), while mlx-c ships no version header at all — its identity is the keg
directory name, which is also the only place the load-bearing revision suffix
appears. A non-keg layout (a wheel, a hand-built tree) yields no version and
the pin stays quiet rather than guessing.

#### Fixing a machine that drifted

Both kegs must already be in the Cellar (`ls /opt/homebrew/Cellar/mlx`):

```sh
ln -sfn ../Cellar/mlx/0.31.2 /opt/homebrew/opt/mlx && \
ln -sfn ../Cellar/mlx-c/0.6.0_2 /opt/homebrew/opt/mlx-c && \
brew pin mlx mlx-c && \
cargo clean -p rmlx-mlx        # required — see below
```

`brew pin` stops a later `brew upgrade` from repointing the symlinks back.

**The `cargo clean` is required, not hygiene.** Cargo re-runs a build script
only when a `rerun-if-changed` path is *newer* than the last run, and it stats
**through** the symlink. Repointing `opt/mlx` at the older validated keg moves
the observed mtime **backwards** (0.31.2's metallib predates 0.32.0's), so a
plain rebuild does not re-run `build.rs` at all: the crate keeps bindings
generated against the wrong headers and a stale baked-in
`RMLX_MLX_BUILD_VERSION`, while the loader picks up the newly-linked pair. That
combination is the ABI abort this section exists to avoid. Touching
`crates/rmlx-mlx/mlx-pin.txt` forces the same re-run if you prefer.

#### Un-pinning when the bottle is fixed

1. `brew unpin mlx mlx-c && brew upgrade mlx mlx-c`
2. Verify the capability actually returned — this is the whole point:
   ```sh
   strings "$(brew --prefix mlx)/lib/mlx.metallib" | grep -c steel_gemm_fused_nax
   # must be non-zero (288 on the 0.31.2 bottle); zero means the bottle is still broken
   ```
   Assert non-zero, not a particular count: the number is a `strings` line count
   that tracks kernel-name spelling, and the build's probe only asks present/absent.
3. Bump **both** lines in `crates/rmlx-mlx/mlx-pin.txt` to the new pair
   (`brew list --versions mlx mlx-c` gives the keg names, revision suffix
   included). Editing the pin file is itself a rebuild trigger, which is what
   makes step 4 mean something.
4. Rebuild and confirm the build emits no MLX warning — but only after
   `cargo clean -p rmlx-mlx`. Upgrading normally moves mtimes forward, so the
   re-run usually happens on its own; a *downgrade* does not, and then "no
   warning" would merely mean the check never ran. Force it and the step is
   real. Confirm `build.rs` actually ran with `cargo build -p rmlx-mlx -vv`
   (its output appears only on a real run).
5. Re-verify on a real prefill cell, not just the kernel count:
   ```sh
   rmlx baseline --model <gemma-4-e2b> --kv-quant none --max-ctx 8192 --prompt-tokens 4096
   ```
   Expect ~15k t/s prefill. A regression to ~7k means the kernels are present
   but unused — reopen [#216](https://github.com/Pushkinist/rMLX/issues/216)
   rather than re-pinning silently.

Note that fixture lengths are nominal: `--prompt-tokens 4096` tokenizes to
~4410 on Gemma, so pass `--max-ctx` deliberately or the prompt is rejected and
the cell reports `prefill_tps=0`.

### Runtime version skew

The build-time pin cannot see a symlink that moves *after* the build. The FFI
layer therefore also compares the version it was compiled against
(`RMLX_MLX_BUILD_VERSION`, baked by `build.rs` from the resolved prefix's
`version.h`) against `mlx_version()` of the library actually loaded, on the
existing one-shot init, and warns on mismatch. Together the two checks cover
both halves: the pin catches a bad stack at compile time, the skew warning
catches a stack that changed underneath a built binary.

### Build pipeline (`build.rs`)

1. Target guard: `aarch64-apple-darwin` only. The assertion fires at compile
   time for any other target.
2. `bindgen` runs against `wrapper.h`, producing `$OUT_DIR/bindings.rs`. Only
   `mlx_*` symbols are allowlisted; `mlx_dtype_` and `mlx_device_type_` are
   emitted as Rust enums.
3. `build.rs` post-processes `bindings.rs` to strip any `#![...]` inner
   attributes bindgen 0.71 emits at the file head — these are illegal inside
   the `include!()` call site in `sys.rs`.
4. The resolved MLX / mlx-c pair is checked against `mlx-pin.txt`, and the
   resolved `mlx.metallib` is scanned for the fast GEMM kernels. Both warn,
   neither fails — see "Pinned MLX / mlx-c pair" above.
5. Rebuild is triggered on `MLX_C_PREFIX`, `MLX_PREFIX`, `wrapper.h`,
   `build_support.rs`, `mlx-pin.txt`, or a *newer* resolved
   `version.h` / `mlx.metallib` (each registered only when it exists — cargo
   treats a missing trigger path as permanently dirty, which would re-run
   bindgen on every build of a non-keg layout).

   **Repointing the `opt` symlink does not reliably re-run these checks.** Cargo
   compares mtimes through the symlink and re-runs only for a newer one, so
   moving to an *older* keg — the recovery direction — looks like nothing
   changed. Use `cargo clean -p rmlx-mlx` when repointing; the runtime
   version-skew warning below is the backstop for a binary that was never
   rebuilt.
6. rpath entries are emitted so the binary finds both dylibs at runtime
   without `DYLD_LIBRARY_PATH`.

`build.rs`'s pure logic (pin parsing, keg-version and header-version
resolution, the metallib scan) lives in `build_support.rs`, `include!`d by both
the build script and `tests/mlx_pin.rs`. A build script cannot be imported by
the crate it builds, and this logic decides whether a known 3.8× perf cliff is
reported at all — so it is covered by tests `cargo test` actually runs.

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

### Per-thread GPU stream context — `ensure_gpu_default_stream`

`with_stream` borrows the process-global default stream, but MLX's
`eval` materialises arrays through a **thread-local** map of
`{stream_index → CommandEncoder}`. The encoder entry for a given stream is
created when `mlx::core::new_stream` runs on that thread. tokio
blocking-pool worker threads never call `new_stream`, so an `Array::eval()`
on such a thread can fail with `There is no Stream(gpu, 0) in current
thread`.

`rmlx_mlx::ensure_gpu_default_stream()` fixes this: it creates a GPU stream
(registering a fresh `CommandEncoder` for the calling thread), sets it as the
thread's default, and stores the handle in a thread-local for the thread's
lifetime. It is idempotent and a no-op when the GPU is unavailable, with zero
ML-semantic effect.

### Per-thread CPU stream context — `ensure_cpu_default_stream`

The same thread-local-encoder problem exists on the CPU side, independently of
the GPU one: MLX's CPU backend resolves a stream's `CommandEncoder` through a
**thread-local** map first
(`mlx/backend/cpu/encoder.cpp::get_command_encoder`), falling back to a
process-global map and otherwise throwing `There is no Stream(cpu, N) in
current thread.` A worker thread whose graph includes a CPU-scheduled op —
e.g. the K8V8 `exit_prefill` quantization's scale reduction, which MLX places
on the CPU stream even though the surrounding quantize dispatch runs on the
GPU device — can fault the same way the GPU path did (issue #206).

`rmlx_mlx::ensure_cpu_default_stream()` is the CPU analog of
`ensure_gpu_default_stream()`: same mechanism (create a stream, register it as
the calling thread's default, leak the handle in a thread-local for the
thread's lifetime), same idempotency guarantee, zero ML-semantic effect.

**Contract:** every blocking-thread inference entry point calls
`ensure_cpu_default_stream()` **unconditionally** (not gated on the resolved
device — a GPU-device forward can still schedule CPU-side ops), before
`ensure_gpu_default_stream()` when both apply. Covered entries: the text
generate dispatch (`arch::generate_greedy`), the image generate dispatch
(`arch::generate_image` and the server's `run_qwen3vl_image`), the
speculative-decode blocking closure, the
audio-transcription blocking closure (`audio.rs` Whisper decode, and the CLI
`transcribe` command), and the embeddings compute closure (`embeddings.rs`
`compute_embeddings`). New blocking-pool entry points that materialise CPU or
GPU arrays must follow the same pattern (both guards, CPU first).

**Bounded leak (serve only).** Both guards deliberately leak their stream
handle on thread exit — freeing it would drop the `CommandEncoder` entry a
still-running eval might reference. Each leaked handle is also backed by its
own MLX-internal OS thread, so the leak is only safe if the **set of distinct
worker threads that ever call these guards is bounded**. `rmlx serve`
(`crates/rmlx-cli/src/commands/serve.rs`) builds its tokio runtime with a
capped `max_blocking_threads` and a long `thread_keep_alive`, so the blocking
pool's worker threads are reused rather than idle-reaped-and-replaced under
sporadic load — bounding the cumulative leak to that cap instead of growing
unbounded over long serve uptime (which would otherwise eventually exhaust the
~2 048 per-process pthread ceiling noted above). One-shot CLI commands
(`chat`, `baseline`, `info`) are unaffected — the leak is bounded by the
process lifetime regardless.

See `docs/KV_CACHE.md` §5.7.5 and `docs/KV_QUANT.md` for the `exit_prefill`
mechanism this guards, including the guard's limitation (it registers the
*worker's own* stream — it does not fix a genuinely cross-thread eval).

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

**Never `eval()` a kernel's inputs before dispatching it.** `Array::eval()`
blocks the calling thread until the GPU has produced the array, so an `eval()`
inside a per-layer dispatcher runs the forward pass one layer at a time with
nothing queued ahead — the host and the GPU stop overlapping. It produces
byte-identical output, which is why the cost hides: it shows up only as a
decode rate several times below what the kernel can reach. A KV flash-decode
dispatcher paying this on every attention layer measured **2.7× below** its own
rate on Ternary-Bonsai-8B once the `eval()` calls were dropped, with the token
digest unchanged. Pass lazy arrays to `MetalKernel::apply` and let MLX schedule
the graph; the row-contiguous guarantee a raw-linear kernel needs comes from
`ensure_row_contiguous` (below), not from forcing evaluation. The CI gate
`make check-no-kernel-input-eval` enforces this for the flash-decode
dispatchers, with an `// eval-ok: <reason>` marker for a genuinely load-bearing
barrier.

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

`ensure_row_contiguous` is what makes it safe for a kernel body to index its
buffers by raw linear offset: MLX copies any input that is not row-contiguous
(a lazy transpose, a strided view) before the dispatch. Callers therefore do
**not** need to materialise inputs themselves — see "Never `eval()` a kernel's
inputs" under Evaluation. It does not fix a *semantic* layout disagreement: an
array whose logical axis order is not the one the kernel indexes is still wrong
after the copy, and that is what the canonical seq-major KV store layout is
for.

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

- Input buffers: `device const T* <name>` for most arrays (auto-typed from
  array dtype) — **but not always**. MLX silently switches an input's
  outer-kernel parameter to the `constant` address space instead of `device`
  when the array is small (measured trip point: fewer than 8 elements, for
  every dtype and shape tried); this is an internal MLX size heuristic, not
  something a caller can pin per-argument. If the kernel body calls a
  header/body helper function whose parameter is hard-declared
  `device const T*`, that call fails the MSL compile the moment a small
  enough array reaches it (`Unable to build metal library from source: no
  matching function ... cannot pass pointer to address space 'constant' as a
  pointer to address space 'device'`) — a **first-dispatch** failure, not a
  `MetalKernel::new`-time one, so it can ship silently until a small enough
  input shows up in production. Any kernel whose input can legitimately be
  tiny at a valid call (e.g. a per-token `norms` buffer at low `kv_seq`) must
  pad that array up past the trip point before dispatch rather than assume
  `device` binding. See `rmlx_kv_quant::flash_decode_common::pad_norms_to_device_floor`
  (floor 16, 2× the measured 8-element trip point for margin) — the general
  fix shared by `iso_flash_decode_symv_sdpa` and `rotor_flash_decode_symv_sdpa`,
  documented per-codec in `docs/KV_QUANT.md`.
- Output buffers: `device T* <name>`.
- Built-ins: `thread_position_in_grid` (uint3),
  `threadgroup_position_in_grid` (uint3),
  `thread_position_in_threadgroup` (uint3).
- `header` is MSL inserted before the kernel body; use it for `constant`
  array declarations and helper functions.

For KV codecs the `source` argument is not a Rust literal — it is
`include_str!` of a `.metal` file. See "`.metal` files + `include_str!`" below.

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

### Where MSL lives

MSL source is split across two crates:

| Crate | Modules | Scope |
|---|---|---|
| `rmlx-kv-quant` | `src/*_msl.rs`, `src/sparse_attn/*_msl.rs` | Every KV-cache codec (q8, TurboQuant, PlanarQuant, IsoQuant, RotorQuant, rot-K, TCQ, TurboFlash, fused-QK, sparse-attn phases) |
| `rmlx-models` | `paroquant_msl.rs`, `gated_delta_msl.rs` | Per-arch kernels — weight-side ParoQuant and GatedDeltaNet. Not KV codecs. |

Each module registers its kernels once as `OnceLock<MetalKernel>` singletons
on first use, and its MSL body matches the CPU reference path in the
corresponding `*quant.rs` file.

### `.metal` files + `include_str!` (KV codecs)

KV kernel bodies live in **`.metal` files** under
`crates/rmlx-kv-quant/src/metal/`, not in Rust string literals. They are
embedded at **compile time**:

```rust
const QUANTIZE_SOURCE: &str = include_str!("metal/q8_quantize.metal");
```

`include_str!` — never a runtime `fs::read`. The binary stays single-file with
no runtime data files (CLAUDE.md hard rule 2).

**Body / header split.** A `.metal` file holds the kernel *body* only — MLX
supplies the function signature and buffer declarations at dispatch. The
`header` argument is separate, and comes from one of two places:

- **Static header** — a `.metal` file of its own (`turboquant_header.metal`,
  `turbo_flash_header.metal`, …), embedded the same way.
- **Runtime-generated header** — a `build_*_header(..)` Rust function that
  emits `constant` / `#define` declarations whose values are computed
  (codebooks, rotation constants, quaternions, eps). These stay in Rust: they
  are derived data, not source text. The kernel is assembled at registration as
  `MetalKernel::new(name, header, include_str!("<body>.metal"), ..)`.

**Parameterised bodies.** Where a body varies by a codec parameter, each
variant gets its own `.metal` file and the builder selects between them
(`iso_fused_qk_b3.metal` / `_b4.metal`; `rot_k_fwht_quantize_d{32..512}.metal`).
The body text stays literal — parameters are not templated back into it at
runtime.

Adding a KV codec means adding a `.metal` decode kernel and a native compile
test; see CLAUDE.md hard rule 10.

### MSL gates (`make ci`, enforced in CI)

Two gates run over `crates/rmlx-kv-quant/src/metal/*.metal`.

| Target | Tool | Checks |
|---|---|---|
| `make check-metal-compiles` | `xcrun -sdk macosx metal` (full Xcode, not just the Command Line Tools) | Every KV kernel compiles natively, so an MSL syntax error surfaces at CI instead of on first GPU dispatch. |
| `make check-metal-format` | `clang-format` (on `PATH` or via `xcrun -f clang-format` — it is not on `PATH` by default) | Every KV kernel is clang-format clean. MSL is a C++14 dialect; style is pinned by `src/metal/.clang-format`. |

**Where they actually run.** Both gates skip when their tool is missing, so a
Command-Line-Tools-only box is not blocked — but a skipping gate protects
nothing, so the skip is local-only. The `msl` job in
`.github/workflows/ci.yml` runs both with `METAL_STRICT=--strict`, which turns
a missing tool into a hard failure. The GitHub macOS runner ships full Xcode,
so the compile gate runs for real there; compiling MSL needs the toolchain,
not a GPU, so it works on a runner with no usable Metal device. Install full
Xcode (`xcode-select -s /Applications/Xcode.app`) to run the compile gate
locally too — on Xcode 16.3+ the compiler is a separate component
(`xcodebuild -downloadComponent MetalToolchain`).

`check-metal-compiles` cannot compile a `.metal` file directly — a body is a
run of statements at file scope, not a translation unit. It assembles a probe
per kernel (`stdlib preamble + header + kernel { buffer aliases + body }`) and
compiles that. `src/metal/probes/kernels.manifest` supplies the header and
buffer list per body; `probes/README.md` documents the layout and how to
refresh the captured header snapshots.

Deliberately **not** wired: `clang-tidy` (wants a compilation database and is
noisy on MSL) and MegaLinter (CI-heavy). The two gates above already cover
syntax and style.

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
  in `rmlx-kv-quant/src/*_msl.rs` map to each codec.
