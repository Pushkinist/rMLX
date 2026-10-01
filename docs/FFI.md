# FFI Bridge Reference

How Rust drives MLX through `mlx-c`, with no Python at runtime.

---

## Overview

`mlx-c` is the stable C ABI over the MLX C++ library (`mlx_array`,
`mlx_closure`, `mlx_fast_metal_kernel`, …). `rmlx-mlx` binds it with
`bindgen` and is the only crate that calls it: the raw bindings are
crate-private. Every other crate uses `rmlx-mlx`'s public API.
`rmlx-kv-quant` and `rmlx-models` ship MSL kernel bodies and dispatch them
through `rmlx-mlx`'s `metal_kernel` module.

`rmlx-mlx` does not depend on `mlx-rs`.

---

## mlx-c Contract

### Versioned ABI

The build links `libmlxc.dylib` and `libmlx.dylib`. Each prefix resolves in
this order:

1. `MLX_C_PREFIX` / `MLX_PREFIX`;
2. `brew --prefix mlx-c` / `brew --prefix mlx`;
3. `/opt/homebrew/opt/mlx-c` / `/opt/homebrew/opt/mlx`.

`build.rs` aborts with an actionable message if a dylib is missing.

Both dylibs' install names are Homebrew `opt` paths
(`/opt/homebrew/opt/mlx/lib/libmlx.dylib`). The `opt` symlink therefore
decides what loads at run time, whatever the build pointed at. Repointing it
retargets an already-built binary with no rebuild and no diagnostic. Building
against the same path the loader uses keeps compile time and run time on one
file.

### Pinned MLX / mlx-c pair

The validated stack is declared in one file,
`crates/rmlx-mlx/mlx-pin.txt`:

```
mlx    0.32.3
mlx-c  0.7.0
```

`src/pin.rs` reads it; bumping a line there is the whole change.

**The two bump together.** mlx-c is compiled against one mlx, and both
resolve the `opt` symlink at run time, so a mismatched pair aborts at load
with a `dyld: Symbol not found` naming an `mlx::core` symbol. The pin names
Homebrew keg directories, so a revision suffix (`0.6.0_4`) is part of a
version.

**Why the pin exists.** MLX builds its Neural-Accelerator (NAX) kernels only
for a deployment target of macOS 26.2 or later
(`mlx/backend/metal/kernels/CMakeLists.txt`). Homebrew builds its macOS 26
(`arm64_tahoe`) bottle for target 26, so that bottle has none; the macOS 27
bottle has them. Without them GEMM-bound prefill is slower while output and
decode look normal. The pinned pair is mlx 0.32.3 built from source for
target 26.2, and mlx-c 0.7.0 with a one-line patch. How to build it, per
host: [`docs/MLX_PAIR.md`](MLX_PAIR.md).

```sh
strings "$(brew --prefix mlx)/lib/mlx.metallib" | grep -c steel_gemm_fused_nax
# non-zero on a NAX build, 0 on the macOS 26 bottle
```

rMLX also compiles against the Homebrew pair, mlx 0.32.1 + mlx-c 0.6.0_4
(see [Two mlx-c C APIs](#two-mlx-c-c-apis)). The gate that checks the loaded
pair, and binds only on M5 and later, is in
[`docs/MLX_PAIR.md`](MLX_PAIR.md#the-pin-gate).

#### Where NAX can appear, and where it cannot

NAX **GEMM** serves every model's matmuls. NAX **attention**
(`mlx/backend/metal/scaled_dot_product_attention.cpp`, v0.32.3) is narrower:

| Path | NAX? | Why |
|---|---|---|
| Prefill attention (`q_seq > 8`), `head_dim` 64, 96 or 128, Q not f32 | yes | `sdpa_full` takes the NAX kernel. 72 and 80 are padded to 96 only for long unmasked half-precision calls. |
| Prefill attention, `head_dim` 256 | when `q_seq >= 1024` with a causal or array mask | Otherwise the composite graph. |
| Prefill attention, `head_dim` 512 | when `q_seq >= 1024`, causal, no array mask, and `B × H × ⌈q_seq/32⌉ >= 1024` | Otherwise the composite graph. |
| Decode attention, any `head_dim`, any codec | no | `q_seq <= 8` routes to `sdpa_vector`, which has no NAX variant. |
| Our own KV decode-attention kernels | no | They run at `q_seq = 1` and stream the KV cache at O(1) FLOPs per byte, so more arithmetic throughput cannot help. No production rMLX kernel uses `mpp::tensor_ops`; only the JIT probe does. |

So no decode path on any codec reaches NAX. `mpp::tensor_ops::matmul2d`
allows an `M` below MLX's shipped tile, so the tile is not what keeps decode
off NAX.

### Two mlx-c C APIs

rMLX compiles against the mlx-c 0.6 and the 0.7 C API, and refuses the one
call whose argument list differs when the loaded `libmlxc.dylib` has the
other C API: [`docs/MLX_PAIR.md`](MLX_PAIR.md#two-mlx-c-c-apis).

### Runtime version skew

The pin cannot see a symlink that moves after the build. `build.rs` bakes the
compiled-against version as `RMLX_MLX_BUILD_VERSION` from `version.h`. On the
one-shot init, the FFI layer compares it with `mlx_version()` of the loaded
library and warns on a mismatch.

### Runtime NAX capability (`src/nax.rs`)

A version match is not a capability match: a distributed binary links the
installing user's MLX through the `opt` symlink. `src/nax.rs` therefore scans
the metallib at run time, on the same one-shot init. It finds the loaded
`libmlx.dylib` in dyld's image list and reads the `mlx.metallib` beside it.

The host-class gate runs first. On a chip below Apple GPU family 10 neither
the image walk nor the file open happens.

| Host | Kernels | Result |
|---|---|---|
| M5 and later | absent | `warn!` naming prefill/TTFT, the metallib, and the fix for this macOS version |
| M5 and later | present | `debug!` only |
| M5 and later | metallib unreadable or not found | `debug!` only; "could not look" is not "absent" |
| M1–M4, or chip unidentifiable | either | `debug!` only; no scan |

The fix the warning names, from `kern.osproductversion`:

| macOS | Fix |
|---|---|
| 27 and later | the Homebrew bottle carries the kernels: `brew upgrade mlx mlx-c`, or `brew reinstall mlx` at the same version |
| 26.2 to 26.x | the source build for target 26.2 ([`docs/MLX_PAIR.md`](MLX_PAIR.md)) |
| below 26.2 | none; MLX builds the kernels only for target 26.2 or later, so update macOS |

The warning names prefill and TTFT only, because NAX is unreachable at decode.

### Build pipeline (`build.rs`)

1. Target guard: `aarch64-apple-darwin` only.
2. `bindgen` runs on `wrapper.h` into `$OUT_DIR/bindings.rs`. It allowlists
   `mlx_*` / `_mlx_*` functions, `mlx_*` types and `MLX_*` vars, and emits
   `mlx_dtype_` and `mlx_device_type_` as Rust enums.
3. Inner attributes (`#![...]`) that bindgen emits at the file head are
   stripped; they are illegal at the `include!` site in `sys.rs`.
4. The MLX version from `version.h` is baked in as `RMLX_MLX_BUILD_VERSION`,
   and the mlx-c C API of the bindings as `cfg(mlxc_c_api_0_7)`.
5. Rebuild triggers: `MLX_C_PREFIX`, `MLX_PREFIX`, `wrapper.h`,
   `build_support.rs`, and a *newer* `mlx/version.h` of mlx or
   `mlx/c/fast.h` / `mlx/c/compile.h` of mlx-c, the two headers that decide
   `cfg(mlxc_c_api_0_7)`. Each file trigger is registered only when the file
   exists. Repointing `opt` to an older keg does not re-run the script; use
   `cargo clean -p rmlx-mlx`. A stale C API choice then fails at run time
   ([Two mlx-c C APIs](#two-mlx-c-c-apis)).
6. rpath entries for both prefixes let the binary run without
   `DYLD_LIBRARY_PATH`.

`read_mlx_version` lives in `build_support.rs`, `include!`d by both the build
script and `tests/mlx_build_version.rs`, so `cargo test` covers it.

### `sys.rs` — raw bindings

`sys.rs` includes `bindings.rs` inside a private `mod ffi` with blanket lint
suppression and re-exports it as `pub(crate) use ffi::*`. `sys::mlx_*` is the
only spelling of an mlx-c call in the crate; the `check-eval-lock` gate keys
on it. Nothing from `sys.rs` is public.

### Error delivery

mlx-c reports errors through a registered callback. `install_error_handler`
registers one per process (`std::sync::Once`). The handler writes the message
into the thread-local `LAST_ERROR: Cell<Option<String>>`. After each mlx-c
call, `check_status(status, context)` takes that message and returns
`Err(Error::Mlx(...))` on a non-zero status. It must run on the same thread,
before any other mlx-c call can overwrite the slot.

### Default stream

`with_stream(device, |s| …) -> Result<T>` gives `f` the stream of the MLX
thread for the device ([below](#the-mlx-thread)). Only `mlx_thread.rs` names an mlx-c stream function:
`every_stream_comes_from_the_mlx_thread` fails on one in any other source file
of the crate. It reads identifiers, not calls, so an alias, a glob import or a
`link_name` counts; a name that a macro builds from parts does not. For the GPU
device `with_stream` checks the latch below first.

### The CPU-device latch

`parse_device` calls `rmlx_mlx::forbid_gpu()` once, before any model work,
when it resolves a device with no claim; `qwen36_diag`'s `cpu` arm does too.
The call sets one process-global `AtomicBool` irreversibly, and makes the CPU
the MLX default device (process-wide). These calls check the latch first,
returning `Err(Error::GpuForbidden { op })`, not the mlx-c call:
`with_stream(Device::Gpu, ..)`, every `rmlx_mlx::metal` fn, and
`CaptureScope::start`.

**Guarantee.** Under `--device cpu`, no rmlx op runs on a GPU stream, and no
rmlx Metal API is called. A KV codec carrying MSL, and `--gpu-capture`, are
refused before the model load; any other GPU request returns `GpuForbidden`.
An op that MLX builds inside another op (the `astype` of bf16 scales in an
affine `quantized_matmul` with f32 input) runs on the CPU too
(`an_op_that_mlx_builds_itself_runs_on_the_cpu_under_cpu`).

Outside it: MLX's allocator opens an `MTLDevice` on first allocation either
way — `metal::allocator()` backs every array buffer. A GPU stream that MLX
creates itself, not from the default device, is uncovered.

### The MLX thread

MLX keeps the command encoder of a stream in the thread that created the
stream, and an evaluation looks for the encoder only in the evaluating thread:
`cpu::get_command_encoder` (`mlx/backend/cpu/encoder.cpp`, MLX 0.32) throws
`There is no Stream(cpu, N) in current thread.`, and the Metal device throws
`There is no Stream(gpu, N) in current thread.` on every supported MLX. Default
streams are per thread too (`mlx/stream.cpp`). An op records its stream when it
is built. So, without more, an array built on one thread fails when another
thread evaluates it, or the evaluation waits forever. `rmlx serve` builds
model state on one thread and uses it on another: a model loads on the preload
thread, on an async worker or inside a request, and generates on a
blocking-pool worker; a cache entry is written by one request and read by a
later one.

`crates/rmlx-mlx/src/mlx_thread.rs` starts one thread, the MLX thread, at the
first op. It creates one CPU and one GPU stream (its own default streams),
and keeps them for the life of the process.

- `with_stream` builds every op, on every thread, on one of the two. The first
  op on a thread also makes both of them the default streams of that thread,
  because MLX builds some ops inside other ops on the default stream of the
  default device, whatever device the caller builds on
  (`a_thread_that_built_an_op_defaults_to_the_mlx_threads_streams`). The GPU
  is left out after `forbid_gpu` ([the latch](#the-cpu-device-latch)) and on a
  Mac without Metal.
- `with_eval_lock` runs every evaluation on the MLX thread, under the
  evaluation lock. The caller waits. The mlx-c error message of the call moves
  back to the calling thread, so `check_status` reads it there. A panic
  continues on the calling thread.
- A hand-off allocates nothing and creates no channel: the job and its result
  stay on the stack of the waiting thread, and one static mailbox passes a
  pointer to the job (`crates/rmlx-mlx/tests/hand_off.rs`). Callers take
  turns (`concurrent_hand_offs_each_get_their_own_value`). Each side parks
  until the other unparks it. A job can borrow from its caller, so no mlx-c
  handle type needs `Send`.
- A call from the MLX thread itself runs in place
  (`a_hand_off_from_the_mlx_thread_runs_in_place`). A second hand-off from
  there would wait forever for the mailbox `turn`, which the thread that
  posted the outer job holds until that job is done.

So an array built on any thread evaluates from any thread, and a new loader,
cache or blocking-pool entry point needs no stream set-up or evaluation before
a hand-over. `a_cpu_op_built_on_one_thread_evaluates_on_another` and
`a_gpu_op_built_on_one_thread_evaluates_on_another` hold this; the
thread-boundary suite (`crates/rmlx-server/tests/thread_boundary*.rs`) holds
the production hand-overs on real models.

Cost: each `Array::eval`, `async_eval` and `Closure::apply` from another
thread is one hand-off. Median, release build, `eval` of an available array:
2 to 4 µs. Decode makes 110 per token on
Ternary-Bonsai-8B and 103 on gemma-4-e2b at the default KV quant; one
`async_eval` of the K and V buffers is one hand-off, not two.

### Null sentinel for optional arguments

An absent optional `mlx_array` argument is a handle with `ctx = null`.
`rmlx-mlx` keeps one process-global sentinel in `EMPTY_ARRAY_SENTINEL:
OnceLock<Array>`, returned by `null_sentinel()`. It is never freed and must
never reach a function that frees or materialises it. Uses: `weight` in
`rms_norm`, `freqs` in the `rope` family, `biases` in `quantized_matmul`,
`dequantize` and `gather_qmm`, `lhs_indices` in `gather_qmm`, `mask` and
`sinks` in `scaled_dot_product_attention`, and `global_scale` in `quantize` /
`dequantize`.

### Mode string cache

`mode_to_cstr` returns a `Cow<'static, CStr>` for a mode string. The fixed
set (`affine`, `mxfp8`, `mxfp4`, `nvfp4`, the SDPA mask modes, the `constant`
pad mode) is cached in `OnceLock<CString>`s. Any other string is allocated
per call.

---

## Array Lifetime and Ownership

### Type

`Array` wraps one `sys::mlx_array`, a ref-counted
`std::shared_ptr<mlx::core::array>`. Each `Array` owns exactly one handle;
`Drop` calls `mlx_array_free`. `Array` is `Send + Sync`, because mlx-c arrays
are ref-counted with `shared_ptr` semantics.

### Construction

| Method | mlx-c call | Ownership |
|---|---|---|
| `Array::from_bytes(data, shape, dtype)` | `mlx_array_new_data` | MLX copies the buffer; `data` need not outlive the call. |
| `Array::from_safetensor_view(view)` | `mlx_array_new_data` | Copied from the mmap view. |
| `scalar_f32(v)` | `mlx_array_new_float` | MLX owns the scalar. |
| `Array::try_clone(&self)` | `mlx_array_set` | Ref-count increment, no data copy. |

`Array::from_bytes` checks `data.len() == product(shape) * dtype.itemsize()`
before calling into C.

### Evaluation (lazy graph)

MLX ops build a graph; evaluation runs it.

- `Array::eval()` wraps `mlx_array_eval` and blocks until the array exists.
- `async_eval(&[..])` calls `mlx_async_eval` once for all its arrays and
  returns at once; `Array::async_eval()` is the one-array form. A later
  `eval()` or `to_bytes()` waits. The decode loop uses it to queue the next
  forward while the current argmax is read back, as mlx-lm does.

**Both are serialised process-wide by `EVAL_LOCK`**
(`crates/rmlx-mlx/src/lib.rs`), through `with_eval_lock`, which holds the lock
across the FFI call and nothing else. `Closure::apply` is too. The lock is
taken on the MLX thread ([above](#the-mlx-thread)), which already runs one
evaluation at a time; the lock keeps serial evaluation a property of the crate
even if an evaluation leaves that thread.

`with_eval_lock` takes a closure instead of returning a guard, so
`let _ = acquire();` cannot drop the guard before the FFI call. `async_eval`
still pipelines: only the graph walk and dispatch hold the lock.

### Which C entry points need the lock

Twenty-five. `scripts/check_eval_lock.sh` records how they were derived; re-run
it when the pin moves. Its automated pass finds 24; it cannot follow the
`std::function` call through which `mlx_closure_apply` evaluates, so do not
delete the closure guard.

| Entry point | Count | Why it evaluates | Called here |
|---|---|---|---|
| `mlx_array_eval` | 1 | directly | yes, guarded |
| `mlx_async_eval` | 1 | directly | yes, guarded |
| `mlx_closure_apply` | 1 | building the fused `Compiled` primitive bakes scalar constants into the kernel name: `print_constant` → `array::item<T>()` → `array::eval()` (`mlx/backend/common/compiled.cpp`) | yes, guarded |
| `mlx_eval` | 1 | directly | no |
| `mlx_array_item_*` | 14 | `array::item<T>()` evaluates before reading | no |
| `mlx_array_tostring` | 1 | `operator<<(ostream&, array)` evaluates | no |
| `mlx_save`, `mlx_save_writer`, `mlx_save_safetensors`, `mlx_save_safetensors_writer`, `mlx_save_gguf`, `mlx_load_gguf` | 6 | serialisation materialises first | no |
| `mlx_array_data_*` | — | does not evaluate; plain pointer accessors | yes, unguarded and correct |

The 22 uncalled entry points are the live risk. `mlx_array_item_float32`
reads as a scalar accessor, `mlx_array_tostring` is what an `impl Debug`
reaches for, and `mlx_save_safetensors` is the write side of `rmlx convert`.
Calling one unguarded makes that evaluation unserialised.

A compiled-closure body runs inside `mlx_closure_apply` with the lock held.
The mutex is not reentrant, so a body must not take the lock: no `eval`, and
no `Closure::apply` of another compiled closure either.

### The three things that hold this together

| | Kind | Catches | Misses |
|---|---|---|---|
| `make check-eval-lock` | text gate, deterministic | an unguarded call to any of the 25 (RULE 1, RULE 2); a closure body that takes the lock (RULE 3) | a lock that no longer locks |
| `under_eval_lock_excludes_concurrent_callers`, `with_eval_lock_holds_the_lock_while_its_job_runs` | unit tests, deterministic | a lock that does not exclude two callers; a `with_eval_lock` that runs its job without the lock | which FFI calls go through `with_eval_lock` |
| `make eval-lock-stress` | end-to-end driver, 400 threads per process | wrong values from concurrent evaluation under the lock | no defect is measured on the linked MLX, so its detection power there is unknown |

The gate and the unit tests are complementary: each is blind to what the other
catches. `make ci` runs both. The hosted `source gates` job runs the gate and
its fixtures (`make check-eval-lock-fixtures`); it runs no `cargo test`. The
stress driver is not in `make ci`; its reproducer
(`concurrent_first_eval_reproducer`) carries `#[ignore]`.

**Never `eval()` a kernel's inputs before dispatching it.** `Array::eval()`
blocks the host until the GPU produces the array. Inside a per-layer
dispatcher that runs the forward one layer at a time with nothing queued
ahead. Output is byte-identical, so the only symptom is a low decode rate.

Pass lazy arrays to `MetalKernel::apply`:

- **Ordering.** `apply` enqueues an MLX `fast::CustomKernel` graph node. MLX
  runs its `eval_gpu` only after every input is materialised, and applies the
  `ensure_row_contiguous` copy inside it. A caller-side `eval()` buys no
  ordering.
- **Layout.** `eval()` materialises but does not relayout: an evaluated
  transpose is still a strided view. Layout comes from `reshape` plus
  `ensure_row_contiguous`.

`make check-no-kernel-input-eval` enforces this across every
custom-kernel dispatcher and shared `*_common.rs` scaffold in the KV codec
layer. It keys on a file constructing a `MetalKernelInvoke`, not on a codec
name. An `// eval-ok: <reason>` marker exempts one call, such as a host
readback before `to_bytes()`. `make check-no-kernel-input-eval-fixtures` is
its recall test.

### A kernel returns the dtype it was given

An MSL dispatcher declares its output dtype
(`invoke.add_output_shape(&[n], Dtype::F32)`). f32 is often right inside the
kernel, for softmax, FWHT and log-sum-exp accumulation. Returning it is not:
MLX promotes any binary op between that f32 result and a bf16 tensor. An
attention output feeds the residual add, so the whole next layer and the
sampler run in f32, with no error. Quantization parameters behave the same:
f32 scales handed to `quantized_matmul` or `dequantize` promote the graph.

Two checks hold the rule:

- `make check-kernel-dtype-contract` (in `make ci`) scans every file that
  constructs a `MetalKernelInvoke` in the three crates of
  `scripts/metal_dirs.sh`. A function declaring an f32 output must cast to a
  *derived* dtype (`.astype(x.dtype(), …)`, `.astype(out_dtype, …)`) or carry
  `// f32-out-ok: <reason>` naming a consumer that cannot promote. Codec
  buffers read back only by our own kernels are the legitimate case.
  `make check-kernel-dtype-contract-fixtures` is its recall test.
- `crates/rmlx-kv-quant/tests/kv_decode_dtype_contract.rs` (GPU, `#[ignore]`)
  runs every `ALL_KV_QUANTS` codec through a decode step, with the default
  policy and with every fused kernel on. It asserts the attention output keeps
  the query's dtype, and a non-zero dispatch count per kernel family.

### Data readback

`Array::to_bytes()` evaluates, then copies the buffer behind
`mlx_array_data_uint8` into a `Vec<u8>`. Callers need no `eval()` first. It
returns `Err` on a null pointer.

**The contract is the array's logical elements in row-major order, for any
input.** A transpose, a strided slice and a broadcast evaluate to the parent's
buffer with adjusted strides, while `mlx_array_nbytes` reports the logical
size. A linear read of such a view returns wrong values, or reads past the end
for a broadcast.

So `to_bytes` reads `_mlx_array_is_row_contiguous` after evaluating. When it
is false, it relays the array out with `contiguous()` on the CPU stream and
reads the copy. The order is eval, classify, read: before evaluation the flag
answers `true` for a transpose. A dense array pays one extra flag read.

`_mlx_array_is_row_contiguous` is an internal mlx-c function with no stability
promise. Recomputing it from strides and shape would duplicate MLX's
definition and drift. `layout_flag_classifies_views_once_they_are_evaluated`
(`crates/rmlx-mlx/src/lib_tests.rs`) pins its answers; re-run it when the pin
moves. The non-contiguous readback cases in the same file (strided slice,
rank-2 window, transpose, the `[b, seq, kv_h, head_dim]` permutation,
broadcast) and `reshape_of_a_transposed_view_is_relaid_out_and_says_so` pin
the rest.

A caller never needs `.contiguous()` just to make a readback correct. Keep it
where it compacts a slice or feeds a raw-linear kernel.

---

## Core Ops

Every op in `src/ops/` and `fast_ops.rs` installs the error handler, allocates
an output handle, calls mlx-c inside `with_stream(device, …)`, and converts
the status with `check_status`.

| Module | Ops |
|---|---|
| `ops/arith.rs` | elementwise math, reductions, `concatenate`, `topk`, `argsort`, `argpartition`, `take_along_axis`, `zeros`, `scatter_add`, `argmax`, `broadcast_to` |
| `ops/matmul.rs` | `matmul`, `quantized_matmul`, `dequantize`, `quantize`, `quantize_mode`, `gather_qmm` |
| `ops/activation.rs` | `gelu`, `gelu_tanh`, `silu`, `tanh`, `softmax`, `softmax_precise` |
| `ops/shape.rs` | `conv1d`, `conv2d`, `conv_transpose1d`, `pad`, `tril`, `arange`, `sin`, `cos`, `maximum` |
| `Array` methods | `slice`, `slice_update`, `take`, `reshape`, `transpose`, `astype`, `contiguous` |

- `quantized_matmul` takes `mode` `"affine"` or an fp mode (`"mxfp8"`, …).
  `biases` is `None` for the fp modes. MLX rejects the legacy `"default"`.
- `dequantize` gives the on-device embedding lookup.
- `gather_qmm` is the batched MoE expert matmul.
- `slice_update` is the write path for pre-allocated KV buffers.

---

## Fast Ops (Fused Metal Kernels)

`fast_ops.rs` wraps the `mlx_fast_*` family.

### `rms_norm`

`x / sqrt(mean(x^2) + eps) * weight`. `weight` is `None` for no-scale norms
(Gemma4 `v_norm`).

### `rope` / `rope_dynamic` / `rope_with_freqs` / `rope_with_freqs_dynamic`

| Variant | Offset | Frequencies |
|---|---|---|
| `rope` | `i32` | base theta |
| `rope_dynamic` | `Array` (0-D i32) | base theta |
| `rope_with_freqs` | `i32` | explicit `[dims/2]` table |
| `rope_with_freqs_dynamic` | `Array` (0-D i32) | explicit table |

Inside a compiled closure, use a `_dynamic` variant: a captured `i32` offset
is a new literal every step and forces a retrace. Gemma4 full-attention layers
use `rope_with_freqs` for proportional RoPE; `base` is then ignored.

### `scaled_dot_product_attention`

Wraps `mlx_fast_scaled_dot_product_attention`. `q`, `k`, `v` are
`[batch, n_heads, seq_len, head_dim]`. `mask_mode` is one of:

- `"causal"`: the kernel masks internally;
- `"array"`: `mask_arr` is an additive mask (sliding-window masks use this);
- `""`: no mask.

`sinks` is always the null sentinel.

It is not always a FlashAttention kernel. `head_dim` decides, silently,
whether the call reaches a fused kernel or a composite graph.

#### Head-dim dispatch and the unfused fallback

`ScaledDotProductAttention::use_fallback`
(`mlx/backend/metal/scaled_dot_product_attention.cpp`, v0.32.3) gates on
`head_dim`, `q_seq`, `kL`, the GQA factor and the mask:

| Route | `head_dim` | Conditions |
|---|---|---|
| `sdpa_full` (fused, `q_seq > 8`) | 64, 72, 80, 96, 128, (96, 64) | the mask is absent, an array, or causal with `q_seq <= kL` (every `sdpa_full` row) |
| `sdpa_full` at 256 | 256 | NAX GPU, `q_seq >= 1024`, causal or array mask; or no NAX, causal, `q_seq >= 2048`, `q_seq == kL`, Q not f32 |
| `sdpa_full` at 512 | 512 | NAX GPU, `q_seq >= 1024`, causal, no array mask, `B × H × ⌈q_seq/32⌉ >= 1024` |
| `sdpa_vector` (fused, `q_seq <= 8`) | 64, 96, 128, 256, (192, 128), (96, 64) | `q_seq <= kL` and `q_seq × GQA factor <= 32`; GQA factors 8, 12 and 16 read K/V once in a two-pass kernel |
| `sdpa_vector` at 512 | 512 | `q_seq == 1`, GQA factor 8, no array mask, `kL >= 1024` (`MLX_SDPA_D512_MIN_KL`) |
| composite graph | any | whatever the rows above reject, including 192 |

So at GQA factor 8 a call with 5 to 8 query rows, a speculative verify round
of that size for one, takes the composite graph.

The composite route is `matmul(q, kᵀ)` → mask → `softmax` → `matmul`, with
the `[B, n_heads, L_q, L_k]` score tensor materialised.

| Family | Windowed / linear layers | Full-attention layers |
|---|---|---|
| Ternary-Bonsai-8B (`Qwen3ForCausalLM`) | none | 128 |
| gemma-4 | 256, sliding window | 512 |
| medgemma (`Gemma3…`) | 256, sliding window | 256 |
| Qwen3.5 / Qwen3.6 (`Qwen3_5…`) | GatedDeltaNet, no SDPA | 256 |

So a gemma-4, medgemma or Qwen3.5 prefill chunk of 1024 or more rows reaches
the fused 256 kernel on an M5, and the 512 layers of gemma-4 reach a fused
kernel only inside the gates above. The composite path computes the whole
causal score rectangle and then masks it, and it holds the score tensor in
memory. `scripts/sdpa_headdim_bench.py` measures the cost against the metallib
it reports.

What rMLX pays on the composite path is bounded two ways:

- Prefill is chunked per arch (`prefill_chunk.rs`: gemma-4 1024, Qwen3.5
  2048), so the score tensor is `[H_q, chunk, kL]`, linear in `kL`.
- On gemma-4 the 256-wide layers are sliding-window, so their `kL` is capped
  at the window.

At decode, `q_seq = 1`, so the composite path has no O(L²) term.

---

## Compiled Kernels (`compile` module)

`compile.rs` wraps `mlx_compile` / `mlx_closure`, the equivalent of
`@mx.compile`.

### `Closure`

An owned handle around `mlx_closure`, freed on drop, `Send + Sync`.
`Closure::from_fn` takes any
`Fn(Vec<Array>) -> Result<Vec<Array>> + Send + Sync + 'static`. It boxes the
function and passes the raw pointer as the closure payload;
`rust_closure_callback` is the C trampoline.

- The callback runs the body under `catch_unwind`. A panic becomes a non-zero
  return and a `tracing::error!`; it never crosses the FFI boundary.
- The C++ side hands the callback `*output` with a null ctx, and appending to
  it is undefined. The callback builds a new vector, appends, then overwrites
  `*output`; the C++ side frees it.

### `compile` / `compile_shapeless`

Both consume a `Closure` and return a compiled one that replays the cached
program.

- `compile` re-traces when input shapes change.
- `compile_shapeless` keeps one program for every shape. Use it for ops called
  at varying sequence lengths, such as chunked prefill.

---

## MSL Kernels (`metal_kernel` module)

`metal_kernel.rs` wraps mlx-c's `mlx_fast_metal_kernel`, which JIT-compiles
MSL bodies and dispatches them inside the MLX graph.

### `MetalKernel`

A kernel handle, freed on drop with `mlx_fast_metal_kernel_free`, and
`Send + Sync`: it is immutable after `new`. Callers hold the process GPU claim
before dispatching.

`MetalKernel::new` builds the name and argument-name vectors and calls
`mlx_fast_metal_kernel_new`; mlx-c copies the strings. Every kernel is created
with `ensure_row_contiguous = true` and `atomic_outputs = false`.
`ensure_row_contiguous` makes MLX copy any non-row-contiguous input before the
dispatch, so a body may index its buffers by linear offset. It does not fix a
*semantic* layout mismatch; that is what the sequence-major KV store layout is
for.

`MetalKernel::apply` consumes a `MetalKernelInvoke`, dispatches through
`mlx_fast_metal_kernel_apply`, and returns the output arrays.

**MLX JIT language version.** MLX compiles custom kernel bodies at Metal 4.0.
`rmlx_nax_probe_gpu` (`crates/rmlx-mlx/src/metal_kernel_tests.rs`, run by
`make gpu-test`) reads `__METAL_VERSION__` from inside a JIT'd body and gets
`400`. It also sees `__HAVE_TENSOR__ == 1`, and a
`constexpr matmul2d_descriptor(8, 32, 128, …)` instantiates with `.m == 8`.
That last value shows the `<MetalPerformancePrimitives/…>` include survives
MLX's source wrapping.

So `mpp::tensor_ops` is reachable from an rMLX kernel body. mlx-c exposes no
compile options, so the version is observed, not forced: re-run the probe after
an MLX bump. The compile gate checks every body at `metal3.1` and `metal4.0`
(see "MSL gates").

**Lazy compile.** `MetalKernel::new` only registers the kernel; the pipeline
compiles on the first `apply()`. The KV layer warms its shader-heavy codecs at
model load with `rmlx_kv_quant::precompile::precompile_kv_codec_msl`; see
`docs/KV_QUANT.md` § "Metal-vs-CPU hot path + load-time MSL precompile".
`gdn_warmup` (`rmlx-models::arch::loader`) warms the GatedDeltaNet graph.

### MSL source conventions

`source` is the kernel **body**. MLX adds the signature and buffer
declarations:

- Inputs are `device const T* <name>`, typed from the array's dtype, except
  that MLX binds a small input (fewer than 8 elements, observed) in the
  `constant` address space. A helper hard-declared `device const T*` then
  fails the JIT compile on first dispatch. Pad such an input past the trip
  point, as `rmlx_kv_quant::flash_decode_common::pad_norms_to_device_floor`
  does (floor 16), or template the helper over the pointer type.
- Outputs are `device T* <name>`.
- Built-ins: `thread_position_in_grid`, `threadgroup_position_in_grid`,
  `thread_position_in_threadgroup` (all `uint3`).
- `header` is MSL placed before the kernel, for `constant` arrays and helper
  functions.

### `MetalKernelInvoke`

A builder for one dispatch. `add_input` takes a ref-counted clone.
`add_output_shape` declares an output. `set_grid` and `set_thread_group` set
the 3-D geometry. `set_template_int` and `set_template_dtype` specialise the
body at JIT time. `set_init_value` initialises outputs before the kernel runs,
which a kernel that accumulates with `atomic_fetch_or_explicit` needs: MLX may
hand it a pooled buffer with stale contents.

### Where MSL lives

Every MSL kernel body a production path can dispatch lives in a `.metal` file,
never in a Rust string literal:

| Directory | Scope |
|---|---|
| `crates/rmlx-kv-quant/src/metal/` | every KV codec kernel, dispatched from `src/*_msl.rs` and `src/sparse_attn/*_msl.rs` |
| `crates/rmlx-models/src/metal/` | GatedDeltaNet and weight-side ParoQuant, from `gated_delta_msl.rs` / `paroquant_msl.rs` |
| `crates/rmlx-mlx/src/metal/` | the MLX JIT language-version probe; not a production kernel |

The gates are scoped by directory, not crate: a kernel outside a listed
`metal/` directory is not gated. Throwaway `#[cfg(test)]` bodies stay inline.

Each module registers its kernels once, as `OnceLock<MetalKernel>`
singletons, and each body matches the CPU reference in its `*quant.rs` file.

### `.metal` files + `include_str!`

Bodies are embedded at compile time, never read at run time:

```rust
const QUANTIZE_SOURCE: &str = include_str!("metal/q8_quantize.metal");
```

A `.metal` file holds a body only. The header comes from one of two places:

- a static header `.metal` file (`turboquant_header.metal`, …);
- a `build_*_header(..)` Rust function that emits `constant` / `#define`
  declarations from computed values (codebooks, rotation constants, eps).

An `#include`, or an `inline` helper, belongs in the header: MLX splices the
body into the kernel function, where neither compiles. The shared code-plane
reader (`crate::code_plane::render_msl_code_plane`) is emitted into each
codec's generated header for that reason.

Parameterised bodies use one of two mechanisms:

- one `.metal` file per variant, when the variants differ in code
  (`planar_fused_qk_b3.metal` / `_b4.metal`,
  `rot_k_fwht_quantize_d{32..512}.metal`);
- MLX template arguments, when they differ only in a constant
  (`gated_delta_step.metal`: `Dk`/`Dv`/`Hk`/`Hv`; `paroquant_rotate.metal`:
  `ROWS_PER_TILE`/`MAX_KROT`/`MAX_GROUP_SIZE`). The bound then lives in the
  Rust const the validation already uses.

A body's text is never rewritten at run time; a `.replace("{X}", ..)` makes
the file uncompilable by the gate.

A new KV codec ships a `.metal` decode kernel and its native compile check;
see `CLAUDE.md` hard rule 10.

### MSL gates (`make ci`, enforced in CI)

`scripts/metal_dirs.sh` lists the three kernel directories, and both gates
source it. The `check-metal-format` pre-commit hook triggers on any
`crates/*/src/metal/*.metal` file but checks only the listed directories. A
crate that starts shipping MSL must be added to the list; nothing else
discovers it.

| Target | Tool | Checks |
|---|---|---|
| `make check-metal-compiles` | `xcrun -sdk macosx metal` | Every body compiles natively at `-std=metal3.1` and `-std=metal4.0`. Fails on a `.metal` file its directory's `probes/kernels.manifest` does not name. |
| `make check-metal-format` | `clang-format` (`PATH` or `xcrun -f clang-format`) | Every kernel is clean against its directory's `.clang-format`. |

`metal4.0` is what production compiles at. `metal3.1` is the floor; it is not
3.0 because `bfloat`, the element type of the codecs' scale and norm planes,
is a Metal 3.1 type. A `#if __HAVE_TENSOR__` body is undefined below 4.0, so
it is compiled at 4.0 or reported `SKIP` and counted, never passed silently.
The capability probe asserts the guard and the tensor includes, not just that
the driver accepts `-std=metal4.0`.

A missing tool, or a toolchain without the 4.0 pass, fails under `--strict`
and prints a counted notice otherwise. The `msl` job in
`.github/workflows/ci.yml` runs both gates with `METAL_STRICT=--strict`,
where an empty file set also fails. Compiling MSL needs the toolchain, not a
GPU. Locally, install full Xcode; on Xcode 16.3+ run
`xcodebuild -downloadComponent MetalToolchain`.

A body is not a translation unit, so the compile gate assembles a probe per
kernel: stdlib preamble, header, then a kernel wrapping buffer aliases,
defines and the body. The manifest gives, per body, the header, the buffer
names and types (`u`, `i`, `f`), and optional `#define NAME VALUE` pairs for
values MLX injects at dispatch (template dtypes and ints, 0-D scalar inputs).
A `#define` that copies a Rust const is pinned by an equality test, as
`probe_manifest_defines_match_rust_consts` does.
`crates/rmlx-kv-quant/src/metal/probes/README.md` documents the layout and how
to refresh the captured header snapshots.

---

## Unsafe Policy

The workspace denies `unsafe_code` and `unsafe_op_in_unsafe_fn`
(`[workspace.lints.rust]`). A file that needs `unsafe` opts in with a
module-level `#![allow(unsafe_code)]` and a comment giving the reason:

```rust
// unsafe_code: mlx-rs FFI bridge — <per-file justification>
#![allow(unsafe_code)]
```

In `rmlx-mlx` that is the FFI. Elsewhere it is mostly zero-copy byte
reinterpretation (`slice::from_raw_parts`) of array data. Because
`unsafe_op_in_unsafe_fn` is denied, every unsafe operation inside an
`unsafe fn` needs its own block and its own `// SAFETY:` comment.

### SAFETY contracts by pattern

| Site | Contract |
|---|---|
| `Array::from_bytes` / `mlx_array_new_data` | MLX copies the buffer; `data` need not outlive the call. |
| `mlx_array_shape`, `mlx_array_data_uint8` | The pointer is valid while the `Array` lives; it is copied into a `Vec` before return. |
| `null_sentinel` | The null handle is only an "absent" argument to an mlx-c function that accepts null. Never store or materialise it. |
| `Stream` (`mlx_thread.rs`) `Send + Sync` | The handle names an immutable `{device, index}` and is never freed. Only the MLX thread uses the encoder behind it. |
| `check_status` | Call immediately after the mlx-c call, on the same thread, before another call can overwrite the error slot. After `with_eval_lock`, the error slot of the calling thread holds the message of the call on the MLX thread. |
| `JobRef` (`mlx_thread.rs`) `Send`, and its call in `serve` | `hand_off` makes it from a live `Send` closure on its own stack, holds `turn` so that the next `done` is for its job, and does not use or drop the closure until it sees `done`: nothing from the post to the end of the wait returns or unwinds. |
| `rust_closure_callback` | `payload` is the boxed function, valid for the closure's life. `input` is borrowed and not freed; `output` is filled here. No panic crosses the boundary. |
| `MetalKernel` / `Closure` `Send + Sync` | The handle is immutable and ref-counted by mlx-c. The Metal device context is process-global; callers hold the Metal claim (`crates/rmlx-server/src/claim.rs`). |

---

## See also

- `docs/KV_CACHE.md`: the KV cache built on `slice_update` and the MSL codec
  kernels.
- `docs/WEIGHT_QUANTS.md`: weight formats behind `quantized_matmul`,
  `dequantize` and `gather_qmm`.
- `docs/KV_CODECS.md`: how the kernels in `rmlx-kv-quant/src/*_msl.rs` map to
  each codec.
