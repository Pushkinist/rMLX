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

#### Where NAX can appear, and where it cannot

The pin buys the NAX **GEMM** path (`steel_gemm_fused_nax_*`), which every
model's matmuls go through. NAX **attention** is a far narrower thing, and the
two are easy to conflate:

| Path | NAX? | Why |
|---|---|---|
| Prefill attention, `head_dim` 64 or 128, Q not f32 | **yes**, `steel_attention_<dtype>_bq64_…` | MLX's `sdpa_full` takes the NAX branch unless `head_dim == 80` or Q is f32 without TF32 |
| Prefill attention, `head_dim` 256 or 512 | no | MLX has no fused prefill kernel at either width, NAX or otherwise. See [Head-dim dispatch](#head-dim-dispatch-and-the-unfused-fallback) — that gap is about a missing kernel *shape*, not about NAX |
| Decode attention, any `head_dim`, any codec | no | `bq64` is a query tile of 64, and decode is `q_seq = 1`. At `head_dim` ≤ 256 MLX routes `q_seq <= 8` to `sdpa_vector`, which has no NAX variant; at 512 there is no vector kernel either and decode falls to the composite path. Neither reaches NAX |
| Our own `.metal` kernels | no | all of them are `q_seq = 1` decode kernels. NAX's one matmul tile shape has an M floor of 16; at `q_seq = 1` the only M available is `heads_per_kv` (4–8 on our models), so the tile can never be more than half filled |

So the bf16 mirror's decode advantage over a quantized codec is bandwidth and
kernel quality, **not** NAX — no decode path on any codec reaches it, and a
hand-written decode kernel could not either. Only prefill is in play, and only
for a 64- or 128-wide head.

Measured on this host (M5 Max, mlx 0.31.2), from the per-pipeline binary
archives inside a GPU capture — the whole `mlx.metallib` is also embedded in a
bundle, so only the small per-pipeline archives are evidence of a *created*
pipeline:

- Ternary-Bonsai-8B (every layer `head_dim` 128) creates three
  `steel_attention_bfloat16_bq64_bk32_bd128_wm4_wn1_mask*` pipelines — the
  causal, aligned-masked and unaligned-masked prefill permutations. NAX is
  engaged and Q is already bf16, so there is no f32 gate to fix.
- gemma-4-e2b creates none, on the same tooling and prompt size. Its sliding
  layers are 256-wide and its full-attention layers 512-wide, and MLX ships no
  fused attention kernel at either. It does create `steel_gemm_fused_nax_*`,
  i.e. NAX GEMM is live for a model whose attention can never use NAX.

Which widths reach which kernel, and what the fallback costs, is
[Head-dim dispatch](#head-dim-dispatch-and-the-unfused-fallback).

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

### Runtime NAX capability (`src/nax.rs`)

A version match is not a capability match, and neither survives distribution.
`RMLX_MLX_NAX` describes the machine that *built* the binary; a Homebrew bottle
or release tarball links `libmlx.dylib` through the moving `opt` symlink, so it
runs against the installing user's MLX, which may be a different bottle of the
same version. `src/nax.rs` therefore repeats the metallib scan at run time, on
the same one-shot init as the skew warning, against the library **dyld actually
loaded** (found by walking dyld's image list for `libmlx.dylib`, then reading
its colocated `mlx.metallib`).

**Host-class gated, and that gate is the point.** The kernels only exist for
the GPU Neural Accelerator, which arrives with M5 — Apple GPU family 10 in
`rmlx_core::apple_gpu`. Every earlier generation legitimately ships zero of
them at every MLX version, so a warning there would be noise on the majority
of Macs and would train people to ignore the one host where the absence costs
something. The gate runs ahead of every other step, and when it says "no Neural
Accelerator" neither the dyld image walk nor the metallib open happens.

| Host | Kernels | Result |
|---|---|---|
| M5+ | absent (confirmed) | `warn!` — names prefill/TTFT, the metallib, and the check command |
| M5+ | present | `debug!` only |
| M5+ | metallib unreadable / not found | `debug!` only — "could not look" is not "absent" |
| M1–M4, or chip unidentifiable | either | `debug!` only; **no scan at all** |

Measured cost on the release binary (M5 Max, warm page cache, best of 5):
**~0.7 ms** when the kernels are present — the first match lands a couple of MB
into the 158 MB metallib and ends the scan — and **~49 ms** for a full pass over
a metallib of the same size with none. On a pre-M5 host it is **~1.3 µs**, which
is the `sysctl` chip query and nothing else: the dyld image walk (~220 ns) and
the file open are both behind the gate. It runs once per process, on the init
that already precedes a multi-second model load.

The wording is scoped to **prefill / TTFT** deliberately: NAX is unreachable at
decode by construction (see "Where NAX can appear, and where it cannot"), so a
warning implying a general slowdown would be wrong.

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

A worker thread whose graph includes a CPU-scheduled op — e.g. the K8V8
`exit_prefill` quantization's scale reduction, which MLX places on the CPU
stream even though the surrounding quantize dispatch runs on the GPU device —
needs a CPU stream of its own, the same way the GPU path did.

**How MLX resolves the CPU encoder is version-specific, and the two versions
behave oppositely — do not reason from the wrong one:**

| | 0.31.x (**the pinned version**) | 0.32.0 |
|---|---|---|
| `cpu::get_command_encoder` map | one **process-global** `unordered_map<int, CommandEncoder>` | `thread_local`, with a process-global fallback |
| Populated | lazily, on first evaluation, **with no synchronisation** | at stream registration |
| Unregistered stream | silently inserted | throws `There is no Stream(cpu, N) in current thread.` |
| Cross-thread eval | succeeds | throws |

Default CPU streams are per-thread either way (`mlx/stream.cpp`:
`static thread_local ... default_streams`), so on the pinned version every
thread that evaluates mints its own stream index and inserts into that one
shared map. That unsynchronised insert is a genuine upstream defect — the
neighbouring `Scheduler::threads_` map in `mlx/scheduler.h` *is* mutex-guarded —
and it is what `EVAL_LOCK` (below) contains. MLX 0.32.0 fixes it by making the
map thread-local; we do not pin that version because its bottle ships no NAX
GEMM kernels.

`rmlx_mlx::ensure_cpu_default_stream()` is the CPU analog of
`ensure_gpu_default_stream()`: same mechanism (create a stream, register it as
the calling thread's default, leak the handle in a thread-local for the
thread's lifetime), same idempotency guarantee, zero ML-semantic effect. On
0.31.x it is not load-bearing for a thread that builds and evaluates its own
graph — MLX self-registers there — but it pins the thread's stream identity
explicitly and is what keeps the code correct if the pin moves to 0.32.0.

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
mechanism this guards. Note the guard's scope: it registers the *worker's own*
stream. On the pinned 0.31.x that is all anyone needs — a cross-thread eval
resolves through the process-global map and succeeds
(`cross_thread_eval_resolves_through_the_process_global_encoder_map` pins this).
On 0.32.0 a cross-thread eval throws and the guard would not help, because it
registers a different stream than the one the foreign array is bound to.

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

**Both are serialised process-wide by `EVAL_LOCK`** (`crates/rmlx-mlx/src/lib.rs`),
a `Mutex<()>` held across the FFI call and nothing else. MLX evaluation is not
safe to drive from two threads at once on the pinned 0.31.x: the CPU
command-encoder table described above is a process-global map filled without
synchronisation, so two concurrent evaluations rehash it under each other and
the process takes SIGSEGV inside MLX — no Rust frame at fault, no failing test
reported, nothing to catch. `cargo test` runs one OS thread per test, which is
how this reached `make ci` as an intermittent crash of the whole test binary.

Three consequences worth knowing:

- **Cost is one uncontended atomic per evaluation.** rMLX runs inference on one
  thread at a time by design (and the server has its own `gpu_gate` above that),
  so the lock is a crash guard, not a throughput bottleneck.
- **`async_eval` still pipelines.** Only the graph walk and dispatch happen
  under the lock; the scheduled work completes after it is released.
- **The lock is not a licence to evaluate concurrently.** It makes concurrent
  callers correct, not parallel — they serialise.

New FFI entry points that can reach `mlx::core::eval_impl` must take the same
lock. Today that is exactly `mlx_array_eval` and `mlx_async_eval`; mlx-c's
data accessors (`mlx_array_data_*`) do not evaluate.

**Never `eval()` a kernel's inputs before dispatching it.** `Array::eval()`
blocks the calling thread until the GPU has produced the array, so an `eval()`
inside a per-layer dispatcher runs the forward pass one layer at a time with
nothing queued ahead — the host and the GPU stop overlapping. It produces
byte-identical output, which is why the cost hides: it shows up only as a
decode rate several times below what the kernel can reach. A KV flash-decode
dispatcher paying this on every attention layer measured **2.7× below** its own
rate on Ternary-Bonsai-8B once the `eval()` calls were dropped, with the token
digest unchanged.

Pass lazy arrays to `MetalKernel::apply` and let MLX schedule the graph. Two
facts make that safe, and both are worth stating because the comment this rule
replaced got each of them wrong:

* **Ordering.** `MetalKernel::apply` enqueues an MLX `fast::CustomKernel` graph
  node; it does not dispatch. MLX runs that node's `eval_gpu` only once every
  input edge is materialised, and applies the `ensure_row_contiguous` copy
  (below) inside that same `eval_gpu`. A kernel cannot read an uncomputed or
  strided buffer, so a caller-side `eval()` buys no ordering — it only changes
  *when* the host blocks.
* **Layout.** `Array::eval()` materialises but does **not** relayout. MLX's
  `Transpose` is a strided view over a shared buffer, so an evaluated transpose
  is still non-row-contiguous. The layout guarantee for a raw-linear kernel
  always came from `reshape` plus `ensure_row_contiguous`, never from forcing
  evaluation.

The CI gate `make check-no-kernel-input-eval` enforces this across every
custom-Metal-kernel dispatcher and shared dispatcher scaffold in the KV codec
layer (keyed on the file constructing a `MetalKernelInvoke`, or being a
`*_common.rs` scaffold — not on a codec name), with an `// eval-ok: <reason>`
marker for a genuinely load-bearing call such as a host readback before
`to_bytes()`. One marker exempts one call.
`make check-no-kernel-input-eval-fixtures` is the gate's own recall test: a
fixture tree per evasion (UFCS `Array::eval(&x)`, an eval moved into a shared
`_common.rs` helper, a dispatcher relocated into a sub-directory, a marker
leaking onto a following loop) pinned to the exit code the gate must produce.

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

Wraps `mlx_fast_scaled_dot_product_attention`. Arguments:

- `q`, `k`, `v`: `[batch, n_heads, seq_len, head_dim]`.
- `scale`: `1/sqrt(head_dim)` or `1.0` (Gemma4 uses `1.0`).
- `mask_mode`: `"causal"` (kernel handles masking internally, fastest),
  `"additive"` (caller supplies additive mask), or `""` (no mask).
- `mask_arr`: ignored when `mask_mode = "causal"`; null sentinel when `None`.
- `sinks`: always null sentinel (not used at this stage).

It is **not** unconditionally a FlashAttention kernel. Whether the call reaches
a fused kernel or a composite graph is decided by `head_dim`, silently.

#### Head-dim dispatch and the unfused fallback

`ScaledDotProductAttention::use_fallback`
(`mlx/backend/metal/scaled_dot_product_attention.cpp:618-636`, v0.31.2) gates
on `head_dim` and on `q_seq`:

| Route | `head_dim` accepted | Other conditions |
|---|---|---|
| `sdpa_full` (fused `steel_attention`) | 64, 80, 128 | `q_seq > 8` (prefill), and the mask is absent, an array, or causal with `q_seq <= kL` |
| `sdpa_vector` (fused) | 64, 96, 128, 256 | `q_seq <= 8` (decode), `q_seq <= kL`, `q_seq × gqa_factor <= 32` |
| composite graph | any | whatever both gates reject |

The composite route is the unfused lambda at `mlx/fast.cpp:717` —
`matmul(q, kᵀ)` → mask → `softmax` → `matmul`, with the
`[B, n_heads, L_q, L_k]` score tensor **materialised**.

The shipped kernel inventory agrees with the gate:

```sh
LIB="$(brew --prefix mlx)/lib/mlx.metallib"
xcrun metal-nm --defined-only "$LIB" | grep -o 'steel_attention[a-z0-9_]*' | sort -u
# ... _bd64_ / _bd80_ / _bd128_ only — no bd256, no bd512
xcrun metal-nm --defined-only "$LIB" | grep -o 'sdpa_vector[a-z0-9_]*' | sort -u
# ... _64_64 / _96_96 / _128_128 / _256_256 — no 512
```

So above `head_dim` 128 there is **no fused prefill kernel at all**, and at
`head_dim` 512 there is no fused decode kernel either.

**Which of our models sit where** (per-layer `head_dim` from each snapshot's
`config.json`; the gemma-4 split is applied in `gemma4/loader.rs`, sliding
layers take `head_dim`, full-attention layers take `global_head_dim`):

| Family | Windowed / linear layers | Full-attention layers | Fused prefill? |
|---|---|---|---|
| Ternary-Bonsai-8B (`Qwen3ForCausalLM`) | — (all 36 are full-attention) | 128 | **yes** |
| gemma-4 e2b / e4b / 26b / 31b | 256, SWA (`kL ≤ window`) | **512** | no, at either width |
| medgemma 1.5 4b (`Gemma3…`) | 256, SWA (`kL ≤ window`) | 256 | no |
| Qwen3.6-35B-A3B, Bonsai-27B (`Qwen3_5…`) | GDN — no SDPA | 256 | no |

Gemma-4 is **not** a `head_dim` 256 model end-to-end: only its
window-bounded layers are 256 wide, and the layers whose `kL` grows with the
prompt are 512 wide. That distinction decides who actually pays.

**What the fallback costs.** Reproduce with
[`scripts/sdpa_headdim_bench.py`](../scripts/sdpa_headdim_bench.py), which
prints the metallib inventory it measured against — **re-run it when the pin
moves**, because these numbers are only valid for the kernel set above. Below:
mlx 0.31.2, M5 Max, bf16, causal, `q_seq = kv_seq = L`, best-of-5 after a
pipeline pre-warm, median of two runs. `L ≥ 8192` reproduced within 6%; the
`L = 2048` row is launch-bound and is not load-bearing.

| q:kv heads | L | `head_dim` 128 | 256 | 512 | 256 ÷ 128 | peak, 256 |
|---|---|---|---|---|---|---|
| 8:1 | 2 048 | 0.71 ms | 1.27 ms | 1.73 ms | 1.79× | 99 MB |
| 8:1 | 8 192 | 3.03 ms | 17.28 ms | 26.17 ms | **5.70×** | 1.25 GB |
| 8:1 | 32 768 | 44.5 ms | 313.0 ms | 489.9 ms | **7.03×** | 18.7 GB |
| 32:8 | 2 048 | 1.07 ms | 4.64 ms | 7.00 ms | **4.35×** | 390 MB |
| 32:8 | 8 192 | 11.9 ms | 72.0 ms | 107.8 ms | **6.06×** | 4.8 GB |
| 32:8 | 32 768 | 200.1 ms | 1 276.1 ms | 2 142.3 ms | **6.38×** | 71.7 GB |

A fused kernel present at both widths would land near 2.0×, since doubling
`head_dim` doubles the FLOPs. Every cell from `L = 2048` up at 32:8, and from
`L = 8192` up at 8:1, is past 4×. Only the smallest cell is inside the
"costs little" band, so the gap is a real cost, not a curiosity.

**Why it is more than 2×.** The two paths do not perform the same work, and
the `causal` section of the harness measures it directly — `causal ÷ unmasked`
at `L = 8192`:

| `head_dim` | 8:1 | 32:8 | reading |
|---|---|---|---|
| 128 (fused) | 0.566 | 0.511 | skips fully-masked tiles — does ~half the rectangle |
| 256 (composite) | 1.318 | 1.331 | computes the whole rectangle, then pays to build and apply the mask |
| 512 (composite) | 1.160 | 1.150 | same |

So going 128 → 256 costs 2× for the wider head **and another 2× for losing the
causal skip** — 4× the arithmetic. Normalising each path by the work it
actually performs (`2·H·L²·D` fused, `4·H·L²·D` composite), the fused path
sustains 44–49 TF/s and the composite 28–32 TF/s. That closes the measurement:
`4 × (46.2 / 30.5) = 6.06×` against 6.06× measured at 32:8 / 8192, and
`4 × (44.0 / 27.6) = 6.38×` against 6.38× at 32:8 / 32768. The composite path
is not catastrophically inefficient per FLOP — it is asked to do four times as
many, and it materialises the score tensor to do them.

Quoting a single dense-equivalent `4·H·L²·D` rate for both would overstate the
fused path by 2×; the ratios above are convention-free either way.

The 512 column is the control. Both 256 and 512 are unfused, and 512 ÷ 256
lands at 1.36–1.68× — *below* the 2.0× FLOP ideal, which is what a shared
composite path predicts: its fixed `[H, L, L]` score cost does not grow with
`head_dim`. The cliff is at the 128 → 256 boundary, not "wider heads are
slower".

Peak memory is the same story told in bytes: at `L = 32768`, 32:8, 671 MB at
`head_dim` 128 against **71.7 GB** at 256 — the materialised score tensor is
32 × 32768² × 2 B = 68.7 GB on its own.

Decode is unaffected in the way that matters: at `q_seq = 1` the score tensor
is `[H, 1, kL]`, so the composite path has no O(L²) term. In the only decode
shape that stays clear of this host's ~200 µs dispatch floor (32:8,
`kL = 32768`, three runs) the unfused 512-wide path reads KV at ≈315 GB/s
against the 256-wide vector kernel's ≈355 GB/s — a modest deficit, not a
cliff. Every other decode cell moved by up to 2.2× run to run and cannot
resolve a kernel-level difference; the harness prints them all so that stays
visible rather than being quoted selectively.

**What rMLX actually pays.** Less than the isolated `L = kL` numbers above,
for two structural reasons:

- Prefill is chunked per arch (`prefill_chunk.rs`: gemma-4 1024,
  Qwen3.5-MoE 2048), so the score tensor is `[H_q, chunk, kL]` — linear in
  `kL`, not quadratic in the prompt. It is still large in absolute terms:
  Qwen3.6-35B-A3B (16 q heads, chunk 2048) materialises 2.1 GB per
  full-attention layer per chunk at `kL = 32768`, 8.6 GB at 128k. Chunking
  bounds the growth, it does not remove the tensor.
- On gemma-4 the 256-wide layers are sliding-window, so their `kL` is capped
  at 512 / 1024 tokens no matter how long the prompt is.

The families that pay a *growing*-`kL` composite cost are Qwen3.5 / Qwen3.6
(10 of 40 layers on Qwen3.6-35B-A3B, 16 of 64 on Bonsai-27B), medgemma
(5 of 34), and gemma-4's global layers — the last at 512, where no upstream
proposal reaches.

**Upstream status** (ml-explore/mlx, checked 2026-08-16). This belongs
upstream, and it is already in flight there; none of it has merged:

| PR | What | State |
|---|---|---|
| [#3293](https://github.com/ml-explore/mlx/pull/3293) | `head_dim=256` in `sdpa_full` + a `bd=256` steel instantiation | closed, unmerged |
| [#3660](https://github.com/ml-explore/mlx/pull/3660) | revival of #3293 for 192/256, routed above `kL > 16384` | closed, unmerged |
| [#3842](https://github.com/ml-explore/mlx/pull/3842) | a NAX `bd=256` full-attention path | open |
| [#4185](https://github.com/ml-explore/mlx/pull/4185) | `force_fused` flag, which also restores 192/256 behind it | open |

#3660 was closed with "we decided to add a `force_fused` option to the API to
let users make the decision, instead of providing builtin heuristics" — so the
direction upstream is an explicit opt-in, not a wider default. **No proposal
covers `head_dim` 512**, so gemma-4's global layers stay composite regardless.

There is nothing to port until a release ships one of these. A hand-written
`bd=256` flash kernel here is not on the table: it duplicates upstream work,
and the one attention-kernel class this repo has hand-written — flash-*decode*
over a quant store — landed at 4–14% of MLX's per-byte throughput
(`docs/PERF_BASELINE.md`). That is a different kernel, but it is the only
calibration we have for what writing one here costs.

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

**MLX JIT language version.** MLX compiles custom kernel bodies at **Metal
4.0**. Observed, not inferred: `rmlx_nax_probe_gpu`
(`crates/rmlx-mlx/src/metal_kernel_tests.rs`, run by `make gpu-test`) reads
`__METAL_VERSION__` from inside a JIT'd body and gets `400`, with
`__HAVE_TENSOR__ == 1` and a `constexpr matmul2d_descriptor(8, 32, 128, …)`
instantiating to `.m == 8`. The third value is the load-bearing one: it proves
the `<MetalPerformancePrimitives/…>` include path survives MLX's source
wrapping, not merely that the macro is defined.

The consequence: `mpp::tensor_ops` **is** reachable from an rMLX kernel body, so
a prefill GEMM with a custom epilogue is a legitimate design option rather than
something gated on an MLX change. mlx-c exposes no compile-options surface (the
kernel config is seven setters in `mlx/c/fast.h`), so this is observed, never
forced — re-run the probe after an MLX bump rather than assuming it holds.

The compile gate mirrors this: it compiles every body at `metal3.0` **and**
`metal4.0`. See "MSL gates" below.

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

Every MSL kernel body lives in a `.metal` file, never in a Rust string literal.
Three directories hold them, one per crate that ships MSL:

| Directory | Scope |
|---|---|
| `crates/rmlx-kv-quant/src/metal/` | Every KV-cache codec (q8, TurboQuant, PlanarQuant, IsoQuant, RotorQuant, rot-K, TCQ, TurboFlash, fused-QK, sparse-attn phases), dispatched from `src/*_msl.rs` and `src/sparse_attn/*_msl.rs` |
| `crates/rmlx-models/src/metal/` | Per-arch kernels — weight-side ParoQuant and GatedDeltaNet, dispatched from `paroquant_msl.rs` / `gated_delta_msl.rs`. Not KV codecs. |
| `crates/rmlx-mlx/src/metal/` | The MLX-JIT language-version probe. Not a production kernel. |

The gates are scoped by **directory, not crate**: a `.metal` file is gated by
where it lives, wherever its Rust dispatcher sits. That distinction is the one
that matters — a kernel inside a gated crate but outside its `metal/` directory
is not gated.

Each module registers its kernels once as `OnceLock<MetalKernel>` singletons
on first use, and its MSL body matches the CPU reference path in the
corresponding `*quant.rs` file.

### `.metal` files + `include_str!`

Bodies are embedded at **compile time**:

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

An `#include` belongs in the header, never the body: MLX splices the body into
the generated kernel *function*, so an include there lands at function scope.

**Parameterised bodies.** Two mechanisms, and the choice is not stylistic:

- **One `.metal` file per variant**, selected by the builder
  (`planar_fused_qk_b3.metal` / `_b4.metal`;
  `rot_k_fwht_quantize_d{32..512}.metal`). Use this when the variants differ in
  *code*, not just in a constant.
- **MLX template arguments** — `set_template_int` / `set_template_dtype`, which
  MLX instantiates per distinct tuple. Use this when the variants differ only by
  a compile-time constant, as `gated_delta_step.metal` does for
  `Dk`/`Dv`/`Hk`/`Hv` and `paroquant_rotate.metal` does for
  `ROWS_PER_TILE`/`MAX_KROT`/`MAX_GROUP_SIZE`. It keeps the bound's single source
  of truth in the Rust const that the validation checks already use.

The body text is never mutated at runtime. A `.replace("{PLACEHOLDER}", ..)` over
a kernel source is what a template argument is for, and it makes the file
uncompilable by the gate.

Adding a KV codec means adding a `.metal` decode kernel and a native compile
test; see CLAUDE.md hard rule 10.

### MSL gates (`make ci`, enforced in CI)

Two gates run over the three kernel directories listed above. The list is
single-sourced in `scripts/metal_dirs.sh`, sourced by both gates and referenced
by the `check-metal-format` pre-commit hook's trigger pattern; a crate that
starts shipping MSL must be added there, since nothing else discovers it.

| Target | Tool | Checks |
|---|---|---|
| `make check-metal-compiles` | `xcrun -sdk macosx metal` (full Xcode, not just the Command Line Tools) | Every kernel compiles natively at `-std=metal3.0` **and** `-std=metal4.0`, so an MSL syntax error surfaces at CI instead of on first GPU dispatch. Also fails if a `.metal` file is missing from its directory's manifest. |
| `make check-metal-format` | `clang-format` (on `PATH` or via `xcrun -f clang-format` — it is not on `PATH` by default) | Every kernel is clang-format clean. MSL is a C++14 dialect; style is pinned by the `.clang-format` in each kernel directory. |

**Two language versions, for two different reasons.** `metal4.0` is what
production compiles at (see "MLX JIT language version" above). `metal3.0` is the
floor, kept so newer syntax cannot creep in unnoticed. The second pass is what
makes a `#if __HAVE_TENSOR__` kernel checkable at all: that macro is undefined
below 4.0, so at `metal3.0` such a body compiles to an empty translation unit
and the gate goes green having validated nothing. Such a body is therefore never
compiled without the guard — it is checked for real, or reported as `SKIP` and
counted, never quietly passed.

The capability is probed by asserting the guard and the cooperative-tensor
includes, not by testing that the driver accepts the `-std` flag. A toolchain
that takes the flag but leaves `__HAVE_TENSOR__` undefined would otherwise
compile a guarded body through its `#else` arm at *both* passes — the same
vacuous pass, reached another way.

**One toolchain policy, not two.** "This box cannot do X" gets the same answer
whether X is the Metal compiler itself or the Metal 4 pass: hard failure under
`--strict` (CI, which must never report green while checking less), and a loud
notice plus a reduced run otherwise. A contributor on an older Xcode keeps a
working `make ci`; what could not be checked is named on stdout and counted in
the summary line. Splitting that rule would break the dev loop for everyone
whose Xcode predates Metal 4, over one diagnostic kernel that ships nothing.

**Manifest coverage is enforced.** Every `.metal` file in a gated directory must
be named by that directory's `probes/kernels.manifest`, as a body or as a
`../`-prefixed header. An unlisted body is compiled by nothing, which is the
same vacuous pass in a different disguise, so the gate hard-fails on it.

**Where they actually run.** Both gates skip when their tool is missing, so a
Command-Line-Tools-only box is not blocked — but a skipping gate protects
nothing, so the skip is local-only. The `msl` job in
`.github/workflows/ci.yml` runs both with `METAL_STRICT=--strict`, which turns
a missing tool into a hard failure — and, for the compile gate, a toolchain that
cannot do the `metal4.0` pass; for the format gate, an empty file set, so a
renamed kernel directory cannot silently disable it while the job stays green. The GitHub macOS runner ships full Xcode,
so the compile gate runs for real there; compiling MSL needs the toolchain,
not a GPU, so it works on a runner with no usable Metal device. Install full
Xcode (`xcode-select -s /Applications/Xcode.app`) to run the compile gate
locally too — on Xcode 16.3+ the compiler is a separate component
(`xcodebuild -downloadComponent MetalToolchain`).

`check-metal-compiles` cannot compile a `.metal` file directly — a body is a
run of statements at file scope, not a translation unit. It assembles a probe
per kernel (`stdlib preamble + header + kernel { buffer aliases + defines +
body }`) and compiles that. Each directory's `probes/kernels.manifest` supplies,
per body: the header to prepend, the buffer names the body expects, and an
optional fourth field of `#define NAME VALUE` pairs for the values MLX injects
at dispatch that are neither buffers nor header constants — template dtypes
(`OutT`, `InT`, `StT`), template ints (`Dk`, `ROWS_PER_TILE`, …) and scalar 0-D
inputs (`T`), which the body sees as numeric literals. Buffer types are `u`
(uint), `i` (int) and `f` (float), matching the dtype the dispatch site declares.
Where such a `#define` duplicates a Rust const, pin it with an equality test as
`probe_manifest_defines_match_rust_consts` does — a hand-copied bound drifts the
same way a captured header snapshot does.
`crates/rmlx-kv-quant/src/metal/probes/README.md` documents the layout and how
to refresh the captured header snapshots; the other two directories follow the
same convention and point back at it.

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
