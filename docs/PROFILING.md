# rMLX Profiling Runbook

Reference: [Rust Perf Book, Chapter 5 — Profiling](https://nnethercote.github.io/perf-book/profiling.html)

## Long-term perf trends

Per-bench TPS / TTFT / RSS land in `metrics/runs.db` (see `docs/METRICS_DB.md`). Time-series queries:
- `rmlx metrics history --backend rmlx --namespace mlx-community --model gemma-4-e2b-it-mxfp8 --weight-quant mxfp8 --kv-quant k8v8 --metric decode_tps_warm` — every observation for one cell.
- `rmlx metrics deltas --since-sha <git-sha> --threshold-pct 5` — what regressed since a commit.
- `rmlx metrics rank --metric decode_tps_warm --limit 20` — top-20 champions.

## Prerequisites (already shipped)

`Cargo.toml [profile.release]` has:
- `debug = "line-tables-only"` — filename + line info for samply/Instruments, no full DWARF.
- `strip = "debuginfo"` — keeps the symbol table (readable backtraces), removes DWARF.
- `split-debuginfo = "packed"` — bundles debug info in `.dSYM` alongside the binary.

`.cargo/config.toml` has:
- `force-frame-pointers=yes` — keeps x29 non-clobbered for stack-walking (samply, Instruments).

These together mean `cargo build --release` already produces samply-readable binaries.

## Tool matrix (Apple Silicon / macOS aarch64)

| Tool | macOS aarch64 | Notes |
|------|--------------|-------|
| **samply** | YES — recommended | Cross-platform sampling profiler. Outputs Firefox Profiler JSON. |
| **Instruments / xctrace** | YES — recommended | Native Apple profiler. Time Profiler, Metal System Trace, Allocations. |
| **cargo flamegraph** | YES (needs sudo for DTrace) | Uses DTrace under the hood on macOS. Lower ergonomics than samply. |
| **dhat-rs** | YES — gated feature | Heap allocation profiling. Gate: `--features dhat-heap`. See below. |
| **counts crate** | YES | Ad-hoc cardinality counting via `eprintln!`. No binary dep. |
| **Intel VTune** | YES (x86 emulation only) | Not recommended on aarch64. |
| **perf + Hotspot** | NO — Linux only | Use Instruments Time Profiler or samply instead. |
| **Cachegrind / Callgrind** | NO — Valgrind ARM64-darwin broken | Use Instruments Counters template (PMU events). |
| **heaptrack / bytehound** | NO — Linux only | Use dhat-rs or Instruments Allocations instead. |
| **Coz (causal profiling)** | NO — macOS support poor | Skip. |
| **AMD uProf** | NO — macOS not supported | Skip. |

## 1. CPU sampling with samply (recommended)

Install:
```bash
cargo install samply
```

Profile a single `rmlx baseline` run:
```bash
samply record --rate 4000 -- \
  ./target/release/rmlx baseline \
    --model $RMLX_O_MODELS_ROOT/mlx-community__gemma-4-e2b-it-mxfp8 \
    --kv-quant k8v8
```

`samply` opens the Firefox Profiler in your browser automatically. Requires no `sudo`.

`--rate 4000` = 4000 Hz sampling (default is 1000 Hz; higher = more resolution, more overhead).

### Make target

```bash
make profile-samply MODEL=/path/to/snapshot
```

## 2. Instruments / xctrace (Apple native)

Time Profiler (CPU sampling, integrates with tracing spans via os_signpost):
```bash
xcrun xctrace record \
  --template 'Time Profiler' \
  --launch -- ./target/release/rmlx baseline \
    --model $RMLX_O_MODELS_ROOT/mlx-community__gemma-4-e2b-it-mxfp8 \
    --kv-quant k8v8
```

Output: `.trace` package. Open with Instruments.app.

Metal System Trace (GPU kernel timings, see §5):
```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --launch -- ./target/release/rmlx baseline \
    --model /path/to/snapshot --kv-quant k8v8
```

### Make target

```bash
make profile-instruments MODEL=/path/to/snapshot
```

## 3. cargo-flamegraph (DTrace-based, needs sudo)

Install:
```bash
cargo install flamegraph
```

Run:
```bash
sudo cargo flamegraph --bin rmlx -- baseline \
  --model $RMLX_O_MODELS_ROOT/mlx-community__gemma-4-e2b-it-mxfp8 \
  --kv-quant k8v8
```

Output: `flamegraph.svg` in the current directory.

Note: DTrace on macOS requires SIP to be partially disabled for kernel stacks. User-space
stacks work without SIP changes when frame pointers are preserved (already configured).

## 4. Heap profiling with dhat-rs (gated feature)

`rmlx-cli` has a `dhat-heap` feature that instruments the global allocator to collect
DHAT-format heap profiles. It is OFF by default and must be explicitly enabled.

Build and run:
```bash
cargo build --features rmlx-cli/dhat-heap --bin rmlx
./target/debug/rmlx baseline \
  --model $RMLX_O_MODELS_ROOT/mlx-community__gemma-4-e2b-it-mxfp8 \
  --kv-quant k8v8
```

On exit, `dhat-heap.json` is written to the current directory.
View it at: https://nnethercote.github.io/dh_view/dh_view.html

Note: run in **debug** mode (or `opt-level=1`) — DHAT's overhead is significant at full opt.
Note: the global allocator is replaced when this feature is active, so jemalloc is disabled.

### What DHAT shows

- Which call sites allocated the most heap bytes.
- Which allocations are short-lived (high alloc + dealloc rate = pressure hot spots).
- Useful for auditing `rmlx-loader` mmap vs full-read behaviour and KV-cache growth.

## 5. Metal GPU capture (the tool for kernel work)

**This is the entry point for MSL kernel questions**, not samply — kernel cost
lives on the GPU, where host stack sampling cannot see it. But a `.gputrace` is
a *frame capture*, not a timeline, and it answers far less on its own than the
Xcode marketing implies. [What a `.gputrace` actually
answers](#what-a-gputrace-actually-answers) below splits the three questions
people conflate; read it before planning a session, because one of the three is
not answerable from a capture at all.

On M5 the Neural Accelerator is **part of the GPU**, so profiling nax needs no
special tooling — the ordinary Metal capture path covers it
([ml-explore/mlx#3182](https://github.com/ml-explore/mlx/issues/3182)).

### How the window works

The capture is a **bounded window of decode steps**, not a whole run: a run is
dominated by weight load and prefill, which is not what kernel work studies, and
a full-run trace is unusably large.

`rmlx_mlx::metal_capture` owns the whole mechanism behind the `metal-capture`
feature. `CaptureScope` is the RAII guard over `mlx_metal_start_capture` /
`mlx_metal_stop_capture`; `Window` is the pure policy that decides when the
scope opens and closes. The one hook is a `step()` call at the top of the
**shared** decode loop (`rmlx_models::decode_loop::pipelined_decode`), so the
window is model- and codec-agnostic — every arch that uses that loop (gemma4,
gemma3, qwen3, qwen3.5-MoE) is covered with no per-arch wiring.

With `--gpu-capture-skip 4 --gpu-capture-steps 8` the scope opens before decode
step 5 and closes before step 13: eight whole steps, no load, no prefill.

**Use at least 8 steps.** `pipelined_decode` is pipelined — a step's work
straddles the boundary — so a 1-step window's kernel set is a strict *subset* of
an 8-step window's (measured: it misses the `gather_front*` embedding lookups).
A narrow window does not just capture less; it misrepresents which kernels
decode runs.

**Off, none of it exists.** Without the feature there is no flag, no hook, no
`Window`, and no undefined reference to `mlx_metal_start_capture` — verifiable
with `nm -u target/release/rmlx | grep mlx_metal_start_capture` (empty).

### Prerequisites

1. Full **Xcode**, not just Command Line Tools — traces cannot be replayed
   without it:

   ```sh
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   ```

2. A binary built with the feature:

   ```sh
   make build-capture     # cargo build --profile release-debug --features rmlx-cli/metal-capture
   ```

3. `MTL_CAPTURE_ENABLED=1` in the process environment. This is **Apple's** —
   Metal inserts the capture layer at launch and there is no in-process way to
   add it later — not an rMLX configuration knob. The wrapper script sets it;
   without it the run aborts before loading the model and says so.

### Capture

```sh
make profile-gputrace CODEC=iso3_sym MODEL=/path/to/snapshot
# or, with the window spelled out:
bash scripts/gpu_capture.sh --kv-quant iso3_sym --model /path/to/snapshot \
  --prompt-tokens 4096 --skip 4 --steps 8
```

Traces land in `.rmlx/traces/` named
`<model>-<codec>-<prompt>tok-<timestamp>.gputrace`; `open` them in Xcode. The
script runs the MLX preflight, refuses a binary built without the feature, and
sizes `--max-ctx` for the prompt.

A capture run's timings are worthless (see below), so `--gpu-capture` forces the
metrics kill switch to `off` for the whole process — no `events` row, no
`observations` row, no `metrics/baseline.csv` append — whatever `--metrics` says.
It also conflicts with `--record`, so asking for a recorded capture is an error
rather than a silent downgrade.

Driving it directly is the same thing without the guard rails:

```sh
MTL_CAPTURE_ENABLED=1 ./target/release-debug/rmlx --metrics off baseline \
  --model /path/to/snapshot --kv-quant none \
  --prompt-tokens 4096 --max-tokens 18 --max-ctx 4700 \
  --gpu-capture .rmlx/traces/run.gputrace --gpu-capture-skip 4 --gpu-capture-steps 8
```

### What it costs

Capture serialises and records every dispatch and snapshots every resident GPU
buffer. Two consequences worth planning for:

- **Decode collapses** to single-digit TPS during the window (measured: ~2.5 TPS
  on gemma-4-e2b against ~127 TPS for the same cell uncaptured). Timings from a
  capture run are meaningless, which is why `--gpu-capture` conflicts with
  `--record` *and* forces `--metrics off` — the flag conflict alone would still
  have let the run write an `events` row and a `baseline.csv` line.
- **Bundles are large** — the resource snapshot dominates, so the floor is
  roughly the model's resident footprint. ~6 GB for an e2b-class model at 4k
  context, near-identical for an 8-step and a 16-step window. Delete traces when
  you are done with them; `.rmlx/traces/` has no size cap.

The *command stream* (`<trace>/capture`) is the part that scales with the
window — measured at ~2.2–2.6 MB per decode step with a fixed floor under 0.2%
of an 8-step stream. That near-zero intercept is the check that a trace really
holds only the decode window: model load or prefill inside it would show up as a
large constant term.

### What a `.gputrace` actually answers

Three questions people run this for. They need three different things, and only
the first is in the bundle you just wrote. Measured across five bundles from
this path: **no `.gpuprofiler_raw`, and zero timestamp, duration or counter
payloads anywhere** — the only `counter`-ish string in a whole 6 GB bundle is
one empty `counterSampleBuffers` category label.

**1. Kernel identity — in the bundle, offline, no Xcode.**

`<trace>/device-resources-0x<addr>` names every pipeline and function the window
referenced, by mangled MSL name, and `<trace>/metadata` counts how many of them
went unused:

```sh
strings <trace>/device-resources-0x* | grep -oE 'gather_front[a-z0-9_]*' | sort -u
plutil -p <trace>/metadata | grep unused   # unusedComputePipelineStateCount, ...
```

That is enough to answer "is the codec's own kernel running at all, or is it
decoding through the bf16 mirror?" — the question that motivated the capture
window in the first place.

**2. Per-dispatch time, limiter counters, occupancy, achieved bandwidth — GUI
only, and they do not exist until you ask for them.**

Open the bundle in Xcode and press **Profile**. That *replays* the capture
on-device with counters enabled, and the replay is what creates
`.gpuprofiler_raw`. There is no headless replay tool: on Xcode 26.6 `xctrace`
offers `record` / `import` / `export` / `symbolicate` and nothing that replays a
capture, so this step cannot be scripted or run over ssh. Background: [Analyzing
Apple GPU performance using
counter
statistics](https://developer.apple.com/documentation/xcode/analyzing-apple-gpu-performance-using-counter-statistics).

**3. Gaps between dispatches as they occurred in your run — not in a frame
capture. Ever.**

A replay has the replay's schedule, not the schedule of the run you captured.
Host round-trips — the blocking `Array::eval()` per layer per step, a per-step
prefix restage — will **not** show up in a `.gputrace`, no matter how it is
replayed. That needs a timeline instrument over the live process:

```sh
xcrun xctrace record --template 'Metal System Trace' --launch -- \
  ./target/release-debug/rmlx --metrics off baseline --model /path/to/snapshot ...
```

Budget the session accordingly: if the hypothesis is "the GPU is idle waiting on
the host", a GPU capture is the wrong artifact and will cost a day proving
nothing.

### Tests

The window policy and the request validation are pure and unit-tested, but the
tests are behind the same feature, so a plain `cargo test` does not compile
them. Run them with:

```sh
make test-capture
```

`make ci` runs that target too — without it, an off-by-one in the window policy
would pass the gate green, since `make test` compiles these tests out entirely.

## 6. Ad-hoc cardinality counting with the `counts` crate

The [counts crate](https://crates.io/crates/counts) is the perf-book's "ad-hoc profiling"
recommendation: sprinkle `eprintln!` on a hot branch, run, pipe to `counts`, get a
frequency table.

No code change needed — add `counts` as a dev-dependency when needed:
```toml
[dev-dependencies]
counts = "0.2"
```

Example use: counting how often mxfp8 vs bf16 dequant paths are taken in `rmlx-quant`:
```rust
eprintln!("dequant_path={}", if is_mxfp8 { "mxfp8" } else { "bf16" });
```
```bash
./target/release/rmlx baseline ... 2>&1 | counts
```

## 7. RUST_LOG tuning for profiling sessions

The default `RUST_LOG=debug,rmlx=trace` setting writes every span enter/exit for
`rmlx_models` to the JSONL log. With `#[instrument]` on generate_greedy boundaries,
this produces ~42 span events per decode step at trace level — tolerable for short runs.

For long runs or when reducing log I/O is important:
```bash
RUST_LOG=debug,rmlx_models=debug ./target/release/rmlx baseline ...
```

To enable per-model trace for a specific module only:
```bash
RUST_LOG=debug,rmlx_models::gemma4=trace ./target/release/rmlx baseline ...
```

## 8. Symbol demangling

If a profiler shows mangled `_ZN` or `_R` prefixed names:
```bash
cargo install rustfilt
some-profiler-output | rustfilt
```

Or build with v0 mangling (more demangler-compatible):
```bash
RUSTFLAGS="-C symbol-mangling-version=v0" cargo build --release
```
(Not set by default — adds it only when needed for a specific profiling session.)

## 9. Process-memory counters: RSS vs phys_footprint vs Metal peak_alloc (J4)

`rmlx_core::mach_mem::read_proc_mem()` exposes six counters from two `task_info` calls.
They are related but not equal; understanding the difference matters for OOM tuning:

| Counter | Source | What it counts | When to use |
|---------|--------|----------------|-------------|
| `rss_bytes` | `MACH_TASK_BASIC_INFO.resident_size` | Pages physically in RAM right now — what `ps -o rss` shows. | Quick sanity check; matches operator intuition. |
| `virtual_bytes` | `MACH_TASK_BASIC_INFO.virtual_size` | Total VM address space committed. | Rarely actionable on Apple Silicon (48-bit VA space). |
| `phys_footprint_bytes` | `TASK_VM_INFO.phys_footprint` | **Apple's pressure metric** — anonymous + file-backed resident + compressed pages counted as "yours". What Activity Monitor shows; what the kernel OOM killer uses. | Use this for pressure decisions (J3 OOM guard). |
| `internal_bytes` | `TASK_VM_INFO.internal` | Anonymous heap pages — jemalloc arenas, KV-cache buffers, Rust Vec allocations. | Track heap growth independently of weights. |
| `compressed_bytes` | `TASK_VM_INFO.compressed` | Pages handed to the macOS memory compressor ("soft swap" — still counts against `phys_footprint`). | Non-zero means the system is already under pressure. |
| `external_bytes` | `TASK_VM_INFO.external` | File-backed pages — in rMLX this is primarily mmap'd safetensors weight files. | `external_bytes ≈ loaded-weight footprint`; grows with model size, shrinks on unload. |

**Metal `peak_alloc_mb` (F3, not yet built)** is a separate counter from the Metal Performance
HUD / `MTLDevice.currentAllocatedSize`.  It counts GPU-private VRAM allocations (weight
tensors, KV-cache MTLBuffers) and is disjoint from the `task_info` counters above — they
measure CPU/UMA host memory, not GPU-private usage.  On Unified Memory Macs the boundaries
blur (all memory is the same physical chips) but the accounting domains are distinct.

**Typical relationship**: `rss_bytes ≤ phys_footprint_bytes ≤ rss_bytes + compressed_bytes`.
`external_bytes` overlaps with `rss_bytes` (mmap'd weight pages that are currently resident).
When weights are evicted by the compressor, `external_bytes` drops and `compressed_bytes` rises.

## 10. Prefill-chunk size knob (J9)

Cold prefill is chunked per-arch by `rmlx_models::prefill_chunk::prefill_chunk_for(arch)`
(all 7 archs route through it). The chunk size trades per-chunk lazy-graph
overhead against MLX scheduler pipelining and the GatedDeltaNet `ts<256`
fast-path (Qwen3.5-MoE). Tuned per-arch from follow-up bench data —
**do not change defaults without an executor-bench sweep** (CLAUDE.md
§"Executor-bench discipline").

Per-arch defaults: `qwen3=256`, `qwen3_5_moe=64`, `gemma3=256`,
`gemma4=512`, `qwen2=256`, `laguna=256`.

Override at runtime (resolution order: **per-arch env > global env >
arch default > 64 fallback**):

- `RMLX_PREFILL_CHUNK=<n>` — global, all archs.
- `RMLX_PREFILL_CHUNK_<ARCH>=<n>` — per-arch, ARCH upper-cased, e.g.
  `RMLX_PREFILL_CHUNK_QWEN3_5_MOE=256`.

Notes:
- `qwen3_5_moe=64` is deliberately low: a larger chunk pushes
  GatedDeltaNet past the `ts<256` fused-kernel fast-path into the slow
  MLX-graph recurrence. The 256 variant is reachable via
  `RMLX_PREFILL_CHUNK_QWEN3_5_MOE=256` for users who want to A/B it.
- `gemma4=512` is bench-justified (−30% cold TTFT at 8K vs 256).
- The 64-is-best-for-qwen3_5_moe verdict was last measured 2026-05-12;
  re-confirm in the next bench sweep that touches the MoE path (J9.5,
  backlog).

## Quick reference

| Goal | Command |
|------|---------|
| CPU profile (recommended) | `make profile-samply MODEL=...` |
| Native Apple profiler | `make profile-instruments MODEL=...` |
| Flamegraph (needs sudo) | `sudo cargo flamegraph --bin rmlx -- baseline ...` |
| Heap profile | `cargo build --features rmlx-cli/dhat-heap && ./target/debug/rmlx baseline ...` |
| GPU capture | Deferred — see §5 above |
| Ad-hoc branch counts | `eprintln!` + `counts` crate |
| Process memory snapshot | `rmlx_core::mach_mem::read_proc_mem()` — see §9 |
| Prefill-chunk override | `RMLX_PREFILL_CHUNK_<ARCH>=<n>` — see §10 |
