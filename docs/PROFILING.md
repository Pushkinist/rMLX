# rMLX Profiling Runbook

Reference: [Rust Perf Book, Chapter 5 — Profiling](https://nnethercote.github.io/perf-book/profiling.html)

Recorded bench figures live in `runs.db`; `docs/METRICS_DB.md` covers the
queries over them. This doc covers taking a profile.

## Ordering on this host

Two configurations compared in slots are compared at different slot
positions. An ABBA block cancels drift that is linear in slot position. On
this host the drift is not linear: slot 2 of a block runs slow for whichever
arm occupies it. ABBA puts the same arm in slot 2 of every block, so that
penalty lands on one arm and looks like a clean result.

- **Two arms**: pair every ABBA block with a BAAB block, so each arm holds
  each slot position equally often. `scripts/perf_ab.sh` alternates ABBA,
  BAAB, ABBA.
- **Three or more levels**: run a Latin square. The level index is
  `(position + row) % n_levels`, one row per cycle, so every level holds every
  slot once per `n_levels` rows. `scripts/prefill_chunk_sweep.sh` runs this
  shape.
- **Report the paired statistic.** Compare each level to the baseline within
  a row and take the median over rows. A pooled median across slots re-admits
  the positional term.

An unpaired block, or a paired block reported pooled, is not evidence here.

## Build profiles

`[profile.release]` in `Cargo.toml` sets `debug = "line-tables-only"`,
`strip = "debuginfo"` and `split-debuginfo = "packed"`. A release binary keeps
its symbol table and line tables, so samply and Instruments can read it.

`[profile.release-debug]` inherits `release` and sets `debug = true` and
`strip = "none"`. Full DWARF lets samply resolve inlined frames under fat LTO.
Build it with `make build-debug`.

No profile forces frame pointers. `.cargo/config.toml` shows the `RUSTFLAGS`
spelling that adds `-C force-frame-pointers=yes` for a local build.

## Tool matrix (Apple Silicon / macOS aarch64)

| Tool | Use | Notes |
|---|---|---|
| samply | CPU sampling | Opens the Firefox Profiler. No `sudo`. |
| Instruments / `xctrace` | CPU and GPU timelines | Time Profiler, Metal System Trace, Allocations. |
| `cargo flamegraph` | CPU sampling | DTrace, needs `sudo`. |
| dhat-rs | Heap allocations | Gated feature `rmlx-cli/dhat-heap`. |
| Metal GPU capture | Kernel identity | Gated feature `rmlx-cli/metal-capture`. |

perf, Hotspot, Valgrind, heaptrack and bytehound do not run on macOS aarch64.

## 1. CPU sampling with samply

```bash
cargo install samply
make profile-samply MODEL=/path/to/snapshot
```

`profile-samply` records `target/release/rmlx baseline --kv-quant k8v8` at
4000 Hz. `make profile-samply-debug MODEL=...` builds `release-debug` first
and records a decode-weighted run: `PROF_PROMPT` (default 1024) prompt tokens
and `PROF_GEN` (default 500) generated tokens.

## 2. Instruments / xctrace (Apple native)

```bash
make profile-instruments MODEL=/path/to/snapshot
```

The target runs the Time Profiler template on
`target/release/rmlx baseline --kv-quant k8v8` and writes a `.trace` package.
For GPU timing use `make profile-mst` (§5).

## 3. cargo-flamegraph (DTrace-based, needs sudo)

```bash
cargo install flamegraph
sudo cargo flamegraph --bin rmlx -- baseline --model /path/to/snapshot --kv-quant k8v8
```

It writes `flamegraph.svg` to the current directory. Kernel stacks need SIP
partly disabled; user-space stacks do not.

## 4. Heap profiling with dhat-rs (gated feature)

`rmlx-cli` has a `dhat-heap` feature that replaces the global allocator with
DHAT's. It is off by default.

```bash
cargo build --features rmlx-cli/dhat-heap --bin rmlx
./target/debug/rmlx baseline --model /path/to/snapshot --kv-quant k8v8
```

On exit the binary writes `dhat-heap.json` to the current directory. Open it
in [dh_view](https://nnethercote.github.io/dh_view/dh_view.html). DHAT is slow
at full optimisation, so profile a debug build.

## 5. Metal GPU capture (the tool for kernel work)

Kernel cost lives on the GPU, where host stack sampling cannot see it. Two
tools answer different questions:

- A `.gputrace` capture answers which kernels a decode window referenced.
- A Metal System Trace answers how long each submission ran and how long the
  GPU sat idle.

Neither gives per-dispatch counters headlessly. Those come from the Xcode
replay below, which needs a person.

On M5 the Neural Accelerator is part of the GPU. A capture names NAX
pipelines like any other: `steel_gemm_fused_nax_*` for matmul and
`steel_attention_*_bq64_*` for attention. `bq64` means the NAX attention
branch ran; `bq32` means it did not. Which paths can reach NAX is in
[`docs/FFI.md`](FFI.md#where-nax-can-appear-and-where-it-cannot).

### How the window works

A capture covers a bounded window of decode steps, not a whole run.
`rmlx_mlx::metal_capture` owns it behind the `metal-capture` feature.
`CaptureScope` is the RAII guard over `mlx_metal_start_capture` and
`mlx_metal_stop_capture`. `Window` decides when the scope opens and closes.
The one hook is `metal_capture::step()` at the top of the shared
`rmlx_models::decode_loop::pipelined_decode`, so every arch on that loop is
covered.

With `--gpu-capture-skip 4 --gpu-capture-steps 8` (the defaults) the scope
opens before decode step 5 and closes before step 13.

**Use at least 8 steps.** Decode is pipelined, so a step's work straddles the
step boundary. A 1-step window misses kernels an 8-step window holds, such as
the `gather_front*` embedding lookups.

Without the feature there is no flag, no hook and no reference to
`mlx_metal_start_capture`. `nm -u target/release/rmlx | grep
mlx_metal_start_capture` prints nothing.

### Prerequisites

To capture:

1. A binary built with the feature. `make build-capture` builds
   `release-debug` with `rmlx-cli/metal-capture` and re-signs it.
2. `MTL_CAPTURE_ENABLED=1` in the process environment. This is Apple's
   variable: Metal inserts the capture layer at launch only. Without it the
   run aborts before loading the model.

For Apple's GPU tools to attach to the process, checked by
`make gputrace-preflight` (`scripts/gputrace_preflight.sh`):

3. Full Xcode selected: `sudo xcode-select -s
   /Applications/Xcode.app/Contents/Developer`.
4. Developer mode: `sudo DevToolsSecurity -enable`.
5. The binary signed with `com.apple.security.get-task-allow`. Cargo emits a
   linker-signed binary with no entitlements. `make build-capture` re-signs it
   with `scripts/rmlx-capture.entitlements`, and re-running it repairs a
   binary a plain `cargo build` re-created. Check with
   `codesign -d --entitlements - target/release-debug/rmlx`.
6. The Metal toolchain, for the shader recompile a replay does
   (`xcodebuild -downloadComponent MetalToolchain`). The preflight only warns
   about this one.

### Capture

```sh
make profile-gputrace CODEC=iso3_sym MODEL=/path/to/snapshot
# or, with the window spelled out:
bash scripts/gpu_capture.sh --kv-quant iso3_sym --model /path/to/snapshot \
  --prompt-tokens 4096 --skip 4 --steps 8
```

Traces land in `.rmlx/traces/` as
`<model>-<codec>-<prompt>tok-<timestamp>.gputrace`. The script runs the MLX
preflight and refuses a binary without the feature. It refuses a host the
tools cannot attach to before it writes anything. It sizes `--max-ctx` for the
prompt and enforces the trace cap afterwards.

Capture serialises every dispatch and snapshots every resident buffer. Decode
drops to a few tokens per second, and a bundle is about the size of the
model's resident footprint. So `--gpu-capture` conflicts with `--record` and
forces the metrics kill switch to `off`: no `events` row, no `observations`
row, no `metrics/baseline.csv` line.

Driving the binary directly:

```sh
MTL_CAPTURE_ENABLED=1 ./target/release-debug/rmlx --metrics off baseline \
  --model /path/to/snapshot --kv-quant none \
  --prompt-tokens 4096 --max-tokens 18 --max-ctx 4700 \
  --gpu-capture .rmlx/traces/run.gputrace --gpu-capture-skip 4 --gpu-capture-steps 8
```

### What a `.gputrace` actually answers

**Kernel identity, offline.** `<trace>/device-resources-0x<addr>` names every
pipeline and function the window referenced.
`unused-device-resources-0x<addr>` holds the ones the capture layer recorded
as unused. That answers whether a codec's own kernel runs or it decodes
through the bf16 mirror. Read them with the scripts below.

**No timing, no counters.** The bundles hold no `.gpuprofiler_raw` and no
timestamp or counter payload. Only Xcode's GUI Profile replay writes timing,
and `xctrace` has no replay verb. On M5 Max
`supportsCounterSampling(atDispatchBoundary)` is false and
`device.counterSets` holds only `GPUTimestamp`. An empty timeline in Xcode is
the expected state of these bundles.

**Wall-clock GPU time and host gaps need Metal System Trace.** A replay runs
on the replay's schedule. Host round-trips, such as a blocking
`Array::eval()` per layer, never show in a `.gputrace`.

```sh
make profile-mst MODEL=/path/to/snapshot
# or, spelled out:
bash scripts/mst_capture.sh --model /path/to/snapshot --kv-quant none \
  --prompt-tokens 4096 --max-tokens 600 --time-limit 18
```

`mst_capture.sh` records the live process, exports the `metal-gpu-intervals`
table, parses it and prints a per-channel table and a CSV. The table gives
each submission's `start` and `duration` in nanoseconds, `gpu-channel-name`,
`start-latency`, `cmdbuffer-id` and `encoder-id`. One row is one encoder.

- **`--attach <pid>` records nothing** for this template. Metal
  instrumentation must be present at launch, so the harness uses `--launch`.
- **Weight load leaves no rows; prefill does.** The table does not mark where
  prefill ends. The harness reads the run's own `decode_profile{prefill_ms}`
  from `<RMLX_HOME>/logs/<run-id>.jsonl` and uses it as the default
  `--skip-ms`. An explicit `--skip-ms` wins. When no event is found, the
  harness says so and summarises the whole window. `--skip-ms` counts from the
  matched process's first submission.
- **Tracing slows decode**, so the harness forces `--metrics off`.
- **The export XML uses `id`/`ref` back-references** and `<sentinel/>` for
  NULL. A naive parser misaligns columns into plausible wrong numbers.
  `rmlx_mlx::xctrace` checks each row's cell count and each cell's tag
  against the schema, and rejects an unresolvable `ref`.
- **Bound the volume** with `--time-limit`. `.rmlx/traces/mst` keeps the
  newest `--keep` bundles (default 5) and prunes the rest on every exit.
  `make traces-gc` does not cover this directory.
- **No kernel names on the timeline.** `metal-shader-profiler-shader-list`
  names the pipelines, but the stock template records
  `Shader Timeline: Disabled`, so no key joins a name to a timed row. Pair the
  timeline with a `.gputrace` identity list.

The summariser tells two refusals apart: `contains no rows` is an empty table;
`holds N rows but none for a process matching …` lists the processes it saw.

#### Reading the timeline

- **Check the window against the run's own log.** The full span should match
  `prefill_ms + step_total_ms`, and the decode span should match
  `step_total_ms`. The harness prints both. Do not compare against a separate
  untraced run: tracing and run-to-run spread both move it.
- **`start-latency` is queue depth, not a host stall.** It is the gap between
  commit and start on the GPU. It is high when the GPU is saturated and falls
  when the host is the bottleneck, because the queue is empty.
- **The host-stall signal is idle GPU time**, `span - busy`, which the harness
  prints as `idle:`. Read `idle` first.

### Working with a bundle

The bundle layout is Apple's, not a stable contract. Each script checks the
structure it reads and fails by name when the layout moves.

| Command | What it answers |
|---|---|
| `bash scripts/gputrace_summary.sh <bundle>` | What was captured (from the harness file name), total and command-stream size, and whether a `.gpuprofiler_raw` is present. |
| `bash scripts/gputrace_kernels.sh <bundle>` | Which Metal functions the window referenced, and which it recorded as unused. `--set used\|unused\|all`, `--names-only`. |
| `bash scripts/gputrace_diff.sh <a> <b>` | What A's window referenced that B's did not, and the reverse. |
| `bash scripts/gputrace_preflight.sh` | The host prerequisites above, each with its fix. Also `make gputrace-preflight`. |

Some function records store their name by object id. The scripts count and
report those (`… 46 named, 12 stored by object id`), so the named list is a
subset. A `.gputrace` holds no dispatch counts: which kernels ran is
answerable offline, how many times is not.

### Keeping `.rmlx/traces` bounded

- `scripts/gpu_capture.sh` keeps the newest 6 bundles and at most 40 GB after
  each capture. Eviction is oldest-first, never the bundle just written, and
  each removal is printed.
- `--keep-all` (`make profile-gputrace … KEEP_ALL=1`) skips the cap.

```sh
make traces-gc                                   # report what is over the caps
make traces-gc APPLY=1                           # enforce them
make traces-gc APPLY=1 MAX_COUNT=12 MAX_TOTAL_GB=80
bash scripts/traces_gc.sh --apply --max-age-days 7   # optional age rule
```

### Tests

The window policy, the request validation and the `xctrace` parser are unit
tests behind the same feature, so `make test` compiles them out.
`make test-capture` runs them, and `make ci` runs that target. Most parser
tests pair a fixture with a mutated twin that must be refused: a dropped
`<sentinel/>`, a one-column shift, a dangling `ref`, a non-numeric duration.

## 6. Log level for profiling sessions

`--log verbose` or `RUST_LOG=...=trace` turns on per-token and per-FFI
events, which add log I/O to decode. Scope trace to one module:

```bash
RUST_LOG=debug,rmlx_models::gemma4=trace ./target/release/rmlx baseline ...
```

## 7. Symbol demangling

If a profiler shows mangled `_ZN` or `_R` names, pipe its output through
`rustfilt` (`cargo install rustfilt`).

## 8. Ad-hoc counting

The [counts crate](https://crates.io/crates/counts) prints a frequency table
of the lines on its stdin. Emit a `trace!` event on the branch under study,
extract its field from the JSONL log, and pipe that through `counts`.

## 9. Process-memory counters: RSS vs phys_footprint vs Metal peak_alloc (J4)

`rmlx_core::mach_mem::read_proc_mem()` returns six counters from two
`task_info` calls:

| Counter | Source | What it counts |
|---|---|---|
| `rss_bytes` | `MACH_TASK_BASIC_INFO.resident_size` | Pages in RAM now, as `ps -o rss` shows. |
| `virtual_bytes` | `MACH_TASK_BASIC_INFO.virtual_size` | Virtual address space. |
| `phys_footprint_bytes` | `TASK_VM_INFO.phys_footprint` | Apple's memory-pressure figure. It counts compressed pages. Activity Monitor shows it. |
| `internal_bytes` | `TASK_VM_INFO.internal` | Anonymous pages. |
| `compressed_bytes` | `TASK_VM_INFO.compressed` | Pages held by the memory compressor. Non-zero means the system is under pressure. |
| `external_bytes` | `TASK_VM_INFO.external` | File-backed pages, mostly the mmap'd safetensors. |

The Metal peak is a different counter. `rmlx_mlx::mlx_peak_memory_bytes()`
reads the MLX allocator's high-water mark of live bytes. The server publishes
it as `metal_peak_alloc_mb`. It is a process-lifetime figure unless a
`PeakBracket` scopes it.

### 9.1 Scoping the Metal peak to a region

`rmlx_mlx::PeakBracket` records the live bytes and zeroes the peak mark at
`open()`, and reads them back at `close()`:

```rust
let bracket = PeakBracket::open();
// ... region under test, evaluated inside the bracket (MLX is lazy) ...
let reading = bracket.close();
```

| Accessor | Meaning |
|---|---|
| `peak_bytes` | Most bytes live at once inside the region. `0` if it allocated nothing. |
| `headroom_bytes()` | `peak - live_at_open`: what the region needed on top of what was resident. Compare this across runs. |
| `transient_bytes()` | `peak - live_at_close`: allocated inside and released again. A scratch buffer smaller than the surviving buffers hides under the peak, so zero does not prove no scratch. |
| `observed_allocation()` | `headroom_bytes() > 0`. Assert this first. `peak_bytes > 0` is no test: MLX sets `peak = max(peak, active)` over the whole live count, so one allocation anywhere lifts it to the full resident total. |
| `measurable()` | The peak mark was zeroed at `open()`. When `false`, every accessor above returns 0. |

Rules the pooling allocator imposes:

- **Never assert on an absolute byte count.** Pooled buffers carry what ran
  before. Bound `headroom_bytes()` by a multiple of the workload's own size,
  as `q8_msl_roundtrip_allocation_stays_within_budget` in
  `crates/rmlx-kv-quant/src/q8_msl_tests.rs` does.
- **Evaluate inside the bracket.** An `eval()` after `close()` allocates after
  the mark was read, and `observed_allocation()` reads false.
- **One bracket at a time.** The peak mark is process-global.

`rmlx baseline` reports `metal_peak_mb` (the peak over prefill and decode) and
`metal_gen_alloc_mb` (that peak minus the bytes live at open). Only the second
compares across runs; the first carries the weights.

## 10. Prefill-chunk size knob

The per-arch chunk defaults and their resolution order are in
[`KV_CACHE.md`](KV_CACHE.md) § "Chunked prefill". The `arch_default` rows in
`crates/rmlx-models/src/prefill_chunk.rs` record what each default was
measured on. Do not change one without a sweep run as a Latin square (see
"Ordering on this host" above).

`scripts/prefill_chunk_sweep.sh` drives the per-arch sweep through
`RMLX_PREFILL_CHUNK_<ARCH>`. It records each cell in the metrics DB under
`decode_config = 'prefill_chunk=<n>'`.

`decode_loop::chunked_prefill` and `speculative::prefill_chunked_for_class`
log the resolved size as `debug!` fields: `prefill_chunk` and
`prefill_chunk_source`. The source is one of `adaptive`, `env_arch`,
`env_global`, `arch_default` or `fallback`. A prefill loop that calls
`prefill_chunk_for` directly logs neither field: the `laguna`, `qwen2` and
`bitnet` generate paths, one `qwen3_5_moe` path and the `qwen3_vl_moe` image
path.

## Quick reference

| Goal | Command |
|---|---|
| CPU profile | `make profile-samply MODEL=...` |
| CPU profile with inlined frames | `make profile-samply-debug MODEL=...` |
| Native Apple profiler | `make profile-instruments MODEL=...` |
| Heap profile | `cargo build --features rmlx-cli/dhat-heap` (§4) |
| Which kernels ran | `make profile-gputrace CODEC=... MODEL=...` (§5) |
| GPU time and idle gaps | `make profile-mst MODEL=...` (§5) |
| Process memory | `rmlx_core::mach_mem::read_proc_mem()` (§9) |
| Prefill-chunk override | `RMLX_PREFILL_CHUNK_<ARCH>=<n>` |

## Xcode GPU counter replay

This is a GUI workflow; an agent cannot drive it. An agent can capture the
bundle and read the exported CSV. A person has to click Profile, so ask the
user rather than report the counters as unavailable.

### Capture

```bash
bash scripts/gputrace_preflight.sh
bash scripts/gpu_capture.sh --kv-quant <codec> --model <snapshot> \
     --prompt-tokens 8192 --skip 32 --steps 8 --keep-all
```

Keep `--steps` at 8 or more. Pass `--keep-all` when a sibling bundle must
survive the default prune.

### Replay (the human step)

1. `open -a Xcode <bundle>`. A `.gputrace` opens as Debugging GPU Workload.
2. In the navigator, open Performance. It reads "Performance data not
   available" with a Profile button.
3. In the Profile sheet set Performance State: Maximum and GPU Execution
   Mode: Serial, then Profile. Maximum keeps full clocks, so the bandwidth
   fraction is not deflated. Serial attributes time per dispatch; it does not
   represent runtime performance.
4. Both arms of a comparison use identical settings, or the comparison is
   void.

### Read

- **Shaders** tab: cost %, `# SIMD Groups`, `# Allocated Registers` and
  `Spilled Bytes` per pipeline. The name suffix shows the kernel dtype
  (`float32` against `bfloat16`), which is how a dtype promotion shows up.
- **Counters** tab: export the CSV, one row per encoder. The decisive
  columns:

| column | reads |
|---|---|
| `Integer and Conditional Limiter` | integer, address and control issue pressure |
| `Last Level Cache Limiter` | memory-bound |
| `Instruction Throughput Limiter` | issue-bound |
| `Kernel Occupancy` | resident threadgroups; falls as allocated registers rise |
| `Device Memory Bandwidth` | achieved GB/s, against the ceiling `scripts/perf_ceiling.py` assumes |

Group encoders by their top limiter. Dispatch counts identify the kernel: 26
codec layers over 8 steps is 208 encoders. Store exports under
`.rmlx/analysis/<probe>/xcode/`.

Under Serial, limiters, occupancy, registers, SIMD groups, kernel identity
and dtype are precise. Absolute time and bandwidth are not production
numbers; compare the ON/OFF ratio. Never compare a profiled millisecond with
a `perf_ab.sh` millisecond.
