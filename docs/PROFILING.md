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

On M5 the Neural Accelerator is part of the GPU, so there is no separate NAX
tool and none is needed
([ml-explore/mlx#3182](https://github.com/ml-explore/mlx/issues/3182)): a
capture *names* NAX pipelines like any other — `steel_gemm_fused_nax_*` for
matmul, `steel_attention_*_bq64_*` for attention, where **`bq64` means the NAX
attention branch was taken and `bq32` means it was not**. It does not *time*
them, on this hardware or any other; see [What a `.gputrace` actually
answers](#what-a-gputrace-actually-answers). Which paths can reach NAX at all
is in [`docs/FFI.md`](FFI.md#where-nax-can-appear-and-where-it-cannot).

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

Capture and *replay* need different things, and the split is why a trace can be
written successfully and still show nothing when opened. All of them are checked
before a run writes anything — `scripts/gpu_capture.sh` refuses up front rather
than after several GB.

**To capture:**

1. A binary built with the feature:

   ```sh
   make build-capture     # cargo build --profile release-debug --features rmlx-cli/metal-capture, then signs it
   ```

2. `MTL_CAPTURE_ENABLED=1` in the process environment. This is **Apple's** —
   Metal inserts the capture layer at launch and there is no in-process way to
   add it later — not an rMLX configuration knob. The wrapper script sets it;
   without it the run aborts before loading the model and says so.

**For Apple's GPU tools to attach to the process** (developer mode plus the
debuggable entitlement are what let them attach at all — a capture taken by a
process they may not attach to is not usable in the Xcode GPU debugger),
checked by `scripts/gputrace_preflight.sh` / `make gputrace-preflight`:

3. Full **Xcode**, not just Command Line Tools:

   ```sh
   sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
   ```

4. **Developer mode** enabled — Xcode's GPU tools cannot attach without it:

   ```sh
   sudo DevToolsSecurity -enable
   ```

5. The capture binary signed with **`com.apple.security.get-task-allow`** —
   Apple's "this process may be attached to" marker. Cargo emits an ad-hoc
   *linker-signed* binary that carries no entitlements at all, so a plain
   `cargo build` fails this. `make build-capture` re-signs with
   `scripts/rmlx-capture.entitlements` as part of the build; `codesign --force`
   is idempotent, so it also repairs a binary a bare `cargo build` re-created.
   The re-sign is inert for throughput — measured on gemma-4-e2b at 4k, decode
   128.5 TPS signed against 128.4 unsigned (+0.10%, inside a 2.2% run-to-run
   range), identical token digest.

   Verify by hand with:

   ```sh
   codesign -d --entitlements - target/release-debug/rmlx
   ```

6. The Metal toolchain, for the shader recompilation a replay does
   (`xcodebuild -downloadComponent MetalToolchain`). Advisory — the preflight
   warns rather than failing, since capture itself does not need it.

### Capture

```sh
make profile-gputrace CODEC=iso3_sym MODEL=/path/to/snapshot
# or, with the window spelled out:
bash scripts/gpu_capture.sh --kv-quant iso3_sym --model /path/to/snapshot \
  --prompt-tokens 4096 --skip 4 --steps 8
```

Traces land in `.rmlx/traces/` named
`<model>-<codec>-<prompt>tok-<timestamp>.gputrace`. The script runs the MLX
preflight, refuses a binary built without the feature, refuses a host that
cannot attach (developer mode, entitlement — *before* the multi-GB write), sizes
`--max-ctx` for the prompt, and enforces the trace-directory cap afterwards
(see [Keeping `.rmlx/traces` bounded](#keeping-rmlxtraces-bounded)).

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
the first is in the bundle you just wrote. Measured across six bundles from this
path — including ones captured with developer mode on and an entitled binary:
**no `.gpuprofiler_raw`, and zero timestamp, duration or counter payloads
anywhere** — the only `counter`-ish string in a whole 6 GB bundle is one empty
`counterSampleBuffers` category label.

**1. Kernel identity — in the bundle, offline, no Xcode. This is the reason to
capture.**

`<trace>/device-resources-0x<addr>` names every pipeline and function the window
referenced, by mangled MSL name; `unused-device-resources-0x<addr>` holds the
ones the capture layer recorded as unused. Read them with the bundle tools
below, not by hand — the record layout has traps (see
[Working with a bundle](#working-with-a-bundle)).

That is enough to answer "is the codec's own kernel running at all, or is it
decoding through the bf16 mirror?" — the question that motivated the capture
window in the first place, and the one that produced the `iso3_sym` ⊃ `none`
finding.

**2. Per-dispatch time, limiter counters, occupancy, achieved bandwidth — do
not plan a session around these. Most of them do not exist on this hardware.**

- Timing appears only in `.gpuprofiler_raw`, and only Xcode's **GUI** Profile
  replay writes it. There is no scriptable equivalent: `xctrace` has no replay
  verb (Xcode 26.6 offers `record` / `import` / `export` / `remodel` /
  `symbolicate`), `/System/Library/CoreServices/MTLReplayer.app` has hidden
  `--replay` / `--counters` flags but hangs and is killed without Xcode's XPC
  session, and Xcode 26's MCP server exposes nothing that touches a gputrace.
- The counters people actually want are **unsupported on M5 Max**:
  `supportsCounterSampling(atDispatchBoundary)` is false, `device.counterSets`
  returns exactly one set (`GPUTimestamp`), and the *Metal GPU Counters*
  template refuses with "Selected counter profile is not supported on target
  device".

So an empty timeline in Xcode is the expected state of these bundles, not a
misconfiguration to chase.

**3. Wall-clock GPU timing and the gaps between submissions — use Metal System
Trace, not a capture. Ever.**

A replay has the replay's schedule, not the schedule of the run you captured.
Host round-trips — the blocking `Array::eval()` per layer per step, a per-step
prefix restage — will **not** show up in a `.gputrace`, no matter how it is
replayed. A timeline instrument over the live process does show them, headlessly
and with nanosecond resolution. That path is built:

```sh
make profile-mst MODEL=/path/to/snapshot
# or, spelled out:
bash scripts/mst_capture.sh --model /path/to/snapshot --kv-quant none \
  --prompt-tokens 4096 --max-tokens 600 --time-limit 18
```

It records the live process, exports the `metal-gpu-intervals` table, parses it
and prints a per-channel table plus a CSV a script can assert on. See
[Reading the timeline](#reading-the-timeline) for what the numbers mean.

The table gives per-GPU-submission `start` and `duration` in nanoseconds,
`gpu-channel-name`, `start-latency` (the CPU→GPU gap), and `cmdbuffer-id` /
`encoder-id`, per process; `metal-application-encoders-list` and
`metal-command-buffer-completed` join on those ids. Five things to know before
using it:

- **`--attach <pid>` does not work** for this template — it reports "No
  configuration information received, will have to guess" and exports zero rows.
  Metal instrumentation has to be present at launch: use `--launch --` (or
  `--all-processes`, which does pick up a running process). The harness uses
  `--launch`, which is why a recording always starts at process launch.
- **Weight load leaves no rows**; prefill does. Load is CPU and file I/O, so it
  is simply absent from a GPU-interval table however long it took, and the
  traced process's very first row is already prefill. Nothing in the table marks
  where prefill ends, and no `--skip-ms` value discovers it — re-summarising one
  export at several skips just slides the window, `span(S) == span(0) - S` to
  under 0.3 ms. The boundary has to come from outside, so the harness reads the
  run's **own** `decode_profile{prefill_ms}` back out of
  `<RMLX_HOME>/logs/<run-id>.jsonl` and defaults `--skip-ms` to it. That event
  is a plain `info!`, so it is present at the default log level even though
  `xctrace --launch` swallowed the child's stdout. An explicit `--skip-ms` still
  wins; when no such event is found the harness says so and summarises the full
  window rather than pretending. `--skip-ms` counts from the matched process's
  own first submission, not from the start of the trace.
- **Tracing costs about 1–3% of decode throughput.** Alternating traced and
  untraced runs of the same cell, comparing each run's own
  `n_steps / step_total_ms`: gemma-4-e2b −2.8% (n=3 pairs, this machine under
  load) and −0.9% (n=3, independently, quiet). That straddles CLAUDE.md's ±1%
  regression band, which is why `--metrics off` is forced: a traced run's
  throughput must never be recorded.
- The export XML uses an **`id`/`ref` back-reference encoding** with positional
  columns and `<sentinel/>` for NULL. A naive parser silently misaligns columns,
  which reads as plausible-but-wrong numbers rather than an error.
  `rmlx_mlx::xctrace` refuses instead: it checks a row's cell count against the
  schema, checks every cell's tag against the column's declared
  `engineering-type`, and rejects an unresolvable `ref`.
- **Volume**: a bundle runs ~300–400 MB and the one-table export ~110 MB for a
  ~15 s recording. Bound it with `--time-limit`; `.rmlx/traces/mst` keeps the
  newest `--keep` bundles (default 5) and prunes the rest, sidecar XML and CSV
  included, on **every** exit — a run that refuses its arguments is exactly when
  bundles pile up, so a bound that only applied on success would not be one.
  `make traces-gc` does not cover this directory: it owns `.gputrace` bundles
  only, and its reported total is scoped to those.

Pipeline and function names do **not** survive the export (only encoder,
command-buffer, buffer and queue labels), and the driver coalesces consecutive
compute encoders into one GPU kick — so one row can cover several encoders. Pair
it with the identity list from a capture when you need to know *which* kernel.

#### Reading the timeline

Measured with `--kv-quant none`, a 4096-token prompt and 600 decoded tokens.
`--skip-ms` is the run's own reported `prefill_ms`, which is what the harness
defaults to:

| Model | `prefill_ms` | skip used | Compute subs | Compute busy / span | `start-latency` p50 |
|---|---|---|---|---|---|
| gemma-4-e2b-it-mxfp8 | 257.5 | 257 ms | 27 055 | 4908 / 5055 ms (97.1%) | 4.18 ms |
| Ternary-Bonsai-8B-mlx-2bit | 1372.7 | 1373 ms | 44 534 | 4504 / 4690 ms (96.0%) | 6.27 ms |

Note how far apart the two prefills are: a single hand-picked skip cannot serve
both. A 500 ms skip leaves roughly a fifth of Bonsai's window as prefill while
overshooting e2b's by 240 ms.

**Cross-check the window within the run, against the same run's log.** The
predictor is exact enough to be a gate:

| Model | full span | `prefill_ms + step_total_ms` | decode span | `step_total_ms` |
|---|---|---|---|---|
| gemma-4-e2b | 5311.96 ms | 5319.9 ms (0.15%) | 5054.86 ms | 5062.4 ms (0.15%) |
| Ternary-Bonsai-8B | 6063.09 ms | 6044.0 ms (0.32%) | 4689.73 ms | 4671.3 ms (0.39%) |

Do **not** cross-check against a decode rate from a *separate* untraced run: the
two runs differ by the observer effect above and by ordinary run-to-run spread,
and the comparison also has to add the prefill term back. Both windows are
printed by the harness for exactly this reason.

Run-to-run spread of the span is **0.04%–0.33%** (n=4 per model): 0.04% on
Bonsai, 0.33% on e2b. The 0.06% quoted for a toy compute workload is a floor,
not a bound.

**`start-latency` measures queueing in both regimes — it is not a host-stall
signal.** It is the gap between a command buffer being *created* (committed) and
starting on the GPU, so it can only ever see backlog. At 97% busy that is what
it reports: a 4.18 ms p50 against a 0.14 ms p50 duration is roughly 30
submissions deep on e2b, ~176 on Bonsai. But it does not rise when the host is
the bottleneck — it falls, because the queue is empty. Measured on a
deliberately host-bound cell (gemma-4-e2b `--kv-quant k_rotor3 --rotor-qjl on`,
which forces the rotor K path onto the CPU and decodes at ~4 TPS):

| cell | Compute busy / span | `dur` p50 | `start-latency` p50 |
|---|---|---|---|
| e2b `none` (saturated) | 4908 / 5055 ms (97.1%) | 0.144 ms | 4.18 ms |
| e2b `k_rotor3` + QJL (host-bound) | 416 / 7196 ms (5.8%) | 0.158 ms | **2.42 ms** |

**The host-stall signal is idle GPU time — `span - busy`**, which the harness
prints as `idle:`. In the host-bound row above that is 6.78 s of a 7.20 s
window; the run's own log agrees, at 249 ms per decode step. Read `idle` first,
and read `start-latency` as queue depth.

### Working with a bundle

A 6 GB bundle is opaque, and its layout is Apple's — not a stable contract. Four
scripts cover the operations that have actually been needed. Each one checks the
structure it depends on and fails loudly, by name, when the layout moves: an
empty list is never printed in place of "could not read this".

| Command | What it answers |
|---|---|
| `bash scripts/gputrace_summary.sh <bundle>` | What was captured (model, codec, prompt size, when — read back from the harness naming convention), total and command-stream size, and whether a `.gpuprofiler_raw` is present. |
| `bash scripts/gputrace_kernels.sh <bundle>` | Which Metal functions the window referenced, and which the capture layer recorded as unused. `--set used\|unused\|all`, `--names-only` for piping. |
| `bash scripts/gputrace_diff.sh <a> <b>` | What A's window referenced that B's did not, and vice versa — the codec-vs-codec or commit-vs-commit A/B. |
| `bash scripts/gputrace_preflight.sh` | The host-side prerequisites above, each with its fix. Also `make gputrace-preflight`. |

Worked example — the same comparison that first had to be done by hand, on two
captures of gemma-4-e2b at 4k taken minutes apart:

```console
$ bash scripts/gputrace_diff.sh <none>.gputrace <iso3_sym>.gputrace
shared: 37
only in A (0):
only in B (9):
  custom_kernel_rmlx_iso_flash_decode_symv_p1_b3
  custom_kernel_rmlx_iso_flash_decode_symv_p2
  custom_kernel_rmlx_iso3_quantize
  ...
```

Two limits worth knowing. Some function records store their name by object id
rather than inline; those are counted and reported (`… 46 named, 12 stored by
object id`) instead of silently dropped, so the named list is a subset, not the
whole set. And a `.gputrace` holds no dispatch *counts* — the command stream
references pipelines by object id, so "which kernels ran" is answerable offline
but "how many times" is not.

### Keeping `.rmlx/traces` bounded

Bundles are ~6 GB each — roughly the model's resident footprint — and a single
A/B session produces several. Unlike `target/`, they are not cheap to
regenerate: each is a model load plus a capture run. So the directory is
**capped**, not expired on a timer:

- keep the newest **6** bundles, and at most **40 GB** total;
- eviction is oldest-first, never the bundle just written, and every removal is
  printed with its reason and the space reclaimed;
- `scripts/gpu_capture.sh` enforces the cap after a successful capture — the
  point of a cap is to stop a session filling the disk, and an advisory the
  operator runs afterwards does not do that. Pass `--keep-all`
  (`make profile-gputrace … KEEP_ALL=1`) for a session that wants more.

```sh
make traces-gc                                   # report: what is over the caps
make traces-gc APPLY=1                           # enforce them
make traces-gc APPLY=1 MAX_COUNT=12 MAX_TOTAL_GB=80
bash scripts/traces_gc.sh --apply --max-age-days 7   # optional extra age rule
```

### Tests

The window policy, the request validation and the `xctrace` export parser are
pure and unit-tested, but the tests are behind the same feature, so a plain
`cargo test` does not compile them. Run them with:

```sh
make test-capture
```

`make ci` runs that target too — without it, an off-by-one in the window policy
would pass the gate green, since `make test` compiles these tests out entirely.

The parser's tests are mostly *refusals*, and deliberately so: each pairs a
fixture carrying one of the export encoding's traps with a mutated twin that
must be rejected — a dropped `<sentinel/>`, a one-column shift that keeps the
cell count intact, a dangling `ref`, a `ref` into a still-open element, a
non-numeric duration, the wrong table, and a filter that selects nothing. Every
one of those, accepted, yields a well-formed table of wrong numbers.

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

**Metal `peak_alloc_mb`** is a separate counter from the Metal Performance
HUD / `MTLDevice.currentAllocatedSize`.  It counts GPU-private VRAM allocations (weight
tensors, KV-cache MTLBuffers) and is disjoint from the `task_info` counters above — they
measure CPU/UMA host memory, not GPU-private usage.  On Unified Memory Macs the boundaries
blur (all memory is the same physical chips) but the accounting domains are distinct.

### 9.1 Scoping the Metal peak to a region

`rmlx_mlx::mlx_peak_memory_bytes()` alone is a process-lifetime high-water mark:
it carries the model load, every prior request, and anything else the process
did. That makes it a dashboard number, not something an assertion can be built
on. `rmlx_mlx::PeakBracket` scopes it:

```rust
let bracket = PeakBracket::open();   // record live bytes, zero the peak mark
// ... region under test, MATERIALISED (MLX is lazy: eval inside the bracket) ...
let reading = bracket.close();
```

| Accessor | Meaning |
|---|---|
| `peak_bytes` | Most bytes live at once inside the region. `0` if it allocated nothing. |
| `headroom_bytes()` | `peak - live_at_open` — what the region needed *on top of* the resident weights. The number to compare across two runs. |
| `transient_bytes()` | `peak - live_at_close` — allocated inside and released again. Catches a scratch buffer *larger* than the region's own steady state; a smaller one hides under the peak the surviving buffers reach anyway, and reads zero. Zero is not a no-scratch proof. |
| `observed_allocation()` | `headroom_bytes() > 0` — this region's live bytes rose above where they started. Assert this first; an upper bound is free to hold against a region that measured nothing. **Not** `peak_bytes > 0`: MLX updates the mark as `peak = max(peak, active)` where `active` is the whole live count, so after a reset one allocation anywhere in the process lifts `peak_bytes` to the full resident total, and the predicate would be true in every real process. |
| `measurable()` | The peak mark was actually zeroed at `open()`. When it is `false` the peak is still process-lifetime, so every accessor above returns 0 rather than a large, stable, plausible-looking delta. |

Two rules the pooling allocator imposes:

- **Never assert on an absolute byte count.** MLX reuses pooled buffers, so an
  absolute figure encodes what ran before as much as what ran now. Bound
  `headroom_bytes()` by a multiple of the workload's own size instead — see
  `q8_msl_roundtrip_allocation_stays_within_budget` in
  `crates/rmlx-kv-quant/src/q8_msl_tests.rs`.
- **Materialise inside the bracket.** MLX is lazy. An `eval()` after `close()`
  allocates after the mark has been read, and the bracket reports
  `peak_bytes: 0` — which `observed_allocation()` exists to catch.

The peak mark is process-global, so two brackets on different threads reset
each other. Scope one at a time.

`rmlx baseline` uses this to report `metal_peak_mb` (peak during
prefill+decode) and `metal_gen_alloc_mb` (that figure minus what was already
live). Only the second is comparable between two runs of the same model; the
first still carries the weights.

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
| GPU kernel identity (which kernels ran) | `make profile-gputrace CODEC=... MODEL=...` — see §5 |
| GPU timing + CPU→GPU gap | `make profile-mst MODEL=...` — see §5 |
| Ad-hoc branch counts | `eprintln!` + `counts` crate |
| Process memory snapshot | `rmlx_core::mach_mem::read_proc_mem()` — see §9 |
| Prefill-chunk override | `RMLX_PREFILL_CHUNK_<ARCH>=<n>` — see §10 |

---

## Xcode GPU counter replay — the only per-kernel counter path on this host

**This step needs a human.** It is a GUI workflow: Xcode's Metal Debugger cannot
be driven headlessly here, the `xcode` MCP bridge never answers `tools/list`, and
accessibility automation can select a navigator row but cannot make Xcode load
its editor. An agent can capture the bundle, open it, and read the exported CSV —
a person has to click Profile. **Ask the user; do not report the counters as
unavailable.**

Headless Metal System Trace is *not* a substitute: it exports zero rows for
`rmlx` on this host, and per-dispatch counter sampling is unsupported on M5 Max.
This GUI path is what remains.

### Capture

```bash
bash scripts/gputrace_preflight.sh          # Xcode selected, developer mode, get-task-allow
bash scripts/gpu_capture.sh --kv-quant <codec> --model <snapshot> \
     --prompt-tokens 8192 --skip 32 --steps 8 --keep-all
```

Keep `--steps >= 8`: a shorter window's kernel set is a strict subset of an
8-step one. Pass `--keep-all` when a sibling bundle must survive — the default
prune is oldest-first and has already deleted a comparison arm once. Bundles run
7-14 GB each.

### Replay (the human step)

1. `open -a Xcode <bundle>` — a `.gputrace` opens as **Debugging GPU Workload**,
   not a project. If no window appears, check the other displays and Spaces:
   `osascript -e 'tell application "System Events" to tell process "Xcode" to get position of every window'`.
2. Navigator: **Performance** (4th row). It reads *"Performance data not
   available"* with a **Profile...** button.
3. In the Profile sheet set **Performance State: Maximum** and **GPU Execution
   Mode: Serial**, then Profile.
   * *Maximum* because the roofline (614 GB/s on M5 Max) and every recorded slope
     assume full clocks; a reduced P-state deflates the bandwidth fraction and can
     make an issue-bound kernel look memory-bound.
   * *Serial* because per-dispatch attribution is the goal. Xcode warns *"Adds
     precision to the data report, but it doesn't represent runtime
     performance"* — that is the intended trade.
4. **Both arms of a comparison must use identical settings**, or the comparison
   is void.

### Read

* **Shaders** tab — cost %, `# SIMD Groups`, `# Allocated Registers`,
  `Spilled Bytes` per pipeline. Kernel dtype is visible in the name suffix
  (`_float_` / `float32` vs `_bfloat16_t` / `bfloat16`), which is how a dtype
  promotion shows up.
* **Counters** tab — export CSV (one row per encoder, ~600 rows, 22 limiter
  columns). The decisive columns:

| column | reads |
|---|---|
| `Integer and Conditional Limiter` | integer/address/control issue pressure |
| `Last Level Cache Limiter` | memory-bound |
| `Instruction Throughput Limiter` | issue-bound |
| `Kernel Occupancy` | resident threadgroups; falls as allocated registers rise |
| `Device Memory Bandwidth` | achieved GB/s — against 614 on M5 Max |

Group encoders by their own top limiter; dispatch counts identify the kernel
(e.g. 26 codec layers x 8 steps = 208 encoders). Store exports under
`.rmlx/analysis/<probe>/xcode/`.

### What is trustworthy under Serial

Limiters, occupancy, register counts, SIMD-group counts, kernel identity and
dtype are the precise half. Absolute wall-clock and absolute bandwidth are **not**
production numbers; the ON/OFF *ratio* is what carries. Never compare a profiled
millisecond against a `perf_ab.sh` millisecond.

Xcode may report `Sampled Cores 12 / 40`. Relative shares hold; absolute
extrapolation carries that uncertainty.
