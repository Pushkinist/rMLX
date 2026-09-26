# Perf Baseline

This doc holds the decode-throughput anchors of the three test-target models.
It also names the tools that measure a build against them. No program reads an
anchor from this doc. `scripts/regression_gate.sh` takes the anchor as
arguments, and `make canary-gate` reads `runs.db`.

## Canary anchors (release-perf)

Each anchor is the median `decode_tps` of `make canary` at the `auto` KV
default. `auto` is unquantised bf16 on every architecture (`docs/KV_QUANT.md`
"The auto default"). The anchors were measured at `bd729e36` on 2026-08-21,
with the `release-perf` profile, by `make canary`.

Hardware: M5 Max, bandwidth ceiling 614 GB/s. That is the host constant
`scripts/perf_ceiling.py` divides by; `--bandwidth-gbs` overrides it.

| model | kv_quant | decode_tps | stddev |
|---|---|---:|---:|
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | `none` (bf16) | 142.33 | 2.91 |
| mlx-community__gemma-4-e4b-it-mxfp8 | `none` (bf16) | 79.57 | 1.01 |
| mlx-community__Qwen3.6-35B-A3B-8bit | `none` (bf16) | 100.53 | 0.34 |

An anchor is a floor for the next run of the same instrument on the same
host. Absolute decode TPS drifts between runs on a busy host. Only an
interleaved A/B run supports a direction between two builds.

**Canary protocol.** `make canary` builds the `release-perf` binary and runs
`scripts/perf_canary.sh`:

- Models: the three above. `bash scripts/perf_canary.sh --include-26b` adds
  `mlx-community__gemma-4-26b-a4b-it-mxfp8`.
- Shape: `rmlx baseline --prompt-tokens 4096 --max-tokens 100
  --max-ctx 8192`, with no `--kv-quant`.
- Runs: 1 warmup discarded, 3 measured; median and sample stddev.
- `decode_tps` is `(n_generated - 1)` over the time between the first and the
  last token callback. Prefill is outside that window
  (`crates/rmlx-cli/src/commands/baseline.rs`).
- CSV: one row per model in `$RMLX_HOME/bench/perf_canary.csv`, with columns
  `ts_utc,git_sha,model,kv_quant,prompt_tokens,decode_tps,stddev,build_profile`.
  `kv_quant` is the codec name the run's own log states, or `auto` if that
  probe fails.
- `runs.db`: one further run per model with `rmlx baseline --record`. A
  failed record is a warning; the CSV row is still written.
- A `k8vturbo3` arm follows each model: 1 warmup, 3 measured, CSV row only.

**Gating.** Two gates read what the canary wrote:

- `make canary-gate SHA=<last-green-sha>` runs `rmlx metrics deltas
  --since-sha <SHA> --threshold-pct 3 --exit-code true` against `runs.db`.
  `CANARY_THRESHOLD_PCT` sets the threshold. Exit 0 is clean and 1 a
  regression. Exit 125 (a `git bisect` skip) means no `runs.db`, or no
  returned row with a baseline. A SHA that returns no rows exits 0.
- `scripts/regression_gate.sh <model> <baseline_tps> <baseline_stddev>
  [--tolerance PCT]` compares the last CSV row naming the model with the
  anchor given as arguments. The tolerance defaults to 3%. It widens to 5%
  when that row's stddev exceeds half the tolerance band. Exit 0 is within
  tolerance, 1 a regression, 125 a failed precondition (arguments, binary,
  CSV, or a missing or unparseable row).

The last CSV row for a model is its `k8vturbo3` arm, because the canary
appends that row after the `auto` row. `regression_gate.sh` therefore
compares the `k8vturbo3` median with the anchor.

**The canary decodes greedily.** `rmlx baseline` samples at temperature 0, on
the GPU-argmax path. A served request resolves its temperature in the order
`docs/SAMPLING.md` § "Defaults and resolution order" gives. A resolved temperature
above 0 takes the host-sampling path, which no canary run observes. Measure
it with `rmlx bench --temperature / --top-p / --top-k / --repetition-penalty`.

**The canary is a short-context instrument.** Its pinned shape is a 4096-token
prompt. A defect that engages only at longer contexts cannot move an anchor.
Read a green canary as "no short-context regression". Put a long-context claim
on a cell that runs long.

**The canary tracks one build over time. It cannot compare two.** All of a
model's measured runs happen together. When it is pointed at two builds in
turn, whichever ran second wears any drift. For any two-arm question, use
`--ab`.

## A/B comparison: `perf_canary.sh --ab` (`scripts/perf_ab.sh`)

Interleaved comparison of two arms. An arm is a binary plus extra
`rmlx baseline` arguments. `make canary-ab ARGS='…'` builds `release-perf`
and runs the same harness.

```bash
# two builds, same flags
bash scripts/perf_canary.sh --ab \
  --binary-a target/release-perf/rmlx.main \
  --binary-b target/release-perf/rmlx \
  --label-a main --label-b patch

# one build, two flag settings, on one model
bash scripts/perf_canary.sh --ab \
  --model "$RMLX_O_MODELS_ROOT/mlx-community__gemma-4-e2b-it-mxfp8" \
  --arm-a "--kv-quant k8v8" --arm-b "--kv-quant k8v4" \
  --allow-token-divergence
```

**Protocol.** The shape defaults to the canary's. Per model, each arm gets one
untimed warmup, which also records its reference token ids. Then `--slots`
measured slots (default 12) run in a balanced `ABBA BAAB ABBA` schedule. Both
arms occupy the same mean slot position, so a monotone drift cancels.
`--invert` complements the pattern. `--slots` must be a multiple of 4.

**Criterion, fixed before the run.** The arms are **SEPARATED** if and only if
their per-slot `decode_tps` ranges are disjoint. Anything else is
**INCONCLUSIVE**: no measured effect, not a small one. Under the null that the
arms are exchangeable, `P(disjoint) = 2 / C(slots, slots/2)`. That is
`2/924 = 0.00216` at 12 slots. A slot count whose null probability exceeds
0.05 is refused, so the floor is 8.

The probability is per comparison, and a run emits one verdict per model. The
header states the family size and computes `1-(1-p)^m`. Read a single
SEPARATED against that family figure.

The header also computes the relative standard error of a sample stddev,
`~1/sqrt(2(n-1))`, from `n = slots/2` per arm. No confidence interval and no
other p-value is computed. Read none into the ratio.

**The verdict and the taint are separate lines.** `VERDICT:` always carries
the rank test's answer. A contaminated run adds a `TAINTED:` line beside it and
exits 125.

**Guards.** Each refuses rather than producing a number that looks fine:

| Guard | Behaviour | Waiver |
|---|---|---|
| Indistinguishable arms (same binary digest *and* same args) | exit 125 before measuring | `--allow-null-arms` |
| `--metrics` in an arm's arguments | exit 125 before measuring | none |
| A non-numeric `--slots` / `--busy-pct` / shape option | exit 125 before measuring | none |
| `--slots` whose null probability exceeds 0.05 | exit 125 before measuring | none |
| Host not quiescent — any foreign process ≥ `--busy-pct` (default 25) of a core | exit 125 before measuring | `--allow-busy-host` (still exits 125 if the result is tainted) |
| A foreign process ran during any slot or across the comparison | `TAINTED:` line, exit 125 | none |
| A slot or the comparison could not be sampled for interference | `TAINTED:` line, exit 125 | none |
| Arms generate different token ids | exit 1 | `--allow-token-divergence` |
| A slot stops reproducing its own arm's warmup token ids | exit 1 | none |
| `rmlx serve` holds the Metal context | exit 125 (reported, never killed) | none |
| A slot emits no `decode_tps` / Metal memory reading / `token_ids` line | exit 125 | none |
| A slot reports `metal_peak_mb=0` — the bracket measured nothing | exit 125 | none |
| A slot generates fewer tokens than `--max-tokens` | exit 125 | none |

Four of those rows read the machine: host quiescence, per-slot and
whole-comparison interference, and the Metal-exclusivity check. The load
average is recorded beside them as context. Every other row reads the arms.

`--metrics` is refused in arm arguments because it is declared
`global = true`. An occurrence after the subcommand overrides the leading
`--metrics off`, and the slot would open the real `runs.db`.

**`--synthetic-arms` is not an escape hatch — it says the run is not a
measurement.** It declares the arms are stubs, for a caller that checks this
script's own logic. The machine is then not consulted: no quiescence probe, no
interference sampling, no exclusivity check and no load average. The run says
so in its header, on every slot line and in `waivers.synthetic_arms`.
`scripts/ingest/perf_ab_ingest.py` refuses such a result with no waiver. The
arm-reading guards still apply.

The boundary lives in `scripts/lib/cpu_snapshot.sh`, as `snapshot_ok` and
`window_not_sampled`. `scripts/bench_llama_ab.sh` takes the same
`--synthetic-arms` flag through it. Its result file carries `synthetic_arms`,
and `scripts/ingest/llama_ab_ingest.py` refuses that result with no waiver.

**Interference measurement.** The figure is the change in a process's
cumulative CPU time across a known window, per slot and across the whole
comparison. `ps -o pcpu` does not serve: on macOS it is a decayed figure that
lags a process pinning a core. Load average does not serve either: it sits at
3–5 on an idle desktop. Only processes in the closing snapshot are scored. A
process that starts and exits inside one window contributes nothing. A window
that could not be sampled reports `unmeasured` and taints.

**Correctness is folded in.** Every slot runs with `--emit-token-ids`. Its
token ids are compared with its arm's warmup reference, and the two references
with each other.

**Residency is reported next to throughput.** Each slot contributes two memory
figures. `metal_gen_alloc_mb` is the generation-scoped allocator peak. The
prefill working set, not the cache, can set that peak, and then a real KV delta
reads `+0.0 MB` there. `kv_cache_bytes` is `KvCache::resident_bytes` off the
`baseline` summary line: the cache itself. Where a slot's KV accounting
refused, the column reads `n/a` for that whole arm, never `0`.

**Never writes `runs.db`.** Every slot runs `--metrics off`, so the file is
never opened. Promoting an accepted comparison is a separate step:
`scripts/ingest/perf_ab_ingest.py` turns one result file into two
`docs/METRICS_DB.md` §8.5 RunRecords (`decode_tps_warm`, `kv_cache_bytes`). It
refuses a TAINTED run unless told otherwise, and carries the taint text into
`notes`.

The result lands in `$RMLX_HOME/bench/perf_ab/<timestamp>.json`, with the
host conditions, the statistics and the binary digests. Each slot is a full
`rmlx` process that writes its own `$RMLX_HOME/logs/<run-id>.jsonl`. A default
three-model run writes 42 of them, and each launch runs the log size-cap
rotation. Point `RMLX_HOME` at a scratch directory when that matters.

**Cost.** `2 + slots` process launches per model: 14 at the default.

**Scope.** Each slot is a separate process, so this is a two-process
comparison. Kernel selections are latched in `OnceLock` at first read, so one
process exercises one dispatch path.

**Selftests.** `make canary-ab-selftest` (`scripts/perf_ab_selftest.sh`, also
in `make ci`) mutation-checks the harness against stub binaries with planted
differences. It must report a planted ratio exactly, and nothing for two equal
arms. Every case passes `--synthetic-arms`. The cases that test host gating
supply the machine as `ps` and `pgrep` shims on `PATH`. Every run counts the
cases that took each route; a case that could reach this machine fails the
suite. `scripts/bench_llama_ab_selftest.sh` carries the same boundary, with
`ps` as its whole host surface.

`make canary-ab-host-gate-fixtures` (`scripts/perf_ab_host_gate_fixtures.sh`,
also in `make ci`) pins that boundary from both sides. The quiescence and
Metal-exclusivity gates still refuse a shimmed hostile host. A hostile and a
quiet host give the same verdict text under `--synthetic-arms`. The flag
waives no arm-reading guard.

## Per-codec × per-model cells

`make bench-codec-cell CODEC=<codec> MODEL=<snapshot>` runs
`scripts/bench_codec_cell.sh`. It runs `rmlx baseline --kv-quant <codec>`, 1
warmup and 3 measured runs, at `--prompt-len 4096` and `--max-tokens 100` by
default. The binary is `$RMLX_BINARY`, else `target/release-perf/rmlx`, else
`target/release/rmlx`. It appends three rows to
`$RMLX_HOME/bench/codec_cells.csv` and gates nothing.

| Column | Type | Meaning |
|---|---|---|
| timestamp | ISO-8601 | When the row was recorded |
| codec | string | The `--kv-quant` value |
| model | string | Snapshot directory basename |
| prompt_len | int | `--prompt-tokens` value |
| max_tokens | int | `--max-tokens` value |
| run_idx | int | 1, 2 or 3 |
| decode_tps | float | Decode tokens per second |
| prefill_tps | float | Prefill tokens per second |
| git_sha | string | `HEAD` at bench time, first 12 characters |

## Reading a decode rate against the bandwidth ceiling

`scripts/perf_ceiling.py` computes the bytes one decode step must stream. It
reads `config.json` and the safetensors headers, adds the KV bytes of the
codec at the given context, and divides by the host bandwidth. It reads no
tensor data and launches no model.

```sh
scripts/perf_ceiling.py --model "$RMLX_O_MODELS_ROOT/<snapshot>" \
    --kv-quant <codec> --ctx 4096 --max-ctx 8192 --no-db --json
```

The figure counts bytes a step must stream, not bytes it moves. Cache
residency, dequant scratch, MoE gather waste and sub-peak bandwidth all land
in the gap between the ceiling and the measured rate.

The KV term transcribes the engine's Rust byte model into Python.
`make check-kv-byte-model-parity` diffs the two per codec, per topology and
per head dimension. The engine is the oracle.

**Compare models on overhead, not on ratio.** The ratio of measured to ideal
step time is `1 + overhead_ms / ideal_ms`. The same fixed per-step cost reads
large on a small model and small on a large one. Across models of different
size, compare the absolute overhead. With `--json`, each row's
`ms_per_token_floor` is the ideal step time:

```
overhead_ms = 1000 / measured_tps - ms_per_token_floor
```

Within one model, the ratio is valid.

**What the census cannot size.** BitNet checkpoints store every packed
ternary linear weight as `u8`. The loader dequantizes each one to bf16 at load
(`dequant_trit_u8` in `crates/rmlx-models/src/bitnet/loader.rs`). The census
counts the packed byte size from the header, so it undercounts what a BitNet
decode step streams.

## Cross-backend cells

`scripts/bench_cell.sh` drives one bench cell on one backend: `rmlx`,
`mlx-lm`, `omlx`, `paroquant` or llama.cpp. Its llama.cpp arms load GGUF
files. No file loads in both families: the MLX backends read MLX safetensors,
llama.cpp reads GGUF. A llama.cpp cell set against an MLX backend compares
near-equivalent weight quants, such as `q8_0` against `mxfp8`, not identical
ones. State that assumption beside every such cell. A pair of two llama.cpp
builds on one GGUF file does not carry it.

## Iso trace phases

The iso update paths time their encode and dequant steps at `trace!` level.
Turn them on with `--log verbose` or `RUST_LOG=rmlx_kv_quant=trace`. No
event sets `target =`, so each event's target is its module path,
`rmlx_kv_quant::kvcache::update_iso`. All of them live in
`crates/rmlx-kv-quant/src/kvcache/update_iso.rs`.

| Function | Codecs | Phases |
|---|---|---|
| `iso_v_encode_decode` | `iso3`, `iso4`, `iso3_sym`, `iso4_sym` | `iso_encode`, `iso_dequant_gpu`, `iso_dequant_cpu`, `iso_vec_to_array` |
| `iso_k_only_k_side` | `k_iso3`, `k_iso4` | the same four |
| `iso_v_bulk_encode` | `iso3` | `iso3_encode` |

Every decode phase carries `phase`, `bits`, `ms`, `s_total`, `kv_h` and
`head_dim`; the V path adds `variant`. Each width emits them, with `bits` 3
or 4. `iso3_encode` carries `phase`, `ms`, `s_total`, `kv_h`, `head_dim` and
`site = "exit_prefill"`. It has no 4-bit counterpart.

**Where each phase is reachable.** None of them fires on the fast paths:

- `iso3_encode` does not fire in a served run. `iso_v_bulk_encode` runs only
  from `exit_prefill_iso_v`, and `exit_prefill` returns before it for `iso3`
  and `iso4`: those codecs build no packed store
  (`KvQuant::materialises_packed_store`).
- The `iso3` / `iso4` decode phases need a cache with no bf16 K seed.
  `exit_prefill` sets that seed for these codecs, and `update_iso_v` then
  decodes through the bf16 mirror.
- The `*_sym` and `k_iso*` decode phases fire only when the fused iso
  flash-decode path declines the step. That path takes every single-query GPU
  step at a supported shape: batch 1, and `head_dim` a power of two, a
  multiple of 4 and at most 512 (`crates/rmlx-kv-quant/src/kvcache/sdpa.rs`).
  The phases fire on the CPU device, on a step with more than one query, and
  at any other shape.

## Iso GPU dequant parity

`iso_v3_dequant_gpu_matches_dequant_cpu` and
`iso_k3_dequant_gpu_matches_dequant_cpu`
(`crates/rmlx-kv-quant/src/isoquant_msl_tests.rs`, GPU, `#[ignore]`) compare
the 3-bit iso GPU dequant with the CPU one. They assert 5e-3 per element and
a strict max|cpu-gpu| ≤ 1e-6, and print the maximum. Their failure message
cites the observed figure in `docs/KV_ROTATION_CODECS.md`.

## Prefix-index bench

`cargo bench -p rmlx-models --bench prefix_index_bench` compares the two
`PrefixIndex` implementations under `crates/rmlx-models/src/prefix_index/`:
`LinearScan` (`linear.rs`) and `RadixTree` (`radix.rs`). For each entry count
N in 1, 4, 16, 64 and 256, it fills a fresh index with N synthetic 8-block
entries. One iteration is a pass of 10 000 random lookups, half hits and half
misses. Criterion times that pass and reports lookups per second. Each run
also times one pass per strategy and N and appends a row to
`$RMLX_HOME/bench/prefix_index.csv`. The row holds ns per lookup and an
estimate of resident bytes.

`--prefix-index {linear|radix}` selects the strategy a prompt cache uses. The
default is `linear`; `radix` is opt-in.
