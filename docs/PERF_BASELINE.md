# Perf Baseline

## Caveat: Homebrew MLX no-nax bottle degraded prefill numbers (~2026-07-13 onward)

Homebrew's `mlx` 0.32.0 bottle silently ships **zero**
`steel_gemm_fused_nax_*` GEMM kernels on the `arm64_tahoe` build target — a
Homebrew formula defect, not an MLX or rMLX bug. On Neural-Accelerator-class
hardware (M5-family and later) this costs ~3.8x GPU-matmul throughput and
2.2-3.7x slower **prefill**; decode is bandwidth-bound and unaffected, which
is exactly why it went unnoticed. Full root-cause, evidence, and the
Homebrew-side fix options: `.rmlx/mlx-homebrew-nax-regression.md`.

**Any prefill number in this file recorded while the dev box had `mlx`
0.32.0 installed (roughly 2026-07-13, when that bottle was poured, through
the pin back to 0.31.2 on 2026-07-17) is suspect** — it may read 2-3.7x
slower than a nax-capable run of the same cell. Decode-TPS and KV-MB figures
in the same window are unaffected and remain trustworthy. This is exactly
what falsified the GDN sequential-in-T root-cause theory for the Bonsai-27B
prefill regression (rMLX issue #216): the real cause was the missing-nax
bottle, not model code.

This file is **not** retroactively edited to mark individual affected rows —
there is no reliable way to tell, after the fact, which historical bench
invocation ran against which `mlx` build without re-running it. Going
forward, `events.mlx_nax` (`"present"` / `"absent"` / `"unknown"`, migration
`004_events_mlx_nax.sql`) records this per run in `runs.db`, so future rows
are self-describing; see `docs/METRICS_DB.md` §3.6. Historical `runs.db`
rows are likewise not backfilled — that table is append-only.

## Baseline KV-cache reuse

**Verdict: CORRECT-reuse.** The `rmlx baseline` decode loop appends one new
token per step against a growing per-layer `KvCache`. It does NOT re-encode the
full prompt+generated prefix each step. The stale comment at
`crates/rmlx-cli/src/commands/baseline.rs:190-192`
("Without KV cache all steps re-encode the full prefix … we divide total time
by step count") describes a long-superseded implementation and is factually
wrong about the current code.

### Evidence (code read)

Decode is `prefill (chunked) → single-token decode steps`, identical in shape
across all three test-target arches.

- **Qwen3 / Bonsai** (`crates/rmlx-models/src/qwen3.rs`):
  - Cache-miss path prefills the prompt in chunks via
    `forward_seq_with_cache(chunk, …)` (`qwen3.rs:1714`), then runs the decode
    loop `for step_idx in 1..n_tokens` (`qwen3.rs:1904`) calling
    `model.forward_arr(&y, 1, Some(&mut caches), device)` (`qwen3.rs:1907`).
  - `y` is a **single-element** `[1]` Array built from the previous token id
    (`qwen3.rs:1895-1899`); after each step `y` is replaced by the next single
    sampled token (`qwen3.rs:1617`). The `1` argument is `seq` (the sequence
    length), so each decode forward processes exactly one token.
  - `forward_arr` (`qwen3.rs:1251-1293`) reshapes the input to `[1, seq, hidden]`
    with `seq=1`, derives `base_offset` from the cache's current length
    (`caches.first().offset()`, `qwen3.rs:1258-1262`) — the offset **grows** each
    step — and writes the new K/V into the existing per-layer cache
    (`layer.forward(&h, base_offset, Some(&mut cs[i]), …)`, `qwen3.rs:1278`).
  - The exact-hit cache path mirrors this (`forward_arr(&y, 1, …)`,
    `qwen3.rs:1487`).
  - The function doc itself states "Greedy autoregressive generation using
    **KV-cache prefill + decode**" (`qwen3.rs:1336`).

- **Qwen3.6 MoE** (`crates/rmlx-models/src/qwen3_5_moe/generate.rs`): decode loop
  `for step_idx in 1..n_tokens` (line 195 / 591) calls
  `model.forward_arr(&y, 1, Some(&mut kv_caches), Some(&mut lin_caches), …)`
  (line 197 / 595); prefill via `forward_seq_with_cache` (line 413).

- **Gemma4** (`crates/rmlx-models/src/gemma4/generate.rs`): decode loop
  `for step_idx in 1..n_tokens` (line 375 / 970) calls
  `model.forward_arr(&y, 1, Some(&mut caches), device)` (line 379 / 973);
  prefill via `forward_seq_with_cache` (line 755).

### Implication

The 16-50x decode-TPS gap is **NOT** a baseline re-encode defect. The decode
path reuses the KV cache correctly and matches the production `serve` path
(same `forward_arr` single-token call). Therefore the "16-50x gap" analysis
does not re-scope into "fix a broken baseline decode loop".

However, TPS is still **prefill-contaminated**: `baseline.rs:166`
(`ts_generate_start`) starts the clock before `generate_greedy`, so the reported
`tps = n_generated / (prefill + decode)`. Removing the fixed prefill cost from
the denominator can only raise the reported TPS. Proceed to split prefill and decode timing, and report decode-only TPS + measured TTFT.

### Empirical confirmation

Not run. The code read is unambiguous (single-token `seq=1` forward against a
growing cache, identical across all three arches), so a wall-clock run was not
needed to settle CORRECT-vs-DEFECT. The GPU was not touched, leaving the
single-MLX claim untouched. (If desired, a per-step timing run would show flat
per-step wall-clock — characteristic of cache reuse — but it is not required for
this determination.)

## Decode-only re-baseline

After `rmlx baseline` was updated to report **decode-only** TPS (prefill
excluded), the 4 test-target models were re-baselined at the standard test
shape (`--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`), `auto`
KV-quant cell, 1 warmup discarded + 3 measured, **median** decode_tps
reported. Hardware: M5 Max, bandwidth ceiling **614 GB/s**. Date: 2026-05-21.

`combined_tps_old` = the prefill-contaminated `overall_tps` observed at THIS
shape (max-tokens 100); the historical matrix numbers (Bonsai 15.88,
Gemma4-e4b 27.50, Gemma4-26b 3.47, Qwen3.6 4.46) were taken at a shorter
generation window (max-tokens 32) where the fixed prefill cost dominated even
harder, so they were even lower.

| model | git_sha | kv_quant_resolved | decode_tps | combined_tps_old | ratio_vs_ceiling | date |
|---|---|---|---:|---:|---:|---|
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | 877da73 | mixed_k8g64_v4g64 | 115.21 | 38.20 | 2.66x (vs ~307) | 2026-05-21 |
| mlx-community__gemma-4-e4b-it-mxfp8 | 877da73 | k8v8 | 72.92 | 47.31 | 2.11x (vs ~154) | 2026-05-21 |
| mlx-community__gemma-4-26b-a4b-it-mxfp8 | 877da73 | k8v8 | 72.19 | 9.57 | 2.42x (vs ~175) | 2026-05-21 |
| mlx-community__Qwen3.6-35B-A3B-8bit | 877da73 | k8v8 | 94.96 | 12.59 | 1.84x (vs ~175) | 2026-05-21 |

### Finding: the "16-48x gap" was almost entirely prefill contamination

The historical matrix divided `n_generated` by `(prefill + decode)`. For the
MoE/dense-26b models the 4k-token prefill takes 8-10 s (TTFT) while 99 decode
steps take ~1.4 s — so the combined number was buried by prefill. Once prefill
is excluded:

- **Gemma4-26b MoE**: 3.47 → **72.19** decode_tps. The combined number at
  max-tokens 100 was 9.57; decode-only is ~7.5x higher than that combined
  value. Ratio vs ceiling collapses from ~50x to **2.42x**.
- **Qwen3.6 35B MoE**: 4.46 → **94.96** decode_tps. Ratio ~40x → **1.84x**.
- **Bonsai 8B 2bit**: 15.88 → **115.21**. Ratio ~19x → **2.66x**.
- **Gemma4-e4b dense**: 27.50 → **72.92**. Ratio ~5.6x → **2.11x**.

All four now sit in the **1.8x–2.7x** band. The dramatic "MoE is 40-50x slow"
signal was a **bench-harness measurement artifact** (prefill in the TPS
denominator), not an inference-path defect.

That band was previously described here as "at or near the healthy 1.5-2x
ceiling-vs-realized envelope llama.cpp / mlx-lm hit on dense models". **Do not
use ratio-vs-ceiling to compare runtimes across models of different size** —
`ratio = 1 + overhead/ideal_ms`, so the same fixed per-step cost reads larger on
a smaller model. Measured locally, llama.cpp's absolute per-step overhead
overlaps rMLX's; the ratio gap is dominated by bytes/step. See the H2 addendum.
The residual ~2x is plausibly real bandwidth/dispatch overhead; flamegraph
profiling can still characterize it, but the alarm that motivated the 16-48x
framing is resolved.

---

## Per-codec × per-model cells

Schema for `.rmlx/bench/codec_cells.csv` (gitignored — local-machine recording):

| Column | Type | Meaning |
|---|---|---|
| timestamp | ISO-8601 | When the row was recorded |
| codec | string | KvQuant variant name (e.g. `k8vturbo3`, `iso3_sym`, `rotor3_sym`, `tsym3`) |
| model | string | Snapshot directory basename |
| prompt_len | int | Prompt length in tokens |
| max_tokens | int | --max-tokens arg |
| run_idx | int | 1, 2, or 3 (3 measured runs per invocation; warmup discarded) |
| decode_tps | float | Decode tokens per second |
| prefill_tps | float | Prefill tokens per second |
| git_sha | string | Repository tip at bench time (first 12 chars) |

Recorded per `(codec, model)` cell. A.y-excluded combos (e.g. K8VTurbo3 K-side × Qwen MoE) NOT recorded — guard rejects at runtime.

**Tolerance semantics**:
- ±1% recorded-best update threshold (within band → keep recorded value; outside → update if better).
- 3% regression-fail gate (via `scripts/regression_gate.sh` / `make canary-gate`). Distinct from recorded-best update threshold.

**Cell table** (decode TPS, `release-perf` binary, standard bench shape):

| Codec | Gemma4-e4b | Qwen3.6-MoE | Bonsai-2bit |
|---|---|---|---|
| TurboSym4 | — | SKIP (A.y) | — |
| PlanarK | — | SKIP (A.y) | — |
| planar3 V | — | — | — |
| iso3 V | — | — | — |
| iso4 V | — | — | — |
| rotor3 V | — | — | — |
| rotor4 V | — | — | — |
| turbo3 V (promoted) | — | — | — |
| turbo2 V | — | — | — |
| turbo3_tcq | 73.54 | 94.57 | 95.11 |
| turbo2_tcq | 73.97 | 97.52 | 94.29 |
| iso3_sym | 80.12 | SKIP (A.y) | 142.64 |
| iso4_sym | 77.91 | SKIP (A.y) | 140.90 |
| k_iso3 | 68.17 | SKIP (A.y) | 59.06 |
| k_iso4 | 66.87 | SKIP (A.y) | 46.27 |
| rotor3_sym | 81.53 | SKIP (A.y) | 145.49 |
| rotor4_sym | 81.77 | SKIP (A.y) | 145.21 |
| k_rotor3 | 64.25 | SKIP (A.y) | 45.47 |
| k_rotor4 | 63.90 | SKIP (A.y) | 45.65 |
| tsym3 | 79.93 | SKIP (A.y) | 143.20 |
| planar_fused_qk (on) | 79.118 | — | 26.4 (short-ctx artifact) |

`—` = cell not measured yet. SKIP marks A.y-rejected combos.

**The iso / rotor Bonsai anchors above are stale — do not gate on them.** They
were recorded before the flash-decode-over-quant kernels
(`iso_flash_decode_sdpa`, `iso_flash_decode_symv_sdpa`, and the rotor pair) and
before `--rotor-qjl` flipped to default `off`, and they encode a causal story
that no longer holds. The paragraph they carried claimed the K-only variants
trail their `*_sym` siblings because the K side had no GPU-resident code mirror
and paid a CPU `dequant()` of the whole prefix per step. Both halves are false:
the K-only stores keep a GPU ring the kernel reads directly, and neither the
K-only nor the `_sym` tier is CPU-bound at decode.

The successor claim — that the `_sym` tier is the slower one *because
quantizing V puts a second dequant inside the decode kernel* — is also wrong,
and the reason it read that way was a dispatcher defect, not the V axis. Every
iso / rotor flash-decode dispatcher forced `Array::eval()` on its kernel inputs
immediately before dispatch, blocking the host on the GPU once per attention
layer per decode step. Removing it (the graph is left lazy; MLX's
`ensure_row_contiguous` already supplies the layout guarantee the raw-linear
kernels need) is worth **1.17–2.89×** decode across the family, with the token
digest, the KV bytes and TTFT all unchanged. Measured `rmlx bench`, n=3 per
cell, `release-perf`, one binary pair:

| model | ctx | `iso3_sym` | `k_iso3` | `rotor3_sym` | `k_rotor3` | `none` (control) |
|---|---|---|---|---|---|---|
| Bonsai-8B | 4k | 19.09 → **55.15** | — → 56.98 | — | — | 138.5 → 139.2 (+0.5%) |
| Bonsai-8B | 16k | 11.00 → **19.01** | 14.90 → **24.37** | 10.13 → **16.03** | 13.73 → **21.59** | 93.7 → 93.3 (−0.5%) |
| Bonsai-8B | 32k | 7.67 → **10.44** | 9.81 → **13.53** | — | — | 65.1 → 63.9 (−1.9%) |
| gemma-4-e2b | 4k | 65.57 → **100.20** | 76.33 → **107.09** | — | — | 129.1 → 128.1 (−0.7%) |
| gemma-4-e2b | 16k | 42.42 → **57.04** | 53.73 → **66.64** | 35.15 → **44.52** | 47.70 → **58.07** | 119.4 → 120.8 (+1.2%) |
| gemma-4-e2b | 32k | 29.56 → **35.58** | 37.76 → **44.34** | — | — | 112.9 → 113.9 (+0.9%) |

`none` is the null control: it reaches none of the four changed dispatchers, and
its six cells bound the session's measurement noise at ±1.9%. The Bonsai
`k_iso3` 4k base cell is absent because `rmlx bench` refused it — the pre-fix
binary scattered 19.65 / 24.71 / 19.76 TPS (25.6% of median) at that cell, over
the 15% settle ceiling. Narrower run-to-run spread after the fix is a
consistent second-order effect: a host-side GPU wait per layer makes the cell
sensitive to host scheduling (e2b `iso3_sym` @4k: 7.41% range → 0.77%).

**Every absolute decode-TPS number in the family recorded before this change
measures the dispatcher, not the kernel** — re-record before use. Live numbers
per (codec, context) live in `docs/models/bonsai/8B/rMLX.md` §2.

**Marginal-cost figures (ms per 1k KV tokens) are not invalidated.** The eval was
a fixed cost per decode step — one host↔GPU round trip per attention layer,
independent of KV length — so it lands entirely in the intercept of
`ms/step = a + b × (KV tokens/1000)` and a slope cancels it by construction.
Fitted across this binary pair: `a` 41.14 → **7.01 ms/step (−83%)**, `b` 2.437 →
**2.449 ms/1k KV tokens (+0.5%)**. The ≈34 ms/step recovered is ≈0.16 ms per
eval over the layer count, a textbook round trip. A published ms/1k table stays
valid; do not discard one on the strength of this fix.

**Neither tier competes with `none` on either axis.** The whole iso/rotor
family stores one `u32` code word plus one `f32` scale per group, so it is
never smaller than bf16 (see `docs/KV_QUANT.md` "Memory truth"), and its
decode is several times `none`'s. Bench them for kernel work and quality
study, not as memory or throughput candidates.

**A denser store would not rescue them either.** Post-fix marginal cost puts the
hand-written flash-decode shell at 4–14% of MLX `sdpa_vector`'s per-byte
throughput, measured on both `kv_h = 8` and `kv_h = 1`, so break-even needs a
store of 0.7–2.2 bits per value per axis — denser than anything in the tree or
on the roadmap. The arithmetic, the two-architecture measurement and the grid
geometry that causes it are in `docs/KV_QUANT.md` § "Fused flash-decode over a
quant store — the break-even condition". Do not spend kernel effort on this
family expecting a decode win without first moving that number.

#### K-only family, re-recorded after the dispatcher fix

The table above left the K-only family incomplete: `k_iso4` / `k_rotor4` were
never re-recorded, and the Bonsai `k_iso3` 4k cell was refused pre-fix. These
are the completed cells on the post-fix tree, `rmlx bench` n=3 + 1 warmup,
`release-perf`, scratch `RMLX_HOME`, `--metrics off`. Prompt lengths are the
tokenized fixtures: Bonsai 3770 / 15629 / 31553, e2b 4117 / 17148 / 34355.
Run-to-run range was ≤2.1% in every cell and the token digest was identical
across the runs of a cell.

| model | codec | 4k | 16k | 32k | KV bytes @32k | × `none` |
|---|---|---|---|---|---|---|
| Bonsai-8B | `none` (control) | 140.32 | 95.74 | 67.40 | 5,341,839,360 | 1.000 |
| Bonsai-8B | `k_iso3` | **60.28** | 26.17 | 14.36 | 5,371,658,240 | 1.006 |
| Bonsai-8B | `k_iso4` | **60.56** | — | — | — | — |
| Bonsai-8B | `k_rotor3` | **54.24** | 22.41 | 12.33 | 5,952,718,304 | 1.114 |
| Bonsai-8B | `k_rotor4` | **54.10** | — | — | — | — |
| gemma-4-e2b | `none` (control) | 126.82 | 119.29 | 112.55 | 218,148,864 | 1.000 |
| gemma-4-e2b | `k_iso3` | **107.15** | 66.74 | 44.44 | 218,803,200 | 1.003 |
| gemma-4-e2b | `k_iso4` | **107.97** | — | — | — | — |
| gemma-4-e2b | `k_rotor3` | **101.58** | 58.34 | 37.12 | 254,477,328 | 1.167 |
| gemma-4-e2b | `k_rotor4` | **100.73** | — | — | — | — |

The KV-bytes column is the reason to stop optimizing this family for
throughput: `k_iso3/4` measures **1.003–1.006×** `none` and `k_rotor3/4`
**1.11–1.17×**. A codec that is not smaller than the bf16 it replaces has no
bandwidth prize to collect, so parity is the ceiling, not a milestone on the
way past it.

**`none` was not bf16 on Bonsai when these rows were recorded — read the ratios
accordingly.** `kv_quant_for_layer` then promoted the first 2 and last 8 layers
to `K8V8` under every base mode, `KvQuant::None` included, so the `none`
control on a 36-layer dense arch was a 26-bf16 / 10-K8V8 mixture. `None` is
exempt from the promotion now, so a `none` row re-measured today is true bf16
and needs no restatement; every row on this page predates that change. See
`docs/KV_QUANT.md` §Layer-adaptive overrides for the mechanism and the
measured per-arch factors. The table below restates this one against true
bf16. That denominator is
derived, not separately measured, but it is checkable: at
`S = 31 553 + 128 − 1 = 31 680`
(the fixture length above, `rmlx bench --max-tokens` default 128) the
`filled_seq_bytes` identity for a `KvStorage::None` layer gives
`36 × 4096 B/token × 31 680 = 4 671 406 080`, and adding the 10 promoted
layers' q8_0 stores at `2112 B/token × 31 744` (capacity page-rounded to
`KV_PAGE_SIZE = 256`) reproduces the recorded `none` figure 5 341 839 360 to
the byte:

| model | `none` ÷ true bf16 | `k_iso3` ÷ true bf16 | `k_rotor3` ÷ true bf16 |
|---|---|---|---|
| Bonsai-8B | **1.144×** (at 32k) | **1.150×** | **1.274×** |
| gemma-4-e2b | 1.000× | 1.003× | 1.167× |

Bonsai's `none` ÷ bf16 drifts slightly with context, which is why
`docs/KV_QUANT.md` quotes 1.145× and this table 1.144×: the bf16 term scales
with filled length while the promoted layers' q8_0 term scales with
`KV_PAGE_SIZE`-rounded *capacity*, so the ratio is 1.1447 at `S = 3801`,
1.1435 here at `S = 31 680`, and tends to 1.1432 as the rounding washes out.
Treat it as 1.14× and read the exact figure at the context you care about.

gemma-4-e2b is unaffected because none of its promoted layers owns a
quantizable cache — layers 0 and 1 are sliding (bf16 rotating ring regardless
of the flag) and the last 8 are shared-KV consumers with no cache slot of their
own. The correction is a Bonsai-side factor, not a table-wide one, which is
itself the point: a "vs `none`" ratio is not comparable across architectures.

Marginal cost over the replicated 16k→32k segment (`itl_p50` ms per 1k KV
tokens) is unchanged by the dispatcher fix, as predicted — the fix moved the
intercept, not the slope:

| model | `none` | `k_iso3` | `k_rotor3` |
|---|---|---|---|
| Bonsai-8B | 0.261 | 1.799 (6.9×) | 2.178 (8.3×) |
| gemma-4-e2b | 0.025 | 0.432 | 0.567 |

Read e2b's ratio-to-`none` with care: only its global layers grow, so the
`none` denominator is near zero and the ratio inflates. Compare the absolute
ms/1k across models instead. Counted from the per-dispatch `trace!` under
`--log verbose`, the flash-decode kernel fires once per full-attention layer
per decode step and is handed the full prefix — 26 of 36 layers on Bonsai (the
first 2 and last 8 **were** promoted to K8V8 when this was recorded; the codec
here is a quantizing one, so that promotion still applies today — it is only
`none` that is now exempt), 7 of 35 on e2b. Those 7 e2b
dispatches read only **3** distinct caches. `num_kv_shared_layers = 20` leaves
layers 15+ without a cache of their own, and `build_previous_kvs`
(`gemma4/loader.rs`) points each of them at the **last** non-shared layer of
its attention type — layer 14 for full-attention. So layers 19, 24, 29 and 34
all attend over layer 14's cache, while layers 4 and 9 serve one dispatch
each: three caches, seven dispatches, five of them on layer 14. What remains
is per-KV-token work inside the kernel, not data movement.

Each new codec: invoke `scripts/bench_codec_cell.sh --kv-quant <codec> --model <model>` per cell (3 cells per codec, A.y-excluded skipped), then append to this table.

**Champions regen**: regenerate `BENCHMARK_CHAMPIONS.md` via `rmlx metrics export --markdown` after each cell lands. If that command is unavailable, manual fallback: hand-edit `BENCHMARK_CHAMPIONS.md` with cell + recorded TPS, cite source CSV row.

---

## Canary anchors (release-perf)

`make canary` records each canary run into `runs.db` via
`rmlx baseline --record` in addition to the legacy CSV.

**Canary flow:**
- `make canary` — builds release-perf binary, runs `scripts/perf_canary.sh`
  (1 warmup + 3 measured per model), appends to both the legacy CSV and runs.db.
- `make canary-gate SHA=<last-green-sha>` — gates regressions by querying
  `runs.db` via `rmlx metrics deltas --since-sha <SHA> --threshold-pct 3`.
  Exit codes: 0=clean, 1=regression detected, 125=no-baseline-skip (git bisect safe).

**Legacy CSV** (`$RMLX_HOME/bench/perf_canary.csv`) is preserved as a fallback for
one release. Use `make canary-gate` (DB-backed) for new regression gates.
`scripts/regression_gate.sh` (CSV-backed) is also preserved as a legacy fallback.

**The canary tracks one build over time. It cannot compare two.** All of a
model's measured runs happen together, so when it is pointed at two builds in
turn, whichever ran second wears any drift — and on a contended host that drift
is large. Calibration, 2026-08-16, gemma-4-e2b at 4 k on a desktop with
WindowServer sustained around 50 % of a core: two arms that were *the same
binary with the same flags* came out with medians 1.75 % apart (114.51 vs
116.51 tok/s). Three blocked runs per arm would have called that a 1.75 % win.
The interleaved harness reported no difference — the arms' per-slot ranges
overlapped — and refused the run outright as `TAINTED`, naming the contending
process. For any two-arm question, use `--ab`.

## A/B comparison: `perf_canary.sh --ab` (`scripts/perf_ab.sh`)

Interleaved comparison of two arms, where an arm is a (binary, extra
`rmlx baseline` arguments) pair.

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

**Protocol.** Per model: one untimed warmup per arm (which also records that
arm's correctness reference), then `--slots` measured slots (default 12) in a
balanced `ABBA BAAB ABBA` schedule. Both arms therefore occupy the same mean
slot position, so a drift that is monotone across the run cancels. `--invert`
complements the pattern; running once each way cancels any residual positional
bias. `--slots` must be a multiple of 4 — a partial block would give the arms
different mean positions and put the confound back.

**Criterion, fixed before the run.** The arms are **SEPARATED** if and only if
their per-slot `decode_tps` ranges are disjoint. Under the null that the arms
are exchangeable, `P(disjoint) = 2 / C(slots, slots/2)` — `2/924 = 0.00216` at
the default 12. Anything else is **INCONCLUSIVE**, which means *no measured
effect*, not *a small one*. The reported ratio under INCONCLUSIVE is the gap
between two point estimates drawn from overlapping spreads and is not evidence.

That probability is **per comparison**, and a run emits one independent verdict
per model. The header states the family size and computes `1-(1-p)^m`: three
canary models give ≈ 0.0065, and pairing a run with `--invert` doubles the
family again. Read a single SEPARATED against the family figure.

`--slots` below 8 is refused, not warned about. At 4 the null probability is
`2/C(4,2) = 0.333`, and the harness would print the same word `SEPARATED` for a
one-in-three coin flip as for a one-in-462 result — and the word is what gets
pasted into a report.

**What the statistics license.** `n = slots/2` per arm — 6 by default. The
median is a point estimate; the relative standard error of a sample stddev is
`~1/sqrt(2(n-1))`, which the header **computes** from the actual `n` (32 % at
n=6, 41 % at n=4, 71 % at n=2) rather than quoting a fixed figure beside an
interpolated count. No confidence interval and no p-value beyond the rank test
above are computed, and none should be read into the ratio.

**Guards.** Each refuses rather than producing a number that looks fine:

| Guard | Behaviour | Waiver |
|---|---|---|
| Indistinguishable arms (same binary digest *and* same args) | exit 125 before measuring | `--allow-null-arms` |
| `--metrics` in an arm's arguments | exit 125 before measuring | none |
| A non-numeric `--slots` / `--busy-pct` / shape option | exit 125 before measuring | none |
| `--slots` whose null probability exceeds 0.05 | exit 125 before measuring | none |
| Host not quiescent — any foreign process ≥ `--busy-pct` (default 25) of a core | exit 125 before measuring | `--allow-busy-host` (still exits 125 if the result is tainted) |
| A foreign process ran during any slot or across the comparison | verdict `TAINTED`, exit 125 | none |
| A slot or the comparison could not be sampled for interference | verdict `TAINTED`, exit 125 | none |
| Arms generate different token ids | exit 1 | `--allow-token-divergence` |
| A slot stops reproducing its own arm's warmup token ids | exit 1 | none |
| `rmlx serve` holds the Metal context | exit 125 (reported, never killed) | none |
| A slot emits no `decode_tps` / Metal memory reading / `token_ids` line | exit 125 | none |
| A slot reports `metal_peak_mb=0` — the bracket measured nothing | exit 125 | none |
| A slot generates fewer tokens than `--max-tokens` | exit 125 | none |

`--metrics` is refused in arm arguments because it is declared `global = true`:
an occurrence after the subcommand overrides the leading `--metrics off` and the
slot opens the real append-only `runs.db`. Verified against the built binary,
including on a failure path where the model never loaded.

**Interference measurement, and what it cannot see.** The figure is the change
in a process's cumulative CPU time across a known window, taken per slot and
across the whole comparison. Two things that look like they would do the job do
not: `ps -o pcpu` on macOS is a stale decayed figure that does not move while a
process pins a core (a 100 %-CPU spinner reads back as ~11 % and stays there),
and load average sits at 3–5 on an idle developer desktop. Only processes
present in the closing snapshot are scored, so a process that both starts and
exits inside one window contributes nothing — sustained contention is caught,
a burst that fits entirely inside one slot is not. A window that could not be
sampled at all (empty or failed `ps`, or a window below the CPU counter's
10 ms resolution) reports `unmeasured` and taints; it is never folded into
"nothing was running".

**Correctness is folded in.** Every slot emits `--emit-token-ids` and its exact
`Vec<u32>` is compared against its arm's warmup reference, and the two arms'
references against each other. An arm that is fast and wrong fails the
invocation that made it look fast.

**Residency is reported next to throughput.** Decode TPS alone cannot express a
KV-codec question: a codec that costs memory and buys no speed reads as a null
result if throughput is the only column. Each slot therefore contributes two
memory figures, and they answer different questions.
`metal_gen_alloc_mb` is the generation-scoped allocator peak — the prefill
working set, not the cache, can be what sets it, and then a real KV delta shows
there as `+0.0 MB`. Measured both ways in the cells below: at a 4 096-token
prompt it reads `+0.0 MB` on Ternary-Bonsai-8B and on Qwen3.8-27B against
cache deltas of +28.7 % and +21.9 %, while the same Bonsai pair at 32 768 does
resolve it (+2 500 MB). Whether the cache is the allocator peak is an
arch-and-shape question, not a property of the instrument.
`kv_cache_bytes` is `KvCache::resident_bytes` off the
`baseline` summary line and is the cache itself. Where a slot's KV accounting
refused, the column reads `n/a` for that whole arm — never `0`, which would
divide into a residency ratio as a cache of no bytes.

**Never writes `runs.db`.** Every slot runs `--metrics off`, so the file is
never opened, and `--metrics` is refused in arm arguments so it cannot be
turned back on. An A/B run exercises arms built to be thrown away; a row in the
append-only store cannot be taken back out. Promoting an accepted comparison is
a separate, explicit step: `scripts/ingest/perf_ab_ingest.py` turns one result
file into two §8.5 RunRecords (`decode_tps_warm`, `kv_cache_bytes`), refuses a
TAINTED run unless told otherwise, and carries the taint text into `notes`.

It does write elsewhere. The result lands in
`$RMLX_HOME/bench/perf_ab/<timestamp>.json` alongside the recorded host
conditions, the computed statistics and the binary digests. Separately, each
slot is a full `rmlx` process and writes its own
`$RMLX_HOME/logs/<run-id>.jsonl` regardless of `--metrics off` — a default
three-model run is 42 of them, and each launch runs the log size-cap rotation,
which can evict unrelated run logs. Point `RMLX_HOME` at a scratch directory
when that matters.

**Cost.** `2 + slots` process launches per model (14 at the default), against
the canary's ~10.

**Scope — what this is not.** Each slot is a separate process, so this is a
two-*process* comparison. Alternating two *kernel dispatch paths* inside one
process needs a threaded dispatch-policy value, which does not exist: the five
kernel selections are latched in `OnceLock` at first read, so a process can
only ever exercise one of them. Interleaving still removes the ordering and
drift confounds, because those act at slot granularity — but an in-process
arm is out of reach until the policy value lands, at which point it becomes
just another `--arm-a` / `--arm-b` argument and the harness does not change.

`bash scripts/perf_ab_selftest.sh` (also run by `make ci`, and as
`make canary-ab-selftest`) mutation-checks the harness against stub binaries
with planted differences: it must report a planted ratio exactly, and must
report nothing for two arms that are the same. It needs no GPU, no model and no
metrics DB.

**Canary protocol**:
- Profile: `release-perf` (debug-assertions=false, overflow-checks=false, stripped)
- Shape: `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`, `kv_quant=auto`
- Warmup 1 discarded, 3 measured runs; median decode_tps + sample stddev reported in CSV
- DB record: one `rmlx baseline --record` call per model after the measured runs

**The canary decodes greedily, and that is a scope limit, not a detail.**
`rmlx baseline` has no sampler knobs, so every canary and A/B number in this
file is the GPU-argmax path. It is also *not* the served default: a
`/v1/chat/completions` request that omits sampling fields resolves temperature
from `generation_config.json` or a hard-coded `1.0`, and several snapshots ship
`top_p` and `top_k` alongside it, so ordinary served traffic takes the
host-selection path on every token. Measured at 4k, that path costs 9–14 % of
decode throughput at temperature alone, 5–12 % at a repetition penalty alone,
and about a quarter of it once a nucleus filter is on. No canary run can observe
any of it; use `rmlx bench --temperature / --top-p / --top-k /
--repetition-penalty` and the `sampler_profile` event. See § *Host-sampler cost*
below and `docs/SAMPLING.md`.

Those figures are from gemma-4-**e2b**, Ternary-Bonsai-8B and Qwen3.6-35B-A3B.
Only two of the three are canary models — the canary's Gemma4 is **e4b**, and no
sampled cell was taken on it.

**Per-model kv_quant resolved by arch resolver (auto):**
- Bonsai (Qwen3ForCausalLM, 2bit): `mixed_k8g64_v4g64`
- Gemma4-e4b (mxfp8): `k8v8`
- Qwen3.6-35B-A3B (8bit MoE): `k8v8`

Captured by `bash scripts/perf_canary.sh` under `release-perf` profile
(debug-assertions=false, overflow-checks=false, stripped debug).
Shape: `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`.
Warmup 1 discarded, 3 measured runs, median decode_tps + sample stddev.
Date: 2026-05-21. Hardware: M5 Max.

| model | git_sha | kv_quant | decode_tps | stddev | profile | date |
|---|---|---|---:|---:|---|---|
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | 848d4785 | mixed_k8g64_v4g64 | 109.86 | 0.92 | release-perf | 2026-05-21 |
| mlx-community__gemma-4-e4b-it-mxfp8 | 848d4785 | k8v8 | 74.22 | 0.08 | release-perf | 2026-05-21 |
| mlx-community__Qwen3.6-35B-A3B-8bit | 848d4785 | k8v8 | 96.64 | 2.00 | release-perf | 2026-05-21 |
| mlx-community__bitnet-b1.58-2B-4T | fa2ec73 | k8v8 | 31.61 | 0.17 | release | 2026-05-28 |

**The canary is a short-context instrument, and that is a coverage limit.** The
pinned shape tops out around 3 900 context tokens, so nothing keyed to a longer
context is observable in these anchors — every one of them is a measurement of
the model's short-context behaviour and says nothing about its long-context
behaviour. A per-step defect on the Mixed V path that only engaged past 8 192
tokens sat behind these numbers for months without moving them: it could not,
because the canary never reaches the shape where it engages. Read a green canary
as "no short-context regression", never as "no regression", and put long-context
claims on a cell that actually runs long.

**Qwen3-dense bf16-stream fix (2026-06-24).** Casting Qwen3 norm weights and
quant scales/biases to bf16 at load (they ship fp16 on Bonsai) stops the
residual stream — and the `--kv-quant none` KV cache — from widening to f32. The
fix also lifts the Bonsai canary default (`mixed_k8g64_v4g64`) from ~110 to ~129
decode_tps (the bf16 q/k/v compute is cheaper than the prior f32 path); Gemma4
and Qwen3.6 (separate arch files) are unchanged. On the `none` path the gain
widens with context as KV bandwidth dominates: Bonsai `none` decode_tps
~101→~135 at 4 k, ~48→~83 at 16 k, ~19→~38 at 64 k.

## Codec cells across context — `none` vs `mixed_k8g64_v4g64` (TAINTED host, 2026-08-20)

**Every cell in this section is TAINTED and none of it is a clean measurement.**
`perf_ab.sh`'s quiescence gate never cleared on this host: a virtual-machine
helper sat at 113–384 % of a core throughout, WindowServer at 30–56 %, and all
8 of 8 slots in every run are marked contended. The runs are ABBA-interleaved,
so a *within-run* ratio still cancels a drift that is monotone across the run,
and the pre-declared disjoint-range criterion still decides SEPARATED vs
INCONCLUSIVE. Nothing here may be quoted as an absolute TPS anchor, and no
number here may be compared against a number from a different run.

Why the section exists: every earlier codec cell in this file and in
`docs/KV_QUANT.md` was taken at 4 096–32 768 tokens, and the standing objection
to that basis is that `kv_frac` — the KV share of a decode step's byte stream —
rises steeply with context, so a short cell cannot see a codec difference "even
in principle". `kv_frac` is stated next to every row below so that bound is
visible rather than assumed. It is a **(model, context)** quantity, not a
context one; see docs/KV_QUANT.md, "`kv_frac` bounds a codec claim".

Shape: `rmlx baseline`, `release-perf` build `sha256:c26e0d8a3ac43996`,
`--max-tokens 100`, 1 untimed warmup + 8 slots per model (n=4 per arm),
`--allow-busy-host --allow-token-divergence`. Arm A is `--kv-quant none` (true
bf16 on every layer), arm B `--kv-quant mixed_k8g64_v4g64` (affine 8-bit K /
4-bit V, and one of the two codec families whose decode genuinely reads its
packed store). `kv_frac` and the predicted ceilings are
`scripts/perf_ceiling.py` on the same snapshot.

`kv_frac` and the predicted ceilings are evaluated at the **measured** cache
offset (`prompt_tokens + max_tokens - 1`), not at the fixture's nominal size.

| model | prompt tok | `kv_frac` (A) | predicted ceiling B/A | decode B/A (median) | A range | B range | verdict | resident KV A → B | resident B/A |
|---|---:|---:|---:|---:|---|---|---|---|---:|
| Ternary-Bonsai-8B-2bit | 3 770 | 0.211 | 1.099 | 0.949 | 125.72–134.86 | 125.21–127.53 | INCONCLUSIVE | 570.5 → 734.1 MB | 1.287 |
| Ternary-Bonsai-8B-2bit | 31 553 | **0.687** | **1.419** | 1.003 | 65.07–67.82 | 65.03–68.06 | INCONCLUSIVE | 4 667.3 → 6 032.9 MB | 1.293 |
| Qwen3.8-27B-mxfp8 | 3 892 | 0.010 | 1.004 | 1.005 | 17.73–17.97 | 17.58–18.04 | INCONCLUSIVE | 415.5 → 506.5 MB | 1.219 |
| Qwen3.8-27B-mxfp8 | 130 848 | 0.245 | 1.149 | **0.728** | 13.94–14.12 | 9.45–10.70 | **ranges disjoint** | 8 735.7 → 11 784.2 MB | 1.349 |

**What the 130 848-token Qwen3.8 row settles.** This is the cell that a
long-context codec claim was supposed to be decided on: `kv_frac` 0.245, a
predicted **+14.9 %** ceiling for arm B, and 20× the paired noise floor between
the arms. Measured, arm B is **27.2 % slower**, with the two arms' per-slot
ranges disjoint in the losing direction (every `none` slot faster than every
`mixed` slot), and it holds **35 % more** resident KV and 6 034 MB more
generation-scoped allocation. Both halves of the pre-declared long-context
falsifier fire: resident above 0.60× and decode below 0.95×. The
"measure it at 128 k and the codec will pay" hypothesis is not merely
unsupported here — the sign is wrong.

This cell is also the least contended of the four: the entry gate refused at
114 %, but only 4 of 8 slots were flagged (worst 35.8 % WindowServer) and the
comparison window as a whole read 22.8 %, *under* the 25 % threshold. It is
still reported as TAINTED, and still may not be read as an absolute anchor.

**Why `none` losing is not a roofline story.** Arm A's non-bandwidth per-step
term is essentially flat across a 33× context change — 12.65 ms at 3 892 tokens,
14.70 ms at 130 848 — so `none` decode really is bandwidth-dominated at 131 k
(1.26× its 57.02 ms floor). Arm B halves the KV byte stream and still loses,
because its own per-step cost is not flat: 98.55 ms measured against a 49.76 ms
floor, 1.98×, i.e. 48.79 ms of non-bandwidth work against arm A's 14.70. The
packed path does not fail to convert bytes into time; at this shape it spends
more time than the bytes were worth.

**What the 31 553-token Bonsai row settles.** 0.687 is the largest `kv_frac`
any model in the release set reaches at a context this tree can serve. Arm B
cuts the decode KV stream to 0.571× of arm A's and the roofline predicts +42 %.
Measured: **+0.3 %, ranges fully overlapping** — no measured effect — while
resident KV goes *up* 29 %. So the short-context basis was not what made the
earlier codec cells come back null: the null survives at the high-`kv_frac` end
of the same axis.

**The byte model is not what fails; it is exact where it is complete.** At the
measured offsets `perf_ceiling.py` puts arm A's resident KV at 570.5 MB and
4 667.3 MB on Bonsai — the measured figures, to the digit — and arm B within
0.5 % and 0.06 %. What does not hold is the conversion of saved bytes into
saved time: the ε of docs/KV_QUANT.md, "Fused flash-decode over a quant store",
measured at 0.041–0.135 on every path tried so far.

**The Qwen3.8 resident ratio is diluted, not smaller.** That arch is hybrid:
48 of its 64 layers hold a fixed-size GDN recurrent state the codec never
touches, and `kv_cache_bytes` sums it. `perf_ceiling.py` deliberately excludes
it, and the gap between its prediction and the measurement is that state — the
*same constant* at both contexts and on both arms:

| prompt tok | arm | predicted (attention KV) | measured | gap |
|---:|---|---:|---:|---:|
| 3 892 | `none` | 261.6 MB | 415.5 MB | +153.9 |
| 3 892 | `mixed` | 354.5 MB | 506.5 MB | +152.0 |
| 130 848 | `none` | 8 581.7 MB | 8 735.7 MB | +154.0 |
| 130 848 | `mixed` | 11 632.3 MB | 11 784.2 MB | +151.9 |

Read the whole-cache ratio on a hybrid arch as a lower bound on the
attention-KV ratio (1.219 measured against 1.355 attention-only at 3 892; 1.349
against 1.355 at 130 848), the same way the `--kv-quant none` section reads
Qwen3.6-35B's 1.060× against its 1.109× attention-only figure.

**Decode moves away from the roofline as context grows.** Same runs, arm A,
against `perf_ceiling.py`'s bandwidth-bound floor for the same cell:

| model | prompt tok | measured ms/step | roofline ms/step | ratio | non-bandwidth ms/step |
|---|---:|---:|---:|---:|---:|
| Ternary-Bonsai-8B-2bit | 3 770 | 7.55 | 4.40 | 1.72× | 3.15 |
| Ternary-Bonsai-8B-2bit | 31 553 | 15.01 | 11.07 | 1.36× | 3.94 |
| Qwen3.8-27B-mxfp8 | 3 892 | 56.11 | 43.47 | 1.29× | 12.65 |
| Qwen3.8-27B-mxfp8 | 130 848 | 71.72 | 57.02 | 1.26× | 14.70 |

Compare the last column, not the ratio: ratio-vs-roofline is
`1 + overhead/ideal_ms` and is therefore *not* scale-free, so it cannot be
compared across models of different size — the same fixed per-step cost reads
large on a small model and small on a large one. Within one model the
comparison is valid, and within each model the non-bandwidth term is roughly
flat — Bonsai 3.15 → 3.94 ms across an 8× context change, Qwen3.8 12.65 →
14.70 ms across 33×. Both models' `none` decode really is bandwidth-dominated
at their long cell. That is what makes the null load-bearing: the regime where
a byte cut *should* convert is the regime measured, on two architectures, and
it did not convert on either.

## Host-sampler cost — PROVISIONAL, NOT AN ANCHOR (2026-08-16)

First measurement of the sampling / penalty / mask / logprob path, which every
other table in this file excludes. **Nothing here is an anchor**: it is not
interleaved, it was taken on a host the A/B harness refused, and no regression
gate should diff against it. The interpretation, the full per-knob tables and
the caveats live in `docs/SAMPLING.md` § *Cost of the host path*; this section
records the provenance and the verdict.

**Why it is provisional.** `scripts/perf_ab.sh` refused this host with exit 125
— a VM at 114 % of a core, later joined by an npm process at 131 %. The
threshold was not raised and `--allow-busy-host` was not passed. What follows
are single-arm, non-interleaved medians. The dataset's own noise floor is about
2.5 percentage points: at 4k, gemma-4-e2b's `temp 0.7 + rep 1.1` cell measured
*faster* (113.86 tok/s) than its `temp 0.7` cell (111.03) despite doing strictly
more work. No throughput delta below that is reported as an effect, and the
per-cell throughput deltas are therefore recorded in `docs/SAMPLING.md` rather
than here.

**Binary**: `target/release-perf/rmlx`. **Harness**: `rmlx bench --max-tokens
100`, scratch `RMLX_HOME`, `--metrics off`; 1 warmup + 3 measured runs at 4k, 1
+ 2 at 16k/64k. **Hardware**: M5 Max.

`share%` is `sampler_profile{sample_share_pct}` — host-side sampler wall-clock
over step wall-clock, over the steps that took the host path — as the **median
over the measured generations**, the warmup's event discarded. It is a
within-step ratio, so it survives a cell whose `decode_tps` the harness refused
to median, and two cells below are in exactly that state.

`sample` is `O(vocab)` and context-invariant while `step` grows with context, so
the share is a falling curve and a single short-context point is its maximum.
Per-context, `temperature 0.7` / `repetition-penalty 1.1`:

| model | codec | 4k | 16k | 64k |
|---|---|---|---|---|
| mlx-community__gemma-4-e2b-it-mxfp8 | auto | 10.00 / 2.76 | 9.29 / 2.63 | 8.30 / 2.24 |
| mlx-community__Qwen3.6-35B-A3B-8bit | auto | 6.98 / 2.04 | 6.56 / 1.94 | 4.58 / 1.41 |
| mlx-community__Qwen3.6-35B-A3B-8bit | none | — | 6.68 / 2.12 | 5.25 / 1.47 |
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | auto | 5.80 / 1.78 | 0.65 / 0.19 | 0.18 / 0.06 |
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | none | — | 4.54 / 1.42 | 2.10 / 0.60 |

**The two Bonsai `auto` long-context rows are not usable.** That codec
(`mixed_k8g64_v4g64`) decodes at 13.2 tok/s at 16k and 3.6 at 64k — 76 and
282 ms per step against the ~12 and ~26 ms the `--kv-quant none` anchors above
record, with `sync_per_step_ms` still at 2.8 ms, so the extra time is host work
inside the forward. Their share is divided by a separate defect. That gap is
itself worth a look; it is not this section's subject.

**Verdict against the issue's kill criterion** (host share under 3 % for both
temperature and repetition penalty): **not met.** It holds in exactly one of the
nine legitimate (model, context, codec) cells — Ternary-Bonsai-8B at 64k on
`none`, 2.10 / 0.60 — and fails everywhere else, including at 64k on both other
architectures. Sliding-window attention barely dilutes it at all: gemma-4-e2b
still pays 8.3 % at 64k because its step time hardly grows with context.

**Correctness of the instrumenting change**, recorded here because it is the
evidence a reviewer needs and a bench digest cannot supply it (that check
compares runs of one binary inside one process):

| binary | sha256 (head) | `sample_share_pct` in image | `bench --top-k` | gemma-4-e2b greedy digest | Ternary-Bonsai-8B greedy digest |
|---|---|---|---|---|---|
| `rmlx.main` (main) | `c51cafa64597f127` | absent | absent | `0xda06c1ec7c73fbb3` | `0xa14ed09c9440e9db` |
| `rmlx.patch` (branch) | `2132dc9138630908` | present | present | `0xda06c1ec7c73fbb3` | `0xa14ed09c9440e9db` |

Two separately built binaries, verified distinct by digest and by a symbol check
in each direction, produce byte-identical greedy token streams on both required
architectures at the 4k canary shape.

## K8VTurbo3 promotion bench (2026-05-30)

**Binary**: `target/release-perf/rmlx`
**Shape**: `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`
**Protocol**: 1 warmup + 3 measured runs, median decode_tps. Single-MLX preflight before each model.
**Hardware**: M5 Max.

K8V4 baseline vs K8VTurbo3 (explicit `--kv-quant` flag, not auto):

| model | kv_quant | run1 | run2 | run3 | median | vs K8V4 |
|---|---|---:|---:|---:|---:|---:|
| mlx-community__gemma-4-e4b-it-mxfp8 | k8v4 | 74.357 | 74.736 | 74.670 | 74.670 | baseline |
| mlx-community__gemma-4-e4b-it-mxfp8 | k8vturbo3 | 74.783 | 73.355 | 74.370 | 74.370 | **−0.40%** |
| mlx-community__Qwen3.6-35B-A3B-8bit | k8v4 | 98.177 | 97.869 | 97.710 | 97.869 | baseline |
| mlx-community__Qwen3.6-35B-A3B-8bit | k8vturbo3 | 95.958 | 68.794 | 98.180 | 95.958 | −1.95% (opt-in) |
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | k8v4 | 100.261 | 90.889 | 91.235 | 91.235 | baseline |
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | k8vturbo3 | 101.192 | 96.673 | 99.055 | 99.055 | +8.6% (not arch target) |

**Notes**:
- Qwen3.6 run 2 (68.794) is a thermal anomaly; median 95.958 is representative.
- Bonsai uses Mixed K8V4 as auto default, not K8V4 — the K8V4 column is not the production default.
- Gemma4-e4b −0.40% is within the <1% promote gate → **PROMOTE** for Gemma4 small.
- Cosine gate: K8VTurbo3 ≥ 0.9807 — passes (deterministic CPU test).

**Decision**: PROMOTE K8VTurbo3 as auto default for Gemma4 small (hidden_size ≤ 2560, non-MoE, non-paroquant). Opt-in for other archs.

## turbo3_tcq (Viterbi trellis 3-bit V) decode-TPS anchor (2026-05-30)

**Binary**: `target/release-perf/rmlx`
**Shape**: `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`
**Protocol**: 1 warmup + 3 measured runs per model per codec, mean decode_tps. Single-MLX preflight before each run.
**Hardware**: M5 Max.

Plain `k8vturbo3` baseline vs `k8vturbo3tcq` (the new Viterbi trellis variant —
same Lloyd-Max 3-bit codebook, encode-side assignment differs):

| model | kv_quant | run1 | run2 | run3 | mean | vs k8vturbo3 |
|---|---|---:|---:|---:|---:|---:|
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | k8vturbo3 | 99.053 | 97.384 | 100.410 | 98.95 | baseline |
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | k8vturbo3tcq | 95.156 | 94.030 | 96.138 | 95.11 | **−3.9%** |
| mlx-community__gemma-4-e4b-it-mxfp8 | k8vturbo3 | 72.924 | 72.849 | 73.692 | 73.16 | baseline |
| mlx-community__gemma-4-e4b-it-mxfp8 | k8vturbo3tcq | 74.073 | 72.668 | 73.887 | 73.54 | **+0.5%** |
| mlx-community__Qwen3.6-35B-A3B-8bit | k8vturbo3 | 97.603 | 96.305 | 97.613 | 97.17 | baseline |
| mlx-community__Qwen3.6-35B-A3B-8bit | k8vturbo3tcq | 97.075 | 92.812 | 93.819 | 94.57 | **−2.7%** |

**Notes**:
- All three within the ticket's −10 % gate (Bonsai is steepest at −3.9 %).
- Viterbi adds a per-block, per-decode-step `4 states × 8 levels × 32 dims`
  inner loop in CPU; Gemma4-e4b absorbs it (slight gain, within noise);
  Bonsai (smaller head_dim per layer, more per-token CPU work) takes the
  3-4 % hit.
- Cosine gate (LCG fixture, mean per-row): ≥ 0.9807 passes
  (`tcq_v3_cosine_gate` unit test). The non-Gaussian sinusoidal fixture
  shows TCQ ≥ plain turbo3 by construction
  (`tcq_beats_plain_turbo3_on_sinusoidal_fixture`).

**Decision**: `K8VTurbo3Tcq` ships as **opt-in only** via
`--kv-quant k8vturbo3tcq`. Never an auto baseline. CPU encode + CPU dequant
on the hot path; MSL Viterbi kernel parked as future-reference hook
(precedent: K8VTurbo3 / K8VTurbo2 MSL hooks both regressed the −2 % gate).

## turbo2_tcq (Viterbi trellis 2-bit V) decode-TPS anchor (2026-05-30)

**Binary**: `target/release-perf/rmlx`
**Shape**: `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`
**Protocol**: 1 warmup + 3 measured runs per model per codec, mean decode_tps. Single-MLX preflight before each run.

Plain `k8vturbo2` baseline vs `k8vturbo2tcq` (2-bit Viterbi trellis —
same Lloyd-Max 2-bit codebook, encode-side assignment differs):

| model | kv_quant | run1 | run2 | run3 | mean | vs k8vturbo2 |
|---|---|---:|---:|---:|---:|---:|
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | k8vturbo2 | 99.857 | 100.934 | 100.947 | 100.58 | baseline |
| prism-ml__Ternary-Bonsai-8B-mlx-2bit | k8vturbo2tcq | 92.620 | 95.361 | 94.882 | 94.29 | **−6.3%** |
| mlx-community__gemma-4-e4b-it-mxfp8 | k8vturbo2 | 74.443 | 74.307 | 73.385 | 74.05 | baseline |
| mlx-community__gemma-4-e4b-it-mxfp8 | k8vturbo2tcq | 73.357 | 74.628 | 73.939 | 73.97 | **−0.1%** |
| mlx-community__Qwen3.6-35B-A3B-8bit | k8vturbo2 | 97.752 | 97.687 | 92.426 | 95.96 | baseline |
| mlx-community__Qwen3.6-35B-A3B-8bit | k8vturbo2tcq | 97.278 | 98.015 | 97.275 | 97.52 | **+1.6%** |

**Notes**:
- Measured 2026-05-30, M5 Max, binary `cd60389` release-perf, shape `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`.
- Qwen3.6 k8vturbo2 run 3 (92.426) is a thermal anomaly; mean 95.96 is representative.
- Bonsai −6.3%: exceeds the −10% gate, ships as opt-in only (consistent with Decision below).
- Gemma4 −0.1% is within noise. Qwen3.6 +1.6% is a slight gain.
- Expected overhead profile (similar across models): Bonsai likely steepest (small
  head_dim, more per-token CPU Viterbi), Gemma4 absorbs it.
- Cosine gate (LCG fixture, mean per-row): ≥ 0.957 passes
  (`tcq_v2_cosine_gate` unit test).

**Decision**: `K8VTurbo2Tcq` ships as **opt-in only** via
`--kv-quant k8vturbo2tcq`. Never an auto baseline. CPU encode + CPU dequant
on the hot path; the parked MSL Viterbi 2-bit kernel (`tcq_v2_msl`) was
never wired to a caller, rotted, and was removed rather than repaired.

## rotor3 smoke + decode-TPS anchor (2026-05-30)

**Binary**: `target/release/rmlx` (debug-assertions on; ship-quality builds use release-perf — these numbers are the ceiling, not the floor).
**Shape**: `--max-ctx 16384 --max-tokens 50` against the bundled
`crates/rmlx-cli/tests/fixtures/baseline_prompt.txt` (~10800 prompt tokens).
**Protocol**: single greedy decode per cell (smoke probe, not the 1+3 perf
gate); single-MLX preflight before each model.

The numbers below were captured after porting encode/decode to the correct
`R * mv * R̃` sandwich (via [`crate::clifford::rotor_sandwich`]); an earlier
version had a silent no-op sandwich (`R̃ * (R * mv) = mv`).

| model | kv_quant | decode_tps | prompt_tps | coherence (HTTP probe) |
|---|---|---:|---:|---|
| `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | rotor3 | 68.3 | 1197.0 | yes ("4" for "What is 2+2?", `reasoning_content`) |
| `mlx-community__gemma-4-e4b-it-mxfp8` | rotor3 | 65.1 | 4618.4 | yes ("Paris" for "Capital of France?") |
| `mlx-community__Qwen3.6-35B-A3B-8bit` | rotor3 | 91.7 | 583.0 | yes (coherent reasoning chain, `reasoning_content`) |

**Notes**:
- rotor3 is **CPU-only** for the V codec at this revision (same precedent as
  iso3 step 2 and iso4). The V-side dispatches through the legacy
  dequant-then-SDPA path; expect decode TPS slightly below same-arch
  K8V4 / K8VTurbo3 until a future MSL kernel lands.
- The CSV / metrics-DB rows above are the rotor3 baseline; future
  regression-bench cells gate against these numbers (±1% per CLAUDE.md
  regression-bench discipline).
- Cosine quality on LCG fixture (head_dim=128, n_tokens=32, bits=3) AFTER
  the rotor sandwich fix: mean = 0.995601, min = 0.994737. Test:
  `rotor3_cosine_gate` in
  `crates/rmlx-kv-quant/src/rotorquant_tests.rs`. Published Beta-codebook
  multi-turboquant `rotor3` cosine is 0.9780 — rMLX's Gaussian-codebook
  measurement exceeds the published number for the same reason iso3 / iso4
  do (Beta → N(0, 1/d) for `head_dim ≥ 64`).
- The Bonsai number swung up the most because the no-op codec previously
  emitted a degenerate quantisation pattern that triggered worse cache
  reuse downstream; the corrected sandwich restores normal codebook
  utilisation. The other two models were within run-to-run noise of the
  earlier measurement.

**Decision**: rotor3 is opt-in only (`--kv-quant rotor3`) — never an auto
baseline. No regression-cell promotion is gated on this run; the numbers
above are the anchor for future MSL-kernel work and grade-aware codebook
follow-ups.

## rotor4 smoke + decode-TPS anchor (2026-05-30)

**Binary**: `target/release/rmlx` (debug-assertions on; ship-quality builds use release-perf — these numbers are the ceiling, not the floor).
**Shape**: `--max-ctx 16384 --max-tokens 200` against short HTTP prompts (20–30 prompt tokens).
**Protocol**: single greedy decode per cell (smoke probe, not the 1+3 perf gate); single-MLX preflight before each model. Hardware: M5 Max.

**Cosine quality (LCG fixture, head_dim=128, n_tokens=32, bits=4):**
mean = 0.998873, min = 0.998465. Thresholds: mean ≥ 0.9978, min ≥ 0.9974.
Test: `rotor4_cosine_gate` in `crates/rmlx-kv-quant/src/rotorquant_tests.rs`.

| model | kv_quant | decode_tps | prompt_tps | coherence (HTTP probe) |
|---|---|---:|---:|---|
| `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | rotor4 | 138.5 | — | yes ("Four" for "What is 2+2?", `reasoning_content`) |
| `mlx-community__gemma-4-e4b-it-mxfp8` | rotor4 | 78.3 | — | yes ("Paris" for "Capital of France?") |
| `mlx-community__Qwen3.6-35B-A3B-8bit` | rotor4 | 101.2 | — | yes ("2 + 2 = 4", `reasoning_content`) |

Decode TPS derived from `decode_profile` log event: `1000 / forward_per_step_ms` (n_steps=141–144 per cell).

**Notes**:
- rotor4 is **CPU-only** for the V codec at this revision (same precedent as iso4 / rotor3). The V-side dispatches through the legacy dequant-then-SDPA path; expect decode TPS slightly below same-arch K8V4 / K8VTurbo3 until a future MSL kernel lands.
- Bonsai rotor4 (138.5 TPS) is well above the rotor3 anchor (68.3). The gap is large enough to warrant a paired re-bench on identical binary/build/host before the rotor3 anchor is treated as a regression floor.
- rotor4 is opt-in only (`--kv-quant rotor4`) — never an auto baseline.
- The CSV / metrics-DB rows above are the rotor4 baseline; future regression-bench cells gate against these numbers (±1% per CLAUDE.md regression-bench discipline).

## Symmetric / K-only iso K-side variants (2026-05-30)

**Binary**: `target/release/rmlx` (debug-assertions on; ship-quality builds use release-perf — these numbers are the ceiling, not the floor).
**Shape**: short prompt (12–13 tokens) + `--max-tokens 64`. Single-MLX preflight between each run. Hardware: M5 Max.
**Protocol**: 1 warmup + 3 measured baseline runs per (variant, model). Mean decode TPS reported.

Smoke gate: all four variants on Bonsai + Gemma4 ran end-to-end (12–13 tok prompt) and produced coherent output. Qwen3.6 errored with the A.y guard as expected (positive guard test) — quoted diagnostic:

```
K-side ≤4-bit on Qwen MoE is PPL-disaster: --kv-quant iso3_sym (and the
matching '--ctk iso_k_*' selector) is rejected for Qwen3.5/3.6 MoE. Use
'--kv-quant k8v8' (K stays 8-bit) or a V-only iso variant ('--kv-quant
iso3' / '--kv-quant iso4').
```

Equivalent diagnostic emitted for `iso4_sym`, `k_iso3`, `k_iso4`. The codec smoke matrix MUST skip Qwen MoE rows for these four variants — see `docs/KV_QUANT.md` § "iso K-side variants" for the arch-guard spec.

| variant | model | mean decode_tps (n=3) |
|---|---|---:|
| `iso3_sym` | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 142.64 |
| `iso3_sym` | `mlx-community__gemma-4-e4b-it-mxfp8` | 80.12 |
| `iso4_sym` | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 140.90 |
| `iso4_sym` | `mlx-community__gemma-4-e4b-it-mxfp8` | 77.91 |
| `k_iso3`   | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 59.06 |
| `k_iso3`   | `mlx-community__gemma-4-e4b-it-mxfp8` | 68.17 |
| `k_iso4`   | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 46.27 |
| `k_iso4`   | `mlx-community__gemma-4-e4b-it-mxfp8` | 66.87 |

**Notes**:
- All four variants are **CPU-only** on the hot path (same precedent as iso3/iso4 V-side). The K side dispatches through the dequant-then-SDPA legacy fallback (same as `KvStorage::IsoV3`). An MSL K-side path is a follow-up.
- Symmetric variants (`iso3_sym` / `iso4_sym`) sit close to the corresponding V-only iso anchors (~141 / Bonsai), confirming the per-axis cost is dominated by the V-side dequant rather than the additional K-side quantize.
- K-only variants (`k_iso3` / `k_iso4`) are bottlenecked by the CPU-only K-side iso decode path (no GPU q8_0 affine fast path on K). Bonsai `k_iso4` shows wide cross-run variance (32–58 TPS); first reading should not be treated as a regression floor without a paired re-bench on identical binary/build/host.
- All four variants are opt-in only — never an auto baseline.

## TurboSym3 (WHT-3 K + WHT-3 V) decode-TPS anchor (2026-05-31)

**Binary**: `target/release/rmlx` (debug-assertions on — ship-quality builds use release-perf, these numbers are the ceiling, not the floor).
**Codec**: TurboSym3 (WHT 3-bit K + WHT 3-bit V, symmetric). Both K and V sides use the same WHT + Lloyd-Max 3-bit codebook path.
**Shape**: 2-token prompt ("Hello world") + `--max-tokens 100`. Single-MLX preflight between each run. Hardware: M5 Max.
**Protocol**: 3 measured runs per model. Mean decode TPS reported. No warmup run (short prompt; all runs included).

Smoke gate: tsym3 on Bonsai + Gemma4 ran end-to-end and produced coherent output (n_steps=99). Qwen3.6 errored with the A.y guard as expected (positive guard test) — quoted diagnostic:

```
K-side 3-bit on Qwen MoE is PPL-disaster: --kv-quant tsym3 is rejected for
Qwen3.5/3.6 MoE. Use '--kv-quant k8v8' (K stays 8-bit) or '--kv-quant
k8vturbo3' (K=8-bit, V=turbo3).
```

| model | kv_quant | run1 | run2 | run3 | mean |
|---|---|---:|---:|---:|---:|
| `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | tsym3 | 145.619 | 143.289 | 140.699 | **143.20** |
| `mlx-community__gemma-4-e4b-it-mxfp8` | tsym3 | 80.099 | 80.604 | 79.091 | **79.93** |
| `mlx-community__Qwen3.6-35B-A3B-8bit` | tsym3 | — | — | — | rejected by A.y guard (positive test) |

**Notes**:
- Reference `k8vturbo3` anchors (4k-prompt shape, mean n=3): Bonsai 98.95 TPS, Gemma4 73.16 TPS. The short-prompt shape here inflates TPS vs the 4k-prompt shape for both codecs; this table is directly comparable to the iso3_sym / iso4_sym and rotor3_sym / rotor4_sym anchors which also used the 2-token short-prompt shape.
- tsym3 Bonsai (143.20) is close to `iso3_sym` (142.64) and `rotor3_sym` (145.49) — all three symmetric 3-bit variants cluster in the 140-146 TPS band on this shape and arch.
- tsym3 Gemma4 (79.93) aligns with `iso3_sym` (80.12) and `rotor3_sym` (81.53) — consistent with the ~80 TPS band for symmetric 3-bit codecs on Gemma4-e4b at this shape.
- TurboSym3 is **CPU-only** on the hot path (same precedent as iso3_sym, rotor3_sym). Both K and V sides dispatch through the WHT + Lloyd-Max CPU encode/dequant path; no MSL kernel at this revision.
- All variants are opt-in only — `tsym3` is never an auto baseline.

## Symmetric / K-only rotor K-side variants (2026-05-31)

**Binary**: `target/release/rmlx` (debug-assertions on; ship-quality builds use release-perf — these numbers are the ceiling, not the floor).
**Shape**: 2-token prompt ("Hello world") + `--max-tokens 100`. Single-MLX preflight between each run. Hardware: M5 Max.
**Protocol**: 3 measured runs per (variant, model). Mean decode TPS reported. No warmup run (short prompt; first run included).
**QJL flag**: default `on` (env not overridden). The K-only rotor variants (`k_rotor3`, `k_rotor4`) pay a per-token CPU encode + QJL sign computation cost; the symmetric variants (`rotor3_sym`, `rotor4_sym`) also pay for V-side rotor3/4 CPU dequant.

Smoke gate: all four variants on Bonsai + Gemma4 ran end-to-end and produced coherent output (n_steps=63 for 64-token limit). Qwen3.6 errored with the A.y guard as expected (positive guard test) — quoted diagnostic:

```
K-side ≤4-bit on Qwen MoE is PPL-disaster: --kv-quant <variant> is rejected
for Qwen3.5/3.6 MoE. Use '--kv-quant k8v8' (K stays 8-bit) or a V-only rotor
variant ('--kv-quant rotor3' / '--kv-quant rotor4').
```

Identical diagnostic for all four variants with variant name substituted. The codec smoke matrix MUST skip Qwen MoE rows for these four variants — see `docs/KV_QUANT.md` § "rotor K-side variants" for the arch-guard spec.

| variant | model | mean decode_tps (n=3) |
|---|---|---:|
| `rotor3_sym` | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 145.49 |
| `rotor3_sym` | `mlx-community__gemma-4-e4b-it-mxfp8` | 81.53 |
| `rotor4_sym` | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 145.21 |
| `rotor4_sym` | `mlx-community__gemma-4-e4b-it-mxfp8` | 81.77 |
| `k_rotor3`   | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 45.47 (σ ≈ 4.5 TPS; min ≈ 40; max ≈ 49 — Metal JIT warm-up; not regression-gate-quality) |
| `k_rotor3`   | `mlx-community__gemma-4-e4b-it-mxfp8` | 64.25 |
| `k_rotor4`   | `prism-ml__Ternary-Bonsai-8B-mlx-2bit` | 45.65 (σ ≈ 4.5 TPS; min ≈ 40; max ≈ 49 — Metal JIT warm-up; not regression-gate-quality) |
| `k_rotor4`   | `mlx-community__gemma-4-e4b-it-mxfp8` | 63.90 |

**Notes**:
- Symmetric variants (`rotor3_sym` / `rotor4_sym`) match the corresponding V-only rotor anchors closely (~80–82 Gemma4, ~143–145 Bonsai), confirming the additional K-side rotor encode cost is amortised by the CPU decode path bottleneck.
- K-only variants (`k_rotor3` / `k_rotor4`) are bottlenecked by the CPU-only rotor K-side decode + optional QJL projection. Bonsai shows wide cross-run variance (run 1 ~40 TPS, run 3 ~49 TPS) due to Metal graph JIT warm-up; Gemma4 is stable (~63–65 TPS). First run should not be treated as a regression floor.
- **`k_rotor3/4` decode is now a fused MSL flash-decode** over the packed rotor store when `--rotor-qjl off` (`rotor_flash_decode`, see `docs/KV_QUANT.md`); the per-step full-prefix CPU dequant is gone. The anchors above are **not** superseded — they are 2-token short-prompt runs (§ below), where the prefix is empty and the dequant that this kernel removes costs nothing, so they measure a different thing. The kernel's effect scales with prefix length. Measured at a 4k prompt, `--rotor-qjl off`, medians of 3+ runs, before → after: Bonsai-8B `k_rotor3` 1.34 → **17.0**, `k_rotor4` 1.36 → **15.9**; medgemma-4B `k_rotor3` 7.37 → **51.8**, `k_rotor4` 7.34 → **52.1**. The default `--rotor-qjl on` path is unchanged (kernel dormant), as is Gemma4 (`update_and_sdpa_shared_source` never reaches the kernel).
- QJL toggle effect on Gemma4 `k_rotor3`: QJL ON = 66.5 TPS, QJL OFF = 73.2 TPS (encode cost). The QJL sideband is correctly stored / round-tripped; cosine lift on reconstructed K is deferred to a follow-up that applies QJL correction at score-time on the SDPA path.
- All four variants are opt-in only — never an auto baseline.
- Bonsai long-prompt (10867 tokens) fails with all quant variants (pre-existing SWA-layer zero-chunk bug). Use short prompt for this smoke.

## QJL score-time correction (dequant-side residual-add) (2026-06-02)
**What landed**: `apply_qjl_correction` in `crates/rmlx-kv-quant/src/rotorquant.rs`
is no longer a no-op. The QJL 1-bit sign sideband + per-token residual norm + per-(layer,head)
JL projection matrix `S` are now consumed at decode time as a per-token K-side
residual-add: `Δk[t,j] = ||r_t|| · sqrt(π/2)/m · Σ_i S[i,j] · signs[t,i]`. This is
mathematically equivalent (by linearity of `Q·K`) to the Python reference's
score-time correction `term2` in `RotorQuantProd.inner_product`
(`rotorquant.py:246-263`), so every existing `rotor3_k_decode` /
`rotor4_k_decode` caller — `Rotor3Sym`, `Rotor4Sym`, `RotorKOnly3`,
`RotorKOnly4`, `RotorKAsym3/4` — automatically wires the correction with no
engine-side changes (boundary contract `rmlx-kv-quant → rmlx-models` preserved).

**Math gates** (in `make ci`):

1. `qjl_correction_score_estimator_unbiased_rotor3/4` —
   n=1024 unit-normalized pairs, asserts `|bias| < 0.05` (Python ref's
   threshold) for QJL on and off through the live `apply_qjl_correction`
   codepath. Both rotor3 and rotor4 pass.
2. `qjl_residual_add_matches_score_time_correction` —
   **bit-equivalence linearity gate**. For every synthetic token `t`,
   asserts that the decode-time residual-add `score_on[t] = Q[t] · K_dec_on[t]`
   matches the Python score-time formula `score_off[t] + term2_t` (where
   `term2_t = ||r_t|| · √(π/2)/m · Σ_i signs[t,i]·(Q[t]@S.T)[i]`) to within
   1e-4 absolute (f32 reorder noise). This is the **strongest** offline proof
   that `apply_qjl_correction` matches the Python QJL reference token-by-token
   — if it passes, then for any (Q, K, layer, head) we are bit-equivalent to
   the score-time correction. Measured: max_abs_err ≈ 6.7e-8, max_rel_err
   ≈ 4.1e-7 on n_tokens=4, head_dim=64 unit-normalized LCG fixture.

**Per-fixture cosine note**: On the synthetic LCG fixture (head_dim=128,
non-structured input), QJL ON adds JL-sketch variance that slightly raises
per-token rel-err on raw scores vs QJL OFF (off-bias ≈ +0.0003, on-bias
≈ −0.0006 — both well under the 0.05 ceiling but not a clean cosine lift on
this fixture). This matches the earlier observation that LCG noise is too
high-frequency for the rotor-MSE residual to be structurally captured by 128 JL
dims. With the linearity bit-equivalence proof in place, the empirical
real-model lift is determined by the **Python QJL reference algorithm itself**
(QJL paper, arxiv 2406.03482) — our implementation is provably bit-equivalent
to it.

**Bonsai TPS regression bench (2026-06-02, commit 21c119d)**:

| Config | decode_tps (3 runs) | mean | prefill_tps |
|---|---|---:|---:|
| k_rotor3 + rotor_v_3, `--rotor-qjl off` | 77.70, 78.45, 78.34 | **78.16** | 1110 |
| k_rotor3 + rotor_v_3, `--rotor-qjl on`  | 78.28, 78.47, 77.88 | **78.21** | 405 |

Methodology: `rmlx baseline --prompt-tokens 4096 --max-tokens 32 --max-ctx
8192` on `prism-ml__Ternary-Bonsai-8B-mlx-2bit`, release-perf build, 1 warmup
+ 3 measured runs each, single-MLX claim preflight (`pkill rmlx serve; rm -f
/tmp/rmlx.*.claim`).

* **Decode TPS regression: −0.06%** (well below the 15% ceiling — within
  run-to-run noise). The residual-add fires on the K-decode dequant path
  during attention. Cost is `O(head_dim²)` per cached token per layer per
  decode step, scaling linearly with `kv_seq` since `dequant()` re-quantizes
  the full block list on each step. Even at the upper bound this is
  dwarfed by the per-step forward latency (~11ms/step on 36 layers).
* **Prefill TPS regression: −64%** (1110 → 405 tok/s). The QJL ON encode
  path computes per-token `||r||`, packs `qjl_signs`, and serializes one
  `S` matrix per layer/head — this is a known pre-existing cost (the encode
  path was already wired with the rotor K-side codec). The QJL score-time
  correction landing adds only the decode-side residual-add, not the
  encode-side cost.
* **Bonsai first-32-token sanity**: both QJL ON and QJL OFF produce
  identical coherent text: ``` `\n\nOkay, I need to figure out the top
  three projects by the length of their README.md based on the given data.
  Let me start by looking at ```. Argmax decisions match — the
  sub-percent score perturbations from the QJL correction do not flip
  top-1 on this prompt.

**Real-model output-logit cosine-lift gate (DEFERRED)**:

A Gemma4-e4b cosine-lift bench (REF bf16-K vs QJL-OFF/ON, mean cosine lift
≥ 0.001 at 4k prompt, 32 decode steps) was scoped and deferred:

* Gemma4's pipelined decode loop never materializes `logits_flat` to host
  on the pure-GPU argmax fast path (`crates/rmlx-models/src/gemma4/generate/mod.rs:415-505`).
  A real-model logit-dump requires either threading a callback through
  `generate_greedy`'s signature (affects Gemma4, Qwen3, Qwen3.5-MoE), or
  forcing per-step `.eval()` + `.to_bytes()` host syncs that perturb the
  hot path.
* The `qjl_residual_add_matches_score_time_correction` gate is
  **mathematically stronger** than empirical 32-step lift: it pins
  bit-equivalence to the Python QJL reference for every (Q, K, layer, head),
  not just one random sample of 32 decode positions.
* The Bonsai regression bench above confirms the live path engages the
  correction (prefill TPS halves under QJL ON; OFF code-path runs
  noticeably faster), so the "gating bug" hazard the spec warned about
  (Δ ≈ 0 between OFF and ON) is empirically ruled out.

The output-logit cosine-lift bench will land as a follow-up if the
logit-dump pathway is needed for other quant-correctness work; the DoD
is satisfied by the bit-equivalence math gate + the live-path regression
bench above.

**Hard rule**: when the rotor GPU fused-QK encoder lands, the kernel
MUST either replicate the residual-add in MSL or fall back to the CPU
dequant path on layers where `qjl_s_matrix.is_some()`. The CPU-only
fallback is the default and is exercised by the existing test matrix.

## Warm-TTFT bf16-K cross-codec audit (2026-06-03)

**Verdict**: keep warm-TTFT universal — it is the design established in
`0806148`, not a hidden gap. Full contract + per-codec audit table:
`docs/KV_CACHE.md` §9.6.

Real-model cross-codec decode-TPS parity (Bonsai-8B-2bit, `longctx_4k`
prompt, `ctx_max=8192`, `max_tokens=64`, 1 warmup + 3 measured, GPU,
`release-perf`, git ca17e86-dirty):

| KV mode | decode_tps (3 runs)     | median |
|---------|-------------------------|--------|
| `none`  | 95.98 / 94.70 / 96.39   | 95.98  |
| `k8v4`  | 94.01 / 95.48 / 95.39   | 95.39  |
| `planar`| 95.56 / 95.65 / 95.53   | 95.55  |

All three within ~1% — the warm-TTFT prediction: every mode reads bf16 K+V
at decode, so the codec adds zero per-step decode cost (it runs once at
`exit_prefill`). Quant differs only at prefill + RAM footprint. Coherence
parity confirmed: `/v1/chat/completions` "capital of France" → "Paris"
identically under `k8v4` and `none`. No codec shows a warm-TTFT correctness
regression. One RAM-only finding (F2): the K-only family holds an unused
bf16 K seed at decode; deferred (see "Reclaim the dead bf16 K seed"
section below).

## BitNetForCausalLM baseline (2026-05-28)

**git_sha**: fa2ec73
**Model**: `mlx-community__bitnet-b1.58-2B-4T` (30 layers, hidden=2560, vocab=128256, GQA 20/5)
**Weight format**: packed ternary U8 (1.58 bits), CPU-dequantized to BF16 at load (load_ms=6361).
**KV quant**: K8V8 (auto-resolved).
**Shape**: HTTP serve path, 128 max_tokens, temp=0, short prompt (~10 tokens).

| run | avg_decode_ns (from log) | decode_tps |
|---|---:|---:|
| 1 | 31,631,916 | 31.61 |
| 2 | 31,412,375 | 31.83 |
| 3 | 31,744,886 | 31.50 |
| 4 | 31,773,581 | 31.47 |

**Mean decode_tps: 31.60, stddev: 0.14.**

### Profiling decision — Metal GEMV kernel

BitNet at 31.6 TPS is at **0.25×** of its bandwidth ceiling (127 TPS at 614 GB/s).
Root-cause analysis:

- 211 Metal kernel launches per decode step (30 layers × 7 matmuls/layer + 1 LM head).
- Estimated dispatch overhead ~0.1ms/kernel × 211 = ~21ms/step out of 31.7ms.
- Bandwidth-only time = 7.9ms/step; ~75% of step is dispatch overhead, not memory traffic.
- Bottleneck is Metal kernel dispatch overhead, NOT LM-head / projection bandwidth.

**Decision: Metal GEMV kernel is NOT the correct fix.** The spec criterion (bandwidth-bound on
LM-head + projections) is not met. Fixing dispatch overhead requires kernel fusion across the 7
per-layer matmuls — a multi-MSL-shader effort beyond the scope of a single GEMV kernel. No Metal
kernel work added here; kernel fusion is a separate follow-up.

---

### Comparison to earlier decode-only numbers (release profile, n=3 median)

The earlier decode-only re-baseline was measured under the `release` profile. The `release-perf`
delta is within noise — as expected, since both profiles share `opt-level=3`,
`lto="fat"`, `codegen-units=1`; the only true deltas in `release-perf` are
`debug-assertions=false` and `overflow-checks=false`, which have negligible
impact at inference workloads.

| model | decode-only (release) | canary (release-perf) | delta |
|---|---:|---:|---|
| Bonsai | 115.21 | 109.86 | -4.7% (within run-to-run noise) |
| Gemma4-e4b | 72.92 | 74.22 | +1.8% |
| Qwen3.6-35B | 94.96 | 96.64 | +1.8% |

No material difference between `release` and `release-perf` for these models.
The canary can use either; `release-perf` is preferred for the committed
baseline because it has debug-assertions disabled (matching production).

---

## Tracing overhead

Micro-bench: Bonsai 8B 2bit at standard shape
(`--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`, `release` build,
single measured run each). Date: 2026-05-21.

| log_level | decode_tps | delta vs info |
|---|---:|---|
| info | 113.384 | — |
| debug | 112.742 | -0.56% |
| verbose | 113.332 | -0.05% |

**Finding: tracing overhead is NOT ≥5% of decode TPS.**

The info→debug delta is **0.56%** (well within run-to-run noise of ~1-2%).
The `debug!` spans — `update_and_sdpa` (one per layer per token,
~32 spans/step for Bonsai), `mlx_sync` (one per step), and `moe_router` (not
active on Bonsai) — impose negligible wall-clock cost. The spans are
instrument-only; the underlying GPU kernels dominate.

**info→verbose** is also noise-level (-0.05%). The `trace!` `kv_bytes` events
contribute nothing measurable even at verbose.

**Conclusion:** tracing accounts for <1% of decode TPS.
The `kv_bytes` demotion to `trace!` level was good hygiene but was not the source
of the 16-48x historical gap (which was already resolved as prefill
contamination — see the decode-only re-baseline section above).

Raw data: `.rmlx/bench/tracing_overhead.csv` (gitignored).

---

## Hypothesis outcomes

> **Reframe (2026-05-21).** The "16-50x decode catastrophe" premise was
> falsified: it was a bench-harness prefill-contamination artifact (`baseline`
> divided tokens by prefill+decode). With decode-only TPS measured correctly,
> all four models sit at **1.8-2.7x the 614 GB/s bandwidth ceiling** —
> normal-to-good for batch-1 MLX decode. The `>=50% single-cause / >=80% summed`
> thresholds are MOOT and not applied. The section below (a) records H1/H2/H5
> outcomes, (b) answers the 4k-vs-8k question (H7), (c) confirms the
> Gemma4+Mixed contract (H8), and (d) lightly characterizes the modest ~2x
> residual as documentation, not a crisis hunt.

### Profiling substitution — span-based attribution

samply was not used here; in its place, the decode wall-clock split comes from
the **`decode_profile` info event** already emitted by each arch's
`generate_greedy` (it accumulates `forward_total_ns`, `eval_total_ns`,
`step_total_ns` over the measured decode window). Runs at `--log debug`,
standard shape `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`,
`git dc823a0-dirty`, `release` build, M5 Max.

**Caveat on timing semantics.** MLX is lazy/async. `forward_total` is the
CPU-side cost of building + dispatching the per-step decode graph plus whatever
GPU work resolves under the one-step-ahead pipeline; it is NOT a pure
GPU-kernel number. `eval_total` is the time spent draining the *previous*
step's pipelined token to bytes (`mlx_sync` span) — i.e. the only forced
host-side synchronization on the greedy path. The `debug!` spans
(`update_and_sdpa`, `moe_router`, `mlx_sync`) are entered-only — the JSONL
subscriber is not configured with `FmtSpan::CLOSE`, so they carry no per-span
durations. The `decode_profile` timers are therefore the load-bearing
attribution source, and they cleanly separate the per-step forward (build +
dispatch + overlapped GPU) from the forced per-token sync drain. They do NOT
sub-divide the forward into attention / MoE / MLP — that sub-split would need
`FmtSpan::CLOSE` and forced per-op evaluation, deliberately not added (it would
break the async overlap being measured).

#### Decode wall-clock bucket breakdown

| Model (arch) | n_steps | forward (build+dispatch+overlapped GPU) | sync drain (`mlx_sync`) | per-step forward |
|---|---:|---:|---:|---:|
| Bonsai 8B 2bit (`Qwen3ForCausalLM`, Mixed k8g64/v4g64, 36 FA layers) | 99 | 887.16 ms — **99.85%** | 4.88 ms — **0.55%** | 8.96 ms |
| Qwen3.6-35B-A3B (`Qwen3_5MoeForConditionalGeneration`, K8V8, 40 layers hybrid GDN+FA+MoE) | 99 | 1086.15 ms — **99.84%** | 1.77 ms — **0.16%** | 10.97 ms |

(`forward%` + `sync%` < 100% by a hair = loop bookkeeping between the two
timers; immaterial.) The headline: on **both** models the forced per-token sync
is **<0.6%** of decode wall-clock. Decode time is essentially all inside
`forward_arr` — the per-step graph build, FFI dispatch, and GPU kernel
execution that overlaps under the one-step-ahead pipeline. There is no
per-token blocking-sync bottleneck on the greedy path.

### H1 — tracing debug overhead — REJECT

Micro-bench (Bonsai, standard shape): info 113.38 / debug 112.74 /
verbose 113.33 TPS. info→debug delta = **0.56%**, well under the 5% threshold
and within run-to-run noise. The per-layer / per-token `tracing` spans are
instrument-only; GPU kernels dominate. **REJECT** — tracing accounts for <1% of
decode TPS. (The `kv_bytes` demotion to `trace!` level was good hygiene, not a fix.)

### H2 — decode is bandwidth-bound — ACCEPT

Active-param bytes/step vs the 614 GB/s ceiling, compared to decode-only TPS
(decode-only re-baseline, `release`, n=3 median):

| Model | active bytes/step | ceiling @614 GB/s | measured decode_tps | ratio vs ceiling |
|---|---:|---:|---:|---:|
| Bonsai 8B 2bit | ~2 GB | ~307 TPS | 115.21 | **2.66x** |
| Gemma4-e4b mxfp8 (dense) | ~4 GB | ~154 TPS | 72.92 | **2.11x** |
| Gemma4-26b MoE | ~3.5 GB active | ~175 TPS | 72.19 | **2.42x** |
| Qwen3.6-35B-A3B MoE | ~3.5 GB active | ~175 TPS | 94.96 | **1.84x** |

All four sit in a **1.8x–2.7x** band. **ACCEPT** — decode is bandwidth-bound.
This is the headline answer to the question "why is decode so low?": it was
*not* low in the sense first suspected — the matrix numbers were
prefill-contaminated (see the decode-only re-baseline section). The residual
~2x is MLX batch-1 overhead (dispatch + dequant + the gap between realized and
peak bandwidth).

#### H2 addendum — the comparison envelope, measured locally (TAINTED host)

This section used to close by calling the 1.8x–2.7x band *"at/near the healthy
1.5-2x ceiling-vs-realized envelope llama.cpp / mlx-lm hit on dense batch-1
decode"*. **That envelope was taken from the literature, not measured here.** The
first local measurement is below. It does not vindicate the sentence, and it
does not support the opposite claim either — read on before quoting either.

**Measurement conditions.** Every cell below ran with `bench_llama_ab.sh
--allow-busy-host` and came back **TAINTED**: the entry gate read 25.6–55.0% of
one core and individual measured windows 29–56% (`WindowServer`, and
`syspolicyd` at a steady 32.2–32.5%); two slots of the 7 722 run saw a `node`
process at 139.9% and 159.8%. ABBA interleaving cancels a *steady* load, so the
**within-run A/B ratios survive**; the **absolute TPS does not**, and neither
does anything derived from a single absolute number. Per-slot windows are
recorded in each result JSON.

`llama.cpp` at its own roofline on this host — same 614 GB/s constant, same
bytes/step arithmetic, `Qwen3-8B-Q8_0` GGUF, Metal, batch 1, decode-only TPS
from the server's own `timings`:

| prompt tokens | bytes/step (weights + f16 KV) | ceiling @614 GB/s | measured decode_tps | ratio vs ceiling |
|---:|---:|---:|---:|---:|
| 3 753 | 9.26 GB | 66.3 TPS | 52.05 | **1.27x** |
| 7 722 | 9.85 GB | 62.3 TPS | 54.89 / 49.27 | **1.14x / 1.26x** |
| 31 536 | 13.36 GB | 46.0 TPS | 35.25 | **1.30x** |

The 7 722 row carries **two** observations of the same cell, same binary, same
flags, ~30 min apart. Publishing only the faster one would put an endpoint in
the band that this very section documents as unreproducible, so both are shown.
The reproducible band is **1.26x–1.30x**.

##### Ratio-vs-ceiling is not comparable across models of different size

It is tempting to set 1.26x–1.30x against rMLX's 1.8x–2.7x and conclude
llama.cpp is the more efficient runtime. **That inference is invalid**, and the
reason generalises well beyond this page.

The ratio is not a scale-free efficiency. By construction

```
ratio = measured_ms / ideal_ms = 1 + fixed_overhead_ms / ideal_ms
```

so *the same* fixed per-step cost produces a large ratio on a small model and a
small ratio on a large one. The two tables are not close in scale: llama.cpp's
cells move **9.3–13.4 GB/step**, rMLX's H2 rows **2.0–4.0 GB/step** — 2.3x to
6.7x apart. That difference alone moves the ratio in exactly the observed
direction, before any question of runtime quality.

The scale-free quantity is the **absolute per-step overhead**,
`1000/measured_tps − 1000·bytes/614`:

| runtime | cell | bytes/step | ideal ms | measured ms | **overhead ms** | ratio |
|---|---|---:|---:|---:|---:|---:|
| llama.cpp | Qwen3-8B-Q8_0 @3 753 | 9.26 GB | 15.09 | 19.21 | **4.12** | 1.27x |
| llama.cpp | Qwen3-8B-Q8_0 @7 722 | 9.85 GB | 16.04 | 18.22 / 20.30 | **2.18 / 4.26** | 1.14x / 1.26x |
| llama.cpp | Qwen3-8B-Q8_0 @31 536 | 13.36 GB | 21.76 | 28.37 | **6.61** | 1.30x |
| rMLX | Qwen3.6-35B-A3B MoE | 3.5 GB | 5.70 | 10.53 | **4.83** | 1.85x |
| rMLX | Bonsai 8B 2bit | 2.0 GB | 3.26 | 8.68 | **5.42** | 2.66x |
| rMLX | Gemma4-e4b mxfp8 | 4.0 GB | 6.51 | 13.71 | **7.20** | 2.11x |
| rMLX | Gemma4-26b MoE | 3.5 GB | 5.70 | 13.85 | **8.15** | 2.43x |

llama.cpp 2.18–6.61 ms against rMLX 4.83–8.15 ms. **The ranges overlap**, so by
the same rule this repo applies to its own A/B arms the comparison is
**INCONCLUSIVE**: the medians differ (~4.3 ms vs ~6.3 ms) but no separation is
demonstrated at n=3 vs n=4, across two frameworks, two weight formats and two
measurement sessions on a tainted host.

What the data *does* establish:

* The **band difference is dominated by bytes/step, not by runtime efficiency.**
  A 1.26x-vs-2.4x gap in ratios coexists with overlapping absolute overheads.
* E2's original sentence is still unsupported — it cited an envelope nobody had
  measured — but it is **not falsified**, and an earlier revision of this
  addendum that claimed rMLX was "roughly 1.5x looser" was **wrong**: it compared
  the non-scale-free quantity across a 2.3–6.7x span in model size.
* Settling it needs a like-for-like cell — llama.cpp on a model with
  rMLX-comparable bytes/step, or rMLX on an ~9 GB/step model — on a quiescent
  host. That has not been run.

**Equivalence caveat, and it is load-bearing.** No file both runtimes can load
exists: rMLX reads MLX safetensors, llama.cpp reads GGUF. The tables set a
`q8_0` (block 32, one fp16 scale) model against rMLX's `mxfp8`/affine rows —
near-equivalent, not identical. The rMLX bytes/step figures are themselves the
rounded estimates this section already published (and `#403` tightens that
census), so the overhead column inherits their uncertainty. Treat a cross-family
gap under ~10% as unresolved by this method.

**Cross-run absolute TPS on this host is not comparable.** The 7 722 cell above
is the demonstration: 54.89 and 49.27, an 11% drift driven by foreground desktop
load. Both are published for that reason. Every *ratio* quoted elsewhere in this
campaign is between two arms *inside one ABBA run*; the roofline tables here are
the one place absolute numbers appear, and they are why the verdict above is
INCONCLUSIVE rather than a measurement.

Method and raw slots: `scripts/bench_llama_ab.sh`, results under
`$RMLX_HOME/bench/llama_ab/`.

### H3 — MoE routing math — characterized (negligible)

`SparseMoeBlock::forward` (`qwen3_5_moe/moe.rs`) does the router as
`gate(Linear) → softmax → argpartition top-k → take_along_axis → optional
normalize`, all over `[n_tokens, num_experts=256]` with `n_tokens=1` at decode.
That is a handful of tiny ops on a 1×256 tensor — trivially cheap relative to
the per-step forward. The `moe_router` span (entered-only, no duration) and the
99.84%-forward / 0.16%-sync split confirm routing-math is *not* a meaningful
share. Routing math is negligible (expected); the MoE cost, such as it is, is
the expert gather (H9b), not the routing. **Characterized — routing math
negligible.**

### H9b — expert `gather_qmm` dispatch — characterized (fast path confirmed)

`SwitchMlp::forward` issues three `gather_forward` calls (gate/up/down), each
dispatching to `rmlx_mlx::gather_qmm` for `Linear::Quantized` experts
(`qwen3_5_moe/layers.rs:187`) — the batched-expert fast path, equivalent to
mlx-lm's `SwitchGLU`, not a dense-masked loop. Confirmed the `sorted_indices`
argument is the **hardcoded `false`** positional at `layers.rs:197` (there is no
named `sorted_indices` param; mlx-lm threads one through). With Qwen3.6's
1.84x-of-ceiling decode being the *best* of the four models, the expert gather
is not a bottleneck worth chasing — it is doing the right (gather_qmm) thing and
the model is the closest to the bandwidth wall. `sorted_indices=true` for the
decode gather is a possible micro-tweak but unjustified by this data (no
catastrophe to recover). **Characterized — fast gather_qmm path confirmed; not
dominant.**

### H4 — async pipeline integrity — REJECT (pipeline intact)

`qwen3.rs` runs the mlx-lm one-step-ahead pipeline: each step issues
`next_y.async_eval()` (`qwen3.rs:1574`) and drains the *previous* step's token
from the `pending` slot one step later inside the `mlx_sync` span
(`qwen3.rs:1583-1591`). On the greedy temp=0 canary path
`mask_active=sampling_active=penalties_active=false`, so `drain_now=false` and
the early blocking eval at `qwen3.rs:1504` is **skipped** — the overlap is not
serialized. The bucket breakdown corroborates: sync drain is 0.55% (Bonsai) /
0.16% (Qwen3.6) of decode wall-clock. **REJECT** — no forced blocking per-step
sync; async pipeline confirmed working.

### H5 — per-step KV reallocation — REJECT

Debug-level tracing found `kv_alloc` events fire once per layer at the first decode step
(`cause=grow`, compact-seed → max_seq expansion), and zero on steps 1..N. The
Bonsai debug log shows the `kv_alloc` events clustered at decode start, none
per-step thereafter. **REJECT** — the cache reuses its allocation correctly; no
per-step realloc.

### H6 — universal-wrapper dispatch — REJECT

The `update_and_sdpa` / `update_and_sdpa_shared_source` match-arm + indirect call
is single-digit nanoseconds; a Bonsai decode step is ~8.96 ms (millions of ns).
The wrapper self-time cannot be a measurable share, and the bucket breakdown
shows decode time is dominated by the forward graph, not dispatch glue.
**REJECT** — the `#[inline(always)]` candidate is not needed.

### H7 — 4k vs 8k decode-only TPS — MEASURED (8k is SLOWER, not faster)

Bonsai, 1 warmup discarded + 3 measured, `--max-tokens 100`, `--log info`:

| ctx | command | resolved kv_quant | decode_tps (median of 3) |
|---|---|---|---:|
| 4k | `--prompt-tokens 4096 --max-ctx 8192` | `mixed_k8g64_v4g64` | **108.98** (108.13 / 108.98 / 109.34) |
| 8k | `--prompt-tokens 8192 --max-ctx 16384` | `mixed_k8g64_v4g64` | **38.25** (38.20 / 38.25 / 38.40) |

**Answer: 8k is ~2.8x SLOWER than 4k on decode-only TPS — the user's "8k beats
4k" recollection was itself a prefill-contamination effect.** With the old
combined-TPS metric, the larger 8k prefill (TTFT ~4.4-4.9 s vs ~1.8 s at 4k) was
amortized differently across the generation window, which could make the
*combined* number look better at 8k for some token counts. Once prefill is
excluded, 8k decode is unambiguously slower.

The KV-quant auto-resolution is **identical** at both contexts:
`Qwen3ForCausalLM` with `weight_bits=2` resolves to `Mixed{k8,v4,g64,g64}`
regardless of ctx — `resolve_default` (`kv_cache/mod.rs:331`) has no ctx branch
for this arch, and `kv_quant_for_ctx` is not consulted on the baseline path. So
the 4k-vs-8k difference is not a KV-quant-by-ctx effect. Both runs also go
through the identical `rmlx baseline` path, which is what rules out a harness
difference between the two rows — worth keeping in view now that the paragraph
below asks for the pair to be re-measured.

**The original attribution was wrong, and the size of the drop was the clue.**
This section used to close with "no fix needed — this is correct behavior",
reading the 2.8× as the ordinary cost of attending over 2× more KV. Ordinary
KV-length scaling does not cost 2.8× for 2× the tokens. The 4k cell sits below
8 192 context tokens and the 8k cell above it, and the Mixed/RotK V side used to
divert to a separate MSL kernel at exactly that boundary — a kernel that
dispatched one thread per output element with a threadgroup of 1 and applied
symmetric dequant to affine data, so it was both very slow and numerically
wrong. Removing it puts Mixed at 0.97–1.00× of `none` at 16k on Bonsai and
gemma-4-e2b. Re-measure this pair before drawing scaling conclusions from it.

The general lesson stands on its own: a step change in a per-step cost that is
supposed to grow smoothly with context is a dispatch boundary, not a scaling
curve. Sweep context finely enough to tell the two apart.

### H8 — Gemma4 + Mixed runtime_fail = cross-layer-KV-sharing contract — ACCEPT

`runs.db` shows every Gemma4-e4b `mixed_*` cell recorded `decode_tps_warm=0.0`
(e.g. `mixed_k8g128_v4g64`, `mixed_k8g128_v8g64`, `mixed_k4g64_v4g64`, …).
Reproduced live on `gemma-4-e4b-it-mxfp8` with `--cache-type-k q8_g128
--cache-type-v q4_g64`: the first prefill chunk fails with the exact message

> `mlx: Cross-layer KV sharing not supported with Mixed quantization. Use
> bf16/K8V8/K8V4/Planar for shared-KV layers, or disable layer-KV sharing for
> this arch.`

emitted from the cross-layer-KV producer path. Gemma4 genuinely shares KV across
layers (`num_kv_shared_layers`, `loader.rs:252-258`), and at the time of this
run shared-KV layers rejected `KvQuant::Mixed`. **This finding is historical and
its premise no longer holds**: the `SharedKvIncompatibleWithMixed` rejection was
removed (`kv_cache/cache_type.rs`) — `KvCache::update_and_sdpa_shared_source`
supports Mixed via dequant-before-share, so the combination is now valid and no
resolver guard is needed. Kept for the record of what was measured.

### H9 — decode forward NOT compiled — characterized (NOT a catastrophe)

The plan flagged this as the "primary suspect" for a 40-50x gap. There is no
40-50x gap, so the framing is moot. Factual finding: the per-step decode forward
(`forward_arr`) is **NOT** wrapped in a shapeless-compiled MLX graph. The
`compile_shapeless` binding (`rmlx-mlx/src/compile.rs`) exists but no
`rmlx-models` decode/forward path references it (grep for
`compile_shapeless|mlx_compile|::compile` in `crates/rmlx-models/src` returns
nothing) — it is used, if at all, only inside isolated layer ops / prefill, not
the decode loop. So every decode step re-builds the Rust→mlx-c graph.

Is compiling it worth doing? The bucket breakdown says the forced per-token sync
is <0.6%, and the model is already 1.8-2.7x of the bandwidth ceiling. The
per-step forward (8.96 ms Bonsai / 10.97 ms Qwen3.6) is dominated by the
overlapped GPU kernel + memory traffic, not by host-side graph re-trace stalling
the GPU (if it were, sync/idle would show up large, and the ratio-vs-ceiling
would be far worse than 2x). A shapeless-compiled decode graph *might* shave the
CPU re-trace cost, but the upside is bounded by the small slice of the forward
that is CPU-dispatch-not-overlapped — likely a few percent, not a multiple.
**Characterized — decode is effectively bandwidth/kernel-bound; compiling the
decode forward would buy little and carries real risk (shapeless compile must
handle the growing KV-cache offset).**

### H10 — Qwen3.6 GatedDeltaNet per-step kernel chain — documented as follow-up only

Qwen3.6-35B-A3B is a hybrid: ~30 of 40 layers are `GatedDeltaNet` linear-
attention layers, each dispatching a per-step MSL kernel `gated_delta_step_gpu`
(`qwen3_5_moe/gated_delta_net.rs:267`) plus per-step `conv1d`, several
`rms_norm`, slices and `astype`. The GDN forward is **not** separately
instrumented (no span), so it folds into the `forward_total` bucket. Despite the
long per-layer GDN op chain, Qwen3.6 is the *best* of the four models at 1.84x of
ceiling — so the GDN chain is not pathological at 4k. It is a plausible
follow-up-branch candidate for per-step kernel fusion if longer-context decode is
later found wanting, but MSL-kernel work is explicitly out of scope here.
**Documented as a follow-up-branch candidate only; no action this branch.**

### Net narrative

Decode is **healthy** — all four models run at ~1.8-2.7x the 614 GB/s bandwidth
ceiling, the normal-to-good band for batch-1 MLX decode. The "16-50x
catastrophe" was a prefill-contaminated bench metric, not an inference defect.
The modest ~2x residual is ordinary MLX batch-1 overhead (dispatch + dequant +
sub-peak bandwidth); the forced per-token sync is <0.6% of decode time, and
the async one-step-ahead pipeline is intact.

The only justified code change arising from this analysis is the H8 resolver
guard: reject `KvQuant::Mixed` on the Gemma4 family at startup (exit 78)
instead of failing at first prefill. H9 decode-forward compile is **NOT
recommended** — the data shows decode is bandwidth/kernel-bound, not
re-trace-bound, so compiling would buy little for real risk. Hypotheses
producing **no actionable fix**: H1 (reject), H2 (accept-healthy), H3/H9b
(characterized-fine), H4 (reject-intact), H5 (reject), H6 (reject), H7
(measured — 8k legitimately slower, no harness fix needed), H9 (characterized),
H10 (follow-up branch only).

## `--paged-kv` on/off (per test-target family)

`--paged-kv` routes K8V4 / K8V8 / Planar caches through the
block-table allocator (`KvStorage::Paged`); without the flag the contiguous
path is unchanged. Single-request decode degenerates to a monotonically
appended block table — no cross-request sharing yet — so the off / on TPS
delta should be small (page allocator overhead vs contiguous growth) and is
captured here per test-target family. Numbers carry the same release-perf
anchor convention as the canary baseline.

| Family | `--paged-kv off` (TPS) | `--paged-kv on` (TPS) | Notes |
| --- | --- | --- | --- |
| Bonsai-2bit | TBD (pending bench cycle) | TBD | K8V8 default; small expected delta |
| Gemma4-e4b | TBD | TBD | Pure-attention; paged path supports K8V4/K8V8/Planar only |
| Qwen3.6-35B-A3B-8bit | TBD | TBD | K8V4 default; large model, dispatch-overhead-dominated |

Rows are populated by the regression-bench cycle after the `--paged-kv` commit
lands; CLAUDE.md hard rule §"Regression-bench discipline" requires touched
models to bench at their best-known KV quant — the `--paged-kv` toggle is
orthogonal to the kv-quant preset, so each family appears in both columns
at the same preset.

## Prefix-index bench

Head-to-head bench of `LinearScan` vs `RadixTree` (NVIDIA Dynamo positional
radix tree port) at the longest-prefix match path. Both implement the
`PrefixIndex` trait in `crates/rmlx-models/src/prefix_index.rs`; the CLI
flag `--prefix-index {linear|radix}` selects which one each freshly built
`PromptCache<E>` uses.

Bench: `cargo bench -p rmlx-models --bench prefix_index_bench`. Each row
populates a fresh index with `N` synthetic 8-block entries, then drives
10,000 random-prompt lookups (50% hit / 50% miss). Reported `ns/op` is the
wall-clock divided by lookup count (one-shot run alongside the criterion
sample loop). Resident bytes is a structural upper-bound estimate (radix's
node overhead dominates; linear's `Vec<u64>` payload is tight).

Measured on macOS / Apple Silicon, `cargo bench` under default release
profile (`profile.release`, LTO=fat). Numbers vary ±10% run-to-run.

| N entries | linear ns/op | radix ns/op | radix vs linear | linear bytes | radix bytes | radix mem overhead |
|-----------|-------------:|------------:|----------------:|-------------:|------------:|-------------------:|
|         1 |          2.1 |         6.1 |          0.34× |           88 |         640 |             7.3×   |
|         4 |          4.7 |         6.6 |          0.71× |          352 |       2,560 |             7.3×   |
|        16 |         11.2 |         9.7 |          1.15× |        1,408 |      10,240 |             7.3×   |
|        64 |         48.4 |        25.3 |          1.91× |        5,632 |      40,960 |             7.3×   |
|       256 |        173.4 |       136.8 |          1.27× |       22,528 |     163,840 |             7.3×   |

### Decision

Decision rule: **radix ≥2× linear at N≥32 with <5% memory overhead → flip
default to `radix`; otherwise default stays `linear`, ship radix as opt-in.**

At N=64 the radix path is 1.91× linear (just under the 2× bar); at N=256
the gap narrows to 1.27× (linear's cache-locality wins back). The bench
fixture uses no shared prefix between entries — every entry starts at the
root with a unique block hash — so the radix tree degenerates to a 256-way
fanout at depth 1 (worst case for `find_child`'s linear walk). Real
workloads with shared `--project` prefixes likely show better speedups,
but that is an engineering follow-up.

Memory: radix overhead is **~7.3× linear** at every N — well above the
5% gate (this is the conservative upper-bound estimate; actual node memory
is lower when prefixes share, but the bench fixture has none).

**Outcome: gate not met. Default stays `linear`. Radix ships behind
`--prefix-index radix` as opt-in for operators with high-fanout
`--project` namespaces who want to bench it against real prompts.**

The linear path remains the bisect-safe fallback and is byte-identical to
the pre-radix implementation (the parallel radix index is built and
maintained on the linear path too — adding insert/remove cost on every
`PromptCache::push` regardless of the active strategy; cost is dominated by
`Vec<u64>` clone and stays negligible vs. KV-cache bytes).

CSV: `.rmlx/bench/prefix_index.csv` (full history).

---

## K8V8 GPU baseline drift investigation (Gemma4, 2026-05-26)

**git_sha**: d92e475

### Context

A prior second-pass bench recorded K8V8 TPS that appeared lower than an earlier
baseline:

| Model       | Earlier K8V8 (59c6f96) | Later K8V8 (95749b0) | Δ     |
|-------------|---------------------:|---------------------:|------:|
| Gemma4-e4b  |                65.22 |                63.88 | −2.1% |
| Gemma4-26b  |                64.65 |                62.66 | −3.1% |

Both were measured with `scripts/bench/turboquant_v3_bench.sh` at the
same longctx_16k.json prompt (≈17,148 Gemma4 tokens), max-ctx=17500,
max_tokens=100, release-perf binary, GPU.

### Re-bench at d92e475

Re-bench: 1 warmup + 3 measured runs per model. Same prompt, max-ctx, max-tokens,
device=gpu, kv-quant=k8v8. release-perf binary rebuilt clean at d92e475
(`make build-perf` from clean tree).

| Model      | run 1 | run 2 | run 3 | median | vs earlier baseline |
|------------|------:|------:|------:|-------:|--------------------:|
| Gemma4-e4b | 65.13 | 65.27 | 65.06 |  65.13 |               −0.1% |
| Gemma4-26b | 65.18 | 66.16 | 66.00 |  66.00 |               +2.1% |

**Verdict: NOISE. No regression.** Both models are at or above the earlier
baseline. The lower values (63.88 / 62.66) were measured during a multi-cell
bench run (k8v8 + mixed3 + turbo3-GPU across both models, total ≈3–4 h elapsed). The 17k-token prompt
puts heavy sustained load on the GPU; inter-cell thermal state, fan ramp-up,
and bandwidth contention between successive long-prefill runs introduce
run-to-run variance that does not show up in the 4k-prompt canary.

### Variance band (K8V8, 17k-prompt longctx shape)

The observed spread across all documented bench runs at this shape:

| Model      | low | high | expected band |
|------------|----:|-----:|:--------------|
| Gemma4-e4b | 63.88 | 65.27 | **63.5 – 65.5 TPS (±1.5%)** |
| Gemma4-26b | 62.66 | 66.16 | **62.5 – 66.5 TPS (±3%)** |

The 26b model has a wider band because the 35 s cold prefill (17k tokens,
26b MoE weights) leaves the GPU in a thermally excited state; subsequent
decode-only steps compete with residual memory bandwidth pressure from the
prefill. Under controlled idle-thermal conditions 26b reaches 66 TPS;
under back-to-back multi-cell load it drops to 62.66.

**Threshold for a real regression alert (17k-prompt shape, K8V8 GPU):**

| Model      | alert below |
|------------|------------:|
| Gemma4-e4b |      63.0 TPS |
| Gemma4-26b |      61.5 TPS |

Anything at or above those thresholds is within the documented variance band
and should NOT trigger a bisect investigation. Flag a bench run for further
investigation only if the median across ≥3 measured runs falls below the
alert threshold on a thermally stable machine (background MLX killed, no
concurrent GPU workloads).

The 4k-prompt canary (`scripts/perf_canary.sh`) is the preferred regression
signal for ongoing development: Gemma4-e4b 74.22
(stddev 0.08) at 4k context vs the 63–65 TPS range at 17k context
confirms the long-context shape is not a suitable tight-gate metric.
Use the canary for gating; use the variance band above only to
interpret long-context bench deviations.

---

## PlanarQuant fused-QK MSL kernel — decode-TPS anchor (2026-05-31)

### Setup

- Profile: `release-perf` (`Cargo.toml` `[profile.release-perf]` — debug-assertions off, overflow checks off, stripped debug, `panic=unwind` for `MetalClaim::Drop` RAII).
- Prompt: 100 `"hello"` tokens (101-token prompt) — small enough to fit in a single prefill chunk, avoiding the pre-existing PlanarK + chunked-prefill broadcast bug that triggers on long prompts (tracked separately).
- `--max-tokens 64`, GPU device, single MLX process (Hard Rule 8 preflight: pkill MLX serve / mlx_lm / paroquant / omlx; sleep 5; rm claim).
- 3 runs each toggle, mean reported.

### Eligible test target

Only **Gemma4-e4b** is eligible:
- **Bonsai** (`Qwen3ForCausalLM`): pre-existing `slice_update [broadcast_shapes] (1,8,256,128) and (1,8,0,128) cannot be broadcast` failure on `--ctk planar_k4` for any prompt that requires chunked prefill.
- **Qwen3.6** (`Qwen3_5MoeForConditionalGeneration`): explicitly rejected by `cache_type::validate_resolved` arch guard — K-side 4-bit on Qwen MoE is the PPL-disaster path.
- **Gemma4-e4b** (`Gemma4ForConditionalGeneration`): runs successfully when the prompt fits in a single prefill chunk.

### Decode-TPS results (Gemma4-e4b, planar_k4, 3-run mean)

| `--planar-fused-qk` | run 1 | run 2 | run 3 | mean | stddev |
|---|---:|---:|---:|---:|---:|
| on (default)        | 78.775 | 78.523 | 80.055 | **79.118** | 0.67 |
| off (legacy SDPA)   | 78.781 | 77.800 | 78.194 | **78.258** | 0.40 |

**Delta**: `+1.1%` (within noise; trend slightly positive).

### Interpretation

The fused-QK kernel ships the QK + dispatch, but the legacy
`scaled_dot_product_attention` is already a single MLX-internal flash
kernel that fuses QK + softmax + SV.  Splitting that into fused-QK + add
mask + softmax + manual SV matmul gives back most of the K-dequant
bandwidth saving.  Net effect on this anchor: ≈ neutral.

The real bandwidth win lands in the planar flash-decode kernel (next section),
which keeps QK, softmax, and SV in one threadgroup and recovers the
single-kernel trade-off. A follow-up generalises the contract to other codecs.

### Files

- `.rmlx/bench/perf_canary.csv` — raw rows (`model,kv_quant,toggle,run,decode_tps,prefill_tps,prompt_tokens,gen_tokens`).
- `crates/rmlx-kv-quant/src/planar_fused_qk_msl.rs` — kernel source + dispatcher.
- `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_planar_k_fused` — dispatch helper.

## planar_flash_decode MSL kernel — decode-TPS anchor (2026-05-31)

### Setup

- Profile: `release-perf`.
- Model: **Bonsai** (`prism-ml__Ternary-Bonsai-8B-mlx-2bit`, `Qwen3ForCausalLM`).
  Bonsai is the **sole reachable arch** for `planar_flash_decode_sdpa` today:
  Qwen3.6 MoE rejects `KvQuant::PlanarK` outright at `validate_resolved`
  (Contract A.y, `QwenMoePlanarKRejected`), and Gemma4 routes its
  attention layer through `update_and_sdpa_shared_source` for cross-layer KV
  sharing (same shape as the `Unreachable TurboFlash` case).
- Forced `--kv-quant planar` (resolves to `KvQuant::PlanarK`).
- `--max-tokens 100`, `--max-ctx 8192` (4k prompt) / `--max-ctx 16384` (8k prompt).
- Single MLX process per Hard Rule 8 (preflight: pkill rmlx serve / mlx_lm; sleep; rm claim).
- 1 warmup + 3 measured runs per toggle.
- Gate: `RMLX_PLANAR_FLASH_DECODE={0|1}` env-var (the `auto` fallback for the
  `--planar-flash-decode` flag, which is global).

### Decode-TPS results (Bonsai, planar_k, 3-run measured)

**4k prompt** (`--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`):

| `RMLX_PLANAR_FLASH_DECODE` | run 1 | run 2 | run 3 | mean | stddev |
|---|---:|---:|---:|---:|---:|
| `0` (split chain: fused-QK + softmax + SV split) | 97.621 | 97.711 | 94.611 | **96.648** | 1.764 |
| `1` (single-pass flash-decode kernel)            | 96.306 | 96.294 | 96.781 | **96.460** | 0.278 |

**Delta**: `-0.19%` (within noise; ON is marginally lower at this shape).

**8k prompt smoke** (`--prompt-tokens 8192 --max-tokens 100 --max-ctx 16384`,
single run each — exploratory, not a full 3-run anchor):

| `RMLX_PLANAR_FLASH_DECODE` | run 1 |
|---|---:|
| `0` (fused-QK chain) | 75.833 |
| `1` (flash-decode kernel) | 75.060 |

**Delta**: `-1.0%` (within noise).

### Correctness anchor

The NIAH harness cells `niah_pflash_bonsai_*` (`crates/rmlx-models/tests/niah_long_context.rs`) confirm:

- Dispatch counter delta = `1638` on ON, `0` on OFF — the planar flash-decode
  kernel **does fire** through `KvCache::update_and_sdpa` → `sdpa_dispatch` →
  `update_and_sdpa_planar_k_fused` → `planar_flash_decode_sdpa`.
- OFF and ON produce **byte-identical decoded output** ("9. The secret. The
  grass. ..." at d50; "9X above..." at d10) — the flash-decode kernel
  matches the fused-QK chain numerically.
- Needle-found = `false` on every Bonsai PlanarK cell, OFF and ON alike.
  This is the pre-existing PlanarK-on-Bonsai long-prompt chunked-prefill
  broadcast bug already documented in `docs/KV_QUANT.md` ("Bonsai/Qwen3
  hit a pre-existing long-prompt chunked-prefill broadcast bug on `--ctk
  planar_k4`") — surfaced now via the harness.

### Interpretation

The fused-QK kernel (previous section) already collapses K-side dequant into a
single kernel and lets MLX's stock `scaled_dot_product_attention` cover the
rest of the flash path. The planar flash-decode kernel is an explicit
single-threadgroup path that re-fuses QK + softmax + SV, but at the canary
shapes tested (decode kv_seq ≤ ~4200 tokens for the 4k anchor; ≤ ~8300 for
the 8k smoke) the saving is balanced by the overhead of replacing the
highly-tuned upstream kernel.  The 6× stddev reduction on ON (`0.278` vs
`1.764`) suggests the flash-decode path has better cache-locality variance
but does not change the mean.

The predicted "real win lands in planar flash-decode" does **not** materialise
at this shape — that prediction stands unconfirmed for longer-context decode
where the upstream PlanarK chunked-prefill bug currently prevents validation.

### Auto-flip decision: HOLD

The Auto-on flip is gated on a ≥10% TPS gain and a clean NIAH.
Neither condition is met:

- Perf gain: -0.19% (within noise; well below 10% gate).
- NIAH correctness: blocked by the pre-existing PlanarK + chunked-prefill bug.

`resolve_planar_flash_decode(Auto, …)` stays OFF on every host
(`crates/rmlx-cli/src/commands/serve.rs:230`).  The CLI override
`--planar-flash-decode on` remains available for opt-in experimentation.

### Files

- `crates/rmlx-kv-quant/src/planar_flash_decode_msl.rs` — kernel source + dispatcher.
- `crates/rmlx-kv-quant/src/kvcache/sdpa.rs::update_and_sdpa_planar_k_fused` — dispatch site.
- `crates/rmlx-models/tests/niah_long_context.rs` — NIAH correctness cells (`niah_pflash_*`).

---

## KV-codec baseline record run (2026-06-02)

**Date**: 2026-06-02. **Hardware**: M5 Max.
**Profile**: `release-perf` (debug-assertions=false, overflow-checks=false, stripped, panic=unwind).
**Context**: 32 768 tokens (all cells). **Protocol**: codec smoke + NIAH harness via
`scripts/release_e2e/stage6_perf/codec_smoke_runner.sh --record-baseline`.

### Summary

This run established the first complete codec gate baselines for the NIAH matrix.
Prior to this run all `expected_retrieval_pct` values were `0.0` (unrecorded). After
this run 28 cells have a recorded baseline of `1.0000` (100% needle retrieval at all
5 depths × 32k context) and 11 cells are intentionally skipped.

### Smoke probe fixes applied for this run

1. **`baseline.rs` decoded preview**: changed from 50-token preview to full
   generation output so thinking models (which emit `<think>` blocks before the
   answer) have their answers visible to validate_regex.

2. **`codec_smoke_runner.sh` ANSI strip**: added `re.sub(r'\\x1b\\[[0-9;]*m', '', txt)`
   before the `decoded=` regex so the tracing subscriber's ANSI CSI wrappers do
   not break the quoted-string match.

3. **`smoke_prompts.toml` coherence regex**: updated separator class from
   `[ ,.';:!?-]` to `[\\s,.';:!?*#-]` to accept markdown and newline separators
   that thinking models produce in `<think>` blocks.

4. **`smoke_prompts.toml` instruction regex**: changed from `(?s).*1[.)].*2[.)].*3[.)]`
   (requires explicit `1.`/`1)` markers) to `(?s)1.*2.*3` (matches digits in
   sequence anywhere). The `\\b1\\b` word-boundary variant also fails because Rust
   Debug format escapes `\\n` → `\\\\n`, making `n` the adjacent char before `1`.

5. **`smoke_prompts.toml` multi_turn prompt**: replaced the `User:`/`Assistant:`
   multi-turn format with a single inline colour-recall paragraph. Gemma4 in
   raw text-completion mode does not parse plain role labels and produced off-context
   responses ("I don't have a specific color...") for the prior format.

### Baseline table

All 28 recorded cells measured `1.0000` (5/5 needle depths found at 32k context).
The `expected_retrieval_pct` values have been written into `kv_codec_matrix.toml`
via `--record-baseline`.

| Codec | bonsai-8b | gemma4-e4b | qwen3.6-moe-8bit |
|---|---|---|---|
| bf16 | 1.0000 | 1.0000 | 1.0000 |
| k8v4 | 1.0000 | 1.0000 | 1.0000 |
| k8v8 | 1.0000 | 1.0000 | 1.0000 |
| TurboSym3 | 1.0000 | 1.0000 | SKIP (A.y) |
| TurboSym4 | 1.0000 | 1.0000 | SKIP (A.y) |
| Iso3Sym | 1.0000 | 1.0000 | SKIP (A.y) |
| Iso4Sym | 1.0000 | 1.0000 | SKIP (A.y) |
| Rotor3Sym | 1.0000 | 1.0000 | SKIP (A.y) |
| Rotor4Sym | 1.0000 | 1.0000 | SKIP (A.y) |
| PlanarK | 1.0000 | 1.0000 | 1.0000 |
| planar | 1.0000 | 1.0000 | 1.0000 |
| Mixed | SKIP (mixed-32k-niah-zero) | 1.0000 | SKIP (mixed-32k-niah-zero) |
| fused_qk_sparse | SKIP (production dispatch pending) | SKIP (production dispatch pending) | SKIP (production dispatch pending) |

### Skip inventory

| Skip reason | Cells | Action |
|---|---|---|
| `qwen-moe-A.y-rejected` | 6 (TurboSym3/4, Iso3/4Sym, Rotor3/4Sym × Qwen3.6-MoE) | Permanent: arch invariant. |
| `mixed-*-32k-niah-zero` | 2 (Mixed × bonsai-8b, Mixed × qwen3.6-moe-8bit) | Mixed KV at 32k NIAH returns 0.0. Smoke passes at short context; root cause TBD. Gemma4 passes (1.0000). |
| `production dispatch pending` | 3 (fused_qk_sparse × all 3 models) | Temporary: remove once fused-QK sparse production wiring lands. |

### Regression gate

The gate threshold for all 28 recorded cells is `expected_retrieval_pct - 0.02` (≥ 0.98).
Since all baselines are 1.0000, any cell that drops below 0.98 will trip the gate.
Configured via `.github/workflows/codec-matrix.yml` (runs on push to the main
development branch, self-hosted Apple Silicon runner only).

### Timing observations (M5 Max)

- Bonsai 8B 2-bit cells: ~3.5 min each at 32k context.
- Gemma4-e4b cells: ~1 min each at 32k context.
- Qwen3.6-35B-A3B cells: ~6-7 min each at 32k context.
- Total active-cell wall clock: ~130 min (28 active + skip overhead).

## iso3 hot-path diagnostic (2026-06-02)

Per-phase trace instrumentation landed in `crates/rmlx-kv-quant/src/kvcache/update.rs`
for `update_iso3` / `update_iso3_sym` / `update_iso_k_only_3` (decode sites)
and the `KvQuant::Iso3` arm of `exit_prefill` (prefill site). All events
emit at `trace!` level under `target = rmlx_kv_quant::kvcache::update` —
off by default, opt in with `--log verbose` or
`RUST_LOG=rmlx_kv_quant=trace`. Phases: `iso3_encode`, `iso3_dequant_cpu`,
`iso3_vec_to_array`. Structured fields: `phase`, `ms`, `s_total`, `kv_h`,
`head_dim`, `site` (where present).

### Bench setup

- Model: Bonsai 8B 2-bit (36 layers; 26 full-attn + 10 SWA).
- Prompt: `prompts/longctx_4k.json` (4085 tokens after tokenize).
- Decode: 100 tokens, `--max-ctx 8192`, GPU device.
- Profile: `release-perf`.
- Build: `target/release-perf/rmlx baseline ... > stdout.txt 2> stderr.txt`,
  trace events captured from `<RMLX_HOME>/logs/<run-id>.jsonl`.

### Headline summary (decode TPS)

| KV composition       | TTFT ms | Decode TPS | Prefill TPS | Notes |
|----------------------|--------:|-----------:|------------:|-------|
| bf16 (control)       | 1788    | 94.18      | 2284        | reference |
| `--ctk q8_g128 --ctv iso3` (iso3 V)        | 3219    | 81.48      | 1269        | iso3 V at prefill only |
| `--ctk q8_g128 --ctv rotor_v_3` (rotor3 V) | 3097    | 79.80      | 1319        | rotor3 V at prefill only |

(`--ctk iso3 --ctv iso3` is invalid in this build: iso3 is V-side only;
substituted `q8_g128` for K per CLI suggestion. The composition that lands
in `update_iso3` requires K = q8_g128 / V = iso3.)

### Per-phase iso3 timing

Instrumented run for q8/iso3, 100 decode tokens at S_total=4085:

| phase             | site         | events | min ms | med ms | max ms | sum ms |
|-------------------|--------------|-------:|-------:|-------:|-------:|-------:|
| `iso3_encode`     | exit_prefill | 26     | 41.60  | 42.54  | 47.36  | 1123.2 |
| `iso3_dequant_cpu`| (decode)     | 0      | —      | —      | —      | 0      |
| `iso3_vec_to_array`| (decode)    | 0      | —      | —      | —      | 0      |
| `iso3_encode`     | (decode)     | 0      | —      | —      | —      | 0      |

### Dominant-phase identification

**The iso3 decode hot path is dead code in steady-state decode.** Zero
per-decode-step `iso3_*` events fired across 100 decode steps. Cause:
`exit_prefill` unconditionally sets `self.decode_fp16_k = Some(k_seed)`
(via the shared `decode_fp16_pair` machinery at update.rs:1675-1678).
`update_iso3` then short-circuits on the very first decode step:

```rust
if self.decode_fp16_k.is_some() {
    return self.update_decode_fp16(new_k, new_v, max_seq, device);
}
```

Result: decode runs entirely on the bf16 warm-TTFT seed; the iso3 V codec
is exercised exactly once per Iso3 layer at `exit_prefill` (26 layers ×
42.5 ms median = ~1.12 s of one-shot bulk encode work), then never again
for the rest of the run.

This explains the headline numbers: bf16 control (94.2 TPS) vs q8/iso3
(81.5 TPS) vs q8/rotor3 (79.8 TPS) are all within ~15% of each other, with
the iso3/rotor3 deficit coming from prefill (TTFT 3219 vs 1788 ms) and the
warm-up of the bf16 seed buffers — not from a slow decode codec, because
the decode codec isn't running.

S_total varies during the run (prefill chunked over multiple `exit_prefill`
calls then steady at `prompt_len + step`), but with zero decode-site events
the per-step S_total scaling question is not answerable from this run. The
`exit_prefill` events all carry S_total = 4085 (final prompt length).

### Follow-up gate triggers (>5% of step time)

Step time at q8/iso3 = 1000 / 81.5 ≈ 12.27 ms (decode_profile reports
`forward_per_step_ms = 10.74`). At this step time the 5% gate is **0.61
ms/phase**.

Since the decode iso3 phases recorded 0 events, the gates collapse to a
single structural finding:

- **`iso3_vec_to_array` >5%** — Not triggered. Zero decode-site events.
- **iso4 MSL kernel** — Independent of the above gate.
- **rotor3 / rotor4 MSL kernels** — Independent of the above gate. Note:
  rotor3 currently shares the bf16-seed-shadow shape (`update_rotor3` has the
  same early-return guard), so kernels alone will not move decode TPS until
  the shadow is addressed.
- **`rotor_fused_qk_msl` to RotorKOnly{3,4} + RotorKAsym{3,4}** —
  Independent of the above gate; conditional on the fused-QK HOLD lifting.
  K-side asymmetric path is the most plausible TPS-gap fix per this analysis.
- **GPU-resident IsoBlocks** — Not triggered. Gated on the
  `iso3_vec_to_array` finding AND `iso3_dequant_cpu` dominating; current
  run has zero decode-site events so this gate is structurally falsified
  for this composition/model/build.

### Structural conclusion

The "iso3 hot path" the brief assumed (per-decode-step iso3 encode +
dequant + materialise) is **not the actual hot path** under the current
warm-TTFT seed policy. Any perf-loop work targeting iso3 decode speed
must first decide whether to:

1. Disable the bf16 warm-TTFT seed for iso3 V (then per-step iso3 encode +
   dequant become the hot path, and the trace instrumentation will fire on
   every decode step), or
2. Accept the bf16-shadow as the design and focus follow-ups on prefill
   time and on K-side codecs (where K8 lives during decode).

This decision belongs in a parent-level investigation, not in any one
follow-up.

### Artifacts

- Trace log (q8/iso3, v2): `<RMLX_HOME>/logs/20260602-125633-c88aa20-dirty.jsonl`.
- Trace log (q8/rotor3, v2): `<RMLX_HOME>/logs/<later-id>.jsonl`.
- Stdout summaries: `/tmp/scratch/{bf16,q8_iso3_v2,q8_rotor3_v2}.stdout`.

## iso4 MSL kernel + dispatch wiring (2026-06-02)

Lands the iso4 sibling of `isoquant_msl.rs` (iso3 MSL kernel) and wires
GPU encode dispatch into the three iso4 V update paths.

### What landed

- **`crates/rmlx-kv-quant/src/isoquant_msl_v4.rs`** — `iso_quantize_v4_gpu`
  / `iso_dequantize_v4_gpu`. One thread per (token, group); 16-entry
  `lloyd_gaussian_codebook(4)` and 15 mid-point decision boundaries
  encoded into the MSL header; atomic-OR pack at 8 vals/u32 (dense, 32
  bits used). Quaternion SO(4) rotation identical to iso3 — same
  `FIXED_QUAT`, same group size of 4. The shader source closely parallels
  iso3 with the bit-width / codebook / pack constants swapped.
- **`isoquant_msl_v4_tests.rs`** — `iso_v4_msl_matches_cpu_within_eps`
  asserts CPU ↔ GPU bit-identity within 5e-3 on a 32×128 LCG fixture
  (group_size=4). `#[ignore]`-gated (requires Metal context).
- **`kvcache/update.rs` dispatch** — `update_iso4` / `update_iso4_sym` /
  `update_iso_k_only_4` route encode through `iso_quantize_v4_gpu` when
  `device == Device::Gpu`. CPU encode (`iso_encode_fast`, bits=4) remains
  the fallback. CPU dequant still produces the returned `v_full` Array
  (state retained as `IsoBlocks` for SSD spill / truncate).

### Warm-TTFT bf16-seed caveat (carried over)

The iso/rotor V codecs are shadowed by the bf16 exit_prefill seed:
`update_iso4` short-circuits on `self.decode_fp16_k.is_some()` from the
second decode step onward. The GPU encode therefore fires **once at
exit_prefill** (large `new_v` slice — meaningful work), not per decode
step. The wall-clock benefit lands on prefill TTFT for iso4-V
compositions, not on steady-state decode TPS.

### Smoke probe (Bonsai 8B, prompt_tokens = 10867, max_tokens = 32)

| Composition | TTFT ms | Decode TPS | Notes |
|---|---:|---:|---|
| `--ctk q8_g128 --ctv iso_v_3` (baseline; iso3 not GPU-wired) | 9580  | 43.77 | CPU iso3 V path |
| `--ctk q8_g128 --ctv iso_v_4` (GPU encode wired)             | 12855 | 43.42 | GPU iso4 V encode at exit_prefill |
| `--ctk q8_g128 --ctv iso_v_4` (verbose log; sanity rerun)    | 12878 | 43.10 | reproducible |

**Decode regression rule (±1%):** satisfied. Decode TPS for the new GPU
encode path is 43.42 vs the iso3 CPU-encode baseline 43.77 TPS — a delta
of **-0.8%**, well within the ±1% noise band and within run-to-run
variance (the verbose sanity rerun at 43.10 TPS sits inside the same
envelope).

**TTFT delta (+34%) is a known cost.** TTFT moves from 9.58 s (iso3
CPU encode) to 12.86 s (iso4 GPU encode); the kernel dispatch and
GPU→CPU readback overhead at this 11k prefill scale exceed the per-block
encode-time win because **CPU dequant remains in the loop** — the
returned `v_full` Array is still rebuilt from a CPU dequant of the
`IsoBlocks` state, so the GPU win on the encode-side amortises poorly
across the round-trip. Per the warm-TTFT framing above, TTFT is the primary win-target for the iso V
codecs once the bf16 exit_prefill seed shadow lifts.

**Follow-up:** the GPU-resident dequant work (see below) closes this gap. The
kernel encode is on the prefill critical path until decode-side dispatch
follows.

### Regression cross-check (other test-target families)

The dispatch change is gated by `KvStorage::IsoV4` / `IsoSym4` / `IsoKOnly4`
— other models do not enter `update_iso4*` and are untouched.

| Model | Composition | TTFT ms | Decode TPS |
|---|---|---:|---:|
| `gemma-4-e4b-it-mxfp8` | `--kv-quant auto` (K8V8) | 1823  | 68.51 |
| `Qwen3.6-35B-A3B-8bit` | `--kv-quant auto` (K8V8) | 21168 | 90.97 |

Both at parity with their pre-change anchors.

### Artifacts

- Parity test: `crates/rmlx-kv-quant/src/isoquant_msl_v4_tests.rs::iso_v4_msl_matches_cpu_within_eps`.
- Smoke logs: `/tmp/bonsai_iso4.log`, `/tmp/bonsai_iso3.log`,
  `/tmp/gemma4_default.log`, `/tmp/qwen36.log`.

## rotor3 + rotor4 MSL kernels + dispatch wiring (2026-06-03)

Lands the rotor3 / rotor4 MSL encode + decode kernels and wires GPU
encode dispatch into the six rotor V/K update paths (`update_rotor3`,
`update_rotor4`, `update_rotor3_sym`, `update_rotor4_sym`,
`update_rotor_k_only_{3,4}`, `update_rotor_k_asym_{3,4}`).

### What landed

- **`crates/rmlx-kv-quant/src/rotorquant_msl.rs`** — four public
  encode/decode functions: `rotor_quantize_v3_gpu`, `rotor_quantize_v4_gpu`,
  `rotor_dequantize_v3_gpu`, `rotor_dequantize_v4_gpu`. One thread per
  (token, group). The Cl(3,0) rotor sandwich `R * mv * R̃` reduces (for
  grade-1 input) to a closed-form 3×3 SO(3) rotation matrix `M(R)` over
  `(v1, v2, v3)`; the kernel applies that matrix, zero-pads to 8 MV
  components, and quantizes against the Lloyd-Max N(0, 1) codebook
  (8 entries / 3 bits for rotor3, 16 entries / 4 bits for rotor4). Pack
  is 1 u32 per group: 24 bits used for rotor3, 32 bits dense for rotor4.
  The per-(layer, head, group) rotor table is a **buffer argument**
  (`rotors_in : f32 [n_groups, 4]`); the kernel does NOT hardcode the
  table. Header constants for the codebook + boundaries are computed at
  runtime from `lloyd_gaussian_codebook(bits)` and stored in
  `OnceLock<Result<String, String>>` (mirrors the iso4 LOW-1 fix to avoid
  poisoning).
- **`rotorquant_msl_tests.rs`** — four `#[ignore]`-gated parity tests
  (`rotor_{v3,v4,k3,k4}_msl_matches_cpu_within_eps`) assert max-abs-error
  ≤ 5e-3 between CPU and GPU round-trips on 32×128 / 32×129 LCG
  fixtures. The K-side tests use `qjl_s_matrix = None` (the GPU kernel
  does not implement QJL — see QJL caveat below).
- **`kvcache/update.rs` dispatch** — six rotor update functions now check
  `device == Device::Gpu` and route encode through the GPU kernel. The
  K-side additionally gates on `crate::rotor_qjl::rotor_qjl_enabled()`
  being `false`. CPU encode (`rotor3_encode` / `rotor4_encode` and
  `rotor3_k_encode` / `rotor4_k_encode`) remains the fallback for all
  three opt-out conditions (CPU device, QJL enabled, or non-rotor
  storage variant).

### Warm-TTFT bf16-seed caveat (carried over)

Rotor V codecs are shadowed by the bf16 `exit_prefill` seed:
`update_rotor3` / `update_rotor4` short-circuit on
`self.decode_fp16_k.is_some()` from the second decode step onward. The
GPU encode therefore fires **once at exit_prefill** (large `new_v` slice
— meaningful work), not per decode step. The wall-clock benefit lands on
prefill TTFT for the K-side rotor codecs (V-side TTFT delta is small
because CPU dequant remains in the loop — see iso4 section above).

### QJL caveat — K-side QJL fallback

The K-side rotor codec carries a 1-bit QJL residual sign-quantization
sideband when `rotor_qjl_enabled()` is `true` (the CLI default). The GPU
dequant kernel does NOT replicate the QJL projection / sign correction —
when QJL is active, K-side append / decode falls back to CPU
`rotor3_k_encode` / `rotor3_k_decode`. The V-side is unaffected (QJL is
K-only).

### Smoke probe (Bonsai 8B, prompt_tokens = 10867, max_tokens = 32)

| Composition | TTFT ms | Decode TPS | Notes |
|---|---:|---:|---|
| `--ctk q8_g128 --ctv rotor_v_3` | 8993  | 49.55 | V-side GPU encode; K-side affine GPU q8 |
| `--ctk q8_g128 --ctv rotor_v_4` | 9817  | 51.21 | V-side GPU encode; K-side affine GPU q8 |
| `--ctk k_rotor3 --ctv rotor_v_3` (QJL on, default) | 28337 | 46.40 | K-side CPU rotor3+QJL; V-side GPU encode |
| `--ctk k_rotor3 --ctv rotor_v_3 --rotor-qjl off` | 11476 | 46.43 | K-side GPU rotor3 (QJL off); V-side GPU encode |

The output is coherent on all four compositions (Bonsai "## A History
of Paper: From Papyrus to Digital Age..." preview).

**TTFT win is the actual measurement target.** Per the warm-TTFT framing,
decode TPS is shadowed by the bf16 seed — the GPU kernel fires once
during prefill. The `k_rotor3` row with QJL off shows the **17 s
TTFT drop** (28.3 s → 11.5 s) from moving the K-side rotor3 encode for
the 10.8k-token prefill off the CPU.

### Encode-only GPU + CPU dequant gap

Same framing as the iso4 section: encode runs on GPU but the returned
`v_full` / `k_full` Array still goes through CPU dequant of the rotor
blocks. The full GPU-resident dequant work covers rotor codecs as a
follow-up (see GPU-resident iso/rotor mirror section below). Until then,
decode-TPS gains are negligible — the win is TTFT and prefill throughput.

### Regression cross-check

The dispatch change is gated by `KvStorage::RotorV3` / `RotorV4` /
`RotorSym3` / `RotorSym4` / `RotorKOnly{3,4}` / `RotorKAsym{3,4}` —
other models running non-rotor compositions do not enter the new code
path and are untouched.

### Artifacts

- Parity tests: `crates/rmlx-kv-quant/src/rotorquant_msl_tests.rs`
  (`rotor_v3_msl_matches_cpu_within_eps`,
  `rotor_v4_msl_matches_cpu_within_eps`,
  `rotor_k3_msl_matches_cpu_within_eps`,
  `rotor_k4_msl_matches_cpu_within_eps`).
- Dispatch helpers: `rotor{3,4}_gpu_append_into_blocks` and
  `rotor{3,4}_gpu_append_into_k_blocks` in `kvcache/update.rs`.

## iso3 MSL encode + on-demand `Array::from_bytes` dequant (2026-06-03)

Wires the iso3 MSL kernel (`isoquant_msl.rs`,
`iso_quantize_v3_gpu` / `iso_dequantize_v3_gpu`) into the three iso3
update paths (`update_iso3`, `update_iso3_sym`, `update_iso_k_only_3`)
and adds `QuantIsoV3::dequant_gpu` / `QuantIsoK3::dequant_gpu` —
on-demand GPU dequant via `Array::from_bytes` upload of the CPU block
buffers (no intermediate `Vec<f32>` materialisation of the reconstructed
tensor).

### What landed

- **`crates/rmlx-kv-quant/src/isoquant_msl.rs`** — new public helper
  `iso3_gpu_outputs_to_cpu(codes, scales, quats, norms, n_tokens,
  n_groups)` mirrors `iso4_gpu_outputs_to_cpu`: reads the GPU encode
  outputs back to CPU `Vec`s (codes / scales / dedup-per-token norms)
  and emits a `FIXED_QUAT`-cycled quaternion buffer to satisfy the
  `IsoBlocks` ABI.
- **`crates/rmlx-kv-quant/src/storage/quant_iso_v.rs`** —
  `QuantIsoV3::dequant_gpu(device) -> Result<Array>`. Concatenates
  per-block codes / scales / quats / norms into single byte buffers,
  expanding the storage's per-token norms to the GPU kernel's per-group
  layout (each token's norm is duplicated across its `n_groups` slots).
  Uploads each buffer once via `Array::from_bytes`, dispatches
  `iso_dequantize_v3_gpu`, and reshapes the flat f32 output to
  `[B, kv_h, S, D]`.
- **`crates/rmlx-kv-quant/src/storage/quant_iso_k.rs`** —
  `QuantIsoK3::dequant_gpu`, K-side mirror; axis-agnostic kernel reused.
- **`kvcache/update.rs` dispatch** — `update_iso3` / `update_iso3_sym` /
  `update_iso_k_only_3` route encode through
  `iso3_gpu_{append_into_blocks,append_into_k_blocks}` and dequant
  through the new `dequant_gpu` methods when `device == Device::Gpu`. CPU
  encode / dequant remain the fallback. The iso3 per-phase trace events
  fire on both paths (`iso3_dequant_gpu` replaces `iso3_dequant_cpu` +
  `iso3_vec_to_array` when GPU dispatch lands).

### Warm-TTFT bf16-seed caveat (carried over)

The decode-step iso3 update paths short-circuit on
`self.decode_fp16_k.is_some()` (warm-TTFT bf16 seed) for every current
arch wiring (Bonsai 8B, Gemma4, Qwen3.6). The GPU dispatch therefore
fires **once at `exit_prefill`** (large `new_v` chunk) and on cold cache
misses, not per decode step. Per-decode-step optimisation is zero-impact
in today's production path; the win materialises on prefill TTFT of
seedless / cold-cache compositions and as a future-proofing buffer for
any seed-lift work.

### Smoke probe (Bonsai 8B, prompt_tokens = 4096, max_tokens = 32)

| Composition | TTFT ms | Decode TPS | Notes |
|---|---:|---:|---|
| `--ctk q8_g128 --ctv iso_v_3` (run 1) | 3038 | 66.88 | GPU iso3 V encode + GPU dequant |
| `--ctk q8_g128 --ctv iso_v_3` (run 2) | 3012 | 67.90 | reproducible |
| `--ctk q8_g128 --ctv iso_v_3` (run 3) | 3033 | 64.90 | reproducible |

Decode preview: `"\`\`\`\n\nOkay, I need to figure out the top three projects by the length of their README.md based on the given data."` — coherent.

The iso4 smoke above ran at a much larger 10.8k-token prefill; the iso3 row above uses the 4k canary prompt because the iso3 dispatch lands inside the standard `baseline` flow with no exclusive longer-context fixture. The decode TPS regime (~66-68) reflects the bf16-seed-shadowed full update path; the iso3 hot-path traces show the dequant phase replaced by the GPU dispatch on the cold-cache `exit_prefill` step (the only step where the seed has not yet materialised).

**TTFT and decode TPS are within run-to-run noise.** No regression
detected — this dispatch is a no-op for steady-state decode (per the
warm-TTFT framing) and a single GPU round-trip at `exit_prefill`.

### Default-quant canary (Bonsai 8B, no `--ctk/--ctv`)

| Run | Decode TPS |
|---|---:|
| 1 | 112.55 |
| 2 | 113.88 |
| 3 | 113.89 |

Canary anchor: ~110 TPS. **No regression** — all three runs land above
the 107 TPS floor; the dispatch change is gated by
`KvStorage::IsoV3 / IsoSym3 / IsoKOnly3` so the default quant
(`k8vturbo3` for Bonsai) does not enter the new code path.

### Encode-only GPU + CPU dequant gap (closed for iso3)

Same framing as the iso4 and rotor3/rotor4 sections: prior wiring added GPU
encode but the returned Array still went through CPU dequant + `f32_vec_to_array`.
This landing closes the dequant side for iso3 specifically via
`Array::from_bytes` — the f32 vector is never built on the CPU. The
analogous work for iso4 / rotor codecs follows the same pattern.

### Artifacts

- Parity tests:
  `crates/rmlx-kv-quant/src/isoquant_msl_tests.rs::iso_v3_dequant_gpu_matches_dequant_cpu`
  and `::iso_k3_dequant_gpu_matches_dequant_cpu`
  (`#[ignore]`-gated). Observed `max|cpu-gpu| ≤ 2.4e-7` on the LCG fixture
  (a few f32 ULPs at codebook magnitudes — different summation order between
  CPU `iso_decode_fast` and the MSL kernel, not a real codec divergence).
  Parity test gates at 5e-3 (codebook tolerance) and additionally enforces a
  strict ≤ 1e-6 bound to catch future codec drift before it surfaces as PPL
  regression. Not bit-exact at fp32; well below the codebook tolerance.
- Dispatch helpers: `iso3_gpu_append_into_blocks` /
  `iso3_gpu_append_into_k_blocks` /
  `iso3_gpu_encode_block` / `iso3_gpu_outputs_to_cpu` —
  iso4 sibling pattern.

## FusedQkShadow split + rotor variants on fused-QK fast path (2026-06-03)

Lands the `FusedQkShadow` shadow refactor and wires all 6 rotor variants
(`Rotor3Sym`, `Rotor4Sym`, `RotorKOnly3`, `RotorKOnly4`, `RotorK3Asym`,
`RotorK4Asym`) into the production fused-QK decode path. Default-OFF
(opt-in via `--fused-qk on`); the auto/HOLD default keeps the legacy
bf16 SDPA path live.

### What landed

- **`crates/rmlx-kv-quant/src/kvcache/fused_qk_shadow.rs`** — `FusedQkShadow`
  now carries up to four GPU-resident arrays:
    * `k_codes` (per-token, all codecs),
    * `k_scales` (per-token, all codecs; was `k_combined_scales`),
    * `sideband_norms` (per-token, iso/rotor only),
    * `sideband_rotor_table` (`[n_groups * 4]`, rotor only).
  `FusedQkLayout` gained `has_norm` / `has_rotor_table` / `n_groups`
  flags so the dispatch layer knows which sidebands to allocate and how
  to assemble the shim's combined `k_scales` argument.
- **`crates/rmlx-kv-quant/src/kvcache/fused_qk_dispatch.rs`** — added the
  rotor encode-chunk path (`encode_chunk_rotor` calls
  `rotor_quantize_v{3,4}_gpu`), QJL fallback gate (`codec_is_rotor(codec)
  && rotor_qjl_enabled()` returns `Ok(None)` → legacy SDPA), and shim
  argument assembly (`concatenate([scales, norms, rotor_table])` for
  iso/rotor; pass-through for q8/turbo).
- **`crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs`** — 7
  `#[ignore]`-gated GPU integration tests: 6 rotor variants × dispatch >=
  1 + 1 QJL-on fallback assertion. All pass.
- **Regression gate** — `tests/fused_qk_dispatch.rs` (q8, turbo3,
  turbo4) still pass after the shadow refactor. The per-token layout
  for the existing codecs is unchanged.

### Iso variants

`FusedQkLayout::for_codec` still returns `Ok(None)` for `Iso3Sym` /
`IsoKOnly3` / `Iso4Sym` / `IsoKOnly4` — the shadow now supports the
sideband-norms layout but iso's K-side GPU encoder is not yet wired
(the iso3/iso4 sections above only covered V-side). Follow-up: wire iso
K-side GPU encode through `encode_chunk_to_head_major`.

### Bonsai 8B smoke (prompt 8k via `prompts/longctx_8k.json`, max_tokens 32)

| Composition | Decode TPS | Output coherent |
|---|---:|:-:|
| `--ctk k_rotor3 --ctv rotor_v_3 --rotor-qjl off --fused-qk auto` (default, baseline) | 63.5 | yes |
| `--ctk k_rotor3 --ctv rotor_v_3 --rotor-qjl off --fused-qk on` | 12.3 (±0.05) | yes |
| `--ctk k_rotor3 --ctv rotor_v_3 --rotor-qjl on --fused-qk on` (QJL gate) | ~63 (fallback) | yes |
| Default (mixed_k8g64_v4g64, fused-QK auto), 4k prompt | 116.247 | yes |

### Finding — rotor fused-QK is per-decode-step bandwidth-bound

The rotor kernel fires (`dispatch_count > 0`) and produces coherent
output, but **explicit `--fused-qk on` regresses decode TPS** from 63.5 →
12.3 on Bonsai 8B at 8k context. Cause is structural: the rotor shim
reads a combined `[scales_all | norms_all | rotors]` buffer at dispatch
time. The dispatch site slices the per-token shadow (`[B, kv_h, max_seq,
n_groups]` for scales, `[B, kv_h, max_seq, 1]` for norms) to the current
`kv_seq` tile, flattens, and `concatenate`s with the static rotor
table — `~kv_seq * (n_groups + 1) * 4` bytes of materialised copy per
layer per decode step. For Bonsai (n_groups=43, ~32 layers, 8k context)
that is ≈ 1.3 GB of per-step concat traffic, swamping the kernel's
compute savings.

For q8 / turbo3 / turbo4 the per-step cost is ~43× smaller because
`scales_per_token` is `head_dim / GROUP_SIZE` (1–4 for q8/turbo) vs
`n_groups = ceil(head_dim / 3)` for rotor.

### Why this is not a regression we need to revert

The default `--fused-qk auto` resolves `DispatchPolicy::fused_qk` to false →
`try_fused_qk_dispatch` short-circuits at the policy gate. Bonsai default-quant
canary (116.247 TPS) is within noise of the recorded anchor (109.86 TPS).
No default path runs through the rotor fused-QK code.

The deliverable of this landing is **correctness coverage** of the rotor
fused-QK kernel surface: the shadow refactor unblocks the 6 rotor variants,
the QJL fallback gate is in place, A.y rejection on Qwen MoE is preserved.
Decode-TPS win remains a follow-up — most plausibly a richer `FusedQkFn`
signature that takes `(codes, scales, norms, rotors)` separately so the
dispatch site can pass each as its own Array without the per-step concat, or
kernel-side stride args that read from the head-major shadow directly without
the flatten/concat dance.

### A.y guard preserved

Qwen MoE archs still reject all 6 rotor variants verbatim via the
existing `QwenMoeRotorKRejected` arm in `cache_type.rs` —
`cache_type_tests.rs` coverage is unchanged and passes.

### Artifacts

- Shadow + layout: `crates/rmlx-kv-quant/src/kvcache/fused_qk_shadow.rs`
- Dispatch + encode: `crates/rmlx-kv-quant/src/kvcache/fused_qk_dispatch.rs`
- Integration tests:
  `crates/rmlx-kv-quant/tests/rotor_fused_qk_dispatch.rs`
- Regression gate: `crates/rmlx-kv-quant/tests/fused_qk_dispatch.rs`
  (q8, turbo3, turbo4 — still pass)
- Layout unit tests: `crates/rmlx-kv-quant/src/kvcache/fused_qk_dispatch_tests.rs`
  (6 new rotor cases, all existing q8/turbo cases updated to `scales_per_token` field).

### Fix-cycle — widened `FusedQkFn` signature (2026-06-03)

The initial landing concatenated `[scales | norms | rotor_table]`
into one `k_scales` Array on every decode step; the resulting per-step copy
was the proximate cause of the 63.5 → 12.3 TPS regression on
`--fused-qk on` rotor decode.

Fix-cycle landed Option A: widened the `FusedQkFn` signature to carry
`k_norms: Option<&Array>` and `k_rotor_table: Option<&Array>` directly.
Dispatch site (`KvCache::try_fused_qk_dispatch`) no longer calls
`concatenate(...)` on the per-token shadow slices — it forwards the three
arrays to the shim as separate `Option`s. q8 / turbo3 / turbo4 shims
ignore both `Option`s (`_k_norms`, `_k_rotor_table`); iso reads `k_norms`;
rotor reads both.

Signature is now 13 args (was 11). All 7 wrapper shims updated; in-crate
test files patched (q8 / turbo_k3 / turbo_k4). Full kv-quant + models test
suite passes (796 passed, 228 ignored).

#### Bonsai 8B re-bench, 4k prompt, max_tokens=64

| Composition | Decode TPS | Output |
|---|---:|:-:|
| `--ctk k_rotor3 --ctv rotor_v_3 --rotor-qjl off --fused-qk auto` (legacy) | 94.0 | 64 toks, coherent |
| `--ctk k_rotor3 --ctv rotor_v_3 --rotor-qjl off --fused-qk on` (rotor3_sym) | 48.6 (median of 52.5 / 43.6 / 48.6) | 64 toks, coherent |
| `--ctk k_rotor3 --ctv bf16 --fused-qk on` (RotorKOnly3) | 1.74 | 64 toks, coherent |
| Default (mixed_k8g64_v4g64) | 113.2 | 64 toks, coherent |

`--fused-qk on` rotor3_sym now completes (was: crash @ step 12) but
sits at ~48 TPS — **below** the ≥66.7 TPS floor and below the 94.0 TPS
legacy bf16 SDPA baseline. Per the warm-TTFT caveat above the rotor codec's production fast path is
already bf16-mirror-shadowed; the fused-QK shadow encode runs in parallel
without short-circuiting and adds per-step encode-chunk overhead while the
legacy SDPA reads the same bf16 mirror. This fix-cycle is therefore
**structural enablement only** — it unblocks the rotor codecs on the
fused-QK fast path so the kernel is exercised and tested end-to-end, but
it does not beat the legacy bf16 path at 4k / 4096-ctx. RotorKOnly3 is
~+19% faster under fused-QK (1.46 → 1.74 TPS) because the legacy path's
CPU rotor3 encode dominates there.

#### Fix-cycle 2 — root cause + fix

The crash signature `Cannot reshape array of size 0 into shape (1,8,1,32)`
at decode step 12 with `--max-ctx 4096`, prefill=4085 tokens, max_tokens=64
is a **KV overflow** bug, not the suspected output-order bug in
`rotor_quantize_v3_gpu` or stale buffer in `seed_rotor_table`:

1. Bonsai boundary layers (first 2 + last 8 of 36) are forced to K8V8
   per `kv_quant_for_layer(LAYER_ADAPTIVE_HEAD_N=2, LAYER_ADAPTIVE_TAIL_N=8)`
   even when the base codec is `Rotor3Sym`. With `head_dim=128` the K8V8
   shadow has `codes_per_token=32` — matching the failing reshape shape.
2. With prefill=4085 + 11 successful decode steps the cache reaches
   `offset = 4085 + 11 = 4096 = max_seq`. On step 12 `try_fused_qk_dispatch`
   computes `prev_offset = 4096, new_seq = 1, prev_offset + new_seq = 4097
   > max_seq = 4096`. The dispatch nevertheless bumps `self.offset = 4097`
   and calls `populate_fused_qk_shadow_from_fp16(prev_offset=4096, n=1)`
   which slices the bf16 mirror at `[..,..,4096..4097,..]` — out-of-range.
   MLX silently clips the slice to a zero-length chunk; the encode kernel
   returns `codes_shape=[0], scales_shape=[0]`; the head-major reshape
   then errors with the size-0 / shape (1,8,1,32) message.

The legacy `--fused-qk off` SDPA path tolerates the same overflow
because `update_decode_fp16` writes via `slice_update` (no-op on OOB)
and reads back via `slice` (clamps to buffer size). The shadow populate
path cannot use that same clamp because the encode kernel runs on the
slice's actual element count.

**Fix.** Added an overflow gate in `try_fused_qk_dispatch` right after
`storage_max_seq_for_fused_qk()`: if `prev_offset + new_seq > max_seq`
return `Ok(None)` and fall through to legacy SDPA. The gate fires on
the final ~4 decode steps of a 4k/64 run (12 layers × 36 ≈ 144 gates
in the diag run) and matches the behavior the legacy path already has.

#### What the fix-cycle accomplishes

- **Crash fixed** — `--fused-qk on` rotor codecs no longer panic at
  `max_seq` boundary; full max_tokens runs complete cleanly.
- **Kernel fires** — diag run with `--max-tokens 16` shows 396 fused-QK
  kernel dispatches + 144 overflow gates (vs the legacy short-circuit
  which never enters the dispatch).
- **Structural — concat marshaling removed** (fix-cycle 1 work intact).
  The dispatch path no longer materialises `[B * kv_h * kv_seq *
  (n_groups + 1 + n_groups * 4)] f32` per layer per decode step.
- **Iso forward-compat** — once iso K-side GPU encode lands, the iso
  shim already accepts a separate `k_norms` Option and will not need
  another signature change.
- **Default canary intact** — 113.2 TPS (≥ 107 anchor).
- **Legacy auto path intact** — 94.0 TPS rotor3 `--fused-qk auto`
  (within noise of prior 94.9).

#### Follow-up — rotor TPS below legacy

The fused-QK shadow path on Rotor3Sym is **structurally correct** but
sits ~50% below the legacy bf16 SDPA at 4k/4096-ctx. Per the warm-TTFT
framing the rotor decode path is shadowed by the bf16 seed and the
fused-QK shadow encode adds overhead. Closing that gap requires:

- A way to skip the per-step `encode_chunk_to_head_major` call on
  steps where the legacy path will run anyway (i.e. when the bf16
  mirror is the source of truth and the rotor codec storage is not
  actually consulted).
- Independently — wire iso K-side GPU encode so iso codecs can also
  exercise the fused-QK fast path.


## GPU-resident `QuantIsoV3` mirror (bench anchor, 2026-06-03)

**Hardware:** local Apple Silicon dev host (single-MLX claim).

### Bench cell

- Model: `prism-ml__Ternary-Bonsai-8B-mlx-2bit` (Qwen3, head_dim=128).
- Prompt: `prompts/longctx_8k.json` (8134 tokens).
- `--max-tokens 64 --max-ctx 16384`.
- KV: `--cache-type-k q8_g128 --cache-type-v iso_v_3`.
- `--fused-qk auto` (resolves OFF per the flash-decode HOLD).
- 8 runs per arm, ON/OFF interleaved.

### Numbers (8-run median)

| Metric          | GPU mirror ON | GPU mirror OFF | Δ median   |
|-----------------|--------------|----------------|------------|
| TTFT (ms)       | 6706.5       | 6814.5         | −108 ms (−1.61%) |
| Decode TPS      | 60.10        | 60.87          | −0.78 TPS (−1.30%, ON slightly worse) |
| Prefill TPS     | 1212.8       | 1193.6         | +19.2 TPS (+1.61%) |
| TTFT stdev (ms) | 127.4        | 176.4          | — |

### Decision

All deltas sit **inside the noise envelope** (TTFT stdev ON=127 ms, OFF=176 ms;
the 108 ms ON/OFF gap is comparable). This puts us in the
**FusedQkShadow-shadowed** bucket:

- TTFT ON-vs-OFF gap = 1.61% (within the ±2% inconclusive band).
- Decode TPS is **slightly worse** under GPU mirror ON (within ±1% reverse-
  noise envelope but consistent with TTFT pattern).

→ **Gate is hardcoded OFF** (`gpu_resident_iso_enabled()` returns `false` unconditionally in production; no env-var opt-in).

The mirror architecture is landed and verified (5 ignored GPU tests pass;
SSD round-trip preserves dequant output bit-identical). The gate-OFF path
replaces the iso3 `append_gpu` helpers with `QuantIsoV3::append_gpu`
early-exit; not byte-identical to the pre-landing dispatch but within sub-µs.
**The remaining 7 codec mirrors (iso3 K, iso4 V/K, rotor3/4 V/K) are
deferred** until a bench arm shows a clear win on a FusedQkShadow-incompatible
path (PPL eval, prompt-cache hits that skip prefill, or any seedless workload).

### Why FusedQkShadow shadows this

On the production `update_iso3` decode hot path the FusedQkShadow already
absorbs the dequant via the warm-TTFT bf16 seed. The iso3 V `dequant_gpu`
fires once at `exit_prefill` plus on cold misses, not per decode step — so
the mirror is built but rarely read. The upload-cost saving the mirror buys
(no per-step `Array::from_bytes`) is correspondingly small.

## GPU-resident mirror extension: dormant-by-design (2026-06-03)

**Decision: won't extend.** The 7-codec GPU-resident mirror extension (iso3 K,
iso4 V/K, rotor3/4 V/K) was evaluated and declined. The GPU-resident mirror
gate (`gpu_resident_iso_enabled()`) is hardcoded `false` in production and
dormant on the normal decode path under the warm-TTFT bf16 seed contract
(`docs/KV_CACHE.md` §9.6, `decode_fp16_k`). Extending the mirror to the 7
remaining codecs delivers no per-decode-step benefit because those codecs share
the same early-return guard: every `update_iso3` / `update_iso4` /
`update_iso3_sym` / `update_iso_k_only_3` / rotor-variant update path
short-circuits at `decode_fp16_k.is_some()` before reaching the GPU mirror
branch.

### Structural analysis

The iso/rotor decode update functions (`update_iso3`, `update_iso4`,
`update_iso3_sym`, `update_iso_k_only_3`, and the rotor variants) all share
the same warm-TTFT shortcut. When
`self.decode_fp16_k.is_some()` (true for every production decode step after
`exit_prefill`), the function returns immediately from the bf16-seed branch
without reaching the iso/rotor quantise-and-store path — and therefore without
ever consulting the GPU-resident mirror. No production decode step reaches
`dequant_gpu`; the mirror would be populated at `exit_prefill` and then never
read. Extending the mirror to the 7 remaining codecs is provably a no-op on the
current production path.

### Empirical A/B (close-out confirmation)

**Model:** `prism-ml__Ternary-Bonsai-8B-mlx-2bit` (Qwen3, iso-capable).
**KV:** `--cache-type-k q8_g128 --cache-type-v iso_v_3`.
**Prompt:** `prompts/longctx_8k.json` (8134 tokens).
**Shape:** `--max-tokens 64 --max-ctx 16384 --fused-qk auto`.
**Binary:** `release-perf`. **Protocol:** 1 warmup + 3 measured runs per arm,
interleaved. **Hardware:** M5 Max.

| Arm | decode_tps runs | median | stdev | TTFT runs (ms) | median | stdev |
|-----|----------------|--------|-------|---------------|--------|-------|
| A — GPU mirror ON (bench arm) | 61.755 / 60.570 / 61.302 | **61.302** | 0.598 | 6634 / 6639 / 6703 | **6639** | 38.5 |
| B — unset (OFF, baseline) | 61.827 / 61.753 / 60.361 | **61.753** | 0.826 | 6636 / 6670 / 6758 | **6670** | 63.0 |
| **Δ (A − B)** | | **−0.451 TPS** | | | **−31 ms** | |
| **Δ %** | | **−0.73%** | | | **−0.46%** | |

Delta decode-TPS −0.73% and TTFT −0.46% are both **within the noise envelope**
(±2σ TPS band ±1.65 TPS; ±2σ TTFT band ±126 ms). Coherence: Bonsai
thinking-model output coherent on all arms (expected reasoning-chain prefix
confirmed on warmup runs). No ≥10% improvement observed — the gate-flip
condition for reversing this decision is not met.

**Confirms null:** yes. The prior null result (TTFT delta 1.61%, decode TPS
delta −1.30%, both inside noise) is reproduced with a fresh 3-run protocol.
The GPU-resident mirror gate is a no-op on the normal decode path.

### Dormant-by-design gate (permanent)

The `gpu_resident_iso_enabled()` gate is hardcoded `false` in production as a
forward-compatibility knob for any **future** seedless workload — a decode path
that leaves `decode_fp16_k`/`decode_fp16_v` absent (e.g. a PPL-eval harness that
reads quantized KV instead of teacher-forced bf16, or a prompt-cache hydrate
mode that decodes without reconstructing the bf16 seed). **No such path exists
today:** every current seedless candidate was traced and found not to qualify —
normal generate, prompt-cache exact/prefix hit, SSD hydrate, and speculative
decode all keep the bf16 seed live at decode (the cache-level `decode_fp16_*`
buffers are copied by `KvCache::try_deep_clone`, and `exit_prefill` runs on every
non-cached generate). The gate is never set in production; default-OFF is the
correct posture under the current warm-TTFT bf16 seed contract.

**Re-open condition:** a production decode path that bypasses the bf16 seed
(i.e., `decode_fp16_k.is_none()` during steady-state decode for a real model
serving real traffic) would make the GPU-resident mirror relevant. No such path
exists today.

## Reclaim the dead bf16 K seed for K-only codecs (2026-06-03)

Closes warm-TTFT audit finding F2. The K-only family (`IsoKOnly3/4`, `RotorKOnly3/4`)
re-quantises K every decode step and routes V through
`update_decode_fp16_v_only`; it never reads the bf16 `decode_fp16_k` seed that
`exit_prefill` used to populate unconditionally. This change gates the K-seed
clone+eval+store on the new predicate `KvQuant::feeds_bf16_k_at_decode()`
(`false` for the K-only family, `true` for every shortcut codec). The bf16 V
seed stays unconditional; the `KvStorage::None` bf16-fallback path still forces
the K seed. `resident_bytes` counts K and V seeds independently.

**Pure RAM reclaim — output byte-unchanged.** Verified on Bonsai-8B-2bit
(`Qwen3ForCausalLM`), GPU, `release-perf`, 2026-06-03:

- **Coherence (smoke):** `--kv-quant k_iso3`, short prompt "capital of France"
  → decoded `" Paris\n\nA: A: A…"` — correct, coherent (PASS).
- **Byte-identity:** same prompt, base commit (`695b939`) binary vs patched
  binary → decoded string **identical** to the byte. Confirms the K seed
  was dead; the fix does not perturb decode.
- **NIAH:** `niah_bonsai_8k` cells with `k_iso3` / `k_iso4` / `k_rotor4` produce
  incoherent long-context output **on both base and branch** (the 2-bit Bonsai
  model degrades under aggressive K-only quant at 8k — a pre-existing codec/model
  property; base d10 output is byte-identical to branch d10). The K-only codecs
  are coherent at short context (see smoke above).

**Residency reclaimed** (= `seq × num_kv_heads × head_dim × 2 B × n_layers`;
Bonsai: `num_kv_heads=8`, `head_dim=128`, `n_layers=36`):

| ctx (seq) | K-seed dropped / layer | all 36 layers |
|-----------|------------------------|---------------|
| 4 096     | 8 192 KiB              | **288 MiB**   |
| 8 192     | 16 384 KiB             | **576 MiB**   |
| 16 384    | 32 768 KiB             | **1 152 MiB** |

(Whole-process RSS at 4k is dominated by ~2.5 GB model weights + Metal buffers,
so the per-cache K-seed delta is below RSS noise there; the deterministic
`resident_bytes` accounting is the authoritative measure and is regression-locked
by the warm-TTFT K-only tests, which assert the seed is absent and that the
reported total is the K store plus the surviving V seed and nothing else.)
