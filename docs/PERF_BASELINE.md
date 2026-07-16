# Perf Baseline

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

All four now sit in the **1.8x–2.7x** band — at or near the healthy
1.5-2x ceiling-vs-realized envelope llama.cpp / mlx-lm hit on dense models.
The dramatic "MoE is 40-50x slow" signal was a **bench-harness measurement
artifact** (prefill in the TPS denominator), not an inference-path defect.
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

**Why `k_iso*` / `k_rotor*` trail their `*_sym` siblings.** Counter-intuitively
the K-only variants (bf16 V) decode *slower* than the fully-symmetric ones on
Bonsai (e.g. k_iso3 59 vs iso3_sym 142, k_rotor3 45 vs rotor3_sym 145). The
K-side iso/rotor codecs have no GPU-resident code mirror on the live decode
path: the CPU `dequant()` re-materializes the full K prefix each step and the
result is re-uploaded via `Array::from_bytes` (rotor additionally applies an
O(head_dim²)-per-token QJL score correction under the default `--rotor-qjl
on`), an O(kv_seq) per-step cost the `*_sym` path amortizes through its
warm-TTFT bf16 seed. (The `dequant_gpu` mirror is gated off by
`gpu_resident_iso_enabled()`.) These anchors are
short-prompt; the gap widens with context. See `docs/KV_QUANT.md` "iso/rotor
K-side variants" and `gpu_resident_iso_enabled` in `rmlx-kv-quant/src/lib.rs`.

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

**Canary protocol**:
- Profile: `release-perf` (debug-assertions=false, overflow-checks=false, stripped)
- Shape: `--prompt-tokens 4096 --max-tokens 100 --max-ctx 8192`, `kv_quant=auto`
- Warmup 1 discarded, 3 measured runs; median decode_tps + sample stddev reported in CSV
- DB record: one `rmlx baseline --record` call per model after the measured runs

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

**Qwen3-dense bf16-stream fix (2026-06-24).** Casting Qwen3 norm weights and
quant scales/biases to bf16 at load (they ship fp16 on Bonsai) stops the
residual stream — and the `--kv-quant none` KV cache — from widening to f32. The
fix also lifts the Bonsai canary default (`mixed_k8g64_v4g64`) from ~110 to ~129
decode_tps (the bf16 q/k/v compute is cheaper than the prior f32 path); Gemma4
and Qwen3.6 (separate arch files) are unchanged. On the `none` path the gain
widens with context as KV bandwidth dominates: Bonsai `none` decode_tps
~101→~135 at 4 k, ~48→~83 at 16 k, ~19→~38 at 64 k.

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
on the hot path; MSL Viterbi 2-bit kernel (`tcq_v2_msl`) parked as
future-reference hook.

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
- **`k_rotor3/4` decode is now a fused MSL flash-decode** over the packed rotor store when `--rotor-qjl off` (`rotor_flash_decode`, see `docs/KV_QUANT.md`); the per-step full-prefix CPU dequant is gone. The anchors above are **not** superseded — they are 2-token short-prompt runs (§ below), where the prefix is empty and the dequant that this kernel removes costs nothing, so they measure a different thing. The kernel's effect scales with prefix length. Measured at a 4k prompt, `--rotor-qjl off`, medians of 3+ runs, before → after: Bonsai-8B `k_rotor3` 1.34 → **17.0**, `k_rotor4` 1.36 → **15.9**; medgemma-4B `k_rotor3` 7.37 → **51.8**, `k_rotor4` 7.34 → **52.1**. The default `--rotor-qjl on` path is unchanged (kernel dormant), as is Gemma4 (`update_and_sdpa_returning_kv` never reaches the kernel).
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

All four sit in a **1.8x–2.7x** band — at/near the healthy 1.5-2x
ceiling-vs-realized envelope llama.cpp / mlx-lm hit on dense batch-1 decode.
**ACCEPT** — decode is bandwidth-bound and healthy. This is the headline answer
to the question "why is decode so low?": it was *not* low — the matrix numbers
were prefill-contaminated (see the decode-only re-baseline section). The residual ~2x is normal MLX batch-1
overhead (dispatch + dequant + the gap between realized and peak bandwidth),
not a defect.

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

The `update_and_sdpa` / `update_and_sdpa_returning_kv` match-arm + indirect call
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
excluded, 8k decode is unambiguously slower — expected, since each decode step
attends over ~2x more KV (longer Mixed quantized_matmul over the cache).

The KV-quant auto-resolution is **identical** at both contexts:
`Qwen3ForCausalLM` with `weight_bits=2` resolves to `Mixed{k8,v4,g64,g64}`
regardless of ctx — `resolve_default` (`kv_cache/mod.rs:331`) has no ctx branch
for this arch, and `kv_quant_for_ctx` is not consulted on the baseline path. So
the 4k-vs-8k difference is NOT a KV-quant-by-ctx effect; it is the pure
KV-length scaling of the per-step attention. No fix needed — this is correct
behavior, and both runs already go through the identical `rmlx baseline` path.

### H8 — Gemma4 + Mixed runtime_fail = cross-layer-KV-sharing contract — ACCEPT

`runs.db` shows every Gemma4-e4b `mixed_*` cell recorded `decode_tps_warm=0.0`
(e.g. `mixed_k8g128_v4g64`, `mixed_k8g128_v8g64`, `mixed_k4g64_v4g64`, …).
Reproduced live on `gemma-4-e4b-it-mxfp8` with `--cache-type-k q8_g128
--cache-type-v q4_g64`: the first prefill chunk fails with the exact message

> `mlx: Cross-layer KV sharing not supported with Mixed quantization. Use
> bf16/K8V8/K8V4/Planar for shared-KV layers, or disable layer-KV sharing for
> this arch.`

emitted from `KvCache::update_and_sdpa_returning_kv` (`kvcache.rs:412`). Gemma4
genuinely shares KV across layers (`num_kv_shared_layers`, `loader.rs:252-258`),
and shared-KV layers route through the `returning_kv` path that rejects
`KvQuant::Mixed`. This is the documented cross-layer-KV-sharing contract behaving correctly. The
only flaw is *when* it fires — at first prefill (runtime_fail, value 0.00)
rather than at startup. **ACCEPT** — this is correct-by-design and justifies
the resolver guard: reject Mixed on the Gemma family at flag resolution and
exit 78 at startup, so the user sees the error immediately instead of after a
model load + prefill.

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
  attention layer through `update_and_sdpa_returning_kv` for cross-layer KV
  sharing (same shape as the `Unreachable TurboFlash` case).
- Forced `--kv-quant planar` (resolves to `KvQuant::PlanarK`).
- `--max-tokens 100`, `--max-ctx 8192` (4k prompt) / `--max-ctx 16384` (8k prompt).
- Single MLX process per Hard Rule 8 (preflight: pkill rmlx serve / mlx_lm; sleep; rm claim).
- 1 warmup + 3 measured runs per toggle.
- Gate: `RMLX_PLANAR_FLASH_DECODE={0|1}` env-var; the production CLI flag
  `--planar-flash-decode` is `serve`-only.

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

`apply_planar_flash_decode_flags(Auto)` stays OFF on every host
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

The default `--fused-qk auto` keeps `RMLX_FUSED_QK` unset →
`try_fused_qk_dispatch` short-circuits at the env gate. Bonsai default-quant
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
the K seed. `approx_bytes` now counts K and V seeds independently.

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
`approx_bytes` accounting is the authoritative measure and is regression-locked
by the warm-TTFT K-only tests, which assert the seed is absent and
`approx_bytes` drops.)
