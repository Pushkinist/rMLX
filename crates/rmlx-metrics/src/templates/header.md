<!-- GENERATED FILE — do not hand-edit. Run: rmlx metrics export --markdown > BENCHMARK_CHAMPIONS.md -->
# BENCHMARK_CHAMPIONS — best observed metrics per (model, backend, quant)

**Append-only highest-record table.** Each cell holds the BEST observed metric across all runs. Update only if new metric is **strictly higher** (decode_tps, prefill_tps) or **strictly lower** (TTFT, peak_rss). Even +0.1% wins update the cell.

**Never delete rows.** Never overwrite with worse numbers.

## Scope (2026-05-09)

**In scope**: Gemma + Qwen + Bonsai families only. 9 models.

**Out of scope (excluded by user direction 2026-05-09)**:
- `mlx-community__Laguna-XS.2-mxfp8` (Laguna arch — out-of-tree custom arch)
- `mlx-community__DR-Venus-4B-RL-mlx-8Bit` (Qwen3 dense, redundant with Bonsai/medgemma)

These models are dropped from the table, regression smokes, and future optimization tasks (#126-#133).

## Methodology

- **Context**: 4096-token prompt (`Cross-Backend-Bench/prompts/longctx_4k.json`), 8192 max-ctx server config, 32 max_tokens decode (server cap respected), 2 runs per cell (cold + warm), warm = 2nd run.
- **Hardware**: M5 Max 128 GB, macOS 25.4.0, single MLX process at a time.
- **Metric units**: TPS = tokens/second; TTFT = milliseconds; RSS = megabytes.
- **Cells**:
  - Numeric value = best observed metric.
  - `x` = backend supports the (model, quant) but produces incorrect output (gibberish, empty, GPU timeout).
  - `N/A` = backend does not support this (model, quant) combination structurally (e.g. ollama doesn't have z-lab/PARO checkpoints).
  - `-` = not yet measured.
- **Source-of-truth**: `metrics/runs.db` (SQLite). Cited per update in commit messages.

## Update protocol

1. Run bench. Capture decode_tps_warm, prefill_tps, ttft_cold, ttft_warm, peak_rss for each (model × backend × weight_quant × kv_quant) cell.
2. Recorder appends to `observations`; the `bests` VIEW picks the champion per cell.
3. Regenerate this file: `rmlx metrics export --markdown --scope config/scope.toml > BENCHMARK_CHAMPIONS.md`.
4. Commit message format: `bench(records): <model> <backend> <quant> +X% <metric>` if a record was beaten.

## Backends

| Code | Path / Description |
|---|---|
| `rmlx` | `<RMLX_ROOT>/target/release/rmlx serve` |
| `mlx-lm` | `<mlx-lm>/.venv/bin/python -m mlx_lm.server` |
| `mlx-lm-tq` | `<mlx-lm-turboquant>/.venv/bin/python -m mlx_lm.server` |
| `oMLX` | `<oMLX>/...` (api_key=1234) |
| `ollama` | `ollama serve` (system app) |
| `paroquant` | `<paroquant>/.venv/bin/python -m paroquant.cli.serve` (venv inside the cloned repo) |

## Quant families

| Code | Description |
|---|---|
| `affine 8b` | Per-group int8 affine, group_size=64 |
| `affine 4b` | Per-group int4 affine, group_size=64 |
| `2-bit ternary` | TheStage-style ternary 2-bit |
| `mxfp8 g32` | Microscaling FP8, group_size=32 |
| `paroquant int4` | Z-Lab pairwise-rotation INT4 (z-lab/*-PARO) |
| `kv-k8v4` | KV-cache: K=q8_0, V=TurboQuant 4-bit |
| `kv-k8v8` | KV-cache: K=q8_0, V=q8_0 |
| `kv-planar` | KV-cache: K=q8_0, V=PlanarQuant 4-bit |

---
