<!-- GENERATED FILE — do not hand-edit. Run: rmlx metrics export --markdown > BENCHMARK_CHAMPIONS.md -->
# BENCHMARK_CHAMPIONS — best observed metrics per (model, backend, quant)

Each cell holds the BEST observed metric across all runs, as the `bests` view picks it: the highest decode_tps and prefill_tps, the lowest TTFT and peak_rss. `observations` is append-only.

## Methodology

- **Context and hardware**: each observation records its own prompt, context and `hardware_tag`; read them in `runs.db`. One MLX process runs at a time.
- **Metric units**: TPS = tokens/second; TTFT = milliseconds; RSS = megabytes.
- **Cells**:
  - Numeric value = best observed metric.
  - `x` = backend supports the (model, quant) but produces incorrect output (gibberish, empty, GPU timeout).
  - `N/A` = backend does not support this (model, quant) combination structurally (e.g. ollama doesn't have z-lab/PARO checkpoints).
  - `-` = not yet measured.
- **Source-of-truth**: `<RMLX_HOME>/metrics/runs.db` (SQLite).

## Update protocol

1. Run bench. Capture decode_tps_warm, prefill_tps, ttft_cold, ttft_warm, peak_rss for each (model × backend × weight_quant × kv_quant) cell.
2. Recorder appends to `observations`; the `bests` VIEW picks the champion per cell.
3. Regenerate this file: `rmlx metrics export --markdown --scope config/scope.toml > BENCHMARK_CHAMPIONS.md` (`make metrics-export`).

## Backends

| Code | Path / Description |
|---|---|
| `rmlx` | `<RMLX_ROOT>/target/release/rmlx serve` |
| `mlx-lm` | `<mlx-lm>/.venv/bin/python -m mlx_lm.server` |
| `mlx-lm-tq` | `<mlx-lm-turboquant>/.venv/bin/python -m mlx_lm.server` |
| `oMLX` | `<oMLX>/...` |
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

---
