# SSD Canary

`scripts/ssd_canary.sh` is an end-to-end long-session harness that proves three properties of the SSD prompt-cache tier on a live rMLX server:

1. **SSD tier serves repeated cold-equivalent prompts** — hit rate climbs as the harness revisits prompts that were previously spilled from RAM to SSD.
2. **LRU eviction holds under budget pressure** — flooding the cache past a 50 MB ceiling keeps `SUM(byte_size)` in `kv_blocks` ≤ budget; oldest rows are gone.
3. **All step-2 timing slices fire** — `ssd_spill_ms`, `ssd_hydrate_ms`, per-slice columns inside `events`, `ssd_bytes_used`, and `ssd_evict_total` all populate `runs.db` and `/metrics`.

## Make targets

The canonical way to run the canary and its regression gate is via `make`:

```bash
# Run full canary (POPULATE + REVISIT + EVICT)
VERIFIER_MODEL=/path/to/mlx-community__gemma-4-e2b-it-mxfp8 make ssd-canary

# Gate against a baseline SHA (exits non-zero on regression)
make ssd-canary-gate SHA=b30c842
```

### Env-var table (make targets)

| Env var | Purpose | Default |
|---|---|---|
| `VERIFIER_MODEL` | abs path to MLX snapshot | required |
| `SSD_GB` | budget passed to canary script | 100 |
| `RMLX_HOME` | hermetic data root | `$PWD/.rmlx` |
| `CANARY_DB` | gate-target DB path override | `$RMLX_HOME/metrics/runs.db` |
| `CANARY_THRESHOLD_PCT` | gate threshold pct | 3 |

`ssd-canary` implicitly calls `make build-perf` first and kills any competing
MLX processes before spawning the server. `ssd-canary-gate` requires `SHA=`
on the command line; it exits 125 (skip-worthy) when `runs.db` has no rows.

## How to run (script directly)

```bash
# Resolve the model path from LOCAL.md (gitignored — never embed in scripts).
VERIFIER_MODEL=/path/to/mlx-community__gemma-4-e2b-it-mxfp8 \
  bash scripts/ssd_canary.sh [--port 62265] [--ssd-gb 100] [--tag my-run]
```

The binary must be built first:

```bash
make build-perf
```

### Flags

| Flag | Default | Meaning |
|---|---|---|
| `--port N` | `62265` | HTTP port for `rmlx serve` |
| `--ssd-gb N` | `100` | SSD-tier budget for POPULATE and REVISIT phases |
| `--tag TAG` | `ssd-canary` | Tag prefix for `runs.db` observation rows |
| `--dry-run` | false | Skip server spawn and DB writes; print what would happen |

### Env vars (script)

| Var | Required | Meaning |
|---|---|---|
| `VERIFIER_MODEL` | yes | Absolute path to model snapshot dir (resolve from LOCAL.md) |
| `PORT` | no | Overrides `--port` |
| `RMLX_HOME` | no | Overrides the hermetic proof directory (default `.rmlx/proofs/step3-canary/`) |
| `RMLX_HARDWARE_TAG` | no | Hardware label recorded in `runs.db` rows |

## Phases

### POPULATE

- Sends all 20 canonical prompts from `prompts/ssd_bench/` back-to-back.
- Server starts with `--prompt-cache-slots 4` (small RAM tier) and `--kv-ssd-cache-gb <SSD_GB>`.
- After each request: parses `/metrics` for `rmlx_ssd_bytes_used`, `rmlx_ssd_evict_total`, the spill/hydrate histogram sums, and reads `ssd_hits` from the chat-completion response.
- Appends one row per request to `phase_populate.csv`.

### REVISIT

- Replays a fixed 10-prompt subset (deterministic: indices 0, 2, 4, ..., 18 of the sorted prompt list).
- Same server configuration as POPULATE; same project namespace (`ssd-canary`) so the SSD blocks from POPULATE are visible.
- Expected: `ssd_hits > 0` on prompts whose RAM slot was evicted but whose block survived on SSD.
- Appends to `phase_revisit.csv`.

### EVICT

- Restarts the server with `--kv-ssd-cache-gb 0.05` (50 MB) and `--project ssd-evict-canary`.
- Sends 8 long prompts designed to exceed the budget.
- After each request, queries `<RMLX_HOME>/cache/kv/ssd-evict-canary/index.db` directly:
  ```sql
  SELECT COUNT(*), SUM(byte_size), MIN(last_used), MAX(last_used) FROM kv_blocks;
  ```
- Asserts `SUM(byte_size) ≤ 52428800` bytes (50 MiB).
- Appends to `phase_evict.csv`.

## Success criteria

| Criterion | How verified |
|---|---|
| `events` ≥ 1 SsdSpill rows | `sqlite3 runs.db "SELECT COUNT(*) FROM events WHERE op='ssd_spill';"` |
| `events` ≥ 1 SsdHydrate rows | `sqlite3 runs.db "SELECT COUNT(*) FROM events WHERE op='ssd_hydrate';"` |
| `observations` has 3 tagged rows | `SELECT description FROM observations WHERE ts_utc >= ...` |
| `ssd_bytes_used` after POPULATE > 0 | Checked in validation block of the script |
| `ssd_evict_total` after EVICT > 0 | Checked in validation block |
| Budget not violated in EVICT | `EVICT_FINAL_SUM_BYTES <= EVICT_BUDGET_BYTES`, script exits 1 if violated |

All criteria are printed in the final summary table and recorded in `iteration_summary.json`.

## Output artifacts

```
.rmlx/proofs/step3-canary/
  phase_populate.csv          one row per request (20 rows)
  phase_revisit.csv           one row per request (10 rows)
  phase_evict.csv             one row per request (8 rows)
  iteration_summary.json      full phase aggregates + validation outcome
  metrics/runs.db             observations table (3 tagged sets)
  metrics/buffer/pending/     flushed by ingest step
```

## Related

- `docs/METRICS_DB.md` — schema and operating rules for `runs.db`.
- `scripts/spec_bench.sh` — decode-TPS canary (template this script mirrors).
- `crates/rmlx-server/tests/ssd_cache_restart.rs` — integration-level SSD cache correctness test (spill → restart → hydrate chain).
- `crates/rmlx-metrics/src/events.rs` — `SsdSpillEvent` and `SsdHydrateEvent` payloads.
