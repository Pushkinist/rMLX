# SSD Canary

`scripts/ssd_canary.sh` drives a live `rmlx serve` through three server
processes. It checks that a new process serves prompts from blocks an
earlier process spilled to SSD, and that startup eviction holds the budget.

## Running it

```bash
make build-perf
VERIFIER_MODEL=/path/to/snapshot bash scripts/ssd_canary.sh \
  [--port 62265] [--ssd-gb 100] [--dry-run]
```

`make ssd-canary` builds `release-perf` and runs the script with
`--ssd-gb ${SSD_GB:-100}`. The script uses
`target/release-perf/rmlx` and exits 125 when that binary is missing.

| Flag or variable | Default | Meaning |
|---|---|---|
| `VERIFIER_MODEL` | required | Snapshot directory to serve |
| `--port`, `PORT` | `62265` | Server port |
| `--ssd-gb`, `SSD_GB` | `100` | `--kv-ssd-cache-gb` for POPULATE and REVISIT |
| `--tag` | — | Parsed and never read; it changes nothing |
| `--dry-run` | off | Keeps the data root; skips the ingest and the `events` and `observations` checks |
| `RMLX_HOME` | `.rmlx/proofs/step3-canary/` | Data root of every server process |
| `RMLX_HARDWARE_TAG` | script default | Hardware label of the `runs.db` rows |

The script deletes its data root whole before the run, except under
`--dry-run`. An `RMLX_HOME` exported in the shell is that data root, so an
exported `RMLX_HOME=$PWD/.rmlx` loses `metrics/runs.db`, `metrics/backups/`,
`cache/` and `logs/`. Unset `RMLX_HOME` before the run.

Before each phase the script kills every `rmlx serve`, `mlx_lm`, `paroquant`
and `omlx` process and removes every `/tmp/rmlx.*.claim` file. The make
target first kills every `rmlx serve` and `mlx_lm` process and removes every
claim file.

## Phases

Every server runs with `--prompt-cache-slots 4`, `--project ssd-canary`
and `--log info`. Every request is non-streaming, `max_tokens` 64,
temperature 0, seed 42. Hits are read from `ssd_hits` in `/metrics/cache`,
as the change since the previous request.

1. **POPULATE.** One server sends all 20 prompts in `prompts/ssd_bench/`, in
   sorted order.
2. **REVISIT.** A new server sends prompts 0, 2, …, 18 of the same list.
   Its RAM cache starts empty, so every hit is a block POPULATE spilled.
3. **EVICT.** A new server starts with `--kv-ssd-cache-gb` set to four times
   the mean block size in the POPULATE index, at least 1 MiB. With no
   POPULATE block, the budget is 0.05 GB. The script reads the namespace
   index `<RMLX_HOME>/cache/kv/ssd-canary/index.db` before the first request,
   then sends the first 8 prompts.

In REVISIT and EVICT, two failed requests in a row restart the phase's
server.

## Checks

The run exits 1 when any FAIL check fails. WARN checks only print a note.

| Check | Level |
|---|---|
| REVISIT served at least one SSD hit | FAIL |
| `ssd_evict_total` is above 0 after EVICT startup (or at the end of EVICT) | FAIL |
| The EVICT index holds no more bytes than the budget right after startup | FAIL |
| `events` has at least one `ssd_spill` row and one `ssd_hydrate` row | WARN |
| `observations` has at least one row per phase tag | WARN |
| `ssd_bytes_used` after POPULATE is above 0 | WARN |

Each phase files one `RunRecord` through `rmlx metrics record`, tagged
`ssd-canary-populate`, `ssd-canary-revisit` or `ssd-canary-evict` whatever
`--tag` says. The POPULATE and REVISIT records carry SSD hits, bytes used,
evictions, and mean spill and hydrate time and rate. The EVICT record carries
bytes used and evictions only.

## Output

Under the data root:

- `phase_populate.csv`, `phase_revisit.csv`, `phase_evict.csv`: one row per
  request.
- `iteration_summary.json`: phase totals and the check results.
- `metrics/runs.db`: the `events` rows and the three observations.

## The regression gate

`make ssd-canary-gate SHA=<sha>` runs `rmlx metrics deltas --since-sha <sha>
--exit-code true` on `CANARY_DB`, at `CANARY_THRESHOLD_PCT` (default 3). It
exits 125 without `SHA=` or without the DB.

`CANARY_DB` defaults to `$RMLX_HOME/metrics/runs.db`, with `RMLX_HOME`
defaulting to `.rmlx`. With `RMLX_HOME` unset, that is not the DB the canary
writes. Either way the canary's DB holds one run, since the script deletes its
data root first.

`docs/METRICS_DB.md` describes `runs.db`. The cross-restart integration test,
one spill and one hydrate, is `crates/rmlx-server/tests/ssd_cache_restart.rs`.
