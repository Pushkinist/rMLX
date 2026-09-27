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
`target/release-perf/rmlx` and exits 125, before anything else, when that
binary is missing.

| Flag or variable | Default | Meaning |
|---|---|---|
| `VERIFIER_MODEL` | required | Snapshot directory to serve |
| `--port`, `PORT` | `62265` | Server port |
| `--ssd-gb`, `SSD_GB` | `100` | `--kv-ssd-cache-gb` for POPULATE and REVISIT |
| `--dry-run` | off | Skips the ingest and the `events` and `observations` checks |
| `RMLX_HOME` | the checkout's `.rmlx/` | Data root: holds the run directory, and its `metrics/runs.db` is the default metrics DB. A relative value is ignored, as by every `rmlx` command |
| `RMLX_METRICS_DB` | `<data root>/metrics/runs.db` | Metrics DB the records go to, as for every `rmlx metrics` command |

The script does not work out either path itself: it asks the binary, from the
checkout, with `rmlx metrics path --home` and `rmlx metrics path`, so the data
root and the DB are the ones every `rmlx` command in the checkout resolves.

Each run creates a new directory,
`<data root>/proofs/ssd-canary-<UTC stamp>.<random>/`, and every phase server
runs with its absolute path as `RMLX_HOME`: POPULATE starts from an empty SSD
tier, and the servers' logs, `events` rows and SSD blocks stay in it. At exit,
however the run ends, the script deletes that directory's `cache/kv/`, the SSD
blocks, and nothing else: the CSVs, the summary, the logs and the `events` DB
are kept. Nothing already in the data root is removed; outside the run
directory, the one write is `rmlx metrics record` appending the three records
to the metrics DB.

Each phase's server takes the Metal claim, and the script stops only the
server it started. When another process holds the claim, the phase server
exits 11 and names the holder, and the canary stops with that status.

## Phases

Every server runs with `--prompt-cache-slots 4`, `--max-ctx 8192`,
`--project ssd-canary` and `--log info`, and the default KV codec. Every request is non-streaming, `max_tokens` 64,
temperature 0, seed 42. Hits are read from `ssd_hits` in `/metrics/cache`,
as the change since the previous request.

1. **POPULATE.** One server sends all 20 prompts in `prompts/ssd_bench/`, in
   sorted order.
2. **REVISIT.** A new server sends prompts 0, 2, …, 18 of the same list.
   Its RAM cache starts empty, so every hit is a block POPULATE spilled.
3. **EVICT.** A new server starts with `--kv-ssd-cache-gb` set to four times
   the mean block size in the POPULATE index, at least 1 MiB. With no
   POPULATE block, the budget is 0.05 GB. The script reads the namespace
   index `<run directory>/cache/kv/ssd-canary/index.db` before the first request,
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
`ssd-canary-populate`, `ssd-canary-revisit` or `ssd-canary-evict`, into the
metrics DB `rmlx metrics path` names. Its `ctx_max` is the 8192 the servers
ran at, and its `kv_quant` is the codec the POPULATE server's log names
(`scripts/lib/server_kv_quant.py`); a log naming none stops the run. The
POPULATE and REVISIT records carry SSD hits, bytes used,
evictions, and mean spill and hydrate time and rate. The EVICT record carries
bytes used and evictions only.

## Output

In the run directory:

- `phase_populate.csv`, `phase_revisit.csv`, `phase_evict.csv`: one row per
  request.
- `iteration_summary.json`: phase totals and the check results.
- `server_<phase>.log`: each phase server's output.
- `metrics/runs.db`: the phase servers' `events` rows.

The three observations go to the metrics DB, beside every earlier run's.

## The regression gate

`make ssd-canary-gate SHA=<sha>` runs `rmlx metrics deltas --since-sha <sha>
--prompt-prefix ssd-canary- --exit-code true` on the DB `rmlx metrics path`
names, at `CANARY_THRESHOLD_PCT` (default 3). That is the DB the canary
writes, so the comparison sees the rows of `<sha>` and of every later run.
`--prompt-prefix` keeps it to the cells whose prompt name starts
`ssd-canary-`, the canary's own; a regression in another bench's cell in the
same DB does not fail it. It exits 125 without `SHA=` or without the DB.

`scripts/ssd_canary_selftest.sh` (`make ssd-canary-selftest`, in `make ci`)
runs `make ssd-canary` and the gate against a stub binary and a stub server,
with `RMLX_HOME` exported, unset, relative, and beside an exported
`RMLX_METRICS_DB`. It checks that a data root survives the run byte for byte,
that each run gets a new absolute directory whose SSD blocks are gone at exit,
that the script removes nothing outside it, that the summary and the records
carry what the stub server wrote into its own data root, and that the gate
reads the DB the records went to, scoped to the canary's prompts.

`docs/METRICS_DB.md` describes `runs.db`. The cross-restart integration test,
one spill and one hydrate, is `crates/rmlx-server/tests/ssd_cache_restart.rs`.
