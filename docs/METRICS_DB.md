# Metrics Database

The canonical store for benchmark measurements from rMLX and other backends.
It is one SQLite file. The append-only `observations` table is the ground
truth. The `bests` view derives the champion of each cell at read time. Read
§13 before you write to the DB.

---

## 1. Why a DB

Every measurement is one `observations` row with its full run context. A
cell's history, its champion and a comparison across backends are each one
query. `BENCHMARK_CHAMPIONS.md` is generated from `bests` (§9).

---

## 2. Path & cross-repo discipline

- **Path**: `<RMLX_HOME>/metrics/runs.db`, from
  `rmlx_core::paths::metrics_db_path()`. `rmlx metrics` takes `--db <path>`
  first, then `RMLX_METRICS_DB`, then that default, and creates the parent
  directory.
- **Git**: `.rmlx/` and `metrics/` are git-ignored. Never commit the DB.
- **Other repos** reach the same file through `--db`, `RMLX_METRICS_DB` or a
  symlink. There is one DB, not one per repo.
- **Connection**: every open sets `journal_mode=WAL`, `synchronous=NORMAL`,
  `foreign_keys=ON` and `busy_timeout=5000` (`schema::apply_pragmas`).
- **Sub-tree** under `<RMLX_HOME>/metrics/`:
  - `buffer/pending/`: the ingest queue (§8.4).
  - `buffer/failed/`: records that `--replay-pending` rejected.
  - `backups/`: `backup` and `restore` snapshots (§10.1).
  - `legacy/`: read-only archive of files from before the DB. No tool reads
    it.
- Take a snapshot with `rmlx metrics backup` before any bulk operation.

---

## 3. Schema

The migrations under `crates/rmlx-metrics/src/migrations/` are the schema.
`schema::MIGRATIONS` embeds them in the binary. There are three user tables
(`prompts`, `observations`, `events`), one view (`bests`) and one bookkeeping
table (`schema_meta`). `bests` is derived at read time from `observations`.

Two access patterns:

- **Champion view** (`bests`): the best value per cell. `rmlx metrics best`,
  `rank` and `export --markdown` read it.
- **Time series** (`observations`): every value of a cell over time. `rmlx
  metrics history`, `timeseries`, `regress` and `deltas` read it.

Every observation is kept. A value worse than the best is inserted and does
not surface in `bests`.

### 3.0 `schema_meta` (versioning + provenance)

```sql
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
```

The migration-1 post-hook `seed_schema_meta` inserts five rows with
`INSERT OR IGNORE`: `schema_version` = `1`,
`created_utc`, `created_by` = `rmlx-metrics@<semver>`, `hardware_tag` =
`m5_max_128gb` and `default_namespace` = `mlx-community`. No code reads them
back, and no later migration updates `schema_version`.

The schema version is `PRAGMA user_version`. `migrate::run_pending` applies
each migration above it in one transaction, then sets `user_version` to that
migration's number. It then rebuilds `bests` if the view is stale (§3.3).
Every writer runs it before its first write: `schema::open_migrated`,
`EventRecorder`, the server's metrics drainer, `init` and `migrate`.

`rmlx metrics init` refuses a path that exists. `rmlx metrics doctor` applies
pending migrations without `--fix`.

### 3.1 `prompts`

The prompt registry. `observations.prompt_id` references it, so a prompt body
is stored once.

```sql
CREATE TABLE prompts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256          TEXT    NOT NULL UNIQUE,  -- hex SHA-256 of body
    name            TEXT    NOT NULL,         -- human label, not unique
    body            TEXT    NOT NULL,         -- exact prompt text
    tokens_approx   INTEGER,                  -- whitespace-split estimate
    first_seen_utc  TEXT    NOT NULL,         -- ISO-8601 UTC
    notes           TEXT                      -- source, intent
);
CREATE INDEX prompts_name_idx ON prompts(name);
```

| Field            | Meaning |
|------------------|---------|
| `id`             | Surrogate key; `observations.prompt_id` points here. |
| `sha256`         | Content hash and idempotency key. The same body returns the existing id. |
| `name`           | Label. One name can carry several bodies over time. |
| `body`           | Exact prompt bytes. |
| `tokens_approx`  | Whitespace-split count. The tokenizer count is `observations.prompt_tokens`. |
| `first_seen_utc` | When this body was first inserted. |
| `notes`          | Why the prompt exists, and its source. |

### 3.2 `observations` (every measurement, append-only)

One row per metric value, with its full run context.

```sql
CREATE TABLE observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    -- cell identity
    backend          TEXT    NOT NULL,  -- §5.4
    model_namespace  TEXT    NOT NULL,  -- §5.1
    model            TEXT    NOT NULL,
    weight_quant     TEXT    NOT NULL,
    kv_quant         TEXT    NOT NULL,
    ctx_max          INTEGER NOT NULL,
    prompt_id        INTEGER NOT NULL REFERENCES prompts(id),
    metric           TEXT    NOT NULL,  -- §4
    -- value
    value            REAL    NOT NULL,
    unit             TEXT    NOT NULL,
    direction        TEXT    NOT NULL
        CHECK (direction IN ('higher_better', 'lower_better')),
    -- run context
    run_id           TEXT    NOT NULL,
    ts_utc           TEXT    NOT NULL,
    git_sha          TEXT,
    build_profile    TEXT,
    backend_version  TEXT,
    hardware_tag     TEXT    NOT NULL,
    -- bench config
    prompt_tokens    INTEGER,
    max_tokens       INTEGER,
    temperature      REAL,
    seed             INTEGER,
    n_warmups        INTEGER,
    n_measure        INTEGER,
    -- side data
    output_first_64  TEXT,
    decode_stddev    REAL,
    notes            TEXT,
    description      TEXT,
    -- bookkeeping
    inserted_utc     TEXT    NOT NULL,
    inserted_by      TEXT    NOT NULL,
    -- cell identity (migration 005)
    decode_config    TEXT
);

CREATE INDEX obs_cell_idx          ON observations(backend, model_namespace,
    model, weight_quant, kv_quant, ctx_max, prompt_id, metric, decode_config);
CREATE INDEX obs_metric_idx        ON observations(metric);
CREATE INDEX obs_ts_idx            ON observations(ts_utc);
CREATE INDEX obs_git_sha_idx       ON observations(git_sha);
CREATE INDEX obs_run_id_idx        ON observations(run_id);
CREATE INDEX obs_backend_idx       ON observations(backend);
CREATE INDEX obs_inserted_idx      ON observations(inserted_utc);
CREATE INDEX obs_decode_config_idx ON observations(decode_config);
```

`decode_config` is added by `ALTER TABLE`, so it is the last column.

**The cell.** `rmlx_metrics::cell::CELL_COLUMNS` is the cell key: `backend`,
`model_namespace`, `model`, `weight_quant`, `kv_quant`, `ctx_max`,
`prompt_id` and `decode_config`. `bests` and every cell query partition on it
plus `metric`. A cell holds many observations over time, so the primary key
is the surrogate `id`.

**Run grouping.** Every observation from one `record` call shares one
`run_id`. `WHERE run_id = '…'` returns the metrics of that run.

**Field semantics**

| Field             | Meaning |
|-------------------|---------|
| `backend`         | Engine name, lowercase, no version (§5.4). |
| `model_namespace` | Who published the model (§5.1). |
| `model`           | Short name within the namespace, no path. |
| `weight_quant`    | Weight quantization on disk; `bf16` if unquantized (§5.2). |
| `kv_quant`        | KV-cache quantization at run time; `none` if unquantized (§5.3). |
| `ctx_max`         | Server max context. It changes the KV cache shape, so it is part of the cell. |
| `prompt_id`       | The prompt. TPS is not comparable across prompts. |
| `metric`          | Registry name (§4). |
| `decode_config`   | Non-default engine configuration; `NULL` is every setting at its default. Grammar below. |
| `value`           | The number, in the registry unit. |
| `unit`            | Registry unit (§4). The recorder takes it from the registry. |
| `direction`       | `higher_better` or `lower_better`, from the registry. `bests` ranks by it. |
| `run_id`          | `<YYYYMMDDHHMMSS>-<6hex>`, minted by the recorder at write time. A tracking string, not a key. |
| `ts_utc`          | When the measurement was taken (ISO-8601 UTC). |
| `git_sha`         | Caller-supplied provenance (§8.5.1). `NULL` unless a caller set it. `deltas --since-sha` also matches `<sha>-dirty`. |
| `build_profile`   | For `rmlx`, the Cargo profile the binary was built with (`release`, `release-perf`, `release-debug`, `debug`). |
| `backend_version` | Semver only. |
| `hardware_tag`    | Hardware identifier, part of the row context. For `rmlx`, `RMLX_HARDWARE_TAG` or the default `m5_max_128gb`. |
| `prompt_tokens`   | Tokenizer count of the prompt at run time. |
| `max_tokens`      | Cap on generated tokens. |
| `temperature`     | Sampling temperature. |
| `seed`            | Sampler seed. |
| `n_warmups`       | Warm-up runs discarded. |
| `n_measure`       | Measured runs behind `value`. |
| `output_first_64` | Start of the generated text, to compare temp-0 output across builds. |
| `decode_stddev`   | Stddev across the measured runs, when the emitter sends one. |
| `notes`           | Machine-written by the emitter. |
| `description`     | Written by a person or an agent: why the run exists and what changed (§6). |
| `inserted_utc`    | When the row was written. |
| `inserted_by`     | `<tool>@<semver>`, e.g. `rmlx-cli@<semver>`. |

**`decode_config` is cell identity, not context.** It names every engine
setting that a run moved off its default. A drafter arm and a plain arm of one
model are different configurations. Ranking one against the other would
publish the drafter's rate as the model's decode rate. A setting that only
describes the run, such as `ctx_max`, `kv_quant` or the prompt, has its own
column.

**Grammar.** Two spellings of one configuration would be two cells, so the
spelling is a contract:

```
decode_config := term ("," term)*
term          := key "=" value
key           := segment ("/" segment)*
segment       := [a-z0-9_]+
value         := [A-Za-z0-9_.+-]+
```

There is no whitespace. Terms are strictly ordered by key. A `/`-path key
names a setting of a subsystem, as in `mtp/block=5`. `NULL` is the engine at
its defaults. The empty string is refused.

`rmlx_metrics::cell::decode_config_is_well_formed` implements the grammar.
`RunRecord::validate` refuses a record that:

- breaks the grammar;
- spells only default values (`decode_config_is_all_defaults`), which must be
  `NULL`;
- names an adaptive drafter's block without its depth term
  (`decode_config_with_inherent_depth`).

**Terms in use.**

| Terms | Setting | Emitted by |
|---|---|---|
| `<drafter>/block=<n>` | speculative arm and its configured block | `rmlx_metrics::cell::decode_config`, called by the round loop, which logs it on its `done` line |
| `<drafter>/depth=<policy>` | how the loop sizes each round's block, when it resizes | same |
| `prefill_chunk=<n>` | non-default prefill chunk | `scripts/prefill_chunk_sweep.sh` |
| `kv_boundary/head=<h>,kv_boundary/tail=<t>` | `--kv-boundary-layers` off its default | `rmlx baseline --record`, `rmlx eval ppl`, `scripts/ingest/{codec_inertness,perf_ab}_ingest.py` |

`<drafter>` is the `DraftKind::as_str` name: `mtp`, `dflash`, `dflash2`,
`eagle3` or `two_model`.

The depth term is absent when the loop drafts the configured block every
round. `cell::ADAPTIVE_DRAFTERS` lists the drafters that always resize: today
`dflash`, with policy `accept_rate`. No production path runs DFlash at a fixed
block, so its arm is always `dflash/block=<n>,dflash/depth=accept_rate`.
Migration 008 rewrites a stored `dflash/block=<n>` to that form.

`scripts/spec_bench.sh` records the string the engine logged.
`scripts/ingest/published_ingest.py` carries the string its result file
recorded from the `done` line. Neither composes a drafter term.

The `kv_boundary/*` pair is always written together, `head` before `tail`.
The shipped default is `NULL`. The two Python ingesters derive the terms from
the run's own recorded arguments: the probe's `kv_boundary` column and
`perf_ab.sh`'s per-arm `args`. They read the default from
`rmlx_core::kv_boundary` through `scripts/lib/kv_boundary_default.py`;
`make check-kv-boundary-default-parity` gates it.

**What does not belong here.** A setting that changes what the number means
is a different metric, not a term. `rmlx eval ppl` has a cacheless scorer and
a cache-bearing one: they record `ppl_<corpus>` and `ppl_<corpus>_cached`.
A term would also fence the rows off from every `mlx_lm` row, which never
carries one.

**Producers of the column.** `rmlx_metrics::cell::decode_config` is the one
producer of drafter terms. The other composers are `rmlx_models::kv_cache`
(the boundary terms), `scripts/ingest/perf_ab_ingest.py`,
`scripts/ingest/codec_inertness_ingest.py` and
`scripts/prefill_chunk_sweep.sh`. `check-kv-boundary-default-parity` holds
their default values together; only `validate` holds their format.

`bests`, the cell queries and the exports partition on the whole cell key.
A new setting's cells separate as soon as its rows carry the term.

### 3.3 `bests` (VIEW — champion per cell, derived)

`rmlx_metrics::bests_view::create_sql` renders the view:

```sql
CREATE VIEW bests AS
WITH ranked AS (
    SELECT
        o.*,
        ROW_NUMBER() OVER (
            PARTITION BY backend, model_namespace, model, weight_quant,
                         kv_quant, ctx_max, prompt_id, decode_config, metric
            ORDER BY
                CASE WHEN direction = 'higher_better' THEN  value END DESC,
                CASE WHEN direction = 'lower_better'  THEN -value END DESC,
                ts_utc DESC
        ) AS rn
    FROM observations o
    WHERE CASE metric
              WHEN 'decode_tps_warm' THEN (value > 0.0 AND value <= 10000.0)
              -- one branch per registry metric (§4.1)
              ELSE 1
          END
)
SELECT * FROM ranked WHERE rn = 1
```

The partition is `cell::partition_columns`. Equal values resolve to the
newest `ts_utc`. The champion row carries every `observations` column,
plus `rn`.

The `WHERE` is generated from the §4 registry: `bests_view::plausible_sql`
renders one branch per metric from the `Bounds` that ingest enforces.
`query::deltas`, `query::regress`, `query::timeseries` and
`query::champions` `AND` in the same predicate.

A row outside its bound is excluded, not re-ranked. The cell falls to its
best plausible row, and drops out of `bests` when there is none. Without the
filter, the largest number in a partition would win, whatever it is.

Who rebuilds the view: `migrate::run_pending`, run by every writer, recreates
it when the stored definition differs from `create_sql`. `rmlx metrics doctor
--fix` does so on demand. A read command never rebuilds it:
`schema::open_checked` refuses a stale view and names `doctor --fix`. A query
must not change what the champion table publishes.

### 3.4 NULL policy (sparse rows are normal)

Backends measure different things, so most columns are nullable. The same
policy applies to `bests`.

**NOT NULL:** the cell columns except `decode_config`; `value`, `unit`,
`direction`; `run_id`, `ts_utc`, `hardware_tag`, `inserted_utc`,
`inserted_by`.

**Nullable:** `decode_config`; `git_sha`, `build_profile`,
`backend_version`; the bench-config columns; `output_first_64`,
`decode_stddev`, `notes`, `description`.

**Sparse metrics.** A metric the run did not measure has no row. An emitter
sends `null` for it and the recorder writes nothing. A row never carries a
placeholder value. Pivot queries read a missing row as `NULL` through
`MAX(CASE WHEN metric = …)`.

The champion export (§9) renders a missing cell as `-`, not `0`. It renders
`N/A` only for a backend that the scope file lists under `unsupported`.

### 3.5 Why no triggers / no UPSERT

The recorder only inserts into `observations`. There is no trigger and no
`INSERT OR REPLACE`. Champions are ranked at read time by `bests` (§3.3), so
nothing can fall out of sync with the observations. The cost is storage for
every observation (§10.2).

### 3.6 `events` table (runtime per-event stream)

Migrations `002_events.sql`, `003_events_identity.sql` and
`004_events_mlx_nax.sql`. `rmlx_metrics::events::EventRecorder::record`
writes one row per event, append-only.

```sql
CREATE TABLE events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT    NOT NULL,
    ts_utc          TEXT    NOT NULL,
    model_path      TEXT    NOT NULL,
    quant_mode      TEXT    NOT NULL,
    stage           TEXT    NOT NULL,
    op              TEXT    NOT NULL,
    value_unit      TEXT    NOT NULL,
    value           REAL    NOT NULL,
    notes           TEXT    NOT NULL DEFAULT '',
    backend_version TEXT,   -- migration 003
    build_profile   TEXT,   -- migration 003
    mlx_nax         TEXT    -- migration 004
);
CREATE INDEX events_run_id_idx ON events(run_id);
CREATE INDEX events_op_idx     ON events(op);
CREATE INDEX events_ts_idx     ON events(ts_utc);
```

**Identity.** `backend_version` and `build_profile` come from the same
`RunIdentity` as `observations` (§8.5.1). Rows written before migration 003
hold `NULL` in both. `events` has no `git_sha` column: only the in-process
recorder writes it, and nothing can supply a commit.

**`mlx_nax`.** Whether the MLX the process loaded ships the
`steel_gemm_fused_nax*` GEMM kernels: `present`, `absent` or `unknown`.
Without them, M5-class GPUs lose matmul and prefill throughput; decode is
bandwidth-bound and looks normal.

`rmlx_mlx::nax_capability()` scans the `mlx.metallib` beside the
`libmlx.dylib` that dyld resolved, once per process. It is read at run time
because the binary links MLX through a package-manager symlink that can move
after the build. `rmlx-metrics` does not depend on `rmlx-mlx`, so
`rmlx-cli`'s `main` passes the value to
`rmlx_metrics::identity::set_mlx_nax` at startup.

A process that never calls it records `unknown`. So does a metallib that
could not be inspected. The value is free-form text, not an enum. Rows
written before migration 004 hold `NULL`.

**`stage` / `op`.** Each writer sets its own strings. The server writes
per-request figures with `stage = 'request'` and `op` set to a §4 metric
name: `ttft_cold_ms` or `ttft_warm_ms`, `prefill_duration_ms`, `itl_p50_ms`,
`itl_p95_ms`, `itl_p99_ms` and `tpot_p50_ms` to `tpot_p99_ms` (unit `ms`),
and `kv_cache_bytes` (unit `bytes`). Other
writers use `baseline`, `stage0`, `stage1`, `audio`, `tts`, `mm_cache` and
`ssd_tier`.

The admission controller (`--adaptive-admission`) writes one row per
controller tick with `stage = 'admission_ctrl'`:

| `op` | When written |
|---|---|
| `admission_insufficient_data` | The regressor holds fewer than 4 points. No depth change. |
| `admission_no_change` | ITL estimate inside the deadband, above target for fewer than 3 ticks, or the depth already at its bound (1 or 256). |
| `admission_scale_down` | ITL estimate above target for 3 consecutive ticks. Depth decremented. |
| `admission_scale_up` | ITL estimate below 80% of target. Depth incremented. |

`value` is the ITL estimate in ms (`0.0` for `admission_insufficient_data`).
`notes` is a JSON object with `depth`, `est_itl_ms`, `itl_target_ms`,
`window` and `reason`. The controller writes no rows when
`--adaptive-admission` is absent. Its anticipatory 503s and its
`--adaptive-prefill-chunk` decisions go to the log, not to `events`.

---

## 4. Metric registry (canonical)

`rmlx_metrics::registry::METRICS` is the source of truth for `metric`,
`unit`, `direction` and bounds. Ingest refuses a name it does not list.

| `metric` | `unit` | `direction` | What it measures |
|---|---|---|---|
| `decode_tps_warm` | `tps` | `higher_better` | Decode tokens/s with prefill excluded, warm. |
| `decode_tps_cold` | `tps` | `higher_better` | Decode tokens/s on the first run after load. |
| `prefill_tps` | `tps` | `higher_better` | Prefill tokens/s. |
| `overall_tps` | `tps` | `higher_better` | Generated tokens over the whole request, prefill included. |
| `ttft_cold_ms` | `ms` | `lower_better` | Time to first token, first request after load. |
| `ttft_warm_ms` | `ms` | `lower_better` | Time to first token, later requests. |
| `itl_p50_ms` | `ms` | `lower_better` | Inter-token latency, median. |
| `itl_p95_ms` | `ms` | `lower_better` | Inter-token latency, 95th percentile. |
| `itl_p99_ms` | `ms` | `lower_better` | Inter-token latency, 99th percentile. |
| `itl_spikes` | `count` | `lower_better` | Intervals above 3 × the request's median ITL. |
| `step_ms_mean` | `ms` | `lower_better` | Mean time per generated token. |
| `prefill_duration_ms` | `ms` | `lower_better` | From `generate` entry to the first token; equal to the request's TTFT. |
| `tpot_p50_ms` | `ms` | `lower_better` | Time per output token over decode intervals, median; the same value as `itl_p50_ms`. |
| `tpot_p95_ms` | `ms` | `lower_better` | As above, 95th percentile. |
| `tpot_p99_ms` | `ms` | `lower_better` | As above, 99th percentile. |
| `model_load_ms` | `ms` | `lower_better` | Load wall time, start to ready. |
| `load_mmap_ms` | `ms` | `lower_better` | Load phase: weight-file mmap. |
| `load_dequant_ms` | `ms` | `lower_better` | Load phase: weight conversion. |
| `load_gpu_residency_ms` | `ms` | `lower_better` | Load phase: making weights GPU-resident. |
| `load_first_kernel_ready_ms` | `ms` | `lower_better` | Load phase: until the first kernel can run. |
| `load_total_ms` | `ms` | `lower_better` | Load phases, total. |
| `peak_rss_mb` | `mb` | `lower_better` | Peak resident set, `MACH_TASK_BASIC_INFO.resident_size`. |
| `peak_phys_footprint_mb` | `mb` | `lower_better` | Peak `TASK_VM_INFO.phys_footprint`, which counts compressed pages (`docs/PROFILING.md` §9). An emitter that samples it states its interval in `notes`: the figure is then a lower bound. |
| `metal_peak_alloc_mb` | `mb` | `lower_better` | Peak Metal allocation over the process lifetime (rMLX: `rmlx_mlx::mlx_peak_memory_bytes`). |
| `kv_cache_bytes` | `bytes` | `lower_better` | Resident KV bytes after decode. See below. |
| `tps_per_gb_ram` | `ratio` | `higher_better` | `decode_tps_warm / peak_rss_gb`. |
| `task_pass_at_1` | `ratio` | `higher_better` | Quality-probe pass rate, 0.0–1.0. |
| `prompt_cache_hits` | `count` | `higher_better` | Prompt-cache prefix matches, cumulative per server. |
| `prompt_cache_misses` | `count` | `lower_better` | Prompt-cache lookups with no match, cumulative per server. |
| `prompt_cache_bytes` | `bytes` | `lower_better` | RAM held by the occupied prompt-cache slots. |
| `prompt_cache_block_hits` | `count` | `higher_better` | 256-token blocks matched, cumulative. |
| `prompt_cache_block_misses` | `count` | `lower_better` | 256-token blocks not matched, cumulative. |
| `prompt_cache_partial_hits` | `count` | `higher_better` | Hits that matched only part of the wanted blocks. |
| `prompt_cache_hot_cache_hits` | `count` | `higher_better` | Hits in the in-RAM tier; the same value as `prompt_cache_hits`. |
| `prompt_cache_hot_cache_evictions` | `count` | `lower_better` | Slots evicted by the RAM LRU. |
| `prompt_cache_ssd_hits` | `count` | `higher_better` | RAM misses served from the SSD tier. |
| `queue_wait_ms` | `ms` | `lower_better` | Wait in the FIFO admission queue for the GPU permit. |
| `queue_depth` | `count` | `lower_better` | Admitted requests in flight at admission, this one included. |
| `prompt_tokens_live` | `count` | `lower_better` | Prompt tokens of one live request. |
| `completion_tokens_live` | `count` | `lower_better` | Completion tokens of one live request. |
| `accept_rate` | `ratio` | `higher_better` | `accept_tokens_total / draft_tokens_total`. |
| `draft_tokens_total` | `count` | `higher_better` | Draft tokens proposed over the request. |
| `accept_tokens_total` | `count` | `higher_better` | Draft tokens the verifier accepted. |
| `draft_rounds_total` | `count` | `higher_better` | Verifier rounds. |
| `accepted_per_step` | `ratio` | `higher_better` | `accept_tokens_total / draft_rounds_total`. |
| `tokens_per_round` | `ratio` | `higher_better` | Tokens the rounds emitted, per round. See below. |
| `draft_ms_per_round` | `ms` | `lower_better` | Time in the drafter call, per round. |
| `verify_ms_per_round` | `ms` | `lower_better` | Time in the verify forward, per round. |
| `loop_ms_per_round` | `ms` | `lower_better` | The rest of the round: rollback, snapshots, acceptance, sampling. |
| `ssd_bytes_used` | `bytes` | `lower_better` | On-disk KV-block footprint of one namespace. |
| `ssd_evict_total` | `count` | `lower_better` | SSD LRU evictions since the previous emit. |
| `ssd_spill_ms` | `ms` | `lower_better` | One spill's duration. The Prometheus histogram `rmlx_ssd_spill_us_bucket` holds the percentiles. |
| `ssd_hydrate_ms` | `ms` | `lower_better` | One hydrate's duration; histogram `rmlx_ssd_hydrate_us_bucket`. |
| `ssd_spill_mb_per_s` | `mb/s` | `higher_better` | Spill throughput. |
| `ssd_hydrate_mb_per_s` | `mb/s` | `higher_better` | Hydrate throughput. |
| `ppl_wikitext2` | `ppl` | `lower_better` | Perplexity on the wikitext-2 raw test split, cacheless scorer (`rmlx eval ppl`). |
| `ppl_wikitext2_cached` | `ppl` | `lower_better` | The same, scored through a per-layer KV cache (`rmlx eval ppl` with a KV codec). |
| `ppl_mean_nll` | `nat` | `lower_better` | Mean negative log-likelihood per scored token. |
| `ppl_scored_tokens` | `count` | `higher_better` | Corpus positions scored. |
| `ppl_windows` | `count` | `higher_better` | Sliding-window forwards run. |
| `ppl_score_ms` | `ms` | `lower_better` | Scorer wall time, load excluded. |

`rmlx eval ppl` supports Qwen3, Gemma4 and Qwen3.5-MoE.

**`kv_cache_bytes`.** The filled part of the KV cache that serves decode.
It counts packed codes, scales, rotation and residual buffers. It counts the
per-position bf16/f32 decode buffers up to the filled length. `KvCache::resident_bytes`
reads real array shapes. The decode mirrors are allocated to `--max-ctx`, so
only `offset` positions of them count. A prompt-cache snapshot is not
counted.

Every arch samples it once, after the decode loop, when a ring-backed codec's
GPU ring is resident. `KvBytesCounter::store` takes a `PostDecode` witness
that only a completed decode loop mints. A run with no decode loop, such as
an immediate EOS, stores nothing.

The counter is a field of each model instance. A caller that records the
figure reads `Architecture::kv_cache_bytes_sample()` (`KvBytesSample { bytes,
seq }`) before and after the generation. It records only if `seq` advanced;
otherwise the count belongs to an earlier generation. `kv_cache_bytes()`
returns the bare count, for display surfaces like `/metrics/cache`; do not
record from it.

**`metal_peak_alloc_mb`** is lifetime-scoped for every backend, so rows stay
comparable. `rmlx baseline` also prints a region-scoped `metal_gen_alloc_mb`
(`rmlx_mlx::PeakBracket`). That figure is stdout only and never recorded.

**`tokens_per_round`** counts at the round loops' emit sites. The bonus token
a sidecar loop emits from the prefill forward, before round one, is excluded.
It equals `1 + accept_rate × (block − 1)` only while every round drafts the
configured block, so it is recorded, not derived.

### 4.1 Plausible-value bounds

Each registry entry carries a `Bounds`: the window of values that can be a
measurement. `crates/rmlx-metrics/src/registry.rs` (`METRICS`) holds the
values; this section is the policy.

The floor is always `0`. A negative value, NaN or an infinity is never a
measurement. The families differ in whether `0` itself is one:

| Family | Floor | Ceiling | Why |
|---|---|---|---|
| Rates (`tps`, `mb/s`, `tps_per_gb_ram`) | `0` excluded | `1e4`; `prefill_tps` and `tps_per_gb_ram` `1e5`; `mb/s` `1e6` | A zero rate means no token was produced. |
| Durations (`ms`) | `0` included | `3.6e6` (1 h) | A sub-ms span rounds to 0. A span past an hour is a hung run. |
| Counters (`count`) | `0` included | `1e12` | Zero cache hits is real. |
| Gauges (`mb`, `bytes`) | `0` included, except `peak_rss_mb` and `peak_phys_footprint_mb` | `1e9` MB, `1e13` B | A live process always has RSS. A run can allocate no Metal. |
| Ratios (`ratio`, except `tps_per_gb_ram`) | `0` included | `1.0`; `accepted_per_step` and `tokens_per_round` `1e3` | Acceptance can be zero. A per-round count is not a fraction. |
| Perplexity (`ppl`, `nat`) | `ppl`: `0` excluded; `nat`: `0` included | `ppl` `1e6`, `nat` `1e3` | Perplexity is at least 1. |

Ceilings are loose. They reject fabrications, not fast machines.

**What a bound can and cannot catch.** It rejects a value orders of magnitude
out of range. It cannot see a wrong value inside the range. Bounds back up a
correct producer; they do not replace one.

Three places enforce the same bounds:

1. **Ingest.** `RunRecord::validate` rejects the whole record with
   `ImplausibleValue`. An emitter with no measurement sends `null`.
2. **`bests`.** The view does not rank a row outside its bound (§3.3).
3. **`rmlx metrics doctor`.** Check 6b reports stored rows outside their
   bound (§10.4). It warns and does not fail: append-only rows cannot be
   corrected, so an error would fail every run for good.

**Where bounds cannot decide.** Some placeholders are valid numbers. CBB's
`summary.csv` writes `task_pass_at_1 = 0.0` when it ran no quality probe,
and `0.0` is also a real score. `migrate::legacy` drops that zero where it
parses the column, and the bound admits `0.0`.

**Archive converters.** `rmlx metrics migrate` replays other tools' CSV and
JSONL exports, which write `0.0` for a column they never measured. It drops
those entries instead of failing the run, and counts them as
`metrics_dropped_implausible` in its report.

#### Known-bad rows already in the DB

`observations` is append-only, so rows that an older producer wrote wrong
stay in the DB. Current producers and ingest write no new rows of these
kinds. The rows are inside their bounds, so a predicate names each.

- **`decode_tps_warm` with prefill in its window.** Older `spec_bench.sh`
  runs divided by a window that started before prefill. Current producers
  take the rate from the engine and write `decode_window=` in `notes`
  (`engine_round_loop` or `engine_itl`). `rmlx baseline`-driven producers and
  `llama-bench` rows exclude prefill at the source and carry no marker, so
  the predicate is scoped to `spec_bench`:

  ```sql
  SELECT * FROM observations
  WHERE metric = 'decode_tps_warm'
    AND (notes IS NULL OR notes NOT LIKE '%decode_window=%')
    AND description LIKE 'spec_bench%';
  ```
- **`spec_bench.sh` rows labelled `kv_quant = 'k8v8'` with
  `prompt_tokens = 14`.** The script wrote both as constants. It now reads
  the codec from the `cache-type resolved` event and the length from
  `usage.prompt_tokens`, and refuses a run that reports neither.

  ```sql
  SELECT * FROM observations
  WHERE description LIKE 'spec_bench%'
    AND kv_quant = 'k8v8'
    AND prompt_tokens = 14
    AND (notes IS NULL OR notes NOT LIKE '%decode_window=%');
  ```
- **Speculative rows with `decode_config IS NULL`.** Migration 006 fills
  `decode_config` from the drafter that `notes` names
  (`rmlx_metrics::cell::decode_config_from_notes`). It writes no measurement.
  A speculative row that names no drafter stays in the plain cell, and no
  predicate finds it. This query returns no rows when the fill is complete:

  ```sql
  SELECT * FROM observations
  WHERE decode_config IS NULL
    AND notes LIKE '%draft_kind=%'
    AND notes NOT LIKE '%draft_kind=none%'
    AND notes NOT LIKE '%config=normal%'
    AND notes NOT LIKE '%config=base%';
  ```
- **`eagle/block=5` beside `eagle3/block=5`.** A bench script wrote the
  drafter as `eagle`; the engine writes `eagle3`. Nothing recorded shows that
  the two populations ran the same drafter, so no migration merges them.
  Re-running the cell fills it.

  ```sql
  SELECT * FROM observations WHERE decode_config LIKE 'eagle/%';
  ```

Read a recorded rate through `bests` or a `query::*` function; both apply
the bound. A consumer that needs the raw distribution, such as a median over
a cell, carries the predicate itself and cites this section.
`scripts/perf_ceiling.py`'s `prefill_anchor` is the one such consumer.

**Backend coverage.** `rmlx_metrics::registry::COVERAGE_MATRIX` states which
backend can emit which metric. A pair it does not list reads as `No`. Its
backends are `identity::BACKEND_WHITELIST` minus
`BACKENDS_WITHOUT_COVERAGE` (`vllm`). Adding a backend means adding its block
there; `registry_tests.rs` walks the whitelist.

For the fifteen cross-backend metrics (`BACKEND_METRIC_SPEC`):

| Metric | rmlx | mlx_lm | mlx_lm_tq | paroquant | omlx | ollama | llama_cpp | llama_cpp_tq |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `decode_tps_warm` | yes | yes | yes | yes | yes | yes | yes | yes |
| `decode_tps_cold` | yes | yes | yes | yes | yes | yes | yes | yes |
| `prefill_tps` | yes | yes | yes | yes | yes | yes | yes | yes |
| `overall_tps` | yes | yes | yes | yes | yes | yes | no | no |
| `ttft_warm_ms` | yes | yes | yes | yes | yes | yes | no | no |
| `ttft_cold_ms` | yes | yes | yes | yes | yes | yes | no | no |
| `itl_p50_ms` | yes | yes | yes | yes | yes | yes | no | no |
| `itl_p95_ms` | yes | yes | yes | yes | yes | yes | no | no |
| `step_ms_mean` | yes | yes | yes | yes | yes | yes | yes | yes |
| `model_load_ms` | yes | yes | yes | yes | yes | yes | yes | yes |
| `peak_rss_mb` | yes | yes | yes | yes | yes | yes | yes | yes |
| `metal_peak_alloc_mb` | yes | yes | yes | yes | yes | no | no | no |
| `kv_cache_bytes` | yes | no | no | no | maybe | no | yes | yes |
| `tps_per_gb_ram` | yes | yes | yes | yes | yes | yes | yes | yes |
| `task_pass_at_1` | no | no | no | no | no | no | no | no |

`no` means the backend cannot measure it. `maybe` means the backend exposes
it and no recording path is wired. The other `rmlx` entries of
`COVERAGE_MATRIX` are `yes` for the prompt-cache, load-phase, queue, live
token, ITL, speculative, SSD, `prefill_duration_ms` and `tpot_*` metrics. The
`ppl_*` metrics and `peak_phys_footprint_mb` have no coverage entry.

The server's metrics drainer writes `observations`; its
`event_kind_to_metrics` (`crates/rmlx-server/src/metrics_drainer.rs`) is the
list of metrics it records. It records every TTFT as `ttft_warm_ms`. The
per-request figures in `events` (§3.6) tell cold and warm TTFT apart.

Rules:

- A new metric gets an entry in `registry::METRICS`, a row in the table
  above, and a `COVERAGE_MATRIX` entry for each backend that emits it.
- Never repurpose a metric name.
- Store raw values in the registry unit.

---

## 5. Identity & cell normalization (mandatory rules)

`RunRecord::validate` checks the identity fields against the lists in
`rmlx_metrics::identity`. It lowercases a value and applies the aliases below
before the lookup. The recorder then stores the value as the record spells
it, so an emitter sends the canonical spelling.

### 5.1 `model_namespace` + `model` canonicalization

`model_namespace` is who published or repackaged the weights. `model` is the
short name within that namespace. Populate both, and never put the namespace
into `model`.

`identity::split_model_path` derives the pair from a path, an HF id or an
ollama tag:

| Input | `model_namespace` | `model` |
|---|---|---|
| `$RMLX_O_MODELS_ROOT/mlx-community__gemma-4-e2b-it-mxfp8` | `mlx-community` | `gemma-4-e2b-it-mxfp8` |
| `$RMLX_O_MODELS_ROOT/z-lab__Qwen3.6-27B-PARO` | `z-lab` | `Qwen3.6-27B-PARO` |
| `$RMLX_O_MODELS_ROOT/prism-ml__Ternary-Bonsai-8B-mlx-2bit` | `prism-ml` | `Ternary-Bonsai-8B-mlx-2bit` |
| an absolute path with no `__` | `local` | the last path component |
| ollama tag `llama3.2:3b` | `ollama` | `llama3.2:3b` |
| HF id `meta-llama/Llama-3.2-3B-Instruct` | `hf` | `meta-llama/Llama-3.2-3B-Instruct` |

A path's namespace must be in `identity::NAMESPACE_WHITELIST`:
`mlx-community`, `z-lab`, `prism-ml`, `paramind`, `paro-team`, `ollama`,
`hf`, `local`. Any other input is an error. `rmlx baseline`, `rmlx eval` and
`rmlx metrics migrate` use this strict form, because their caller can fix
the path.

`identity::split_model_id` is the lenient form that `RunRecordBuilder::rmlx`
uses. It splits a snapshot name on its first `__` and falls back to `local`.

**Ingest does not check either field against a list.** They are free-form
labels, like `kv_quant` (§5.3), so a new host or a renamed fine-tune still
records. `rmlx metrics doctor` does sweep `model_namespace` against
`NAMESPACE_WHITELIST` and reports a value outside it as an error (§10.4).

### 5.2 `weight_quant` canonicalization

Must be in `identity::WEIGHT_QUANT_WHITELIST`: `bf16`, `fp16`, `mxfp8`,
`mxfp4`, `nvfp4`, `q8_0`, `q4_k_m`, `2bit`, `3bit`, `4bit`, `5bit`, `6bit`,
`8bit`, `paro`. `bf16` means unquantized. `identity::infer_weight_quant`
reads the value from a snapshot name for `RunRecordBuilder::rmlx` and falls
back to `bf16`.

### 5.3 `kv_quant` canonicalization

`kv_quant` is a free-form label, not checked against a codec list.
`rmlx-metrics` does not depend on `rmlx-kv-quant`, so any list here would be
a hand-kept copy of the codec enum. It would go stale with the next codec and
drop that codec's rows at ingest.

`identity::canonicalize_kv_quant` trims and lowercases the value and maps
`bf16` and `f16` to `none`, and `rotor_v_3` / `rotor_v_4` to `rotor3` /
`rotor4`. Any other value passes through. `RunRecordBuilder::rmlx` and the
legacy importer (§7) apply it; a JSON record is stored as it spells the
field.

### 5.4 `backend` whitelist

Only these strings are allowed. The list mirrors
`rmlx_metrics::identity::BACKEND_WHITELIST`. Extend both in the same change,
or the doc says one thing and the validator does another:

- `rmlx`
- `mlx_lm` (Apple stock)
- `mlx_lm_tq` (the TurboQuant fork)
- `paroquant`
- `omlx`
- `ollama`
- `vllm` (no runner and no coverage entries; see §4)
- `llama_cpp` (upstream `ggml-org/llama.cpp`)
- `llama_cpp_tq` (the `llama-cpp-turboquant` fork)

Before the lookup, `llama.cpp` / `llama-cpp` / `llamacpp` read as
`llama_cpp`, and `llama-cpp-turboquant` / `llama.cpp-turboquant` /
`llama_cpp_turboquant` read as `llama_cpp_tq`.

**A fork is its own backend id, never a `kv_quant` value on the upstream id.**
`llama_cpp_tq` carries codecs (`turbo2` / `turbo3` / `turbo4`) that upstream
cannot load. Under `llama_cpp`, a turbo row would name a binary that cannot
produce it, and `bests` would rank two builds in one backend column.

### 5.5 `hardware_tag`

One string per host, such as `m5_max_128gb`. For `rmlx`,
`RMLX_HARDWARE_TAG` sets it, and `m5_max_128gb` is the default. A different
host needs a different tag.

`hardware_tag` is run context, not cell identity. `bests` does not partition
on it, so rows from two hosts compete for one champion. Filter on the column
when a comparison must stay on one host.

---

## 6. The `description` field (operating instruction)

`description` is on every `observations` row, and `bests` carries the
champion's. An emitter can send it in the record, and `rmlx metrics
describe` sets it afterwards on one `--observation-id` or on every row of a
`--run-id`. `notes` is what the emitter writes by machine. `description` is
what a person or an agent writes after reading the run.

Format: one to three lines, no headings.

```
<git-sha>: <one-line summary of what changed>
[why it improved or regressed]
[the doc that explains it]
```

Rules:

- A row that beats a champion cites the commit, or states that no commit
  changed.
- A regression states the suspected cause.
- Quote a commit subject verbatim. If unsure, leave the field empty.
- Never write `synthetic=true` into `notes` or `description`: ingest refuses
  such a record (§8.5).

---

## 7. Migration plan

`rmlx metrics migrate` (`migrate::legacy::migrate_all`) imports archives
written before the DB existed. It applies pending schema migrations first
and prints a JSON summary of rows read, inserted and skipped. It is a
one-shot importer, not a recording path.

It records through `Recorder::legacy_archive`, the one recorder that skips
the `backend_version` check (§8.5.1). Archive rows have no version to state.

It is idempotent. Each imported row carries `legacy_run_key=<hex>` at the
start of `notes`, and a row whose key is already present is skipped.

It drops archive entries that write `0.0` for an unmeasured column (§4.1).

### 7.1 Sources

| Flag | Source |
|---|---|
| `--rmlx-glob <glob>` | rMLX JSONL rows carrying `ts_utc`, `model_path`, `kv_quant` and `decode_tps_mean`. The walk goes at most three directory levels deep. |
| `--cbb-csv <path>` | A Cross-Backend-Bench `summary.csv`. |
| `--records-md <path>` | A hand-written records table in Markdown. |
| `--hardware-tag <tag>` | Stamped on every imported row. Default `m5_max_128gb`. |

Each flag is optional; a pass whose flag is absent is skipped.

### 7.2 Field mapping

- `run_id` is minted anew by the recorder. A source `run_id` is discarded.
- JSONL: `decode_tps_mean` becomes `decode_tps_warm`, with
  `decode_tps_stddev` as its `decode_stddev`. `step_ms_mean` becomes
  `step_ms_mean`, and `first_32_words` becomes `output_first_64`. The model
  comes from `model_path` through `split_model_path` (§5.1), and the weight
  quant from the model name's suffix. A file whose name contains `-dirty`
  gets `-dirty` appended to its `git_sha`. Every row is `ctx_max = 8192`,
  `prompt_tokens = 4096`, `max_tokens = 32`, `temperature = 0`, `seed = 0`,
  one warm-up and three measured runs.
- CSV: `timestamp_utc`, `backend`, `backend_version`, `model_id`,
  `quant_signature` and `device` map to the run context. `quant_signature`
  splits on `/` into `weight_quant` and `kv_quant`. The columns
  `decode_tps`, `overall_tps`, `ttft_ms`, `itl_p50_ms`, `itl_p95_ms`,
  `peak_rss_mb` and `task_pass_at_1` each become one metric row. A row
  whose `success` is `false` or `0` is skipped.

### 7.3 Prompt-body recovery for legacy rows

Archive rows store `prompt_tokens`, not the prompt. The importer reads
`prompts/longctx_4k.json` under the repo root (`RMLX_REPO_ROOT`, or the
working directory). A row with `prompt_tokens = 4096` gets that prompt. Any
other count gets a placeholder prompt named `legacy_unknown_<N>`.

---

## 8. Tooling — Rust

Every metrics operation is an `rmlx metrics` subcommand
(`crates/rmlx-cli/src/commands/metrics/`) over the `rmlx-metrics` crate.
Scripts in other languages shell out to it and never write the DB directly.

### 8.1 Who writes `observations`

- `rmlx metrics record`, from a buffer file, an argument or stdin.
- `rmlx baseline --record` and `rmlx eval ppl`. Each writes a buffer file,
  then records it in-process (§8.4).
- The server's metrics drainer (`crates/rmlx-server/src/metrics_drainer.rs`),
  through `RunRecordBuilder::rmlx`.
- `rmlx metrics migrate`, for archives only (§7).

The `events` table is written only by `EventRecorder` in the running binary
(§3.6).

### 8.1.1 What is NOT a recording path: `rmlx bench`

`rmlx bench` (see [`docs/CLI.md`](CLI.md#bench)) measures TTFT, ITL, decode
TPS and `kv_cache_bytes` over repeated runs of one cell. It prints medians
with the observed range and **writes nothing**: no buffer file and no row.
`observations` is append-only, and `bench` exists to establish a number and
its spread, including the runs that get thrown away. `rmlx baseline --record`
writes a figure worth keeping.

`bench` refuses five things: a run served from the prompt cache, a KV-byte
figure the run did not report, a metric that trended, runs that decoded
different tokens, and `--runs 1`. A recording path that measures the same
quantities refuses on the same conditions.

The two paths that write `kv_cache_bytes` hold that rule: `rmlx baseline
--record` and the server's speculative-decode request boundary. Each samples
`kv_cache_bytes_sample()` before and after the generation. When the store
sequence did not advance, the figure belongs to an earlier generation. Both
paths then `warn!` and omit the row.

### 8.2 Subcommands

`--db <path>` applies to every subcommand. A command that fails exits 1
unless the table says otherwise.

| Subcommand | What it does |
|---|---|
| `init` | Creates the DB and applies every migration. Refuses a path that exists. |
| `doctor [--fix]` | Checks the DB (§10.4) and applies pending migrations. |
| `backup [--out <path>] [--keep N]` | `VACUUM INTO` copy (§10.1). |
| `restore --from <path>` | Snapshots the current DB, then replaces it (§10.1). |
| `record --inline <json> \| --file <path> \| --stdin [--dry-run]` | Ingests one §8.5 record and prints the outcome as JSON. `--dry-run` validates and writes nothing. |
| `record --replay-pending [--dry-run]` | Ingests every `buffer/pending/*.json` (§8.4). Exits 2 when any file failed. |
| `identity [--json]` | Prints this binary's run identity (§8.5.1). |
| `validate --file <path> \| --stdin` | Runs the ingest validator and writes nothing (§8.5.2). |
| `best` | The champion row of one cell and metric, as JSON. Exits 1 when there is none. |
| `rank --metric M [--backend B] [--limit N]` | Top-N champions of one metric, default 20. |
| `compare --backends a,b --metric M` | The champion of each listed backend, per cell. |
| `history` | Every observation of one cell, oldest first; `--metric` and `--since <date>` filter. |
| `timeseries --metric M [--bucket day\|week]` | Mean per day or week for one cell and metric. |
| `champions [--backend B] [--jsonl]` | One row per model, weight quant and KV quant, with one column per metric. |
| `regress --model <substring> --metric M [--kv K] [--threshold-pct P]` | Latest observation against the champion. Exits 0 within `P` % (default 1.0), 1 on a regression, 125 with no champion or no observation. |
| `deltas --since-sha <sha> [--threshold-pct P] [--exit-code false]` | Per cell and metric: the best after the SHA's first row (else the champion) against the best up to it. Prints the moves beyond `P` % (default 5.0) and the cells with no value up to it. Exits 1 on a regression, 125 when no printed cell has a value up to it. `--exit-code false` always exits 0. A SHA with no rows is an error. |
| `describe --observation-id N \| --run-id R --text T` | Sets `description` (§6). |
| `query "<SELECT …>"` | Runs one statement that starts with `SELECT` and prints TSV with a header. |
| `open [--readonly]` | Starts `sqlite3` on the DB; `--readonly` passes `-readonly`. |
| `export --markdown \| --json \| --csv \| --jsonl [--scope <toml>]` | Prints `bests` (§9). |
| `prompts list \| get --name N \| add --file F \| sync` | The prompt registry (§8.7). |
| `migrate` | Imports archives (§7). |

`best`, `history` and `timeseries` take the whole cell: `--backend`,
`--namespace`, `--model`, `--weight-quant`, `--kv-quant`, `--ctx-max`
(default 8192), `--decode-config` (omit it for the default configuration)
and one of `--prompt-id` or `--prompt-name`. A name resolves to the newest
prompt carrying it.

How each command opens the DB:

- `record`, `prompts add` / `sync` and `champions` open with
  `schema::open_migrated`, which applies pending migrations first.
- The other read commands, and `describe`, open with `schema::open_checked`.
  It refuses a path that does not exist and a stale `bests` view (§3.3), and
  it never migrates.

### 8.2.1 Atomicity contract

One `record` call is one transaction:

- `RunRecord::validate` runs first. A rejected record opens no transaction
  and changes nothing.
- The prompt is resolved or inserted, the `run_id` is minted, and one
  `observations` row is inserted per non-null metric, inside the transaction.
- An error inside it rolls the transaction back. The buffer file stays where
  it was (§8.4).

### 8.4 JSON buffer pattern (per-run write→ingest→delete)

An emitter writes one §8.5 JSON file per run to
`<RMLX_HOME>/metrics/buffer/pending/<ts>-<id>.json`, then ingests it. A
record that fails to ingest is still on disk, so no measurement is lost to a
locked or missing DB.

- `rmlx metrics record --file <path>` deletes the file on success. On
  failure it exits 1 and leaves the file in place.
- `rmlx baseline --record` and `rmlx eval ppl` ingest in-process, delete the
  file on success, and leave it in `pending/` on failure.
- A script that calls `record --file` moves a rejected file to
  `buffer/failed/` for triage (`scripts/perf-iter/bench_decode_tps.sh`).
- `rmlx metrics record --replay-pending` walks `pending/` in name order. It
  deletes each file that ingests and moves each one that fails to
  `buffer/failed/`. It then prints `ok` and `fail` counts and exits 2 if any
  failed. `--dry-run` moves and deletes nothing.

Nothing expires `buffer/failed/`. Fix the cause, move the files back to
`pending/`, and replay.

### 8.5 Ingest contract (universal — every backend uses this shape)

One JSON object per run. The recorder writes one `observations` row per
metric whose value is not `null`, in one transaction (§8.2.1).

```json
{
  "schema_version":  1,
  "backend":         "rmlx",
  "backend_version": "0.2.8",
  "model_namespace": "mlx-community",
  "model":           "gemma-4-e2b-it-mxfp8",
  "weight_quant":    "mxfp8",
  "kv_quant":        "k8v8",
  "ctx_max":         8192,
  "decode_config":   null,
  "prompt": {
    "name":  "longctx_4k",
    "body":  "You are an expert ...",
    "notes": null
  },
  "ts_utc":          "2026-05-10T07:30:00Z",
  "git_sha":         null,
  "build_profile":   "release-perf",
  "hardware_tag":    "m5_max_128gb",
  "prompt_tokens":   4096,
  "max_tokens":      32,
  "temperature":     0.0,
  "seed":            0,
  "n_warmups":       1,
  "n_measure":       3,
  "output_first_64": null,
  "notes":           null,
  "description":     null,
  "metrics": [
    { "name": "decode_tps_warm", "value": 119.14, "stddev": 0.64 },
    { "name": "ttft_warm_ms",    "value": null }
  ]
}
```

`RunRecord` (`crates/rmlx-metrics/src/ingest.rs`) parses it. An unknown key
is ignored. **Required keys**, and what `RunRecord::validate` checks:

| Field | Check |
|---|---|
| `backend` | In the §5.4 whitelist, after aliases. |
| `model_namespace` | None; free-form (§5.1). |
| `model` | None; free-form. `model_id` is accepted as the key. |
| `weight_quant` | In the §5.2 whitelist. |
| `kv_quant` | None; free-form (§5.3). |
| `ctx_max` | Greater than 0. |
| `prompt` | See below. |
| `ts_utc` | Parses as ISO-8601. |
| `hardware_tag` | Not empty. |
| `metrics` | At least one entry with a non-null `value`. |

**Optional keys**: `schema_version` (default `1`), `backend_version`,
`git_sha`, `build_profile`, `prompt_tokens`, `max_tokens`, `temperature`,
`seed`, `n_warmups`, `n_measure`, `output_first_64`, `decode_config`,
`notes` and `description`. The checks on them:

- `schema_version` above `RECORD_SCHEMA_VERSION` (`1`) is refused.
- `backend_version` is required, and must be semver, when `backend` is
  `rmlx` (§8.5.1).
- `temperature` lies in `0.0..=2.0`.
- `decode_config` is cell identity, not context (§3.2). Omit it or send
  `null` for a run at every default. A value must follow the §3.2 grammar. A
  value that spells only the defaults is refused, and so is an adaptive
  drafter written with a fixed depth.
- `notes` or `description` containing `synthetic=true` refuses the record.
  To test whether a record would be accepted, use `record --dry-run`.

**Metric entry**: `{ "name": …, "value": …, "stddev": … }`.

- `name` must be in the §4 registry.
- `value` `null` writes no row. A number outside the metric's §4.1 bounds
  refuses the whole record.
- `stddev` is optional and stored as `decode_stddev` for any metric.

**Prompt**: one of two forms.

- `{ "name", "body", "notes"?, "tokens_approx"? }`. `name` is not empty, and
  `body` is any JSON value except `null`. The recorder hashes the body
  (`ingest::prompt_body_sha256`), reuses the `prompts` row with that hash, or
  inserts one.
- `{ "sha256": "<64 hex>" }` names a registered prompt. The recorder refuses
  a hash that is not in `prompts`.

`rmlx metrics record` also accepts two older buffer shapes, converted by
`legacy_ingest::try_parse_legacy` and `legacy_ingest::try_parse_cbb`. New
emitters write the shape above.

### 8.5.1 Run identity (hard rule)

The identity fields `backend`, `backend_version`, `build_profile` and
`hardware_tag` say which binary produced the number. No emitter writes them
by hand. Each language surface takes them from one place:

| Surface | Source |
|---|---|
| Rust, borrowed | `rmlx_metrics::identity::RunIdentity::get()`: resolved once per process, with no I/O. |
| Rust records | `rmlx_metrics::ingest::RunRecordBuilder::rmlx(...)`: fills identity, `model_namespace`, `model`, `weight_quant`, `kv_quant`, `ts_utc` and `schema_version`. The caller adds the measurement. |
| Shell, Python | `rmlx metrics identity --json`. `scripts/lib/identity.sh` exports it as `RMLX_IDENTITY_JSON`, and each record merges it. |

`identity --json` also prints `mlx_nax`, the MLX nax-kernel capability the
process loaded. It is written to `events` only (§3.6); a record ignores it.

**`git_sha` is not an identity field.** The binary does no git of any kind,
at build time or at run time. The directory a process starts in is not
necessarily its own checkout. `git_sha` is provenance that the caller
supplies:

- A bench script runs `git rev-parse` in its own repo and sets `"git_sha"`
  after merging the identity block.
- `rmlx baseline --git-sha <sha>` and `rmlx eval ppl --git-sha <sha>` stamp
  it. Without the flag it is `NULL`.

The server drainer has no such input, so its rows carry `NULL`.
`deltas --since-sha` also matches `<sha>-dirty`, a suffix that older rows
carry.

**Enforcement.** `RunRecord::validate` runs on every ingest path: `record`,
`--replay-pending` and the in-process `Recorder`.

- An `rmlx` record needs a semver-shaped `backend_version`: `MAJOR.MINOR.PATCH`
  with an optional `-pre` or `+build` suffix. A missing, empty or other value
  is refused and exits 1, and the buffer file stays for triage.
- Other backends keep `backend_version` optional and free-form.
- `git_sha` is never required.

The check proves the shape, not the source. A hand-written buffer file can
carry any semver-shaped version. `RunRecord` is `#[non_exhaustive]`, and its
identity fields are `pub(crate)` behind getters. Rust code outside the crate
therefore builds one only through `RunRecordBuilder` and cannot change one
after it is built.

**Identity is stamped at emit time, into the buffer file, never at ingest
time.** A buffer replayed by a newer binary keeps the identity of the build
that produced it. `inserted_by` (`<tool>@<semver>`) separately names the tool
that inserted the row. Build it with `RunIdentity::inserted_by(tool)`.

**`build_profile` is the real Cargo profile name**: `release`,
`release-perf`, `release-debug` or `debug`. `crates/rmlx-core/build.rs` reads
it from `OUT_DIR`: the component before the last `build` component. Do not
use `cfg!(debug_assertions)`. It is off in all three release profiles, so a
cross-profile comparison would read as a same-profile one.

### 8.5.2 Validating a record without writing it

```bash
rmlx metrics validate --file <buffer.json>
rmlx metrics validate --stdin
```

It runs `RunRecord::validate`, the recorder's own check, and prints one `ok`
line or exits 1. There is no separate JSON Schema file; a second copy of the
contract would drift. Two things pass here and can still fail in `record`: a
`sha256` prompt that is not registered, and a transaction error. `validate`
does not try the older shapes that `record` converts.

### 8.7 Prompt ownership — rMLX is the source-of-truth

Bench prompts live in this repo under `prompts/`. Each top-level
`prompts/*.json` file holds `name`, `body`, and optionally `tokens_approx`
and `notes`.

- `rmlx metrics prompts sync` registers every top-level `*.json` file in
  `prompts/` under the repo root (`RMLX_REPO_ROOT`, or the working
  directory). It prints how many it inserted.
- `prompts add --file <path> [--name N] [--notes T]` registers one file.
- `prompts get --name N` prints the newest body carrying that name.

The `prompts` table is content-addressed (§3.1). A changed body is a new row
with a new id, and older observations keep the old id. A bench record always
carries the full body, and the recorder dedups it by hash.

---

## 9. `BENCHMARK_CHAMPIONS.md` regeneration

The champion table is generated, never edited by hand:

```bash
make metrics-export     # rmlx metrics export --markdown > BENCHMARK_CHAMPIONS.md
```

The file is git-ignored. It is a function of a machine-local DB, so each host
has its own.

`export --markdown` (`rmlx_metrics::export::export_markdown`) renders:

1. A fixed header (`crates/rmlx-metrics/src/templates/header.md`).
2. `## Records`: one table per model with one row per backend and KV quant.
   With `--scope <toml>` (`config/scope.toml`), only the listed models
   appear, in its order. A backend the scope lists as `unsupported` shows
   `N/A`. Without `--scope`, every model in `bests` appears.
3. A speculative-decoding table, when any champion's `decode_config` names a
   drafter.
4. A champion summary: the best decode rate per model and the rMLX gap.
5. A provenance block: export time, `bests` row count, cells rendered.

A missing value renders as `-`.

`--json`, `--csv` and `--jsonl` print one record per `bests` row, including
`git_sha`. Exactly one format flag is allowed.

### 9.1 What the `Updated` column means

Every metric column of a row is a separate `bests` lookup, one partition per
cell **and metric** (§3.3). A row's decode record and its memory record can
come from two runs. The column reports provenance for the metric columns
only:

- **One run behind all of them**: its date, its `run_id` and its notes.
- **More than one**: `no single run —`, then each `run_id` with the columns
  it backs. Naming one run would put its id and notes beside numbers it does
  not contain.
- **No metric column printed**: `-`.

`git_sha` is not in this column. From a Markdown row, look it up in
`observations` by `run_id`.

`KV GB` and `reduction vs bf16` are outside the column's scope. Both are
minima over every cell that matches the model, across backends and prompts,
so no single observation backs them.

---

## 10. Operational concerns

### 10.1 Backups

`rmlx metrics backup` writes a `VACUUM INTO` copy, which is consistent while
a writer is active. The default path is
`<RMLX_HOME>/metrics/backups/runs-<YYYYMMDD-HHMMSS>.db`; `--out <path>`
overrides it. `--keep N` then deletes all but the newest N `runs-*.db` files
in that directory. `make metrics-backup` runs it with `--keep 30`.

`rmlx metrics restore --from <path>` first copies the current DB to
`backups/pre-restore-<ts>.db`. It then copies the source beside the DB and
renames it into place.

Backups are git-ignored. Off-machine copies are the owner's job.

### 10.1.1 Turning metrics off (`--metrics`)

A global CLI flag, like `--log`. The default is `full`.

| Mode | `events` | `observations` | Effect |
|---|---|---|---|
| `full` (default) | yes | yes | Everything records. |
| `events` | yes | no | The runtime event stream only. |
| `off` | no | no | **No DB writes.** The drainer is never spawned, and `runs.db` is never opened or created. |

```bash
rmlx --metrics off serve --model <snapshot>
```

`rmlx_metrics::mode::init` sets the mode once, at process start, and every
writer reads it there. `off` stops the producers, so no record is built. A
GPU-capture run is forced to `off`, because capture makes every timing false.
`off` also stops the `metrics/baseline.csv` append of `rmlx baseline`.

`off` stops telemetry writes only. `rmlx metrics` read commands work in
every mode. `rmlx metrics record` and `migrate` are explicit commands, not
telemetry, and are not gated.

### 10.2 Retention policy

- `observations` is append-only. No command deletes a row. Two commands
  update one: `describe` sets `description`, and `doctor --fix` corrects
  `unit` and `direction` from the registry.
- `bests` is a view and holds no rows.
- `prompts` rows stay; `observations.prompt_id` references them. A changed
  prompt body is a new row.
- `buffer/failed/` has no automatic expiry (§8.4).

### 10.3 Audit

- Every row carries `inserted_by` (`<tool>@<semver>`), such as
  `rmlx-cli@<semver>` or `migrate@<semver>`.
- `schema_meta.created_by` names the version that created the DB (§3.0).
- `SELECT inserted_by, COUNT(*) FROM observations GROUP BY inserted_by;`
  shows which tools wrote the data.

### 10.4 Integrity & validation (`rmlx metrics doctor`)

`rmlx metrics doctor` (`crates/rmlx-cli/src/commands/metrics/admin.rs`) runs
these checks in order. It exits 1 when any check reports an error; warnings
do not fail it.

1. `PRAGMA integrity_check`. Error.
2. `PRAGMA foreign_key_check`. Error.
3. Schema version. Pending migrations are applied, without `--fix`.
3b. `bests` against the §4 registry. The view is generated, so a DB at the
    latest version can still hold one built from an older registry. Warning;
    `--fix` rebuilds it.
4. Whitelist sweep of `observations`: `backend`, `model_namespace` and
   `weight_quant` against their §5 lists, and `metric` against the registry.
   Error, naming the value and a row id. The `kv_quant` sweep cannot fail,
   because the field is free-form (§5.3).
5. `direction` against the registry. Error; `--fix` corrects it.
6. `unit` against the registry. Error; `--fix` corrects it.
6b. Values outside their §4.1 bounds, per metric, with a count and the first
    row id. Warning, never repaired: the rows cannot be corrected, only
    re-measured.
7. Prompts that observations reference and whose hash matches no
   `prompts/*.json` in the working directory. Warning.
8. `COVERAGE_MATRIX` pairs marked `Yes` with no row. Warning.
9. Champion cells whose `kv_quant` is not `none`, `k8v8` or `k4v4`. Warning:
   their fidelity is not covered by an automated check.

### 10.5 Concurrency

- SQLite WAL serializes writers and lets readers run in parallel.
- `busy_timeout=5000` makes a contending writer wait up to 5 s instead of
  failing.
- There is no lock file. A write is one short transaction, and the single
  MLX process rule (`CLAUDE.md`) keeps bench writers from running in
  parallel.

---

## 12. CI integration

### 12.1 Pre-push gate (`make ci`)

`make ci` runs `ci-metrics`: `rmlx metrics doctor` against
`$RMLX_HOME/metrics/runs.db` (default `.rmlx/`), skipped when that file is
absent. CI never runs a bench. It checks the DB's structure, its identity
whitelists, its registry agreement and its §4.1 plausibility.

It does not diff `BENCHMARK_CHAMPIONS.md`. That file is git-ignored and
differs on every host (§9). Regenerate it with `make metrics-export` after a
change to what `bests` publishes, such as a §4.1 bounds change.

---

## 13. Operating rules (instruction summary)

1. **The DB is the source of truth.** `metrics/legacy/` is an archive; never
   read or extend it.
2. **Path**: `<RMLX_HOME>/metrics/runs.db`, git-ignored (§2). Back up before
   bulk operations.
3. **Tables**: `prompts`, `observations` and `events`, the `bests` view, and
   `schema_meta` (§3). A new table is a new migration and a §3 section.
4. **`observations` is the ground truth**: every measurement, append-only.
   `bests` derives the champion per cell. No triggers, no UPSERT.
5. **One row per measured (cell, metric) per run.** A metric the run did not
   measure has no row and never a placeholder.
6. **Identity fields follow §5.** Send canonical spellings; the recorder
   stores what it is given.
7. **Metric names come from the §4 registry.** A new metric follows the §4
   rules. Never reuse a name.
8. **`description`** is written after reading the run and cites the commit
   (§6). Write one whenever a new champion appears.
9. **`run_id` is minted at write time**, `<YYYYMMDDHHMMSS>-<6hex>`. Never
   reuse an external id.
10. **One DB across repos.** Never fork it.
11. **`BENCHMARK_CHAMPIONS.md` is generated** with `make metrics-export`,
    never hand-edited (§9).
12. **`hardware_tag` is run context**, not cell identity (§5.5).
13. **WAL and `foreign_keys=ON`** are set on every connection. Never disable
    them.
14. **All tooling is `rmlx metrics …`.** Other languages shell out and never
    write the DB directly.
15. **Every backend emits the §8.5 shape.** One run is one record and one
    transaction.
16. **Prompts live in `prompts/*.json`** and are content-addressed (§8.7).
17. **Buffer every record** (§8.4). Replay with `--replay-pending`.
18. **Every row carries `inserted_by`.**
19. **Run `rmlx metrics doctor`** after migrations and before any operation
    on a DB in a suspect state (§10.4).
20. **Identity comes from `RunIdentity`** or `rmlx metrics identity --json`,
    never from a literal (§8.5.1).
21. **A/B experiments never write here.** `scripts/perf_ab.sh` runs every
    slot with `--metrics off`, so `runs.db` is not opened. A row from a
    discarded arm would be permanent. Record the surviving arm once with
    `rmlx baseline --record`.
22. **A new rule, metric or backend** updates this doc, `CLAUDE.md` and
    `crates/rmlx-metrics/README.md` in the same change.
