# Metrics Schema

The tables, the `bests` view and the metric registry of the metrics database,
with the bounds a stored value must hold. Where the database lives, the
identity rules, ingest, the `rmlx metrics` tooling and the operating rules are
in [`METRICS_DB.md`](METRICS_DB.md).

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
    backend          TEXT    NOT NULL,  -- METRICS_DB.md §5.4
    model_namespace  TEXT    NOT NULL,  -- METRICS_DB.md §5.1
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
| `backend`         | Engine name, lowercase, no version (`METRICS_DB.md` §5.4). |
| `model_namespace` | Who published the model (`METRICS_DB.md` §5.1). |
| `model`           | Short name within the namespace, no path. |
| `weight_quant`    | Weight quantization on disk; `bf16` if unquantized (`METRICS_DB.md` §5.2). |
| `kv_quant`        | KV-cache quantization at run time; `none` if unquantized (`METRICS_DB.md` §5.3). |
| `ctx_max`         | Server max context. It changes the KV cache shape, so it is part of the cell. |
| `prompt_id`       | The prompt. TPS is not comparable across prompts. |
| `metric`          | Registry name (§4). |
| `decode_config`   | Non-default engine configuration; `NULL` is every setting at its default. Grammar below. |
| `value`           | The number, in the registry unit. |
| `unit`            | Registry unit (§4). The recorder takes it from the registry. |
| `direction`       | `higher_better` or `lower_better`, from the registry. `bests` ranks by it. |
| `run_id`          | `<YYYYMMDDHHMMSS>-<6hex>`, minted by the recorder at write time. A tracking string, not a key. |
| `ts_utc`          | When the measurement was taken (ISO-8601 UTC). |
| `git_sha`         | Caller-supplied provenance (`METRICS_DB.md` §8.5.1). `NULL` unless a caller set it. `deltas --since-sha` also matches `<sha>-dirty`. |
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
| `description`     | Written by a person or an agent: why the run exists and what changed (`METRICS_DB.md` §6). |
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

The champion export (`METRICS_DB.md` §9) renders a missing cell as `-`, not `0`. It renders
`N/A` only for a backend that the scope file lists under `unsupported`.

### 3.5 Why no triggers / no UPSERT

The recorder only inserts into `observations`. There is no trigger and no
`INSERT OR REPLACE`. Champions are ranked at read time by `bests` (§3.3), so
nothing can fall out of sync with the observations. The cost is storage for
every observation (`METRICS_DB.md` §10.2).

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
`RunIdentity` as `observations` (`METRICS_DB.md` §8.5.1). Rows written before migration 003
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
   bound (`METRICS_DB.md` §10.4). It warns and does not fail: append-only rows cannot be
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
