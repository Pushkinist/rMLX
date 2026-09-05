# Metrics Database

Canonical store for all benchmark observations across rMLX and sibling backends. SQLite, append-only `observations` table as ground truth, `bests` exposed as a derived VIEW for the champion-per-cell record. Designed for both CLI summary (`BENCHMARK_CHAMPIONS.md` regen) and time-series UI (Grafana).

This document is also the operating instruction for any future Claude session that records, queries, or migrates metrics. Read before touching `metrics/runs.db`.

---

## 1. Why a DB

Today metrics are scattered:

- `metrics/<run-id>.jsonl` — rMLX runtime traces (tracing layer).
- `metrics/perf-iter/*.jsonl` — perf-book optimization sweep results.
- `Cross-Backend-Bench/metrics/summary.csv` — CBB longitudinal record (rMLX + mlx_lm + paroquant + omlx + ollama).
- `BENCHMARK_CHAMPIONS.md` — hand-curated highest-record table.

Pain:
- "Has any backend ever beaten rMLX on cell X?" needs `awk` over a 500 KB CSV plus three JSONL trees.
- `BENCHMARK_CHAMPIONS.md` is hand-edited and lossy — only the top number, no commit, no run context.
- Description fields (why-better / why-worse) live in commit messages or report markdown, not next to the number.

**Day-1 approach**: DB is THE ground truth from the migration commit forward. Existing JSONL/CSV is ingested once, then archived under `metrics/legacy/` and never extended. New runs write a per-run JSON buffer file (§8.4), recorder ingests + deletes on success. No dual-write era — JSONL doesn't get richer over time; the DB does.

---

## 2. Path & cross-repo discipline

- **Primary path**: `<RMLX_HOME>/metrics/runs.db` (resolved via `rmlx_core::paths::home()`).
- **Git**: file is git-ignored. Never commit binary DB.
- **Cross-repo access**: Cross-Backend-Bench reaches in via symlink:
  ```bash
  ln -s <rMLX>/metrics/runs.db \
        <Cross-Backend-Bench>/metrics/runs.db
  ```
  Symlink committed in CBB repo. Both repos write/read same DB.
- **Backups**: routine via `rmlx metrics backup` (§10.1). Manual snapshot before bulk operations:
  ```bash
  rmlx metrics backup --out metrics/backups/runs-pre-<reason>-$(date +%Y%m%d-%H%M).db
  ```
- **Concurrency**: SQLite `WAL` mode mandatory (`PRAGMA journal_mode=WAL`). Single-MLX-process rule keeps writer contention near zero, but readers still need WAL to avoid blocking on long writes.
- **Legacy archive**: `metrics/legacy/<YYYYMMDD>-pre-db/` holds the JSONL/CSV that existed at migration time. Read-only archive; tooling never re-reads. Kept for archaeology.
- **Buffer dirs**: `metrics/buffer/pending/` (recorder consumes), `metrics/buffer/failed/` (triage). See §8.4.
- **Backup dir**: `metrics/backups/` (auto-created by `backup` command).
- **All under** `metrics/` and gitignored.

---

## 3. Schema

Three user tables (`prompts`, `observations`, `bests`) plus one bookkeeping table (`schema_meta`). `bests` is a VIEW, not a base table — champions are derived at read time, not maintained by triggers. Storage truth = `observations`.

This shape supports **two access patterns**:

- **Champion view** (`bests` view) — "what's the best decode_tps_warm for cell X?" Used by CLI `best`/`rank`/`export --markdown` and the canonical `BENCHMARK_CHAMPIONS.md`.
- **Time-series** (`observations` table) — "how did decode_tps_warm for cell X evolve over the last 90 days?" Used by Grafana, regression detection, and `rmlx metrics history`.

No data is ever discarded. Worse-than-best observations are still inserted — they just don't surface in `bests`.

### 3.0 `schema_meta` (versioning + provenance)

```sql
CREATE TABLE schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- seeded on init:
INSERT INTO schema_meta(key, value) VALUES
    ('schema_version',    '1'),                       -- bump when DDL changes
    ('created_utc',       '<ISO-8601>'),
    ('created_by',        'rmlx-cli@<semver>'),
    ('hardware_tag',      'm5_max_128gb'),            -- default for this DB instance
    ('default_namespace', 'mlx-community');

PRAGMA user_version = 1;                              -- mirrors schema_meta.schema_version
```

`rmlx metrics init` rejects re-init if file exists. `rmlx metrics doctor` reads `schema_version`, applies pending migrations from `crates/rmlx-metrics/migrations/NNN_<name>.sql` in order. Each migration must be idempotent within a transaction.



### 3.1 `prompts`

Prompt registry. Referenced from `observations.prompt_id`. Avoids embedding 4 KB prompt text in every observation row and tracks prompt evolution over time.

```sql
CREATE TABLE prompts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256          TEXT    NOT NULL UNIQUE,        -- hex of SHA-256(body)
    name            TEXT    NOT NULL,               -- human label, e.g. "longctx_4k"
    body            TEXT    NOT NULL,               -- full prompt text
    tokens_approx   INTEGER,                        -- tokenizer-agnostic estimate (whitespace split)
    first_seen_utc  TEXT    NOT NULL,               -- ISO-8601 UTC
    notes           TEXT                            -- free text: source, intent, change reason
);
CREATE INDEX prompts_name_idx ON prompts(name);
```

**Field semantics**

| Field            | Meaning                                                                                  |
|------------------|------------------------------------------------------------------------------------------|
| `id`             | Surrogate key. Used as FK from `observations.prompt_id`.                                 |
| `sha256`         | Content hash. Idempotency key — re-inserting same prompt body returns existing id.       |
| `name`           | Human-readable label. NOT unique — same name can have multiple revisions over time.      |
| `body`           | Full prompt text, exact bytes. Stored once, referenced N times.                          |
| `tokens_approx`  | Whitespace-split count for quick reference. Real tokenizer count goes on `observations`. |
| `first_seen_utc` | When this exact body was first inserted. Lets us track prompt drift.                     |
| `notes`          | Why this prompt exists, what changed vs prior version, source URL if any.                |

### 3.2 `observations` (every measurement, append-only)

Ground-truth append-only table. Every metric value from every run lands here, with full run context. `bests` is derived from this. Worse-than-current-best observations are inserted unchanged — they're how regression detection and time-series visualization work.

```sql
CREATE TABLE observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    -- cell identity
    backend          TEXT    NOT NULL,             -- 'rmlx', 'mlx_lm', 'paroquant', 'omlx', 'ollama', ...
    model_namespace  TEXT    NOT NULL,             -- 'mlx-community', 'z-lab', 'prism-ml', 'ollama', 'hf', 'local'
    model            TEXT    NOT NULL,             -- short name within namespace, e.g. 'gemma-4-e2b-it-mxfp8'
    weight_quant     TEXT    NOT NULL,             -- 'mxfp8', 'mxfp4', 'q8_0', 'paro', '2bit', 'bf16', ...
    kv_quant         TEXT    NOT NULL,             -- 'none', 'k8v4', 'k8v8', 'planar', 'turbo4', ...
    ctx_max          INTEGER NOT NULL,             -- server max-ctx setting at run time
    prompt_id        INTEGER NOT NULL REFERENCES prompts(id),
    metric           TEXT    NOT NULL,             -- see §4 metric registry
    decode_config    TEXT,                         -- non-default engine configuration; NULL = every setting at its default, e.g. 'mtp/block=5', 'prefill_chunk=1024' (migration 005)
    -- value
    value            REAL    NOT NULL,             -- numeric measurement
    unit             TEXT    NOT NULL,             -- 'tps', 'ms', 'mb', 'bytes', 'count', 'ratio'
    direction        TEXT    NOT NULL              -- 'higher_better' | 'lower_better'
        CHECK (direction IN ('higher_better','lower_better')),
    -- run context (the run that produced THIS observation)
    run_id           TEXT    NOT NULL,             -- minted at DB write: '<YYYYMMDDHHMMSS>-<6hex>'
    ts_utc           TEXT    NOT NULL,             -- when measurement was taken (ISO-8601 UTC)
    git_sha          TEXT,                         -- caller-supplied provenance, see §8.5.1; NULL unless a caller set it
    build_profile    TEXT,                         -- 'release', 'release+pgo', 'debug'
    backend_version  TEXT,                         -- semver only, e.g. '0.0.1'
    hardware_tag     TEXT    NOT NULL,             -- 'm5_max_128gb' — context, no special invalidation logic
    -- bench config
    prompt_tokens    INTEGER,
    max_tokens       INTEGER,
    temperature      REAL,
    seed             INTEGER,
    n_warmups        INTEGER,
    n_measure        INTEGER,
    -- side data
    output_first_64  TEXT,                         -- first 64 chars/words of decode (equality probe)
    decode_stddev    REAL,                         -- if metric=decode_tps_*, paired stddev
    notes            TEXT,                         -- machine-recorded auto-summary (from bench script)
    description      TEXT,                         -- written by human OR Claude: why this run, what changed
    -- bookkeeping
    inserted_utc     TEXT    NOT NULL,
    inserted_by      TEXT    NOT NULL              -- audit: tool@semver
);

CREATE INDEX obs_cell_idx       ON observations(backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id, metric, decode_config);
CREATE INDEX obs_metric_idx     ON observations(metric);
CREATE INDEX obs_ts_idx         ON observations(ts_utc);
CREATE INDEX obs_git_sha_idx    ON observations(git_sha);
CREATE INDEX obs_run_id_idx     ON observations(run_id);
CREATE INDEX obs_backend_idx    ON observations(backend);
CREATE INDEX obs_inserted_idx   ON observations(inserted_utc);
```

**`decode_config` is cell identity, not context.** A speculative-decode arm and
a plain-decode arm of one model at one quant and one prompt are not two
measurements of the same thing — the drafter changes what produced the tokens,
and ranking one against the other publishes the drafter's rate as the model's
decode throughput. It is `NULL` for ordinary decode, which is also what every
row written before migration 005 carries, so legacy plain-decode rows keep
their cells unchanged. The population that column cannot sort out — speculative
rows written before it existed — is named in §4.1.

**It is the general discriminator for a non-default engine configuration, not
a speculative-only field.** The drafter was the first setting to need one, but
the argument is not about drafters: any engine setting a run deliberately moved
off its default produces measurements that are not alternatives to the default
configuration's, and a column per setting would mean a migration, a `bests`
partition change and an edit to every consumer in `cell.rs` each time one
appears. A prefill-chunk sweep is the same shape as a drafter arm — its cells
are `prefill_chunk=<n>` — and lands here for the same reason. Settings that
merely describe the run rather than change what was measured (`ctx_max`,
`kv_quant`, `prompt`) have their own columns and do not belong here.

**Grammar.** Because it is cell identity, the spelling is a contract: two
emitters describing one configuration in two spellings put its measurements in
two cells, where neither ranks against the other and both look like champions.

```
decode_config := term ("," term)*
term          := key "=" value
key           := segment ("/" segment)*
segment       := [a-z0-9_]+
value         := [A-Za-z0-9_.+-]+
```

No whitespace anywhere, and terms are **strictly ordered by key** — that
ordering is what makes one configuration one string. A key may carry a `/`-path
when the setting belongs to a named subsystem, which is what makes the
speculative arm's `mtp/block=5` a term of this grammar rather than an
exception to it. `NULL` is the engine at its defaults; the empty string is not
a spelling of that and is refused.

**Settings that currently use it.**

| Terms | Setting | Emitted by |
|---|---|---|
| `<drafter>/block=<n>` | speculative-decode arm and the block it was configured with | `rmlx_metrics::cell::decode_config`, composed by the round loop and logged on its `done` line |
| `<drafter>/depth=<policy>` | how the loop picks each round's block, when it does not simply take the configured one | same |
| `prefill_chunk=<n>` | non-default prefill chunk size | prefill-chunk sweeps |
| `kv_boundary/head=<h>,kv_boundary/tail=<t>` | `--kv-boundary-layers` off its default | `rmlx baseline --record`, `rmlx eval ppl`, `scripts/ingest/{codec_inertness,perf_ab}_ingest.py` |

The `<drafter>/depth` term is absent when the loop drafts the configured block
every round. It is present when the loop resizes: DFlash halves and grows its
block from the recent accept rate, so its arm is
`dflash/block=16,dflash/depth=accept_rate` and is a different cell from a fixed
arm at block 16 — which it must be, because the two do not emit the same number
of tokens per round.

**DFlash has no fixed-block arm at all**, and never had: its production call
site has always passed `prefer_requested = false`, and the only caller passing
`true` is a unit test. So the bare `dflash/block=<n>` names a configuration that
has never run. `rmlx_metrics::cell::ADAPTIVE_DRAFTERS` is the one list of
drafters in that position — the engine's per-loop accessor reads it,
`decode_config_from_notes` reads it when recovering a row from notes,
`RunRecord::validate` refuses a record that contradicts it, and **migration
008** rewrote the eight rows (ids 122743–122750) that predated it. Rows for
every other drafter keep the cell they have always been in. The round loop composes both terms and puts the result on
its `done` line, and `scripts/spec_bench.sh` records what it finds there; a
bench script that spelled the string itself would file a run under a
configuration the engine did not use, which is the defect that motivated
moving it.

The `kv_boundary/*` pair is always written together and always in that order
(`head` sorts before `tail`), because a head count without a tail count does
not name a boundary. `NULL` is the shipped default, which is what keeps a
default run ranking against every row recorded before the flag existed. The two
Python ingesters derive the terms from the run's own recorded arguments — the
probe's `kv_boundary` CSV column and `perf_ab.sh`'s per-arm `args` string —
rather than from a flag on the ingester, so the term cannot describe a
configuration the run did not use, and they read the default itself from
`rmlx_core::kv_boundary` rather than restating it
(`scripts/lib/kv_boundary_default.py`, gated by
`make check-kv-boundary-default-parity`).

**What does not belong here: a change to what the number means.** `rmlx eval
ppl` has two scorers — a cacheless full-window forward and a cache-bearing
teacher-forced one — and they do not measure the same quantity. That is a
*metric*, `ppl_<corpus>` against `ppl_<corpus>_cached`, not a `decode_config`
term. Putting it here would have fenced both off from every `mlx_lm` row, which
can never carry a term this engine invented. The test is the one stated above:
a setting the engine moved off its default belongs here; a different
measurement is a different metric.

**One producer of the drafter terms, not of the column.**
`rmlx_metrics::cell::decode_config` is the only site that composes a
`<drafter>/block` or `<drafter>/depth` term, and `scripts/spec_bench.sh` records
the string the engine logged rather than spelling it again. The column has other
composers — `rmlx_models::kv_cache`'s boundary terms,
`scripts/ingest/perf_ab_ingest.py`, `scripts/ingest/codec_inertness_ingest.py`
and `scripts/prefill_chunk_sweep.sh` — whose *values* are held together by
`check-kv-boundary-default-parity` but whose *format* is not.

`rmlx_metrics::cell::decode_config_is_well_formed` is the one implementation,
enforced at ingest (`RunRecord::validate`) so a private spelling is rejected
rather than stored. `bests`, `rmlx metrics best` / `compare` / `history` /
`timeseries` / `deltas` and the CSV / Markdown exports need nothing beyond
this: they already partition on the whole cell key from `CELL_COLUMNS`, so a
new setting's cells separate the moment its rows carry the term.

**No PK on the cell columns.** A cell can have N observations over time — that's the whole point. PK is the surrogate `id`.

**Run grouping**: every observation from the same `record` invocation shares the same `run_id`. Lets us reconstruct "this run produced these N metrics" trivially: `WHERE run_id = '...'`.

### 3.3 `bests` (VIEW — champion per cell, derived)

```sql
CREATE VIEW bests AS
WITH ranked AS (
    SELECT
        o.*,
        ROW_NUMBER() OVER (
            PARTITION BY backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id, metric, decode_config
            ORDER BY
                CASE WHEN direction = 'higher_better' THEN  value END DESC,
                CASE WHEN direction = 'lower_better'  THEN -value END DESC,
                ts_utc DESC                           -- tie-breaker: newer wins
        ) AS rn
    FROM observations o
    WHERE CASE metric                                 -- §4.1 plausible-value bounds
              WHEN 'decode_tps_warm' THEN (value > 0.0 AND value <= 10000.0)
              WHEN 'prompt_cache_hits' THEN (value >= 0.0 AND value <= 1e12)
              -- … one branch per §4 metric, generated from the registry …
              ELSE 1
          END
)
SELECT * FROM ranked WHERE rn = 1;
```

The `WHERE` is **generated from the §4 registry**, not written by hand:
`rmlx_metrics::bests_view::plausible_sql` renders one branch per metric from
the same `Bounds` the ingest validator enforces. Retyping the bound in SQL is
how the view and the gate would drift apart, so every consumer that ranks or
aggregates `observations` in its own SQL — `query::deltas` (the
`--exit-code` CI gate), `query::regress`, `query::timeseries` — `AND`s in the
same generated predicate rather than carrying its own copy.

Who installs it: `migrate::run_pending` (i.e. every *writer*) recreates the
view whenever the stored definition and the registry disagree, and
`rmlx metrics doctor --fix` does so explicitly. A **read** command never
rebuilds it — `schema::open_checked` refuses to run against a DB whose view
was built from a different registry and names the repair. A query must not
change what the champion table publishes as a side effect of being run.

Rows outside the bound are **excluded, not re-ranked**: the cell falls to the
best plausible runner-up, and disappears from `bests` when there is none. That
is the intended outcome — an absent anchor is honest, an implausible one is
not. The ranking has no plausibility floor of its own, so without this filter
the largest number in a partition wins by construction, which is how
`prefill_tps = (prompt_tokens - 242) * 1000` rows published as champions.

Every column from `observations` is preserved on the champion row, so callers see the full run context that set the record (git_sha, description, etc.).

Tie-breaking by `ts_utc DESC` means equal-best observations resolve to the most recent — important for description freshness.

Performance: scan of `observations` per query. With the cell index above, even at 1M rows the GROUP BY/ROW_NUMBER plan stays under 100ms. If we ever exceed that, materialize `bests` as a trigger-maintained table — drop-in swap, same surface.

**Field semantics**

Identity (PK):

| Field             | Meaning                                                                                    |
|-------------------|--------------------------------------------------------------------------------------------|
| `backend`         | Engine binary. Lowercase, no version. Versions go in `backend_version`.                    |
| `model_namespace` | Source / manufacturer of the model. Tracks who repackaged or trained it. See §5.1.         |
| `model`           | Short name within the namespace — strip path and namespace prefix.                         |
| `weight_quant`    | Weight quantization scheme — what's on disk. `bf16` if unquantized.                        |
| `kv_quant`        | KV-cache quantization at runtime. `none` for unquantized.                                  |
| `ctx_max`         | Server-side max context. Different ctx = different cell because KV cache shape changes.    |
| `prompt_id`       | FK to `prompts`. Different prompt ⇒ different cell — TPS isn't comparable across prompts.  |
| `metric`          | Canonical metric name from §4. One row per metric per cell.                                |

Value:

| Field       | Meaning                                                                  |
|-------------|--------------------------------------------------------------------------|
| `value`     | The number. Always SI / human units, never normalized.                   |
| `unit`      | Unit string from §4 registry. Trust the registry, do not invent units.   |
| `direction` | Optimization direction. UPSERT trigger uses this to compare new vs old.  |

Run context (snapshot):

| Field             | Meaning                                                                                       |
|-------------------|-----------------------------------------------------------------------------------------------|
| `run_id`          | Minted at DB write time. Format `<YYYYMMDDHHMMSS>-<6hex>`. Not a FK, just a tracking string.  |
| `ts_utc`          | When the measurement was taken (ISO-8601 UTC). Distinct from `inserted_utc`.                  |
| `git_sha`         | Caller-supplied provenance, not derived by the binary — a bench script's own `git rev-parse`, or `--git-sha` on `rmlx baseline` / `rmlx eval ppl`. `NULL` unless a caller set it (e.g. every drainer/server-produced row). Some historical rows predating this contract carry a `-dirty` suffix that `deltas --since-sha` still matches; the binary never mints that suffix now. See §8.5.1. |
| `build_profile`   | Cargo profile + flags hint. `release`, `release+pgo`, `release+lto`, `debug`.                 |
| `backend_version` | Semver only, e.g. `0.0.1`, `0.21.0`. The backend name lives in `backend`.                     |
| `hardware_tag`    | Hardware identifier, e.g. `m5_max_128gb`. Treated as a regular context column — different value = different cell, no special "invalidation" semantics. Mint a new tag when migrating to new hardware (`m5_ultra_256gb`). |

Bench config:

| Field           | Meaning                                                                     |
|-----------------|-----------------------------------------------------------------------------|
| `prompt_tokens` | Actual tokenizer count of the prompt at run time (not whitespace estimate). |
| `max_tokens`    | Hard cap on generated tokens.                                               |
| `temperature`   | Sampling temperature. 0 for deterministic equality checks.                  |
| `seed`          | Sampler seed.                                                               |
| `n_warmups`     | Warmup runs discarded before measurement.                                   |
| `n_measure`     | Number of measurement runs averaged into `value`.                           |

Denormalized side data:

| Field            | Meaning                                                                                    |
|------------------|--------------------------------------------------------------------------------------------|
| `output_first_64`| First N words/chars of generated text. Used to verify temp=0 equality across commits.      |
| `decode_stddev`  | Stddev of the N measurement runs. Only populated for `decode_tps_*` metrics.               |
| `notes`          | Machine-written by the bench script (auto-summary, e.g. `"step_ms_mean=wall/completion_tokens"`). |
| `description`    | Written by **human OR Claude** after analyzing the run. Why this is better/worse, what changed, which commit/report explains it. See §6. |

Bookkeeping:

| Field           | Meaning                                                                       |
|-----------------|-------------------------------------------------------------------------------|
| `inserted_utc`  | When this DB row was created or last updated. `ts_utc` is the measurement time, this is the row write time. |
| `inserted_by`   | Tool that wrote the row, with version: `rmlx-cli@0.0.1`, `cbb-runner@0.5.3`, `migrate@0.0.1`. Audit trail for debugging anomalous rows. |

**Removed columns** (vs first draft, with rationale):
- `git_dirty` (BOOLEAN) → folded into `git_sha` as `-dirty` suffix. One field, no boolean logic.
- `host` (TEXT) → single machine for now. Re-add only if multi-machine becomes real.
- `source_path` (TEXT) → traceability via (`run_id`, `ts_utc`, `git_sha`); JSONL/CSV go away after migration.

### 3.4 NULL policy (sparse rows are normal)

Different backends measure different things. ollama doesn't expose `metal_peak_alloc_mb`. mlx_lm doesn't return `kv_cache_bytes`. Historic JSONL didn't have `decode_stddev`. Schema must tolerate this.

Applies to `observations` (and inherited by `bests`).

**NOT NULL columns** (always required):
- All cell columns: `backend`, `model_namespace`, `model`, `weight_quant`, `kv_quant`, `ctx_max`, `prompt_id`, `metric`.
- Value identity: `value`, `unit`, `direction`.
- Provenance: `run_id`, `ts_utc`, `hardware_tag`, `inserted_utc`, `inserted_by`.

**Nullable columns** (backend may not provide):
- `git_sha`, `build_profile`, `backend_version`.
- `prompt_tokens`, `max_tokens`, `temperature`, `seed`, `n_warmups`, `n_measure`.
- `output_first_64`, `decode_stddev`, `notes`, `description`.

**Sparse-metrics rule**: if a backend doesn't measure a given metric, the recorder writes **no row** for that (cell, metric) pair. NEVER write a row with `value=NULL`. "No row" is the canonical "not measured" signal — pivot queries handle it via `MAX(CASE WHEN metric=…)` returning NULL.

`bests`/Grafana queries should always tolerate missing rows for a given metric in a given cell. The champion-view export (§9) renders missing cells as blank, not as `0` or `N/A` (reserve `N/A` for "backend doesn't support this configuration", set via `description`).

### 3.5 Why no triggers / no UPSERT

Earlier drafts had a `bests_strictly_better_only` trigger and an `INSERT OR REPLACE` recording path. **Both removed.**

- Recorder INSERTs into `observations` unconditionally. No write-time comparison.
- Champions are derived at READ time by the `bests` view (§3.3 ROW_NUMBER over partitions).
- "Strictly better" semantics are now encoded in the view's `ORDER BY direction`, not in DDL.

Net effect from the user's perspective:
- `bests` still returns one champion per cell.
- `observations` exposes the full history for free (Grafana / regression detection).
- No trigger code to debug, no UPSERT race conditions, no risk of `bests` getting out of sync with reality.

Tradeoff: every observation costs storage. Acceptable — see retention §10.2.

### 3.6 `events` table (runtime per-event stream)

Schema migrations `002_events.sql` + `003_events_identity.sql` +
`004_events_mlx_nax.sql`. Written by `rmlx_metrics::events::EventRecorder`.
One row per runtime event; append-only; WAL absorbs concurrent writers.

**Columns:** `id` (INTEGER PK), `run_id` (TEXT), `ts_utc` (TEXT), `model_path`
(TEXT), `quant_mode` (TEXT), `stage` (TEXT), `op` (TEXT), `value_unit` (TEXT),
`value` (REAL), `notes` (TEXT), plus the run-identity columns added by 003:
`backend_version` (TEXT NULL), `build_profile` (TEXT NULL), and by 004:
`mlx_nax` (TEXT NULL).

**Identity.** `backend_version` and `build_profile` are stamped from the same
`RunIdentity` source `observations` uses (§8.5.1) — the binary genuinely
knows both about itself, which makes an `events` row self-describing on its
own. **`events` has no `git_sha` column.** Unlike `observations.git_sha`
(caller-supplied provenance: a bench script's own `git rev-parse`, or
`--git-sha` on `rmlx baseline` / `rmlx eval ppl`), `events` is written only by
the in-process `EventRecorder` — there is no `rmlx serve` flag or script that
could ever supply a commit SHA for it, so migration `003_events_identity.sql`
deliberately does not add the column. Before migration 003 `events` carried
no identity at all; rows written before 003 keep NULL for `backend_version` /
`build_profile`, and the table is append-only, not backfilled.

An earlier draft of migration 003 briefly did add `events.git_sha` before
this contract settled; a small number of already-migrated databases may still
carry that stray, permanently-`NULL` column from before the migration was
amended. It is harmless — nothing selects it (every query names its columns
explicitly) — and is not something to "fix" by hand-editing that database's
schema.

**`mlx_nax` (migration `004_events_mlx_nax.sql`).** Whether the MLX this
process **loaded** ships the `steel_gemm_fused_nax*` GEMM kernels:
`"present"` / `"absent"` / `"unknown"`. On Neural-Accelerator-class hardware
(M5-family and later) their absence costs ~3.8x GPU-matmul throughput and
2.2-3.7x slower prefill — decode is bandwidth-bound and unaffected, which is
exactly why this silently passed for weeks (see
`.rmlx/mlx-homebrew-nax-regression.md`). A bench row is otherwise unable to
say whether it ran against a nax-capable build.

Same "binary genuinely knows this about itself" category as
`backend_version` / `build_profile`, but the fact originates in a different
crate: `rmlx_mlx::nax_capability()` scans the `mlx.metallib` beside the
`libmlx.dylib` **dyld resolved for this process**, once per process. It is
read at run time, not baked in at compile time, because a binary links MLX
through a package-manager `opt` symlink — so it loads whatever the installing
user has, and that symlink can move after the build on the same machine.
Anything stamped by a build script answers for the machine that compiled the
binary, which is the wrong machine, and cargo cannot even re-run that script
when the symlink moves backwards.

`rmlx-metrics` cannot call it directly: it deliberately does not depend on
`rmlx-mlx` (whose build script hard-requires a working Homebrew MLX/mlx-c
install — a dependency this generic, cross-backend metrics crate must not
carry). `rmlx-cli::main()` is the one binary that links both, so it calls
`rmlx_metrics::identity::set_mlx_nax(rmlx_mlx::nax_capability())` once at
startup — the same process-wide one-shot-`OnceLock` pattern already used for
`install_rotor_qjl` / `install_planar_fused_qk` — before the first
`RunIdentity::get()` / `EventRecorder::record`. A process that never calls it
(unit tests, tools that don't link `rmlx-mlx`) reads `"unknown"`, which is
honest: no capability was ever supplied. `"unknown"` also covers a metallib
that could not be inspected — "did not look" is never recorded as either
presence or absence. Free-form TEXT, not validated against an enum — same rule
as every other recorded label (kv_quant/model, #214).

Nullable: rows written before migration 004 keep NULL. Append-only — no
backfill, no UPDATE. Historical rows recorded roughly 2026-07-13 through the
Homebrew fix landing may have run against the no-nax `mlx` 0.32.0 bottle with
degraded prefill numbers (decode unaffected) — see
`.rmlx/mlx-homebrew-nax-regression.md` for the suspect window and the
evidence. Those pre-migration rows are **not** retroactively annotated with
`mlx_nax` — they simply carry NULL, and doing otherwise would violate this
table's append-only/no-UPDATE contract; a bench row from before this column
existed has no honest way to state its nax capability after the fact.

**`stage` / `op` vocabulary:**

| `stage` | `op` value | When written |
|---|---|---|
| `admission_ctrl` | `admission_insufficient_data` | Tick fires but regressor has < 4 points. No depth change. |
| `admission_ctrl` | `admission_no_change` | ITL estimate within SLA bounds. Depth unchanged. |
| `admission_ctrl` | `admission_scale_down` | ITL estimate above target for 3 consecutive ticks. Depth decremented. |
| `admission_ctrl` | `admission_scale_up` | ITL estimate below deadband (80 % of target). Depth incremented. |
| `admission_ctrl` | `admission_anticipatory_503` | Per-request anticipatory rejection (503 returned to caller). Increments the `admission_sla_503` error counter. |
| `admission_ctrl` | `prefill_chunk_raise` | Adaptive prefill chunk raised (`--adaptive-prefill-chunk`). |
| `admission_ctrl` | `prefill_chunk_lower` | Adaptive prefill chunk lowered after 3 overload ticks. |
| `admission_ctrl` | `prefill_chunk_hold` | Adaptive prefill chunk evaluated, no change (deadband or hold gate). |

The `Disabled` reason (`admission_disabled`) is never written — the controller is
entirely inactive when `--adaptive-admission` is absent, producing zero events.
The three `prefill_chunk_*` ops are only written when `--adaptive-prefill-chunk` is also set.

`value` carries `est_itl` (ms) and `label` carries `itl_target` (ms) on tick events.
Both are `NaN` for `admission_insufficient_data` rows.

---

## 4. Metric registry (canonical)

Source of truth for `observations.metric`, `.unit`, `.direction`. Add to this table when introducing a new metric — never invent ad-hoc names in the recording code.

| `metric`              | `unit`  | `direction`     | What it measures                                                        |
|-----------------------|---------|-----------------|-------------------------------------------------------------------------|
| `decode_tps_warm`     | `tps`   | `higher_better` | Warm-cache decode tokens/sec, mean of measurement runs.                 |
| `decode_tps_cold`     | `tps`   | `higher_better` | First-run decode TPS (model just loaded, JIT not warm).                 |
| `prefill_tps`         | `tps`   | `higher_better` | Prefill tokens/sec on the prompt.                                       |
| `overall_tps`         | `tps`   | `higher_better` | (prompt_tokens + completion_tokens) / wall_time.                        |
| `ttft_cold_ms`        | `ms`    | `lower_better`  | Time-to-first-token, cold (first request after load). Kept for backward compat alongside `prefill_duration_ms`. |
| `ttft_warm_ms`        | `ms`    | `lower_better`  | Time-to-first-token, warm (subsequent requests). Kept for backward compat alongside `prefill_duration_ms`. |
| `itl_p50_ms`          | `ms`    | `lower_better`  | Inter-token latency, median. Kept for backward compat alongside `tpot_p50_ms`. |
| `itl_p95_ms`          | `ms`    | `lower_better`  | Inter-token latency, 95th percentile. Kept for backward compat alongside `tpot_p95_ms`. |
| `step_ms_mean`        | `ms`    | `lower_better`  | Mean per-token wall time (= `wall / completion_tokens`).                |
| `model_load_ms`       | `ms`    | `lower_better`  | Wall time from `serve` start to "ready" log line.                       |
| `peak_rss_mb`         | `mb`    | `lower_better`  | Peak resident set during the run.                                       |
| `metal_peak_alloc_mb` | `mb`    | `lower_better`  | Peak Metal device allocation over the **process lifetime** (rMLX: `rmlx_mlx::mlx_peak_memory_bytes`; other backends: `mx.metal.get_peak_memory()`). Deliberately unscoped, so rows stay comparable across backends that expose only the lifetime figure. The region-scoped variant (`rmlx baseline`'s `metal_gen_alloc_mb`, from `rmlx_mlx::PeakBracket`) is a stdout diagnostic for A/B work and is **not** recorded here — mixing the two definitions into one column would make the history meaningless. |
| `kv_cache_bytes`      | `bytes` | `lower_better`  | Live-inference KV resident bytes at end of generation: the *filled* prefix of the cache that actually serves decode (packed codes + scales + rotation/residual buffers, plus the per-position bf16/f32 decode buffers scaled to the filled length). Reads real Array shapes × dtype via `KvCache::resident_bytes`, but counts only `offset` positions of the seq-scaled buffers — the bf16/f32 decode mirrors are pre-allocated to the `--max-ctx` ceiling, so counting the whole allocation would inflate the figure and make bytes-per-KV-token depend on the ceiling rather than the prompt. Excludes any prompt-cache snapshot clone (held separately, never summed). Sampled at ONE lifecycle point on every arch — **after the decode loop**, when the decode-time GPU ring of a ring-backed codec is resident — so the figure is comparable across archs and across prompt-cache hit/miss. "One lifecycle point" means post-decode *and only when a decode ran*: an immediate-EOS run (no decode loop) does not refresh this metric and keeps the prior value — uniform across archs, and lossless since with no decode there is no ring. A NaN prefill is not in that category: it aborts the request outright, so there is no run to attribute a byte count to. The requirement is gated by a witness: `KvBytesCounter::store` takes a `PostDecode` minted only by a completed decode loop, so re-co-locating the store at the prefill snapshot fails to compile (the loop's witness is not in scope there). It is a raised bar + review convention backed by a manual-GPU re-drift test, not an unforgeable compile guarantee — `seal()` is `pub(crate)`. **Reading it back:** the "keeps the prior value" case above means a bare byte count cannot be attributed to a particular generation. `Architecture::kv_cache_bytes_sample()` therefore returns `(bytes, seq)`, where `seq` counts the stores on **that model instance** — the counter is a field on the model struct (`kv_bytes::KvBytesCounter`), not a per-arch static, so a second model of the same architecture generating concurrently cannot advance this one's sequence and have its byte count read back under this one's name. A caller that *records* the figure samples the pair before and after the generation and requires `seq` to have advanced; if it did not, the readable count belongs to an earlier generation (or is the unset `0` initialiser) and must be refused, not written. `Architecture::kv_cache_bytes()` returns the bare count and is for display surfaces (`/metrics/cache`) that have no generation boundary to check against — do not record from it. |
| `tps_per_gb_ram`      | `ratio` | `higher_better` | `decode_tps_warm / peak_rss_gb` — runtime efficiency.                   |
| `task_pass_at_1`      | `ratio` | `higher_better` | Quality probe pass rate (CBB-style). 0.0–1.0.                           |
| `prompt_cache_block_hits`        | `count` | `higher_better` | Cumulative 256-tok blocks served from a cached prefix.   |
| `prompt_cache_block_misses`      | `count` | `lower_better`  | Cumulative 256-tok blocks that had to be (re-)prefilled. |
| `prompt_cache_partial_hits`      | `count` | `higher_better` | Requests that matched a non-empty proper block prefix.   |
| `queue_wait_ms`                  | `ms`    | `lower_better`  | Milliseconds a request waited in the FIFO admission queue for the GPU permit. |
| `queue_depth`                    | `count` | `lower_better`  | In-flight admitted-request count observed at admission (this request inclusive). |
| `prompt_tokens_live`             | `count` | `lower_better`  | Per-request prompt (input) token count from live HTTP handler. Suffixed `_live` to distinguish from bench-config metadata columns. rmlx-only. |
| `completion_tokens_live`         | `count` | `lower_better`  | Per-request completion (output) token count from live HTTP handler. rmlx-only. |
| `itl_p99_ms`                     | `ms`    | `lower_better`  | Inter-token latency 99th percentile per request. Emitted alongside itl_p50_ms/itl_p95_ms. rmlx-only. |
| `itl_spikes`                     | `count` | `lower_better`  | Count of ITL intervals exceeding 3×median per request (spike threshold = 3×p50). Diagnostic for GC pauses and Metal pipeline stalls. rmlx-only. |
| `accept_rate`                    | `ratio` | `higher_better` | Speculative decoding: `accept_tokens_total / draft_tokens_total` (0.0–1.0). Per request. rmlx-only (MTP / DFlash / Eagle3 drafters). |
| `draft_tokens_total`             | `count` | `higher_better` | Speculative decoding: cumulative draft-proposed token count over the request. rmlx-only. |
| `accept_tokens_total`            | `count` | `higher_better` | Speculative decoding: cumulative verifier-accepted token count over the request. rmlx-only. |
| `draft_rounds_total`             | `count` | `higher_better` | Speculative decoding: number of verifier rounds (one round = drafter proposes block, verifier accepts prefix). rmlx-only. |
| `accepted_per_step`              | `ratio` | `higher_better` | Speculative decoding: `accept_tokens_total / draft_rounds_total` (mean accepted tokens per verifier step). rmlx-only. |
| `tokens_per_round`               | `ratio` | `higher_better` | Speculative decoding: tokens the **rounds** produced per round, counted at the loops' own emit sites. The sidecar loops emit one bonus token out of the prefill forward before the first round and it is excluded, or their rows would read `+1/rounds` above a two-model row that did the same work (measured +1.35% and +0.98%). `1 + accept_rate x (block - 1)` only while every round drafts the configured block, which an adaptive drafter does not, so it is recorded and not derived. rmlx-only. |
| `draft_ms_per_round`             | `ms`    | `lower_better`  | Speculative decoding: wall clock inside the drafter call, per round. rmlx-only. |
| `verify_ms_per_round`            | `ms`    | `lower_better`  | Speculative decoding: wall clock inside the verify forward, per round. rmlx-only. |
| `loop_ms_per_round`              | `ms`    | `lower_better`  | Speculative decoding: the round loop's own overhead per round — rollback, snapshot and restore, acceptance walks, sampling. A residual: the three `*_ms_per_round` partition one round's wall clock. rmlx-only. |
| `ssd_bytes_used`                 | `bytes` | `lower_better`  | SSD-tier: current on-disk KV-block cache footprint per namespace. Unbounded growth is a budget risk; LowerBetter keeps regression gate alert on runaway accumulation. rmlx-only. |
| `ssd_evict_total`                | `count` | `lower_better`  | SSD-tier: lifetime LRU eviction count. More evictions = more cache thrash. rmlx-only. |
| `ssd_spill_ms`                   | `ms`    | `lower_better`  | SSD-tier: raw per-spill duration observation (drain thread, off-hot-path). One SQLite row per event. Real p50/p99 aggregation via Prometheus histogram `rmlx_ssd_spill_us_bucket`. rmlx-only. |
| `ssd_hydrate_ms`                 | `ms`    | `lower_better`  | SSD-tier: raw per-hydrate duration observation (on request thread, RAM-miss cold path). One SQLite row per event. Real p50/p99 aggregation via Prometheus histogram `rmlx_ssd_hydrate_us_bucket`. rmlx-only. |
| `ssd_spill_mb_per_s`             | `mb/s`  | `higher_better` | SSD-tier: spill throughput (bytes_written / dur_us). rmlx-only. |
| `ssd_hydrate_mb_per_s`           | `mb/s`  | `higher_better` | SSD-tier: hydrate throughput (bytes_read / dur_us). rmlx-only. |
| `ppl_wikitext2`                  | `ppl`   | `lower_better`  | Sliding-window perplexity over the wikitext-2 raw test split, computed by `rmlx eval ppl`. Architecture support: Qwen3, Gemma4, Qwen3.5. rmlx-only. |
| `ppl_mean_nll`                   | `nat`   | `lower_better`  | Per-token mean negative log-likelihood from the same scorer (natural-log nats). Audit field paired with `ppl_*`. rmlx-only. |
| `ppl_scored_tokens`              | `count` | `higher_better` | Number of corpus positions scored. Audit field. rmlx-only. |
| `ppl_windows`                    | `count` | `higher_better` | Number of sliding-window forwards the scorer ran. Audit field. rmlx-only. |
| `ppl_score_ms`                   | `ms`    | `lower_better`  | Wall-clock spent in the scorer (excludes model load). rmlx-only. |
| `prefill_duration_ms`            | `ms`    | `lower_better`  | Wall-clock from `generate(...)` entry to first OK token. Numerically equal to the request's `ttft_warm_ms` / `ttft_cold_ms` row but emitted under its own canonical op so honest stage attribution does not require parsing `op` for a `ttft` prefix. Phase=`Prefill`. rmlx-only. |
| `tpot_p50_ms`                    | `ms`    | `lower_better`  | Time-Per-Output-Token, median over decode-only intervals (post-first-token gaps). In v1 numerically identical to `itl_p50_ms` (both definitions skip the first interval) — named separately for the TPOT convention + future divergence (e.g. tool-call stalls). Phase=`Decode`. rmlx-only. |
| `tpot_p95_ms`                    | `ms`    | `lower_better`  | TPOT 95th percentile over decode-only intervals. Mirrors `itl_p95_ms` in v1. Phase=`Decode`. rmlx-only. |
| `tpot_p99_ms`                    | `ms`    | `lower_better`  | TPOT 99th percentile over decode-only intervals. Mirrors `itl_p99_ms` in v1. Phase=`Decode`. rmlx-only. |

**Dropped metrics** (vs first draft): `model_disk_gb` (not a perf metric, derivable from `du -sh <model_path>` on demand) and `tps_per_gb_disk` (depends on disk_gb).

### 4.1 Plausible-value bounds

Every registry entry also carries a `Bounds` — the window of values that can be
a *measurement* of that metric. Per-metric values live in
`crates/rmlx-metrics/src/registry.rs` (`METRICS`); this section is the policy
they follow.

Each metric counts, times or rates a physical quantity, so the floor is always
`0.0` and a negative value is never a measurement. What differs is whether the
floor is itself a measurement:

| Family | Floor | Ceiling | Why |
|---|---|---|---|
| Rates (`tps`, `mb/s`, `tps_per_gb_ram`) | `0` **excluded** | 1e4–1e6 | `tokens / seconds` is zero only when no token was produced — nothing was measured. |
| Durations (`ms`) | `0` included | 3.6e6 (1 h) | Millisecond resolution rounds a sub-ms span to zero. A single span past an hour is a hung run. |
| Counters (`count`) | `0` included | 1e12 | Zero cache hits is a real observation. |
| Gauges (`mb`, `bytes`) | `0` included, except `peak_rss_mb` | 1e9 MB / 1e13 B | A live process always has RSS; a run can genuinely allocate no Metal. |
| Ratios (`ratio`) | `0` included | 1.0 (or 1e3 for `accepted_per_step` / `tokens_per_round`) | Rates of acceptance are honestly zero, and a per-round count is not a fraction. |

Ceilings are deliberately loose — several × the best value ever recorded — so
they reject fabrications, not fast machines. NaN and infinities are outside
every window.

**What a bound can and cannot catch.** It rejects a value orders of magnitude
out of range; it cannot detect a wrong value that lands inside the range. The
fabricated `(prompt_tokens - 242) * 1000` form below stays under the
`prefill_tps` ceiling for any prompt shorter than ~342 tokens. Bounds are
defence in depth behind a correct producer, never a substitute for one.

Three places enforce the same bounds, from the same table:

1. **Ingest** — `RunRecord::validate` rejects the whole record with
   `ImplausibleValue`. An emitter with no measurement must send `null`; the
   recorder writes no row for a null and that is the supported way to say
   "not measured". A placeholder number is not.
2. **`bests`** — the view does not rank rows outside the bound (§3.3).
3. **`rmlx metrics doctor`** — check 6b reports rows already in the DB
   (§10.4). A *warning*, not an error: `observations` is append-only, so those
   rows cannot be deleted or corrected, and failing on them would make every
   `make ci` run permanently red — a stuck light, not a gate. The gate that
   fails is (1); 6b reports what it would now refuse.

**Where bounds cannot decide.** A placeholder that is indistinguishable from a
real measurement *as a number* has to be dropped where the convention is known,
not here. CBB's `summary.csv` writes `task_pass_at_1 = 0.0` when it ran no
quality probe, and `0.0` pass@1 is also a legitimate score for a model that
failed every task — so `migrate::legacy` drops that column's zero at the parse
site, and the §4.1 bound for `task_pass_at_1` deliberately admits `0.0`.

**Archive converters are the one exception.** `rmlx metrics migrate` replays
another tool's CSV/JSONL exports, which write `0.0` in a column they never
measured. Those entries are dropped rather than failing the whole historical
run, and counted in the migrate report as `metrics_dropped_implausible`.

#### Known-bad rows already in the DB

Rows written before these gates existed are still there — `observations` is
append-only and nothing is deleted. Three known populations. The first two fall
outside the plausible-value bound, so `bests` already drops them and `doctor`
reports them. The third does not: it is the wrong-but-in-range case §4.1 says a
bound cannot reach, so the predicate that identifies it is written down here
because nothing enforces it.

- **`prefill_tps` ≥ 1e5** — 20 rows from a legacy buffer replay storing
  `(prompt_tokens - 242) * 1000` under `unit='tps'`. The producing script
  derived `prefill_s` from `elapsed_s - n_comp / (n_comp / elapsed_s)`, which
  is identically zero, so it always hit a `0.001 s` floor and reported
  `prompt_tokens × 1000` as a rate. `value >= 1e5` identifies them; the
  `notes` string they carry does not, since the emitting code no longer writes
  it.
- **rate metrics `= 0.0`** — an early-stopped run recording a fabricated zero
  instead of nothing. These win any cell whose rows are all zeros, so an
  upper-only bound would have *promoted* them; the bound has to be two-sided.
- **`decode_tps_warm` from a whole-request or prefill-inclusive stopwatch** —
  several bench scripts wrote this column from a rate whose window started
  before the prompt prefill, which is `overall_tps` under another metric's name
  and reads low by whatever the prefill cost. The producers:
  `scripts/spec_bench.sh` on both arms (the speculative one divided the round
  loop's `emitted` by its `elapsed_ms`, 35-62% low on the 4k-prompt runs still
  in `<RMLX_HOME>/logs`; the no-drafter one divided the completion tokens by the
  whole curl request, 9.6% low on the 14-token prompt those rows used);
  `scripts/perf-iter/bench_decode_tps.sh`; and six `scripts/bench/` campaign
  drivers now deleted (`t1`/`t2`/`t3_final_bench.sh`,
  `fullctx_regression_bench.sh`, `gemma_matrix_bench.sh`,
  `final_matrix_bench.sh`). Some of the `spec_bench` rows win their `bests` cell
  today.

  Every one of these producers now takes the rate from the engine and records
  `decode_window=` in `notes` — `engine_round_loop` for a speculative round
  loop's own figure, `engine_itl` for the server's per-request inter-token
  aggregate. That marker is a **positive** provenance claim, so its absence is
  "no window was recorded", not "the value is wrong": rows from
  `rmlx baseline`-driven producers (`ingest/perf_ab_ingest.py`, `bench_cell.sh`)
  and `llama-bench` token-generation rows are prefill-excluded at the source and
  also lack it. The predicate that identifies the population above is therefore
  producer-scoped, not marker-scoped alone:

  ```sql
  SELECT * FROM observations
  WHERE metric = 'decode_tps_warm'
    AND (notes IS NULL OR notes NOT LIKE '%decode_window=%')
    AND description LIKE 'spec_bench%';
  ```

  The `perf-iter` and deleted-campaign rows carry no `description` to key on;
  they are identified only by `ts_utc` predating this change, which is why the
  marker exists from here on. Re-measuring out-ranks any of them on merit only
  where the corrected number is larger.
- **`spec_bench.sh` rows labelled `kv_quant = 'k8v8'` with `prompt_tokens = 14`**
  — the same rows, from the other direction. That script wrote both as
  constants: it started its server with no `--kv-quant` and recorded `k8v8`
  regardless, while the engine resolved `none` and said so in its startup log,
  and it recorded a 14-token prompt for all three prompt files it is run with.
  So those rows are filed under a codec the run did not use, at a prompt length
  it did not have, and no re-measurement can out-rank them because a correctly
  labelled run lands in a different cell. Both fields are now read back from the
  run — the codec from the `cache-type resolved` event, the length from the
  response's `usage.prompt_tokens` — and a run reporting neither is refused
  rather than recorded. `k8v8` is a codec a run can legitimately use and 14 is a
  length a prompt can legitimately have, so the constants alone do not identify
  the population — the predicate is bounded by the same provenance marker as the
  row above:

  ```sql
  SELECT * FROM observations
  WHERE description LIKE 'spec_bench%'
    AND kv_quant = 'k8v8'
    AND prompt_tokens = 14
    AND (notes IS NULL OR notes NOT LIKE '%decode_window=%');
  ```

  No row written after this change matches: every row the script emits now
  carries `decode_window=`, whatever codec and prompt length it measured.
- **`ppl_wikitext2` rows from the first two cache-bearing sweeps** — the scorer
  gained a second mode (teacher-forcing each window through a real per-layer KV
  cache), and it took two attempts to file it correctly. The first sweep said
  nothing at all, putting a cacheless number and a bf16-cache number in one
  cell. The second said it in `decode_config` (`ppl/scorer=cached`), which is
  the wrong column: which scorer ran changes what the number *means*, not how
  the engine was configured, and a `decode_config` term additionally fences
  those rows off from every `mlx_lm` row, which can never carry a term this
  engine invented. Both are now `ppl_wikitext2_cached`, a metric of its own.
  Eighteen rows, `2026-09-03`/`2026-09-04`, on `Ternary-Bonsai-8B-mlx-2bit` and
  `gemma-4-12B-it-mxfp8` at `ctx_max = 2048`. No migration reaches the first
  batch: nothing in those rows says which scorer produced them, and guessing
  from the value is the substitution this list exists to refuse. They are
  superseded, not corrected — a correctly named re-measurement lands under a
  different metric and cannot out-rank them.

  ```sql
  SELECT * FROM observations
  WHERE metric = 'ppl_wikitext2'
    AND ts_utc >= '2026-09-03'
    AND prompt_id IN (SELECT id FROM prompts WHERE name LIKE 'wikitext-2_ctx2048%');
  ```

- **35 `ppl_*` rows on `Ternary-Bonsai-8B-mlx-2bit`, scored with a warm-up one
  slot too late** — the non-BOS scorer skipped `ctx_window - stride` leading
  slots where the slot that scores the first unseen corpus position is
  `ctx_window - stride - 1`, so exactly one corpus position per window boundary
  was never scored. `ppl_scored_tokens` is the metric that moved most: it *is*
  the denominator that changed, by `windows - 1`, and `ppl_mean_nll` and
  `ppl_windows` come off the same runs — which is why the predicate below covers
  the whole `ppl_` family and not just the headline metric.

  **Three conditions, all necessary**, and the selector is narrow because a
  predicate that flags good rows is one readers learn to ignore:

  - **Not Gemma4.** That scorer prepends BOS, which shifts each target one slot,
    and it already subtracted the one. It was always correct. 46 rows.
  - **`stride < ctx_window`.** At `stride == ctx_window` the old expression gives
    `ctx_window - stride = 0` and the new gives `0.saturating_sub(1) = 0`:
    identical, and the scored sets are byte-identical. The whole `2026-09-03`
    Bonsai batch is in that case. 16 rows.
  - **A pre-fix binary.** `git_sha` is the discriminator, not the date. This
    change was made on a branch, so a run from a pre-fix binary *after* the fix
    landed files an affected row that no date predicate catches; the four shas
    below are what the DB holds today, and a fifth would have to be added here
    rather than inferred. No affected row has a NULL `git_sha`.

  At the default `--ctx-window 4096 --stride 2048` the effect is one position in
  2048 — 0.05% of the denominator — and the resulting shift in `ppl` is far under
  any gate it has been used for. At `stride == 1` it was total: every window
  after the first scored nothing.

  ```sql
  SELECT * FROM observations
  WHERE metric LIKE 'ppl_%'
    AND model NOT LIKE 'gemma%'
    AND notes NOT LIKE '%ctx_window=' || ctx_max || ' stride=' || ctx_max || '%'
    AND git_sha IN ('2bcf206', '2bcf206-dirty', '6eeb4ae-dirty', 'a71d88b-dirty');
  ```

- **`dflash/*` rows on `Qwen3.8-27B-4bit`, measured against a DFlash 2
  checkpoint** — the drafter loader implements the earlier DFlash architecture
  and reads none of the candidate-selector or per-layer dynamic-convolution
  tensors a DFlash 2 snapshot ships. It used to build the drafter out of the
  rest and serve, so the rows are honest measurements of *that* drafter and not
  of the published one. `decode_config` says `dflash/block=N` either way and
  cannot tell them apart, which is why the loader now refuses such a snapshot
  outright: no further row of this kind can be written. The `2026-09-04`
  block-16 rows and the `2026-09-05` block-8 ones are the ones already here. Do
  not compare them against a row taken once the full drafter is implemented.
  `z-lab/Qwen3.6-35B-A3B-DFlash` reads every tensor it ships and is unaffected
  by the refusal.

  ```sql
  SELECT * FROM observations
  WHERE decode_config LIKE 'dflash/%'
    AND model = 'Qwen3.8-27B-4bit';
  ```

- **Two synthetic rows from an ingest-refusal probe** — a review of the
  boundary-layer work exercised the ingest path's refusals by handing it
  near-real records, and two of them were accepted instead of refused. They
  carry real identity, a real cell, and a placeholder `kv_cache_bytes` of
  `123456`, so **124693 currently wins `bests`** for
  `Ternary-Bonsai-8B-mlx-2bit / iso3_sym / 40960 /
  kv_boundary/head=2,kv_boundary/tail=4` on the smallest-cache ranking. Nothing
  measured them. Deleting rows is an ask-before item and the table is
  append-only, so they stay and are named:

  ```sql
  SELECT * FROM observations WHERE id IN (124693, 124694);
  ```

  The general fix is on both sides. `RunRecord::validate` now refuses any record
  whose `notes` or `description` carries
  [`rmlx_metrics::ingest::SYNTHETIC_MARKER`] (`synthetic=true`), so a probe can
  declare itself and be turned away before a transaction opens. And a probe that
  only needs to know *whether* a record would be accepted does not need a record
  at all: `rmlx metrics record --dry-run` runs the whole of `validate` and
  returns before the commit — verified to leave both the row count and the
  buffer file untouched.

- **`perf_ab.sh` rows whose `notes` say `ABBA` for an inverted leg** — the
  ingester asserted the interleave pattern as a constant instead of reading the
  result file's own `pattern` field, so a `--invert` leg (BAAB) was recorded as
  ABBA. Only that one word is wrong: the medians, spreads, `n` and verdicts are
  per-leg and correct, and the arms are still correctly paired. The ingester now
  interpolates the recorded pattern and writes `pattern-unrecorded` when a
  result file predates the field, so no new row asserts one. Affected:
  ids 124477–124480 and 124553–124556. The predicate cannot be written from the
  notes alone — `ABBA` is also what a genuine straight leg says — so it is the
  id range, bounded by the fix:

  ```sql
  SELECT * FROM observations
  WHERE id IN (124477, 124478, 124479, 124480, 124553, 124554, 124555, 124556);
  ```

- **`decode_config IS NULL` on a row that was speculative and never said so** —
  migration 005 added the column and left every existing row NULL, which is what
  ordinary decode carries, so a speculative row from before it kept sharing a
  cell with the plain row it ranks against and kept winning it. Migration 006
  reads the drafter back out of `notes`, which the bench scripts have recorded
  since long before the column existed
  (`rmlx_metrics::cell::decode_config_from_notes`), and classifies every row
  whose own fields say what it was. What it cannot reach is a speculative run
  that recorded no drafter marker anywhere: nothing distinguishes it from
  ordinary decode, so it stays in the plain cell and no predicate finds it. That
  the backfill is complete over what *is* recorded is checkable, and the check
  is that this returns no rows:

  ```sql
  SELECT * FROM observations
  WHERE decode_config IS NULL
    AND notes LIKE '%draft_kind=%'
    AND notes NOT LIKE '%draft_kind=none%'
    AND notes NOT LIKE '%config=normal%'
    AND notes NOT LIKE '%config=base%';
  ```

  Filling this column is not an exception to append-only. It classifies a row
  from that row's own fields into a column that was NULL for want of existing;
  no measurement is written, corrected or moved.

- **`eagle/block=5` beside `eagle3/block=5`** — two cells, one drafter, from a
  bench script that wrote the drafter's name as `eagle` where the engine writes
  `eagle3` (`DraftKind::as_str`). 16 rows carry the old name (ids 122751–122766)
  and rank against nothing. Unlike the DFlash split that migration 008 closed,
  this is a **name** and not a policy the loop always had: nothing recorded says
  the two populations ran the same drafter on the same code, so reclassifying
  them would be an assertion rather than a correction. Re-running the cell is
  what fills it.

  ```sql
  SELECT * FROM observations WHERE decode_config LIKE 'eagle/%';
  ```

Anything anchoring on a recorded rate — a roofline, a champion table, a
`rmlx metrics rank` — should read `bests`, or one of the `query::*` functions,
all of which apply the bound already. A consumer that genuinely needs the raw
distribution (a *median* over a cell's measurements, say, which the one-row-per-cell
view cannot give) has to carry the predicate itself; it must then be pinned to
this section by name and reviewed with it. `scripts/perf_ceiling.py`'s
`prefill_anchor` is the one such consumer in the tree.

**Backend coverage matrix** (which backend can emit which metric — sparse is fine, see §3.3).

This table mirrors `rmlx_metrics::registry::COVERAGE_MATRIX`; the columns are
`identity::BACKEND_WHITELIST` minus the entries declared in
`BACKENDS_WITHOUT_COVERAGE`. **Adding a backend means adding a column here and a
block there in the same change** — `coverage()` falls back to `No` for a pair it
cannot find, so a half-wired backend answers "cannot emit" for everything and
nothing says otherwise. `mlx_lm_tq`, `llama_cpp` and `llama_cpp_tq` all reached
the whitelist without coverage rows before the test was driven off the whitelist
itself.

| Metric                | rmlx | mlx_lm | mlx_lm_tq | paroquant | omlx | ollama | llama_cpp | llama_cpp_tq |
|-----------------------|:----:|:------:|:---------:|:---------:|:----:|:------:|:---------:|:------------:|
| `decode_tps_warm`     | yes | yes | yes | yes | yes | yes | yes | yes |
| `decode_tps_cold`     | yes | yes | yes | yes | yes | yes | yes | yes |
| `prefill_tps`         | yes | yes | yes | yes | yes | yes | yes | yes |
| `overall_tps`         | yes | yes | yes | yes | yes | yes | no | no |
| `ttft_warm_ms`        | yes | yes | yes | yes | yes | yes | no | no |
| `ttft_cold_ms`        | yes | yes | yes | yes | yes | yes | no | no |
| `itl_p50_ms`          | yes | yes | yes | yes | yes | yes | no | no |
| `itl_p95_ms`          | yes | yes | yes | yes | yes | yes | no | no |
| `step_ms_mean`        | yes | yes | yes | yes | yes | yes | yes | yes |
| `model_load_ms`       | yes | yes | yes | yes | yes | yes | yes | yes |
| `peak_rss_mb`         | yes | yes | yes | yes | yes | yes | yes | yes |
| `metal_peak_alloc_mb` | yes | yes | yes | yes | yes | no | no | no |
| `kv_cache_bytes`      | yes | no | no | no | maybe | no | yes | yes |
| `tps_per_gb_ram`      | yes | yes | yes | yes | yes | yes | yes | yes |
| `task_pass_at_1`      | no | no | no | no | no | no | no | no |
| `prompt_cache_block_hits`         | yes | no | no | no | no | no | no | no |
| `prompt_cache_block_misses`       | yes | no | no | no | no | no | no | no |
| `prompt_cache_partial_hits`       | yes | no | no | no | no | no | no | no |
| `queue_wait_ms`                   | yes | no | no | no | no | no | no | no |
| `queue_depth`                     | yes | no | no | no | no | no | no | no |
| `prompt_tokens_live`              | yes | no | no | no | no | no | no | no |
| `completion_tokens_live`          | yes | no | no | no | no | no | no | no |
| `itl_p99_ms`                      | yes | no | no | no | no | no | no | no |
| `itl_spikes`                      | yes | no | no | no | no | no | no | no |
| `accept_rate`                     | yes | no | no | no | no | no | no | no |
| `draft_tokens_total`              | yes | no | no | no | no | no | no | no |
| `accept_tokens_total`             | yes | no | no | no | no | no | no | no |
| `draft_rounds_total`              | yes | no | no | no | no | no | no | no |
| `accepted_per_step`               | yes | no | no | no | no | no | no | no |
| `tokens_per_round`                | yes | no | no | no | no | no | no | no |
| `draft_ms_per_round`              | yes | no | no | no | no | no | no | no |
| `verify_ms_per_round`             | yes | no | no | no | no | no | no | no |
| `loop_ms_per_round`               | yes | no | no | no | no | no | no | no |

`no` = backend genuinely can't measure. `maybe` = backend exposes it but recording path not wired. rMLX TTFT/ITL/kv_cache_bytes are wired via the EventRecorder → `events` table; cold/warm TTFT is distinguished by a first-load flag. Metal peak alloc is also wired.

Rules:
- New metric → add row here, then add `Metric` enum variant in recording code, then write code.
- Never repurpose an existing metric name.
- Units are SI / human-friendly, not normalized — store raw numbers.

---

## 5. Identity & cell normalization (mandatory rules)

These rules prevent duplicate cells and keep cross-backend comparisons honest.

### 5.1 `model_namespace` + `model` canonicalization

Two columns. `model_namespace` = source / manufacturer (who repackaged or trained the weights). `model` = short name within that namespace. Always populate both — never bake namespace into `model`. Disambiguates forks and tracks provenance.

Parser rules (filesystem layout `<root>/<namespace>__<model>` or `<root>/<namespace>/<model>` or ollama tag):

| Input                                                                              | `model_namespace` | `model`                              |
|------------------------------------------------------------------------------------|-------------------|--------------------------------------|
| `$RMLX_O_MODELS_ROOT/mlx-community__gemma-4-e2b-it-mxfp8`                               | `mlx-community`   | `gemma-4-e2b-it-mxfp8`               |
| `$RMLX_O_MODELS_ROOT/mlx-community__Qwen3.6-35B-A3B-8bit/`                              | `mlx-community`   | `Qwen3.6-35B-A3B-8bit`               |
| `$RMLX_O_MODELS_ROOT/z-lab__Qwen3.6-27B-PARO`                                           | `z-lab`           | `Qwen3.6-27B-PARO`                   |
| `$RMLX_O_MODELS_ROOT/prism-ml__Ternary-Bonsai-8B-mlx-2bit`                              | `prism-ml`        | `Ternary-Bonsai-8B-mlx-2bit`         |
| ollama tag `llama3.2:3b`                                                           | `ollama`          | `llama3.2:3b`                        |
| HF id `meta-llama/Llama-3.2-3B-Instruct`                                           | `hf`              | `meta-llama/Llama-3.2-3B-Instruct`   |
| local fine-tune `<root>/my-finetune-v1`                                            | `local`           | `my-finetune-v1`                     |

Known namespaces (extend by editing `identity::NAMESPACE_WHITELIST`): `mlx-community`, `z-lab`, `prism-ml`, `paramind`, `paro-team`, `ollama`, `hf`, `local`. This list is consulted by `identity::split_model_path` — the strict path-splitting helper `rmlx baseline` / `rmlx eval` / legacy migration use when deriving a namespace from a caller-supplied path, where an unrecognized namespace is a genuine "I could not parse this path" error the caller can act on.

**`model` and `model_namespace` are otherwise free-form recorded labels, same as `kv_quant` (§5.3).** The §8.5 ingest gate (`RunRecord::validate`) does not check either against a whitelist — an unrecognized namespace (a new model host, a typo, a local finetune someone renamed) still records rather than silently dropping into `metrics/buffer/failed/`. Only `split_model_path`'s stricter, opt-in path-derivation keeps whitelist-based rejection, because there the caller receives the error directly and can fix the input, instead of a record vanishing from an unattended bench run.

Why a separate column rather than baking into the string:
- Query "all `mlx-community` repackagings of Qwen3.6-35B" trivially.
- Query "rMLX vs ollama on the same logical model" needs namespace-aware joins.
- Same `model` name can exist in multiple namespaces (forks); composite PK handles it cleanly.

### 5.2 `weight_quant` canonicalization

Lowercase, no spaces. Match the cell-naming convention in `BENCHMARK_CHAMPIONS.md`. Examples: `mxfp8`, `mxfp4`, `nvfp4`, `q8_0`, `q4_k_m`, `2bit`, `4bit`, `8bit`, `bf16`, `fp16`, `paro`.

### 5.3 `kv_quant` canonicalization

`kv_quant` is a **free-form recorded label, not validated against a fixed codec set**. `rmlx_metrics::identity::canonicalize_kv_quant` lowercases/trims the input and normalizes a tiny, stable set of aliases — `bf16`/`f16` → `none`, `rotor_v_3`/`rotor_v_4` → `rotor3`/`rotor4` — then records everything else, including a codec name this binary has never heard of, verbatim.

This is deliberate, not an oversight. An earlier version of this function hand-mirrored the closed `KvQuant` enum grammar (`crates/rmlx-kv-quant/src/quant.rs`) as an allow-list; `rmlx-metrics` must not depend on `rmlx-kv-quant` (Ask-before dep edge — see CLAUDE.md workspace dep graph), so that mirror could only ever be a hand-maintained copy, and it went stale the moment a new codec shipped, silently dropping every metrics row for it. There is no fix to "mirror it better" — any fixed allow-list on a free-form recorded label re-introduces the same drift. The metrics DB is a measurement log, not a type-checker for the codec registry; an unrecognized `kv_quant` value is exactly as valid a data point as a recognized one, and a typo is caught by eyeballing `rmlx metrics doctor` / query output, not by rejecting the row at ingest.

The same reasoning applies to `model` and `model_namespace` — see §5.1.

### 5.3.1 Legacy note

Earlier revisions of this document described `kv_quant` as validated against a fixed list (`none`, `k8v4`, `k8v8`, `planar`, `turbo4`, `turbo8`, `mixed_k<kb>g<kg>_v<vb>g<vg>`) and, briefly, against the full `KvQuant` Display grammar with a cross-crate drift-guard test. Both were removed — the whitelist for silently dropping valid rows, the drift-guard because it kept the same enum coupling in a different shape (the moment a codec's `Display` form changed or a new one was added without a corresponding `rmlx-models` test update, rows would drop again). `canonicalize_kv_quant` is now the single, permanent implementation.

### 5.4 `backend` whitelist

Only these strings allowed. The list here mirrors
`rmlx_metrics::identity::BACKEND_WHITELIST` — extend **both**, in the same
change, or the doc says one thing and the validator does another:

- `rmlx`
- `mlx_lm` (Apple stock + sampler-fast)
- `mlx_lm_tq` (TurboQuant fork)
- `paroquant`
- `omlx`
- `ollama`
- `vllm` (future)
- `llama_cpp` (upstream `ggml-org/llama.cpp`)
- `llama_cpp_tq` (the `llama-cpp-turboquant` fork)

`canonicalize` normalizes a few spellings before the lookup: `llama.cpp` /
`llama-cpp` / `llamacpp` → `llama_cpp`, and `llama-cpp-turboquant` /
`llama.cpp-turboquant` / `llama_cpp_turboquant` → `llama_cpp_tq`.

**A fork is its own backend id, never a `kv_quant` value on the upstream id.**
`llama_cpp_tq` exists for the same reason `mlx_lm_tq` does: it is a different
build carrying codecs (`turbo2` / `turbo3` / `turbo4`) that upstream cannot
load at all. Recording a turbo cell under `llama_cpp` would attribute a
measurement to a binary that cannot produce it, and `bests` would then rank the
two builds inside one backend column.

### 5.5 `hardware_tag`

Single string: `m5_max_128gb`. New hardware = new tag (`m5_ultra_256gb`, etc.). Same tag must not span genuinely different hardware.

No special invalidation logic — `hardware_tag` is a regular context column like `backend_version`. Different value = different cell candidate via the PK. Old hardware rows stay in the table; champion view filters by current `hardware_tag`.

---

## 6. The `description` field (operating instruction)

`description` lives on `observations` (every row). `bests` view inherits the champion observation's description.

It's the **analysis-written** column. Either a human or Claude writes it after analyzing the run. `notes` is for machine-emitted auto-summary; `description` is for the post-bench narrative.

Format (1–3 lines, no headings):

```
<git-sha>: <one-line summary of what changed>
[optional second line: why it improved / regressed]
[optional third line: which doc/report explains it]
```

Examples:

```
599fb89: chunks_exact_mut + #[cold] err helpers in affine.rs / mxfp.rs
+8% decode TPS via bounds-check elision and better cold-path layout
```

```
0c41a3c: SWA per-layer mask dispatch for Gemma3/4
Fixes 4k-context <turn|>-only output on e2b k8v8; +95 TPS from 0
```

```
N/A: backend doesn't support 2-bit weights
```

Authoring rules:

- When a record is BEATEN, the new row's `description` MUST cite the commit (or "no-commit, env change") and the report path if any.
- Regressions: same field; describe suspected cause and link to investigation report.
- Claude-written descriptions: include the verbatim commit subject, do not paraphrase. If unsure, leave blank rather than guess — `notes` already captures the machine summary.
- If a dedicated `regressions` table is added later (currently deferred), it will reference `observations.id`; until then, regression context lives in the `description` field.

---

## 7. Migration plan

One-shot script to seed the DB from existing files. Idempotent — re-running with same inputs produces the same DB state (because UPSERT-on-strictly-better is idempotent).

### 7.1 Sources

| Source                                            | Format | Notes                                                                                |
|---------------------------------------------------|--------|--------------------------------------------------------------------------------------|
| `metrics/perf-iter/*.jsonl`                       | JSONL  | rMLX perf-book sweep. Has `decode_tps_mean`, `step_ms_mean`, `first_32_words`.       |
| `metrics/*.jsonl` (top-level)                     | JSONL  | rMLX runtime tracing. Mixed schemas — extract only rows matching bench schema.       |
| `Cross-Backend-Bench/metrics/summary.csv`         | CSV    | Cross-backend longitudinal. Has full schema (ttft, itl, decode_tps, peak_rss, etc.). |
| `metrics/perf-iter/*.jsonl`                        | JSONL  | Per-perf-iteration benches. Same schema as `perf-iter/`.                             |
| `BENCHMARK_CHAMPIONS.md`                            | MD     | Hand-curated records. Parse into `observations` as fallback for cells with no JSONL/CSV. Synthetic `ts_utc=2026-01-01T00:00:00Z`, `notes='migrated from BENCHMARK_CHAMPIONS.md'`. |

### 7.2 Field mapping

`run_id` is **always re-minted** at DB insert time (`<YYYYMMDDHHMMSS>-<6hex>`). Source IDs in JSONL/CSV are discarded. This is the clean break from old IDs the user requested.

CBB CSV → `bests`:

| CSV column        | `observations` column                | Notes                                                                       |
|-------------------|--------------------------------------|-----------------------------------------------------------------------------|
| (re-minted)       | `run_id`                             | New format `<YYYYMMDDHHMMSS>-<6hex>`. CSV `run_id` discarded.               |
| `timestamp_utc`   | `ts_utc`                             |                                                                             |
| `backend`         | `backend`                            | Apply §5.4 whitelist, normalize.                                            |
| `backend_version` | `backend_version`                    | Strip backend name if present, keep semver only.                            |
| `model_id`        | `model_namespace` + `model`          | Split per §5.1.                                                             |
| `quant_signature` | `weight_quant` + `kv_quant`          | Split: e.g. `mxfp8/k8v8` → `mxfp8` + `k8v8`. Document split rules in code.  |
| `device`          | `hardware_tag`                       | Map `m5_max` → `m5_max_128gb`.                                              |
| `prompt_tokens`   | `prompt_tokens`                      |                                                                             |
| `max_tokens`      | `max_tokens`                         |                                                                             |
| `ttft_ms`         | one row, metric=`ttft_warm_ms`       | `direction=lower_better`.                                                   |
| `itl_p50_ms`      | one row, metric=`itl_p50_ms`         |                                                                             |
| `itl_p95_ms`      | one row, metric=`itl_p95_ms`         |                                                                             |
| `decode_tps`      | one row, metric=`decode_tps_warm`    | `direction=higher_better`.                                                  |
| `overall_tps`     | one row, metric=`overall_tps`        |                                                                             |
| `peak_rss_mb`     | one row, metric=`peak_rss_mb`        |                                                                             |
| `task_pass_at_1`  | one row, metric=`task_pass_at_1`     |                                                                             |
| `output_first_64` | `output_first_64`                    | Denormalized into every row from the same run.                              |

CSV columns dropped: `model_disk_gb`, `model_ram_gb`, `tps_per_gb_disk`, `tps_per_gb_ram` (the last is computable from `decode_tps_warm` + `peak_rss_mb` if needed). `run_id` from CSV not stored.

rMLX perf-iter JSONL → `bests`:

| JSONL field           | `observations` column                                                                  |
|-----------------------|----------------------------------------------------------------------------------------|
| (re-minted)           | `run_id`                                                                               |
| `ts_utc`              | `ts_utc`                                                                               |
| `model_path`          | `model_namespace` + `model` (apply §5.1)                                               |
| `kv_quant`            | `kv_quant`                                                                             |
| `decode_tps_mean`     | one row, metric=`decode_tps_warm`, `decode_stddev` populated                           |
| `decode_tps_stddev`   | `decode_stddev` field                                                                  |
| `step_ms_mean`        | one row, metric=`step_ms_mean`                                                         |
| `first_32_words`      | `output_first_64` (joined with spaces, truncated to 64 chars)                          |
| `git_sha`             | `git_sha` (append `-dirty` if source `run_id` was `*-dirty.jsonl`)                     |
| `build_profile`       | `build_profile`                                                                        |
| `notes`               | `notes`                                                                                |

Defaults for rMLX JSONL (not present in source):
- `backend = 'rmlx'`
- `weight_quant` = inferred from model name suffix (`-mxfp8` → `mxfp8`, `-2bit` → `2bit`, ...)
- `ctx_max = 8192` (CBB-methodology default; verify per source)
- `prompt_id` = lookup-or-insert with body from `Cross-Backend-Bench/prompts/longctx_4k.json`
- `prompt_tokens = 4096`, `max_tokens = 32`, `temperature = 0`, `seed = 0`
- `hardware_tag = 'm5_max_128gb'`
- `n_warmups = 1`, `n_measure = 3`

### 7.3 Prompt-body recovery for legacy rows

Historic JSONL/CSV stored only `prompt_tokens`, no body. Migration reconstructs the prompt by hardcoded mapping from prompt_tokens → known prompt file. Today there is exactly one canonical CBB bench prompt:

| `prompt_tokens` (legacy row) | Prompt file (read at migration time)               | `prompts.name` |
|------------------------------|----------------------------------------------------|----------------|
| `4096`                       | `Cross-Backend-Bench/prompts/longctx_4k.json`      | `longctx_4k`   |

Migration script behaviour:
- Read each prompt file once, insert into `prompts` table, cache the resulting id.
- For each legacy row, look up by `prompt_tokens` → file → id.
- Token count mismatch ⇒ log warning, still insert, mark `observations.notes` with `"prompt_tokens=N from row vs M from file"`.
- Unknown `prompt_tokens` value ⇒ insert sentinel prompt `name="legacy_unknown_<N>"`, body `"<UNKNOWN BODY: legacy CBB row, prompt_tokens=N>"`, flag in description for later cleanup.

This is a **one-shot migration concern**. Post-migration, every new row carries its prompt body inline (or sha256 ref to an already-registered prompt) — see §8.4. No more guessing.

### 7.4 Migration steps

```bash
# 1. Snapshot
cp metrics/runs.db metrics/runs.db.bak-$(date +%Y%m%d-%H%M) 2>/dev/null || true

# 2. Init schema
sqlite3 metrics/runs.db < scripts/metrics/schema.sql

# 3. Seed prompt registry
uv run scripts/metrics/seed_prompts.py    # reads Cross-Backend-Bench/prompts/*.json

# 4. Migrate sources (idempotent — Rust subcommand)
rmlx metrics migrate \
    --rmlx-glob   "metrics/**/*.jsonl" \
    --cbb-csv     "Cross-Backend-Bench/metrics/summary.csv" \
    --records-md  "BENCHMARK_CHAMPIONS.md" \
    --hardware-tag "m5_max_128gb"

# 5. Verify counts
sqlite3 metrics/runs.db "SELECT backend, COUNT(*) FROM observations GROUP BY backend;"
sqlite3 metrics/runs.db "SELECT backend, COUNT(*) FROM bests GROUP BY backend;"     # via VIEW
sqlite3 metrics/runs.db "SELECT * FROM prompts;"

# 6. Archive legacy
mkdir -p "metrics/legacy/$(date -u +%Y%m%d)-pre-db"
mv metrics/*.jsonl metrics/perf-iter metrics/perf-* "metrics/legacy/$(date -u +%Y%m%d)-pre-db/" 2>/dev/null || true
```

### 7.5 Re-runnability

Migration is idempotent on the (`run_id`, cell, metric) tuple — recorder skips duplicates by checking for an existing observation matching the source row. Concretely:

- Re-importing the same JSONL twice = no new rows.
- Importing newer JSONL = appends fresh observations. Worse numbers don't displace better ones; both coexist in `observations`. `bests` view still picks the champion.
- No "reset" semantics needed for new hardware — just bench under a new `hardware_tag`. Old observations stay; champion view filters by tag.

If a true reset is ever required (testing, schema bug):

```bash
rmlx metrics backup                   # snapshot first
sqlite3 metrics/runs.db "DELETE FROM observations;"
rmlx metrics migrate ...              # re-import
```

`bests` is a view, no DROP needed — it always reflects current `observations`.

---

## 8. Tooling — Rust

All metrics tooling lives in the `rmlx` binary as a `metrics` subcommand. Single tool covers record, query, export, migrate. CBB Python runners shell out to it.

### 8.1 Why Rust (not Python)

| Concern                                          | Rust       | Python                |
|--------------------------------------------------|------------|-----------------------|
| Reuse workspace types (`Quant`, `KvQuant` enums) | yes, compile-time | redefine in py |
| Single binary, no venv                           | yes        | needs uv              |
| Type safety on `Metric`/`Direction` enums        | enum match | runtime strings       |
| Tracing integration                              | shared     | separate logger       |
| Bench-script call site                           | `rmlx metrics record …` | `python record.py` |
| CBB Python interop                               | shell-out  | direct import         |
| Migration speed-to-write                         | slower (~1 day) | faster (~2 hours) |
| Long-term drift between migrate/record paths     | none — same code | risk of two paths |

Decision: Rust everywhere. Migration cost paid once; drift between recording and migration paths avoided forever (both paths share the canonicalization, registry lookup, and validation code).

Crate plan:
- `crates/rmlx-metrics/` — new lib crate. Owns DB connection, schema migrations, canonicalization (§5), metric registry (§4), prompt registry, ingest API.
- `crates/rmlx-cli` — adds `metrics` subcommand wrapping the lib.
- Deps: `rusqlite` (bundled feature for static SQLite), `serde` for JSONL row deserialization.

### 8.1.1 What is NOT a recording path: `rmlx bench`

`rmlx bench` (see [`docs/CLI.md`](CLI.md#bench)) measures the same quantities —
TTFT, ITL p50/p99, decode TPS, `kv_cache_bytes` — over N repeated runs of one
cell and prints medians with the observed range. It **writes nothing**: no
buffer file, no `observations` row, no `bests` entry. Looking for its numbers in
`runs.db` will find nothing, by design.

The split is deliberate. `runs.db` is append-only, so a row recorded under the
wrong conditions is permanent; `bench` exists to *establish* a number and its
spread interactively, including the runs that get thrown away. When a figure is
worth keeping, `rmlx baseline --record` is the path that writes it.

`bench` is also where the refusal rules live in executable form — a
prompt-cache-served run, a KV-byte figure the run did not itself report, a
metric that trended across the runs instead of settling, runs that decoded
different tokens, and a `--runs 1` invocation are all hard errors there. Any
recording path that measures the same quantities refuses on the same conditions
rather than record a plausible-looking value.

The `kv_cache_bytes` rule is already carried by the two paths that *write*:
`rmlx baseline --record` and the server's speculative-decode request boundary
both sample `kv_cache_bytes_sample()` before and after the generation and
require the store sequence to have advanced. When it has not, the figure
readable belongs to an earlier generation, and both paths `warn!` and omit the
row rather than append it — the refusal matters more here than in `bench`,
because `bench` can be re-run and an `observations` row cannot be taken back.

### 8.2 Subcommands

Lifecycle:

```bash
rmlx metrics init                              # create schema, seed schema_meta (idempotent: refuses if exists)
rmlx metrics doctor                            # verify schema version, apply pending migrations, validate FKs/orphans, integrity check
rmlx metrics backup [--out <path>]             # WAL-checkpointed copy to <path> (default metrics/backups/runs-<ts>.db)
rmlx metrics restore --from <path>             # replace DB from backup, snapshot current first
```

Recording:

```bash
rmlx metrics record --inline '<json>'          # ingest one universal §8.5 record from arg
rmlx metrics record --file <path>              # ingest one universal §8.5 record from file (preferred — see §8.4 buffer)
rmlx metrics record --stdin                    # read record from stdin
rmlx metrics record --inline '<json>' --dry-run  # validate + show what WOULD be written, no commit
rmlx metrics record --replay-pending           # re-ingest every buffer/pending/ file
```

Run identity (§8.5.1):

```bash
rmlx metrics identity --json                   # the identity block, for shell/python emitters
rmlx metrics identity                          # same, human-readable
rmlx metrics validate --file <path>            # dry-run the ingest validator, write nothing
```

Migration (one-shot, idempotent):

```bash
rmlx metrics migrate --rmlx-glob "metrics/**/*.jsonl" \
                     --cbb-csv "Cross-Backend-Bench/metrics/summary.csv" \
                     [--records-md "BENCHMARK_CHAMPIONS.md"]
```

Query / read API:

```bash
rmlx metrics best --backend rmlx --namespace mlx-community \
                  --model gemma-4-e2b-it-mxfp8 --metric decode_tps_warm
rmlx metrics rank --metric decode_tps_warm [--backend rmlx] [--limit 20]    # top-N champions for one metric
rmlx metrics compare --backends rmlx,mlx_lm --metric decode_tps_warm        # side-by-side champions per cell
rmlx metrics history --cell '<json>' [--metric M] [--since <date>]          # all observations for one cell, ordered by ts
rmlx metrics timeseries --cell '<json>' --metric M [--since <date>] [--bucket day|week]  # bucketed for plotting
rmlx metrics deltas --since-sha <git-sha> [--threshold-pct N]               # what regressed/improved since a commit (per cell, per metric)
rmlx metrics describe --observation-id <N> --text "<description>"           # set/update description on one observation row
rmlx metrics describe --run-id <run_id>     --text "<description>"          # set on every observation in a run
rmlx metrics query "<sql>"                                                  # raw SQL, read-only
rmlx metrics open --readonly                                                # open in `sqlite3 -readonly` for ad-hoc browsing
```

Grafana datasource:

```bash
rmlx metrics serve --port 9821 [--bind 127.0.0.1]                           # HTTP read-only datasource for Grafana — see §10.6
rmlx metrics serve --readonly --grafana-json                                # JSON datasource format (Grafana plugin: simpod/grafana-json-datasource)
```

Export:

```bash
rmlx metrics export --markdown > BENCHMARK_CHAMPIONS.md   # canonical records table
rmlx metrics export --json     > /tmp/bests.json        # full dump for tooling
rmlx metrics export --csv      > /tmp/bests.csv         # spreadsheet-friendly
rmlx metrics export --jsonl    > /tmp/bests.jsonl       # one row per line
```

Prompts:

```bash
rmlx metrics prompts list                                # list registry
rmlx metrics prompts get --name longctx_4k               # print body to stdout
rmlx metrics prompts add  --name <n> --file <path> [--notes "..."]
rmlx metrics prompts sync                                # ingest rMLX/prompts/*.json (see §8.7)
```

DB connection setup (applied on every command opening the DB):
- `PRAGMA journal_mode=WAL` (set on init, persists in DB header).
- `PRAGMA synchronous=NORMAL` for write throughput. Still durable across crashes; only loses last txn on power loss.
- `PRAGMA foreign_keys=ON` (off by default in SQLite — must be set per connection).
- `PRAGMA busy_timeout=5000` (wait up to 5s on locked DB).
- Transaction-wrap each `record` invocation. One run = one transaction = atomic.

Read-only mode (`open --readonly`, `query`, `best`, `rank`, `compare`, `history`, `export`) opens with `mode=ro` URI flag — guaranteed not to write, safe to run while another writer is active.

For ad-hoc shell access, `sqlite3 metrics/runs.db` works fine — the DB is plain SQLite, no extensions.

### 8.2.1 Atomicity contract

Per `record` invocation:
- **Either** all metric rows for the run land in `bests` AND any new `prompts` row lands, **or** nothing changes.
- Validation failure (unknown backend, unknown metric, missing required field) → entire run rejected, exit code non-zero, stderr explains why.
- Strictly-better trigger fires per row inside the transaction. Worse-or-equal rows silently dropped (`RAISE(IGNORE)`); better rows REPLACE existing.
- If the recorder crashes mid-transaction: SQLite rolls back; re-run picks up from JSON buffer (§8.4).

### 8.2.2 Single-writer guard

Multiple bench scripts running in parallel = same DB. SQLite WAL handles concurrent reads + serializes writes, but bench scripts that sleep mid-run can hold a writer lock too long.

- All recorder invocations are short-lived (single transaction, milliseconds).
- `PRAGMA busy_timeout=5000` makes contending writers retry instead of failing.
- The single-MLX-process rule (CLAUDE.md) means parallel bench scripts shouldn't exist anyway. Recorder contention = configuration bug.
- For belt-and-braces: `metrics/runs.db.write.lock` PID file. Recorder takes flock, refuses to start if lock held by live PID. Released on exit/crash.

### 8.3 Bench script integration (day-1 DB-only)

No transition era. The migration commit:

1. Runs `rmlx metrics migrate` to backfill DB from existing JSONL/CSV (§7).
2. Moves `metrics/*.jsonl`, `metrics/perf-iter/*.jsonl` into `metrics/legacy/<YYYYMMDD>-pre-db/` — preserved for archaeology, never re-read by tooling.
3. Adds `metrics/legacy/` to `.gitignore` (already ignored via `metrics/`).
4. Switches `scripts/perf-iter/bench_decode_tps.sh` and `Cross-Backend-Bench/runners/run_one.py` to the JSON-buffer pattern (§8.4).
5. CBB `summary.csv` continues to be appended for legacy compatibility ONLY for one release; scripts also call recorder. Two releases later, drop the CSV write.

After this commit, the DB is the only source consulted for new analysis. Old JSONL stays on disk; nobody reads it.

### 8.4 JSON buffer pattern (per-run write→ingest→delete)

The bench script does NOT write directly to the DB process. It writes one JSON file per run, then invokes `rmlx metrics record --file <path>`. On success, recorder deletes the file. On failure, file is kept for retry / debugging.

Why a buffer file (not direct stdin):
- **Crash recovery**: if recorder dies mid-insert, file is still on disk. Re-run `rmlx metrics record --file <path>` once recorder is fixed.
- **Debug visibility**: human-inspectable artifact during early days. Same shape as the §8.4 ingest contract.
- **Audit trail**: `metrics/buffer/failed/` preserves every rejected record for triage.
- **Decouples bench from DB availability**: bench can complete even if DB is locked by `doctor`.

Layout:

```
metrics/buffer/
  pending/   <ts>-<uuid>.json    # written by bench, queued for recorder
  failed/    <ts>-<uuid>.json    # recorder rejected (validation failed); inspect, fix, retry
```

Bench script flow:

```bash
# scripts/perf-iter/bench_decode_tps.sh (post-migration)
record_path="metrics/buffer/pending/$(date -u +%Y%m%d%H%M%S)-$(uuidgen | head -c8).json"
# ... run bench, build JSON ...
echo "$json" > "$record_path"
if rmlx metrics record --file "$record_path"; then
    rm -f "$record_path"                           # recorder deleted on success, defensive rm
else
    mv "$record_path" "metrics/buffer/failed/"
    echo "WARN: record rejected; see ${record_path/pending/failed}"
fi
```

CBB Python runner equivalent:

```python
import json, os, subprocess, time, uuid, pathlib
buf = pathlib.Path("metrics/buffer/pending"); buf.mkdir(parents=True, exist_ok=True)
ts  = time.strftime("%Y%m%d%H%M%S", time.gmtime())
record_path = buf / f"{ts}-{uuid.uuid4().hex[:8]}.json"
record_path.write_text(json.dumps(build_record()))
result = subprocess.run(["rmlx", "metrics", "record", "--file", str(record_path)],
                        capture_output=True, text=True)
if result.returncode == 0:
    record_path.unlink(missing_ok=True)
else:
    failed = pathlib.Path("metrics/buffer/failed"); failed.mkdir(parents=True, exist_ok=True)
    record_path.rename(failed / record_path.name)
    raise RuntimeError(f"record rejected: {result.stderr}")
```

Recovery on next run: optional `rmlx metrics record --replay-pending` walks `metrics/buffer/pending/`, attempts each, deletes on success, moves to `failed/` on rejection. Mainly for crash-recovery; routine runs don't need it.

### 8.5 Ingest contract (universal — every backend uses this shape)

Single JSON object per run. The recorder fans out into N `observations` rows (one per non-null metric) inside one transaction. Atomic per-run insert. `bests` view re-evaluates lazily on next read.

`schema_version` declares the wire shape (current: `1`). Absent ⇒ assumed `1`, so archived buffer files still replay. A record declaring a **higher** version is rejected loudly rather than silently mis-parsed.

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
    "notes": "ch2-18 perf-book sweep prompt"
  },
  "ts_utc":          "2026-05-10T07:30:00Z",
  "git_sha":         "599fb89",
  "build_profile":   "release",
  "hardware_tag":    "m5_max_128gb",
  "prompt_tokens":   4096,
  "max_tokens":      32,
  "temperature":     0.0,
  "seed":            0,
  "n_warmups":       1,
  "n_measure":       3,
  "output_first_64": "## The Roman Empire: A Detailed History The history of",
  "notes":           "step_ms_mean=wall/completion_tokens",
  "description":     null,
  "metrics": [
    { "name": "decode_tps_warm", "value": 119.14, "stddev": 0.64 },
    { "name": "step_ms_mean",    "value": 8.4 },
    { "name": "ttft_warm_ms",    "value": null },
    { "name": "peak_rss_mb",     "value": null }
  ]
}
```

**Required fields** (recorder rejects row if missing or empty):

| Field             | Why                                                      |
|-------------------|----------------------------------------------------------|
| `backend`         | PK column, must be in §5.4 whitelist                     |
| `model_namespace` | PK column, free-form label (§5.1) — not whitelist-checked here |
| `model`           | PK column, free-form label. `model_id` accepted as a deserialize alias |
| `weight_quant`    | PK column, must be in §5.2 whitelist                     |
| `kv_quant`        | PK column, free-form label (§5.3) — not whitelist-checked |
| `ctx_max`         | PK column                                                |
| `prompt`          | PK column (resolved/inserted in `prompts` first)         |
| `ts_utc`          | provenance, ISO-8601 UTC                                 |
| `hardware_tag`    | provenance                                               |
| `metrics`         | array, ≥1 entry with non-null value                      |
| `backend_version` | **when `backend == "rmlx"`** — see §8.5.1                |

**Optional fields** (recorder accepts missing or null):

`schema_version` (defaults to 1), `git_sha`, `build_profile`, `prompt_tokens`, `max_tokens`, `temperature`, `seed`, `n_warmups`, `n_measure`, `output_first_64`, `notes`, `description`. Also `backend_version` — but only for non-rMLX backends (§8.5.1).

`decode_config` is optional but is **cell identity, not context** (§3.2): absent
or null means every engine setting at its default, and an emitter measuring any
non-default configuration — a speculative arm, a swept prefill chunk — must set
it or its rows land in the default cell and rank against it. It is validated
against the §3.2 grammar and a record outside that grammar is refused.

### 8.5.1 Run identity (hard rule)

The four identity fields — `backend`, `backend_version`, `build_profile`, `hardware_tag` — say **which binary produced the number**. Without them a metrics row cannot be compared across versions, and a wrong value is worse than no value.

They are produced in exactly **one place per language surface**. No emitter hand-rolls them, ever.

| Surface | Source | Notes |
|---|---|---|
| Rust (borrow) | `rmlx_metrics::identity::RunIdentity::get()` | `&'static RunIdentity`, resolved once per process, no allocation after the first call — a few `env!()` reads, no I/O, no subprocess. Prefer this. |
| Rust (owned) | `rmlx_metrics::identity::RunIdentity::rmlx()` | `get().clone()`. Only for the one call site that must move the fields into an owned `RunRecord` (`RunRecordBuilder::rmlx`). |
| Rust records | `rmlx_metrics::ingest::RunRecordBuilder::rmlx(...)` | Fills identity **and** `model_namespace` / `model` / `weight_quant` / `kv_quant` / `ts_utc` / `schema_version`. Caller supplies the measurement only. |
| Shell / Python | `rmlx metrics identity --json` | `scripts/lib/identity.sh` exports it as `RMLX_IDENTITY_JSON`; merged into each record's JSON. |

**`git_sha` is not an identity field.** The binary cannot honestly know the
commit it was built from or the tree it runs against: the working directory a
process is launched from is not necessarily its own source checkout (`rmlx
serve` is normally started from a user's project, not this repo), and
compile-time approaches fare no better — every attempt to bake a commit SHA in
at build time, then detect whether the source tree had since gone "dirty"
relative to it, required a workspace-root-discovery + rerun-if-changed +
runtime-probe apparatus that produced new defects across several review
rounds (wrong-repo detection, stale-commit detection, untracked-file false
positives) for a value nothing downstream actually needed the binary to
guess. **The binary does no git of any kind, at build time or runtime, in any
mode.**

`git_sha` is instead ordinary **caller-supplied provenance**, exactly like
`hardware_tag` already was before this section existed — the recording
surface accepts it; it does not invent it. Two ways a caller supplies it:

- A bench script that already runs its own `git -C <repo> rev-parse --short
  HEAD` stamps `"git_sha": "<sha>"` directly into the §8.5 JSON it emits,
  after the identity spread so the explicit value wins.
- `rmlx baseline --git-sha <sha>` / `rmlx eval ppl --git-sha <sha>` — optional
  CLI flags on the two record-producing bench commands. Absent by default;
  `git_sha` is then `NULL`, never guessed or derived.

The server drainer (`EventRecorder`, and any `RunRecordBuilder::rmlx(...)`
caller) has no `--git-sha` input at all, so `git_sha` is always `NULL` on
those rows. That is honest, not a regression — see §8.1 for what the drainer
records.

`RunRecord` is `#[non_exhaustive]` **and** its five `pub(crate)` fields
(`backend`, `backend_version`, `git_sha`, `build_profile`, `hardware_tag`) are
read via getters (`RunRecord::backend()`, `.backend_version()`, `.git_sha()`,
…). Two separate holes, two separate closures: `#[non_exhaustive]` blocks a
struct *literal* from outside `rmlx-metrics`; field privacy blocks mutating an
*already-built* record (`let mut r = builder.build()?; r.backend_version =
Some("0.0.1".into());` compiled and bypassed the validator before this). Rust
emitters must go through `RunRecordBuilder`; nothing outside this crate can
construct or mutate a `RunRecord`'s identity fields at all. `git_sha` stays
`pub(crate)` too even though it is caller-supplied, not binary-derived — the
same mutation-hole closure applies to it, and `RunRecordBuilder` has no
`.git_sha(...)` setter today (no Rust caller needs one yet; the two CLI flags
above set the field directly on the assembled JSON, not through the builder).

This is a mutation-hole closure, not a provenance guarantee. `RunRecord::validate()`
checks that `backend_version` is semver-*shaped*; it does not — cannot — verify
that the value is authentic. A hand-written JSON buffer file with a
fabricated-but-well-formed `"backend_version": "0.0.1"` still ingests cleanly.
The fields being un-mutable only closes the in-*Rust* hole; a hand-edited
buffer file is JSON text, not a `RunRecord` value, and goes through
`Deserialize` (which, being generated inside this crate, can populate
`pub(crate)` fields — that path is exactly what the validator gates).

**Enforcement.** `RunRecord::validate()` is the single chokepoint — every ingest path (`record --file` / `--inline` / `--stdin`, `--replay-pending`, the in-process `Recorder`) runs it. Rules:

- `backend == "rmlx"` **must** carry a semver-*shaped* `backend_version` (`MAJOR.MINOR.PATCH`, optional `-pre` / `+build` suffix — see previous paragraph on what "shaped" does and doesn't guarantee). Missing, empty, or non-semver ⇒ ingest **fails loudly**, exit 1, buffer file preserved for triage.
- Every other backend keeps `backend_version` free-form and optional — llama.cpp has no semver, it emits a `build_commit`. Cross-backend ingest is unaffected.
- `git_sha` is never required and never validated beyond being an optional string — it is provenance, not identity.

Why rMLX only: it is our own binary, so it always knows its own version. A NULL there is a bug, not a limitation.

**Identity is stamped at emit time, into the buffer file — never at ingest time.** A buffer replayed later by a newer binary keeps the identity of the build that produced it. `inserted_by` (`<tool>@<semver>`) separately records which tool did the insert. Build it with `RunIdentity::inserted_by(tool)`, not a literal.

**`build_profile` is the real Cargo profile name** — `release`, `release-perf`, `release-debug`, `debug` — stamped by `crates/rmlx-core/build.rs` from `OUT_DIR` (the path component immediately before the LAST `build` component, so a `CARGO_TARGET_DIR` that itself contains an earlier `build/` component does not misfire). Do **not** use `cfg!(debug_assertions)`: it is off for all three release profiles and collapses them to one `"release"` label, which silently turns a cross-profile comparison into what looks like a same-profile one. This is the *only* thing `build.rs` stamps — no git of any kind runs at build time.

**Pre-existing rows.** The DB still holds rows that predate this rule (NULL versions, `'0.0.1'` literals, git SHAs in the semver column) and rows written while the binary still minted `git_sha` itself, some carrying a `-dirty` suffix. `<RMLX_HOME>/metrics/` is append-only — they are **not** backfilled or rewritten. `rmlx metrics deltas --since-sha` still matches the `-dirty` suffix on historical rows (`crates/rmlx-metrics/src/query/read.rs`) even though the binary will never mint it again.

**One-time pending-buffer quarantine.** Buffer files written by an `rmlx` binary from before this contract landed (`backend: "rmlx"`, no `backend_version` key at all) are now **rejected** by `rmlx metrics record --replay-pending` instead of being silently ingested as another NULL row. Expect, on the first `--replay-pending` after upgrading past this point:

- Each pre-contract file is moved to `metrics/buffer/failed/` (nothing is deleted).
- `rmlx metrics record --replay-pending` exits **2** (its normal "some records failed" exit code) if any pre-contract files were present.
- This is expected, one-time behavior for files that predate the contract — **not a new bug** and not something to "fix" by adding a bypass flag. There is deliberately no `--legacy-archive` (or similar) escape on `record`/`--replay-pending`: the only door around the identity check (`Recorder::legacy_archive`) is `pub(crate)`, reachable only from the one-shot `rmlx metrics migrate` importer for pre-DB archives, and an operator-facing bypass would turn an exceptional path into a routine one.
- Those quarantined records carry a real `git_sha`; if ever wanted, they can be reconstructed via the same sha→tag map mentioned above and re-submitted with a real version. Backfilling them is out of scope here, same as the pre-existing DB rows.

### 8.5.2 Validating a record without writing it

```bash
rmlx metrics validate --file <buffer.json>     # dry-run of the SAME validator the recorder runs
rmlx metrics validate --stdin
```

Deliberately not a separate JSON Schema file: a second machine-readable copy of the contract is a second source of truth, and would drift — the exact failure this section exists to prevent.

**Per-metric entry**:

```json
{ "name": "decode_tps_warm", "value": 119.14, "stddev": 0.64 }
```

| Field    | Required | Notes                                                                  |
|----------|----------|------------------------------------------------------------------------|
| `name`   | yes      | Must be in §4 metric registry. Unknown name = recorder rejects.        |
| `value`  | conditional | If `null`, recorder skips this metric (no row written). Sparse OK. |
| `stddev` | optional | Only meaningful for `decode_tps_*`. Stored in `bests.decode_stddev`.   |

**Prompt resolution**: recorder hashes `prompt.body`, looks up by sha256:
- Hit: reuse existing `prompts.id`.
- Miss: insert new row with `name`, `body`, `notes`, return new id.

If body is too large to inline (>1 MB), instead pass `"prompt": { "sha256": "abc123…" }` referencing an already-registered prompt — recorder errors if not found.

### 8.6 Per-backend recording examples

**rMLX bench** (Rust, calls lib directly):

```rust
use rmlx_metrics::{Recorder, RunRecord, Metric};
let rec = Recorder::open("metrics/runs.db")?;
rec.record_run(&RunRecord { backend: "rmlx", ... metrics: vec![...], .. })?;
```

**CBB Python (mlx_lm, paroquant, omlx, ollama)** — same `_common.py` builds the dict, shells out:

```python
def record_run(row: dict) -> None:
    subprocess.run(
        ["rmlx", "metrics", "record", "--inline", json.dumps(row)],
        check=True, capture_output=True,
    )
```

**Backend-specific NULL behaviour** (illustrative, refine in code):

| Backend     | `metal_peak_alloc_mb` | `kv_cache_bytes` | `ttft_warm_ms` |
|-------------|-----------------------|------------------|----------------|
| `rmlx`      | yes                   | yes (planned)    | yes            |
| `mlx_lm`    | yes (`mx.metal.get_peak_memory()`) | no   | yes (`/v1/chat/completions` streaming) |
| `paroquant` | yes                   | no               | yes            |
| `omlx`      | yes                   | maybe            | yes            |
| `ollama`    | no                    | no               | yes (HTTP)     |

Each row in the `metrics: []` array with `value: null` is silently skipped. No backend needs to lie about a metric it can't measure — and it must not: a placeholder `0.0` in place of a `null` is rejected at ingest by the §4.1 bounds, because once stored it ranks and publishes exactly like a measured zero.

### 8.7 Prompt ownership — rMLX is the source-of-truth

Long-term the prompts used by every backend bench live in **this repo**, not in CBB. Today CBB owns `Cross-Backend-Bench/prompts/longctx_4k.json` for historic reasons. Plan to move them.

#### Layout (post-migration)

```
rMLX/prompts/
  longctx_4k.json
  shortctx_512.json
  code_eval_2k.json
  ...
```

Each file:

```json
{
  "name": "longctx_4k",
  "body": "You are an expert ...",
  "tokens_approx": 4096,
  "notes": "ch2-18 perf-book sweep prompt; original from CBB"
}
```

#### Sync into DB

```bash
rmlx metrics prompts sync                    # ingest every rMLX/prompts/*.json into prompts table
rmlx metrics prompts sync --dry-run          # show what would change
```

Hash on `body` — file edit = new sha256 = new `prompts.id` row (old id retained, since old runs reference it). Bench scripts always reference the CURRENT id by name.

#### How backends consume

Two access modes:

1. **Read-from-file** (preferred — no shell-out per bench):
   - rMLX repo is the canonical home.
   - CBB symlinks `Cross-Backend-Bench/prompts → ../rMLX/prompts` (one symlink, not per-file).
   - Both repos read the same files.

2. **Read-from-DB** (for tools that don't have repo checkout):
   ```bash
   rmlx metrics prompts get --name longctx_4k > /tmp/prompt.txt
   ```

Either way, when the bench runner emits the §8.5 ingest record, it includes the FULL body in `prompt.body` — recorder hashes it and reuses the existing `prompts.id`. No "trust me it's the same prompt" — content-addressed always.

#### Migration sequence (concrete)

1. Today: prompts live in `Cross-Backend-Bench/prompts/`, one file (`longctx_4k.json`).
2. Migration commit: `git mv Cross-Backend-Bench/prompts rMLX/prompts` + add CBB symlink `Cross-Backend-Bench/prompts → ../rMLX/prompts`.
3. `rmlx metrics prompts sync` seeds the registry.
4. CBB Python `_common.py` continues reading `Cross-Backend-Bench/prompts/longctx_4k.json` (transparent via symlink).
5. New prompts only added to `rMLX/prompts/` going forward.

#### Why this works for cross-backend bench

CBB stays operational with zero code change (symlink is invisible). rMLX owns prompt evolution. DB tracks every prompt body ever benched against by content hash, so we can always reproduce any historic run.

### 8.8 Schema-evolution policy for new metrics

When a new backend exposes a metric we don't yet track:

1. Add row to §4 metric registry (this doc).
2. Add `Metric` enum variant in `crates/rmlx-metrics/src/registry.rs`.
3. No DB migration needed — `metric` is a TEXT column, new value just appears.
4. Old rows for other backends remain row-less for that metric. That's correct ("not measured by them"), no backfill needed.

When a new backend appears (not in §5.4 whitelist):

1. Add to §5.4 whitelist.
2. Add to backend enum in `crates/rmlx-metrics/src/registry.rs`.
3. No schema change.

---

## 9. `BENCHMARK_CHAMPIONS.md` regeneration

The markdown table becomes a **derived artifact**. Single command:

```bash
rmlx metrics export --markdown > BENCHMARK_CHAMPIONS.md
```

This:
1. Queries `bests` for the canonical 5-model × N-KV grid.
2. Renders cells per the existing layout in `BENCHMARK_CHAMPIONS.md`.
3. Marks unsupported cells `N/A`, broken-output cells `x` (manual flag in `description`).
4. Footer lines list the run_id + git_sha behind each champion cell.

The hand-edit rule remains: if you didn't run a bench, don't touch the file. Now: if you didn't UPSERT a strictly-better row, the file won't change.

---

## 10. Operational concerns

### 10.1 Backups

Daily snapshot via `rmlx metrics backup`. Output dir: `metrics/backups/runs-<YYYYMMDD-HHMMSS>.db`.

- WAL-checkpointed copy (uses SQLite's `VACUUM INTO` so output is consistent even with concurrent writes).
- Cron suggestion (user-installed, not auto):
  ```cron
  0 3 * * * cd <rMLX> && rmlx metrics backup --keep 30
  ```
- `--keep N` retains last N snapshots, deletes older. Default unlimited.
- Backups are git-ignored. iCloud or external-disk sync is the user's responsibility.
- `rmlx metrics restore --from <path>` snapshots current DB to `metrics/backups/pre-restore-<ts>.db` first, then atomic-rename the restore target into place.

### 10.1.1 Turning metrics off (`--metrics`)

Global CLI flag, mirroring `--log`. Default `full` — existing behaviour, nothing changes for existing users.

| Mode | `events` | `observations` | Effect |
|---|---|---|---|
| `full` (default) | yes | yes | Current behaviour. |
| `events` | yes | no | Runtime event stream only; no bench observations. |
| `off` | no | no | **No DB writes at all.** The drainer task is never spawned and `runs.db` is never opened or created. |

```bash
rmlx --metrics off serve --model <snapshot>    # zero telemetry; no SQLite file appears
rmlx --metrics events serve --model <snapshot>
```

The mode is resolved **once** at process start (`rmlx_metrics::mode::init`) and every writer reads it from there — there is no per-call-site toggle. `off` is a no-op at the *producer*: records are not built and thrown away, they are never built.

`off` disables **writing**, not the whole subsystem. `rmlx metrics best|rank|compare|history|export|query` read the DB in every mode, and the explicitly user-invoked `rmlx metrics record` / `migrate` writes are commands, not telemetry — they are not gated.

Not an environment variable: new env vars are an "Ask before" item in CLAUDE.md, and `--log` already establishes the flag pattern.

### 10.2 Retention policy

- `observations` table append-only forever. No row deletion under normal operation. `bests` is a view, has no rows of its own.
- Old `hardware_tag` rows kept (e.g. when migrating from `m5_max_128gb` to `m5_ultra_256gb`). Champion view filters by current tag; old rows still queryable.
- `prompts` rows retained forever (foreign-key referenced by `bests`). Editing a prompt file = new sha = new row, old row stays.
- If table exceeds 1M rows (unlikely for years), revisit. Indexes scale fine until then.
- `metrics/buffer/failed/` should be inspected and cleaned periodically — no automatic TTL. Use `rmlx metrics record --replay-pending` after fixing root cause.

### 10.3 Audit

- `bests.inserted_by` = `<tool>@<semver>`. Surfaced in every read query.
- `schema_meta.created_by` = which rmlx-cli version initialized the DB.
- Bug triage: `SELECT inserted_by, COUNT(*) FROM observations GROUP BY inserted_by;` shows tool-version distribution.
- Schema migrations log into `schema_meta`: `INSERT INTO schema_meta(key, value) VALUES ('migration_<N>_at', '<utc>')`.

### 10.4 Integrity & validation (`rmlx metrics doctor`)

Runs in this order, exits non-zero on any failure:

1. `PRAGMA integrity_check` — SQLite block-level check.
2. `PRAGMA foreign_key_check` — orphan refs (`bests.prompt_id` → `prompts.id`).
3. Schema version vs code expectation. Apply pending migrations from `crates/rmlx-metrics/migrations/`.
4. Whitelist sweep — `SELECT DISTINCT backend FROM bests EXCEPT VALUES ('rmlx'),('mlx_lm'),(…)` etc. for `backend`, `model_namespace`, `weight_quant`, `kv_quant`, `metric`. Any row outside whitelist = error with PK printed.
5. Direction sanity — every `metric` value must have correct `direction` per §4 registry. Mismatch = error.
6. Unit sanity — every `metric` value must have correct `unit` per registry. Mismatch = error.
3b. `bests` view vs the §4 registry — the view is generated, not pinned to a schema version, so a DB at the latest `user_version` can still carry a definition built from an older registry. Warns; rebuilds only under `--fix`.
6b. Value plausibility — every `value` must be inside its §4.1 bounds. Unit sanity checks the *label*; this checks the *number*. Reported per metric with a count and the first offending id. **Warning**, never auto-fixed: these rows predate the ingest gate, `observations` is append-only, and the true value is not recoverable — only re-measurable.
7. Stale-prompt check — prompts referenced from `bests` whose body sha256 doesn't match any `prompts/*.json` file. Warn (not error — a removed prompt file might still have valid history).

`rmlx metrics doctor --fix` attempts safe auto-repairs (re-derives `unit`/`direction` from registry, rebuilds the `bests` view from the registry; fixes none of the others, including 6b).

### 10.5 Concurrency

- `observations` INSERTs serialized by SQLite WAL (one writer at a time, readers parallel).
- `busy_timeout=5000` makes contending writers retry rather than fail.
- Bench scripts MUST NOT run in parallel against the same model anyway (single MLX process per Mac, CLAUDE.md hard rule).
- External PID-flock on `metrics/runs.db.write.lock` to refuse concurrent recorder invocations.
- Grafana datasource (§10.6) opens DB read-only — never blocks writers.

### 10.6 Grafana / dashboards (DB as a datasource)

DB is designed for two consumers from day one: CLI (champion view, exports) and time-series UI (Grafana). The `observations` table IS a time-series store keyed by `ts_utc`. Grafana queries it directly.

#### Two integration paths

**Path A: SQLite datasource plugin** (preferred for local single-user setup)

- Plugin: `frser/sqlite-datasource` (Grafana plugin marketplace).
- Point at `metrics/runs.db` (read-only mount).
- Write SQL panels directly:
  ```sql
  -- Decode TPS over time for one cell
  SELECT
      strftime('%s', ts_utc) * 1000 AS time,
      value
  FROM observations
  WHERE backend = 'rmlx'
    AND model_namespace = 'mlx-community'
    AND model = 'gemma-4-e2b-it-mxfp8'
    AND weight_quant = 'mxfp8' AND kv_quant = 'k8v8'
    AND metric = 'decode_tps_warm'
    AND ts_utc >= datetime('now', '-90 day')
  ORDER BY ts_utc;
  ```
- No rMLX runtime required — Grafana reads the DB directly.

**Path B: HTTP datasource** (`rmlx metrics serve`, for remote / multi-tenant later)

- Plugin: `simpod/grafana-json-datasource` or built-in JSON API datasource.
- `rmlx metrics serve --port 9821 --grafana-json` exposes:
  - `GET /metrics` — list of available metric names (`decode_tps_warm`, …)
  - `GET /cells?metric=M` — list of cells emitting metric M
  - `POST /query` — Grafana JSON query format → returns datapoints
  - `POST /annotations` — returns description-bearing observations as Grafana annotations on the timeline
- DB opened with `mode=ro`. Server is stateless, safe to restart.
- Add `--bind 127.0.0.1` to keep it local-only by default.

#### Suggested dashboards

1. **"Cell over time"** — pick (backend, model, quant, kv, metric), plot `value` vs `ts_utc`. Annotations from `description` field render as flag pins on the line. Lets us see "this commit broke perf" at a glance.
2. **"Backend rivalry"** — same metric, multiple backends side-by-side per model. Each line a backend.
3. **"Champion delta"** — for each cell, current best vs 7d/30d/90d ago. Bar chart of % change.
4. **"Run summary"** — per `run_id`, all metrics emitted in that run. Useful for forensic: "what happened in this one run".
5. **"Regression watchlist"** — cells where the most recent observation regressed >5% from the all-time best. Auto-refresh.

#### Why this matters

Without Grafana, regressions surface only when a human looks at `BENCHMARK_CHAMPIONS.md` diffs. With Grafana, "decode_tps degraded over the last 3 commits" is a panel that updates automatically. The DB schema choice (`observations` as ground truth, not `bests` upserts) was made specifically to enable this.

#### Annotations (commit/release markers)

Observations carry `description` and `git_sha`. Grafana annotations panel can render:
- `description IS NOT NULL` rows as flag annotations.
- Releases (when we have them) as global annotations: `SELECT ts_utc, description FROM observations WHERE notes LIKE '%release:%'`.

No additional schema needed.

---

## 11. Test fixtures

Unit and integration tests for `crates/rmlx-metrics` use SQLite `:memory:` databases — same code path as on-disk, but disposable.

```rust
#[cfg(test)]
mod tests {
    use rmlx_metrics::Recorder;

    fn fresh() -> Recorder {
        let r = Recorder::open_memory().unwrap();
        r.init_schema().unwrap();
        r.seed_default_prompts().unwrap();   // loads rMLX/prompts/*.json
        r
    }

    #[test]
    fn upsert_only_strictly_better() { /* ... */ }

    #[test]
    fn unknown_metric_rejected() { /* ... */ }

    #[test]
    fn sparse_metric_writes_no_row() { /* ... */ }

    #[test]
    fn prompt_dedup_by_sha256() { /* ... */ }
}
```

Integration tests under `crates/rmlx-metrics/tests/` use temp-dir on-disk DB to exercise WAL + busy-timeout + flock semantics. Cleanup via `tempfile::TempDir`.

Golden-file fixtures: a tiny corpus of legacy JSONL/CSV in `crates/rmlx-metrics/tests/fixtures/legacy/`. Migration test ingests them and asserts exact row count + sample values.

---

## 12. CI integration

### 12.1 Pre-push gate (`make ci`)

`make ci` runs `ci-metrics`, which is `rmlx metrics doctor` against
`<RMLX_HOME>/metrics/runs.db`, skipped when that file is absent. CI never RUNS
benches (slow, hardware-bound); it validates the DB's structure, identity
whitelists, unit/direction registry agreement and §4.1 value plausibility.

It does **not** diff `BENCHMARK_CHAMPIONS.md`. An earlier draft of this section
proposed that, and it cannot work: the champions table is a pure function of a
machine-local, gitignored DB, so the file is gitignored too
(`.gitignore`, and `make metrics-export` says so on its help line). There is no
committed copy for CI to compare against, and a per-machine one would differ on
every host. Regenerate it locally with `make metrics-export` after any change
that alters what `bests` publishes — for example a §4.1 bounds change.

### 12.2 Bench-records sweep automation

`scripts/bench-records-sweep.sh` (existing) writes to JSONL today. Post-migration, also writes JSON buffer files via §8.4. After the sweep:

```bash
rmlx metrics doctor
rmlx metrics export --markdown > BENCHMARK_CHAMPIONS.md
git diff BENCHMARK_CHAMPIONS.md     # human-review the deltas before committing
```

Sweep script never auto-commits records changes.

### 12.3 Regression detector

`rmlx metrics rank --metric decode_tps_warm --since-sha <git-sha>` (future) compares current bests against the bests as of a given commit. Fails if any cell regressed >5%. Optional pre-push check, off by default.

---

## 13. Operating rules (instruction summary)

For any future Claude session that touches metrics:

1. **DB is THE source-of-truth.** Day-1, no transition era. Old JSONL/CSV under `metrics/legacy/` is archive only — never read or extend.
2. **DB path**: `metrics/runs.db`. Git-ignored. Daily backup via `rmlx metrics backup`. Manual snapshot before bulk ops.
3. **Three user tables**: `prompts`, `observations` (append-only), `bests` (VIEW). Plus bookkeeping `schema_meta`. Don't add user tables without updating this doc.
4. **`observations` is the ground truth** — every measurement, append-only, never updated. `bests` is a VIEW that derives champions per cell. No triggers, no UPSERT.
5. **One observation row per measured (cell, metric) per run.** Many observations per cell over time = the time-series feature. Sparse metrics still write no row.
6. **Canonicalize identity fields** per §5 — split namespace from model, lowercase quant strings, match whitelists.
7. **Use the metric registry §4.** Don't invent metric names. Don't reuse names. New metric = update §4 + add enum variant + done (TEXT col, no DDL).
8. **`description` written by human OR Claude post-bench**, lives on `observations` rows. Cite git sha and report. Mandatory whenever a new champion appears (`bests` will surface it).
9. **`run_id` minted at DB write**, format `<YYYYMMDDHHMMSS>-<6hex>`. Never reuse external IDs.
10. **Cross-repo via symlink** (CBB → rMLX). Don't fork the DB across repos.
11. **`BENCHMARK_CHAMPIONS.md` regenerated** via `rmlx metrics export --markdown`, never hand-edited. CI gate enforces parity.
12. **`hardware_tag` is regular context** — different value = different cell. New hardware = new tag.
13. **WAL mode mandatory** — set on schema init, never disable. `foreign_keys=ON` per connection.
14. **All tooling Rust** — `rmlx metrics …` subcommand. Python runners shell out, never write DB directly.
15. **Sparse rows are normal.** If a backend doesn't measure a metric, write no row. Never store `value=NULL`. Pivot queries handle missing via `MAX(CASE WHEN metric=…)`.
16. **Universal ingest contract** — every backend (rMLX, mlx_lm, paroquant, omlx, ollama, future) emits the §8.5 JSON shape. One run = one JSON = one transaction. Recorder fans out into N rows, validates against §4 registry and §5 whitelists, rejects unknowns.
17. **Prompts owned by rMLX repo** — `rMLX/prompts/*.json` is canonical. CBB symlinks. Content-addressed by sha256, so editing a file = new prompt id; old runs still reference old id. Bench runners include full body in ingest record; recorder hashes + dedups.
18. **JSON buffer per run** (§8.4) — bench writes `metrics/buffer/pending/<ts>-<uuid>.json`, calls recorder, deletes on success or moves to `failed/` on rejection. Crash recovery via `--replay-pending`.
19. **`inserted_by` audit field** — every row carries `<tool>@<semver>`. Never NULL. Triage rows by tool when anomalies appear.
20. **`rmlx metrics doctor`** — run after migrations or before suspect-state operations. Validates schema version, FKs, whitelists, unit/direction sanity, and §4.1 value plausibility.
21. **No parallel writers.** Single MLX process rule already prevents this. PID-flock available as belt-and-braces.
22. **Atomicity per run** — one `record` invocation = one transaction = all observations from that run land or none do.
23. **DB is also a Grafana datasource** — the `observations` table is a time-series. Schema choices privilege read-time derivation over write-time pruning specifically to keep history queryable. Never delete observations to "clean up" — that breaks the dashboards.
24. **Doc propagation is mandatory** — see §14. New metric/backend/rule here without CLAUDE.md + crate-README updates = drift.
25. **A/B experiments never write here.** `scripts/perf_ab.sh` runs every slot with `--metrics off`, so `runs.db` is not opened. An A/B run exercises arms built to be discarded; because `observations` is append-only, a row from a discarded arm is permanent and cannot be undone. Record the arm that survives, once, with `rmlx baseline --record`.

---

## 14. Documentation propagation (mandatory before/after impl)

This doc is the spec. The following docs MUST be kept in sync — adding a metric/backend/rule here without updating them causes drift.

### 14.1 Files to update at implementation time (single PR)

| File | What to add |
|---|---|
| `CLAUDE.md` (this repo) | New top-level "Metrics database" section (≤30 lines) — points at this doc, surfaces the 3-table model, the §13 operating rules, and the day-1 DB-only rule. Replace the current `BENCHMARK_CHAMPIONS.md (hard rule — append-only highest-record table)` section since records become a derived export. |
| `Cross-Backend-Bench/CLAUDE.md` (if exists) | Section: "Recording metrics" — points at §8.5 ingest contract + §8.4 buffer pattern + the symlink to `metrics/runs.db`. |
| `BENCHMARK_CHAMPIONS.md` | Header note: "**Generated by `rmlx metrics export --markdown`. Do not hand-edit.**" Plus a footer with the `inserted_by` + `git_sha` of the last regen. |
| `README.md` (rMLX) | One-paragraph "Metrics" section in the project layout list. Mention `metrics/runs.db`, link this doc. |
| `Makefile` | New targets: `metrics-init`, `metrics-doctor`, `metrics-export`, `metrics-backup`, `ci-metrics`. |
| `.gitignore` | Confirm `metrics/` covers `metrics/buffer/`, `metrics/legacy/`, `metrics/backups/`, `*.db*` (WAL files). |
| `docs/00-overview.md` | Stage plan section already notes metrics DB as Stage-1 deliverable; cross-link this doc. |
| `docs/PROFILING.md` | Add note: "perf measurements land in `metrics/runs.db`; query via `rmlx metrics history` for trends." |
| `crates/rmlx-metrics/README.md` (new) | Lift §10 + §13 into a crate-level README. The spec stays in `docs/METRICS_DB.md`; the README mirrors the operating rules for the lib's primary surface. |

### 14.2 Rules-of-the-road for future doc edits

- **New metric** (registry §4) → also add to: backend coverage matrix in §4, `Metric` enum in `crates/rmlx-metrics/src/registry.rs`, this doc's example queries if useful.
- **New backend** (§5.4) → also add to: backend coverage matrix in §4, `Backend` enum, `inserted_by` token format in your bench runner.
- **New table** → update §3 and renumber subsections; update §13 operating rules ("N user tables"); add migration under `crates/rmlx-metrics/migrations/`.
- **Behavior change in recorder** → bump `crates/rmlx-metrics` semver; the new version flows into `bests.inserted_by` automatically.
- **CLAUDE.md drift** — `make ci` (§12.1) only diffs `BENCHMARK_CHAMPIONS.md`. CLAUDE.md sync is on the human reviewer; flag in PR description.

### 14.4 Concrete CLAUDE.md insert (template — paste at impl time)

```markdown
## Metrics database (hard rule)

All bench metrics from any backend land in `metrics/runs.db` (SQLite, gitignored). Three user tables: `prompts`, `observations` (append-only, every measurement), `bests` (VIEW over observations). Schema and operating rules: `docs/METRICS_DB.md`.

- DB is source-of-truth from day-1. Old `metrics/*.jsonl` archived under `metrics/legacy/`, never read or extended.
- New runs: bench script writes `metrics/buffer/pending/<ts>-<uuid>.json` → `rmlx metrics record --file <path>` → recorder ingests + deletes.
- `BENCHMARK_CHAMPIONS.md` regenerated via `rmlx metrics export --markdown`. Never hand-edited.
- Cross-backend recording: every backend (rMLX, mlx_lm, paroquant, omlx, ollama) emits the §8.5 universal JSON shape.
- Prompts owned by `rMLX/prompts/*.json` (content-addressed). CBB symlinks.
- Grafana datasource: SQLite plugin reads `metrics/runs.db` directly.

Do not add tables, hand-edit `BENCHMARK_CHAMPIONS.md`, or write directly to the DB from non-Rust code. See `docs/METRICS_DB.md` §13 for the full operating rules.
```

(Replaces the current `## BENCHMARK_CHAMPIONS.md (hard rule …)` section in `CLAUDE.md` after impl lands.)

---

## 14.5 SSD cross-namespace LRU

`--kv-ssd-global-gb` (rMLX serve) installs a pool-wide ceiling across every
`<RMLX_HOME>/cache/kv/<ns>/` namespace, distinct from the per-namespace
`--kv-ssd-cache-gb` budget. At every model load (`ssd_tier::install_config`
+ `attach_at_load`) the tier scans every namespace's `index.db` once, builds
a merged `(last_used, namespace, hash, layout_key, byte_size)` row list, and
evicts oldest-first across the union until the pool sum is ≤
`global_budget_bytes`. Each eviction deletes the `.kvb` file and the index
row in the owning namespace's DB. The active namespace's per-namespace
`evict_lru_until` then runs as today — bounded by
`min(per_namespace_budget, global_budget)`. The sweep is one-shot at
startup; the decode hot path holds no cross-namespace lock. Per sweep, one
`tracing::info!` event (`event = "ssd_pool_lru_eviction"`) records
`bytes_freed`, `blocks_evicted`, `namespaces_touched`, `pool_bytes_before`,
`pool_bytes_after`, and `global_budget_bytes` for post-hoc audit.

## 14.6 Prefix-index strategy

`--prefix-index {linear|radix}` (default `linear`) selects how the in-RAM
prompt cache resolves the longest-prefix block-hash match. Both paths
implement the `PrefixIndex` trait in
`crates/rmlx-models/src/prefix_index/mod.rs`:

- `linear` — O(slots × n_blocks) scan over `Vec<Slot>`. Bisect-safe fallback.
- `radix` — NVIDIA Dynamo positional radix tree port (single-payload
  variant), O(n_blocks) lookup independent of slot count.

**SQLite stays — radix is a read accelerator, not a replacement.** The
SQLite `kv_blocks` table (`ssd_index.rs`) remains the single durability +
LRU source of truth: every spilled block is persisted there, the
cross-namespace pool eviction reads only from SQLite, and the composite
`(hash, layout_key)` PK keeps multi-layout disambiguation correct. The radix tree lives in-process memory only; it is rebuilt from
the SQLite snapshot at model load via
`PrefixIndex::insert(chained_hashes, layout_key, slot_id)` driven by the
`ssd_tier::attach_at_load` startup path and is **never persisted**.

Lock order on the spill path: tree → SQLite (insert into the prefix index
*before* `ssd_index.record`, evict from the prefix index *before* deleting
the `.kvb` files). The in-process tree mutation is cheap; doing it first
keeps the index ⇔ slot vector invariant intact even if a later SQLite
call panics.

The default strategy stays `linear` (see
`docs/PERF_BASELINE.md` §"Prefix-index bench"). Radix is opt-in pending
real-workload bench cycles.

## 15. Open items (deferred)

- True TTFT measurement still pending. Until then, `ttft_*` rows from rMLX bench come only from CBB, not from `metrics/perf-iter/`.
- `regressions` notes table — currently `description` doubles as regression note. If we accumulate >50 regressions, split into `regressions(id, bests_pk_cols, ts_utc, run_id, value, delta_pct, suspected_cause)`.
- Quality probe (`task_pass_at_1`) only present in CBB rows. rMLX bench script doesn't compute it yet.
- Multi-machine support — `host` column was dropped. Re-add only if a second machine joins; until then `hardware_tag` is sufficient.
- Regression-detector flag for `rmlx metrics rank --since-sha` (§12.3) — not in v1, deferred until we have CI bench infra.
- TUI / dashboard — `rmlx metrics tui` for live champion view. Cosmetic, deferred.
- Parquet export — for analytics workflows. Add when there's a real consumer.
- Streaming write API — if multi-tenant bench harness emerges, add a long-lived recorder daemon over Unix socket. Not needed for single-machine.
- This doc itself is provisional. After implementation lands, prune outdated drafting commentary (e.g. "Removed columns" / "Dropped metrics" subsections) and lift the operating rules §13 into either CLAUDE.md or `crates/rmlx-metrics/README.md`.
