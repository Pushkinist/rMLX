# Metrics Database

The canonical store for benchmark measurements from rMLX and other backends:
where it lives, the identity rules, ingest, the `rmlx metrics` tooling and the
operating rules. The tables, the `bests` view and the metric registry are in
[`METRICS_SCHEMA.md`](METRICS_SCHEMA.md). It is one SQLite file. The
append-only `observations` table is the ground truth. The `bests` view derives
the champion of each cell at read time. Read §13 before you write to the DB.

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
- **Connection**: every open through `rmlx_metrics::schema` sets
  `journal_mode=WAL`, `synchronous=NORMAL`, `foreign_keys=ON` and
  `busy_timeout=5000` (`schema::apply_pragmas`).
- **Sub-tree** under `<RMLX_HOME>/metrics/`:
  - `buffer/pending/`: the ingest queue (§8.4).
  - `buffer/failed/`: records that `--replay-pending` rejected.
  - `backups/`: `backup` and `restore` snapshots (§10.1).
- Take a snapshot with `rmlx metrics backup` before any bulk operation.

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
JSONL pass of the legacy importer (§7) apply it. A JSON record is stored as
it spells the field.

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
- `vllm` (no runner and no coverage entries; see `METRICS_SCHEMA.md` §4)
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

It drops every archive value outside its `METRICS_SCHEMA.md` §4.1 bounds and counts the drops
as `metrics_dropped_implausible`. The CSV pass also drops
`task_pass_at_1 = 0`, which CBB writes when it ran no quality probe. The CSV
and Markdown passes store `kv_quant` through the importer's own
`normalize_kv_quant`.

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
- `rmlx baseline --record`, and `rmlx eval ppl` when given `--corpus`. Each
  writes a buffer file, then records it in-process (§8.4).
- The server's metrics drainer (`crates/rmlx-server/src/metrics_drainer.rs`).
  It builds records with `RunRecordBuilder::rmlx` and inserts them directly,
  with no buffer file.
- `rmlx metrics migrate`, for archives only (§7).

The `events` table is written only by `EventRecorder` in the running binary
(`METRICS_SCHEMA.md` §3.6).

### 8.1.1 What is NOT a recording path: `rmlx bench`

`rmlx bench` (see [`docs/CLI.md`](CLI.md#bench)) measures TTFT, ITL, decode
TPS and `kv_cache_bytes` over repeated runs of one cell. It prints medians
with the observed range and **writes nothing**: no buffer file and no row.
`observations` is append-only, and `bench` exists to establish a number and
its spread, including runs that the operator throws away. `rmlx baseline --record`
writes a figure worth keeping.

`bench` refuses a figure it cannot attribute to the measured run. Examples
are a run served from the prompt cache or the SSD tier, a KV-byte count the
run did not report, and a metric that trended. The full set is in
`crates/rmlx-cli/src/commands/bench.rs`. A recording path that measures the
same quantities refuses on the same conditions.

The three paths that write `kv_cache_bytes` hold that rule: `rmlx baseline
--record`, the server's plain generation path
(`crates/rmlx-server/src/engine/arch_generator.rs`) and its speculative
request boundary (`engine/speculative.rs`). Each samples
`kv_cache_bytes_sample()` before and after the generation. When the store
sequence did not advance, the figure belongs to an earlier generation. Each
path then `warn!`s and omits the row.

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
| `deltas --since-sha <sha> [--threshold-pct P] [--exit-code false]` | Per cell and metric: the best after the SHA's first row (else the champion) against the best up to it. Prints the moves beyond `P` % (default 5.0) and the cells with no value up to it. Exits 1 on a regression. Exits 125 when it prints cells and none has a value up to it, and 0 when it prints nothing. `--exit-code false` always exits 0. A SHA with no rows is an error. |
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
  It refuses a path that does not exist and a stale `bests` view (`METRICS_SCHEMA.md` §3.3), and
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
- `decode_config` is cell identity, not context (`METRICS_SCHEMA.md` §3.2). Omit it or send
  `null` for a run at every default. A value must follow the `METRICS_SCHEMA.md` §3.2 grammar. A
  value that spells only the defaults is refused, and so is an adaptive
  drafter written with a fixed depth.
- `notes` or `description` containing `synthetic=true` refuses the record.
  To test whether a record would be accepted, use `record --dry-run`.

**Metric entry**: `{ "name": …, "value": …, "stddev": … }`.

- `name` must be in the `METRICS_SCHEMA.md` §4 registry.
- `value` `null` writes no row. A number outside the metric's `METRICS_SCHEMA.md` §4.1 bounds
  refuses the whole record.
- `stddev` is optional and stored as `decode_stddev` for any metric.

**Prompt**: one of two forms.

- `{ "name", "body", "notes"?, "tokens_approx"? }`. `name` is not empty, and
  `body` is any JSON value except `null`. The recorder hashes the body
  (`ingest::prompt_body_sha256`), reuses the `prompts` row with that hash, or
  inserts one.
- `{ "sha256": "<64 hex>" }` names a registered prompt. The recorder refuses
  a hash that is not in `prompts`.

`rmlx metrics record` also accepts two older buffer shapes. The legacy
bench-script shape (`model_name`, `max_ctx`, `observations`) is converted by
`legacy_ingest::try_parse_legacy`. The CBB runner shape (compound
`weight_quant`, display backend names) is converted by
`legacy_ingest::try_parse_cbb`. New emitters write the shape above.

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
process loaded. It is written to `events` only (`METRICS_SCHEMA.md` §3.6); a record ignores it.

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
  is refused. `record` exits 1 and leaves the file in place;
  `--replay-pending` moves it to `buffer/failed/` and exits 2.
- Other backends keep `backend_version` optional and free-form.
- `git_sha` is never required.

The check proves the shape, not the source. A hand-written buffer file can
carry any semver-shaped version. `RunRecord` is `#[non_exhaustive]`, and its
identity fields are `pub(crate)` behind getters. Its other fields are `pub`.
Rust code outside the crate gets a `RunRecord` from `RunRecordBuilder` or by
deserializing JSON (`serde_json::from_value`, as `rmlx baseline` and `rmlx
eval` do). Either way it cannot change the identity fields afterwards.

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
does not try the legacy and CBB shapes that `record` converts.

### 8.7 Prompt ownership — rMLX is the source-of-truth

Bench prompts live in this repo under `prompts/`. Each top-level
`prompts/*.json` file holds `name`, `body`, and optionally `tokens_approx`
and `notes`.

- `rmlx metrics prompts sync` registers every top-level `*.json` file in
  `prompts/` under the repo root (`RMLX_REPO_ROOT`, or the working
  directory). It prints how many it inserted.
- `prompts add --file <path> [--name N] [--notes T]` registers one file.
- `prompts get --name N` prints the newest body carrying that name.

The `prompts` table is content-addressed (`METRICS_SCHEMA.md` §3.1). A changed body is a new row
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
cell **and metric** (`METRICS_SCHEMA.md` §3.3). A row's decode record and its memory record can
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

- `observations` is append-only. No command deletes a row. Three things
  update one: `describe` sets `description`; `doctor --fix` corrects `unit`
  and `direction` from the registry; the post-hooks of migrations 6, 7 and 8
  rewrite `decode_config` (`migrate::schema_runner`).
- `bests` is a view and holds no rows.
- `prompts` rows stay; `observations.prompt_id` references them. A changed
  prompt body is a new row.
- `buffer/failed/` has no automatic expiry (§8.4).

### 10.3 Audit

- Every row carries `inserted_by` (`<tool>@<semver>`), such as
  `rmlx-cli@<semver>` or `migrate@<semver>`.
- `schema_meta.created_by` names the version that created the DB (`METRICS_SCHEMA.md` §3.0).
- `SELECT inserted_by, COUNT(*) FROM observations GROUP BY inserted_by;`
  shows which tools wrote the data.

### 10.4 Integrity & validation (`rmlx metrics doctor`)

`rmlx metrics doctor` (`crates/rmlx-cli/src/commands/metrics/admin.rs`) runs
these checks in order. It exits 1 when any check reports an error; warnings
do not fail it.

1. `PRAGMA integrity_check`. Error.
2. `PRAGMA foreign_key_check`. Error.
3. Schema version. Pending migrations are applied, without `--fix`.
3b. `bests` against the `METRICS_SCHEMA.md` §4 registry. The view is generated, so a DB at the
    latest version can still hold one built from an older registry. Warning;
    `--fix` rebuilds it.
4. Whitelist sweep of `observations`: `backend`, `model_namespace` and
   `weight_quant` against their §5 lists, and `metric` against the registry.
   Error, naming the value and a row id. The `kv_quant` sweep cannot fail,
   because the field is free-form (§5.3).
5. `direction` against the registry. Error; `--fix` corrects it, and the
   run that corrects it still exits 1.
6. `unit` against the registry. Error; `--fix` corrects it, and the run that
   corrects it still exits 1.
6b. Values outside their `METRICS_SCHEMA.md` §4.1 bounds, per metric, with a count and the first
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
whitelists, its registry agreement and its `METRICS_SCHEMA.md` §4.1 plausibility.

It does not diff `BENCHMARK_CHAMPIONS.md`. That file is git-ignored and
differs on every host (§9). Regenerate it with `make metrics-export` after a
change to what `bests` publishes, such as a `METRICS_SCHEMA.md` §4.1 bounds change.

---

## 13. Operating rules (instruction summary)

1. **The DB is the source of truth.**
2. **Path**: `<RMLX_HOME>/metrics/runs.db`, git-ignored (§2). Back up before
   bulk operations.
3. **Tables**: `prompts`, `observations` and `events`, the `bests` view, and
   `schema_meta` (`METRICS_SCHEMA.md` §3). Do not add tables.
4. **`observations` is the ground truth**: every measurement, append-only.
   `bests` derives the champion per cell. No triggers, no UPSERT.
5. **One row per measured (cell, metric) per run.** A metric the run did not
   measure has no row and never a placeholder.
6. **Identity fields follow §5.** Send canonical spellings; the recorder
   stores what it is given.
7. **Metric names come from the `METRICS_SCHEMA.md` §4 registry.** A new metric follows the `METRICS_SCHEMA.md` §4
   rules. Never reuse a name.
8. **`description`** is written after reading the run and cites the commit
   (§6). Write one whenever a new champion appears.
9. **`run_id` is minted at write time**, `<YYYYMMDDHHMMSS>-<6hex>`. Never
   reuse an external id.
10. **One DB across repos.** Never fork it.
11. **`BENCHMARK_CHAMPIONS.md` is generated** with `make metrics-export`,
    never hand-edited (§9).
12. **`hardware_tag` is run context**, not cell identity (§5.5).
13. **WAL and `foreign_keys=ON`** are set on every connection opened through
    `rmlx_metrics::schema`. Never disable them.
14. **All tooling is `rmlx metrics …`.** Other languages shell out and never
    write the DB directly.
15. **Every backend emits the §8.5 shape.** One run is one record and one
    transaction.
16. **Prompts live in `prompts/*.json`** and are content-addressed (§8.7).
17. **Buffer every record** (§8.4). Replay with `--replay-pending`. The
    server's metrics drainer is the one writer that inserts directly.
18. **Every row carries `inserted_by`.**
19. **Run `rmlx metrics doctor`** after migrations and before any operation
    on a DB in a suspect state (§10.4).
20. **Identity comes from `RunIdentity`** or `rmlx metrics identity --json`,
    never from a literal (§8.5.1).
21. **A/B experiments never write here.** `scripts/perf_ab.sh` runs every
    slot with `--metrics off`, so `runs.db` is not opened. A row from a
    discarded arm would be permanent. Record the surviving arm once with
    `rmlx baseline --record`.
22. **A new rule, metric or backend** updates this doc or `METRICS_SCHEMA.md`,
    `CLAUDE.md` and `crates/rmlx-metrics/README.md` in the same change.
