# `BENCHMARK_CHAMPIONS.md` and the exports

Split out of [`METRICS_DB.md`](METRICS_DB.md) to keep that doc under the size cap.

## Regeneration

The champion table is generated, never edited by hand:

```bash
make metrics-export     # rmlx metrics export --markdown > BENCHMARK_CHAMPIONS.md
```

The file is git-ignored. It is a function of a machine-local DB, so each host
has its own. `bests` ranks `gate` rows only (`METRICS_DB.md` §8.5), so no A/B slot or
diagnostic run reaches the table.

`export --markdown` (`rmlx_metrics::export::export_markdown`) renders:

1. A fixed header (`crates/rmlx-metrics/src/templates/header.md`).
2. `## Records`: one table per model with one row per (backend, KV quant,
   decode config) cell. With `--scope <toml>` (`config/scope.toml`), only the listed models
   appear, in its order. A backend the scope lists as `unsupported` shows
   `N/A`. Without `--scope`, every model in `bests` appears.
3. A speculative-decoding table, when any champion's `decode_config` names a
   drafter.
4. A champion summary: the best decode rate per model and the rMLX gap.
5. A provenance block: export time, `bests` row count, cells rendered.

A missing value renders as `-`.

`--json`, `--csv` and `--jsonl` print one record per `bests` row, including
`git_sha`. Exactly one format flag is allowed.

### What the `Updated` column means

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
minima over `kv_cache_bytes` rows keyed on `(namespace, model, ctx_max,
kv_quant)` — so the minimum runs across every backend, prompt and decode
config at the row's own `ctx_max` and `kv_quant` (`ctx_max` taken from the
row's alphabetically-first metric), not across every `ctx_max`. No single
observation backs them.
