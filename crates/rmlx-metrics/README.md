# rmlx-metrics

SQLite-backed metrics store and ingest API for rMLX bench data. Spec: [`docs/METRICS_DB.md`](../../docs/METRICS_DB.md).

## Tables

- `prompts` — registry, sha256-deduped.
- `observations` — every measurement, append-only ground truth.
- `bests` — VIEW (champion per cell via ROW_NUMBER).
- `schema_meta` — version + audit bookkeeping.

## Public API (Rust)

- `rmlx_metrics::schema::open(path)` / `open_memory()` / `open_readonly(path)`
- `rmlx_metrics::migrate::run_pending(&mut conn)` — schema migrations.
- `rmlx_metrics::migrate::migrate_all(&mut conn, &MigrateOptions)` — legacy data ingest.
- `rmlx_metrics::ingest::RunRecord` — universal §8.5 ingest contract type.
- `rmlx_metrics::recorder::Recorder::new(&mut conn, "rmlx-cli@0.0.1").record_run(&run)` — atomic per-run insert.
- `rmlx_metrics::query::{best, rank, compare, history, timeseries, deltas}` — read API.
- `rmlx_metrics::export::{export_markdown, export_json, export_csv, export_jsonl}` — serializers.
- `rmlx_metrics::prompts::{PromptStore, sync_dir}` — prompt registry.

## CLI

See `rmlx metrics --help`. Subcommands: `init`, `doctor`, `backup`, `restore`, `record`, `best`, `rank`, `compare`, `history`, `timeseries`, `deltas`, `describe`, `query`, `open`, `export`, `prompts`, `migrate`. Grafana `serve` is intentionally not implemented.

## Operating rules

1. DB at `metrics/runs.db`. Day-1 source of truth. Legacy JSONL archived under `metrics/legacy/`.
2. Three user tables (prompts, observations, bests view). No others.
3. `observations` append-only. NEVER delete; champions derived at read.
4. Sparse rows are normal (backend doesn't measure → no row). NEVER store `value=NULL`.
5. `description` written by human OR Claude post-bench, lives on observations rows.
6. `run_id` minted at DB write (`<YYYYMMDDHHMMSS>-<6hex>`).
7. JSON-buffer pattern (§8.4): bench writes `metrics/buffer/pending/<ts>-<uuid>.json`, recorder consumes/deletes.
8. `BENCHMARK_CHAMPIONS.md` regenerated via `make metrics-export`. Never hand-edited.
9. Backups via `make metrics-backup`. Restore: `rmlx metrics restore --from <backup>`.
10. Doctor (`make metrics-doctor`) validates schema + FKs + whitelists + units/directions.
11. New metric / new backend → update `docs/METRICS_DB.md` §4 / §5 first, then code.

See `docs/METRICS_DB.md` §13 for the full 24-rule list.
