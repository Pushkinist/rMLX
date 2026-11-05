# Metrics DB import

Split out of [`METRICS_DB.md`](METRICS_DB.md) to keep that doc under the size cap.

## `rmlx metrics import`

`rmlx metrics import --from <db> [--run-class C]` (`rmlx_metrics::import`)
copies every `observations` row of another metrics DB, and the prompts those
rows reference, into the DB `--db` names. Use it to keep the rows a scratch
`RMLX_HOME` recorded.

- The source is opened read-only and never migrated, so its rows and schema
  do not change. SQLite may still create `-wal` and `-shm` files beside a
  WAL-mode source. A WAL source with a leftover `-wal` may refuse a read-only
  open: take a copy first with `rmlx metrics --db <source> backup --out
  <copy>` (it does not migrate) and import the copy.
- The run class: a source at schema version 8, the last one without the
  column, needs `--run-class`, and every row takes it. A source from version 9
  up to this binary's last migration keeps each row's own class, and
  `--run-class` is refused. Any other version is refused, naming it.
- The columns copied are the target's own `observations` and `prompts`
  columns, read from the target at run time. A source missing one of them is
  refused, naming the column; `run_class` may be missing only from a version
  8 source.
- A row is copied as it stands, identity and `inserted_by` included, except
  `id`, `prompt_id` and `run_class`. A source row with no class is copied
  with none, and still reaches no gate. A prompt is matched by `sha256` and
  inserted when absent, so prompts stay content-addressed.
- A row whose `run_id`, `metric`, `ts_utc` and `inserted_utc` are already in
  the target is skipped, so a second import adds nothing.
- One transaction. It prints a JSON report and logs the same fields as one
  `info` event: `source_version`, `stated_run_class`, `read`, `inserted`,
  `skipped`, `classless` (rows copied with no class, from a source that
  carries the column) and `prompts_inserted`.
- `events` is not imported.
