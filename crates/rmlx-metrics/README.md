# rmlx-metrics

The SQLite metrics store behind `rmlx metrics`: the schema and its
migrations, the ingest of bench records, the runtime `events` recorder, the
DB-to-DB import, and the read, query and export API.

Every `observations` row carries a run class (`run_class::RunClass`: `gate`,
`ab_arm` or `diagnostic`). Ingest refuses a record that states none, and only
`gate` rows reach `bests` and the gate queries
(`bests_view::gate_rows_sql`). `migrate::run_pending` refuses a DB at a schema
version past the last migration this crate carries, since a newer binary
migrated it (`docs/METRICS_DB.md` §13, rule 23). A recorded `rmlx baseline` or `rmlx eval ppl`
run whose record does not reach the DB keeps its buffer file and exits 75
(rule 24).

[`docs/METRICS_SCHEMA.md`](../../docs/METRICS_SCHEMA.md) is the reference for
the tables (§3) and the metric registry (§4).
[`docs/METRICS_DB.md`](../../docs/METRICS_DB.md) is the reference for the
ingest record and tooling (§8) and the operating rules (§13). `rmlx metrics --help` lists the subcommands.
