# rmlx-metrics

The SQLite metrics store behind `rmlx metrics`: the schema and its
migrations, the ingest of bench records, the runtime `events` recorder, and
the read, query and export API.

[`docs/METRICS_SCHEMA.md`](../../docs/METRICS_SCHEMA.md) is the reference for
the tables (§3) and the metric registry (§4).
[`docs/METRICS_DB.md`](../../docs/METRICS_DB.md) is the reference for the
ingest record and tooling (§8) and the operating rules (§13). `rmlx metrics --help` lists the subcommands.
