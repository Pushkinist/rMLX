-- Migration 002: per-event runtime metrics table.
--
-- Replaces the legacy `metrics/<run-id>.jsonl` per-event stream that used
-- to be written by `rmlx_core::metrics::MetricsSink`. The DB is now the
-- single source-of-truth for both:
--   * bench observations (table `observations`, schema 001)
--   * runtime events    (table `events`, this migration)
--
-- Schema mirrors the old JSONL row format so callers do not need to
-- re-derive fields (model_basename, quant_mode, stage, op, value, unit,
-- notes). Indexed on (run_id), (op), (ts_utc) so the common queries
-- ("everything for this run", "all model-load ops", "events in a time
-- window") are cheap.

CREATE TABLE IF NOT EXISTS events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT    NOT NULL,
    ts_utc      TEXT    NOT NULL,
    model_path  TEXT    NOT NULL,
    quant_mode  TEXT    NOT NULL,
    stage       TEXT    NOT NULL,
    op          TEXT    NOT NULL,
    value_unit  TEXT    NOT NULL,
    value       REAL    NOT NULL,
    notes       TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS events_run_id_idx ON events(run_id);
CREATE INDEX IF NOT EXISTS events_op_idx     ON events(op);
CREATE INDEX IF NOT EXISTS events_ts_idx     ON events(ts_utc);
