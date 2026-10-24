-- Migration 009: why a run was recorded.
--
-- `run_class` is `gate`, `ab_arm` or `diagnostic` (docs/METRICS_SCHEMA.md §3.2).
-- Only `gate` rows reach `bests` and the queries a gate or a published figure
-- reads, so A/B slots and investigation runs can be recorded without moving a
-- champion or a gate verdict.
--
-- No DEFAULT: an INSERT that names no class stores NULL, and NULL never equals
-- 'gate', so a row written around the recorder stays out of the gates.
--
-- Every row written before this column existed was eligible for the gates, so
-- the UPDATE files it as `gate` and no champion or verdict moves. It writes no
-- measurement.

ALTER TABLE observations ADD COLUMN run_class TEXT
    CHECK (run_class IN ('gate', 'ab_arm', 'diagnostic'));

UPDATE observations SET run_class = 'gate';

CREATE INDEX IF NOT EXISTS obs_run_class_idx ON observations(run_class);
