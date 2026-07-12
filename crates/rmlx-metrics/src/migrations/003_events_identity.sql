-- Migration 003: run identity on the `events` table.
--
-- `events` (migration 002) carried no identity at all. The version of the
-- binary that produced a row was recoverable only from the short SHA embedded
-- in `run_id` (`YYYYMMDD-HHMMSS-<sha>`) — never as a semver, and not at all
-- once a run_id was minted by a non-`make_run_id` caller.
--
-- `observations` and `events` are now stamped from the SAME identity source
-- (`rmlx_metrics::identity::RunIdentity`). Leaving one of the two tables
-- identity-free would re-open on the events side exactly the gap being closed
-- on the observations side.
--
-- Nullable: rows written before this migration keep NULL. Append-only table —
-- no backfill, no UPDATE. Cost is ~30 bytes/row.

ALTER TABLE events ADD COLUMN backend_version TEXT;
ALTER TABLE events ADD COLUMN git_sha         TEXT;
ALTER TABLE events ADD COLUMN build_profile   TEXT;
