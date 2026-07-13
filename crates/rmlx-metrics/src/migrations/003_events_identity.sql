-- Migration 003: run identity on the `events` table.
--
-- `events` (migration 002) carried no identity at all. The version of the
-- binary that produced a row was recoverable only from the short SHA embedded
-- in `run_id` (`YYYYMMDD-HHMMSS-<sha>`) — never as a semver, and not at all
-- once a run_id was minted by a non-`make_run_id` caller.
--
-- `events` is written only by `EventRecorder::record` — i.e. only by the
-- binary, never by a bench script or CLI flag the way `observations.git_sha`
-- is. `backend_version` and `build_profile` are things the binary genuinely
-- knows about itself and that make an `events` row self-describing on its
-- own; `git_sha` is not — the binary cannot honestly know the commit it was
-- built from (see `rmlx_core::runinfo`'s module doc), so there is no caller
-- of `EventRecorder::record` that could ever populate a `git_sha` column on
-- this table. It is deliberately not added here.
--
-- Nullable: rows written before this migration keep NULL. Append-only table —
-- no backfill, no UPDATE. Cost is ~20 bytes/row.

ALTER TABLE events ADD COLUMN backend_version TEXT;
ALTER TABLE events ADD COLUMN build_profile   TEXT;
