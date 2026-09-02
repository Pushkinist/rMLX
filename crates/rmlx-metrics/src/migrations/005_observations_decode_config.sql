-- Migration 005: the decode configuration a bench row was measured under.
--
-- The `bests` cell key names what was measured (backend, model, quants,
-- context, prompt, metric) but nothing about *how the tokens were produced*.
-- A speculative-decode arm and a plain-decode arm of the same model at the
-- same quant and prompt therefore land in one partition, and the view's rule
-- — largest `higher_better` value wins — publishes the drafter's rate as that
-- model's champion decode throughput. On gemma-4-e2b-it-mxfp8 that is 276
-- tok/s standing in for the 142 a request without a drafter gets. Neither
-- number is wrong; they are answers to different questions, and ranking them
-- against each other is a category error the bound in §4.1 cannot see.
--
-- `decode_config` names the arm: NULL (or absent) is ordinary decode, and a
-- speculative arm records its drafter and block size, e.g. `mtp/block=5`.
-- Free-form TEXT, not an enum or CHECK: identity columns in this schema are
-- recorded strings, never validated against a closed set, so a drafter this
-- binary has never heard of still records honestly (see
-- `canonicalize_kv_quant`'s doc in `rmlx-metrics::identity`).
--
-- Nullable: every row written before this keeps NULL, which is also what
-- ordinary decode writes — so legacy plain-decode rows keep their cells and
-- keep competing with each other exactly as before. The one population this
-- does not sort out is the speculative rows written before the column
-- existed; they are NULL too, and `docs/METRICS_DB.md` names them under
-- "Known-bad rows already in the DB". Append-only table — no backfill, no
-- UPDATE.
--
-- The cell index gains the column so the `bests` window function keeps its
-- covering scan.

ALTER TABLE observations ADD COLUMN decode_config TEXT;

DROP INDEX IF EXISTS obs_cell_idx;
CREATE INDEX IF NOT EXISTS obs_cell_idx ON observations(
    backend, model_namespace, model, weight_quant, kv_quant, ctx_max,
    prompt_id, metric, decode_config
);
