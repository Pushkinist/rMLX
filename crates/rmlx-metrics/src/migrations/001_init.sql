-- Migration 001: initial schema
-- docs/METRICS_SCHEMA.md §3: schema_meta, prompts, observations, bests VIEW
-- No triggers per docs/METRICS_SCHEMA.md §3.5. bests is a VIEW, not a base
-- table (docs/METRICS_SCHEMA.md §3.3).

-- docs/METRICS_SCHEMA.md §3.0 Versioning + provenance
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- docs/METRICS_SCHEMA.md §3.1 Prompt registry
CREATE TABLE IF NOT EXISTS prompts (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256         TEXT    NOT NULL UNIQUE,
    name           TEXT    NOT NULL,
    body           TEXT    NOT NULL,
    tokens_approx  INTEGER,
    first_seen_utc TEXT    NOT NULL,
    notes          TEXT
);

CREATE INDEX IF NOT EXISTS prompts_name_idx ON prompts(name);

-- docs/METRICS_SCHEMA.md §3.2 Observations — append-only ground truth
-- PK is surrogate INTEGER only; no composite PK on cell columns
-- (docs/METRICS_SCHEMA.md §3.2).
CREATE TABLE IF NOT EXISTS observations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    -- cell identity
    backend          TEXT    NOT NULL,
    model_namespace  TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    weight_quant     TEXT    NOT NULL,
    kv_quant         TEXT    NOT NULL,
    ctx_max          INTEGER NOT NULL,
    prompt_id        INTEGER NOT NULL REFERENCES prompts(id),
    metric           TEXT    NOT NULL,
    -- value
    value            REAL    NOT NULL,
    unit             TEXT    NOT NULL,
    direction        TEXT    NOT NULL
        CHECK (direction IN ('higher_better', 'lower_better')),
    -- run context
    run_id           TEXT    NOT NULL,
    ts_utc           TEXT    NOT NULL,
    git_sha          TEXT,
    build_profile    TEXT,
    backend_version  TEXT,
    hardware_tag     TEXT    NOT NULL,
    -- bench config (nullable — docs/METRICS_SCHEMA.md §3.4 sparse-rows policy)
    prompt_tokens    INTEGER,
    max_tokens       INTEGER,
    temperature      REAL,
    seed             INTEGER,
    n_warmups        INTEGER,
    n_measure        INTEGER,
    -- side data
    output_first_64  TEXT,
    decode_stddev    REAL,
    notes            TEXT,
    description      TEXT,
    -- bookkeeping
    inserted_utc     TEXT    NOT NULL,
    inserted_by      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS obs_cell_idx      ON observations(backend, model_namespace, model, weight_quant, kv_quant, ctx_max, prompt_id, metric);
CREATE INDEX IF NOT EXISTS obs_metric_idx    ON observations(metric);
CREATE INDEX IF NOT EXISTS obs_ts_idx        ON observations(ts_utc);
CREATE INDEX IF NOT EXISTS obs_git_sha_idx   ON observations(git_sha);
CREATE INDEX IF NOT EXISTS obs_run_id_idx    ON observations(run_id);
CREATE INDEX IF NOT EXISTS obs_backend_idx   ON observations(backend);
CREATE INDEX IF NOT EXISTS obs_inserted_idx  ON observations(inserted_utc);

-- docs/METRICS_SCHEMA.md §3.3 bests VIEW — champion per cell.
-- Not created here: the definition is generated from the
-- docs/METRICS_SCHEMA.md §4 metric registry
-- (it carries the docs/METRICS_SCHEMA.md §4.1 plausibility filter, which this file cannot know), and
-- `migrate::run_pending` installs it via `bests_view::ensure` after the last
-- migration. Edit `bests_view::create_sql`.
-- Must remain a VIEW; do NOT convert to a base table
-- (docs/METRICS_SCHEMA.md §3.3).
-- No triggers (docs/METRICS_SCHEMA.md §3.5).
