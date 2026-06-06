-- Migration 001: initial schema
-- docs/METRICS_DB.md §3: schema_meta, prompts, observations, bests VIEW
-- No triggers per §3.5. bests is a VIEW, not a base table (§3.3).

-- §3.0 Versioning + provenance
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- §3.1 Prompt registry
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

-- §3.2 Observations — append-only ground truth
-- PK is surrogate INTEGER only; no composite PK on cell columns (§3.2 rule).
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
    -- bench config (nullable — §3.4 sparse-rows policy)
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

-- §3.3 bests VIEW — champion per cell via ROW_NUMBER, tie-break newer ts_utc wins
-- Must remain a VIEW; do NOT convert to a base table (§3.3 note).
-- No triggers (§3.5).
CREATE VIEW IF NOT EXISTS bests AS
WITH ranked AS (
    SELECT
        o.*,
        ROW_NUMBER() OVER (
            PARTITION BY backend, model_namespace, model, weight_quant, kv_quant,
                         ctx_max, prompt_id, metric
            ORDER BY
                CASE WHEN direction = 'higher_better' THEN  value END DESC,
                CASE WHEN direction = 'lower_better'  THEN -value END DESC,
                ts_utc DESC
        ) AS rn
    FROM observations o
)
SELECT * FROM ranked WHERE rn = 1;
