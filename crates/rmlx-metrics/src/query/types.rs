//! Public data types for the read-only query API.
//!
//! All types in this module are re-exported from [`crate::query`].

use serde::{Deserialize, Serialize};

// ── Public types ──────────────────────────────────────────────────────────────

/// Cell coordinates — the columns of [`crate::cell::CELL_COLUMNS`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(
    clippy::exhaustive_structs,
    reason = "closed cell-coordinate struct — these fields are the complete PK of a metrics cell, mirroring crate::cell::CELL_COLUMNS; constructed with struct-literal from rmlx-cli, so adding a column has to reach every construction site"
)]
pub struct Cell {
    /// Canonical backend identifier (e.g. `"rmlx"`, `"mlx_lm"`).
    pub backend: String,
    /// Model namespace from the identity whitelist.
    pub model_namespace: String,
    /// Model repository name within the namespace.
    pub model: String,
    /// Canonical weight quantization string.
    pub weight_quant: String,
    /// Canonical KV-cache quantization string.
    pub kv_quant: String,
    /// Maximum context length (tokens) used for this cell.
    pub ctx_max: i64,
    /// Row ID of the prompt in the `prompts` table.
    pub prompt_id: i64,
    /// How the tokens were produced; `None` is ordinary decode. Part of the
    /// key: a speculative arm is a different configuration, not a better
    /// measurement of the plain one.
    #[serde(default)]
    pub decode_config: Option<String>,
}

/// One row from the `bests` VIEW (champion per cell+metric).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct BestRow {
    /// Row ID of the underlying observation.
    pub observation_id: i64,
    /// Cell coordinates (backend, model, quant, context, prompt).
    pub cell: Cell,
    /// Metric name (§4 canonical).
    pub metric: String,
    /// Champion metric value.
    pub value: f64,
    /// Unit string from the registry (e.g. `"tps"`, `"ms"`).
    pub unit: String,
    /// `"higher_better"` or `"lower_better"`.
    pub direction: String,
    /// Run ID of the champion observation (`<YYYYMMDDHHMMSS>-<6hex>`).
    pub run_id: String,
    /// ISO-8601 UTC timestamp of the champion observation.
    pub ts_utc: String,
    /// Git SHA of the binary that produced the champion, if recorded.
    pub git_sha: Option<String>,
    /// Backend version string of the champion run, if recorded.
    pub backend_version: Option<String>,
    /// Hardware tag of the machine where the champion was measured.
    pub hardware_tag: String,
    /// Human-readable description of the champion run.
    pub description: Option<String>,
    /// Free-form notes on the champion observation.
    pub notes: Option<String>,
    /// Audit field: who inserted this row.
    pub inserted_by: String,
}

/// One observation row (for history / timeseries queries).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct ObservationRow {
    /// Row ID in the `observations` table.
    pub id: i64,
    /// Cell coordinates for this observation.
    pub cell: Cell,
    /// Metric name (§4 canonical).
    pub metric: String,
    /// Measured value.
    pub value: f64,
    /// ISO-8601 UTC timestamp of this observation.
    pub ts_utc: String,
    /// Git SHA of the binary, if recorded.
    pub git_sha: Option<String>,
    /// Run ID for this observation batch.
    pub run_id: String,
    /// Human-readable description, if present.
    pub description: Option<String>,
}

/// Per-cell, per-backend champion row group returned by `compare`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct CompareRow {
    /// Model namespace for this comparison cell.
    pub model_namespace: String,
    /// Model name for this comparison cell.
    pub model: String,
    /// Weight quantization for this comparison cell.
    pub weight_quant: String,
    /// KV quantization for this comparison cell.
    pub kv_quant: String,
    /// Maximum context length for this comparison cell.
    pub ctx_max: i64,
    /// Prompt ID for this comparison cell.
    pub prompt_id: i64,
    /// Decode configuration for this comparison cell; `None` is ordinary decode.
    pub decode_config: Option<String>,
    /// Champion per backend: `(backend, Option<BestRow>)`, ordered by `backends` slice.
    pub per_backend: Vec<(String, Option<BestRow>)>,
}

/// Bucket granularity for `timeseries`.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[allow(
    clippy::exhaustive_enums,
    reason = "closed enum — two time-bucket granularities; adding a granularity requires updating the SQL time-bucket expressions"
)]
pub enum Bucket {
    /// Aggregate by calendar day (UTC midnight boundaries).
    Day,
    /// Aggregate by calendar week (Monday UTC boundaries).
    Week,
}

/// One bucketed data point from `timeseries`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct TimeseriesPoint {
    /// ISO-8601 UTC start of the bucket (day or week).
    pub bucket_start_utc: String,
    /// Mean value of all observations in this bucket.
    pub mean_value: f64,
    /// Number of observations in this bucket.
    pub n: i64,
}

/// One regression/improvement row from `deltas`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct DeltaRow {
    /// Cell coordinates for this delta.
    pub cell: Cell,
    /// Metric name (§4 canonical).
    pub metric: String,
    /// `"higher_better"` or `"lower_better"`.
    pub direction: String,
    /// Value of the baseline observation (oldest in range), if present.
    pub baseline_value: Option<f64>,
    /// Value of the most-recent observation.
    pub current_value: f64,
    /// Percentage change from baseline to current, if baseline is present.
    pub delta_pct: Option<f64>,
    /// `true` when the delta moves in the wrong direction beyond the threshold.
    pub regressed: bool,
}

/// Result of a `regress` check: comparison of the latest observation vs the
/// all-time champion (`bests` VIEW) for one (model, metric) scope.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct RegressResult {
    /// The model name used for matching (partial substring match against
    /// `bests.model`).
    pub model: String,
    /// Metric name (§4 canonical).
    pub metric: String,
    /// `"higher_better"` or `"lower_better"`.
    pub direction: String,
    /// Champion value from the `bests` VIEW. `None` if no champion exists.
    pub champion_value: Option<f64>,
    /// Value of the most-recent observation matching the (model, metric) scope.
    /// `None` if no observations exist.
    pub latest_value: Option<f64>,
    /// `(latest - champion) / |champion| * 100`, or `None` when either value
    /// is absent.
    pub delta_pct: Option<f64>,
    /// `true` when `delta_pct` violates the threshold in the wrong direction.
    pub regressed: bool,
    /// Threshold used for the gate (percentage).
    pub threshold_pct: f64,
    /// One-line human-readable summary.
    pub message: String,
}

/// One cell in the champion table: the best observed value for a specific metric
/// within a (model_namespace, model, weight_quant, kv_quant) key.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct ChampionCell {
    /// Champion metric value.
    pub value: f64,
    /// Unit string from the registry (e.g. `"tps"`, `"ms"`).
    pub unit: String,
    /// Backend that produced the champion observation.
    pub backend: String,
    /// Run ID of the champion observation.
    pub run_id: String,
    /// Git SHA of the binary that produced the champion, if recorded.
    pub git_sha: Option<String>,
    /// ISO-8601 UTC timestamp of the champion observation.
    pub ts_utc: String,
}

/// One row per (model_namespace, model, weight_quant, kv_quant). Columns =
/// canonical metrics from §4. Each cell = champion observation for that
/// (cell × metric), or absent if no observation present. Optionally filter
/// to one backend (returns the per-backend champion).
///
/// If `backend` is None, picks the OVERALL champion across all backends
/// per metric — useful for the BENCHMARK_CHAMPIONS.md headline view.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub struct ChampionRow {
    /// Model namespace for this champion row.
    pub model_namespace: String,
    /// Model name for this champion row.
    pub model: String,
    /// Weight quantization for this champion row.
    pub weight_quant: String,
    /// KV quantization for this champion row.
    pub kv_quant: String,
    /// Per-metric: (value, unit, backend, run_id, git_sha, ts_utc).
    /// BTreeMap keyed by metric name. Missing metric = absent on render.
    pub metrics: std::collections::BTreeMap<String, ChampionCell>,
}
