//! Error types for rmlx-metrics.

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Top-level error enum.
#[derive(thiserror::Error, Debug)]
#[non_exhaustive]
pub enum Error {
    /// SQLite driver error from rusqlite.
    #[error("sqlite: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// I/O error (file read, write, or directory access).
    #[error("io: {0}")]
    Io(#[from] std::io::Error),

    /// Schema setup or migration failure.
    #[error("schema: {0}")]
    Schema(String),

    /// A required identity field contained a value not in the allowed whitelist.
    #[error("identity: '{value}' is not a valid {field}; allowed: {allowed:?}")]
    IdentityNotInWhitelist {
        /// Name of the field that failed validation (e.g. `"backend"`).
        field: String,
        /// The value that was rejected.
        value: String,
        /// The set of values the field accepts.
        allowed: Vec<String>,
    },

    /// The model path string could not be parsed into `(namespace, model)`.
    #[error("identity: cannot parse model path: {0}")]
    IdentityModelPath(String),

    /// The metric name is not registered in the §4 METRICS registry.
    #[error("unknown metric: '{0}' (not in registry; see docs/METRICS_DB.md §4)")]
    UnknownMetric(String),

    /// The direction string is not `"higher_better"` or `"lower_better"`.
    #[error("unknown direction: '{0}' (must be 'higher_better' or 'lower_better')")]
    UnknownDirection(String),

    /// The `ts_utc` field could not be parsed as an ISO-8601 UTC timestamp.
    #[error("ingest: invalid timestamp '{0}' — must be ISO-8601 UTC")]
    InvalidTimestamp(String),

    /// The `prompt` field was missing or structurally invalid.
    #[error("ingest: missing or invalid prompt — {0}")]
    InvalidPrompt(String),

    /// The `metrics` array was empty or every entry had a null value.
    #[error("ingest: metrics array is empty or all values are null")]
    NoMeasurements,

    /// A specific ingest field had an invalid value.
    #[error("ingest: invalid value for {field}: {message}")]
    InvalidIngestField {
        /// Name of the field that failed validation.
        field: String,
        /// Human-readable description of why the value was rejected.
        message: String,
    },

    /// Recorder-layer error (DB insert or transaction failure).
    #[error("recorder: {0}")]
    Recorder(String),

    /// Query-layer error (invalid filter, SQL execution failure).
    #[error("query: {0}")]
    Query(String),

    /// Scope-file parse or load error.
    #[error("scope: {0}")]
    Scope(String),
}
