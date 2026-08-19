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

    /// The value cannot be a measurement of the metric it is filed under.
    ///
    /// A rate of exactly `0.0`, or a value orders of magnitude past what the
    /// hardware can produce, is a missing or miscomputed field, not a record —
    /// and once stored it wins the `bests` view and publishes as a champion.
    /// Emitters must send `null` for a measurement they do not have.
    #[error(
        "ingest: {value} is not a plausible '{metric}' — the registry bounds are {bounds} \
         (see docs/METRICS_DB.md §4). Send null, not a placeholder, for a metric \
         this run did not measure."
    )]
    ImplausibleValue {
        /// Registry name of the metric the value was filed under.
        metric: String,
        /// The value that was rejected.
        value: f64,
        /// Human-readable rendering of the plausible window, e.g. `(0, 100000]`.
        bounds: String,
    },

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

    /// An `rmlx` record carried a missing, empty, or non-semver `backend_version`.
    ///
    /// Only rMLX is held to this: it is our own binary, so it always knows its
    /// own version. Other backends (llama.cpp emits a `build_commit`, not a
    /// semver) keep the field free-form and optional.
    #[error(
        "ingest: backend 'rmlx' requires a semver backend_version (MAJOR.MINOR.PATCH), got {got}. \
         Shell emitters must take it from `rmlx metrics identity --json`; \
         Rust emitters from `rmlx_metrics::ingest::RunRecordBuilder::rmlx()` \
         (or `RunIdentity::get().stamp_json(&mut record)` for `json!`-built records). \
         See docs/METRICS_DB.md §8.5."
    )]
    MissingBackendVersion {
        /// The offending value, quoted, or `<null>` when the key was absent.
        got: String,
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
