//! Schema migration runner **and** legacy-data ingester.
//!
//! Two sub-modules:
//! - [`schema_runner`] — `run_pending` (schema migrations).
//! - [`legacy`] — `migrate_all` + `MigrateOptions` + `MigrateReport`
//!   (docs/METRICS_DB.md §7, legacy-data ingestion).
//!
//! Usage:
//! ```ignore
//! let mut conn = schema::open(&path)?;
//! migrate::run_pending(&mut conn)?;
//! let report = migrate::migrate_all(&mut conn, &opts)?;
//! ```

mod legacy;
mod schema_runner;

pub use legacy::{infer_weight_quant_from_model, migrate_all, MigrateOptions, MigrateReport};
pub use schema_runner::run_pending;
