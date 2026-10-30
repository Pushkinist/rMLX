// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::Path;

use anyhow::Context as _;
use rmlx_metrics::run_class::RunClass;
use rmlx_metrics::{import, schema};

pub(super) fn cmd_import(
    db_path: &Path,
    from: &Path,
    run_class: Option<RunClass>,
) -> anyhow::Result<()> {
    let mut conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;
    let report = import::import_db(&mut conn, from, run_class)
        .with_context(|| format!("import from {}", from.display()))?;
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}
