// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rmlx_metrics::{migrate, schema};

use super::repo_root;

// ---------------------------------------------------------------------------
// migrate
// ---------------------------------------------------------------------------

pub(super) fn cmd_migrate(
    db_path: &Path,
    rmlx_glob: Option<String>,
    cbb_csv: Option<PathBuf>,
    records_md: Option<PathBuf>,
    hardware_tag: &str,
) -> anyhow::Result<()> {
    let root = repo_root();
    let prompts_dir = root.join("prompts");

    const VERSION: &str = env!("CARGO_PKG_VERSION");
    let opts = migrate::MigrateOptions {
        rmlx_glob,
        cbb_csv,
        records_md,
        hardware_tag: hardware_tag.to_owned(),
        prompts_dir,
        inserted_by: format!("migrate@{VERSION}"),
    };

    let mut conn =
        schema::open(db_path).with_context(|| format!("open DB at {}", db_path.display()))?;
    // Ensure schema is up to date before migrating.
    migrate::run_pending(&mut conn).context("run_pending migrations")?;

    let report = migrate::migrate_all(&mut conn, &opts).map_err(|e| anyhow::anyhow!("{e}"))?;

    // Serialize MigrateReport fields manually (struct doesn't derive Serialize).
    let summary = serde_json::json!({
        "rmlx_jsonl_files_read": report.rmlx_jsonl_files_read,
        "rmlx_jsonl_rows_total": report.rmlx_jsonl_rows_total,
        "rmlx_jsonl_rows_inserted": report.rmlx_jsonl_rows_inserted,
        "rmlx_jsonl_rows_skipped": report.rmlx_jsonl_rows_skipped,
        "rmlx_jsonl_parse_failures": report.rmlx_jsonl_parse_failures.len(),
        "cbb_csv_rows_total": report.cbb_csv_rows_total,
        "cbb_csv_rows_inserted": report.cbb_csv_rows_inserted,
        "cbb_csv_rows_skipped": report.cbb_csv_rows_skipped,
        "cbb_csv_parse_failures": report.cbb_csv_parse_failures.len(),
        "records_md_cells_added": report.records_md_cells_added,
    });
    println!("{}", serde_json::to_string(&summary)?);
    Ok(())
}
