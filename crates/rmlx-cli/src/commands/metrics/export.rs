// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::Path;

use anyhow::Context as _;
use rmlx_metrics::{export, schema};

// ---------------------------------------------------------------------------
// export
// ---------------------------------------------------------------------------

pub(super) fn cmd_export(
    db_path: &Path,
    markdown: bool,
    json: bool,
    csv: bool,
    jsonl: bool,
    scope_path: Option<&Path>,
) -> anyhow::Result<()> {
    let flags = [markdown, json, csv, jsonl];
    let set_count = flags.iter().filter(|&&f| f).count();
    if set_count != 1 {
        anyhow::bail!(
            "exactly one of --markdown, --json, --csv, --jsonl must be set (got {set_count})"
        );
    }
    if scope_path.is_some() && !markdown {
        anyhow::bail!("--scope only applies to --markdown");
    }

    let conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;

    let scope = match scope_path {
        Some(p) => Some(
            rmlx_metrics::scope::ScopeFile::load(p)
                .with_context(|| format!("load scope file {}", p.display()))?,
        ),
        None => None,
    };

    let output = if markdown {
        export::export_markdown(&conn, scope.as_ref()).map_err(|e| anyhow::anyhow!("{e}"))?
    } else if json {
        export::export_json(&conn).map_err(|e| anyhow::anyhow!("{e}"))?
    } else if csv {
        export::export_csv(&conn).map_err(|e| anyhow::anyhow!("{e}"))?
    } else {
        export::export_jsonl(&conn).map_err(|e| anyhow::anyhow!("{e}"))?
    };

    print!("{output}");
    Ok(())
}
