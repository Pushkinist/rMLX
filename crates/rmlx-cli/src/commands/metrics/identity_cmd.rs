// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! `rmlx metrics identity` and `rmlx metrics validate`.
//!
//! `identity` is how a non-Rust emitter learns who the measured binary is. The
//! shell benches used to hard-code `'0.0.1'`, guess `"release-perf"`, or omit
//! the fields entirely; now they ask the binary and merge the answer verbatim.
//!
//! `validate` is a dry-run of the *same* `RunRecord::validate` the recorder
//! runs — deliberately not a second, parallel schema that could drift.

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rmlx_metrics::identity::RunIdentity;

/// Print the §8.5 run-identity block for this binary.
pub(super) fn cmd_identity(json: bool) -> anyhow::Result<()> {
    let ident = RunIdentity::get();

    if json {
        println!("{}", serde_json::to_string(ident)?);
        return Ok(());
    }

    println!("backend         {}", ident.backend());
    println!("backend_version {}", ident.backend_version());
    println!("build_profile   {}", ident.build_profile());
    println!("hardware_tag    {}", ident.hardware_tag());
    Ok(())
}

/// Validate a §8.5 record without writing anything.
pub(super) fn cmd_validate(file: Option<PathBuf>, stdin: bool) -> anyhow::Result<()> {
    let payload = read_payload(file.as_deref(), stdin)?;

    let run: rmlx_metrics::ingest::RunRecord =
        serde_json::from_str(&payload).context("parse RunRecord JSON")?;
    run.validate().map_err(|e| anyhow::anyhow!("{e}"))?;

    let measured = run.metrics.iter().filter(|m| m.value.is_some()).count();
    println!(
        "ok: backend={} version={} profile={} model={}/{} metrics={}",
        run.backend(),
        run.backend_version().unwrap_or("-"),
        run.build_profile().unwrap_or("-"),
        run.model_namespace,
        run.model,
        measured,
    );
    Ok(())
}

fn read_payload(file: Option<&Path>, stdin: bool) -> anyhow::Result<String> {
    if let Some(path) = file {
        return std::fs::read_to_string(path)
            .with_context(|| format!("read record file {}", path.display()));
    }
    if stdin {
        use std::io::Read as _;
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("read record from stdin")?;
        return Ok(buf);
    }
    anyhow::bail!("one of --file or --stdin is required")
}
