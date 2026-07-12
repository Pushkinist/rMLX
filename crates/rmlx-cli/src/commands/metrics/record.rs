// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rmlx_metrics::identity::RunIdentity;
use rmlx_metrics::{recorder::Recorder, schema};

// ---------------------------------------------------------------------------
// record
// ---------------------------------------------------------------------------

pub(super) fn cmd_record(
    db_path: &Path,
    inline: Option<String>,
    file: Option<PathBuf>,
    stdin: bool,
    dry_run: bool,
    replay_pending: bool,
) -> anyhow::Result<()> {
    if replay_pending {
        return cmd_record_replay(db_path, dry_run);
    }
    let payload = read_payload(inline, file.as_deref(), stdin)?;
    let outcome = ingest_one(db_path, &payload, dry_run)?;
    if let Some(o) = outcome {
        println!("{}", serde_json::to_string(&o)?);
    }
    // On success in --file mode: delete the file (§8.4 buffer pattern).
    if !dry_run {
        if let Some(path) = file.as_deref() {
            match std::fs::remove_file(path) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    eprintln!("warn: buffer file already gone (race): {}", path.display());
                }
                Err(e) => {
                    return Err(e)
                        .with_context(|| format!("rm post-record buffer file {}", path.display()));
                }
            }
        }
    }
    Ok(())
}

/// Read the JSON payload from exactly one of: inline string, file path, or stdin.
/// Returns an error if none of the three sources is provided.
fn read_payload(
    inline: Option<String>,
    file: Option<&Path>,
    stdin: bool,
) -> anyhow::Result<String> {
    if let Some(s) = inline {
        return Ok(s);
    }
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
    anyhow::bail!("one of --inline, --file, or --stdin is required");
}

/// Validate and optionally commit one JSON record.
///
/// Returns `Some(RecordOutcome)` on a real insert, `None` on `--dry-run`.
///
/// Tries converters in order until one produces a record that validates:
/// 1. Canonical §8.5 [`RunRecord`] — parse + validate.
/// 2. Legacy shape (model_name / max_ctx / observations) via
///    [`rmlx_metrics::legacy_ingest::try_parse_legacy`].
/// 3. CBB May-10 shape (compound weight_quant, display backend names) via
///    [`rmlx_metrics::legacy_ingest::try_parse_cbb`].
///
/// The CBB converter is tried after the canonical path because CBB files ARE
/// structurally valid §8.5 JSON (same field names) — they just carry
/// non-whitelisted values that only `validate()` catches.
///
/// If all three fail the canonical parse/validate error is returned.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex::lock() cannot poison; Option/Result unwrap on values established by construction in this fn"
)]
pub(super) fn ingest_one(
    db_path: &Path,
    json_text: &str,
    dry_run: bool,
) -> anyhow::Result<Option<rmlx_metrics::recorder::RecordOutcome>> {
    use rmlx_metrics::ingest::RunRecord;

    // ── Attempt 1: canonical §8.5 parse + validate ────────────────────────────
    let run: RunRecord = match serde_json::from_str::<RunRecord>(json_text) {
        Err(canonical_err) => {
            // Structural parse failed — try legacy shape first.
            if let Some(converted) = rmlx_metrics::legacy_ingest::try_parse_legacy(json_text) {
                converted
            } else if let Some(converted) = rmlx_metrics::legacy_ingest::try_parse_cbb(json_text) {
                converted
            } else {
                return Err(canonical_err).context("parse RunRecord JSON");
            }
        }
        Ok(r) => {
            // Parse succeeded. Validate before accepting. If validate fails
            // and the file looks like a CBB record, try the CBB converter.
            if r.validate().is_ok() {
                r
            } else if let Some(converted) = rmlx_metrics::legacy_ingest::try_parse_cbb(json_text) {
                // CBB May-10: compound weight_quant / display backend names.
                converted
            } else {
                // Neither canonical nor CBB — surface the original validate error.
                // validate() is deterministic on &self and was Err at the is_ok()
                // check above, so it is Err here too. Re-call, map to anyhow, and
                // extract via unwrap_err() to satisfy the Result return type
                // without an unreachable!() path.
                return Err(r
                    .validate()
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .unwrap_err());
            }
        }
    };

    // validate() catches bad backend/metric/etc. without opening a transaction.
    run.validate().map_err(|e| anyhow::anyhow!("{e}"))?;

    if dry_run {
        let non_null = run.metrics.iter().filter(|m| m.value.is_some()).count();
        let prompt_name = match &run.prompt {
            rmlx_metrics::ingest::PromptRef::ByBody { name, .. } => name.as_str(),
            rmlx_metrics::ingest::PromptRef::BySha256 { sha256 } => sha256.as_str(),
        };
        println!(
            "dry-run: backend={} model={}/{} weight={} kv={} metrics={} prompt={} ts={}",
            run.backend(),
            run.model_namespace,
            run.model,
            run.weight_quant,
            run.kv_quant,
            non_null,
            prompt_name,
            run.ts_utc,
        );
        return Ok(None);
    }

    let mut conn =
        schema::open(db_path).with_context(|| format!("open DB at {}", db_path.display()))?;

    let inserted_by = RunIdentity::get().inserted_by("rmlx-cli");
    let mut rec = Recorder::new(&mut conn, inserted_by);
    let outcome = rec.record_run(&run).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(Some(outcome))
}

/// Walk `metrics/buffer/pending/`, attempt to ingest each `.json` file.
/// Successes: file removed. Failures: file moved to `metrics/buffer/failed/`.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex::lock() cannot poison; Option/Result unwrap on values established by construction in this fn"
)]
fn cmd_record_replay(db_path: &Path, dry_run: bool) -> anyhow::Result<()> {
    let pending = rmlx_core::paths::ingest_buffer_dir();
    let failed = rmlx_core::paths::metrics_dir()
        .join("buffer")
        .join("failed");
    std::fs::create_dir_all(&failed).ok();

    let mut count_ok = 0usize;
    let mut count_fail = 0usize;

    let mut entries: Vec<PathBuf> = std::fs::read_dir(&pending)
        .with_context(|| format!("read pending dir {}", pending.display()))?
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("json"))
        .collect();
    entries.sort(); // deterministic order

    for path in &entries {
        let payload = match std::fs::read_to_string(path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("fail: {}: {e}", path.display());
                count_fail += 1;
                continue;
            }
        };
        let res = ingest_one(db_path, &payload, dry_run);
        match res {
            Ok(_) => {
                if !dry_run {
                    std::fs::remove_file(path).ok();
                }
                count_ok += 1;
                println!("ok: {}", path.display());
            }
            Err(e) => {
                if !dry_run {
                    let dst = failed.join(path.file_name().unwrap());
                    std::fs::rename(path, &dst).ok();
                }
                count_fail += 1;
                eprintln!("fail: {}: {e}", path.display());
            }
        }
    }

    println!("replay summary: ok={count_ok}, fail={count_fail}");
    if count_fail > 0 {
        std::process::exit(2);
    }
    Ok(())
}
