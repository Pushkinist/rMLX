// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]

use crate::exit::ExitWith;
use std::path::{Path, PathBuf};

use anyhow::Context as _;
use rmlx_metrics::identity::RunIdentity;
use rmlx_metrics::run_class::RunClass;
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
    run_class: Option<RunClass>,
) -> anyhow::Result<()> {
    if replay_pending {
        return cmd_record_replay(db_path, dry_run, run_class);
    }
    let payload = read_payload(inline, file.as_deref(), stdin)?;
    let outcome = ingest_one(db_path, &payload, dry_run, run_class)?;
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
/// `run_class` fills a record that states no class, whichever converter
/// produced it, before it is validated. A record that states a different class
/// is refused.
///
/// If all three fail the canonical parse/validate error is returned.
pub(super) fn ingest_one(
    db_path: &Path,
    json_text: &str,
    dry_run: bool,
    run_class: Option<RunClass>,
) -> anyhow::Result<Option<rmlx_metrics::recorder::RecordOutcome>> {
    use rmlx_metrics::ingest::RunRecord;

    let classed = |run: RunRecord| apply_run_class(run, run_class);

    // ── Attempt 1: canonical §8.5 parse + validate ────────────────────────────
    let run: RunRecord = match serde_json::from_str::<RunRecord>(json_text) {
        Err(canonical_err) => {
            // Structural parse failed — try legacy shape first.
            if let Some(converted) = rmlx_metrics::legacy_ingest::try_parse_legacy(json_text) {
                classed(converted)?
            } else if let Some(converted) = rmlx_metrics::legacy_ingest::try_parse_cbb(json_text) {
                classed(converted)?
            } else {
                return Err(canonical_err).context("parse RunRecord JSON");
            }
        }
        Ok(r) => {
            // Parse succeeded. Validate before accepting. If validate fails
            // and the file looks like a CBB record, try the CBB converter.
            let r = classed(r)?;
            match r.validate() {
                Ok(()) => r,
                Err(canonical_err) => {
                    match rmlx_metrics::legacy_ingest::try_parse_cbb(json_text) {
                        // CBB May-10: compound weight_quant / display backend names.
                        Some(converted) => classed(converted)?,
                        // Neither canonical nor CBB — surface the original error.
                        None => return Err(anyhow::anyhow!("{canonical_err}")),
                    }
                }
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
            "dry-run: backend={} model={}/{} weight={} kv={} metrics={} prompt={} ts={} run_class={}",
            run.backend(),
            run.model_namespace,
            run.model,
            run.weight_quant,
            run.kv_quant,
            non_null,
            prompt_name,
            run.ts_utc,
            run.run_class.map_or("none", RunClass::as_str),
        );
        return Ok(None);
    }

    let mut conn = schema::open_migrated(db_path)
        .with_context(|| format!("open DB at {}", db_path.display()))?;

    let inserted_by = RunIdentity::get().inserted_by("rmlx-cli");
    let mut rec = Recorder::new(&mut conn, inserted_by);
    let outcome = rec.record_run(&run).map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok(Some(outcome))
}

/// Set `flag` on a record that states no class; refuse one that states another.
fn apply_run_class(
    mut run: rmlx_metrics::ingest::RunRecord,
    flag: Option<RunClass>,
) -> anyhow::Result<rmlx_metrics::ingest::RunRecord> {
    match (run.run_class, flag) {
        (Some(stated), Some(flag)) if stated != flag => anyhow::bail!(
            "the record states run_class {stated} and --run-class says {flag}; \
             a record keeps the class its producer stated"
        ),
        (None, Some(flag)) => run.run_class = Some(flag),
        _ => {}
    }
    Ok(run)
}

/// Walk `metrics/buffer/pending/`, attempt to ingest each `.json` file.
/// Successes: file removed. Failures: file moved to `metrics/buffer/failed/`.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex::lock() cannot poison; Option/Result unwrap on values established by construction in this fn"
)]
fn cmd_record_replay(
    db_path: &Path,
    dry_run: bool,
    run_class: Option<RunClass>,
) -> anyhow::Result<()> {
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
        let res = ingest_one(db_path, &payload, dry_run, run_class);
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
        return Err(ExitWith(2).into());
    }
    Ok(())
}
