// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
//! `rmlx metrics …` — metrics database management subcommands.
//!
//! Dispatches to the metrics DB operation selected by [`MetricsAction`].
//! All subcommands operate on the SQLite `runs.db` resolved via
//! `--db <path>` or `RMLX_METRICS_DB` (defaults to the workspace DB).
//!
//! # Subcommands
//!
//! - `init` — create or migrate the DB schema.
//! - `doctor` — check schema integrity; `--fix` applies repairs.
//! - `record` — ingest one universal §8.5 JSON payload into `observations`.
//! - `record-replay` — re-ingest all pending buffer files.
//! - `best` — print the best-known measurement per metric × model × backend.
//! - `rank` / `compare` — leaderboard and pairwise diff views.
//! - `export` — write `BENCHMARK_CHAMPIONS.md` from current bests.
//! - `backup` / `restore` — `VACUUM INTO` snapshots and restore from one.
//! - `prompts` — manage the content-addressed prompt registry.
//!
//! # Public API
//!
//! - [`MetricsCmd`] — top-level clap struct.
//! - [`MetricsAction`] — variant per subcommand.
//! - [`dispatch`] — entry point called from the CLI dispatch table.
//!
//! # See also
//!
//! - `docs/METRICS_DB.md` — schema, §8.2 API contract, §8.5 ingest shape.

#![allow(
    clippy::assigning_clones,
    clippy::cognitive_complexity,
    clippy::fn_params_excessive_bools,
    clippy::items_after_statements,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]

mod admin;
mod export;
mod identity_cmd;
mod migrate_cmd;
mod prompts_cmds;
mod query_cmds;
mod record;

use std::path::PathBuf;

use clap::{Args, Subcommand};

#[derive(Debug, Args)]
pub(crate) struct MetricsCmd {
    #[command(subcommand)]
    pub action: MetricsAction,

    /// Path to the metrics DB (defaults to env RMLX_METRICS_DB or metrics/runs.db).
    #[arg(long, global = true)]
    pub db: Option<PathBuf>,
}

#[derive(Debug, Subcommand)]
pub(crate) enum MetricsAction {
    /// Create schema, seed schema_meta. Refuses if file exists.
    Init,
    /// Verify schema version, integrity, FKs, whitelists, units/directions.
    Doctor {
        #[arg(long)]
        fix: bool,
    },
    /// WAL-checkpointed copy of the DB to the given path or default backups dir.
    Backup {
        #[arg(long)]
        out: Option<PathBuf>,
        #[arg(long)]
        keep: Option<usize>,
    },
    /// Replace the DB from a backup, snapshotting current first.
    Restore {
        #[arg(long)]
        from: PathBuf,
    },
    /// Ingest one §8.5 RunRecord JSON into observations.
    Record {
        /// Inline JSON object (mutually exclusive with --file/--stdin).
        #[arg(long, conflicts_with_all = ["file", "stdin"])]
        inline: Option<String>,
        /// Read JSON from path (preferred — see §8.4 buffer pattern).
        #[arg(long, conflicts_with = "stdin")]
        file: Option<PathBuf>,
        /// Read JSON from stdin.
        #[arg(long, default_value_t = false)]
        stdin: bool,
        /// Validate and show what WOULD be written; do not commit.
        #[arg(long, default_value_t = false)]
        dry_run: bool,
        /// Walk metrics/buffer/pending/, attempt each, move failures to failed/.
        #[arg(long, default_value_t = false, conflicts_with_all = ["inline", "file", "stdin"])]
        replay_pending: bool,
    },
    /// Print this binary's §8.5 run-identity block (backend, version, git sha,
    /// build profile, hardware tag). Shell emitters merge this instead of
    /// hand-rolling or hard-coding the fields.
    Identity {
        /// Emit as a single JSON object (the form bench scripts consume).
        #[arg(long, default_value_t = false)]
        json: bool,
    },
    /// Validate a §8.5 record without writing it. Same validator the recorder
    /// runs — not a second copy of the contract.
    Validate {
        /// Read JSON from path.
        #[arg(long, conflicts_with = "stdin")]
        file: Option<PathBuf>,
        /// Read JSON from stdin.
        #[arg(long, default_value_t = false)]
        stdin: bool,
    },
    // ── Query / read API (§8.2) ───────────────────────────────────────────────
    /// Champion row for one (cell, metric). Resolves --prompt-name via PromptStore if --prompt-id absent.
    Best {
        #[arg(long)]
        backend: String,
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        weight_quant: String,
        #[arg(long)]
        kv_quant: String,
        /// Server max-ctx at run time (default 8192).
        #[arg(long, default_value_t = 8192)]
        ctx_max: i64,
        /// How the tokens were produced; omit for ordinary decode. Part of the
        /// cell key, so a speculative arm is a different cell from a plain one.
        #[arg(long)]
        decode_config: Option<String>,
        /// Prompt id (FK into prompts table). Mutually exclusive with --prompt-name.
        #[arg(long, conflicts_with = "prompt_name")]
        prompt_id: Option<i64>,
        /// Resolve prompt id by name (latest revision). Mutually exclusive with --prompt-id.
        #[arg(long, conflicts_with = "prompt_id")]
        prompt_name: Option<String>,
        #[arg(long)]
        metric: String,
    },

    /// Top-N champions for one metric across all cells.
    Rank {
        #[arg(long)]
        metric: String,
        #[arg(long)]
        backend: Option<String>,
        /// Number of rows to return (default 20).
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },

    /// Side-by-side champions per cell for two or more backends (comma-separated).
    Compare {
        /// Comma-separated backends, e.g. rmlx,mlx_lm
        #[arg(long)]
        backends: String,
        #[arg(long)]
        metric: String,
        #[arg(long)]
        namespace: Option<String>,
        #[arg(long)]
        model: Option<String>,
        #[arg(long)]
        weight_quant: Option<String>,
        #[arg(long)]
        kv_quant: Option<String>,
    },

    /// All observations for one cell, ordered oldest-first.
    History {
        #[arg(long)]
        backend: String,
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        weight_quant: String,
        #[arg(long)]
        kv_quant: String,
        #[arg(long, default_value_t = 8192)]
        ctx_max: i64,
        /// How the tokens were produced; omit for ordinary decode. Part of the
        /// cell key, so a speculative arm is a different cell from a plain one.
        #[arg(long)]
        decode_config: Option<String>,
        #[arg(long, conflicts_with = "prompt_name")]
        prompt_id: Option<i64>,
        #[arg(long, conflicts_with = "prompt_id")]
        prompt_name: Option<String>,
        /// Filter to one metric (optional).
        #[arg(long)]
        metric: Option<String>,
        /// ISO-8601 date lower bound (inclusive), e.g. 2026-01-01.
        #[arg(long)]
        since: Option<String>,
    },

    /// Bucketed mean per period for one (cell, metric).
    Timeseries {
        #[arg(long)]
        backend: String,
        #[arg(long)]
        namespace: String,
        #[arg(long)]
        model: String,
        #[arg(long)]
        weight_quant: String,
        #[arg(long)]
        kv_quant: String,
        #[arg(long, default_value_t = 8192)]
        ctx_max: i64,
        /// How the tokens were produced; omit for ordinary decode. Part of the
        /// cell key, so a speculative arm is a different cell from a plain one.
        #[arg(long)]
        decode_config: Option<String>,
        #[arg(long, conflicts_with = "prompt_name")]
        prompt_id: Option<i64>,
        #[arg(long, conflicts_with = "prompt_id")]
        prompt_name: Option<String>,
        #[arg(long)]
        metric: String,
        #[arg(long)]
        since: Option<String>,
        /// Bucket granularity: day or week (default day).
        #[arg(long, default_value = "day")]
        bucket: String,
    },

    /// Champion-scoped regression gate for one model + metric.
    ///
    /// Compares the LATEST observation for the given (model, metric) scope
    /// against the all-time champion in the `bests` VIEW.
    ///
    /// Exit codes:
    ///   0   — within tolerance (or improvement)
    ///   1   — regressed beyond `--threshold-pct`
    ///   125 — no champion or no observations found (skip, bisect-safe)
    Regress {
        /// Model name substring to match (e.g. "bonsai" matches any model
        /// whose name contains "bonsai").
        #[arg(long)]
        model: String,
        /// Metric name (must be a canonical metric from the registry,
        /// e.g. decode_tps_warm, peak_rss_mb).
        #[arg(long)]
        metric: String,
        /// Optional kv_quant filter (e.g. k8v8). When omitted, matches all.
        #[arg(long)]
        kv: Option<String>,
        /// Regression threshold percentage (default 1.0).
        #[arg(long, default_value_t = 1.0)]
        threshold_pct: f64,
    },

    /// Regressions/improvements since a git SHA (per cell, per metric).
    Deltas {
        #[arg(long)]
        since_sha: String,
        /// Delta threshold percentage (default 5.0).
        #[arg(long, default_value_t = 5.0)]
        threshold_pct: f64,
        /// Exit 1 when any regression is found (default true).
        /// Pass --exit-code=false to suppress the non-zero exit (always exit 0).
        #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
        exit_code: bool,
    },

    /// Set/update the description field on one observation or all observations in a run.
    Describe {
        /// Observation id to annotate. Mutually exclusive with --run-id.
        #[arg(long, conflicts_with = "run_id")]
        observation_id: Option<i64>,
        /// Run id: annotates every observation with this run_id. Mutually exclusive with --observation-id.
        #[arg(long, conflicts_with = "observation_id")]
        run_id: Option<String>,
        #[arg(long)]
        text: String,
    },

    /// Run a raw SELECT against the DB (TSV output). Refuses non-SELECT.
    Query { sql: String },

    /// Open the DB in an interactive sqlite3 shell.
    Open {
        /// Open read-only (passes -readonly to sqlite3).
        #[arg(long, default_value_t = false)]
        readonly: bool,
    },

    /// Export the bests view to a specified format.
    Export {
        /// Emit BENCHMARK_CHAMPIONS.md markdown.
        #[arg(long, default_value_t = false)]
        markdown: bool,
        /// Emit compact JSON array.
        #[arg(long, default_value_t = false)]
        json: bool,
        /// Emit CSV with header row.
        #[arg(long, default_value_t = false)]
        csv: bool,
        /// Emit JSONL (one row per line).
        #[arg(long, default_value_t = false)]
        jsonl: bool,
        /// Optional `config/scope.toml` path. Filters and orders the
        /// markdown output. Only applies to `--markdown`.
        #[arg(long)]
        scope: Option<PathBuf>,
    },

    /// Prompt registry subcommands.
    Prompts {
        #[command(subcommand)]
        action: PromptsAction,
    },

    /// One row per (model_namespace, model, weight_quant, kv_quant) with each canonical metric as a column.
    Champions {
        /// Filter to one backend (e.g. `--backend rmlx`).
        #[arg(long)]
        backend: Option<String>,
        /// Output as JSONL instead of Markdown.
        #[arg(long, default_value_t = false)]
        jsonl: bool,
    },

    /// Ingest legacy JSONL/CSV/MD into the DB (one-shot, idempotent).
    Migrate {
        /// Glob for rMLX JSONL files, e.g. "metrics/**/*.jsonl".
        #[arg(long)]
        rmlx_glob: Option<String>,
        /// Path to Cross-Backend-Bench/metrics/summary.csv.
        #[arg(long)]
        cbb_csv: Option<PathBuf>,
        /// Path to BENCHMARK_CHAMPIONS.md fallback.
        #[arg(long)]
        records_md: Option<PathBuf>,
        /// Hardware tag stamped on every migrated observation.
        #[arg(long, default_value = "m5_max_128gb")]
        hardware_tag: String,
    },
}

/// Sub-subcommands for `rmlx metrics prompts`.
#[derive(Debug, Subcommand)]
pub(crate) enum PromptsAction {
    /// List all registered prompts.
    List,
    /// Print the body of the latest prompt with the given name to stdout.
    Get {
        #[arg(long)]
        name: String,
    },
    /// Register a prompt from a JSON file.
    Add {
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        file: PathBuf,
        #[arg(long)]
        notes: Option<String>,
    },
    /// Sync all *.json files in rMLX/prompts/ into the registry.
    Sync,
}

pub(crate) fn dispatch(cmd: MetricsCmd) -> anyhow::Result<()> {
    let db_path = resolve_db_path(cmd.db)?;
    match cmd.action {
        MetricsAction::Init => admin::cmd_init(&db_path),
        MetricsAction::Doctor { fix } => admin::cmd_doctor(&db_path, fix),
        MetricsAction::Backup { out, keep } => admin::cmd_backup(&db_path, out, keep),
        MetricsAction::Restore { from } => admin::cmd_restore(&db_path, &from),
        MetricsAction::Record {
            inline,
            file,
            stdin,
            dry_run,
            replay_pending,
        } => record::cmd_record(&db_path, inline, file, stdin, dry_run, replay_pending),
        MetricsAction::Identity { json } => identity_cmd::cmd_identity(json),
        MetricsAction::Validate { file, stdin } => identity_cmd::cmd_validate(file, stdin),
        MetricsAction::Best {
            backend,
            namespace,
            model,
            weight_quant,
            kv_quant,
            ctx_max,
            decode_config,
            prompt_id,
            prompt_name,
            metric,
        } => query_cmds::cmd_best(
            &db_path,
            &backend,
            &namespace,
            &model,
            &weight_quant,
            &kv_quant,
            ctx_max,
            decode_config,
            prompt_id,
            prompt_name.as_deref(),
            &metric,
        ),
        MetricsAction::Rank {
            metric,
            backend,
            limit,
        } => query_cmds::cmd_rank(&db_path, &metric, backend.as_deref(), limit),
        MetricsAction::Compare {
            backends,
            metric,
            namespace,
            model,
            weight_quant,
            kv_quant,
        } => query_cmds::cmd_compare(
            &db_path,
            &backends,
            &metric,
            namespace.as_deref(),
            model.as_deref(),
            weight_quant.as_deref(),
            kv_quant.as_deref(),
        ),
        MetricsAction::History {
            backend,
            namespace,
            model,
            weight_quant,
            kv_quant,
            ctx_max,
            decode_config,
            prompt_id,
            prompt_name,
            metric,
            since,
        } => query_cmds::cmd_history(
            &db_path,
            &backend,
            &namespace,
            &model,
            &weight_quant,
            &kv_quant,
            ctx_max,
            decode_config,
            prompt_id,
            prompt_name.as_deref(),
            metric.as_deref(),
            since.as_deref(),
        ),
        MetricsAction::Timeseries {
            backend,
            namespace,
            model,
            weight_quant,
            kv_quant,
            ctx_max,
            decode_config,
            prompt_id,
            prompt_name,
            metric,
            since,
            bucket,
        } => query_cmds::cmd_timeseries(
            &db_path,
            &backend,
            &namespace,
            &model,
            &weight_quant,
            &kv_quant,
            ctx_max,
            decode_config,
            prompt_id,
            prompt_name.as_deref(),
            &metric,
            since.as_deref(),
            &bucket,
        ),
        MetricsAction::Regress {
            model,
            metric,
            kv,
            threshold_pct,
        } => query_cmds::cmd_regress(&db_path, &model, &metric, kv.as_deref(), threshold_pct),
        MetricsAction::Deltas {
            since_sha,
            threshold_pct,
            exit_code,
        } => query_cmds::cmd_deltas(&db_path, &since_sha, threshold_pct, exit_code),
        MetricsAction::Describe {
            observation_id,
            run_id,
            text,
        } => query_cmds::cmd_describe(&db_path, observation_id, run_id.as_deref(), &text),
        MetricsAction::Query { sql } => query_cmds::cmd_query(&db_path, &sql),
        MetricsAction::Open { readonly } => query_cmds::cmd_open(&db_path, readonly),
        MetricsAction::Export {
            markdown,
            json,
            csv,
            jsonl,
            scope,
        } => export::cmd_export(&db_path, markdown, json, csv, jsonl, scope.as_deref()),
        MetricsAction::Prompts { action } => prompts_cmds::cmd_prompts(&db_path, action),
        MetricsAction::Champions { backend, jsonl } => {
            prompts_cmds::cmd_champions(&db_path, backend.as_deref(), jsonl)
        }
        MetricsAction::Migrate {
            rmlx_glob,
            cbb_csv,
            records_md,
            hardware_tag,
        } => migrate_cmd::cmd_migrate(&db_path, rmlx_glob, cbb_csv, records_md, &hardware_tag),
    }
}

// ---------------------------------------------------------------------------
// Path resolution
// ---------------------------------------------------------------------------

/// Resolve the DB path from (in priority order):
///   1. `--db` flag value
///   2. `RMLX_METRICS_DB` env var
///   3. `metrics/runs.db` (relative to cwd at invocation)
///
/// Creates the parent directory if it does not exist.
fn resolve_db_path(flag: Option<PathBuf>) -> anyhow::Result<PathBuf> {
    use anyhow::Context as _;
    let path = if let Some(p) = flag {
        p
    } else if let Ok(env) = std::env::var("RMLX_METRICS_DB") {
        PathBuf::from(env)
    } else {
        rmlx_core::paths::metrics_db_path()
    };

    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create parent dir {}", parent.display()))?;
        }
    }

    Ok(path)
}

/// Return the repo root path.
///
/// Checks `RMLX_REPO_ROOT` env var first; otherwise uses cwd.
/// (Users run `rmlx` from repo root per convention.)
pub(super) fn repo_root() -> PathBuf {
    if let Ok(root) = std::env::var("RMLX_REPO_ROOT") {
        return PathBuf::from(root);
    }
    std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
