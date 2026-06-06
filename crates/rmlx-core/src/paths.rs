//! Canonical data-root resolver for rMLX runtime artifacts.
//!
//! All log files, metrics, the SQLite DB, prompt buffers and any other
//! on-disk state live under a single root directory referred to as
//! `.rmlx/`. The root is resolved once at process start in this exact order:
//!
//! 1. `$RMLX_HOME` (if set; must be absolute).
//! 2. `<workspace>/.rmlx/` — found by walking up from the current working
//!    directory until `Cargo.lock` appears. This is the **dev default**:
//!    state is co-located with the checkout, gitignored, and trivially
//!    wiped (`rm -rf .rmlx`).
//! 3. `$HOME/.rmlx/` — installed-binary default, persists across sessions.
//!
//! Sub-directories:
//!
//! ```text
//! .rmlx/
//!   logs/                 per-run JSON logs (rotated by total-size cap)
//!   metrics/
//!     runs.db             SQLite metrics DB (source-of-truth)
//!     summary.csv         rolling CSV mirror
//!     backups/            VACUUM INTO snapshots
//!     buffer/pending/     universal-shape ingest queue
//!     legacy/             archived per-run jsonls (read-only)
//!   cache/                future model/weight cache
//!   tmp/                  transient files, may be wiped at startup
//! ```
//!
//! All callers in this codebase MUST go through [`home`], [`logs_dir`],
//! [`metrics_dir`], [`metrics_db_path`], etc. Hard-coded `"logs"` /
//! `"metrics"` string paths are a bug — they resolve against the caller's
//! cwd, which produces orphan files in `crates/rmlx-cli/` when tests or
//! `cargo run` set cwd to the crate directory.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

const ENV_HOME: &str = "RMLX_HOME";

/// Cached resolution. The root is fixed for the lifetime of the process.
static RESOLVED_HOME: OnceLock<PathBuf> = OnceLock::new();

/// Return the resolved `.rmlx/` root for this process. Creates the
/// directory tree on first call. Idempotent and cheap on subsequent calls.
pub fn home() -> PathBuf {
    RESOLVED_HOME
        .get_or_init(|| {
            let root = resolve();
            // Best-effort create. Callers that need stronger guarantees
            // (e.g. metrics db open) will re-attempt and surface their own
            // errors.
            let _ = std::fs::create_dir_all(&root);
            tracing::debug!(path = %root.display(), "rmlx home resolved");
            root
        })
        .clone()
}

/// Return `<home>/logs/`. Created on demand.
pub fn logs_dir() -> PathBuf {
    let p = home().join("logs");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Return `<home>/metrics/`. Created on demand.
pub fn metrics_dir() -> PathBuf {
    let p = home().join("metrics");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Path to the canonical SQLite metrics DB.
pub fn metrics_db_path() -> PathBuf {
    metrics_dir().join("runs.db")
}

/// Path to the rolling summary CSV.
pub fn summary_csv_path() -> PathBuf {
    metrics_dir().join("summary.csv")
}

/// `<home>/metrics/buffer/pending/` — §8.5 ingest queue.
pub fn ingest_buffer_dir() -> PathBuf {
    let p = metrics_dir().join("buffer").join("pending");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// `<home>/metrics/legacy/` — read-only archive of pre-DB jsonls.
pub fn legacy_metrics_dir() -> PathBuf {
    let p = metrics_dir().join("legacy");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// `<home>/cache/` — model/weight cache (future use).
pub fn cache_dir() -> PathBuf {
    let p = home().join("cache");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// `<home>/cache/kv/<namespace>/` — per-namespace KV-block SSD cache.
///
/// Each namespace (typically `<model_id>/<kv_quant>`) gets its own
/// sub-directory so block files and the index DB are collocated.
/// Created on demand, matching the pattern of the other path helpers.
pub fn kv_cache_dir(namespace: &str) -> PathBuf {
    let p = cache_dir().join("kv").join(namespace);
    let _ = std::fs::create_dir_all(&p);
    p
}

/// `<home>/bench/` — micro-benchmark artefacts (CSV rows, ad-hoc baselines).
///
/// review LOW-4: canonical location for criterion / shell-script
/// bench outputs (`perf_canary.csv`, `prefix_index.csv`, …) so benches and
/// `scripts/perf_canary.sh` agree on the same root and respect `$RMLX_HOME`.
pub fn bench_dir() -> PathBuf {
    let p = home().join("bench");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// `<home>/tmp/` — transient files; may be wiped at process start.
pub fn tmp_dir() -> PathBuf {
    let p = home().join("tmp");
    let _ = std::fs::create_dir_all(&p);
    p
}

/// Path to the named-server-profiles TOML (`<home>/profiles.toml`).
///
/// holds `[profile.<name>]` launch presets for `rmlx serve --profile`.
/// Resolved against [`home`], never the caller's cwd. The file is optional;
/// callers handle "does not exist" as "no profiles defined".
pub fn profiles_path() -> PathBuf {
    home().join("profiles.toml")
}

/// Path to the per-project cap defaults TOML (`<home>/projects.toml`).
///
/// holds `[global]` default caps and `[project.<name>]` per-project
/// SSD / RAM overrides for `rmlx serve --project`. Resolved against [`home`],
/// never the caller's cwd. The file is optional; callers handle "does not
/// exist" as "no project config" (built-in defaults apply).
pub fn projects_toml_path() -> PathBuf {
    home().join("projects.toml")
}

// ── Resolution ───────────────────────────────────────────────────────────────

fn resolve() -> PathBuf {
    // 1. Env var override (dev shells, CI, installed-binary on non-standard layouts).
    if let Some(p) = env_home() {
        return p;
    }
    // 2. Workspace-local for dev: walk up from cwd for Cargo.lock.
    if let Some(ws) = workspace_root() {
        return ws.join(".rmlx");
    }
    // 3. User home — installed-binary default.
    user_home_rmlx().unwrap_or_else(|| PathBuf::from(".rmlx"))
}

fn env_home() -> Option<PathBuf> {
    let raw = std::env::var_os(ENV_HOME)?;
    let p = PathBuf::from(raw);
    if !p.is_absolute() {
        tracing::warn!(
            env = ENV_HOME,
            path = %p.display(),
            "RMLX_HOME must be absolute; ignoring"
        );
        return None;
    }
    Some(p)
}

fn workspace_root() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let mut cur: &Path = cwd.as_path();
    loop {
        if cur.join("Cargo.lock").is_file() {
            return Some(cur.to_path_buf());
        }
        match cur.parent() {
            Some(p) => cur = p,
            None => return None,
        }
    }
}

fn user_home_rmlx() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(PathBuf::from(home).join(".rmlx"))
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "paths_tests.rs"]
mod tests;
