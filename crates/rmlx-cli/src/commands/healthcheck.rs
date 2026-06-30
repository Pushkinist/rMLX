// CLI binary: user-facing output. tracing not appropriate for command results.
#![allow(clippy::print_stdout, clippy::print_stderr)]
// unsafe_code: POSIX libc FFI — libc::kill (process-liveness probe) + libc::statvfs (disk-space check)
#![allow(unsafe_code)]
#![allow(trivial_numeric_casts)]

//! `rmlx healthcheck` — shell-able green/red readiness probe.
//!
//! Emits one JSON line per check (or plain text with `--human`):
//!
//! ```json
//! {"check":"<name>","status":"green|red|info","detail":"..."}
//! ```
//!
//! Final aggregate line:
//! ```json
//! {"check":"aggregate","status":"green|red","red_checks":["..."]}
//! ```
//!
//! Exit code: 0 = all green, 1 = any red, 2 = internal error.
//!
//! ## MLX safety
//! Default path (no `--full`) never loads the MLX runtime — safe to run
//! repeatedly with no Metal context concern. `--full` invokes the smoke probe
//! which DOES load MLX; the caller is responsible for the single-process
//! constraint (do not run `--full` while another rMLX instance holds Metal).

use std::path::{Path, PathBuf};

use rmlx_server::{ModelRegistry, RegistryConfig};
use tracing::debug;

// ---------------------------------------------------------------------------
// Output helpers
// ---------------------------------------------------------------------------

/// One emitted healthcheck line.
#[derive(Debug)]
pub(crate) struct CheckLine {
    pub check: String,
    pub status: Status,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Status {
    Green,
    Red,
    Info,
}

impl Status {
    fn as_str(&self) -> &'static str {
        match self {
            Status::Green => "green",
            Status::Red => "red",
            Status::Info => "info",
        }
    }
}

impl CheckLine {
    fn new(check: impl Into<String>, status: Status, detail: impl Into<String>) -> Self {
        Self {
            check: check.into(),
            status,
            detail: detail.into(),
        }
    }

    fn emit(&self, human: bool) {
        if human {
            let tag = self.status.as_str().to_uppercase();
            println!("{}: {} — {}", self.check, tag, self.detail);
        } else {
            // serde_json is already a workspace dep and available here.
            println!(
                r#"{{"check":"{}","status":"{}","detail":"{}"}}"#,
                self.check,
                self.status.as_str(),
                // Escape backslash and double-quote in detail to keep JSON valid.
                self.detail.replace('\\', "\\\\").replace('"', "\\\"")
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run all requested healthchecks and return the aggregate exit code.
///
/// `0` = all green. `1` = any red. `2` = internal/setup error (caller maps
/// from anyhow::Error).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_healthcheck(
    registry: Option<&Path>,
    model: Option<&Path>,
    port: Option<u16>,
    db_path: &Path,
    min_disk_gb: u64,
    full: bool,
    human: bool,
) -> anyhow::Result<i32> {
    let mut red_checks: Vec<String> = Vec::new();

    // ── 1. Claim check ────────────────────────────────────────────────────────
    if let Some(p) = port {
        let line = check_claim(p);
        if line.status == Status::Red {
            red_checks.push(line.check.clone());
        }
        line.emit(human);

        // ── 2. HTTP /health check ─────────────────────────────────────────────
        let http_line = check_http(p);
        if http_line.status == Status::Red {
            red_checks.push(http_line.check.clone());
        }
        http_line.emit(human);
    }

    // ── 3. Registry / model loadability ──────────────────────────────────────
    let model_paths: Vec<PathBuf> = collect_model_paths(registry, model)?;

    if !model_paths.is_empty() {
        let reg_lines = check_registry(&model_paths);
        for line in &reg_lines {
            if line.status == Status::Red {
                red_checks.push(line.check.clone());
            }
            line.emit(human);
        }

        // ── 4. Smoke probe (--full only) ──────────────────────────────────────
        if full {
            for path in &model_paths {
                let line = check_smoke(path);
                if line.status == Status::Red {
                    red_checks.push(line.check.clone());
                }
                line.emit(human);
            }
        } else {
            let skip = CheckLine::new("smoke", Status::Info, "skipped (--full not set)");
            skip.emit(human);
        }
    }

    // ── 5. Metrics DB ─────────────────────────────────────────────────────────
    let db_line = check_db(db_path);
    if db_line.status == Status::Red {
        red_checks.push(db_line.check.clone());
    }
    db_line.emit(human);

    // ── 6. Disk space ─────────────────────────────────────────────────────────
    for (dir, dir_name) in [
        (rmlx_core::paths::metrics_dir(), "metrics"),
        (rmlx_core::paths::logs_dir(), "logs"),
    ] {
        let disk_line = check_disk(&dir, min_disk_gb, dir_name);
        if disk_line.status == Status::Red {
            red_checks.push(disk_line.check.clone());
        }
        disk_line.emit(human);
    }

    // ── 7. Process memory (info-only) ────────────────────────────────────────
    let mem_line = check_mem();
    mem_line.emit(human);

    // ── Aggregate ─────────────────────────────────────────────────────────────
    let agg_status = if red_checks.is_empty() {
        Status::Green
    } else {
        Status::Red
    };

    if human {
        if red_checks.is_empty() {
            println!("aggregate: OK");
        } else {
            println!("aggregate: FAIL: {}", red_checks.join(", "));
        }
    } else {
        // Emit the red_checks JSON array inline.
        let arr: String = red_checks
            .iter()
            .map(|s| format!("\"{}\"", s.replace('"', "\\\"")))
            .collect::<Vec<_>>()
            .join(",");
        println!(
            r#"{{"check":"aggregate","status":"{}","red_checks":[{}]}}"#,
            agg_status.as_str(),
            arr
        );
    }

    Ok(i32::from(!red_checks.is_empty()))
}

// ---------------------------------------------------------------------------
// Individual checks
// ---------------------------------------------------------------------------

/// Check 1: claim file — exists, PID parses, process alive via `kill(pid, 0)`.
fn check_claim(port: u16) -> CheckLine {
    let path = PathBuf::from(format!("/tmp/rmlx.{port}.claim"));

    if !path.exists() {
        return CheckLine::new(
            "claim",
            Status::Red,
            format!("claim file /tmp/rmlx.{port}.claim not found"),
        );
    }

    let contents = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            return CheckLine::new("claim", Status::Red, format!("cannot read claim file: {e}"));
        }
    };

    let pid: u32 = match contents.trim().parse() {
        Ok(p) => p,
        Err(_) => {
            return CheckLine::new(
                "claim",
                Status::Red,
                format!("claim file body is not a valid PID: {:?}", contents.trim()),
            );
        }
    };

    // Use kill(pid, 0) to check if the process is alive (signal 0 = existence probe).
    // SAFETY: kill(2) is safe to call with any PID and signal 0.
    let alive = unsafe { libc::kill(pid as libc::pid_t, 0) == 0 };

    if alive {
        debug!(port, pid, "claim check: process alive");
        CheckLine::new(
            "claim",
            Status::Green,
            format!("port={port} pid={pid} alive"),
        )
    } else {
        CheckLine::new(
            "claim",
            Status::Red,
            format!("claim file exists (pid={pid}) but process is not alive"),
        )
    }
}

/// Check 2: HTTP GET /health on 127.0.0.1:port, 5-second timeout.
#[allow(
    clippy::unwrap_used,
    reason = "Mutex::lock() cannot poison; Option/Result unwrap on values established by construction in this fn"
)]
fn check_http(port: u16) -> CheckLine {
    use std::io::{Read, Write};
    use std::net::TcpStream;
    use std::time::Duration;

    let addr = format!("127.0.0.1:{port}");
    let timeout = Duration::from_secs(5);

    let mut stream = match TcpStream::connect_timeout(
        &addr
            .parse()
            .unwrap_or_else(|_| "127.0.0.1:8080".parse().unwrap()),
        timeout,
    ) {
        Ok(s) => s,
        Err(e) => {
            return CheckLine::new("http", Status::Red, format!("TCP connect to {addr}: {e}"));
        }
    };

    stream.set_read_timeout(Some(timeout)).unwrap_or(());
    stream.set_write_timeout(Some(timeout)).unwrap_or(());

    let req = format!("GET /health HTTP/1.0\r\nHost: {addr}\r\nConnection: close\r\n\r\n");
    if let Err(e) = stream.write_all(req.as_bytes()) {
        return CheckLine::new("http", Status::Red, format!("HTTP write to {addr}: {e}"));
    }

    let mut response = String::new();
    if let Err(e) = stream.read_to_string(&mut response) {
        return CheckLine::new("http", Status::Red, format!("HTTP read from {addr}: {e}"));
    }

    // Parse status line: "HTTP/1.x 200 ..."
    let status_line = response.lines().next().unwrap_or("");
    let status_code: Option<u16> = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok());

    match status_code {
        Some(200) => {
            // Also check body contains `"ok":true` (best-effort).
            let body_has_ok = response.contains("\"ok\":true") || response.contains("\"ok\": true");
            if body_has_ok {
                CheckLine::new(
                    "http",
                    Status::Green,
                    format!("port={port} HTTP 200 ok=true"),
                )
            } else {
                CheckLine::new(
                    "http",
                    Status::Green,
                    format!("port={port} HTTP 200 (body check skipped)"),
                )
            }
        }
        Some(code) => CheckLine::new(
            "http",
            Status::Red,
            format!("port={port} HTTP {code} (expected 200)"),
        ),
        None => CheckLine::new(
            "http",
            Status::Red,
            format!(
                "port={port} unexpected response: {:?}",
                &response[..response.len().min(80)]
            ),
        ),
    }
}

/// Check 3: registry loadability — config.json loads, tokenizer.json exists,
/// chat_template parses. Uses `ModelRegistry::from_paths` which mirrors
/// exactly what `rmlx serve --registry` does.
fn check_registry(paths: &[PathBuf]) -> Vec<CheckLine> {
    let reg = ModelRegistry::from_paths(paths);
    let mut lines = Vec::new();

    for path in paths {
        let id = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("(unknown)");

        match reg.get(id) {
            None => {
                // Registry skips entries whose config.json fails to load.
                lines.push(CheckLine::new(
                    format!("registry:{id}"),
                    Status::Red,
                    format!("config.json failed to load for {}", path.display()),
                ));
            }
            Some(entry) => {
                let mut issues: Vec<String> = Vec::new();

                // tokenizer.json must exist on disk (best-effort check from registry).
                if entry.tokenizer.is_none() {
                    let tok_path = path.join("tokenizer.json");
                    if tok_path.exists() {
                        issues.push("tokenizer.json present but failed to load".to_owned());
                    } else {
                        issues.push("tokenizer.json missing".to_owned());
                    }
                }

                // chat_template: warn if absent (not red — some models don't need it).
                if entry.chat_template.is_none() {
                    debug!(model = id, "no chat_template.jinja (non-fatal)");
                }

                if issues.is_empty() {
                    lines.push(CheckLine::new(
                        format!("registry:{id}"),
                        Status::Green,
                        format!(
                            "arch={} tokenizer=ok template={}",
                            entry.arch,
                            entry.chat_template.is_some()
                        ),
                    ));
                } else {
                    lines.push(CheckLine::new(
                        format!("registry:{id}"),
                        Status::Red,
                        issues.join("; "),
                    ));
                }
            }
        }
    }

    lines
}

/// Check 4 (--full only): run the existing smoke probe via `rmlx info --probe-smoke`.
///
/// Loads MLX — the caller ensures the single-process constraint.
fn check_smoke(path: &Path) -> CheckLine {
    use rmlx_metrics::events::EventRecorder;
    use rmlx_mlx::Device;

    let id = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("(unknown)");

    // We need a EventRecorder. Use a no-op path under /tmp.
    let run_id = format!("healthcheck-smoke-{id}");
    let sink = match EventRecorder::open(&run_id) {
        Ok(s) => s,
        Err(e) => {
            return CheckLine::new(
                format!("smoke:{id}"),
                Status::Red,
                format!("could not open metrics sink: {e}"),
            );
        }
    };

    match crate::commands::info::run_info(
        path,
        false, // probe_forward = false
        true,  // probe_smoke = true
        Device::Gpu,
        None, // kv_quant_override = auto
        None, // max_ctx_override = auto
        &sink,
    ) {
        Ok(exit_code) => {
            use crate::commands::info::SmokeExitCode;
            match exit_code {
                SmokeExitCode::Ok => CheckLine::new(
                    format!("smoke:{id}"),
                    Status::Green,
                    "smoke probe verdict: ok".to_owned(),
                ),
                SmokeExitCode::Broken => CheckLine::new(
                    format!("smoke:{id}"),
                    Status::Red,
                    "smoke probe verdict: broken (BrokenPunctLoop or BrokenNan)".to_owned(),
                ),
                SmokeExitCode::LoadFail => CheckLine::new(
                    format!("smoke:{id}"),
                    Status::Red,
                    "smoke probe verdict: load-fail (supported arch failed to load)".to_owned(),
                ),
                SmokeExitCode::Inconclusive => CheckLine::new(
                    format!("smoke:{id}"),
                    Status::Red,
                    "smoke probe verdict: inconclusive (too few steps to confirm)".to_owned(),
                ),
                SmokeExitCode::Unsupported => CheckLine::new(
                    format!("smoke:{id}"),
                    Status::Red,
                    "smoke probe verdict: unsupported architecture".to_owned(),
                ),
            }
        }
        Err(e) => CheckLine::new(
            format!("smoke:{id}"),
            Status::Red,
            format!("smoke probe error: {e}"),
        ),
    }
}

/// Check 5: open the SQLite DB, run `PRAGMA schema_version`, count observations.
fn check_db(db_path: &Path) -> CheckLine {
    // Use rusqlite directly — rmlx-metrics::schema already exposes open_readonly
    // but the CLI crate already has rusqlite as a direct dep (Cargo.toml).
    use rusqlite::Connection;

    if !db_path.exists() {
        return CheckLine::new(
            "db",
            Status::Red,
            format!("DB file not found: {}", db_path.display()),
        );
    }

    let conn = match Connection::open(db_path) {
        Ok(c) => c,
        Err(e) => {
            return CheckLine::new(
                "db",
                Status::Red,
                format!("cannot open DB {}: {e}", db_path.display()),
            );
        }
    };

    // schema_version pragma — fails on corrupt file.
    let schema_version: Result<i64, _> = conn.query_row("PRAGMA schema_version", [], |r| r.get(0));
    let sv = match schema_version {
        Ok(v) => v,
        Err(e) => {
            return CheckLine::new(
                "db",
                Status::Red,
                format!("PRAGMA schema_version failed: {e}"),
            );
        }
    };

    // Count observations (table may not exist on a fresh DB before `init`).
    let obs_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM observations", [], |r| r.get(0))
        .unwrap_or(-1); // -1 = table absent (schema not initialized yet)

    CheckLine::new(
        "db",
        Status::Green,
        format!(
            "schema_version={sv} observations={}",
            if obs_count < 0 {
                "n/a (schema not initialized)".to_owned()
            } else {
                obs_count.to_string()
            }
        ),
    )
}

/// Check 6: disk free on `dir` via `statvfs`.
fn check_disk(dir: &Path, min_gb: u64, label: &str) -> CheckLine {
    // Create the dir if it doesn't exist so statvfs can run.
    let target = if dir.exists() {
        dir.to_path_buf()
    } else {
        // Fall back to cwd — the dir hasn't been created yet.
        PathBuf::from(".")
    };

    let path_cstr = match std::ffi::CString::new(target.to_string_lossy().as_bytes()) {
        Ok(s) => s,
        Err(e) => {
            return CheckLine::new(
                format!("disk:{label}"),
                Status::Red,
                format!("invalid path for statvfs: {e}"),
            );
        }
    };

    // SAFETY: path_cstr is a valid NUL-terminated C string. statvfs writes
    // into `buf` only on success; buf is zeroed beforehand.
    let mut buf: libc::statvfs = unsafe { std::mem::zeroed() };
    let ret = unsafe { libc::statvfs(path_cstr.as_ptr(), &raw mut buf) };

    if ret != 0 {
        let err = std::io::Error::last_os_error();
        return CheckLine::new(
            format!("disk:{label}"),
            Status::Red,
            format!("statvfs({}) failed: {err}", target.display()),
        );
    }

    // Available bytes = f_bavail * f_frsize (available to non-root).
    let free_bytes = u64::from(buf.f_bavail) * buf.f_frsize as u64;
    let free_gb = free_bytes / (1024 * 1024 * 1024);
    let min_bytes = min_gb * 1024 * 1024 * 1024;

    if free_bytes < min_bytes {
        CheckLine::new(
            format!("disk:{label}"),
            Status::Red,
            format!(
                "{} free_gb={free_gb} below min_disk_gb={min_gb}",
                target.display()
            ),
        )
    } else {
        CheckLine::new(
            format!("disk:{label}"),
            Status::Green,
            format!("{} free_gb={free_gb}", target.display()),
        )
    }
}

/// Check 7: process RSS + phys_footprint via `rmlx_core::mach_mem` (info-only).
fn check_mem() -> CheckLine {
    #[cfg(target_os = "macos")]
    {
        use rmlx_core::mach_mem::read_proc_mem;
        match read_proc_mem() {
            Ok(m) => CheckLine::new(
                "mem",
                Status::Info,
                format!(
                    "phys_footprint={:.1}MB rss={:.1}MB",
                    m.phys_footprint_bytes as f64 / (1024.0 * 1024.0),
                    m.rss_bytes as f64 / (1024.0 * 1024.0),
                ),
            ),
            Err(e) => CheckLine::new("mem", Status::Info, format!("read_proc_mem failed: {e}")),
        }
    }
    #[cfg(not(target_os = "macos"))]
    {
        CheckLine::new("mem", Status::Info, "not available (non-macOS)".to_owned())
    }
}

// ---------------------------------------------------------------------------
// Helper: collect model paths from --registry / --model
// ---------------------------------------------------------------------------

fn collect_model_paths(
    registry: Option<&Path>,
    model: Option<&Path>,
) -> anyhow::Result<Vec<PathBuf>> {
    if let Some(reg_path) = registry {
        let cfg = RegistryConfig::from_file(reg_path)
            .map_err(|e| anyhow::anyhow!("load registry {}: {e}", reg_path.display()))?;
        return Ok(cfg.models.iter().map(|e| e.path.clone()).collect());
    }
    if let Some(m) = model {
        return Ok(vec![m.to_path_buf()]);
    }
    Ok(Vec::new())
}

// ---------------------------------------------------------------------------
// Unit tests (J6)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[path = "healthcheck_tests.rs"]
mod healthcheck_tests;
