//! The kernel-gate flags must resolve for **every** subcommand, not just `serve`.
//!
//! `--turbo-flash`, `--turbo-flash-lock` and `--planar-flash-decode` drive
//! process-wide `OnceLock` gates in `rmlx-kv-quant`. When they were resolved
//! inside `run_serve`, a measurement command (`bench`, `baseline`, `eval`) ran
//! with the gates unresolved while `serve` on the same host resolved
//! `--turbo-flash=auto` to ON — so the instrument benchmarked a different
//! kernel set than production. These tests pin the flags to the top level
//! (`global = true`, resolved in `main`) so that cannot drift back.
//!
//! `rmlx profile list` is the probe: it is a pure file-read admin command that
//! short-circuits before any model load or Metal claim, yet it runs *after* the
//! gate resolution in `main`. That makes it a model-free, GPU-free witness for
//! "the gates were resolved for this subcommand".

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    clippy::print_stderr
)]

use std::process::Command;

struct RunResult {
    exit_code: i32,
    stderr: String,
}

/// Run the built `rmlx` binary with an isolated `RMLX_HOME` and `RUST_LOG=info`
/// so the gate-resolution `tracing::info!` lines are emitted.
fn run(args: &[&str]) -> RunResult {
    let rmlx_home = std::env::temp_dir().join(format!("rmlx_gate_{}_{}", std::process::id(), {
        use std::sync::atomic::{AtomicU64, Ordering};
        static CTR: AtomicU64 = AtomicU64::new(0);
        CTR.fetch_add(1, Ordering::Relaxed)
    }));
    let out = Command::new(env!("CARGO_BIN_EXE_rmlx"))
        .env("RUST_LOG", "info")
        // A stale shell value would latch the OnceLock and mask the flag.
        .env_remove("RMLX_TURBO_FLASH")
        .env_remove("RMLX_TURBO_FLASH_LOCK")
        .env_remove("RMLX_PLANAR_FLASH_DECODE")
        .env("RMLX_HOME", &rmlx_home)
        .args(args)
        .output()
        .expect("failed to spawn rmlx subprocess");
    RunResult {
        exit_code: out.status.code().unwrap_or(-1),
        stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
    }
}

/// Both gates resolve on a non-`serve` subcommand, and the flags parse before it.
#[test]
fn kernel_gates_resolve_for_non_serve_subcommand() {
    let r = run(&[
        "--turbo-flash",
        "off",
        "--planar-flash-decode",
        "off",
        "profile",
        "list",
    ]);
    assert_eq!(r.exit_code, 0, "expected exit 0; stderr was: {}", r.stderr);
    assert!(
        r.stderr.contains("--turbo-flash resolved OFF"),
        "TurboFlash gate was not resolved for `profile list`; stderr: {}",
        r.stderr
    );
    assert!(
        r.stderr.contains("--planar-flash-decode resolved OFF"),
        "planar-flash-decode gate was not resolved for `profile list`; stderr: {}",
        r.stderr
    );
}

/// The gates resolve even when no flag is passed — `auto` still runs through
/// the hardware probe for every subcommand.
#[test]
fn kernel_gates_resolve_without_any_flag() {
    let r = run(&["profile", "list"]);
    assert_eq!(r.exit_code, 0, "expected exit 0; stderr was: {}", r.stderr);
    assert!(
        r.stderr.contains("--turbo-flash resolved"),
        "TurboFlash `auto` was not resolved for `profile list`; stderr: {}",
        r.stderr
    );
    assert!(
        r.stderr.contains("--planar-flash-decode resolved"),
        "planar-flash-decode `auto` was not resolved for `profile list`; stderr: {}",
        r.stderr
    );
}

/// `global = true` keeps the pre-existing `rmlx serve --turbo-flash …` spelling
/// working: the flag is still accepted *after* the subcommand.
#[test]
fn kernel_gate_flags_accepted_after_the_subcommand() {
    let r = run(&["profile", "list", "--turbo-flash", "on"]);
    assert_eq!(
        r.exit_code, 0,
        "post-subcommand flag position must still parse; stderr: {}",
        r.stderr
    );
    assert!(
        r.stderr.contains("--turbo-flash resolved ON"),
        "explicit `on` did not resolve ON; stderr: {}",
        r.stderr
    );
}

/// `--turbo-flash-lock` is a global toggle too, and only sets its env var when
/// passed.
#[test]
fn turbo_flash_lock_is_global_and_opt_in() {
    let off = run(&["profile", "list"]);
    assert!(
        !off.stderr.contains("--turbo-flash-lock flag set"),
        "lock must stay off when the flag is absent; stderr: {}",
        off.stderr
    );
    let on = run(&["--turbo-flash-lock", "profile", "list"]);
    assert_eq!(on.exit_code, 0, "stderr: {}", on.stderr);
    assert!(
        on.stderr.contains("--turbo-flash-lock flag set"),
        "lock flag was not honoured on a non-serve subcommand; stderr: {}",
        on.stderr
    );
}
