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
///
/// `RMLX_HOME` is a `TempDir` dropped at the end of the call, so the run's logs
/// and metrics DB are removed instead of accumulating under the system temp dir
/// on every `cargo test`.
fn run(args: &[&str]) -> RunResult {
    run_with_env(args, &[])
}

/// [`run`] plus extra environment variables, applied after the gate vars are
/// cleared so a test can deliberately pre-set one of them.
fn run_with_env(args: &[&str], extra_env: &[(&str, &str)]) -> RunResult {
    let rmlx_home = tempfile::TempDir::new().expect("failed to create RMLX_HOME tempdir");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rmlx"));
    cmd.env("RUST_LOG", "info")
        // A stale shell value would latch the OnceLock and mask the flag.
        .env_remove("RMLX_TURBO_FLASH")
        .env_remove("RMLX_TURBO_FLASH_LOCK")
        .env_remove("RMLX_PLANAR_FLASH_DECODE")
        .env("RMLX_HOME", rmlx_home.path())
        .args(args);
    for (k, v) in extra_env {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("failed to spawn rmlx subprocess");
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

/// `auto` honours a pre-existing `RMLX_TURBO_FLASH=1` for back-compat, which
/// means the flag resolves OFF while the kernel actually runs. That combination
/// is a known 2.0-4.25x decode loss, so it must be operator-visible rather than
/// silent: the resolution logs at `warn!` and names the cost.
#[test]
fn turbo_flash_auto_warns_when_the_env_opt_in_is_set() {
    let r = run_with_env(&["profile", "list"], &[("RMLX_TURBO_FLASH", "1")]);
    assert_eq!(r.exit_code, 0, "expected exit 0; stderr was: {}", r.stderr);
    assert!(
        r.stderr.contains("WARN") && r.stderr.contains("the kernel stays ON"),
        "a pre-set RMLX_TURBO_FLASH=1 under `auto` must warn that the kernel is \
         still ON; stderr: {}",
        r.stderr
    );
    // The plain "env untouched" info line is the *other* branch: it must not be
    // what an opted-in operator sees. Match the flag name too — the sibling
    // gates emit the same suffix on their own quiet branches.
    assert!(
        !r.stderr
            .contains("--turbo-flash resolved OFF; env untouched"),
        "the quiet OFF line must not fire when RMLX_TURBO_FLASH=1 is set; \
         stderr: {}",
        r.stderr
    );
}

/// With the env var absent, `auto` takes the quiet branch — no warn, no claim
/// that anything is still on.
#[test]
fn turbo_flash_auto_is_quiet_without_the_env_opt_in() {
    let r = run(&["profile", "list"]);
    assert_eq!(r.exit_code, 0, "expected exit 0; stderr was: {}", r.stderr);
    assert!(
        r.stderr
            .contains("--turbo-flash resolved OFF; env untouched"),
        "auto with no env opt-in must log the quiet OFF line; stderr: {}",
        r.stderr
    );
    assert!(
        !r.stderr.contains("the kernel stays ON"),
        "the opt-in warn must not fire when RMLX_TURBO_FLASH is unset; stderr: {}",
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
