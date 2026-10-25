//! Shared launchers for the integration tests that run the `rmlx` binary.
//!
//! Every launch names its runtime home, so no run reaches the checkout's or the
//! user's real `.rmlx/`.

// Each test binary uses a subset of these items.
#![allow(dead_code, unreachable_pub)]

use std::path::Path;
use std::process::Command;

/// The `rmlx` binary under test with `RMLX_HOME` set to `home`.
///
/// `home` must be absolute and outlive the command.
pub fn rmlx_at(home: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rmlx"));
    cmd.env("RMLX_HOME", home).env_remove("RMLX_METRICS_DB");
    cmd
}

/// `rmlx metrics --db <db>` with the directory holding `db` as its runtime home.
///
/// `db` must sit directly in a directory the test owns.
#[allow(
    clippy::expect_used,
    reason = "test launcher: a db path without a parent is a broken fixture"
)]
pub fn rmlx_metrics(db: &Path) -> Command {
    let home = db.parent().expect("a db path has a parent directory");
    let mut cmd = rmlx_at(home);
    cmd.args(["metrics", "--db"]).arg(db);
    cmd
}
