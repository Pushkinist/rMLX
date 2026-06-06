//! rMLX E2E feature-proof harness — entry point.
//!
//! Parses `tests/e2e/manifest.toml`, drives the real `rmlx` binary per case
//! (CLI subprocess or `rmlx serve` + HTTP), asserts on real output, and writes
//! the PASS/FAIL grid to `<RMLX_HOME>/e2e/report.{json,md}`.
//!
//! `#[ignore]` + `--test-threads=1` are mandatory: only one MLX process may
//! hold the Metal context per Mac (CLAUDE.md hard rule 8). Run via `make e2e`
//! or:
//!
//! ```bash
//! cargo test -p rmlx-cli --test e2e_harness -- --ignored --test-threads=1 --nocapture
//! ```
//!
//! When no Bonsai snapshot resolves, every model-gated case records SKIP and
//! the test passes — safe on machines without snapshots.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout
)]

#[path = "e2e/mod.rs"]
mod e2e;

use e2e::runner;

/// The manifest is embedded at compile time so the test binary is self-contained
/// (it still runs the *real* `rmlx` binary; only the case list is embedded).
const MANIFEST: &str = include_str!("e2e/manifest.toml");

#[ignore = "drives the real rmlx binary + Bonsai snapshot; run with --ignored --test-threads=1"]
#[test]
fn e2e_feature_proof_grid() {
    let report = runner::run_manifest(MANIFEST);
    let dir = runner::report_dir();
    let written = report.write(&dir).expect("write e2e report");

    let cases = report.results();
    let total = cases.len();
    println!("\n===== rMLX E2E grid =====");
    println!("report: {}", written.join("report.md").display());
    println!("json:   {}", written.join("report.json").display());
    println!("cases:  {total}");
    println!("=========================\n");

    // Fail the test only on a genuine FAIL verdict — SKIP / PENDING are fine.
    assert!(
        !report.any_failed(),
        "one or more E2E feature cases FAILED — see {}",
        written.join("report.md").display()
    );
}
