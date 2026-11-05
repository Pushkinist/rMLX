//! Integration test for the `baseline` subcommand.
//!
//! Runs the baseline harness against the primary snapshot with `--max-tokens 4`
//! and a short temporary prompt file.
//!
//! **Why `#[ignore]`**: Even with `--max-tokens 4`, model load + 4 decode steps
//! on GPU takes 30–120 s for the 4B Gemma4 mxfp8 model. This is unsuitable for
//! routine `cargo test` runs.
//!
//! Run explicitly:
//! cargo test -p rmlx-cli --test baseline_smoke -- --ignored --nocapture
//!
//! # Runtime home
//! Each test runs the binary with its own temporary `RMLX_HOME` and reads
//! `metrics/baseline.csv` from there.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::print_stdout,
    clippy::print_stderr,
    clippy::unreachable,
    clippy::todo,
    clippy::unimplemented,
    clippy::float_cmp,
    clippy::ignore_without_reason,
    clippy::unnecessary_debug_formatting
)]

mod common;

use std::io::Write as _;
use std::path::PathBuf;

use common::rmlx_at;

fn primary_model_dir() -> Option<PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(PathBuf::from)
}

/// End-to-end baseline subcommand smoke test — GPU device.
///
/// Skips when the primary snapshot is absent (offline dev, CI without model).
/// Asserts: stdout contains "baseline: " prefix; metrics/baseline.csv has at
/// least 1 data row (beyond the header) after the run.
#[test]
#[ignore]
fn baseline_command_gpu_produces_output_and_csv_row() {
    let Some(model_path) = primary_model_dir() else {
        eprintln!("[baseline_smoke] skipping: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    if !model_path.exists() {
        eprintln!("[baseline_smoke] primary snapshot absent at {model_path:?} — skipping");
        return;
    }

    let home = tempfile::tempdir().expect("create RMLX_HOME tempdir");

    // Write a tiny prompt to a temp file so we avoid the multi-hour full fixture.
    let mut prompt_file = tempfile::NamedTempFile::new().expect("create temp prompt file");
    writeln!(prompt_file, "The history of paper begins in ancient China.").expect("write prompt");
    let prompt_path = prompt_file.path().to_path_buf();

    let model_str = model_path.to_str().expect("model path is valid UTF-8");
    let output = rmlx_at(home.path())
        .args([
            "baseline",
            "--model",
            model_str,
            "--prompt",
            prompt_path.to_str().expect("prompt path is valid UTF-8"),
            "--device",
            "gpu",
            "--max-tokens",
            "4",
        ])
        .output()
        .expect("failed to launch rmlx binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("[baseline_smoke/gpu] stdout:\n{stdout}");
    eprintln!(
        "[baseline_smoke/gpu] stderr (first 2000 chars):\n{}",
        &stderr[..stderr.len().min(2000)]
    );

    assert!(
        output.status.success(),
        "[baseline_smoke/gpu] rmlx baseline exited with non-zero status: {}",
        output.status
    );

    assert!(
        stdout.contains("baseline: "),
        "[baseline_smoke/gpu] stdout does not contain 'baseline: ' summary line:\n{stdout}"
    );

    // `baseline` appends its row to `<RMLX_HOME>/metrics/baseline.csv`.
    let found_csv = home.path().join("metrics/baseline.csv");
    assert!(
        found_csv.exists(),
        "[baseline_smoke/gpu] metrics/baseline.csv was not created at {found_csv:?}"
    );

    let csv_content = std::fs::read_to_string(&found_csv).expect("read metrics/baseline.csv");
    let data_rows: Vec<&str> = csv_content
        .lines()
        .skip(1) // skip header
        .filter(|l| !l.trim().is_empty())
        .collect();

    assert!(
        !data_rows.is_empty(),
        "[baseline_smoke/gpu] metrics/baseline.csv has no data rows (only header):\n{csv_content}"
    );

    eprintln!(
        "[baseline_smoke/gpu] metrics/baseline.csv has {} data row(s) — ok",
        data_rows.len()
    );
}

/// CPU variant — kept as a faster fallback for offline/CI environments.
///
/// Uses `--device cpu` for compatibility when no GPU context is claimed.
/// Model load on CPU takes 2–5 minutes for the 4B snapshot.
#[test]
#[ignore]
fn baseline_command_cpu_produces_output_and_csv_row() {
    let Some(model_path) = primary_model_dir() else {
        eprintln!("[baseline_smoke] skipping: RMLX_TEST_MODEL_GEMMA4_E4B not set");
        return;
    };
    if !model_path.exists() {
        eprintln!("[baseline_smoke] primary snapshot absent at {model_path:?} — skipping");
        return;
    }

    let home = tempfile::tempdir().expect("create RMLX_HOME tempdir");

    let mut prompt_file = tempfile::NamedTempFile::new().expect("create temp prompt file");
    writeln!(prompt_file, "The history of paper begins in ancient China.").expect("write prompt");
    let prompt_path = prompt_file.path().to_path_buf();

    let model_str = model_path.to_str().expect("model path is valid UTF-8");
    let output = rmlx_at(home.path())
        .args([
            "baseline",
            "--model",
            model_str,
            "--prompt",
            prompt_path.to_str().expect("prompt path is valid UTF-8"),
            "--device",
            "cpu",
            "--max-tokens",
            "4",
        ])
        .output()
        .expect("failed to launch rmlx binary");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("[baseline_smoke/cpu] stdout:\n{stdout}");
    eprintln!(
        "[baseline_smoke/cpu] stderr (first 2000 chars):\n{}",
        &stderr[..stderr.len().min(2000)]
    );

    assert!(
        output.status.success(),
        "[baseline_smoke/cpu] rmlx baseline exited with non-zero status: {}",
        output.status
    );

    assert!(
        stdout.contains("baseline: "),
        "[baseline_smoke/cpu] stdout does not contain 'baseline: ' summary line:\n{stdout}"
    );

    let found_csv = home.path().join("metrics/baseline.csv");
    assert!(
        found_csv.exists(),
        "[baseline_smoke/cpu] metrics/baseline.csv was not created at {found_csv:?}"
    );

    let csv_content = std::fs::read_to_string(&found_csv).expect("read metrics/baseline.csv");
    let data_rows: Vec<&str> = csv_content
        .lines()
        .skip(1)
        .filter(|l| !l.trim().is_empty())
        .collect();

    assert!(
        !data_rows.is_empty(),
        "[baseline_smoke/cpu] metrics/baseline.csv has no data rows (only header):\n{csv_content}"
    );

    eprintln!(
        "[baseline_smoke/cpu] metrics/baseline.csv has {} data row(s) — ok",
        data_rows.len()
    );
}
