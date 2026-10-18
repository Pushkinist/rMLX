//! `rmlx eval spec-answer` exits 0 for a near tie, 1 for a divergence the
//! oracle refuses and 2 for an input it cannot judge, and prints its verdict as
//! `key=value` lines on stdout.
//!
//! Each test subprocesses the built `rmlx` binary over the response bodies in
//! `tests/fixtures/spec_answer/`. Nothing loads a model or takes the claim.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test assertions: a failed setup step is a test failure"
)]

use std::path::{Path, PathBuf};
use std::process::Command;

struct Run {
    code: Option<i32>,
    stdout: String,
}

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/spec_answer")
        .join(name)
}

fn spec_answer(home: &Path, plain: &Path, spec: &Path) -> Run {
    let out = Command::new(env!("CARGO_BIN_EXE_rmlx"))
        .args(["eval", "spec-answer", "--plain"])
        .arg(plain)
        .arg("--spec")
        .arg(spec)
        .env("RMLX_HOME", home)
        .env("HOME", home)
        .output()
        .expect("launch rmlx");
    Run {
        code: out.status.code(),
        stdout: String::from_utf8(out.stdout).expect("utf-8 stdout"),
    }
}

fn field<'a>(run: &'a Run, key: &str) -> Option<&'a str> {
    run.stdout
        .lines()
        .find_map(|line| line.strip_prefix(key)?.strip_prefix('='))
}

#[test]
fn a_flip_at_a_tied_decision_exits_zero() {
    let home = tempfile::tempdir().unwrap();
    let run = spec_answer(
        home.path(),
        &fixture("plain.json"),
        &fixture("spec_near_tie.json"),
    );
    assert_eq!(run.code, Some(0), "{}", run.stdout);
    assert_eq!(field(&run, "verdict"), Some("near_tie"));
    assert_eq!(field(&run, "position"), Some("5"));
    assert_eq!(field(&run, "margin"), Some("0.0000"));
    assert_eq!(field(&run, "rank"), Some("0.0000"));
    assert!(field(&run, "reason").is_some(), "{}", run.stdout);
}

#[test]
fn a_flip_at_a_confident_decision_exits_one() {
    let home = tempfile::tempdir().unwrap();
    let run = spec_answer(
        home.path(),
        &fixture("plain.json"),
        &fixture("spec_confident.json"),
    );
    assert_eq!(run.code, Some(1), "{}", run.stdout);
    assert_eq!(field(&run, "verdict"), Some("refused"));
    assert_eq!(field(&run, "position"), Some("2"));
    assert_eq!(field(&run, "margin"), Some("5.0000"));
    assert!(field(&run, "rank").is_some(), "{}", run.stdout);
}

#[test]
fn a_body_that_is_not_json_exits_two_and_names_the_file() {
    let home = tempfile::tempdir().unwrap();
    let run = spec_answer(
        home.path(),
        &fixture("plain.json"),
        &fixture("malformed.json"),
    );
    assert_eq!(run.code, Some(2), "{}", run.stdout);
    assert_eq!(field(&run, "verdict"), Some("unjudgeable"));
    assert!(
        field(&run, "reason").is_some_and(|r| r.contains("malformed.json")),
        "{}",
        run.stdout
    );
    assert_eq!(field(&run, "position"), None);
}

#[test]
fn a_missing_file_exits_two_and_names_the_file() {
    let home = tempfile::tempdir().unwrap();
    let run = spec_answer(
        home.path(),
        &fixture("plain.json"),
        &home.path().join("absent.json"),
    );
    assert_eq!(run.code, Some(2), "{}", run.stdout);
    assert!(
        field(&run, "reason").is_some_and(|r| r.contains("absent.json")),
        "{}",
        run.stdout
    );
}

#[test]
fn two_identical_bodies_exit_two() {
    let home = tempfile::tempdir().unwrap();
    let run = spec_answer(home.path(), &fixture("plain.json"), &fixture("plain.json"));
    assert_eq!(run.code, Some(2), "{}", run.stdout);
    assert_eq!(field(&run, "verdict"), Some("unjudgeable"));
}
