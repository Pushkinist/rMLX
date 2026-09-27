//! `rmlx metrics path [--home]` prints what every `rmlx metrics` command in the
//! same environment resolves: `--db` over `RMLX_METRICS_DB` over
//! `<RMLX_HOME>/metrics/runs.db`, and a relative `RMLX_HOME` ignored in favour
//! of the checkout found by walking up for `Cargo.lock`. Stdout is the path and
//! nothing else, because scripts read it with `$(...)`.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::Command;

fn rmlx_bin() -> PathBuf {
    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set by cargo");
    let debug = PathBuf::from(manifest_dir)
        .parent()
        .and_then(Path::parent)
        .expect("workspace root from CARGO_MANIFEST_DIR")
        .join("target/debug/rmlx");
    assert!(
        debug.exists(),
        "target/debug/rmlx not found at {}",
        debug.display()
    );
    debug
}

/// Run `rmlx metrics <args>` in `cwd` with exactly the given `RMLX_HOME` /
/// `RMLX_METRICS_DB`, and return stdout, which must be one line.
fn path(cwd: &Path, home: &str, metrics_db: Option<&Path>, args: &[&str]) -> String {
    let mut cmd = Command::new(rmlx_bin());
    cmd.current_dir(cwd)
        .arg("metrics")
        .args(args)
        .env("RMLX_HOME", home)
        .env("HOME", cwd)
        .env_remove("RMLX_METRICS_DB");
    if let Some(db) = metrics_db {
        cmd.env("RMLX_METRICS_DB", db);
    }
    let out = cmd.output().expect("launch rmlx");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).unwrap();
    assert_eq!(
        stdout.lines().count(),
        1,
        "stdout must be the path alone: {stdout:?}"
    );
    stdout.trim_end().to_owned()
}

fn tempdir() -> (tempfile::TempDir, PathBuf) {
    let td = tempfile::tempdir().unwrap();
    let real = td.path().canonicalize().unwrap();
    (td, real)
}

#[test]
fn home_db_is_under_rmlx_home() {
    let (_td, root) = tempdir();
    let home = root.join("home");
    let h = home.to_str().unwrap();
    assert_eq!(
        path(&root, h, None, &["path"]),
        format!("{h}/metrics/runs.db")
    );
    assert_eq!(path(&root, h, None, &["path", "--home"]), h);
}

#[test]
fn rmlx_metrics_db_beats_rmlx_home() {
    let (_td, root) = tempdir();
    let home = root.join("home");
    let env_db = root.join("env/runs.db");
    assert_eq!(
        path(&root, home.to_str().unwrap(), Some(&env_db), &["path"]),
        env_db.to_str().unwrap()
    );
}

#[test]
fn db_flag_beats_rmlx_metrics_db() {
    let (_td, root) = tempdir();
    let home = root.join("home");
    let env_db = root.join("env/runs.db");
    let flag_db = root.join("flag/runs.db");
    assert_eq!(
        path(
            &root,
            home.to_str().unwrap(),
            Some(&env_db),
            &["--db", flag_db.to_str().unwrap(), "path"]
        ),
        flag_db.to_str().unwrap()
    );
}

#[test]
fn relative_rmlx_home_is_ignored() {
    let (_td, root) = tempdir();
    std::fs::write(root.join("Cargo.lock"), "").unwrap();
    let expected = root.join(".rmlx");
    assert_eq!(
        path(&root, "relative-home", None, &["path", "--home"]),
        expected.to_str().unwrap()
    );
    assert_eq!(
        path(&root, "relative-home", None, &["path"]),
        expected.join("metrics/runs.db").to_str().unwrap()
    );
    assert!(!root.join("relative-home").exists());
}
