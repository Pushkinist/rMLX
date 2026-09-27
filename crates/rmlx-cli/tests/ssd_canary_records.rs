//! The three phase records `scripts/ssd_canary.sh` files are three cells.
//!
//! Prompts are content-addressed and part of the cell key, so records that
//! share a prompt body share a cell whatever their prompt names say. The
//! records here have the shape the script's `emit_and_ingest` builds, identity
//! block included, and go through `rmlx metrics record` like the script's do.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::{json, Value};

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

/// `rmlx <args>` with its data root in `home`, so no run log lands elsewhere.
fn rmlx(home: &Path, args: &[&str]) -> std::process::Output {
    Command::new(rmlx_bin())
        .args(args)
        .env("RMLX_HOME", home)
        .env_remove("RMLX_METRICS_DB")
        .output()
        .expect("launch rmlx")
}

/// One phase record as the canary builds it, with `content` as the prompt's
/// one message.
fn phase_record(identity: &Value, tag: &str, content: &str) -> Value {
    let mut rec = identity.clone();
    let fields = json!({
        "git_sha": "abc1234",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e2b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "bf16",
        "ctx_max": 8192,
        "prompt": {
            "name": tag,
            "body": [{"role": "user", "content": content}],
        },
        "ts_utc": "2026-09-27T10:00:00Z",
        "temperature": 0.0,
        "seed": 42,
        "notes": format!("ssd_canary {tag} phase"),
        "description": format!("ssd_canary tag={tag} sha=abc1234"),
        "metrics": [{"name": "ssd_bytes_used", "value": 4096.0}],
    });
    rec.as_object_mut()
        .unwrap()
        .extend(fields.as_object().unwrap().clone());
    rec
}

/// Ingest one record per phase tag and count the distinct prompt cells.
fn cells_for(content_of: impl Fn(&str) -> String) -> i64 {
    let td = tempfile::tempdir().unwrap();
    let home = td.path().join("home");
    let db = td.path().join("runs.db");
    let out = rmlx(&home, &["metrics", "identity", "--json"]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let identity: Value = serde_json::from_slice(&out.stdout).expect("identity JSON");
    for tag in [
        "ssd-canary-populate",
        "ssd-canary-revisit",
        "ssd-canary-evict",
    ] {
        let file = td.path().join(format!("{tag}.json"));
        std::fs::write(
            &file,
            phase_record(&identity, tag, &content_of(tag)).to_string(),
        )
        .unwrap();
        let out = rmlx(
            &home,
            &[
                "metrics",
                "--db",
                db.to_str().unwrap(),
                "record",
                "--file",
                file.to_str().unwrap(),
            ],
        );
        assert!(
            out.status.success(),
            "{}",
            String::from_utf8_lossy(&out.stderr)
        );
    }
    let conn = rusqlite::Connection::open(&db).unwrap();
    conn.query_row(
        "SELECT COUNT(DISTINCT prompt_id) FROM observations",
        [],
        |r| r.get(0),
    )
    .unwrap()
}

#[test]
fn each_phase_body_is_its_own_cell() {
    assert_eq!(cells_for(|tag| format!("ssd_canary {tag}")), 3);
}

/// The control: one body for all three phases collapses them into one cell,
/// which is what the per-phase body exists to prevent.
#[test]
fn one_shared_body_is_one_cell() {
    assert_eq!(cells_for(|_| "ssd_canary batch".to_owned()), 1);
}
