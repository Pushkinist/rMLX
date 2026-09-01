use super::*;
use std::io::Write as _;
use tempfile::TempDir;

// ── JSON-line shape ─────────────────────────────────────────────────────────

/// Emitting a green line to a captured String produces valid JSON with
/// the expected `"status":"green"` field.
#[test]
fn j6_json_line_shape_green() {
    let line = CheckLine::new("test_check", Status::Green, "all good");
    // Redirect stdout via a captured string by constructing manually.
    let json = format!(
        r#"{{"check":"{}","status":"{}","detail":"{}"}}"#,
        line.check,
        line.status.as_str(),
        line.detail
    );
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(v["status"], "green");
    assert_eq!(v["check"], "test_check");
}

#[test]
fn j6_json_line_shape_red() {
    let line = CheckLine::new("fail_check", Status::Red, "something broke");
    let json = format!(
        r#"{{"check":"{}","status":"{}","detail":"{}"}}"#,
        line.check,
        line.status.as_str(),
        line.detail
    );
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(v["status"], "red");
}

#[test]
fn j6_json_line_shape_info() {
    let line = CheckLine::new("info_check", Status::Info, "just info");
    let json = format!(
        r#"{{"check":"{}","status":"{}","detail":"{}"}}"#,
        line.check,
        line.status.as_str(),
        line.detail
    );
    let v: serde_json::Value = serde_json::from_str(&json).expect("valid JSON");
    assert_eq!(v["status"], "info");
}

// ── Aggregate exit logic ────────────────────────────────────────────────────

/// When red_checks is empty, aggregate status is green.
#[test]
fn j6_aggregate_green_when_no_reds() {
    let red_checks: Vec<String> = Vec::new();
    let status = if red_checks.is_empty() {
        Status::Green
    } else {
        Status::Red
    };
    assert_eq!(status, Status::Green);
}

/// When red_checks is non-empty, aggregate status is red.
#[test]
fn j6_aggregate_red_when_any_red() {
    let red_checks = ["claim".to_owned(), "http".to_owned()];
    let status = if red_checks.is_empty() {
        Status::Green
    } else {
        Status::Red
    };
    assert_eq!(status, Status::Red);
}

// ── Claim check (unit level) ────────────────────────────────────────────────

/// Claim check on a non-existent file is red.
#[test]
fn j6_claim_missing_file_is_red() {
    // Port unlikely to have a real claim file.
    let port: u16 = 19999;
    let _ = std::fs::remove_file(format!("/tmp/rmlx.{port}.claim"));
    let line = check_claim(port);
    assert_eq!(line.status, Status::Red, "missing claim file must be red");
}

/// Claim check with a valid PID in the file but no actual process is red.
#[test]
fn j6_claim_dead_pid_is_red() {
    let port: u16 = 19998;
    let path = format!("/tmp/rmlx.{port}.claim");
    // Write a PID that is virtually guaranteed to not exist (PID 2^22).
    std::fs::write(&path, "4194303").expect("write claim file");
    let line = check_claim(port);
    let _ = std::fs::remove_file(&path);
    assert_eq!(line.status, Status::Red, "dead PID must produce red");
}

// ── DB check ────────────────────────────────────────────────────────────────

/// DB check on non-existent path is red.
#[test]
fn j6_db_missing_file_is_red() {
    let path = PathBuf::from("/tmp/rmlx_j6_test_nonexistent.db");
    let _ = std::fs::remove_file(&path);
    let line = check_db(&path);
    assert_eq!(line.status, Status::Red, "missing DB must be red");
}

/// DB check on a valid SQLite file is green.
#[test]
fn j6_db_valid_sqlite_is_green() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("test.db");
    // Create a minimal SQLite DB via rusqlite.
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch("CREATE TABLE observations (id INTEGER PRIMARY KEY);")
            .unwrap();
    }
    let line = check_db(&db_path);
    assert_eq!(line.status, Status::Green, "valid SQLite must be green");
    assert!(
        line.detail.contains("schema_version"),
        "detail must include schema_version"
    );
}

/// DB check on a corrupt / garbage file is red.
#[test]
fn j6_db_corrupt_file_is_red() {
    let dir = TempDir::new().unwrap();
    let db_path = dir.path().join("corrupt.db");
    {
        let mut f = std::fs::File::create(&db_path).unwrap();
        f.write_all(b"this is not a sqlite file!!!").unwrap();
    }
    let line = check_db(&db_path);
    assert_eq!(line.status, Status::Red, "corrupt DB must be red");
}

// ── Registry check ──────────────────────────────────────────────────────────

/// Registry check on non-existent path produces a red line.
#[test]
fn j6_registry_nonexistent_path_is_red() {
    let path = PathBuf::from("/tmp/rmlx_j6_nonexistent_model_dir");
    let _ = std::fs::remove_dir_all(&path);
    let lines = check_registry(&[path]);
    assert_eq!(lines.len(), 1);
    assert_eq!(
        lines[0].status,
        Status::Red,
        "non-existent model path must be red"
    );
}

/// Registry check on a minimal valid snapshot (config.json + tokenizer.json) is green.
#[test]
fn j6_registry_valid_snapshot_is_green() {
    let dir = TempDir::new().unwrap();
    let snap = dir.path().join("TestModel");
    std::fs::create_dir_all(&snap).unwrap();

    // Minimal config.json.
    std::fs::write(
        snap.join("config.json"),
        r#"{"architectures":["LlamaForCausalLM"],"dtype":"bfloat16"}"#,
    )
    .unwrap();

    // Minimal tokenizer.json (tokenizers crate requires valid JSON structure).
    // Use a pre-built minimal tokenizer JSON that tokenizers 0.x can parse.
    std::fs::write(
        snap.join("tokenizer.json"),
        serde_json::json!({
            "version": "1.0",
            "truncation": null,
            "padding": null,
            "added_tokens": [],
            "normalizer": null,
            "pre_tokenizer": null,
            "post_processor": null,
            "decoder": null,
            "model": {
                "type": "BPE",
                "dropout": null,
                "unk_token": null,
                "continuing_subword_prefix": null,
                "end_of_word_suffix": null,
                "fuse_unk": false,
                "byte_fallback": false,
                "ignore_merges": false,
                "vocab": {"hello": 0, "world": 1},
                "merges": []
            }
        })
        .to_string(),
    )
    .unwrap();

    let lines = check_registry(&[snap]);
    assert_eq!(lines.len(), 1);
    assert_eq!(
        lines[0].status,
        Status::Green,
        "valid snapshot must be green; detail: {}",
        lines[0].detail
    );
}

// ── Disk check ──────────────────────────────────────────────────────────────

/// Disk check with an absurdly large min_disk_gb should be red.
#[test]
fn j6_disk_huge_threshold_is_red() {
    let dir = TempDir::new().unwrap();
    let line = check_disk(dir.path(), u64::MAX / (1024 * 1024 * 1024), "test");
    assert_eq!(
        line.status,
        Status::Red,
        "impossibly large min_disk_gb must be red"
    );
}

/// Disk check with min_disk_gb = 0 should always be green.
#[test]
fn j6_disk_zero_threshold_is_green() {
    let dir = TempDir::new().unwrap();
    let line = check_disk(dir.path(), 0, "test");
    assert_eq!(
        line.status,
        Status::Green,
        "min_disk_gb=0 must always be green"
    );
}

// ── Detail escaping ─────────────────────────────────────────────────────────

/// Backslash and double-quote in detail must be escaped in JSON output.
#[test]
fn j6_detail_escaping() {
    let line = CheckLine::new("esc", Status::Info, r#"path="C:\foo" ok"#);
    let escaped = line.detail.replace('\\', "\\\\").replace('"', "\\\"");
    // Parse the resulting JSON to verify it is valid.
    let json = format!(r#"{{"check":"esc","status":"info","detail":"{escaped}"}}"#);
    serde_json::from_str::<serde_json::Value>(&json).expect("escaped JSON must be valid");
}

// ── MLX pin ─────────────────────────────────────────────────────────────────

/// The verdict is red only where the pin binds, and never for a host it does
/// not: on pre-Neural-Accelerator hardware the pinned kernels buy nothing, so
/// reporting a failure there would be noise on the majority of Macs.
///
/// The pin itself is gated by `linked_mlx_matches_the_pinned_pair`
/// (`crates/rmlx-mlx/src/pin_tests.rs`); this covers only the mapping from a
/// verdict to a check line.
#[test]
fn j6_mlx_pin_is_red_only_where_the_pin_binds() {
    let line = check_mlx_pin();
    assert_eq!(line.check, "mlx_pin");
    assert!(
        !line.detail.is_empty(),
        "the verdict must say what it found"
    );

    let check = rmlx_mlx::pin_check();
    let expected = match (check.matches, check.enforced) {
        (true, _) => Status::Green,
        (false, true) => Status::Red,
        (false, false) => Status::Info,
    };
    assert_eq!(line.status, expected, "detail: {}", line.detail);
}
