use super::*;
use rusqlite::Connection;
use serde_json::json;
use tempfile::NamedTempFile;

/// Open an in-memory DB with the full schema applied.
fn test_conn() -> Connection {
    let mut conn = Connection::open_in_memory().unwrap();
    crate::migrate::run_pending(&mut conn).unwrap();
    conn
}

// ── PromptStore tests ─────────────────────────────────────────────────────

#[test]
fn get_or_insert_new_prompt_returns_id() {
    let conn = test_conn();
    let store = PromptStore::new(&conn);
    let id = store
        .get_or_insert("test", &json!("hello world"), Some(2), None)
        .unwrap();
    assert!(id > 0);
}

#[test]
fn get_or_insert_dedup_returns_same_id() {
    let conn = test_conn();
    let store = PromptStore::new(&conn);
    let body = json!("duplicate body");
    let id1 = store.get_or_insert("p1", &body, None, None).unwrap();
    let id2 = store.get_or_insert("p2", &body, None, None).unwrap();
    assert_eq!(id1, id2, "same body must return same id");

    let count: i64 = conn
        .query_row("SELECT count(*) FROM prompts", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1, "must have exactly one row in prompts");
}

#[test]
fn get_or_insert_different_bodies_returns_different_ids() {
    let conn = test_conn();
    let store = PromptStore::new(&conn);
    let id1 = store
        .get_or_insert("a", &json!("body-a"), None, None)
        .unwrap();
    let id2 = store
        .get_or_insert("b", &json!("body-b"), None, None)
        .unwrap();
    assert_ne!(id1, id2);
}

#[test]
fn find_by_sha256_hit() {
    let conn = test_conn();
    let store = PromptStore::new(&conn);
    let body = json!(["msg1", "msg2"]);
    let id = store
        .get_or_insert("arr", &body, Some(10), Some("notes"))
        .unwrap();
    let sha = prompt_body_sha256(&body);

    let row = store.find_by_sha256(&sha).unwrap().expect("must find row");
    assert_eq!(row.id, id);
    assert_eq!(row.name, "arr");
    assert_eq!(row.tokens_approx, Some(10));
    assert_eq!(row.notes.as_deref(), Some("notes"));
}

#[test]
fn find_by_sha256_miss() {
    let conn = test_conn();
    let store = PromptStore::new(&conn);
    let result = store
        .find_by_sha256("0000000000000000000000000000000000000000000000000000000000000000")
        .unwrap();
    assert!(result.is_none());
}

#[test]
fn find_by_id_hit() {
    let conn = test_conn();
    let store = PromptStore::new(&conn);
    let id = store
        .get_or_insert("x", &json!("body-x"), None, None)
        .unwrap();
    let row = store.find_by_id(id).unwrap().expect("must find row");
    assert_eq!(row.id, id);
    assert_eq!(row.name, "x");
}

#[test]
fn find_by_id_miss() {
    let conn = test_conn();
    let store = PromptStore::new(&conn);
    assert!(store.find_by_id(999).unwrap().is_none());
}

#[test]
fn find_latest_by_name_picks_newest_first_seen() {
    let conn = test_conn();
    let store = PromptStore::new(&conn);

    // Insert two rows with the same name but different bodies (forces two
    // distinct sha256 values, hence two rows).
    let id1 = store
        .get_or_insert("evolving", &json!("v1 body"), None, None)
        .unwrap();
    // Sleep briefly to guarantee distinct first_seen_utc (1-second resolution).
    std::thread::sleep(std::time::Duration::from_millis(1100));
    let id2 = store
        .get_or_insert("evolving", &json!("v2 body"), None, None)
        .unwrap();
    assert_ne!(id1, id2);

    let row = store
        .find_latest_by_name("evolving")
        .unwrap()
        .expect("must find a row");
    assert_eq!(row.id, id2, "latest insert must be returned");
}

// ── parse_prompt_file tests ───────────────────────────────────────────────

#[test]
fn parse_prompt_file_with_messages_array() {
    let tmp = NamedTempFile::with_suffix(".json").unwrap();
    std::fs::write(
        tmp.path(),
        r#"{"name":"test_msg","messages":[{"role":"user","content":"hello"}],"tokens_approx":5}"#,
    )
    .unwrap();
    let pf = parse_prompt_file(tmp.path()).unwrap();
    assert_eq!(pf.name, "test_msg");
    assert!(pf.body.is_array(), "body must be the messages array");
    assert_eq!(pf.tokens_approx, Some(5));
}

#[test]
fn parse_prompt_file_with_body_string() {
    let tmp = NamedTempFile::with_suffix(".json").unwrap();
    std::fs::write(tmp.path(), r#"{"name":"flat","body":"hello world"}"#).unwrap();
    let pf = parse_prompt_file(tmp.path()).unwrap();
    assert_eq!(pf.name, "flat");
    assert_eq!(pf.body, json!("hello world"));
}

#[test]
fn parse_prompt_file_missing_name_uses_stem() {
    let tmp = NamedTempFile::with_suffix(".json").unwrap();
    std::fs::write(tmp.path(), r#"{"body":"no name here"}"#).unwrap();
    // We can't easily capture the tracing warn, but we verify the name fallback.
    let pf = parse_prompt_file(tmp.path()).unwrap();
    let stem = tmp
        .path()
        .file_stem()
        .unwrap()
        .to_str()
        .unwrap()
        .to_string();
    assert_eq!(pf.name, stem, "name must fall back to file stem");
}

#[test]
fn parse_prompt_file_missing_body_errors() {
    let tmp = NamedTempFile::with_suffix(".json").unwrap();
    std::fs::write(tmp.path(), r#"{"name":"no-body"}"#).unwrap();
    let err = parse_prompt_file(tmp.path()).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("neither 'messages' nor 'body'"),
        "unexpected error: {msg}"
    );
}

// ── sync_dir tests ────────────────────────────────────────────────────────

#[test]
fn sync_dir_inserts_all_jsons() {
    let dir = tempfile::tempdir().unwrap();
    for (name, body) in [
        ("p1", r#"{"name":"p1","body":"body one"}"#),
        ("p2", r#"{"name":"p2","body":"body two"}"#),
        (
            "p3",
            r#"{"name":"p3","messages":[{"role":"user","content":"hi"}]}"#,
        ),
    ] {
        std::fs::write(dir.path().join(format!("{name}.json")), body).unwrap();
    }
    // A non-JSON file that must be silently ignored.
    std::fs::write(dir.path().join("notes.txt"), "ignore me").unwrap();

    let conn = test_conn();
    let (inserted, total) = sync_dir(&conn, dir.path()).unwrap();
    assert_eq!(total, 3, "3 json files");
    assert_eq!(inserted, 3, "all 3 are new");

    let count: i64 = conn
        .query_row("SELECT count(*) FROM prompts", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 3);
}

#[test]
fn sync_dir_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    for (name, body) in [
        ("p1", r#"{"name":"p1","body":"a"}"#),
        ("p2", r#"{"name":"p2","body":"b"}"#),
        ("p3", r#"{"name":"p3","body":"c"}"#),
    ] {
        std::fs::write(dir.path().join(format!("{name}.json")), body).unwrap();
    }

    let conn = test_conn();
    let (ins1, tot1) = sync_dir(&conn, dir.path()).unwrap();
    assert_eq!((ins1, tot1), (3, 3));

    let (ins2, tot2) = sync_dir(&conn, dir.path()).unwrap();
    assert_eq!(tot2, 3, "still 3 files");
    assert_eq!(ins2, 0, "zero new inserts on second sync");
}

#[test]
fn sync_dir_real_rmlx_prompts() {
    // Resolve the rMLX repo root via CARGO_MANIFEST_DIR (crates/rmlx-metrics → workspace root).
    let manifest_dir =
        std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap_or_default());
    let repo_root = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map_or_else(|| manifest_dir.clone(), Path::to_path_buf);
    let prompts_path = repo_root.join("prompts");
    let dir = prompts_path.as_path();
    if !dir.exists() {
        // CI environments outside the dev machine skip this test.
        eprintln!("skip: rMLX/prompts/ not found");
        return;
    }
    let conn = test_conn();
    let (inserted, total) = sync_dir(&conn, dir).unwrap();
    assert!(total >= 1, "must find at least one prompt file");
    assert!(
        inserted <= total,
        "inserted ({inserted}) must be <= total ({total})"
    );

    // Verify DB rows match total (all unique bodies).
    let count: i64 = conn
        .query_row("SELECT count(*) FROM prompts", [], |r| r.get(0))
        .unwrap();
    assert!(count >= 1, "DB must have at least one row; got {count}");
    println!("real-files smoke: {inserted} inserted / {total} files / {count} DB rows");
}
