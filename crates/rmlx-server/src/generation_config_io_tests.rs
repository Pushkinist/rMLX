use super::*;
use std::io::Write;
use tempfile::tempdir;

#[test]
fn absent_file_returns_ok_none() {
    let tmp = tempdir().unwrap();
    // No generation_config.json created.
    let result = load_generation_config(tmp.path());
    assert!(result.is_ok(), "absent file must not be an error");
    assert!(result.unwrap().is_none(), "absent file must yield None");
}

#[test]
fn unparseable_json_returns_err() {
    let tmp = tempdir().unwrap();
    let mut f = std::fs::File::create(tmp.path().join("generation_config.json")).unwrap();
    f.write_all(b"{ not valid json }").unwrap();
    let result = load_generation_config(tmp.path());
    assert!(result.is_err(), "malformed JSON must return Err");
}

#[test]
fn partial_keys_parsed_correctly() {
    let tmp = tempdir().unwrap();
    let mut f = std::fs::File::create(tmp.path().join("generation_config.json")).unwrap();
    // Only temperature present; other keys absent.
    f.write_all(br#"{"temperature": 0.7, "bos_token_id": 1}"#)
        .unwrap();
    let result = load_generation_config(tmp.path()).expect("parse");
    let cfg = result.expect("should be Some");
    assert_eq!(cfg.temperature, Some(0.7_f32));
    assert!(cfg.top_p.is_none());
    assert!(cfg.top_k.is_none());
    assert!(cfg.repetition_penalty.is_none());
    assert!(cfg.max_new_tokens.is_none());
}

#[test]
fn all_keys_parsed_correctly() {
    let tmp = tempdir().unwrap();
    let mut f = std::fs::File::create(tmp.path().join("generation_config.json")).unwrap();
    f.write_all(
        br#"{
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.1,
            "max_new_tokens": 512
        }"#,
    )
    .unwrap();
    let result = load_generation_config(tmp.path()).expect("parse");
    let cfg = result.expect("should be Some");
    assert_eq!(cfg.temperature, Some(1.0_f32));
    assert_eq!(cfg.top_p, Some(0.95_f32));
    assert_eq!(cfg.top_k, Some(20u32));
    assert_eq!(cfg.repetition_penalty, Some(1.1_f32));
    assert_eq!(cfg.max_new_tokens, Some(512u32));
}
