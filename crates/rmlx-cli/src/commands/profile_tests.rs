use super::*;

#[test]
fn toml_round_trip_preserves_profile() {
    let src = r#"
[profile.myrun]
model = "/models/gemma"
port = 9001
host = "0.0.0.0"
kv_quant = "k8v4"
max_ctx = 8192
default_temperature = 0.0

[profile.fast]
port = 9100
"#;
    let parsed = ProfilesFile::from_toml(src).expect("parse");
    assert_eq!(parsed.profile.len(), 2);

    let myrun = parsed.get("myrun").expect("myrun present");
    assert_eq!(myrun.model, Some(PathBuf::from("/models/gemma")));
    assert_eq!(myrun.port, Some(9001));
    assert_eq!(myrun.host.as_deref(), Some("0.0.0.0"));
    assert_eq!(myrun.kv_quant.as_deref(), Some("k8v4"));
    assert_eq!(myrun.max_ctx, Some(8192));
    assert_eq!(myrun.default_temperature, Some(0.0));
    // Unset fields fall through to None (CLI default wins).
    assert_eq!(myrun.idle_timeout_secs, None);

    let fast = parsed.get("fast").expect("fast present");
    assert_eq!(fast.port, Some(9100));
    assert_eq!(fast.model, None);

    // Round-trip: serialise → re-parse must be identical.
    let serialised = parsed.to_toml().expect("serialise");
    let reparsed = ProfilesFile::from_toml(&serialised).expect("reparse");
    assert_eq!(parsed, reparsed);
}

#[test]
fn missing_profile_lists_known_names() {
    let parsed = ProfilesFile::from_toml("[profile.a]\nport = 1\n").expect("parse");
    let err = parsed.get("nope").unwrap_err().to_string();
    assert!(err.contains("'nope' not found"), "got: {err}");
    assert!(err.contains("known profiles: [a]"), "got: {err}");
}

#[test]
fn empty_document_parses() {
    let parsed = ProfilesFile::from_toml("").expect("parse empty");
    assert!(parsed.profile.is_empty());
}
