use super::*;

#[test]
fn build_record_includes_ppl_op_name() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("fake-model");
    std::fs::create_dir_all(&model_dir).expect("mkdir fake-model");
    let report = ppl::PplReport {
        ppl: 12.34,
        mean_nll: 2.5,
        scored_tokens: 1000,
        windows: 3,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-deadbeef",
        &model_dir,
        "wikitext-2",
        4096,
        2048,
        5000,
        &report,
        123.4,
        7777.0,
        "2bit",
        None,
    )
    .expect("record builds");
    let metrics = rec["metrics"].as_array().expect("metrics array");
    let names: Vec<&str> = metrics.iter().filter_map(|m| m["name"].as_str()).collect();
    assert!(
        names.contains(&"ppl_wikitext2"),
        "expected ppl_wikitext2 metric, got {names:?}"
    );
    // Audit fields present.
    assert!(names.contains(&"ppl_mean_nll"));
    assert!(names.contains(&"ppl_scored_tokens"));
    assert!(names.contains(&"ppl_windows"));
    assert_eq!(rec["backend"], "rmlx");
    // Op-name derivation strips hyphens (`wikitext-2` -> `wikitext2`).
    assert_eq!(rec["prompt"]["name"], "wikitext-2_ctx4096_stride2048");
}

/// Identity comes from the one Rust source, not from re-deriving a SHA by
/// string-splitting the run_id — that reimplementation is what this record
/// used to do, and it is why `backend_version` and `build_profile` were missing
/// entirely (the run_id carries neither). `git_sha` is caller-supplied
/// provenance (the `--git-sha` flag), not part of `RunIdentity` at all.
#[test]
fn build_record_stamps_identity_from_the_single_source() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("m");
    std::fs::create_dir_all(&model_dir).expect("mkdir m");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-0.2.8",
        &model_dir,
        "wikitext-2",
        16,
        8,
        1,
        &report,
        0.0,
        0.0,
        "bf16",
        Some("cafebabe"),
    )
    .expect("record builds");

    let ident = RunIdentity::get();
    assert_eq!(rec["backend"], "rmlx");
    assert_eq!(rec["backend_version"], ident.backend_version());
    assert_eq!(rec["build_profile"], ident.build_profile());
    assert_eq!(rec["git_sha"], "cafebabe");
    assert_eq!(rec["hardware_tag"], ident.hardware_tag());

    assert_eq!(rec["weight_quant"], "bf16");
    assert_eq!(rec["kv_quant"], "none");
}

/// `--git-sha` absent ⇒ `git_sha` is `null`, never guessed.
#[test]
fn build_record_git_sha_absent_is_null() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("m2");
    std::fs::create_dir_all(&model_dir).expect("mkdir m2");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-0.2.8",
        &model_dir,
        "wikitext-2",
        16,
        8,
        1,
        &report,
        0.0,
        0.0,
        "bf16",
        None,
    )
    .expect("record builds");

    assert!(rec["git_sha"].is_null());
}

/// `--git-sha ""` is not provenance either — normalized to the same `null`
/// an absent flag gets, not stamped as a literal empty string.
#[test]
fn build_record_git_sha_blank_string_is_null() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("m3");
    std::fs::create_dir_all(&model_dir).expect("mkdir m3");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-0.2.8",
        &model_dir,
        "wikitext-2",
        16,
        8,
        1,
        &report,
        0.0,
        0.0,
        "bf16",
        Some(""),
    )
    .expect("record builds");

    assert!(rec["git_sha"].is_null());
}
