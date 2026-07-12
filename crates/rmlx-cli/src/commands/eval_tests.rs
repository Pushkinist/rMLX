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
/// entirely (the run_id carries neither).
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
        "20260526-120000-cafebabe-dirty",
        &model_dir,
        "wikitext-2",
        16,
        8,
        1,
        &report,
        0.0,
        0.0,
        "bf16",
    )
    .expect("record builds");

    let ident = RunIdentity::rmlx();
    assert_eq!(rec["backend"], "rmlx");
    assert_eq!(rec["backend_version"], ident.backend_version);
    assert_eq!(rec["build_profile"], ident.build_profile);
    assert_eq!(
        rec["git_sha"],
        serde_json::to_value(&ident.git_sha).expect("git_sha to json")
    );
    assert_eq!(rec["hardware_tag"], ident.hardware_tag);

    assert_eq!(rec["weight_quant"], "bf16");
    assert_eq!(rec["kv_quant"], "none");
}
