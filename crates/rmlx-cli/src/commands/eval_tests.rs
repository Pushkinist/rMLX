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
        None,
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
        None,
        None,
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
        None,
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
        None,
        None,
    )
    .expect("record builds");

    assert!(rec["git_sha"].is_null());
}

/// A cacheless run carries `kv_quant = none` and no `decode_config`: it ranks
/// in the same cell as every PPL row recorded before the flag existed. A run
/// through a codec carries that codec, and a run at a non-default boundary
/// carries the boundary term as well.
#[test]
fn build_record_reports_the_codec_and_boundary_it_scored_at() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("m4");
    std::fs::create_dir_all(&model_dir).expect("mkdir m4");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let build = |kv: Option<rmlx_kv_quant::KvQuant>, dc: Option<&str>| {
        build_ppl_run_record(
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
            kv,
            dc,
        )
        .expect("record builds")
    };

    let cacheless = build(None, None);
    assert_eq!(cacheless["kv_quant"], "none");
    assert!(cacheless["decode_config"].is_null());

    let scored = build(
        Some(rmlx_kv_quant::KvQuant::Iso3Sym),
        Some("kv_boundary/head=2,kv_boundary/tail=4"),
    );
    assert_eq!(scored["kv_quant"], "iso3_sym");
    assert_eq!(
        scored["decode_config"],
        "kv_boundary/head=2,kv_boundary/tail=4"
    );
}

/// The two scorers are two engine configurations, so they do not share a cell.
///
/// Without the `ppl/scorer` term a cacheless run and a bf16-cache run both land
/// as `kv_quant = 'none'` with `decode_config` NULL, and `bests` ranks a number
/// produced by a full-window forward against one produced by a decode loop.
#[test]
fn the_two_scorers_do_not_share_a_cell() {
    assert_eq!(
        ppl_decode_config(None),
        None,
        "the default scorer is the default"
    );
    assert_eq!(
        ppl_decode_config(Some(rmlx_kv_quant::KvQuant::None)).as_deref(),
        Some("ppl/scorer=cached"),
    );
    let config = ppl_decode_config(Some(rmlx_kv_quant::KvQuant::Iso3Sym))
        .expect("a cached run carries a term");
    assert!(
        rmlx_metrics::cell::decode_config_is_well_formed(&config),
        "{config} is not a well-formed decode_config"
    );
}
