use super::*;

#[test]
fn build_record_includes_ppl_op_name() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("fake-model");
    std::fs::create_dir_all(&model_dir).expect("mkdir fake-model");
    std::fs::write(model_dir.join("config.json"), "{}").expect("write config.json");
    let report = ppl::PplReport {
        ppl: 12.34,
        mean_nll: 2.5,
        scored_tokens: 1000,
        windows: 3,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-deadbeef",
        &resolve_record_identity(&model_dir).expect("identity"),
        "wikitext-2",
        4096,
        2048,
        5000,
        &report,
        123.4,
        7777.0,
        None,
        Some(RunClass::Gate),
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
    std::fs::write(model_dir.join("config.json"), "{}").expect("write config.json");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-0.2.8",
        &resolve_record_identity(&model_dir).expect("identity"),
        "wikitext-2",
        16,
        8,
        1,
        &report,
        0.0,
        0.0,
        Some("cafebabe"),
        Some(RunClass::Gate),
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
    std::fs::write(model_dir.join("config.json"), "{}").expect("write config.json");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-0.2.8",
        &resolve_record_identity(&model_dir).expect("identity"),
        "wikitext-2",
        16,
        8,
        1,
        &report,
        0.0,
        0.0,
        None,
        Some(RunClass::Gate),
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
    std::fs::write(model_dir.join("config.json"), "{}").expect("write config.json");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-0.2.8",
        &resolve_record_identity(&model_dir).expect("identity"),
        "wikitext-2",
        16,
        8,
        1,
        &report,
        0.0,
        0.0,
        Some(""),
        Some(RunClass::Gate),
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
    std::fs::write(model_dir.join("config.json"), "{}").expect("write config.json");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let build = |kv: Option<rmlx_kv_quant::KvQuant>, dc: Option<&str>| {
        build_ppl_run_record(
            "20260526-120000-0.2.8",
            &resolve_record_identity(&model_dir).expect("identity"),
            "wikitext-2",
            16,
            8,
            1,
            &report,
            0.0,
            0.0,
            None,
            Some(RunClass::Gate),
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

/// The two scorers measure different quantities, so they carry different
/// metric names — not one name plus a `decode_config` term.
///
/// Sharing a name would rank a full-window-forward number against a decode-loop
/// number in `bests` and in `metrics export --markdown`, and a `decode_config`
/// discriminator would additionally split those cells away from every `mlx_lm`
/// row, which can never carry a term this engine invented.
#[test]
fn the_two_scorers_do_not_share_a_metric() {
    assert_eq!(ppl_metric_name("wikitext-2", None), "ppl_wikitext2");
    assert_eq!(
        ppl_metric_name("wikitext-2", Some(rmlx_kv_quant::KvQuant::Iso3Sym)),
        "ppl_wikitext2_cached"
    );
    assert_ne!(
        ppl_metric_name("wikitext-2", None),
        ppl_metric_name("wikitext-2", Some(rmlx_kv_quant::KvQuant::None)),
        "a bf16 cache is still a cache: it does not share the cacheless metric"
    );
}

/// `decode_config` is the boundary's alone. A cached run at the default
/// boundary carries none, so it ranks against every row of its own metric
/// rather than being fenced off in a cell of one.
#[test]
fn the_scorer_mode_is_not_a_decode_config_term() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("m5");
    std::fs::create_dir_all(&model_dir).expect("mkdir m5");
    std::fs::write(model_dir.join("config.json"), "{}").expect("write config.json");
    let report = ppl::PplReport {
        ppl: 1.0,
        mean_nll: 0.0,
        scored_tokens: 1,
        windows: 1,
    };
    let rec = build_ppl_run_record(
        "20260526-120000-0.2.8",
        &resolve_record_identity(&model_dir).expect("identity"),
        "wikitext-2",
        16,
        8,
        1,
        &report,
        0.0,
        0.0,
        None,
        Some(RunClass::Gate),
        Some(rmlx_kv_quant::KvQuant::Iso3Sym),
        None,
    )
    .expect("record builds");
    assert!(rec["decode_config"].is_null());
    let names: Vec<&str> = rec["metrics"]
        .as_array()
        .expect("metrics array")
        .iter()
        .filter_map(|m| m["name"].as_str())
        .collect();
    assert!(
        names.contains(&"ppl_wikitext2_cached"),
        "the cached scorer names its own metric: {names:?}"
    );
}

/// An identity that cannot be recorded stops the run before the corpus is read
/// or a model is loaded: the corpus path and the model directory here do not
/// exist, so reaching either would fail with a different message.
#[test]
fn ppl_refuses_an_unrecordable_identity_before_any_work() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("__m");
    let err = run_ppl(
        &model_dir,
        &tmp.path().join("no-such-corpus.txt"),
        4096,
        2048,
        "wikitext-2",
        &ClaimedDevice::cpu(),
        0,
        "20260526-120000-deadbeef",
        None,
        Some(RunClass::Gate),
        None,
    )
    .expect_err("identity is refused");
    let msg = err.to_string();
    assert!(msg.contains("model_namespace"), "{msg}");
    assert!(!msg.contains("cannot read text file"), "{msg}");
}

fn snapshot_with_quant_bits(root: &Path, name: &str, bits: u8) -> PathBuf {
    let dir = root.join(name);
    std::fs::create_dir_all(&dir).expect("mkdir snapshot");
    std::fs::write(
        dir.join("config.json"),
        format!(r#"{{"quantization": {{"group_size": 64, "bits": {bits}}}}}"#),
    )
    .expect("write config.json");
    dir
}

/// A bit width the metrics DB has no label for would be refused at ingest,
/// after the run: it must stop the run before the corpus is read.
#[test]
fn ppl_refuses_an_unrecordable_weight_quant_before_any_work() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = snapshot_with_quant_bits(tmp.path(), "host__m", 7);
    let err = run_ppl(
        &model_dir,
        &tmp.path().join("no-such-corpus.txt"),
        4096,
        2048,
        "wikitext-2",
        &ClaimedDevice::cpu(),
        0,
        "20260526-120000-deadbeef",
        None,
        Some(RunClass::Gate),
        None,
    )
    .expect_err("weight_quant is refused");
    let msg = err.to_string();
    assert!(msg.contains("weight_quant"), "{msg}");
    assert!(!msg.contains("cannot read text file"), "{msg}");
}

#[test]
fn record_identity_carries_the_snapshot_weight_quant() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = snapshot_with_quant_bits(tmp.path(), "host__m", 4);
    let id = resolve_record_identity(&model_dir).expect("identity");
    assert_eq!(
        (
            id.namespace.as_str(),
            id.model.as_str(),
            id.weight_quant.as_str()
        ),
        ("host", "m", "4bit")
    );
}

#[test]
fn record_identity_of_a_snapshot_with_no_config_is_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("host__m");
    std::fs::create_dir_all(&model_dir).expect("mkdir snapshot");
    let err = resolve_record_identity(&model_dir)
        .err()
        .expect("a snapshot with no config.json has no weight_quant to record");
    let msg = err.to_string();
    assert!(msg.contains("no config.json"), "{msg}");
}

#[test]
fn record_identity_of_a_snapshot_whose_config_does_not_parse_is_refused() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("host__m");
    std::fs::create_dir_all(&model_dir).expect("mkdir snapshot");
    std::fs::write(model_dir.join("config.json"), "{ not json").expect("write config.json");
    let err = resolve_record_identity(&model_dir)
        .err()
        .expect("an unreadable config is not an unquantized snapshot");
    let msg = format!("{err:#}");
    assert!(msg.contains("malformed config.json"), "{msg}");
}

#[test]
fn record_identity_of_a_config_without_quantization_is_bf16() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("host__m");
    std::fs::create_dir_all(&model_dir).expect("mkdir snapshot");
    std::fs::write(model_dir.join("config.json"), "{}").expect("write config.json");
    let id = resolve_record_identity(&model_dir).expect("identity");
    assert_eq!(id.weight_quant, "bf16");
}
