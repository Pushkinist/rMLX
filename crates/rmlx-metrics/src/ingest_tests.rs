use super::*;
use serde_json::json;

fn valid_record() -> RunRecord {
    RunRecord {
        schema_version: RECORD_SCHEMA_VERSION,
        backend: "rmlx".to_string(),
        backend_version: Some("0.2.8".to_string()),
        model_namespace: "mlx-community".to_string(),
        model: "gemma-4-e4b-it-mxfp8".to_string(),
        weight_quant: "mxfp8".to_string(),
        kv_quant: "k8v8".to_string(),
        ctx_max: 8192,
        prompt: PromptRef::ByBody {
            name: "t".to_string(),
            body: json!("hi"),
            notes: None,
            tokens_approx: None,
        },
        ts_utc: "2026-05-10T07:30:00Z".to_string(),
        git_sha: None,
        build_profile: None,
        hardware_tag: "m5_max_128gb".to_string(),
        prompt_tokens: None,
        max_tokens: None,
        temperature: None,
        seed: None,
        n_warmups: None,
        n_measure: None,
        output_first_64: None,
        notes: None,
        description: None,
        metrics: vec![MetricEntry {
            name: "decode_tps_warm".to_string(),
            value: Some(100.0),
            stddev: None,
        }],
    }
}

#[test]
fn valid_record_passes() {
    valid_record().validate().unwrap();
}

#[test]
fn unknown_backend_rejected() {
    let mut r = valid_record();
    r.backend = "pytorch".to_string();
    let err = r.validate().unwrap_err();
    assert!(matches!(err, Error::IdentityNotInWhitelist { .. }));
}

#[test]
fn bad_timestamp_rejected() {
    let mut r = valid_record();
    r.ts_utc = "not-a-time".to_string();
    let err = r.validate().unwrap_err();
    assert!(matches!(err, Error::InvalidTimestamp(_)));
}

#[test]
fn unknown_metric_rejected() {
    let mut r = valid_record();
    r.metrics[0].name = "foo".to_string();
    let err = r.validate().unwrap_err();
    assert!(matches!(err, Error::UnknownMetric(_)));
}

#[test]
fn all_null_values_rejected() {
    let mut r = valid_record();
    r.metrics[0].value = None;
    let err = r.validate().unwrap_err();
    assert!(matches!(err, Error::NoMeasurements));
}

#[test]
fn empty_metrics_rejected() {
    let mut r = valid_record();
    r.metrics = vec![];
    let err = r.validate().unwrap_err();
    assert!(matches!(err, Error::NoMeasurements));
}

#[test]
fn prompt_by_body_str() {
    let json_str = r#"{
        "backend": "rmlx",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "t", "body": "hello" },
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": "decode_tps_warm", "value": 100.0 }]
    }"#;
    let record: RunRecord = serde_json::from_str(json_str).unwrap();
    assert!(matches!(&record.prompt, PromptRef::ByBody { body, .. } if body == &json!("hello")));
    // round-trip
    let s = serde_json::to_string(&record).unwrap();
    let _: RunRecord = serde_json::from_str(&s).unwrap();
}

#[test]
fn prompt_by_body_messages() {
    let json_str = r#"{
        "backend": "rmlx",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": {
            "name": "t",
            "body": [{"role": "user", "content": "hi"}]
        },
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": "decode_tps_warm", "value": 100.0 }]
    }"#;
    let record: RunRecord = serde_json::from_str(json_str).unwrap();
    assert!(matches!(&record.prompt, PromptRef::ByBody { body, .. } if body.is_array()));
    // round-trip
    let s = serde_json::to_string(&record).unwrap();
    let _: RunRecord = serde_json::from_str(&s).unwrap();
}

#[test]
fn prompt_by_sha256_only() {
    let sha = "a".repeat(64);
    let json_str = format!(
        r#"{{
            "backend": "rmlx",
            "model_namespace": "mlx-community",
            "model": "gemma-4-e4b-it-mxfp8",
            "weight_quant": "mxfp8",
            "kv_quant": "k8v8",
            "ctx_max": 8192,
            "prompt": {{ "sha256": "{sha}" }},
            "ts_utc": "2026-05-10T07:30:00Z",
            "hardware_tag": "m5_max_128gb",
            "metrics": [{{ "name": "decode_tps_warm", "value": 100.0 }}]
        }}"#
    );
    let record: RunRecord = serde_json::from_str(&json_str).unwrap();
    assert!(matches!(&record.prompt, PromptRef::BySha256 { .. }));
    // round-trip
    let s = serde_json::to_string(&record).unwrap();
    let _: RunRecord = serde_json::from_str(&s).unwrap();
}

#[test]
fn prompt_sha256_too_short_rejected() {
    let mut r = valid_record();
    r.prompt = PromptRef::BySha256 {
        sha256: "abc123".to_string(),
    };
    let err = r.validate().unwrap_err();
    assert!(matches!(err, Error::InvalidPrompt(_)));
}

#[test]
fn sparse_metric_kept_through_serde() {
    let json_str = r#"{ "name": "ttft_warm_ms", "value": null }"#;
    let entry: MetricEntry = serde_json::from_str(json_str).unwrap();
    assert!(entry.value.is_none());
}

#[test]
fn sha256_dedup_stable() {
    let h1 = prompt_body_sha256(&json!("foo"));
    let h2 = prompt_body_sha256(&json!("foo"));
    assert_eq!(h1, h2);
}

#[test]
fn sha256_changes_with_body() {
    let h1 = prompt_body_sha256(&json!("foo"));
    let h2 = prompt_body_sha256(&json!("bar"));
    assert_ne!(h1, h2);
}

#[test]
fn sha256_messages_array_stable() {
    let body = json!([{"role": "user", "content": "x"}]);
    let h1 = prompt_body_sha256(&body);
    let h2 = prompt_body_sha256(&body);
    assert_eq!(h1, h2);
    assert_eq!(h1.len(), 64);
}

// ── Run-identity contract (§8.5) ──────────────────────────────────────────────

#[test]
fn rmlx_record_without_backend_version_is_rejected() {
    let mut r = valid_record();
    r.backend_version = None;
    let err = r.validate().unwrap_err();
    assert!(
        matches!(err, Error::MissingBackendVersion { .. }),
        "expected MissingBackendVersion, got {err:?}"
    );
}

#[test]
fn rmlx_record_with_empty_backend_version_is_rejected() {
    let mut r = valid_record();
    r.backend_version = Some(String::new());
    assert!(matches!(
        r.validate().unwrap_err(),
        Error::MissingBackendVersion { .. }
    ));
}

/// The exact junk found in the live DB: git SHAs and refs stuffed into a
/// semver column, plus a bare integer.
#[test]
fn rmlx_record_with_non_semver_backend_version_is_rejected() {
    for junk in [
        "head", "379dcea", "a156173", "1257883", "0.2", "v0.2.8", "dirty",
    ] {
        let mut r = valid_record();
        r.backend_version = Some(junk.to_string());
        assert!(
            matches!(
                r.validate().unwrap_err(),
                Error::MissingBackendVersion { .. }
            ),
            "{junk:?} should have been rejected"
        );
    }
}

#[test]
fn rmlx_record_with_semver_backend_version_passes() {
    for good in ["0.0.1", "0.2.8", "1.0.0", "0.3.0-rc.1", "1.2.3+build7"] {
        let mut r = valid_record();
        r.backend_version = Some(good.to_string());
        assert!(r.validate().is_ok(), "{good:?} should have been accepted");
    }
}

/// Cross-backend: llama.cpp has no semver — it emits a `build_commit`. It must
/// keep ingesting cleanly, or we break every non-rMLX bench.
#[test]
fn non_rmlx_backend_may_omit_backend_version() {
    let mut r = valid_record();
    r.backend = "llama_cpp".to_string();
    r.backend_version = None;
    r.validate().unwrap();
}

#[test]
fn non_rmlx_backend_may_carry_non_semver_version() {
    let mut r = valid_record();
    r.backend = "llama_cpp".to_string();
    r.backend_version = Some("b4567-cafebabe".to_string());
    r.validate().unwrap();
}

/// The one door around the check: the one-shot pre-contract archive import.
#[test]
fn legacy_archive_policy_allows_missing_backend_version() {
    let mut r = valid_record();
    r.backend_version = None;
    assert!(r.validate_with(IdentityPolicy::LegacyArchive).is_ok());
    // ...but the default policy still rejects it.
    assert!(r.validate_with(IdentityPolicy::Enforce).is_err());
}

// ── Wire schema version ───────────────────────────────────────────────────────

#[test]
fn record_from_the_future_is_rejected_loudly() {
    let mut r = valid_record();
    r.schema_version = RECORD_SCHEMA_VERSION + 1;
    let err = r.validate().unwrap_err();
    assert!(
        matches!(err, Error::InvalidIngestField { ref field, .. } if field == "schema_version"),
        "expected schema_version rejection, got {err:?}"
    );
}

#[test]
fn absent_schema_version_defaults_to_v1() {
    // Buffer files written before the field existed must still replay.
    let raw = json!({
        "backend": "rmlx",
        "backend_version": "0.2.8",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": { "name": "t", "body": "hi" },
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{ "name": "decode_tps_warm", "value": 100.0 }],
    });
    let r: RunRecord = serde_json::from_value(raw).unwrap();
    assert_eq!(r.schema_version, 1);
    r.validate().unwrap();
}

// ── Builder ───────────────────────────────────────────────────────────────────

#[test]
fn builder_fills_identity_and_canonicalization() {
    let rec = RunRecordBuilder::rmlx(
        "mlx-community__gemma-4-e4b-it-mxfp8",
        "K8V8",
        8192,
        PromptRef::ByBody {
            name: "t".to_string(),
            body: json!("hi"),
            notes: None,
            tokens_approx: None,
        },
    )
    .unwrap()
    .metric("decode_tps_warm", Some(119.14), Some(0.6))
    .build()
    .unwrap();

    // Identity: inherited, never supplied by the caller.
    assert_eq!(rec.backend, "rmlx");
    assert_eq!(
        rec.backend_version.as_deref(),
        Some(rmlx_core::runinfo::backend_version())
    );
    assert_eq!(
        rec.build_profile.as_deref(),
        Some(rmlx_core::runinfo::build_profile())
    );
    assert_eq!(rec.schema_version, RECORD_SCHEMA_VERSION);

    // Canonicalization: derived from the model id, not hand-passed.
    assert_eq!(rec.model_namespace, "mlx-community");
    assert_eq!(rec.model, "gemma-4-e4b-it-mxfp8");
    assert_eq!(rec.weight_quant, "mxfp8");
    assert_eq!(rec.kv_quant, "k8v8");
    assert!(!rec.ts_utc.is_empty());
}

#[test]
fn builder_infers_weight_quant_and_namespace_from_a_path() {
    let rec = RunRecordBuilder::rmlx(
        "/models/prism-ml__Ternary-Bonsai-8B-mlx-2bit",
        "none",
        4096,
        PromptRef::ByBody {
            name: "t".to_string(),
            body: json!("hi"),
            notes: None,
            tokens_approx: None,
        },
    )
    .unwrap()
    .metric("decode_tps_warm", Some(110.0), None)
    .build()
    .unwrap();

    assert_eq!(rec.model_namespace, "prism-ml");
    assert_eq!(rec.model, "Ternary-Bonsai-8B-mlx-2bit");
    assert_eq!(rec.weight_quant, "2bit");
}

#[test]
fn builder_rejects_a_record_ingest_would_reject() {
    // No metrics → NoMeasurements. build() validates, so a record that cannot
    // be recorded cannot be built either.
    let err = RunRecordBuilder::rmlx(
        "mlx-community__gemma-4-e4b-it-mxfp8",
        "none",
        8192,
        PromptRef::ByBody {
            name: "t".to_string(),
            body: json!("hi"),
            notes: None,
            tokens_approx: None,
        },
    )
    .unwrap()
    .build()
    .unwrap_err();
    assert!(matches!(err, Error::NoMeasurements), "got {err:?}");
}
