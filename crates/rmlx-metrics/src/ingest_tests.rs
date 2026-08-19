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

// ── kv_quant is a free-form recorded label, full record ingest ───────────────

/// A couple of real rotation-family codec names — the exact class the old
/// hand-maintained allow-list dropped — ingest fine at the full-record
/// level (not just the bare `canonicalize_kv_quant` call).
#[test]
fn record_ingests_for_real_rotation_codec_names() {
    for token in ["rotor4_sym", "k_rotor3"] {
        let mut r = valid_record();
        r.kv_quant = token.to_string();
        r.validate()
            .unwrap_or_else(|e| panic!("kv_quant {token:?} rejected: {e}"));
    }
}

/// The core of the fix: a full record with a `kv_quant` token this binary
/// has never heard of — a codec that does not exist yet, a typo, anything —
/// still ingests. No allow-list, no drift; `is_valid_kv_quant_token`-style
/// grammar mirrors are gone for good.
#[test]
fn record_ingests_with_unknown_kv_quant_token() {
    let mut r = valid_record();
    r.kv_quant = "some_future_codec_v9".to_string();
    r.validate().unwrap();
}

/// `bf16`/`f16` still alias to `none` at the full-record level.
#[test]
fn record_kv_quant_bf16_alias_still_normalizes() {
    for alias in ["bf16", "f16"] {
        let mut r = valid_record();
        r.kv_quant = alias.to_string();
        r.validate()
            .unwrap_or_else(|e| panic!("kv_quant {alias:?} rejected: {e}"));
    }
}

// ── model_namespace is a free-form recorded label too ────────────────────────

/// An unrecognized `model_namespace` — a new model host, a typo, a local
/// finetune — must still record. Same footgun class as kv_quant: a fixed
/// whitelist on a free-form recorded label silently drops valid rows.
#[test]
fn record_ingests_with_unknown_model_namespace() {
    let mut r = valid_record();
    r.model_namespace = "some-new-model-host".to_string();
    r.validate().unwrap();
}

// ── §8.5 CBB record shape (`model` / `model_id` tolerance) ───────────────────

/// A record shaped exactly like the CBB cross-backend §8.5 emitter (see
/// `../Cross-Backend-Bench/runners/_common.py::_build_metrics_record`)
/// deserializes and ingests cleanly.
#[test]
fn cbb_shaped_record_with_model_field_ingests() {
    let json_str = r#"{
        "backend": "rmlx",
        "backend_version": "0.2.8",
        "model_namespace": "prism-ml",
        "model": "Ternary-Bonsai-8B-mlx-2bit",
        "weight_quant": "2bit",
        "kv_quant": "none",
        "ctx_max": 8192,
        "prompt": {"name": "longctx_4k", "tokens_approx": 4096, "body": "hi"},
        "ts_utc": "2026-05-10T07:30:00Z",
        "git_sha": null,
        "build_profile": null,
        "hardware_tag": "m5_max_128gb",
        "notes": "cbb-runner",
        "metrics": [{"name": "decode_tps_warm", "value": 100.0}]
    }"#;
    let r: RunRecord = serde_json::from_str(json_str).unwrap();
    r.validate().unwrap();
}

/// A record using `model_id` instead of the canonical `model` key (a shape a
/// future or foreign emitter could plausibly send) still ingests — the
/// ingest tolerates the alias rather than silently landing in
/// `metrics/buffer/failed/`.
#[test]
fn record_with_model_id_alias_ingests() {
    let json_str = r#"{
        "backend": "rmlx",
        "backend_version": "0.2.8",
        "model_namespace": "prism-ml",
        "model_id": "Ternary-Bonsai-8B-mlx-2bit",
        "weight_quant": "2bit",
        "kv_quant": "none",
        "ctx_max": 8192,
        "prompt": {"name": "longctx_4k", "tokens_approx": 4096, "body": "hi"},
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{"name": "decode_tps_warm", "value": 100.0}]
    }"#;
    let r: RunRecord = serde_json::from_str(json_str).unwrap();
    assert_eq!(r.model, "Ternary-Bonsai-8B-mlx-2bit");
    r.validate().unwrap();
}

// ── §4 value plausibility ─────────────────────────────────────────────────

#[test]
fn zero_rate_rejected() {
    // A run that generated no tokens has no rate to report — the recorder
    // skips a `null` entry, so an emitter has no reason to send `0.0`.
    let mut r = valid_record();
    r.metrics[0].value = Some(0.0);
    let err = r.validate().unwrap_err();
    assert!(
        matches!(err, Error::ImplausibleValue { ref metric, value, .. }
                 if metric == "decode_tps_warm" && value == 0.0),
        "expected ImplausibleValue, got {err}"
    );
}

#[test]
fn out_of_range_rate_rejected() {
    // `(prompt_tokens - 242) * 1000` for a 131k-token prompt: the closed-form
    // non-rate that this gate exists to stop at the door.
    let mut r = valid_record();
    r.metrics[0].name = "prefill_tps".to_string();
    r.metrics[0].value = Some(130_810_000.0);
    let err = r.validate().unwrap_err();
    assert!(
        matches!(err, Error::ImplausibleValue { ref metric, .. } if metric == "prefill_tps"),
        "expected ImplausibleValue, got {err}"
    );
}

#[test]
fn non_finite_value_rejected() {
    let mut r = valid_record();
    r.metrics[0].value = Some(f64::NAN);
    assert!(matches!(
        r.validate().unwrap_err(),
        Error::ImplausibleValue { .. }
    ));
    r.metrics[0].value = Some(f64::INFINITY);
    assert!(matches!(
        r.validate().unwrap_err(),
        Error::ImplausibleValue { .. }
    ));
}

#[test]
fn zero_counter_accepted() {
    // The floor is per metric: zero cache hits is a measurement, and
    // rejecting it would drop a valid record.
    let mut r = valid_record();
    r.metrics[0].name = "prompt_cache_hits".to_string();
    r.metrics[0].value = Some(0.0);
    r.validate().unwrap();
}

#[test]
fn zero_duration_accepted() {
    // Millisecond resolution rounds a sub-millisecond span to zero; that is
    // a coarse measurement, not a missing one.
    let mut r = valid_record();
    r.metrics[0].name = "ttft_warm_ms".to_string();
    r.metrics[0].value = Some(0.0);
    r.validate().unwrap();
}

#[test]
fn null_value_is_the_way_to_say_not_measured() {
    let mut r = valid_record();
    r.metrics.push(MetricEntry {
        name: "prefill_tps".to_string(),
        value: None,
        stddev: None,
    });
    r.validate().unwrap();
}

// ── archive-only placeholder drop ─────────────────────────────────────────

#[test]
fn drop_implausible_metrics_removes_only_the_placeholders() {
    let mut r = valid_record();
    r.metrics = vec![
        MetricEntry {
            name: "decode_tps_warm".to_string(),
            value: Some(0.0), // a rate of zero: not a measurement
            stddev: None,
        },
        MetricEntry {
            name: "prefill_tps".to_string(),
            value: Some(130_810_000.0), // out of range
            stddev: None,
        },
        MetricEntry {
            name: "peak_rss_mb".to_string(),
            value: Some(35_392.0), // real
            stddev: None,
        },
        MetricEntry {
            name: "prompt_cache_hits".to_string(),
            value: Some(0.0), // a counter's zero IS a measurement
            stddev: None,
        },
        MetricEntry {
            name: "ttft_warm_ms".to_string(),
            value: None, // already "not measured"
            stddev: None,
        },
    ];

    assert_eq!(r.drop_implausible_metrics(), 2);
    let kept: Vec<&str> = r.metrics.iter().map(|m| m.name.as_str()).collect();
    assert_eq!(
        kept,
        vec!["peak_rss_mb", "prompt_cache_hits", "ttft_warm_ms"],
        "wrong entries survived the archive drop"
    );
    // What survives must then pass the gate.
    r.validate().unwrap();
}

#[test]
fn drop_implausible_metrics_leaves_an_unregistered_name_for_validate() {
    let mut r = valid_record();
    r.metrics[0].name = "not_a_metric".to_string();
    assert_eq!(
        r.drop_implausible_metrics(),
        0,
        "an unknown name is validate's to reject, not this function's to hide"
    );
    assert!(matches!(r.validate().unwrap_err(), Error::UnknownMetric(_)));
}
