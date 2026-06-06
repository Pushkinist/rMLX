use super::*;
use serde_json::json;

fn valid_record() -> RunRecord {
    RunRecord {
        backend: "rmlx".to_string(),
        backend_version: None,
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
