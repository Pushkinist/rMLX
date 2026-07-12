use super::*;
use serde_json::json;

fn legacy_json_with_prompt() -> &'static str {
    r#"{
        "run_id":         "20260511-124538-a2f649b",
        "ts_utc":         "2026-05-11T12:45:33Z",
        "backend":        "rmlx",
        "model_namespace":"mlx-community",
        "model_name":     "Qwen3.6-35B-A3B-8bit",
        "weight_quant":   "q8_0",
        "kv_quant":       "k8v4",
        "max_ctx":        8192,
        "prompt_tokens":  4096,
        "max_tokens":     30,
        "git_sha":        "a2f649b",
        "prompt_body":    [{"role":"user","content":"hello"}],
        "observations": [
            {"metric":"decode_tps", "value":85.95,  "unit":"tps", "direction":"higher_is_better", "run_type":"warm","notes":"warm-run"},
            {"metric":"prefill_tps","value":500.0,  "unit":"tps", "direction":"higher_is_better", "run_type":"warm","notes":"warm-run"},
            {"metric":"ttft_ms",    "value":120.0,  "unit":"ms",  "direction":"lower_is_better",  "run_type":"warm","notes":"warm-run"}
        ],
        "first_32_tokens": ["The","user","wants"]
    }"#
}

fn legacy_json_no_prompt() -> &'static str {
    r#"{
        "run_id":         "20260511-130000-b3c4d5e",
        "ts_utc":         "2026-05-11T13:00:00Z",
        "backend":        "rmlx",
        "model_namespace":"mlx-community",
        "model_name":     "Qwen3.6-35B-A3B-8bit",
        "weight_quant":   "q8_0",
        "kv_quant":       "k8v8",
        "max_ctx":        8192,
        "prompt_tokens":  4096,
        "max_tokens":     30,
        "git_sha":        "b3c4d5e",
        "observations": [
            {"metric":"decode_tps","value":90.0,"unit":"tps","direction":"higher_is_better","run_type":"warm","notes":"final-matrix"}
        ]
    }"#
}

#[test]
fn try_parse_legacy_with_prompt_body() {
    let run = try_parse_legacy(legacy_json_with_prompt()).expect("should parse");
    assert_eq!(run.model, "Qwen3.6-35B-A3B-8bit");
    assert_eq!(run.ctx_max, 8192);
    assert_eq!(run.hardware_tag, "m5_max_128gb");
    assert_eq!(run.metrics[0].name, "decode_tps_warm");
    assert_eq!(run.metrics[1].name, "prefill_tps");
    assert_eq!(run.metrics[2].name, "ttft_warm_ms");
    assert_eq!(run.metrics[0].value, Some(85.95));
    // prompt_body → PromptRef::ByBody
    assert!(matches!(&run.prompt, PromptRef::ByBody { name, body, .. }
        if name == "longctx_4k" && body == &json!([{"role":"user","content":"hello"}])));
    // first_32_tokens → output_first_64
    assert_eq!(run.output_first_64.as_deref(), Some("The user wants"));
}

#[test]
fn try_parse_legacy_no_prompt_body() {
    let run = try_parse_legacy(legacy_json_no_prompt()).expect("should parse");
    assert_eq!(run.model, "Qwen3.6-35B-A3B-8bit");
    assert_eq!(run.metrics[0].name, "decode_tps_warm");
    // Placeholder prompt body
    assert!(matches!(&run.prompt, PromptRef::ByBody { name, body, .. }
        if name == "longctx_4k" && body.is_string()));
    assert!(run.output_first_64.is_none());
}

#[test]
fn try_parse_legacy_returns_none_for_canonical() {
    // A canonical §8.5 record (has `model`, not `model_name`) must return None.
    let canonical = r#"{
        "backend": "rmlx",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": {"name":"t","body":"hi"},
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{"name":"decode_tps_warm","value":100.0}]
    }"#;
    assert!(try_parse_legacy(canonical).is_none());
}

#[test]
fn map_metric_name_decode_tps() {
    assert_eq!(map_metric_name("decode_tps"), "decode_tps_warm");
}

#[test]
fn map_metric_name_ttft_ms() {
    assert_eq!(map_metric_name("ttft_ms"), "ttft_warm_ms");
}

#[test]
fn map_metric_name_prefill_tps() {
    assert_eq!(map_metric_name("prefill_tps"), "prefill_tps");
}

#[test]
fn map_kv_quant_bf16_to_none() {
    assert_eq!(map_kv_quant("bf16"), "none");
}

#[test]
fn map_kv_quant_pass_through() {
    assert_eq!(map_kv_quant("k8v4"), "k8v4");
    assert_eq!(map_kv_quant("k8v8"), "k8v8");
    assert_eq!(map_kv_quant("none"), "none");
}

#[test]
fn legacy_bf16_kv_quant_converts_to_none() {
    let json = r#"{
        "run_id":         "20260511-163526-abc",
        "ts_utc":         "2026-05-11T16:35:26Z",
        "backend":        "rmlx",
        "model_namespace":"mlx-community",
        "model_name":     "Qwen3.6-35B-A3B-8bit",
        "weight_quant":   "q8_0",
        "kv_quant":       "bf16",
        "max_ctx":        8192,
        "prompt_tokens":  4096,
        "max_tokens":     30,
        "git_sha":        "abc1234",
        "observations": [
            {"metric":"decode_tps","value":90.74,"unit":"tps","direction":"higher_is_better","run_type":"warm","notes":"legacy-buffer"}
        ]
    }"#;
    let run = try_parse_legacy(json).expect("should parse");
    assert_eq!(run.kv_quant, "none");
}

/// A pre-§8.5 rMLX buffer file carries no `backend_version` — the shape simply
/// had no such key. Conversion still works, but the record is NOT ingestable:
/// we will not invent a semver for it, and silently writing another NULL is the
/// bug being fixed. It goes to `buffer/failed/` for triage.
#[test]
fn converted_legacy_rmlx_record_is_rejected_for_missing_identity() {
    let run = try_parse_legacy(legacy_json_with_prompt()).expect("should parse");
    assert_eq!(run.backend, "rmlx");
    assert!(run.backend_version.is_none());
    assert!(matches!(
        run.validate().unwrap_err(),
        crate::error::Error::MissingBackendVersion { .. }
    ));
}

// ── parse_cbb_weight_quant ────────────────────────────────────────────────

#[test]
fn cbb_wq_mxfp8_kv_bf16() {
    let (w, k) = parse_cbb_weight_quant("mxfp8 g32 + kv-bf16").unwrap();
    assert_eq!(w, "mxfp8");
    assert_eq!(k, "none");
}

#[test]
fn cbb_wq_affine_kv_k8v8() {
    let (w, k) = parse_cbb_weight_quant("affine g64 b8 + kv-k8v8").unwrap();
    assert_eq!(w, "8bit");
    assert_eq!(k, "k8v8");
}

#[test]
fn cbb_wq_turbo4_compound() {
    let (w, k) = parse_cbb_weight_quant("affine g64 b8 + kv-turbo4").unwrap();
    assert_eq!(w, "8bit");
    assert_eq!(k, "turbo4");
}

#[test]
fn cbb_wq_turbo3_v4_compound() {
    let (w, k) = parse_cbb_weight_quant("affine g64 b8 + kv-turbo3_v4").unwrap();
    assert_eq!(w, "8bit");
    assert_eq!(k, "turbo4");
}

#[test]
fn cbb_wq_ternary_kv_k8v4() {
    let (w, k) = parse_cbb_weight_quant("2-bit ternary + kv-k8v4").unwrap();
    assert_eq!(w, "2bit");
    assert_eq!(k, "k8v4");
}

#[test]
fn cbb_wq_ternary_kv_none() {
    let (w, k) = parse_cbb_weight_quant("2-bit ternary + kv-none").unwrap();
    assert_eq!(w, "2bit");
    assert_eq!(k, "none");
}

#[test]
fn cbb_wq_mxfp8_kv_planar() {
    let (w, k) = parse_cbb_weight_quant("mxfp8 g32 + kv-planar").unwrap();
    assert_eq!(w, "mxfp8");
    assert_eq!(k, "planar");
}

#[test]
fn cbb_wq_bare_canonical_no_kv() {
    // Bare canonical string with no " + " separator → kv = none.
    let (w, k) = parse_cbb_weight_quant("8bit").unwrap();
    assert_eq!(w, "8bit");
    assert_eq!(k, "none");
}

#[test]
fn cbb_wq_unknown_base_returns_none() {
    assert!(parse_cbb_weight_quant("unknown-quant + kv-bf16").is_none());
}

#[test]
fn cbb_wq_unknown_kv_suffix_returns_none() {
    assert!(parse_cbb_weight_quant("mxfp8 g32 + kv-future99").is_none());
}

// ── map_cbb_backend ───────────────────────────────────────────────────────

#[test]
fn cbb_backend_mlx_lm_turboquant() {
    assert_eq!(map_cbb_backend("mlx-lm-turboquant").unwrap(), "mlx_lm_tq");
}

#[test]
fn cbb_backend_mlx_lm() {
    assert_eq!(map_cbb_backend("mlx-lm").unwrap(), "mlx_lm");
}

#[test]
fn cbb_backend_omlx_passes() {
    assert_eq!(map_cbb_backend("omlx").unwrap(), "omlx");
}

#[test]
fn cbb_backend_unknown_returns_none() {
    assert!(map_cbb_backend("unknown-backend").is_none());
}

// ── try_parse_cbb full record ─────────────────────────────────────────────

fn cbb_json_rmlx_mxfp8_kv_k8v8() -> &'static str {
    r#"{
        "backend": "rmlx",
        "backend_version": "379dcea",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e2b-it-mxfp8",
        "weight_quant": "mxfp8 g32 + kv-k8v8",
        "kv_quant": "none",
        "ctx_max": 8192,
        "prompt": {
            "name": "longctx_4k",
            "tokens_approx": 4096,
            "body": [{"role":"user","content":"hello"}]
        },
        "ts_utc": "2026-05-10T17:12:12.087501+00:00",
        "hardware_tag": "m5_max_128gb",
        "prompt_tokens": 4096,
        "max_tokens": 32,
        "temperature": 0.0,
        "seed": 0,
        "n_warmups": 1,
        "n_measure": 1,
        "output_first_64": "llama.cpp: Longest README content.",
        "notes": "cbb-runner",
        "metrics": [
            {"name":"decode_tps_warm","value":120.74},
            {"name":"peak_rss_mb","value":8555.8},
            {"name":"ttft_warm_ms","value":1450.7},
            {"name":"overall_tps","value":19.2},
            {"name":"task_pass_at_1","value":null}
        ]
    }"#
}

#[test]
fn try_parse_cbb_rmlx_mxfp8_k8v8() {
    let run = try_parse_cbb(cbb_json_rmlx_mxfp8_kv_k8v8()).expect("should parse");
    assert_eq!(run.backend, "rmlx");
    assert_eq!(run.weight_quant, "mxfp8");
    assert_eq!(run.kv_quant, "k8v8");
    assert_eq!(run.model, "gemma-4-e2b-it-mxfp8");
    assert_eq!(run.ctx_max, 8192);
    // decode_tps_warm present and non-null
    let dtps = run
        .metrics
        .iter()
        .find(|m| m.name == "decode_tps_warm")
        .unwrap();
    assert!((dtps.value.unwrap() - 120.74).abs() < 0.01);
}

/// This CBB fixture carries `"backend_version": "379dcea"` — a git SHA stuffed
/// into the semver column. It is one of the exact junk values found in the live
/// DB. Conversion still works; ingest now rejects it instead of recording it.
#[test]
fn cbb_rmlx_record_with_git_sha_as_version_is_rejected() {
    let run = try_parse_cbb(cbb_json_rmlx_mxfp8_kv_k8v8()).unwrap();
    assert_eq!(run.backend_version.as_deref(), Some("379dcea"));
    assert!(matches!(
        run.validate().unwrap_err(),
        crate::error::Error::MissingBackendVersion { .. }
    ));
}

/// Cross-backend legacy ingest must keep working: a non-rMLX CBB record has no
/// semver to give and is accepted as before.
#[test]
fn cbb_non_rmlx_record_still_validates_without_a_version() {
    let json = r#"{
        "backend": "mlx-lm-turboquant",
        "model_namespace": "mlx-community",
        "model": "Qwen3.6-35B-A3B-8bit",
        "weight_quant": "affine g64 b8 + kv-turbo4",
        "kv_quant": "none",
        "ctx_max": 8192,
        "prompt": {"name":"longctx_4k","body":"hi","tokens_approx":4096},
        "ts_utc": "2026-05-10T18:45:00+00:00",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{"name":"decode_tps_warm","value":55.3}]
    }"#;
    let run = try_parse_cbb(json).expect("should parse");
    assert_eq!(run.backend, "mlx_lm_tq");
    assert!(run.backend_version.is_none());
    run.validate()
        .expect("non-rMLX legacy record must still ingest");
}

#[test]
fn try_parse_cbb_turboquant_backend() {
    let json = r#"{
        "backend": "mlx-lm-turboquant",
        "model_namespace": "mlx-community",
        "model": "Qwen3.6-35B-A3B-8bit",
        "weight_quant": "affine g64 b8 + kv-turbo4",
        "kv_quant": "none",
        "ctx_max": 8192,
        "prompt": {"name":"longctx_4k","body":"hi","tokens_approx":4096},
        "ts_utc": "2026-05-10T18:45:00+00:00",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{"name":"decode_tps_warm","value":55.3}]
    }"#;
    let run = try_parse_cbb(json).expect("should parse");
    assert_eq!(run.backend, "mlx_lm_tq");
    assert_eq!(run.weight_quant, "8bit");
    assert_eq!(run.kv_quant, "turbo4");
    run.validate().expect("should validate");
}

#[test]
fn try_parse_cbb_returns_none_for_legacy_n74_shape() {
    // Legacy bench-script shape has model_name not model — must NOT be parsed by try_parse_cbb.
    let n74_json = r#"{
        "backend": "rmlx",
        "model_namespace": "mlx-community",
        "model_name": "Qwen3.6-35B-A3B-8bit",
        "weight_quant": "q8_0",
        "kv_quant": "k8v4",
        "max_ctx": 8192,
        "ts_utc": "2026-05-11T12:00:00Z",
        "hardware_tag": "m5_max_128gb",
        "observations": [{"metric":"decode_tps","value":85.0}]
    }"#;
    // Legacy bench-script shape has no `metrics` key so try_parse_cbb should return None.
    assert!(try_parse_cbb(n74_json).is_none());
}

#[test]
fn try_parse_cbb_returns_none_for_canonical_record() {
    // A perfectly canonical §8.5 record with no compound weight_quant
    // and a canonical backend should NOT be captured by try_parse_cbb.
    let canonical = r#"{
        "backend": "rmlx",
        "model_namespace": "mlx-community",
        "model": "gemma-4-e4b-it-mxfp8",
        "weight_quant": "mxfp8",
        "kv_quant": "k8v8",
        "ctx_max": 8192,
        "prompt": {"name":"t","body":"hi"},
        "ts_utc": "2026-05-10T07:30:00Z",
        "hardware_tag": "m5_max_128gb",
        "metrics": [{"name":"decode_tps_warm","value":100.0}]
    }"#;
    assert!(try_parse_cbb(canonical).is_none());
}
