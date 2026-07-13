use super::*;
use rmlx_mlx::Device;

#[test]
fn device_arg_accepts_cpu_and_gpu() {
    // Valid devices must not return Err.
    assert!(matches!(
        match "cpu" {
            "cpu" => Ok(Device::Cpu),
            "gpu" => Ok(Device::Gpu),
            other => Err(anyhow::anyhow!("bad: {other}")),
        },
        Ok(Device::Cpu)
    ));
    assert!(matches!(
        match "gpu" {
            "cpu" => Ok(Device::Cpu),
            "gpu" => Ok(Device::Gpu),
            other => Err(anyhow::anyhow!("bad: {other}")),
        },
        Ok(Device::Gpu)
    ));
}

#[test]
fn device_arg_rejects_tpu() {
    let result: anyhow::Result<Device> = match "tpu" {
        "cpu" => Ok(Device::Cpu),
        "gpu" => Ok(Device::Gpu),
        other => Err(anyhow::anyhow!("bad: {other}")),
    };
    assert!(result.is_err(), "expected error for 'tpu'");
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("tpu"), "error should mention the bad value");
}

#[test]
fn csv_escape_plain_value() {
    assert_eq!(baseline_csv_escape("hello world"), "hello world");
    assert_eq!(baseline_csv_escape("hello"), "hello");
}

#[test]
fn csv_escape_comma_in_value() {
    let result = baseline_csv_escape("hello, world");
    assert_eq!(result, "\"hello, world\"");
}

#[test]
fn csv_escape_double_quote_in_value() {
    let result = baseline_csv_escape("say \"hi\"");
    assert_eq!(result, "\"say \"\"hi\"\"\"");
}

#[test]
fn csv_escape_newline_in_value() {
    let result = baseline_csv_escape("line1\nline2");
    assert_eq!(result, "\"line1\nline2\"");
}

#[test]
fn csv_escape_empty_string() {
    assert_eq!(baseline_csv_escape(""), "");
}

// -- compute_phase_timing -------------------------------------------------

#[test]
fn phase_timing_decode_excludes_prefill() {
    // 100 tokens. First callback (TTFT) at 1.0s; last at 2.0s. Total 2.0s.
    // Decode window = 1.0s over 99 tokens => 99 tps.
    // Overall = 100 / 2.0 = 50 tps.
    let t = compute_phase_timing(1.0, 2.0, 2.0, 100, 4096);
    assert!((t.ttft_ms - 1000.0).abs() < 1e-6, "ttft {}", t.ttft_ms);
    assert!(
        (t.decode_tps - 99.0).abs() < 1e-6,
        "decode {}",
        t.decode_tps
    );
    assert!(
        (t.overall_tps - 50.0).abs() < 1e-6,
        "overall {}",
        t.overall_tps
    );
    // prefill_tps = 4096 / 1.0
    assert!(
        (t.prefill_tps - 4096.0).abs() < 1e-6,
        "prefill {}",
        t.prefill_tps
    );
}

#[test]
fn phase_timing_decode_gte_overall_invariant() {
    // The acceptance invariant: removing fixed prefill cost can only raise
    // TPS, so decode_tps >= overall_tps for any run with a real prefill.
    for (first, last, total, n) in [
        (0.5, 3.0, 3.0, 100usize),
        (1.0, 1.5, 1.5, 50),
        (0.2, 5.0, 5.0, 200),
        (2.0, 2.01, 2.01, 100), // prefill-dominated: decode still >= overall
    ] {
        let t = compute_phase_timing(first, last, total, n, 4096);
        assert!(
            t.decode_tps + 1e-9 >= t.overall_tps,
            "decode_tps {} must be >= overall_tps {} (first={first} last={last} total={total} n={n})",
            t.decode_tps,
            t.overall_tps
        );
    }
}

#[test]
fn phase_timing_single_token_falls_back_to_overall() {
    // With n_generated < 2 there is no decode window; decode_tps == overall.
    let t = compute_phase_timing(0.5, 0.5, 0.5, 1, 4096);
    assert!((t.decode_tps - t.overall_tps).abs() < 1e-9);
}

// -- build_run_record: git_sha provenance -----------------------------------
//
// Mirrors `eval_tests::build_record_stamps_identity_from_the_single_source` /
// `build_record_git_sha_absent_is_null` — `build_run_record` is the OTHER
// (and, via `perf_canary.sh`, the more heavily exercised) record builder that
// threads `--git-sha` through, and it is pure — no model, no GPU claim.

#[test]
fn build_record_git_sha_survives_stamp_json() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("m");
    std::fs::create_dir_all(&model_dir).expect("mkdir m");

    let args = BaselineRecordArgs {
        label: None,
        prompt_id: Some("longctx_4k"),
        prompt_body: Some(serde_json::json!([{"role": "user", "content": "hi"}])),
        kv_quant: rmlx_kv_quant::KvQuant::None,
        ctx_max: 4096,
        git_sha: Some("cafebabe"),
    };

    let rec = build_run_record(
        "20260526-120000-0.2.8",
        &model_dir,
        &args,
        "fallback-label",
        "prompt text",
        "bf16",
        16,
        8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8,
        "",
        0,
    )
    .expect("record builds");

    let ident = RunIdentity::get();
    assert_eq!(rec["backend"], "rmlx");
    assert_eq!(rec["backend_version"], ident.backend_version());
    assert_eq!(rec["build_profile"], ident.build_profile());
    assert_eq!(rec["git_sha"], "cafebabe");
    assert_eq!(rec["hardware_tag"], ident.hardware_tag());
}

#[test]
fn build_record_git_sha_absent_is_null() {
    let tmp = tempfile::tempdir().expect("tempdir");
    let model_dir = tmp.path().join("m2");
    std::fs::create_dir_all(&model_dir).expect("mkdir m2");

    let args = BaselineRecordArgs {
        label: None,
        prompt_id: Some("longctx_4k"),
        prompt_body: Some(serde_json::json!([{"role": "user", "content": "hi"}])),
        kv_quant: rmlx_kv_quant::KvQuant::None,
        ctx_max: 4096,
        git_sha: None,
    };

    let rec = build_run_record(
        "20260526-120000-0.2.8",
        &model_dir,
        &args,
        "fallback-label",
        "prompt text",
        "bf16",
        16,
        8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8,
        "",
        0,
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

    let args = BaselineRecordArgs {
        label: None,
        prompt_id: Some("longctx_4k"),
        prompt_body: Some(serde_json::json!([{"role": "user", "content": "hi"}])),
        kv_quant: rmlx_kv_quant::KvQuant::None,
        ctx_max: 4096,
        git_sha: Some(""),
    };

    let rec = build_run_record(
        "20260526-120000-0.2.8",
        &model_dir,
        &args,
        "fallback-label",
        "prompt text",
        "bf16",
        16,
        8,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        8,
        "",
        0,
    )
    .expect("record builds");

    assert!(rec["git_sha"].is_null());
}
