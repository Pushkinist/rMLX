// Device-policy tests: `Device::Gpu` here is a selector value passed to pure
// functions (`resolve_prompt_truncation`, device-arg parsing), never a Metal
// dispatch. Each such test carries a per-fn `gpu-test-gate: exempt` marker so
// the shape gate does not treat the value as a GPU test — while any genuinely
// Metal-driving test added to this file still trips the gate.
use super::*;
use rmlx_mlx::Device;

// gpu-test-gate: exempt
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

// gpu-test-gate: exempt
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

// ── resolve_prompt_truncation ────────────────────────────────────────────
// Model-agnostic: pure function over (prompt_len, cap, device, flags), no
// model load / GPU context involved.

// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_under_cap_is_a_noop_on_gpu() {
    let len =
        resolve_prompt_truncation(1_000, 65_536, Device::Gpu, false, false).expect("under cap");
    assert_eq!(len, 1_000);
}

#[test]
fn resolve_prompt_truncation_under_cap_is_a_noop_on_cpu() {
    let len =
        resolve_prompt_truncation(1_000, 65_536, Device::Cpu, false, false).expect("under cap");
    assert_eq!(len, 1_000);
}

/// Equality boundary on the GPU-default (no opt-in) path: a prompt exactly
/// at the cap must not error -- pins the `<=` guard against a `<` mutation.
// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_at_cap_exactly_is_a_noop() {
    let len = resolve_prompt_truncation(65_536, 65_536, Device::Gpu, false, false)
        .expect("prompt exactly at the default cap must not error");
    assert_eq!(len, 65_536);
}

/// The bug this fixes: a >65536-token prompt on `--device gpu` with the
/// default cap and no opt-in must fail loudly, not silently truncate down to
/// a shorter measurement that looks like a full-length one.
// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_default_cap_over_limit_errors_loudly() {
    let err = resolve_prompt_truncation(131_072, 65_536, Device::Gpu, false, false)
        .expect_err("must error, not silently truncate");
    let msg = err.to_string();
    assert!(msg.contains("131072"), "{msg}");
    assert!(msg.contains("65536"), "{msg}");
    assert!(msg.contains("--max-prompt-tokens"), "{msg}");
    assert!(msg.contains("--allow-truncate"), "{msg}");
}

// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_explicit_cap_over_limit_truncates() {
    // An explicit `--max-prompt-tokens` is itself the opt-in.
    let len = resolve_prompt_truncation(131_072, 65_536, Device::Gpu, true, false)
        .expect("explicit cap truncates instead of erroring");
    assert_eq!(len, 65_536);
}

// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_allow_truncate_over_limit_truncates() {
    let len = resolve_prompt_truncation(131_072, 65_536, Device::Gpu, false, true)
        .expect("--allow-truncate opts into truncation");
    assert_eq!(len, 65_536);
}

/// `--device gpu` measuring the full length requires raising the cap; when
/// the caller does that, the full prompt is measured (not truncated).
// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_full_length_when_cap_raised() {
    let len = resolve_prompt_truncation(131_072, 131_072, Device::Gpu, true, false)
        .expect("prompt exactly at the raised cap");
    assert_eq!(len, 131_072);
}

#[test]
fn resolve_prompt_truncation_cpu_over_limit_always_truncates() {
    // CPU forward is genuinely O(N^2); the historical silent-truncate
    // behavior is preserved regardless of explicit/allow-truncate flags.
    let len = resolve_prompt_truncation(131_072, 65_536, Device::Cpu, false, false)
        .expect("cpu always truncates");
    assert_eq!(len, 65_536);
}

// ── chat-JSON fixture tokenization (bug: baseline tokenized the raw JSON
// envelope, not the message content) ────────────────────────────────────
//
// A chat-JSON bench fixture (`prompts/longctx_<N>k.json`) is a JSON envelope
// around a `messages` array. `run_baseline` must tokenize the *rendered
// message content* through the model's chat_template.jinja, matching the
// HTTP chat-completions path -- not the raw JSON envelope + syntax text.
// `write_chat_fixture_model` builds a tiny WordLevel tokenizer +
// chat_template.jinja pair (mirrors
// `chat_template_tests::write_smoke_fixture`) so these tests run with no
// real model snapshot.

/// Fixture JSON: two messages plus non-content envelope fields
/// (`prompt_tokens`, `label`) that must never end up in the token count.
const CHAT_FIXTURE_JSON: &str = r#"{"messages": [{"role": "system", "content": "System prompt here"}, {"role": "user", "content": "User message here"}], "prompt_tokens": 999, "label": "test"}"#;

fn write_chat_fixture_model(dir: &Path) {
    // "messages"/"role"/"content"/"prompt_tokens"/"label"/"test" are JSON
    // envelope/key vocabulary -- present in the raw fixture text but absent
    // from the rendered message content, so their token ids are the
    // discriminator between the buggy raw-tokenize path and the fixed
    // template-rendered path.
    let vocab = r#"{
        "<unk>":2,"<bos>":0,
        "<start_of_turn>":10,"<end_of_turn>":11,
        "system":12,"user":13,"model":14,
        "System":20,"prompt":21,"here":22,"User":23,"message":24,
        "messages":30,"role":31,"content":32,"prompt_tokens":33,"label":34,"test":35
    }"#;
    let added = r#"[
        {"id":0,"content":"<bos>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        {"id":10,"content":"<start_of_turn>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true},
        {"id":11,"content":"<end_of_turn>","single_word":false,"lstrip":false,"rstrip":false,"normalized":false,"special":true}
    ]"#;
    let tok_json = format!(
        r#"{{"version":"1.0","truncation":null,"padding":null,"added_tokens":{added},"normalizer":null,"pre_tokenizer":{{"type":"Whitespace"}},"post_processor":null,"decoder":null,"model":{{"type":"WordLevel","vocab":{vocab},"unk_token":"<unk>"}}}}"#
    );
    std::fs::write(dir.join("tokenizer.json"), tok_json).expect("write tokenizer.json");
    std::fs::write(
        dir.join("tokenizer_config.json"),
        r#"{"bos_token":"<bos>","eos_token":"<eos>"}"#,
    )
    .expect("write tokenizer_config.json");
    let tpl = "{{ bos_token }} {% for m in messages %}<start_of_turn> {{ m.role }} {{ m.content }} <end_of_turn> {% endfor %}{% if add_generation_prompt %}<start_of_turn> model{% endif %}";
    std::fs::write(dir.join("chat_template.jinja"), tpl).expect("write chat_template.jinja");
}

#[test]
fn parse_chat_fixture_detects_messages_array() {
    let messages = parse_chat_fixture(CHAT_FIXTURE_JSON).expect("chat fixture detected");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[1].content, "User message here");
}

#[test]
fn parse_chat_fixture_none_for_plain_text() {
    assert!(parse_chat_fixture("Just plain prose, not JSON at all.").is_none());
}

#[test]
fn parse_chat_fixture_none_for_json_without_messages() {
    // Mirrors prompts/calibration_default.json's shape (`prompts`, not
    // `messages`) -- must fall through to raw-text tokenization, not error.
    assert!(parse_chat_fixture(r#"{"prompts": ["a", "b"]}"#).is_none());
}

#[test]
fn parse_chat_fixture_none_for_empty_messages_array() {
    assert!(parse_chat_fixture(r#"{"messages": []}"#).is_none());
}

/// The bug this fixes: tokenizing the RAW chat-JSON fixture text (the old
/// `run_baseline` behavior -- `tokenizer.encode(prompt_text, true)` with no
/// fixture detection) counts the JSON envelope key `"messages"` as a real
/// prompt token even though it is pure structure, not content.
#[test]
fn raw_json_tokenize_counts_envelope_keys_as_tokens() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_chat_fixture_model(tmp.path());
    let tk = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json")).expect("load tok");

    let raw_ids = tk
        .encode(CHAT_FIXTURE_JSON, true)
        .expect("encode raw")
        .get_ids()
        .to_vec();
    // id 30 == "messages", the envelope key -- present only because the raw
    // JSON text (not the rendered content) was tokenized.
    assert!(
        raw_ids.contains(&30),
        "expected raw-JSON tokenize to include the envelope key token: {raw_ids:?}"
    );
}

/// `tokenize_chat_fixture` renders only the message content through the chat
/// template -- the JSON envelope key `"messages"` must never appear in the
/// tokenized output, and the actual content words must.
#[test]
fn tokenize_chat_fixture_excludes_envelope_tokens() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_chat_fixture_model(tmp.path());
    let tk = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json")).expect("load tok");

    let messages = parse_chat_fixture(CHAT_FIXTURE_JSON).expect("chat fixture detected");
    let ids = tokenize_chat_fixture(tmp.path(), &tk, &messages).expect("tokenize chat fixture");

    // id 30 == "messages" (JSON envelope key) must be absent.
    assert!(
        !ids.contains(&30),
        "chat-template tokenization must not include the JSON envelope key: {ids:?}"
    );
    // Real message content must be present: System(20), prompt(21), here(22),
    // User(23), message(24).
    for t in [20u32, 21, 22, 23, 24] {
        assert!(ids.contains(&t), "missing content token {t}: {ids:?}");
    }
    // Turn-structured: begins with BOS.
    assert_eq!(ids.first(), Some(&0), "must begin with BOS: {ids:?}");
}
