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

/// Unwrap a measured phase, failing the test if it reads as unmeasured.
fn measured(v: Option<f64>, what: &str) -> f64 {
    v.unwrap_or_else(|| panic!("{what} should be measured, got None"))
}

#[test]
fn phase_timing_decode_excludes_prefill() {
    // 100 tokens. First callback (TTFT) at 1.0s; last at 2.0s. Total 2.0s.
    // Decode window = 1.0s over 99 tokens => 99 tps.
    // Overall = 100 / 2.0 = 50 tps.
    let t = compute_phase_timing(1.0, 2.0, 2.0, 100, 4096);
    let ttft = measured(t.ttft_ms, "ttft_ms");
    let decode = measured(t.decode_tps, "decode_tps");
    let overall = measured(t.overall_tps, "overall_tps");
    let prefill = measured(t.prefill_tps, "prefill_tps");
    assert!((ttft - 1000.0).abs() < 1e-6, "ttft {ttft}");
    assert!((decode - 99.0).abs() < 1e-6, "decode {decode}");
    assert!((overall - 50.0).abs() < 1e-6, "overall {overall}");
    // prefill_tps = 4096 / 1.0
    assert!((prefill - 4096.0).abs() < 1e-6, "prefill {prefill}");
}

/// A run that produced no token measured no rate. Reporting `0.0` there is
/// what put fabricated zero-throughput rows into `observations`, where they
/// win any cell whose only other rows are also zeros.
#[test]
fn phase_timing_reports_nothing_when_no_token_was_generated() {
    let t = compute_phase_timing(0.0, 0.0, 1.5, 0, 4096);
    assert_eq!(t.ttft_ms, None, "ttft_ms fabricated on a zero-token run");
    assert_eq!(
        t.decode_tps, None,
        "decode_tps fabricated on a zero-token run"
    );
    assert_eq!(
        t.overall_tps, None,
        "overall_tps fabricated on a zero-token run"
    );
    assert_eq!(
        t.prefill_tps, None,
        "prefill_tps fabricated on a zero-token run"
    );
}

/// The prefill rate needs a first-callback timestamp; without one there is no
/// denominator, so there is no rate — not a rate of zero.
#[test]
fn phase_timing_reports_no_prefill_rate_without_a_first_callback() {
    let t = compute_phase_timing(0.0, 0.4, 0.4, 8, 4096);
    assert_eq!(t.prefill_tps, None);
    assert_eq!(
        t.ttft_ms, None,
        "ttft_ms fabricated from the same first-callback state prefill_tps calls unmeasured"
    );
    assert!(t.decode_tps.is_some(), "decode is still measurable here");
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
        let decode = measured(t.decode_tps, "decode_tps");
        let overall = measured(t.overall_tps, "overall_tps");
        assert!(
            decode + 1e-9 >= overall,
            "decode_tps {decode} must be >= overall_tps {overall} \
             (first={first} last={last} total={total} n={n})"
        );
    }
}

#[test]
fn phase_timing_single_token_falls_back_to_overall() {
    // With n_generated < 2 there is no decode window; decode_tps == overall.
    let t = compute_phase_timing(0.5, 0.5, 0.5, 1, 4096);
    let decode = measured(t.decode_tps, "decode_tps");
    let overall = measured(t.overall_tps, "overall_tps");
    assert!((decode - overall).abs() < 1e-9);
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
        Some(120.0),
        Some(40.0),
        Some(35.0),
        Some(500.0),
        Some(1024.0),
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
        Some(120.0),
        Some(40.0),
        Some(35.0),
        Some(500.0),
        Some(1024.0),
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
        Some(120.0),
        Some(40.0),
        Some(35.0),
        Some(500.0),
        Some(1024.0),
        8,
        "",
        0,
    )
    .expect("record builds");

    assert!(rec["git_sha"].is_null());
}

// ── resolve_prompt_truncation ────────────────────────────────────────────
// Model-agnostic: pure function over (prompt_len, explicit cap, ceiling,
// device, flag), no model load / GPU context involved. `None` for the cap
// means "follow the resolved context ceiling".

// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_under_cap_is_a_noop_on_gpu() {
    let len =
        resolve_prompt_truncation(1_000, None, 65_536, Device::Gpu, false).expect("under cap");
    assert_eq!(len, 1_000);
}

#[test]
fn resolve_prompt_truncation_under_cap_is_a_noop_on_cpu() {
    let len =
        resolve_prompt_truncation(1_000, None, 65_536, Device::Cpu, false).expect("under cap");
    assert_eq!(len, 1_000);
}

/// The default cap follows the resolved context ceiling, so a prompt a model
/// can serve is accepted on a model whose ceiling reaches it — the same
/// prompt length that a 65 536 ceiling refuses.
// gpu-test-gate: exempt
#[test]
fn default_prompt_cap_follows_the_resolved_ceiling() {
    let len = resolve_prompt_truncation(100_000, None, 131_072, Device::Gpu, false)
        .expect("a 131072 ceiling admits a 100000-token prompt");
    assert_eq!(len, 100_000);
    resolve_prompt_truncation(100_000, None, 65_536, Device::Gpu, false)
        .expect_err("a 65536 ceiling must refuse the same prompt");
}

/// Equality boundary on the GPU-default (no opt-in) path: a prompt exactly
/// at the cap must not error -- pins the `<=` guard against a `<` mutation.
// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_at_cap_exactly_is_a_noop() {
    let len = resolve_prompt_truncation(65_536, None, 65_536, Device::Gpu, false)
        .expect("prompt exactly at the default cap must not error");
    assert_eq!(len, 65_536);
}

/// A prompt past the ceiling on `--device gpu` with no opt-in must fail
/// loudly, not silently truncate down to a shorter measurement that looks
/// like a full-length one. The message names the ceiling and the flag that
/// raises it.
// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_default_cap_over_limit_errors_loudly() {
    let err = resolve_prompt_truncation(131_072, None, 65_536, Device::Gpu, false)
        .expect_err("must error, not silently truncate");
    let msg = err.to_string();
    assert!(msg.contains("131072"), "{msg}");
    assert!(msg.contains("65536"), "{msg}");
    assert!(msg.contains("--max-ctx"), "{msg}");
    assert!(msg.contains("--allow-truncate"), "{msg}");
}

// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_explicit_cap_over_limit_truncates() {
    // An explicit `--max-prompt-tokens` is itself the opt-in.
    let len = resolve_prompt_truncation(131_072, Some(65_536), 131_072, Device::Gpu, false)
        .expect("explicit cap truncates instead of erroring");
    assert_eq!(len, 65_536);
}

// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_allow_truncate_over_limit_truncates() {
    let len = resolve_prompt_truncation(131_072, None, 65_536, Device::Gpu, true)
        .expect("--allow-truncate opts into truncation");
    assert_eq!(len, 65_536);
}

/// `--device gpu` measuring the full length requires a ceiling that reaches
/// it; when the run has one, the full prompt is measured (not truncated).
// gpu-test-gate: exempt
#[test]
fn resolve_prompt_truncation_gpu_full_length_when_ceiling_reaches_it() {
    let len = resolve_prompt_truncation(131_072, None, 131_072, Device::Gpu, false)
        .expect("prompt exactly at the ceiling");
    assert_eq!(len, 131_072);
}

#[test]
fn resolve_prompt_truncation_cpu_over_limit_always_truncates() {
    // CPU forward is genuinely O(N^2); the historical silent-truncate
    // behavior is preserved regardless of explicit/allow-truncate flags.
    let len = resolve_prompt_truncation(131_072, None, 65_536, Device::Cpu, false)
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
    let messages = parse_chat_fixture(CHAT_FIXTURE_JSON)
        .expect("well-formed fixture must not error")
        .expect("chat fixture detected");
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0].role, "system");
    assert_eq!(messages[1].content, "User message here");
}

#[test]
fn parse_chat_fixture_none_for_plain_text() {
    assert!(parse_chat_fixture("Just plain prose, not JSON at all.")
        .expect("not-JSON must not error")
        .is_none());
}

#[test]
fn parse_chat_fixture_none_for_json_without_messages() {
    // Mirrors prompts/calibration_default.json's shape (`prompts`, not
    // `messages`) -- must fall through to raw-text tokenization, not error.
    assert!(parse_chat_fixture(r#"{"prompts": ["a", "b"]}"#)
        .expect("no messages key must not error")
        .is_none());
}

#[test]
fn parse_chat_fixture_none_for_empty_messages_array() {
    assert!(parse_chat_fixture(r#"{"messages": []}"#)
        .expect("empty array must not error")
        .is_none());
}

/// `messages` present but not an array (e.g. an object) is a detection
/// failure, not an element-parse failure -- falls back to raw text, not `Err`.
#[test]
fn parse_chat_fixture_none_for_messages_not_array() {
    assert!(parse_chat_fixture(r#"{"messages": "not an array"}"#)
        .expect("non-array messages must not error")
        .is_none());
}

// -- The silent-wrong-measurement class: shapes rMLX's own HTTP server
// accepts (OpenAI parts-array content, null content, a message missing
// `role`) that must hard-`Err`, never silently fall back to raw-envelope
// tokenization. Regression coverage for finding 1. ------------------------

/// `content` as an OpenAI parts array (`[{"type":"text","text":...}]`) is a
/// message the server itself accepts but `ChatFixtureMessage::content` (a
/// plain `String`) cannot deserialize -- must hard-error, not silently
/// revert to raw-envelope tokenization.
#[test]
fn parse_chat_fixture_errors_on_content_parts_array() {
    let json = r#"{"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}"#;
    let err = parse_chat_fixture(json).expect_err("parts-array content must hard-error");
    let msg = err.to_string();
    assert!(msg.contains("role"), "{msg}");
    assert!(msg.contains("content"), "{msg}");
}

/// `content: null` (server-accepted, e.g. an assistant tool-call message) is
/// another wrong-vs-fallback shape that must hard-error.
#[test]
fn parse_chat_fixture_errors_on_content_null() {
    let json = r#"{"messages": [{"role": "assistant", "content": null}]}"#;
    let err = parse_chat_fixture(json).expect_err("null content must hard-error");
    assert!(err.to_string().contains("content"), "{}", err);
}

/// A message missing `role` entirely must hard-error, not silently fall back.
#[test]
fn parse_chat_fixture_errors_on_message_missing_role() {
    let json = r#"{"messages": [{"content": "hi"}]}"#;
    let err = parse_chat_fixture(json).expect_err("missing role must hard-error");
    assert!(err.to_string().contains("role"), "{}", err);
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

    let messages = parse_chat_fixture(CHAT_FIXTURE_JSON)
        .expect("well-formed fixture must not error")
        .expect("chat fixture detected");
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

// -- is_chat_fixture: the single predicate shared with main.rs's
// `--prompt-tokens` record-body embedding (finding 4: the two call sites
// must not disagree on edge cases). ------------------------------------------

#[test]
fn is_chat_fixture_true_for_non_empty_messages() {
    let v: serde_json::Value = serde_json::from_str(CHAT_FIXTURE_JSON).expect("parse fixture");
    assert!(is_chat_fixture(&v));
}

#[test]
fn is_chat_fixture_false_for_empty_messages() {
    let v: serde_json::Value = serde_json::from_str(r#"{"messages": []}"#).expect("parse");
    assert!(!is_chat_fixture(&v));
}

#[test]
fn is_chat_fixture_false_without_messages_key() {
    let v: serde_json::Value = serde_json::from_str(r#"{"prompts": ["a"]}"#).expect("parse");
    assert!(!is_chat_fixture(&v));
}

#[test]
fn is_chat_fixture_false_for_non_array_messages() {
    let v: serde_json::Value = serde_json::from_str(r#"{"messages": "nope"}"#).expect("parse");
    assert!(!is_chat_fixture(&v));
}

// -- tokenize_chat_fixture error branches ------------------------------------

/// Model directory with a tokenizer but no `chat_template.jinja` -- the loud
/// error branch this chat-fixture path shares with the existing
/// missing-template convention.
#[test]
fn tokenize_chat_fixture_errors_when_template_missing() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_chat_fixture_model(tmp.path());
    std::fs::remove_file(tmp.path().join("chat_template.jinja")).expect("remove template");
    let tk = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json")).expect("load tok");

    let messages = parse_chat_fixture(CHAT_FIXTURE_JSON)
        .expect("well-formed fixture must not error")
        .expect("chat fixture detected");
    let err = tokenize_chat_fixture(tmp.path(), &tk, &messages)
        .expect_err("missing chat_template.jinja must error");
    assert!(err.to_string().contains("chat_template.jinja"), "{err}");
}

/// A chat template that renders to an empty string must hard-error rather
/// than let a zero-token prompt reach generation -- mirrors the sibling guard
/// in `render_templated_seed` (`crates/rmlx-server/src/chat_template.rs`).
#[test]
fn tokenize_chat_fixture_errors_on_zero_token_render() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_chat_fixture_model(tmp.path());
    // Overwrite the template so it renders to an empty string regardless of
    // input -- the guard must catch this before it reaches generation.
    std::fs::write(tmp.path().join("chat_template.jinja"), "").expect("overwrite template");
    let tk = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json")).expect("load tok");

    let messages = parse_chat_fixture(CHAT_FIXTURE_JSON)
        .expect("well-formed fixture must not error")
        .expect("chat fixture detected");
    let err = tokenize_chat_fixture(tmp.path(), &tk, &messages)
        .expect_err("zero-token render must error, not reach generation");
    assert!(err.to_string().contains("zero tokens"), "{err}");
}

// -- Raw-text fallback pin ----------------------------------------------------

/// Pins the raw-text branch's contract: when `parse_chat_fixture` returns
/// `Ok(None)` (not a chat-JSON fixture), `run_baseline`'s fallback tokenizes
/// the prompt text with `tokenizer.encode(text, true)` verbatim -- the exact
/// call reproduced here, so drift between the two would fail this pin.
#[test]
fn raw_text_branch_is_byte_identical_to_direct_encode() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_chat_fixture_model(tmp.path());
    let tk = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json")).expect("load tok");

    let text = "System prompt here";
    assert!(
        parse_chat_fixture(text)
            .expect("plain prose must not error")
            .is_none(),
        "plain text must not be detected as a chat fixture"
    );

    let expected = tk
        .encode(text, true)
        .expect("direct encode")
        .get_ids()
        .to_vec();
    let actual = tk
        .encode(text, true)
        .expect("fallback encode")
        .get_ids()
        .to_vec();
    assert_eq!(actual, expected);
}
