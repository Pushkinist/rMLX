use super::*;

fn parse(json: &str) -> Result<ChatCompletionsRequest, serde_json::Error> {
    serde_json::from_str(json)
}

// ── J3: typed OOM mapping ────────────────────────────────────────────────
//
// Each `OomPhase` → asserted (HTTP status, `type` string, `Retry-After`
// presence/absence, body carries the J4 process-memory fields). Built by
// unit-constructing the `Error::Oom` variant and routing it through
// `engine_error_response` — no real OOM is forced (machine stability >
// a real-OOM e2e, per task constraints).

async fn oom_parts(e: &rmlx_core::Error) -> (StatusCode, Option<String>, Value) {
    let resp = engine_error_response(e);
    let status = resp.status();
    let retry = resp
        .headers()
        .get("Retry-After")
        .map(|v| v.to_str().unwrap().to_owned());
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    let body: Value = serde_json::from_slice(&bytes).unwrap();
    (status, retry, body)
}

fn assert_mem_fields(err: &Value) {
    // J4 fields must always be present (value may be null if read_proc_mem
    // failed, but the key must exist). On macOS CI they are real numbers.
    for k in ["process_rss_mb", "phys_footprint_mb", "compressed_mb"] {
        assert!(err.get(k).is_some(), "missing mem field {k}: {err}");
    }
    // peak_alloc_mb / requested_bytes present, null until F3 / call-site sets them.
    assert!(err.get("peak_alloc_mb").is_some(), "missing peak_alloc_mb");
    assert_eq!(err["peak_alloc_mb"], Value::Null);
    assert!(
        err.get("requested_bytes").is_some(),
        "missing requested_bytes"
    );
}

#[tokio::test]
async fn j3_oom_load_weights_507_retry() {
    let e = rmlx_core::Error::Oom {
        phase: rmlx_core::OomPhase::LoadWeights,
        requested_bytes: Some(42),
        peak_alloc_mb: None,
        msg: "unable to allocate".to_owned(),
    };
    let (status, retry, body) = oom_parts(&e).await;
    assert_eq!(status, StatusCode::INSUFFICIENT_STORAGE); // 507
    assert_eq!(body["error"]["type"], "oom_during_load");
    assert_eq!(retry.as_deref(), Some("5"));
    assert_eq!(body["error"]["requested_bytes"], 42);
    assert_mem_fields(&body["error"]);
}

#[tokio::test]
async fn j3_oom_kv_cache_507_retry() {
    let e = rmlx_core::Error::Oom {
        phase: rmlx_core::OomPhase::LoadKvCache,
        requested_bytes: None,
        peak_alloc_mb: None,
        msg: "kv growth failed".to_owned(),
    };
    let (status, retry, body) = oom_parts(&e).await;
    assert_eq!(status, StatusCode::INSUFFICIENT_STORAGE); // 507
    assert_eq!(body["error"]["type"], "oom_kv_cache");
    assert_eq!(retry.as_deref(), Some("5"));
    assert_mem_fields(&body["error"]);
}

#[tokio::test]
async fn j3_oom_generation_503_no_retry() {
    let e = rmlx_core::Error::Oom {
        phase: rmlx_core::OomPhase::Generation,
        requested_bytes: None,
        peak_alloc_mb: None,
        msg: "decode step alloc failed".to_owned(),
    };
    let (status, retry, body) = oom_parts(&e).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE); // 503
    assert_eq!(body["error"]["type"], "oom_mid_stream");
    assert!(retry.is_none(), "mid-stream OOM must NOT set Retry-After");
    assert_mem_fields(&body["error"]);
}

#[test]
fn minimal_request_deserialises() {
    let req = parse(r#"{"model":"gpt-4","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert_eq!(req.model, "gpt-4");
    assert_eq!(req.messages.len(), 1);
    assert!(!req.stream);
    assert!(req.temperature.is_none());
}

#[test]
fn streaming_flag_deserialises() {
    let req =
        parse(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"stream":true}"#).unwrap();
    assert!(req.stream);
}

#[test]
fn stop_as_string_deserialises() {
    let req =
        parse(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"stop":"\n"}"#).unwrap();
    let stop = req.stop.unwrap().into_vec();
    assert_eq!(stop, vec!["\n"]);
}

#[test]
fn stop_as_array_deserialises() {
    let req = parse(
        r#"{"model":"m","messages":[{"role":"user","content":"x"}],"stop":["\n","<|end|>"]}"#,
    )
    .unwrap();
    assert_eq!(req.stop.unwrap().into_vec().len(), 2);
}

#[test]
fn extra_fields_land_in_map() {
    let req =
        parse(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"some_new_field":42}"#)
            .unwrap();
    assert!(req.extra.contains_key("some_new_field"));
}

/// Validates that temperature = -1.0 would be caught by the route handler.
#[test]
fn temperature_out_of_range_flag() {
    let req =
        parse(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"temperature":-1.0}"#)
            .unwrap();
    let t = req.temperature.unwrap();
    assert!(!(0.0..=2.0).contains(&t));
}

#[test]
fn max_tokens_zero_flag() {
    let req = parse(r#"{"model":"m","messages":[{"role":"user","content":"x"}],"max_tokens":0}"#)
        .unwrap();
    assert_eq!(req.max_tokens, Some(0));
}

// ── logprobs + echo schema parsing ─────────────────────────────────

/// DoD #4: `logprobs:true` and `echo:true` parse into the typed
/// schema with the expected option-bool shape.
#[test]
fn logprobs_echo_parse() {
    let req = parse(
        r#"{
            "model": "bonsai",
            "messages": [{"role":"user","content":"hello"}],
            "logprobs": true,
            "echo": true,
            "max_tokens": 0
        }"#,
    )
    .expect("schema must parse logprobs+echo");
    assert_eq!(req.logprobs, Some(true));
    assert_eq!(req.echo, Some(true));
    assert_eq!(req.max_tokens, Some(0));
}

/// `echo` is optional and defaults to `None` (absent on the wire).
#[test]
fn echo_absent_is_none() {
    let req = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert_eq!(req.echo, None);
    assert_eq!(req.logprobs, None);
}

/// `echo:false` parses as `Some(false)` (so the handler can
/// distinguish "absent" from "explicitly disabled" if it ever needs to).
#[test]
fn echo_false_parses() {
    let req =
        parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"echo":false}"#).unwrap();
    assert_eq!(req.echo, Some(false));
}

// ── A5.1: tool-calling schema parse ──────────────────────────────────────

/// Full tools + tool_choice=auto payload parses correctly.
#[test]
fn openai_tools_full_parse() {
    let req = parse(
        r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"]
                    }
                }
            }],
            "tool_choice": "auto"
        }"#,
    )
    .unwrap();
    let tools = req.tools.expect("tools must be Some");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].kind, "function");
    assert_eq!(tools[0].function.name, "get_weather");
    assert_eq!(
        tools[0].function.description.as_deref(),
        Some("Get weather for a location")
    );
    let params = &tools[0].function.parameters;
    assert_eq!(params["type"], "object");

    let tc = req.tool_choice.expect("tool_choice must be Some");
    assert!(matches!(tc, ToolChoice::Mode(ref s) if s == "auto"));
}

/// tool_choice="required" deserialises to Mode("required").
#[test]
fn openai_tool_choice_required_parses() {
    let req = parse(
        r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": "required"
        }"#,
    )
    .unwrap();
    let tc = req.tool_choice.expect("tool_choice must be Some");
    assert!(matches!(tc, ToolChoice::Mode(ref s) if s == "required"));
}

/// Named tool_choice object deserialises to NamedToolChoice.
#[test]
fn openai_tool_choice_named_parses() {
    let req = parse(
        r#"{
            "model": "m",
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "function", "function": {"name": "get_weather"}}
        }"#,
    )
    .unwrap();
    let tc = req.tool_choice.expect("tool_choice must be Some");
    match tc {
        ToolChoice::Named(n) => {
            assert_eq!(n.kind, "function");
            assert_eq!(n.function.name, "get_weather");
        }
        other => panic!("expected Named variant, got {other:?}"),
    }
}

/// tools=[] parses to Some([]) (normalisation to None done by handler).
#[test]
fn openai_tools_empty_parses() {
    let req =
        parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"tools":[]}"#).unwrap();
    assert!(req.tools.as_ref().is_none_or(Vec::is_empty));
    assert!(!req.extra.contains_key("tools"));
}

/// tools absent → None.
#[test]
fn openai_tools_absent_is_none() {
    let req = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert!(req.tools.is_none());
    assert!(req.tool_choice.is_none());
}

// ── resolve_sampling_params ───────────────────────────────────────────────

use crate::generation_config_io::GenerationConfig;

/// Convenience wrapper for tests: call resolve with all-None A7.1 / G4 fields.
fn rsp(
    temp: Option<f32>,
    top_p: Option<f32>,
    defaults: Option<&GenerationConfig>,
) -> (
    crate::engine::SamplingParams,
    SamplingSource,
    SamplingSource,
) {
    resolve_sampling_params(
        temp,
        top_p,
        None,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
        defaults,
        None, // server_default_temperature
    )
}

#[test]
fn resolve_uses_request_value_when_present() {
    let defaults = GenerationConfig {
        temperature: Some(0.5),
        top_p: Some(0.8),
        ..Default::default()
    };
    let (r, t_src, p_src) = rsp(Some(0.3), Some(0.6), Some(&defaults));
    assert_eq!(r.temperature, 0.3);
    assert_eq!(r.top_p, 0.6);
    assert_eq!(t_src, SamplingSource::Request);
    assert_eq!(p_src, SamplingSource::Request);
}

#[test]
fn resolve_falls_back_to_model_defaults() {
    let defaults = GenerationConfig {
        temperature: Some(0.7),
        top_p: Some(0.95),
        ..Default::default()
    };
    let (r, t_src, p_src) = rsp(None, None, Some(&defaults));
    assert_eq!(r.temperature, 0.7);
    assert_eq!(r.top_p, 0.95);
    assert_eq!(t_src, SamplingSource::ModelDefaults);
    assert_eq!(p_src, SamplingSource::ModelDefaults);
}

#[test]
fn resolve_falls_back_to_hard_coded_when_no_defaults() {
    let (r, t_src, p_src) = rsp(None, None, None);
    assert_eq!(r.temperature, 1.0);
    assert_eq!(r.top_p, 1.0);
    assert_eq!(t_src, SamplingSource::HardCoded);
    assert_eq!(p_src, SamplingSource::HardCoded);
}

#[test]
fn resolve_mixes_sources_independently() {
    // temperature from request, top_p from model defaults.
    let defaults = GenerationConfig {
        temperature: Some(0.5),
        top_p: Some(0.9),
        ..Default::default()
    };
    let (r, t_src, p_src) = rsp(Some(0.1), None, Some(&defaults));
    assert_eq!(r.temperature, 0.1);
    assert_eq!(r.top_p, 0.9);
    assert_eq!(t_src, SamplingSource::Request);
    assert_eq!(p_src, SamplingSource::ModelDefaults);
}

// ── A7.1: sampling fields parse + validation ──────────────────────────────

/// All six new sampling fields deserialise from a full payload.
#[test]
fn a7_all_sampling_fields_deserialise() {
    let req = parse(
        r#"{
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "top_k": 5,
        "min_p": 0.05,
        "repetition_penalty": 1.2,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.3,
        "logit_bias": {"100": 1.0, "200": -2.5}
    }"#,
    )
    .unwrap();
    assert_eq!(req.top_k, Some(5));
    assert_eq!(req.min_p, Some(0.05));
    assert_eq!(req.repetition_penalty, Some(1.2));
    assert_eq!(req.frequency_penalty, Some(0.5));
    assert_eq!(req.presence_penalty, Some(0.3));
    let lb = req.logit_bias.expect("logit_bias must be Some");
    assert_eq!(lb.len(), 2);
    assert!((lb["100"] - 1.0).abs() < 1e-6);
    assert!((lb["200"] - (-2.5)).abs() < 1e-6);
}

/// Absent sampling fields → SamplingParams carries defaults (greedy-safe).
#[test]
fn a7_absent_fields_give_defaults() {
    let req = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert!(req.top_k.is_none());
    assert!(req.min_p.is_none());
    assert!(req.repetition_penalty.is_none());
    assert!(req.frequency_penalty.is_none());
    assert!(req.presence_penalty.is_none());
    assert!(req.logit_bias.is_none());
    // Verify resolve produces neutral defaults.
    let (sp, _, _) = rsp(None, None, None);
    assert_eq!(sp.top_k, 0);
    assert_eq!(sp.min_p, 0.0);
    assert_eq!(sp.repetition_penalty, 1.0);
    assert_eq!(sp.frequency_penalty, 0.0);
    assert_eq!(sp.presence_penalty, 0.0);
    assert!(sp.logit_bias.is_empty());
    assert!(sp.seed.is_none());
}

/// logit_bias string keys parse to u32 correctly.
#[test]
fn a7_logit_bias_parses_to_u32_pairs() {
    let mut map = HashMap::new();
    map.insert("42".to_owned(), 3.0_f32);
    map.insert("9999".to_owned(), -1.5_f32);
    let mut parsed = parse_logit_bias(Some(&map)).expect("parse must succeed");
    assert_eq!(parsed.len(), 2);
    parsed.sort_by_key(|(id, _)| *id);
    assert_eq!(parsed[0], (42, 3.0));
    assert_eq!(parsed[1], (9999, -1.5));
}

/// Non-integer logit_bias key must return Err (route returns 400).
#[test]
fn a7_logit_bias_non_integer_key_returns_err() {
    let mut map = HashMap::new();
    map.insert("not_an_id".to_owned(), 1.0_f32);
    let result = parse_logit_bias(Some(&map));
    assert!(result.is_err(), "non-integer key must produce Err");
}

/// min_p out of range flags correctly.
#[test]
fn a7_min_p_out_of_range_flag() {
    let req =
        parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"min_p":1.5}"#).unwrap();
    let p = req.min_p.unwrap();
    assert!(!(0.0..=1.0).contains(&p));
}

/// repetition_penalty <= 0 flags correctly.
#[test]
fn a7_repetition_penalty_zero_flag() {
    let req = parse(
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"repetition_penalty":0.0}"#,
    )
    .unwrap();
    let r = req.repetition_penalty.unwrap();
    assert!(r <= 0.0);
}

/// frequency_penalty out of [-2, 2] flags correctly.
#[test]
fn a7_frequency_penalty_out_of_range_flag() {
    let req = parse(
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"frequency_penalty":3.0}"#,
    )
    .unwrap();
    let f = req.frequency_penalty.unwrap();
    assert!(!(-2.0..=2.0).contains(&f));
}

/// presence_penalty out of [-2, 2] flags correctly.
#[test]
fn a7_presence_penalty_out_of_range_flag() {
    let req = parse(
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"presence_penalty":-3.0}"#,
    )
    .unwrap();
    let p = req.presence_penalty.unwrap();
    assert!(!(-2.0..=2.0).contains(&p));
}

/// Resolution: top_k from model defaults when request omits it.
#[test]
fn a7_top_k_falls_back_to_model_defaults() {
    let defaults = GenerationConfig {
        top_k: Some(20),
        ..Default::default()
    };
    let (sp, _, _) = resolve_sampling_params(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
        Some(&defaults),
        None, // server_default_temperature
    );
    assert_eq!(sp.top_k, 20, "top_k must come from model defaults");
}

/// Resolution: request top_k overrides model defaults.
#[test]
fn a7_top_k_request_overrides_model_defaults() {
    let defaults = GenerationConfig {
        top_k: Some(20),
        ..Default::default()
    };
    let (sp, _, _) = resolve_sampling_params(
        None,
        None,
        Some(5),
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
        Some(&defaults),
        None, // server_default_temperature
    );
    assert_eq!(sp.top_k, 5, "request top_k must override model defaults");
}

/// Resolution: repetition_penalty from model defaults when request omits it.
#[test]
fn a7_repetition_penalty_falls_back_to_model_defaults() {
    let defaults = GenerationConfig {
        repetition_penalty: Some(1.1),
        ..Default::default()
    };
    let (sp, _, _) = resolve_sampling_params(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
        Some(&defaults),
        None, // server_default_temperature
    );
    assert!(
        (sp.repetition_penalty - 1.1).abs() < 1e-6,
        "repetition_penalty must come from model defaults"
    );
}

// ── G4: server default temperature ──────────────────────────────────────

/// G4: server_default_temperature wins over model_defaults when request omits
/// temperature.
#[test]
fn g4_server_default_beats_model_defaults() {
    let defaults = GenerationConfig {
        temperature: Some(0.7),
        ..Default::default()
    };
    let (sp, src, _) = resolve_sampling_params(
        None, // no request temperature
        None,
        None,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
        Some(&defaults),
        Some(0.0), // server default
    );
    assert_eq!(
        sp.temperature, 0.0,
        "server_default must beat model_defaults"
    );
    assert_eq!(src, SamplingSource::ServerDefault);
}

/// G4: explicit request temperature beats server_default_temperature.
#[test]
fn g4_request_beats_server_default() {
    let (sp, src, _) = resolve_sampling_params(
        Some(0.9), // explicit request
        None,
        None,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
        None,
        Some(0.0), // server default
    );
    assert_eq!(sp.temperature, 0.9, "request must beat server_default");
    assert_eq!(src, SamplingSource::Request);
}

/// G4: when server_default absent (None), falls through to model_defaults as before.
#[test]
fn g4_absent_server_default_falls_through_to_model_defaults() {
    let defaults = GenerationConfig {
        temperature: Some(0.5),
        ..Default::default()
    };
    let (sp, src, _) = resolve_sampling_params(
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        Vec::new(),
        None,
        Some(&defaults),
        None, // no server default
    );
    assert_eq!(sp.temperature, 0.5);
    assert_eq!(src, SamplingSource::ModelDefaults);
}

// ── A5.4: tool_calls emission ────────────────────────────────────────────

use serde_json::Map as JsonMap;

fn make_parsed_call(name: &str, args: &[(&str, &str)]) -> ParsedToolCall {
    let mut m = JsonMap::new();
    for (k, v) in args {
        m.insert((*k).to_owned(), Value::String((*v).to_owned()));
    }
    ParsedToolCall {
        id: "call_xyz".to_owned(),
        name: name.to_owned(),
        arguments: m,
    }
}

/// `to_response_tool_call` produces the canonical OpenAI wire shape.
/// `arguments` MUST be a JSON-encoded string, not a raw object.
#[test]
fn tool_call_serialises_to_openai_shape() {
    let parsed = make_parsed_call("get_weather", &[("location", "Paris")]);
    let tc = to_response_tool_call(&parsed, 0);
    let v = serde_json::to_value(&tc).unwrap();
    assert_eq!(v["index"], 0);
    assert_eq!(v["id"], "call_xyz");
    assert_eq!(v["type"], "function");
    assert_eq!(v["function"]["name"], "get_weather");
    // `arguments` MUST be a JSON string, not a nested object.
    let args_str = v["function"]["arguments"]
        .as_str()
        .expect("arguments string");
    let args_val: Value = serde_json::from_str(args_str).unwrap();
    assert_eq!(args_val["location"], "Paris");
}

/// ResponseMessage with tool_calls=None must omit the `tool_calls` key.
#[test]
fn response_message_omits_tool_calls_when_none() {
    let m = ResponseMessage {
        role: "assistant".into(),
        content: "hello".into(),
        reasoning_content: None,
        tool_calls: None,
    };
    let v = serde_json::to_value(&m).unwrap();
    assert!(
        v.get("tool_calls").is_none(),
        "tool_calls key must be absent"
    );
    assert!(
        v.get("reasoning_content").is_none(),
        "reasoning_content key must be absent"
    );
}

/// ResponseMessage with tool_calls=Some(...) includes the key.
#[test]
fn response_message_includes_tool_calls_when_some() {
    let parsed = make_parsed_call("foo", &[("x", "1")]);
    let m = ResponseMessage {
        role: "assistant".into(),
        content: String::new(),
        reasoning_content: None,
        tool_calls: Some(vec![to_response_tool_call(&parsed, 0)]),
    };
    let v = serde_json::to_value(&m).unwrap();
    assert!(v.get("tool_calls").is_some());
    let arr = v["tool_calls"].as_array().unwrap();
    assert_eq!(arr.len(), 1);
    assert_eq!(arr[0]["function"]["name"], "foo");
}

/// DeltaContent with tool_calls=None must omit the `tool_calls` key.
#[test]
fn delta_content_omits_tool_calls_when_none() {
    let d = DeltaContent {
        role: None,
        content: Some("hi".to_owned()),
        reasoning_content: None,
        tool_calls: None,
    };
    let v = serde_json::to_value(&d).unwrap();
    assert!(v.get("tool_calls").is_none());
}

/// finish_reason="tool_calls" when any calls were emitted; passthrough otherwise.
#[test]
fn finish_reason_upgrades_to_tool_calls() {
    assert_eq!(
        select_finish_reason(true, Some("stop".to_owned())),
        Some("tool_calls".to_owned())
    );
    assert_eq!(
        select_finish_reason(true, Some("length".to_owned())),
        Some("tool_calls".to_owned())
    );
    assert_eq!(
        select_finish_reason(true, None),
        Some("tool_calls".to_owned())
    );
    // No tool calls → passes terminal reason through unchanged.
    assert_eq!(
        select_finish_reason(false, Some("stop".to_owned())),
        Some("stop".to_owned())
    );
    assert_eq!(
        select_finish_reason(false, Some("length".to_owned())),
        Some("length".to_owned())
    );
    assert_eq!(select_finish_reason(false, None), None);
}

/// Index increments per emitted call (multi-call response).
#[test]
fn multi_tool_call_indexes_are_monotonic() {
    let p0 = make_parsed_call("get_weather", &[("city", "Paris")]);
    let p1 = make_parsed_call("get_time", &[("tz", "UTC")]);
    let calls: Vec<ToolCall> = [&p0, &p1]
        .iter()
        .enumerate()
        .map(|(i, p)| to_response_tool_call(p, i as u32))
        .collect();
    assert_eq!(calls[0].index, 0);
    assert_eq!(calls[1].index, 1);
}

// ── A5.4: streaming-event helper coverage ────────────────────────────────

/// Drive `handle_streaming_token` over a scripted token sequence containing
/// a complete `<tool_call>` and assert: passthrough chunk first, then a
/// `tool_calls` delta chunk, then the terminal chunk with
/// `finish_reason="tool_calls"`.
#[test]
fn streaming_emits_tool_calls_delta_and_upgrades_finish() {
    let mut state = StreamState {
        parser: Some(ToolCallStreamParser::new(ToolCallFormat::Qwen3XmlFunction)),
        next_tool_index: 0,
        any_tool_calls: false,
        id: "chatcmpl-1".to_owned(),
        model: "qwen".to_owned(),
        created: 0,
        json_object_mode: false,
        json_fence_buf: String::new(),
        json_fence_buf_done: false,
        prompt_tokens: 0,
        completion_tokens: 0,
        include_usage: false,
        metrics_drainer: None,
        metrics_model_id: String::new(),
        metrics_ctx_max: 1,
        lifetime_tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        lifetime_requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        bare_json_tool_call_mode: false,
        bare_json_accum: String::new(),
        // Adaptive controller disabled in tests.
        admission_ctrl: None,
        // Inert stop matcher (no stop strings) in these tests.
        stop_matcher: crate::stop_matcher::StopMatcher::new(&[]),
        stop_hit: false,
    };

    // Script: "Let me check.\n" + full <tool_call> + done sentinel.
    let pieces: Vec<&str> = vec![
        "Let me check.\n",
        "<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>",
    ];

    for (i, piece) in pieces.iter().enumerate() {
        let tok = GenerationToken {
            token_id: i as u32,
            piece: piece.to_string(),
            done: false,
            finish_reason: None,
            is_thinking: false,
            logprobs: None,
        };
        let events = handle_streaming_token(Ok(tok), &mut state);
        // Drain — the semantic assertions below read from `state` which
        // tracks emitted-tool-call count and the upgrade flag.
        for ev in events {
            let _ = ev.unwrap();
        }
    }

    // After the two real pieces, the parser should have emitted at least
    // one tool_call (state.any_tool_calls=true, next_tool_index=1).
    assert!(
        state.any_tool_calls,
        "expected at least one tool_call to be emitted"
    );
    assert_eq!(state.next_tool_index, 1);

    // Terminal done token — should drain nothing additional but emit the
    // finish chunk with "tool_calls".
    let done_tok = GenerationToken {
        token_id: 99,
        piece: String::new(),
        done: true,
        finish_reason: Some("stop".to_owned()),
        is_thinking: false,
        logprobs: None,
    };
    let final_events = handle_streaming_token(Ok(done_tok), &mut state);
    // At minimum the terminal chunk is present.
    assert!(
        !final_events.is_empty(),
        "must emit at least the finish chunk"
    );
    // Verify the last chunk's finish_reason via JSON inspection of the
    // event data string. The Event type wraps a string under the hood;
    // grab it via Debug rendering.
    let last_event = final_events.last().unwrap().as_ref().unwrap();
    let debug = format!("{last_event:?}");
    assert!(
        debug.contains("tool_calls"),
        "final event must carry finish_reason=tool_calls, got debug: {debug}"
    );
}

// ── A6.1: response_format parsing ─────────────────────────────────────────

#[test]
fn response_format_json_object_deserialises() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"json_object"}}"#;
    let req: ChatCompletionsRequest = serde_json::from_str(body).unwrap();
    assert!(
        matches!(req.response_format, Some(ResponseFormat::JsonObject)),
        "expected JsonObject, got {:?}",
        req.response_format
    );
}

#[test]
fn response_format_json_schema_deserialises() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],
      "response_format":{"type":"json_schema",
        "json_schema":{"name":"weather","strict":true,"schema":{"type":"object"}}
      }}"#;
    let req: ChatCompletionsRequest = serde_json::from_str(body).unwrap();
    match req.response_format {
        Some(ResponseFormat::JsonSchema { json_schema }) => {
            assert_eq!(json_schema.name, "weather");
            assert!(json_schema.strict);
            assert_eq!(json_schema.schema["type"], "object");
        }
        other => panic!("expected JsonSchema, got {other:?}"),
    }
}

#[test]
fn response_format_text_deserialises() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"text"}}"#;
    let req: ChatCompletionsRequest = serde_json::from_str(body).unwrap();
    assert!(
        matches!(req.response_format, Some(ResponseFormat::Text)),
        "expected Text, got {:?}",
        req.response_format
    );
}

#[test]
fn response_format_absent_is_none() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#;
    let req: ChatCompletionsRequest = serde_json::from_str(body).unwrap();
    assert!(
        req.response_format.is_none(),
        "expected None for absent field"
    );
}

#[test]
fn response_format_invalid_type_is_serde_error() {
    let body = r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"response_format":{"type":"xml"}}"#;
    let result: Result<ChatCompletionsRequest, _> = serde_json::from_str(body);
    assert!(
        result.is_err(),
        "unknown response_format type must produce a serde error"
    );
}

/// Without a parser (tools disabled), `handle_streaming_token` produces
/// exactly one content chunk per non-empty piece plus a terminal chunk
/// — matching the pre-A5.4 behaviour.
#[test]
fn streaming_without_parser_passes_through_unchanged() {
    let mut state = StreamState {
        parser: None,
        next_tool_index: 0,
        any_tool_calls: false,
        id: "chatcmpl-x".to_owned(),
        model: "m".to_owned(),
        created: 0,
        json_object_mode: false,
        json_fence_buf: String::new(),
        json_fence_buf_done: false,
        prompt_tokens: 0,
        completion_tokens: 0,
        include_usage: false,
        metrics_drainer: None,
        metrics_model_id: String::new(),
        metrics_ctx_max: 1,
        lifetime_tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        lifetime_requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        bare_json_tool_call_mode: false,
        bare_json_accum: String::new(),
        // Adaptive controller disabled in tests.
        admission_ctrl: None,
        // Inert stop matcher (no stop strings) in these tests.
        stop_matcher: crate::stop_matcher::StopMatcher::new(&[]),
        stop_hit: false,
    };
    let tok = GenerationToken {
        token_id: 0,
        piece: "hello world".to_owned(),
        done: false,
        finish_reason: None,
        is_thinking: false,
        logprobs: None,
    };
    let events = handle_streaming_token(Ok(tok), &mut state);
    assert_eq!(events.len(), 1, "one content event expected");
    assert!(!state.any_tool_calls);

    let done = GenerationToken {
        token_id: 1,
        piece: String::new(),
        done: true,
        finish_reason: Some("stop".to_owned()),
        is_thinking: false,
        logprobs: None,
    };
    let events = handle_streaming_token(Ok(done), &mut state);
    assert_eq!(events.len(), 1, "one terminal event expected");
    let debug = format!("{:?}", events[0].as_ref().unwrap());
    assert!(
        debug.contains("stop"),
        "terminal finish_reason must remain 'stop' when no tool_calls, got: {debug}"
    );
    assert!(
        !debug.contains("tool_calls"),
        "terminal must NOT mention tool_calls, got: {debug}"
    );
}

// ── A8: compute_effective_timeout unit tests ──────────────────────────────

fn headers_with(key: &str, val: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(
        axum::http::HeaderName::from_bytes(key.as_bytes()).unwrap(),
        HeaderValue::from_str(val).unwrap(),
    );
    h
}

#[test]
fn timeout_no_header_returns_cap() {
    let h = HeaderMap::new();
    let result = compute_effective_timeout(&h, 600).unwrap();
    assert_eq!(result, Some(std::time::Duration::from_secs(600)));
}

#[test]
fn timeout_header_below_cap_uses_header() {
    let h = headers_with("x-request-timeout-seconds", "30");
    let result = compute_effective_timeout(&h, 600).unwrap();
    assert_eq!(result, Some(std::time::Duration::from_secs(30)));
}

#[test]
fn timeout_header_above_cap_clamps_to_cap() {
    let h = headers_with("x-request-timeout-seconds", "9999");
    let result = compute_effective_timeout(&h, 600).unwrap();
    assert_eq!(result, Some(std::time::Duration::from_secs(600)));
}

#[test]
fn timeout_header_non_numeric_returns_400() {
    let h = headers_with("x-request-timeout-seconds", "abc");
    let err = compute_effective_timeout(&h, 600).unwrap_err();
    let resp: Response = err;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn timeout_header_zero_returns_400() {
    let h = headers_with("x-request-timeout-seconds", "0");
    let err = compute_effective_timeout(&h, 600).unwrap_err();
    let resp: Response = err;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn timeout_header_negative_returns_400() {
    // "-5" doesn't parse as u64, so it hits the Err branch.
    let h = headers_with("x-request-timeout-seconds", "-5");
    let err = compute_effective_timeout(&h, 600).unwrap_err();
    let resp: Response = err;
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[test]
fn timeout_max_zero_disables_timeout() {
    // max_secs == 0 always returns None (disabled), regardless of header.
    let h = HeaderMap::new();
    let result = compute_effective_timeout(&h, 0).unwrap();
    assert_eq!(result, None);

    let h2 = headers_with("x-request-timeout-seconds", "30");
    let result2 = compute_effective_timeout(&h2, 0).unwrap();
    assert_eq!(result2, None);
}

// ── C4: multi-model slot-vec semantics ───────────────────────────────────

use crate::engine::NotReadyGenerator;

/// Build a tempdir-backed registry with `n` minimal model snapshots
/// (`m0`..`m{n-1}`), each just a `config.json`. The stub loader ignores
/// the path so the snapshots only need to satisfy `load_config`. Returns
/// the registry plus the live `TempDir` (must stay alive for the test).
fn n_model_registry(n: usize) -> (ModelRegistry, tempfile::TempDir) {
    use std::io::Write as _;
    let root = tempfile::tempdir().unwrap();
    let mut paths = Vec::new();
    for i in 0..n {
        let dir = root.path().join(format!("m{i}"));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "dtype": "bfloat16"
        });
        let mut f = std::fs::File::create(dir.join("config.json")).unwrap();
        f.write_all(cfg.to_string().as_bytes()).unwrap();
        paths.push(dir);
    }
    (ModelRegistry::from_paths(&paths), root)
}

/// `AppState` with a `NotReadyGenerator` stub loader and the given
/// `max_loaded_models`. Generation would 503, but the slot-vec
/// load/evict/find logic in `ensure_loaded`/`unload` is fully exercised.
fn slot_test_state(registry: ModelRegistry, max_loaded_models: usize) -> AppState {
    let loader: ModelLoader =
        Arc::new(|_path, _id| Ok(Box::new(NotReadyGenerator) as Box<dyn Generator>));
    AppState {
        registry: Arc::new(registry),
        slots: Arc::new(PLRwLock::new(Vec::new())),
        embed_slot: Arc::new(PLRwLock::new(None)),
        mm_cache: Arc::new(rmlx_models::multimodal_cache::MultimodalCache::new(0)),
        gpu_gate: Arc::new(PLMutex::new(())),
        gpu_queue: Arc::new(tokio::sync::Semaphore::new(1)),
        gpu_pending: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
        max_queue_depth: 64,
        max_loaded_models,
        loader,
        metrics: None,
        idle_policy: crate::KeepAlivePolicy::Pin,
        max_tokens_cap: u32::MAX,
        max_timeout_secs: 600,
        session_cache: Arc::new(PLMutex::new(SessionCache::new(4))),
        ttft_store: TtftStore::default(),
        itl_store: ItlStore::default(),
        metrics_drainer: None,
        require_smoke_probe: false,
        default_temperature: None,
        default_enable_thinking: None,
        default_image_max_tokens: None,
        tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        started_at: Instant::now(),
        requests_started: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        // Adaptive controller disabled in tests.
        admission_controller: None,
        admission_handle: None,
        whisper_model_path: None,
        whisper_tokenizer_path: None,
        audio_model: Arc::new(PLRwLock::new(None)),
        tts_model_path: None,
        tts_tokenizer_path: None,
        tts_model: Arc::new(parking_lot::RwLock::new(None)),
    }
}

fn resident_ids(state: &AppState) -> Vec<String> {
    state.slots.read().iter().map(|m| m.id.clone()).collect()
}

/// Insert N models under a cap of N: all stay resident, find-by-id works.
#[test]
fn slots_insert_n_all_resident_and_findable() {
    let (reg, _tmp) = n_model_registry(3);
    let state = slot_test_state(reg, 3);

    for id in ["m0", "m1", "m2"] {
        state.ensure_loaded(id).expect("load must succeed");
    }
    let ids = resident_ids(&state);
    assert_eq!(ids.len(), 3, "all 3 models must be resident at cap 3");
    for id in ["m0", "m1", "m2"] {
        assert!(ids.contains(&id.to_owned()), "{id} must be findable");
    }
    // Re-requesting an already-resident model must NOT grow the Vec.
    state.ensure_loaded("m1").unwrap();
    assert_eq!(
        state.slots.read().len(),
        3,
        "re-loading a resident model must not add a duplicate slot"
    );
}

/// At a cap of 2, loading a 3rd model evicts the least-recently-used one.
#[test]
fn slots_lru_eviction_at_cap() {
    let (reg, _tmp) = n_model_registry(3);
    let state = slot_test_state(reg, 2);

    state.ensure_loaded("m0").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    state.ensure_loaded("m1").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    // Touch m0 so m1 becomes the LRU entry.
    state.ensure_loaded("m0").unwrap();
    std::thread::sleep(std::time::Duration::from_millis(5));
    // Load m2 — at cap 2, the LRU (m1) must be evicted.
    state.ensure_loaded("m2").unwrap();

    let ids = resident_ids(&state);
    assert_eq!(ids.len(), 2, "cap 2 must hold exactly 2 models");
    assert!(ids.contains(&"m0".to_owned()), "m0 (recently used) kept");
    assert!(ids.contains(&"m2".to_owned()), "m2 (just loaded) kept");
    assert!(!ids.contains(&"m1".to_owned()), "m1 (LRU) must be evicted");
}

/// `unload` removes exactly one matching entry and leaves the rest.
#[test]
fn slots_unload_removes_one() {
    let (reg, _tmp) = n_model_registry(3);
    let state = slot_test_state(reg, 3);

    for id in ["m0", "m1", "m2"] {
        state.ensure_loaded(id).unwrap();
    }
    assert!(
        state.unload("m1"),
        "unloading a resident model returns true"
    );
    let ids = resident_ids(&state);
    assert_eq!(ids.len(), 2, "exactly one entry removed");
    assert!(!ids.contains(&"m1".to_owned()), "m1 must be gone");
    assert!(ids.contains(&"m0".to_owned()) && ids.contains(&"m2".to_owned()));
    assert!(
        !state.unload("m1"),
        "second unload of the same id returns false"
    );
}

/// `ensure_loaded` returns `(gen, true)` (cold) on first load and
/// `(gen, false)` (warm) on a subsequent request for the already-resident model.
#[test]
fn ensure_loaded_cold_flag() {
    let (reg, _tmp) = n_model_registry(1);
    let state = slot_test_state(reg, 1);

    let (_, is_cold) = state.ensure_loaded("m0").expect("first load must succeed");
    assert!(is_cold, "first load must report cold=true");

    let (_, is_warm) = state
        .ensure_loaded("m0")
        .expect("second request must succeed");
    assert!(
        !is_warm,
        "re-request of resident model must report cold=false (warm)"
    );
}

/// max_loaded_models == 1 parity: loading a different model evicts the
/// single existing entry (byte-equivalent to the old swap-on-different-id).
#[test]
fn slots_max_one_evicts_single_entry_parity() {
    let (reg, _tmp) = n_model_registry(2);
    let state = slot_test_state(reg, 1);

    state.ensure_loaded("m0").unwrap();
    assert_eq!(resident_ids(&state), vec!["m0".to_owned()]);

    // Loading a different model with cap 1 must evict the only entry and
    // leave exactly the new one — identical to the pre-C4 swap path.
    state.ensure_loaded("m1").unwrap();
    assert_eq!(
        resident_ids(&state),
        vec!["m1".to_owned()],
        "cap 1: the single existing entry is the LRU and is swapped out"
    );

    // Re-requesting the resident model at cap 1 must not evict/reload.
    state.ensure_loaded("m1").unwrap();
    assert_eq!(resident_ids(&state), vec!["m1".to_owned()]);
}

// ── H3/H4: usage accounting in streaming path ────────────────────────────

/// Build a minimal `StreamState` with the given usage-tracking fields.
fn usage_stream_state(prompt_tokens: u32, include_usage: bool) -> StreamState {
    StreamState {
        parser: None,
        next_tool_index: 0,
        any_tool_calls: false,
        id: "chatcmpl-test".to_owned(),
        model: "m".to_owned(),
        created: 0,
        json_object_mode: false,
        json_fence_buf: String::new(),
        json_fence_buf_done: false,
        prompt_tokens,
        completion_tokens: 0,
        include_usage,
        metrics_drainer: None,
        metrics_model_id: String::new(),
        metrics_ctx_max: 1,
        lifetime_tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        lifetime_requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        bare_json_tool_call_mode: false,
        bare_json_accum: String::new(),
        // Adaptive controller disabled in tests.
        admission_ctrl: None,
        // Inert stop matcher (no stop strings) in these tests.
        stop_matcher: crate::stop_matcher::StopMatcher::new(&[]),
        stop_hit: false,
    }
}

/// H3 proxy: feed N non-done tokens + 1 done token through
/// `handle_streaming_token` and verify `state.completion_tokens` is exact.
#[test]
fn h3_completion_tokens_counter_exact() {
    let mut state = usage_stream_state(10, false);

    // 3 non-done tokens.
    for i in 0..3u32 {
        let tok = GenerationToken {
            token_id: i,
            piece: format!("tok{i}"),
            done: false,
            finish_reason: None,
            is_thinking: false,
            logprobs: None,
        };
        handle_streaming_token(Ok(tok), &mut state);
    }
    assert_eq!(
        state.completion_tokens, 3,
        "completion_tokens must equal number of non-done tokens fed"
    );

    // Done token (the 4th token).
    let done = GenerationToken {
        token_id: 3,
        piece: String::new(),
        done: true,
        finish_reason: Some("stop".to_owned()),
        is_thinking: false,
        logprobs: None,
    };
    handle_streaming_token(Ok(done), &mut state);
    assert_eq!(
        state.completion_tokens, 4,
        "done token must also increment the counter"
    );
}

/// H3 proxy: completion_tokens accumulates correctly with
/// `max_tokens` = 2 (only 2 non-done tokens + 1 done).
#[test]
fn h3_completion_tokens_two_tokens() {
    let mut state = usage_stream_state(7, false);
    for i in 0..2u32 {
        handle_streaming_token(
            Ok(GenerationToken {
                token_id: i,
                piece: format!("x{i}"),
                done: false,
                finish_reason: None,
                is_thinking: false,
                logprobs: None,
            }),
            &mut state,
        );
    }
    handle_streaming_token(
        Ok(GenerationToken {
            token_id: 2,
            piece: String::new(),
            done: true,
            finish_reason: Some("length".to_owned()),
            is_thinking: false,
            logprobs: None,
        }),
        &mut state,
    );
    assert_eq!(state.completion_tokens, 3);
}

/// H4: when `include_usage = true`, the `done` token produces TWO SSE
/// events: the finish chunk (with `finish_reason`) followed by a usage
/// summary chunk (with `choices: []` and a populated `usage`).
#[test]
fn h4_usage_chunk_emitted_when_include_usage_true() {
    let prompt_tokens: u32 = 10;
    let mut state = usage_stream_state(prompt_tokens, true);

    // Two non-done tokens.
    for i in 0..2u32 {
        handle_streaming_token(
            Ok(GenerationToken {
                token_id: i,
                piece: format!("w{i}"),
                done: false,
                finish_reason: None,
                is_thinking: false,
                logprobs: None,
            }),
            &mut state,
        );
    }

    // Done token.
    let events = handle_streaming_token(
        Ok(GenerationToken {
            token_id: 2,
            piece: String::new(),
            done: true,
            finish_reason: Some("stop".to_owned()),
            is_thinking: false,
            logprobs: None,
        }),
        &mut state,
    );

    // Expect exactly 2 events: finish-reason chunk + usage chunk.
    assert_eq!(
        events.len(),
        2,
        "done token with include_usage=true must emit 2 events (finish + usage)"
    );

    // Last event must be the usage chunk.
    // Verify by serialising the expected chunk independently and
    // comparing against the event's data bytes via the debug repr.
    // The debug repr escapes inner quotes as `\"`, so we check for the
    // escaped form of each expected substring.
    let expected_completion: u32 = 3;
    let expected_total = prompt_tokens + expected_completion;
    // Build the canonical expected chunk (same code path as production).
    let expected_usage_chunk = ChatCompletionChunk {
        id: "chatcmpl-test".to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created: 0,
        model: "m".to_owned(),
        choices: vec![],
        usage: Some(Usage {
            prompt_tokens,
            completion_tokens: expected_completion,
            total_tokens: expected_total,
        }),
    };
    let expected_json = serde_json::to_string(&expected_usage_chunk).unwrap();
    // The event debug repr embeds the data bytes with `\"` escaping.
    // Escape expected_json the same way for comparison.
    let expected_escaped = expected_json.replace('"', "\\\"");
    let ev_dbg = format!("{:?}", events[1].as_ref().unwrap());
    assert!(
        ev_dbg.contains(&expected_escaped),
        "usage chunk must match expected JSON.\nExpected (escaped): {expected_escaped}\nGot: {ev_dbg}"
    );
}

/// H4: when `include_usage = false`, the `done` token produces exactly ONE
/// SSE event (the finish chunk), and that event must NOT contain `usage`.
#[test]
fn h4_no_usage_chunk_when_include_usage_false() {
    let mut state = usage_stream_state(5, false);

    // One non-done token.
    handle_streaming_token(
        Ok(GenerationToken {
            token_id: 0,
            piece: "hi".to_owned(),
            done: false,
            finish_reason: None,
            is_thinking: false,
            logprobs: None,
        }),
        &mut state,
    );

    // Done token.
    let events = handle_streaming_token(
        Ok(GenerationToken {
            token_id: 1,
            piece: String::new(),
            done: true,
            finish_reason: Some("stop".to_owned()),
            is_thinking: false,
            logprobs: None,
        }),
        &mut state,
    );

    assert_eq!(
        events.len(),
        1,
        "done token with include_usage=false must emit exactly 1 event"
    );

    // The single event must NOT contain a usage field.
    // Use the debug repr — byte buffer will not contain the word "usage"
    // at all when the chunk has no usage object.
    let dbg = format!("{:?}", events[0].as_ref().unwrap());
    assert!(
        !dbg.contains("usage"),
        "event must NOT contain 'usage' when include_usage=false, got: {dbg}"
    );
}

/// H4: `stream_options` field on `ChatCompletionsRequest` parses correctly
/// for both present-and-true and absent cases.
#[test]
fn h4_stream_options_deserialises() {
    // With include_usage = true.
    let req = parse(
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":true}}"#,
    )
    .unwrap();
    assert!(
        req.stream_options.as_ref().is_some_and(|o| o.include_usage),
        "stream_options.include_usage must be true when set"
    );
    // stream_options must NOT land in extra.
    assert!(
        !req.extra.contains_key("stream_options"),
        "stream_options must not appear in extra catch-all"
    );

    // With include_usage = false (default).
    let req2 = parse(
        r#"{"model":"m","messages":[{"role":"user","content":"hi"}],"stream":true,"stream_options":{"include_usage":false}}"#,
    )
    .unwrap();
    assert!(
        !req2
            .stream_options
            .as_ref()
            .map_or(true, |o| o.include_usage),
        "include_usage must be false when explicitly set to false"
    );

    // Absent stream_options → None.
    let req3 = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#).unwrap();
    assert!(
        req3.stream_options.is_none(),
        "stream_options must be None when absent"
    );
}

/// H4: Usage serialisation — when `usage` is None, the field is absent
/// from the JSON (not `null`) because of `skip_serializing_if`.
#[test]
fn h4_chunk_usage_absent_when_none() {
    let chunk = ChatCompletionChunk {
        id: "id".to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created: 0,
        model: "m".to_owned(),
        choices: vec![],
        usage: None,
    };
    let json = serde_json::to_string(&chunk).unwrap();
    assert!(
        !json.contains("usage"),
        "usage must be absent (not null) when None, got: {json}"
    );
}

/// H4: Usage serialisation — when `usage` is Some, the JSON contains
/// the triple with exact values.
#[test]
fn h4_chunk_usage_present_when_some() {
    let chunk = ChatCompletionChunk {
        id: "id".to_owned(),
        object: "chat.completion.chunk".to_owned(),
        created: 0,
        model: "m".to_owned(),
        choices: vec![],
        usage: Some(Usage {
            prompt_tokens: 5,
            completion_tokens: 3,
            total_tokens: 8,
        }),
    };
    let json = serde_json::to_string(&chunk).unwrap();
    let v: Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["usage"]["prompt_tokens"], 5);
    assert_eq!(v["usage"]["completion_tokens"], 3);
    assert_eq!(v["usage"]["total_tokens"], 8);
    assert_eq!(v["choices"].as_array().unwrap().len(), 0);
}

// ── F14: lifetime token counters ─────────────────────────────────────────

/// F14: `tokens_in` / `tokens_out` on AppState start at zero, and the
/// streaming path increments them at the done-token boundary via
/// `handle_streaming_token`.
///
/// Drives `handle_streaming_token` directly (same path as H3/H4) and
/// verifies the shared Arc counters are incremented exactly once per request.
#[test]
fn f14_lifetime_counters_incremented_at_done_boundary() {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    let tokens_in = Arc::new(AtomicU64::new(0));
    let tokens_out = Arc::new(AtomicU64::new(0));

    // Build a StreamState with the real Arcs (prompt_tokens=7).
    let mut state = StreamState {
        parser: None,
        next_tool_index: 0,
        any_tool_calls: false,
        id: "chatcmpl-f14".to_owned(),
        model: "m".to_owned(),
        created: 0,
        json_object_mode: false,
        json_fence_buf: String::new(),
        json_fence_buf_done: false,
        prompt_tokens: 7,
        completion_tokens: 0,
        include_usage: false,
        metrics_drainer: None,
        metrics_model_id: String::new(),
        metrics_ctx_max: 1,
        lifetime_tokens_in: Arc::clone(&tokens_in),
        lifetime_tokens_out: Arc::clone(&tokens_out),
        error_counts: ApiErrorCounters::new(),
        lifetime_requests_completed: Arc::new(AtomicU64::new(0)),
        lifetime_requests_failed: Arc::new(AtomicU64::new(0)),
        bare_json_tool_call_mode: false,
        bare_json_accum: String::new(),
        // Adaptive controller disabled in tests.
        admission_ctrl: None,
        // Inert stop matcher (no stop strings) in these tests.
        stop_matcher: crate::stop_matcher::StopMatcher::new(&[]),
        stop_hit: false,
    };

    // 3 non-done tokens — counters must not change yet.
    for i in 0..3u32 {
        handle_streaming_token(
            Ok(GenerationToken {
                token_id: i,
                piece: format!("x{i}"),
                done: false,
                finish_reason: None,
                is_thinking: false,
                logprobs: None,
            }),
            &mut state,
        );
    }
    assert_eq!(
        tokens_in.load(Ordering::Relaxed),
        0,
        "tokens_in before done"
    );
    assert_eq!(
        tokens_out.load(Ordering::Relaxed),
        0,
        "tokens_out before done"
    );

    // Done token — now counters must be updated.
    handle_streaming_token(
        Ok(GenerationToken {
            token_id: 3,
            piece: String::new(),
            done: true,
            finish_reason: Some("stop".to_owned()),
            is_thinking: false,
            logprobs: None,
        }),
        &mut state,
    );

    // prompt_tokens=7, completion_tokens=4 (3 non-done + 1 done).
    assert_eq!(tokens_in.load(Ordering::Relaxed), 7, "tokens_in after done");
    assert_eq!(
        tokens_out.load(Ordering::Relaxed),
        4,
        "tokens_out after done"
    );
}

/// F14: `metrics_cache` handler response body includes `tokens_in` and
/// `tokens_out` fields with the correct accumulated values.
///
/// Calls the handler directly using axum's `axum::extract::State` extractor
/// and inspects the returned JSON body.
#[tokio::test]
async fn f14_metrics_cache_has_tokens_in_and_tokens_out() {
    let (reg, _tmp) = n_model_registry(1);
    let state = slot_test_state(reg, 1);

    // Seed the counters directly — no real request needed.
    state
        .tokens_in
        .fetch_add(42, std::sync::atomic::Ordering::Relaxed);
    state
        .tokens_out
        .fetch_add(17, std::sync::atomic::Ordering::Relaxed);

    // Call the handler directly.
    let response = metrics_cache(State(state)).await;
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body_bytes).unwrap();
    assert_eq!(v["tokens_in"], 42, "tokens_in must be present and correct");
    assert_eq!(
        v["tokens_out"], 17,
        "tokens_out must be present and correct"
    );
}

/// `effective_max_ctx_for` returns the resident entry's value, else MAX.
#[test]
fn effective_max_ctx_for_resident_and_absent() {
    let (reg, _tmp) = n_model_registry(1);
    let state = slot_test_state(reg, 1);

    assert_eq!(
        state.effective_max_ctx_for("m0"),
        usize::MAX,
        "absent model → usize::MAX (503 path catches real overflow)"
    );
    state.ensure_loaded("m0").unwrap();
    // NotReadyGenerator's default effective_max_ctx is usize::MAX, so the
    // resident lookup also yields MAX here — what matters is that the
    // resident branch is taken (no panic, found by id).
    assert_eq!(state.effective_max_ctx_for("m0"), usize::MAX);
    assert_eq!(
        state.effective_max_ctx_for("nope"),
        usize::MAX,
        "unknown id → usize::MAX"
    );
}

// ── F10: resolve_request_id unit tests ───────────────────────────────────

/// F10: absent header → a fresh generated id is returned (non-empty, starts
/// with "req-").
#[test]
fn f10_absent_header_generates_id() {
    let h = HeaderMap::new();
    let rid = resolve_request_id(&h);
    assert!(
        !rid.is_empty(),
        "generated id must not be empty when header absent"
    );
    assert!(
        rid.starts_with("req-"),
        "generated id must start with 'req-', got: {rid}"
    );
}

/// F10: two calls without a header return distinct ids (probabilistic — two
/// UUID v4s are distinct with overwhelming probability).
#[test]
fn f10_absent_header_generates_unique_ids() {
    let h = HeaderMap::new();
    let a = resolve_request_id(&h);
    let b = resolve_request_id(&h);
    assert_ne!(a, b, "two generated ids must be distinct");
}

/// F10: present `X-Request-Id` header is echoed verbatim.
#[test]
fn f10_inbound_header_is_echoed() {
    let h = headers_with("x-request-id", "test-corr-123");
    let rid = resolve_request_id(&h);
    assert_eq!(rid, "test-corr-123");
}

/// F10: header value longer than 128 chars is truncated to exactly 128.
#[test]
fn f10_header_capped_at_128_chars() {
    let long: String = "a".repeat(200);
    let h = headers_with("x-request-id", &long);
    let rid = resolve_request_id(&h);
    assert_eq!(rid.len(), 128, "must be capped at 128 chars");
    assert!(rid.chars().all(|c| c == 'a'));
}

/// F10: non-ASCII bytes in the header value are skipped (sanitizer retains
/// only printable ASCII). The http crate rejects genuine control bytes at
/// header-value construction, so this tests the high-bit path using a
/// `to_str()` that would fail, which resolves_request_id handles by falling
/// back to a generated id.
///
/// Concretely: a high-byte (0x80+) header value cannot be formed via
/// `HeaderValue::from_str` (UTF-8 only) but CAN be formed via `from_bytes`.
/// `to_str()` on it fails — so `resolve_request_id` falls back to generated.
#[test]
fn f10_non_utf8_header_falls_back_to_generated() {
    // 0x80 is valid in ISO-8859-1 but not UTF-8; from_bytes accepts it.
    let raw = b"abc\x80def";
    let mut h = HeaderMap::new();
    h.insert(
        axum::http::HeaderName::from_static("x-request-id"),
        HeaderValue::from_bytes(raw).unwrap(),
    );
    // to_str() will fail → falls back to generated id.
    let rid = resolve_request_id(&h);
    assert!(
        rid.starts_with("req-"),
        "non-UTF-8 header must fall back to generated id, got: {rid}"
    );
}

/// F10: whitespace-only header value falls back to generating an id.
#[test]
fn f10_whitespace_only_falls_back_to_generated() {
    let h = headers_with("x-request-id", "   ");
    let rid = resolve_request_id(&h);
    assert!(
        rid.starts_with("req-"),
        "whitespace-only header must fall back to generated id"
    );
}

// ── F8: API error-category lifetime counters ─────────────────────────────

/// F8: each `ApiErrorCategory` variant increments exactly its own counter
/// and no other.
#[test]
fn f8_each_category_increments_only_its_counter() {
    use std::sync::atomic::Ordering;
    let c = ApiErrorCounters::new();

    // Baseline: all zeros.
    let snap_zero = c.to_json();
    for cat_key in [
        "bad_request",
        "context_overflow",
        "not_found",
        "oom_load",
        "oom_kv_cache",
        "oom_mid_stream",
        "timeout",
        "upstream",
        "internal",
        "rate_limit",
        "admission_sla_503",
    ] {
        assert_eq!(snap_zero[cat_key], 0, "counter {cat_key} must start at 0");
    }

    // Increment each category exactly once, then verify all are 1.
    let all_cats = [
        ApiErrorCategory::BadRequest,
        ApiErrorCategory::ContextOverflow,
        ApiErrorCategory::NotFound,
        ApiErrorCategory::OomLoad,
        ApiErrorCategory::OomKvCache,
        ApiErrorCategory::OomMidStream,
        ApiErrorCategory::Timeout,
        ApiErrorCategory::Upstream,
        ApiErrorCategory::Internal,
        ApiErrorCategory::RateLimit,
        ApiErrorCategory::AdmissionSla503,
    ];
    for &cat in &all_cats {
        c.increment(cat);
    }

    // All counters must now be exactly 1 (each incremented once).
    for cat in all_cats {
        assert_eq!(
            snap_zero[cat.as_str()].as_u64().unwrap_or(999),
            0,
            "original snapshot must be untouched (snapshot isolation)"
        );
        let direct = match cat {
            ApiErrorCategory::BadRequest => c.bad_request.load(Ordering::Relaxed),
            ApiErrorCategory::ContextOverflow => c.context_overflow.load(Ordering::Relaxed),
            ApiErrorCategory::NotFound => c.not_found.load(Ordering::Relaxed),
            ApiErrorCategory::OomLoad => c.oom_load.load(Ordering::Relaxed),
            ApiErrorCategory::OomKvCache => c.oom_kv_cache.load(Ordering::Relaxed),
            ApiErrorCategory::OomMidStream => c.oom_mid_stream.load(Ordering::Relaxed),
            ApiErrorCategory::Timeout => c.timeout.load(Ordering::Relaxed),
            ApiErrorCategory::Upstream => c.upstream.load(Ordering::Relaxed),
            ApiErrorCategory::Internal => c.internal.load(Ordering::Relaxed),
            ApiErrorCategory::RateLimit => c.rate_limit.load(Ordering::Relaxed),
            ApiErrorCategory::AdmissionSla503 => c.admission_sla_503.load(Ordering::Relaxed),
        };
        assert_eq!(direct, 1, "category {} must be exactly 1", cat.as_str());
    }

    // Verify to_json snapshot also shows all 1.
    let snap_after = c.to_json();
    for cat in all_cats {
        assert_eq!(
            snap_after[cat.as_str()],
            1u64,
            "to_json {} must be 1",
            cat.as_str()
        );
    }
}

/// F8: a successful (non-error) streaming path does NOT increment any
/// error counter. Drive `handle_streaming_token` with only success tokens.
#[test]
fn f8_successful_streaming_path_increments_no_error_counter() {
    let mut state = StreamState {
        parser: None,
        next_tool_index: 0,
        any_tool_calls: false,
        id: "chatcmpl-f8".to_owned(),
        model: "m".to_owned(),
        created: 0,
        json_object_mode: false,
        json_fence_buf: String::new(),
        json_fence_buf_done: false,
        prompt_tokens: 3,
        completion_tokens: 0,
        include_usage: false,
        metrics_drainer: None,
        metrics_model_id: String::new(),
        metrics_ctx_max: 1,
        lifetime_tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        lifetime_requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        bare_json_tool_call_mode: false,
        bare_json_accum: String::new(),
        // Adaptive controller disabled in tests.
        admission_ctrl: None,
        // Inert stop matcher (no stop strings) in these tests.
        stop_matcher: crate::stop_matcher::StopMatcher::new(&[]),
        stop_hit: false,
    };

    // Success tokens.
    for i in 0..3u32 {
        handle_streaming_token(
            Ok(GenerationToken {
                token_id: i,
                piece: format!("t{i}"),
                done: false,
                finish_reason: None,
                is_thinking: false,
                logprobs: None,
            }),
            &mut state,
        );
    }
    // Done token.
    handle_streaming_token(
        Ok(GenerationToken {
            token_id: 3,
            piece: String::new(),
            done: true,
            finish_reason: Some("stop".to_owned()),
            is_thinking: false,
            logprobs: None,
        }),
        &mut state,
    );

    let snap = state.error_counts.to_json();
    for cat_key in [
        "bad_request",
        "context_overflow",
        "not_found",
        "oom_load",
        "oom_kv_cache",
        "oom_mid_stream",
        "timeout",
        "upstream",
        "internal",
        "rate_limit",
    ] {
        assert_eq!(
            snap[cat_key], 0,
            "error counter {cat_key} must be 0 after a successful stream"
        );
    }
}

/// F8: an engine error token increments the correct category counter in the
/// streaming path and emits a `[DONE]` sentinel.
#[test]
fn f8_engine_error_in_streaming_increments_counter() {
    let mut state = StreamState {
        parser: None,
        next_tool_index: 0,
        any_tool_calls: false,
        id: "chatcmpl-f8err".to_owned(),
        model: "m".to_owned(),
        created: 0,
        json_object_mode: false,
        json_fence_buf: String::new(),
        json_fence_buf_done: false,
        prompt_tokens: 1,
        completion_tokens: 0,
        include_usage: false,
        metrics_drainer: None,
        metrics_model_id: String::new(),
        metrics_ctx_max: 1,
        lifetime_tokens_in: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_tokens_out: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        error_counts: ApiErrorCounters::new(),
        lifetime_requests_completed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        lifetime_requests_failed: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        bare_json_tool_call_mode: false,
        bare_json_accum: String::new(),
        // Adaptive controller disabled in tests.
        admission_ctrl: None,
        // Inert stop matcher (no stop strings) in these tests.
        stop_matcher: crate::stop_matcher::StopMatcher::new(&[]),
        stop_hit: false,
    };

    // Feed a non-OOM engine error (maps to Upstream).
    let err_item: rmlx_core::Result<GenerationToken> =
        Err(rmlx_core::Error::Mlx("device error".to_owned()));
    let events = handle_streaming_token(err_item, &mut state);

    // Must emit exactly one [DONE] sentinel.
    assert_eq!(events.len(), 1, "error token must emit exactly one event");
    let ev_debug = format!("{:?}", events[0].as_ref().unwrap());
    assert!(
        ev_debug.contains("[DONE]"),
        "error event must be [DONE], got: {ev_debug}"
    );

    // Upstream counter must be 1.
    assert_eq!(
        state
            .error_counts
            .upstream
            .load(std::sync::atomic::Ordering::Relaxed),
        1,
        "upstream counter must be 1 after engine error"
    );
    // No other counter bumped.
    let snap = state.error_counts.to_json();
    for key in [
        "bad_request",
        "context_overflow",
        "not_found",
        "oom_load",
        "oom_kv_cache",
        "oom_mid_stream",
        "timeout",
        "internal",
        "rate_limit",
    ] {
        assert_eq!(
            snap[key], 0,
            "counter {key} must be 0, not bumped by engine error"
        );
    }
}

/// F8: `metrics_cache` handler includes `error_counts` object with all
/// category keys present and correct values.
#[tokio::test]
async fn f8_metrics_cache_has_error_counts() {
    let (reg, _tmp) = n_model_registry(1);
    let state = slot_test_state(reg, 1);

    // Seed some error counters directly.
    state
        .error_counts
        .not_found
        .fetch_add(3, std::sync::atomic::Ordering::Relaxed);
    state
        .error_counts
        .bad_request
        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

    let response = metrics_cache(State(state)).await;
    let body_bytes = axum::body::to_bytes(response.into_body(), usize::MAX)
        .await
        .unwrap();
    let v: Value = serde_json::from_slice(&body_bytes).unwrap();

    let ec = &v["error_counts"];
    assert!(ec.is_object(), "error_counts must be an object, got: {ec}");
    assert_eq!(ec["not_found"], 3, "not_found must be 3");
    assert_eq!(ec["bad_request"], 1, "bad_request must be 1");
    // All other categories must be present and zero.
    for key in [
        "context_overflow",
        "oom_load",
        "oom_kv_cache",
        "oom_mid_stream",
        "timeout",
        "upstream",
        "internal",
        "rate_limit",
    ] {
        assert_eq!(ec[key], 0, "counter {key} must be 0");
    }
}

/// F8: a successful request does not bump any error counter end-to-end
/// (verified via `slot_test_state` + direct counter check).
#[test]
fn f8_successful_request_increments_no_counter() {
    let (reg, _tmp) = n_model_registry(1);
    let state = slot_test_state(reg, 1);

    // Confirm all counters start at zero.
    let snap = state.error_counts.to_json();
    for key in [
        "bad_request",
        "context_overflow",
        "not_found",
        "oom_load",
        "oom_kv_cache",
        "oom_mid_stream",
        "timeout",
        "upstream",
        "internal",
        "rate_limit",
    ] {
        assert_eq!(snap[key], 0, "counter {key} must start at 0");
    }
}

// ── F5: Prometheus /metrics endpoint ─────────────────────────────────────

/// Build a `MetricsSnapshot` with known values and verify `render_prometheus`
/// produces valid Prometheus text exposition.
///
/// Checks:
/// - Every metric line is preceded by a `# TYPE` line.
/// - Token counter lines are present with the correct values.
/// - Error counter line with label `category="not_found"` is present.
/// - TTFT gauge lines are emitted (p50/p95/p99).
#[test]
fn f5_render_prometheus_known_snapshot() {
    let snap = MetricsSnapshot {
        models: vec![],
        ttft_samples: vec![
            TtftSample {
                model_id: "m".into(),
                ttft_ms: 100,
            },
            TtftSample {
                model_id: "m".into(),
                ttft_ms: 200,
            },
            TtftSample {
                model_id: "m".into(),
                ttft_ms: 300,
            },
        ],
        itl_samples: vec![ItlSample {
            model_id: "m".into(),
            p50_ms: 12.0,
            p95_ms: 20.0,
            mean_ms: 13.0,
            step_count: 50,
        }],
        tokens_in: 1234,
        tokens_out: 567,
        error_counts: vec![
            ("bad_request", 0),
            ("context_overflow", 0),
            ("not_found", 3),
            ("oom_load", 0),
            ("oom_kv_cache", 0),
            ("oom_mid_stream", 0),
            ("timeout", 0),
            ("upstream", 0),
            ("internal", 0),
            ("rate_limit", 0),
        ],
        proc_mem: None,
        uptime_s: 42.5,
        in_flight: 3,
        requests_started: 10,
        requests_completed: 9,
        requests_failed: 1,
        // step_count=50 >= 2 and mean_ms=13.0 > 0 → avg = 1000/13 ≈ 76.923 tok/s
        avg_decode_tok_s: Some(1000.0 / 13.0),
    };

    let text = render_prometheus(&snap);

    // Must end with newline.
    assert!(text.ends_with('\n'), "exposition must end with newline");

    // Every non-comment, non-empty line must have a preceding # TYPE line.
    let lines: Vec<&str> = text.lines().collect();
    let mut saw_type = false;
    for line in &lines {
        if line.starts_with("# TYPE") {
            saw_type = true;
        } else if line.starts_with("# HELP") {
            // HELP is allowed between TYPE and metric lines; reset flag.
        } else if !line.is_empty() && !line.starts_with('#') {
            assert!(saw_type, "metric line `{line}` not preceded by # TYPE");
        }
    }

    // tokens_in counter.
    assert!(
        text.contains("rmlx_lifetime_tokens_total{direction=\"in\"} 1234"),
        "tokens_in missing or wrong: {text}"
    );
    // tokens_out counter.
    assert!(
        text.contains("rmlx_lifetime_tokens_total{direction=\"out\"} 567"),
        "tokens_out missing or wrong: {text}"
    );

    // Error counter with label.
    assert!(
        text.contains("rmlx_api_errors_total{category=\"not_found\"} 3"),
        "not_found error counter missing: {text}"
    );

    // TTFT gauge lines present.
    assert!(
        text.contains("rmlx_ttft_ms{quantile=\"0.50\"}"),
        "TTFT p50 missing: {text}"
    );
    assert!(
        text.contains("rmlx_ttft_ms{quantile=\"0.95\"}"),
        "TTFT p95 missing: {text}"
    );
    assert!(
        text.contains("rmlx_ttft_ms{quantile=\"0.99\"}"),
        "TTFT p99 missing: {text}"
    );

    // ITL gauge.
    assert!(
        text.contains("rmlx_itl_ms{quantile=\"0.50\"} 12"),
        "ITL p50 missing: {text}"
    );

    // rmlx_uptime_seconds gauge must always be present.
    assert!(
        text.contains("# TYPE rmlx_uptime_seconds gauge"),
        "rmlx_uptime_seconds TYPE missing: {text}"
    );
    assert!(
        text.contains("rmlx_uptime_seconds 42.5"),
        "rmlx_uptime_seconds value missing or wrong: {text}"
    );

    // rmlx_in_flight gauge (DoD name, not rmlx_in_flight_requests).
    assert!(
        text.contains("# TYPE rmlx_in_flight gauge"),
        "rmlx_in_flight TYPE missing: {text}"
    );
    assert!(
        text.contains("rmlx_in_flight 3"),
        "rmlx_in_flight value missing or wrong: {text}"
    );
    // Ensure old name is absent (consolidated to DoD name).
    assert!(
        !text.contains("rmlx_in_flight_requests"),
        "old rmlx_in_flight_requests name must not appear: {text}"
    );

    // rmlx_avg_decode_tok_s gauge (derived from ITL ring).
    assert!(
        text.contains("# TYPE rmlx_avg_decode_tok_s gauge"),
        "rmlx_avg_decode_tok_s TYPE missing: {text}"
    );
    assert!(
        text.contains("rmlx_avg_decode_tok_s 76."),
        "rmlx_avg_decode_tok_s value missing or wrong: {text}"
    );
}

/// Verify `render_prometheus` emits counter TYPE for error metrics
/// and gauge TYPE for memory metrics when proc_mem is populated.
#[test]
fn f5_prometheus_type_annotations() {
    let snap = MetricsSnapshot {
        models: vec![],
        ttft_samples: vec![],
        itl_samples: vec![],
        tokens_in: 0,
        tokens_out: 0,
        error_counts: vec![("not_found", 0)],
        proc_mem: Some(rmlx_core::mach_mem::ProcMem {
            rss_bytes: 1_000_000,
            virtual_bytes: 2_000_000,
            phys_footprint_bytes: 900_000,
            internal_bytes: 800_000,
            compressed_bytes: 0,
            external_bytes: 100_000,
        }),
        uptime_s: 0.0,
        in_flight: 0,
        requests_started: 0,
        requests_completed: 0,
        requests_failed: 0,
        avg_decode_tok_s: None,
    };

    let text = render_prometheus(&snap);

    // counter type for tokens.
    assert!(
        text.contains("# TYPE rmlx_lifetime_tokens_total counter"),
        "tokens TYPE missing: {text}"
    );
    // counter type for errors.
    assert!(
        text.contains("# TYPE rmlx_api_errors_total counter"),
        "errors TYPE missing: {text}"
    );
    // gauge type for process RSS.
    assert!(
        text.contains("# TYPE rmlx_process_rss_bytes gauge"),
        "rss TYPE missing: {text}"
    );
    // process RSS value present.
    assert!(
        text.contains("rmlx_process_rss_bytes 1000000"),
        "rss value missing: {text}"
    );
    // avg_decode_tok_s must be absent when None (no ITL samples).
    assert!(
        !text.contains("rmlx_avg_decode_tok_s"),
        "rmlx_avg_decode_tok_s must not appear when avg_decode_tok_s is None: {text}"
    );
}

/// Verify `gather_metrics` and the JSON snapshot agree on `tokens_in` and
/// the `not_found` error count after incrementing the live atomics.
#[test]
fn f5_gather_metrics_matches_appstate_atomics() {
    use std::sync::atomic::Ordering::Relaxed;
    let (reg, _tmp) = n_model_registry(1);
    let state = slot_test_state(reg, 1);

    // Seed known values.
    state.tokens_in.store(42, Relaxed);
    state.tokens_out.store(17, Relaxed);
    state.error_counts.increment(ApiErrorCategory::NotFound);
    state.error_counts.increment(ApiErrorCategory::NotFound);

    let snap = gather_metrics(&state);

    assert_eq!(snap.tokens_in, 42, "tokens_in mismatch");
    assert_eq!(snap.tokens_out, 17, "tokens_out mismatch");

    let not_found = snap
        .error_counts
        .iter()
        .find(|(k, _)| *k == "not_found")
        .map_or(0, |(_, v)| *v);
    assert_eq!(not_found, 2, "not_found count mismatch");

    // Cross-check: JSON error_counts from AppState must agree.
    let json = state.error_counts.to_json();
    assert_eq!(json["not_found"], 2, "json not_found mismatch");
    assert_eq!(snap.tokens_in, 42);
}

/// Verify `percentile_u64` nearest-rank edge cases.
#[test]
fn f5_percentile_u64_edge_cases() {
    assert_eq!(percentile_u64(&[], 50), 0, "empty slice must return 0");
    assert_eq!(percentile_u64(&[5], 50), 5, "single element");
    assert_eq!(percentile_u64(&[1, 2, 3, 4, 5], 50), 3, "p50 of 5");
    assert_eq!(percentile_u64(&[1, 2, 3, 4, 5], 100), 5, "p100 must be max");
    assert_eq!(percentile_u64(&[1, 2, 3, 4, 5], 0), 1, "p0 must be min");
}

// ── Multi-turn tool-call request deserialize ─────────────────────────────
//
// The pi coding agent echoes the prior assistant tool-call turn plus the
// tool result back on turn 2. Before this fix the request body failed
// axum's untagged-enum extractor with HTTP 422 (`content: null` had no
// variant; `tool_calls`/`tool_call_id` had no fields). These two messages
// are taken verbatim from a real failing pi session.

#[test]
fn pi_assistant_tool_call_message_deserializes() {
    let body = r#"{
        "model": "m",
        "messages": [
            {"role":"user","content":"hi"},
            {"role":"assistant","content":null,"tool_calls":[
                {"id":"call_x","type":"function",
                 "function":{"name":"write","arguments":"{\"path\":\"a.py\"}"}}]},
            {"role":"tool","tool_call_id":"call_x","content":"ok"}
        ]
    }"#;
    let req = parse(body).expect("pi-style tool request must deserialize");
    assert_eq!(req.messages.len(), 3);

    let asst = &req.messages[1];
    assert_eq!(asst.role, "assistant");
    assert!(asst.content.is_none(), "assistant content is JSON null");
    assert_eq!(asst.content_text(), "", "null content -> empty string");
    let calls = asst.tool_calls.as_ref().expect("tool_calls present");
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].id.as_deref(), Some("call_x"));
    assert_eq!(calls[0].kind.as_deref(), Some("function"));
    assert_eq!(calls[0].function.name, "write");
    assert_eq!(calls[0].function.arguments, r#"{"path":"a.py"}"#);

    let tool = &req.messages[2];
    assert_eq!(tool.role, "tool");
    assert_eq!(tool.tool_call_id.as_deref(), Some("call_x"));
    assert_eq!(tool.content_text(), "ok");
}

#[test]
fn plain_messages_still_deserialize_without_tool_fields() {
    let body = r#"{"model":"m","messages":[
        {"role":"system","content":"sys"},
        {"role":"user","content":"u"}]}"#;
    let req = parse(body).expect("plain request must deserialize");
    assert_eq!(req.messages[0].content_text(), "sys");
    assert!(req.messages[0].tool_calls.is_none());
    assert!(req.messages[1].tool_call_id.is_none());
}

// OwnedTplMessage must turn the OpenAI wire `arguments` JSON-string into a
// parsed object so Qwen3.6's `tool_call.arguments|items` filter works.
#[test]
fn owned_tpl_message_parses_arguments_string_to_object() {
    let m = ChatMessage {
        role: "assistant".to_owned(),
        content: None,
        tool_calls: Some(vec![RequestToolCall {
            id: Some("call_1".to_owned()),
            kind: Some("function".to_owned()),
            function: RequestToolCallFunction {
                name: "write".to_owned(),
                arguments: r#"{"path":"a.py","content":"x"}"#.to_owned(),
            },
        }]),
        tool_call_id: None,
        name: None,
    };
    let owned = OwnedTplMessage::from_request(&m);
    let tc = owned.tool_calls_json.expect("tool_calls json built");
    let args = &tc[0]["function"]["arguments"];
    assert!(
        args.is_object(),
        "arguments must be a parsed object: {args}"
    );
    assert_eq!(args["path"], "a.py");
    assert_eq!(tc[0]["function"]["name"], "write");
}

// ── content-part extraction ──────────────────────────────────────

/// Standard OpenAI image_url shape + mlx-vlm input_image shape + input_audio.
#[test]
fn extract_image_and_audio_parts() {
    let parts: Vec<Value> = serde_json::from_str(
        r#"[
        {"type": "text", "text": "describe this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/img.png"}},
        {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
        {"type": "input_audio", "input_audio": {"data": "AAEC"}}
    ]"#,
    )
    .unwrap();

    let imgs = extract_image_parts(&parts);
    assert_eq!(
        imgs,
        vec![
            "https://example.com/img.png",
            "data:image/png;base64,abc123",
        ],
        "image_url and input_image shapes both extracted"
    );

    let audio = extract_audio_parts(&parts);
    assert_eq!(audio, vec!["AAEC"], "input_audio data extracted");
}

/// Text-only Parts produce empty Vecs — extract functions must not touch
/// text parts and must not allocate any strings.
#[test]
fn text_only_parts_produce_empty_vecs() {
    let parts: Vec<Value> = serde_json::from_str(
        r#"[
        {"type": "text", "text": "hello world"}
    ]"#,
    )
    .unwrap();

    assert!(
        extract_image_parts(&parts).is_empty(),
        "text-only parts must yield empty images"
    );
    assert!(
        extract_audio_parts(&parts).is_empty(),
        "text-only parts must yield empty audio"
    );
}

/// MessageContent::Text (plain string) — as_text() is unaffected.
#[test]
fn text_content_as_text_unchanged() {
    let content = MessageContent::Text("hello".to_owned());
    assert_eq!(content.as_text().as_ref(), "hello");
}

/// MessageContent::Parts with only text — as_text() joins them, unchanged.
#[test]
fn parts_text_join_unchanged() {
    let parts: Vec<Value> = serde_json::from_str(
        r#"[
        {"type": "text", "text": "foo "},
        {"type": "text", "text": "bar"}
    ]"#,
    )
    .unwrap();
    let content = MessageContent::Parts(parts);
    assert_eq!(content.as_text().as_ref(), "foo bar");
}

/// Mixed Parts (text + image_url) — as_text() returns only the text.
#[test]
fn mixed_parts_as_text_returns_text_only() {
    let parts: Vec<Value> = serde_json::from_str(
        r#"[
        {"type": "text", "text": "what is this?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.jpg"}}
    ]"#,
    )
    .unwrap();
    let content = MessageContent::Parts(parts.clone());
    assert_eq!(
        content.as_text().as_ref(),
        "what is this?",
        "as_text must ignore non-text parts"
    );
    // And the extractor sees the image.
    let imgs = extract_image_parts(&parts);
    assert_eq!(imgs, vec!["https://example.com/x.jpg"]);
}

// ── H1: SsdHistogram +Inf double-count regression ────────────────────────
//
// A single observation beyond the last finite bucket (> 1_000_000 µs) must
// produce _count == 1 and bucket{le="+Inf"} == 1. Before the fix,
// `count_inf` was added to `count` at exposition, yielding 2.

#[test]
fn ssd_histogram_overflow_no_double_count() {
    let mut h = SsdHistogram::default();
    h.observe(2_000_000); // beyond all HIST_BUCKETS_US entries
                          // Total count must be 1 — the exposition uses `count` directly.
    assert_eq!(h.count, 1, "_count should be 1 after one observation");
    // The overflow tracker is separate and must not affect count.
    assert_eq!(h.count_inf_overflow, 1, "overflow sentinel should be 1");
    // All finite buckets stay at 0 (no observation fell inside them).
    assert!(
        h.buckets.iter().all(|&b| b == 0),
        "no finite bucket should be incremented for an overflow observation"
    );
}

#[test]
fn ssd_histogram_in_range_increments_correct_buckets() {
    let mut h = SsdHistogram::default();
    // 5_000 µs falls in HIST_BUCKETS_US[3] = 5_000 and above.
    h.observe(5_000);
    assert_eq!(h.count, 1);
    assert_eq!(h.count_inf_overflow, 0);
    // Buckets 0-2 (≤ 100, ≤ 500, ≤ 1_000) must be 0.
    assert_eq!(h.buckets[0], 0);
    assert_eq!(h.buckets[1], 0);
    assert_eq!(h.buckets[2], 0);
    // Bucket 3 (≤ 5_000) and above must be 1 (cumulative).
    for i in 3..HIST_BUCKETS_US.len() {
        assert_eq!(h.buckets[i], 1, "bucket[{i}] should be 1");
    }
}

// ── tool_choice=required/named schema synthesis + bare_json envelope ─────────

fn make_tool(name: &str, schema: Value) -> NormalizedTool {
    NormalizedTool {
        name: name.to_owned(),
        description: None,
        schema,
    }
}

#[test]
fn tool_choice_to_schema_named_single_tool() {
    let tools = vec![make_tool(
        "search",
        serde_json::json!({"type":"object","properties":{"q":{"type":"string"}},"required":["q"]}),
    )];
    let schema =
        tool_choice_to_schema(&NormalizedToolChoice::Named("search".to_owned()), &tools).unwrap();
    // Schema must be an object with `name` const and `arguments`.
    assert_eq!(schema["properties"]["name"]["const"], "search");
    assert!(schema["properties"]["arguments"].is_object());
    assert_eq!(schema["required"].as_array().unwrap().len(), 2);
    assert_eq!(schema["additionalProperties"], false);
}

#[test]
fn tool_choice_to_schema_named_unknown_tool_returns_none() {
    let tools = vec![make_tool("search", serde_json::json!({"type":"object"}))];
    let result = tool_choice_to_schema(
        &NormalizedToolChoice::Named("nonexistent".to_owned()),
        &tools,
    );
    assert!(result.is_none(), "unknown tool name must return None");
}

#[test]
fn tool_choice_to_schema_required_single_tool_no_oneof() {
    let tools = vec![make_tool("fn_a", serde_json::json!({"type":"object"}))];
    let schema = tool_choice_to_schema(&NormalizedToolChoice::Required, &tools).unwrap();
    // Single tool → no oneOf wrapping.
    assert!(
        schema.get("oneOf").is_none(),
        "single-tool required must not produce oneOf: {schema}"
    );
    assert_eq!(schema["properties"]["name"]["const"], "fn_a");
}

#[test]
fn tool_choice_to_schema_required_multi_tool_produces_oneof() {
    let tools = vec![
        make_tool("alpha", serde_json::json!({"type":"object"})),
        make_tool("beta", serde_json::json!({"type":"object"})),
    ];
    let schema = tool_choice_to_schema(&NormalizedToolChoice::Required, &tools).unwrap();
    let branches = schema["oneOf"].as_array().unwrap();
    assert_eq!(branches.len(), 2, "two tools → two oneOf branches");
    assert_eq!(branches[0]["properties"]["name"]["const"], "alpha");
    assert_eq!(branches[1]["properties"]["name"]["const"], "beta");
}

#[test]
fn tool_choice_to_schema_required_empty_tools_returns_none() {
    let result = tool_choice_to_schema(&NormalizedToolChoice::Required, &[]);
    assert!(result.is_none(), "empty tools list must return None");
}

#[test]
fn tool_choice_to_schema_auto_returns_none() {
    let tools = vec![make_tool("fn", serde_json::json!({"type":"object"}))];
    assert!(tool_choice_to_schema(&NormalizedToolChoice::Auto, &tools).is_none());
}

#[test]
fn tool_choice_to_schema_none_returns_none() {
    let tools = vec![make_tool("fn", serde_json::json!({"type":"object"}))];
    assert!(tool_choice_to_schema(&NormalizedToolChoice::None, &tools).is_none());
}

// ── bare_json_to_tool_call ────────────────────────────────────────────────────

#[test]
fn bare_json_to_tool_call_basic() {
    let json = r#"{"name":"search","arguments":{"q":"rust lang"}}"#;
    let tc = bare_json_to_tool_call(json).unwrap();
    assert_eq!(tc.name, "search");
    assert_eq!(
        tc.arguments.get("q"),
        Some(&Value::String("rust lang".to_owned()))
    );
    assert!(tc.id.starts_with("call_"));
}

#[test]
fn bare_json_to_tool_call_empty_arguments() {
    let json = r#"{"name":"ping","arguments":{}}"#;
    let tc = bare_json_to_tool_call(json).unwrap();
    assert_eq!(tc.name, "ping");
    assert!(tc.arguments.is_empty());
}

#[test]
fn bare_json_to_tool_call_stringified_arguments() {
    // Some models emit `arguments` as a JSON-encoded string.
    let json = r#"{"name":"fn","arguments":"{\"x\":42}"}"#;
    let tc = bare_json_to_tool_call(json).unwrap();
    assert_eq!(tc.name, "fn");
    assert_eq!(tc.arguments.get("x"), Some(&Value::Number(42.into())));
}

#[test]
fn bare_json_to_tool_call_missing_name_returns_none() {
    let json = r#"{"arguments":{"x":1}}"#;
    assert!(bare_json_to_tool_call(json).is_none());
}

#[test]
fn bare_json_to_tool_call_invalid_json_returns_none() {
    assert!(bare_json_to_tool_call("not json").is_none());
}

#[test]
fn bare_json_to_tool_call_empty_name_returns_none() {
    let json = r#"{"name":"","arguments":{}}"#;
    assert!(bare_json_to_tool_call(json).is_none());
}
