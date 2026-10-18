use super::*;
use crate::openai::errors::error_response;
use crate::openai::request::OwnedTplMessage;

fn parse(json: &str) -> Result<MessagesRequest, serde_json::Error> {
    serde_json::from_str(json)
}

// ── Typed OOM mapping (Anthropic error types) ────────────────────────

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
    for k in [
        "process_rss_mb",
        "phys_footprint_mb",
        "compressed_mb",
        "peak_alloc_mb",
        "requested_bytes",
    ] {
        assert!(err.get(k).is_some(), "missing mem field {k}: {err}");
    }
}

fn engine_error_cases() -> Vec<rmlx_core::Error> {
    use rmlx_core::OomPhase;
    vec![
        rmlx_core::Error::SmokeProbe("nan".to_owned()),
        rmlx_core::Error::Oom {
            phase: OomPhase::LoadWeights,
            requested_bytes: None,
            peak_alloc_mb: None,
            msg: "oom".to_owned(),
        },
        rmlx_core::Error::Oom {
            phase: OomPhase::LoadKvCache,
            requested_bytes: None,
            peak_alloc_mb: None,
            msg: "oom".to_owned(),
        },
        rmlx_core::Error::Oom {
            phase: OomPhase::Generation,
            requested_bytes: None,
            peak_alloc_mb: None,
            msg: "oom".to_owned(),
        },
        // wildcard arm — the common case for a mid-decode engine fault.
        rmlx_core::Error::Mlx("device error".to_owned()),
        rmlx_core::Error::Other("engine panic".to_owned()),
        rmlx_core::Error::InvalidInput("input_audio decode failed: no audio track".to_owned()),
        rmlx_core::Error::KvCeilingExceeded {
            requested: 641,
            ceiling: 640,
        },
    ]
}

/// This surface's `engine_error_type` must name every error exactly as this
/// surface's `engine_error_response` spells it.
///
/// `/v1/messages` streaming can only use the former (its status and headers are
/// already sent) while blocking uses the latter. Drift means a streamed client
/// sees a different `type` than a blocking one for the identical fault.
#[tokio::test]
async fn engine_error_type_matches_response() {
    for e in &engine_error_cases() {
        let (_status, _retry, body) = oom_parts(e).await;
        let from_response = body["error"]["type"]
            .as_str()
            .unwrap_or_else(|| panic!("response body carries no error.type for {e:?}: {body}"));
        assert_eq!(
            from_response,
            errors::engine_error_type(e),
            "anthropic engine_error_type drifted from anthropic engine_error_response for {e:?}"
        );
    }
}

/// Input the client got wrong is a 400 `invalid_request_error` whose message is
/// the fact, not a retryable 503; a fault of the server stays a 503.
#[tokio::test]
async fn invalid_input_is_400_invalid_request_error_with_its_fact() {
    let e = rmlx_core::Error::InvalidInput(
        "input_audio decode failed: the file holds no audio track".to_owned(),
    );
    let (status, retry, body) = oom_parts(&e).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "invalid_request_error");
    assert_eq!(
        body["error"]["message"],
        "input_audio decode failed: the file holds no audio track"
    );
    assert_eq!(retry, None);
    assert_eq!(errors::engine_error_type(&e), "invalid_request_error");

    let server_fault = rmlx_core::Error::Other("the audio tower did not answer".to_owned());
    let (status, _, body) = oom_parts(&server_fault).await;
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["type"], "service_unavailable_error");
}

/// A sequence past the KV ceiling is the 400 `context_length_exceeded` the
/// route gives a prompt over `max_ctx`; the type keeps the OpenAI spelling.
#[tokio::test]
async fn kv_ceiling_exceeded_is_400_context_length_exceeded() {
    let e = rmlx_core::Error::KvCeilingExceeded {
        requested: 641,
        ceiling: 640,
    };
    let (status, retry, body) = oom_parts(&e).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["error"]["type"], "context_length_exceeded");
    assert_eq!(
        body["error"]["message"],
        "the request needs 641 tokens of context, max_ctx is 640"
    );
    assert_eq!(retry, None);
    assert_eq!(errors::engine_error_type(&e), "context_length_exceeded");
}

/// The Anthropic type strings deliberately diverge from OpenAI's: this surface
/// carries an `_error` suffix on its non-OOM types. Pinned so nobody
/// "deduplicates" the two mirrors into one shared function — that collapse is
/// exactly what made `/v1/messages` streaming emit OpenAI spellings.
///
/// Only the OOM keys coincide, and that is intentional.
#[test]
fn engine_error_type_diverges_from_openai_surface() {
    assert_eq!(
        errors::engine_error_type(&rmlx_core::Error::Mlx("x".to_owned())),
        "service_unavailable_error"
    );
    assert_eq!(
        errors::engine_error_type(&rmlx_core::Error::SmokeProbe("x".to_owned())),
        "internal_server_error"
    );
    // The OpenAI surface spells the same two faults without the suffix.
    assert_eq!(
        crate::openai::errors::engine_error_type(&rmlx_core::Error::Mlx("x".to_owned())),
        "service_unavailable"
    );
    assert_eq!(
        crate::openai::errors::engine_error_type(&rmlx_core::Error::SmokeProbe("x".to_owned())),
        "internal_error"
    );
}

#[tokio::test]
async fn j3_oom_load_weights_507_retry() {
    let e = rmlx_core::Error::Oom {
        phase: rmlx_core::OomPhase::LoadWeights,
        requested_bytes: Some(99),
        peak_alloc_mb: None,
        msg: "unable to allocate".to_owned(),
    };
    let (status, retry, body) = oom_parts(&e).await;
    assert_eq!(status, StatusCode::INSUFFICIENT_STORAGE);
    assert_eq!(body["error"]["type"], "oom_during_load");
    assert_eq!(retry.as_deref(), Some("5"));
    assert_eq!(body["error"]["requested_bytes"], 99);
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
    assert_eq!(status, StatusCode::INSUFFICIENT_STORAGE);
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
    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(body["error"]["type"], "oom_mid_stream");
    assert!(retry.is_none(), "mid-stream OOM must NOT set Retry-After");
    assert_mem_fields(&body["error"]);
}

// ── Deserialization ───────────────────────────────────────────────────────

#[test]
fn minimal_request_deserialises() {
    let req = parse(
        r#"{"model":"claude-3","max_tokens":1024,"messages":[{"role":"user","content":"hi"}]}"#,
    )
    .unwrap();
    assert_eq!(req.model, "claude-3");
    assert_eq!(req.max_tokens, 1024);
    assert_eq!(req.messages.len(), 1);
    assert!(!req.stream);
    assert!(req.system.is_none());
}

#[test]
fn system_as_string_deserialises() {
    let req = parse(
        r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"system":"You are helpful."}"#,
    )
    .unwrap();
    let sys = req.system.unwrap();
    assert_eq!(sys.as_text(), "You are helpful.");
}

#[test]
fn system_as_array_deserialises() {
    let req = parse(
        r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"system":[{"type":"text","text":"Be concise."}]}"#,
    )
    .unwrap();
    let sys = req.system.unwrap();
    assert_eq!(sys.as_text(), "Be concise.");
}

#[test]
fn full_request_deserialises() {
    let json = r#"{
        "model": "claude-3-opus",
        "max_tokens": 512,
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": [{"type":"text","text":"world"}]}
        ],
        "system": "You are a helpful assistant.",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "stop_sequences": ["<END>"],
        "stream": false,
        "metadata": {"user_id": "u123"}
    }"#;
    let req = parse(json).unwrap();
    assert_eq!(req.model, "claude-3-opus");
    assert_eq!(req.max_tokens, 512);
    assert_eq!(req.messages.len(), 2);
    assert_eq!(req.temperature, Some(0.7));
    assert_eq!(req.top_p, Some(0.95));
    assert_eq!(req.top_k, Some(40));
    assert_eq!(
        req.stop_sequences.as_deref(),
        Some(["<END>".to_owned()].as_slice())
    );
    assert!(req.metadata.is_some());
}

// ── Validation (deserialization-level) ────────────────────────────────────

#[test]
fn missing_max_tokens_is_serde_error() {
    // max_tokens is a plain u32 (no Option) — missing field is a serde error.
    let result = parse(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#);
    assert!(
        result.is_err(),
        "expected serde error for missing max_tokens"
    );
}

// ── Validation flags (caught by handler, not serde) ───────────────────────

#[test]
fn max_tokens_zero_flag() {
    let req = parse(r#"{"model":"m","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}"#)
        .unwrap();
    assert_eq!(req.max_tokens, 0); // handler rejects this
}

#[test]
fn temperature_out_of_range_flag() {
    let req = parse(
        r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"temperature":1.5}"#,
    )
    .unwrap();
    let t = req.temperature.unwrap();
    // Anthropic range is [0, 1]; 1.5 is out.
    assert!(!(0.0..=1.0).contains(&t));
}

#[test]
fn top_p_out_of_range_flag() {
    let req = parse(
        r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"top_p":2.0}"#,
    )
    .unwrap();
    assert!(!(0.0..=1.0).contains(&req.top_p.unwrap()));
}

#[test]
fn top_k_zero_flag() {
    let req = parse(
        r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"top_k":0}"#,
    )
    .unwrap();
    assert_eq!(req.top_k, Some(0)); // handler rejects this
}

#[test]
fn empty_messages_flag() {
    let req = parse(r#"{"model":"m","max_tokens":10,"messages":[]}"#).unwrap();
    assert!(req.messages.is_empty()); // handler rejects this
}

// ── Tools + tool_choice are parsed fields ──────────────────────────

/// `tools=[]` deserialises to an empty Vec (not extra).
#[test]
fn tools_empty_array_parsed_as_field() {
    let req = parse(
        r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"tools":[]}"#,
    )
    .unwrap();
    // No longer in extra — it's a first-class field.
    assert!(!req.extra.contains_key("tools"));
    assert!(req.tools.as_ref().is_none_or(Vec::is_empty));
}

/// Full tools payload parses all fields correctly.
#[test]
fn anthropic_tools_full_parse() {
    let req = parse(
        r#"{
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{
                "name": "get_weather",
                "description": "Get weather",
                "input_schema": {"type": "object", "properties": {}, "required": []}
            }],
            "tool_choice": {"type": "auto"}
        }"#,
    )
    .unwrap();
    let tools = req.tools.expect("tools must be Some");
    assert_eq!(tools.len(), 1);
    assert_eq!(tools[0].name, "get_weather");
    assert_eq!(tools[0].description.as_deref(), Some("Get weather"));
    let tc = req.tool_choice.expect("tool_choice must be Some");
    assert_eq!(tc.kind, "auto");
    assert!(tc.name.is_none());
}

/// tool_choice with type="tool" parses name correctly.
#[test]
fn anthropic_tool_choice_named_parse() {
    let req = parse(
        r#"{
            "model": "m",
            "max_tokens": 10,
            "messages": [{"role": "user", "content": "hi"}],
            "tool_choice": {"type": "tool", "name": "get_weather"}
        }"#,
    )
    .unwrap();
    let tc = req.tool_choice.expect("tool_choice must be Some");
    assert_eq!(tc.kind, "tool");
    assert_eq!(tc.name.as_deref(), Some("get_weather"));
}

/// `tool_choice` is not in extra.
#[test]
fn tool_choice_not_in_extra() {
    let req = parse(
        r#"{"model":"m","max_tokens":10,"messages":[{"role":"user","content":"hi"}],"tool_choice":{"type":"auto"}}"#,
    )
    .unwrap();
    assert!(!req.extra.contains_key("tool_choice"));
    assert!(req.tool_choice.is_some());
}

// ── stop_reason mapping ───────────────────────────────────────────────────

/// EOS (engine "stop") maps to Anthropic "end_turn", not
/// "stop_sequence". "stop_sequence" is only set by the stop-matcher path when a
/// real stop string matched; that branch bypasses map_stop_reason entirely.
#[test]
fn stop_reason_eos_is_end_turn() {
    assert_eq!(map_stop_reason(Some("stop")), "end_turn");
}

/// Length cap maps to "max_tokens".
#[test]
fn stop_reason_length_is_max_tokens() {
    assert_eq!(map_stop_reason(Some("length")), "max_tokens");
}

/// Engine "tool_calls" maps to "tool_use".
#[test]
fn stop_reason_tool_calls_is_tool_use() {
    assert_eq!(map_stop_reason(Some("tool_calls")), "tool_use");
}

/// None (stream ended without a finish token) maps to "end_turn".
#[test]
fn stop_reason_none_is_end_turn() {
    assert_eq!(map_stop_reason(None), "end_turn");
}

/// An unrecognised finish reason must NOT be laundered into the successful
/// "end_turn"; it surfaces as an explicit "error" stop reason.
#[test]
fn stop_reason_unknown_is_error_not_success() {
    let mapped = map_stop_reason(Some("whatever"));
    assert_ne!(
        mapped, "end_turn",
        "unknown finish reason must never map to the successful end_turn"
    );
    assert_eq!(mapped, "error");
}

/// A finish reason of "error" (e.g. an engine-emitted error terminal) maps to
/// an explicit "error" stop reason — the masked-failure case that motivated
/// making this mapping non-laundering. It must not become a successful stop.
#[test]
fn stop_reason_error_terminal_is_not_success() {
    let mapped = map_stop_reason(Some("error"));
    assert_ne!(mapped, "end_turn");
    assert_ne!(mapped, "max_tokens");
    assert_ne!(mapped, "tool_use");
    assert_eq!(mapped, "error");
}

/// Compose blocking path — EOS finish → stop_reason:"end_turn",
/// stop_sequence:None. Mirrors the logic in generate_blocking.
#[test]
fn blocking_eos_finish_composes_end_turn() {
    // Engine emits finish_reason="stop" for natural EOS.
    let terminal = map_stop_reason(Some("stop"));
    assert_eq!(terminal, "end_turn");
    // No tool_use → passes through.
    let stop_reason = select_anthropic_stop_reason(false, terminal);
    assert_eq!(stop_reason, "end_turn");
    // On the EOS branch matched_stop is None, so the response stop_sequence
    // field stays null — asserted end-to-end in stop_sequence_smoke.rs.
}

/// Compose blocking path — max_tokens finish → "max_tokens".
#[test]
fn blocking_length_finish_composes_max_tokens() {
    let terminal = map_stop_reason(Some("length"));
    assert_eq!(terminal, "max_tokens");
    let stop_reason = select_anthropic_stop_reason(false, terminal);
    assert_eq!(stop_reason, "max_tokens");
}

/// Stop-sequence match path is independent — the explicit
/// "stop_sequence" branch bypasses map_stop_reason in both handlers.
/// This test encodes the invariant so a future refactor cannot break it.
#[test]
fn stop_sequence_match_path_is_independent_of_map_stop_reason() {
    // map_stop_reason never returns "stop_sequence".
    for reason in [Some("stop"), Some("length"), Some("tool_calls"), None] {
        assert_ne!(
            map_stop_reason(reason),
            "stop_sequence",
            "map_stop_reason({reason:?}) must never return stop_sequence"
        );
    }
    // The stop-matcher path sets stop_reason="stop_sequence" directly (not via
    // map_stop_reason), mimicked here:
    let matched: Option<String> = Some("<END>".to_owned());
    let stop_reason = if matched.is_some() {
        "stop_sequence".to_owned()
    } else {
        map_stop_reason(Some("stop"))
    };
    assert_eq!(stop_reason, "stop_sequence");
    assert_eq!(matched.as_deref(), Some("<END>"));
}

// ── tool_use serialisation + stop_reason upgrade ───────────────────

use serde_json::Map as JsonMap;

fn make_parsed_call(name: &str, args: &[(&str, &str)]) -> ParsedToolCall {
    let mut m = JsonMap::new();
    for (k, v) in args {
        m.insert((*k).to_owned(), Value::String((*v).to_owned()));
    }
    ParsedToolCall {
        id: "call_abc".to_owned(),
        name: name.to_owned(),
        arguments: m,
    }
}

/// `ContentBlock::ToolUse` serialises as the Anthropic wire shape:
/// `{"type":"tool_use", "id":..., "name":..., "input": {...}}`.
/// Note: `input` is a JSON OBJECT, not a JSON-stringified string.
#[test]
fn tool_use_block_serialises_to_anthropic_shape() {
    let parsed = make_parsed_call("get_weather", &[("location", "Paris")]);
    let block = to_tool_use_block(&parsed);
    let v = serde_json::to_value(&block).unwrap();
    assert_eq!(v["type"], "tool_use");
    assert_eq!(v["id"], "call_abc");
    assert_eq!(v["name"], "get_weather");
    // `input` MUST be a JSON object, not a stringified blob.
    assert!(v["input"].is_object(), "input must be a JSON object");
    assert_eq!(v["input"]["location"], "Paris");
}

/// `select_anthropic_stop_reason` upgrades to `"tool_use"` when any
/// tool_use was emitted; otherwise the terminal reason passes through.
#[test]
fn stop_reason_upgrades_to_tool_use() {
    assert_eq!(
        select_anthropic_stop_reason(true, "end_turn".to_owned()),
        "tool_use"
    );
    assert_eq!(
        select_anthropic_stop_reason(true, "max_tokens".to_owned()),
        "tool_use"
    );
    assert_eq!(
        select_anthropic_stop_reason(true, "stop_sequence".to_owned()),
        "tool_use"
    );
    // No tool calls → passthrough.
    assert_eq!(
        select_anthropic_stop_reason(false, "end_turn".to_owned()),
        "end_turn"
    );
    assert_eq!(
        select_anthropic_stop_reason(false, "max_tokens".to_owned()),
        "max_tokens"
    );
}

/// content[] assembly: thinking → text → tool_use in that order. Mirrors
/// the non-streaming `generate_blocking` arrangement.
#[test]
fn content_array_assembly_with_tool_use() {
    let parsed = make_parsed_call("get_weather", &[("location", "Paris")]);
    let thinking = "t1".to_owned();
    let text = "hello".to_owned();

    let mut content: Vec<ContentBlock> = Vec::new();
    if !thinking.is_empty() {
        content.push(ContentBlock::Thinking { thinking });
    }
    if !text.is_empty() {
        content.push(ContentBlock::Text { text });
    }
    content.push(to_tool_use_block(&parsed));

    let v = serde_json::to_value(&content).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[0]["type"], "thinking");
    assert_eq!(arr[0]["thinking"], "t1");
    assert_eq!(arr[1]["type"], "text");
    assert_eq!(arr[1]["text"], "hello");
    assert_eq!(arr[2]["type"], "tool_use");
    assert_eq!(arr[2]["name"], "get_weather");
    assert_eq!(arr[2]["input"]["location"], "Paris");
}

/// ContentBlock::Text still serialises to the legacy shape.
#[test]
fn text_block_shape_unchanged() {
    let b = ContentBlock::Text {
        text: "hi".to_owned(),
    };
    let v = serde_json::to_value(&b).unwrap();
    assert_eq!(v["type"], "text");
    assert_eq!(v["text"], "hi");
}

/// ContentBlock::Thinking still serialises to the legacy shape.
#[test]
fn thinking_block_shape_unchanged() {
    let b = ContentBlock::Thinking {
        thinking: "t".to_owned(),
    };
    let v = serde_json::to_value(&b).unwrap();
    assert_eq!(v["type"], "thinking");
    assert_eq!(v["thinking"], "t");
}

/// Multiple tool_use blocks each carry their own (id, name, input).
#[test]
fn multi_tool_use_blocks_each_independent() {
    let p0 = make_parsed_call("get_weather", &[("city", "Paris")]);
    let p1 = make_parsed_call("get_time", &[("tz", "UTC")]);
    let content: Vec<ContentBlock> = vec![
        ContentBlock::Text {
            text: "ok".to_owned(),
        },
        to_tool_use_block(&p0),
        to_tool_use_block(&p1),
    ];
    let v = serde_json::to_value(&content).unwrap();
    let arr = v.as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert_eq!(arr[1]["name"], "get_weather");
    assert_eq!(arr[1]["input"]["city"], "Paris");
    assert_eq!(arr[2]["name"], "get_time");
    assert_eq!(arr[2]["input"]["tz"], "UTC");
}

// ── Streaming-side helper coverage ─────────────────────────────────

fn event_data_string(ev: &Event) -> String {
    // axum's `Event` is opaque; rely on Debug rendering to access the
    // serialised data field, the same trick the openai tests use.
    format!("{ev:?}")
}

/// `enqueue_tool_use_block` emits the Anthropic-spec triplet:
/// content_block_start(tool_use, id+name) →
/// content_block_delta(input_json_delta, partial_json) →
/// content_block_stop.
/// `current_block` resets to `None` after.
///
/// `Event`'s Debug rendering is byte-quoted: a JSON `"` appears as the
/// two characters `\` `"` in the printed form. Match needles include
/// the escape backslash to compensate.
#[test]
fn streaming_tool_use_emits_full_block_sequence() {
    let parsed = make_parsed_call("get_weather", &[("location", "Paris")]);
    let mut queue: std::collections::VecDeque<Result<Event, std::convert::Infallible>> =
        std::collections::VecDeque::new();
    let mut current_block: Option<BlockKind> = None;
    let mut current_index: u32 = 0;

    enqueue_tool_use_block(&mut queue, &mut current_block, &mut current_index, &parsed);

    // No prior block → no opening cb_stop. Expect exactly: start, delta, stop.
    assert_eq!(queue.len(), 3);
    let start_dbg = event_data_string(queue[0].as_ref().unwrap());
    let delta_dbg = event_data_string(queue[1].as_ref().unwrap());
    let stop_dbg = event_data_string(queue[2].as_ref().unwrap());

    // Debug-rendered byte buffer escapes JSON `"` as `\\"`.
    assert!(
        start_dbg.contains(r"event: content_block_start"),
        "start event header: {start_dbg}"
    );
    assert!(
        start_dbg.contains(r#"\"type\":\"tool_use\""#),
        "start event payload type: {start_dbg}"
    );
    assert!(
        start_dbg.contains(r#"\"name\":\"get_weather\""#),
        "start event name: {start_dbg}"
    );
    assert!(
        start_dbg.contains(r#"\"id\":\"call_abc\""#),
        "start event id: {start_dbg}"
    );

    assert!(
        delta_dbg.contains(r"event: content_block_delta"),
        "delta event header: {delta_dbg}"
    );
    assert!(
        delta_dbg.contains(r#"\"type\":\"input_json_delta\""#),
        "delta type: {delta_dbg}"
    );
    // partial_json carries the FULL serialised input. The inner JSON
    // is doubly-escaped because it's a string field whose value is
    // itself JSON, which is then byte-escaped by Debug for the buffer.
    assert!(
        delta_dbg.contains("location") && delta_dbg.contains("Paris"),
        "delta should carry partial_json containing input: {delta_dbg}"
    );

    assert!(
        stop_dbg.contains(r"event: content_block_stop"),
        "stop event header: {stop_dbg}"
    );

    // current_block resets after a tool_use; current_index advanced by 1.
    assert!(current_block.is_none());
    assert_eq!(current_index, 1);
}

/// When a text block was already open, `enqueue_tool_use_block` first
/// emits content_block_stop for that block (closing it) and bumps the
/// index, then emits the tool_use triplet at the next index.
#[test]
fn streaming_tool_use_closes_prior_text_block() {
    let parsed = make_parsed_call("foo", &[]);
    let mut queue: std::collections::VecDeque<Result<Event, std::convert::Infallible>> =
        std::collections::VecDeque::new();
    let mut current_block: Option<BlockKind> = Some(BlockKind::Text);
    let mut current_index: u32 = 0;

    enqueue_tool_use_block(&mut queue, &mut current_block, &mut current_index, &parsed);

    // Prior text → cb_stop, then start, delta, stop = 4 total.
    assert_eq!(queue.len(), 4);
    let prior_stop_dbg = event_data_string(queue[0].as_ref().unwrap());
    assert!(
        prior_stop_dbg.contains(r"event: content_block_stop"),
        "prior stop header: {prior_stop_dbg}"
    );
    assert!(
        prior_stop_dbg.contains(r#"\"index\":0"#),
        "prior stop index: {prior_stop_dbg}"
    );

    // The tool_use block lives at index 1 (post-increment).
    let start_dbg = event_data_string(queue[1].as_ref().unwrap());
    assert!(
        start_dbg.contains(r#"\"index\":1"#),
        "start index: {start_dbg}"
    );
    assert_eq!(current_index, 2); // advanced past the tool_use block.
    assert!(current_block.is_none());
}

// ── Anthropic sampling schema ──────────────────────────────────────

/// Anthropic MessagesRequest accepts `top_k`; OpenAI-only knobs
/// (min_p, repetition_penalty, frequency_penalty, presence_penalty, logit_bias)
/// are NOT fields on the struct — compile-time guarantee.
///
/// We verify that a payload carrying `top_k` deserialises correctly,
/// and that the struct does not have the OpenAI-only field names
/// (checked by confirming extra unknown fields are accepted as `extra`
/// rather than blowing up deserialization — already covered by the
/// `#[serde(flatten)] extra` catch-all).
#[test]
fn a7_anthropic_top_k_parses() {
    let req = parse(
        r#"{
        "model": "claude-3-sonnet",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": "hi"}],
        "top_k": 15
    }"#,
    )
    .unwrap();
    assert_eq!(req.top_k, Some(15));
}

/// The bytes of an Anthropic error are the error envelope with `message`
/// before `type` and nothing else.
#[tokio::test]
async fn an_error_body_is_the_envelope_bytes() {
    let resp = error_response(StatusCode::BAD_REQUEST, "invalid_request_error", "m");
    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
    let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX)
        .await
        .unwrap();
    assert_eq!(
        bytes.as_ref(),
        br#"{"error":{"message":"m","type":"invalid_request_error"}}"#
    );
}

// ── Template messages of earlier turns ────────────────────────────────────

fn tpl_messages(messages: &str) -> Vec<OwnedTplMessage> {
    let req = parse(&format!(
        r#"{{"model":"m","max_tokens":1,"messages":{messages}}}"#
    ))
    .unwrap();
    req.messages
        .iter()
        .flat_map(OwnedTplMessage::from_anthropic)
        .collect()
}

/// Thinking text, text and every tool call of an assistant turn make one
/// template message, with each value as the client sent it.
#[test]
fn an_assistant_turn_keeps_thinking_text_and_tool_calls() {
    let msgs = tpl_messages(
        r#"[{"role":"assistant","content":[
            {"type":"thinking","thinking":"plan","signature":"sig"},
            {"type":"redacted_thinking","data":"opaque"},
            {"type":"text","text":"ok"},
            {"type":"tool_use","id":"t1","name":"get_weather","input":{"city":"Paris","days":2}},
            {"type":"tool_use","id":"t2","name":"get_time","input":{}}]}]"#,
    );
    assert_eq!(msgs.len(), 1);
    let m = &msgs[0];
    assert_eq!(m.role, "assistant");
    assert_eq!(m.content, "ok");
    assert_eq!(m.reasoning_content.as_deref(), Some("plan"));
    assert_eq!(
        m.tool_calls_json,
        Some(serde_json::json!([
            {"id":"t1","type":"function",
             "function":{"name":"get_weather","arguments":{"city":"Paris","days":2}}},
            {"id":"t2","type":"function","function":{"name":"get_time","arguments":{}}}
        ]))
    );
}

/// Each tool result is a `tool` message with its call id, string content or
/// text blocks. The remaining text is one `user` message after them.
#[test]
fn a_user_turn_gives_one_tool_message_per_tool_result_then_its_text() {
    let msgs = tpl_messages(
        r#"[{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"t1","content":"sunny","is_error":true},
            {"type":"tool_result","tool_use_id":"t2","content":[
                {"type":"text","text":"late"},
                {"type":"image","source":{"type":"base64","data":"AAEC"}},
                {"type":"text","text":" night"}]},
            {"type":"text","text":"thanks"}]}]"#,
    );
    let got: Vec<(&str, &str, Option<&str>)> = msgs
        .iter()
        .map(|m| {
            (
                m.role.as_str(),
                m.content.as_str(),
                m.tool_call_id.as_deref(),
            )
        })
        .collect();
    assert_eq!(
        got,
        [
            ("tool", "sunny", Some("t1")),
            ("tool", "late night", Some("t2")),
            ("user", "thanks", None),
        ]
    );
}

/// A user turn of tool results alone gives no empty `user` message.
#[test]
fn a_user_turn_of_tool_results_alone_gives_tool_messages_only() {
    let msgs =
        tpl_messages(r#"[{"role":"user","content":[{"type":"tool_result","tool_use_id":"t1"}]}]"#);
    assert_eq!(msgs.len(), 1);
    assert_eq!(
        (msgs[0].role.as_str(), msgs[0].content.as_str()),
        ("tool", "")
    );
}

/// A string content is one message as before; text blocks join with nothing.
#[test]
fn plain_content_is_one_message_with_no_extra_fields() {
    let msgs = tpl_messages(
        r#"[{"role":"user","content":"hi"},
            {"role":"assistant","content":[{"type":"text","text":"a"},{"type":"text","text":"b"}]}]"#,
    );
    assert_eq!(msgs.len(), 2);
    for (m, role, content) in [(&msgs[0], "user", "hi"), (&msgs[1], "assistant", "ab")] {
        assert_eq!((m.role.as_str(), m.content.as_str()), (role, content));
        assert!(m.reasoning_content.is_none() && m.tool_calls_json.is_none());
        assert!(m.tool_call_id.is_none() && m.name.is_none());
    }
}

/// Blocks with no template field leave no message and no field.
#[test]
fn redacted_thinking_leaves_no_field() {
    let msgs = tpl_messages(
        r#"[{"role":"assistant","content":[
            {"type":"redacted_thinking","data":"opaque"},
            {"type":"text","text":"ok"}]}]"#,
    );
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].content, "ok");
    assert!(msgs[0].reasoning_content.is_none());
    assert!(msgs[0].tool_calls_json.is_none());
}

/// Thinking blocks of one message join with nothing, an empty one still gives
/// the field, and a message without one gives none.
#[test]
fn thinking_blocks_join_and_an_empty_one_gives_the_field() {
    let msgs = tpl_messages(
        r#"[{"role":"assistant","content":[
                {"type":"thinking","thinking":"a"},{"type":"thinking","thinking":"b"}]},
            {"role":"assistant","content":[{"type":"thinking","thinking":""}]},
            {"role":"assistant","content":[{"type":"text","text":"t"}]}]"#,
    );
    let got: Vec<Option<&str>> = msgs
        .iter()
        .map(|m| m.reasoning_content.as_deref())
        .collect();
    assert_eq!(got, [Some("ab"), Some(""), None]);
}

/// A message that holds a tool result and a thinking block or a tool call
/// gives the tool message and then its own message with that field. The own
/// message has no text.
#[test]
fn a_tool_result_beside_thinking_or_a_tool_call_gives_both_messages() {
    let msgs = tpl_messages(
        r#"[{"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"r"},
                {"type":"thinking","thinking":"plan"}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t2","content":"r"},
                {"type":"tool_use","id":"t3","name":"n","input":{}}]}]"#,
    );
    let roles: Vec<&str> = msgs.iter().map(|m| m.role.as_str()).collect();
    assert_eq!(roles, ["tool", "user", "tool", "user"]);
    assert_eq!(msgs[1].reasoning_content.as_deref(), Some("plan"));
    assert!(msgs[1].tool_calls_json.is_none() && msgs[1].content.is_empty());
    assert!(msgs[3].tool_calls_json.is_some() && msgs[3].reasoning_content.is_none());
    assert_eq!(msgs[3].content, "");
}

/// A turn of tool calls alone is one message with no text. A turn with no
/// block the template reads is one empty message, as a string content "" is.
#[test]
fn a_tool_call_turn_and_an_empty_turn_are_one_message_each() {
    let msgs = tpl_messages(
        r#"[{"role":"assistant","content":[
                {"type":"tool_use","id":"t1","name":"n","input":{"a":1}}]},
            {"role":"user","content":[]},
            {"role":"user","content":[{"type":"image","source":{"type":"base64","data":"AAEC"}}]}]"#,
    );
    assert_eq!(msgs.len(), 3);
    assert!(msgs[0].content.is_empty() && msgs[0].tool_calls_json.is_some());
    for m in &msgs[1..] {
        assert_eq!(m.role, "user");
        assert!(m.content.is_empty() && m.tool_calls_json.is_none());
    }
}

/// A tool result with no id has no call id, and a tool call with no id or
/// with a `null` input is passed on as sent. An absent input is an empty
/// object.
#[test]
fn missing_ids_and_odd_inputs_are_passed_on_as_sent() {
    let msgs = tpl_messages(
        r#"[{"role":"user","content":[{"type":"tool_result","content":"r"}]},
            {"role":"assistant","content":[
                {"type":"tool_use","name":"a","input":null},
                {"type":"tool_use","id":"t","name":"b"}]}]"#,
    );
    assert_eq!(msgs[0].tool_call_id, None);
    assert_eq!(
        msgs[1].tool_calls_json,
        Some(serde_json::json!([
            {"id":null,"type":"function","function":{"name":"a","arguments":null}},
            {"id":"t","type":"function","function":{"name":"b","arguments":{}}}
        ]))
    );
}

/// Blocks inside a tool result are read for text only: an audio block there
/// is neither text nor audio for the engine.
#[test]
fn an_audio_block_inside_a_tool_result_is_ignored() {
    let req = parse(
        r#"{"model":"m","max_tokens":1,"messages":[{"role":"user","content":[
            {"type":"tool_result","tool_use_id":"t1","content":[
                {"type":"input_audio","source":{"type":"base64","data":"AAEC"}},
                {"type":"text","text":"heard"}]}]}]}"#,
    )
    .unwrap();
    assert_eq!(req.messages[0].audio_b64().count(), 0);
    let msgs = OwnedTplMessage::from_anthropic(&req.messages[0]);
    assert_eq!(msgs.len(), 1);
    assert_eq!(msgs[0].content, "heard");
}

/// One history sent to `/v1/messages` and to `/v1/chat/completions` gives the
/// Qwen3.8 template the same input: the rendered text is equal, and it holds
/// the reasoning, the calls and the results.
#[test]
fn the_qwen38_template_renders_one_history_alike_from_both_routes() {
    const SNAPSHOT: &str = "mlx-community__Qwen3.8-27B-4bit";
    const TEST: &str = "the_qwen38_template_renders_one_history_alike_from_both_routes";
    let Some(root) = std::env::var_os("RMLX_O_MODELS_ROOT").filter(|r| !r.is_empty()) else {
        println!("SKIP {TEST}: RMLX_O_MODELS_ROOT is not set");
        return;
    };
    let root = std::path::PathBuf::from(root);
    assert!(
        root.is_dir(),
        "{TEST}: RMLX_O_MODELS_ROOT is set but is not a directory"
    );
    let dir = root.join(SNAPSHOT);
    if !dir.is_dir() {
        println!("SKIP {TEST}: RMLX_O_MODELS_ROOT does not hold {SNAPSHOT}");
        return;
    }
    let source = crate::chat_template::load_template_source(&dir)
        .unwrap_or_else(|e| panic!("{TEST}: {SNAPSHOT} is present but its template fails: {e}"));
    let tpl = crate::chat_template::ChatTemplate::new(source).unwrap();

    let anthropic: Vec<OwnedTplMessage> = tpl_messages(
        r#"[{"role":"user","content":"weather?"},
            {"role":"assistant","content":[
                {"type":"thinking","thinking":"plan the lookup","signature":"sig"},
                {"type":"text","text":"checking"},
                {"type":"tool_use","id":"t1","name":"get_weather","input":{"city":"Paris"}},
                {"type":"tool_use","id":"t2","name":"get_time","input":{"city":"Rome"}}]},
            {"role":"user","content":[
                {"type":"tool_result","tool_use_id":"t1","content":"sunny"},
                {"type":"tool_result","tool_use_id":"t2","content":[{"type":"text","text":"late"}]},
                {"type":"text","text":"thanks"}]}]"#,
    );
    let openai_req: crate::openai::request::ChatCompletionsRequest = serde_json::from_str(
        r#"{"model":"m","messages":[
            {"role":"user","content":"weather?"},
            {"role":"assistant","content":"checking","reasoning_content":"plan the lookup",
             "tool_calls":[
                {"id":"t1","type":"function",
                 "function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}},
                {"id":"t2","type":"function",
                 "function":{"name":"get_time","arguments":"{\"city\":\"Rome\"}"}}]},
            {"role":"tool","tool_call_id":"t1","content":"sunny"},
            {"role":"tool","tool_call_id":"t2","content":"late"},
            {"role":"user","content":"thanks"}]}"#,
    )
    .unwrap();
    let openai: Vec<OwnedTplMessage> = openai_req
        .messages
        .iter()
        .map(OwnedTplMessage::from_request)
        .collect();

    let render = |owned: &[OwnedTplMessage]| -> String {
        let msgs: Vec<crate::chat_template::ChatMessageTpl<'_>> =
            owned.iter().map(OwnedTplMessage::as_tpl).collect();
        let opts = crate::chat_template::RenderOpts {
            bos_token: "",
            eos_token: "",
            add_generation_prompt: true,
            tools: &[],
            variables: None,
        };
        tpl.render(&msgs, &opts).unwrap().text
    };
    let from_anthropic = render(&anthropic);
    assert_eq!(from_anthropic, render(&openai));
    for held in ["plan the lookup", "get_weather", "Paris", "sunny", "late"] {
        assert!(from_anthropic.contains(held), "{held}: {from_anthropic}");
    }
}

/// The audio blocks of a message are listed in order; a plain string has none.
#[test]
fn input_audio_blocks_are_listed_in_order() {
    let req = parse(
        r#"{"model":"m","max_tokens":1,"messages":[
            {"role":"user","content":[
                {"type":"input_audio","source":{"type":"base64","data":"AAEC"}},
                {"type":"text","text":"x"},
                {"type":"input_audio","source":{"type":"base64","data":"AwQF"}}]},
            {"role":"user","content":"plain"}]}"#,
    )
    .unwrap();
    assert_eq!(
        req.messages[0].audio_b64().collect::<Vec<_>>(),
        ["AAEC", "AwQF"]
    );
    assert_eq!(req.messages[1].audio_b64().count(), 0);
}
