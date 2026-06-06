use super::*;

fn parse(json: &str) -> Result<MessagesRequest, serde_json::Error> {
    serde_json::from_str(json)
}

// ── J3: typed OOM mapping (Anthropic error types) ────────────────────────

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

// ── A5.1: tools + tool_choice are now first-class fields ─────────────────

/// A5.1: tools=[] deserialises to an empty Vec (not extra).
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

/// A5.1: full tools payload parses all fields correctly.
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

/// A5.1: tool_choice with type="tool" parses name correctly.
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

/// A5.1: tool_choice is no longer in extra.
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

/// Unknown reason falls back to "end_turn".
#[test]
fn stop_reason_unknown_is_end_turn() {
    assert_eq!(map_stop_reason(Some("whatever")), "end_turn");
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

// ── A5.5: tool_use serialisation + stop_reason upgrade ───────────────────

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

// ── A5.5: streaming-side helper coverage ─────────────────────────────────

fn event_data_string(ev: &Event) -> String {
    // axum's `Event` is opaque; rely on Debug rendering to access the
    // serialised data field, the same trick A5.4 used in openai tests.
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

// ── A7.1: Anthropic sampling schema ──────────────────────────────────────

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
