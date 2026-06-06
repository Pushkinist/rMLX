use super::*;

/// Feed `input` to a fresh parser one piece at a time and return the
/// finished parser.
fn run_pieces(pieces: &[&str]) -> ToolCallStreamParser {
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3XmlFunction);
    for piece in pieces {
        p.push(piece);
    }
    p
}

fn run_whole(input: &str) -> ToolCallStreamParser {
    run_pieces(&[input])
}

const HAPPY: &str = "I'll check the weather.\n\n<tool_call>\n<function=get_weather>\n<parameter=location>\nParis\n</parameter>\n</function>\n</tool_call>";

#[test]
fn happy_path_single_tool() {
    let mut p = run_whole(HAPPY);
    assert_eq!(p.passthrough_text, "I'll check the weather.\n\n");
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        calls[0].arguments.get("location"),
        Some(&Value::String("Paris".to_string()))
    );
    assert!(calls[0].id.starts_with("call_"));
}

#[test]
fn multi_arg() {
    let input = "<tool_call>\n<function=add>\n<parameter=a>\n1\n</parameter>\n<parameter=b>\n2\n</parameter>\n</function>\n</tool_call>";
    let mut p = run_whole(input);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "add");
    assert_eq!(
        calls[0].arguments.get("a"),
        Some(&Value::String("1".to_string()))
    );
    assert_eq!(
        calls[0].arguments.get("b"),
        Some(&Value::String("2".to_string()))
    );
}

#[test]
fn multi_line_value() {
    let input = "<tool_call>\n<function=note_taker>\n<parameter=note>\nline one\nline two\nline three\n</parameter>\n</function>\n</tool_call>";
    let mut p = run_whole(input);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].arguments.get("note"),
        Some(&Value::String("line one\nline two\nline three".to_string()))
    );
}

#[test]
fn plain_text_no_tool_call() {
    let input = "Hello there, this is just text with no tool call at all.";
    let mut p = run_whole(input);
    assert_eq!(p.passthrough_text, input);
    assert!(p.take_parsed().is_empty());
    assert!(!p.in_tool_call());
}

#[test]
fn two_calls_in_sequence() {
    let input = "First, I'll look up the weather.\n<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n</tool_call>\nThen check time.\n<tool_call>\n<function=get_time>\n<parameter=tz>\nUTC\n</parameter>\n</function>\n</tool_call>";
    let mut p = run_whole(input);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 2);
    assert_eq!(calls[0].name, "get_weather");
    assert_eq!(
        calls[0].arguments.get("city"),
        Some(&Value::String("Paris".to_string()))
    );
    assert_eq!(calls[1].name, "get_time");
    assert_eq!(
        calls[1].arguments.get("tz"),
        Some(&Value::String("UTC".to_string()))
    );
    // Text outside any <tool_call> goes to passthrough; that includes
    // the inter-call "\nThen check time.\n" because state returns to
    // Outside between the two calls.
    assert!(p
        .passthrough_text
        .starts_with("First, I'll look up the weather.\n"));
    assert!(p.passthrough_text.contains("Then check time."));
}

/// Deterministic XorShift64 — used to generate reproducible random byte
/// splits of a UTF-8 string.
struct XorShift64(u64);
impl XorShift64 {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Split `s` at `n_splits` random byte offsets, returning the pieces.
/// Offsets are snapped to UTF-8 char boundaries to avoid breaking
/// multi-byte chars (all chars in our test input are single-byte ASCII,
/// but be safe).
fn random_split<'a>(s: &'a str, n_splits: usize, rng: &mut XorShift64) -> Vec<&'a str> {
    let len = s.len();
    if len == 0 {
        return vec![s];
    }
    let mut points: Vec<usize> = (0..n_splits)
        .map(|_| (rng.next() as usize) % len)
        .map(|p| {
            // Snap to char boundary.
            let mut p = p;
            while p > 0 && !s.is_char_boundary(p) {
                p -= 1;
            }
            p
        })
        .collect();
    points.sort_unstable();
    points.dedup();
    let mut pieces = Vec::with_capacity(points.len() + 1);
    let mut prev = 0;
    for p in points {
        if p > prev {
            pieces.push(&s[prev..p]);
            prev = p;
        }
    }
    if prev < len {
        pieces.push(&s[prev..]);
    }
    pieces
}

#[test]
fn random_bpe_splits_match_whole_parse() {
    // Reference: parse the whole string in one shot.
    let mut whole_parser = run_whole(HAPPY);
    let whole_calls = whole_parser.take_parsed();
    let whole_passthrough = whole_parser.passthrough_text.clone();

    let mut rng = XorShift64(0xABCD);
    for trial in 0..5 {
        let n_splits = 4 + trial * 3; // 4, 7, 10, 13, 16
        let pieces = random_split(HAPPY, n_splits, &mut rng);
        let mut p = run_pieces(&pieces);
        let calls = p.take_parsed();
        assert_eq!(
            calls.len(),
            whole_calls.len(),
            "trial {trial}: call count differs (pieces = {pieces:?})"
        );
        assert_eq!(
            calls[0].name, whole_calls[0].name,
            "trial {trial}: name differs"
        );
        assert_eq!(
            calls[0].arguments, whole_calls[0].arguments,
            "trial {trial}: arguments differ"
        );
        assert_eq!(
            p.passthrough_text, whole_passthrough,
            "trial {trial}: passthrough differs"
        );
    }
}

#[test]
fn parser_pass_through_no_markers() {
    // Future-proofing: parser keyed on format; no markers => all passes.
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3XmlFunction);
    p.push("hello ");
    p.push("world, ");
    p.push("nothing to see here.");
    assert_eq!(p.passthrough_text, "hello world, nothing to see here.");
    assert!(p.take_parsed().is_empty());
    assert!(!p.in_tool_call());
    assert!(!p.has_calls());
}

#[test]
fn arch_string_mapping() {
    assert_eq!(
        arch_to_tool_call_format("Qwen3ForCausalLM"),
        Some(ToolCallFormat::Qwen3XmlFunction)
    );
    assert_eq!(
        arch_to_tool_call_format("Qwen3MoeForCausalLM"),
        Some(ToolCallFormat::Qwen3XmlFunction)
    );
    assert_eq!(
        arch_to_tool_call_format("Qwen3_5MoeForConditionalGeneration"),
        Some(ToolCallFormat::Qwen3XmlFunction)
    );
    assert_eq!(
        arch_to_tool_call_format("Gemma4ForConditionalGeneration"),
        None
    );
    assert_eq!(arch_to_tool_call_format("LagunaForCausalLM"), None);
    assert_eq!(arch_to_tool_call_format(""), None);
}

#[test]
fn one_byte_at_a_time_split() {
    // Most aggressive split: every byte is its own piece. Must still
    // produce the same parse as the whole string.
    let mut whole_parser = run_whole(HAPPY);
    let whole_calls = whole_parser.take_parsed();
    let whole_pass = whole_parser.passthrough_text.clone();

    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3XmlFunction);
    for ch in HAPPY.chars() {
        let mut buf = [0u8; 4];
        let s = ch.encode_utf8(&mut buf);
        p.push(s);
    }
    let calls = p.take_parsed();
    assert_eq!(calls.len(), whole_calls.len());
    assert_eq!(calls[0].name, whole_calls[0].name);
    assert_eq!(calls[0].arguments, whole_calls[0].arguments);
    assert_eq!(p.passthrough_text, whole_pass);
}

// ── Hermes JSON format (Qwen3JsonToolCall) ───────────────────────────

fn run_json(pieces: &[&str]) -> ToolCallStreamParser {
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3JsonToolCall);
    for piece in pieces {
        p.push(piece);
    }
    p
}

#[test]
fn hermes_json_single_call() {
    let input = "Sure.\n<tool_call>\n{\"name\": \"write\", \"arguments\": {\"path\": \"a.py\", \"content\": \"x = 1\"}}\n</tool_call>";
    let mut p = run_json(&[input]);
    assert_eq!(p.passthrough_text, "Sure.\n");
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "write");
    assert_eq!(
        calls[0].arguments.get("path"),
        Some(&Value::String("a.py".to_string()))
    );
    assert_eq!(
        calls[0].arguments.get("content"),
        Some(&Value::String("x = 1".to_string()))
    );
    assert!(calls[0].id.starts_with("call_"));
}

#[test]
fn hermes_json_split_pieces_match_whole() {
    let input = "<tool_call>{\"name\":\"ls\",\"arguments\":{\"dir\":\"/tmp\"}}</tool_call>";
    let whole = {
        let mut p = run_json(&[input]);
        p.take_parsed()
    };
    // Byte-by-byte split must yield the identical parse.
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3JsonToolCall);
    for ch in input.chars() {
        let mut buf = [0u8; 4];
        p.push(ch.encode_utf8(&mut buf));
    }
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls.len(), whole.len());
    assert_eq!(calls[0].name, "ls");
    assert_eq!(
        calls[0].arguments.get("dir"),
        Some(&Value::String("/tmp".to_string()))
    );
}

#[test]
fn hermes_json_stringified_arguments() {
    // Some models emit `"arguments"` as a JSON-encoded string.
    let input = r#"<tool_call>{"name":"f","arguments":"{\"k\":1}"}</tool_call>"#;
    let mut p = run_json(&[input]);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "f");
    assert_eq!(calls[0].arguments.get("k"), Some(&Value::Number(1.into())));
}

#[test]
fn hermes_json_no_call_passthrough() {
    let mut p = run_json(&["just a plain answer, no tools."]);
    assert_eq!(p.passthrough_text, "just a plain answer, no tools.");
    assert!(p.take_parsed().is_empty());
}

// ── Gemma format (GemmaToolCall) ─────────────────────────────────────

fn run_gemma(pieces: &[&str]) -> ToolCallStreamParser {
    let mut p = ToolCallStreamParser::new(ToolCallFormat::GemmaToolCall);
    for piece in pieces {
        p.push(piece);
    }
    p
}

#[test]
fn gemma_string_arg() {
    // <|"|> is the Gemma string sentinel from the chat template's
    // format_argument macro.
    let input = "<|tool_call>call:write{path:<|\"|>a.py<|\"|>}<tool_call|>";
    let mut p = run_gemma(&[input]);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "write");
    assert_eq!(
        calls[0].arguments.get("path"),
        Some(&Value::String("a.py".to_string()))
    );
    assert!(calls[0].id.starts_with("call_"));
}

#[test]
fn gemma_multi_arg_mixed_types() {
    let input = "<|tool_call>call:run{cmd:<|\"|>ls -la<|\"|>,timeout:30,verbose:true}<tool_call|>";
    let mut p = run_gemma(&[input]);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "run");
    assert_eq!(
        calls[0].arguments.get("cmd"),
        Some(&Value::String("ls -la".to_string()))
    );
    assert_eq!(
        calls[0].arguments.get("timeout"),
        Some(&Value::Number(30.into()))
    );
    assert_eq!(calls[0].arguments.get("verbose"), Some(&Value::Bool(true)));
}

#[test]
fn gemma_nested_object_and_array() {
    let input =
        "<|tool_call>call:cfg{opts:{depth:2,tags:[<|\"|>a<|\"|>,<|\"|>b<|\"|>]}}<tool_call|>";
    let mut p = run_gemma(&[input]);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "cfg");
    let opts = calls[0].arguments.get("opts").and_then(|v| v.as_object());
    let opts = opts.expect("opts object");
    assert_eq!(opts.get("depth"), Some(&Value::Number(2.into())));
    assert_eq!(
        opts.get("tags"),
        Some(&Value::Array(vec![
            Value::String("a".to_string()),
            Value::String("b".to_string()),
        ]))
    );
}

#[test]
fn gemma_string_with_braces_and_commas() {
    // Sentinel-wrapped strings must be opaque to brace/comma scanning.
    let input =
        "<|tool_call>call:write{content:<|\"|>def f(): return {1,2}<|\"|>,path:<|\"|>x.py<|\"|>}<tool_call|>";
    let mut p = run_gemma(&[input]);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(
        calls[0].arguments.get("content"),
        Some(&Value::String("def f(): return {1,2}".to_string()))
    );
    assert_eq!(
        calls[0].arguments.get("path"),
        Some(&Value::String("x.py".to_string()))
    );
}

#[test]
fn gemma_split_pieces_match_whole() {
    let input = "thinking...<|tool_call>call:get{id:<|\"|>42<|\"|>}<tool_call|>done";
    let whole = {
        let mut p = run_gemma(&[input]);
        (p.take_parsed(), p.passthrough_text.clone())
    };
    let mut p = ToolCallStreamParser::new(ToolCallFormat::GemmaToolCall);
    for ch in input.chars() {
        let mut buf = [0u8; 4];
        p.push(ch.encode_utf8(&mut buf));
    }
    let calls = p.take_parsed();
    assert_eq!(calls.len(), whole.0.len());
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "get");
    assert_eq!(
        calls[0].arguments.get("id"),
        Some(&Value::String("42".to_string()))
    );
    // Passthrough text outside the markers is preserved identically.
    assert_eq!(p.passthrough_text, whole.1);
    assert!(p.passthrough_text.contains("thinking..."));
    assert!(p.passthrough_text.contains("done"));
}

#[test]
fn gemma_no_call_passthrough() {
    let mut p = run_gemma(&["plain Gemma answer with no tool call."]);
    assert_eq!(p.passthrough_text, "plain Gemma answer with no tool call.");
    assert!(p.take_parsed().is_empty());
}

// ── detect_tool_call_format against the THREE real templates ─────────

/// Read a real snapshot's chat_template.jinja. Returns `None` (test
/// skips the assertion) if the Open Models path is absent on this host.
fn read_template(model_dir_slug: &str) -> Option<String> {
    // Resolve via env var if set; skip silently when absent (no model downloads in CI).
    let omodels_root = std::env::var_os("RMLX_O_MODELS_ROOT").map_or_else(
        || std::path::PathBuf::from("models"),
        std::path::PathBuf::from,
    );
    let path = omodels_root
        .join(model_dir_slug)
        .join("chat_template.jinja");
    std::fs::read_to_string(path).ok()
}

#[test]
fn detect_qwen36_xml_from_real_template() {
    if let Some(src) = read_template("mlx-community__Qwen3.6-35B-A3B-8bit") {
        assert_eq!(
            detect_tool_call_format(Some(&src), "Qwen3_5MoeForConditionalGeneration"),
            Some(ToolCallFormat::Qwen3XmlFunction),
            "Qwen3.6 template must resolve to XML (regression guard)"
        );
    }
}

#[test]
fn detect_ternary_bonsai_json_from_real_template() {
    if let Some(src) = read_template("prism-ml__Ternary-Bonsai-8B-mlx-2bit") {
        assert_eq!(
            detect_tool_call_format(Some(&src), "Qwen3ForCausalLM"),
            Some(ToolCallFormat::Qwen3JsonToolCall),
            "Ternary-Bonsai Hermes-JSON template must resolve to JSON"
        );
    }
}

#[test]
fn detect_gemma4_from_real_template() {
    if let Some(src) = read_template("mlx-community__gemma-4-26b-a4b-it-mxfp8") {
        assert_eq!(
            detect_tool_call_format(Some(&src), "Gemma4ForConditionalGeneration"),
            Some(ToolCallFormat::GemmaToolCall),
            "Gemma4 template must resolve to GemmaToolCall"
        );
    }
}

#[test]
fn detect_falls_back_to_arch_when_no_template() {
    // No template src → arch fallback only.
    assert_eq!(
        detect_tool_call_format(None, "Qwen3ForCausalLM"),
        Some(ToolCallFormat::Qwen3XmlFunction)
    );
    assert_eq!(
        detect_tool_call_format(None, "Gemma4ForConditionalGeneration"),
        None
    );
}

#[test]
fn detect_minimal_representative_snippets() {
    // XML: contains `<function=`.
    let xml_tpl = "...for each call emit <tool_call>\n<function=NAME>\n<parameter=K>\nV\n</parameter>\n</function>\n</tool_call>...";
    assert_eq!(
        detect_tool_call_format(Some(xml_tpl), "Qwen3ForCausalLM"),
        Some(ToolCallFormat::Qwen3XmlFunction)
    );
    // Hermes JSON: `<tool_call>` + a `{"name"` instruction, no
    // `<function=`.
    let json_tpl = "return a json object within <tool_call></tool_call> tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args>}\n</tool_call>";
    assert_eq!(
        detect_tool_call_format(Some(json_tpl), "Qwen3ForCausalLM"),
        Some(ToolCallFormat::Qwen3JsonToolCall)
    );
    // Gemma: contains `<|tool_call>`.
    let gemma_tpl = "{{- '<|tool_call>call:' + name + '{' -}} ... {{- '}<tool_call|>' -}}";
    assert_eq!(
        detect_tool_call_format(Some(gemma_tpl), "Gemma4ForConditionalGeneration"),
        Some(ToolCallFormat::GemmaToolCall)
    );
}

#[test]
fn engine_marker_allowlist_is_tight() {
    use crate::engine::tests_support_is_reconstructible_tool_marker as f;
    assert!(f("<|tool_call>"));
    assert!(f("<tool_call|>"));
    assert!(f("<|\"|>"));
    // Must NOT reconstruct turn / channel / tool-def markers.
    assert!(!f("<turn|>"));
    assert!(!f("<|turn>"));
    assert!(!f("<|channel>"));
    assert!(!f("<channel|>"));
    assert!(!f("<|tool>"));
    assert!(!f("plain"));
}

// ── allow_eof_recovery invariant + JSON balancer ──────────────────────────────

use super::balance_truncated_json;

#[test]
fn balance_already_complete_returns_none() {
    // A complete JSON object → no balancing needed, returns None.
    assert_eq!(
        balance_truncated_json(r#"{"name":"foo","arguments":{"x":1}}"#),
        None
    );
}

#[test]
fn balance_truncated_unclosed_object() {
    let truncated = r#"{"name":"foo","arguments":{"x":1"#;
    let repaired = balance_truncated_json(truncated).expect("must balance");
    let v: Value = serde_json::from_str(&repaired).expect("must be valid JSON");
    assert_eq!(v["name"], "foo");
}

#[test]
fn balance_truncated_mid_string() {
    // Truncated mid-string value.
    let truncated = r#"{"name":"foo","arguments":{"q":"hel"#;
    let repaired = balance_truncated_json(truncated).expect("must balance");
    serde_json::from_str::<Value>(&repaired).expect("repaired must parse");
}

#[test]
fn balance_truncated_after_backslash() {
    // EOF immediately after a `\` inside a string — the escape must be closed.
    let truncated = r#"{"k":"a\"#;
    let repaired = balance_truncated_json(truncated).expect("must balance");
    serde_json::from_str::<Value>(&repaired).expect("repaired must parse");
}

#[test]
fn balance_nested_containers() {
    let truncated = r#"{"name":"fn","arguments":{"a":[1,2"#;
    let repaired = balance_truncated_json(truncated).expect("must balance");
    serde_json::from_str::<Value>(&repaired).expect("repaired must parse");
}

// allow_eof_recovery invariant — streaming keeps false.
#[test]
fn streaming_parser_does_not_allow_eof_recovery() {
    // A freshly-constructed parser must have allow_eof_recovery=false
    // (verified indirectly: in-flight Bonsai block at EOS must NOT be recovered
    // before finalize() is called).
    let input = "<tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"rust\"";
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3JsonToolCall);
    p.push(input);
    // No close marker arrived — streaming path → no calls yet.
    assert!(
        p.take_parsed().is_empty(),
        "streaming: truncated call must not be emitted before finalize"
    );
}

// JSON balancer for Bonsai EOF recovery — finalize path.
#[test]
fn finalize_recovers_truncated_hermes_call() {
    // Simulate max_tokens truncation: `<tool_call>` consumed, then EOF.
    let input = "<tool_call>{\"name\":\"search\",\"arguments\":{\"q\":\"rust\"";
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3JsonToolCall);
    p.push(input);
    // Before finalize: no calls.
    assert!(p.take_parsed().is_empty());
    // Finalize: EOF recovery runs.
    p.finalize();
    let calls = p.take_parsed();
    assert_eq!(
        calls.len(),
        1,
        "finalize must recover truncated Hermes call"
    );
    assert_eq!(calls[0].name, "search");
    assert!(calls[0].id.starts_with("call_"));
}

#[test]
fn finalize_complete_hermes_call_unaffected() {
    // A fully-closed call must be returned even without finalize.
    let input = "<tool_call>{\"name\":\"add\",\"arguments\":{\"a\":1,\"b\":2}}</tool_call>";
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3JsonToolCall);
    p.push(input);
    let calls = p.take_parsed();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].name, "add");
    // finalize must be idempotent — no second call emitted.
    p.finalize();
    let extra = p.take_parsed();
    assert!(
        extra.is_empty(),
        "finalize must not re-emit already-parsed calls"
    );
}

#[test]
fn finalize_is_idempotent() {
    let input = "<tool_call>{\"name\":\"fn\",\"arguments\":{}";
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3JsonToolCall);
    p.push(input);
    p.finalize();
    let calls1 = p.take_parsed();
    // Second finalize must be a no-op.
    p.finalize();
    let calls2 = p.take_parsed();
    assert_eq!(calls1.len(), 1);
    assert!(
        calls2.is_empty(),
        "second finalize must not emit duplicate calls"
    );
}

#[test]
fn finalize_qwen3xml_partial_state_dropped_gracefully() {
    // XML parser in InToolCall (saw `<tool_call>` but no `<function=`) — finalize
    // must not panic and must produce zero calls (no name captured).
    let input = "<tool_call>\n";
    let mut p = ToolCallStreamParser::new(ToolCallFormat::Qwen3XmlFunction);
    p.push(input);
    p.finalize();
    let calls = p.take_parsed();
    assert!(
        calls.is_empty(),
        "empty XML block must be dropped on finalize"
    );
}

#[test]
fn finalize_gemma_truncated_dropped() {
    // Gemma format: truncated block is dropped (no balancer for the custom grammar).
    let input = "<|tool_call>call:search{q:";
    let mut p = ToolCallStreamParser::new(ToolCallFormat::GemmaToolCall);
    p.push(input);
    p.finalize();
    let calls = p.take_parsed();
    assert!(
        calls.is_empty(),
        "truncated Gemma block must be dropped on finalize"
    );
}
