use super::*;

fn primary_snap_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

fn qwen36_snap_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_QWEN36").map(std::path::PathBuf::from)
}

fn dr_venus_snap_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_DR_VENUS").map(std::path::PathBuf::from)
}

fn laguna_snap_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_LAGUNA").map(std::path::PathBuf::from)
}

// ── Minimal hand-written template ──────────────────────────────────────────

#[test]
fn minimal_template_renders_correctly() {
    let src = "{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}".to_owned();
    let tpl = ChatTemplate::new(src).expect("compile");
    let messages = vec![
        ChatMessageTpl {
            role: "user",
            content: "Hello",
            ..Default::default()
        },
        ChatMessageTpl {
            role: "assistant",
            content: "Hi",
            ..Default::default()
        },
    ];
    let opts = RenderOpts {
        bos_token: "<bos>",
        eos_token: "<eos>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let rendered = tpl.render(&messages, &opts).expect("render");
    assert_eq!(
        rendered.text, "<|user|>Hello<|assistant|>Hi<|assistant|>",
        "unexpected rendered output: {:?}",
        rendered.text
    );
    assert!(rendered.add_generation_prompt);
}

// ── raise_exception filter ────────────────────────────────────────────────

#[test]
fn raise_exception_returns_error() {
    let src = r#"{{ "oops" | raise_exception }}"#.to_owned();
    let tpl = ChatTemplate::new(src).expect("compile");
    let result = tpl.render(
        &[],
        &RenderOpts {
            bos_token: "",
            eos_token: "",
            add_generation_prompt: false,
            tools: &[],
            enable_thinking: None,
        },
    );
    assert!(result.is_err(), "raise_exception must produce an error");
}

// ── Generic HF byte-compare across model snapshots ────────────────────────
//
// Every supported architecture must render identical bytes to HF
// `apply_chat_template(messages, add_generation_prompt=True)`.
// References below were captured via Python +transformers tokenizer
// for each snapshot. Tests skip silently when the snapshot is absent
// (CI or alternate dev machines).
//
// Adding a new architecture: capture the HF render for the simple
// ("system: 'You are helpful.'", "user: 'Hi'") fixture, append a row.
fn render_matches_hf_or_skip(snap: &str, bos: &str, eos: &str, expected: &str) {
    let p = Path::new(snap);
    if !p.exists() {
        tracing::warn!(path = snap, "snapshot absent — skipping render-vs-HF test");
        return;
    }
    let source = load_template_source(p).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile template");
    let messages = vec![
        ChatMessageTpl {
            role: "system",
            content: "You are helpful.",
            ..Default::default()
        },
        ChatMessageTpl {
            role: "user",
            content: "Hi",
            ..Default::default()
        },
    ];
    let opts = RenderOpts {
        bos_token: bos,
        eos_token: eos,
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let r = tpl.render(&messages, &opts).expect("render");
    assert_eq!(
        r.text, expected,
        "{snap} render diverges from HF reference.\n  got:      {:?}\n  expected: {:?}",
        r.text, expected
    );
}

#[test]
fn render_matches_hf_gemma4_e4b() {
    let Some(dir) = primary_snap_dir() else {
        tracing::warn!(
            "RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping render_matches_hf_gemma4_e4b"
        );
        return;
    };
    render_matches_hf_or_skip(
        dir.to_str().unwrap_or(""),
        "<bos>",
        "<eos>",
        "<bos><|turn>system\nYou are helpful.<turn|>\n<|turn>user\nHi<turn|>\n<|turn>model\n",
    );
}

#[test]
fn render_matches_hf_qwen36_moe() {
    let Some(dir) = qwen36_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_QWEN36 not set — skipping render_matches_hf_qwen36_moe");
        return;
    };
    render_matches_hf_or_skip(
        dir.to_str().unwrap_or(""),
        "",
        "<|im_end|>",
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
}

#[test]
fn render_matches_hf_dr_venus() {
    // Same Qwen3-style chat template family as Qwen3.6.
    let Some(dir) = dr_venus_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_DR_VENUS not set — skipping render_matches_hf_dr_venus");
        return;
    };
    render_matches_hf_or_skip(
        dir.to_str().unwrap_or(""),
        "",
        "<|im_end|>",
        "<|im_start|>system\nYou are helpful.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n",
    );
}

#[test]
fn render_matches_hf_laguna_xs2() {
    let Some(dir) = laguna_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_LAGUNA not set — skipping render_matches_hf_laguna_xs2");
        return;
    };
    render_matches_hf_or_skip(
        dir.to_str().unwrap_or(""),
        "〈|EOS|〉",
        "〈|EOS|〉",
        "〈|EOS|〉<system>\n\nYou are helpful.\n</system>\n<user>\nHi\n</user>\n<assistant>\n</think>",
    );
}

// ── Real Gemma4 template ──────────────────────────────────────────────────

#[test]
fn real_gemma4_template_renders_basic_prompt() {
    let Some(snap_buf) = primary_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping real_gemma4_template_renders_basic_prompt");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping real_gemma4_template_renders_basic_prompt"
        );
        return;
    }

    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Gemma4 template");

    let messages = vec![ChatMessageTpl {
        role: "user",
        content: "What is the capital of France?",
        ..Default::default()
    }];
    let opts = RenderOpts {
        bos_token: "<bos>",
        eos_token: "<eos>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let rendered = tpl
        .render(&messages, &opts)
        .expect("render Gemma4 template");

    // Must contain the user content.
    assert!(
        rendered.text.contains("What is the capital of France?"),
        "rendered text must contain user message: {:?}",
        rendered.text
    );
    // Gemma4 adds `<|turn>model\n` before assistant turn when add_generation_prompt=true.
    assert!(
        rendered.text.contains("<|turn>model"),
        "rendered text must end with model turn marker: {:?}",
        rendered.text
    );
}

// ── Regression: Qwen3.6 render must match HF apply_chat_template byte-for-byte ──
//
// Reference (HF `apply_chat_template` with `add_generation_prompt=True`):
// bos_token=None eos_token="<|im_end|>"
// ends with: '<|im_start|>assistant\n<think>\n'
//
// Earlier rMLX hardcoded `enable_thinking=false` in the Jinja context,
// which triggered Qwen3.6's template branch that injects an empty
// `<think>\n\n</think>\n\n` block — closing the assistant's thinking
// window before generation. Result: cross-backend bench saw the model
// emit garbage like "lisnsolls - lisnsolls - lisnsolls". This guard
// prevents that regression.
#[test]
fn qwen36_render_matches_hf_reference_byte_for_byte() {
    let Some(snap_buf) = qwen36_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_QWEN36 not set — skipping");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "Qwen3.6 snapshot absent — skipping qwen36_render_matches_hf_reference_byte_for_byte"
        );
        return;
    }
    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Qwen3.6 template");
    let messages = vec![
        ChatMessageTpl {
            role: "system",
            content: "You are a senior Python developer. DO NOT think out loud. DO NOT explain. Output exactly one ```python ... ``` block containing the full file. No prose before or after.",
            ..Default::default()
        },
        ChatMessageTpl {
            role: "user",
            content: "Write a Python function add(a, b) returning a+b.\n\n/no_think",
            ..Default::default()
        },
    ];
    let opts = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let r = tpl.render(&messages, &opts).expect("render");
    let expected = "<|im_start|>system\nYou are a senior Python developer. DO NOT think out loud. DO NOT explain. Output exactly one ```python ... ``` block containing the full file. No prose before or after.<|im_end|>\n<|im_start|>user\nWrite a Python function add(a, b) returning a+b.\n\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n";
    assert_eq!(
        r.text, expected,
        "Qwen3.6 render diverges from HF reference.\n  got:      {:?}\n  expected: {:?}",
        r.text, expected
    );
}

// ── enable_thinking=Some(false) closes the <think> block ────────────
//
// When `enable_thinking: Some(false)` is set in `RenderOpts`, the Qwen3.6
// template branch `enable_thinking is defined and enable_thinking is false`
// fires, producing a closed `<think>\n\n</think>\n\n` block (no-think mode).
//
// When `enable_thinking: None` (the default), the variable is absent from
// the Jinja context, `is defined` evaluates false, and the template emits
// the open `<think>\n` block — byte-identical to HF output (guarded by
// `qwen36_render_matches_hf_reference_byte_for_byte` above).
#[test]
fn qwen36_enable_thinking_false_yields_closed_think_block() {
    let Some(snap_buf) = qwen36_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_QWEN36 not set — skipping");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "Qwen3.6 snapshot absent — skipping qwen36_enable_thinking_false_yields_closed_think_block"
        );
        return;
    }
    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Qwen3.6 template");
    let messages = vec![ChatMessageTpl {
        role: "user",
        content: "Hi",
        ..Default::default()
    }];

    // Some(false) → no-think mode: closed <think></think> block.
    let opts_no_think = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: Some(false),
    };
    let r_no_think = tpl
        .render(&messages, &opts_no_think)
        .expect("render (no-think)");
    assert!(
        r_no_think.text.contains("<think>\n\n</think>"),
        "enable_thinking=Some(false) must produce a closed <think></think> block; got: {:?}",
        r_no_think.text
    );

    // None → HF default: open <think> block (thinking enabled).
    let opts_default = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let r_default = tpl
        .render(&messages, &opts_default)
        .expect("render (default)");
    assert!(
        r_default.text.ends_with("<think>\n"),
        "enable_thinking=None must produce an open <think> block; got: {:?}",
        r_default.text
    );
    assert!(
        !r_default.text.contains("</think>"),
        "enable_thinking=None must NOT close the think block; got: {:?}",
        r_default.text
    );
}

// ── Regression: Gemma4 render must match HF apply_chat_template byte-for-byte ──
//
// Reference (HF `apply_chat_template` with `add_generation_prompt=True`):
// input: [{"role":"user","content":"What is 2+2?"}]
// rendered: '<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n'
//
// This test guards against future minijinja / template regressions.
// Skip gracefully when the primary snapshot is absent (CI without model files).
#[test]
fn gemma4_render_matches_hf_reference_byte_for_byte() {
    let Some(snap_buf) = primary_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_GEMMA4_E4B not set — skipping gemma4_render_matches_hf_reference_byte_for_byte");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "primary snapshot absent — skipping gemma4_render_matches_hf_reference_byte_for_byte"
        );
        return;
    }
    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Gemma4 template");
    let messages = vec![ChatMessageTpl {
        role: "user",
        content: "What is 2+2?",
        ..Default::default()
    }];
    let opts = RenderOpts {
        bos_token: "<bos>",
        eos_token: "<eos>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let rendered = tpl.render(&messages, &opts).expect("render");
    let expected = "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n";
    assert_eq!(
        rendered.text, expected,
        "Gemma4 render diverges from HF reference.\n  got:      {:?}\n  expected: {:?}",
        rendered.text, expected
    );
}

// ── A5.2: tool injection tests ────────────────────────────────────────────

/// Empty tools slice must NOT produce a `<tools>` block.
///
/// The `{% if tools %}` branch in Qwen3 (and most other templates) requires
/// the tools list to be truthy. An empty list is falsy in Jinja, so the
/// render output must be byte-identical to pre-A5.2 output.
#[test]
fn qwen3_empty_tools_no_tools_block() {
    let Some(snap_buf) = qwen36_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_QWEN36 not set — skipping");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "Qwen3.6 snapshot absent — skipping qwen3_empty_tools_no_tools_block"
        );
        return;
    }
    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Qwen3.6 template");
    let messages = vec![ChatMessageTpl {
        role: "user",
        content: "Hi",
        ..Default::default()
    }];
    let opts = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let rendered = tpl.render(&messages, &opts).expect("render");
    assert!(
        !rendered.text.contains("<tools>"),
        "empty tools must not produce <tools> block: {:?}",
        rendered.text
    );
    assert!(
        !rendered.text.contains("# Tools"),
        "empty tools must not produce # Tools header: {:?}",
        rendered.text
    );
}

/// Non-empty tools slice MUST produce a `<tools>` block containing the
/// function schema.
#[test]
fn qwen3_template_emits_tools_block() {
    let Some(snap_buf) = qwen36_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_QWEN36 not set — skipping");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "Qwen3.6 snapshot absent — skipping qwen3_template_emits_tools_block"
        );
        return;
    }
    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Qwen3.6 template");
    let messages = vec![ChatMessageTpl {
        role: "user",
        content: "What is the weather in Paris?",
        ..Default::default()
    }];
    let tool = serde_json::json!({
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"]
            }
        }
    });
    let tools = vec![tool];
    let opts = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &tools,
        enable_thinking: None,
    };
    let rendered = tpl.render(&messages, &opts).expect("render");
    assert!(
        rendered.text.contains("<tools>"),
        "missing <tools> block: {}",
        rendered.text
    );
    assert!(
        rendered.text.contains("get_weather"),
        "missing function name: {}",
        rendered.text
    );
    assert!(
        rendered.text.contains("What is the weather in Paris?"),
        "missing user message: {}",
        rendered.text
    );
}

/// Full multi-turn tool conversation (system, user, assistant+tool_calls,
/// tool result) must render through the real Qwen3.6 template with the
/// function name, the arguments, and the tool-result string all present.
/// This is the exact shape pi sends on turn 2 of a tool session.
#[test]
fn qwen3_renders_full_tool_conversation() {
    let Some(snap_buf) = qwen36_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_QWEN36 not set — skipping");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "Qwen3.6 snapshot absent — skipping qwen3_renders_full_tool_conversation"
        );
        return;
    }
    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Qwen3.6 template");

    // arguments arrive as a parsed object (OwnedTplMessage does the
    // wire-string -> object conversion before render; Qwen3.6 uses
    // `tool_call.arguments|items`).
    let tool_calls = serde_json::json!([{
        "id": "call_x",
        "type": "function",
        "function": {
            "name": "write_file",
            "arguments": {"path": "fizzbuzz.py"}
        }
    }]);
    let messages = vec![
        ChatMessageTpl {
            role: "system",
            content: "You are a coding agent.",
            ..Default::default()
        },
        ChatMessageTpl {
            role: "user",
            content: "Create fizzbuzz.py",
            ..Default::default()
        },
        ChatMessageTpl {
            role: "assistant",
            content: "",
            tool_calls: Some(&tool_calls),
            ..Default::default()
        },
        ChatMessageTpl {
            role: "tool",
            content: "wrote fizzbuzz.py (12 lines)",
            tool_call_id: Some("call_x"),
            ..Default::default()
        },
    ];
    let opts = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let r = tpl.render(&messages, &opts).expect("render");
    assert!(
        r.text.contains("<function=write_file>"),
        "missing tool-call function name in render:\n{}",
        r.text
    );
    assert!(
        r.text.contains("fizzbuzz.py"),
        "missing tool-call arguments in render:\n{}",
        r.text
    );
    assert!(
        r.text.contains("<tool_response>") && r.text.contains("wrote fizzbuzz.py (12 lines)"),
        "missing tool-result string in render:\n{}",
        r.text
    );
}

/// A5.2 invariant: a plain system+user render must be byte-identical to the
/// pre-tool-support output. Same fixture as
/// `qwen36_render_matches_hf_reference_byte_for_byte`; guards that the
/// optional tool keys stay absent (and the context map stays {role,
/// content}) for non-tool messages.
#[test]
fn qwen3_plain_render_byte_identical_after_tool_support() {
    let Some(snap_buf) = qwen36_snap_dir() else {
        tracing::warn!("RMLX_TEST_MODEL_QWEN36 not set — skipping");
        return;
    };
    let snap = snap_buf.as_path();
    if !snap.exists() {
        tracing::warn!(
            path = %snap.display(),
            "Qwen3.6 snapshot absent — skipping qwen3_plain_render_byte_identical_after_tool_support"
        );
        return;
    }
    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Qwen3.6 template");
    let messages = vec![
        ChatMessageTpl {
            role: "system",
            content: "You are a senior Python developer. DO NOT think out loud. DO NOT explain. Output exactly one ```python ... ``` block containing the full file. No prose before or after.",
            ..Default::default()
        },
        ChatMessageTpl {
            role: "user",
            content: "Write a Python function add(a, b) returning a+b.\n\n/no_think",
            ..Default::default()
        },
    ];
    let opts = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let r = tpl.render(&messages, &opts).expect("render");
    let expected = "<|im_start|>system\nYou are a senior Python developer. DO NOT think out loud. DO NOT explain. Output exactly one ```python ... ``` block containing the full file. No prose before or after.<|im_end|>\n<|im_start|>user\nWrite a Python function add(a, b) returning a+b.\n\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n";
    assert_eq!(
        r.text, expected,
        "plain render diverged after tool-support change (A5.2 invariant)"
    );
}

// ── Smoke-probe prompt: template-shaped vs bare-seed fallback ───────────────
//
// Regression guard for the gemma4-unified 4-bit false-positive: the smoke probe
// must feed turn-structured input when the snapshot ships a chat template, so a
// healthy instruction-tuned model is exercised the same way it is served. A bare
// instruction prompt can make even a healthy model loop a filler token; rendering
// through the template removes that false Broken* verdict. These two tests assert
// the template path is taken when present and the bare-seed fallback otherwise —
// no model snapshot required.

/// A minimal WordLevel `tokenizer.json` whose vocab covers the turn markers, the
/// `user`/`model` role words, and every whitespace-split token of `SMOKE_PROMPT`.
/// Token ids are arbitrary but distinct so the encoded id sequence is checkable.
fn write_smoke_fixture(dir: &Path, with_template: bool) {
    // Whitespace pre-tokenizer splits on spaces and punctuation, so `France?`
    // becomes `France` + `?`; the vocab lists both pieces. The `?` id is 26.
    let vocab = r#"{
        "<bos>":0,"<eos>":1,"<unk>":2,
        "<start_of_turn>":10,"<end_of_turn>":11,"user":12,"model":13,
        "What":20,"is":21,"the":22,"capital":23,"of":24,"France":25,"?":26
    }"#;
    // The angle-bracket markers must tokenize atomically (the Whitespace
    // pre-tokenizer would otherwise split `<bos>` into `< bos >`), so they are
    // registered as added/special tokens — mirroring real HF tokenizers.
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
    if with_template {
        // Gemma-style turn markers. The template emits its own BOS. Tokens are
        // space-separated so the Whitespace pre-tokenizer yields exact vocab ids.
        let tpl = "{{ bos_token }} {% for m in messages %}<start_of_turn> {{ m.role }} {{ m.content }} <end_of_turn> {% endfor %}{% if add_generation_prompt %}<start_of_turn> model{% endif %}";
        std::fs::write(dir.join("chat_template.jinja"), tpl).expect("write chat_template.jinja");
    }
}

#[test]
fn smoke_prompt_uses_chat_template_when_present() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_smoke_fixture(tmp.path(), true);
    let tk = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json")).expect("load tok");

    let ids = smoke_prompt_ids(tmp.path(), &tk, 0).expect("templated smoke prompt");

    // Must be turn-structured: starts with BOS(0) + <start_of_turn>(10) user(12),
    // contains the prompt tokens, and ends on the model-turn opener (10, 13).
    assert_eq!(
        ids.first(),
        Some(&0),
        "templated prompt must begin with BOS"
    );
    assert!(
        ids.windows(2).any(|w| w == [10, 12]),
        "expected a <start_of_turn> user span: {ids:?}"
    );
    assert_eq!(
        &ids[ids.len() - 2..],
        &[10, 13],
        "templated prompt must end on the <start_of_turn> model opener: {ids:?}"
    );
    // The prompt body tokens are present (What=20, capital=23, France=25).
    for t in [20u32, 23, 25] {
        assert!(ids.contains(&t), "missing prompt token {t} in {ids:?}");
    }
}

#[test]
fn smoke_prompt_falls_back_to_bare_seed_without_template() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_smoke_fixture(tmp.path(), false); // no chat_template.jinja
    let tk = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json")).expect("load tok");

    let ids = smoke_prompt_ids(tmp.path(), &tk, 0).expect("bare-seed smoke prompt");

    // Bare seed = [bos] + SMOKE_PROMPT tokens, no turn markers (10/13 absent).
    assert_eq!(ids.first(), Some(&0), "bare seed must begin with BOS");
    assert!(
        !ids.contains(&10) && !ids.contains(&13),
        "bare-seed fallback must not contain turn markers: {ids:?}"
    );
    assert!(
        ids.contains(&20) && ids.contains(&25),
        "missing prompt body: {ids:?}"
    );
}
