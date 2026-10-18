use super::*;

fn primary_snap_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_GEMMA4_E4B").map(std::path::PathBuf::from)
}

fn qwen36_snap_dir() -> Option<std::path::PathBuf> {
    std::env::var_os("RMLX_TEST_MODEL_QWEN36").map(std::path::PathBuf::from)
}

fn qwen38_snap_dir() -> std::path::PathBuf {
    let root = std::env::var_os("RMLX_O_MODELS_ROOT").map_or_else(
        || std::path::PathBuf::from("models"),
        std::path::PathBuf::from,
    );
    root.join("mlx-community__Qwen3.8-27B-4bit")
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
        variables: None,
    };
    let rendered = tpl.render(&messages, &opts).expect("render");
    assert_eq!(
        rendered.text, "<|user|>Hello<|assistant|>Hi<|assistant|>",
        "unexpected rendered output: {:?}",
        rendered.text
    );
    assert!(rendered.add_generation_prompt);
}

/// `reasoning_content` is a context key only when the message carries it: absent
/// is undefined to the template, an empty string is defined.
#[test]
fn reasoning_content_is_a_context_key_only_when_present() {
    let src = "{% for m in messages %}{% if m.reasoning_content is defined %}D[{{ m.reasoning_content }}]{% else %}U{% endif %}{% endfor %}".to_owned();
    let tpl = ChatTemplate::new(src).expect("compile");
    let render = |reasoning_content| {
        let messages = [ChatMessageTpl {
            role: "assistant",
            content: "a",
            reasoning_content,
            ..Default::default()
        }];
        tpl.render(&messages, &NO_OPTS).expect("render").text
    };
    assert_eq!(render(None), "U");
    assert_eq!(render(Some("")), "D[]");
    assert_eq!(render(Some("x")), "D[x]");
}

// ── raise_exception and reasoning_effort ──────────────────────────────────

const NO_OPTS: RenderOpts<'static> = RenderOpts {
    bos_token: "",
    eos_token: "",
    add_generation_prompt: false,
    tools: &[],
    variables: None,
};

/// A template's `raise_exception(msg)` call, the form HuggingFace defines, is
/// a rejection that carries the template's own message. Any other failure is
/// not.
#[test]
fn raise_exception_is_a_rejection_carrying_the_template_message() {
    let tpl = ChatTemplate::new("{{ raise_exception('Unexpected role ' ~ 'x' ~ '.') }}".to_owned())
        .expect("compile");
    match tpl.render(&[], &NO_OPTS) {
        Err(RenderError::Rejected(msg)) => assert_eq!(msg, "Unexpected role x."),
        other => panic!("want Rejected(\"Unexpected role x.\"), got {other:?}"),
    }

    let tpl = ChatTemplate::new("{{ messages.items() }}".to_owned()).expect("compile");
    assert!(
        matches!(tpl.render(&[], &NO_OPTS), Err(RenderError::Rejected(_))),
        "an engine error on a value is the request's, as it cannot be told from a template bug"
    );
}

/// A filter, test or function the engine does not have fails every request
/// alike. That is a defect, not a rejection, even in a branch that only some
/// requests reach.
#[test]
fn an_unknown_name_is_a_defect() {
    for source in [
        "{{ messages|no_such_filter }}",
        "{% if messages is no_such_test %}{% endif %}",
        "{{ no_such_function() }}",
    ] {
        let tpl = ChatTemplate::new(source.to_owned()).expect("compile");
        match tpl.render(&[], &NO_OPTS) {
            Err(RenderError::Defect(msg)) => assert!(msg.contains("no_such"), "{source}: {msg}"),
            other => panic!("{source}: want Defect, got {other:?}"),
        }
    }
}

/// A variable is defined in the template context only when the request sets
/// it, so the template's own `default` applies otherwise.
#[test]
fn reasoning_effort_is_defined_only_when_the_request_names_one() {
    let tpl = ChatTemplate::new(
        "{{ reasoning_effort|default('template default') }}|\
         {% if reasoning_effort is defined %}defined{% else %}undefined{% endif %}"
            .to_owned(),
    )
    .expect("compile");
    let render = |effort| {
        let variables = request_variables(None, None, effort, None);
        let opts = RenderOpts {
            variables: Some(&variables),
            ..NO_OPTS
        };
        tpl.render(&[], &opts).map(|r| r.text)
    };
    assert_eq!(
        render(None).ok().as_deref(),
        Some("template default|undefined")
    );
    assert_eq!(
        render(Some("medium")).ok().as_deref(),
        Some("medium|defined")
    );
}

fn vars(json: serde_json::Value) -> serde_json::Map<String, serde_json::Value> {
    let serde_json::Value::Object(map) = json else {
        panic!("not an object");
    };
    map
}

/// Which `enable_thinking` the template gets. The top-level field and the
/// server default keep the meaning they have always had: `false` is passed,
/// `true` leaves the template to its default. `chat_template_kwargs` is passed
/// as sent, `true` and values that are not bools included. A top-level field
/// wins over the same key in `chat_template_kwargs` for every value, and
/// `chat_template_kwargs` wins over the server default.
#[test]
fn enable_thinking_resolves_top_level_then_kwargs_then_default() {
    use serde_json::json;
    let resolved = |kw: Option<serde_json::Value>, top: Option<bool>, default: Option<bool>| {
        let kw = kw.map(|v| vars(json!({ "enable_thinking": v })));
        request_variables(kw, top, None, default)
            .get("enable_thinking")
            .cloned()
    };
    for nested in [json!(true), json!(false), json!("no")] {
        for default in [None, Some(true), Some(false)] {
            assert_eq!(
                resolved(Some(nested.clone()), Some(true), default),
                None,
                "top true, nested {nested}, default {default:?}"
            );
            assert_eq!(
                resolved(Some(nested.clone()), Some(false), default),
                Some(json!(false)),
                "top false, nested {nested}, default {default:?}"
            );
            assert_eq!(
                resolved(Some(nested.clone()), None, default),
                Some(nested.clone()),
                "no top, nested {nested}, default {default:?}"
            );
        }
    }
    assert_eq!(resolved(None, None, Some(false)), Some(json!(false)));
    assert_eq!(resolved(None, None, Some(true)), None);
    assert_eq!(resolved(None, None, None), None);
}

/// With no `chat_template_kwargs`, the template sees the variable exactly as
/// on the release before `chat_template_kwargs`: that release passed
/// `enable_thinking` only when the top-level field, else the server default,
/// was `false`. Checked on all nine pairs of (field, default).
#[test]
fn without_kwargs_enable_thinking_reaches_the_template_as_it_always_did() {
    let tpl = ChatTemplate::new(
        "{% if enable_thinking is defined %}defined:{{ enable_thinking }}\
         {% else %}undefined{% endif %}"
            .to_owned(),
    )
    .expect("compile");
    for top in [None, Some(true), Some(false)] {
        for default in [None, Some(true), Some(false)] {
            let release_passes_false = top.or(default) == Some(false);
            let want = if release_passes_false {
                "defined:False"
            } else {
                "undefined"
            };
            let variables = request_variables(None, top, None, default);
            let opts = RenderOpts {
                variables: Some(&variables),
                ..NO_OPTS
            };
            assert_eq!(
                tpl.render(&[], &opts).map(|r| r.text).ok().as_deref(),
                Some(want),
                "field {top:?}, default {default:?}"
            );
        }
    }
}

/// Same for `reasoning_effort`: the top-level field wins, and no value is
/// rewritten.
#[test]
fn reasoning_effort_resolves_top_level_over_kwargs_and_is_not_rewritten() {
    use serde_json::json;
    let kwargs = vars(json!({ "reasoning_effort": "low", "other": [1, 2] }));
    let resolved = request_variables(Some(kwargs.clone()), None, Some("high"), None);
    assert_eq!(resolved.get("reasoning_effort"), Some(&json!("high")));
    assert_eq!(resolved.get("other"), Some(&json!([1, 2])));
    let nested_only = request_variables(Some(kwargs), None, None, None);
    assert_eq!(nested_only.get("reasoning_effort"), Some(&json!("low")));
}

/// A client variable reaches the template with its JSON type.
#[test]
fn client_variables_reach_the_template_with_their_types() {
    let tpl = ChatTemplate::new(
        "{{ s }}|{{ n + 1 }}|{{ l|length }}|{{ o.k }}|{{ 'yes' if b else 'no' }}|\
         {{ 'none' if z is none else 'set' }}"
            .to_owned(),
    )
    .expect("compile");
    let variables = vars(serde_json::json!({
        "s": "text", "n": 41, "l": [1, 2, 3], "o": {"k": "v"}, "b": true, "z": null
    }));
    let opts = RenderOpts {
        variables: Some(&variables),
        ..NO_OPTS
    };
    assert_eq!(
        tpl.render(&[], &opts).map(|r| r.text).ok().as_deref(),
        Some("text|42|3|v|yes|none")
    );
}

/// A client variable cannot replace one the server sets: the server must
/// account for the prompt it renders.
#[test]
fn client_variables_cannot_replace_the_variables_the_server_sets() {
    let tpl = ChatTemplate::new(
        "{{ bos_token }}|{{ eos_token }}|{{ add_generation_prompt }}|{{ tools|length }}|\
         {% for m in messages %}{{ m.content }}{% endfor %}"
            .to_owned(),
    )
    .expect("compile");
    let variables = vars(serde_json::json!({
        "bos_token": "evil", "eos_token": "evil", "add_generation_prompt": true,
        "tools": [1, 2, 3], "messages": [{"role": "user", "content": "evil"}]
    }));
    let messages = [ChatMessageTpl {
        role: "user",
        content: "real",
        ..Default::default()
    }];
    let opts = RenderOpts {
        bos_token: "BOS",
        eos_token: "EOS",
        variables: Some(&variables),
        ..NO_OPTS
    };
    assert_eq!(
        tpl.render(&messages, &opts).map(|r| r.text).ok().as_deref(),
        Some("BOS|EOS|False|0|real")
    );
}

/// minijinja wraps an error raised inside an include in a `BadInclude` error.
/// The refusal under it is still a refusal, with the template's message.
#[test]
fn a_refusal_inside_an_include_is_still_a_rejection() {
    let tpl = ChatTemplate::new(
        "{% if inner is defined %}{{ raise_exception('refused inside') }}\
         {% else %}{% set inner = true %}{% include 'chat' %}{% endif %}"
            .to_owned(),
    )
    .expect("compile");
    match tpl.render(&[], &NO_OPTS) {
        Err(RenderError::Rejected(msg)) => assert_eq!(msg, "refused inside"),
        other => panic!("want Rejected(\"refused inside\"), got {other:?}"),
    }
}

/// A template can copy request text into its refusal (Qwen3.8 copies
/// `reasoning_effort`). The message the server logs and replies with is
/// bounded: a 1 MiB value leaves at most 257 characters, starting with the
/// template's own text.
#[test]
fn a_refusal_message_is_bounded() {
    let huge = "x".repeat(1 << 20);
    let messages = [ChatMessageTpl {
        role: "user",
        content: &huge,
        ..Default::default()
    }];
    let tpl = ChatTemplate::new(
        "{{ raise_exception('Unexpected role ' ~ messages[0].content ~ '.') }}".to_owned(),
    )
    .expect("compile");
    match tpl.render(&messages, &NO_OPTS) {
        Err(RenderError::Rejected(msg)) => {
            assert!(
                msg.chars().count() <= 257,
                "{} characters",
                msg.chars().count()
            );
            assert!(msg.starts_with("Unexpected role xxxx"), "{}", &msg[..40]);
        }
        other => panic!(
            "want Rejected, got {:?}",
            other.map(|r| r.text.len()).map_err(|e| e.to_string().len())
        ),
    }
}

/// A render failure can copy request text too (a missing template named by a
/// variable). Its message is bounded like a refusal.
#[test]
fn a_render_failure_message_is_bounded() {
    let huge = "x".repeat(1 << 20);
    let messages = [ChatMessageTpl {
        role: "user",
        content: &huge,
        ..Default::default()
    }];
    let tpl = ChatTemplate::new("{% include messages[0].content %}".to_owned()).expect("compile");
    match tpl.render(&messages, &NO_OPTS) {
        Err(RenderError::Rejected(msg)) => {
            assert!(
                msg.chars().count() <= 257,
                "{} characters",
                msg.chars().count()
            );
        }
        other => panic!(
            "want Rejected, got {:?}",
            other.map(|r| r.text.len()).map_err(|e| e.to_string().len())
        ),
    }
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
        variables: None,
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
        variables: None,
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
        variables: None,
    };
    let r = tpl.render(&messages, &opts).expect("render");
    let expected = "<|im_start|>system\nYou are a senior Python developer. DO NOT think out loud. DO NOT explain. Output exactly one ```python ... ``` block containing the full file. No prose before or after.<|im_end|>\n<|im_start|>user\nWrite a Python function add(a, b) returning a+b.\n\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n";
    assert_eq!(
        r.text, expected,
        "Qwen3.6 render diverges from HF reference.\n  got:      {:?}\n  expected: {:?}",
        r.text, expected
    );
}

// ── enable_thinking=false closes the <think> block ────────────
//
// With `enable_thinking` set to `false` in `RenderOpts::variables`, the Qwen3.6
// template branch `enable_thinking is defined and enable_thinking is false`
// fires, producing a closed `<think>\n\n</think>\n\n` block (no-think mode).
//
// With no `variables`, `enable_thinking` is absent from the Jinja context,
// `is defined` evaluates false, and the template emits the open `<think>\n`
// block, byte-identical to HF output (guarded by
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

    // enable_thinking=false → no-think mode: closed <think></think> block.
    let no_think = request_variables(None, Some(false), None, None);
    let opts_no_think = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &[],
        variables: Some(&no_think),
    };
    let r_no_think = tpl
        .render(&messages, &opts_no_think)
        .expect("render (no-think)");
    assert!(
        r_no_think.text.contains("<think>\n\n</think>"),
        "enable_thinking=false must produce a closed <think></think> block; got: {:?}",
        r_no_think.text
    );

    // No variables → HF default: open <think> block (thinking enabled).
    let opts_default = RenderOpts {
        bos_token: "",
        eos_token: "<|im_end|>",
        add_generation_prompt: true,
        tools: &[],
        variables: None,
    };
    let r_default = tpl
        .render(&messages, &opts_default)
        .expect("render (default)");
    assert!(
        r_default.text.ends_with("<think>\n"),
        "no variables must produce an open <think> block; got: {:?}",
        r_default.text
    );
    assert!(
        !r_default.text.contains("</think>"),
        "no variables must NOT close the think block; got: {:?}",
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
        variables: None,
    };
    let rendered = tpl.render(&messages, &opts).expect("render");
    let expected = "<bos><|turn>user\nWhat is 2+2?<turn|>\n<|turn>model\n";
    assert_eq!(
        rendered.text, expected,
        "Gemma4 render diverges from HF reference.\n  got:      {:?}\n  expected: {:?}",
        rendered.text, expected
    );
}

// ── Tool injection tests ────────────────────────────────────────────

/// Empty tools slice must NOT produce a `<tools>` block.
///
/// The `{% if tools %}` branch in Qwen3 (and most other templates) requires
/// the tools list to be truthy. An empty list is falsy in Jinja, so the
/// render output must be byte-identical to the tool-free render.
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
        variables: None,
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
        variables: None,
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
/// tool result) must render through each real Qwen template present on this
/// host (Qwen3.6 by variable, Qwen3.8 by slug under the models root) with the
/// function name, the arguments, and the tool-result string all present.
/// This is the exact shape pi sends on turn 2 of a tool session.
#[test]
fn qwen3_renders_full_tool_conversation() {
    const TEST: &str = "qwen3_renders_full_tool_conversation";
    let mut snaps = Vec::new();
    for (name, dir) in [
        ("Qwen3.6 (RMLX_TEST_MODEL_QWEN36)", qwen36_snap_dir()),
        ("Qwen3.8 (RMLX_O_MODELS_ROOT)", Some(qwen38_snap_dir())),
    ] {
        match dir.filter(|d| d.exists()) {
            Some(dir) => snaps.push(dir),
            None => println!("SKIP {TEST}: no {name} snapshot present"),
        }
    }
    for snap in snaps {
        assert_full_tool_conversation_renders(&snap);
    }
}

fn assert_full_tool_conversation_renders(snap: &Path) {
    let source = load_template_source(snap).expect("load template source");
    let tpl = ChatTemplate::new(source).expect("compile Qwen template");

    // arguments arrive as a parsed object (OwnedTplMessage does the
    // wire-string -> object conversion before render; the Qwen templates use
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
        variables: None,
    };
    let r = tpl.render(&messages, &opts).expect("render");
    assert!(
        r.text.contains("<function=write_file>"),
        "{}: missing tool-call function name in render:\n{}",
        snap.display(),
        r.text
    );
    assert!(
        r.text.contains("fizzbuzz.py"),
        "{}: missing tool-call arguments in render:\n{}",
        snap.display(),
        r.text
    );
    assert!(
        r.text.contains("<tool_response>") && r.text.contains("wrote fizzbuzz.py (12 lines)"),
        "{}: missing tool-result string in render:\n{}",
        snap.display(),
        r.text
    );
}

/// Invariant: a plain system+user render must be byte-identical to the
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
        variables: None,
    };
    let r = tpl.render(&messages, &opts).expect("render");
    let expected = "<|im_start|>system\nYou are a senior Python developer. DO NOT think out loud. DO NOT explain. Output exactly one ```python ... ``` block containing the full file. No prose before or after.<|im_end|>\n<|im_start|>user\nWrite a Python function add(a, b) returning a+b.\n\n/no_think<|im_end|>\n<|im_start|>assistant\n<think>\n";
    assert_eq!(
        r.text, expected,
        "plain render diverged from the tool-free render"
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

    let ids = smoke_prompt_ids(tmp.path(), &tk).expect("templated smoke prompt");

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

/// Without a chat template, `smoke_prompt_ids` must return `None` so the caller
/// (`run_smoke_probe` / the CLI probe) builds the bare seed itself with its own
/// canonical BOS resolution. The function must NOT invent a token id — earlier
/// the server probe hard-coded a magic id when no `<bos>` resolved, seeding the
/// probe wrong; returning `None` keeps the BOS fallback chain in one place.
#[test]
fn smoke_prompt_falls_back_to_bare_seed_without_template() {
    let tmp = tempfile::tempdir().expect("tempdir");
    write_smoke_fixture(tmp.path(), false); // no chat_template.jinja
    let tk = tokenizers::Tokenizer::from_file(tmp.path().join("tokenizer.json")).expect("load tok");

    // No template ⇒ None (the templated path is the only thing this fn owns).
    assert!(
        smoke_prompt_ids(tmp.path(), &tk).is_none(),
        "no chat template must yield None so the caller resolves BOS itself"
    );

    // The bare-seed builder the caller falls back to is the shared
    // `arch::smoke_prompt_ids`, which takes the canonically-resolved BOS id and
    // never substitutes a magic literal. Seeded with a real BOS (0 here) it
    // produces [bos] + SMOKE_PROMPT, no turn markers.
    let ids = rmlx_models::arch::smoke_prompt_ids(&tk, 0).expect("bare-seed smoke prompt");
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
