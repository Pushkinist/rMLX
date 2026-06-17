// unsafe_code: String::from_utf8_unchecked over serde_json output (Python-compatible JSON formatter, ASCII-safe)
#![allow(unsafe_code)]

//! Chat-template rendering via minijinja.
//!
//! Loads `chat_template.jinja` from a model directory and renders it with
//! the given messages and options. The rendered string is ready for tokenization.
//!
//! Stage 1.7 — prompt pipeline.

#![allow(clippy::needless_pass_by_value)]
use std::path::Path;

use minijinja::{value::ValueKind, Environment, Error as MjError, ErrorKind, Value};

use rmlx_core::{Error, Result};

// ── Python-compatible JSON serialiser ─────────────────────────────────────────
//
// Python's `json.dumps` uses `": "` (colon + space) and `", "` (comma +
// space) separators by default. minijinja's built-in `tojson` filter calls
// `serde_json::to_string` which produces compact JSON (no spaces).
//
// Chat templates (Qwen3, Qwen3.5MoE, …) pass tool specs through `| tojson`
// and the rendered string must be byte-identical to HF's output. We register
// a custom `tojson` filter that produces Python-compatible spacing.

struct PythonCompatFormatter;

impl serde_json::ser::Formatter for PythonCompatFormatter {
    #[inline]
    fn begin_object_value<W>(&mut self, writer: &mut W) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        writer.write_all(b": ")
    }

    #[inline]
    fn begin_array_value<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }

    #[inline]
    fn begin_object_key<W>(&mut self, writer: &mut W, first: bool) -> std::io::Result<()>
    where
        W: ?Sized + std::io::Write,
    {
        if first {
            Ok(())
        } else {
            writer.write_all(b", ")
        }
    }
}

/// Serialise a minijinja `Value` to a JSON string using Python-compatible
/// separators (`": "` and `", "`).
fn value_to_python_json(value: &Value) -> std::result::Result<String, serde_json::Error> {
    let mut out = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut out, PythonCompatFormatter);
    serde::Serialize::serialize(value, &mut ser)?;
    // SAFETY: serde_json only writes valid UTF-8.
    Ok(unsafe { String::from_utf8_unchecked(out) })
}

// ── Public types ─────────────────────────────────────────────────────────────

/// Rendered chat string produced by [`ChatTemplate::render`].
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed output struct — two fields are the complete render-result contract; adding a field requires updating all ChatTemplate::render callers"
)]
#[derive(Debug, Clone)]
pub struct RenderedPrompt {
    /// The fully rendered prompt string ready for tokenization.
    pub text: String,
    /// True iff the request asked for an assistant continuation.
    pub add_generation_prompt: bool,
}

/// One message passed to the Jinja template.
///
/// Tool fields are only populated for multi-turn tool conversations and are
/// injected into the per-message Jinja context **only when present**, so plain
/// (no-tool) messages render byte-identically to pre-tool-support output (the
/// A5.2 invariant: `{% if message.tool_calls %}` stays falsy).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed template-context struct — five fields are the complete per-message Jinja context contract; adding a field requires updating all ChatMessageTpl construction sites"
)]
#[derive(Debug, Clone, Default)]
pub struct ChatMessageTpl<'a> {
    /// Message role: `"user"`, `"assistant"`, `"system"`, or `"tool"`.
    pub role: &'a str,
    /// Decoded text content for this message turn.
    pub content: &'a str,
    /// `assistant`-turn tool calls, OpenAI-shaped:
    /// `[{"id":..,"type":"function","function":{"name":..,"arguments":<obj>}}]`.
    /// `None` for non-tool messages (key omitted from context).
    pub tool_calls: Option<&'a serde_json::Value>,
    /// `tool`-role result link id (`None` ⇒ key omitted).
    pub tool_call_id: Option<&'a str>,
    /// Optional message/function name (`None` ⇒ key omitted).
    pub name: Option<&'a str>,
}

/// Options forwarded to the Jinja template context.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed options struct — four fields are the complete render-options contract; adding a field requires updating all RenderOpts construction sites and ChatTemplate::render"
)]
#[derive(Debug, Clone)]
pub struct RenderOpts<'a> {
    /// BOS token string injected into the Jinja context.
    pub bos_token: &'a str,
    /// EOS token string injected into the Jinja context.
    pub eos_token: &'a str,
    /// When `true`, appends the model's assistant-start token after the last message.
    pub add_generation_prompt: bool,
    /// JSON tool specs to inject into the template's `tools` variable.
    ///
    /// Each element should be an OpenAI-shaped function spec:
    /// `{"type":"function","function":{"name":...,"description":...,"parameters":{...}}}`.
    ///
    /// An empty slice causes the Jinja `{% if tools %}` branch to evaluate
    /// false, so renders without tools are byte-identical to pre-A5.2 output.
    pub tools: &'a [serde_json::Value],
    /// Controls the Qwen3.6 `enable_thinking` template variable.
    ///
    /// `Some(false)` → inject `enable_thinking = false` into the Jinja context,
    /// which triggers the `enable_thinking is defined and enable_thinking is false`
    /// branch in the Qwen3.6 template and produces a closed `<think></think>` block
    /// (no-think mode).
    ///
    /// `None` or `Some(true)` → leave the variable **undefined** in the context.
    /// An undefined variable is indistinguishable from absent in Jinja's
    /// `is defined` test, so the template falls through to its default behaviour:
    /// an open `<think>\n` block, byte-identical to HF `apply_chat_template`.
    ///
    /// **Never** define the variable for `Some(true)` — doing so would cause
    /// `enable_thinking is defined and enable_thinking is false` to evaluate
    /// false (because the value is true), which is the same as the None/absent
    /// path. Only `Some(false)` changes behaviour.
    pub enable_thinking: Option<bool>,
}

/// Compiled chat template backed by a minijinja environment.
///
/// Construct once per model; `render` may be called many times.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed template struct — private minijinja environment field; public API is ChatTemplate::new() and render(); adding a field requires updating ChatTemplate::new"
)]
pub struct ChatTemplate {
    env: Environment<'static>,
}

impl std::fmt::Debug for ChatTemplate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChatTemplate").finish_non_exhaustive()
    }
}

// ── Source preprocessing ─────────────────────────────────────────────────────

/// Replace `{% generation %}` / `{% endgeneration %}` markers with empty
/// `{# … #}` comments. These are HuggingFace finetuning markers used to tag
/// the assistant span for loss-masking; they have no effect on the rendered
/// text. minijinja rejects unknown statements, so we neutralise them up-front.
fn strip_generation_markers(src: &str) -> String {
    // Matches `{%-? generation -?%}` and `{%-? endgeneration -?%}` with any
    // surrounding whitespace control. We replace each with a comment so byte
    // offsets in error messages remain meaningful.
    let mut out = String::with_capacity(src.len());
    let mut rest = src;
    while let Some(open) = rest.find("{%") {
        out.push_str(&rest[..open]);
        let after = &rest[open..];
        let close = if let Some(c) = after.find("%}") {
            c + 2
        } else {
            out.push_str(after);
            rest = "";
            break;
        };
        let tag = &after[..close];
        // Inner stripped of `{%-?` / `-?%}` to inspect the keyword.
        let inner = tag
            .trim_start_matches("{%")
            .trim_end_matches("%}")
            .trim_matches('-')
            .trim();
        if inner == "generation" || inner == "endgeneration" {
            out.push_str("{##}");
        } else {
            out.push_str(tag);
        }
        rest = &after[close..];
    }
    out.push_str(rest);
    out
}

// ── I/O ──────────────────────────────────────────────────────────────────────

/// Load `<model_dir>/chat_template.jinja` and return its raw source.
pub fn load_template_source(model_dir: &Path) -> Result<String> {
    let path = model_dir.join("chat_template.jinja");
    std::fs::read_to_string(&path)
        .map_err(|e| Error::Other(format!("cannot read {}: {e}", path.display())))
}

// ── Smoke-probe prompt ─────────────────────────────────────────────────────────

/// Build the smoke-probe input token ids for a model snapshot by rendering the
/// canonical seed through the model's real `chat_template.jinja`, so the probe
/// exercises the same prompt shape the model is served with in production.
///
/// Instruction-tuned models are trained to continue *turn-structured* input
/// (`<start_of_turn>user … <start_of_turn>model`). Fed a bare instruction with
/// no turn scaffolding, a healthy model can still degenerate into a repeated
/// filler token — a behaviour the reference loader reproduces identically. That
/// made the bare-seed probe raise false `Broken*` verdicts for snapshots that
/// generate correctly through `serve`. Rendering the canonical seed through
/// `chat_template.jinja` removes that false positive generally, for any arch
/// whose bare-prompt continuation is degenerate.
///
/// Returns `None` when the snapshot has no usable `chat_template.jinja` (or the
/// render/encode fails) — base / non-chat snapshots. The caller then passes
/// `None` to `run_smoke_probe`, which builds the shared bare seed itself with
/// its own canonical BOS resolution. Keeping the bare-seed construction out of
/// this function means the BOS fallback chain lives in exactly one place per
/// entry point and no token id is hard-coded here.
///
/// The template emits its own `<bos>`, so the rendered text is tokenized with
/// `add_special_tokens = false` (mirrors the production request path).
pub fn smoke_prompt_ids(model_dir: &Path, tokenizer: &tokenizers::Tokenizer) -> Option<Vec<u32>> {
    match render_templated_seed(model_dir, tokenizer) {
        Ok(ids) => Some(ids),
        Err(reason) => {
            // Expected for base / non-chat snapshots. Recorded at debug level so
            // a run's `.jsonl` shows whether the probe ran templated or fell back
            // to the bare seed, and why — without warning on the normal case.
            tracing::debug!(
                reason,
                "smoke_prompt_ids: no usable chat template — using bare seed"
            );
            None
        }
    }
}

/// Render `arch::SMOKE_PROMPT` as a single user turn through the model's chat
/// template and tokenize the result. Returns the encoded ids on success, or a
/// human-readable reason string on any miss (no template, compile/render/encode
/// failure, or empty output) so the caller can log it once at the fallback
/// boundary.
fn render_templated_seed(
    model_dir: &Path,
    tokenizer: &tokenizers::Tokenizer,
) -> std::result::Result<Vec<u32>, String> {
    let src = load_template_source(model_dir).map_err(|e| format!("load template: {e}"))?;
    let tpl = ChatTemplate::new(src).map_err(|e| format!("compile template: {e}"))?;

    let cfg = crate::tokenizer_io::load_tokenizer_config(model_dir)
        .map_err(|e| format!("load tokenizer_config: {e}"))?;
    let bos_token = cfg.bos_token.as_deref().unwrap_or("");
    let eos_token = cfg.eos_token.as_deref().unwrap_or("");

    let messages = [ChatMessageTpl {
        role: "user",
        content: rmlx_models::arch::SMOKE_PROMPT,
        ..ChatMessageTpl::default()
    }];
    let opts = RenderOpts {
        bos_token,
        eos_token,
        add_generation_prompt: true,
        tools: &[],
        enable_thinking: None,
    };
    let rendered = tpl
        .render(&messages, &opts)
        .map_err(|e| format!("render template: {e}"))?;

    // The template already emits the BOS marker, so encode without re-adding
    // special tokens (mirrors the production request path).
    let enc = tokenizer
        .encode(rendered.text.as_str(), false)
        .map_err(|e| format!("encode rendered prompt: {e}"))?;
    let ids = enc.get_ids().to_vec();
    if ids.is_empty() {
        return Err("rendered prompt encoded to zero tokens".to_owned());
    }
    Ok(ids)
}

// ── ChatTemplate ──────────────────────────────────────────────────────────────

impl ChatTemplate {
    /// Compile a minijinja template from raw source.
    ///
    /// Registers compatibility shims for Python-dict/string methods that
    /// real HuggingFace chat templates commonly call:
    ///
    /// - `dict.get(key)` → attribute access (returns `undefined` if missing)
    /// - `str.split(sep)` → split string into sequence
    ///
    /// Also registers a `raise_exception` filter so templates that call it
    /// surface the message as a render error rather than a silent no-op.
    pub fn new(source: String) -> Result<Self> {
        let mut env = Environment::<'static>::new();

        // ── Python compat: .get() and .split() via unknown_method_callback ──
        env.set_unknown_method_callback(|_state, value, method, args| {
            match method {
                "get" => {
                    // dict.get(key) or dict.get(key, default)
                    let key = args.first().cloned().unwrap_or(Value::UNDEFINED);
                    let default_val = args.get(1).cloned().unwrap_or(Value::UNDEFINED);
                    if value.kind() == ValueKind::Map {
                        let result = value.get_attr(&key.to_string());
                        match result {
                            Ok(v) if !v.is_undefined() => Ok(v),
                            _ => Ok(default_val),
                        }
                    } else {
                        Err(MjError::from(ErrorKind::UnknownMethod))
                    }
                }
                "split" => {
                    // str.split(sep) → sequence of strings
                    let sep = args
                        .first()
                        .and_then(|v| v.as_str().map(str::to_owned))
                        .unwrap_or_default();
                    if value.kind() == ValueKind::String {
                        let s = value.as_str().unwrap_or("");
                        if sep.is_empty() {
                            let parts: Vec<Value> = s.split_whitespace().map(Value::from).collect();
                            Ok(Value::from(parts))
                        } else {
                            let parts: Vec<Value> =
                                s.split(sep.as_str()).map(Value::from).collect();
                            Ok(Value::from(parts))
                        }
                    } else {
                        Err(MjError::from(ErrorKind::UnknownMethod))
                    }
                }
                "startswith" | "endswith" => {
                    // str.startswith(prefix) / str.endswith(suffix)
                    if value.kind() != ValueKind::String {
                        return Err(MjError::from(ErrorKind::UnknownMethod));
                    }
                    let s = value.as_str().unwrap_or("");
                    let needle = args
                        .first()
                        .and_then(|v| v.as_str().map(str::to_owned))
                        .unwrap_or_default();
                    let result = if method == "startswith" {
                        s.starts_with(&needle)
                    } else {
                        s.ends_with(&needle)
                    };
                    Ok(Value::from(result))
                }
                "strip" | "lstrip" | "rstrip" => {
                    if value.kind() != ValueKind::String {
                        return Err(MjError::from(ErrorKind::UnknownMethod));
                    }
                    let s = value.as_str().unwrap_or("");
                    let trimmed = match method {
                        "strip" => s.trim().to_owned(),
                        "lstrip" => s.trim_start().to_owned(),
                        "rstrip" => s.trim_end().to_owned(),
                        // Outer match arm `"strip" | "lstrip" | "rstrip"` makes
                        // this arm structurally unreachable. Returning an error
                        // instead of panicking ensures safety if the pattern
                        // ever diverges.
                        _ => return Err(MjError::from(ErrorKind::UnknownMethod)),
                    };
                    Ok(Value::from(trimmed))
                }
                _ => Err(MjError::from(ErrorKind::UnknownMethod)),
            }
        });

        // ── raise_exception filter ────────────────────────────────────────────
        // Some HF templates call `raise_exception(msg)`. Propagate as error.
        env.add_filter(
            "raise_exception",
            |msg: String| -> std::result::Result<String, MjError> {
                Err(MjError::new(
                    ErrorKind::InvalidOperation,
                    format!("template raised exception: {msg}"),
                ))
            },
        );

        // ── tojson override — Python-compatible spacing ───────────────────────
        // HF `apply_chat_template` renders tool specs via Jinja `| tojson`
        // which uses Python's `json.dumps` (separators `": "` / `", "`).
        // minijinja's built-in `tojson` uses compact format (no spaces).
        // We override the filter so the rendered string is byte-identical to
        // HF output on templates that pass tool specs through `| tojson`.
        env.add_filter(
            "tojson",
            |value: Value| -> std::result::Result<String, MjError> {
                value_to_python_json(&value).map_err(|e| {
                    MjError::new(ErrorKind::InvalidOperation, format!("tojson failed: {e}"))
                })
            },
        );

        // Strip HuggingFace-specific `{% generation %}...{% endgeneration %}`
        // markers. They tag the assistant span for finetuning loss masks; for
        // pure inference rendering they're no-ops. minijinja does not recognise
        // them. We blank the opening/closing tags rather than removing the
        // body so the rendered text is byte-identical to HF.
        let source = strip_generation_markers(&source);

        env.add_template_owned("chat", source)
            .map_err(|e| Error::Other(format!("failed to compile chat template: {e}")))?;

        Ok(ChatTemplate { env })
    }

    /// Render the template with the given messages and options.
    ///
    /// The template context exposes:
    /// - `messages` — list of `{role, content}` maps
    /// - `add_generation_prompt` — bool
    /// - `bos_token` — string
    /// - `eos_token` — string
    /// - `tools` — list of OpenAI-shaped function specs (empty = no tools)
    /// - `enable_thinking` — injected ONLY when `opts.enable_thinking == Some(false)`;
    ///   absent (undefined) otherwise (= HF default behaviour, open `<think>` block).
    pub fn render(
        &self,
        messages: &[ChatMessageTpl<'_>],
        opts: &RenderOpts<'_>,
    ) -> Result<RenderedPrompt> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| Error::Other(format!("template lookup failed: {e}")))?;

        // Build the messages list as a minijinja Value.
        //
        // Plain messages emit exactly {role, content} — byte-identical to
        // pre-tool-support output (A5.2 invariant). Tool keys are added ONLY
        // when present so `{% if message.tool_calls %}` / `message.role ==
        // "tool"` branches stay inert for ordinary conversation turns.
        let msgs_val: Vec<Value> = messages
            .iter()
            .map(|m| {
                let mut map: std::collections::BTreeMap<&str, Value> =
                    std::collections::BTreeMap::new();
                map.insert("role", Value::from(m.role));
                map.insert("content", Value::from(m.content));
                if let Some(tc) = m.tool_calls {
                    map.insert("tool_calls", Value::from_serialize(tc));
                }
                if let Some(id) = m.tool_call_id {
                    map.insert("tool_call_id", Value::from(id));
                }
                if let Some(name) = m.name {
                    map.insert("name", Value::from(name));
                }
                Value::from_serialize(map)
            })
            .collect();

        // inject `enable_thinking` into the Jinja context ONLY when
        // `opts.enable_thinking == Some(false)`.
        //
        // Qwen3.6 chat_template.jinja tests:
        // `enable_thinking is defined and enable_thinking is false`
        //
        // When this condition holds the template appends a closed
        // `<think>\n\n</think>\n\n` block — no-think mode. When the variable
        // is UNDEFINED (= absent from the context), the `is defined` check
        // fails and the template falls through to the open `<think>\n` default,
        // which is byte-identical to HF `apply_chat_template` output.
        //
        // CRITICAL: do NOT inject the variable for `Some(true)` or `None`.
        // Injecting `enable_thinking = true` would make `is defined` true but
        // `is false` false — the condition still fails and the template opens
        // the think block, same as `None`. The undefined path is the contract;
        // only `Some(false)` changes the rendered output.

        // A5.2: convert the (possibly empty) tools slice into a minijinja
        // Value. An empty slice serialises as an empty JSON array, which
        // Jinja evaluates as falsy — so `{% if tools %}` is false when no
        // tools are present and renders byte-identically to pre-A5.2 output.
        let tools_val = Value::from_serialize(opts.tools);

        // build the base context; conditionally extend with enable_thinking.
        let ctx = if opts.enable_thinking == Some(false) {
            minijinja::context! {
                messages => msgs_val,
                add_generation_prompt => opts.add_generation_prompt,
                bos_token => opts.bos_token,
                eos_token => opts.eos_token,
                tools => tools_val,
                enable_thinking => false,
            }
        } else {
            minijinja::context! {
                messages => msgs_val,
                add_generation_prompt => opts.add_generation_prompt,
                bos_token => opts.bos_token,
                eos_token => opts.eos_token,
                tools => tools_val,
            }
        };

        let text = tmpl
            .render(ctx)
            .map_err(|e| Error::Other(format!("template render failed: {e}")))?;

        Ok(RenderedPrompt {
            text,
            add_generation_prompt: opts.add_generation_prompt,
        })
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "chat_template_tests.rs"]
mod chat_template_tests;
