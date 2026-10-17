// unsafe_code: String::from_utf8_unchecked over serde_json output (Python-compatible JSON formatter, ASCII-safe)
#![allow(unsafe_code)]

//! Chat-template rendering via minijinja.
//!
//! Loads `chat_template.jinja` from a model directory and renders it with
//! the given messages and options. The rendered string is ready for tokenization.
//!
//! Prompt pipeline.

#![allow(clippy::needless_pass_by_value)]
use std::path::Path;

use minijinja::{
    value::{Serde, ValueKind},
    Environment, Error as MjError, ErrorKind, Value,
};

use rmlx_core::{Error, Result};

// ── Python-compatible JSON serialiser ─────────────────────────────────────────
//
// Python's `json.dumps` uses `": "` (colon + space) and `", "` (comma +
// space) separators by default and does not escape HTML characters.
// minijinja's built-in `tojson` filter escapes `<`, `>`, `&` and `'`.
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
/// (no-tool) messages render byte-identically to a tool-free render (the
/// invariant: `{% if message.tool_calls %}` stays falsy).
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed template-context struct — six fields are the complete per-message Jinja context contract; adding a field requires updating all ChatMessageTpl construction sites"
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
    /// `assistant`-turn reasoning text as the client sent it (`None` ⇒ key
    /// omitted, so a template testing `is string` sees it undefined).
    pub reasoning_content: Option<&'a str>,
}

/// Options forwarded to the Jinja template context.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed options struct — its fields are the complete render-options contract; adding a field requires updating all RenderOpts construction sites and ChatTemplate::render"
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
    /// false, so renders without tools are byte-identical to the tool-free render.
    pub tools: &'a [serde_json::Value],
    /// The variables the request sets, as [`request_variables`] resolves them.
    /// The template reads each one as sent and alone decides which values it
    /// accepts: it refuses others with `raise_exception`
    /// ([`RenderError::Rejected`]). A variable the server sets (`messages`,
    /// `tools`, `add_generation_prompt`, `bos_token`, `eos_token`) replaces one
    /// of the same name here.
    pub variables: Option<&'a serde_json::Map<String, serde_json::Value>>,
}

/// The template variables of one request.
///
/// `chat_template_kwargs` goes to the template as sent, every key and value.
/// The top-level `enable_thinking` and the server default keep their meaning:
/// `false` is passed, and `true` leaves the template to its own default. A
/// top-level `enable_thinking` wins over the same key in `chat_template_kwargs`
/// for every value (top-level `true` removes it). `chat_template_kwargs` wins
/// over the server default. `reasoning_effort` is passed as sent and wins over
/// the same key in `chat_template_kwargs`.
#[must_use]
pub fn request_variables(
    chat_template_kwargs: Option<serde_json::Map<String, serde_json::Value>>,
    enable_thinking: Option<bool>,
    reasoning_effort: Option<&str>,
    default_enable_thinking: Option<bool>,
) -> serde_json::Map<String, serde_json::Value> {
    let mut variables = chat_template_kwargs.unwrap_or_default();
    match enable_thinking {
        Some(false) => {
            variables.insert("enable_thinking".to_owned(), serde_json::Value::Bool(false));
        }
        Some(true) => {
            variables.remove("enable_thinking");
        }
        None if default_enable_thinking == Some(false) => {
            variables
                .entry("enable_thinking")
                .or_insert(serde_json::Value::Bool(false));
        }
        None => {}
    }
    if let Some(effort) = reasoning_effort {
        variables.insert(
            "reasoning_effort".to_owned(),
            serde_json::Value::String(effort.to_owned()),
        );
    }
    variables
}

/// The variables the server sets from the request. A client variable of one of
/// these names is dropped.
const SERVER_VARIABLES: [&str; 5] = [
    "messages",
    "add_generation_prompt",
    "bos_token",
    "eos_token",
    "tools",
];

/// Why [`ChatTemplate::render`] produced no prompt.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed outcome set: the template refused the request, or the render failed"
)]
#[derive(Debug)]
pub enum RenderError {
    /// The template called `raise_exception`: it refuses the request's
    /// messages or options. Holds the template's own message, cut to 256
    /// characters.
    Rejected(String),
    /// Any other failure: compile, lookup, an unsupported construct. Holds the
    /// message cut to 256 characters, as it can copy request text.
    Failed(String),
}

impl std::fmt::Display for RenderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderError::Rejected(msg) => write!(f, "the chat template refuses the request: {msg}"),
            RenderError::Failed(msg) => write!(f, "template render failed: {msg}"),
        }
    }
}

impl std::error::Error for RenderError {}

/// The message a template passed to `raise_exception`, carried as the source
/// of the minijinja error so `render` can tell a refusal from a failure.
#[derive(Debug)]
struct TemplateRefusal(String);

impl std::fmt::Display for TemplateRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for TemplateRefusal {}

/// The most characters of an error message that can hold request text that the
/// server keeps. It goes into the reply and the log.
const MESSAGE_MAX_CHARS: usize = 256;

/// `msg` cut to [`MESSAGE_MAX_CHARS`] characters, with `…` when cut.
pub(crate) fn bounded_message(msg: &str) -> String {
    match msg.char_indices().nth(MESSAGE_MAX_CHARS) {
        Some((cut, _)) => format!("{}…", &msg[..cut]),
        None => msg.to_owned(),
    }
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
        variables: None,
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
    /// Also registers the `raise_exception(msg)` function HuggingFace gives
    /// templates: a call ends the render with [`RenderError::Rejected`].
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

        env.add_function(
            "raise_exception",
            |msg: String| -> std::result::Result<String, MjError> {
                Err(MjError::new(
                    ErrorKind::InvalidOperation,
                    "the template raised an exception",
                )
                .with_source(TemplateRefusal(msg)))
            },
        );

        // ── tojson override — Python-compatible spacing ───────────────────────
        // HF `apply_chat_template` renders tool specs via Jinja `| tojson`
        // which uses Python's `json.dumps` (separators `": "` / `", "`).
        // minijinja's built-in `tojson` escapes `<`, `>`, `&` and `'`.
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

    /// Whether the server or the template environment provides `name`: a
    /// variable the server sets, or a global of the environment (a function
    /// such as `raise_exception`). A client variable of that name replaces
    /// nothing; it is dropped.
    fn provides(&self, name: &str) -> bool {
        SERVER_VARIABLES.contains(&name) || self.env.globals().any(|(global, _)| global == name)
    }

    /// The names in `variables` that [`ChatTemplate::render`] drops. Each is a
    /// server variable or an environment global, never client text.
    pub fn dropped_variables<'v>(
        &self,
        variables: &'v serde_json::Map<String, serde_json::Value>,
    ) -> Vec<&'v str> {
        variables
            .keys()
            .map(String::as_str)
            .filter(|name| self.provides(name))
            .collect()
    }

    /// Render the template with the given messages and options.
    ///
    /// The template context exposes:
    /// - `messages` — list of `{role, content}` maps
    /// - `add_generation_prompt` — bool
    /// - `bos_token` — string
    /// - `eos_token` — string
    /// - `tools` — list of OpenAI-shaped function specs (empty = no tools)
    /// - every entry of [`RenderOpts::variables`] that the server and the
    ///   template environment do not provide
    ///   ([`ChatTemplate::dropped_variables`] names the others).
    pub fn render(
        &self,
        messages: &[ChatMessageTpl<'_>],
        opts: &RenderOpts<'_>,
    ) -> std::result::Result<RenderedPrompt, RenderError> {
        let tmpl = self
            .env
            .get_template("chat")
            .map_err(|e| RenderError::Failed(format!("template lookup failed: {e}")))?;

        // Build the messages list as a minijinja Value.
        //
        // Plain messages emit exactly {role, content} — byte-identical to
        // tool-free output. Tool keys are added ONLY
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
                    map.insert("tool_calls", Value::from(Serde(tc)));
                }
                if let Some(id) = m.tool_call_id {
                    map.insert("tool_call_id", Value::from(id));
                }
                if let Some(name) = m.name {
                    map.insert("name", Value::from(name));
                }
                if let Some(reasoning) = m.reasoning_content {
                    map.insert("reasoning_content", Value::from(reasoning));
                }
                Value::from(Serde(map))
            })
            .collect();

        let mut ctx: std::collections::BTreeMap<String, Value> = opts
            .variables
            .into_iter()
            .flatten()
            .filter(|(name, _)| !self.provides(name))
            .map(|(name, value)| (name.clone(), Value::from(Serde(value))))
            .collect();
        // An empty tools slice serialises as an empty JSON array, which Jinja
        // evaluates as falsy, so `{% if tools %}` is false when no tools are
        // present and renders byte-identically to the tool-free render.
        let server_values = [
            Value::from(msgs_val),
            Value::from(opts.add_generation_prompt),
            Value::from(opts.bos_token),
            Value::from(opts.eos_token),
            Value::from(Serde(opts.tools)),
        ];
        for (name, value) in SERVER_VARIABLES.into_iter().zip(server_values) {
            ctx.insert(name.to_owned(), value);
        }

        let text = tmpl.render(Value::from(ctx)).map_err(|e| {
            // minijinja wraps an error raised inside an include or a super block.
            let refusal = std::iter::successors(std::error::Error::source(&e), |s| s.source())
                .find_map(|s| s.downcast_ref::<TemplateRefusal>());
            match refusal {
                Some(TemplateRefusal(msg)) => RenderError::Rejected(bounded_message(msg)),
                None => RenderError::Failed(bounded_message(&e.to_string())),
            }
        })?;

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
