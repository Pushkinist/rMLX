//! A5.3: Stream parser for Qwen3.6-style `<tool_call>` blocks.
//!
//! Qwen3.6's chat_template.jinja instructs the model to emit tool calls in
//! the following shape (verified against the Qwen3.6-35B-A3B-8bit snapshot's
//! `chat_template.jinja`):
//!
//! ```text
//! <tool_call>
//! <function=example_function_name>
//! <parameter=example_parameter_1>
//! value_1
//! </parameter>
//! <parameter=example_parameter_2>
//! This is the value for the second parameter
//! that can span
//! multiple lines
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! This is NOT the canonical JSON `<tool_call>{"name":..., "arguments":...}`
//! shape — the spec doc is outdated; trust the actual template.
//!
//! Reasoning text BEFORE the `<tool_call>` is allowed and passes through
//! into `passthrough_text`. Multiple `<tool_call>` blocks may appear in
//! sequence.
//!
//! The parser is stream-friendly: pieces fed in arbitrary BPE-aligned splits
//! produce the same parse result as the same string fed all at once. It
//! maintains a small pending buffer (everything from the rightmost `<`) so
//! split tag fragments are re-assembled across `push()` calls.
//!
//! ## `allow_eof_recovery` invariant
//!
//! **Streaming** callers MUST keep `allow_eof_recovery = false` (the default).
//! This prevents the parser from claiming a complete tool call mid-stream
//! before the closing marker has actually arrived.
//!
//! **Finalize / aggregate** callers (non-streaming, after all tokens have been
//! collected) MUST call [`ToolCallStreamParser::finalize`], which flips the
//! flag to `true` and runs EOF recovery: for the `Qwen3JsonToolCall`
//! (Bonsai/Hermes) format, a truncated `<tool_call>{json(unclosed)` is
//! balanced and still yields a valid `ParsedToolCall`.
//!
//! Wired end-to-end into the decode loop: OpenAI `tool_calls` emission (A5.4)
//! and Anthropic `tool_use` emission (A5.5) are live, streaming + non-streaming,
//! with the Anthropic `stop_reason` upgrade and the tool-choice
//! constrained-schema path. Proven by the E2E `tool_call` rows
//! (Qwen3.6 XML + Bonsai Hermes-JSON, `make e2e`).

#![allow(dead_code)]
#![allow(clippy::manual_let_else)]
use serde_json::{Map, Value};

/// One parsed tool call.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed output struct — three fields are the complete parsed-tool-call contract; adding a field requires updating all ParsedToolCall construction sites in the parser"
)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedToolCall {
    /// Stable random ID assigned at parse time. Format: `call_<32-hex>`.
    pub id: String,
    /// Function name from `<function=NAME>`.
    pub name: String,
    /// Arguments as a JSON object. Each `<parameter=KEY>VALUE</parameter>`
    /// adds one key. Multi-line VALUEs keep interior newlines verbatim;
    /// the single leading / trailing newline around the value (template
    /// formatting) is stripped.
    pub arguments: Map<String, Value>,
}

/// Per-arch parser format.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — three tool-call output formats (Qwen3XmlFunction/Qwen3JsonToolCall/GemmaToolCall); adding a format requires updating arch_to_tool_call_format() and all ToolCallFormat match arms"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolCallFormat {
    /// Qwen3.6 XML-style `<function=...><parameter=...>...</parameter>`
    /// wrapped in `<tool_call>...</tool_call>`. Used by `Qwen3.6-35B-A3B`
    /// whose chat_template instructs the XML form.
    Qwen3XmlFunction,
    /// Hermes / Qwen2 JSON form: `<tool_call>{"name":...,"arguments":{...}}
    /// </tool_call>`. The inner blob is a single JSON object with `name`
    /// (string) and `arguments` (object). Used by `Qwen3ForCausalLM`
    /// snapshots whose chat_template instructs the JSON form (e.g.
    /// `Ternary-Bonsai-8B`).
    Qwen3JsonToolCall,
    /// Gemma-4 form: `<|tool_call>call:NAME{key:val,key2:val2}<tool_call|>`.
    /// Argument values use the template's custom encoding — strings wrapped
    /// in the `<|"|>` sentinel, bare numbers / booleans, nested `{}` objects
    /// and `[]` arrays. Used by `Gemma4ForConditionalGeneration`.
    ///
    /// NOTE: the `<|tool_call>` / `<tool_call|>` / `<|"|>` markers are
    /// registered as *special tokens* in the Gemma tokenizer and are
    /// stripped by `tokenizer.decode(skip_special=true)` — the engine
    /// reconstructs them from the raw token ids before feeding the parser
    /// (see `GenerationRequest::emit_tool_markers`).
    GemmaToolCall,
    // Future: Llama3PythonTag, ...
}

/// Map a model architecture string (`config.architectures[0]`) to the
/// tool-call output format that arch is known to emit, or `None` if rMLX has
/// no parser for it.
///
/// Keeping this as a string match avoids a dep cycle between
/// `rmlx-models` (where the `Architecture` enum lives) and `rmlx-server`
/// (where the parser + `ToolCallFormat` live). The set of arch strings is
/// small and stable; mirroring the strings used in
/// `rmlx-models/src/arch.rs::load_model` is sufficient.
pub fn arch_to_tool_call_format(arch_str: &str) -> Option<ToolCallFormat> {
    match arch_str {
        // Qwen3 dense and Qwen3.5 / Qwen3.6 MoE all use the same XML shape
        // documented in their chat templates.
        "Qwen3ForCausalLM"
        | "Qwen3MoeForCausalLM"
        | "Qwen3_5MoeForConditionalGeneration"
        | "MapleForCausalLM" => Some(ToolCallFormat::Qwen3XmlFunction),
        _ => None,
    }
}

/// Resolve the tool-call output format from the model's `chat_template.jinja`
/// **source text**, falling back to the coarse arch-string map.
///
/// The same architecture string can emit different tool conventions
/// depending on the snapshot's chat template (verified against the three
/// standing-gate models' templates):
///
/// - `Qwen3.6-35B-A3B` (`Qwen3_5MoeForConditionalGeneration`) — template
///   instructs the XML form `<function=NAME><parameter=KEY>…`.
/// - `Ternary-Bonsai-8B` (`Qwen3ForCausalLM`) — template instructs the
///   Hermes JSON form `<tool_call>{"name":…,"arguments":…}</tool_call>`.
/// - `gemma-4-26b-a4b-it` (`Gemma4ForConditionalGeneration`) — template
///   emits `<|tool_call>call:NAME{…}<tool_call|>`.
///
/// Detection order (first match wins; deliberately ordered so the working
/// Qwen3.6 XML path is preserved exactly — regression-safe):
///
/// 1. contains `<|tool_call>` → [`ToolCallFormat::GemmaToolCall`]
/// 2. else contains `<function=` → [`ToolCallFormat::Qwen3XmlFunction`]
/// 3. else contains `<tool_call>` AND a JSON `name`/`arguments` instruction
///    block → [`ToolCallFormat::HfToolCall`].
/// 4. else fall back to [`arch_to_tool_call_format`].
///
/// `chat_template_src` is the raw template text retained on the registry
/// entry (`ModelEntry::chat_template_src`). When it is `None` (no template
/// file) only the arch fallback applies.
pub fn detect_tool_call_format(
    chat_template_src: Option<&str>,
    arch_str: &str,
) -> Option<ToolCallFormat> {
    if let Some(src) = chat_template_src {
        if src.contains("<|tool_call>") {
            return Some(ToolCallFormat::GemmaToolCall);
        }
        if src.contains("<function=") {
            return Some(ToolCallFormat::Qwen3XmlFunction);
        }
        if src.contains("<tool_call>")
            && (src.contains("{\"name\"") || src.contains("\\\"name\\\""))
        {
            return Some(ToolCallFormat::Qwen3JsonToolCall);
        }
    }
    arch_to_tool_call_format(arch_str)
}

#[derive(Debug)]
enum ParserState {
    /// Outside any tool_call. Text goes to `passthrough_text`.
    Outside,
    /// Saw `<tool_call>`, waiting for `<function=NAME>`. Stray text between
    /// the two tags is discarded (template formatting newline).
    InToolCall,
    /// Inside the function body, scanning for `<parameter=KEY>`,
    /// `</function>`, or `</tool_call>`. Stray text discarded.
    InFunction,
    /// Inside a parameter value, accumulating into `value_buf` until
    /// `</parameter>` closes it.
    InParameter { key: String, value_buf: String },
}

#[derive(Debug)]
struct InFlightCall {
    name: String,
    arguments: Map<String, Value>,
}

/// Stream parser. Feed pieces with `push`, drain results with `take_parsed`.
///
/// ## `allow_eof_recovery` invariant
///
/// Default: `allow_eof_recovery = false`. The streaming path MUST keep it
/// false to prevent false-positive completion mid-stream. Call
/// [`finalize`](Self::finalize) once all tokens have been accumulated
/// (non-streaming path / aggregate path); it flips `allow_eof_recovery` to
/// `true` and runs EOF recovery on any still-open tool-call block.
#[derive(Debug)]
pub struct ToolCallStreamParser {
    format: ToolCallFormat,
    state: ParserState,
    /// Text emitted outside any `<tool_call>` block. Callers (A5.4 / A5.5)
    /// stream this back to the client as ordinary assistant content.
    pub passthrough_text: String,
    /// Completed tool calls, drained by `take_parsed`.
    parsed: Vec<ParsedToolCall>,
    /// Tail bytes from the last `push` that may be the start of a marker —
    /// everything from the rightmost `<` onwards. Re-scanned on next push.
    pending: String,
    /// In-progress call (Some between `<tool_call>` and `</tool_call>`).
    current: Option<InFlightCall>,
    /// Controls EOF recovery. MUST be `false` on the streaming path (prevents
    /// premature "call complete" before the end-marker arrives). Flipped to
    /// `true` by [`finalize`](Self::finalize) for the non-streaming path.
    allow_eof_recovery: bool,
}

impl ToolCallStreamParser {
    /// Create a new parser for the given tool-call wire format.
    ///
    /// `allow_eof_recovery` starts as `false` (safe for streaming callers).
    /// Call [`finalize`](Self::finalize) at end-of-stream for non-streaming paths.
    pub fn new(format: ToolCallFormat) -> Self {
        Self {
            format,
            state: ParserState::Outside,
            passthrough_text: String::new(),
            parsed: Vec::new(),
            pending: String::new(),
            current: None,
            allow_eof_recovery: false,
        }
    }

    /// Finalize the parser after all tokens have been received (non-streaming /
    /// aggregate path).
    ///
    /// This flips `allow_eof_recovery` to `true` and attempts to recover any
    /// in-flight tool-call block that was truncated by `max_tokens` / EOS:
    ///
    /// - `Qwen3JsonToolCall` (Bonsai/Hermes): if `pending` contains a
    ///   `<tool_call>{json…` that was never closed, the JSON tail is balanced
    ///   (unclosed strings/braces/brackets are closed) and the call is emitted.
    /// - Other formats: any remaining in-flight XML state is finalized best-effort.
    ///
    /// Safe to call multiple times (idempotent after first call).
    pub fn finalize(&mut self) {
        if self.allow_eof_recovery {
            return; // already finalized
        }
        self.allow_eof_recovery = true;
        self.run_eof_recovery();
    }

    /// Run EOF recovery once `allow_eof_recovery` has been set. Extracts any
    /// in-flight `Qwen3JsonToolCall` block that never received its close marker.
    fn run_eof_recovery(&mut self) {
        match self.format {
            ToolCallFormat::Qwen3JsonToolCall => {
                // The pending buffer may contain `<tool_call>{json…` without a
                // closing `</tool_call>`. Extract the JSON tail, balance
                // truncated structures, and emit the call.
                if matches!(self.state, ParserState::InToolCall) {
                    // `pending` holds the buffered body since `<tool_call>` was
                    // consumed. Trim and attempt recovery.
                    let raw = std::mem::take(&mut self.pending);
                    self.state = ParserState::Outside;
                    let trimmed = raw.trim();
                    if trimmed.starts_with('{') || trimmed.starts_with('[') {
                        let recovered =
                            balance_truncated_json(trimmed).unwrap_or_else(|| trimmed.to_owned());
                        if let Some((name, args)) = parse_hermes_json(&recovered) {
                            if !name.is_empty() {
                                tracing::debug!(
                                    name = %name,
                                    "tool_parser: EOF-recovery recovered truncated Hermes JSON call"
                                );
                                self.parsed.push(ParsedToolCall {
                                    id: new_call_id(),
                                    name,
                                    arguments: args,
                                });
                            }
                        }
                    }
                }
            }
            // Qwen3XmlFunction: partial XML state at EOS — finalize current call
            // with whatever parameters were accumulated so far.
            ToolCallFormat::Qwen3XmlFunction => {
                if !matches!(self.state, ParserState::Outside) {
                    self.finalize_current_call();
                    self.state = ParserState::Outside;
                }
            }
            // GemmaToolCall: custom grammar inside the block — no balancer;
            // a truncated Gemma call is best dropped.
            ToolCallFormat::GemmaToolCall => {}
        }
    }

    /// True if currently inside a tool_call block (parsed but not yet closed).
    pub fn in_tool_call(&self) -> bool {
        !matches!(self.state, ParserState::Outside)
    }

    /// True if at least one complete tool_call has been parsed.
    pub fn has_calls(&self) -> bool {
        !self.parsed.is_empty()
    }

    /// Drain completed tool calls. Call once at end of stream.
    pub fn take_parsed(&mut self) -> Vec<ParsedToolCall> {
        std::mem::take(&mut self.parsed)
    }

    /// Feed a piece of decoded text. May span any portion of a tag.
    pub fn push(&mut self, piece: &str) {
        match self.format {
            ToolCallFormat::Qwen3XmlFunction => {
                self.pending.push_str(piece);
                // Loop: try to consume a marker. Each successful consume
                // changes state and advances pending. When no marker fully
                // fits, flush the safe prefix (everything before the
                // rightmost `<`) and stop.
                loop {
                    if !self.try_consume_one() {
                        break;
                    }
                }
                // Flush the safe prefix (everything before the rightmost `<`).
                self.flush_safe_prefix();
            }
            ToolCallFormat::Qwen3JsonToolCall => self.push_delimited(piece, ToolBlockKind::Json),
            ToolCallFormat::GemmaToolCall => self.push_delimited(piece, ToolBlockKind::Gemma),
        }
    }

    /// Streaming consumer for the two delimited-block formats
    /// (`Qwen3JsonToolCall`, `GemmaToolCall`). Both share the same shape:
    /// reasoning / content text outside an OPEN..CLOSE marker pair passes
    /// through; everything between the markers is buffered raw and decoded
    /// once on CLOSE by the format-specific [`Self::finalize_block`].
    ///
    /// Stream-safe: an OPEN / CLOSE marker split across `push` calls is
    /// re-assembled because the tail from the rightmost `<` is retained in
    /// `pending` until it is proven not to be a partial marker.
    #[allow(
        clippy::wildcard_enum_match_arm,
        reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
    )]
    fn push_delimited(&mut self, piece: &str, kind: ToolBlockKind) {
        self.pending.push_str(piece);
        let (open, close) = kind.markers();
        loop {
            match &self.state {
                ParserState::Outside => {
                    if let Some(idx) = self.pending.find(open) {
                        let prefix: String = self.pending.drain(..idx).collect();
                        self.passthrough_text.push_str(&prefix);
                        self.pending.drain(..open.len());
                        self.state = ParserState::InToolCall;
                        continue;
                    }
                    // No OPEN yet. Flush everything that cannot be the start
                    // of a future OPEN marker (retain the rightmost
                    // `<`-anchored viable prefix).
                    let cut = safe_flush_cut(&self.pending, &[open]);
                    if cut > 0 {
                        let prefix: String = self.pending.drain(..cut).collect();
                        self.passthrough_text.push_str(&prefix);
                    }
                    break;
                }
                ParserState::InToolCall => {
                    if let Some(idx) = self.pending.find(close) {
                        let body: String = self.pending.drain(..idx).collect();
                        self.pending.drain(..close.len());
                        self.finalize_block(&body, kind);
                        self.state = ParserState::Outside;
                        continue;
                    }
                    // CLOSE not seen yet — keep buffering the whole body in
                    // `pending` (it is raw block content, not passthrough).
                    break;
                }
                // The delimited formats only use Outside / InToolCall.
                _ => {
                    self.state = ParserState::Outside;
                }
            }
        }
    }

    /// Decode one finished tool-call block body into a [`ParsedToolCall`].
    /// Best-effort: a malformed block is dropped (logged at debug) rather
    /// than aborting the stream — mirrors the Qwen3 path's defensive
    /// "drop degenerate block" behaviour.
    fn finalize_block(&mut self, body: &str, kind: ToolBlockKind) {
        let parsed = match kind {
            ToolBlockKind::Json => parse_hermes_json(body),
            ToolBlockKind::Gemma => parse_gemma_call(body),
        };
        match parsed {
            Some((name, arguments)) if !name.is_empty() => {
                self.parsed.push(ParsedToolCall {
                    id: new_call_id(),
                    name,
                    arguments,
                });
            }
            _ => {
                tracing::debug!(
                    kind = ?kind,
                    body = %body,
                    "tool_parser: dropped unparseable tool-call block"
                );
            }
        }
    }

    /// Attempt to consume exactly one marker from the head of `pending`.
    /// Returns `true` if a marker was consumed (state may have changed),
    /// `false` if no further consume is possible right now.
    fn try_consume_one(&mut self) -> bool {
        match self.format {
            ToolCallFormat::Qwen3XmlFunction => self.try_consume_qwen3(),
            // The delimited formats never reach here — `push` dispatches
            // them straight to `push_delimited`. Kept exhaustive for safety.
            ToolCallFormat::Qwen3JsonToolCall | ToolCallFormat::GemmaToolCall => false,
        }
    }

    fn try_consume_qwen3(&mut self) -> bool {
        match &self.state {
            ParserState::Outside => {
                // Look for `<tool_call>` at the start (after we've already
                // flushed safe prefix on the previous iteration). If it's
                // not at the start, we need to find it inside pending.
                if let Some(idx) = self.pending.find("<tool_call>") {
                    // Flush text before the marker to passthrough.
                    let prefix: String = self.pending.drain(..idx).collect();
                    self.passthrough_text.push_str(&prefix);
                    // Consume the marker itself.
                    self.pending.drain(..LEN_TOOL_CALL_OPEN);
                    self.state = ParserState::InToolCall;
                    self.current = Some(InFlightCall {
                        name: String::new(),
                        arguments: Map::new(),
                    });
                    return true;
                }
                false
            }
            ParserState::InToolCall => {
                // Need `<function=NAME>` (primary) or `</tool_call>`
                // (defensive — empty block). Whichever appears earliest.
                let fn_idx = self.pending.find("<function=");
                let tc_close_idx = self.pending.find("</tool_call>");
                match (fn_idx, tc_close_idx) {
                    (Some(fi), Some(ti)) if ti < fi => {
                        self.pending.drain(..ti + LEN_TOOL_CALL_CLOSE);
                        self.finalize_current_call();
                        self.state = ParserState::Outside;
                        true
                    }
                    (Some(fi), _) => {
                        self.pending.drain(..fi);
                        if let Some(end_rel) = self.pending[LEN_FUNCTION_PREFIX..].find('>') {
                            let name = self.pending
                                [LEN_FUNCTION_PREFIX..LEN_FUNCTION_PREFIX + end_rel]
                                .to_string();
                            let total = LEN_FUNCTION_PREFIX + end_rel + 1;
                            self.pending.drain(..total);
                            if let Some(call) = self.current.as_mut() {
                                call.name = name;
                            }
                            self.state = ParserState::InFunction;
                            true
                        } else {
                            // Have `<function=` but no `>` yet — wait.
                            false
                        }
                    }
                    (None, Some(ti)) => {
                        self.pending.drain(..ti + LEN_TOOL_CALL_CLOSE);
                        self.finalize_current_call();
                        self.state = ParserState::Outside;
                        true
                    }
                    (None, None) => false,
                }
            }
            ParserState::InFunction => {
                // Look for the earliest of `<parameter=…>`, `</function>`,
                // `</tool_call>`. Use a helper to find earliest occurrence.
                let p_idx = self.pending.find("<parameter=");
                let f_close_idx = self.pending.find("</function>");
                let tc_close_idx = self.pending.find("</tool_call>");

                let (which, idx) = match earliest(&[
                    p_idx.map(|i| (MarkerKind::ParamOpen, i)),
                    f_close_idx.map(|i| (MarkerKind::FunctionClose, i)),
                    tc_close_idx.map(|i| (MarkerKind::ToolCallClose, i)),
                ]) {
                    Some(v) => v,
                    None => return false,
                };

                // Discard inter-tag whitespace.
                self.pending.drain(..idx);
                match which {
                    MarkerKind::ParamOpen => {
                        // Need a `>` to close the open tag.
                        if let Some(end_rel) = self.pending[LEN_PARAM_PREFIX..].find('>') {
                            let key = self.pending[LEN_PARAM_PREFIX..LEN_PARAM_PREFIX + end_rel]
                                .to_string();
                            let total = LEN_PARAM_PREFIX + end_rel + 1;
                            self.pending.drain(..total);
                            self.state = ParserState::InParameter {
                                key,
                                value_buf: String::new(),
                            };
                            true
                        } else {
                            // Have prefix but no `>` yet — wait.
                            false
                        }
                    }
                    MarkerKind::FunctionClose => {
                        self.pending.drain(..LEN_FUNCTION_CLOSE);
                        // Stay in InFunction; expect </tool_call> next.
                        // Use a marker state — we treat </function> as a
                        // noop close marker; on next iteration we'll find
                        // </tool_call> or unexpected text.
                        true
                    }
                    MarkerKind::ToolCallClose => {
                        self.pending.drain(..LEN_TOOL_CALL_CLOSE);
                        self.finalize_current_call();
                        self.state = ParserState::Outside;
                        true
                    }
                }
            }
            ParserState::InParameter { .. } => {
                // Look for `</parameter>` only. Everything else accumulates
                // into the value buffer. The value buffer is updated by
                // `flush_safe_prefix` — here we just consume the close tag
                // when it's available.
                if let Some(idx) = self.pending.find("</parameter>") {
                    // The chunk up to idx is value content.
                    let value_chunk: String = self.pending.drain(..idx).collect();
                    self.pending.drain(..LEN_PARAM_CLOSE);
                    if let ParserState::InParameter { key, value_buf } = &mut self.state {
                        value_buf.push_str(&value_chunk);
                        // Strip the single leading / trailing newline that
                        // the template puts around the value.
                        let trimmed = strip_one_outer_newline(value_buf);
                        if let Some(call) = self.current.as_mut() {
                            call.arguments
                                .insert(key.clone(), Value::String(trimmed.to_string()));
                        }
                    }
                    self.state = ParserState::InFunction;
                    return true;
                }
                false
            }
        }
    }

    /// Flush bytes from `pending` that cannot possibly be the start of a
    /// future marker, sending them to the correct destination for the
    /// current state.
    ///
    /// Rule: all markers begin with `<`. The retained tail must be a
    /// `<`-anchored substring that is a prefix of at least one expected
    /// marker for the current state. Everything else flushes.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn flush_safe_prefix(&mut self) {
        // Find the largest `cut` such that `pending[cut..]` is a viable
        // partial marker for the current state. Walk from the rightmost `<`
        // forward; if that suffix isn't a viable prefix, advance past it.
        let pending_bytes = self.pending.as_bytes();
        let mut cut = self.pending.len();
        let mut search_from = 0;
        while let Some(rel) = pending_bytes[search_from..].iter().position(|&b| b == b'<') {
            let candidate = search_from + rel;
            let tail = &self.pending[candidate..];
            if self.is_viable_marker_prefix(tail) {
                cut = candidate;
                break;
            }
            // Not a viable prefix — skip past this `<` and try again.
            search_from = candidate + 1;
        }
        // If no `<` viable, cut stays = pending.len() => flush all.
        if cut == 0 {
            return;
        }
        let prefix: String = self.pending.drain(..cut).collect();
        match &mut self.state {
            ParserState::Outside => self.passthrough_text.push_str(&prefix),
            ParserState::InToolCall | ParserState::InFunction => {
                // Inter-tag template whitespace — discard.
            }
            ParserState::InParameter { value_buf, .. } => {
                value_buf.push_str(&prefix);
            }
        }
    }

    /// Is `tail` a non-empty prefix of any marker expected in the current
    /// state? Used to decide whether to keep `tail` buffered.
    fn is_viable_marker_prefix(&self, tail: &str) -> bool {
        let expected: &[&str] = match self.state {
            ParserState::Outside => &[TOOL_CALL_OPEN],
            ParserState::InToolCall => &[FUNCTION_PREFIX, TOOL_CALL_CLOSE],
            ParserState::InFunction => &[PARAM_PREFIX, FUNCTION_CLOSE, TOOL_CALL_CLOSE],
            ParserState::InParameter { .. } => &[PARAM_CLOSE],
        };
        for marker in expected {
            // `tail` is a viable prefix if either it's a prefix of `marker`
            // (still incoming), or it starts with `marker` (complete, will
            // be consumed on the next try_consume_one).
            if marker.starts_with(tail) || tail.starts_with(marker) {
                return true;
            }
            // For variable-length open tags (`<function=`, `<parameter=`),
            // tail may extend past the prefix while we wait for `>`.
            if (*marker == FUNCTION_PREFIX || *marker == PARAM_PREFIX) && tail.starts_with(marker) {
                return true;
            }
        }
        false
    }

    fn finalize_current_call(&mut self) {
        if let Some(call) = self.current.take() {
            // Only emit if at least a name was captured (defensive — empty
            // <tool_call></tool_call> blocks are degenerate, drop them).
            if !call.name.is_empty() {
                self.parsed.push(ParsedToolCall {
                    id: new_call_id(),
                    name: call.name,
                    arguments: call.arguments,
                });
            }
        }
    }
}

// ── Constants and helpers ─────────────────────────────────────────────────────

const TOOL_CALL_OPEN: &str = "<tool_call>";
const TOOL_CALL_CLOSE: &str = "</tool_call>";
const FUNCTION_PREFIX: &str = "<function=";
const FUNCTION_CLOSE: &str = "</function>";
const PARAM_PREFIX: &str = "<parameter=";
const PARAM_CLOSE: &str = "</parameter>";

const LEN_TOOL_CALL_OPEN: usize = TOOL_CALL_OPEN.len();
const LEN_TOOL_CALL_CLOSE: usize = TOOL_CALL_CLOSE.len();
const LEN_FUNCTION_PREFIX: usize = FUNCTION_PREFIX.len();
const LEN_FUNCTION_CLOSE: usize = FUNCTION_CLOSE.len();
const LEN_PARAM_PREFIX: usize = PARAM_PREFIX.len();
const LEN_PARAM_CLOSE: usize = PARAM_CLOSE.len();

#[derive(Debug, Clone, Copy)]
enum MarkerKind {
    ParamOpen,
    FunctionClose,
    ToolCallClose,
}

fn earliest(opts: &[Option<(MarkerKind, usize)>]) -> Option<(MarkerKind, usize)> {
    let mut best: Option<(MarkerKind, usize)> = None;
    for o in opts.iter().copied().flatten() {
        if best.is_none_or(|(_, bi)| o.1 < bi) {
            best = Some(o);
        }
    }
    best
}

fn strip_one_outer_newline(s: &str) -> &str {
    let s = s.strip_prefix('\n').unwrap_or(s);
    s.strip_suffix('\n').unwrap_or(s)
}

pub(crate) fn new_call_id() -> String {
    let hex = uuid::Uuid::new_v4().simple().to_string();
    format!("call_{hex}")
}

/// Balance a JSON string truncated by `max_tokens` / EOS. Walks the input
/// tracking string state and brace/bracket nesting; on EOF closes any open
/// string and pops outstanding closers.
///
/// Returns `Some(repaired)` only when at least one closer was appended (the
/// input was actually truncated). Returns `None` when the input is already
/// syntactically complete so callers can avoid re-parsing.
///
/// Ported from `dynamo/lib/parsers/src/tool_calling/json/base_json_parser.rs`
/// (`try_repair_truncated_json`). Used by the [`ToolCallStreamParser::finalize`]
/// EOF-recovery path for `Qwen3JsonToolCall` (Bonsai/Hermes format).
pub(crate) fn balance_truncated_json(s: &str) -> Option<String> {
    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape = false;
    for c in s.chars() {
        if escape {
            escape = false;
            continue;
        }
        if in_string {
            match c {
                '\\' => escape = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => stack.push('}'),
            '[' => stack.push(']'),
            '}' | ']' => {
                stack.pop();
            }
            _ => {}
        }
    }
    if !escape && !in_string && stack.is_empty() {
        return None; // already complete
    }
    let mut repaired = s.to_string();
    // EOF mid-escape: pair the trailing `\` with another `\` so the
    // closing quote we append next isn't itself escaped.
    if escape {
        repaired.push('\\');
    }
    if in_string {
        repaired.push('"');
    }
    while let Some(closer) = stack.pop() {
        repaired.push(closer);
    }
    Some(repaired)
}

// ── Delimited-block formats (Hermes JSON / Gemma) ─────────────────────────────

/// The two delimited-block tool-call formats parsed by [`push_delimited`].
///
/// [`ToolCallStreamParser::push_delimited`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ToolBlockKind {
    /// Hermes / Qwen2 JSON: `<tool_call>{...}</tool_call>`.
    Json,
    /// Gemma-4: `<|tool_call>call:NAME{...}<tool_call|>`.
    Gemma,
}

impl ToolBlockKind {
    /// `(open_marker, close_marker)` for this block kind.
    fn markers(self) -> (&'static str, &'static str) {
        match self {
            ToolBlockKind::Json => (TOOL_CALL_OPEN, TOOL_CALL_CLOSE),
            ToolBlockKind::Gemma => (GEMMA_TOOL_CALL_OPEN, GEMMA_TOOL_CALL_CLOSE),
        }
    }
}

const GEMMA_TOOL_CALL_OPEN: &str = "<|tool_call>";
const GEMMA_TOOL_CALL_CLOSE: &str = "<tool_call|>";
/// Sentinel that the Gemma chat template wraps string argument values in
/// (`format_argument` macro: `'<|"|>' + value + '<|"|>'`).
const GEMMA_STR_SENTINEL: &str = "<|\"|>";

/// Largest prefix length of `s` that is safe to flush as plain text given
/// the set of `markers` whose start we must not split across the buffer
/// boundary. Returns the byte offset of the rightmost `<`-anchored suffix
/// that is still a viable (possibly partial) prefix of some marker; bytes
/// before that offset cannot begin any marker and are safe to emit.
///
/// Mirrors the invariant the Qwen3 path's `flush_safe_prefix` enforces, but
/// generic over an explicit marker list (the delimited formats have a
/// single OPEN marker to guard against).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
fn safe_flush_cut(s: &str, markers: &[&str]) -> usize {
    let bytes = s.as_bytes();
    let mut search_from = 0;
    while let Some(rel) = bytes[search_from..].iter().position(|&b| b == b'<') {
        let candidate = search_from + rel;
        let tail = &s[candidate..];
        let viable = markers
            .iter()
            .any(|m| m.starts_with(tail) || tail.starts_with(*m));
        if viable {
            return candidate;
        }
        search_from = candidate + 1;
    }
    // No `<` can begin a marker — the whole buffer is safe to flush.
    s.len()
}

/// Parse a Hermes / Qwen2 JSON tool-call body: a single JSON object with
/// `name` (string) and `arguments` (object). Returns `None` if the body is
/// not valid JSON or is missing a string `name`. `arguments` defaults to an
/// empty object when absent or non-object (lenient — some models emit a
/// JSON-encoded string there; callers re-serialize anyway).
///
/// Reference: `mlx-lm/mlx_lm/tool_parsers/json_tools.py` (`json.loads` of
/// the inner text, take `name` + `arguments`).
fn parse_hermes_json(body: &str) -> Option<(String, Map<String, Value>)> {
    let v: Value = serde_json::from_str(body.trim()).ok()?;
    let obj = v.as_object()?;
    let name = obj.get("name")?.as_str()?.to_owned();
    let arguments = match obj.get("arguments") {
        Some(Value::Object(m)) => m.clone(),
        // Some models emit `"arguments": "{\"k\":1}"` (stringified JSON).
        Some(Value::String(s)) => serde_json::from_str::<Value>(s)
            .ok()
            .and_then(|v| v.as_object().cloned())
            .unwrap_or_default(),
        _ => Map::new(),
    };
    Some((name, arguments))
}

/// Parse a Gemma-4 tool-call body: `call:NAME{key:val,key2:val2}`.
///
/// The argument list uses the chat template's `format_argument` encoding
/// (verified against
/// `mlx-community__gemma-4-26b-a4b-it-mxfp8/chat_template.jinja`):
///
/// - string → `<|"|>text<|"|>` (the `<|"|>` sentinel; `text` is verbatim)
/// - boolean → bare `true` / `false`
/// - null → bare `null`
/// - number → bare numeric literal
/// - object → `{key:val,...}` (keys are NOT sentinel-wrapped — tool-call
///   args render with `escape_keys=False`)
/// - array → `[item,item,...]`
///
/// Decodes that into a JSON arguments object. Returns `None` if the
/// `call:NAME{...}` shell is malformed.
fn parse_gemma_call(body: &str) -> Option<(String, Map<String, Value>)> {
    let body = body.trim();
    let rest = body.strip_prefix("call:")?;
    let brace = rest.find('{')?;
    let name = rest[..brace].trim().to_owned();
    if name.is_empty() {
        return None;
    }
    // The args span is `{ .. }` — take the matching outer braces.
    let args_src = &rest[brace..];
    let inner = strip_matching_braces(args_src)?;
    let mut p = GemmaArgParser::new(inner);
    let value = p.parse_object()?;
    let arguments = value.as_object()?.clone();
    Some((name, arguments))
}

/// If `s` starts with `{` return the slice between it and the matching `}`
/// (sentinel-aware: braces inside a `<|"|>...<|"|>` string do not nest).
#[allow(
    clippy::indexing_slicing,
    reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
)]
fn strip_matching_braces(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    if bytes.first() != Some(&b'{') {
        return None;
    }
    let mut depth = 0usize;
    let mut i = 0usize;
    while i < s.len() {
        if s[i..].starts_with(GEMMA_STR_SENTINEL) {
            // Skip an entire sentinel-delimited string verbatim.
            let after_open = i + GEMMA_STR_SENTINEL.len();
            let close_rel = s[after_open..].find(GEMMA_STR_SENTINEL)?;
            i = after_open + close_rel + GEMMA_STR_SENTINEL.len();
            continue;
        }
        match bytes[i] {
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[1..i]);
                }
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// Recursive-descent decoder for the Gemma argument value grammar. Operates
/// on the raw inner text (between the outer `{` `}`); not streaming — the
/// whole block is buffered before this runs.
struct GemmaArgParser<'a> {
    s: &'a str,
    pos: usize,
}

impl<'a> GemmaArgParser<'a> {
    fn new(s: &'a str) -> Self {
        Self { s, pos: 0 }
    }

    fn rest(&self) -> &'a str {
        &self.s[self.pos..]
    }

    fn skip_ws(&mut self) {
        while let Some(c) = self.rest().chars().next() {
            if c.is_whitespace() {
                self.pos += c.len_utf8();
            } else {
                break;
            }
        }
    }

    /// Parse a comma-separated `key:value` list (no surrounding braces) into
    /// a JSON object. The caller supplies the inner span.
    fn parse_object(&mut self) -> Option<Value> {
        let mut map = Map::new();
        self.skip_ws();
        if self.rest().is_empty() {
            return Some(Value::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_key()?;
            self.skip_ws();
            if !self.rest().starts_with(':') {
                return None;
            }
            self.pos += 1; // ':'
            self.skip_ws();
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            if self.rest().starts_with(',') {
                self.pos += 1;
                continue;
            }
            break;
        }
        Some(Value::Object(map))
    }

    /// A key is either a sentinel-wrapped string or a bare identifier up to
    /// the next `:`.
    fn parse_key(&mut self) -> Option<String> {
        if self.rest().starts_with(GEMMA_STR_SENTINEL) {
            return self.parse_sentinel_string();
        }
        let rel = self.rest().find(':')?;
        let key = self.rest()[..rel].trim().to_owned();
        self.pos += rel;
        if key.is_empty() {
            None
        } else {
            Some(key)
        }
    }

    fn parse_value(&mut self) -> Option<Value> {
        self.skip_ws();
        let r = self.rest();
        if r.starts_with(GEMMA_STR_SENTINEL) {
            return self.parse_sentinel_string().map(Value::String);
        }
        if r.starts_with('{') {
            // Nested object: take the matching-brace span and recurse.
            let span = strip_matching_braces(r)?;
            let consumed = span.len() + 2; // include both braces
            self.pos += consumed;
            let mut sub = GemmaArgParser::new(span);
            return sub.parse_object();
        }
        if r.starts_with('[') {
            return self.parse_array();
        }
        // Bare scalar: read up to the next top-level `,` `}` `]`.
        let end = r.find([',', '}', ']']).unwrap_or(r.len());
        let raw = r[..end].trim();
        self.pos += end;
        Some(scalar_to_json(raw))
    }

    fn parse_array(&mut self) -> Option<Value> {
        // Consume '['.
        self.pos += 1;
        let mut items = Vec::new();
        loop {
            self.skip_ws();
            if self.rest().starts_with(']') {
                self.pos += 1;
                break;
            }
            let v = self.parse_value()?;
            items.push(v);
            self.skip_ws();
            if self.rest().starts_with(',') {
                self.pos += 1;
                continue;
            }
            if self.rest().starts_with(']') {
                self.pos += 1;
                break;
            }
            return None;
        }
        Some(Value::Array(items))
    }

    /// Parse `<|"|>...<|"|>` returning the inner text verbatim. Assumes the
    /// cursor is at the opening sentinel.
    fn parse_sentinel_string(&mut self) -> Option<String> {
        let r = self.rest();
        let after_open = GEMMA_STR_SENTINEL.len();
        let close_rel = r[after_open..].find(GEMMA_STR_SENTINEL)?;
        let inner = r[after_open..after_open + close_rel].to_owned();
        self.pos += after_open + close_rel + GEMMA_STR_SENTINEL.len();
        Some(inner)
    }
}

/// Decode a bare Gemma scalar (`true` / `false` / `null` / number / else
/// treat as an unquoted string — the robust-fallback behaviour from
/// `oMLX/omlx/api/tool_calling.py::_gemma4_args_to_json_robust`).
fn scalar_to_json(raw: &str) -> Value {
    match raw {
        "true" => return Value::Bool(true),
        "false" => return Value::Bool(false),
        "null" => return Value::Null,
        _ => {}
    }
    if let Ok(n) = serde_json::from_str::<serde_json::Number>(raw) {
        return Value::Number(n);
    }
    Value::String(raw.to_owned())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[path = "tool_parser_tests.rs"]
mod tool_parser_tests;
