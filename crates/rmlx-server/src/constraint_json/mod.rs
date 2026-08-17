//! A6 — tokenizer-aware JSON-syntax + JSON-Schema constraint engine for
//! `response_format: {"type":"json_object"}` (A6.3) and
//! `response_format: {"type":"json_schema", "json_schema":{…}}` (A6.4/A6.5).
//!
//! # Two-layer design
//!
//! 1. **State machine** (pure Rust, no MLX deps): a character-level JSON

//!
//! 2. **Token prefix map**: at constraint construction we decode every token

//!
//! # EOS gating
//!
//! Special / control tokens — including EOS — typically decode to an empty
//! or zero first-byte; the state machine never allows byte value 0, so they
//! are naturally masked-out mid-JSON. At terminal states we *additionally*
//! force the EOS ids to `true` in the mask so the decode loop's EOS-stop
//! predicate can fire.
//!
//! # Hot-path cost
//!
//! Construction: one `tokenizer.decode` call per vocab id. For Qwen3.6's
//! ~152K vocab this is ~600 ms on the M5 Max (one-time per constrained
//! request, hidden behind TTFT). Per-step mask build: O(vocab) byte-set
//! membership test, ~1 ms on Qwen3.6. The hot UNCONSTRAINED path never
//! reaches this file.
//!
//! # A6 SUPPORTED / DEGRADED matrix
//!
//! ## SUPPORTED (fully enforced)
//!
//! | Feature | Notes |
//! |---|---|
//! | `json_object` | Any syntactically valid JSON at top level. |
//! | `json_schema` `type:object` | Properties in schema-declared order; `required`; `additionalProperties`. |
//! | `json_schema` `type:array` | Homogeneous `items`; `minItems`/`maxItems`. |
//! | `json_schema` `type:string` | Free-form or restricted by `enum`. |
//! | `json_schema` `type:number` / `type:integer` | Integer forbids `.`/`e`/`E`. |
//! | `json_schema` `type:boolean` / `type:null` | Exact literal. |
//! | `enum` (all-string) | Trie-forced to one of the listed literals. |
//! | `const` | Exact JSON serialisation forced via trie. |
//! | `oneOf`/`anyOf` (discriminated) | Union of literals → merged trie. |
//! | `strict` mode | All properties required, `additionalProperties:false`. |
//! | Think-phase warm-up | `<think>…</think>` blocks on Qwen3/DeepSeek-R1 pass through without engaging. |
//! | Scalar-root `Immediate` engage (A6.5) | Scalar-root schemas engage at the **first post-think token** regardless of its bytes; no waiting for `{`/`[` that never come. |
//! | Markdown-fence suppression (A6.5) | Leading ` ```json\n ` wrapper stripped from `content` in both blocking and streaming paths. |
//!
//! ## A6.4 schema keyword coverage — see [`schema`] module docs for the
//! full gap table (every keyword × status × strict-mode decision).
//!
//! Newly enforced: local `$ref` → `#/$defs/…` / `#/definitions/…`
//! resolution, `$defs`/`definitions`, single-branch `allOf` flattening.
//!
//! ## DEGRADED (non-strict: accepted with warn, narrowing not enforced;
//! ## strict: HTTP 400 `unsupported_schema_keyword`)
//!
//! | Feature | Non-strict behaviour |
//! |---|---|
//! | `pattern`, `format` | Keyword dropped; bare type enforced. |
//! | `minimum`/`maximum`, `minLength`/`maxLength` | Keyword dropped. |
//! | `prefixItems` / tuple `items` | Degrades to `Any`-items array. |
//! | multi-branch `allOf`, `not`, `if`/`then`/`else`, `unevaluatedProperties` | Degrade node to `Any`. |
//! | `additionalProperties` schema (non-bool) | Treated as permissive (value type not checked). |
//!
//! ## ENFORCED with documented simplification (no mis-accept)
//!
//! | Feature | Behaviour |
//! |---|---|
//! | Free-order object keys | Keys emitted in schema-declared order only (one valid ordering). |
//! | Structural `oneOf`/`anyOf` (mixed containers) | First-byte branch selection; no full backtracking. |
//!
//! ## ERROR (both modes — schema cannot be honoured)
//!
//! | Feature | Behaviour |
//! |---|---|
//! | Remote / dangling `$ref` | HTTP 400 `invalid_request_error` (`UnresolvableRef`). |

#![allow(
    clippy::items_after_statements,
    clippy::missing_fields_in_debug,
    clippy::needless_pass_by_value
)]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rmlx_models::ConstraintEngine;

pub mod schema;

pub use schema::{EngagePolicy, SchemaConstraint, SchemaError, SchemaNode};

/// Whitespace bytes allowed outside strings.
const WS: [u8; 4] = *b" \t\n\r";

/// Longest run of consecutive *insignificant* whitespace bytes either JSON
/// engine accepts between two structural tokens.
///
/// JSON itself puts no bound on that run, and an unbounded one is not a
/// harmless permissiveness: a greedy (temp=0) decoder whose highest-scoring
/// continuation is a whitespace piece will re-pick it forever, because the
/// mask keeps offering whitespace and keeps withholding EOS (the value is not
/// complete, so EOS is illegal). The request then runs to `max_tokens` and
/// returns HTTP 200 carrying nothing but indentation.
///
/// Bounding the run removes the cycle without removing any *value*: once the
/// bound is hit the mask stops offering whitespace and the model must emit
/// the next structural byte, so every JSON document is still reachable — only
/// its indentation is clipped. This mirrors the bounded `space` rule llama.cpp
/// generates from a JSON schema for the same reason.
///
/// The bound is per structural position: any content or structural byte resets
/// the counter, so a pretty-printed document pays it once per newline. The run
/// at a structural position is `1 + indent_width * depth`, so 64 clips 4-space
/// indentation only past depth 15 and 2-space past depth 31 — deep enough that
/// no realistic document is reformatted, and still finite, which is all
/// termination needs.
pub(crate) const MAX_INSIGNIFICANT_WS_RUN: u32 = 64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum JsonState {
    /// Top-level — expecting any JSON value to start.
    Start,
    /// Inside a string literal (after `"`). The `is_key` flag tells
    /// `after_string` whether to transition to `ObjectExpectColon` (key)
    /// or `AfterValue`/`Done` (value).
    InString { is_key: bool },
    /// Just consumed `\` inside a string.
    InStringEscape { is_key: bool },
    /// Inside `\u`, expecting `n_hex` more hex digits (0..4).
    InStringUnicode { is_key: bool, n_consumed: u8 },
    /// Inside a numeric value (any of digits / `.` / `e` / `E` / `+` / `-`).
    InNumber,
    /// Matching `true` / `false` / `null` at index `pos`.
    InLiteral { which: LiteralKind, pos: u8 },
    /// Inside `{...}` — expecting `"` key or `}` close.
    ObjectExpectKeyOrEnd { just_after_comma: bool },
    /// After an object key string, expecting `:`.
    ObjectExpectColon,
    /// Inside `[...]` — expecting a value or `]` close.
    ArrayExpectValueOrEnd { just_after_comma: bool },
    /// After a value, expecting `,` / `}` / `]`.
    AfterValue,
    /// Top-level value complete. Only whitespace allowed by the grammar;
    /// EOS is added to the mask layer separately.
    Done,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LiteralKind {
    True,
    False,
    Null,
}

impl LiteralKind {
    fn bytes(self) -> &'static [u8] {
        match self {
            LiteralKind::True => b"true",
            LiteralKind::False => b"false",
            LiteralKind::Null => b"null",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Frame {
    Object,
    Array,
}

/// Pure-Rust JSON state machine. Feed bytes via [`JsonGrammar::step`];
/// query the byte-set legal at the current position via
/// [`JsonGrammar::allowed_bytes`].
#[derive(Debug, Clone)]
pub struct JsonGrammar {
    state: JsonState,
    stack: Vec<Frame>,
    /// Consecutive insignificant-whitespace bytes accepted since the last
    /// content or structural byte. Capped at [`MAX_INSIGNIFICANT_WS_RUN`] so a
    /// greedy decoder cannot sit in the whitespace no-op forever.
    ws_run: u32,
}

impl Default for JsonGrammar {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonGrammar {
    /// Create a new grammar starting in the top-level value state.
    pub fn new() -> Self {
        Self {
            state: JsonState::Start,
            stack: Vec::new(),
            ws_run: 0,
        }
    }

    /// Return `true` when the grammar has consumed a complete top-level JSON value.
    pub fn is_done(&self) -> bool {
        // A top-level open number is considered complete: the masker would
        // only allow EOS at this point anyway (followers like `,` / `}` /
        // `]` require an enclosing frame). Treat InNumber-at-top-level as
        // Done for the `is_done` query so termination tests work on bare
        // literals like `42` and `-1.5`.
        if self.state == JsonState::Done {
            return true;
        }
        if self.state == JsonState::InNumber && self.stack.is_empty() {
            return true;
        }
        false
    }

    fn in_object(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Object))
    }

    fn in_array(&self) -> bool {
        matches!(self.stack.last(), Some(Frame::Array))
    }

    /// Set of bytes legal as the next byte in the current state.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub fn allowed_bytes(&self) -> [bool; 256] {
        let mut out = [false; 256];
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
        )]
        fn add_ws(out: &mut [bool; 256]) {
            for &b in &WS {
                out[b as usize] = true;
            }
        }
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
        )]
        fn add_value_starters(out: &mut [bool; 256]) {
            for &b in b"{[\"tfn-" {
                out[b as usize] = true;
            }
            for b in b'0'..=b'9' {
                out[b as usize] = true;
            }
        }
        match self.state {
            JsonState::Start => {
                add_ws(&mut out);
                add_value_starters(&mut out);
            }
            JsonState::ArrayExpectValueOrEnd { just_after_comma } => {
                add_ws(&mut out);
                add_value_starters(&mut out);
                if !just_after_comma {
                    out[b']' as usize] = true;
                }
            }
            JsonState::InString { .. } => {
                // Any non-control byte; `"` and `\` are state transitions.
                // C0 control bytes must be escaped, so `step` rejects them.
                for b in 0x20u16..=255 {
                    out[b as usize] = true;
                }
            }
            JsonState::InStringEscape { .. } => {
                for &b in b"\"\\/bfnrtu" {
                    out[b as usize] = true;
                }
            }
            JsonState::InStringUnicode { .. } => {
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
                for b in b'a'..=b'f' {
                    out[b as usize] = true;
                }
                for b in b'A'..=b'F' {
                    out[b as usize] = true;
                }
            }
            JsonState::InNumber => {
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
                for &b in b".eE+-" {
                    out[b as usize] = true;
                }
                add_ws(&mut out);
                if self.in_object() {
                    out[b',' as usize] = true;
                    out[b'}' as usize] = true;
                } else if self.in_array() {
                    out[b',' as usize] = true;
                    out[b']' as usize] = true;
                }
                // Top-level number ends at first non-numeric byte; mask layer
                // will allow EOS too once is_done().
            }
            JsonState::InLiteral { which, pos } => {
                let bytes = which.bytes();
                let i = pos as usize;
                if i < bytes.len() {
                    out[bytes[i] as usize] = true;
                }
            }
            JsonState::ObjectExpectKeyOrEnd { just_after_comma } => {
                add_ws(&mut out);
                out[b'"' as usize] = true;
                if !just_after_comma {
                    out[b'}' as usize] = true;
                }
            }
            JsonState::ObjectExpectColon => {
                add_ws(&mut out);
                out[b':' as usize] = true;
            }
            JsonState::AfterValue => {
                add_ws(&mut out);
                match self.stack.last() {
                    Some(Frame::Object) => {
                        out[b',' as usize] = true;
                        out[b'}' as usize] = true;
                    }
                    Some(Frame::Array) => {
                        out[b',' as usize] = true;
                        out[b']' as usize] = true;
                    }
                    None => {
                        // Top-level after-value should already have
                        // transitioned to Done in step(); defensive only.
                    }
                }
            }
            JsonState::Done => {
                add_ws(&mut out);
            }
        }
        // Keep the allow-set in step with what `step` will accept: past the
        // run cap, insignificant whitespace is no longer legal.
        if self.ws_run >= MAX_INSIGNIFICANT_WS_RUN {
            for &b in &WS {
                out[b as usize] = false;
            }
        }
        out
    }

    /// Consume one byte. Returns `Err(())` if illegal at the current state.
    #[allow(clippy::result_unit_err)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub fn step(&mut self, byte: u8) -> Result<(), ()> {
        // Whitespace outside strings is a no-op for state, and a closer for
        // open InNumber. The run is capped: an unbounded no-op is a cycle a
        // greedy decoder never leaves — see `MAX_INSIGNIFICANT_WS_RUN`.
        let in_string = matches!(self.state, JsonState::InString { .. })
            || matches!(self.state, JsonState::InStringEscape { .. })
            || matches!(self.state, JsonState::InStringUnicode { .. });
        if !in_string && WS.contains(&byte) {
            if self.ws_run >= MAX_INSIGNIFICANT_WS_RUN {
                return Err(());
            }
            self.ws_run += 1;
            if self.state == JsonState::InNumber {
                self.close_number();
            }
            return Ok(());
        }
        // A content or structural byte ends the whitespace run.
        self.ws_run = 0;

        match self.state {
            JsonState::Start => self.enter_value(byte, /*as_object_key=*/ false),
            JsonState::ArrayExpectValueOrEnd { just_after_comma } => {
                if byte == b']' {
                    if just_after_comma {
                        return Err(());
                    }
                    return self.close_container(Frame::Array);
                }
                self.enter_value(byte, false)
            }
            JsonState::InString { is_key } => match byte {
                b'"' => {
                    if is_key {
                        self.state = JsonState::ObjectExpectColon;
                    } else {
                        self.transit_value_complete();
                    }
                    Ok(())
                }
                b'\\' => {
                    self.state = JsonState::InStringEscape { is_key };
                    Ok(())
                }
                // Raw C0 control bytes (incl. tab / newline / CR) are illegal
                // inside a JSON string — they must be escaped. Rejecting them
                // also stops a greedy decoder from looping on raw whitespace
                // inside a string value.
                0..=0x1f => Err(()),
                _ => Ok(()),
            },
            JsonState::InStringEscape { is_key } => {
                if byte == b'u' {
                    self.state = JsonState::InStringUnicode {
                        is_key,
                        n_consumed: 0,
                    };
                    Ok(())
                } else if b"\"\\/bfnrt".contains(&byte) {
                    self.state = JsonState::InString { is_key };
                    Ok(())
                } else {
                    Err(())
                }
            }
            JsonState::InStringUnicode { is_key, n_consumed } => {
                let is_hex = byte.is_ascii_digit()
                    || (b'a'..=b'f').contains(&byte)
                    || (b'A'..=b'F').contains(&byte);
                if !is_hex {
                    return Err(());
                }
                if n_consumed + 1 >= 4 {
                    self.state = JsonState::InString { is_key };
                } else {
                    self.state = JsonState::InStringUnicode {
                        is_key,
                        n_consumed: n_consumed + 1,
                    };
                }
                Ok(())
            }
            JsonState::InNumber => {
                if byte.is_ascii_digit() || b".eE+-".contains(&byte) {
                    Ok(())
                } else {
                    // Number is closed; treat current byte as a follower.
                    self.close_number();
                    // Re-dispatch the byte through the new state.
                    self.step(byte)
                }
            }
            JsonState::InLiteral { which, pos } => {
                let bytes = which.bytes();
                let i = pos as usize;
                if i >= bytes.len() || byte != bytes[i] {
                    return Err(());
                }
                if i + 1 == bytes.len() {
                    self.transit_value_complete();
                } else {
                    self.state = JsonState::InLiteral {
                        which,
                        pos: pos + 1,
                    };
                }
                Ok(())
            }
            JsonState::ObjectExpectKeyOrEnd { just_after_comma } => match byte {
                b'"' => {
                    self.state = JsonState::InString { is_key: true };
                    Ok(())
                }
                b'}' if !just_after_comma => self.close_container(Frame::Object),
                _ => Err(()),
            },
            JsonState::ObjectExpectColon => {
                if byte == b':' {
                    self.state = JsonState::Start;
                    Ok(())
                } else {
                    Err(())
                }
            }
            JsonState::AfterValue => self.handle_after_value(byte),
            JsonState::Done => Err(()),
        }
    }

    fn enter_value(&mut self, byte: u8, _as_object_key: bool) -> Result<(), ()> {
        match byte {
            b'{' => {
                self.stack.push(Frame::Object);
                self.state = JsonState::ObjectExpectKeyOrEnd {
                    just_after_comma: false,
                };
                Ok(())
            }
            b'[' => {
                self.stack.push(Frame::Array);
                self.state = JsonState::ArrayExpectValueOrEnd {
                    just_after_comma: false,
                };
                Ok(())
            }
            b'"' => {
                self.state = JsonState::InString { is_key: false };
                Ok(())
            }
            b't' => {
                self.state = JsonState::InLiteral {
                    which: LiteralKind::True,
                    pos: 1,
                };
                Ok(())
            }
            b'f' => {
                self.state = JsonState::InLiteral {
                    which: LiteralKind::False,
                    pos: 1,
                };
                Ok(())
            }
            b'n' => {
                self.state = JsonState::InLiteral {
                    which: LiteralKind::Null,
                    pos: 1,
                };
                Ok(())
            }
            b'-' => {
                self.state = JsonState::InNumber;
                Ok(())
            }
            d if d.is_ascii_digit() => {
                self.state = JsonState::InNumber;
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// Called when a value has just been fully consumed. Transitions
    /// to `Done` if the structural stack is empty, else `AfterValue`.
    fn transit_value_complete(&mut self) {
        if self.stack.is_empty() {
            self.state = JsonState::Done;
        } else {
            self.state = JsonState::AfterValue;
        }
    }

    /// Called when an open number is closed by a non-numeric byte. Sets
    /// the transition state correctly so the caller can dispatch the
    /// closing byte through `step`.
    fn close_number(&mut self) {
        self.transit_value_complete();
    }

    fn handle_after_value(&mut self, byte: u8) -> Result<(), ()> {
        match self.stack.last() {
            None => Err(()),
            Some(Frame::Object) => match byte {
                b',' => {
                    self.state = JsonState::ObjectExpectKeyOrEnd {
                        just_after_comma: true,
                    };
                    Ok(())
                }
                b'}' => self.close_container(Frame::Object),
                _ => Err(()),
            },
            Some(Frame::Array) => match byte {
                b',' => {
                    self.state = JsonState::ArrayExpectValueOrEnd {
                        just_after_comma: true,
                    };
                    Ok(())
                }
                b']' => self.close_container(Frame::Array),
                _ => Err(()),
            },
        }
    }

    fn close_container(&mut self, expected: Frame) -> Result<(), ()> {
        match self.stack.pop() {
            Some(f) if f == expected => {
                self.transit_value_complete();
                Ok(())
            }
            _ => Err(()),
        }
    }
}

// ────────────────── Token bytes map ─────────────────────────────────────────

/// Per-token id → decoded bytes. Built once at constraint construction by
/// calling `tokenizer.decode(&[id], false)` for every vocab id. Required
/// (not just first byte) because the per-step mask must verify ALL bytes
/// of a candidate token pass the grammar, not just the leading byte —
/// BPE tokens often have a Ġ-style space prefix whose first byte is space
/// (always allowed by the grammar's whitespace rule) but whose subsequent
/// bytes encode arbitrary text. Naive first-byte masking lets the model
/// emit ` Apple` from a JSON-Start state because space is legal.
///
/// Layout: a single flat `bytes` buffer + `offsets[id..=id+1]` slice
/// indexing. Saves ~1.3 MB of `String` overhead on a 152K vocab vs a
/// `Vec<Vec<u8>>`.
#[derive(Debug)]
pub struct TokenBytesMap {
    bytes: Vec<u8>,
    /// `offsets[i]..offsets[i+1]` is the byte slice for token id `i`.
    /// Length = `vocab_size + 1`.
    offsets: Vec<u32>,
    vocab_size: usize,
}

impl TokenBytesMap {
    /// Build by decoding every vocab id once. Cost: ~600 ms for Qwen3.6's
    /// ~152K vocab on the M5 Max. Paid once per constrained request.
    pub fn new(tokenizer: &tokenizers::Tokenizer) -> Self {
        let vocab_size = tokenizer.get_vocab_size(true);
        let mut bytes: Vec<u8> = Vec::with_capacity(vocab_size * 4);
        let mut offsets: Vec<u32> = Vec::with_capacity(vocab_size + 1);
        offsets.push(0);
        for id in 0..vocab_size {
            let decoded = tokenizer.decode(&[id as u32], false).unwrap_or_default();
            bytes.extend_from_slice(decoded.as_bytes());
            offsets.push(bytes.len() as u32);
        }
        Self {
            bytes,
            offsets,
            vocab_size,
        }
    }

    /// Return the number of tokens in the vocabulary table.
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Bytes for token id `i`, or `&[]` if out of range.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub fn token_bytes(&self, id: usize) -> &[u8] {
        if id + 1 >= self.offsets.len() {
            return &[];
        }
        let lo = self.offsets[id] as usize;
        let hi = self.offsets[id + 1] as usize;
        &self.bytes[lo..hi]
    }
}

// ────────────────── shared allow-mask probe ──────────────────────────────────

/// A grammar the shared allow-mask probe can drive: cheaply reset to a
/// reference state, then fed one byte at a time. Implemented by both the
/// free-form [`JsonGrammar`] and the schema-driven `SchemaGrammar` so the
/// O(vocab) mask sweep — the hot loop of every constrained decode step — lives
/// in one place instead of being duplicated per engine.
pub(crate) trait ProbeGrammar {
    /// Overwrite `self` with `src`'s state, reusing `self`'s own buffers where
    /// the frame type allows (no per-call allocation for a `Copy` frame stack).
    fn reset_from(&mut self, src: &Self);

    /// Advance one byte. `Err(())` means the byte is illegal at the current
    /// position; the caller stops feeding that candidate token.
    fn feed(&mut self, byte: u8) -> Result<(), ()>;
}

impl ProbeGrammar for JsonGrammar {
    fn reset_from(&mut self, src: &Self) {
        // `state` is `Copy`; `Frame` is `Copy`, so refilling the stack is a
        // memcpy into the reused buffer — no per-token allocation.
        self.state = src.state;
        self.ws_run = src.ws_run;
        self.stack.clear();
        self.stack.extend_from_slice(&src.stack);
    }

    fn feed(&mut self, byte: u8) -> Result<(), ()> {
        self.step(byte)
    }
}

/// Fill `mask` (length `vocab_size`) with the per-token allow bits by feeding
/// each token's decoded bytes through `grammar`'s current state. `scratch` is a
/// throwaway grammar reused across tokens so the probe allocates at most once
/// per decode step (when the frame type is `Copy`) instead of once per token.
///
/// Cost on a ~152K vocab is ~1–3 ms per decode step on the constrained path:
/// most tokens are rejected on their first byte, so the average probe is far
/// shorter than a full token and the once-assumed "~1M ops" never materializes.
/// The dominant remaining term is the O(vocab) sweep itself, which only a
/// per-state feasible-token index would remove.
pub(crate) fn fill_allow_mask<G: ProbeGrammar>(
    grammar: &G,
    scratch: &mut G,
    bytes_map: &TokenBytesMap,
    vocab_size: usize,
    mask: &mut [bool],
) {
    let n = bytes_map.vocab_size().min(vocab_size);
    for id in 0..n {
        let bytes = bytes_map.token_bytes(id);
        if bytes.is_empty() {
            continue; // specials decode to "" → never allowed mid-grammar
        }
        scratch.reset_from(grammar);
        let mut ok = true;
        for &b in bytes {
            if scratch.feed(b).is_err() {
                ok = false;
                break;
            }
        }
        if ok {
            if let Some(m) = mask.get_mut(id) {
                *m = true;
            }
        }
    }
}

// ────────────────── ConstraintEngine impl ────────────────────────────────────

/// `ConstraintEngine` for `response_format: json_object`. Holds the
/// state machine, the precomputed bytes map, and the model's EOS ids.
///
/// # Warm-up before engaging
///
/// Many instruction-tuned models (Qwen3, DeepSeek-R1, etc.) emit a
/// special `<think>` block *before* the visible answer. Those tokens'
/// decoded bytes (`<`, `/`, letters) would all fail the JSON grammar at
/// `Start` state. To avoid forcing the model into a dead state during
/// reasoning, the constraint operates in a two-phase mode:
///
/// 1. **Warm-up**: until we have seen the first byte that the grammar's
///    `Start` state can accept, all tokens are allowed.
/// 2. **Engaged**: as soon as the *previously-sampled token* contains a
///    valid JSON start byte, the constraint enforces the grammar.
///
/// This works at temp=0 because the model, when asked "Return JSON ...",
/// will (after thinking) deterministically emit a token whose bytes
/// start a JSON value. We engage on that token and constrain everything
/// from there. The mask never blocks the model during the think phase.
pub struct JsonObjectConstraint {
    grammar: JsonGrammar,
    bytes_map: Arc<TokenBytesMap>,
    eos_ids: Vec<u32>,
    mask: Vec<bool>,
    /// Pre-engagement state — see the doc comment above.
    engaged: bool,
    /// Mirror of `engaged` the route keeps a clone of. The engine itself is
    /// moved into the decode thread, so this is the only way the route can
    /// learn — after the stream drains — whether the grammar was ever applied.
    engaged_flag: Arc<AtomicBool>,
    /// Text accumulated from tokens seen before engagement. Used to detect
    /// and suppress a leading markdown-fence (` ```json\n `) wrapper.
    /// Cleared when engagement fires.
    pre_engage_buf: String,
    /// Shared `is_thinking` flag updated by the route's step_fn after
    /// the think-splitter classifies each emitted token. When `true`,
    /// the constraint refuses to engage on `{` because the model is
    /// inside its reasoning channel (Qwen3, DeepSeek-R1 — they emit
    /// example JSON in markdown code blocks as part of the chain of
    /// thought, and engaging there locks the grammar). When `false`
    /// (the answer phase or non-think models), engagement on `{`
    /// proceeds. Defaults to `false` (treat non-think models as
    /// non-thinking from the start).
    is_thinking: Arc<AtomicBool>,
}

impl std::fmt::Debug for JsonObjectConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JsonObjectConstraint")
            .field("done", &self.grammar.is_done())
            .field("eos_ids", &self.eos_ids)
            .field("vocab_size", &self.bytes_map.vocab_size)
            .finish()
    }
}

impl JsonObjectConstraint {
    /// Build a constraint by precomputing the token-bytes map.
    ///
    /// `eos_ids` is the model's stop-token set from `config.json`; the mask
    /// allows these only when the grammar reaches a terminal state.
    pub fn new(tokenizer: Arc<tokenizers::Tokenizer>, eos_ids: Vec<u32>) -> Self {
        let bytes_map = Arc::new(TokenBytesMap::new(&tokenizer));
        Self {
            grammar: JsonGrammar::new(),
            bytes_map,
            eos_ids,
            mask: Vec::new(),
            engaged: false,
            engaged_flag: Arc::new(AtomicBool::new(false)),
            pre_engage_buf: String::new(),
            is_thinking: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Construct directly from a precomputed bytes map. Used by unit tests
    /// to avoid loading a real tokenizer.
    pub fn from_bytes_map(bytes_map: Arc<TokenBytesMap>, eos_ids: Vec<u32>) -> Self {
        Self {
            grammar: JsonGrammar::new(),
            bytes_map,
            eos_ids,
            mask: Vec::new(),
            engaged: false,
            engaged_flag: Arc::new(AtomicBool::new(false)),
            pre_engage_buf: String::new(),
            is_thinking: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Force-engage the constraint, bypassing warm-up. Used by tests
    /// that drive the mask directly via [`feed_bytes`].
    pub fn force_engage(&mut self) {
        self.mark_engaged();
    }

    /// The one place `engaged` flips, so the route-visible mirror can never
    /// drift from the field the mask logic reads.
    fn mark_engaged(&mut self) {
        self.engaged = true;
        self.engaged_flag.store(true, Ordering::Release);
    }

    /// Shared handle the route reads after generation to learn whether the
    /// engine ever engaged. `false` at end of stream means the response was
    /// never checked against the requested grammar.
    pub fn engaged_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.engaged_flag)
    }

    /// Handle to the shared `is_thinking` flag. The route's step_fn
    /// updates it after each emitted token; the constraint reads it on
    /// each `advance` call to decide whether to scan for engagement.
    pub fn is_thinking_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.is_thinking)
    }

    /// Returns `true` when the text accumulated before engagement is only
    /// whitespace and/or a markdown fence header (```` ```json ```` /
    /// ```` ``` ````). The route handler may discard that prefix from
    /// `content` rather than leaking it.
    pub fn pre_engage_is_fence(&self) -> bool {
        schema::is_only_fence_or_whitespace(&self.pre_engage_buf)
    }

    /// Feed raw bytes into the grammar (used by tests + by `advance`).
    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if self.grammar.step(b).is_err() {
                // Model emitted an illegal byte despite the mask. Clamp the
                // grammar to Done so the next step returns an EOS-only mask
                // and decode terminates promptly. Should be unreachable
                // when the mask is built correctly (every allowed token
                // bytes-feeds through a clone of the grammar in
                // `step_mask`); kept as a defensive guard.
                tracing::warn!(
                    byte = b,
                    "JsonObjectConstraint: grammar rejected byte; forcing terminal"
                );
                self.grammar.state = JsonState::Done;
                return;
            }
        }
    }

    /// Feed bytes into the grammar, stopping at the first invalid byte but
    /// NOT clamping to Done. Used for the engagement token where the mask
    /// has not yet had a chance to filter the token's suffix bytes — we
    /// accept as many valid bytes as possible and leave the grammar in a
    /// valid (non-terminal) state so the next `step_mask` can steer output.
    fn feed_bytes_partial(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if self.grammar.step(b).is_err() {
                tracing::debug!(
                    byte = b,
                    "JsonObjectConstraint: engagement token suffix byte rejected (expected for multi-byte tokens); stopping feed"
                );
                return; // stop, do NOT clamp to Done
            }
        }
    }
}

impl ConstraintEngine for JsonObjectConstraint {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn step_mask(&mut self, vocab_size: usize) -> &[bool] {
        if self.mask.len() == vocab_size {
            self.mask.fill(false);
        } else {
            self.mask = vec![false; vocab_size];
        }
        // Warm-up: before the model has emitted a JSON-start byte, allow
        // every token. This is how we tolerate `<think>` / reasoning
        // prefixes on Qwen3 and similar models. We engage as soon as
        // `advance` sees a byte that's legal at Start.
        if !self.engaged {
            self.mask.fill(true);
            return &self.mask;
        }
        // Fast path: grammar is done — only EOS is legal. The grammar's
        // `step` method allows trailing whitespace when done (JSON spec allows
        // trailing whitespace after the root value), which would otherwise let
        // whitespace-only tokens through and cause the model to emit infinite
        // TABs/spaces instead of stopping.
        if self.grammar.is_done() {
            for &eid in &self.eos_ids {
                if (eid as usize) < vocab_size {
                    self.mask[eid as usize] = true;
                }
            }
            return &self.mask;
        }
        // Per-step correctness: a candidate token is allowed only if ALL its
        // bytes feed cleanly through the current grammar. The O(vocab) probe is
        // shared with the schema engine via `fill_allow_mask`; `scratch` is a
        // throwaway grammar reused across tokens so the sweep allocates at most
        // once per decode step (see the helper for the cost model).
        let mut scratch = self.grammar.clone();
        fill_allow_mask(
            &self.grammar,
            &mut scratch,
            &self.bytes_map,
            vocab_size,
            &mut self.mask,
        );
        &self.mask
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn advance(&mut self, token_id: u32) {
        if self.eos_ids.contains(&token_id) {
            return;
        }
        let bytes = self.bytes_map.token_bytes(token_id as usize).to_vec();
        if self.engaged {
            self.feed_bytes(&bytes);
        } else {
            // If the route signalled `is_thinking == true` for the last
            // emitted token, skip the engagement scan entirely — the
            // model is inside its reasoning channel and any `{` here is
            // example JSON inside the chain of thought, not the answer.
            if self.is_thinking.load(Ordering::Relaxed) {
                return;
            }
            // Accumulate pre-engagement text so the handler can detect and
            // suppress a leading markdown-fence wrapper.
            if let Ok(s) = std::str::from_utf8(&bytes) {
                self.pre_engage_buf.push_str(s);
            }
            // Pre-engagement: engage on the next `{` or `[` byte. By
            // construction we only reach here when `is_thinking == false`,
            // so a structural opener signals the start of the answer
            // payload. `[` is included because some `json_object` prompts
            // get answered with a top-level array (3-fruits-list, etc.),
            // which is still valid JSON; the grammar accepts top-level
            // arrays.
            if let Some(idx) = bytes.iter().position(|&b| b == b'{' || b == b'[') {
                self.mark_engaged();
                tracing::info!(
                    token_id,
                    engage_byte = bytes[idx],
                    "JsonObjectConstraint: engaging from this token"
                );
                // Use partial feed: the engagement token's suffix bytes have
                // not been filtered by the mask (mask was all-true during
                // warm-up). Accept as many valid bytes as possible and stop
                // at the first invalid byte WITHOUT clamping to Done, so the
                // next step_mask can steer subsequent tokens correctly.
                self.feed_bytes_partial(&bytes[idx..]);
            }
        }
    }

    fn finished(&self) -> bool {
        self.engaged && self.grammar.is_done()
    }

    fn engaged(&self) -> bool {
        self.engaged
    }

    fn engaged_handle(&self) -> Option<Arc<AtomicBool>> {
        Some(Arc::clone(&self.engaged_flag))
    }

    /// Always returns `true` so the pipelined generate loop uses the masked
    /// branch from the very first token. During warm-up, `step_mask` returns
    /// an all-true mask (every token allowed); this is identical in outcome
    /// to the unconstrained argmax path but avoids the one-step pipeline lag
    /// that would otherwise let one unmasked token slip through immediately
    /// after engagement (which is the source of `grammar rejected byte` in
    /// multi-byte-token vocabs like Gemma4's `{<unused…>` composites).
    fn wants_mask(&self) -> bool {
        true
    }
}

// ───── tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests;
