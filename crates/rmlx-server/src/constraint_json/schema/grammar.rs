//! Schema-driven JSON byte state machine and supporting utilities.
//!
//! Contains:
//! - `is_only_fence_or_whitespace` — markdown-fence detection helper (A6.5)
//! - `literal_bytes` / `union_literals` — literal serialization helpers
//! - `LiteralTrie` — byte-level literal match tracker for `enum`/`const`/`oneOf`
//! - `FreeKeyNext` / `free_key_step` — additional-property key parser
//! - `Frame`, `ObjPhase`, `ArrPhase`, `Leaf`, `AfterTarget` — state machine types
//! - `SchemaGrammar` + full `impl` — the schema-driven byte state machine
//!
//! ## LOC exemption
//!
//! This file is ~1400 LOC, exceeding the 1000 LOC split target. The
//! `SchemaGrammar` state machine integrates object-key trie tracking,
//! array bounds enforcement, literal-branch selection, free-key parsing,
//! epilogue handling, and the buffer-reusing per-token `reset_from` into a
//! single cohesive unit. Splitting the state machine across files would
//! require threading mutable `self` through function arguments or boxing
//! frame state, adding indirection for no correctness gain.

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::missing_fields_in_debug,
    clippy::needless_pass_by_value,
    clippy::self_only_used_in_recursion,
    clippy::unused_self,
    unreachable_pub
)]

use std::sync::Arc;

use serde_json::Value;

use super::super::MAX_INSIGNIFICANT_WS_RUN;
use super::types::SchemaNode;

/// Whitespace bytes allowed outside strings (mirrors `super::super::WS`).
pub(super) const WS: [u8; 4] = *b" \t\n\r";

// ────────────────── fence-suppression helper ────────────────────────────────

/// Returns `true` when `s` is entirely whitespace optionally followed by a
/// markdown code-fence header (` ```json` or ` ``` `) and nothing else. Used
/// to decide whether to discard the pre-engagement buffer rather than
/// leaking it into `content`.
///
/// Pattern: `^\s*(```(json)?\s*)?$`
pub(crate) fn is_only_fence_or_whitespace(s: &str) -> bool {
    let trimmed = s.trim_start_matches(|c: char| c.is_ascii_whitespace());
    if trimmed.is_empty() {
        return true;
    }
    // Must start with ` ``` `, otherwise it is real prose → false.
    let Some(after_backticks) = trimmed.strip_prefix("```") else {
        return false;
    };
    // Optionally followed by `json`.
    let after_lang = after_backticks
        .strip_prefix("json")
        .unwrap_or(after_backticks);
    // Remainder must be whitespace only (newline etc.).
    after_lang.chars().all(|c| c.is_ascii_whitespace())
}

// ────────────────── literal trie (enum / const) ─────────────────────────────

/// Canonical JSON serialization of a literal, used as the exact byte
/// sequence the decoder must emit for a `const` / string-`enum` value.
/// Mirrors llama.cpp `_generate_constant_rule = format_literal(dump())`.
pub(super) fn literal_bytes(v: &Value) -> Vec<u8> {
    serde_json::to_string(v).unwrap_or_default().into_bytes()
}

/// If every branch of a union is a `const` or a string-`enum` (i.e. the
/// union is fully discriminated by literal values), return the flattened
/// list of canonical literal byte-sequences. Otherwise `None`.
#[allow(
    clippy::wildcard_enum_match_arm,
    reason = "wildcard arm is the correct fallthrough for unsupported arch/quant/error variants"
)]
pub(super) fn union_literals(branches: &[Arc<SchemaNode>]) -> Option<Vec<Vec<u8>>> {
    let mut out = Vec::new();
    for b in branches {
        match b.as_ref() {
            SchemaNode::Const(v) => out.push(literal_bytes(v)),
            SchemaNode::Str {
                enum_: Some(lits), ..
            } => {
                for s in lits {
                    out.push(literal_bytes(&Value::String(s.clone())));
                }
            }
            _ => return None,
        }
    }
    if out.is_empty() {
        None
    } else {
        Some(out)
    }
}

/// Position inside a set of literal alternatives. Tracks how many bytes of
/// the chosen literal(s) have been emitted; the still-viable set is the
/// literals whose prefix matches what has been emitted so far.
///
/// `lits` is the immutable alternative set, held behind `Arc` so cloning the
/// trie (per candidate token in the mask probe) shares it instead of copying
/// every literal. `pos`/`viable` are the mutable match progress and clone by
/// value.
#[derive(Debug, Clone)]
pub(super) struct LiteralTrie {
    pub(super) lits: Arc<[Vec<u8>]>,
    /// byte offset matched so far (same for all viable lits — they share
    /// the matched prefix).
    pub(super) pos: usize,
    /// indices into `lits` still viable given the emitted prefix.
    pub(super) viable: Vec<usize>,
}

impl LiteralTrie {
    pub(super) fn new(lits: Vec<Vec<u8>>) -> Self {
        let viable = (0..lits.len()).collect();
        Self {
            lits: Arc::from(lits),
            pos: 0,
            viable,
        }
    }

    /// Refill from `src` reusing the `viable` buffer (the per-token reset on
    /// the mask sweep). `lits` is `Arc`-shared (refcount bump).
    fn reset_from(&mut self, src: &Self) {
        self.lits.clone_from(&src.lits);
        self.pos = src.pos;
        self.viable.clone_from(&src.viable);
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub(super) fn allowed(&self, out: &mut [bool; 256]) {
        for &i in &self.viable {
            if let Some(&b) = self.lits[i].get(self.pos) {
                out[b as usize] = true;
            }
        }
    }

    /// True when at least one viable literal is fully emitted.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub(super) fn complete(&self) -> bool {
        self.viable.iter().any(|&i| self.lits[i].len() == self.pos)
    }

    /// Consume one byte. `Err` if no viable literal accepts it.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub(super) fn step(&mut self, b: u8) -> Result<(), ()> {
        let next: Vec<usize> = self
            .viable
            .iter()
            .copied()
            .filter(|&i| self.lits[i].get(self.pos) == Some(&b))
            .collect();
        if next.is_empty() {
            return Err(());
        }
        self.viable = next;
        self.pos += 1;
        Ok(())
    }
}

/// Outcome of stepping one byte through a free-form (additional-property)
/// JSON string key.
enum FreeKeyNext {
    Continue { escape: bool, unicode: u8 },
    Done,
}

/// Drive one byte of a free-form JSON string key (mirrors the `Leaf::InStr`
/// transition rules). The opening `"` was already consumed by the caller.
fn free_key_step(byte: u8, escape: bool, unicode: u8) -> Result<FreeKeyNext, ()> {
    if unicode > 0 {
        let is_hex =
            byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte) || (b'A'..=b'F').contains(&byte);
        if !is_hex {
            return Err(());
        }
        return Ok(FreeKeyNext::Continue {
            escape: false,
            unicode: if unicode >= 4 { 0 } else { unicode + 1 },
        });
    }
    if escape {
        return if byte == b'u' {
            Ok(FreeKeyNext::Continue {
                escape: false,
                unicode: 1,
            })
        } else if b"\"\\/bfnrt".contains(&byte) {
            Ok(FreeKeyNext::Continue {
                escape: false,
                unicode: 0,
            })
        } else {
            Err(())
        };
    }
    match byte {
        b'"' => Ok(FreeKeyNext::Done),
        b'\\' => Ok(FreeKeyNext::Continue {
            escape: true,
            unicode: 0,
        }),
        // Raw C0 control bytes (incl. tab / newline / CR) are illegal inside a
        // JSON string — they must be escaped. Same rule as `step_instr`; a key
        // string is a string.
        0..=0x1f => Err(()),
        _ => Ok(FreeKeyNext::Continue {
            escape: false,
            unicode: 0,
        }),
    }
}

// ────────────────── schema-driven byte state machine ───────────────────────

/// One stack frame describing where in the schema tree the decoder is.
#[derive(Debug, Clone)]
enum Frame {
    /// Inside `{ … }`. `idx` = next property index to emit; `emitted` =
    /// keys already written; phase tracks the `"key":value,` micro-cycle.
    ///
    /// `node_props` / `required` are the immutable schema (shared via `Arc`);
    /// `emitted` / `phase` are the mutable progress.
    Object {
        node_props: Arc<[(String, Arc<SchemaNode>)]>,
        required: Arc<[String]>,
        additional: bool,
        emitted: Vec<String>,
        phase: ObjPhase,
    },
    /// Inside `[ … ]`. `count` = elements committed so far.
    ///
    /// `items` is the immutable element schema (shared via `Arc`); `count` /
    /// `phase` are the mutable progress.
    Array {
        items: Arc<SchemaNode>,
        min: Option<usize>,
        max: Option<usize>,
        count: usize,
        phase: ArrPhase,
    },
}

impl Frame {
    /// Refill from `src` reusing this frame's heap buffers when the variant is
    /// unchanged: the `Object` `emitted` vector and the in-key trie's `viable`
    /// vector are refilled in place via `clone_from`. On a variant mismatch we
    /// fall back to a plain clone.
    fn reset_from(&mut self, src: &Self) {
        match (self, src) {
            (
                Frame::Object {
                    node_props,
                    required,
                    additional,
                    emitted,
                    phase,
                },
                Frame::Object {
                    node_props: s_props,
                    required: s_req,
                    additional: s_add,
                    emitted: s_emitted,
                    phase: s_phase,
                },
            ) => {
                node_props.clone_from(s_props);
                required.clone_from(s_req);
                *additional = *s_add;
                emitted.clone_from(s_emitted);
                phase.reset_from(s_phase);
            }
            (
                Frame::Array {
                    items,
                    min,
                    max,
                    count,
                    phase,
                },
                Frame::Array {
                    items: s_items,
                    min: s_min,
                    max: s_max,
                    count: s_count,
                    phase: s_phase,
                },
            ) => {
                items.clone_from(s_items);
                *min = *s_min;
                *max = *s_max;
                *count = *s_count;
                *phase = s_phase.clone();
            }
            (dst, src) => *dst = src.clone(),
        }
    }
}

impl ObjPhase {
    /// Refill from `src`, reusing the in-key trie's `viable` buffer when both
    /// sides are mid-key; otherwise a plain clone (the other variants are
    /// `Copy`-sized and allocate nothing).
    fn reset_from(&mut self, src: &Self) {
        match (self, src) {
            (ObjPhase::InKey { trie }, ObjPhase::InKey { trie: s_trie }) => {
                trie.reset_from(s_trie);
            }
            (dst, src) => *dst = src.clone(),
        }
    }
}

#[derive(Debug, Clone)]
enum ObjPhase {
    /// Expecting a key string `"` or `}`.
    ExpectKeyOrEnd { just_after_comma: bool },
    /// Inside a key string matched against the still-pending property
    /// names (trie over `"name"` literals incl. quotes).
    InKey { trie: LiteralTrie },
    /// Inside a free-form additional-property key string (no schema
    /// property governs it). `escape`/`unicode` mirror the string leaf.
    InFreeKey { escape: bool, unicode: u8 },
    /// After key + closing quote, expecting `:`. `which` = property index
    /// (None ⇒ free additional key → value schema is `Any`).
    ExpectColon { which: Option<usize> },
    /// Value sub-machine is active; on its completion → expect `,`/`}`.
    AfterValue,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ArrPhase {
    /// Expecting a value or `]`.
    ExpectValueOrEnd { just_after_comma: bool },
    /// A value sub-machine is active; on return expect `,` or `]`.
    AfterValue,
}

/// Active scalar/leaf sub-state when emitting a non-container value.
#[derive(Debug, Clone)]
enum Leaf {
    /// Awaiting the first byte of a value governed by `node` (shared via `Arc`).
    Start(Arc<SchemaNode>),
    /// Inside a JSON string (free-form, e.g. `type:string` w/o enum).
    InStr {
        escape: bool,
        unicode: u8,
        done_to: AfterTarget,
    },
    /// Inside a literal trie (`const`, `enum`, `bool`, `null`).
    InLit {
        trie: LiteralTrie,
        done_to: AfterTarget,
    },
    /// Inside a number (integer or float).
    InNum {
        integer: bool,
        seen_digit: bool,
        done_to: AfterTarget,
    },
}

impl Leaf {
    /// Refill from `src`, reusing the `InLit` trie's `viable` buffer when both
    /// sides are inside a literal trie; otherwise a plain clone.
    fn reset_from(&mut self, src: &Self) {
        match (self, src) {
            (
                Leaf::InLit { trie, done_to },
                Leaf::InLit {
                    trie: s_trie,
                    done_to: s_done_to,
                },
            ) => {
                trie.reset_from(s_trie);
                *done_to = *s_done_to;
            }
            (dst, src) => *dst = src.clone(),
        }
    }
}

/// Where control returns after a leaf/value completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AfterTarget {
    /// The value is the top-level value: set `done = true`.
    Top,
    /// Inside the innermost container frame (advance its phase).
    Container,
}

/// Schema-driven JSON byte state machine. Parallels `super::JsonGrammar`
/// but the valid-byte set is derived from the schema tree position rather
/// than "any JSON".
#[derive(Debug, Clone)]
pub struct SchemaGrammar {
    stack: Vec<Frame>,
    leaf: Option<Leaf>,
    pub(super) done: bool,
    /// Consecutive insignificant-whitespace bytes accepted since the last
    /// content or structural byte. Capped at [`MAX_INSIGNIFICANT_WS_RUN`] so a
    /// greedy decoder cannot sit in the whitespace no-op forever.
    ws_run: u32,
}

impl super::super::ProbeGrammar for SchemaGrammar {
    fn reset_from(&mut self, src: &Self) {
        // The immutable schema inside each `Frame` / `Leaf` is held behind
        // `Arc` (property list, element schema, literal alternatives), so a
        // frame reset is a few refcount bumps plus a copy of the tiny mutable
        // progress (`emitted` keys, trie `viable` indices, `phase`). No schema
        // subtree is ever deep-copied.
        //
        // Beyond that, this reset reuses the scratch's existing per-frame
        // buffers: on the O(vocab) mask sweep the scratch is reset once per
        // candidate token, and where its stack shape still matches the
        // reference (same variant at the same depth) the `emitted` / `viable`
        // vectors are refilled in place via `clone_from` rather than
        // reallocated. The derived `clone_from` would instead reassign each
        // frame wholesale, dropping and re-growing those vectors every token.
        self.done = src.done;
        self.ws_run = src.ws_run;
        reset_stack(&mut self.stack, &src.stack);
        // Reuse a literal-trie's `viable` buffer when both leaves are the same
        // trie-bearing variant; otherwise fall back to a plain clone.
        match (self.leaf.as_mut(), src.leaf.as_ref()) {
            (Some(d), Some(s)) => d.reset_from(s),
            (_, Some(s)) => self.leaf = Some(s.clone()),
            (_, None) => self.leaf = None,
        }
    }

    fn feed(&mut self, byte: u8) -> Result<(), ()> {
        self.step(byte)
    }
}

/// Refill `dst` from `src`, reusing each frame's heap buffers in place where
/// the variant at that depth is unchanged (the common case across consecutive
/// probe tokens). Frames whose variant differs, and any tail beyond `src`'s
/// length, fall back to a plain clone / truncate.
fn reset_stack(dst: &mut Vec<Frame>, src: &[Frame]) {
    let common = dst.len().min(src.len());
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        d.reset_from(s);
    }
    dst.truncate(common);
    dst.extend(src.iter().skip(common).cloned());
}

impl SchemaGrammar {
    /// Create a new grammar that will enforce the given `root` schema node.
    pub fn new(root: SchemaNode) -> Self {
        Self {
            stack: Vec::new(),
            leaf: Some(Leaf::Start(Arc::new(root))),
            done: false,
            ws_run: 0,
        }
    }

    /// Return `true` when the grammar has consumed a complete, schema-valid JSON value.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub fn is_done(&self) -> bool {
        if self.done {
            return true;
        }
        // A top-level bare scalar has no follower byte to close it (the
        // stream just ends / EOS). Treat a fully-formed open top-level
        // number or literal as complete — mirrors A6.3
        // `JsonGrammar::is_done` for `InNumber` at top level.
        if !self.stack.is_empty() {
            return false;
        }
        match &self.leaf {
            Some(Leaf::InNum {
                seen_digit,
                done_to: AfterTarget::Top,
                ..
            }) => *seen_digit,
            Some(Leaf::InLit {
                trie,
                done_to: AfterTarget::Top,
            }) => trie.complete() && !trie.viable.iter().any(|&i| trie.lits[i].len() > trie.pos),
            _ => false,
        }
    }

    /// Set of bytes legal as the next byte at the current position.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    pub fn allowed_bytes(&self) -> [bool; 256] {
        let mut out = [false; 256];
        if self.done {
            for &b in &WS {
                out[b as usize] = true;
            }
            self.clamp_ws(&mut out);
            return out;
        }
        if let Some(leaf) = &self.leaf {
            self.leaf_allowed(leaf, &mut out);
            self.clamp_ws(&mut out);
            return out;
        }
        // No active leaf ⇒ we are between structural tokens inside a frame.
        match self.stack.last() {
            Some(Frame::Object {
                node_props,
                required,
                additional,
                emitted,
                phase,
            }) => self.object_allowed(
                node_props.as_ref(),
                required.as_ref(),
                *additional,
                emitted,
                phase,
                &mut out,
            ),
            Some(Frame::Array {
                min,
                max,
                count,
                phase,
                ..
            }) => self.array_allowed(*min, *max, *count, phase, &mut out),
            None => {
                // top-level value already finished
                for &b in &WS {
                    out[b as usize] = true;
                }
            }
        }
        self.clamp_ws(&mut out);
        out
    }

    /// Drop the insignificant-whitespace bytes from an allow-set once the run
    /// cap is reached, so `allowed_bytes` agrees with what `step` will accept.
    /// A whitespace byte that is string *content* (mid-key) was never added by
    /// the phase arms, so this only ever removes separators.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn clamp_ws(&self, out: &mut [bool; 256]) {
        if self.ws_run < MAX_INSIGNIFICANT_WS_RUN {
            return;
        }
        for &b in &WS {
            out[b as usize] = false;
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn leaf_allowed(&self, leaf: &Leaf, out: &mut [bool; 256]) {
        match leaf {
            Leaf::Start(node) => self.value_starters(node.as_ref(), out),
            Leaf::InStr {
                escape, unicode, ..
            } => {
                if *unicode > 0 {
                    for b in b'0'..=b'9' {
                        out[b as usize] = true;
                    }
                    for b in b'a'..=b'f' {
                        out[b as usize] = true;
                    }
                    for b in b'A'..=b'F' {
                        out[b as usize] = true;
                    }
                } else if *escape {
                    for &b in b"\"\\/bfnrtu" {
                        out[b as usize] = true;
                    }
                } else {
                    // C0 control bytes must be escaped inside a JSON string —
                    // `step_instr` rejects them, so they are not allowed here.
                    for b in 0x20u16..=255 {
                        out[b as usize] = true;
                    }
                }
            }
            Leaf::InLit { trie, .. } => trie.allowed(out),
            Leaf::InNum {
                integer,
                seen_digit,
                done_to,
            } => {
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
                out[b'-' as usize] = true;
                if !*integer {
                    for &b in b".eE+" {
                        out[b as usize] = true;
                    }
                }
                if *seen_digit {
                    self.followers(*done_to, out);
                    for &b in &WS {
                        out[b as usize] = true;
                    }
                }
            }
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn value_starters(&self, node: &SchemaNode, out: &mut [bool; 256]) {
        for &b in &WS {
            out[b as usize] = true;
        }
        match node {
            SchemaNode::Object { .. } => out[b'{' as usize] = true,
            SchemaNode::Array { .. } => out[b'[' as usize] = true,
            SchemaNode::Str { .. } => out[b'"' as usize] = true,
            SchemaNode::Num { .. } => {
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
                out[b'-' as usize] = true;
            }
            SchemaNode::Bool => {
                out[b't' as usize] = true;
                out[b'f' as usize] = true;
            }
            SchemaNode::Null => out[b'n' as usize] = true,
            SchemaNode::Const(v) => {
                if let Some(&b) = literal_bytes(v).first() {
                    out[b as usize] = true;
                }
            }
            SchemaNode::Union(branches) => {
                for br in branches.iter() {
                    self.value_starters(br.as_ref(), out);
                }
            }
            SchemaNode::Any => {
                for &b in b"{[\"tfn-" {
                    out[b as usize] = true;
                }
                for b in b'0'..=b'9' {
                    out[b as usize] = true;
                }
            }
        }
    }

    /// Bytes that may follow a completed value, given where control returns.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn followers(&self, target: AfterTarget, out: &mut [bool; 256]) {
        match target {
            AfterTarget::Top => {} // only WS/EOS (added by caller / mask)
            AfterTarget::Container => match self.stack.last() {
                Some(Frame::Object { .. }) => {
                    out[b',' as usize] = true;
                    out[b'}' as usize] = true;
                }
                Some(Frame::Array { .. }) => {
                    out[b',' as usize] = true;
                    out[b']' as usize] = true;
                }
                None => {}
            },
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn object_allowed(
        &self,
        node_props: &[(String, Arc<SchemaNode>)],
        required: &[String],
        additional: bool,
        emitted: &[String],
        phase: &ObjPhase,
        out: &mut [bool; 256],
    ) {
        // Whitespace is an insignificant separator only between structural
        // tokens. `InKey` / `InFreeKey` are inside a key *string*, where a
        // whitespace byte is content the key sub-machine judges — the trie
        // and the free-key arm below add it themselves when it is legal.
        if !matches!(phase, ObjPhase::InKey { .. } | ObjPhase::InFreeKey { .. }) {
            for &b in &WS {
                out[b as usize] = true;
            }
        }
        let missing_required = required.iter().any(|r| !emitted.contains(r));
        let more_props = node_props.iter().any(|(k, _)| !emitted.contains(k));
        match phase {
            ObjPhase::ExpectKeyOrEnd { just_after_comma } => {
                if more_props || additional {
                    out[b'"' as usize] = true;
                }
                if !just_after_comma && !missing_required {
                    out[b'}' as usize] = true;
                }
            }
            ObjPhase::InKey { trie } => trie.allowed(out),
            ObjPhase::InFreeKey { escape, unicode } => {
                if *unicode > 0 {
                    for b in b'0'..=b'9' {
                        out[b as usize] = true;
                    }
                    for b in b'a'..=b'f' {
                        out[b as usize] = true;
                    }
                    for b in b'A'..=b'F' {
                        out[b as usize] = true;
                    }
                } else if *escape {
                    for &b in b"\"\\/bfnrtu" {
                        out[b as usize] = true;
                    }
                } else {
                    // C0 control bytes must be escaped inside a JSON string —
                    // `free_key_step` rejects them, so they are not allowed here.
                    for b in 0x20u16..=255 {
                        out[b as usize] = true;
                    }
                }
            }
            ObjPhase::ExpectColon { .. } => out[b':' as usize] = true,
            ObjPhase::AfterValue => {
                if more_props || additional {
                    out[b',' as usize] = true;
                }
                if !missing_required {
                    out[b'}' as usize] = true;
                }
            }
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn array_allowed(
        &self,
        min: Option<usize>,
        max: Option<usize>,
        count: usize,
        phase: &ArrPhase,
        out: &mut [bool; 256],
    ) {
        for &b in &WS {
            out[b as usize] = true;
        }
        let at_max = max.is_some_and(|m| count >= m);
        let at_min = min.is_none_or(|m| count >= m);
        match phase {
            ArrPhase::ExpectValueOrEnd { just_after_comma } => {
                if !at_max {
                    // delegate to the items node's starters
                    if let Some(Frame::Array { items, .. }) = self.stack.last() {
                        self.value_starters(items.as_ref(), out);
                    }
                }
                if !just_after_comma && at_min {
                    out[b']' as usize] = true;
                }
            }
            ArrPhase::AfterValue => {
                if !at_max {
                    out[b',' as usize] = true;
                }
                if at_min {
                    out[b']' as usize] = true;
                }
            }
        }
    }

    /// `true` when the innermost frame is an object that is mid-key — the one
    /// place inside a container where a whitespace byte is string content
    /// rather than an insignificant separator.
    fn in_object_key(&self) -> bool {
        matches!(
            self.stack.last(),
            Some(Frame::Object {
                phase: ObjPhase::InKey { .. } | ObjPhase::InFreeKey { .. },
                ..
            })
        )
    }

    /// Accept one insignificant-whitespace byte, refusing once the run reaches
    /// [`MAX_INSIGNIFICANT_WS_RUN`]. Refusing is what makes the whitespace
    /// no-op terminating: the per-token mask probe feeds candidate tokens
    /// through this same `step`, so a rejected byte drops every
    /// whitespace-only token out of the mask and the decoder must emit the
    /// next structural byte.
    fn accept_insignificant_ws(&mut self) -> Result<(), ()> {
        if self.ws_run >= MAX_INSIGNIFICANT_WS_RUN {
            return Err(());
        }
        self.ws_run += 1;
        Ok(())
    }

    /// Consume one byte. `Err(())` if illegal at the current position.
    #[allow(clippy::result_unit_err)]
    pub fn step(&mut self, byte: u8) -> Result<(), ()> {
        if self.done {
            if WS.contains(&byte) {
                return self.accept_insignificant_ws();
            }
            return Err(());
        }

        // Whitespace handling. JSON allows insignificant whitespace only
        // *between* tokens — never inside a string, literal, enum, or before
        // the root value. Treating it as an unconditional no-op let a greedy
        // (temp=0) decoder loop on JSON-legal whitespace forever instead of
        // emitting the value, so we no-op it only where the spec actually
        // permits it:
        let ws_noop = match &self.leaf {
            // Inside a string, literal, or enum trie: whitespace is never an
            // insignificant separator. String whitespace is content (handled by
            // `step_instr`); a literal/enum can't contain mid-token whitespace.
            // Reject it (don't skip) so the trie advances and a greedy decoder
            // can't loop.
            Some(Leaf::InStr { .. } | Leaf::InLit { .. }) => false,
            // Value-start: leading whitespace before an *interior* value is
            // valid JSON (`: <ws> value`), but before the *root* value it is a
            // pointless no-op a greedy decoder loops on, so reject it there.
            Some(Leaf::Start(_)) => !self.stack.is_empty(),
            // Inside a number, trailing whitespace closes it (below).
            Some(Leaf::InNum { .. }) => true,
            // No active leaf ⇒ inside a container frame. Whitespace there is an
            // insignificant separator only *between* structural tokens. Inside
            // an object key string it is string content and must reach the key
            // sub-machine, which decides whether it is legal (a space is; a raw
            // control byte is not) and advances the key trie.
            None => !self.in_object_key(),
        };
        if ws_noop && WS.contains(&byte) {
            if let Some(Leaf::InNum { seen_digit, .. }) = &self.leaf {
                if *seen_digit {
                    self.close_leaf();
                }
            }
            return self.accept_insignificant_ws();
        }
        // A content or structural byte ends the whitespace run.
        self.ws_run = 0;

        if self.leaf.is_some() {
            return self.step_leaf(byte);
        }

        // Structural step inside a container frame.
        let frame = self.stack.last_mut().ok_or(())?;
        match frame {
            Frame::Object { .. } => self.step_object(byte),
            Frame::Array { .. } => self.step_array(byte),
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn step_leaf(&mut self, byte: u8) -> Result<(), ()> {
        // Take the leaf out to satisfy the borrow checker; restore unless a
        // transition replaces it.
        let leaf = self.leaf.take().ok_or(())?;
        match leaf {
            Leaf::Start(node) => self.enter_value(node, byte),
            Leaf::InStr {
                escape,
                unicode,
                done_to,
            } => self.step_instr(byte, escape, unicode, done_to),
            Leaf::InLit { mut trie, done_to } => {
                trie.step(byte)?;
                // Done when some viable literal exactly matches the
                // consumed length and none can extend further.
                if trie.complete() && !trie.viable.iter().any(|&i| trie.lits[i].len() > trie.pos) {
                    self.value_complete(done_to);
                } else {
                    self.leaf = Some(Leaf::InLit { trie, done_to });
                }
                Ok(())
            }
            Leaf::InNum {
                integer,
                seen_digit,
                done_to,
            } => {
                let is_num_byte = byte.is_ascii_digit()
                    || byte == b'-'
                    || byte == b'+'
                    || (!integer && matches!(byte, b'.' | b'e' | b'E'));
                if is_num_byte {
                    self.leaf = Some(Leaf::InNum {
                        integer,
                        seen_digit: seen_digit || byte.is_ascii_digit(),
                        done_to,
                    });
                    Ok(())
                } else if seen_digit {
                    // number closed; re-dispatch this byte as a follower
                    self.value_complete(done_to);
                    self.step(byte)
                } else {
                    Err(())
                }
            }
        }
    }

    fn step_instr(
        &mut self,
        byte: u8,
        escape: bool,
        unicode: u8,
        done_to: AfterTarget,
    ) -> Result<(), ()> {
        if unicode > 0 {
            let is_hex = byte.is_ascii_digit()
                || (b'a'..=b'f').contains(&byte)
                || (b'A'..=b'F').contains(&byte);
            if !is_hex {
                return Err(());
            }
            self.leaf = Some(Leaf::InStr {
                escape: false,
                unicode: if unicode >= 4 { 0 } else { unicode + 1 },
                done_to,
            });
            return Ok(());
        }
        if escape {
            if byte == b'u' {
                self.leaf = Some(Leaf::InStr {
                    escape: false,
                    unicode: 1,
                    done_to,
                });
                Ok(())
            } else if b"\"\\/bfnrt".contains(&byte) {
                self.leaf = Some(Leaf::InStr {
                    escape: false,
                    unicode: 0,
                    done_to,
                });
                Ok(())
            } else {
                Err(())
            }
        } else {
            match byte {
                b'"' => {
                    self.value_complete(done_to);
                    Ok(())
                }
                b'\\' => {
                    self.leaf = Some(Leaf::InStr {
                        escape: true,
                        unicode: 0,
                        done_to,
                    });
                    Ok(())
                }
                // Raw C0 control bytes (incl. tab / newline / CR) are illegal
                // inside a JSON string — they must be escaped. Rejecting them
                // also stops a greedy decoder from looping on raw whitespace
                // inside a free-form string value.
                0..=0x1f => Err(()),
                _ => {
                    self.leaf = Some(Leaf::InStr {
                        escape: false,
                        unicode: 0,
                        done_to,
                    });
                    Ok(())
                }
            }
        }
    }

    /// Begin a value governed by `node` starting at `byte`.
    ///
    /// `node` is the shared schema sub-tree (`Arc`); container arms clone the
    /// already-`Arc`-wrapped property list / element schema (a refcount bump,
    /// not a deep copy) into the pushed frame.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn enter_value(&mut self, node: Arc<SchemaNode>, byte: u8) -> Result<(), ()> {
        let done_to = if self.stack.is_empty() {
            AfterTarget::Top
        } else {
            AfterTarget::Container
        };
        match node.as_ref() {
            SchemaNode::Union(branches) => {
                // Discriminated union: every branch is a `const` or a
                // string-`enum` literal → merge into one literal trie and
                // prune as bytes commit (true discriminated-union support).
                if let Some(lits) = union_literals(branches) {
                    let mut trie = LiteralTrie::new(lits);
                    trie.step(byte)?;
                    self.finish_or_keep_lit(trie, done_to);
                    return Ok(());
                }
                // Structural union: pick the first branch whose starter set
                // accepts `byte` (v1 limitation: no full backtracking). The
                // branch sub-schema is already `Arc`-wrapped, so entering it
                // is a refcount bump, not a deep clone.
                let mut chosen: Option<Arc<SchemaNode>> = None;
                for br in branches.iter() {
                    let mut s = [false; 256];
                    self.value_starters(br.as_ref(), &mut s);
                    if s[byte as usize] {
                        chosen = Some(Arc::clone(br));
                        break;
                    }
                }
                match chosen {
                    Some(n) => {
                        self.leaf = Some(Leaf::Start(n));
                        self.step(byte)
                    }
                    None => Err(()),
                }
            }
            SchemaNode::Const(v) => {
                let mut trie = LiteralTrie::new(vec![literal_bytes(v)]);
                trie.step(byte)?;
                if trie.complete() && trie.lits[0].len() == trie.pos {
                    self.value_complete(done_to);
                } else {
                    self.leaf = Some(Leaf::InLit { trie, done_to });
                }
                Ok(())
            }
            SchemaNode::Str { enum_: Some(lits) } => {
                let blits: Vec<Vec<u8>> = lits
                    .iter()
                    .map(|s| literal_bytes(&Value::String(s.clone())))
                    .collect();
                let mut trie = LiteralTrie::new(blits);
                trie.step(byte)?;
                if trie.complete() && !trie.viable.iter().any(|&i| trie.lits[i].len() > trie.pos) {
                    self.value_complete(done_to);
                } else {
                    self.leaf = Some(Leaf::InLit { trie, done_to });
                }
                Ok(())
            }
            SchemaNode::Str { enum_: None } => {
                if byte != b'"' {
                    return Err(());
                }
                self.leaf = Some(Leaf::InStr {
                    escape: false,
                    unicode: 0,
                    done_to,
                });
                Ok(())
            }
            SchemaNode::Num { integer } => {
                if !(byte.is_ascii_digit() || byte == b'-') {
                    return Err(());
                }
                self.leaf = Some(Leaf::InNum {
                    integer: *integer,
                    seen_digit: byte.is_ascii_digit(),
                    done_to,
                });
                Ok(())
            }
            SchemaNode::Bool => {
                let lits = vec![b"true".to_vec(), b"false".to_vec()];
                let mut trie = LiteralTrie::new(lits);
                trie.step(byte)?;
                self.finish_or_keep_lit(trie, done_to);
                Ok(())
            }
            SchemaNode::Null => {
                let mut trie = LiteralTrie::new(vec![b"null".to_vec()]);
                trie.step(byte)?;
                self.finish_or_keep_lit(trie, done_to);
                Ok(())
            }
            SchemaNode::Object {
                props,
                required,
                additional,
            } => {
                if byte != b'{' {
                    return Err(());
                }
                self.stack.push(Frame::Object {
                    node_props: Arc::clone(props),
                    required: Arc::clone(required),
                    additional: *additional,
                    emitted: Vec::new(),
                    phase: ObjPhase::ExpectKeyOrEnd {
                        just_after_comma: false,
                    },
                });
                Ok(())
            }
            SchemaNode::Array { items, min, max } => {
                if byte != b'[' {
                    return Err(());
                }
                self.stack.push(Frame::Array {
                    items: Arc::clone(items),
                    min: *min,
                    max: *max,
                    count: 0,
                    phase: ArrPhase::ExpectValueOrEnd {
                        just_after_comma: false,
                    },
                });
                Ok(())
            }
            SchemaNode::Any => {
                // Free-form JSON value (delegate to A6.3's flat grammar by
                // value-class on the first byte).
                self.enter_any(byte, done_to)
            }
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn finish_or_keep_lit(&mut self, trie: LiteralTrie, done_to: AfterTarget) {
        if trie.complete() && !trie.viable.iter().any(|&i| trie.lits[i].len() > trie.pos) {
            self.value_complete(done_to);
        } else {
            self.leaf = Some(Leaf::InLit { trie, done_to });
        }
    }

    /// `Any` node: classify the first byte and drive a permissive JSON
    /// value via a sub-`SchemaGrammar` is overkill — instead reuse the
    /// flat A6.3 semantics by mapping to leaf states with `Any` recursion.
    fn enter_any(&mut self, byte: u8, done_to: AfterTarget) -> Result<(), ()> {
        match byte {
            b'{' => {
                self.stack.push(Frame::Object {
                    node_props: Arc::from([]),
                    required: Arc::from([]),
                    additional: true,
                    emitted: Vec::new(),
                    phase: ObjPhase::ExpectKeyOrEnd {
                        just_after_comma: false,
                    },
                });
                Ok(())
            }
            b'[' => {
                self.stack.push(Frame::Array {
                    items: Arc::new(SchemaNode::Any),
                    min: None,
                    max: None,
                    count: 0,
                    phase: ArrPhase::ExpectValueOrEnd {
                        just_after_comma: false,
                    },
                });
                Ok(())
            }
            b'"' => {
                self.leaf = Some(Leaf::InStr {
                    escape: false,
                    unicode: 0,
                    done_to,
                });
                Ok(())
            }
            b't' | b'f' => {
                let lits = vec![b"true".to_vec(), b"false".to_vec()];
                let mut trie = LiteralTrie::new(lits);
                trie.step(byte)?;
                self.finish_or_keep_lit(trie, done_to);
                Ok(())
            }
            b'n' => {
                let mut trie = LiteralTrie::new(vec![b"null".to_vec()]);
                trie.step(byte)?;
                self.finish_or_keep_lit(trie, done_to);
                Ok(())
            }
            b'-' | b'0'..=b'9' => {
                self.leaf = Some(Leaf::InNum {
                    integer: false,
                    seen_digit: byte.is_ascii_digit(),
                    done_to,
                });
                Ok(())
            }
            _ => Err(()),
        }
    }

    /// Called when a leaf/structural value completes. `target` says where
    /// control returns; the innermost frame's phase is advanced.
    fn value_complete(&mut self, target: AfterTarget) {
        self.leaf = None;
        match target {
            AfterTarget::Top => {
                self.done = true;
            }
            AfterTarget::Container => match self.stack.last_mut() {
                Some(Frame::Object { phase, .. }) => {
                    *phase = ObjPhase::AfterValue;
                }
                Some(Frame::Array { count, phase, .. }) => {
                    *count += 1;
                    *phase = ArrPhase::AfterValue;
                }
                None => {
                    self.done = true;
                }
            },
        }
    }

    fn close_leaf(&mut self) {
        if let Some(Leaf::InNum { done_to, .. }) = self.leaf {
            self.value_complete(done_to);
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or validated before call"
    )]
    fn step_object(&mut self, byte: u8) -> Result<(), ()> {
        // Snapshot the bits we need so we can mutate the frame in place after.
        // `node_props`/`required` are `Arc`-shared schema → these clones are
        // refcount bumps, not deep copies; only `emitted`/`phase` are real
        // (tiny) copies.
        let (props, required, additional, emitted, phase) = match self.stack.last() {
            Some(Frame::Object {
                node_props,
                required,
                additional,
                emitted,
                phase,
            }) => (
                Arc::clone(node_props),
                Arc::clone(required),
                *additional,
                emitted.clone(),
                phase.clone(),
            ),
            _ => return Err(()),
        };

        match phase {
            ObjPhase::ExpectKeyOrEnd { just_after_comma } => match byte {
                b'"' => {
                    // Trie over still-pending property keys (quoted, in
                    // schema order). Mirrors llama.cpp `_build_object_rule`
                    // which emits required props in declared order.
                    let pending: Vec<Vec<u8>> = props
                        .iter()
                        .filter(|(k, _)| !emitted.contains(k))
                        .map(|(k, _)| literal_bytes(&Value::String(k.clone())))
                        .collect();
                    let new_phase = if pending.is_empty() {
                        if !additional {
                            return Err(());
                        }
                        // Free additional key: permissive string.
                        ObjPhase::InFreeKey {
                            escape: false,
                            unicode: 0,
                        }
                    } else {
                        let mut trie = LiteralTrie::new(pending);
                        trie.step(byte)?;
                        ObjPhase::InKey { trie }
                    };
                    if let Some(Frame::Object { phase, .. }) = self.stack.last_mut() {
                        *phase = new_phase;
                    }
                    Ok(())
                }
                b'}' if !just_after_comma => {
                    if required.iter().any(|r| !emitted.contains(r)) {
                        return Err(());
                    }
                    self.close_container_object()
                }
                _ => Err(()),
            },
            ObjPhase::InKey { mut trie } => {
                trie.step(byte)?;
                // A property key is done once a full quoted literal is
                // matched and no longer literal can extend the prefix.
                let key_done =
                    trie.complete() && !trie.viable.iter().any(|&i| trie.lits[i].len() > trie.pos);
                if key_done {
                    let matched = trie.viable.first().map(|&i| {
                        // strip surrounding quotes
                        String::from_utf8_lossy(&trie.lits[i][1..trie.lits[i].len() - 1])
                            .into_owned()
                    });
                    let which = matched
                        .as_ref()
                        .and_then(|m| props.iter().position(|(k, _)| k == m));
                    if let Some(Frame::Object { emitted, phase, .. }) = self.stack.last_mut() {
                        if let Some(m) = matched {
                            emitted.push(m);
                        }
                        *phase = ObjPhase::ExpectColon { which };
                    }
                } else if let Some(Frame::Object { phase, .. }) = self.stack.last_mut() {
                    *phase = ObjPhase::InKey { trie };
                }
                Ok(())
            }
            ObjPhase::InFreeKey { escape, unicode } => {
                let next = free_key_step(byte, escape, unicode)?;
                if let Some(Frame::Object { phase, .. }) = self.stack.last_mut() {
                    *phase = match next {
                        FreeKeyNext::Continue { escape, unicode } => {
                            ObjPhase::InFreeKey { escape, unicode }
                        }
                        FreeKeyNext::Done => ObjPhase::ExpectColon { which: None },
                    };
                }
                Ok(())
            }
            ObjPhase::ExpectColon { which } => {
                if byte != b':' {
                    return Err(());
                }
                // A schema'd property's value-schema is already `Arc`-wrapped,
                // so entering it is a refcount bump. A free additional key
                // (`which == None`) falls back to `Any`.
                let value_node = which
                    .and_then(|i| props.get(i).map(|(_, n)| Arc::clone(n)))
                    .unwrap_or_else(|| Arc::new(SchemaNode::Any));
                self.leaf = Some(Leaf::Start(value_node));
                if let Some(Frame::Object { phase, .. }) = self.stack.last_mut() {
                    *phase = ObjPhase::AfterValue;
                }
                Ok(())
            }
            ObjPhase::AfterValue => match byte {
                b',' => {
                    if let Some(Frame::Object { phase, .. }) = self.stack.last_mut() {
                        *phase = ObjPhase::ExpectKeyOrEnd {
                            just_after_comma: true,
                        };
                    }
                    Ok(())
                }
                b'}' => {
                    if required.iter().any(|r| !emitted.contains(r)) {
                        return Err(());
                    }
                    self.close_container_object()
                }
                _ => Err(()),
            },
        }
    }

    /// Record the just-emitted key into the frame's `emitted` set. Called
    /// from `value_complete` path indirectly — but we resolve it lazily by
    /// noting the key at ExpectColon time. To keep `emitted` correct we
    /// add the key when we leave `ExpectColon` (value pushed).
    fn close_container_object(&mut self) -> Result<(), ()> {
        match self.stack.pop() {
            Some(Frame::Object { .. }) => {
                if self.stack.is_empty() {
                    self.done = true;
                } else {
                    self.value_complete(AfterTarget::Container);
                }
                Ok(())
            }
            _ => Err(()),
        }
    }

    fn step_array(&mut self, byte: u8) -> Result<(), ()> {
        let (items, min, max, count, phase) = match self.stack.last() {
            Some(Frame::Array {
                items,
                min,
                max,
                count,
                phase,
            }) => (Arc::clone(items), *min, *max, *count, phase.clone()),
            _ => return Err(()),
        };
        let at_min = min.is_none_or(|m| count >= m);
        let at_max = max.is_some_and(|m| count >= m);
        match phase {
            ArrPhase::ExpectValueOrEnd { just_after_comma } => {
                if byte == b']' && !just_after_comma && at_min {
                    return self.close_container_array();
                }
                if byte == b']' {
                    return Err(());
                }
                if at_max {
                    // already at maxItems — no further element may start
                    return Err(());
                }
                // start a value governed by `items`
                self.leaf = Some(Leaf::Start(items));
                if let Some(Frame::Array { phase, .. }) = self.stack.last_mut() {
                    *phase = ArrPhase::AfterValue;
                }
                self.step(byte)
            }
            ArrPhase::AfterValue => match byte {
                b',' if !at_max => {
                    if let Some(Frame::Array { phase, .. }) = self.stack.last_mut() {
                        *phase = ArrPhase::ExpectValueOrEnd {
                            just_after_comma: true,
                        };
                    }
                    Ok(())
                }
                b']' if at_min => self.close_container_array(),
                _ => Err(()),
            },
        }
    }

    fn close_container_array(&mut self) -> Result<(), ()> {
        match self.stack.pop() {
            Some(Frame::Array { .. }) => {
                if self.stack.is_empty() {
                    self.done = true;
                } else {
                    self.value_complete(AfterTarget::Container);
                }
                Ok(())
            }
            _ => Err(()),
        }
    }
}
