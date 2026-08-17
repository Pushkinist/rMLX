//! A6.4/A6.5 `SchemaConstraint` — `ConstraintEngine` impl for `response_format: json_schema`.
//!
//! Reuses the `TokenBytesMap` + think-phase warm-up + engage logic from A6.3;
//! the inner grammar is schema-driven (`SchemaGrammar`).

#![allow(
    clippy::cognitive_complexity,
    clippy::items_after_statements,
    clippy::missing_fields_in_debug,
    clippy::needless_pass_by_value,
    unreachable_pub
)]

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use rmlx_models::ConstraintEngine;
use serde_json::Value;

use super::super::TokenBytesMap;
use super::grammar::{is_only_fence_or_whitespace, SchemaGrammar};
use super::types::{EngagePolicy, SchemaError, SchemaNode};

/// `ConstraintEngine` for `response_format: json_schema`. Reuses A6.3's
/// `TokenBytesMap` + think-phase warm-up + engage logic; the inner grammar
/// is schema-driven.
///
/// # Engage policy
///
/// Container roots (`object`, `array`) use [`EngagePolicy::ValueStarter`]:
/// engage when the first non-whitespace byte is a legal value-starter for
/// this schema (typically `{` or `[`).
///
/// Scalar roots (`string`/`enum`/`const`/`number`/`integer`/`boolean`/
/// `null`) use [`EngagePolicy::Immediate`]: engage at the **first
/// post-think token**, no byte inspection. Without this, a model that emits
/// a bare unquoted word (e.g. `medium` instead of `"medium"`) would never
/// trigger engagement — the grammar stays inert and the constraint never
/// corrects the output.
pub struct SchemaConstraint {
    grammar: SchemaGrammar,
    pub(super) bytes_map: Arc<TokenBytesMap>,
    eos_ids: Vec<u32>,
    mask: Vec<bool>,
    pub(super) engaged: bool,
    /// Mirror of `engaged` the route keeps a clone of. The engine itself is
    /// moved into the decode thread, so this is the only way the route can
    /// learn — after the stream drains — whether the grammar was ever applied.
    engaged_flag: Arc<AtomicBool>,
    engage_policy: EngagePolicy,
    /// Text accumulated from tokens seen before engagement. Used to detect
    /// and suppress a leading markdown-fence (```` ```json\n ````). Cleared
    /// when engagement fires.
    pre_engage_buf: String,
    is_thinking: Arc<AtomicBool>,
}

impl std::fmt::Debug for SchemaConstraint {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchemaConstraint")
            .field("done", &self.grammar.is_done())
            .field("eos_ids", &self.eos_ids)
            .field("vocab_size", &self.bytes_map.vocab_size())
            .field("engage_policy", &self.engage_policy)
            .finish()
    }
}

impl SchemaConstraint {
    /// Build a constraint: parse the schema, then precompute the
    /// token-bytes map (same ~600 ms one-time cost as A6.3).
    ///
    /// Returns `Err(SchemaError)` if the schema is malformed; the handler
    /// maps this to HTTP 400.
    ///
    /// `engage_policy_override`: when `Some(p)`, use `p` unconditionally;
    /// when `None`, derive from the schema root type (`Immediate` for
    /// scalars, `ValueStarter` for containers). Pass
    /// `Some(EngagePolicy::Immediate)` for `tool_choice=required/named`
    /// so the constraint engages on the first sampled token even when the
    /// schema root is an object (the model may emit a prefix before `{`).
    pub fn new(
        tokenizer: Arc<tokenizers::Tokenizer>,
        eos_ids: Vec<u32>,
        schema: &Value,
        strict: bool,
        engage_policy_override: Option<EngagePolicy>,
    ) -> Result<Self, SchemaError> {
        let node = SchemaNode::parse(schema, strict)?;
        let policy = engage_policy_override.unwrap_or_else(|| {
            if node.is_scalar_root() {
                EngagePolicy::Immediate
            } else {
                EngagePolicy::ValueStarter
            }
        });
        let bytes_map = Arc::new(TokenBytesMap::new(&tokenizer));
        Ok(Self {
            grammar: SchemaGrammar::new(node),
            bytes_map,
            eos_ids,
            mask: Vec::new(),
            engaged: false,
            engaged_flag: Arc::new(AtomicBool::new(false)),
            engage_policy: policy,
            pre_engage_buf: String::new(),
            is_thinking: Arc::new(AtomicBool::new(false)),
        })
    }

    /// Test constructor from a precomputed bytes map + parsed node.
    ///
    /// `engage_policy_override`: same semantics as [`SchemaConstraint::new`].
    pub fn from_parts(
        bytes_map: Arc<TokenBytesMap>,
        eos_ids: Vec<u32>,
        node: SchemaNode,
        engage_policy_override: Option<EngagePolicy>,
    ) -> Self {
        let policy = engage_policy_override.unwrap_or_else(|| {
            if node.is_scalar_root() {
                EngagePolicy::Immediate
            } else {
                EngagePolicy::ValueStarter
            }
        });
        Self {
            grammar: SchemaGrammar::new(node),
            bytes_map,
            eos_ids,
            mask: Vec::new(),
            engaged: false,
            engaged_flag: Arc::new(AtomicBool::new(false)),
            engage_policy: policy,
            pre_engage_buf: String::new(),
            is_thinking: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Force the constraint into the engaged state immediately, bypassing the engage policy.
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
    /// never checked against the requested schema.
    pub fn engaged_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.engaged_flag)
    }

    /// Return a shared handle to the `is_thinking` flag for external budget enforcement.
    pub fn is_thinking_handle(&self) -> Arc<AtomicBool> {
        Arc::clone(&self.is_thinking)
    }

    /// Feed decoded bytes from a delivered token into the grammar state machine.
    pub fn feed_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if self.grammar.step(b).is_err() {
                tracing::warn!(
                    byte = b,
                    all_bytes = ?bytes,
                    grammar_was_done_before = self.grammar.done,
                    "SchemaConstraint: grammar rejected byte; forcing terminal"
                );
                self.grammar.done = true;
                return;
            }
        }
    }

    /// Feed bytes into the grammar, stopping at the first invalid byte but
    /// NOT clamping to Done. Used for the engagement token where the mask
    /// has not yet filtered the token's suffix bytes — we accept as many
    /// valid bytes as possible and leave the grammar in a valid state so
    /// the next `step_mask` can steer subsequent tokens correctly.
    fn feed_bytes_partial(&mut self, bytes: &[u8]) {
        for &b in bytes {
            if self.grammar.step(b).is_err() {
                tracing::debug!(
                    byte = b,
                    "SchemaConstraint: engagement token suffix byte rejected (expected for multi-byte tokens); stopping feed"
                );
                return; // stop, do NOT clamp to Done
            }
        }
    }

    /// Returns `true` when the text accumulated before engagement matches
    /// the pattern `^\s*(```(json)?\s*)?$` — i.e. it is only whitespace
    /// and/or a markdown code-fence header with no real prose. The caller
    /// (route handler) may discard that prefix from `content`.
    pub fn pre_engage_is_fence(&self) -> bool {
        is_only_fence_or_whitespace(&self.pre_engage_buf)
    }
}

impl ConstraintEngine for SchemaConstraint {
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
        if !self.engaged {
            // Immediate policy: engage at the first step_mask call once
            // we are past the think phase. This handles the pipelined
            // decode architecture where the very first decode token is
            // computed before any advance() is called (pending = None in
            // the first loop iteration). Without this, the first token
            // would escape the mask entirely, producing a garbage prefix
            // before the real JSON value.
            if self.engage_policy == EngagePolicy::Immediate
                && !self.is_thinking.load(Ordering::Acquire)
            {
                tracing::info!(
                    policy = "immediate",
                    "SchemaConstraint: proactively engaging at first step_mask (scalar root)"
                );
                self.mark_engaged();
            // Fall through to the filtered mask path below.
            } else {
                self.mask.fill(true);
                return &self.mask;
            }
        }
        // Fast path: grammar is already done — only EOS is legal. Skip the
        // O(vocab) token loop entirely; the grammar's `step` method allows
        // trailing whitespace when done, which would otherwise let whitespace-
        // only tokens through and cause the model to emit infinite TABs/spaces
        // after the JSON value instead of stopping.
        if self.grammar.is_done() {
            for &eid in &self.eos_ids {
                if (eid as usize) < vocab_size {
                    self.mask[eid as usize] = true;
                }
            }
            tracing::debug!(
                allowed_count = self.mask.iter().filter(|&&b| b).count(),
                grammar_done = true,
                "SchemaConstraint::step_mask result (fast-path: grammar done)"
            );
            return &self.mask;
        }

        // O(vocab) probe shared with the json_object engine via
        // `fill_allow_mask`; `scratch` is a throwaway grammar reused across
        // tokens. The parsed schema inside `SchemaGrammar` is `Arc`-shared, so
        // the per-token reset is a few refcount bumps plus a buffer-reusing
        // refill of the small mutable progress — not a schema deep-copy. See
        // `SchemaGrammar::reset_from`.
        let mut scratch = self.grammar.clone();
        super::super::fill_allow_mask(
            &self.grammar,
            &mut scratch,
            &self.bytes_map,
            vocab_size,
            &mut self.mask,
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            let allowed_count = self.mask.iter().filter(|&&b| b).count();
            let n = self.bytes_map.vocab_size().min(vocab_size);
            let byte60_allowed_ids: Vec<usize> = (0..n)
                .filter(|&id| {
                    self.mask.get(id) == Some(&true)
                        && self.bytes_map.token_bytes(id).first() == Some(&60)
                })
                .take(5)
                .collect();
            tracing::debug!(
                allowed_count,
                byte60_allowed = byte60_allowed_ids.len(),
                byte60_examples = ?byte60_allowed_ids,
                grammar_done = self.grammar.done,
                "SchemaConstraint::step_mask result"
            );
        }
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
        tracing::debug!(
            token_id,
            bytes = ?bytes,
            grammar_done = self.grammar.done,
            engaged = self.engaged,
            "SchemaConstraint::advance"
        );
        if self.engaged {
            self.feed_bytes(&bytes);
        } else {
            if self.is_thinking.load(Ordering::Relaxed) {
                // Still inside the <think> block — do not engage yet.
                return;
            }
            // Accumulate pre-engagement text so the handler can detect and
            // suppress a leading markdown-fence wrapper.
            if let Ok(s) = std::str::from_utf8(&bytes) {
                self.pre_engage_buf.push_str(s);
            }
            match self.engage_policy {
                EngagePolicy::Immediate => {
                    // Scalar root: engage at the first post-think token
                    // regardless of its bytes. We do NOT feed the triggering
                    // token to the grammar — the model may emit an arbitrary
                    // word as its first answer token (e.g. "Here", "medium"),
                    // and feeding it would immediately reject and clamp the
                    // grammar to Done. Instead, just set `engaged = true` so
                    // that the NEXT `step_mask` call produces the correct
                    // first-byte constraint (force `"` for string/enum,
                    // digit/`-` for number, `t`/`f` for boolean, etc.), and
                    // the decode loop then samples a conforming token.
                    //
                    // Special case: if the triggering token already starts
                    // with a valid JSON value-starter byte (e.g. `"medium"`),
                    // feed it so we don't throw away a correctly formatted
                    // token that arrived on the engagement step.
                    if !bytes.is_empty() {
                        self.mark_engaged();
                        let starters = self.grammar.allowed_bytes();
                        // Skip leading whitespace, then check first non-WS byte.
                        let first_json_idx = bytes
                            .iter()
                            .position(|&b| b != b' ' && b != b'\t' && b != b'\n' && b != b'\r');
                        if let Some(idx) = first_json_idx {
                            if starters[bytes[idx] as usize] {
                                // First non-WS byte is a valid JSON starter.
                                tracing::info!(
                                    token_id,
                                    engage_byte = bytes[idx],
                                    policy = "immediate",
                                    "SchemaConstraint: engaging (scalar root, first post-think token, valid starter)"
                                );
                                // Partial feed: the engagement token's suffix may
                                // contain invalid bytes that the mask couldn't filter.
                                self.feed_bytes_partial(&bytes[idx..]);
                            } else {
                                // First non-WS byte is NOT a valid JSON
                                // starter (e.g. `H` in "Here"). Discard this
                                // token's bytes — the mask will enforce the
                                // correct opener on the next sampled token.
                                tracing::info!(
                                    token_id,
                                    first_byte = bytes[idx],
                                    policy = "immediate",
                                    "SchemaConstraint: engaging (scalar root, non-JSON token discarded, mask will correct)"
                                );
                            }
                        } else {
                            // Token is all-whitespace. Do NOT feed it: the
                            // grammar rejects whitespace before the root value
                            // (it is a pure no-op there), and `feed_bytes`
                            // clamps to done on rejection, which would answer
                            // the request with an empty value. Engage and let
                            // the next mask force a real value-starter.
                            tracing::info!(
                                token_id,
                                policy = "immediate",
                                "SchemaConstraint: engaging (scalar root, \
                                 whitespace token discarded, mask will correct)"
                            );
                        }
                    }
                }
                EngagePolicy::ValueStarter => {
                    // Container root: engage when the first non-whitespace
                    // byte is a legal value-starter for this schema (e.g.
                    // `{` or `[`). This is the A6.3 strategy — safe because
                    // instruction-tuned models reliably emit the structural
                    // opener as their first answer byte for object/array
                    // schemas.
                    let starters = self.grammar.allowed_bytes();
                    if let Some(idx) = bytes.iter().position(|&b| {
                        b != b' ' && b != b'\t' && b != b'\n' && b != b'\r' && starters[b as usize]
                    }) {
                        self.mark_engaged();
                        tracing::info!(
                            token_id,
                            engage_byte = bytes[idx],
                            policy = "value_starter",
                            "SchemaConstraint: engaging from this token"
                        );
                        // Use partial feed: the engagement token's suffix bytes have
                        // not been filtered by the mask (mask was all-true during
                        // warm-up). Accept as many valid bytes as possible and stop
                        // at the first invalid byte WITHOUT clamping to Done.
                        self.feed_bytes_partial(&bytes[idx..]);
                    }
                }
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
