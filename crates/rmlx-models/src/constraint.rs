//! Sampler-side constraint engine.
//!
//! `ConstraintEngine` is a per-request decoding constraint. It produces a
//! boolean allow-mask over the vocabulary for each sampling step; the sampler
//! consults the mask to suppress disallowed token ids by adding
//! `f32::NEG_INFINITY` to their logits before `argmax` or, when sampling,
//! before softmax.
//!
//! This crate defines the trait and `NoOpConstraint` (mask all-`true`). The
//! real grammars (`JsonObjectConstraint`, `SchemaConstraint`) live in
//! `rmlx-server`'s `constraint_json`.
//!
//! # Hot-path cost
//!
//! The no-constraint path skips the trait entirely — `GenerationRequest`
//! carries `Option<Box<dyn ConstraintEngine>>` and the per-arch decode loop
//! pattern-matches on `Option::as_mut()`. `None` is a single discriminant
//! check before the existing `argmax` call. Only the
//! `response_format ∈ {json_object, json_schema}` path pays the mask cost.

/// Per-step decoding constraint.
///
/// Implementations track grammar / schema state internally and emit a
/// vocabulary-sized boolean allow-mask on each call to [`step_mask`].
/// The sampler then suppresses logits where the mask is `false`.
///
/// All methods take `&mut self`; the engine is owned by `GenerationRequest`
/// and threaded through the decode loop as
/// `Option<&mut dyn ConstraintEngine>`. The decode loop calls
/// [`step_mask`] right before the sampler, then [`advance`] with the
/// sampled token id.
///
/// `Send` is required because `GenerationRequest` is moved into a
/// `tokio::task::spawn_blocking` closure. `Sync` is not strictly required
/// because only one thread ever touches the engine, but adding it costs
/// nothing for the trivial impls and keeps the bound consistent with the
/// other dyn-trait fields on `GenerationRequest`.
///
/// [`step_mask`]: ConstraintEngine::step_mask
/// [`advance`]: ConstraintEngine::advance
pub trait ConstraintEngine: Send + Sync + std::fmt::Debug {
    /// Build the allow-mask for the next token.
    ///
    /// Returned slice has length `vocab_size`. Index `i` is the allow-bit for
    /// token id `i` — `true` means the token may be sampled, `false` means
    /// its logit is replaced with `f32::NEG_INFINITY` before `argmax`.
    ///
    /// Implementations cache the mask buffer internally so the same
    /// `&[bool]` is returned each step when nothing has changed. Callers
    /// must NOT mutate the returned slice.
    fn step_mask(&mut self, vocab_size: usize) -> &[bool];

    /// Inform the engine that `token_id` was sampled.
    ///
    /// The engine advances its internal state so that the next
    /// [`step_mask`](Self::step_mask) call reflects the consequence of
    /// having emitted `token_id`. The decode loop calls this after every
    /// sampled token, regardless of whether the mask was actually consulted
    /// (it always is, on the masked branch).
    fn advance(&mut self, token_id: u32);

    /// True if the constraint has reached an accept state — the caller may
    /// stop decoding (treated like EOS).
    ///
    /// Returning `false` always is safe; `NoOpConstraint` does exactly that.
    /// The JSON grammars flip this to `true` when the JSON / schema parse
    /// completes.
    fn finished(&self) -> bool;

    /// True when the engine wants the sampler to apply its mask.
    ///
    /// `false` is a hint to the decode loop that this step should use the
    /// fast unmasked `argmax` path. Implementations return `false` while
    /// in a warm-up / inert phase (e.g. before the model has emitted the
    /// first byte that engages the grammar — see
    /// `constraint_json::JsonObjectConstraint`).
    ///
    /// Default `true` keeps the existing contract for engines that always
    /// want masking (NoOp / strict-from-start grammars).
    fn wants_mask(&self) -> bool {
        true
    }

    /// True when the engine is actually enforcing its grammar.
    ///
    /// Engines with a warm-up phase stay `false` until the model emits
    /// something the grammar can latch onto. A generation that *ends* with
    /// this still `false` was never constrained at all — its output is
    /// unchecked, and byte-for-byte indistinguishable from output the
    /// grammar inspected and permitted. The decode loop reports that case so
    /// silent non-enforcement leaves a trace.
    ///
    /// Default `true`: an engine with no warm-up enforces from step one.
    fn engaged(&self) -> bool {
        true
    }

    /// Where the grammar engaged, in a handle that outlives the borrow.
    ///
    /// The engine is moved into the decode thread and dropped there. A route
    /// that cuts the reply at the engagement byte holds a clone of this from
    /// before the move. `None` for engines with no warm-up: the whole
    /// generation is the value.
    fn engagement(&self) -> Option<std::sync::Arc<Engagement>> {
        None
    }
}

/// The place where a constraint's grammar engaged: the first byte of the
/// value it enforces.
///
/// The constraint is the only writer. It names the token by its place in the
/// sequence of [`ConstraintEngine::advance`] calls (the first is 1), and the
/// byte by its distance from the end of that token's text. A detokenizer can
/// change the start of a token's text (a leading space), and that does not
/// move the byte. A decode loop shows a token to `advance` before it hands the
/// token on, so a reader that has the n-th token can ask about it.
#[derive(Debug, Default)]
pub struct Engagement(std::sync::atomic::AtomicU64);

impl Engagement {
    const WHOLE_TOKEN: u32 = u32::MAX;

    /// The value starts `bytes_to_end` bytes before the end of the text of
    /// the `token`-th token.
    pub fn report(&self, token: u32, bytes_to_end: usize) {
        let bytes_to_end = u32::try_from(bytes_to_end).unwrap_or(Self::WHOLE_TOKEN);
        self.0.store(
            (u64::from(token) << 32) | u64::from(bytes_to_end),
            std::sync::atomic::Ordering::Release,
        );
    }

    /// The value starts at the first byte of the `token`-th token.
    pub fn report_token_start(&self, token: u32) {
        self.report(token, Self::WHOLE_TOKEN as usize);
    }

    /// How many bytes at the start of `piece`, the text of the `token`-th
    /// token, are before the engagement byte. All of them while the grammar
    /// has not engaged at or before that token.
    pub fn bytes_before(&self, token: u32, piece: &str) -> usize {
        let place = self.0.load(std::sync::atomic::Ordering::Acquire);
        let (engaged_token, bytes_to_end) = ((place >> 32) as u32, place as u32 as usize);
        if engaged_token == 0 || token < engaged_token {
            return piece.len();
        }
        if token > engaged_token {
            return 0;
        }
        let before = piece.len().saturating_sub(bytes_to_end);
        if piece.is_char_boundary(before) {
            before
        } else {
            0
        }
    }
}

/// Engine-specific extension trait — opt-in. Decode loops won't call this
/// (they only know `ConstraintEngine`). The route's step_fn callback,
/// which knows whether the just-emitted token is reasoning vs answer,
/// downcasts via this trait to signal phase transitions.
pub trait ConstraintThinkSignal {
    /// True while the model is in its reasoning channel
    /// (`<think>...</think>` for Qwen3). Engines may use this to defer
    /// engagement until the answer phase begins.
    fn set_thinking(&mut self, _is_thinking: bool) {}
}

/// No-op constraint: every token is allowed at every step, never finishes.
///
/// Output is byte-identical to the unconstrained path at `temp=0` because
/// every logit is preserved.
///
/// The internal mask buffer is lazily-sized: the first call to
/// [`step_mask`](ConstraintEngine::step_mask) records the vocab size and
/// allocates once. Subsequent calls return the cached slice in O(1). A
/// vocab-size mismatch (multi-model swap mid-decode is impossible today)
/// triggers a re-allocation.
#[derive(Debug, Default)]
pub struct NoOpConstraint {
    mask: Vec<bool>,
}

impl NoOpConstraint {
    /// Construct a fresh engine with no preallocated buffer.
    pub fn new() -> Self {
        Self { mask: Vec::new() }
    }
}

impl ConstraintEngine for NoOpConstraint {
    fn step_mask(&mut self, vocab_size: usize) -> &[bool] {
        if self.mask.len() != vocab_size {
            self.mask = vec![true; vocab_size];
        }
        &self.mask
    }

    fn advance(&mut self, _token_id: u32) {}

    fn finished(&self) -> bool {
        false
    }
}

#[cfg(test)]
#[path = "constraint_tests.rs"]
pub(crate) mod tests;
