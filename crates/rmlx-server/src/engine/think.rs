//! `<think>...</think>` stripping state machine for reasoning models (A3).

/// Whether the rendered prompt leaves the assistant turn *inside* an open
/// thinking block — i.e. the first token the model generates is reasoning.
///
/// **`rendered` must be the assistant-turn suffix the template appended, not
/// the whole prompt.** Message content is client-controlled and may contain a
/// literal delimiter; scanning the whole prompt lets a user message decide the
/// channel. Callers obtain the suffix by re-rendering with
/// `add_generation_prompt: false` and taking the delta.
///
/// This drives [`ThinkSplitter`]'s initial channel. Whether a checkpoint's
/// chat template prefills an open
/// `<think>`, prefills a *closed* `<think></think>`, or prefills nothing and
/// lets the model open the block itself is a property of that template, not of
/// the architecture — checkpoints of the same arch disagree, and a template is
/// free to ignore `enable_thinking` entirely. Guessing from either source
/// mis-classifies the disagreeing checkpoint, and the mistake does not
/// self-correct: a splitter that starts open when the prompt already closed the
/// block never sees the `</think>` that would close it, so every piece is
/// routed to `reasoning_content` and every consumer of the "model is thinking"
/// signal stays latched for the whole request.
///
/// Open iff the last thinking-start delimiter appears after the last
/// thinking-end delimiter. Assumes neither delimiter is a substring of the
/// other, which holds for `<think>` / `</think>` and for any sane override.
pub(crate) fn prompt_leaves_think_open(rendered: &str, start: &str, end: &str) -> bool {
    // An empty delimiter matches at every offset — `rfind("")` returns the end
    // of the haystack, which would make every prompt look like it left the
    // block open. The request boundary rejects that input; this keeps the
    // function total for any other caller.
    if start.is_empty() || end.is_empty() {
        return false;
    }
    match (rendered.rfind(start), rendered.rfind(end)) {
        (Some(s), Some(e)) => s > e,
        (Some(_), None) => true,
        (None, _) => false,
    }
}

// ── A3: <think>/</think> stripping state machine ─────────────────────────────

/// Tracks `<think>...</think>` boundaries across token pieces and routes
/// the visible text on either the normal content channel or the reasoning
/// channel (`is_thinking == true`).
///
/// The initial channel comes from [`prompt_leaves_think_open`] applied to the
/// rendered prompt — templates in the wild do all three of "prefill an open
/// `<think>`", "prefill a closed `<think></think>`" and "prefill nothing",
/// sometimes within one architecture, so it must be read off the prompt rather
/// than assumed. Both transitions (`<think>` opener and `</think>` closer) are
/// implemented because which one the model produces depends on that same
/// prefill.
#[derive(Debug, Clone)]
pub(crate) struct ThinkSplitter {
    open: bool,
    /// delimiter string that opens the thinking block (default `"<think>"`).
    thinking_start_token: String,
    /// delimiter string that closes the thinking block (default `"</think>"`).
    thinking_end_token: String,
    /// number of pieces routed under the thinking channel so far.
    /// Counts only pieces emitted while `open == true`. Used to enforce
    /// a per-request thinking budget.
    thinking_token_count: u32,
    /// optional per-request cap on thinking-channel pieces. `None`
    /// (the default) disables the budget entirely — `step` then never
    /// touches the counter-comparison branch beyond a single `Option`
    /// discriminant check, keeping the no-budget path zero-overhead.
    thinking_budget: Option<u32>,
    /// latched once `thinking_token_count` exceeds `thinking_budget`.
    /// Distinct from `force_close` so the budget is reported as exceeded
    /// even after the forced close has been requested.
    budget_exceeded: bool,
    /// set on the step that pushes the count past the budget. The
    /// decode loop reads this (via [`ThinkSplitter::take_force_close`])
    /// and injects the thinking_end_token id as the next decode input,
    /// returning the model to the answer channel. One-shot: cleared once
    /// taken so it fires exactly once per request.
    force_close: bool,
}

impl ThinkSplitter {
    /// Default constructor for architectures that do NOT prefill an open
    /// `<think>` in the assistant prompt. Initial state: closed.
    ///
    /// `pub(crate)` for future archs whose templates don't prefill a
    /// thinking block — today no production path constructs the splitter
    /// this way (all reasoning archs in the registry are Qwen3-family),
    /// so the production code uses `new_qwen3_prefilled` and this
    /// constructor is exercised only by unit tests.
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self {
            open: false,
            thinking_start_token: "<think>".to_owned(),
            thinking_end_token: "</think>".to_owned(),
            thinking_token_count: 0,
            thinking_budget: None,
            budget_exceeded: false,
            force_close: false,
        }
    }

    /// Constructor for Qwen3-family models whose chat template ends with
    /// `...<|im_start|>assistant\n<think>\n` — i.e. the literal `<think>`
    /// is already in the prefilled prompt and the model emits reasoning
    /// text directly until it produces `</think>`. Initial state: open.
    ///
    /// production now constructs the splitter via `new_for_request`
    /// (which threads the per-request `enable_thinking` + budget); this
    /// constructor remains for the unit tests that exercise the open-state
    /// machine directly.
    #[allow(dead_code)]
    pub(crate) fn new_qwen3_prefilled() -> Self {
        Self {
            open: true,
            thinking_start_token: "<think>".to_owned(),
            thinking_end_token: "</think>".to_owned(),
            thinking_token_count: 0,
            thinking_budget: None,
            budget_exceeded: false,
            force_close: false,
        }
    }

    /// Build the splitter for one request.
    ///
    /// `prompt_think_open` is [`prompt_leaves_think_open`] evaluated against
    /// the rendered prompt: `true` starts in the reasoning channel (the prompt
    /// left a `<think>` open), `false` starts in answer-mode so output routes
    /// to `content`. It already reflects `enable_thinking`, because the
    /// template rendered with that flag is what produced the prompt.
    ///
    /// `thinking_budget` is the optional per-request reasoning cap; `None`
    /// disables budget enforcement (zero-overhead default).
    ///
    /// `thinking_start_token` / `thinking_end_token` let callers
    /// redirect the splitter to non-default delimiter strings. `None`
    /// defaults to `"<think>"` / `"</think>"` to preserve existing behavior.
    ///
    /// An **empty** delimiter is treated as absent. `step`'s scanner advances
    /// by the length of the tag it matched, and an empty tag matches at offset
    /// 0 of every remainder — the loop would never shorten `rest` and would
    /// spin forever on a blocking-pool thread. The OpenAI route rejects the
    /// empty override at the boundary; this keeps the type total for every
    /// caller rather than relying on that one guard.
    pub(crate) fn new_for_request(
        prompt_think_open: bool,
        thinking_budget: Option<u32>,
        thinking_start_token: Option<String>,
        thinking_end_token: Option<String>,
    ) -> Self {
        Self {
            open: prompt_think_open,
            thinking_start_token: thinking_start_token
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "<think>".to_owned()),
            thinking_end_token: thinking_end_token
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "</think>".to_owned()),
            thinking_token_count: 0,
            thinking_budget,
            budget_exceeded: false,
            force_close: false,
        }
    }

    /// take the one-shot forced-close flag. Returns `true` exactly
    /// once, on the first call after the budget was exceeded; the decode
    /// loop uses this to inject the `</think>` end-token id and resume
    /// answer generation. Returns `false` on the zero-budget hot path.
    #[inline]
    pub(crate) fn take_force_close(&mut self) -> bool {
        if self.force_close {
            self.force_close = false;
            true
        } else {
            false
        }
    }

    /// Process a piece. Returns `(visible_text, is_thinking)`.
    ///
    /// If the piece contains the configured start/end delimiter, the tag
    /// is stripped from the visible text. If a transition happens
    /// mid-piece, the chunk before the tag is routed under the OLD
    /// state and the chunk after under the NEW state — but since the
    /// per-call return is a single `(String, bool)` pair, we pick the
    /// dominant channel: the state AFTER the last transition. In
    /// practice BPE emits delimiters as standalone tokens, so
    /// the pre-/post-split case is rare and the dominant-channel
    /// approximation is fine.
    ///
    /// Returns an empty visible string when the piece was nothing but a
    /// tag literal (callers should NOT emit an empty delta in that case).
    pub(crate) fn step(&mut self, piece: &str) -> (String, bool) {
        let start = &self.thinking_start_token;
        let end = &self.thinking_end_token;
        // Common case: no tag in piece, route under current state.
        if !piece.contains(start.as_str()) && !piece.contains(end.as_str()) {
            self.account_thinking_piece();
            return (piece.to_owned(), self.open);
        }

        // Slow path: strip tag literals, flip state on each occurrence.
        // Scan left-to-right; the "last state seen" wins for the visible
        // text concatenation (see docstring).
        let mut out = String::with_capacity(piece.len());
        let mut rest = piece;
        loop {
            // Find nearest tag (whichever comes first).
            let open_pos = rest.find(start.as_str());
            let close_pos = rest.find(end.as_str());
            let next = match (open_pos, close_pos) {
                (None, None) => {
                    out.push_str(rest);
                    break;
                }
                (Some(o), None) => Some((o, start.as_str(), true)),
                (None, Some(c)) => Some((c, end.as_str(), false)),
                (Some(o), Some(c)) => {
                    if o < c {
                        Some((o, start.as_str(), true))
                    } else {
                        Some((c, end.as_str(), false))
                    }
                }
            };
            let Some((pos, tag, new_open)) = next else {
                break;
            };
            // Emit prefix under current state, then flip.
            out.push_str(&rest[..pos]);
            self.open = new_open;
            rest = &rest[pos + tag.len()..];
        }
        // A piece that produced a `</think>` transition is the model's own
        // close; the dominant post-transition state decides whether the
        // residual text counts as thinking. (BPE almost always emits tags
        // standalone, so `out` is empty here in practice.)
        self.account_thinking_piece();
        (out, self.open)
    }

    /// count one emitted piece against the thinking budget when the
    /// splitter is currently in the reasoning channel, latching
    /// `budget_exceeded` / `force_close` once the count passes the cap.
    ///
    /// Zero-overhead on the default path: `thinking_budget == None` is a
    /// single `Option` discriminant check and returns immediately, so the
    /// budget-unset request pays nothing beyond that branch.
    #[inline]
    fn account_thinking_piece(&mut self) {
        let Some(budget) = self.thinking_budget else {
            return;
        };
        if !self.open || self.budget_exceeded {
            return;
        }
        self.thinking_token_count += 1;
        if self.thinking_token_count > budget {
            self.budget_exceeded = true;
            self.force_close = true;
        }
    }
}
