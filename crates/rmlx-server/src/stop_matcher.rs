//! Stop-sequence truncation shared by both API surfaces.
//!
//! The `stop` request parameter (OpenAI `stop`, Anthropic `stop_sequences`)
//! must (a) halt generation and (b) **exclude** the matched stop string from
//! the returned content. Stop matching is done on the *detokenized* text, not
//! on raw token ids — a stop string can straddle token boundaries and span
//! multiple decoded pieces (e.g. the model emits `"char"` then `"lie"` and the
//! stop is `"charlie"`). Matching the concatenated text is the only correct
//! approach.
//!
//! One matching rule, two callers:
//!
//! - [`StopMatcher`] — stateful; buffers a held-back tail that could still be
//!   the prefix of a stop string until it is confirmed not to be one, so no
//!   post-stop content ever leaves the server. The streaming routes feed it
//!   piece by piece.
//! - [`ContentGate`] — the non-streaming routes' accumulator over a matcher.
//!   It appends the content that is safe to keep and, at a match, sets the end
//!   flag of the request so the engine ends the decode.
//!
//! The matcher runs on the content channel only: not on reasoning and not on
//! tool-call markup, which the tool parser takes out before the text reaches it.
//!
//! Matching rule (shared): the **earliest byte offset** wins; among stops that
//! match at the same offset, the one **earliest in the `stops` slice** wins
//! (OpenAI "first match in the array"). Empty stop strings are ignored — an
//! empty needle would match at offset 0 and truncate everything.

/// Result of scanning a string for stop sequences.
#[derive(Debug, PartialEq, Eq)]
struct StopHit {
    /// Byte offset in the text where the matched stop string begins. The
    /// caller truncates `text[..offset]` so the stop string is excluded.
    offset: usize,
    /// The stop string that matched.
    matched: String,
}

/// Find the first stop-sequence match in `text`.
///
/// Returns `None` when no stop string is present (no truncation needed).
/// Empty stop strings in `stops` are skipped.
#[must_use]
fn find_stop_match(text: &str, stops: &[String]) -> Option<StopHit> {
    let mut best: Option<StopHit> = None;
    for s in stops {
        if s.is_empty() {
            continue;
        }
        if let Some(off) = text.find(s.as_str()) {
            let take = match &best {
                // Strictly-earlier offset wins. Equal offset keeps the
                // earlier-in-slice stop (already stored), so only replace on
                // strictly-smaller offset.
                Some(b) => off < b.offset,
                None => true,
            };
            if take {
                best = Some(StopHit {
                    offset: off,
                    matched: s.clone(),
                });
            }
        }
    }
    best
}

/// Outcome of feeding one decoded piece to a [`StopMatcher`].
#[derive(Debug, Default, PartialEq, Eq)]
#[allow(
    clippy::exhaustive_structs,
    reason = "closed result type — (emit, stopped, matched) is the complete per-push contract; callers read all three fields"
)]
pub struct StopPush {
    /// Text that is now safe to emit to the client (stop string excluded).
    pub emit: String,
    /// `true` once a stop string has been matched. The caller must stop
    /// emitting and treat generation as finished with reason `"stop"`.
    pub stopped: bool,
    /// Which stop string matched (only set when `stopped == true`).
    pub matched: Option<String>,
}

/// Stop matcher.
///
/// Holds back the smallest suffix of the running text that could still grow
/// into a stop string, so a stop that straddles token boundaries is never
/// half-emitted. The longest stop string bounds how much text can be held
/// back at any time.
#[derive(Debug)]
pub struct StopMatcher {
    /// Non-empty stop strings, in request order (first-match precedence).
    stops: Vec<String>,
    /// Length of the longest stop string in bytes — bounds the held-back tail.
    max_len: usize,
    /// Text seen but not yet safe to emit (a potential stop-string prefix).
    pending: String,
    /// Set once a stop has matched; further pushes are no-ops.
    done: bool,
}

impl StopMatcher {
    /// Build a matcher from the request stop list. Empty stop strings are
    /// dropped. When the resulting list is empty the matcher is inert
    /// (`is_active() == false`) and every push passes through unchanged.
    #[must_use]
    pub fn new(stops: &[String]) -> Self {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let max_len = stops.iter().map(String::len).max().unwrap_or(0);
        Self {
            stops,
            max_len,
            pending: String::new(),
            done: false,
        }
    }

    /// `true` when at least one non-empty stop string is configured.
    #[must_use]
    pub fn is_active(&self) -> bool {
        !self.stops.is_empty()
    }

    /// Feed one decoded piece. Returns the text safe to emit and whether a
    /// stop was hit. After `stopped == true`, subsequent calls emit nothing.
    pub fn push(&mut self, piece: &str) -> StopPush {
        if self.done {
            return StopPush::default();
        }
        if self.stops.is_empty() {
            // Inert matcher: pass through verbatim.
            return StopPush {
                emit: piece.to_owned(),
                stopped: false,
                matched: None,
            };
        }
        self.pending.push_str(piece);
        self.drain()
    }

    /// Flush at end-of-stream: no further tokens can arrive, so any held-back
    /// tail cannot grow into a stop string and is emitted verbatim.
    pub fn finalize(&mut self) -> String {
        if self.done {
            return String::new();
        }
        std::mem::take(&mut self.pending)
    }

    /// Core of the streaming algorithm. Looks for a full match first; failing
    /// that, holds back the longest suffix that is a prefix of some stop
    /// string and emits the rest.
    fn drain(&mut self) -> StopPush {
        // 1. Full match anywhere in the pending buffer → emit up to it, stop.
        if let Some(hit) = find_stop_match(&self.pending, &self.stops) {
            let emit = self.pending[..hit.offset].to_owned();
            self.pending.clear();
            self.done = true;
            return StopPush {
                emit,
                stopped: true,
                matched: Some(hit.matched),
            };
        }

        // 2. No full match. Compute the safe-emit boundary: the longest suffix
        //    of `pending` that is a (proper) prefix of any stop string must be
        //    retained, because the next token could complete it. Everything
        //    before that suffix is safe to emit now.
        //
        //    Only the trailing `max_len - 1` bytes can possibly be such a
        //    suffix, so we scan suffix start positions from there forward and
        //    keep the earliest one that is a prefix of a stop string. Start
        //    positions are clamped to char boundaries so slicing is always
        //    valid UTF-8.
        let len = self.pending.len();
        let scan_from = len.saturating_sub(self.max_len.saturating_sub(1));
        let mut hold_at = len; // default: nothing held back
        let mut i = scan_from;
        while i < len {
            if self.pending.is_char_boundary(i) {
                let suffix = &self.pending[i..];
                if self.stops.iter().any(|s| s.starts_with(suffix)) {
                    hold_at = i;
                    break;
                }
            }
            i += 1;
        }

        let emit = self.pending[..hold_at].to_owned();
        // Retain only the held-back tail.
        self.pending.drain(..hold_at);
        StopPush {
            emit,
            stopped: false,
            matched: None,
        }
    }
}

/// Ask the engine to end the decode: a stop string matched. This is the one
/// place that writes the flag.
pub(crate) fn request_end(end_requested: &std::sync::Arc<std::sync::atomic::AtomicBool>) {
    end_requested.store(true, std::sync::atomic::Ordering::SeqCst);
}

/// The content channel of a non-streaming route, behind a [`StopMatcher`].
///
/// [`push`](Self::push) appends the text that is safe to keep. At a match it
/// sets the end flag of the request, so the engine ends the decode, and from
/// then on it keeps nothing: the caller also stops feeding reasoning and the
/// tool parser. The route still reads the generation to its terminal item.
#[derive(Debug)]
pub struct ContentGate {
    matcher: StopMatcher,
    matched: Option<String>,
    end_requested: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl ContentGate {
    /// A gate over `matcher` that sets `end_requested` at a match.
    #[must_use]
    pub fn new(
        matcher: StopMatcher,
        end_requested: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        Self {
            matcher,
            matched: None,
            end_requested,
        }
    }

    /// `true` once a stop string has matched.
    #[must_use]
    pub fn is_stopped(&self) -> bool {
        self.matched.is_some()
    }

    /// The stop string that matched, if one did.
    #[must_use]
    pub fn matched(&self) -> Option<&str> {
        self.matched.as_deref()
    }

    /// Append the part of `piece` that is safe to keep to `text`.
    pub fn push(&mut self, text: &mut String, piece: &str) {
        if self.matched.is_some() {
            return;
        }
        let pushed = self.matcher.push(piece);
        text.push_str(&pushed.emit);
        if pushed.stopped {
            self.matched = pushed.matched;
            request_end(&self.end_requested);
        }
    }

    /// At the end of the generation, append the held-back tail: it cannot grow
    /// into a stop string now.
    pub fn flush(&mut self, text: &mut String) {
        if self.matched.is_none() {
            text.push_str(&self.matcher.finalize());
        }
    }
}

#[cfg(test)]
#[path = "stop_matcher_tests.rs"]
mod stop_matcher_tests;
