//! The content of a reply that a constraint shaped: `response_format`
//! (`json_object`, `json_schema`) or a forced tool call.
//!
//! The reply is the answer text from the byte at which the grammar engaged to
//! the end of the returned text. The constraint reports that byte
//! ([`rmlx_models::Engagement`]); this module holds no rule for it. Both the
//! streamed and the non-streamed path read the reply through [`JsonReply`].

#[cfg(test)]
#[path = "json_reply_tests.rs"]
mod json_reply_tests;

use std::sync::Arc;

use rmlx_models::Engagement;

/// Error type of a `response_format` reply whose returned text ends before
/// the engagement byte: HTTP 502 non-streamed, an error event streamed.
pub(crate) const NOT_ENGAGED_TYPE: &str = "constraint_not_engaged";

/// Message of [`NOT_ENGAGED_TYPE`].
pub(crate) const NOT_ENGAGED_MESSAGE: &str = "the reply holds no value that the requested \
     `response_format` grammar enforced, so it is unchecked and was not returned. Retry, or \
     drop `response_format`.";

/// Where the reply starts in the answer text of one request.
pub(crate) struct JsonReply {
    engagement: Arc<Engagement>,
    /// Bytes of answer text received.
    received: usize,
    /// Bytes of answer text released to the reply side.
    released: usize,
    /// The engagement byte as an offset in the answer text, once its token
    /// has arrived.
    start: Option<usize>,
}

impl JsonReply {
    pub(crate) fn new(engagement: Arc<Engagement>) -> Self {
        Self {
            engagement,
            received: 0,
            released: 0,
            start: None,
        }
    }

    /// Take `piece`, the answer text of the `token`-th token of the
    /// generation (the first is 1).
    pub(crate) fn receive(&mut self, token: u32, piece: &str) {
        if self.start.is_none() {
            let before = self.engagement.bytes_before(token, piece);
            if before < piece.len() {
                self.start = Some(self.received + before);
            }
        }
        self.received += piece.len();
    }

    /// The part of `text` that is in the reply. `text` is the next answer
    /// text in order: all of it, or what a stop matcher let through.
    pub(crate) fn release<'a>(&mut self, text: &'a str) -> &'a str {
        let from = self.released;
        self.released += text.len();
        self.start
            .and_then(|start| text.get(start.saturating_sub(from)..))
            .unwrap_or_default()
    }

    /// `true` when released text reached the engagement byte.
    pub(crate) fn engaged(&self) -> bool {
        self.start.is_some_and(|start| self.released > start)
    }

    /// Bytes of answer text before the reply, when it has started.
    pub(crate) fn start(&self) -> Option<usize> {
        self.start
    }
}
