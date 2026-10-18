//! Decode-side stop condition of a NIAH cell: the run ends as soon as the
//! decoded text holds the needle.

use tokenizers::Tokenizer;

/// Collects the generated token ids and reports when their decoded text first
/// contains the needle.
pub(crate) struct NeedleWatch<'a> {
    tokenizer: &'a Tokenizer,
    needle: &'a str,
    ids: Vec<u32>,
    found: bool,
}

impl<'a> NeedleWatch<'a> {
    pub(crate) fn new(tokenizer: &'a Tokenizer, needle: &'a str) -> Self {
        Self {
            tokenizer,
            needle,
            ids: Vec::new(),
            found: false,
        }
    }

    /// Record one generated token; true once the needle is in the decoded text.
    pub(crate) fn push(&mut self, id: u32) -> bool {
        self.ids.push(id);
        if !self.found {
            let text = self
                .tokenizer
                .decode(&self.ids, false)
                .expect("decode the generated ids");
            self.found = text.contains(self.needle);
        }
        self.found
    }

    pub(crate) fn found(&self) -> bool {
        self.found
    }

    pub(crate) fn into_ids(self) -> Vec<u32> {
        self.ids
    }
}

#[cfg(test)]
#[path = "needle_watch_tests.rs"]
mod needle_watch_tests;
