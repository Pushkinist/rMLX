//! O(N) linear scan implementation of [`super::PrefixIndex`].

#![allow(clippy::manual_let_else, clippy::semicolon_if_nothing_returned)]

use super::PrefixIndex;

/// O(N) linear scan over a `Vec<Entry>`. Byte-identical to the pre-
/// `PromptCache::find_best_prefix` body — the index keeps a parallel copy of
/// every (chained_hashes, layout_key) the cache holds, and the longest-prefix
/// scan walks them all.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed index impl — field is private Vec<LinearEntry>; public API is the PrefixIndex trait; adding a field requires updating LinearScan::new and Default"
)]
#[derive(Debug, Default)]
pub struct LinearScan {
    entries: Vec<LinearEntry>,
}

#[derive(Debug, Clone)]
struct LinearEntry {
    chained: Vec<u64>,
    layout_key: u64,
    slot_id: u64,
}

impl LinearScan {
    /// Create a new empty `LinearScan` index.
    pub fn new() -> Self {
        Self::default()
    }

    fn position(&self, chained: &[u64], layout_key: u64) -> Option<usize> {
        self.entries
            .iter()
            .position(|e| e.layout_key == layout_key && e.chained == chained)
    }
}

impl PrefixIndex for LinearScan {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn insert(&mut self, chained_hashes: &[u64], layout_key: u64, slot_id: u64) {
        if chained_hashes.is_empty() {
            return;
        }
        if let Some(idx) = self.position(chained_hashes, layout_key) {
            self.entries[idx].slot_id = slot_id;
            return;
        }
        self.entries.push(LinearEntry {
            chained: chained_hashes.to_vec(),
            layout_key,
            slot_id,
        });
    }

    fn remove(&mut self, chained_hashes: &[u64], layout_key: u64) {
        if let Some(idx) = self.position(chained_hashes, layout_key) {
            self.entries.swap_remove(idx);
        }
    }

    fn match_best(&self, prompt_chained: &[u64], layout_key: u64) -> Option<(u64, usize)> {
        if prompt_chained.is_empty() {
            return None;
        }
        let mut best: Option<(u64, usize)> = None;
        for entry in &self.entries {
            if entry.layout_key != layout_key {
                continue;
            }
            let matched = prompt_chained
                .iter()
                .zip(entry.chained.iter())
                .take_while(|(a, b)| a == b)
                .count();
            if matched == 0 {
                continue;
            }
            match best {
                Some((_, prev)) if prev >= matched => {}
                _ => best = Some((entry.slot_id, matched)),
            }
        }
        best
    }

    fn clear(&mut self) {
        self.entries.clear();
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}
