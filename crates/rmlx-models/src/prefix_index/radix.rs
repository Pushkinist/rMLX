//! Positional radix tree implementation of [`super::PrefixIndex`]
//! (NVIDIA Dynamo port, single-payload variant).

#![allow(clippy::manual_let_else, clippy::semicolon_if_nothing_returned)]

use super::PrefixIndex;

/// Single-payload positional radix tree (NVIDIA Dynamo port).
///
/// Each node holds one block-hash + layout-key + optional payload (the
/// `slot_id` of the cache entry whose chained-hash sequence ends here).
/// Children are stored as a small `Vec<NodeId>`; fanout is bounded by the
/// number of distinct continuations cached under the same prefix and stays
/// tiny in practice (each parent has at most one child per (hash,layout_key)
/// pair the cache currently holds).
///
/// ## Lookup
///
/// `match_best` walks one block at a time, descending to the child whose
/// `(block_hash, layout_key)` matches the next element of `prompt_chained`.
/// The deepest visited node that carries a payload wins. Stops on the first
/// mismatch.
///
/// ## Insert
///
/// Walks/creates nodes along the chained-hash sequence, then stamps the
/// payload on the leaf. Repeated insert at the same key overwrites the
/// payload.
///
/// ## Remove
///
/// Clears the payload at the matching leaf, then walks back up pruning
/// payload-less, child-less nodes. Eviction-then-reinsert leaves no orphan
/// nodes — verified by the `eviction_then_reinsert_leaves_no_orphans` test.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed index impl — fields are private arena Vec + counter; public API is the PrefixIndex trait; adding a field requires updating RadixTree::new and Default"
)]
#[derive(Debug)]
pub struct RadixTree {
    pub(super) nodes: Vec<RadixNode>,
    /// Total inserted entries with a payload (drives `len()`).
    n_entries: usize,
}

#[derive(Debug)]
pub(super) struct RadixNode {
    /// Block hash at this node (root sentinel uses 0; never matched).
    pub(super) block_hash: u64,
    /// Layout key at this node (root sentinel uses 0; never matched).
    pub(super) layout_key: u64,
    /// Live entries whose chained-hash path *passes through* this node.
    /// Each entry contributes one tuple `(slot_id, leaf_depth)`. `match_best`
    /// picks the entry with the maximum `leaf_depth` so longest-prefix
    /// semantics matches `LinearScan` exactly: a query terminating at depth
    /// `d` returns the slot id of the entry with the **deepest** leaf in
    /// the subtree (capped by `d`).
    ///
    /// Memory: a single entry of n blocks contributes n tuples (16 B each),
    /// so a populated tree's memory is `Σ entries · depth · 16 B`. For the
    /// bench fanout (256 entries × 8 blocks) this is ~32 KiB total —
    /// negligible vs. KV-cache bytes.
    pub(super) entries: Vec<(u64, u32)>,
    /// Child node indices into `RadixTree::nodes`.
    pub(super) children: Vec<u32>,
}

impl RadixNode {
    pub(super) fn root() -> Self {
        Self {
            block_hash: 0,
            layout_key: 0,
            entries: Vec::new(),
            children: Vec::new(),
        }
    }

    fn new(block_hash: u64, layout_key: u64) -> Self {
        Self {
            block_hash,
            layout_key,
            entries: Vec::new(),
            children: Vec::new(),
        }
    }

    /// Slot id of the entry with the maximum `leaf_depth` (longest-prefix
    /// winner at this node). `None` when no live entry passes through.
    pub(super) fn best_entry(&self) -> Option<(u64, u32)> {
        self.entries.iter().copied().max_by_key(|(_, d)| *d)
    }
}

impl Default for RadixTree {
    fn default() -> Self {
        Self::new()
    }
}

impl RadixTree {
    /// Create a new empty `RadixTree` index.
    pub fn new() -> Self {
        Self {
            nodes: vec![RadixNode::root()],
            n_entries: 0,
        }
    }

    pub(super) const ROOT: u32 = 0;

    /// Find an existing child of `parent` matching `(block_hash, layout_key)`.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    pub(super) fn find_child(&self, parent: u32, block_hash: u64, layout_key: u64) -> Option<u32> {
        for &child in &self.nodes[parent as usize].children {
            let n = &self.nodes[child as usize];
            if n.block_hash == block_hash && n.layout_key == layout_key {
                return Some(child);
            }
        }
        None
    }

    /// Append a fresh child to `parent` and return its index.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn add_child(&mut self, parent: u32, block_hash: u64, layout_key: u64) -> u32 {
        let id = self.nodes.len() as u32;
        self.nodes.push(RadixNode::new(block_hash, layout_key));
        self.nodes[parent as usize].children.push(id);
        id
    }

    /// review MEDIUM-2 helper: evict a specific `(slot_id, leaf_depth)`
    /// tuple along `chained_hashes` from leaf to root, pruning empty nodes.
    /// Decrements `n_entries` if the tuple was found at the leaf. Used by
    /// `insert` to enforce the LinearScan-equivalent overwrite contract at a
    /// colliding `(chained, layout_key)` key with a different slot id.
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn evict_slot_path(
        &mut self,
        chained_hashes: &[u64],
        layout_key: u64,
        slot_id: u64,
        leaf_depth: u32,
    ) {
        // Re-walk to collect the path (find_child requires &self; we hold &mut
        // only across the mutation phase).
        let mut path: Vec<u32> = Vec::with_capacity(chained_hashes.len() + 1);
        path.push(Self::ROOT);
        let mut cursor = Self::ROOT;
        for &h in chained_hashes {
            match self.find_child(cursor, h, layout_key) {
                Some(c) => {
                    cursor = c;
                    path.push(c);
                }
                None => return,
            }
        }
        let mut found_at_leaf = false;
        for window in (1..path.len()).rev() {
            let child = path[window];
            let parent = path[window - 1];
            let n = &mut self.nodes[child as usize];
            if let Some(pos) = n
                .entries
                .iter()
                .position(|(s, d)| *s == slot_id && *d == leaf_depth)
            {
                n.entries.swap_remove(pos);
                if window == path.len() - 1 {
                    found_at_leaf = true;
                }
            }
            if n.entries.is_empty() && n.children.is_empty() {
                self.nodes[parent as usize].children.retain(|&c| c != child);
            }
        }
        if found_at_leaf {
            self.n_entries -= 1;
        }
    }

    /// Compute a stable canonical-traversal hash of the populated tree.
    /// Used by tests to assert "rebuild from same snapshot twice → equal
    /// trees" without depending on insertion order.
    ///
    /// Children are sorted by `(block_hash, layout_key)` before folding so
    /// the result is independent of insertion order at every node. Entry
    /// tuples per node are also sorted before folding.
    #[cfg(test)]
    pub fn canonical_hash(&self) -> u64 {
        #[allow(
            clippy::indexing_slicing,
            reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
        )]
        fn fold(tree: &RadixTree, node: u32) -> u64 {
            let n = &tree.nodes[node as usize];
            // FNV-1a-64 mix
            let mut h: u64 = 0xcbf2_9ce4_8422_2325;
            let mut mix = |x: u64| {
                for byte in x.to_le_bytes() {
                    h ^= u64::from(byte);
                    h = h.wrapping_mul(0x0000_0100_0000_01b3);
                }
            };
            mix(n.block_hash);
            mix(n.layout_key);
            let mut sorted_entries = n.entries.clone();
            sorted_entries.sort_unstable();
            mix(sorted_entries.len() as u64);
            for (s, d) in sorted_entries {
                mix(s);
                mix(u64::from(d));
            }
            let mut child_keys: Vec<(u64, u64, u32)> = n
                .children
                .iter()
                .map(|&c| {
                    let cn = &tree.nodes[c as usize];
                    (cn.block_hash, cn.layout_key, c)
                })
                .collect();
            child_keys.sort_unstable();
            for (_, _, c) in child_keys {
                mix(fold(tree, c));
            }
            h
        }
        fold(self, Self::ROOT)
    }
}

impl PrefixIndex for RadixTree {
    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn insert(&mut self, chained_hashes: &[u64], layout_key: u64, slot_id: u64) {
        if chained_hashes.is_empty() {
            return;
        }
        let leaf_depth = u32::try_from(chained_hashes.len()).unwrap_or(u32::MAX);
        // review MEDIUM-2: before walking/creating the new path,
        // detect whether the leaf at `(chained_hashes, layout_key)` already
        // carries a tuple at the same depth from a *different* slot id —
        // i.e. an overwrite-with-different-slot. LinearScan overwrites at
        // the matching `(chained, layout_key)` key; we mirror that contract
        // here by removing the prior slot's path tuples before inserting
        // the new one. Same-slot re-insert is handled by the dedupe inside
        // the stamping loop below.
        let mut existing_slots_to_evict: Vec<u64> = Vec::new();
        {
            let mut cursor = Self::ROOT;
            let mut found_full_path = true;
            for &h in chained_hashes {
                if let Some(c) = self.find_child(cursor, h, layout_key) {
                    cursor = c
                } else {
                    found_full_path = false;
                    break;
                }
            }
            if found_full_path {
                for &(s, d) in &self.nodes[cursor as usize].entries {
                    if d == leaf_depth && s != slot_id {
                        existing_slots_to_evict.push(s);
                    }
                }
            }
        }
        for s in existing_slots_to_evict {
            self.evict_slot_path(chained_hashes, layout_key, s, leaf_depth);
        }

        let mut cursor = Self::ROOT;
        // First: walk/create the path, building the node sequence.
        let mut path_nodes: Vec<u32> = Vec::with_capacity(chained_hashes.len());
        for &h in chained_hashes {
            cursor = match self.find_child(cursor, h, layout_key) {
                Some(c) => c,
                None => self.add_child(cursor, h, layout_key),
            };
            path_nodes.push(cursor);
        }
        // Stamp `(slot_id, leaf_depth)` at every node along the path.
        // If a tuple with the same slot_id already exists (re-insert of the
        // same slot — happens when the same prompt is pushed twice and the
        // outer PromptCache::push triggers a re-insert), update its
        // leaf_depth instead of appending a duplicate.
        let mut first_insert = true;
        for &node in &path_nodes {
            let n = &mut self.nodes[node as usize];
            if let Some(pos) = n.entries.iter().position(|(s, _)| *s == slot_id) {
                n.entries[pos].1 = leaf_depth;
                first_insert = false;
            } else {
                n.entries.push((slot_id, leaf_depth));
            }
        }
        if first_insert {
            self.n_entries += 1;
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn remove(&mut self, chained_hashes: &[u64], layout_key: u64) {
        if chained_hashes.is_empty() {
            return;
        }
        // Walk down recording the path. If any segment is missing the key
        // is not present — silent no-op.
        let mut path: Vec<u32> = Vec::with_capacity(chained_hashes.len() + 1);
        path.push(Self::ROOT);
        let mut cursor = Self::ROOT;
        for &h in chained_hashes {
            match self.find_child(cursor, h, layout_key) {
                Some(c) => {
                    cursor = c;
                    path.push(c);
                }
                None => return,
            }
        }
        let leaf_depth = u32::try_from(chained_hashes.len()).unwrap_or(u32::MAX);
        // The terminal node MUST carry a tuple matching this depth; pick
        // any slot id matching `leaf_depth`. There is no guaranteed unique
        // slot per (chained_hashes, layout_key) — if two entries inserted
        // the same key with the same depth they share path tuples, so we
        // pop exactly one. Use the **first** matching tuple at the leaf,
        // then pop that same slot id from every parent node.
        // review LOW-1: defensive `else { return; }` instead of
        // `unwrap()`. `path` always contains `ROOT + chained_hashes.len()`
        // nodes by construction above, so this branch is unreachable, but
        // the explicit early-return keeps the function panic-free.
        let Some(&leaf_node) = path.last() else {
            return;
        };
        let slot_id = match self.nodes[leaf_node as usize]
            .entries
            .iter()
            .find(|(_, d)| *d == leaf_depth)
            .map(|(s, _)| *s)
        {
            Some(s) => s,
            None => return,
        };
        // Walk path leaf-to-root and remove the (slot_id, leaf_depth) tuple
        // from each node. Where a node empties out of entries AND has no
        // remaining children, detach from parent (orphan; nodes vec is
        // append-only to keep indices stable — orphaned nodes leak until
        // `clear`, but that is bounded by the radix tree's working set).
        let mut found_at_leaf = false;
        for window in (1..path.len()).rev() {
            let child = path[window];
            let parent = path[window - 1];
            let n = &mut self.nodes[child as usize];
            if let Some(pos) = n
                .entries
                .iter()
                .position(|(s, d)| *s == slot_id && *d == leaf_depth)
            {
                n.entries.swap_remove(pos);
                if window == path.len() - 1 {
                    found_at_leaf = true;
                }
            }
            if n.entries.is_empty() && n.children.is_empty() {
                self.nodes[parent as usize].children.retain(|&c| c != child);
            }
        }
        if found_at_leaf {
            self.n_entries -= 1;
        }
    }

    #[allow(
        clippy::indexing_slicing,
        reason = "bounds established by construction: buffer sized at init, loop indices bounded by slice length, or layer index validated before call"
    )]
    fn match_best(&self, prompt_chained: &[u64], layout_key: u64) -> Option<(u64, usize)> {
        if prompt_chained.is_empty() {
            return None;
        }
        let mut cursor = Self::ROOT;
        let mut best: Option<(u64, usize)> = None;
        for (depth, &h) in prompt_chained.iter().enumerate() {
            match self.find_child(cursor, h, layout_key) {
                Some(c) => {
                    cursor = c;
                    // The deepest entry passing through this node gives the
                    // longest-prefix slot. Depth (matched blocks) is the
                    // cursor depth, NOT the leaf_depth of the chosen entry
                    // — we matched only `depth + 1` blocks of the prompt.
                    if let Some((slot, _leaf_depth)) = self.nodes[cursor as usize].best_entry() {
                        best = Some((slot, depth + 1));
                    }
                }
                None => break,
            }
        }
        best
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.nodes.push(RadixNode::root());
        self.n_entries = 0;
    }

    fn len(&self) -> usize {
        self.n_entries
    }
}
