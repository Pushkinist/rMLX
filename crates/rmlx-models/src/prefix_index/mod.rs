//! Pluggable longest-prefix index for the prompt cache.
//!
//! Two implementations behind the [`PrefixIndex`] trait:
//!
//! - [`LinearScan`] — O(slots × n_blocks) walk, byte-identical to the
//!   pre-`PromptCache::find_best_prefix` body.
//! - [`RadixTree`] — port of NVIDIA Dynamo's `PositionalRadixTree`
//!   (single-payload variant). Lookup is
//!   O(n_blocks · avg_fanout · avg_entries_per_node), not the textbook
//!   O(n_blocks) — per-step `find_child` linear-scans the child vec and
//!   `best_entry` linear-scans the cursor's `entries`. In practice fanout
//!   stays small (≤ working-set distinct continuations) and entry counts
//!   per node stay bounded, so the radix path is still effectively
//!   independent of total slot count once branch density saturates. See
//!   `docs/PERF_BASELINE.md` bench for empirical ns/op numbers.
//!
//! ## Contract
//!
//! Both implementations key on chained 256-token block digests
//! ([`crate::prompt_cache::chained_block_hashes_seeded`]) under a fixed
//! `layout_key`. Entries with the same `chained_hashes` but different
//! `layout_key` are kept disjoint — this mirrors the composite-PK rule
//! on the SQLite side and prevents a prompt cached at one KV layout from
//! falsely matching the same prompt cached at another layout.
//!
//! ## Payload semantics
//!
//! The trait stores a single user-supplied `slot_id: u64` per entry. For the
//! in-RAM `PromptCache<E>` integration the id is the `Slot::seq_id` (a
//! monotonically increasing counter never reused after `swap_remove`). The
//! Dynamo `RamSlot` / `SsdRow` enum is *not* implemented here — the prompt
//! cache already has its own RAM vs SSD dispatch and treats the radix tree as
//! a pure read accelerator over RAM slots.
//!
//! ## Lock order
//!
//! When wired into `PromptCache<E>`: tree mutation runs *before* the SQLite
//! eviction call (eviction of a RAM slot also calls `remove` on the tree
//! first, then drops the entry). This invariant is documented at the call
//! site in `prompt_cache.rs`; never reverse it.

#![allow(clippy::manual_let_else, clippy::semicolon_if_nothing_returned)]

mod linear;
mod radix;

pub use linear::LinearScan;
pub use radix::RadixTree;

use std::str::FromStr;

use crate::prompt_cache::FNV_OFFSET;

/// Error type for [`PrefixIndexKind::from_str`].
///
/// review LOW-2: replaces the previous `String` error so non-clap call
/// sites (config files, profile loader) get a typed error they can match on.
/// The CLI path itself routes through `clap::ValueEnum` (see MEDIUM-3) and
/// never invokes this `FromStr` impl.
#[non_exhaustive]
#[derive(Debug, thiserror::Error)]
pub enum PrefixIndexKindParseError {
    /// The supplied string is not one of the supported variants.
    #[error("invalid --prefix-index value '{0}' (expected 'linear' or 'radix')")]
    Invalid(String),
}

// ---------------------------------------------------------------------------
// Kind + CLI parse
// ---------------------------------------------------------------------------

/// Which [`PrefixIndex`] strategy a freshly constructed `PromptCache` uses.
///
/// Selected at process start via `--prefix-index {linear|radix}` (default
/// `linear`). The choice is global to the process — all per-arch
/// `PromptCache<E>` instances built after [`install_prefix_index_kind`] runs
/// pick up the same strategy.
#[allow(
    clippy::exhaustive_enums,
    reason = "closed dispatch enum — two prefix-index strategies (Linear/Radix); adding a strategy requires updating install_prefix_index_kind, active_prefix_index_kind, and all PromptCache construction sites"
)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefixIndexKind {
    /// O(slots × n_blocks) linear scan over `Vec<Slot>`. Pre-default.
    #[default]
    Linear,
    /// Positional radix tree (NVIDIA Dynamo port). Lookup is
    /// O(n_blocks · avg_fanout · avg_entries_per_node) — see module docs.
    Radix,
}

impl FromStr for PrefixIndexKind {
    type Err = PrefixIndexKindParseError;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "linear" => Ok(Self::Linear),
            "radix" => Ok(Self::Radix),
            other => Err(PrefixIndexKindParseError::Invalid(other.to_string())),
        }
    }
}

static PREFIX_INDEX_KIND: std::sync::OnceLock<PrefixIndexKind> = std::sync::OnceLock::new();

/// Install the process-global [`PrefixIndexKind`] (first call wins).
///
/// Called once at `rmlx serve` startup before any model loads. A second call
/// with a different value is dropped with a `warn!` (mirrors the
/// `install_ram_cap` / `ssd_tier::install_config` idempotency pattern).
pub fn install_prefix_index_kind(kind: PrefixIndexKind) {
    if PREFIX_INDEX_KIND.set(kind).is_err() {
        let existing = PREFIX_INDEX_KIND.get().copied().unwrap_or_default();
        if existing != kind {
            tracing::warn!(
                ?existing,
                requested = ?kind,
                "install_prefix_index_kind called more than once; keeping the first value"
            );
        }
        return;
    }
    tracing::info!(?kind, "prefix-index strategy installed");
}

/// Active [`PrefixIndexKind`] (default [`PrefixIndexKind::Linear`] when
/// [`install_prefix_index_kind`] has not been called — tests / unit paths).
pub fn active_prefix_index_kind() -> PrefixIndexKind {
    PREFIX_INDEX_KIND.get().copied().unwrap_or_default()
}

// ---------------------------------------------------------------------------
// PrefixIndex trait
// ---------------------------------------------------------------------------

/// Longest-prefix index over `(chained_hashes, layout_key)` keys.
///
/// Implementations:
/// - [`LinearScan`] — `Vec<Entry>` + linear walk (byte-identical to pre-).
/// - [`RadixTree`] — positional radix tree (Dynamo port).
///
/// All methods are `&mut self` because both impls mutate internal state on
/// every write (insert / remove / clear). Concurrency is the caller's
/// responsibility — the prompt cache already serialises every access through
/// its outer `Mutex`.
pub trait PrefixIndex: Send + Sync + std::fmt::Debug {
    /// Insert `(chained_hashes, layout_key) → slot_id`. Overwrites any
    /// existing entry with the same key. `chained_hashes.is_empty()` is a
    /// silent no-op (prompts shorter than one full block are never indexed).
    fn insert(&mut self, chained_hashes: &[u64], layout_key: u64, slot_id: u64);

    /// Remove the entry at `(chained_hashes, layout_key)`. No-op if absent.
    fn remove(&mut self, chained_hashes: &[u64], layout_key: u64);

    /// Longest-prefix match against `prompt_chained`. Returns
    /// `Some((slot_id, n_matched_blocks))` for the deepest entry whose key
    /// shares ≥1 block with `prompt_chained` *and* the same `layout_key`, or
    /// `None` otherwise.
    #[must_use]
    fn match_best(&self, prompt_chained: &[u64], layout_key: u64) -> Option<(u64, usize)>;

    /// Drop every entry. Used by `PromptCache::clear`.
    fn clear(&mut self);

    /// Entry count (test introspection).
    #[must_use]
    fn len(&self) -> usize;

    /// `len() == 0`.
    #[must_use]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

// ---------------------------------------------------------------------------
// Factory
// ---------------------------------------------------------------------------

/// Build a [`PrefixIndex`] of the kind currently installed via
/// [`install_prefix_index_kind`]. Called by every fresh `PromptCache<E>`.
pub fn build_active_index() -> Box<dyn PrefixIndex> {
    match active_prefix_index_kind() {
        PrefixIndexKind::Linear => Box::new(LinearScan::new()),
        PrefixIndexKind::Radix => Box::new(RadixTree::new()),
    }
}

/// startup-rebuild helper. Walks every `(hash, layout_key)` row in
/// `rows` (already ordered by `last_used ASC` — oldest first — as the SQLite
/// dump returns them) and inserts into a fresh radix tree, returning the
/// (tree, build_duration_ms). Tree is in-memory only; never persisted.
///
/// Returns an index keyed by the **last chained hash** of each prompt, since
/// SQLite stores one row per matched block and the chained-hash invariant
/// means that a single (last-block, layout_key) row uniquely identifies the
/// whole prefix that produced it. The caller (model load) drives a fresh
/// `chained_block_hashes_seeded(ids, FNV_OFFSET ^ layout_key)` over the
/// prompt_ids when one is available; for the SSD index this is not the case
/// (prompt_ids are not stored), so this method just rebuilds a flat single-
/// node-per-row tree as a warm cache for the longest-prefix lookup path.
pub fn rebuild_from_sqlite_rows(
    rows: impl IntoIterator<Item = RebuildRow>,
) -> (RadixTree, std::time::Duration) {
    let t0 = std::time::Instant::now();
    let mut tree = RadixTree::new();
    for row in rows {
        // Each SQLite row stores a single chained-block hash; we insert at
        // depth 1 (one-block prefix). Multi-block prefixes are not directly
        // reconstructible from the SQLite shape without the original
        // `prompt_ids`, so the rebuilt tree is a flat overlay that the
        // hydrator already consults as a per-block accelerator (the
        // `SsdKvIndex::lookup_longest_prefix` walk degenerates to a per-block
        // hashmap lookup for any longer prefix).
        tree.insert(&[row.last_block_hash], row.layout_key, row.row_id);
    }
    let elapsed = t0.elapsed();
    tracing::info!(
        rows = tree.len(),
        build_ms = elapsed.as_millis() as u64,
        "radix tree rebuilt from SQLite snapshot"
    );
    (tree, elapsed)
}

/// One row from the SQLite dump used by [`rebuild_from_sqlite_rows`].
///
/// Mirrors the composite PK + a synthetic `row_id` payload (any unique
/// u64 — the SQLite `rowid` works, or a packed `(last_used, byte_size)` if
/// the caller cares). `last_block_hash` is the chained FNV-1a-64 digest at
/// the **last** stored block — the same byte under `(hash, layout_key)` the
/// SsdKvIndex queries.
#[allow(
    clippy::exhaustive_structs,
    reason = "internal closed rebuild-row struct — three fields are the complete SsdKvIndex rebuild-scan contract; adding a field requires updating all SsdKvIndex::rebuild_from_snapshot callers"
)]
#[derive(Debug, Clone, Copy)]
/// Row descriptor for rebuilding a `PrefixIndex` from a snapshot.
pub struct RebuildRow {
    /// Hash of the last block in the rebuilt chain.
    pub last_block_hash: u64,
    /// Layout key of this chain.
    pub layout_key: u64,
    /// Row identifier from the source snapshot.
    pub row_id: u64,
}

/// Lightweight helper: `FNV_OFFSET ^ 0 == FNV_OFFSET` is the unsalted
/// default (kept here for `bench` callers that need to mint synthetic data
/// without depending on `prompt_cache::FNV_OFFSET` directly through the
/// `pub(crate)` boundary).
pub fn default_seed() -> u64 {
    FNV_OFFSET
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests;
